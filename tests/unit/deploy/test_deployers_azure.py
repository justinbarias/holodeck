"""Unit tests for Azure Container Apps deployer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from holodeck.deploy.deployers.azure_containerapps import AzureContainerAppsDeployer
from holodeck.lib.errors import CloudSDKNotInstalledError, DeploymentError
from holodeck.models.deployment import AzureContainerAppsConfig


class DummyModel:
    """Simple container for Azure SDK model attributes."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_fixture() -> dict[str, Any]:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "deploy"
        / "mock_responses"
        / "azure"
        / "container_app_create.json"
    )
    return cast(dict[str, Any], json.loads(fixture_path.read_text(encoding="utf-8")))


def _build_result_from_fixture(data: dict[str, Any]) -> DummyModel:
    properties = data.get("properties", {})
    ingress_data = properties.get("configuration", {}).get("ingress", {})
    return DummyModel(
        id=data.get("id"),
        name=data.get("name"),
        provisioning_state=properties.get("provisioningState"),
        configuration=DummyModel(ingress=DummyModel(fqdn=ingress_data.get("fqdn"))),
    )


@pytest.fixture
def azure_sdk(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install mocked Azure SDK modules into sys.modules."""

    class DummyContainerAppsAPIClient:
        def __init__(self, credential: object, subscription_id: str) -> None:
            self.credential = credential
            self.subscription_id = subscription_id
            self.container_apps: MagicMock = MagicMock()

    # Mock azure.core.exceptions
    core_exceptions_module = types.ModuleType("azure.core.exceptions")
    core_exceptions_module.ClientAuthenticationError = type(  # type: ignore[attr-defined]
        "ClientAuthenticationError", (Exception,), {}
    )
    core_exceptions_module.HttpResponseError = type(  # type: ignore[attr-defined]
        "HttpResponseError", (Exception,), {"status_code": 500, "message": "error"}
    )
    core_exceptions_module.ResourceNotFoundError = type(  # type: ignore[attr-defined]
        "ResourceNotFoundError", (Exception,), {}
    )
    core_exceptions_module.ServiceRequestError = type(  # type: ignore[attr-defined]
        "ServiceRequestError", (Exception,), {}
    )

    identity_module = types.ModuleType("azure.identity")
    identity_module.DefaultAzureCredential = MagicMock(  # type: ignore[attr-defined]
        return_value="credential"
    )

    appcontainers_module = types.ModuleType("azure.mgmt.appcontainers")
    appcontainers_module.ContainerAppsAPIClient = DummyContainerAppsAPIClient  # type: ignore[attr-defined]

    models_module = types.ModuleType("azure.mgmt.appcontainers.models")
    for name in (
        "Configuration",
        "Container",
        "ContainerApp",
        "ContainerAppProbe",
        "ContainerAppProbeHttpGet",
        "ContainerResources",
        "EnvironmentVar",
        "Ingress",
        "Scale",
        "Template",
        "TrafficWeight",
    ):
        setattr(models_module, name, DummyModel)

    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.core", types.ModuleType("azure.core"))
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", core_exceptions_module)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt", types.ModuleType("azure.mgmt"))
    monkeypatch.setitem(sys.modules, "azure.mgmt.appcontainers", appcontainers_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt.appcontainers.models", models_module)

    return DummyContainerAppsAPIClient


class TestAzureContainerAppsDeployer:
    """Tests for AzureContainerAppsDeployer behavior."""

    def test_missing_sdk_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError results in CloudSDKNotInstalledError."""
        import builtins

        real_import = builtins.__import__

        def fake_import(
            name: str,
            globals: dict[str, Any] | None = None,
            locals: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] | list[str] = (),
            level: int = 0,
        ) -> Any:
            if name.startswith("azure"):
                raise ImportError("No module named azure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="test-rg",
            environment_name="test-env",
        )

        with pytest.raises(CloudSDKNotInstalledError):
            AzureContainerAppsDeployer(config)

    def test_deploy_calls_begin_create_or_update(self, azure_sdk: type) -> None:
        """Test deploy uses begin_create_or_update poller."""
        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="test-rg",
            environment_name="test-env",
            location="eastus",
            cpu=0.5,
            memory="1Gi",
            ingress_external=True,
            min_replicas=1,
            max_replicas=3,
        )
        deployer = AzureContainerAppsDeployer(config)

        client = deployer._client
        container_apps = cast(MagicMock, client.container_apps)
        poller = MagicMock()
        fixture = _load_fixture()
        result = _build_result_from_fixture(fixture)
        poller.result.return_value = result
        container_apps.begin_create_or_update.return_value = poller

        response = deployer.deploy(
            service_name="test-agent",
            image_uri="ghcr.io/holodeck/test-agent:abc1234",
            port=8080,
            env_vars={"OPENAI_API_KEY": "secret"},
        )

        call_kwargs = container_apps.begin_create_or_update.call_args.kwargs
        assert call_kwargs["resource_group_name"] == "test-rg"
        assert call_kwargs["container_app_name"] == "test-agent"

        envelope = call_kwargs["container_app_envelope"]
        assert envelope.location == "eastus"
        assert envelope.managed_environment_id.endswith("/managedEnvironments/test-env")

        container = envelope.template.containers[0]
        assert container.name == "test-agent"
        assert container.image == "ghcr.io/holodeck/test-agent:abc1234"
        assert container.resources.cpu == config.cpu
        assert container.resources.memory == config.memory
        assert container.env[0].name == "OPENAI_API_KEY"
        assert container.env[0].value == "secret"

        # Verify health probe configuration (uses default /health path)
        assert len(container.probes) == 1
        probe = container.probes[0]
        assert probe.type == "Liveness"
        assert probe.http_get.port == 8080
        assert probe.http_get.path == "/health"
        assert probe.initial_delay_seconds == 10
        assert probe.period_seconds == 30
        assert probe.failure_threshold == 3
        assert probe.timeout_seconds == 5

        scale = envelope.template.scale
        assert scale.min_replicas == 1
        assert scale.max_replicas == 3

        ingress = envelope.configuration.ingress
        assert ingress.external is True
        assert ingress.target_port == 8080
        assert ingress.traffic[0].weight == 100
        assert ingress.traffic[0].latest_revision is True

        assert response.service_id == fixture["id"]
        assert response.service_name == fixture["name"]
        assert response.status == fixture["properties"]["provisioningState"]
        assert (
            response.url
            == f"https://{fixture['properties']['configuration']['ingress']['fqdn']}"
        )

    def test_get_status_returns_url_and_status(self, azure_sdk: type) -> None:
        """Test get_status parses provisioning state and URL."""
        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="test-rg",
            environment_name="test-env",
        )
        deployer = AzureContainerAppsDeployer(config)
        client = deployer._client
        container_apps = cast(MagicMock, client.container_apps)

        container_apps.get.return_value = DummyModel(
            provisioning_state="RUNNING",
            configuration=DummyModel(ingress=DummyModel(fqdn="test.example.com")),
        )

        status = deployer.get_status("test-agent")

        container_apps.get.assert_called_once_with(
            resource_group_name="test-rg",
            container_app_name="test-agent",
        )
        assert status.status == "RUNNING"
        assert status.url == "https://test.example.com"

    def test_destroy_calls_begin_delete(self, azure_sdk: type) -> None:
        """Test destroy issues begin_delete and waits for completion."""
        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="test-rg",
            environment_name="test-env",
        )
        deployer = AzureContainerAppsDeployer(config)
        client = deployer._client
        container_apps = cast(MagicMock, client.container_apps)

        poller = MagicMock()
        container_apps.begin_delete.return_value = poller

        deployer.destroy("test-agent")

        container_apps.begin_delete.assert_called_once_with(
            resource_group_name="test-rg",
            container_app_name="test-agent",
        )
        poller.result.assert_called_once()

    def test_deploy_with_custom_health_check_path(self, azure_sdk: type) -> None:
        """Test deploy configures custom health check path."""
        config = AzureContainerAppsConfig(
            subscription_id="00000000-0000-0000-0000-000000000000",
            resource_group="test-rg",
            environment_name="test-env",
            location="eastus",
        )
        deployer = AzureContainerAppsDeployer(config)

        client = deployer._client
        container_apps = cast(MagicMock, client.container_apps)
        poller = MagicMock()
        fixture = _load_fixture()
        result = _build_result_from_fixture(fixture)
        poller.result.return_value = result
        container_apps.begin_create_or_update.return_value = poller

        deployer.deploy(
            service_name="test-agent",
            image_uri="ghcr.io/holodeck/test-agent:abc1234",
            port=8080,
            env_vars={},
            health_check_path="/api/healthz",
        )

        call_kwargs = container_apps.begin_create_or_update.call_args.kwargs
        envelope = call_kwargs["container_app_envelope"]
        container = envelope.template.containers[0]

        # Verify custom health check path is used
        probe = container.probes[0]
        assert probe.http_get.path == "/api/healthz"
        assert probe.http_get.port == 8080

    def test_resolve_container_app_name_valid_simple_name(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name returns simple names unchanged."""
        assert (
            AzureContainerAppsDeployer._resolve_container_app_name("my-app") == "my-app"
        )

    def test_resolve_container_app_name_valid_resource_id(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name extracts name from resource ID."""
        resource_id = (
            "/subscriptions/00000000-0000-0000-0000-000000000000/"
            "resourceGroups/my-rg/providers/Microsoft.App/containerApps/my-app"
        )
        assert (
            AzureContainerAppsDeployer._resolve_container_app_name(resource_id)
            == "my-app"
        )

    def test_resolve_container_app_name_valid_with_trailing_slash(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name handles trailing slashes."""
        resource_id = (
            "/subscriptions/00000000-0000-0000-0000-000000000000/"
            "resourceGroups/my-rg/providers/Microsoft.App/containerApps/my-app/"
        )
        assert (
            AzureContainerAppsDeployer._resolve_container_app_name(resource_id)
            == "my-app"
        )

    def test_resolve_container_app_name_invalid_empty(self, azure_sdk: type) -> None:
        """Test _resolve_container_app_name raises error for empty string."""
        with pytest.raises(DeploymentError, match="empty or whitespace"):
            AzureContainerAppsDeployer._resolve_container_app_name("")

    def test_resolve_container_app_name_invalid_whitespace(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name raises error for whitespace."""
        with pytest.raises(DeploymentError, match="empty or whitespace"):
            AzureContainerAppsDeployer._resolve_container_app_name("   ")

    def test_resolve_container_app_name_invalid_slash_only(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name raises error for '/' only."""
        with pytest.raises(DeploymentError, match="could not extract"):
            AzureContainerAppsDeployer._resolve_container_app_name("/")

    def test_resolve_container_app_name_invalid_double_slash(
        self, azure_sdk: type
    ) -> None:
        """Test _resolve_container_app_name raises error for '//' only."""
        with pytest.raises(DeploymentError, match="could not extract"):
            AzureContainerAppsDeployer._resolve_container_app_name("//")
