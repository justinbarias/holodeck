"""ACA deployer emits EmptyDir volumes for ephemeral scratch (spec 034 P2a).

ACA does NOT expose securityContext primitives at any API version — see
the research note in Task 11 of the P2 plan. This test covers only the
volumes/volume_mounts surface, which IS exposed.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.mgmt.appcontainers.models import StorageType

from holodeck.deploy.deployers.azure_containerapps import (
    AzureContainerAppsDeployer,
)
from holodeck.models.deployment import AzureContainerAppsConfig


class _DummyModel:
    """Simple stand-in for Azure SDK model objects."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def deployer(monkeypatch: pytest.MonkeyPatch) -> AzureContainerAppsDeployer:
    """Build a deployer with the Azure SDK client mocked out.

    We keep the real ``azure.mgmt.appcontainers.models`` so that
    ``Volume``, ``VolumeMount``, and ``StorageType`` resolve to the true
    SDK objects and the storage_type comparison works correctly.
    """

    class _DummyClient:
        def __init__(self, credential: object, subscription_id: str) -> None:
            self.container_apps: MagicMock = MagicMock()

    # Patch azure.identity so DefaultAzureCredential doesn't need real creds.
    identity_module = types.ModuleType("azure.identity")
    identity_module.DefaultAzureCredential = MagicMock(return_value="cred")  # type: ignore[attr-defined]

    # Patch azure.mgmt.appcontainers (the client factory) while keeping
    # the real models module intact.
    appcontainers_module = types.ModuleType("azure.mgmt.appcontainers")
    appcontainers_module.ContainerAppsAPIClient = _DummyClient  # type: ignore[attr-defined]

    # Build a models module that delegates every name to the real SDK,
    # except the ones we haven't added yet (handled in the deployer).
    import azure.mgmt.appcontainers.models as _real_models

    monkeypatch.setitem(sys.modules, "azure.identity", identity_module)
    monkeypatch.setitem(sys.modules, "azure.mgmt.appcontainers", appcontainers_module)
    # Keep the real models module so real types are used for Volume etc.
    monkeypatch.setitem(sys.modules, "azure.mgmt.appcontainers.models", _real_models)

    config = AzureContainerAppsConfig(
        subscription_id="00000000-0000-0000-0000-000000000000",
        resource_group="rg",
        environment_name="env",
        location="eastus",
    )
    return AzureContainerAppsDeployer(config)


def _stub_poller_result() -> MagicMock:
    result = MagicMock(spec=[])
    result.id = "x"
    result.name = "t"
    result.provisioning_state = "Succeeded"
    ingress = MagicMock(spec=[])
    ingress.fqdn = "x.example"
    config = MagicMock(spec=[])
    config.ingress = ingress
    result.configuration = config
    return result


@pytest.mark.unit
def test_deploy_emits_tmpfs_volumes_at_template_level(
    deployer: AzureContainerAppsDeployer,
) -> None:
    """Template.volumes contains two EMPTY_DIR volumes."""
    poller = MagicMock()
    poller.result.return_value = _stub_poller_result()
    deployer._client.container_apps.begin_create_or_update.return_value = poller

    deployer.deploy(
        service_name="t",
        image_uri="ghcr.io/foo:bar",
        port=8080,
        env_vars={},
    )

    envelope = deployer._client.container_apps.begin_create_or_update.call_args.kwargs[
        "container_app_envelope"
    ]
    volumes = envelope.template.volumes or []
    names = {v.name for v in volumes}
    assert names == {"tmp", "sdk-scratch"}
    for v in volumes:
        assert v.storage_type == StorageType.EMPTY_DIR


@pytest.mark.unit
def test_deploy_emits_volume_mounts_on_container(
    deployer: AzureContainerAppsDeployer,
) -> None:
    """Container.volume_mounts wires /tmp and /var/holodeck/work."""
    poller = MagicMock()
    poller.result.return_value = _stub_poller_result()
    deployer._client.container_apps.begin_create_or_update.return_value = poller

    deployer.deploy(
        service_name="t",
        image_uri="ghcr.io/foo:bar",
        port=8080,
        env_vars={},
    )

    envelope = deployer._client.container_apps.begin_create_or_update.call_args.kwargs[
        "container_app_envelope"
    ]
    mounts = envelope.template.containers[0].volume_mounts or []
    paths = {m.mount_path for m in mounts}
    assert paths == {"/tmp", "/var/holodeck/work"}  # noqa: S108
    volume_names = {m.volume_name for m in mounts}
    assert volume_names == {"tmp", "sdk-scratch"}


@pytest.mark.unit
def test_deploy_does_not_attempt_to_set_security_context(
    deployer: AzureContainerAppsDeployer,
) -> None:
    """Regression guard: Container does NOT carry a security_context attr.

    ACA's Container resource has no security_context field at any API
    version. If a future SDK release adds one, this test will fail and
    prompt revisiting Tasks 11/12/14 of spec 034 P2.
    """
    poller = MagicMock()
    poller.result.return_value = _stub_poller_result()
    deployer._client.container_apps.begin_create_or_update.return_value = poller

    deployer.deploy(
        service_name="t",
        image_uri="ghcr.io/foo:bar",
        port=8080,
        env_vars={},
    )

    envelope = deployer._client.container_apps.begin_create_or_update.call_args.kwargs[
        "container_app_envelope"
    ]
    container = envelope.template.containers[0]
    assert (
        not hasattr(container, "security_context") or container.security_context is None
    )


@pytest.mark.unit
def test_echo_resolved_config_mentions_p2a_posture(
    deployer: AzureContainerAppsDeployer, caplog: pytest.LogCaptureFixture
) -> None:
    """Deploy-time echo names the ACA-enforced and image-enforced posture."""
    import logging

    with caplog.at_level(
        logging.INFO, logger="holodeck.deploy.deployers.azure_containerapps"
    ):
        deployer._echo_resolved_config(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080
        )
    msgs = " ".join(r.message for r in caplog.records).lower()
    # ACA-enforced (EmptyDir volumes)
    assert "emptydir" in msgs or "/var/holodeck/work" in msgs
    assert "/tmp" in msgs  # noqa: S108
    # Image-enforced (Dockerfile USER + chmod)
    assert "non-root" in msgs
    assert "read-only" in msgs


@pytest.mark.unit
def test_echo_resolved_config_does_not_falsely_claim_aca_securitycontext(
    deployer: AzureContainerAppsDeployer, caplog: pytest.LogCaptureFixture
) -> None:
    """Don't claim runAsNonRoot/capabilities.drop — ACA can't express them."""
    import logging

    with caplog.at_level(
        logging.INFO, logger="holodeck.deploy.deployers.azure_containerapps"
    ):
        deployer._echo_resolved_config(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080
        )
    msgs = " ".join(r.message for r in caplog.records).lower()
    assert "runasnonroot" not in msgs
    assert "allowprivilegeescalation" not in msgs
    assert "capabilities.drop" not in msgs
    assert "capabilities dropped" not in msgs
