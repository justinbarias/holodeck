"""Azure Container Apps deployer implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from holodeck.deploy.deployers.base import BaseDeployer
from holodeck.lib.errors import CloudSDKNotInstalledError, DeploymentError
from holodeck.models.deployment import AzureContainerAppsConfig

if TYPE_CHECKING:
    from azure.mgmt.appcontainers import ContainerAppsAPIClient
    from azure.mgmt.appcontainers.models import (
        Configuration,
        Container,
        ContainerApp,
        ContainerResources,
        EnvironmentVar,
        Ingress,
        Scale,
        Template,
        TrafficWeight,
    )


class AzureContainerAppsDeployer(BaseDeployer):
    """Deploy HoloDeck agents to Azure Container Apps."""

    def __init__(self, config: AzureContainerAppsConfig) -> None:
        """Initialize Azure Container Apps deployer.

        Args:
            config: Azure Container Apps configuration

        Raises:
            DeploymentError: If configuration is missing required values
            CloudSDKNotInstalledError: If Azure SDK dependencies are missing
        """
        if not config.environment_name:
            raise DeploymentError(
                operation="deploy",
                message="Azure Container Apps requires environment_name to be set.",
            )

        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.appcontainers import ContainerAppsAPIClient
            from azure.mgmt.appcontainers.models import (
                Configuration,
                Container,
                ContainerApp,
                ContainerResources,
                EnvironmentVar,
                Ingress,
                Scale,
                Template,
                TrafficWeight,
            )
        except ImportError as exc:
            raise CloudSDKNotInstalledError(
                provider="azure", sdk_name="azure-mgmt-appcontainers"
            ) from exc

        self._config = config
        self._client: ContainerAppsAPIClient = ContainerAppsAPIClient(
            DefaultAzureCredential(), config.subscription_id
        )
        self._Configuration: type[Configuration] = Configuration
        self._Container: type[Container] = Container
        self._ContainerApp: type[ContainerApp] = ContainerApp
        self._ContainerResources: type[ContainerResources] = ContainerResources
        self._EnvironmentVar: type[EnvironmentVar] = EnvironmentVar
        self._Ingress: type[Ingress] = Ingress
        self._Scale: type[Scale] = Scale
        self._Template: type[Template] = Template
        self._TrafficWeight: type[TrafficWeight] = TrafficWeight
        self._environment_id = (
            f"/subscriptions/{config.subscription_id}/resourceGroups/"
            f"{config.resource_group}/providers/Microsoft.App/managedEnvironments/"
            f"{config.environment_name}"
        )

    def deploy(
        self,
        *,
        service_name: str,
        image_uri: str,
        port: int,
        env_vars: dict[str, str],
        health_check_path: str = "/health",
        **kwargs: Any,
    ) -> dict[str, str | None]:
        """Deploy a container to Azure Container Apps."""
        env_list = [
            self._EnvironmentVar(name=key, value=value)
            for key, value in env_vars.items()
        ]

        container = self._Container(
            name=service_name,
            image=image_uri,
            resources=self._ContainerResources(
                cpu=self._config.cpu,
                memory=self._config.memory,
            ),
            env=env_list if env_list else None,
        )

        scale = self._Scale(
            min_replicas=self._config.min_replicas,
            max_replicas=self._config.max_replicas,
        )

        template = self._Template(containers=[container], scale=scale)
        ingress = self._Ingress(
            external=self._config.ingress_external,
            target_port=port,
            traffic=[self._TrafficWeight(percentage=100, latest_revision=True)],
        )
        configuration = self._Configuration(ingress=ingress)

        container_app = self._ContainerApp(
            location=self._config.location,
            template=template,
            configuration=configuration,
            managed_environment_id=self._environment_id,
        )

        try:
            poller = self._client.container_apps.begin_create_or_update(
                resource_group_name=self._config.resource_group,
                container_app_name=service_name,
                container_app_envelope=container_app,
            )
            result = poller.result()
        except Exception as exc:
            raise DeploymentError(
                operation="deploy",
                message=f"Azure Container Apps deployment failed: {exc}",
            ) from exc

        url = None
        if result.configuration and result.configuration.ingress:
            fqdn = result.configuration.ingress.fqdn
            if fqdn:
                url = f"https://{fqdn}"

        service_id = result.id or result.name or service_name
        service_name = result.name or service_name
        status = result.provisioning_state or "UNKNOWN"

        return {
            "service_id": service_id,
            "service_name": service_name,
            "url": url,
            "status": status,
        }

    def get_status(self, service_id: str) -> dict[str, str | None]:
        """Get status for an Azure Container App."""
        container_app_name = self._resolve_container_app_name(service_id)
        try:
            app = self._client.container_apps.get(
                resource_group_name=self._config.resource_group,
                container_app_name=container_app_name,
            )
        except Exception as exc:
            raise DeploymentError(
                operation="status",
                message=f"Failed to fetch Azure deployment status: {exc}",
            ) from exc

        url = None
        if app.configuration and app.configuration.ingress:
            fqdn = app.configuration.ingress.fqdn
            if fqdn:
                url = f"https://{fqdn}"

        status = app.provisioning_state or "UNKNOWN"
        return {"status": status, "url": url}

    def destroy(self, service_id: str) -> None:
        """Destroy an Azure Container App deployment."""
        container_app_name = self._resolve_container_app_name(service_id)
        try:
            poller = self._client.container_apps.begin_delete(
                resource_group_name=self._config.resource_group,
                container_app_name=container_app_name,
            )
            poller.result()
        except Exception as exc:
            raise DeploymentError(
                operation="destroy",
                message=f"Failed to destroy Azure deployment: {exc}",
            ) from exc

    def stream_logs(self, service_id: str) -> Iterable[str]:
        """Stream logs for Azure Container Apps (not implemented)."""
        raise NotImplementedError(
            "Log streaming is not implemented for Azure Container Apps."
        )

    @staticmethod
    def _resolve_container_app_name(service_id: str) -> str:
        """Resolve container app name from a service identifier."""
        if "/" in service_id:
            return service_id.rstrip("/").split("/")[-1]
        return service_id
