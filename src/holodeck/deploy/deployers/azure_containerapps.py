"""Azure Container Apps deployer implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from holodeck.deploy.deployers.base import BaseDeployer
from holodeck.lib.errors import CloudSDKNotInstalledError, DeploymentError
from holodeck.models.deployment import (
    AzureContainerAppsConfig,
    DeployResult,
    StatusResult,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from azure.core.exceptions import (
        ClientAuthenticationError,
        HttpResponseError,
        ResourceNotFoundError,
        ServiceRequestError,
    )
    from azure.mgmt.appcontainers import ContainerAppsAPIClient
    from azure.mgmt.appcontainers.models import (
        Configuration,
        Container,
        ContainerApp,
        ContainerAppProbe,
        ContainerAppProbeHttpGet,
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
            from azure.core.exceptions import (
                ClientAuthenticationError,
                HttpResponseError,
                ResourceNotFoundError,
                ServiceRequestError,
            )
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.appcontainers import ContainerAppsAPIClient
            from azure.mgmt.appcontainers.models import (
                Configuration,
                Container,
                ContainerApp,
                ContainerAppProbe,
                ContainerAppProbeHttpGet,
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

        # Store exception types for use in methods
        self._ClientAuthenticationError: type[ClientAuthenticationError] = (
            ClientAuthenticationError
        )
        self._HttpResponseError: type[HttpResponseError] = HttpResponseError
        self._ResourceNotFoundError: type[ResourceNotFoundError] = ResourceNotFoundError
        self._ServiceRequestError: type[ServiceRequestError] = ServiceRequestError

        self._config = config
        self._client: ContainerAppsAPIClient = ContainerAppsAPIClient(
            DefaultAzureCredential(), config.subscription_id
        )
        self._Configuration: type[Configuration] = Configuration
        self._Container: type[Container] = Container
        self._ContainerApp: type[ContainerApp] = ContainerApp
        self._ContainerAppProbe: type[ContainerAppProbe] = ContainerAppProbe
        self._ContainerAppProbeHttpGet: type[ContainerAppProbeHttpGet] = (
            ContainerAppProbeHttpGet
        )
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
        readiness_path: str = "/ready",
        **kwargs: Any,
    ) -> DeployResult:
        """Deploy a container to Azure Container Apps."""
        self._echo_resolved_config(
            service_name=service_name, image_uri=image_uri, port=port
        )

        env_list = [
            self._EnvironmentVar(name=key, value=value)
            for key, value in env_vars.items()
        ]

        # Liveness probe — "is this process still alive". Failure → ACA
        # restarts the replica.
        liveness_probe = self._ContainerAppProbe(
            type="Liveness",
            http_get=self._ContainerAppProbeHttpGet(
                port=port,
                path=health_check_path,
            ),
            initial_delay_seconds=10,
            period_seconds=30,
            failure_threshold=3,
            timeout_seconds=5,
        )

        # Readiness probe — "is this replica ready to receive traffic". The
        # serve layer's /ready endpoint reflects server state; ACA holds
        # traffic off until this returns 200 so cold replicas don't race
        # their first request against init work (spec 034 P1a).
        readiness_probe = self._ContainerAppProbe(
            type="Readiness",
            http_get=self._ContainerAppProbeHttpGet(
                port=port,
                path=readiness_path,
            ),
            initial_delay_seconds=5,
            period_seconds=10,
            failure_threshold=3,
            timeout_seconds=5,
        )

        container = self._Container(
            name=service_name,
            image=image_uri,
            resources=self._ContainerResources(
                cpu=self._config.cpu,
                memory=self._config.memory,
            ),
            env=env_list if env_list else None,
            probes=[liveness_probe, readiness_probe],
        )

        scale = self._Scale(
            min_replicas=self._config.min_replicas,
            max_replicas=self._config.max_replicas,
        )

        template = self._Template(containers=[container], scale=scale)
        ingress = self._Ingress(
            external=self._config.ingress_external,
            target_port=port,
            traffic=[self._TrafficWeight(weight=100, latest_revision=True)],
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
        except self._ClientAuthenticationError as exc:
            raise DeploymentError(
                operation="deploy",
                message=f"Azure authentication failed: {exc}. "
                "Check your Azure credentials configuration.",
            ) from exc
        except self._ServiceRequestError as exc:
            raise DeploymentError(
                operation="deploy",
                message=f"Network error connecting to Azure: {exc}. "
                "Check your network connectivity.",
            ) from exc
        except self._HttpResponseError as exc:
            raise DeploymentError(
                operation="deploy",
                message=f"Azure API error (HTTP {exc.status_code}): {exc.message}",
            ) from exc
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

        result_service_id = result.id or result.name or service_name
        result_service_name = result.name or service_name
        result_status = (
            str(result.provisioning_state) if result.provisioning_state else "UNKNOWN"
        )

        return DeployResult(
            service_id=result_service_id,
            service_name=result_service_name,
            url=url,
            status=result_status,
        )

    def get_status(self, service_id: str) -> StatusResult:
        """Get status for an Azure Container App."""
        container_app_name = self._resolve_container_app_name(service_id)
        try:
            app = self._client.container_apps.get(
                resource_group_name=self._config.resource_group,
                container_app_name=container_app_name,
            )
        except self._ResourceNotFoundError as exc:
            raise DeploymentError(
                operation="status",
                message=f"Container app '{container_app_name}' not found: {exc}",
            ) from exc
        except self._ClientAuthenticationError as exc:
            raise DeploymentError(
                operation="status",
                message=f"Azure authentication failed: {exc}. "
                "Check your Azure credentials configuration.",
            ) from exc
        except self._ServiceRequestError as exc:
            raise DeploymentError(
                operation="status",
                message=f"Network error connecting to Azure: {exc}. "
                "Check your network connectivity.",
            ) from exc
        except self._HttpResponseError as exc:
            raise DeploymentError(
                operation="status",
                message=f"Azure API error (HTTP {exc.status_code}): {exc.message}",
            ) from exc
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

        status = str(app.provisioning_state) if app.provisioning_state else "UNKNOWN"
        return StatusResult(status=status, url=url)

    def destroy(self, service_id: str) -> None:
        """Destroy an Azure Container App deployment."""
        container_app_name = self._resolve_container_app_name(service_id)
        try:
            poller = self._client.container_apps.begin_delete(
                resource_group_name=self._config.resource_group,
                container_app_name=container_app_name,
            )
            poller.result()
        except self._ResourceNotFoundError as exc:
            raise DeploymentError(
                operation="destroy",
                message=f"Container app '{container_app_name}' not found: {exc}",
            ) from exc
        except self._ClientAuthenticationError as exc:
            raise DeploymentError(
                operation="destroy",
                message=f"Azure authentication failed: {exc}. "
                "Check your Azure credentials configuration.",
            ) from exc
        except self._ServiceRequestError as exc:
            raise DeploymentError(
                operation="destroy",
                message=f"Network error connecting to Azure: {exc}. "
                "Check your network connectivity.",
            ) from exc
        except self._HttpResponseError as exc:
            raise DeploymentError(
                operation="destroy",
                message=f"Azure API error (HTTP {exc.status_code}): {exc.message}",
            ) from exc
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

    def _echo_resolved_config(
        self, *, service_name: str, image_uri: str, port: int
    ) -> None:
        """Log the resolved deployment configuration before applying it.

        Surfaces the values that will land on the Container App so an
        operator running ``holodeck deploy run`` can see the new spec
        034 defaults (1 CPU / 2 GiB, internal ingress) before the apply
        completes. Cheap, visible, and the only output an operator sees
        for the runtime sizing decisions.
        """
        import math

        # Mirror the serve layer's derivation so the operator sees the
        # cap they'll get for this replica.
        derived_session_cap = max(1, math.floor(self._config.cpu * 2))
        ingress_mode = (
            "external (public)" if self._config.ingress_external else "internal"
        )

        logger.info("Deploying %s to Azure Container Apps", service_name)
        logger.info("  Image: %s", image_uri)
        logger.info("  Port: %d", port)
        logger.info(
            "  Replica: %.2f CPU / %s memory",
            self._config.cpu,
            self._config.memory,
        )
        logger.info(
            "  Concurrent Claude sessions per replica (default): %d",
            derived_session_cap,
        )
        logger.info(
            "  Replicas: min=%d, max=%d",
            self._config.min_replicas,
            self._config.max_replicas,
        )
        logger.info("  Ingress: %s", ingress_mode)
        logger.info("  Probes: /health (liveness), /ready (readiness)")

        if self._config.cpu < 1.0:
            logger.warning(
                "Replica CPU %.2f is below Anthropic's recommended minimum "
                "of 1.0 per Claude SDK instance. Concurrent sessions will "
                "cap at %d. Consider raising `deployment.target.azure.cpu`.",
                self._config.cpu,
                derived_session_cap,
            )
        if self._config.ingress_external:
            logger.warning(
                "Container App will be reachable from the PUBLIC INTERNET "
                "(ingress_external=true). Set `deployment.target.azure."
                "ingress_external: false` if this is unintentional."
            )

    @staticmethod
    def _resolve_container_app_name(service_id: str) -> str:
        """Resolve container app name from a service identifier.

        Args:
            service_id: Service identifier, either a container app name
                or a full Azure resource ID.

        Returns:
            The container app name extracted from the service_id.

        Raises:
            DeploymentError: If service_id is invalid (empty or resolves to empty).
        """
        if not service_id or not service_id.strip():
            raise DeploymentError(
                operation="resolve",
                message=f"Invalid service_id: '{service_id}' (empty or whitespace)",
            )

        if "/" in service_id:
            name = service_id.rstrip("/").split("/")[-1]
            if not name:
                raise DeploymentError(
                    operation="resolve",
                    message=f"Invalid service_id: '{service_id}' "
                    "(could not extract container app name)",
                )
            return name
        return service_id
