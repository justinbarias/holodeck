"""Base interface for cloud deployers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from holodeck.models.deployment import DeployResult, StatusResult


class BaseDeployer(ABC):
    """Abstract base class for cloud deployers."""

    @abstractmethod
    def deploy(
        self,
        *,
        service_name: str,
        image_uri: str,
        port: int,
        env_vars: dict[str, str],
        health_check_path: str = "/health",
        **kwargs: Any,
    ) -> DeployResult:
        """Deploy a containerized service and return deployment details.

        Args:
            service_name: Name for the deployed service.
            image_uri: Full container image URI including tag.
            port: Container port to expose.
            env_vars: Environment variables to set in the container.
            health_check_path: HTTP path for health checks (default: /health).
            **kwargs: Provider-specific deployment options.

        Returns:
            DeployResult containing service_id, service_name, url, and status.

        Raises:
            DeploymentError: If deployment fails.
        """

    @abstractmethod
    def get_status(self, service_id: str) -> StatusResult:
        """Retrieve deployment status and URL by service identifier.

        Args:
            service_id: Unique identifier for the deployed service.

        Returns:
            StatusResult containing current status and URL.

        Raises:
            DeploymentError: If status check fails.
        """

    @abstractmethod
    def destroy(self, service_id: str) -> None:
        """Destroy a deployed service by identifier.

        Args:
            service_id: Unique identifier for the deployed service.

        Raises:
            DeploymentError: If destroy operation fails.
        """

    @abstractmethod
    def stream_logs(self, service_id: str) -> Iterable[str]:
        """Stream deployment logs for a service identifier.

        Args:
            service_id: Unique identifier for the deployed service.

        Returns:
            Iterable of log lines.

        Raises:
            NotImplementedError: When log streaming is not supported by the provider.
            DeploymentError: If log streaming fails.
        """
