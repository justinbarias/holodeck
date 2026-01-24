"""Base interface for cloud deployers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


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
    ) -> dict[str, str | None]:
        """Deploy a containerized service and return deployment details."""

    @abstractmethod
    def get_status(self, service_id: str) -> dict[str, str | None]:
        """Retrieve deployment status and URL by service identifier."""

    @abstractmethod
    def destroy(self, service_id: str) -> None:
        """Destroy a deployed service by identifier."""

    @abstractmethod
    def stream_logs(self, service_id: str) -> Iterable[str]:
        """Stream deployment logs for a service identifier.

        Implementations may raise NotImplementedError when log streaming
        is not supported by the provider.
        """
