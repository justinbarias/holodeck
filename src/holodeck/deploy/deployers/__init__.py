"""Cloud deployers for HoloDeck agents."""

from __future__ import annotations

from holodeck.deploy.deployers.base import BaseDeployer
from holodeck.lib.errors import DeploymentError
from holodeck.models.deployment import CloudProvider, CloudTargetConfig


def create_deployer(target: CloudTargetConfig) -> BaseDeployer:
    """Create a cloud deployer based on the target configuration."""
    if target.provider == CloudProvider.AZURE:
        if not target.azure:
            raise DeploymentError(
                operation="deploy",
                message="Azure configuration is required for Azure deployments.",
            )
        from holodeck.deploy.deployers.azure_containerapps import (
            AzureContainerAppsDeployer,
        )

        return AzureContainerAppsDeployer(target.azure)

    if target.provider == CloudProvider.AWS:
        raise DeploymentError(
            operation="deploy",
            message=(
                "AWS App Runner deployer is not implemented yet. "
                "Azure Container Apps is the only supported provider for now."
            ),
        )

    if target.provider == CloudProvider.GCP:
        raise DeploymentError(
            operation="deploy",
            message=(
                "GCP Cloud Run deployer is not implemented yet. "
                "Azure Container Apps is the only supported provider for now."
            ),
        )

    raise DeploymentError(
        operation="deploy",
        message=f"Unsupported cloud provider: {target.provider}",
    )


__all__ = ["BaseDeployer", "create_deployer"]
