"""Deployment state models for persisted deployments."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from holodeck.models.deployment import CloudProvider


class DeploymentRecord(BaseModel):
    """Persisted deployment record for a single agent."""

    model_config = ConfigDict(extra="forbid")

    provider: CloudProvider = Field(
        ..., description="Cloud provider for this deployment"
    )
    service_id: str = Field(..., description="Provider-specific service identifier")
    service_name: str = Field(..., description="Human-readable service name")
    url: str | None = Field(default=None, description="Deployment URL")
    status: str = Field(..., description="Deployment status")
    created_at: datetime | None = Field(
        default=None, description="Initial deployment timestamp"
    )
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )
    image_uri: str = Field(..., description="Deployed container image URI")
    config_hash: str = Field(..., description="Deployment configuration hash")


class DeploymentState(BaseModel):
    """Top-level deployment state stored on disk."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0", description="State file version")
    deployments: dict[str, DeploymentRecord] = Field(
        default_factory=dict, description="Deployments keyed by agent name"
    )
