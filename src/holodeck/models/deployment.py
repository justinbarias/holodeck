"""Pydantic models for deployment configuration.

This module defines the configuration schema for HoloDeck agent deployments,
including registry, cloud provider, and container settings.
"""

import re
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class TagStrategy(str, Enum):
    """Strategy for generating container image tags."""

    GIT_SHA = "git_sha"
    GIT_TAG = "git_tag"
    LATEST = "latest"
    CUSTOM = "custom"


class CloudProvider(str, Enum):
    """Supported cloud providers for deployment."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class RuntimeType(str, Enum):
    """Container runtime types."""

    CONTAINER = "container"


class ProtocolType(str, Enum):
    """API protocol types for agent serving."""

    REST = "rest"
    AG_UI = "ag-ui"
    BOTH = "both"


# Regex patterns for validation
AWS_ARN_PATTERN = re.compile(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+$")
GCP_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
GCP_MEMORY_PATTERN = re.compile(r"^\d+(Mi|Gi)$")
AZURE_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
REPOSITORY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._/-]*[a-z0-9]$|^[a-z0-9]$")


class RegistryConfig(BaseModel):
    """Container registry configuration.

    Attributes:
        url: Registry URL (e.g., ghcr.io, docker.io, ECR URL)
        repository: Repository name (e.g., org/agent-name)
        tag_strategy: Strategy for generating image tags
        custom_tag: Custom tag when tag_strategy is CUSTOM
        credentials_env_prefix: Prefix for credential environment variables
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Registry URL (e.g., ghcr.io)")
    repository: str = Field(..., description="Repository name (e.g., org/agent-name)")
    tag_strategy: TagStrategy = Field(
        default=TagStrategy.GIT_SHA, description="Strategy for generating image tags"
    )
    custom_tag: str | None = Field(
        default=None, description="Custom tag when tag_strategy is CUSTOM"
    )
    credentials_env_prefix: str | None = Field(
        default=None, description="Prefix for credential environment variables"
    )

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, v: str) -> str:
        """Validate repository name pattern."""
        if not REPOSITORY_PATTERN.match(v):
            raise ValueError(
                f"Invalid repository name: {v}. "
                "Must contain only lowercase letters, numbers, '.', '_', '/', '-'"
            )
        return v

    @model_validator(mode="after")
    def validate_custom_tag(self) -> "RegistryConfig":
        """Validate that custom_tag is provided when tag_strategy is CUSTOM."""
        if self.tag_strategy == TagStrategy.CUSTOM and not self.custom_tag:
            raise ValueError("custom_tag is required when tag_strategy is 'custom'")
        return self


class AWSAppRunnerConfig(BaseModel):
    """AWS App Runner deployment configuration.

    Attributes:
        region: AWS region for deployment
        cpu: vCPU allocation (1, 2, or 4)
        memory: Memory allocation in MB (2048, 3072, 4096, 8192, or 12288)
        ecr_role_arn: IAM role ARN for ECR access
        health_check_path: Path for health checks
        auto_scaling_min: Minimum instances
        auto_scaling_max: Maximum instances
    """

    model_config = ConfigDict(extra="forbid")

    region: str = Field(..., description="AWS region for deployment")
    cpu: int = Field(default=1, description="vCPU allocation (1, 2, or 4)")
    memory: int = Field(default=2048, description="Memory allocation in MB")
    ecr_role_arn: str | None = Field(
        default=None, description="IAM role ARN for ECR access"
    )
    health_check_path: str = Field(
        default="/health", description="Path for health checks"
    )
    auto_scaling_min: int = Field(
        default=1, ge=1, le=25, description="Minimum instances"
    )
    auto_scaling_max: int = Field(
        default=5, ge=1, le=25, description="Maximum instances"
    )

    @field_validator("ecr_role_arn")
    @classmethod
    def validate_ecr_role_arn(cls, v: str | None) -> str | None:
        """Validate ECR role ARN format."""
        if v is not None and not AWS_ARN_PATTERN.match(v):
            raise ValueError(
                f"Invalid ECR role ARN: {v}. "
                "Must match pattern: arn:aws:iam::<account-id>:role/<role-name>"
            )
        return v

    @model_validator(mode="after")
    def validate_scaling_range(self) -> "AWSAppRunnerConfig":
        """Validate that auto_scaling_min <= auto_scaling_max."""
        if self.auto_scaling_min > self.auto_scaling_max:
            raise ValueError(
                f"auto_scaling_min ({self.auto_scaling_min}) must be <= "
                f"auto_scaling_max ({self.auto_scaling_max})"
            )
        return self


class GCPCloudRunConfig(BaseModel):
    """GCP Cloud Run deployment configuration.

    Attributes:
        project_id: GCP project ID
        region: GCP region for deployment
        memory: Memory allocation (e.g., 512Mi, 1Gi)
        cpu: vCPU allocation (1, 2, or 4)
        concurrency: Maximum concurrent requests per instance
        timeout: Request timeout in seconds
        min_instances: Minimum instances
        max_instances: Maximum instances
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="GCP project ID")
    region: str = Field(default="us-central1", description="GCP region for deployment")
    memory: str = Field(
        default="512Mi", description="Memory allocation (e.g., 512Mi, 1Gi)"
    )
    cpu: int = Field(default=1, description="vCPU allocation")
    concurrency: int = Field(
        default=80,
        ge=1,
        le=1000,
        description="Maximum concurrent requests per instance",
    )
    timeout: int = Field(
        default=300, ge=1, le=3600, description="Request timeout in seconds"
    )
    min_instances: int = Field(default=0, ge=0, description="Minimum instances")
    max_instances: int = Field(default=100, ge=1, description="Maximum instances")

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Validate GCP project ID format."""
        if not GCP_PROJECT_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid GCP project ID: {v}. "
                "Must be 6-30 lowercase letters, numbers, and hyphens, "
                "starting with a letter and not ending with a hyphen."
            )
        return v

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, v: str) -> str:
        """Validate memory format (e.g., 512Mi, 1Gi)."""
        if not GCP_MEMORY_PATTERN.match(v):
            raise ValueError(
                f"Invalid memory format: {v}. Must be a number followed by Mi or Gi."
            )
        return v


class AzureContainerAppsConfig(BaseModel):
    """Azure Container Apps deployment configuration.

    Attributes:
        subscription_id: Azure subscription ID (UUID)
        resource_group: Azure resource group name
        environment_name: Container Apps environment name
        location: Azure region for deployment
        cpu: vCPU allocation
        memory: Memory allocation (e.g., 2Gi)
        ingress_external: Whether ingress is external
        min_replicas: Minimum replicas
        max_replicas: Maximum replicas
    """

    model_config = ConfigDict(extra="forbid")

    subscription_id: str = Field(..., description="Azure subscription ID (UUID)")
    resource_group: str = Field(..., description="Azure resource group name")
    environment_name: str | None = Field(
        default=None, description="Container Apps environment name"
    )
    location: str = Field(default="eastus", description="Azure region for deployment")
    cpu: float = Field(
        default=0.5,
        description="vCPU allocation (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)",
    )
    memory: str = Field(default="1Gi", description="Memory allocation (e.g., 2Gi)")
    ingress_external: bool = Field(
        default=True, description="Whether ingress is external"
    )
    min_replicas: int = Field(default=0, ge=0, description="Minimum replicas")
    max_replicas: int = Field(default=10, ge=1, description="Maximum replicas")

    @field_validator("subscription_id")
    @classmethod
    def validate_subscription_id(cls, v: str) -> str:
        """Validate Azure subscription ID is a valid UUID."""
        if not AZURE_UUID_PATTERN.match(v):
            raise ValueError(
                f"Invalid Azure subscription ID: {v}. Must be a valid UUID."
            )
        return v

    @field_validator("cpu")
    @classmethod
    def validate_cpu(cls, v: float) -> float:
        """Validate CPU is a valid Azure Container Apps value."""
        valid_cpus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        if v not in valid_cpus:
            raise ValueError(f"Invalid CPU value: {v}. Must be one of: {valid_cpus}")
        return v


class CloudTargetConfig(BaseModel):
    """Cloud deployment target configuration.

    Uses a discriminated union pattern to ensure provider-specific
    configuration matches the selected provider.

    Attributes:
        provider: Cloud provider (aws, gcp, azure)
        aws: AWS App Runner configuration
        gcp: GCP Cloud Run configuration
        azure: Azure Container Apps configuration
    """

    model_config = ConfigDict(extra="forbid")

    provider: CloudProvider = Field(..., description="Cloud provider")
    aws: AWSAppRunnerConfig | None = Field(
        default=None, description="AWS App Runner configuration"
    )
    gcp: GCPCloudRunConfig | None = Field(
        default=None, description="GCP Cloud Run configuration"
    )
    azure: AzureContainerAppsConfig | None = Field(
        default=None, description="Azure Container Apps configuration"
    )

    @model_validator(mode="after")
    def validate_provider_config(self) -> "CloudTargetConfig":
        """Validate that provider-specific config is provided and matches."""
        if self.provider == CloudProvider.AWS:
            if self.aws is None:
                raise ValueError("aws configuration is required when provider is 'aws'")
            if self.gcp is not None or self.azure is not None:
                raise ValueError(
                    "Only aws configuration should be provided when provider is 'aws'"
                )
        elif self.provider == CloudProvider.GCP:
            if self.gcp is None:
                raise ValueError("gcp configuration is required when provider is 'gcp'")
            if self.aws is not None or self.azure is not None:
                raise ValueError(
                    "Only gcp configuration should be provided when provider is 'gcp'"
                )
        elif self.provider == CloudProvider.AZURE:
            if self.azure is None:
                raise ValueError(
                    "azure configuration is required when provider is 'azure'"
                )
            if self.aws is not None or self.gcp is not None:
                raise ValueError(
                    "Only azure config should be set when provider is 'azure'"
                )
        return self


class DeploymentConfig(BaseModel):
    """Main deployment configuration model.

    Attributes:
        runtime: Runtime type (currently only container)
        registry: Container registry configuration
        target: Cloud deployment target configuration
        protocol: API protocol type
        port: Container port to expose
        health_check_path: HTTP path for health checks (e.g., /health, /healthz)
        environment: Environment variables for the container
        platform: Target platform for container image (e.g., linux/amd64, linux/arm64)
    """

    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeType = Field(
        default=RuntimeType.CONTAINER, description="Runtime type"
    )
    registry: RegistryConfig = Field(
        ..., description="Container registry configuration"
    )
    target: CloudTargetConfig = Field(
        ..., description="Cloud deployment target configuration"
    )
    protocol: ProtocolType = Field(
        default=ProtocolType.REST, description="API protocol type"
    )
    port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=8080, description="Container port to expose"
    )
    health_check_path: str = Field(
        default="/health",
        description="HTTP path for health checks (e.g., /health, /healthz)",
    )
    environment: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the container"
    )
    platform: str = Field(
        default="linux/amd64",
        description="Target platform for container image (e.g., linux/amd64)",
    )


class DeployResult(BaseModel):
    """Result of a deployment operation.

    Attributes:
        service_id: Unique identifier for the deployed service (e.g., Azure resource ID)
        service_name: Human-readable name of the deployed service
        url: Public URL where the service is accessible (if available)
        status: Current deployment status (e.g., "Running", "Provisioning", "Failed")
    """

    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(
        ..., description="Unique identifier for the deployed service"
    )
    service_name: str = Field(..., description="Human-readable name of the service")
    url: str | None = Field(default=None, description="Public URL for the service")
    status: str = Field(..., description="Current deployment status")


class StatusResult(BaseModel):
    """Result of a status check operation.

    Attributes:
        status: Current deployment status (e.g., "Running", "Stopped", "Failed")
        url: Public URL where the service is accessible (if available)
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Current deployment status")
    url: str | None = Field(default=None, description="Public URL for the service")
