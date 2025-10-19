"""Global configuration models.

This module defines global configuration models stored in ~/.holodeck/config.yaml
for sharing default settings across multiple agents.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.models.llm import LLMProvider


class VectorstoreConfig(BaseModel):
    """Vectorstore configuration for global defaults.

    Specifies connection details and options for a specific vectorstore backend.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        ..., description="Vectorstore provider (postgres, redis, etc.)"
    )
    connection_string: str = Field(
        ..., description="Connection string for the vectorstore"
    )
    options: dict[str, Any] | None = Field(
        None, description="Provider-specific options"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is not empty."""
        if not v or not v.strip():
            raise ValueError("provider must be a non-empty string")
        return v

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate connection_string is not empty."""
        if not v or not v.strip():
            raise ValueError("connection_string must be a non-empty string")
        return v


class DeploymentConfig(BaseModel):
    """Deployment configuration for global defaults.

    Specifies deployment platform and settings for deploying agents.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Deployment type (docker, kubernetes, etc.)")
    settings: dict[str, Any] | None = Field(
        None, description="Deployment-specific settings"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate type is not empty."""
        if not v or not v.strip():
            raise ValueError("type must be a non-empty string")
        return v


class GlobalConfig(BaseModel):
    """Global configuration entity.

    Configuration stored in ~/.holodeck/config.yaml for sharing defaults
    across multiple agents, including LLM providers, vectorstores, and
    deployment settings.
    """

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, LLMProvider] | None = Field(
        None, description="Named LLM provider configurations"
    )
    vectorstores: dict[str, VectorstoreConfig] | None = Field(
        None, description="Named vectorstore configurations"
    )
    deployment: DeploymentConfig | None = Field(
        None, description="Deployment configuration"
    )
