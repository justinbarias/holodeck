"""LLM provider models for agent configuration.

This module defines the LLMProvider model used in agent.yaml configuration
and global configuration files.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderEnum(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMProvider(BaseModel):
    """LLM provider configuration.

    Specifies which LLM provider and model to use, along with model parameters.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderEnum = Field(..., description="LLM provider")
    name: str = Field(..., description="Model name or identifier")
    temperature: float | None = Field(default=0.3, description="Temperature (0.0-2.0)")
    max_tokens: int | None = Field(
        default=1000, description="Maximum tokens to generate"
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter")
    endpoint: str | None = Field(
        None, description="API endpoint (required for Azure OpenAI and Ollama)"
    )
    api_key: str | None = Field(None, description="API Key for LLM Provider")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float | None) -> float | None:
        """Validate temperature is in valid range."""
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int | None) -> int | None:
        """Validate max_tokens is positive."""
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float | None) -> float | None:
        """Validate top_p is in valid range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def check_endpoint_required(self) -> "LLMProvider":
        """Validate endpoint is provided for Azure OpenAI and Ollama."""
        if self.provider in (ProviderEnum.AZURE_OPENAI,) and (
            not self.endpoint or not self.endpoint.strip()
        ):
            raise ValueError(f"endpoint is required for {self.provider.value} provider")
        return self
