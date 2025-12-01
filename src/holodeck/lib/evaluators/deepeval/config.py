"""Configuration models for DeepEval evaluators.

This module provides the DeepEvalModelConfig class that adapts HoloDeck's
LLM provider configuration to DeepEval's native model classes.
"""

from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from holodeck.models.llm import ProviderEnum

if TYPE_CHECKING:
    from deepeval.models import AnthropicModel, AzureOpenAIModel, GPTModel, OllamaModel

# Type alias for DeepEval model return types
DeepEvalModel = Union["GPTModel", "AzureOpenAIModel", "AnthropicModel", "OllamaModel"]


class DeepEvalModelConfig(BaseModel):
    """Configuration adapter for DeepEval model classes.

    This class bridges HoloDeck's LLMProvider configuration to DeepEval's
    native model classes (GPTModel, AzureOpenAIModel, AnthropicModel, OllamaModel).

    The default configuration uses Ollama with gpt-oss:20b for local evaluation
    without requiring API keys.

    Attributes:
        provider: LLM provider to use (defaults to Ollama)
        model_name: Name of the model (defaults to gpt-oss:20b)
        api_key: API key for cloud providers (not required for Ollama)
        endpoint: API endpoint URL (required for Azure, optional for Ollama)
        api_version: Azure OpenAI API version
        deployment_name: Azure OpenAI deployment name
        temperature: Temperature for generation (defaults to 0.0 for determinism)

    API Key Behavior:
        - **OpenAI**: API key can be provided via `api_key` field or the
          `OPENAI_API_KEY` environment variable. If neither is set, DeepEval's
          GPTModel will raise an error at runtime.
        - **Anthropic**: API key can be provided via `api_key` field or the
          `ANTHROPIC_API_KEY` environment variable. If neither is set,
          DeepEval's AnthropicModel will raise an error at runtime.
        - **Azure OpenAI**: The `api_key` field is **required** and validated
          at configuration time (no environment variable fallback).
        - **Ollama**: No API key required (local inference).

    Example:
        >>> config = DeepEvalModelConfig()  # Default Ollama
        >>> model = config.to_deepeval_model()

        >>> openai_config = DeepEvalModelConfig(
        ...     provider=ProviderEnum.OPENAI,
        ...     model_name="gpt-4o",
        ...     api_key="sk-..."  # Or set OPENAI_API_KEY env var
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderEnum = Field(
        default=ProviderEnum.OLLAMA,
        description="LLM provider (openai, azure_openai, anthropic, ollama)",
    )
    model_name: str = Field(
        default="gpt-oss:20b",
        description="Model name or identifier",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for cloud providers",
    )
    endpoint: str | None = Field(
        default=None,
        description="API endpoint URL (required for Azure OpenAI)",
    )
    api_version: str | None = Field(
        default="2025-04-01-preview",
        description="Azure OpenAI API version",
    )
    deployment_name: str | None = Field(
        default=None,
        description="Azure OpenAI deployment name",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature (0.0 for deterministic evaluation)",
    )

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "DeepEvalModelConfig":
        """Validate that required fields are present for each provider.

        Raises:
            ValueError: If required fields are missing for the provider
        """
        if self.provider == ProviderEnum.AZURE_OPENAI:
            if not self.endpoint:
                raise ValueError("endpoint is required for Azure OpenAI provider")
            if not self.deployment_name:
                raise ValueError(
                    "deployment_name is required for Azure OpenAI provider"
                )
            if not self.api_key:
                raise ValueError("api_key is required for Azure OpenAI provider")
        return self

    def to_deepeval_model(self) -> DeepEvalModel:
        """Convert configuration to native DeepEval model class.

        Returns the appropriate DeepEval model class instance based on
        the configured provider.

        Returns:
            DeepEval model instance (GPTModel, AzureOpenAIModel,
            AnthropicModel, or OllamaModel)

        Raises:
            ValueError: If provider is not supported
        """
        if self.provider == ProviderEnum.OPENAI:
            from deepeval.models import GPTModel

            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "temperature": self.temperature,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            return GPTModel(**kwargs)

        elif self.provider == ProviderEnum.AZURE_OPENAI:
            from deepeval.models import AzureOpenAIModel

            return AzureOpenAIModel(
                model_name=self.model_name,
                deployment_name=self.deployment_name,
                azure_endpoint=self.endpoint,
                openai_api_version=self.api_version,
                azure_openai_api_key=self.api_key,
                temperature=1.0,  # reasoning models require temperature=1.0
            )

        elif self.provider == ProviderEnum.ANTHROPIC:
            from deepeval.models import AnthropicModel

            kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            return AnthropicModel(**kwargs)

        elif self.provider == ProviderEnum.OLLAMA:
            from deepeval.models import OllamaModel

            return OllamaModel(
                model=self.model_name,
                base_url=self.endpoint or "http://localhost:11434",
                temperature=self.temperature,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


# Default configuration for convenience
DEFAULT_MODEL_CONFIG = DeepEvalModelConfig()
