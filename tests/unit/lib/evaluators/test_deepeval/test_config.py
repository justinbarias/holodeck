"""Unit tests for DeepEval model configuration."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from holodeck.lib.evaluators.deepeval.config import (
    DEFAULT_MODEL_CONFIG,
    DeepEvalModelConfig,
)
from holodeck.models.llm import ProviderEnum


class TestDeepEvalModelConfigDefaults:
    """Tests for default configuration values."""

    def test_default_provider_is_ollama(self) -> None:
        """Default provider should be Ollama."""
        config = DeepEvalModelConfig()
        assert config.provider == ProviderEnum.OLLAMA

    def test_default_model_name(self) -> None:
        """Default model should be gpt-oss:20b."""
        config = DeepEvalModelConfig()
        assert config.model_name == "gpt-oss:20b"

    def test_default_temperature_is_zero(self) -> None:
        """Default temperature should be 0.0 for deterministic evaluation."""
        config = DeepEvalModelConfig()
        assert config.temperature == 0.0

    def test_default_api_key_is_none(self) -> None:
        """Default API key should be None (not required for Ollama)."""
        config = DeepEvalModelConfig()
        assert config.api_key is None

    def test_default_config_constant(self) -> None:
        """DEFAULT_MODEL_CONFIG should have correct default values."""
        assert DEFAULT_MODEL_CONFIG.provider == ProviderEnum.OLLAMA
        assert DEFAULT_MODEL_CONFIG.model_name == "gpt-oss:20b"
        assert DEFAULT_MODEL_CONFIG.temperature == 0.0


class TestDeepEvalModelConfigValidation:
    """Tests for configuration validation rules."""

    def test_temperature_min_boundary(self) -> None:
        """Temperature at minimum boundary (0.0) should be valid."""
        config = DeepEvalModelConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_max_boundary(self) -> None:
        """Temperature at maximum boundary (2.0) should be valid."""
        config = DeepEvalModelConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_temperature_below_min_raises_error(self) -> None:
        """Temperature below 0.0 should raise validation error."""
        with pytest.raises(ValidationError):
            DeepEvalModelConfig(temperature=-0.1)

    def test_temperature_above_max_raises_error(self) -> None:
        """Temperature above 2.0 should raise validation error."""
        with pytest.raises(ValidationError):
            DeepEvalModelConfig(temperature=2.1)

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise validation error."""
        with pytest.raises(ValidationError):
            DeepEvalModelConfig(unknown_field="value")


class TestAzureOpenAIValidation:
    """Tests for Azure OpenAI provider validation."""

    def test_azure_requires_endpoint(self) -> None:
        """Azure OpenAI requires endpoint field."""
        with pytest.raises(ValueError, match="endpoint is required"):
            DeepEvalModelConfig(
                provider=ProviderEnum.AZURE_OPENAI,
                model_name="gpt-4o",
                deployment_name="my-deployment",
                api_key="test-key",
            )

    def test_azure_requires_deployment_name(self) -> None:
        """Azure OpenAI requires deployment_name field."""
        with pytest.raises(ValueError, match="deployment_name is required"):
            DeepEvalModelConfig(
                provider=ProviderEnum.AZURE_OPENAI,
                model_name="gpt-4o",
                endpoint="https://test.openai.azure.com/",
                api_key="test-key",
            )

    def test_azure_requires_api_key(self) -> None:
        """Azure OpenAI requires api_key field."""
        with pytest.raises(ValueError, match="api_key is required"):
            DeepEvalModelConfig(
                provider=ProviderEnum.AZURE_OPENAI,
                model_name="gpt-4o",
                endpoint="https://test.openai.azure.com/",
                deployment_name="my-deployment",
            )

    def test_azure_valid_config(self) -> None:
        """Azure OpenAI with all required fields should be valid."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.AZURE_OPENAI,
            model_name="gpt-4o",
            endpoint="https://test.openai.azure.com/",
            deployment_name="my-deployment",
            api_key="test-key",
        )
        assert config.provider == ProviderEnum.AZURE_OPENAI
        assert config.endpoint == "https://test.openai.azure.com/"


class TestOpenAIConfig:
    """Tests for OpenAI provider configuration."""

    def test_openai_config_valid_without_api_key(self) -> None:
        """OpenAI config without api_key should be valid (can use env var)."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )
        assert config.provider == ProviderEnum.OPENAI
        assert config.model_name == "gpt-4o"

    def test_openai_config_with_api_key(self) -> None:
        """OpenAI config with api_key should be valid."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
            api_key="sk-test-key",
        )
        assert config.api_key == "sk-test-key"


class TestAnthropicConfig:
    """Tests for Anthropic provider configuration."""

    def test_anthropic_config_valid_without_api_key(self) -> None:
        """Anthropic config without api_key should be valid (can use env var)."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
        )
        assert config.provider == ProviderEnum.ANTHROPIC
        assert config.model_name == "claude-3-5-sonnet-latest"

    def test_anthropic_config_with_api_key(self) -> None:
        """Anthropic config with api_key should be valid."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-opus-latest",
            api_key="sk-ant-test-key",
        )
        assert config.api_key == "sk-ant-test-key"


class TestOllamaConfig:
    """Tests for Ollama provider configuration."""

    def test_ollama_default_endpoint(self) -> None:
        """Ollama should use localhost:11434 by default."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OLLAMA,
            model_name="llama3",
        )
        assert config.endpoint is None  # Default applied in to_deepeval_model()

    def test_ollama_custom_endpoint(self) -> None:
        """Ollama should accept custom endpoint."""
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OLLAMA,
            model_name="llama3",
            endpoint="http://remote-ollama:11434",
        )
        assert config.endpoint == "http://remote-ollama:11434"


class TestToDeepEvalModel:
    """Tests for to_deepeval_model() conversion method."""

    @patch("deepeval.models.OllamaModel")
    def test_ollama_model_creation(self, mock_ollama_model: MagicMock) -> None:
        """to_deepeval_model() should create OllamaModel for Ollama provider."""
        mock_ollama_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OLLAMA,
            model_name="llama3",
            temperature=0.5,
        )

        config.to_deepeval_model()

        mock_ollama_model.assert_called_once_with(
            model="llama3",
            base_url="http://localhost:11434",
            temperature=0.5,
        )

    @patch("deepeval.models.OllamaModel")
    def test_ollama_custom_endpoint(self, mock_ollama_model: MagicMock) -> None:
        """to_deepeval_model() should use custom endpoint for Ollama."""
        mock_ollama_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OLLAMA,
            model_name="llama3",
            endpoint="http://custom:11434",
        )

        config.to_deepeval_model()

        mock_ollama_model.assert_called_once_with(
            model="llama3",
            base_url="http://custom:11434",
            temperature=0.0,
        )

    @patch("deepeval.models.GPTModel")
    def test_openai_model_creation(self, mock_gpt_model: MagicMock) -> None:
        """to_deepeval_model() should create GPTModel for OpenAI provider."""
        mock_gpt_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
            api_key="sk-test",
            temperature=0.3,
        )

        config.to_deepeval_model()

        mock_gpt_model.assert_called_once_with(
            model="gpt-4o",
            temperature=0.3,
            api_key="sk-test",
        )

    @patch("deepeval.models.GPTModel")
    def test_openai_without_api_key(self, mock_gpt_model: MagicMock) -> None:
        """to_deepeval_model() should work without api_key (uses env var)."""
        mock_gpt_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.OPENAI,
            model_name="gpt-4o",
        )

        config.to_deepeval_model()

        mock_gpt_model.assert_called_once_with(
            model="gpt-4o",
            temperature=0.0,
        )

    @patch("deepeval.models.AzureOpenAIModel")
    def test_azure_model_creation(self, mock_azure_model: MagicMock) -> None:
        """to_deepeval_model() should create AzureOpenAIModel for Azure provider."""
        mock_azure_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.AZURE_OPENAI,
            model_name="gpt-4o",
            endpoint="https://test.openai.azure.com/",
            deployment_name="my-deployment",
            api_key="azure-key",
            api_version="2024-02-15-preview",
            temperature=0.1,  # ignored for Azure - uses 1.0 for reasoning models
        )

        config.to_deepeval_model()

        # Azure always uses temperature=1.0 for reasoning model compatibility
        mock_azure_model.assert_called_once_with(
            model_name="gpt-4o",
            deployment_name="my-deployment",
            azure_endpoint="https://test.openai.azure.com/",
            openai_api_version="2024-02-15-preview",
            azure_openai_api_key="azure-key",
            temperature=1.0,
        )

    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_model_creation(self, mock_anthropic_model: MagicMock) -> None:
        """to_deepeval_model() should create AnthropicModel for Anthropic provider."""
        mock_anthropic_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-5-sonnet-latest",
            api_key="sk-ant-key",
            temperature=0.2,
        )

        config.to_deepeval_model()

        mock_anthropic_model.assert_called_once_with(
            model="claude-3-5-sonnet-latest",
            temperature=0.2,
            api_key="sk-ant-key",
        )

    @patch("deepeval.models.AnthropicModel")
    def test_anthropic_without_api_key(self, mock_anthropic_model: MagicMock) -> None:
        """to_deepeval_model() should work without api_key (uses env var)."""
        mock_anthropic_model.return_value = MagicMock()
        config = DeepEvalModelConfig(
            provider=ProviderEnum.ANTHROPIC,
            model_name="claude-3-opus-latest",
        )

        config.to_deepeval_model()

        mock_anthropic_model.assert_called_once_with(
            model="claude-3-opus-latest",
            temperature=0.0,
        )
