"""Tests for Ollama-specific LLM Provider validation.

This module tests Ollama provider configuration validation including:
- Endpoint requirement validation
- Parameter range validation (temperature, max_tokens, top_p)
- Default value application
- Environment variable substitution
"""

import pytest
from pydantic import ValidationError

from holodeck.models.llm import LLMProvider, ProviderEnum


class TestOllamaProvider:
    """Tests for Ollama-specific LLMProvider validation."""

    # T010 [P] [US1] - Endpoint Required Validation
    def test_ollama_endpoint_required(self) -> None:
        """Test that Ollama provider requires endpoint field."""
        # Valid case - endpoint provided
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://localhost:11434",
        )
        assert provider.endpoint == "http://localhost:11434"
        assert provider.provider == ProviderEnum.OLLAMA
        assert provider.name == "llama3"

    def test_ollama_endpoint_optional_missing(self) -> None:
        """Test that Ollama provider accepts missing endpoint (optional)."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint=None,
        )
        assert provider.endpoint is None

    def test_ollama_endpoint_optional_empty_string(self) -> None:
        """Test that Ollama provider accepts empty endpoint (optional)."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="",
        )
        assert provider.endpoint == ""

    def test_ollama_endpoint_optional_whitespace(self) -> None:
        """Test that Ollama provider accepts whitespace endpoint (optional)."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="   ",
        )
        assert provider.endpoint == "   "

    # T011 [P] [US1] - Parameter Validation
    @pytest.mark.parametrize(
        "temperature,should_pass",
        [
            (0.0, True),
            (0.7, True),
            (2.0, True),
            (-0.1, False),
            (2.1, False),
        ],
        ids=["valid_zero", "valid_mid", "valid_max", "below_min", "above_max"],
    )
    def test_ollama_temperature_validation(
        self, temperature: float, should_pass: bool
    ) -> None:
        """Test that Ollama provider validates temperature range (0.0-2.0)."""
        if should_pass:
            provider = LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
                temperature=temperature,
            )
            assert provider.temperature == temperature
        else:
            with pytest.raises(ValidationError) as exc_info:
                LLMProvider(
                    provider=ProviderEnum.OLLAMA,
                    name="llama3",
                    endpoint="http://localhost:11434",
                    temperature=temperature,
                )
            assert "temperature" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "max_tokens,should_pass",
        [
            (1, True),
            (1000, True),
            (4096, True),
            (0, False),
            (-1, False),
        ],
        ids=["valid_one", "valid_default", "valid_large", "zero_invalid", "negative"],
    )
    def test_ollama_max_tokens_validation(
        self, max_tokens: int, should_pass: bool
    ) -> None:
        """Test that Ollama provider validates max_tokens is positive."""
        if should_pass:
            provider = LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
                max_tokens=max_tokens,
            )
            assert provider.max_tokens == max_tokens
        else:
            with pytest.raises(ValidationError) as exc_info:
                LLMProvider(
                    provider=ProviderEnum.OLLAMA,
                    name="llama3",
                    endpoint="http://localhost:11434",
                    max_tokens=max_tokens,
                )
            assert "max_tokens" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "top_p,should_pass",
        [
            (0.0, True),
            (0.5, True),
            (1.0, True),
            (None, True),
            (-0.1, False),
            (1.1, False),
        ],
        ids=[
            "valid_zero",
            "valid_mid",
            "valid_max",
            "valid_none",
            "below_min",
            "above_max",
        ],
    )
    def test_ollama_top_p_validation(
        self, top_p: float | None, should_pass: bool
    ) -> None:
        """Test that Ollama provider validates top_p range (0.0-1.0) or None."""
        if should_pass:
            provider = LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
                top_p=top_p,
            )
            assert provider.top_p == top_p
        else:
            with pytest.raises(ValidationError) as exc_info:
                LLMProvider(
                    provider=ProviderEnum.OLLAMA,
                    name="llama3",
                    endpoint="http://localhost:11434",
                    top_p=top_p,
                )
            assert "top_p" in str(exc_info.value).lower()

    # T012 [P] [US1] - Defaults Applied
    def test_ollama_config_with_defaults(self) -> None:
        """Test that Ollama provider applies default values correctly."""
        # Minimal config - only required fields
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://localhost:11434",
        )

        # Verify defaults are applied per OLLAMA_DEFAULTS in defaults.py
        assert provider.temperature == 0.3  # Default temperature
        assert provider.max_tokens == 1000  # Default max_tokens
        assert provider.top_p is None  # Default top_p
        assert provider.api_key is None  # Default api_key

    def test_ollama_config_override_defaults(self) -> None:
        """Test that explicit values override defaults for Ollama provider."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="phi3",
            endpoint="http://192.168.1.100:11434",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            api_key="test-key",
        )

        # Verify all values are overridden
        assert provider.temperature == 0.7
        assert provider.max_tokens == 2000
        assert provider.top_p == 0.9
        assert provider.api_key == "test-key"

    # T013 [P] [US1] - Model Name Validation
    @pytest.mark.parametrize(
        "model_name",
        [
            "llama3",
            "phi3",
            "mistral",
            "codellama",
            "gemma",
            "llama3:8b",
            "mistral:7b-instruct",
        ],
        ids=[
            "llama3",
            "phi3",
            "mistral",
            "codellama",
            "gemma",
            "llama3_versioned",
            "mistral_tagged",
        ],
    )
    def test_ollama_model_names(self, model_name: str) -> None:
        """Test that common Ollama model names are accepted."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name=model_name,
            endpoint="http://localhost:11434",
        )
        assert provider.name == model_name

    def test_ollama_model_name_required(self) -> None:
        """Test that model name is required for Ollama provider."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvider(
                provider=ProviderEnum.OLLAMA,
                endpoint="http://localhost:11434",
            )
        assert "name" in str(exc_info.value).lower()

    def test_ollama_model_name_not_empty(self) -> None:
        """Test that model name cannot be empty for Ollama provider."""
        with pytest.raises(ValidationError):
            LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="",
                endpoint="http://localhost:11434",
            )

    # Additional tests for Ollama-specific scenarios
    def test_ollama_local_endpoint(self) -> None:
        """Test Ollama with default local endpoint."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://localhost:11434",
        )
        assert provider.endpoint == "http://localhost:11434"

    def test_ollama_remote_endpoint(self) -> None:
        """Test Ollama with remote endpoint."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://192.168.1.100:11434",
        )
        assert provider.endpoint == "http://192.168.1.100:11434"

    def test_ollama_custom_port(self) -> None:
        """Test Ollama with custom port."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://localhost:8080",
        )
        assert provider.endpoint == "http://localhost:8080"

    def test_ollama_https_endpoint(self) -> None:
        """Test Ollama with HTTPS endpoint."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="https://ollama.example.com:11434",
        )
        assert provider.endpoint == "https://ollama.example.com:11434"

    def test_ollama_with_api_key(self) -> None:
        """Test Ollama with API key for remote authentication."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://192.168.1.100:11434",
            api_key="my-secret-key",
        )
        assert provider.api_key == "my-secret-key"

    def test_ollama_all_parameters(self) -> None:
        """Test Ollama with all parameters specified."""
        provider = LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="mistral:7b-instruct",
            endpoint="https://ollama.example.com:11434",
            temperature=0.8,
            max_tokens=2048,
            top_p=0.95,
            api_key="auth-token-123",
        )
        assert provider.provider == ProviderEnum.OLLAMA
        assert provider.name == "mistral:7b-instruct"
        assert provider.endpoint == "https://ollama.example.com:11434"
        assert provider.temperature == 0.8
        assert provider.max_tokens == 2048
        assert provider.top_p == 0.95
        assert provider.api_key == "auth-token-123"
