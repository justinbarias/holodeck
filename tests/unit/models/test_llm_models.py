"""Tests for LLM Provider models in holodeck.models.llm."""

import pytest
from pydantic import ValidationError

from holodeck.models.llm import LLMProvider, ProviderEnum


class TestLLMProvider:
    """Tests for LLMProvider model."""

    def test_llm_provider_valid_openai(self) -> None:
        """Test creating a valid OpenAI LLMProvider."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.provider == ProviderEnum.OPENAI
        assert provider.name == "gpt-4o"

    def test_llm_provider_valid_azure_openai(self) -> None:
        """Test creating a valid Azure OpenAI LLMProvider."""
        provider = LLMProvider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="gpt-4o",
            endpoint="https://myinstance.openai.azure.com",
        )
        assert provider.provider == ProviderEnum.AZURE_OPENAI
        assert provider.endpoint == "https://myinstance.openai.azure.com"

    def test_llm_provider_valid_anthropic(self) -> None:
        """Test creating a valid Anthropic LLMProvider."""
        provider = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-opus",
        )
        assert provider.provider == ProviderEnum.ANTHROPIC
        assert provider.name == "claude-3-opus"

    def test_llm_provider_field_required(self) -> None:
        """Test that provider field is required."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvider(name="gpt-4o")
        assert "provider" in str(exc_info.value).lower()

    def test_llm_provider_name_required(self) -> None:
        """Test that name field is required."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvider(provider=ProviderEnum.OPENAI)
        assert "name" in str(exc_info.value).lower()

    def test_llm_provider_name_not_empty(self) -> None:
        """Test that name cannot be empty string."""
        with pytest.raises(ValidationError):
            LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="",
            )

    @pytest.mark.parametrize(
        "temperature,should_pass",
        [
            (0.0, True),
            (0.7, True),
            (2.0, True),
            (-0.1, False),
            (2.1, False),
        ],
        ids=["valid_low", "valid_mid", "valid_high", "below_min", "above_max"],
    )
    def test_llm_provider_temperature_range_validation(
        self, temperature: float, should_pass: bool
    ) -> None:
        """Test temperature validation for all boundary conditions."""
        if should_pass:
            provider = LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                temperature=temperature,
            )
            assert provider.temperature == temperature
        else:
            with pytest.raises(ValidationError) as exc_info:
                LLMProvider(
                    provider=ProviderEnum.OPENAI,
                    name="gpt-4o",
                    temperature=temperature,
                )
            assert "temperature" in str(exc_info.value).lower()

    def test_llm_provider_temperature_optional(self) -> None:
        """Test that temperature is optional."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.temperature == 0.3  # the default temperature

    @pytest.mark.parametrize(
        "max_tokens,should_pass",
        [
            (2000, True),
            (1, True),
            (0, False),
            (-100, False),
        ],
        ids=["valid_positive", "valid_one", "zero_invalid", "negative_invalid"],
    )
    def test_llm_provider_max_tokens_validation(
        self, max_tokens: int, should_pass: bool
    ) -> None:
        """Test max_tokens validation for all boundary conditions."""
        if should_pass:
            provider = LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                max_tokens=max_tokens,
            )
            assert provider.max_tokens == max_tokens
        else:
            with pytest.raises(ValidationError) as exc_info:
                LLMProvider(
                    provider=ProviderEnum.OPENAI,
                    name="gpt-4o",
                    max_tokens=max_tokens,
                )
            assert "max_tokens" in str(exc_info.value).lower()

    def test_llm_provider_max_tokens_optional(self) -> None:
        """Test that max_tokens is optional."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.max_tokens == 1000  # default max_tokens

    def test_llm_provider_endpoint_required_for_azure(self) -> None:
        """Test that endpoint is required for Azure OpenAI."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4o",
            )
        assert "endpoint" in str(exc_info.value).lower()

    def test_llm_provider_endpoint_not_required_for_openai(self) -> None:
        """Test that endpoint is not required for standard OpenAI."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.endpoint is None

    def test_llm_provider_endpoint_not_required_for_anthropic(self) -> None:
        """Test that endpoint is not required for Anthropic."""
        provider = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-opus",
        )
        assert provider.endpoint is None

    def test_llm_provider_invalid_provider(self) -> None:
        """Test that invalid provider is rejected."""
        with pytest.raises(ValidationError):
            LLMProvider(
                provider="invalid_provider",  # type: ignore
                name="gpt-4o",
            )

    def test_llm_provider_all_fields(self) -> None:
        """Test LLMProvider with all optional fields."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
        )
        assert provider.temperature == 0.7
        assert provider.max_tokens == 2000
        assert provider.top_p == 0.9
