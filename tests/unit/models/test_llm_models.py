"""Tests for LLM Provider models in holodeck.models.llm."""

import logging

import pytest
from pydantic import ValidationError

from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestLLMProvider:
    """Tests for LLMProvider model."""

    @pytest.mark.parametrize(
        "provider,name,extra_kwargs",
        [
            pytest.param(
                ProviderEnum.OPENAI,
                "gpt-4o",
                {},
                id="openai",
            ),
            pytest.param(
                ProviderEnum.AZURE_OPENAI,
                "gpt-4o",
                {"endpoint": "https://myinstance.openai.azure.com"},
                id="azure_openai",
            ),
            pytest.param(
                ProviderEnum.ANTHROPIC,
                "claude-3-opus",
                {},
                id="anthropic",
            ),
        ],
    )
    def test_llm_provider_valid_creation(
        self, provider: ProviderEnum, name: str, extra_kwargs: dict
    ) -> None:
        """Test creating valid LLMProvider instances for each provider type."""
        llm = LLMProvider(provider=provider, name=name, **extra_kwargs)
        assert llm.provider == provider
        assert llm.name == name

    def test_llm_provider_name_not_empty(self) -> None:
        """Test that name cannot be empty string (custom validator)."""
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

    def test_llm_provider_endpoint_required_for_azure(self) -> None:
        """Test that endpoint is required for Azure OpenAI (custom validator)."""
        with pytest.raises(ValidationError) as exc_info:
            LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4o",
            )
        assert "endpoint" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "provider_enum",
        [
            pytest.param(ProviderEnum.OPENAI, id="openai"),
            pytest.param(ProviderEnum.ANTHROPIC, id="anthropic"),
        ],
    )
    def test_llm_provider_endpoint_not_required_for_non_azure(
        self, provider_enum: ProviderEnum
    ) -> None:
        """Test that endpoint is not required for non-Azure providers."""
        provider = LLMProvider(
            provider=provider_enum,
            name="model-name",
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

    def test_llm_provider_defaults(self) -> None:
        """Test that optional fields have correct defaults."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.temperature == 0.3
        assert provider.max_tokens == 1000
        assert provider.endpoint is None

    def test_custom_auth_without_endpoint_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning emitted when auth_provider=custom but no endpoint set."""
        with caplog.at_level(logging.WARNING, logger="holodeck.models.llm"):
            LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="llama3.1",
                auth_provider=AuthProvider.custom,
            )
        assert "No endpoint is currently configured" in caplog.text

    def test_custom_auth_with_endpoint_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when auth_provider=custom and endpoint is set."""
        with caplog.at_level(logging.WARNING, logger="holodeck.models.llm"):
            LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="llama3.1",
                auth_provider=AuthProvider.custom,
                endpoint="http://localhost:11434/v1",
            )
        assert "No endpoint is currently configured" not in caplog.text
