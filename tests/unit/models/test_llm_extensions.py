"""Tests for LLMProvider extensions in holodeck.models.llm (auth_provider field)."""

import logging

import pytest
from pydantic import ValidationError

from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestLLMProviderAuthProvider:
    """Tests for the auth_provider field on LLMProvider."""

    def test_auth_provider_defaults_to_none(self) -> None:
        """Test that auth_provider defaults to None."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        assert provider.auth_provider is None

    @pytest.mark.parametrize(
        "auth",
        [
            AuthProvider.api_key,
            AuthProvider.oauth_token,
            AuthProvider.bedrock,
            AuthProvider.vertex,
            AuthProvider.foundry,
        ],
        ids=["api_key", "oauth_token", "bedrock", "vertex", "foundry"],
    )
    def test_auth_provider_accepts_valid_values_for_anthropic(
        self, auth: AuthProvider
    ) -> None:
        """Test that auth_provider accepts all valid enum values for Anthropic."""
        provider = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-sonnet-4-20250514",
            auth_provider=auth,
        )
        assert provider.auth_provider == auth

    def test_auth_provider_invalid_value_rejected(self) -> None:
        """Test that invalid auth_provider value is rejected."""
        with pytest.raises(ValidationError):
            LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="claude-sonnet-4-20250514",
                auth_provider="invalid_auth",
            )

    def test_auth_provider_warning_for_non_anthropic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning when auth_provider set for non-Anthropic."""
        with caplog.at_level(logging.WARNING):
            provider = LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                auth_provider=AuthProvider.api_key,
            )

        assert provider.auth_provider == AuthProvider.api_key
        assert len(caplog.records) >= 1
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("auth_provider" in msg for msg in warning_messages)
        assert any("openai" in msg for msg in warning_messages)

    def test_no_warning_for_anthropic_provider(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test no warning when auth_provider set for Anthropic."""
        with caplog.at_level(logging.WARNING):
            LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="claude-sonnet-4-20250514",
                auth_provider=AuthProvider.api_key,
            )

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert not any("auth_provider" in msg for msg in warning_messages)

    @pytest.mark.parametrize(
        "non_anthropic_provider",
        [ProviderEnum.OPENAI, ProviderEnum.AZURE_OPENAI, ProviderEnum.OLLAMA],
        ids=["openai", "azure_openai", "ollama"],
    )
    def test_auth_provider_warning_all_non_anthropic(
        self,
        non_anthropic_provider: ProviderEnum,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning for all non-Anthropic providers."""
        kwargs: dict = {
            "provider": non_anthropic_provider,
            "name": "some-model",
            "auth_provider": AuthProvider.bedrock,
        }
        # Azure requires endpoint
        if non_anthropic_provider == ProviderEnum.AZURE_OPENAI:
            kwargs["endpoint"] = "https://test.openai.azure.com"
        # Ollama doesn't require endpoint per the model validator
        with caplog.at_level(logging.WARNING):
            LLMProvider(**kwargs)

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("auth_provider" in msg for msg in warning_messages)
