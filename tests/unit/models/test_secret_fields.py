"""Type-driven tests for SecretStr-bearing fields.

These tests assert that every secret-bearing field in the HoloDeck model tree
is declared as `SecretStr | None` (or `SecretStr`), enabling the eval-run
redactor's type-driven rule to cover them automatically.

Scope (per `specs/031-eval-runs-dashboard/tasks-us1.md` T003-T004):
- Fields that EXIST in the current codebase at feature-start.
- Anthropic `auth_token`, AWS creds, and similar do NOT exist as fields yet
  and are out-of-scope for this migration — any future provider secret must
  be typed `SecretStr` from day one per the invariant documented in
  `src/holodeck/models/llm.py`.
"""

from __future__ import annotations

import types
import typing
from typing import get_args, get_origin

import pytest
from pydantic import SecretStr

from holodeck.models.llm import LLMProvider
from holodeck.models.observability import AzureMonitorExporterConfig
from holodeck.models.tool import DatabaseConfig, KeywordIndexConfig


def _is_optional_secret_str(annotation: typing.Any) -> bool:
    """Return True if annotation is `SecretStr | None` or `Optional[SecretStr]`."""
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = get_args(annotation)
        return SecretStr in args and type(None) in args
    return annotation is SecretStr


@pytest.mark.unit
class TestLLMProviderSecrets:
    """LLMProvider.api_key must be SecretStr | None."""

    def test_api_key_annotation_is_secret_str(self) -> None:
        annotation = LLMProvider.model_fields["api_key"].annotation
        assert _is_optional_secret_str(
            annotation
        ), f"LLMProvider.api_key must be SecretStr | None, got {annotation!r}"


@pytest.mark.unit
class TestDatabaseConfigSecrets:
    """DatabaseConfig.connection_string must be SecretStr | None."""

    def test_connection_string_annotation_is_secret_str(self) -> None:
        annotation = DatabaseConfig.model_fields["connection_string"].annotation
        assert _is_optional_secret_str(annotation), (
            "DatabaseConfig.connection_string must be SecretStr | None, "
            f"got {annotation!r}"
        )


@pytest.mark.unit
class TestKeywordIndexConfigSecrets:
    """KeywordIndexConfig password and api_key must be SecretStr | None."""

    def test_password_annotation_is_secret_str(self) -> None:
        annotation = KeywordIndexConfig.model_fields["password"].annotation
        assert _is_optional_secret_str(
            annotation
        ), f"KeywordIndexConfig.password must be SecretStr | None, got {annotation!r}"

    def test_api_key_annotation_is_secret_str(self) -> None:
        annotation = KeywordIndexConfig.model_fields["api_key"].annotation
        assert _is_optional_secret_str(
            annotation
        ), f"KeywordIndexConfig.api_key must be SecretStr | None, got {annotation!r}"


@pytest.mark.unit
class TestAzureMonitorExporterConfigSecrets:
    """AzureMonitorExporterConfig.connection_string must be SecretStr | None."""

    def test_connection_string_annotation_is_secret_str(self) -> None:
        annotation = AzureMonitorExporterConfig.model_fields[
            "connection_string"
        ].annotation
        assert _is_optional_secret_str(annotation), (
            "AzureMonitorExporterConfig.connection_string must be "
            f"SecretStr | None, got {annotation!r}"
        )


@pytest.mark.unit
class TestSecretStrRoundTrip:
    """Secret values round-trip through model_validate without coercion surprises."""

    def test_llm_provider_accepts_secret_str(self) -> None:
        provider = LLMProvider(
            provider="openai",
            name="gpt-4o",
            api_key=SecretStr("sk-test-value"),
        )
        assert isinstance(provider.api_key, SecretStr)
        assert provider.api_key.get_secret_value() == "sk-test-value"

    def test_llm_provider_accepts_plain_str_input(self) -> None:
        """Pydantic coerces plain str → SecretStr transparently on construction."""
        provider = LLMProvider(
            provider="openai",
            name="gpt-4o",
            api_key="sk-test-value",
        )
        assert isinstance(provider.api_key, SecretStr)
        assert provider.api_key.get_secret_value() == "sk-test-value"

    def test_llm_provider_dump_masks_secret(self) -> None:
        """model_dump() produces SecretStr('**********') — never the raw value."""
        provider = LLMProvider(
            provider="openai",
            name="gpt-4o",
            api_key="sk-test-value",
        )
        dumped = provider.model_dump()
        assert "sk-test-value" not in str(
            dumped
        ), f"raw secret leaked in model_dump(): {dumped!r}"
