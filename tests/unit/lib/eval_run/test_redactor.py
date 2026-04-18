"""Unit tests for the eval-run redactor (T015).

Two-rule policy from research.md R5:
1. Name allowlist (`api_key`, `password`, `secret`) → `"***"` at any leaf.
2. `SecretStr`-typed field → `"***"` regardless of name.
3. Non-matching fields persisted verbatim.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from holodeck.lib.eval_run.redactor import (
    REDACTED_FIELD_NAMES,
    REDACTED_PLACEHOLDER,
    redact,
)
from holodeck.models.agent import Agent


class _SecretField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    something: SecretStr | None = None
    visible: str = "shown"


@pytest.mark.unit
class TestRedactedFieldNamesConstant:
    def test_is_frozenset(self):
        assert isinstance(REDACTED_FIELD_NAMES, frozenset)

    def test_contents(self):
        assert frozenset({"api_key", "password", "secret"}) == REDACTED_FIELD_NAMES

    def test_placeholder_is_three_stars(self):
        assert REDACTED_PLACEHOLDER == "***"


@pytest.mark.unit
class TestRedactor:
    def test_name_allowlist_redacts_api_key(self):
        agent = Agent(
            name="agent-1",
            model={
                "provider": "openai",
                "name": "gpt-4o",
                "api_key": "sk-fake-not-real",
            },
            instructions={"inline": "hi"},
        )
        redacted = redact(agent)
        # api_key is SecretStr-typed; redactor stores SecretStr(REDACTED_PLACEHOLDER).
        assert redacted.model.api_key is not None
        assert redacted.model.api_key.get_secret_value() == REDACTED_PLACEHOLDER
        # No raw secret is present in any serialised representation.
        assert "sk-fake-not-real" not in redacted.model_dump_json()
        assert "sk-fake-not-real" not in str(redacted.model.api_key)

    def test_non_secret_fields_preserved(self):
        agent = Agent(
            name="agent-1",
            description="hello world",
            model={"provider": "openai", "name": "gpt-4o"},
            instructions={"inline": "hi"},
        )
        redacted = redact(agent)
        assert redacted.model_dump()["description"] == "hello world"
        assert redacted.model_dump()["name"] == "agent-1"

    def test_secretstr_typed_field_redacted_even_without_name_match(self):
        # Build a sub-model where the field name is *not* in the allowlist
        # but its annotation is `SecretStr | None`. The redactor must still
        # mask the value via the type-driven rule.
        instance = _SecretField(something=SecretStr("this-is-secret"))
        redacted = redact(instance)
        assert redacted.something is not None
        assert redacted.something.get_secret_value() == REDACTED_PLACEHOLDER
        assert "this-is-secret" not in redacted.model_dump_json()
        # Visible field preserved.
        assert redacted.visible == "shown"
