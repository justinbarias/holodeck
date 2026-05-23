"""Tests for HoloDeck-provided default hooks (spec 034 P2b)."""

from __future__ import annotations

import pytest

from holodeck.lib.backends.claude_hooks import (
    CREDENTIAL_PATTERNS,
    build_default_hooks,
    redact_credentials,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected_marker",
    [
        ("auth=sk-ant-api03-" + "a" * 95, "[REDACTED:anthropic-key]"),
        ("key=AKIA" + "B" * 16, "[REDACTED:aws-access-key]"),
        ("token: ghp_" + "c" * 36, "[REDACTED:github-token]"),
        (
            "jwt: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signaturepart",
            "[REDACTED:jwt]",
        ),
        ("Authorization: Bearer abc.def-123_xyz", "Bearer [REDACTED]"),
    ],
)
def test_redact_credentials_replaces_known_shapes(
    raw: str, expected_marker: str
) -> None:
    """Each credential shape is replaced by its marker."""
    redacted = redact_credentials(raw)
    assert expected_marker in redacted


@pytest.mark.unit
def test_redact_credentials_leaves_clean_text_unchanged() -> None:
    """No false positives on plain output."""
    text = "Retrieved 42 rows from table customers"
    assert redact_credentials(text) == text


@pytest.mark.unit
def test_redact_credentials_handles_nested_structures() -> None:
    """Redacts strings inside dicts/lists recursively."""
    payload = {
        "auth": {"token": "ghp_" + "x" * 36},
        "rows": ["AKIA" + "Y" * 16, "ok"],
    }
    redacted = redact_credentials(payload)
    assert "[REDACTED:github-token]" in redacted["auth"]["token"]
    assert "[REDACTED:aws-access-key]" in redacted["rows"][0]
    assert redacted["rows"][1] == "ok"


@pytest.mark.unit
def test_credential_patterns_documented_count() -> None:
    """Five credential shapes per spec 034 §'Default hook 2'."""
    assert len(CREDENTIAL_PATTERNS) == 5


@pytest.mark.unit
def test_build_default_hooks_returns_post_tool_use_only() -> None:
    """build_default_hooks wires only the PostToolUse credential hook.

    Bash deny hook removed 2026-05-23 — see Task 2 note. Bash hardening
    is handled by SDK permission rules + P1b auto-disallow.
    """
    hooks = build_default_hooks()
    assert "PreToolUse" not in hooks
    assert "PostToolUse" in hooks
    assert len(hooks["PostToolUse"]) == 1
