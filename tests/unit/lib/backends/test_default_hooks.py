"""Tests for HoloDeck-provided default hooks (spec 034 P2b)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from holodeck.lib.backends.claude_hooks import (
    CREDENTIAL_PATTERNS,
    _post_tool_credential_redaction,
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
def test_credential_patterns_include_expected_kinds() -> None:
    """All expected credential kinds are present; tolerates new patterns added later."""
    markers = {marker for marker, _ in CREDENTIAL_PATTERNS}
    expected = {
        "[REDACTED:anthropic-key]",
        "[REDACTED:aws-access-key]",
        "[REDACTED:github-token]",
        "[REDACTED:jwt]",
        "Bearer [REDACTED]",
    }
    assert expected <= markers


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


@pytest.mark.unit
def test_redact_credentials_bounded_by_depth_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recursion guard returns subtree unchanged past the cap, no crash."""
    # Build a nest deeper than the cap so the guard fires.
    payload: Any = "leaf"
    for _ in range(250):
        payload = {"x": payload}

    with caplog.at_level(logging.WARNING):
        out = redact_credentials(payload)

    # Doesn't crash; some subtree was returned as-is (the guard fired).
    assert out is not None
    assert sum(1 for r in caplog.records if "recursion depth cap" in r.message) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_tool_credential_redaction_bash_dict_returns_updated_output() -> (
    None
):
    """SDK contract: PostToolUseHookSpecificOutput.updatedToolOutput is populated."""
    payload = {
        "tool_name": "Bash",
        "tool_use_id": "t1",
        "tool_response": {
            "stdout": "echoed GH_TOKEN=ghp_" + "a" * 36,
            "stderr": "",
            "interrupted": False,
        },
    }
    output = await _post_tool_credential_redaction(payload, "t1", None)  # type: ignore[arg-type]
    hso = output.get("hookSpecificOutput")
    assert hso is not None
    assert hso.get("hookEventName") == "PostToolUse"
    updated = hso.get("updatedToolOutput")
    assert updated is not None
    assert "[REDACTED:github-token]" in updated["stdout"]
    assert updated["stderr"] == ""
    assert updated["interrupted"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_tool_credential_redaction_passthrough_on_clean_output() -> None:
    """No hookSpecificOutput when there is nothing to redact."""
    payload = {
        "tool_name": "Read",
        "tool_use_id": "t2",
        "tool_response": "Retrieved 42 rows",
    }
    output = await _post_tool_credential_redaction(payload, "t2", None)  # type: ignore[arg-type]
    assert (
        "hookSpecificOutput" not in output or output.get("hookSpecificOutput") is None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_tool_credential_redaction_handles_mcp_list_shape() -> None:
    """MCP tools return a list of content blocks; redaction must preserve shape."""
    payload = {
        "tool_name": "mcp__filesystem__read",
        "tool_use_id": "t3",
        "tool_response": [
            {"type": "text", "text": "config token: ghp_" + "z" * 36},
            {"type": "text", "text": "no creds here"},
        ],
    }
    output = await _post_tool_credential_redaction(payload, "t3", None)  # type: ignore[arg-type]
    hso = output.get("hookSpecificOutput")
    assert hso is not None
    updated = hso.get("updatedToolOutput")
    assert isinstance(updated, list)
    assert len(updated) == 2
    assert updated[0]["type"] == "text"
    assert "[REDACTED:github-token]" in updated[0]["text"]
    assert updated[1]["text"] == "no creds here"
