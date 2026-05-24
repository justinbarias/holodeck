"""Prompt-injection scenario: credentials in tool output are scrubbed before
they reach the model context.

Runs against the hook in isolation (no live SDK call — quota-free CI).
The hook is invoked with realistic SDK-shaped payloads that simulate a
tool returning a response containing a credential-shaped substring.
"""

from __future__ import annotations

import pytest

from holodeck.lib.backends.claude_hooks import (
    _post_tool_credential_redaction,
    redact_credentials,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_github_token_redacted_in_tool_response():
    """A GitHub token in tool_response is replaced before model context."""
    payload = {
        "tool_name": "Bash",
        "tool_use_id": "t1",
        "tool_response": ("exported GH_TOKEN=ghp_" + "a" * 36 + " for the workflow"),
    }
    output = await _post_tool_credential_redaction(
        payload,
        "t1",
        None,  # type: ignore[arg-type]
    )
    hso = output.get("hookSpecificOutput")
    assert hso is not None
    updated = hso.get("updatedToolOutput")
    assert updated is not None
    assert "[REDACTED:github-token]" in updated
    assert "ghp_" + "a" * 36 not in updated


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clean_response_is_passthrough():
    """No-op when the tool response has no credential-shaped content."""
    payload = {
        "tool_name": "Read",
        "tool_use_id": "t2",
        "tool_response": "Retrieved 42 rows from table customers",
    }
    output = await _post_tool_credential_redaction(
        payload,
        "t2",
        None,  # type: ignore[arg-type]
    )
    # No-op output: no hookSpecificOutput section.
    assert (
        "hookSpecificOutput" not in output or output.get("hookSpecificOutput") is None
    )


@pytest.mark.integration
def test_nested_tool_response_redacted():
    """Direct test of the helper on a JSON-shaped payload."""
    payload = {
        "logs": [
            {"line": "auth: Bearer eyJfoo.eyJbar.signature"},
            {"line": "ok"},
        ],
        "headers": {"Authorization": "Bearer abc.def-1"},
    }
    redacted = redact_credentials(payload)
    assert (
        "Bearer [REDACTED]" in redacted["logs"][0]["line"]
        or "[REDACTED:jwt]" in redacted["logs"][0]["line"]
    )
    assert redacted["logs"][1]["line"] == "ok"
    assert "Bearer [REDACTED]" in redacted["headers"]["Authorization"]
