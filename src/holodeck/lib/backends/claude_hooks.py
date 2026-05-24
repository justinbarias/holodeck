"""HoloDeck-provided default hooks for Claude Agent SDK.

Ships a single PostToolUse hook that scrubs credential-shaped substrings
out of tool outputs before they re-enter the model context. The
`redact_credentials` helper is also imported by the OTel
RedactingSpanProcessor (spec 034 P2b, Task 5) so trace attributes are
scrubbed independently.

Opt-out via ``claude.disable_default_hooks: true`` — emits a load-time
warning. OTel attribute redaction is NOT disabled by that flag and runs
independently.

Spec 034 P2b — Prompt-injection defenses.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookContext, HookEvent, SyncHookJSONOutput

logger = logging.getLogger(__name__)

# One-shot guard: emit the depth-cap warning at most once per process to
# avoid flooding logs when a deeply-nested payload hits the recursion limit.
_warned_depth_cap = False


# ---------------------------------------------------------------------------
# Credential redaction patterns. Order matters — earlier patterns win on
# overlap. Each entry is (replacement_marker, compiled_pattern).
# ---------------------------------------------------------------------------

CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "[REDACTED:anthropic-key]",
        re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{90,}"),
    ),
    (
        "[REDACTED:aws-access-key]",
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ),
    (
        "[REDACTED:github-token]",
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
    ),
    (
        "[REDACTED:jwt]",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    (
        "Bearer [REDACTED]",
        re.compile(r"Bearer\s+[A-Za-z0-9_.=\-]{20,}"),
    ),
)


def _redact_string(text: str) -> str:
    """Apply every credential pattern to *text* in order."""
    for replacement, pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_MAX_REDACTION_DEPTH = 200


def redact_credentials(value: Any, _depth: int = 0) -> Any:
    """Recursively redact credential-shaped substrings inside *value*.

    Walks strings, dicts, lists, and tuples. Non-string scalars (int,
    float, bool, None) are returned unchanged. Returns a new structure —
    does not mutate *value* in place. The OTel span processor (Task 5)
    reuses this helper, so the behaviour must stay safe for arbitrary
    JSON-shaped input.

    Recursion is bounded at ``_MAX_REDACTION_DEPTH`` levels. If the cap is
    reached the subtree is returned unredacted and a warning is logged once.

    Args:
        value: Tool output, span attribute, or arbitrary JSON-shaped data.
        _depth: Internal recursion counter; callers should not set this.

    Returns:
        The same structure with credential-shaped strings replaced by
        ``[REDACTED:<kind>]`` markers.
    """
    global _warned_depth_cap
    if _depth >= _MAX_REDACTION_DEPTH:
        if not _warned_depth_cap:
            logger.warning(
                "redact_credentials: recursion depth cap (%d) reached "
                "(further occurrences suppressed); returning subtree unredacted",
                _MAX_REDACTION_DEPTH,
            )
            _warned_depth_cap = True
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        return {k: redact_credentials(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_credentials(v, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_credentials(v, _depth + 1) for v in value)
    return value


async def _post_tool_credential_redaction(
    input_data: Any,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PostToolUse hook: scrub credentials out of tool_response before context.

    When a credential-shaped string is found, returns a hook output that
    replaces the tool output via ``updatedToolOutput``. Otherwise returns
    an empty output so the SDK skips any replacement.
    """
    tool_response = input_data.get("tool_response")
    if tool_response is None:
        return SyncHookJSONOutput()

    redacted = redact_credentials(tool_response)
    if redacted == tool_response:
        return SyncHookJSONOutput()

    return SyncHookJSONOutput(
        hookSpecificOutput={
            "hookEventName": "PostToolUse",
            "updatedToolOutput": redacted,
        },
    )


def build_default_hooks() -> dict[HookEvent, list[HookMatcher]]:
    """Build the HoloDeck-provided default hook chain.

    Returns:
        Hook dict suitable for merging into ``ClaudeAgentOptions.hooks``.
        Contains only the PostToolUse credential-redaction matcher; Bash
        hardening is owned by SDK permission rules + P1b auto-disallow
        (see spec 034 P2 plan Task 2 for rationale).
    """
    return {
        "PostToolUse": [HookMatcher(hooks=[_post_tool_credential_redaction])],
    }
