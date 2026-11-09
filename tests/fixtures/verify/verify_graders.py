"""Code graders for the verify-suite agents (``type: code`` metrics).

The test runner puts the agent directory on ``sys.path``, so the fixtures
reference these as ``verify_graders:<name>``.
"""

from __future__ import annotations

from holodeck.lib.test_runner.code_grader import GraderContext

SMOKE_TOOL_RESULT = "SMOKE-TOOL-RESULT: "


def smoke_tool_succeeded(ctx: GraderContext) -> bool:
    """Pass only when ``smoke_echo`` ran and returned its success marker.

    ``expected_tools`` proves the tool was called; this proves it worked. A
    tool that raised records ``result=None`` or an error string, and fails.
    """
    return any(
        invocation.name.endswith("smoke_echo")
        and SMOKE_TOOL_RESULT in str(invocation.result)
        for invocation in ctx.tool_invocations
    )


# ---------------------------------------------------------------------------
# Live-tier graders (live-full-*.yaml). Each reads its target from the turn's
# ``turn_config`` so one grader serves every tool type.
# ---------------------------------------------------------------------------


def _config_str(ctx: GraderContext, key: str) -> str:
    value = ctx.turn_config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"turn_config.{key} must be a non-empty string")
    return value


def tool_result_contains(ctx: GraderContext) -> bool:
    """Pass when a call to ``turn_config.tool`` returned ``turn_config.marker``.

    Tool names are matched as substrings, like ``expected_tools``, so the
    Claude names (``mcp__<server>__<tool>``) and the OpenAI names both match.
    """
    tool = _config_str(ctx, "tool")
    marker = _config_str(ctx, "marker")
    return any(
        tool in invocation.name and marker in str(invocation.result)
        for invocation in ctx.tool_invocations
    )


def response_contains(ctx: GraderContext) -> bool:
    """Pass when the reply contains ``turn_config.marker`` (case-insensitive).

    Used for handoff targets (OpenAI subagents and skills): only the target
    agent's instructions produce the marker.
    """
    return _config_str(ctx, "marker").lower() in ctx.agent_response.lower()


# The Claude CLI exposes subagents through its ``Task`` tool; newer CLIs name
# it ``Agent``. Accept both rather than pin a CLI version.
CLAUDE_SUBAGENT_TOOLS = frozenset({"Task", "Agent"})


def claude_subagent_ran(ctx: GraderContext) -> bool:
    """Pass when ``turn_config.subagent`` ran and produced ``turn_config.marker``.

    The parent must launch the named subagent through Task/Agent, and the
    marker (which only the subagent's prompt produces) must appear in that
    call's result or in the final reply.
    """
    subagent = _config_str(ctx, "subagent")
    marker = _config_str(ctx, "marker").lower()
    return any(
        invocation.name in CLAUDE_SUBAGENT_TOOLS
        and invocation.args.get("subagent_type") == subagent
        and (
            marker in str(invocation.result).lower()
            or marker in ctx.agent_response.lower()
        )
        for invocation in ctx.tool_invocations
    )
