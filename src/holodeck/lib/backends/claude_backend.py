"""Claude Agent SDK backend for HoloDeck.

Implements ``ClaudeBackend`` (AgentBackend) and ``ClaudeSession`` (AgentSession)
for the ``provider: anthropic`` execution path. Every invocation — single-turn
and multi-turn — uses the top-level ``query()`` function; multi-turn sessions
thread state via ``resume=<sdk_session_id>`` (spec 034 P4).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from pathlib import Path
from typing import Any, cast

import claude_agent_sdk
import jsonschema
from ag_ui.core import (
    AssistantMessage as AguiAssistantMessage,
)
from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    MessagesSnapshotEvent,
    ReasoningEncryptedValueEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.core import (
    FunctionCall as AguiFunctionCall,
)
from ag_ui.core import (
    Message as AguiMessage,
)
from ag_ui.core import (
    Tool as AguiTool,
)
from ag_ui.core import (
    ToolCall as AguiToolCall,
)
from ag_ui.core import (
    ToolMessage as AguiToolMessage,
)
from ag_ui.core import (
    UserMessage as AguiUserMessage,
)
from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ProcessError,
    tool,
)
from claude_agent_sdk._errors import CLIConnectionError, MessageParseError
from claude_agent_sdk.types import (
    AgentDefinition,
    HookContext,
    HookEvent,
    HookInput,
    McpSdkServerConfig,
    PostToolUseFailureHookInput,
    PostToolUseHookInput,
    PreToolUseHookInput,
    SyncHookJSONOutput,
)
from exceptiongroup import BaseExceptionGroup
from ulid import ULID

from holodeck.lib.backends.base import (
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
    ToolEvent,
)
from holodeck.lib.backends.mcp_bridge import build_claude_mcp_configs
from holodeck.lib.backends.otel_bridge import translate_observability
from holodeck.lib.backends.tool_adapters import (
    build_holodeck_sdk_server,
    create_tool_adapters,
)
from holodeck.lib.backends.validators import (
    validate_credentials,
    validate_embedding_provider,
    validate_nodejs,
    validate_response_format,
    validate_working_directory,
)
from holodeck.lib.errors import ConfigError
from holodeck.lib.instruction_resolver import resolve_instructions
from holodeck.lib.observability import get_observability_context
from holodeck.models.agent import Agent
from holodeck.models.claude_config import AuthProvider, PermissionMode
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool import HierarchicalDocumentToolConfig, MCPTool
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1
_AGUI_MCP_SERVER_NAME = "ag_ui"
_AGUI_STATE_TOOL_NAME = "ag_ui_update_state"
_AGUI_STATE_TOOL_FULL_NAME = "mcp__ag_ui__ag_ui_update_state"
_AGUI_FORWARDED_PROPS: frozenset[str] = frozenset(
    {
        "resume",
        "fork_session",
        "model",
        "fallback_model",
        "max_turns",
        "max_budget_usd",
        "max_thinking_tokens",
        "include_partial_messages",
        "strict_mcp_config",
        "betas",
        "enable_file_checkpointing",
        "effort",
        "thinking",
        "output_format",
    }
)
_AGUI_STATE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "state_updates": {
            "type": "object",
            "additionalProperties": True,
        }
    },
    "required": ["state_updates"],
}


@dataclasses.dataclass(frozen=True)
class _AguiStreamStateUpdate:
    """Internal stream item carrying updated AG-UI state to the caller."""

    state: Any


@dataclasses.dataclass(frozen=True)
class _AguiStreamResult:
    """Internal stream item carrying Claude result metadata to the caller."""

    result: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _AguiSdkToolUseBlock:
    """Normalized SDK tool-use block for AG-UI conversion."""

    tool_id: str
    raw_name: str
    display_name: str
    tool_input: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _AguiSdkToolResultBlock:
    """Normalized SDK tool-result block for AG-UI conversion."""

    tool_use_id: str
    content: Any


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _wrap_prompt(message: str) -> AsyncGenerator[dict[str, Any], None]:
    """Wrap a string prompt as an async iterable of user message dicts.

    The Claude Agent SDK ``query()`` function treats string prompts differently
    from async-iterable prompts: strings trigger an immediate ``end_input()``
    which closes stdin.  When SDK MCP servers are configured, the subprocess
    needs stdin kept open for bidirectional tool-call communication.  Passing
    an async iterable instead routes through ``stream_input()``, which keeps
    stdin open until the first result arrives.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": message},
        "parent_tool_use_id": None,
    }


async def _streaming_user_envelope(
    message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Wrap a user message in the streaming-mode envelope ``query()`` expects.

    ``query()`` accepts either a ``str`` prompt or an ``AsyncIterable[dict]``.
    The ``str`` form writes the message and immediately closes stdin via
    ``end_input()``, which deadlocks any subsequent SDK-MCP tool callback
    (control-channel writes fail with ``ProcessTransport is not ready for
    writing``). Streaming mode keeps stdin open via the SDK's background
    ``stream_input`` task, so tool callbacks can write responses back.

    See spec 034 P4 spike v1 (2026-05-19) for the surfacing.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": message},
        "parent_tool_use_id": None,
    }


def _enrich_tool_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add ``name`` to each tool result by matching ``call_id`` to tool calls.

    The SDK yields tool names only in ``ToolUseBlock`` (inside
    ``AssistantMessage``), while tool results (``ToolResultBlock`` inside
    ``UserMessage``) only carry ``call_id``.  Downstream consumers like
    ``build_retrieval_context_from_tools`` expect a ``name`` key on each
    result dict.
    """
    id_to_name: dict[str, str] = {
        tc["call_id"]: tc["name"] for tc in tool_calls if "call_id" in tc
    }
    for tr in tool_results:
        if "name" not in tr and tr.get("call_id") in id_to_name:
            tr["name"] = id_to_name[tr["call_id"]]
    return tool_results


def _fix_surrogates(value: str) -> str:
    """Reassemble UTF-16 surrogate pairs before Pydantic serialization."""
    try:
        return value.encode("utf-16", "surrogatepass").decode("utf-16")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value.encode("utf-8", "replace").decode("utf-8")


def _fix_surrogates_deep(value: Any) -> Any:
    """Recursively fix surrogate pairs in nested structures."""
    if isinstance(value, str):
        return _fix_surrogates(value)
    if isinstance(value, dict):
        return {
            _fix_surrogates(k) if isinstance(k, str) else k: _fix_surrogates_deep(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_fix_surrogates_deep(v) for v in value]
    return value


def _strip_mcp_prefix(tool_name: str) -> str:
    """Return frontend-facing tool name for SDK MCP tool identifiers."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


def _is_agui_state_tool(tool_name: str | None) -> bool:
    """Return True when *tool_name* is the internal AG-UI state tool."""
    return tool_name in {_AGUI_STATE_TOOL_NAME, _AGUI_STATE_TOOL_FULL_NAME}


def _agui_tool_names(tools: list[AguiTool] | None) -> list[str]:
    """Extract frontend tool names from AG-UI tool definitions."""
    return [tool_def.name for tool_def in tools or [] if tool_def.name]


def _agui_message_content(message: AguiMessage) -> str:
    """Extract plain text content from an AG-UI message object."""
    if isinstance(message, AguiAssistantMessage):
        return str(message.content or "")
    if isinstance(message, AguiToolMessage):
        return str(message.content)
    if isinstance(message, AguiUserMessage):
        user_content = message.content
        if isinstance(user_content, str):
            return user_content
        return " ".join(block.text for block in user_content if block.type == "text")
    other_content = message.content
    if isinstance(other_content, str):
        return other_content
    return ""


def _agui_message_role(message: AguiMessage) -> str:
    """Extract the AG-UI role from a message object."""
    return str(message.role)


def _agui_tool_call_id(message: AguiMessage) -> str:
    """Extract the AG-UI tool call id from a tool message."""
    if isinstance(message, AguiToolMessage):
        return str(message.tool_call_id)
    return ""


def _agui_message_transcript_line(message: AguiMessage) -> str:
    """Render one AG-UI message into a compact prompt transcript line."""
    role = _agui_message_role(message)
    content = _agui_message_content(message)
    if isinstance(message, AguiAssistantMessage):
        rendered_calls: list[str] = []
        for call in message.tool_calls or []:
            rendered_calls.append(
                "tool_call "
                f"id={call.id} "
                f"name={call.function.name} "
                f"arguments={call.function.arguments}"
            )
        if rendered_calls:
            call_text = "; ".join(rendered_calls)
            return f"assistant: {content}\nassistant_tool_calls: {call_text}"
    if isinstance(message, AguiToolMessage):
        return f"tool_result id={_agui_tool_call_id(message)}: {content}"
    return f"{role}: {content}"


def _agui_frontend_tool_resume_prompt(input_data: RunAgentInput) -> str:
    """Build a continuation prompt when AG-UI resumes after a frontend tool."""
    messages = input_data.messages or []
    transcript = "\n".join(
        _agui_message_transcript_line(message) for message in messages[-20:]
    )
    return (
        "Continue the AG-UI conversation from this transcript.\n\n"
        "The latest frontend tool result has already been executed by the UI. "
        "Use that result to continue the conversation. Do not call the same "
        "frontend tool again unless the result is missing, invalid, or the user "
        "explicitly asks to run it again.\n\n"
        f"{transcript}"
    )


def _agui_prompt_from_input(
    input_data: RunAgentInput,
    message_override: str | None = None,
) -> str:
    """Return the latest AG-UI message payload to send to Claude."""
    if message_override is not None:
        return message_override
    messages = input_data.messages or []
    if not messages:
        return ""
    if _agui_message_role(messages[-1]) == "tool":
        return _agui_frontend_tool_resume_prompt(input_data)
    return _agui_message_content(messages[-1])


def _agui_state_context_addendum(input_data: RunAgentInput) -> str:
    """Build a system-prompt addendum for AG-UI context and shared state."""
    parts: list[str] = []
    if input_data.context:
        parts.append("## Context from the application")
        for context in input_data.context:
            parts.append(f"- {context.description}: {context.value}")
        parts.append("")
    if input_data.state is not None:
        parts.append("## Current Shared State")
        parts.append("This state is shared with the frontend UI and can be updated.")
        try:
            state_json = json.dumps(input_data.state, indent=2)
        except (TypeError, ValueError):
            state_json = str(input_data.state)
        parts.append(f"```json\n{state_json}\n```")
        parts.append("")
        parts.append(
            "To update this state, use the `ag_ui_update_state` tool with your "
            "changes."
        )
        parts.append("")
    return "\n".join(parts)


def _make_agui_frontend_tool(tool_def: AguiTool) -> Any:
    """Convert an AG-UI Tool definition to a Claude SDK MCP proxy tool."""

    @tool(tool_def.name, tool_def.description, tool_def.parameters or {})
    async def frontend_tool_stub(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": "Tool call forwarded to AG-UI client"}]
        }

    return frontend_tool_stub


def _make_agui_state_tool() -> Any:
    """Create the internal AG-UI state update tool."""

    @tool(
        _AGUI_STATE_TOOL_NAME,
        "Update the shared application state visible to the AG-UI frontend.",
        _AGUI_STATE_TOOL_SCHEMA,
    )
    async def update_state_tool(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "State updated"}]}

    return update_state_tool


def _clone_options(options: ClaudeAgentOptions, **updates: Any) -> ClaudeAgentOptions:
    """Clone ``ClaudeAgentOptions`` without mutating the session base options."""
    try:
        dataclasses.fields(options)
    except TypeError:
        cloned = copy.copy(options)
        for key, value in updates.items():
            setattr(cloned, key, value)
        return cloned
    else:
        return dataclasses.replace(options, **updates)


def _agui_tool_result_content(content: Any) -> str:
    """Normalize Claude SDK tool-result content for AG-UI."""
    if content is None:
        return ""
    try:
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                text = str(first.get("text", ""))
                try:
                    return json.dumps(json.loads(text))
                except (json.JSONDecodeError, ValueError):
                    return _fix_surrogates(text)
            return _fix_surrogates(json.dumps(content))
        return _fix_surrogates(json.dumps(content))
    except (TypeError, ValueError):
        return _fix_surrogates(str(content))


def _sdk_content_blocks(sdk_message: object) -> list[Any]:
    """Return Claude SDK content blocks while containing dynamic SDK access.

    The Claude SDK content block classes are not stable public typing surfaces,
    and unit tests use lightweight SDK-shaped doubles. Keep the ``getattr``
    boundary here so stream translation stays typed after normalization.
    """
    content = getattr(sdk_message, "content", None)
    if content is None:
        return []
    if isinstance(content, list):
        return content
    try:
        return list(cast(Any, content))
    except TypeError:
        return []


def _agui_sdk_tool_use_blocks(sdk_message: object) -> list[_AguiSdkToolUseBlock]:
    """Normalize Claude SDK tool-use blocks for AG-UI stream handling."""
    tool_blocks: list[_AguiSdkToolUseBlock] = []
    for block in _sdk_content_blocks(sdk_message):
        if block.__class__.__name__ != "ToolUseBlock":
            continue
        tool_id = getattr(block, "id", None)
        if not tool_id:
            continue
        raw_name = str(getattr(block, "name", "unknown") or "unknown")
        tool_input = getattr(block, "input", {}) or {}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}
        tool_blocks.append(
            _AguiSdkToolUseBlock(
                tool_id=str(tool_id),
                raw_name=raw_name,
                display_name=_strip_mcp_prefix(raw_name),
                tool_input=tool_input,
            )
        )
    return tool_blocks


def _agui_sdk_tool_result_blocks(sdk_message: object) -> list[_AguiSdkToolResultBlock]:
    """Normalize Claude SDK tool-result blocks for AG-UI stream handling."""
    result_blocks: list[_AguiSdkToolResultBlock] = []
    for block in _sdk_content_blocks(sdk_message):
        if block.__class__.__name__ != "ToolResultBlock":
            continue
        tool_use_id = getattr(block, "tool_use_id", None)
        if not tool_use_id:
            continue
        result_blocks.append(
            _AguiSdkToolResultBlock(
                tool_use_id=str(tool_use_id),
                content=getattr(block, "content", None),
            )
        )
    return result_blocks


async def _close_message_stream(message_stream: object) -> None:
    """Close an SDK async stream immediately when AG-UI must halt."""
    aclose = getattr(message_stream, "aclose", None)
    if aclose is None:
        return
    close_result = aclose()
    if inspect.isawaitable(close_result):
        await close_result


def _agui_assistant_message_from_sdk(
    sdk_message: Any,
    message_id: str,
) -> AguiAssistantMessage | None:
    """Convert a complete Claude SDK AssistantMessage to AG-UI history."""
    text = ""
    tool_calls: list[AguiToolCall] = []
    for block in getattr(sdk_message, "content", []) or []:
        block_type = getattr(block, "type", None)
        block_cls = block.__class__.__name__
        if block_type == "text" or block_cls == "TextBlock":
            text += getattr(block, "text", "")
        elif block_type == "tool_use" or block_cls == "ToolUseBlock":
            raw_name = getattr(block, "name", "")
            if _is_agui_state_tool(raw_name):
                continue
            tool_id = getattr(block, "id", "")
            tool_input = getattr(block, "input", {}) or {}
            tool_calls.append(
                AguiToolCall(
                    id=tool_id,
                    type="function",
                    function=AguiFunctionCall(
                        name=_strip_mcp_prefix(raw_name),
                        arguments=json.dumps(tool_input),
                    ),
                )
            )
    if not text and not tool_calls:
        return None
    return AguiAssistantMessage(
        id=message_id,
        role="assistant",
        content=text or None,
        tool_calls=tool_calls or None,
    )


def _agui_tool_message(tool_use_id: str, content: Any) -> AguiToolMessage:
    """Convert a Claude SDK ToolResultBlock to AG-UI tool history."""
    return AguiToolMessage(
        id=f"{tool_use_id}-result",
        role="tool",
        content=_agui_tool_result_content(content),
        tool_call_id=tool_use_id,
    )


def _extract_result_text(content: list[Any]) -> str:
    """Extract concatenated text from SDK content blocks.

    Handles ``TextBlock`` instances, dicts with a ``text`` key, and plain strings.

    Args:
        content: List of content blocks from an ``AssistantMessage``.

    Returns:
        Concatenated text content.
    """
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text") and block.__class__.__name__ == "TextBlock":
            parts.append(block.text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _process_message(
    msg: Any,
    text_parts: list[str],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    thinking_parts: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Extract text, thinking, tool calls, and tool results from SDK messages.

    Handles both ``AssistantMessage`` (text + thinking + tool-use blocks) and
    ``UserMessage`` (tool-result blocks from MCP tool execution).
    Mutates the provided lists in-place and also returns them for convenience.

    Args:
        msg: A message from the SDK response stream.
        text_parts: Accumulator for text content.
        tool_calls: Accumulator for tool call dicts.
        tool_results: Accumulator for tool result dicts.
        thinking_parts: Accumulator for extended-thinking text. Pass ``None``
            (the default) when the caller does not surface thinking; a fresh
            list is allocated and returned in that case.

    Returns:
        ``(text_parts, tool_calls, tool_results, thinking_parts)``.
    """
    if thinking_parts is None:
        thinking_parts = []

    msg_type = msg.__class__.__name__
    if msg_type not in ("AssistantMessage", "UserMessage"):
        return text_parts, tool_calls, tool_results, thinking_parts

    for block in msg.content:
        cls_name = block.__class__.__name__
        if cls_name == "TextBlock":
            text_parts.append(block.text)
        elif cls_name == "ThinkingBlock":
            # The SDK exposes thinking text on the ``thinking`` attribute.
            thinking_text = getattr(block, "thinking", None)
            if isinstance(thinking_text, str) and thinking_text:
                thinking_parts.append(thinking_text)
        elif cls_name == "ToolUseBlock":
            tool_calls.append(
                {
                    "name": block.name,
                    "arguments": block.input,
                    "call_id": block.id,
                }
            )
        elif cls_name == "ToolResultBlock":
            result_text = _extract_result_text(block.content)
            tool_results.append(
                {
                    "call_id": block.tool_use_id,
                    "result": result_text,
                    "is_error": block.is_error,
                }
            )
    return text_parts, tool_calls, tool_results, thinking_parts


_RISKY_BUILTIN_TOOLS: frozenset[str] = frozenset({"Bash", "Write", "Edit", "WebFetch"})


def _declared_builtin_tools(claude: Any) -> set[str]:
    """Return the set of risky built-in SDK tools the operator has declared.

    A tool counts as declared when the HoloDeck schema explicitly opts in
    via a dedicated field (claude.bash.enabled, claude.file_system.write,
    etc.) or when the tool name appears in claude.allowed_tools.

    Args:
        claude: ``ClaudeConfig`` instance or ``None``.

    Returns:
        Subset of ``_RISKY_BUILTIN_TOOLS`` that the operator has declared.
    """
    declared: set[str] = set()
    if claude is None:
        return declared
    if claude.bash is not None and claude.bash.enabled:
        declared.add("Bash")
    if claude.file_system is not None:
        if claude.file_system.write:
            declared.add("Write")
        if claude.file_system.edit:
            declared.add("Edit")
    if claude.allowed_tools:
        declared.update(claude.allowed_tools)
    return declared


def _build_permission_mode(
    claude: Any,
    mode: str,
) -> str | None:
    """Map HoloDeck permission mode to SDK permission literal.

    Decision tree (P1b — spec 034):
      * ``manual`` + serve/chat context -> SDK ``acceptEdits``. The legacy
        mapping to ``default`` wedges in serve mode because there is no
        operator at the terminal to answer the SDK's permission prompts.
        ``acceptEdits`` respects ``allowed_tools``/``disallowed_tools``
        and auto-approves Edit/Write/MultiEdit; the auto-disallow of
        dangerous built-ins (see ``_declared_builtin_tools``) is what
        actually fails closed for un-declared tools.
      * ``manual`` + test -> SDK ``default``. Unchanged; ``holodeck test
        run`` has an operator at the terminal who can answer prompts.
      * ``acceptEdits`` -> SDK ``acceptEdits``. Unchanged.
      * ``acceptAll`` -> SDK ``bypassPermissions`` **only when**
        ``claude.i_understand_this_is_unsafe`` is ``True``; else raise
        ``ConfigError`` with a migration message. ``bypassPermissions``
        disables the SDK permission system entirely.

    Removed: the legacy ``if mode == "test" and permission_mode !=
    manual: sdk_mode = bypassPermissions`` escalation silently turned
    every test-mode ``acceptEdits`` into ``bypassPermissions``, so an
    operator who declared "auto-approve edits, prompt for Bash" got
    "approve everything" in ``holodeck test run``. ``acceptAll`` +
    explicit ``i_understand_this_is_unsafe`` is now the only path to
    ``bypassPermissions`` in any mode.

    Args:
        claude: ``ClaudeConfig`` instance or ``None``.
        mode: Execution mode (``"test"`` or ``"chat"``).

    Returns:
        SDK permission mode literal, or ``None`` when no claude config.

    Raises:
        ConfigError: When ``permission_mode=acceptAll`` is set without
            ``i_understand_this_is_unsafe=true``.
    """
    if claude is None:
        return None

    permission_mode: PermissionMode = claude.permission_mode

    if permission_mode == PermissionMode.acceptAll:
        if not claude.i_understand_this_is_unsafe:
            raise ConfigError(
                field="claude.permission_mode",
                message=(
                    "permission_mode='acceptAll' disables the Claude SDK "
                    "permission system entirely, allowing any tool (including "
                    "Bash, Write, Edit, WebFetch) to execute without "
                    "restriction. Add `claude.i_understand_this_is_unsafe: "
                    "true` to opt in, or — preferred — declare the specific "
                    "tools your agent needs via `claude.bash.enabled`, "
                    "`claude.file_system.*`, `claude.web_search`, or "
                    "`claude.allowed_tools` and use `permission_mode: manual` "
                    "(the default)."
                ),
            )
        return "bypassPermissions"

    if permission_mode == PermissionMode.acceptEdits:
        return "acceptEdits"

    # PermissionMode.manual
    if mode == "chat":
        return "acceptEdits"
    return "default"


def _build_output_format(
    response_format: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    """Translate HoloDeck response_format to SDK output_format dict.

    Args:
        response_format: JSON Schema dict, file path string, or ``None``.

    Returns:
        SDK-compatible output_format dict, or ``None``.
    """
    if response_format is None:
        return None

    if isinstance(response_format, str):
        # File path — load and parse
        import pathlib

        schema_path = pathlib.Path(response_format)
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
        else:
            return None
    else:
        schema = response_format

    return {"type": "json_schema", "schema": schema}


# ---------------------------------------------------------------------------
# build_options()
# ---------------------------------------------------------------------------


_DEFAULT_MAX_TURNS = 20
"""Bound on agent loop iterations when the operator does not set one.

The hosting guide explicitly calls out maxTurns as the way to prevent
the SDK from getting stuck in a tool-call loop. 20 is high enough for
multi-step ConvFinQA-style reasoning and low enough to bound runaway loops.
"""


def build_options(
    *,
    agent: Agent,
    tool_server: McpSdkServerConfig | None,
    tool_names: list[str],
    mcp_configs: dict[str, Any],
    auth_env: dict[str, str],
    otel_env: dict[str, str],
    mode: str,
) -> ClaudeAgentOptions:
    """Assemble ``ClaudeAgentOptions`` from agent config and bridge outputs.

    Args:
        agent: The agent configuration.
        tool_server: In-process MCP server for vectorstore/hierarchical-doc tools.
        tool_names: Allowed tool names from the in-process server.
        mcp_configs: External MCP server configs from ``build_claude_mcp_configs()``.
        auth_env: Auth env vars from ``validate_credentials()``.
        otel_env: OTel env vars from ``translate_observability()``.
        mode: Execution mode (``"test"`` or ``"chat"``).

    Returns:
        Configured ``ClaudeAgentOptions`` ready for ``query()``.
    """
    claude = agent.claude
    system_prompt = resolve_instructions(agent.instructions)

    # MCP servers
    mcp_servers: dict[str, Any] = dict(mcp_configs)
    if tool_server is not None:
        mcp_servers["holodeck_tools"] = tool_server

    # Env vars — unset CLAUDECODE to prevent the "nested session" guard
    # when HoloDeck runs inside a terminal with Claude Code active.
    # SDK merges: {**os.environ, **options.env}, so "" overrides "1".
    env: dict[str, str] = {"CLAUDECODE": "", **auth_env, **otel_env}

    # Custom base URL — when a custom auth provider is used with an explicit
    # endpoint, forward it as ANTHROPIC_BASE_URL so the SDK subprocess targets
    # the third-party endpoint.  Only inject for auth_provider=custom to avoid
    # accidentally overriding the default Anthropic URL for proxy/rate-limit
    # setups that use endpoint with a standard auth provider.
    if agent.model.endpoint and agent.model.auth_provider == AuthProvider.custom:
        env["ANTHROPIC_BASE_URL"] = agent.model.endpoint

    # Permission mode (raises ConfigError for acceptAll without unsafe opt-in)
    perm_mode = _build_permission_mode(claude, mode)

    # Extended thinking
    max_thinking_tokens = None
    if claude and claude.extended_thinking and claude.extended_thinking.enabled:
        max_thinking_tokens = claude.extended_thinking.budget_tokens

    # Allowed tools
    allowed_tools: list[str] = list(tool_names)
    if claude and claude.allowed_tools:
        allowed_tools.extend(claude.allowed_tools)

    # Built-in capabilities
    if claude and claude.web_search:
        allowed_tools.append("WebSearch")

    # Output format
    output_format = _build_output_format(agent.response_format)

    # Working directory — fall back to agent.yaml's directory so that
    # relative paths in MCP args (e.g. "./data") resolve correctly.
    from holodeck.config.context import agent_base_dir

    cwd = claude.working_directory if claude else None
    if cwd is None:
        cwd = agent_base_dir.get()

    # Max turns — default to _DEFAULT_MAX_TURNS when unset (spec 034 P1a)
    max_turns = (
        claude.max_turns
        if claude and claude.max_turns is not None
        else _DEFAULT_MAX_TURNS
    )

    # Disallowed tools (P1b — spec 034). Auto-disallow risky built-in SDK
    # tools (Bash, Write, Edit, WebFetch) that the operator has not declared
    # via the dedicated schema fields or claude.allowed_tools. This is the
    # real fail-closed lever — permission_mode is only a tiebreaker for tools
    # not covered by either list.
    declared = _declared_builtin_tools(claude)
    auto_disallow = sorted(_RISKY_BUILTIN_TOOLS - declared)
    explicit_disallow = (
        list(claude.disallowed_tools) if claude and claude.disallowed_tools else []
    )
    disallowed_tools: list[str] | None = (
        sorted(set(explicit_disallow) | set(auto_disallow))
        if (explicit_disallow or auto_disallow)
        else None
    )
    if auto_disallow:
        logger.info(
            "Auto-disallowed risky built-in SDK tools not declared in agent "
            "config: %s. Declare them via claude.bash.enabled, "
            "claude.file_system.*, or claude.allowed_tools to opt in.",
            ", ".join(auto_disallow),
        )

    # Default [] → CLI subprocess never inherits ~/.claude plugins/skills/hooks.
    setting_sources: list[str] = (
        list(claude.setting_sources)
        if claude is not None and claude.setting_sources is not None
        else []
    )

    # Build the options dict
    opts_kwargs: dict[str, Any] = {
        "model": agent.model.name,
        "system_prompt": system_prompt,
        "permission_mode": perm_mode,
        "max_turns": max_turns,
        "mcp_servers": mcp_servers,
        "allowed_tools": allowed_tools,
        "env": env,
        "cwd": cwd,
        "output_format": output_format,
        "setting_sources": setting_sources,
    }
    if disallowed_tools is not None:
        opts_kwargs["disallowed_tools"] = disallowed_tools

    if max_thinking_tokens is not None:
        opts_kwargs["max_thinking_tokens"] = max_thinking_tokens

    if claude is not None:
        if claude.effort is not None:
            opts_kwargs["effort"] = claude.effort
        if claude.max_budget_usd is not None:
            opts_kwargs["max_budget_usd"] = claude.max_budget_usd
        if claude.fallback_model is not None:
            opts_kwargs["fallback_model"] = claude.fallback_model
        if claude.agents:
            opts_kwargs["agents"] = {
                name: AgentDefinition(
                    description=spec.description,
                    prompt=spec.prompt or "",
                    tools=spec.tools,
                    model=spec.model,
                )
                for name, spec in claude.agents.items()
            }

    # Spec 034 P2b — merge HoloDeck default hooks (credential-redaction
    # PostToolUse) ahead of any user-supplied hooks. Opt-out by setting
    # `claude.disable_default_hooks: true` in agent.yaml. Bash hardening
    # is handled by SDK permission rules + P1b auto-disallow.
    from holodeck.lib.backends.claude_hooks import build_default_hooks

    user_hooks: dict[Any, list[Any]] = opts_kwargs.get("hooks") or {}
    if claude is not None and claude.disable_default_hooks:
        logger.warning(
            "HoloDeck default hooks DISABLED for agent '%s' (credential "
            "redaction PostToolUse hook will NOT run). OTel attribute "
            "redaction is unaffected. To re-enable, remove "
            "`claude.disable_default_hooks: true` from agent.yaml.",
            agent.name,
        )
        if user_hooks:
            opts_kwargs["hooks"] = user_hooks
    else:
        default_hooks = build_default_hooks()
        merged: dict[Any, list[Any]] = {}
        # Default hooks first so they short-circuit when they deny.
        for event, matchers in default_hooks.items():
            merged[event] = list(matchers)
        for event, matchers in user_hooks.items():
            if event in merged:
                merged[event] = merged[event] + list(matchers)
            else:
                merged[event] = list(matchers)
        if merged:
            opts_kwargs["hooks"] = merged

    # Spec 034 P2b — default-on subprocess env scrubbing. These two flags
    # tell the Claude CLI to strip Anthropic/cloud creds from tool
    # subprocesses (Bash, etc.) and to spawn stdio MCP servers with only
    # their declared env (not the full inherited shell env). They don't
    # affect the SDK subprocess's own env (which inherits everything from
    # this serve process — see P3 for the structural fix). Opt out via
    # `claude.disable_subprocess_env_scrub: true` in agent.yaml.
    env_overrides = dict(opts_kwargs.get("env") or {})
    if claude is not None and claude.disable_subprocess_env_scrub:
        logger.warning(
            "Subprocess env scrubbing DISABLED for agent '%s' "
            "(tool subprocesses and stdio MCP servers will inherit the "
            "full agent container env including credentials). To re-enable, "
            "remove `claude.disable_subprocess_env_scrub: true` from "
            "agent.yaml.",
            agent.name,
        )
    else:
        env_overrides.setdefault("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "1")
        # Note: this scrubs the SDK-spawned subprocess env for stdio MCP servers.
        # Operators must declare any inherited env vars (HOME, PATH, provider creds)
        # on the MCP tool's `env` block — see docs/security/prompt-injection-defenses.md
        # §"Operator footgun".
        env_overrides.setdefault("CLAUDE_CODE_MCP_ALLOWLIST_ENV", "1")
    if env_overrides:
        opts_kwargs["env"] = env_overrides

    return ClaudeAgentOptions(**opts_kwargs)


# ---------------------------------------------------------------------------
# ClaudeSession
# ---------------------------------------------------------------------------


def _build_tool_hooks(
    queue: asyncio.Queue[ToolEvent],
) -> dict[HookEvent, list[HookMatcher]]:
    """Build Claude SDK hook matchers that push :class:`ToolEvent` to *queue*.

    Three hooks are registered:

    * ``PreToolUse`` — fires **before** the SDK executes a tool.
    * ``PostToolUse`` — fires **after** a tool succeeds.
    * ``PostToolUseFailure`` — fires **after** a tool fails.

    Each callback returns an empty dict (no-op passthrough) so that the SDK
    continues execution normally.

    Args:
        queue: Destination queue for tool events.

    Returns:
        Hook dict suitable for merging into ``ClaudeAgentOptions.hooks``.
    """

    async def _on_pre_tool(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_input = cast(PreToolUseHookInput, input_data)
        await queue.put(
            ToolEvent(
                kind="start",
                tool_name=tool_input["tool_name"],
                tool_use_id=tool_input["tool_use_id"],
                tool_input=tool_input.get("tool_input"),
            )
        )
        return SyncHookJSONOutput()

    async def _on_post_tool(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_input = cast(PostToolUseHookInput, input_data)
        await queue.put(
            ToolEvent(
                kind="end",
                tool_name=tool_input["tool_name"],
                tool_use_id=tool_input["tool_use_id"],
                tool_response=str(tool_input.get("tool_response", "")),
            )
        )
        return SyncHookJSONOutput()

    async def _on_post_tool_failure(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        tool_input = cast(PostToolUseFailureHookInput, input_data)
        await queue.put(
            ToolEvent(
                kind="error",
                tool_name=tool_input["tool_name"],
                tool_use_id=tool_input["tool_use_id"],
                error=str(tool_input.get("error", "")),
            )
        )
        return SyncHookJSONOutput()

    return {
        "PreToolUse": [HookMatcher(hooks=[_on_pre_tool])],
        "PostToolUse": [HookMatcher(hooks=[_on_post_tool])],
        "PostToolUseFailure": [HookMatcher(hooks=[_on_post_tool_failure])],
    }


def _maybe_emit_thinking_blocks(
    msg: Any,
    queue: asyncio.Queue[ToolEvent],
) -> None:
    """Push one ``ToolEvent(kind='thinking')`` per ``ThinkingBlock`` on *msg*.

    Streams extended-thinking text as it arrives so consumers (e.g. the
    AG-UI emitter) can interleave reasoning bubbles with tool-call cards
    in the order the model produced them.  Each block gets its own ULID
    so downstream protocols can correlate the start/content/end markers
    of a single reasoning chunk.

    Args:
        msg: A message from the SDK response stream.
        queue: Destination event queue.
    """
    if msg.__class__.__name__ != "AssistantMessage":
        return
    content = getattr(msg, "content", []) or []
    for block in content:
        if block.__class__.__name__ != "ThinkingBlock":
            continue
        thinking_text = getattr(block, "thinking", None)
        if not isinstance(thinking_text, str) or not thinking_text:
            continue
        # Best-effort surface; never block message processing on a full queue.
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(
                ToolEvent(
                    kind="thinking",
                    tool_name="",
                    tool_use_id=str(ULID()),
                    text=thinking_text,
                )
            )


def _maybe_emit_subagent_message(
    msg: Any,
    queue: asyncio.Queue[ToolEvent],
) -> None:
    """Push ``subagent_message`` and ``parent_link`` events for subagent traffic.

    The Claude SDK yields nested messages from subagents (Task tool) on the
    same response stream as the parent.  Each subagent ``AssistantMessage``
    carries a non-null ``parent_tool_use_id`` pointing at the launching
    Task's tool use id.  We use it for two things:

    1. Surface the latest assistant-text snapshot via ``subagent_message`` so
       the chat tools panel can display in-flight subagent activity.
    2. Emit a ``parent_link`` for every ``ToolUseBlock`` carried by the
       subagent message, so the panel knows which top-level tool entries are
       actually nested under a subagent and can render them indented.

    Args:
        msg: A message from ``client.receive_response()``.
        queue: Destination event queue.
    """
    if msg.__class__.__name__ != "AssistantMessage":
        return
    parent_id = getattr(msg, "parent_tool_use_id", None)
    if not parent_id:
        return
    content = getattr(msg, "content", []) or []
    text = _extract_result_text(content).strip()
    # Best-effort surface; never block message processing on a full queue.
    with contextlib.suppress(asyncio.QueueFull):
        if text:
            queue.put_nowait(
                ToolEvent(
                    kind="subagent_message",
                    tool_name="Task",
                    tool_use_id=parent_id,
                    parent_tool_use_id=parent_id,
                    text=text,
                )
            )
        for block in content:
            if block.__class__.__name__ != "ToolUseBlock":
                continue
            queue.put_nowait(
                ToolEvent(
                    kind="parent_link",
                    tool_name=getattr(block, "name", ""),
                    tool_use_id=getattr(block, "id", ""),
                    parent_tool_use_id=parent_id,
                )
            )


def _transcript_path(session_id: str, cwd: Path | str | None = None) -> Path:
    """Return the on-disk JSONL transcript path for a session_id.

    The Claude CLI writes per-session transcripts to
    ``~/.claude/projects/<encoded-cwd>/<session_id>.jsonl`` where
    ``encoded-cwd`` is the absolute cwd with ``/`` replaced by ``-``.

    Args:
        session_id: CLI-assigned conversation id (from ResultMessage).
        cwd: The subprocess cwd the SDK passed to the CLI. Must match
            exactly what was set on ``ClaudeAgentOptions.cwd``; do not
            resolve symlinks here because the CLI's encoder uses the
            raw string passed via stdin. Defaults to ``Path.cwd()``
            for agents that don't override cwd.
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    if not base.is_absolute():
        base = Path.cwd() / base
    encoded = str(base).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded / f"{session_id}.jsonl"


class ClaudeSession:
    """Stateful multi-turn session backed by ``query(resume=...)`` (spec 034 P4).

    Each ``send()`` / ``send_streaming()`` opens a fresh CLI subprocess via
    the top-level ``query()`` function. Turn 1 has no ``resume``; the CLI
    assigns a session id which is captured from ``ResultMessage`` and
    stored on ``_sdk_session_id``. Subsequent turns pass that id via
    ``options.resume`` so the CLI rehydrates the JSONL transcript at
    ``~/.claude/projects/<encoded-cwd>/<sdk_session_id>.jsonl``.

    The ``_base_options`` reference is **never mutated**. Turn-specific
    options are created as new ``ClaudeAgentOptions`` instances.
    """

    def __init__(self, options: ClaudeAgentOptions) -> None:
        """Initialize session with base options.

        Args:
            options: Base options (immutable reference for the session lifetime).
        """
        self._base_options = options
        self._turn_count: int = 0
        self._tool_event_queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        # spec 034 P4 — hybrid-session state.
        # CLI-assigned conversation id; captured from ResultMessage on turn 1
        # and fed into ``options.resume`` on turn 2+ so each fresh subprocess
        # rehydrates the JSONL transcript at
        # ``~/.claude/projects/<encoded-cwd>/<sdk_session_id>.jsonl``.
        self._sdk_session_id: str | None = None
        # Serialises concurrent send() / send_streaming() on the same session.
        # Two concurrent turns with the same resume= would race the transcript.
        self._send_lock: asyncio.Lock = asyncio.Lock()

    @property
    def tool_events(self) -> asyncio.Queue[ToolEvent]:
        """Queue of real-time tool events emitted via SDK hooks."""
        return self._tool_event_queue

    def _options_with_hooks(self) -> ClaudeAgentOptions:
        """Return base options with tool-event hooks merged in.

        Appends to any existing hooks (e.g. from OTel) rather than replacing.
        Uses ``dataclasses.replace`` to produce a new options instance,
        leaving ``_base_options`` untouched.
        """
        import dataclasses

        tool_hooks = _build_tool_hooks(self._tool_event_queue)
        existing: dict[HookEvent, list[HookMatcher]] = self._base_options.hooks or {}
        merged: dict[HookEvent, list[HookMatcher]] = dict(existing)
        for event_name, matchers in tool_hooks.items():
            if event_name in merged:
                merged[event_name] = list(merged[event_name]) + matchers
            else:
                merged[event_name] = matchers

        try:
            return dataclasses.replace(self._base_options, hooks=merged)
        except TypeError:
            # Fallback for non-dataclass options (e.g. test mocks)
            self._base_options.hooks = merged
            return self._base_options

    def _options_for_agui(self, input_data: RunAgentInput) -> ClaudeAgentOptions:
        """Return per-run Claude options for an AG-UI request.

        This keeps HoloDeck's base options authoritative while adding only the
        AG-UI-specific runtime surface: partial streaming, state/context prompt
        addendum, dynamic frontend tools, and the internal state update tool.
        """
        options = self._base_options
        fields = {field.name for field in dataclasses.fields(ClaudeAgentOptions)}

        updates: dict[str, Any] = {"include_partial_messages": True}

        if self._sdk_session_id is not None:
            updates["resume"] = self._sdk_session_id

        if isinstance(input_data.forwarded_props, dict):
            for key, value in input_data.forwarded_props.items():
                if key in _AGUI_FORWARDED_PROPS and key in fields and value is not None:
                    updates[key] = value
                elif key not in _AGUI_FORWARDED_PROPS:
                    logger.warning("Ignoring unsupported AG-UI forwardedProp: %s", key)

        addendum = _agui_state_context_addendum(input_data)
        if addendum:
            base_prompt = updates.get("system_prompt", options.system_prompt) or ""
            updates["system_prompt"] = (
                f"{base_prompt}\n\n{addendum}" if base_prompt else addendum
            )

        frontend_tools = [_make_agui_frontend_tool(t) for t in input_data.tools or []]
        agui_tools = [*frontend_tools, _make_agui_state_tool()]

        allowed_tools = list(options.allowed_tools or [])
        for name in _agui_tool_names(input_data.tools):
            full_name = f"mcp__{_AGUI_MCP_SERVER_NAME}__{name}"
            if full_name not in allowed_tools:
                allowed_tools.append(full_name)
        if _AGUI_STATE_TOOL_FULL_NAME not in allowed_tools:
            allowed_tools.append(_AGUI_STATE_TOOL_FULL_NAME)
        updates["allowed_tools"] = allowed_tools

        if agui_tools:
            from claude_agent_sdk import create_sdk_mcp_server

            existing_servers = (
                dict(options.mcp_servers)
                if isinstance(options.mcp_servers, dict)
                else {}
            )
            existing_servers[_AGUI_MCP_SERVER_NAME] = create_sdk_mcp_server(
                _AGUI_MCP_SERVER_NAME,
                "1.0.0",
                tools=agui_tools,
            )
            updates["mcp_servers"] = existing_servers

        return _clone_options(options, **updates)

    async def prepare(self) -> None:
        """No-op under spec 034 P4.

        Retained for backwards compatibility with the chat executor's
        ``_TaskBoundSession``. Under the hybrid-session model the SDK's
        anyio task group is created inside each ``query()`` call frame,
        so there is no task-binding to do up front.
        """
        return None

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and collect the full response.

        Args:
            message: User message text.

        Returns:
            ``ExecutionResult`` with the agent's response.

        Raises:
            BackendSessionError: On subprocess or SDK error.
        """
        async with self._send_lock:
            turn_no = self._turn_count + 1
            logger.debug(
                "[trace] ClaudeSession.send turn=%d: entering, resume=%s",
                turn_no,
                self._sdk_session_id,
            )
            try:
                options = self._options_with_hooks()
                if self._sdk_session_id is not None:
                    import dataclasses

                    try:
                        options = dataclasses.replace(
                            options, resume=self._sdk_session_id
                        )
                    except TypeError:
                        # Fallback for non-dataclass options (test mocks).
                        options.resume = self._sdk_session_id

                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []
                thinking_parts: list[str] = []
                token_usage = TokenUsage.zero()
                num_turns = 1
                structured_output: Any = None

                msg_count = 0
                async for msg in claude_agent_sdk.query(
                    prompt=_streaming_user_envelope(message), options=options
                ):
                    msg_count += 1
                    logger.debug(
                        "[trace] ClaudeSession.send turn=%d: msg #%d type=%s",
                        turn_no,
                        msg_count,
                        msg.__class__.__name__,
                    )
                    _maybe_emit_thinking_blocks(msg, self._tool_event_queue)
                    _maybe_emit_subagent_message(msg, self._tool_event_queue)
                    text_parts, tool_calls, tool_results, thinking_parts = (
                        _process_message(
                            msg,
                            text_parts,
                            tool_calls,
                            tool_results,
                            thinking_parts,
                        )
                    )
                    if msg.__class__.__name__ == "ResultMessage":
                        rm = cast(Any, msg)
                        usage = rm.usage or {}
                        prompt_tokens = usage.get("input_tokens", 0)
                        completion = usage.get("output_tokens", 0)
                        token_usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion,
                            total_tokens=prompt_tokens + completion,
                        )
                        num_turns = rm.num_turns
                        structured_output = rm.structured_output
                        if self._sdk_session_id is None:
                            captured = getattr(rm, "session_id", None)
                            if isinstance(captured, str) and captured:
                                self._sdk_session_id = captured

                logger.debug(
                    "[trace] ClaudeSession.send turn=%d: exited, "
                    "msg_count=%d, num_turns=%d, sdk_session_id=%s",
                    turn_no,
                    msg_count,
                    num_turns,
                    self._sdk_session_id,
                )
                self._turn_count += 1

                _enrich_tool_results(tool_calls, tool_results)

                response_text = "".join(text_parts)
                if structured_output is not None:
                    response_text = json.dumps(structured_output)

                return ExecutionResult(
                    response=response_text,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    token_usage=token_usage,
                    structured_output=structured_output,
                    num_turns=num_turns,
                    thinking="".join(thinking_parts),
                )
            except (ProcessError, CLIConnectionError) as exc:
                raise BackendSessionError(
                    f"subprocess terminated unexpectedly: {exc}"
                ) from exc

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and yield text chunks progressively.

        Args:
            message: User message text.

        Yields:
            Text chunks as they arrive from the SDK.

        Raises:
            BackendSessionError: On subprocess or SDK error.
        """
        async with self._send_lock:
            options = self._options_with_hooks()
            if self._sdk_session_id is not None:
                import dataclasses

                try:
                    options = dataclasses.replace(options, resume=self._sdk_session_id)
                except TypeError:
                    # Fallback for non-dataclass options (test mocks).
                    options.resume = self._sdk_session_id

            try:
                async for msg in claude_agent_sdk.query(
                    prompt=_streaming_user_envelope(message), options=options
                ):
                    _maybe_emit_thinking_blocks(msg, self._tool_event_queue)
                    _maybe_emit_subagent_message(msg, self._tool_event_queue)
                    if msg.__class__.__name__ == "AssistantMessage":
                        for block in cast(Any, msg).content:
                            if block.__class__.__name__ == "TextBlock" and block.text:
                                yield block.text
                    elif msg.__class__.__name__ == "ResultMessage":
                        self._turn_count += 1
                        if self._sdk_session_id is None:
                            captured = getattr(msg, "session_id", None)
                            if isinstance(captured, str) and captured:
                                self._sdk_session_id = captured
            except (ProcessError, CLIConnectionError) as exc:
                raise BackendSessionError(
                    f"subprocess terminated unexpectedly: {exc}"
                ) from exc

    async def send_agui(
        self,
        input_data: RunAgentInput,
        message_override: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Send an AG-UI request through this Claude session.

        Yields AG-UI events translated directly from Claude SDK stream messages.
        """
        async with self._send_lock:
            thread_id = input_data.thread_id
            run_id = input_data.run_id
            options = self._options_for_agui(input_data)
            prompt = _agui_prompt_from_input(input_data, message_override)
            frontend_tool_names = set(_agui_tool_names(input_data.tools))
            current_state = input_data.state
            result_data: dict[str, Any] | None = None

            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=thread_id,
                run_id=run_id,
                parent_run_id=input_data.parent_run_id,
                input=input_data,
            )
            if input_data.state is not None:
                yield StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT,
                    snapshot=input_data.state,
                )

            try:
                message_stream = claude_agent_sdk.query(
                    prompt=_streaming_user_envelope(prompt),
                    options=options,
                )
                async for event in self._stream_agui_messages(
                    message_stream=message_stream,
                    thread_id=thread_id,
                    run_id=run_id,
                    input_data=input_data,
                    frontend_tool_names=frontend_tool_names,
                    current_state=current_state,
                ):
                    if isinstance(event, _AguiStreamStateUpdate):
                        current_state = event.state
                        continue
                    if isinstance(event, _AguiStreamResult):
                        result_data = event.result
                        continue
                    yield event

                self._turn_count += 1
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=thread_id,
                    run_id=run_id,
                    result=result_data,
                )
            except (ProcessError, CLIConnectionError) as exc:
                raise BackendSessionError(
                    f"subprocess terminated unexpectedly: {exc}"
                ) from exc

    async def _stream_agui_messages(
        self,
        *,
        message_stream: AsyncIterable[object],
        thread_id: str,
        run_id: str,
        input_data: RunAgentInput,
        frontend_tool_names: set[str],
        current_state: Any,
    ) -> AsyncGenerator[BaseEvent | _AguiStreamStateUpdate | _AguiStreamResult, None]:
        """Translate Claude SDK message stream to AG-UI events."""
        current_message_id: str | None = None
        has_streamed_text = False
        in_reasoning = False
        reasoning_message_id: str | None = None
        accumulated_signature = ""
        current_tool_call_id: str | None = None
        current_tool_call_name: str | None = None
        current_tool_display_name: str | None = None
        accumulated_tool_json = ""
        processed_tool_ids: set[str] = set()
        halt_stream = False
        run_messages: list[AguiAssistantMessage | AguiToolMessage] = []
        pending_msg: dict[str, Any] | None = None

        def upsert_message(message: AguiAssistantMessage | AguiToolMessage) -> None:
            msg_id = message.id
            for idx, existing in enumerate(run_messages):
                if existing.id == msg_id:
                    run_messages[idx] = message
                    return
            run_messages.append(message)

        def flush_pending_msg() -> None:
            nonlocal pending_msg
            if pending_msg is None:
                return
            has_content = bool(pending_msg.get("content"))
            has_tools = bool(pending_msg.get("tool_calls"))
            if has_content or has_tools:
                upsert_message(
                    AguiAssistantMessage(
                        id=pending_msg["id"],
                        role="assistant",
                        content=pending_msg["content"] if has_content else None,
                        tool_calls=pending_msg["tool_calls"] if has_tools else None,
                    )
                )
            pending_msg = None

        async for message in message_stream:
            if halt_stream:
                await _close_message_stream(message_stream)
                break

            cls_name = message.__class__.__name__

            if cls_name == "StreamEvent":
                event_data = getattr(message, "event", {}) or {}
                event_type = event_data.get("type")

                if event_type == "message_start":
                    current_message_id = str(ULID())
                    has_streamed_text = False
                    pending_msg = {
                        "id": current_message_id,
                        "content": "",
                        "tool_calls": [],
                    }

                elif event_type == "content_block_start":
                    block_data = event_data.get("content_block", {}) or {}
                    block_type = block_data.get("type")
                    if block_type == "thinking":
                        in_reasoning = True
                        reasoning_message_id = str(ULID())
                        yield ReasoningStartEvent(
                            type=EventType.REASONING_START,
                            message_id=reasoning_message_id,
                        )
                        yield ReasoningMessageStartEvent(
                            type=EventType.REASONING_MESSAGE_START,
                            message_id=reasoning_message_id,
                            role="reasoning",
                        )
                    elif block_type == "tool_use":
                        current_tool_call_id = block_data.get("id")
                        current_tool_call_name = block_data.get("name", "unknown")
                        current_tool_display_name = _strip_mcp_prefix(
                            current_tool_call_name
                        )
                        accumulated_tool_json = ""
                        if current_tool_call_id:
                            processed_tool_ids.add(current_tool_call_id)
                            yield ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=current_tool_call_id,
                                tool_call_name=current_tool_display_name,
                                parent_message_id=current_message_id,
                            )

                elif event_type == "content_block_delta":
                    delta_data = event_data.get("delta", {}) or {}
                    delta_type = delta_data.get("type")
                    if delta_type == "text_delta":
                        text = _fix_surrogates(delta_data.get("text", ""))
                        if text and current_message_id:
                            if not has_streamed_text:
                                yield TextMessageStartEvent(
                                    type=EventType.TEXT_MESSAGE_START,
                                    message_id=current_message_id,
                                    role="assistant",
                                )
                            has_streamed_text = True
                            if pending_msg is not None:
                                pending_msg["content"] += text
                            yield TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=current_message_id,
                                delta=text,
                            )
                    elif delta_type == "thinking_delta":
                        text = _fix_surrogates(delta_data.get("thinking", ""))
                        if text and reasoning_message_id:
                            yield ReasoningMessageContentEvent(
                                type=EventType.REASONING_MESSAGE_CONTENT,
                                message_id=reasoning_message_id,
                                delta=text,
                            )
                    elif delta_type == "signature_delta":
                        accumulated_signature += delta_data.get("signature", "")
                    elif delta_type == "input_json_delta":
                        partial_json = delta_data.get("partial_json", "")
                        if partial_json and current_tool_call_id:
                            accumulated_tool_json += partial_json
                            yield ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=current_tool_call_id,
                                delta=_fix_surrogates(partial_json),
                            )

                elif event_type == "content_block_stop":
                    if in_reasoning and reasoning_message_id:
                        yield ReasoningMessageEndEvent(
                            type=EventType.REASONING_MESSAGE_END,
                            message_id=reasoning_message_id,
                        )
                        yield ReasoningEndEvent(
                            type=EventType.REASONING_END,
                            message_id=reasoning_message_id,
                        )
                        if accumulated_signature and current_message_id:
                            yield ReasoningEncryptedValueEvent(
                                type=EventType.REASONING_ENCRYPTED_VALUE,
                                subtype="message",
                                entity_id=current_message_id,
                                encrypted_value=accumulated_signature,
                            )
                        in_reasoning = False
                        reasoning_message_id = None
                        accumulated_signature = ""

                    if current_tool_call_id:
                        if _is_agui_state_tool(current_tool_call_name):
                            try:
                                parsed = json.loads(
                                    _fix_surrogates(accumulated_tool_json)
                                )
                                updates = parsed.get("state_updates", parsed)
                                if isinstance(updates, str):
                                    updates = json.loads(updates)
                                if isinstance(current_state, dict) and isinstance(
                                    updates, dict
                                ):
                                    current_state = {**current_state, **updates}
                                else:
                                    current_state = updates
                                current_state = _fix_surrogates_deep(current_state)
                                yield StateSnapshotEvent(
                                    type=EventType.STATE_SNAPSHOT,
                                    snapshot=current_state,
                                )
                                yield _AguiStreamStateUpdate(state=current_state)
                            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                                yield CustomEvent(
                                    type=EventType.CUSTOM,
                                    name="state_update_error",
                                    value={"error": str(exc)},
                                )
                        elif pending_msg is not None and current_tool_display_name:
                            pending_msg["tool_calls"].append(
                                AguiToolCall(
                                    id=current_tool_call_id,
                                    type="function",
                                    function=AguiFunctionCall(
                                        name=current_tool_display_name,
                                        arguments=_fix_surrogates(
                                            accumulated_tool_json
                                        ),
                                    ),
                                )
                            )

                        yield ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=current_tool_call_id,
                        )
                        if current_tool_display_name in frontend_tool_names:
                            flush_pending_msg()
                            if current_message_id and has_streamed_text:
                                yield TextMessageEndEvent(
                                    type=EventType.TEXT_MESSAGE_END,
                                    message_id=current_message_id,
                                )
                                current_message_id = None
                                has_streamed_text = False
                            halt_stream = True
                        current_tool_call_id = None
                        current_tool_call_name = None
                        current_tool_display_name = None
                        accumulated_tool_json = ""

                elif event_type == "message_stop":
                    flush_pending_msg()
                    if current_message_id and has_streamed_text:
                        yield TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=current_message_id,
                        )
                    current_message_id = None
                if halt_stream:
                    await _close_message_stream(message_stream)
                    break
                continue

            if cls_name == "AssistantMessage":
                msg_id = current_message_id or str(ULID())
                agui_message = _agui_assistant_message_from_sdk(message, msg_id)
                if agui_message is not None:
                    upsert_message(agui_message)

                for tool_block in _agui_sdk_tool_use_blocks(message):
                    if tool_block.tool_id in processed_tool_ids:
                        continue
                    if _is_agui_state_tool(tool_block.raw_name):
                        updates = tool_block.tool_input.get(
                            "state_updates", tool_block.tool_input
                        )
                        if isinstance(updates, str):
                            updates = json.loads(updates)
                        if isinstance(current_state, dict) and isinstance(
                            updates, dict
                        ):
                            current_state = {**current_state, **updates}
                        else:
                            current_state = updates
                        current_state = _fix_surrogates_deep(current_state)
                        yield StateSnapshotEvent(
                            type=EventType.STATE_SNAPSHOT,
                            snapshot=current_state,
                        )
                        yield _AguiStreamStateUpdate(state=current_state)
                        continue
                    processed_tool_ids.add(tool_block.tool_id)
                    yield ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tool_block.tool_id,
                        tool_call_name=tool_block.display_name,
                        parent_message_id=msg_id,
                    )
                    yield ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tool_block.tool_id,
                        delta=json.dumps(tool_block.tool_input),
                    )
                    yield ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_block.tool_id,
                    )
                    if tool_block.display_name in frontend_tool_names:
                        halt_stream = True
                        break

            elif cls_name == "UserMessage":
                for tool_result in _agui_sdk_tool_result_blocks(message):
                    upsert_message(
                        _agui_tool_message(
                            tool_result.tool_use_id,
                            tool_result.content,
                        )
                    )
                    yield ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        message_id=f"{tool_result.tool_use_id}-result",
                        tool_call_id=tool_result.tool_use_id,
                        content=_agui_tool_result_content(tool_result.content),
                        role="tool",
                    )

            elif cls_name == "SystemMessage":
                subtype = getattr(message, "subtype", "") or "unknown"
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name=f"system:{subtype}",
                    value=getattr(message, "data", {}) or {},
                )

            elif cls_name == "ResultMessage":
                result_data = {
                    "is_error": getattr(message, "is_error", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "duration_api_ms": getattr(message, "duration_api_ms", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "usage": getattr(message, "usage", None),
                    "structured_output": getattr(message, "structured_output", None),
                }
                captured = getattr(message, "session_id", None)
                if (
                    self._sdk_session_id is None
                    and isinstance(captured, str)
                    and captured
                ):
                    self._sdk_session_id = captured
                result_text = getattr(message, "result", None)
                if result_text and not has_streamed_text:
                    result_msg_id = str(ULID())
                    yield TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=result_msg_id,
                        role="assistant",
                    )
                    yield TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=result_msg_id,
                        delta=result_text,
                    )
                    yield TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=result_msg_id,
                    )
                    upsert_message(
                        AguiAssistantMessage(
                            id=result_msg_id,
                            role="assistant",
                            content=result_text,
                        )
                    )
                yield _AguiStreamResult(result=result_data)

            if halt_stream:
                await _close_message_stream(message_stream)
                break

        if current_tool_call_id:
            yield ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=current_tool_call_id,
            )
        if in_reasoning and reasoning_message_id:
            yield ReasoningMessageEndEvent(
                type=EventType.REASONING_MESSAGE_END,
                message_id=reasoning_message_id,
            )
            yield ReasoningEndEvent(
                type=EventType.REASONING_END,
                message_id=reasoning_message_id,
            )
        if has_streamed_text and current_message_id:
            yield TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=current_message_id,
            )
        flush_pending_msg()
        if run_messages:
            yield MessagesSnapshotEvent(
                type=EventType.MESSAGES_SNAPSHOT,
                messages=[*(input_data.messages or []), *run_messages],
            )

    async def release_transport(self) -> None:
        """No-op under spec 034 P4.

        Retained for backwards compatibility with the chat executor's
        ``_TaskBoundSession``. Under the hybrid-session model each turn's
        subprocess is created and torn down inside ``query()``; there is no
        persistent transport to release between turns.
        """
        return None

    async def close(self) -> None:
        """Delete the on-disk JSONL transcript and clear session state.

        Under spec 034 P4 the session has no persistent subprocess to
        disconnect. Conversation state lives on disk at
        ``~/.claude/projects/<encoded-cwd>/<sdk_session_id>.jsonl``. Closing
        the session permanently discards that transcript so the next
        open of the same threadId starts fresh.
        """
        if self._sdk_session_id is not None:
            cwd_value = getattr(self._base_options, "cwd", None)
            path = _transcript_path(self._sdk_session_id, cwd=cwd_value)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to delete transcript %s: %s", path, exc)
            self._sdk_session_id = None


# ---------------------------------------------------------------------------
# ClaudeBackend
# ---------------------------------------------------------------------------


class ClaudeBackend:
    """Backend implementation for the Claude Agent SDK.

    Implements the ``AgentBackend`` protocol. Both single-turn invocations
    and multi-turn sessions are built on the top-level ``query()`` function;
    multi-turn state is threaded via ``resume=<sdk_session_id>`` inside
    ``ClaudeSession``.

    The constructor stores config only — no I/O, no subprocess spawned.
    Initialization is deferred to ``initialize()`` (called lazily on first use).
    """

    def __init__(
        self,
        agent: Agent,
        tool_instances: dict[str, Any] | None = None,
        mode: str = "test",
    ) -> None:
        """Store configuration without performing any I/O.

        Args:
            agent: Agent configuration.
            tool_instances: Initialized vectorstore/hierarchical-doc tool instances.
            mode: Execution mode (``"test"`` or ``"chat"``).
        """
        self._agent = agent
        self._tool_instances = tool_instances or {}
        self._mode = mode
        self._initialized = False
        self._options: ClaudeAgentOptions | None = None
        self._owned_tools: list[Any] = []  # Tools created during initialize()
        self._instrumentor: Any = None

    async def _ensure_initialized(self) -> None:
        """Lazy-init guard — call ``initialize()`` if not yet done."""
        if not self._initialized:
            await self.initialize()

    async def initialize(self) -> None:
        """Initialize the backend — validate config, build options.

        Idempotent: calling multiple times is a no-op after the first.

        Raises:
            BackendInitError: On validation or configuration failure.
        """
        if self._initialized:
            return

        try:
            agent = self._agent
            claude = agent.claude

            # 1. Node.js prerequisite
            validate_nodejs(agent)

            # 2. Credentials
            auth_env = validate_credentials(agent.model)

            # 3. Embedding provider (vectorstore tools)
            validate_embedding_provider(agent)

            # 4. Auto-initialize vectorstore/hierarchical-doc tools if needed
            await self._initialize_tools()

            # 5. Tool adapters
            from pathlib import Path

            from holodeck.config.context import agent_base_dir as _agent_base_dir

            _base_dir_str = _agent_base_dir.get()
            _base_dir = Path(_base_dir_str) if _base_dir_str else None
            adapters = create_tool_adapters(
                tool_configs=agent.tools or [],
                tool_instances=self._tool_instances,
                base_dir=_base_dir,
            )
            tool_server, tool_names = build_holodeck_sdk_server(adapters)

            # 6. External MCP configs
            mcp_tools = [t for t in (agent.tools or []) if isinstance(t, MCPTool)]
            mcp_configs = build_claude_mcp_configs(mcp_tools)

            # 7. OTel env vars
            otel_env: dict[str, str] = {}
            if agent.observability:
                otel_env = translate_observability(agent.observability)

            # 8. Build options
            self._options = build_options(
                agent=agent,
                tool_server=tool_server if adapters else None,
                tool_names=tool_names,
                mcp_configs=mcp_configs,
                auth_env=auth_env,
                otel_env=otel_env,
                mode=self._mode,
            )

            # 9. Working directory collision check
            wd = claude.working_directory if claude else None
            validate_working_directory(wd)

            # 10. Response format validation
            validate_response_format(agent.response_format)

            # 11. GenAI instrumentation (optional, non-blocking)
            self._activate_instrumentation()

            self._initialized = True

        except Exception as exc:
            if not isinstance(exc, BackendInitError):
                raise BackendInitError(
                    f"Claude backend initialization failed: {exc}"
                ) from exc
            raise

    async def invoke_once(
        self, message: str, context: list[dict[str, Any]] | None = None
    ) -> ExecutionResult:
        """Invoke the agent for a single turn.

        Automatically initializes if not yet done. Retries on ``ProcessError``
        (subprocess crash) with exponential backoff.

        Args:
            message: User message text.
            context: Optional conversation context (unused for Claude backend).

        Returns:
            ``ExecutionResult`` with the agent's response.

        Raises:
            BackendSessionError: After max retries exhausted.
        """
        await self._ensure_initialized()

        last_error: BaseException | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._invoke_query(message)
            except (ProcessError, CLIConnectionError, MessageParseError) as exc:
                last_error = exc
            except BaseExceptionGroup as exc:
                # anyio TaskGroup wraps subprocess errors in ExceptionGroup
                last_error = exc

            if attempt < _MAX_RETRIES - 1:
                backoff = _BACKOFF_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Claude subprocess error (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                    last_error,
                )
                await asyncio.sleep(backoff)

        raise BackendSessionError(
            f"Claude subprocess failed after {_MAX_RETRIES} retries: {last_error}"
        )

    async def _invoke_query(self, message: str) -> ExecutionResult:
        """Execute a single query invocation.

        Args:
            message: User message text.

        Returns:
            ``ExecutionResult`` with response, tools, and token usage.
        """
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        thinking_parts: list[str] = []
        token_usage = TokenUsage.zero()
        num_turns = 1
        structured_output: Any = None

        prompt_iter = _wrap_prompt(message)
        async for msg in claude_agent_sdk.query(
            prompt=prompt_iter, options=self._options
        ):
            text_parts, tool_calls, tool_results, thinking_parts = _process_message(
                msg, text_parts, tool_calls, tool_results, thinking_parts
            )
            if msg.__class__.__name__ == "ResultMessage":
                rm = cast(Any, msg)
                usage = rm.usage or {}
                prompt = usage.get("input_tokens", 0)
                completion = usage.get("output_tokens", 0)
                token_usage = TokenUsage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                )
                num_turns = rm.num_turns
                structured_output = rm.structured_output

        # Enrich tool results with names from tool calls
        _enrich_tool_results(tool_calls, tool_results)

        response_text = "".join(text_parts)

        thinking_text = "".join(thinking_parts)

        # Structured output handling
        if structured_output is not None:
            result = self._validate_structured_output(structured_output, response_text)
            if result is not None:
                return ExecutionResult(
                    response=result["response"],
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    token_usage=token_usage,
                    structured_output=result.get("structured_output"),
                    num_turns=num_turns,
                    is_error=result.get("is_error", False),
                    error_reason=result.get("error_reason"),
                    thinking=thinking_text,
                )

        # Max turns exceeded detection
        max_turns = (
            self._agent.claude.max_turns
            if self._agent.claude and self._agent.claude.max_turns
            else None
        )
        if max_turns is not None and num_turns >= max_turns:
            return ExecutionResult(
                response=response_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                token_usage=token_usage,
                num_turns=num_turns,
                is_error=True,
                error_reason="max_turns limit reached",
                thinking=thinking_text,
            )

        return ExecutionResult(
            response=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            token_usage=token_usage,
            structured_output=structured_output,
            num_turns=num_turns,
            thinking=thinking_text,
        )

    def _validate_structured_output(
        self, output: Any, response_text: str
    ) -> dict[str, Any] | None:
        """Validate structured output against the configured JSON schema.

        Args:
            output: The structured output from ``ResultMessage``.
            response_text: The text response from content blocks.

        Returns:
            Dict with response/structured_output/error info, or ``None`` to skip.
        """
        rf = self._agent.response_format
        if rf is None or not isinstance(rf, dict):
            # No schema to validate against — pass through
            return {
                "response": json.dumps(output),
                "structured_output": output,
            }

        try:
            jsonschema.validate(instance=output, schema=rf)
            return {
                "response": json.dumps(output),
                "structured_output": output,
            }
        except jsonschema.ValidationError as exc:
            return {
                "response": response_text,
                "structured_output": output,
                "is_error": True,
                "error_reason": (
                    f"Structured output schema validation failed: {exc.message}"
                ),
            }

    async def create_session(self, *, eager_connect: bool = True) -> ClaudeSession:
        """Create a new multi-turn session.

        Automatically initializes if not yet done. Under spec 034 P4 the
        session no longer holds a persistent ``ClaudeSDKClient``; each
        turn opens its own subprocess via ``query(resume=session_id)``.
        ``eager_connect`` is retained as a no-op for API compatibility —
        ``ClaudeSession.prepare()`` is itself a no-op under P4.

        Args:
            eager_connect: Retained for backwards compatibility; has no
                effect under spec 034 P4.

        Returns:
            A new ``ClaudeSession`` instance.
        """
        await self._ensure_initialized()
        if self._options is None:
            raise BackendInitError("Backend options not set after initialization")
        session = ClaudeSession(options=self._options)
        if eager_connect:
            await session.prepare()
        return session

    async def _initialize_tools(self) -> None:
        """Initialize vectorstore/hierarchical-doc tools using shared module.

        Populates ``self._tool_instances`` so ``create_tool_adapters()`` can
        find them. Skips if no tools require initialization.
        """
        if not self._agent.tools:
            return

        has_vs = any(isinstance(t, VectorstoreToolConfig) for t in self._agent.tools)
        has_hd = any(
            isinstance(t, HierarchicalDocumentToolConfig) for t in self._agent.tools
        )
        if not has_vs and not has_hd:
            return

        from holodeck.config.context import agent_base_dir
        from holodeck.lib.tool_initializer import ToolInitializerError, initialize_tools

        try:
            instances = await initialize_tools(
                agent=self._agent,
                force_ingest=False,  ##TODO: figure out how to pass this down
                execution_config=self._agent.execution,
                base_dir=agent_base_dir.get(),
            )
            self._tool_instances = instances
            self._owned_tools = list(instances.values())
        except ToolInitializerError:
            raise
        except Exception as exc:
            raise BackendInitError(f"Failed to initialize tools: {exc}") from exc

    def _activate_instrumentation(self) -> None:
        """Activate GenAI semantic convention instrumentation if configured.

        Lazily imports ``ClaudeAgentSdkInstrumentor`` from the optional
        ``otel-instrumentation-claude-agent-sdk`` package. When observability
        is enabled with traces, instruments the Claude Agent SDK so that
        ``query()`` calls automatically emit GenAI-convention spans.

        This method is synchronous (``instrument()`` does monkey-patching,
        no I/O) and non-blocking — failures are logged as warnings without
        crashing initialization.
        """
        obs = self._agent.observability
        if obs is None or not obs.enabled or not obs.traces.enabled:
            return

        try:
            from opentelemetry.instrumentation.claude_agent_sdk import (
                ClaudeAgentSdkInstrumentor,
            )
        except ImportError:
            logger.warning(
                "otel-instrumentation-claude-agent-sdk not installed; "
                "GenAI instrumentation disabled. "
                "Install with: uv add holodeck[claude-otel]"
            )
            return

        try:
            ctx = get_observability_context()
            if ctx is None:
                logger.warning(
                    "Observability context not initialized; "
                    "GenAI instrumentation disabled"
                )
                return

            meter_provider = ctx.meter_provider if obs.metrics.enabled else None

            instrumentor = ClaudeAgentSdkInstrumentor()
            already = instrumentor.is_instrumented_by_opentelemetry
            logger.debug(
                "GenAI instrumentor: already_instrumented=%s, "
                "tracer_provider=%s, agent=%s",
                already,
                type(ctx.tracer_provider).__name__,
                self._agent.name,
            )
            instrumentor.instrument(
                tracer_provider=ctx.tracer_provider,
                meter_provider=meter_provider,
                agent_name=self._agent.name,
                capture_content=obs.traces.capture_content,
            )
            self._instrumentor = instrumentor
            logger.info(
                "GenAI instrumentation activated for agent '%s'",
                self._agent.name,
            )
        except Exception as exc:
            logger.warning("Failed to activate GenAI instrumentation: %s", exc)

    async def teardown(self) -> None:
        """Reset backend state, releasing any built options."""
        # Deactivate GenAI instrumentation
        if self._instrumentor is not None:
            try:
                self._instrumentor.uninstrument()
            except Exception as exc:
                logger.warning("Error deactivating GenAI instrumentation: %s", exc)
            self._instrumentor = None

        # Cleanup owned tools
        for tool_inst in self._owned_tools:
            if hasattr(tool_inst, "cleanup"):
                try:
                    await tool_inst.cleanup()
                except Exception as exc:
                    logger.warning("Error cleaning up tool: %s", exc)
        self._owned_tools = []
        self._tool_instances = {}
        self._initialized = False
        self._options = None
