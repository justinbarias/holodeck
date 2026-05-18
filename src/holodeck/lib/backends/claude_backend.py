"""Claude Agent SDK backend for HoloDeck.

Implements ``ClaudeBackend`` (AgentBackend) and ``ClaudeSession`` (AgentSession)
for the ``provider: anthropic`` execution path. Single-turn invocations use the
top-level ``query()`` function; multi-turn chat sessions use ``ClaudeSDKClient``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

import claude_agent_sdk
import jsonschema
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ProcessError,
)
from claude_agent_sdk._errors import CLIConnectionError, MessageParseError
from claude_agent_sdk.types import (
    AgentDefinition,
    HookContext,
    HookEvent,
    McpSdkServerConfig,
    SyncHookJSONOutput,
)
from exceptiongroup import BaseExceptionGroup

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
    validate_tool_filtering,
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

# Session ID passed to every ``client.query()`` for the lifetime of a
# connected ``ClaudeSDKClient``. The CLI subprocess tracks conversation
# state internally in interactive streaming mode; rotating to the
# CLI-assigned id surfaced on ``ResultMessage`` wedges the CLI on the
# next turn (query write succeeds but no response messages arrive).
_DEFAULT_SESSION_ID = "default"

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
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract text, tool calls, and tool results from SDK messages.

    Handles both ``AssistantMessage`` (text + tool-use blocks) and
    ``UserMessage`` (tool-result blocks from MCP tool execution).
    Mutates the provided lists in-place and also returns them for convenience.

    Args:
        msg: A message from the SDK response stream.
        text_parts: Accumulator for text content.
        tool_calls: Accumulator for tool call dicts.
        tool_results: Accumulator for tool result dicts.

    Returns:
        The (text_parts, tool_calls, tool_results) tuple.
    """
    msg_type = msg.__class__.__name__
    if msg_type not in ("AssistantMessage", "UserMessage"):
        return text_parts, tool_calls, tool_results

    for block in msg.content:
        cls_name = block.__class__.__name__
        if cls_name == "TextBlock":
            text_parts.append(block.text)
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
    return text_parts, tool_calls, tool_results


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
        Configured ``ClaudeAgentOptions`` ready for ``query()`` or ``ClaudeSDKClient``.
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

    return ClaudeAgentOptions(**opts_kwargs)


# ---------------------------------------------------------------------------
# ClaudeSession
# ---------------------------------------------------------------------------


def _patch_hooks_for_context_propagation(client: ClaudeSDKClient) -> None:
    """Wrap hook callbacks so they re-inject the ContextVar from the instance.

    Works around a ContextVar timing mismatch in the OTel instrumentor:
    ``connect()`` spawns a ``_read_messages`` background task *before*
    ``_wrap_client_query`` sets the ``InvocationContext`` in the ContextVar.
    asyncio copies ContextVars at task-creation time, so the background task
    sees ``None`` forever.

    The instrumentor's ``_wrap_client_query`` *does* store the context on the
    instance as ``_otel_invocation_ctx``.  This function wraps each hook
    callback so that, just before it runs, it reads from the instance attribute
    and re-sets the ContextVar — bridging the gap.

    No-op when the instrumentor package is not installed.
    """
    try:
        from opentelemetry.instrumentation.claude_agent_sdk._context import (
            set_invocation_context,
        )
    except ImportError:
        return  # Instrumentor not installed — nothing to patch

    options = getattr(client, "options", None)
    hooks = getattr(options, "hooks", None) if options else None
    if not hooks:
        return

    for matchers in hooks.values():
        if not matchers:
            continue
        for matcher in matchers:
            original_hooks = matcher.hooks if hasattr(matcher, "hooks") else []
            wrapped: list[Any] = []
            for hook_fn in original_hooks:

                async def _ctx_wrapper(
                    input_data: Any,
                    tool_use_id: str | None = None,
                    context: Any = None,
                    *,
                    _orig: Any = hook_fn,
                    _cli: ClaudeSDKClient = client,
                    **kwargs: Any,
                ) -> dict[str, Any]:
                    ctx = getattr(_cli, "_otel_invocation_ctx", None)
                    if ctx is not None:
                        set_invocation_context(ctx)
                    result: dict[str, Any] = await _orig(
                        input_data, tool_use_id, context, **kwargs
                    )
                    return result

                wrapped.append(_ctx_wrapper)
            if hasattr(matcher, "hooks"):
                matcher.hooks = wrapped


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
        input_data: Any,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        await queue.put(
            ToolEvent(
                kind="start",
                tool_name=input_data["tool_name"],
                tool_use_id=input_data["tool_use_id"],
                tool_input=input_data.get("tool_input"),
            )
        )
        return SyncHookJSONOutput()

    async def _on_post_tool(
        input_data: Any,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        await queue.put(
            ToolEvent(
                kind="end",
                tool_name=input_data["tool_name"],
                tool_use_id=input_data["tool_use_id"],
                tool_response=str(input_data.get("tool_response", "")),
            )
        )
        return SyncHookJSONOutput()

    async def _on_post_tool_failure(
        input_data: Any,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        await queue.put(
            ToolEvent(
                kind="error",
                tool_name=input_data["tool_name"],
                tool_use_id=input_data["tool_use_id"],
                error=str(input_data.get("error", "")),
            )
        )
        return SyncHookJSONOutput()

    return {
        "PreToolUse": [HookMatcher(hooks=[_on_pre_tool])],
        "PostToolUse": [HookMatcher(hooks=[_on_post_tool])],
        "PostToolUseFailure": [HookMatcher(hooks=[_on_post_tool_failure])],
    }


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


class ClaudeSession:
    """Stateful multi-turn session backed by ``ClaudeSDKClient``.

    Multi-turn state lives **inside the connected CLI subprocess**, not on
    this object. ``ClaudeSDKClient`` in interactive streaming mode keeps a
    single subprocess open for the session's lifetime; each ``send()``
    writes one user message on stdin and reads the response stream off
    stdout. The CLI tracks conversation history internally, so every
    ``client.query()`` for that connected lifetime uses
    ``session_id=_DEFAULT_SESSION_ID`` (the module-level constant) —
    rotating it to the CLI-assigned id surfaced on ``ResultMessage``
    wedges the CLI on the next turn (the query write succeeds but no
    response messages ever come back).

    The ``_base_options`` reference is **never mutated**. Turn-specific
    options are created as new ``ClaudeAgentOptions`` instances.
    """

    def __init__(self, options: ClaudeAgentOptions) -> None:
        """Initialize session with base options.

        Args:
            options: Base options (immutable reference for the session lifetime).
        """
        self._base_options = options
        self._client: ClaudeSDKClient | None = None
        self._turn_count: int = 0
        self._tool_event_queue: asyncio.Queue[ToolEvent] = asyncio.Queue()

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

    async def prepare(self) -> None:
        """Open the SDK transport in the calling task's context.

        Connects the underlying ``ClaudeSDKClient`` if not already
        connected. The SDK's anyio task group + ``_read_messages``
        background reader bind to whichever task calls ``connect()``,
        so callers needing a specific task to own that lifecycle should
        call ``prepare()`` from that task before the first ``send``.
        """
        await self._ensure_client()

    async def _ensure_client(self) -> ClaudeSDKClient:
        """Lazily create, connect, and return the SDK client.

        Uses ``claude_agent_sdk.ClaudeSDKClient`` (module-level attribute
        access) instead of the locally-imported name so that wrapt
        monkey-patches applied by the OTel instrumentor are resolved at
        call time.

        After ``__init__`` (which triggers the instrumentor's hook injection)
        but before ``connect()`` (which spawns the ``_read_messages`` task),
        we patch hook callbacks to re-inject the ContextVar from the instance.
        See ``_patch_hooks_for_context_propagation`` for details.
        """
        if self._client is None:
            options = self._options_with_hooks()
            self._client = claude_agent_sdk.ClaudeSDKClient(options=options)
            _patch_hooks_for_context_propagation(self._client)
            await self._client.connect()
        return self._client

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and collect the full response.

        Args:
            message: User message text.

        Returns:
            ``ExecutionResult`` with the agent's response.

        Raises:
            BackendSessionError: On subprocess or SDK error.
        """
        turn_no = self._turn_count + 1
        logger.debug(
            "[trace] ClaudeSession.send turn=%d: entering, client_cached=%s",
            turn_no,
            self._client is not None,
        )
        try:
            client = await self._ensure_client()
            logger.debug(
                "[trace] ClaudeSession.send turn=%d: client ready, calling query",
                turn_no,
            )
            await client.query(message, session_id=_DEFAULT_SESSION_ID)
            logger.debug(
                "[trace] ClaudeSession.send turn=%d: query written, awaiting "
                "receive_response",
                turn_no,
            )

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            token_usage = TokenUsage.zero()
            num_turns = 1
            structured_output: Any = None

            msg_count = 0
            async for msg in client.receive_response():
                msg_count += 1
                logger.debug(
                    "[trace] ClaudeSession.send turn=%d: msg #%d type=%s",
                    turn_no,
                    msg_count,
                    msg.__class__.__name__,
                )
                _maybe_emit_subagent_message(msg, self._tool_event_queue)
                text_parts, tool_calls, tool_results = _process_message(
                    msg, text_parts, tool_calls, tool_results
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

            logger.debug(
                "[trace] ClaudeSession.send turn=%d: receive_response exited, "
                "msg_count=%d, num_turns=%d",
                turn_no,
                msg_count,
                num_turns,
            )
            self._turn_count += 1

            # Enrich tool results with names from tool calls so downstream
            # consumers (eval_kwargs_builder.build_retrieval_context_from_tools)
            # can match each result to its source tool. The SDK yields names
            # only on ToolUseBlock; results arrive on ToolResultBlock carrying
            # just call_id.
            _enrich_tool_results(tool_calls, tool_results)

            response_text = "".join(text_parts)

            # When ``response_format`` is set the SDK delivers the validated
            # payload on ``ResultMessage.structured_output``; the text
            # content blocks carry the model's prose reasoning, which the
            # CLI does NOT constrain to the schema. The structured payload
            # is the authoritative answer — prefer it so downstream graders
            # (response_path, equality/numeric over JSON envelopes) operate
            # on the schema-validated value, not the unconstrained prose.
            # Mirrors ``invoke_once`` which routes through
            # ``_validate_structured_output``.
            if structured_output is not None:
                response_text = json.dumps(structured_output)

            return ExecutionResult(
                response=response_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                token_usage=token_usage,
                structured_output=structured_output,
                num_turns=num_turns,
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
        try:
            client = await self._ensure_client()
            await client.query(message, session_id=_DEFAULT_SESSION_ID)

            async for msg in client.receive_response():
                _maybe_emit_subagent_message(msg, self._tool_event_queue)
                if msg.__class__.__name__ == "AssistantMessage":
                    for block in cast(Any, msg).content:
                        if block.__class__.__name__ == "TextBlock" and block.text:
                            yield block.text
                elif msg.__class__.__name__ == "ResultMessage":
                    self._turn_count += 1
        except (ProcessError, CLIConnectionError) as exc:
            raise BackendSessionError(
                f"subprocess terminated unexpectedly: {exc}"
            ) from exc

    async def release_transport(self) -> None:
        """Disconnect the underlying ``ClaudeSDKClient`` connection.

        After calling this, the next ``send()`` / ``send_streaming()`` call
        will create and connect a fresh ``ClaudeSDKClient``. Note that the
        new client starts a new CLI subprocess with no awareness of the
        prior conversation — any conversation continuity needs to be
        rebuilt at the application layer (e.g., by replaying history).

        Provided primarily for cross-task migration scenarios (the SDK's
        anyio task group is bound to the task that called ``connect()``)
        and for explicit resource release in tests; it is **not** part of
        the normal per-turn flow in ``holodeck serve`` — that path keeps
        one connected client for the session's lifetime via
        ``_TaskBoundSession``.
        """
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def close(self) -> None:
        """Disconnect the SDK client and release resources."""
        await self.release_transport()


# ---------------------------------------------------------------------------
# ClaudeBackend
# ---------------------------------------------------------------------------


class ClaudeBackend:
    """Backend implementation for the Claude Agent SDK.

    Implements the ``AgentBackend`` protocol. Single-turn invocations use the
    top-level ``query()`` function. Multi-turn sessions use ``ClaudeSession``
    wrapping ``ClaudeSDKClient``.

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
            validate_nodejs()

            # 2. Credentials
            auth_env = validate_credentials(agent.model)

            # 3. Embedding provider (vectorstore tools)
            validate_embedding_provider(agent)

            # 4. Tool filtering (warning only)
            validate_tool_filtering(agent)

            # 4b. Auto-initialize vectorstore/hierarchical-doc tools if needed
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
        token_usage = TokenUsage.zero()
        num_turns = 1
        structured_output: Any = None

        prompt_iter = _wrap_prompt(message)
        async for msg in claude_agent_sdk.query(
            prompt=prompt_iter, options=self._options
        ):
            text_parts, tool_calls, tool_results = _process_message(
                msg, text_parts, tool_calls, tool_results
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
            )

        return ExecutionResult(
            response=response_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            token_usage=token_usage,
            structured_output=structured_output,
            num_turns=num_turns,
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

        Automatically initializes if not yet done. By default eagerly
        connects the SDK client so the anyio cancel scope is entered on
        the caller's task — required for non-actor callers because
        deferring ``connect()`` to the first ``send()`` would bind the
        scope to whatever task wraps ``send`` (e.g. an inner task spawned
        by ``asyncio.wait_for`` on Python <3.11) and a later ``close()``
        on the outer task would raise ``RuntimeError: Attempted to exit
        cancel scope in a different task``.

        Pass ``eager_connect=False`` when the caller will own the
        connect/disconnect lifecycle in a different task (e.g. when
        wrapping the session in ``_TaskBoundSession`` — that actor must
        be the task that calls ``connect()`` so the SDK's anyio task
        group + ``_read_messages`` background reader live as long as
        the actor, not the HTTP request task that created the session).

        Args:
            eager_connect: When True (default), connect the SDK client
                synchronously before returning. When False, return an
                unconnected session; the first ``send()`` will connect.

        Returns:
            A new ``ClaudeSession`` instance, connected unless
            ``eager_connect=False``.
        """
        await self._ensure_initialized()
        if self._options is None:
            raise BackendInitError("Backend options not set after initialization")
        session = ClaudeSession(options=self._options)
        if eager_connect:
            await session._ensure_client()
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
