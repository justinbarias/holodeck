"""Claude Agent SDK backend for HoloDeck.

Implements ``ClaudeBackend`` (AgentBackend) and ``ClaudeSession`` (AgentSession)
for the ``provider: anthropic`` execution path. Single-turn invocations use the
top-level ``query()`` function; multi-turn chat sessions use ``ClaudeSDKClient``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

import jsonschema
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ProcessError,
    query,
)
from claude_agent_sdk.types import McpSdkServerConfig

from holodeck.lib.backends.base import (
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
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
from holodeck.lib.instruction_resolver import resolve_instructions
from holodeck.models.agent import Agent
from holodeck.models.claude_config import PermissionMode
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool import MCPTool

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _process_assistant(
    msg: Any,
    text_parts: list[str],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract text, tool calls, and tool results from an AssistantMessage.

    Mutates the provided lists in-place and also returns them for convenience.
    Skips non-AssistantMessage types.

    Args:
        msg: A message from the SDK response stream.
        text_parts: Accumulator for text content.
        tool_calls: Accumulator for tool call dicts.
        tool_results: Accumulator for tool result dicts.

    Returns:
        The (text_parts, tool_calls, tool_results) tuple.
    """
    if msg.__class__.__name__ != "AssistantMessage":
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


def _build_permission_mode(
    permission_mode: PermissionMode | None,
    mode: str,
    allow_side_effects: bool,
) -> str | None:
    """Map HoloDeck permission mode to SDK permission literal.

    Args:
        permission_mode: HoloDeck ``PermissionMode`` enum value, or ``None``.
        mode: Execution mode (``"test"`` or ``"chat"``).
        allow_side_effects: Whether side effects are allowed in test mode.

    Returns:
        SDK permission mode string, or ``None`` if not configured.
    """
    if permission_mode is None:
        return None

    mapping: dict[PermissionMode, str] = {
        PermissionMode.manual: "default",
        PermissionMode.acceptEdits: "acceptEdits",
        PermissionMode.acceptAll: "bypassPermissions",
    }

    sdk_mode = mapping[permission_mode]

    # In test mode, override non-manual to bypassPermissions for automation
    if mode == "test" and permission_mode != PermissionMode.manual:
        sdk_mode = "bypassPermissions"

    return sdk_mode


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


def build_options(
    *,
    agent: Agent,
    tool_server: McpSdkServerConfig | None,
    tool_names: list[str],
    mcp_configs: dict[str, Any],
    auth_env: dict[str, str],
    otel_env: dict[str, str],
    mode: str,
    allow_side_effects: bool,
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
        allow_side_effects: Whether side effects are allowed in test mode.

    Returns:
        Configured ``ClaudeAgentOptions`` ready for ``query()`` or ``ClaudeSDKClient``.
    """
    claude = agent.claude
    system_prompt = resolve_instructions(agent.instructions)

    # MCP servers
    mcp_servers: dict[str, Any] = dict(mcp_configs)
    if tool_server is not None:
        mcp_servers["holodeck_tools"] = tool_server

    # Env vars
    env: dict[str, str] = {**auth_env, **otel_env}

    # Permission mode
    perm_mode = None
    if claude is not None:
        perm_mode = _build_permission_mode(
            claude.permission_mode, mode, allow_side_effects
        )

    # Extended thinking
    max_thinking_tokens = None
    if claude and claude.extended_thinking and claude.extended_thinking.enabled:
        max_thinking_tokens = claude.extended_thinking.budget_tokens

    # Allowed tools
    allowed_tools: list[str] = list(tool_names)
    if claude and claude.allowed_tools:
        allowed_tools.extend(claude.allowed_tools)

    # Output format
    output_format = _build_output_format(agent.response_format)

    # Working directory
    cwd = claude.working_directory if claude else None

    # Max turns
    max_turns = claude.max_turns if claude else None

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

    if max_thinking_tokens is not None:
        opts_kwargs["max_thinking_tokens"] = max_thinking_tokens

    return ClaudeAgentOptions(**opts_kwargs)


# ---------------------------------------------------------------------------
# ClaudeSession
# ---------------------------------------------------------------------------


class ClaudeSession:
    """Stateful multi-turn session backed by ``ClaudeSDKClient``.

    Each session maintains a ``session_id`` across turns. Multi-turn state is
    opt-in: after the first turn, subsequent turns pass
    ``continue_conversation=True`` and ``resume=session_id`` to the SDK.

    The ``_base_options`` reference is **never mutated**. Turn-specific options
    are created as new ``ClaudeAgentOptions`` instances.
    """

    def __init__(self, options: ClaudeAgentOptions) -> None:
        """Initialize session with base options.

        Args:
            options: Base options (immutable reference for the session lifetime).
        """
        self._base_options = options
        self._session_id: str | None = None
        self._client: ClaudeSDKClient | None = None
        self._turn_count: int = 0

    async def _ensure_client(self) -> ClaudeSDKClient:
        """Lazily create and return the SDK client."""
        if self._client is None:
            self._client = ClaudeSDKClient(options=self._base_options)
        return self._client

    def _build_turn_options(self) -> ClaudeAgentOptions | None:
        """Build options for the current turn.

        Turn 1 returns ``None`` (use base options). Turn 2+ returns a NEW
        ``ClaudeAgentOptions`` with ``continue_conversation=True`` and
        ``resume=session_id``.
        """
        if self._turn_count == 0:
            return None  # Use base options

        return ClaudeAgentOptions(
            continue_conversation=True,
            resume=self._session_id,
        )

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and collect the full response.

        Args:
            message: User message text.

        Returns:
            ``ExecutionResult`` with the agent's response.

        Raises:
            BackendSessionError: On subprocess or SDK error.
        """
        try:
            client = await self._ensure_client()
            turn_opts = self._build_turn_options()

            if turn_opts is not None:
                await client.query(message, options=turn_opts)  # type: ignore[call-arg]
            else:
                await client.query(message)

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            token_usage = TokenUsage.zero()
            num_turns = 1

            async for msg in client.receive_response():
                text_parts, tool_calls, tool_results = _process_assistant(
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
                    self._session_id = rm.session_id

            self._turn_count += 1

            return ExecutionResult(
                response="".join(text_parts),
                tool_calls=tool_calls,
                tool_results=tool_results,
                token_usage=token_usage,
                num_turns=num_turns,
            )
        except ProcessError as exc:
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
            turn_opts = self._build_turn_options()

            if turn_opts is not None:
                await client.query(message, options=turn_opts)  # type: ignore[call-arg]
            else:
                await client.query(message)

            async for msg in client.receive_response():
                if msg.__class__.__name__ == "AssistantMessage":
                    for block in cast(Any, msg).content:
                        if block.__class__.__name__ == "TextBlock" and block.text:
                            yield block.text
                elif msg.__class__.__name__ == "ResultMessage":
                    rm = cast(Any, msg)
                    self._session_id = rm.session_id
                    self._turn_count += 1
        except ProcessError as exc:
            raise BackendSessionError(
                f"subprocess terminated unexpectedly: {exc}"
            ) from exc

    async def close(self) -> None:
        """Disconnect the SDK client and release resources."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None


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
        allow_side_effects: bool = False,
    ) -> None:
        """Store configuration without performing any I/O.

        Args:
            agent: Agent configuration.
            tool_instances: Initialized vectorstore/hierarchical-doc tool instances.
            mode: Execution mode (``"test"`` or ``"chat"``).
            allow_side_effects: Allow bash/file_system.write in test mode.
        """
        self._agent = agent
        self._tool_instances = tool_instances
        self._mode = mode
        self._allow_side_effects = allow_side_effects
        self._initialized = False
        self._options: ClaudeAgentOptions | None = None

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

            # 5. Tool adapters
            adapters = create_tool_adapters(
                tool_configs=agent.tools or [],
                tool_instances=self._tool_instances or {},
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
                allow_side_effects=self._allow_side_effects,
            )

            # 9. Working directory collision check
            wd = claude.working_directory if claude else None
            validate_working_directory(wd)

            # 10. Response format validation
            validate_response_format(agent.response_format)

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

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._invoke_query(message)
            except ProcessError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BACKOFF_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        "Claude subprocess error (attempt %d/%d), "
                        "retrying in %ds: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                        exc,
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

        async for msg in query(prompt=message, options=self._options):
            text_parts, tool_calls, tool_results = _process_assistant(
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
                    "Structured output schema validation failed: " f"{exc.message}"
                ),
            }

    async def create_session(self) -> ClaudeSession:
        """Create a new multi-turn session.

        Automatically initializes if not yet done.

        Returns:
            A new ``ClaudeSession`` instance.
        """
        await self._ensure_initialized()
        if self._options is None:
            raise BackendInitError("Backend options not set after initialization")
        return ClaudeSession(options=self._options)

    async def teardown(self) -> None:
        """Reset backend state, releasing any built options."""
        self._initialized = False
        self._options = None
