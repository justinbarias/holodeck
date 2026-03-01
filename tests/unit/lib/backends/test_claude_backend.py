"""Unit tests for holodeck.lib.backends.claude_backend (Phase 8A TDD).

Tests T001–T016 covering:
- build_options() — option assembly from Agent + ClaudeConfig
- Permission mode mapping — HoloDeck enum → SDK literals
- ClaudeBackend init, initialize, invoke_once, teardown
- ClaudeSession send, send_streaming, close, multi-turn state
- Retry logic, structured output validation, lazy-init guard
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from exceptiongroup import BaseExceptionGroup

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.lib.backends.claude_backend import (
    ClaudeBackend,
    ClaudeSession,
    _build_output_format,
    _build_permission_mode,
    _enrich_tool_results,
    _extract_result_text,
    _process_message,
    _wrap_prompt,
    build_options,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.claude_config import (
    ClaudeConfig,
    ExtendedThinkingConfig,
    PermissionMode,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.observability import (
    MetricsConfig,
    ObservabilityConfig,
    TracingConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SDK_MODULE = "holodeck.lib.backends.claude_backend"
_CAS_MODULE = "claude_agent_sdk"


def _make_agent(
    *,
    claude: ClaudeConfig | None = None,
    response_format: dict[str, Any] | str | None = None,
    tools: list[Any] | None = None,
    observability: ObservabilityConfig | None = None,
) -> Agent:
    """Create a minimal Anthropic-provider Agent for testing."""
    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-6"),
        instructions=Instructions(inline="Be helpful."),
        claude=claude,
        response_format=response_format,
        tools=tools,
        observability=observability,
    )


def _make_text_block(text: str) -> MagicMock:
    """Create a mock TextBlock."""
    block = MagicMock()
    block.text = text
    block.__class__.__name__ = "TextBlock"
    return block


def _make_tool_use_block(
    tool_id: str = "toolu_01", name: str = "kb_search", inp: dict | None = None
) -> MagicMock:
    """Create a mock ToolUseBlock."""
    block = MagicMock()
    block.id = tool_id
    block.name = name
    block.input = inp or {"query": "refund"}
    block.__class__.__name__ = "ToolUseBlock"
    return block


def _make_tool_result_block(
    tool_use_id: str = "toolu_01",
    text: str = "30-day guarantee",
    is_error: bool = False,
) -> MagicMock:
    """Create a mock ToolResultBlock."""
    block = MagicMock()
    block.tool_use_id = tool_use_id
    block.content = [_make_text_block(text)]
    block.is_error = is_error
    block.__class__.__name__ = "ToolResultBlock"
    return block


def _make_assistant_message(content: list[Any] | None = None) -> MagicMock:
    """Create a mock AssistantMessage."""
    msg = MagicMock()
    msg.content = content or [_make_text_block("Hello world")]
    msg.__class__.__name__ = "AssistantMessage"
    return msg


def _make_result_message(
    *,
    is_error: bool = False,
    num_turns: int = 1,
    session_id: str = "sess-abc",
    usage: dict[str, Any] | None = None,
    structured_output: Any = None,
) -> MagicMock:
    """Create a mock ResultMessage."""
    msg = MagicMock()
    msg.is_error = is_error
    msg.num_turns = num_turns
    msg.session_id = session_id
    msg.usage = usage or {"input_tokens": 10, "output_tokens": 5}
    msg.structured_output = structured_output
    msg.__class__.__name__ = "ResultMessage"
    return msg


async def _async_iter(items: list[Any]):
    """Create an async iterator from a list."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# T001 — build_options()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildOptions:
    """Tests for build_options() — Agent + ClaudeConfig → ClaudeAgentOptions."""

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_minimal(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """T001a: No claude block → sane defaults."""
        agent = _make_agent()
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        mock_opts_cls.assert_called_once()
        kwargs = mock_opts_cls.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["system_prompt"] == "Be helpful."
        assert kwargs["max_turns"] is None

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_full_claude_block(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """T001b: Full claude block → all fields mapped."""
        claude = ClaudeConfig(
            working_directory="/var/holodeck/test",
            permission_mode=PermissionMode.acceptAll,
            max_turns=5,
            extended_thinking=ExtendedThinkingConfig(
                enabled=True, budget_tokens=20_000
            ),
            web_search=True,
        )
        agent = _make_agent(claude=claude)
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="chat",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert kwargs["cwd"] == "/var/holodeck/test"
        assert kwargs["max_turns"] == 5
        assert kwargs["max_thinking_tokens"] == 20_000

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_response_format(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """T001c: response_format dict → output_format translated."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        agent = _make_agent(response_format=schema)
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert kwargs["output_format"] is not None

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_auth_env_merged(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """T010: Auth env vars are merged into ClaudeAgentOptions.env."""
        agent = _make_agent()
        build_options(
            agent=agent,
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
            otel_env={"CLAUDE_CODE_ENABLE_TELEMETRY": "1"},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert kwargs["env"]["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert kwargs["env"]["AWS_REGION"] == "us-east-1"
        assert kwargs["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_mcp_servers_merged(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """Tool server + external MCP configs merged into mcp_servers."""
        tool_server = MagicMock()
        external_mcp = {"ext_server": MagicMock()}
        build_options(
            agent=_make_agent(),
            tool_server=tool_server,
            tool_names=["mcp__holodeck_tools__kb_search"],
            mcp_configs=external_mcp,
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert "holodeck_tools" in kwargs["mcp_servers"]
        assert "ext_server" in kwargs["mcp_servers"]

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_web_search_adds_allowed_tool(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """web_search=True adds WebSearch to allowed_tools."""
        claude = ClaudeConfig(web_search=True)
        build_options(
            agent=_make_agent(claude=claude),
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert "WebSearch" in kwargs["allowed_tools"]

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_build_options_web_search_false_omits_tool(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """web_search=False (default) does not add WebSearch."""
        claude = ClaudeConfig(web_search=False)
        build_options(
            agent=_make_agent(claude=claude),
            tool_server=None,
            tool_names=[],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert "WebSearch" not in kwargs["allowed_tools"]


# ---------------------------------------------------------------------------
# T002 — Permission mode mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPermissionMode:
    """Tests for _build_permission_mode() — HoloDeck enum → SDK literals."""

    def test_manual_maps_to_default(self) -> None:
        """manual → 'default'."""
        assert _build_permission_mode(PermissionMode.manual, "chat", False) == "default"

    def test_accept_edits_maps_to_accept_edits(self) -> None:
        """acceptEdits → 'acceptEdits'."""
        assert (
            _build_permission_mode(PermissionMode.acceptEdits, "chat", False)
            == "acceptEdits"
        )

    def test_accept_all_maps_to_bypass_permissions(self) -> None:
        """acceptAll → 'bypassPermissions'."""
        assert (
            _build_permission_mode(PermissionMode.acceptAll, "chat", False)
            == "bypassPermissions"
        )

    def test_test_mode_overrides_to_bypass(self) -> None:
        """In test mode, non-manual permission → 'bypassPermissions'."""
        assert (
            _build_permission_mode(PermissionMode.acceptEdits, "test", False)
            == "bypassPermissions"
        )

    def test_test_mode_manual_keeps_default(self) -> None:
        """In test mode, manual keeps 'default' (no override)."""
        assert _build_permission_mode(PermissionMode.manual, "test", False) == "default"

    def test_none_permission_mode(self) -> None:
        """No permission mode configured → None."""
        assert _build_permission_mode(None, "test", False) is None


# ---------------------------------------------------------------------------
# T003 — ClaudeBackend.__init__() stores config only
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendInit:
    """Tests for ClaudeBackend constructor — no I/O, stores config only."""

    def test_init_stores_config_no_io(self) -> None:
        """T003: Constructor stores config without triggering I/O."""
        agent = _make_agent()
        backend = ClaudeBackend(agent=agent, tool_instances=None, mode="test")

        assert backend._initialized is False
        assert backend._options is None
        assert backend._agent is agent

    def test_init_accepts_tool_instances(self) -> None:
        """Constructor stores tool_instances."""
        agent = _make_agent()
        tools = {"kb": MagicMock()}
        backend = ClaudeBackend(
            agent=agent, tool_instances=tools, mode="chat", allow_side_effects=True
        )

        assert backend._tool_instances is tools
        assert backend._mode == "chat"
        assert backend._allow_side_effects is True

    def test_implements_backend_protocol(self) -> None:
        """ClaudeBackend satisfies the AgentBackend protocol."""
        assert isinstance(ClaudeBackend(_make_agent()), AgentBackend)


# ---------------------------------------------------------------------------
# T004 — ClaudeBackend.initialize() validator call order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendInitialize:
    """Tests for ClaudeBackend.initialize() — validator ordering + state."""

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options")
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_validators_called_in_order(
        self,
        mock_nodejs: MagicMock,
        mock_creds: MagicMock,
        mock_embed: MagicMock,
        mock_tool_filter: MagicMock,
        mock_create_adapters: MagicMock,
        mock_build_server: MagicMock,
        mock_mcp_configs: MagicMock,
        mock_otel: MagicMock,
        mock_build_opts: MagicMock,
        mock_validate_wd: MagicMock,
        mock_validate_rf: MagicMock,
    ) -> None:
        """T004: Validators called in documented order; _initialized set True."""
        mock_build_opts.return_value = MagicMock()
        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        await backend.initialize()

        # Verify ordering via call sequence
        mock_nodejs.assert_called_once()
        mock_creds.assert_called_once_with(agent.model)
        mock_embed.assert_called_once_with(agent)
        mock_tool_filter.assert_called_once_with(agent)
        mock_build_opts.assert_called_once()

        assert backend._initialized is True
        assert backend._options is not None

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options")
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_initialize_idempotent(
        self,
        mock_nodejs: MagicMock,
        mock_creds: MagicMock,
        mock_embed: MagicMock,
        mock_tool_filter: MagicMock,
        mock_create_adapters: MagicMock,
        mock_build_server: MagicMock,
        mock_mcp_configs: MagicMock,
        mock_otel: MagicMock,
        mock_build_opts: MagicMock,
        mock_validate_wd: MagicMock,
        mock_validate_rf: MagicMock,
    ) -> None:
        """T004b: Calling initialize() twice only runs validators once."""
        mock_build_opts.return_value = MagicMock()
        backend = ClaudeBackend(_make_agent())

        await backend.initialize()
        await backend.initialize()

        mock_nodejs.assert_called_once()


# ---------------------------------------------------------------------------
# T005 — invoke_once() happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendInvokeOnce:
    """Tests for ClaudeBackend.invoke_once() — execution and result mapping."""

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_happy_path(self, mock_query: MagicMock) -> None:
        """T005: Mocked query() → correct ExecutionResult fields."""
        assistant = _make_assistant_message([_make_text_block("Hello world")])
        result_msg = _make_result_message()
        mock_query.return_value = _async_iter([assistant, result_msg])

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("Hi")

        assert isinstance(result, ExecutionResult)
        assert result.response == "Hello world"
        assert result.is_error is False
        assert result.token_usage.prompt_tokens == 10
        assert result.token_usage.completion_tokens == 5
        assert result.num_turns == 1

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_tool_extraction(self, mock_query: MagicMock) -> None:
        """T006: Tool calls and results extracted from content blocks."""
        tool_use = _make_tool_use_block("toolu_01", "kb_search", {"query": "refund"})
        tool_result = _make_tool_result_block("toolu_01", "30-day guarantee", False)
        assistant = _make_assistant_message([tool_use, tool_result])
        result_msg = _make_result_message()
        mock_query.return_value = _async_iter([assistant, result_msg])

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("What is the refund policy?")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0] == {
            "name": "kb_search",
            "arguments": {"query": "refund"},
            "call_id": "toolu_01",
        }
        assert len(result.tool_results) == 1
        assert result.tool_results[0] == {
            "call_id": "toolu_01",
            "name": "kb_search",
            "result": "30-day guarantee",
            "is_error": False,
        }

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_structured_output(self, mock_query: MagicMock) -> None:
        """T007: Structured output returned and response set to JSON string."""
        structured = {"name": "Widget", "price": 9.99}
        result_msg = _make_result_message(structured_output=structured)
        assistant = _make_assistant_message([_make_text_block("")])
        mock_query.return_value = _async_iter([assistant, result_msg])

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
        }
        backend = ClaudeBackend(_make_agent(response_format=schema))
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("Get product info")

        assert result.structured_output == {"name": "Widget", "price": 9.99}
        assert result.response == json.dumps(structured)

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_structured_output_schema_failure(
        self, mock_query: MagicMock
    ) -> None:
        """T007b: Schema validation failure → is_error=True."""
        structured = {"name": 123}  # name should be string
        result_msg = _make_result_message(structured_output=structured)
        assistant = _make_assistant_message([_make_text_block("")])
        mock_query.return_value = _async_iter([assistant, result_msg])

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        backend = ClaudeBackend(_make_agent(response_format=schema))
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("Get product info")

        assert result.is_error is True
        assert "schema validation" in result.error_reason.lower()

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_max_turns_exceeded(self, mock_query: MagicMock) -> None:
        """T008: max_turns exceeded → is_error=True with partial response."""
        assistant = _make_assistant_message([_make_text_block("Partial response")])
        result_msg = _make_result_message(num_turns=10)
        mock_query.return_value = _async_iter([assistant, result_msg])

        claude = ClaudeConfig(max_turns=10)
        backend = ClaudeBackend(_make_agent(claude=claude))
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("Complex task")

        assert result.is_error is True
        assert "max_turns limit reached" in result.error_reason
        assert result.response == "Partial response"


# ---------------------------------------------------------------------------
# T009 — Subprocess crash retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendRetry:
    """Tests for invoke_once() retry on ProcessError."""

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.asyncio")
    @patch(f"{_CAS_MODULE}.query")
    async def test_retry_on_process_error_then_success(
        self, mock_query: MagicMock, mock_asyncio: MagicMock
    ) -> None:
        """T009a: First call raises ProcessError, second succeeds."""
        process_error = type("ProcessError", (Exception,), {})()

        assistant = _make_assistant_message()
        result_msg = _make_result_message()

        mock_query.side_effect = [
            process_error,
            _async_iter([assistant, result_msg]),
        ]
        mock_asyncio.sleep = AsyncMock()

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        # Patch ProcessError type check
        with patch(f"{_SDK_MODULE}.ProcessError", type(process_error)):
            result = await backend.invoke_once("Hello")

        assert result.response == "Hello world"
        mock_asyncio.sleep.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.asyncio")
    @patch(f"{_CAS_MODULE}.query")
    async def test_retry_all_failures_raises(
        self, mock_query: MagicMock, mock_asyncio: MagicMock
    ) -> None:
        """T009b: All 3 retries fail → BackendSessionError."""
        process_error = type("ProcessError", (Exception,), {})()

        mock_query.side_effect = [process_error, process_error, process_error]
        mock_asyncio.sleep = AsyncMock()

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        with (
            patch(f"{_SDK_MODULE}.ProcessError", type(process_error)),
            pytest.raises(BackendSessionError, match="3 retries"),
        ):
            await backend.invoke_once("Hello")

        assert mock_asyncio.sleep.await_count == 2  # backoff between retries


# ---------------------------------------------------------------------------
# T011 — ClaudeSession.send_streaming()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeSessionStreaming:
    """Tests for ClaudeSession.send_streaming() — progressive text chunks."""

    @pytest.mark.asyncio
    async def test_send_streaming_yields_chunks(self) -> None:
        """T011: Chunks arrive progressively, not all at once."""
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        chunk1 = _make_assistant_message([_make_text_block("Hello ")])
        chunk2 = _make_assistant_message([_make_text_block("world!")])
        result_msg = _make_result_message()

        def mock_receive():
            return _async_iter([chunk1, chunk2, result_msg])

        mock_client.receive_response = mock_receive

        session = ClaudeSession(options=MagicMock())
        session._client = mock_client

        chunks: list[str] = []
        async for chunk in session.send_streaming("Hi"):
            chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]


# ---------------------------------------------------------------------------
# T012 — ClaudeSession.send() (non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeSessionSend:
    """Tests for ClaudeSession.send() — non-streaming, full response."""

    @pytest.mark.asyncio
    async def test_send_returns_execution_result(self) -> None:
        """T012: send() returns ExecutionResult with concatenated text."""
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        assistant = _make_assistant_message(
            [_make_text_block("Hello "), _make_text_block("world!")]
        )
        result_msg = _make_result_message()

        def mock_receive():
            return _async_iter([assistant, result_msg])

        mock_client.receive_response = mock_receive

        session = ClaudeSession(options=MagicMock())
        session._client = mock_client

        result = await session.send("Hi")

        assert isinstance(result, ExecutionResult)
        assert result.response == "Hello world!"
        assert result.token_usage.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_send_multi_turn_state_tracking(self) -> None:
        """T014: Multi-turn state tracking passes session_id on turn 2+."""
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        # First turn
        result_msg_1 = _make_result_message(session_id="sess-001")
        turn1_items = [_make_assistant_message(), result_msg_1]

        def mock_receive_turn1():
            return _async_iter(turn1_items)

        mock_client.receive_response = mock_receive_turn1

        base_options = MagicMock()
        session = ClaudeSession(options=base_options)
        session._client = mock_client

        await session.send("Turn 1")

        # After first turn, session_id should be captured
        assert session._session_id == "sess-001"
        assert session._turn_count == 1

        # First turn uses default session_id
        mock_client.query.assert_called_with("Turn 1", session_id="default")

        # Second turn
        result_msg_2 = _make_result_message(session_id="sess-002")
        turn2_items = [_make_assistant_message(), result_msg_2]

        def mock_receive_turn2():
            return _async_iter(turn2_items)

        mock_client.receive_response = mock_receive_turn2

        await session.send("Turn 2")

        # Second turn should pass captured session_id
        mock_client.query.assert_called_with("Turn 2", session_id="sess-001")
        assert session._session_id == "sess-002"
        assert session._turn_count == 2


# ---------------------------------------------------------------------------
# T013 — ClaudeSession.close()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeSessionClose:
    """Tests for ClaudeSession.close() — client disconnect."""

    @pytest.mark.asyncio
    async def test_close_calls_disconnect(self) -> None:
        """T013: close() calls client.disconnect() exactly once."""
        mock_client = AsyncMock()
        session = ClaudeSession(options=MagicMock())
        session._client = mock_client

        await session.close()

        mock_client.disconnect.assert_awaited_once()
        assert session._client is None

    @pytest.mark.asyncio
    async def test_close_no_client_is_noop(self) -> None:
        """close() with no client is a no-op."""
        session = ClaudeSession(options=MagicMock())
        await session.close()  # Should not raise

    def test_implements_session_protocol(self) -> None:
        """ClaudeSession satisfies the AgentSession protocol."""
        assert isinstance(ClaudeSession(options=MagicMock()), AgentSession)


# ---------------------------------------------------------------------------
# T015 — Lazy-init guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendLazyInit:
    """Tests for lazy-init — auto-initialize on first use."""

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_once_auto_initializes(self, mock_query: MagicMock) -> None:
        """T015a: invoke_once() without explicit initialize() auto-inits."""
        mock_query.return_value = _async_iter(
            [_make_assistant_message(), _make_result_message()]
        )

        backend = ClaudeBackend(_make_agent())
        assert backend._initialized is False

        with patch.object(backend, "initialize", new_callable=AsyncMock) as mock_init:
            # After initialize is called, set _initialized to True and _options
            async def side_effect():
                backend._initialized = True
                backend._options = MagicMock()

            mock_init.side_effect = side_effect
            await backend.invoke_once("Hello")
            mock_init.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_session_auto_initializes(self) -> None:
        """T015b: create_session() without explicit initialize() auto-inits."""
        backend = ClaudeBackend(_make_agent())

        with patch.object(backend, "initialize", new_callable=AsyncMock) as mock_init:

            async def side_effect():
                backend._initialized = True
                backend._options = MagicMock()

            mock_init.side_effect = side_effect
            session = await backend.create_session()
            mock_init.assert_awaited_once()
            assert isinstance(session, ClaudeSession)

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_no_double_init(self, mock_query: MagicMock) -> None:
        """T015c: Explicit initialize() + invoke_once() → no second init."""
        mock_query.return_value = _async_iter(
            [_make_assistant_message(), _make_result_message()]
        )

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        with patch.object(backend, "initialize", new_callable=AsyncMock) as mock_init:
            await backend.invoke_once("Hello")
            mock_init.assert_not_awaited()


# ---------------------------------------------------------------------------
# T016 — teardown()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendTeardown:
    """Tests for ClaudeBackend.teardown() — state reset."""

    @pytest.mark.asyncio
    async def test_teardown_resets_state(self) -> None:
        """T016a: teardown() resets _initialized and _options."""
        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        await backend.teardown()

        assert backend._initialized is False
        assert backend._options is None

    @pytest.mark.asyncio
    @patch(f"{_CAS_MODULE}.query")
    async def test_invoke_after_teardown_reinitializes(
        self, mock_query: MagicMock
    ) -> None:
        """T016b: invoke_once() after teardown() triggers re-initialization."""
        mock_query.return_value = _async_iter(
            [_make_assistant_message(), _make_result_message()]
        )

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        await backend.teardown()

        with patch.object(backend, "initialize", new_callable=AsyncMock) as mock_init:

            async def side_effect():
                backend._initialized = True
                backend._options = MagicMock()

            mock_init.side_effect = side_effect
            await backend.invoke_once("Hello")
            mock_init.assert_awaited_once()


# ---------------------------------------------------------------------------
# T006–T009 — ClaudeBackend instrumentation activation
# ---------------------------------------------------------------------------


def _mock_instrumentor_module():
    """Create a mock ``opentelemetry.instrumentation.claude_agent_sdk`` module.

    Returns:
        (mock_module, mock_cls, mock_instrumentor_instance)
    """
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)
    mock_module = MagicMock()
    mock_module.ClaudeAgentSdkInstrumentor = mock_cls
    return mock_module, mock_cls, mock_instance


@pytest.mark.unit
class TestClaudeBackendInstrumentation:
    """Tests for GenAI instrumentation activation during initialize()."""

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options", return_value=MagicMock())
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_instrument_called_with_tracer_and_meter_providers(
        self, *_mocks: MagicMock
    ) -> None:
        """T006: instrument() called with both providers when fully enabled."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=True, capture_content=True),
                metrics=MetricsConfig(enabled=True),
            ),
        )

        mock_module, mock_cls, mock_instance = _mock_instrumentor_module()
        mock_ctx = MagicMock()
        mock_tp = MagicMock(name="tracer_provider")
        mock_mp = MagicMock(name="meter_provider")
        mock_ctx.tracer_provider = mock_tp
        mock_ctx.meter_provider = mock_mp

        backend = ClaudeBackend(agent=agent)

        with (
            patch(f"{_SDK_MODULE}.get_observability_context", return_value=mock_ctx),
            patch.dict(
                sys.modules,
                {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
            ),
        ):
            await backend.initialize()

        mock_cls.assert_called_once()
        mock_instance.instrument.assert_called_once_with(
            tracer_provider=mock_tp,
            meter_provider=mock_mp,
            agent_name="test-agent",
            capture_content=True,
        )
        assert backend._instrumentor is mock_instance

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options", return_value=MagicMock())
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_instrument_called_without_meter_provider_when_metrics_disabled(
        self, *_mocks: MagicMock
    ) -> None:
        """T007: meter_provider=None when metrics disabled."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=True, capture_content=False),
                metrics=MetricsConfig(enabled=False),
            ),
        )

        mock_module, mock_cls, mock_instance = _mock_instrumentor_module()
        mock_ctx = MagicMock()
        mock_tp = MagicMock(name="tracer_provider")
        mock_mp = MagicMock(name="meter_provider")
        mock_ctx.tracer_provider = mock_tp
        mock_ctx.meter_provider = mock_mp

        backend = ClaudeBackend(agent=agent)

        with (
            patch(f"{_SDK_MODULE}.get_observability_context", return_value=mock_ctx),
            patch.dict(
                sys.modules,
                {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
            ),
        ):
            await backend.initialize()

        mock_instance.instrument.assert_called_once_with(
            tracer_provider=mock_tp,
            meter_provider=None,
            agent_name="test-agent",
            capture_content=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "obs_config",
        [
            ObservabilityConfig(enabled=False),
            None,
        ],
        ids=["disabled", "absent"],
    )
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options", return_value=MagicMock())
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_instrument_not_called_when_observability_disabled(
        self,
        _m_nodejs: MagicMock,
        _m_creds: MagicMock,
        _m_embed: MagicMock,
        _m_tool_filter: MagicMock,
        _m_adapters: MagicMock,
        _m_server: MagicMock,
        _m_mcp: MagicMock,
        _m_otel: MagicMock,
        _m_opts: MagicMock,
        _m_wd: MagicMock,
        _m_rf: MagicMock,
        obs_config: ObservabilityConfig | None,
    ) -> None:
        """T008: No instrumentation when observability disabled or absent."""
        agent = _make_agent(observability=obs_config)
        mock_module, mock_cls, _mock_instance = _mock_instrumentor_module()

        backend = ClaudeBackend(agent=agent)

        with patch.dict(
            sys.modules,
            {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
        ):
            await backend.initialize()

        mock_cls.assert_not_called()
        assert backend._instrumentor is None

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_response_format")
    @patch(f"{_SDK_MODULE}.validate_working_directory")
    @patch(f"{_SDK_MODULE}.build_options", return_value=MagicMock())
    @patch(f"{_SDK_MODULE}.translate_observability", return_value={})
    @patch(f"{_SDK_MODULE}.build_claude_mcp_configs", return_value={})
    @patch(f"{_SDK_MODULE}.build_holodeck_sdk_server", return_value=(None, []))
    @patch(f"{_SDK_MODULE}.create_tool_adapters", return_value=[])
    @patch(f"{_SDK_MODULE}.validate_tool_filtering")
    @patch(f"{_SDK_MODULE}.validate_embedding_provider")
    @patch(f"{_SDK_MODULE}.validate_credentials", return_value={})
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_instrument_not_called_when_traces_disabled(
        self, *_mocks: MagicMock
    ) -> None:
        """T009: No instrumentation when traces disabled."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=False),
            ),
        )
        mock_module, mock_cls, _mock_instance = _mock_instrumentor_module()

        backend = ClaudeBackend(agent=agent)

        with patch.dict(
            sys.modules,
            {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
        ):
            await backend.initialize()

        mock_cls.assert_not_called()
        assert backend._instrumentor is None


# ---------------------------------------------------------------------------
# T017 — _patch_hooks_for_context_propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatchHooksForContextPropagation:
    """Tests for the ContextVar re-injection hook wrapper."""

    @pytest.mark.asyncio
    async def test_wrapper_sets_contextvar_from_instance(self) -> None:
        """Hook wrapper re-injects ContextVar from client instance."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        # Record what the original hook callback sees
        seen_args: list[Any] = []

        async def fake_hook(
            input_data: Any,
            tool_use_id: str | None = None,
            context: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            seen_args.append(input_data)
            return {}

        # Build a mock client with HookMatcher-like objects
        matcher = MagicMock()
        matcher.hooks = [fake_hook]

        mock_options = MagicMock()
        mock_options.hooks = {"PreToolUse": [matcher]}

        mock_client = MagicMock()
        mock_client.options = mock_options

        fake_invocation_ctx = MagicMock(name="InvocationContext")
        mock_client._otel_invocation_ctx = fake_invocation_ctx

        # Capture set_invocation_context calls
        set_ctx_calls: list[Any] = []

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.instrumentation.claude_agent_sdk._context": (
                    MagicMock(
                        set_invocation_context=lambda c: set_ctx_calls.append(c),
                    )
                ),
            },
        ):
            _patch_hooks_for_context_propagation(mock_client)

        # Hooks were replaced with wrappers
        assert matcher.hooks != [fake_hook]
        assert len(matcher.hooks) == 1

        # Call the wrapped hook
        await matcher.hooks[0]({"tool": "WebSearch"}, "toolu_01")

        # Original hook was called with correct args
        assert seen_args == [{"tool": "WebSearch"}]
        # ContextVar was set from instance attribute
        assert set_ctx_calls == [fake_invocation_ctx]

    def test_noop_when_instrumentor_not_installed(self) -> None:
        """No-op when otel-instrumentation-claude-agent-sdk is missing."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        mock_client = MagicMock()
        mock_client.options = MagicMock()
        mock_client.options.hooks = {"PreToolUse": [MagicMock()]}

        # Remove the module so ImportError fires
        with patch.dict(
            sys.modules,
            {"opentelemetry.instrumentation.claude_agent_sdk._context": None},
        ):
            _patch_hooks_for_context_propagation(mock_client)

    def test_noop_when_no_hooks(self) -> None:
        """No-op when client has no hooks on options."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        mock_client = MagicMock()
        mock_client.options = MagicMock()
        mock_client.options.hooks = None

        _patch_hooks_for_context_propagation(mock_client)


# ---------------------------------------------------------------------------
# T018 — _wrap_prompt() async generator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWrapPrompt:
    """Tests for _wrap_prompt() — wraps string as async iterable."""

    @pytest.mark.asyncio
    async def test_yields_user_message_dict(self) -> None:
        """_wrap_prompt yields a single user-message dict."""
        items = [item async for item in _wrap_prompt("Hello")]
        assert len(items) == 1
        assert items[0]["type"] == "user"
        assert items[0]["message"]["role"] == "user"
        assert items[0]["message"]["content"] == "Hello"
        assert items[0]["session_id"] == ""
        assert items[0]["parent_tool_use_id"] is None


# ---------------------------------------------------------------------------
# T019 — _extract_result_text() branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractResultText:
    """Tests for _extract_result_text() — multiple content block types."""

    def test_text_block_extraction(self) -> None:
        """TextBlock instances produce text."""
        block = _make_text_block("hello")
        assert _extract_result_text([block]) == "hello"

    def test_dict_with_text_key(self) -> None:
        """Dict blocks with 'text' key produce text."""
        assert _extract_result_text([{"text": "from dict"}]) == "from dict"

    def test_plain_string(self) -> None:
        """Plain string blocks produce text."""
        assert _extract_result_text(["plain"]) == "plain"

    def test_mixed_content_blocks(self) -> None:
        """Mixed block types are concatenated."""
        blocks = [_make_text_block("A"), {"text": "B"}, "C"]
        assert _extract_result_text(blocks) == "ABC"

    def test_unknown_block_type_ignored(self) -> None:
        """Unknown block types are silently skipped."""
        unknown = MagicMock()
        unknown.__class__.__name__ = "ImageBlock"
        assert _extract_result_text([unknown]) == ""

    def test_empty_content(self) -> None:
        """Empty content list returns empty string."""
        assert _extract_result_text([]) == ""


# ---------------------------------------------------------------------------
# T020 — _build_output_format() file path branch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildOutputFormat:
    """Tests for _build_output_format() — dict, file path, None."""

    def test_none_returns_none(self) -> None:
        """None input → None output."""
        assert _build_output_format(None) is None

    def test_dict_returns_json_schema_format(self) -> None:
        """Dict input → wrapped in json_schema format."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = _build_output_format(schema)
        assert result == {"type": "json_schema", "schema": schema}

    def test_file_path_loads_schema(self, tmp_path) -> None:
        """Valid file path → loads and wraps schema."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema))

        result = _build_output_format(str(schema_file))
        assert result == {"type": "json_schema", "schema": schema}

    def test_missing_file_path_returns_none(self) -> None:
        """Non-existent file path → None."""
        result = _build_output_format("/nonexistent/schema.json")
        assert result is None


# ---------------------------------------------------------------------------
# T021 — _enrich_tool_results()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnrichToolResults:
    """Tests for _enrich_tool_results() — name injection."""

    def test_enriches_results_with_names(self) -> None:
        """Adds name from tool_calls to matching tool_results."""
        tool_calls = [{"call_id": "t1", "name": "search"}]
        tool_results = [{"call_id": "t1", "result": "data"}]
        _enrich_tool_results(tool_calls, tool_results)
        assert tool_results[0]["name"] == "search"

    def test_skips_already_named_results(self) -> None:
        """Doesn't overwrite existing name."""
        tool_calls = [{"call_id": "t1", "name": "search"}]
        tool_results = [{"call_id": "t1", "result": "data", "name": "original"}]
        _enrich_tool_results(tool_calls, tool_results)
        assert tool_results[0]["name"] == "original"

    def test_no_match_leaves_result_unchanged(self) -> None:
        """Unmatched call_id leaves result without name."""
        tool_calls = [{"call_id": "t1", "name": "search"}]
        tool_results = [{"call_id": "t2", "result": "data"}]
        _enrich_tool_results(tool_calls, tool_results)
        assert "name" not in tool_results[0]


# ---------------------------------------------------------------------------
# T022 — _process_message() skips non-Assistant/UserMessage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessMessage:
    """Tests for _process_message() — unknown message types."""

    def test_unknown_message_type_skipped(self) -> None:
        """Non-AssistantMessage/UserMessage types are no-ops."""
        msg = MagicMock()
        msg.__class__.__name__ = "SystemMessage"
        text, calls, results = _process_message(msg, [], [], [])
        assert text == []
        assert calls == []
        assert results == []

    def test_user_message_with_tool_result(self) -> None:
        """UserMessage containing ToolResultBlock extracts results."""
        tool_result_block = _make_tool_result_block("toolu_01", "answer", False)
        msg = MagicMock()
        msg.__class__.__name__ = "UserMessage"
        msg.content = [tool_result_block]

        text: list[str] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        _process_message(msg, text, calls, results)

        assert len(results) == 1
        assert results[0]["call_id"] == "toolu_01"
        assert results[0]["result"] == "answer"


# ---------------------------------------------------------------------------
# T023 — build_options() allowed_tools from claude.allowed_tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildOptionsAllowedTools:
    """Tests for build_options() — claude.allowed_tools merging."""

    @patch(f"{_SDK_MODULE}.ClaudeAgentOptions")
    @patch(f"{_SDK_MODULE}.resolve_instructions", return_value="Be helpful.")
    def test_allowed_tools_merged(
        self, mock_resolve: MagicMock, mock_opts_cls: MagicMock
    ) -> None:
        """claude.allowed_tools extends allowed_tools list."""
        claude = ClaudeConfig(allowed_tools=["Bash", "Read", "Write"])
        build_options(
            agent=_make_agent(claude=claude),
            tool_server=None,
            tool_names=["mcp__holodeck__search"],
            mcp_configs={},
            auth_env={},
            otel_env={},
            mode="test",
            allow_side_effects=False,
        )

        kwargs = mock_opts_cls.call_args[1]
        assert "Bash" in kwargs["allowed_tools"]
        assert "Read" in kwargs["allowed_tools"]
        assert "Write" in kwargs["allowed_tools"]
        assert "mcp__holodeck__search" in kwargs["allowed_tools"]


# ---------------------------------------------------------------------------
# T024 — _patch_hooks no-hooks early return (line 358)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPatchHooksNoOptions:
    """Test _patch_hooks_for_context_propagation early returns."""

    def test_noop_when_options_is_none(self) -> None:
        """No-op when client.options is None."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        mock_client = MagicMock()
        mock_client.options = None

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.instrumentation.claude_agent_sdk._context": MagicMock(
                    set_invocation_context=MagicMock(),
                ),
            },
        ):
            _patch_hooks_for_context_propagation(mock_client)

    def test_noop_when_hooks_empty_dict(self) -> None:
        """No-op when hooks is an empty dict."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        mock_client = MagicMock()
        mock_client.options.hooks = {}

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.instrumentation.claude_agent_sdk._context": MagicMock(
                    set_invocation_context=MagicMock(),
                ),
            },
        ):
            _patch_hooks_for_context_propagation(mock_client)

    def test_noop_when_matchers_is_none(self) -> None:
        """No-op when hooks value is None."""
        from holodeck.lib.backends.claude_backend import (
            _patch_hooks_for_context_propagation,
        )

        mock_client = MagicMock()
        mock_client.options.hooks = {"PreToolUse": None}

        with patch.dict(
            sys.modules,
            {
                "opentelemetry.instrumentation.claude_agent_sdk._context": MagicMock(
                    set_invocation_context=MagicMock(),
                ),
            },
        ):
            _patch_hooks_for_context_propagation(mock_client)


# ---------------------------------------------------------------------------
# T025 — ClaudeSession._ensure_client() lazy creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeSessionEnsureClient:
    """Tests for ClaudeSession._ensure_client() — lazy client creation."""

    @pytest.mark.asyncio
    async def test_ensure_client_creates_and_connects(self) -> None:
        """First call creates client and connects."""
        mock_client_instance = AsyncMock()
        mock_sdk_module = MagicMock()
        mock_sdk_module.ClaudeSDKClient.return_value = mock_client_instance

        session = ClaudeSession(options=MagicMock())

        with (
            patch(f"{_SDK_MODULE}.claude_agent_sdk", mock_sdk_module),
            patch(f"{_SDK_MODULE}._patch_hooks_for_context_propagation") as mock_patch,
        ):
            client = await session._ensure_client()

        assert client is mock_client_instance
        mock_client_instance.connect.assert_awaited_once()
        mock_patch.assert_called_once_with(mock_client_instance)

    @pytest.mark.asyncio
    async def test_ensure_client_reuses_existing(self) -> None:
        """Second call reuses existing client without reconnecting."""
        existing_client = AsyncMock()
        session = ClaudeSession(options=MagicMock())
        session._client = existing_client

        client = await session._ensure_client()

        assert client is existing_client
        existing_client.connect.assert_not_awaited()


# ---------------------------------------------------------------------------
# T026 — Session send/send_streaming error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeSessionErrorHandling:
    """Tests for ClaudeSession error paths."""

    @pytest.mark.asyncio
    async def test_send_raises_on_process_error(self) -> None:
        """send() wraps ProcessError in BackendSessionError."""
        from claude_agent_sdk import ProcessError

        mock_client = MagicMock()
        mock_client.query = AsyncMock(side_effect=ProcessError("boom"))

        session = ClaudeSession(options=MagicMock())
        session._client = mock_client

        with pytest.raises(BackendSessionError, match="subprocess terminated"):
            await session.send("Hi")

    @pytest.mark.asyncio
    async def test_send_streaming_raises_on_cli_connection_error(self) -> None:
        """send_streaming() wraps CLIConnectionError in BackendSessionError."""
        from claude_agent_sdk._errors import CLIConnectionError

        mock_client = MagicMock()
        mock_client.query = AsyncMock(side_effect=CLIConnectionError("disconnected"))

        session = ClaudeSession(options=MagicMock())
        session._client = mock_client

        with pytest.raises(BackendSessionError, match="subprocess terminated"):
            chunks = []
            async for chunk in session.send_streaming("Hi"):
                chunks.append(chunk)


# ---------------------------------------------------------------------------
# T027 — initialize() wraps non-BackendInitError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendInitializeErrors:
    """Tests for initialize() error wrapping."""

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_non_backend_init_error_wrapped(self, mock_nodejs: MagicMock) -> None:
        """Non-BackendInitError exceptions are wrapped in BackendInitError."""
        from holodeck.lib.backends.base import BackendInitError

        mock_nodejs.side_effect = RuntimeError("node not found")

        backend = ClaudeBackend(_make_agent())

        with pytest.raises(BackendInitError, match="node not found"):
            await backend.initialize()

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.validate_nodejs")
    async def test_backend_init_error_not_double_wrapped(
        self, mock_nodejs: MagicMock
    ) -> None:
        """BackendInitError is re-raised without wrapping."""
        from holodeck.lib.backends.base import BackendInitError

        mock_nodejs.side_effect = BackendInitError("already an init error")

        backend = ClaudeBackend(_make_agent())

        with pytest.raises(BackendInitError, match="already an init error"):
            await backend.initialize()


# ---------------------------------------------------------------------------
# T028 — invoke_once() BaseExceptionGroup handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendExceptionGroup:
    """Tests for invoke_once() BaseExceptionGroup retry."""

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.asyncio")
    @patch(f"{_CAS_MODULE}.query")
    async def test_exception_group_triggers_retry(
        self, mock_query: MagicMock, mock_asyncio: MagicMock
    ) -> None:
        """BaseExceptionGroup triggers retry logic."""
        exc_group = BaseExceptionGroup(
            "subprocess errors",
            [RuntimeError("process died")],
        )

        assistant = _make_assistant_message()
        result_msg = _make_result_message()

        mock_query.side_effect = [
            exc_group,
            _async_iter([assistant, result_msg]),
        ]
        mock_asyncio.sleep = AsyncMock()

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        result = await backend.invoke_once("Hello")

        assert result.response == "Hello world"
        mock_asyncio.sleep.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    @patch(f"{_SDK_MODULE}.asyncio")
    @patch(f"{_CAS_MODULE}.query")
    async def test_exception_group_all_retries_raises(
        self, mock_query: MagicMock, mock_asyncio: MagicMock
    ) -> None:
        """BaseExceptionGroup on all retries → BackendSessionError."""
        exc_group = BaseExceptionGroup(
            "subprocess errors",
            [RuntimeError("process died")],
        )

        mock_query.side_effect = [exc_group, exc_group, exc_group]
        mock_asyncio.sleep = AsyncMock()

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()

        with pytest.raises(BackendSessionError, match="3 retries"):
            await backend.invoke_once("Hello")


# ---------------------------------------------------------------------------
# T029 — _validate_structured_output() no-schema pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateStructuredOutputNoSchema:
    """Tests for _validate_structured_output() without a schema."""

    def test_no_response_format_passes_through(self) -> None:
        """When response_format is None, output passes through."""
        backend = ClaudeBackend(_make_agent(response_format=None))
        backend._initialized = True

        result = backend._validate_structured_output({"key": "val"}, "fallback text")

        assert result is not None
        assert result["structured_output"] == {"key": "val"}
        assert result["response"] == json.dumps({"key": "val"})
        assert "is_error" not in result

    def test_string_response_format_passes_through(self) -> None:
        """When response_format is a string path, output passes through."""
        backend = ClaudeBackend(_make_agent(response_format="/path/to/schema.json"))
        backend._initialized = True

        result = backend._validate_structured_output({"x": 1}, "text")

        assert result is not None
        assert result["structured_output"] == {"x": 1}


# ---------------------------------------------------------------------------
# T030 — create_session() raises when options is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateSessionError:
    """Tests for create_session() when options not set."""

    @pytest.mark.asyncio
    async def test_raises_when_options_none_after_init(self) -> None:
        """BackendInitError when _options is None after initialization."""
        from holodeck.lib.backends.base import BackendInitError

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = None

        with pytest.raises(BackendInitError, match="options not set"):
            await backend.create_session()


# ---------------------------------------------------------------------------
# T031 — _initialize_tools()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitializeTools:
    """Tests for ClaudeBackend._initialize_tools()."""

    @pytest.mark.asyncio
    async def test_no_tools_is_noop(self) -> None:
        """No tools configured → early return."""
        agent = _make_agent(tools=None)
        backend = ClaudeBackend(agent=agent)
        await backend._initialize_tools()
        assert backend._tool_instances == {}

    @pytest.mark.asyncio
    async def test_no_vs_or_hd_tools_is_noop(self) -> None:
        """Only non-vectorstore/non-hierarchical-doc tools → skip init."""
        # Use a mock tool that isinstance checks won't match VS or HD
        mock_tool = MagicMock()
        mock_tool.__class__ = type("SomeOtherTool", (), {})

        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        # Bypass Pydantic by setting tools directly
        backend._agent = MagicMock()
        backend._agent.tools = [mock_tool]

        await backend._initialize_tools()
        assert backend._owned_tools == []

    @pytest.mark.asyncio
    async def test_vectorstore_tool_triggers_init(self) -> None:
        """VectorstoreTool in tools list triggers initialize_tools()."""
        from holodeck.models.tool import VectorstoreTool

        vs_tool = MagicMock(spec=VectorstoreTool)

        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        backend._agent = MagicMock()
        backend._agent.tools = [vs_tool]

        mock_instances = {"kb": MagicMock()}
        with (
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new_callable=AsyncMock,
                return_value=mock_instances,
            ),
            patch("holodeck.config.context.agent_base_dir") as mock_ctx,
        ):
            mock_ctx.get.return_value = "/base"
            await backend._initialize_tools()

        assert backend._tool_instances == mock_instances
        assert len(backend._owned_tools) == 1

    @pytest.mark.asyncio
    async def test_hierarchical_doc_tool_triggers_init(self) -> None:
        """HierarchicalDocumentToolConfig triggers initialize_tools()."""
        from holodeck.models.tool import HierarchicalDocumentToolConfig

        hd_tool = MagicMock(spec=HierarchicalDocumentToolConfig)

        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        backend._agent = MagicMock()
        backend._agent.tools = [hd_tool]

        mock_instances = {"hd": MagicMock()}
        with (
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new_callable=AsyncMock,
                return_value=mock_instances,
            ),
            patch("holodeck.config.context.agent_base_dir") as mock_ctx,
        ):
            mock_ctx.get.return_value = "/base"
            await backend._initialize_tools()

        assert backend._tool_instances == mock_instances

    @pytest.mark.asyncio
    async def test_tool_init_generic_error_wrapped(self) -> None:
        """Generic exception in initialize_tools → BackendInitError."""
        from holodeck.lib.backends.base import BackendInitError
        from holodeck.models.tool import VectorstoreTool

        vs_tool = MagicMock(spec=VectorstoreTool)

        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        backend._agent = MagicMock()
        backend._agent.tools = [vs_tool]

        with (
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new_callable=AsyncMock,
                side_effect=RuntimeError("embedding failed"),
            ),
            patch("holodeck.config.context.agent_base_dir") as mock_ctx,
            pytest.raises(BackendInitError, match="Failed to initialize tools"),
        ):
            mock_ctx.get.return_value = "/base"
            await backend._initialize_tools()

    @pytest.mark.asyncio
    async def test_tool_initializer_error_reraised(self) -> None:
        """ToolInitializerError is re-raised directly."""
        from holodeck.lib.tool_initializer import ToolInitializerError
        from holodeck.models.tool import VectorstoreTool

        vs_tool = MagicMock(spec=VectorstoreTool)

        agent = _make_agent()
        backend = ClaudeBackend(agent=agent)
        backend._agent = MagicMock()
        backend._agent.tools = [vs_tool]

        with (
            patch("holodeck.config.context.agent_base_dir") as mock_ctx,
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new_callable=AsyncMock,
                side_effect=ToolInitializerError("bad tool config"),
            ),
            pytest.raises(ToolInitializerError, match="bad tool config"),
        ):
            mock_ctx.get.return_value = "/base"
            await backend._initialize_tools()


# ---------------------------------------------------------------------------
# T032 — _activate_instrumentation() error branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActivateInstrumentationErrors:
    """Tests for _activate_instrumentation() failure modes."""

    def test_import_error_logs_warning(self) -> None:
        """ImportError → logs warning, no crash."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=True),
            ),
        )
        backend = ClaudeBackend(agent=agent)

        # Remove the module so ImportError fires
        with patch.dict(
            sys.modules,
            {"opentelemetry.instrumentation.claude_agent_sdk": None},
        ):
            backend._activate_instrumentation()

        assert backend._instrumentor is None

    def test_observability_context_none_logs_warning(self) -> None:
        """get_observability_context() returning None → logs warning."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=True),
            ),
        )
        backend = ClaudeBackend(agent=agent)

        mock_module, mock_cls, _instance = _mock_instrumentor_module()

        with (
            patch(f"{_SDK_MODULE}.get_observability_context", return_value=None),
            patch.dict(
                sys.modules,
                {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
            ),
        ):
            backend._activate_instrumentation()

        mock_cls.assert_not_called()
        assert backend._instrumentor is None

    def test_instrument_exception_logs_warning(self) -> None:
        """instrument() raising → logs warning, no crash."""
        agent = _make_agent(
            observability=ObservabilityConfig(
                enabled=True,
                traces=TracingConfig(enabled=True),
            ),
        )
        backend = ClaudeBackend(agent=agent)

        mock_module, mock_cls, mock_instance = _mock_instrumentor_module()
        mock_instance.instrument.side_effect = RuntimeError("otel failed")

        mock_ctx = MagicMock()
        mock_ctx.tracer_provider = MagicMock()
        mock_ctx.meter_provider = MagicMock()

        with (
            patch(f"{_SDK_MODULE}.get_observability_context", return_value=mock_ctx),
            patch.dict(
                sys.modules,
                {"opentelemetry.instrumentation.claude_agent_sdk": mock_module},
            ),
        ):
            backend._activate_instrumentation()

        # Should not have stored the failed instrumentor
        assert backend._instrumentor is None


# ---------------------------------------------------------------------------
# T033 — teardown() owned tool cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeBackendTeardownToolCleanup:
    """Tests for teardown() cleaning up owned tools."""

    @pytest.mark.asyncio
    async def test_teardown_calls_cleanup_on_owned_tools(self) -> None:
        """teardown() calls cleanup() on each owned tool."""
        tool1 = AsyncMock()
        tool1.cleanup = AsyncMock()
        tool2 = AsyncMock()
        tool2.cleanup = AsyncMock()

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()
        backend._owned_tools = [tool1, tool2]
        backend._tool_instances = {"t1": tool1, "t2": tool2}

        await backend.teardown()

        tool1.cleanup.assert_awaited_once()
        tool2.cleanup.assert_awaited_once()
        assert backend._owned_tools == []
        assert backend._tool_instances == {}

    @pytest.mark.asyncio
    async def test_teardown_tool_cleanup_error_logged_not_raised(self) -> None:
        """Tool cleanup error is logged but doesn't crash teardown."""
        tool = MagicMock()
        tool.cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()
        backend._owned_tools = [tool]

        # Should not raise
        await backend.teardown()

        assert backend._owned_tools == []
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_teardown_tool_without_cleanup_attr(self) -> None:
        """Tools without cleanup attribute are silently skipped."""
        tool = MagicMock(spec=[])  # No attributes at all

        backend = ClaudeBackend(_make_agent())
        backend._initialized = True
        backend._options = MagicMock()
        backend._owned_tools = [tool]

        # Should not raise
        await backend.teardown()

        assert backend._owned_tools == []
