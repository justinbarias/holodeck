"""Unit tests for agent execution orchestrator."""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock

import pytest

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    """Create a minimal Agent instance for tests."""
    return Agent(
        name="test-agent",
        description="Test agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            api_key="test-key",
        ),
        instructions=Instructions(inline="Be helpful."),
    )


def _make_mock_backend(
    response_text: str = "Hello!",
    tool_calls: list | None = None,
    token_usage: TokenUsage | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Create mock AgentBackend/AgentSession pair.

    Returns:
        Tuple of (mock_backend, mock_session).
    """
    if token_usage is None:
        token_usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

    mock_session = AsyncMock(spec=AgentSession)
    mock_session.send.return_value = ExecutionResult(
        response=response_text,
        tool_calls=tool_calls or [],
        token_usage=token_usage,
    )

    async def _stream_chunks(message: str):
        for chunk in ["Hello", " ", "world"]:
            yield chunk

    mock_session.send_streaming = _stream_chunks

    mock_backend = AsyncMock(spec=AgentBackend)
    mock_backend.create_session.return_value = mock_session
    return mock_backend, mock_session


# ---------------------------------------------------------------------------
# T021: TestAgentExecutorInitialization — rewritten without AgentFactory
# ---------------------------------------------------------------------------


class TestAgentExecutorInitialization:
    """Test AgentExecutor initialization with backend injection."""

    def test_executor_initialization(self) -> None:
        """Executor initializes with agent config."""
        agent_config = _make_agent()
        executor = AgentExecutor(agent_config)
        assert executor is not None

    def test_executor_stores_agent_config(self) -> None:
        """Executor stores agent configuration."""
        agent_config = _make_agent()
        executor = AgentExecutor(agent_config)
        assert executor.agent_config == agent_config

    def test_executor_accepts_backend_injection(self) -> None:
        """Executor accepts an injected backend."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend()
        executor = AgentExecutor(agent_config, backend=mock_backend)
        assert executor._backend is mock_backend

    def test_executor_lazy_init_no_session(self) -> None:
        """Executor starts with no session (lazy init)."""
        agent_config = _make_agent()
        executor = AgentExecutor(agent_config)
        assert executor._session is None

    def test_executor_empty_history_on_init(self) -> None:
        """Executor starts with empty history."""
        agent_config = _make_agent()
        executor = AgentExecutor(agent_config)
        assert executor.get_history() == []


# ---------------------------------------------------------------------------
# T022: TestAgentExecutorExecution — rewritten without AgentFactory
# ---------------------------------------------------------------------------


class TestAgentExecutorExecution:
    """Test message execution using mock backends."""

    @pytest.mark.asyncio
    async def test_execute_turn_success(self) -> None:
        """Successful execution returns AgentResponse."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend("Hi there!")

        executor = AgentExecutor(agent_config, backend=mock_backend)
        response = await executor.execute_turn("Hello")

        assert isinstance(response, AgentResponse)
        assert response.content == "Hi there!"
        assert response.tool_executions == []
        mock_session.send.assert_awaited_once_with("Hello")

    @pytest.mark.asyncio
    async def test_execute_turn_with_tool_calls(self) -> None:
        """Tool calls extracted from ExecutionResult."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend(
            response_text="Searching...",
            tool_calls=[
                {"name": "search", "arguments": {"query": "test"}},
            ],
        )

        executor = AgentExecutor(agent_config, backend=mock_backend)
        response = await executor.execute_turn("Search for test")

        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "search"
        assert response.tool_executions[0].status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_turn_returns_agent_response(self) -> None:
        """Execution returns properly structured AgentResponse."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend("Response")

        executor = AgentExecutor(agent_config, backend=mock_backend)
        response = await executor.execute_turn("Test")

        assert hasattr(response, "content")
        assert hasattr(response, "tool_executions")
        assert hasattr(response, "tokens_used")
        assert hasattr(response, "execution_time")


# ---------------------------------------------------------------------------
# T023: TestAgentExecutorHistory — rewritten without AgentFactory/ChatHistory
# ---------------------------------------------------------------------------


class TestAgentExecutorHistory:
    """Test history management via local list[dict]."""

    def test_get_history_returns_empty_list(self) -> None:
        """History returns empty list before any turns."""
        agent_config = _make_agent()
        executor = AgentExecutor(agent_config)
        history = executor.get_history()

        assert isinstance(history, list)
        assert history == []

    @pytest.mark.asyncio
    async def test_history_tracks_user_assistant_pairs(self) -> None:
        """Each turn appends user + assistant entries."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend("Reply")

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Hello")

        history = executor.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Reply"}

    @pytest.mark.asyncio
    async def test_clear_history_resets_and_closes_session(self) -> None:
        """clear_history() closes session and empties history."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend()

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.clear_history()

        mock_session.close.assert_awaited_once()
        assert executor._session is None
        assert executor.get_history() == []

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self) -> None:
        """Shutdown closes session and tears down backend."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend()

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.shutdown()

        mock_session.close.assert_awaited_once()
        mock_backend.teardown.assert_awaited_once()
        assert executor._session is None
        assert executor._backend is None


# ---------------------------------------------------------------------------
# T024: TestAgentResponseStructure — ExecutionResult → AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponseStructure:
    """Test AgentResponse data structure and conversion."""

    def test_agent_response_creation(self) -> None:
        """AgentResponse can be created with required fields."""
        response = AgentResponse(
            content="Test response",
            tool_executions=[],
            tokens_used=None,
            execution_time=0.5,
        )
        assert response.content == "Test response"
        assert response.tool_executions == []
        assert response.execution_time == 0.5

    def test_agent_response_with_tools(self) -> None:
        """AgentResponse can include tool executions."""
        tool_exec = ToolExecution(
            tool_name="search",
            parameters={"q": "test"},
            result="found",
            status=ToolStatus.SUCCESS,
        )
        response = AgentResponse(
            content="Using search tool",
            tool_executions=[tool_exec],
            tokens_used=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            execution_time=1.0,
        )
        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "search"
        assert response.tokens_used is not None
        assert response.tokens_used.total_tokens == 15

    @pytest.mark.asyncio
    async def test_execution_result_to_agent_response(self) -> None:
        """Full ExecutionResult → AgentResponse conversion."""
        agent_config = _make_agent()
        token_usage = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )
        mock_backend, _ = _make_mock_backend(
            response_text="Converted response",
            tool_calls=[
                {"name": "search", "arguments": {"q": "test"}},
            ],
            token_usage=token_usage,
        )

        executor = AgentExecutor(agent_config, backend=mock_backend)
        response = await executor.execute_turn("Convert this")

        # (1) result.response → response.content
        assert response.content == "Converted response"

        # (2) result.tool_calls → response.tool_executions
        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "search"
        assert isinstance(response.tool_executions[0], ToolExecution)

        # (3) result.token_usage → response.tokens_used
        assert response.tokens_used is not None
        assert response.tokens_used.total_tokens == 30

        # (4) response.execution_time is a positive float
        assert isinstance(response.execution_time, float)
        assert response.execution_time > 0


# ---------------------------------------------------------------------------
# TestBackendExecutorExecution (T001–T007) — already uses mock backends
# ---------------------------------------------------------------------------


class TestBackendExecutorExecution:
    """Test AgentExecutor with backend abstraction layer (T001-T007)."""

    @pytest.mark.asyncio
    async def test_execute_turn_uses_session_send(self) -> None:
        """T001: backend= injection, session.send(), returns AgentResponse."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend("Hi there!")

        executor = AgentExecutor(agent_config, backend=mock_backend)
        response = await executor.execute_turn("Hello")

        assert isinstance(response, AgentResponse)
        assert response.content == "Hi there!"
        mock_session.send.assert_awaited_once_with("Hello")

    @pytest.mark.asyncio
    async def test_execute_turn_streaming_yields_chunks(self) -> None:
        """T002: execute_turn_streaming() yields chunks."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend()

        executor = AgentExecutor(agent_config, backend=mock_backend)
        chunks: list[str] = []
        async for chunk in executor.execute_turn_streaming("Hello"):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_auto_select_backend(self) -> None:
        """T003: No backend= → BackendSelector.select() called."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend()

        with mock.patch(
            "holodeck.chat.executor.BackendSelector.select",
            new_callable=AsyncMock,
            return_value=mock_backend,
        ) as mock_select:
            executor = AgentExecutor(agent_config)
            await executor.execute_turn("Hello")

            mock_select.assert_awaited_once_with(
                agent_config,
                tool_instances=None,
                mode="chat",
                allow_side_effects=False,
            )

    @pytest.mark.asyncio
    async def test_get_history_tracks_messages(self) -> None:
        """T004: 2 turns → 4 history entries."""
        agent_config = _make_agent()
        mock_backend, _ = _make_mock_backend("Response 1")

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Message 1")

        # Update mock for second turn
        mock_backend_session = await mock_backend.create_session()
        mock_backend_session.send.return_value = ExecutionResult(
            response="Response 2",
            tool_calls=[],
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

        await executor.execute_turn("Message 2")

        history = executor.get_history()
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "Message 1"}
        assert history[1] == {
            "role": "assistant",
            "content": "Response 1",
        }
        assert history[2] == {"role": "user", "content": "Message 2"}
        assert history[3] == {
            "role": "assistant",
            "content": "Response 2",
        }

    @pytest.mark.asyncio
    async def test_clear_history_closes_session(self) -> None:
        """T005: clear_history() closes session and empties history."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend()

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.clear_history()

        mock_session.close.assert_awaited_once()
        assert executor._session is None
        assert executor.get_history() == []

    @pytest.mark.asyncio
    async def test_shutdown_closes_session_and_backend(self) -> None:
        """T006: shutdown() closes session + tears down backend."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend()

        executor = AgentExecutor(agent_config, backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.shutdown()

        mock_session.close.assert_awaited_once()
        mock_backend.teardown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_backend_error_wrapped(self) -> None:
        """T007: BackendSessionError → wrapped in RuntimeError."""
        agent_config = _make_agent()
        mock_backend, mock_session = _make_mock_backend()
        mock_session.send.side_effect = BackendSessionError("Connection lost")

        executor = AgentExecutor(agent_config, backend=mock_backend)

        with pytest.raises(RuntimeError, match="Connection lost"):
            await executor.execute_turn("Hello")
