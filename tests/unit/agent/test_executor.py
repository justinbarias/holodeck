"""Unit tests for agent execution orchestrator."""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock

import pytest

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.lib.backends.base import (
    BackendSessionError,
    ExecutionResult,
)
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus

# ---------------------------------------------------------------------------
# T021: TestAgentExecutorInitialization
# ---------------------------------------------------------------------------


class TestAgentExecutorInitialization:
    """Test AgentExecutor initialization with backend injection."""

    def test_executor_initialization(self, make_agent) -> None:
        """Executor initializes with agent config."""
        executor = AgentExecutor(make_agent())
        assert executor is not None

    def test_executor_stores_agent_config(self, make_agent) -> None:
        """Executor stores agent configuration."""
        agent_config = make_agent()
        executor = AgentExecutor(agent_config)
        assert executor.agent_config == agent_config

    def test_executor_accepts_backend_injection(
        self, make_agent, make_mock_backend
    ) -> None:
        """Executor accepts an injected backend."""
        mock_backend, _ = make_mock_backend()
        executor = AgentExecutor(make_agent(), backend=mock_backend)
        assert executor._backend is mock_backend

    def test_executor_lazy_init_no_session(self, make_agent) -> None:
        """Executor starts with no session (lazy init)."""
        executor = AgentExecutor(make_agent())
        assert executor._session is None

    def test_executor_empty_history_on_init(self, make_agent) -> None:
        """Executor starts with empty history."""
        executor = AgentExecutor(make_agent())
        assert executor.get_history() == []


# ---------------------------------------------------------------------------
# T022: TestAgentExecutorExecution
# ---------------------------------------------------------------------------


class TestAgentExecutorExecution:
    """Test message execution using mock backends."""

    @pytest.mark.asyncio
    async def test_execute_turn_success(self, make_agent, make_mock_backend) -> None:
        """Successful execution returns AgentResponse."""
        mock_backend, mock_session = make_mock_backend("Hi there!")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Hello")

        assert isinstance(response, AgentResponse)
        assert response.content == "Hi there!"
        assert response.tool_executions == []
        mock_session.send.assert_awaited_once_with("Hello")

    @pytest.mark.asyncio
    async def test_execute_turn_with_tool_calls(
        self, make_agent, make_mock_backend
    ) -> None:
        """Tool calls extracted from ExecutionResult."""
        mock_backend, _ = make_mock_backend(
            response_text="Searching...",
            tool_calls=[
                {"name": "search", "arguments": {"query": "test"}},
            ],
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Search for test")

        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "search"
        assert response.tool_executions[0].status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_turn_returns_agent_response(
        self, make_agent, make_mock_backend
    ) -> None:
        """Execution returns properly structured AgentResponse."""
        mock_backend, _ = make_mock_backend("Response")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Test")

        assert hasattr(response, "content")
        assert hasattr(response, "tool_executions")
        assert hasattr(response, "tokens_used")
        assert hasattr(response, "execution_time")


# ---------------------------------------------------------------------------
# T023: TestAgentExecutorHistory
# ---------------------------------------------------------------------------


class TestAgentExecutorHistory:
    """Test history management via local list[dict]."""

    def test_get_history_returns_empty_list(self, make_agent) -> None:
        """History returns empty list before any turns."""
        executor = AgentExecutor(make_agent())
        history = executor.get_history()

        assert isinstance(history, list)
        assert history == []

    @pytest.mark.asyncio
    async def test_history_tracks_user_assistant_pairs(
        self, make_agent, make_mock_backend
    ) -> None:
        """Each turn appends user + assistant entries."""
        mock_backend, _ = make_mock_backend("Reply")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        history = executor.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Reply"}

    @pytest.mark.asyncio
    async def test_clear_history_resets_and_closes_session(
        self, make_agent, make_mock_backend
    ) -> None:
        """clear_history() closes session and empties history."""
        mock_backend, mock_session = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.clear_history()

        mock_session.close.assert_awaited_once()
        assert executor._session is None
        assert executor.get_history() == []

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, make_agent, make_mock_backend) -> None:
        """Shutdown closes session and tears down backend."""
        mock_backend, mock_session = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.shutdown()

        mock_session.close.assert_awaited_once()
        mock_backend.teardown.assert_awaited_once()
        assert executor._session is None
        assert executor._backend is None


# ---------------------------------------------------------------------------
# T024: TestAgentResponseStructure
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
    async def test_execution_result_to_agent_response(
        self, make_agent, make_mock_backend
    ) -> None:
        """Full ExecutionResult to AgentResponse conversion."""
        token_usage = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )
        mock_backend, _ = make_mock_backend(
            response_text="Converted response",
            tool_calls=[
                {"name": "search", "arguments": {"q": "test"}},
            ],
            token_usage=token_usage,
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Convert this")

        # (1) result.response -> response.content
        assert response.content == "Converted response"

        # (2) result.tool_calls -> response.tool_executions
        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "search"
        assert isinstance(response.tool_executions[0], ToolExecution)

        # (3) result.token_usage -> response.tokens_used
        assert response.tokens_used is not None
        assert response.tokens_used.total_tokens == 30

        # (4) response.execution_time is a positive float
        assert isinstance(response.execution_time, float)
        assert response.execution_time > 0


# ---------------------------------------------------------------------------
# TestBackendExecutorExecution (T001-T007)
# ---------------------------------------------------------------------------


class TestBackendExecutorExecution:
    """Test AgentExecutor with backend abstraction layer (T001-T007)."""

    @pytest.mark.asyncio
    async def test_execute_turn_uses_session_send(
        self, make_agent, make_mock_backend
    ) -> None:
        """T001: backend= injection, session.send(), returns AgentResponse."""
        mock_backend, mock_session = make_mock_backend("Hi there!")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Hello")

        assert isinstance(response, AgentResponse)
        assert response.content == "Hi there!"
        mock_session.send.assert_awaited_once_with("Hello")

    @pytest.mark.asyncio
    async def test_execute_turn_streaming_yields_chunks(
        self, make_agent, make_mock_backend
    ) -> None:
        """T002: execute_turn_streaming() yields chunks."""
        mock_backend, _ = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        chunks: list[str] = []
        async for chunk in executor.execute_turn_streaming("Hello"):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_auto_select_backend(self, make_agent, make_mock_backend) -> None:
        """T003: No backend= means BackendSelector.select() is called."""
        mock_backend, _ = make_mock_backend()

        with mock.patch(
            "holodeck.chat.executor.BackendSelector.select",
            new_callable=AsyncMock,
            return_value=mock_backend,
        ) as mock_select:
            executor = AgentExecutor(make_agent())
            await executor.execute_turn("Hello")

            mock_select.assert_awaited_once_with(
                executor.agent_config,
                tool_instances=None,
                mode="chat",
                allow_side_effects=False,
            )

    @pytest.mark.asyncio
    async def test_get_history_tracks_messages(
        self, make_agent, make_mock_backend
    ) -> None:
        """T004: 2 turns produce 4 history entries."""
        mock_backend, _ = make_mock_backend("Response 1")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
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
    async def test_clear_history_closes_session(
        self, make_agent, make_mock_backend
    ) -> None:
        """T005: clear_history() closes session and empties history."""
        mock_backend, mock_session = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.clear_history()

        mock_session.close.assert_awaited_once()
        assert executor._session is None
        assert executor.get_history() == []

    @pytest.mark.asyncio
    async def test_shutdown_closes_session_and_backend(
        self, make_agent, make_mock_backend
    ) -> None:
        """T006: shutdown() closes session + tears down backend."""
        mock_backend, mock_session = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        await executor.shutdown()

        mock_session.close.assert_awaited_once()
        mock_backend.teardown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_backend_error_wrapped(self, make_agent, make_mock_backend) -> None:
        """T007: BackendSessionError wrapped in RuntimeError."""
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.side_effect = BackendSessionError("Connection lost")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(RuntimeError, match="Connection lost"):
            await executor.execute_turn("Hello")
