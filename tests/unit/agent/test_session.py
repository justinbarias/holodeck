"""Unit tests for chat session management."""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.chat.message import MessageValidator
from holodeck.chat.session import ChatSessionManager
from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    ExecutionResult,
)
from holodeck.models.chat import SessionState
from holodeck.models.token_usage import TokenUsage


class TestChatSessionManagerInitialization:
    """Test ChatSessionManager initialization."""

    def test_session_manager_initialization(self, make_agent, make_config) -> None:
        """SessionManager initializes with config."""
        manager = ChatSessionManager(make_agent(), make_config())
        assert manager is not None

    def test_session_manager_stores_config(self, make_agent, make_config) -> None:
        """SessionManager stores configuration."""
        config = make_config()
        manager = ChatSessionManager(make_agent(), config)
        assert manager.config == config

    def test_session_manager_stores_agent_config(self, make_agent, make_config) -> None:
        """SessionManager stores agent configuration."""
        agent_config = make_agent()
        manager = ChatSessionManager(agent_config, make_config())
        assert manager.agent_config == agent_config


class TestChatSessionLifecycle:
    """Test session lifecycle (start, process, terminate)."""

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_start(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        make_agent,
        make_config,
    ) -> None:
        """Session transitions to ACTIVE on start."""
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(make_agent(), make_config())
        await manager.start()

        assert manager.session is not None
        assert manager.session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_process_message(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        make_agent,
        make_config,
    ) -> None:
        """Session processes messages and increments count."""
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_response = AgentResponse(
            content="Hello there!",
            tool_executions=[],
            tokens_used=None,
            execution_time=0.5,
        )
        mock_executor.execute_turn.return_value = mock_response
        mock_executor_class.return_value = mock_executor

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (True, None)
        mock_validator_class.return_value = mock_validator

        manager = ChatSessionManager(make_agent(), make_config())
        await manager.start()

        initial_count = manager.session.message_count
        response = await manager.process_message("Hello")

        assert response == mock_response
        assert manager.session.message_count > initial_count

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_validate_message_before_processing(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        make_agent,
        make_config,
    ) -> None:
        """Session validates message before sending to executor."""
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_response = AgentResponse(
            content="Hello there!",
            tool_executions=[],
            tokens_used=None,
            execution_time=0.5,
        )
        mock_executor.execute_turn.return_value = mock_response
        mock_executor_class.return_value = mock_executor

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (True, None)
        mock_validator_class.return_value = mock_validator

        manager = ChatSessionManager(make_agent(), make_config())
        await manager.start()
        await manager.process_message("Test message")

        mock_validator.validate.assert_called()

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_rejects_invalid_message(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        make_agent,
        make_config,
    ) -> None:
        """Session rejects invalid messages."""
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (False, "Message too long")
        mock_validator_class.return_value = mock_validator

        manager = ChatSessionManager(make_agent(), make_config())
        await manager.start()

        with pytest.raises(ValueError):
            await manager.process_message("x" * 50000)

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_terminate(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        make_agent,
        make_config,
    ) -> None:
        """Session transitions to TERMINATED on terminate."""
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(make_agent(), make_config())
        await manager.start()

        assert manager.session.state == SessionState.ACTIVE

        await manager.terminate()

        assert manager.session.state == SessionState.TERMINATED
        mock_executor.shutdown.assert_called_once()


class TestContextLimitWarning:
    """Test context limit warning mechanism."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message_count, max_messages, expected_warn",
        [
            pytest.param(50, 100, False, id="50pct_below_threshold"),
            pytest.param(80, 100, True, id="80pct_at_threshold"),
            pytest.param(10, 10, True, id="100pct_at_max"),
        ],
    )
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_should_warn_context_limit(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
        message_count: int,
        max_messages: int,
        expected_warn: bool,
        make_agent,
        make_config,
    ) -> None:
        """Context limit warning fires at correct thresholds."""
        config = make_config(max_messages=max_messages)
        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(make_agent(), config)
        await manager.start()

        manager.session.message_count = message_count

        assert manager.should_warn_context_limit() is expected_warn


class TestSessionStreamingSupport:
    """Test streaming support through ChatSessionManager (T008-T009)."""

    def _make_mock_backend(self) -> tuple[AsyncMock, AsyncMock]:
        """Create mock AgentBackend/AgentSession pair."""
        mock_session = AsyncMock(spec=AgentSession)

        async def _stream_chunks(message: str):
            for chunk in ["Streamed", " ", "response"]:
                yield chunk

        mock_session.send_streaming = _stream_chunks
        mock_session.send.return_value = ExecutionResult(
            response="Streamed response",
            tool_calls=[],
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        mock_backend = AsyncMock(spec=AgentBackend)
        mock_backend.create_session.return_value = mock_session
        return mock_backend, mock_session

    @pytest.mark.asyncio
    async def test_process_message_streaming_yields_chunks(
        self, make_agent, make_config
    ) -> None:
        """T008: process_message_streaming() yields chunks, increments message_count."""
        agent_config = make_agent()
        config = make_config()
        mock_backend, _ = self._make_mock_backend()

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (True, None)
        with mock.patch(
            "holodeck.chat.session.MessageValidator", return_value=mock_validator
        ):
            manager = ChatSessionManager(agent_config, config)

        # Inject a mock executor that has streaming support
        mock_executor = AsyncMock(spec=AgentExecutor)

        async def _stream_exec(message: str):
            for chunk in ["Streamed", " ", "response"]:
                yield chunk

        mock_executor.execute_turn_streaming = _stream_exec
        manager._executor = mock_executor

        # Create session manually so we can test
        from holodeck.models.chat import ChatSession

        manager.session = ChatSession(
            agent_config=agent_config,
            history=[],
            state=SessionState.ACTIVE,
        )

        initial_count = manager.session.message_count
        chunks: list[str] = []
        async for chunk in manager.process_message_streaming("Hello"):
            chunks.append(chunk)

        assert chunks == ["Streamed", " ", "response"]
        assert manager.session.message_count > initial_count

    @pytest.mark.asyncio
    async def test_process_message_streaming_validates_message(
        self, make_agent, make_config
    ) -> None:
        """T009: Empty message raises ValueError before streaming starts."""
        agent_config = make_agent()
        config = make_config()

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (False, "Message cannot be empty")

        with mock.patch(
            "holodeck.chat.session.MessageValidator", return_value=mock_validator
        ):
            manager = ChatSessionManager(agent_config, config)

        # Create session manually
        from holodeck.models.chat import ChatSession

        manager.session = ChatSession(
            agent_config=agent_config,
            history=[],
            state=SessionState.ACTIVE,
        )
        manager._executor = AsyncMock(spec=AgentExecutor)

        with pytest.raises(ValueError, match="Message cannot be empty"):
            async for _ in manager.process_message_streaming(""):
                pass
