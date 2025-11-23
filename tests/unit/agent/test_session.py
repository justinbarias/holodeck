"""Unit tests for chat session management."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.chat.message import MessageValidator
from holodeck.chat.session import ChatSessionManager
from holodeck.models.agent import Agent, Instructions
from holodeck.models.chat import ChatConfig, SessionState
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestChatSessionManagerInitialization:
    """Test ChatSessionManager initialization."""

    def _make_agent(self) -> Agent:
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

    def _make_config(self, agent_config_path: Path | None = None) -> ChatConfig:
        """Create a minimal ChatConfig for tests."""
        if agent_config_path is None:
            # Create a temp file for testing
            with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                agent_config_path = Path(f.name)

        return ChatConfig(
            agent_config_path=agent_config_path,
            verbose=False,
            enable_observability=False,
            max_messages=50,
        )

    @mock.patch("holodeck.chat.session.AgentExecutor")
    def test_session_manager_initialization(
        self, mock_executor_class: MagicMock
    ) -> None:
        """SessionManager initializes with config."""
        agent_config = self._make_agent()
        config = self._make_config()

        # Should initialize without error
        with mock.patch("holodeck.chat.session.MessageValidator"):
            manager = ChatSessionManager(agent_config, config)
            assert manager is not None

    @mock.patch("holodeck.chat.session.AgentExecutor")
    def test_session_manager_stores_config(
        self, mock_executor_class: MagicMock
    ) -> None:
        """SessionManager stores configuration."""
        agent_config = self._make_agent()
        config = self._make_config()

        with mock.patch("holodeck.chat.session.MessageValidator"):
            manager = ChatSessionManager(agent_config, config)
            assert manager.config == config

    @mock.patch("holodeck.chat.session.AgentExecutor")
    def test_session_manager_stores_agent_config(
        self, mock_executor_class: MagicMock
    ) -> None:
        """SessionManager stores agent configuration."""
        agent_config = self._make_agent()
        config = self._make_config()

        with mock.patch("holodeck.chat.session.MessageValidator"):
            manager = ChatSessionManager(agent_config, config)
            assert manager.agent_config == agent_config


class TestChatSessionLifecycle:
    """Test session lifecycle (start, process, terminate)."""

    def _make_agent(self) -> Agent:
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

    def _make_config(self, agent_config_path: Path | None = None) -> ChatConfig:
        """Create a minimal ChatConfig for tests."""
        if agent_config_path is None:
            with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                agent_config_path = Path(f.name)

        return ChatConfig(
            agent_config_path=agent_config_path,
            verbose=False,
            enable_observability=False,
            max_messages=50,
        )

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_start(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
    ) -> None:
        """Session transitions to ACTIVE on start."""
        agent_config = self._make_agent()
        config = self._make_config()

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(agent_config, config)
        await manager.start()

        # Session should be created and state should be ACTIVE
        assert manager.session is not None
        assert manager.session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_session_process_message(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
    ) -> None:
        """Session processes messages and increments count."""
        agent_config = self._make_agent()
        config = self._make_config()

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

        manager = ChatSessionManager(agent_config, config)
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
    ) -> None:
        """Session validates message before sending to executor."""
        agent_config = self._make_agent()
        config = self._make_config()

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

        manager = ChatSessionManager(agent_config, config)
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
    ) -> None:
        """Session rejects invalid messages."""
        agent_config = self._make_agent()
        config = self._make_config()

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        mock_validator = MagicMock(spec=MessageValidator)
        mock_validator.validate.return_value = (False, "Message too long")
        mock_validator_class.return_value = mock_validator

        manager = ChatSessionManager(agent_config, config)
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
    ) -> None:
        """Session transitions to TERMINATED on terminate."""
        agent_config = self._make_agent()
        config = self._make_config()

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(agent_config, config)
        await manager.start()

        assert manager.session.state == SessionState.ACTIVE

        await manager.terminate()

        assert manager.session.state == SessionState.TERMINATED
        mock_executor.shutdown.assert_called_once()


class TestContextLimitWarning:
    """Test context limit warning mechanism."""

    def _make_agent(self) -> Agent:
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

    def _make_config(self, max_messages: int = 10) -> ChatConfig:
        """Create a ChatConfig with specified max_messages."""
        with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            agent_config_path = Path(f.name)

        return ChatConfig(
            agent_config_path=agent_config_path,
            verbose=False,
            enable_observability=False,
            max_messages=max_messages,
        )

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_should_warn_context_limit_below_threshold(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
    ) -> None:
        """No warning when below 80% of message limit."""
        agent_config = self._make_agent()
        config = self._make_config(max_messages=100)

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(agent_config, config)
        await manager.start()

        # Set message count below 80%
        manager.session.message_count = 50

        should_warn = manager.should_warn_context_limit()
        assert should_warn is False

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_should_warn_context_limit_above_threshold(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
    ) -> None:
        """Warning triggered at 80% of message limit."""
        agent_config = self._make_agent()
        config = self._make_config(max_messages=100)

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(agent_config, config)
        await manager.start()

        # Set message count at 80%
        manager.session.message_count = 80

        should_warn = manager.should_warn_context_limit()
        assert should_warn is True

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.session.AgentExecutor")
    @mock.patch("holodeck.chat.session.MessageValidator")
    async def test_should_warn_at_max_limit(
        self,
        mock_validator_class: MagicMock,
        mock_executor_class: MagicMock,
    ) -> None:
        """Warning triggered at max message limit."""
        agent_config = self._make_agent()
        config = self._make_config(max_messages=10)

        mock_executor = AsyncMock(spec=AgentExecutor)
        mock_executor_class.return_value = mock_executor

        manager = ChatSessionManager(agent_config, config)
        await manager.start()

        # Set message count at max
        manager.session.message_count = 10

        should_warn = manager.should_warn_context_limit()
        assert should_warn is True
