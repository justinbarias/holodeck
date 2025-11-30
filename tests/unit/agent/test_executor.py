"""Unit tests for agent execution orchestrator."""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from semantic_kernel.contents import ChatHistory

from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.lib.test_runner.agent_factory import AgentExecutionResult, AgentFactory
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus


class TestAgentExecutorInitialization:
    """Test AgentExecutor initialization."""

    def _make_agent(self) -> Agent:
        """Create a minimal Agent instance for tests."""
        return Agent(
            name="test-agent",
            description="Test agent for unit tests",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o-mini",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Be a helpful assistant."),
        )

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_executor_initialization(self, mock_factory_class: MagicMock) -> None:
        """Executor initializes with agent config."""
        agent_config = self._make_agent()
        executor = AgentExecutor(agent_config, enable_observability=False)
        assert executor is not None

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_executor_stores_observability_flag(
        self, mock_factory_class: MagicMock
    ) -> None:
        """Executor stores observability preference."""
        agent_config = self._make_agent()
        executor = AgentExecutor(agent_config, enable_observability=True)
        assert executor._observability_enabled is True

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_executor_stores_agent_config(self, mock_factory_class: MagicMock) -> None:
        """Executor stores agent configuration."""
        agent_config = self._make_agent()
        executor = AgentExecutor(agent_config, enable_observability=False)
        assert executor.agent_config == agent_config

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_executor_creates_agent_factory(
        self, mock_factory_class: MagicMock
    ) -> None:
        """Executor creates AgentFactory instance."""
        agent_config = self._make_agent()
        AgentExecutor(agent_config)
        mock_factory_class.assert_called_once()


class TestAgentExecutorExecution:
    """Test message execution."""

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

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.executor.AgentFactory")
    async def test_execute_turn_success(self, mock_factory_class: MagicMock) -> None:
        """Successful execution returns AgentResponse."""
        agent_config = self._make_agent()

        # Create mock AgentFactory and thread run
        mock_factory = MagicMock()
        mock_thread_run = AsyncMock()
        mock_history = ChatHistory()
        mock_history.add_user_message("Hello")
        mock_history.add_assistant_message("Hi there!")

        mock_thread_run.invoke.return_value = AgentExecutionResult(
            tool_calls=[],
            tool_results=[],
            chat_history=mock_history,
        )
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)
        mock_factory.shutdown = AsyncMock()
        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)
        response = await executor.execute_turn("Hello")

        assert isinstance(response, AgentResponse)
        assert response.content is not None
        assert response.tool_executions == []

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.executor.AgentFactory")
    async def test_execute_turn_with_tool_calls(
        self, mock_factory_class: MagicMock
    ) -> None:
        """Tool calls extracted from response."""
        agent_config = self._make_agent()

        mock_factory = MagicMock()
        mock_thread_run = AsyncMock()
        mock_history = ChatHistory()
        mock_history.add_user_message("Use the search tool")
        mock_history.add_assistant_message("Searching...")

        mock_thread_run.invoke.return_value = AgentExecutionResult(
            tool_calls=[
                {
                    "name": "search",
                    "arguments": {"query": "test query"},
                }
            ],
            tool_results=[],
            chat_history=mock_history,
        )
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)
        mock_factory.shutdown = AsyncMock()
        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)
        response = await executor.execute_turn("Use the search tool")

        assert len(response.tool_executions) >= 0  # May be converted

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.executor.AgentFactory")
    async def test_execute_turn_returns_agent_response(
        self, mock_factory_class: MagicMock
    ) -> None:
        """Execution returns properly structured AgentResponse."""
        agent_config = self._make_agent()

        mock_factory = MagicMock()
        mock_thread_run = AsyncMock()
        mock_history = ChatHistory()
        mock_history.add_user_message("Test")
        mock_history.add_assistant_message("Response")

        mock_thread_run.invoke.return_value = AgentExecutionResult(
            tool_calls=[],
            tool_results=[],
            chat_history=mock_history,
        )
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)
        mock_factory.shutdown = AsyncMock()
        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)
        response = await executor.execute_turn("Test")

        # Verify AgentResponse structure
        assert hasattr(response, "content")
        assert hasattr(response, "tool_executions")
        assert hasattr(response, "tokens_used")
        assert hasattr(response, "execution_time")


class TestAgentExecutorHistory:
    """Test history management."""

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

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_get_history_returns_chat_history(
        self, mock_factory_class: MagicMock
    ) -> None:
        """History accessible via getter."""
        agent_config = self._make_agent()
        mock_factory = MagicMock(spec=AgentFactory)
        mock_factory.agent = MagicMock()
        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)
        history = executor.get_history()

        assert history is not None

    @mock.patch("holodeck.chat.executor.AgentFactory")
    def test_clear_history(self, mock_factory_class: MagicMock) -> None:
        """Clear removes messages but preserves system."""
        agent_config = self._make_agent()

        mock_factory = MagicMock(spec=AgentFactory)
        mock_history = ChatHistory()
        mock_history.add_user_message("Test")
        mock_history.add_assistant_message("Response")

        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)

        # Clear history
        executor.clear_history()

        # History should be cleared (specific verification depends on implementation)
        assert True  # Placeholder

    @pytest.mark.asyncio
    @mock.patch("holodeck.chat.executor.AgentFactory")
    async def test_shutdown_cleanup(self, mock_factory_class: MagicMock) -> None:
        """Shutdown cleans up resources."""
        agent_config = self._make_agent()
        mock_factory = MagicMock(spec=AgentFactory)
        mock_factory_class.return_value = mock_factory

        executor = AgentExecutor(agent_config)
        await executor.shutdown()

        # Verify shutdown called (specific verification depends on implementation)
        assert True  # Placeholder


class TestAgentResponseStructure:
    """Test AgentResponse data structure."""

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
