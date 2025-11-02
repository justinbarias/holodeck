"""Unit tests for Agent model.

Tests Agent configuration including execution config field.
"""

from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestAgentExecutionConfig:
    """Tests for Agent model with ExecutionConfig."""

    def test_agent_with_execution_config(self) -> None:
        """Test Agent with execution config field."""
        execution = ExecutionConfig(
            file_timeout=60,
            llm_timeout=120,
            download_timeout=60,
            cache_enabled=True,
            cache_dir=".holodeck/cache",
            verbose=False,
            quiet=False,
        )

        agent = Agent(
            name="Test Agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
            ),
            instructions=Instructions(inline="Test instructions"),
            execution=execution,
        )

        assert agent.execution is not None
        assert agent.execution.file_timeout == 60
        assert agent.execution.llm_timeout == 120
        assert agent.execution.download_timeout == 60
        assert agent.execution.cache_enabled is True
        assert agent.execution.cache_dir == ".holodeck/cache"
        assert agent.execution.verbose is False
        assert agent.execution.quiet is False

    def test_agent_without_execution_config(self) -> None:
        """Test Agent without execution config (optional)."""
        agent = Agent(
            name="Test Agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
            ),
            instructions=Instructions(inline="Test instructions"),
        )

        assert agent.execution is None

    def test_agent_with_partial_execution_config(self) -> None:
        """Test Agent with partially filled execution config."""
        execution = ExecutionConfig(
            file_timeout=30,
            llm_timeout=60,
        )

        agent = Agent(
            name="Test Agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
            ),
            instructions=Instructions(inline="Test instructions"),
            execution=execution,
        )

        assert agent.execution is not None
        assert agent.execution.file_timeout == 30
        assert agent.execution.llm_timeout == 60
        assert agent.execution.download_timeout is None
        assert agent.execution.cache_enabled is None

    def test_agent_execution_config_validation(self) -> None:
        """Test that execution config is properly validated."""
        # ExecutionConfig should accept valid values
        execution = ExecutionConfig(file_timeout=45)

        agent = Agent(
            name="Test Agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
            ),
            instructions=Instructions(inline="Test instructions"),
            execution=execution,
        )

        assert agent.execution is not None
        assert agent.execution.file_timeout == 45

    def test_agent_with_empty_execution_config(self) -> None:
        """Test Agent with empty ExecutionConfig."""
        execution = ExecutionConfig()

        agent = Agent(
            name="Test Agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
            ),
            instructions=Instructions(inline="Test instructions"),
            execution=execution,
        )

        assert agent.execution is not None
        assert agent.execution.file_timeout is None
        assert agent.execution.llm_timeout is None
        assert agent.execution.cache_enabled is None
