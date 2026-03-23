"""Unit tests for Agent model.

Tests Agent configuration including execution config field.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import MCPTool, VectorstoreTool


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


class TestToolNameUniqueness:
    """Tests for Agent-level tool name uniqueness validation."""

    def _make_model(self) -> LLMProvider:
        return LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o")

    def _make_instructions(self) -> Instructions:
        return Instructions(inline="test")

    def test_agent_duplicate_tool_names_raises(self) -> None:
        """Two vectorstore tools with the same name must raise ValidationError."""
        tool_a = VectorstoreTool(
            name="search_kb", description="KB A", type="vectorstore", source="./a"
        )
        tool_b = VectorstoreTool(
            name="search_kb", description="KB B", type="vectorstore", source="./b"
        )
        with pytest.raises(ValidationError, match="search_kb"):
            Agent(
                name="dup-agent",
                model=self._make_model(),
                instructions=self._make_instructions(),
                tools=[tool_a, tool_b],
            )

    def test_agent_duplicate_names_across_types_raises(self) -> None:
        """A vectorstore and MCP tool sharing a name must raise ValidationError."""
        vs_tool = VectorstoreTool(
            name="my_tool", description="VS", type="vectorstore", source="./data"
        )
        mcp_tool = MCPTool(
            name="my_tool",
            description="MCP",
            type="mcp",
            command="npx",
            args=["-y", "server"],
        )
        with pytest.raises(ValidationError, match="my_tool"):
            Agent(
                name="cross-type-agent",
                model=self._make_model(),
                instructions=self._make_instructions(),
                tools=[vs_tool, mcp_tool],
            )

    def test_agent_unique_tool_names_passes(self) -> None:
        """Distinct tool names must be accepted without error."""
        tool_a = VectorstoreTool(
            name="tool_alpha", description="A", type="vectorstore", source="./a"
        )
        tool_b = VectorstoreTool(
            name="tool_beta", description="B", type="vectorstore", source="./b"
        )
        agent = Agent(
            name="ok-agent",
            model=self._make_model(),
            instructions=self._make_instructions(),
            tools=[tool_a, tool_b],
        )
        assert len(agent.tools) == 2  # type: ignore[arg-type]

    def test_agent_no_tools_passes(self) -> None:
        """Agent with tools=None must not error."""
        agent = Agent(
            name="no-tools-agent",
            model=self._make_model(),
            instructions=self._make_instructions(),
            tools=None,
        )
        assert agent.tools is None

    def test_agent_duplicate_error_lists_names(self) -> None:
        """Error message must include each duplicate name."""
        tool1 = VectorstoreTool(
            name="dup_one", description="D1", type="vectorstore", source="./d1"
        )
        tool2 = VectorstoreTool(
            name="dup_one", description="D1b", type="vectorstore", source="./d1b"
        )
        tool3 = VectorstoreTool(
            name="dup_two", description="D2", type="vectorstore", source="./d2"
        )
        tool4 = VectorstoreTool(
            name="dup_two", description="D2b", type="vectorstore", source="./d2b"
        )
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="multi-dup-agent",
                model=self._make_model(),
                instructions=self._make_instructions(),
                tools=[tool1, tool2, tool3, tool4],
            )
        error_text = str(exc_info.value)
        assert "dup_one" in error_text
        assert "dup_two" in error_text
