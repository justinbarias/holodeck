"""Tests for Agent model in holodeck.models.agent."""

import pytest
from pydantic import ValidationError

from holodeck.models.agent import Agent, Instructions
from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_case import TestCase
from holodeck.models.tool import VectorstoreTool


class TestInstructions:
    """Tests for Instructions model."""

    def test_instructions_with_file(self) -> None:
        """Test Instructions with file reference."""
        instructions = Instructions(file="prompts/system.md")
        assert instructions.file == "prompts/system.md"
        assert instructions.inline is None

    def test_instructions_with_inline(self) -> None:
        """Test Instructions with inline text."""
        instructions = Instructions(inline="You are a helpful assistant.")
        assert instructions.inline == "You are a helpful assistant."
        assert instructions.file is None

    def test_instructions_file_and_inline_mutually_exclusive(self) -> None:
        """Test that file and inline are mutually exclusive."""
        with pytest.raises(ValidationError):
            Instructions(
                file="prompts/system.md",
                inline="You are a helpful assistant.",
            )

    def test_instructions_file_or_inline_required(self) -> None:
        """Test that either file or inline is required."""
        with pytest.raises(ValidationError):
            Instructions()

    def test_instructions_file_not_empty(self) -> None:
        """Test that file cannot be empty string."""
        with pytest.raises(ValidationError):
            Instructions(file="")

    def test_instructions_inline_not_empty(self) -> None:
        """Test that inline cannot be empty string."""
        with pytest.raises(ValidationError):
            Instructions(inline="")

    def test_instructions_file_whitespace_only(self) -> None:
        """Test that file cannot be whitespace-only string."""
        with pytest.raises(ValidationError):
            Instructions(file="   ")

    def test_instructions_inline_whitespace_only(self) -> None:
        """Test that inline cannot be whitespace-only string."""
        with pytest.raises(ValidationError):
            Instructions(inline="   ")

    def test_instructions_file_with_whitespace(self) -> None:
        """Test that file with whitespace is accepted."""
        instructions = Instructions(file="  prompts/system.md  ")
        # Pydantic normalizes, but doesn't strip input strings by default
        assert "prompts/system.md" in instructions.file

    def test_instructions_no_extra_fields(self) -> None:
        """Test that Instructions rejects extra fields."""
        with pytest.raises(ValidationError):
            Instructions(inline="Test", extra_field="should_fail")


class TestAgent:
    """Tests for Agent model."""

    def test_agent_valid_creation(self) -> None:
        """Test creating a valid Agent."""
        model = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        agent = Agent(
            name="my_agent",
            model=model,
            instructions=Instructions(inline="You are helpful."),
        )
        assert agent.name == "my_agent"
        assert agent.model.provider == ProviderEnum.OPENAI
        assert agent.instructions.inline == "You are helpful."

    def test_agent_name_required(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )
        assert "name" in str(exc_info.value).lower()

    def test_agent_name_not_empty(self) -> None:
        """Test that name cannot be empty string."""
        with pytest.raises(ValidationError):
            Agent(
                name="",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_model_required(self) -> None:
        """Test that model is required."""
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="test",
                instructions=Instructions(inline="Test"),
            )
        assert "model" in str(exc_info.value).lower()

    def test_agent_instructions_required(self) -> None:
        """Test that instructions are required."""
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="test",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            )
        assert "instructions" in str(exc_info.value).lower()

    def test_agent_description_optional(self) -> None:
        """Test that description is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.description is None

    def test_agent_with_description(self) -> None:
        """Test Agent with description."""
        agent = Agent(
            name="test",
            description="A test agent",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.description == "A test agent"

    def test_agent_tools_optional(self) -> None:
        """Test that tools is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.tools is None or isinstance(agent.tools, list)

    def test_agent_with_tools(self) -> None:
        """Test Agent with tools."""
        tool = VectorstoreTool(
            name="search",
            description="Search documents",
            type="vectorstore",
            source="data.txt",
        )
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            tools=[tool],
        )
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search"

    def test_agent_evaluations_optional(self) -> None:
        """Test that evaluations is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.evaluations is None

    def test_agent_with_evaluations(self) -> None:
        """Test Agent with evaluations."""
        eval_config = EvaluationConfig(
            metrics=[
                EvaluationMetric(metric="groundedness"),
            ]
        )
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            evaluations=eval_config,
        )
        assert agent.evaluations is not None
        assert len(agent.evaluations.metrics) == 1

    def test_agent_test_cases_optional(self) -> None:
        """Test that test_cases is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.test_cases is None

    def test_agent_with_test_cases(self) -> None:
        """Test Agent with test cases."""
        test_case = TestCase(input="What is 2+2?")
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            test_cases=[test_case],
        )
        assert len(agent.test_cases) == 1
        assert agent.test_cases[0].input == "What is 2+2?"

    def test_agent_instructions_with_file(self) -> None:
        """Test Agent with file-based instructions."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(file="prompts/system.md"),
        )
        assert agent.instructions.file == "prompts/system.md"

    def test_agent_author_optional(self) -> None:
        """Test that author is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.author is None

    def test_agent_with_author(self) -> None:
        """Test Agent with author field."""
        agent = Agent(
            name="test",
            author="Alice Johnson",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.author == "Alice Johnson"

    def test_agent_author_not_empty(self) -> None:
        """Test that author cannot be empty string."""
        with pytest.raises(ValidationError):
            Agent(
                name="test",
                author="",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_author_whitespace_only(self) -> None:
        """Test that author cannot be whitespace-only string."""
        with pytest.raises(ValidationError):
            Agent(
                name="test",
                author="   ",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_name_whitespace_only(self) -> None:
        """Test that name cannot be whitespace-only string."""
        with pytest.raises(ValidationError):
            Agent(
                name="   ",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_description_whitespace_only(self) -> None:
        """Test that description cannot be whitespace-only string."""
        with pytest.raises(ValidationError):
            Agent(
                name="test",
                description="   ",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_description_not_empty(self) -> None:
        """Test that description cannot be empty string."""
        with pytest.raises(ValidationError):
            Agent(
                name="test",
                description="",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
            )

    def test_agent_tools_max_limit(self) -> None:
        """Test that agent cannot have more than 50 tools."""
        tools = [
            VectorstoreTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                type="vectorstore",
                source="data.txt",
            )
            for i in range(51)
        ]
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="test",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
                tools=tools,
            )
        assert "50" in str(exc_info.value).lower()

    def test_agent_tools_at_max_limit(self) -> None:
        """Test that agent can have exactly 50 tools."""
        tools = [
            VectorstoreTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                type="vectorstore",
                source="data.txt",
            )
            for i in range(50)
        ]
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            tools=tools,
        )
        assert len(agent.tools) == 50

    def test_agent_test_cases_max_limit(self) -> None:
        """Test that agent cannot have more than 100 test cases."""
        test_cases = [TestCase(input=f"Test case {i}") for i in range(101)]
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="test",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
                test_cases=test_cases,
            )
        assert "100" in str(exc_info.value).lower()

    def test_agent_test_cases_at_max_limit(self) -> None:
        """Test that agent can have exactly 100 test cases."""
        test_cases = [TestCase(input=f"Test case {i}") for i in range(100)]
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            test_cases=test_cases,
        )
        assert len(agent.test_cases) == 100

    def test_agent_no_extra_fields(self) -> None:
        """Test that Agent rejects extra fields."""
        with pytest.raises(ValidationError):
            Agent(
                name="test",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
                extra_field="should_fail",
            )

    def test_agent_all_fields(self) -> None:
        """Test Agent with all optional fields."""
        model = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-opus",
        )
        tool = VectorstoreTool(
            name="search",
            description="Search",
            type="vectorstore",
            source="data.txt",
        )
        eval_config = EvaluationConfig(
            metrics=[EvaluationMetric(metric="groundedness")]
        )
        test_case = TestCase(input="Test")

        agent = Agent(
            name="comprehensive_agent",
            description="An agent with all features",
            author="Alice Johnson",
            model=model,
            instructions=Instructions(inline="Instructions"),
            tools=[tool],
            evaluations=eval_config,
            test_cases=[test_case],
        )

        assert agent.name == "comprehensive_agent"
        assert agent.description == "An agent with all features"
        assert agent.author == "Alice Johnson"
        assert agent.model.provider == ProviderEnum.ANTHROPIC
        assert len(agent.tools) == 1
        assert agent.evaluations is not None
        assert len(agent.test_cases) == 1
