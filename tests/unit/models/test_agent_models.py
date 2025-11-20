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

    @pytest.mark.parametrize(
        "field,invalid_value",
        [
            ("file", ""),
            ("file", "   "),
            ("inline", ""),
            ("inline", "   "),
        ],
        ids=["file_empty", "file_whitespace", "inline_empty", "inline_whitespace"],
    )
    def test_instructions_string_fields_not_empty_or_whitespace(
        self, field: str, invalid_value: str
    ) -> None:
        """Test that string fields cannot be empty or whitespace-only."""
        kwargs = {field: invalid_value}
        with pytest.raises(ValidationError):
            Instructions(**kwargs)

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

    @pytest.mark.parametrize(
        "missing_field,kwargs",
        [
            (
                "name",
                {
                    "model": LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                    "instructions": Instructions(inline="Test"),
                },
            ),
            (
                "model",
                {
                    "name": "test",
                    "instructions": Instructions(inline="Test"),
                },
            ),
            (
                "instructions",
                {
                    "name": "test",
                    "model": LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                },
            ),
        ],
        ids=["name_required", "model_required", "instructions_required"],
    )
    def test_agent_required_fields(self, missing_field: str, kwargs: dict) -> None:
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            Agent(**kwargs)
        assert missing_field in str(exc_info.value).lower()

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

    @pytest.mark.parametrize(
        "field,invalid_value",
        [
            ("name", ""),
            ("name", "   "),
            ("description", ""),
            ("description", "   "),
            ("author", ""),
            ("author", "   "),
        ],
        ids=[
            "name_empty",
            "name_whitespace",
            "description_empty",
            "description_whitespace",
            "author_empty",
            "author_whitespace",
        ],
    )
    def test_agent_string_fields_not_empty_or_whitespace(
        self, field: str, invalid_value: str
    ) -> None:
        """Test that string fields cannot be empty or whitespace-only."""
        kwargs = {
            "name": "test",
            "model": LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            "instructions": Instructions(inline="Test"),
        }
        kwargs[field] = invalid_value
        with pytest.raises(ValidationError):
            Agent(**kwargs)

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

    def test_agent_response_format_optional(self) -> None:
        """Test that response_format is optional."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
        )
        assert agent.response_format is None

    def test_agent_response_format_inline_dict(self) -> None:
        """Test Agent with inline response_format dict."""
        response_format = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            response_format=response_format,
        )
        assert agent.response_format == response_format
        assert agent.response_format["type"] == "object"

    def test_agent_response_format_file_path(self) -> None:
        """Test Agent with response_format as file path."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            response_format="schemas/response.json",
        )
        assert agent.response_format == "schemas/response.json"

    def test_agent_response_format_null(self) -> None:
        """Test Agent with response_format explicitly set to null."""
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            response_format=None,
        )
        assert agent.response_format is None

    def test_agent_response_format_complex_nested_schema(self) -> None:
        """Test Agent with complex nested response_format schema."""
        response_format = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "name"],
                    },
                }
            },
            "required": ["results"],
        }
        agent = Agent(
            name="test",
            model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
            instructions=Instructions(inline="Test"),
            response_format=response_format,
        )
        assert agent.response_format == response_format

    def test_agent_response_format_invalid_type_raises_error(self) -> None:
        """Test that response_format with invalid type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Agent(
                name="test",
                model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o"),
                instructions=Instructions(inline="Test"),
                response_format=123,  # Invalid: number instead of dict/string/null
            )
        assert "response_format" in str(exc_info.value).lower()

    def test_agent_response_format_with_all_fields(self) -> None:
        """Test Agent with response_format and all other fields."""
        model = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o")
        tool = VectorstoreTool(
            name="search",
            description="Search",
            type="vectorstore",
            source="data.txt",
        )
        response_format = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }
        eval_config = EvaluationConfig(
            metrics=[EvaluationMetric(metric="groundedness")]
        )
        test_case = TestCase(input="Test")

        agent = Agent(
            name="comprehensive_with_response_format",
            description="Agent with response format",
            author="Test Author",
            model=model,
            instructions=Instructions(inline="Test instructions"),
            response_format=response_format,
            tools=[tool],
            evaluations=eval_config,
            test_cases=[test_case],
        )

        assert agent.name == "comprehensive_with_response_format"
        assert agent.response_format == response_format
        assert len(agent.tools) == 1
        assert agent.evaluations is not None
        assert len(agent.test_cases) == 1
