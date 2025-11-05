"""Unit tests for test executor module.

Tests cover:
- Configuration resolution (CLI > YAML > env > defaults)
- Tool call validation
- Timeout handling
- Main executor flow
- File processing and error handling
- Agent invocation with timeout and exceptions
- Evaluation metrics with different types and errors
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_case import FileInput, TestCaseModel
from holodeck.models.test_result import ProcessedFileInput


class TestToolCallValidation:
    """Tests for T049: Tool call validation against expected tools."""

    def test_exact_match_passes(self):
        """Tool calls exactly matching expected tools passes validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_mismatch_fails(self):
        """Tool calls not matching expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "get_weather"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_no_expected_tools_skips_validation(self):
        """When expected_tools is None, validation is skipped."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator"]
        expected = None

        result = validate_tool_calls(actual, expected)

        assert result is None

    def test_empty_expected_tools_with_calls_fails(self):
        """Agent calling tools when none expected fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = ["search_tool"]
        expected: list[str] = []

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_empty_actual_with_expected_fails(self):
        """Agent not calling expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = []
        expected: list[str] = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_order_independent_matching(self):
        """Tool call order doesn't matter, only set membership."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["calculator", "search_tool"]  # Different order
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_subset_of_expected_fails(self):
        """Calling subset of expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool"]  # Only one of two expected
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_superset_of_expected_fails(self):
        """Calling more tools than expected fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator", "extra_tool"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False


class TestExecutorMainFlow:
    """Tests for T053-T054: Main executor orchestration flow."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """TestExecutor initializes with agent config path and dependencies."""
        agent_config_path = "tests/fixtures/agents/test_agent.yaml"

        # Create mocks for dependencies
        mock_loader = Mock(spec=ConfigLoader)
        mock_file_processor = Mock(spec=FileProcessor)
        mock_agent_factory = Mock(spec=AgentFactory)

        # Setup mock config
        mock_config = Mock(spec=Agent)
        mock_config.name = "test_agent"
        mock_config.test_cases = []
        mock_config.evaluations = None
        mock_config.execution = None
        mock_loader.load_agent_yaml.return_value = mock_config

        # Inject dependencies
        executor = TestExecutor(
            agent_config_path=agent_config_path,
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_agent_factory,
        )

        assert executor.agent_config_path == agent_config_path
        assert executor.agent_config is not None
        assert executor.file_processor is mock_file_processor
        assert executor.agent_factory is mock_agent_factory
        mock_loader.load_agent_yaml.assert_called_once_with(agent_config_path)

    @pytest.mark.asyncio
    async def test_execute_test_cases_with_agent_response(self):
        """Single test case execution captures agent response."""
        from unittest.mock import AsyncMock

        from holodeck.lib.test_runner.agent_factory import AgentExecutionResult
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Create test case with ground truth
        test_case = TestCaseModel(
            name="test_1",
            input="What is 2+2?",
            expected_tools=None,
            ground_truth="The answer is 4",
            files=None,
            evaluations=None,
        )

        # Create evaluation config with METEOR and BLEU metrics
        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="meteor",
                    threshold=0.7,
                    enabled=True,
                ),
                EvaluationMetric(
                    metric="bleu",
                    threshold=0.6,
                    enabled=True,
                ),
            ],
        )

        # Create a real Agent instance
        agent_config = Agent(
            name="test_agent",
            description="Test agent for unit testing",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        # Mock config loader
        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        # Mock chat history with assistant response
        # Use Mock instead of MagicMock to allow direct attribute assignment
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "The answer is 4"
        mock_chat_history.messages = [mock_message]

        # Mock agent execution result
        mock_result = AgentExecutionResult(
            tool_calls=[],
            chat_history=mock_chat_history,
        )

        # Mock agent factory
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        # Mock file processor
        mock_file_processor = Mock(spec=FileProcessor)

        # Create executor with mocks
        executor = TestExecutor(
            agent_config_path="tests/fixtures/agents/test_agent.yaml",
            config_loader=mock_loader,
            agent_factory=mock_factory,
            file_processor=mock_file_processor,
        )

        # Execute all tests (which includes our single test case)
        report = await executor.execute_tests()

        # Verify report structure
        assert report.agent_name == "test_agent"
        assert report.summary.total_tests == 1
        assert report.summary.passed == 1
        assert report.summary.failed == 0
        assert report.summary.pass_rate == 1.0
        assert len(report.results) == 1

        # Verify test result
        result = report.results[0]
        assert result.test_name == "test_1"
        assert result.test_input == "What is 2+2?"
        assert result.agent_response == "The answer is 4"
        assert result.tool_calls == []
        assert result.expected_tools is None
        assert result.tools_matched is None
        assert result.passed is True
        assert result.execution_time_ms > 0
        assert result.errors == []
        assert result.timestamp is not None
        assert result.ground_truth == "The answer is 4"

        # Verify metric results
        assert len(result.metric_results) == 2
        metric_names = {m.metric_name for m in result.metric_results}
        assert "meteor" in metric_names
        assert "bleu" in metric_names

        # Verify each metric has expected fields
        for metric_result in result.metric_results:
            assert metric_result.score is not None
            assert metric_result.threshold is not None
            assert metric_result.passed is not None
            assert metric_result.scale == "0-1"

        # Verify agent factory was called
        mock_factory.invoke.assert_called_once()
        call_args = mock_factory.invoke.call_args[0][0]
        assert "What is 2+2?" in call_args


class TestFileProcessing:
    """Tests for file processing in test execution."""

    @pytest.mark.asyncio
    async def test_file_processing_with_error(self):
        """Test case with file processing error includes error in result."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Create test case with file input
        test_case = TestCaseModel(
            name="test_with_file",
            input="Analyze this file",
            expected_tools=None,
            ground_truth=None,
            files=[FileInput(path="test.txt", type="text")],
            evaluations=None,
        )

        # Create agent config
        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        # Mock config loader
        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        # Mock file processor with error
        mock_file_processor = Mock(spec=FileProcessor)
        mock_processed = Mock(spec=ProcessedFileInput)
        mock_processed.error = "File not found"
        mock_processed.markdown_content = None
        mock_file_processor.process_file.return_value = mock_processed

        # Mock agent factory
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        # Create executor
        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        # Execute tests
        report = await executor.execute_tests()

        # Verify error was recorded
        assert report.summary.total_tests == 1
        assert report.summary.failed == 1
        assert "File error: File not found" in report.results[0].errors

    @pytest.mark.asyncio
    async def test_file_processing_exception(self):
        """Test case with file processing exception records error."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_with_exception",
            input="Process file",
            expected_tools=None,
            ground_truth=None,
            files=[FileInput(path="test.txt", type="text")],
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        # Mock file processor to raise exception
        mock_file_processor = Mock(spec=FileProcessor)
        mock_file_processor.process_file.side_effect = OSError("Disk error")

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify exception was caught and recorded
        assert report.summary.failed == 1
        assert "File processing error: Disk error" in report.results[0].errors


class TestAgentInvocation:
    """Tests for agent invocation and error handling."""

    @pytest.mark.asyncio
    async def test_agent_timeout_error(self):
        """Test case records timeout error during agent invocation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_timeout",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=5
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent factory to raise TimeoutError
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(side_effect=TimeoutError())

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify timeout was recorded
        assert report.summary.failed == 1
        assert "Agent invocation timeout after 5s" in report.results[0].errors

    @pytest.mark.asyncio
    async def test_agent_invocation_generic_exception(self):
        """Test case records generic exception during agent invocation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_error",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent factory to raise generic exception
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(side_effect=ValueError("Invalid API key"))

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify error was recorded
        assert report.summary.failed == 1
        assert "Agent invocation error: Invalid API key" in report.results[0].errors


class TestFileContentInAgent:
    """Tests for file content inclusion in agent input."""

    @pytest.mark.asyncio
    async def test_file_content_included_in_agent_input(self):
        """Test that processed file content is included in agent input."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_with_file",
            input="Analyze the file",
            expected_tools=None,
            ground_truth=None,
            files=[FileInput(path="test.md", type="text")],
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        # Mock file processor with markdown content
        mock_file_processor = Mock(spec=FileProcessor)
        mock_processed = Mock(spec=ProcessedFileInput)
        mock_processed.error = None
        mock_processed.markdown_content = "# File Content\nThis is test content"
        mock_processed.original = Mock()
        mock_processed.original.path = "test.md"
        mock_file_processor.process_file.return_value = mock_processed

        # Track what gets passed to agent
        captured_input: str | None = None

        async def capture_invoke(agent_input: str):
            nonlocal captured_input
            captured_input = agent_input
            mock_result = Mock()
            mock_result.tool_calls = []
            mock_chat_history = Mock()
            mock_message = Mock()
            mock_message.role = "assistant"
            mock_message.content = "Analysis complete"
            mock_chat_history.messages = [mock_message]
            mock_result.chat_history = mock_chat_history
            return mock_result

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(side_effect=capture_invoke)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        await executor.execute_tests()

        # Verify file content was included in agent input
        assert captured_input is not None
        assert "File: test.md" in captured_input
        assert "# File Content" in captured_input
        assert "Analyze the file" in captured_input


class TestChatHistoryHandling:
    """Tests for extracting response from chat history."""

    @pytest.mark.asyncio
    async def test_empty_chat_history(self):
        """Test that empty chat history returns empty response."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_empty_history",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent factory with empty chat history
        mock_factory = Mock(spec=AgentFactory)
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = None  # Empty chat history
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify response is empty string
        assert report.results[0].agent_response == ""


class TestEvaluationMetrics:
    """Tests for different evaluation metrics and error handling."""

    @pytest.mark.asyncio
    async def test_groundedness_evaluator(self):
        """Test groundedness metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_groundedness",
            input="Query",
            expected_tools=None,
            ground_truth="Expected answer",
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="groundedness",
                    threshold=0.7,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Agent response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        # Mock evaluator
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.8})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"groundedness": mock_evaluator},
        )

        report = await executor.execute_tests()

        # Verify metric was evaluated
        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "groundedness"
        assert report.results[0].metric_results[0].score == 0.8

    @pytest.mark.asyncio
    async def test_relevance_evaluator(self):
        """Test relevance metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_relevance",
            input="What is AI?",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="relevance",
                    threshold=0.6,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "AI is artificial intelligence"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.9})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"relevance": mock_evaluator},
        )

        report = await executor.execute_tests()

        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "relevance"
        assert report.results[0].metric_results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_bleu_evaluator(self):
        """Test BLEU NLP metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_bleu",
            input="Translate hello",
            expected_tools=None,
            ground_truth="hola",
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="bleu",
                    threshold=0.5,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "hola"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"bleu": 1.0})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"bleu": mock_evaluator},
        )

        report = await executor.execute_tests()

        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "bleu"

    @pytest.mark.asyncio
    async def test_evaluation_failure_recorded(self):
        """Test that evaluation failures are recorded in metric results."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_eval_error",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="groundedness",
                    threshold=0.7,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        # Mock evaluator that raises exception
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            side_effect=RuntimeError("Evaluator failed")
        )

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"groundedness": mock_evaluator},
        )

        report = await executor.execute_tests()

        # Verify error was recorded
        assert len(report.results[0].metric_results) == 1
        metric = report.results[0].metric_results[0]
        assert metric.error == "Evaluator failed"
        assert metric.passed is False


class TestToolCallValidationInExecution:
    """Tests for tool call validation during execution."""

    @pytest.mark.asyncio
    async def test_tool_calls_validated_in_test(self):
        """Test that tool calls are validated and recorded."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_tools",
            input="Use these tools",
            expected_tools=["search", "calculator"],
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Result"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = [
            {"name": "search"},
            {"name": "calculator"},
        ]
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify tool calls were validated
        assert report.results[0].tool_calls == ["search", "calculator"]
        assert report.results[0].tools_matched is True


class TestMoreEvaluators:
    """Tests for additional evaluator types."""

    @pytest.mark.asyncio
    async def test_coherence_evaluator(self):
        """Test coherence metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_coherence",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="coherence",
                    threshold=0.7,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Coherent response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.85})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"coherence": mock_evaluator},
        )

        report = await executor.execute_tests()

        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "coherence"

    @pytest.mark.asyncio
    async def test_fluency_evaluator(self):
        """Test fluency metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_fluency",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="fluency",
                    threshold=0.75,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Fluent response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.9})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"fluency": mock_evaluator},
        )

        report = await executor.execute_tests()

        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "fluency"

    @pytest.mark.asyncio
    async def test_rouge_evaluator(self):
        """Test ROUGE NLP metric evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_rouge",
            input="Summarize",
            expected_tools=None,
            ground_truth="Expected summary",
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="rouge",
                    threshold=0.5,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Summary"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"rouge": 0.75})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"rouge": mock_evaluator},
        )

        report = await executor.execute_tests()

        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "rouge"


class TestContextInEvaluation:
    """Tests for context inclusion in evaluations."""

    @pytest.mark.asyncio
    async def test_context_passed_to_groundedness_metric(self):
        """Test that file context is passed to groundedness evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_groundedness_context",
            input="Question",
            expected_tools=None,
            ground_truth=None,
            files=[FileInput(path="context.txt", type="text")],
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="groundedness",
                    threshold=0.7,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        # Mock file processor with content
        mock_file_processor = Mock(spec=FileProcessor)
        mock_processed = Mock(spec=ProcessedFileInput)
        mock_processed.error = None
        mock_processed.markdown_content = "Context information"
        mock_processed.original = Mock()
        mock_processed.original.path = "context.txt"
        mock_file_processor.process_file.return_value = mock_processed

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response based on context"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        # Track what gets passed to evaluator
        evaluation_kwargs: dict = {}

        async def capture_evaluate(**kwargs) -> dict:
            evaluation_kwargs.update(kwargs)
            return {"score": 0.85}

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(side_effect=capture_evaluate)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"groundedness": mock_evaluator},
        )

        await executor.execute_tests()

        # Verify context was passed to evaluator
        assert "context" in evaluation_kwargs
        assert "Context information" in evaluation_kwargs["context"]


class TestNoMetricsConfigured:
    """Tests for cases where no metrics are configured."""

    @pytest.mark.asyncio
    async def test_no_evaluations_configured(self):
        """Test execution when no evaluations are configured."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_no_eval",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,  # No evaluations
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify test passed with no metrics
        assert report.results[0].passed is True
        assert len(report.results[0].metric_results) == 0


class TestReportGeneration:
    """Tests for test report generation and summary statistics."""

    @pytest.mark.asyncio
    async def test_report_summary_statistics(self):
        """Test that report summary calculates correct statistics."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Create multiple test cases
        test_case_1 = TestCaseModel(
            name="test_1",
            input="Query 1",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        test_case_2 = TestCaseModel(
            name="test_2",
            input="Query 2",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="meteor",
                    threshold=0.5,
                    enabled=True,
                ),
            ],
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case_1, test_case_2],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={"meteor": 0.8})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"meteor": mock_evaluator},
        )

        report = await executor.execute_tests()

        # Verify summary
        assert report.summary.total_tests == 2
        assert report.summary.passed == 2
        assert report.summary.failed == 0
        assert report.summary.pass_rate == 1.0
        assert "meteor" in report.summary.metrics_evaluated

    @pytest.mark.asyncio
    async def test_version_import_fallback(self):
        """Test that version fallback works if import fails."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_version",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[test_case],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.chat_history = mock_chat_history

        mock_factory = Mock(spec=AgentFactory)
        mock_factory.invoke = AsyncMock(return_value=mock_result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        # Simulate import failure
        with patch("builtins.__import__", side_effect=ImportError()):
            report = await executor.execute_tests()

        # Verify fallback version is used
        assert report.holodeck_version == "0.1.0"
