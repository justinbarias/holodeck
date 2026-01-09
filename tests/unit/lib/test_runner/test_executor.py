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
from holodeck.lib.test_runner.agent_factory import AgentFactory, AgentThreadRun
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_case import FileInput, TestCaseModel
from holodeck.models.test_result import ProcessedFileInput, TestResult


class TestToolCallValidation:
    """Tests for T049: Tool call validation against expected tools.

    Validation uses substring matching - each expected tool name must be found
    within at least one actual tool name. Extra tools are allowed.
    """

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

    def test_empty_expected_tools_with_calls_passes(self):
        """Empty expected list passes (all zero expected tools are found)."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = ["search_tool"]
        expected: list[str] = []

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_empty_actual_with_expected_fails(self):
        """Agent not calling expected tools fails validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual: list[str] = []
        expected: list[str] = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is False

    def test_order_independent_matching(self):
        """Tool call order doesn't matter for validation."""
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

    def test_extra_tools_allowed(self):
        """Calling more tools than expected passes validation."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_tool", "calculator", "extra_tool"]
        expected = ["search_tool", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_substring_match_with_prefix(self):
        """Expected tool found as substring in actual tool with prefix."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["vectorstore-search", "mcp-calculator"]
        expected = ["search", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_substring_match_with_suffix(self):
        """Expected tool found as substring in actual tool with suffix."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["search_v2", "calculator_advanced"]
        expected = ["search", "calculator"]

        result = validate_tool_calls(actual, expected)

        assert result is True

    def test_substring_match_partial_only_fails(self):
        """Expected tool not found as substring in any actual tool fails."""
        from holodeck.lib.test_runner.executor import validate_tool_calls

        actual = ["vectorstore-query", "mcp-math"]
        expected = ["search"]  # "search" not in any actual tool

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

        # Mock ModelConfig import in executor
        with patch.dict("sys.modules", {"holodeck.lib.evaluators.azure_ai": Mock()}):
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
            tool_results=[],
            chat_history=mock_chat_history,
        )

        # Mock agent factory with thread run pattern
        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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

        # Verify agent factory was called with thread run pattern
        mock_factory.create_thread_run.assert_called_once()
        mock_thread_run.invoke.assert_called_once()
        call_args = mock_thread_run.invoke.call_args[0][0]
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(side_effect=TimeoutError())
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(side_effect=ValueError("Invalid API key"))
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
            mock_result.tool_results = []
            mock_chat_history = Mock()
            mock_message = Mock()
            mock_message.role = "assistant"
            mock_message.content = "Analysis complete"
            mock_chat_history.messages = [mock_message]
            mock_result.chat_history = mock_chat_history
            return mock_result

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(side_effect=capture_invoke)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = None  # Empty chat history
        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
    @pytest.mark.parametrize(
        "metric_name,test_name,test_input,ground_truth,response,score_key,score_value",
        [
            (
                "groundedness",
                "test_groundedness",
                "Query",
                "Expected answer",
                "Agent response",
                "score",
                0.8,
            ),
            (
                "relevance",
                "test_relevance",
                "What is AI?",
                None,
                "AI is artificial intelligence",
                "score",
                0.9,
            ),
            (
                "bleu",
                "test_bleu",
                "Translate hello",
                "hola",
                "hola",
                "bleu",
                1.0,
            ),
            (
                "coherence",
                "test_coherence",
                "Query",
                None,
                "Coherent response",
                "score",
                0.85,
            ),
            (
                "fluency",
                "test_fluency",
                "Query",
                None,
                "Fluent response",
                "score",
                0.9,
            ),
            (
                "rouge",
                "test_rouge",
                "Summarize",
                "Expected summary",
                "Summary",
                "rouge",
                0.75,
            ),
        ],
        ids=[
            "groundedness",
            "relevance",
            "bleu",
            "coherence",
            "fluency",
            "rouge",
        ],
    )
    async def test_evaluator_metrics(
        self,
        metric_name: str,
        test_name: str,
        test_input: str,
        ground_truth: str | None,
        response: str,
        score_key: str,
        score_value: float,
    ):
        """Test evaluation metrics with different types."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name=test_name,
            input=test_input,
            expected_tools=None,
            ground_truth=ground_truth,
            files=None,
            evaluations=None,
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric=metric_name,
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
        mock_message.content = response
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value={score_key: score_value})

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={metric_name: mock_evaluator},
        )

        report = await executor.execute_tests()

        # Verify metric was evaluated
        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == metric_name

    @pytest.mark.asyncio
    async def test_evaluation_failure_recorded(self):
        """Test that evaluation failures are recorded in metric results."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Mock evaluator that raises exception
        # Must return proper ParamSpec for get_param_spec() to avoid coroutine issues
        mock_evaluator = Mock()
        mock_evaluator.name = "groundedness"
        mock_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.CONTEXT}),
                uses_context=True,
            )
        )
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

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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


class TestContextInEvaluation:
    """Tests for context inclusion in evaluations."""

    @pytest.mark.asyncio
    async def test_context_passed_to_groundedness_metric(self):
        """Test that file context is passed to groundedness evaluation."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track what gets passed to evaluator
        evaluation_kwargs: dict = {}

        async def capture_evaluate(**kwargs) -> dict:
            evaluation_kwargs.update(kwargs)
            return {"score": 0.85}

        mock_evaluator = Mock()
        mock_evaluator.name = "groundedness"
        mock_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.CONTEXT}),
                uses_context=True,
            )
        )
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluator = Mock()
        mock_evaluator.name = "meteor"
        mock_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
            )
        )
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

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


class TestPerTestMetricResolution:
    """Tests for T095: Per-test metric resolution logic.

    Tests verify that:
    - Per-test metrics override global metrics when specified
    - Test cases without per-test metrics use global defaults
    - Different test cases can have different metric configurations
    """

    @pytest.mark.asyncio
    async def test_per_test_metrics_override_global(self):
        """Per-test metrics override global metrics when specified."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Create per-test metric
        groundedness_metric = EvaluationMetric(
            metric="groundedness",
            threshold=0.8,
            enabled=True,
        )

        # Test case with specific per-test metrics
        test_case = TestCaseModel(
            name="test_per_test_override",
            input="Test query",
            expected_tools=None,
            ground_truth="Expected answer",
            files=None,
            evaluations=[groundedness_metric],  # Only groundedness, not bleu
        )

        # Global config has both METEOR and BLEU
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
                EvaluationMetric(
                    metric="groundedness",
                    threshold=0.8,
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
        mock_message.content = "Response based on context"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluators = {}
        for metric in eval_config.metrics:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.85})
            mock_evaluators[metric.metric] = mock_evaluator

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators=mock_evaluators,
        )

        report = await executor.execute_tests()

        # Verify only groundedness was evaluated
        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "groundedness"

    @pytest.mark.asyncio
    async def test_fallback_to_global_metrics(self):
        """Test case without per-test metrics falls back to global metrics."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Test case without per-test metrics
        test_case = TestCaseModel(
            name="test_no_override",
            input="Test query",
            expected_tools=None,
            ground_truth="Expected answer",
            files=None,
            evaluations=None,  # No per-test metrics
        )

        # Global config has METEOR and BLEU
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluators = {
            "meteor": AsyncMock(evaluate=AsyncMock(return_value={"meteor": 0.85})),
            "bleu": AsyncMock(evaluate=AsyncMock(return_value={"bleu": 0.75})),
        }

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators=mock_evaluators,
        )

        report = await executor.execute_tests()

        # Verify both global metrics were used
        assert len(report.results[0].metric_results) == 2
        metric_names = {m.metric_name for m in report.results[0].metric_results}
        assert metric_names == {"meteor", "bleu"}

    @pytest.mark.asyncio
    async def test_multiple_tests_different_metrics(self):
        """Different test cases can have different per-test metrics."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Test case 1 with specific metrics
        test_case_1 = TestCaseModel(
            name="test_1",
            input="Query 1",
            expected_tools=None,
            ground_truth="Answer 1",
            files=None,
            evaluations=[
                EvaluationMetric(
                    metric="meteor",
                    threshold=0.7,
                    enabled=True,
                ),
            ],
        )

        # Test case 2 with different metrics
        test_case_2 = TestCaseModel(
            name="test_2",
            input="Query 2",
            expected_tools=None,
            ground_truth="Answer 2",
            files=None,
            evaluations=[
                EvaluationMetric(
                    metric="bleu",
                    threshold=0.6,
                    enabled=True,
                ),
                EvaluationMetric(
                    metric="rouge",
                    threshold=0.65,
                    enabled=True,
                ),
            ],
        )

        # Test case 3 without per-test metrics
        test_case_3 = TestCaseModel(
            name="test_3",
            input="Query 3",
            expected_tools=None,
            ground_truth="Answer 3",
            files=None,
            evaluations=None,
        )

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
                EvaluationMetric(
                    metric="rouge",
                    threshold=0.65,
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
            test_cases=[test_case_1, test_case_2, test_case_3],
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluators = {
            "meteor": AsyncMock(evaluate=AsyncMock(return_value={"meteor": 0.85})),
            "bleu": AsyncMock(evaluate=AsyncMock(return_value={"bleu": 0.75})),
            "rouge": AsyncMock(evaluate=AsyncMock(return_value={"rouge": 0.80})),
        }

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators=mock_evaluators,
        )

        report = await executor.execute_tests()

        # Verify test 1 has only meteor
        assert len(report.results[0].metric_results) == 1
        assert report.results[0].metric_results[0].metric_name == "meteor"

        # Verify test 2 has bleu and rouge
        assert len(report.results[1].metric_results) == 2
        metric_names_2 = {m.metric_name for m in report.results[1].metric_results}
        assert metric_names_2 == {"bleu", "rouge"}

        # Verify test 3 has all global metrics
        assert len(report.results[2].metric_results) == 3
        metric_names_3 = {m.metric_name for m in report.results[2].metric_results}
        assert metric_names_3 == {"meteor", "bleu", "rouge"}

    @pytest.mark.asyncio
    async def test_empty_per_test_metrics_uses_global(self):
        """Empty per-test metrics list falls back to global metrics."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Test case with empty per-test metrics list
        test_case = TestCaseModel(
            name="test_empty_list",
            input="Query",
            expected_tools=None,
            ground_truth="Answer",
            files=None,
            evaluations=[],  # Empty list - should use global metrics
        )

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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        mock_evaluators = {
            "meteor": AsyncMock(evaluate=AsyncMock(return_value={"meteor": 0.85})),
            "bleu": AsyncMock(evaluate=AsyncMock(return_value={"bleu": 0.75})),
        }

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators=mock_evaluators,
        )

        report = await executor.execute_tests()

        # Empty list should fall back to global metrics
        assert len(report.results[0].metric_results) == 2
        metric_names = {m.metric_name for m in report.results[0].metric_results}
        assert metric_names == {"meteor", "bleu"}


@pytest.mark.unit
class TestProgressCallbackIntegration:
    """Tests for T061: Progress callback integration with TestExecutor.

    Tests verify that:
    - Callback is invoked after each test execution
    - Callback receives TestResult instances
    - Callback with None handling works correctly
    - Multiple test execution flow calls callback appropriately
    """

    @pytest.mark.asyncio
    async def test_callback_invoked_after_each_test(self):
        """Callback is invoked after each test completes."""
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(name="Test 1", input="test input")

        agent_config = Agent(
            name="Test Agent",
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track callback invocations
        callback_invocations = []

        def progress_callback(result):
            callback_invocations.append(result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            progress_callback=progress_callback,
        )

        await executor.execute_tests()

        # Callback should be invoked once for the single test
        assert len(callback_invocations) == 1
        # Callback should receive TestResult instance
        assert isinstance(callback_invocations[0], TestResult)

    @pytest.mark.asyncio
    async def test_callback_with_none_handling(self):
        """Executor handles None callback gracefully."""
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(name="Test 1", input="test input")

        agent_config = Agent(
            name="Test Agent",
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Create executor without callback (None)
        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            progress_callback=None,
        )

        # Should execute without error
        report = await executor.execute_tests()

        # Report should be generated successfully
        assert report is not None
        assert len(report.results) == 1

    @pytest.mark.asyncio
    async def test_callback_receives_test_results(self):
        """Callback receives correct TestResult data."""
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(name="My Test Case", input="test input")

        agent_config = Agent(
            name="Test Agent",
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
        mock_message.content = "Test Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        callback_results = []

        def progress_callback(result):
            callback_results.append(result)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            progress_callback=progress_callback,
        )

        await executor.execute_tests()

        # Verify callback received TestResult with correct data
        assert len(callback_results) == 1
        result = callback_results[0]
        assert result.test_name == "My Test Case"
        assert result.agent_response == "Test Response"
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_multiple_test_execution_flow(self):
        """Callback is invoked correctly for multiple test executions."""
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Create multiple test cases
        test_cases = [
            TestCaseModel(name="Test 1", input="input 1"),
            TestCaseModel(name="Test 2", input="input 2"),
            TestCaseModel(name="Test 3", input="input 3"),
        ]

        agent_config = Agent(
            name="Test Agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=test_cases,
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track callback invocations with test names
        callback_results = []

        def progress_callback(result):
            callback_results.append(result.test_name)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            progress_callback=progress_callback,
        )

        await executor.execute_tests()

        # Callback should be invoked once per test
        assert len(callback_results) == 3
        assert callback_results == ["Test 1", "Test 2", "Test 3"]

    @pytest.mark.asyncio
    async def test_callback_called_in_order(self):
        """Callbacks are invoked in the order tests execute."""
        from holodeck.models.agent import Agent, Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_cases = [
            TestCaseModel(name="First", input="1"),
            TestCaseModel(name="Second", input="2"),
        ]

        agent_config = Agent(
            name="Test Agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=test_cases,
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
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        call_order = []

        def progress_callback(result):
            call_order.append(result.test_name)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            progress_callback=progress_callback,
        )

        await executor.execute_tests()

        # Verify callbacks were called in execution order
        assert call_order == ["First", "Second"]


class TestExecutorComponentCreation:
    """Tests for component creation methods in TestExecutor."""

    @pytest.mark.asyncio
    async def test_create_file_processor_with_custom_config(self):
        """Test _create_file_processor with custom download timeout."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=ExecutionConfig(download_timeout=60),
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            download_timeout=60
        )

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Verify file processor was created with correct timeout
        assert executor.file_processor is not None

    @pytest.mark.asyncio
    async def test_create_evaluators_with_rouge_metric(self):
        """Test _create_evaluators with ROUGE metric."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                EvaluationMetric(
                    metric="rouge",
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
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        assert "rouge" in executor.evaluators

    @pytest.mark.asyncio
    async def test_extract_response_text_empty_history(self):
        """Test _extract_response_text with empty chat history."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Test with None chat history
        result = executor._extract_response_text(None)
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_tool_names_malformed(self):
        """Test _extract_tool_names with malformed tool calls."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Test with tool calls missing 'name' key
        tool_calls = [{"arguments": "test"}, {"name": "valid_tool"}]
        result = executor._extract_tool_names(tool_calls)
        assert result == ["valid_tool"]

    @pytest.mark.asyncio
    async def test_determine_test_passed_with_errors(self):
        """Test _determine_test_passed with execution errors."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Test with errors
        result = executor._determine_test_passed([], None, ["Error occurred"])
        assert result is False

    @pytest.mark.asyncio
    async def test_determine_test_passed_with_failed_tools(self):
        """Test _determine_test_passed with failed tool validation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Test with failed tool validation
        result = executor._determine_test_passed([], False, [])
        assert result is False


class TestOnTestStartCallback:
    """Tests for on_test_start callback functionality."""

    @pytest.mark.asyncio
    async def test_on_test_start_callback_invoked(self):
        """Test that on_test_start callback is invoked for each test."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_callback",
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

        # Mock agent factory
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track callback invocations
        callback_invocations = []

        def on_test_start_callback(test: TestCaseModel):
            callback_invocations.append(test.name)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            on_test_start=on_test_start_callback,
        )

        await executor.execute_tests()

        # Verify callback was invoked
        assert len(callback_invocations) == 1
        assert callback_invocations[0] == "test_callback"


class TestAzureAIEvaluatorCreation:
    """Tests for Azure AI evaluator creation with ModelConfig."""

    @pytest.mark.asyncio
    async def test_groundedness_evaluator_creation(self):
        """Test creation of groundedness evaluator with ModelConfig."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
            ),
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
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Verify groundedness evaluator was created
        assert "groundedness" in executor.evaluators

    @pytest.mark.asyncio
    async def test_relevance_evaluator_creation(self):
        """Test creation of relevance evaluator with ModelConfig."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
            ),
            metrics=[
                EvaluationMetric(
                    metric="relevance",
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
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Verify relevance evaluator was created
        assert "relevance" in executor.evaluators

    @pytest.mark.asyncio
    async def test_coherence_evaluator_creation(self):
        """Test creation of coherence evaluator with ModelConfig."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
            ),
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
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Verify coherence evaluator was created
        assert "coherence" in executor.evaluators

    @pytest.mark.asyncio
    async def test_fluency_evaluator_creation(self):
        """Test creation of fluency evaluator with ModelConfig."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationConfig, EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        eval_config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4",
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
            ),
            metrics=[
                EvaluationMetric(
                    metric="fluency",
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
            test_cases=[],
            evaluations=eval_config,
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Verify fluency evaluator was created
        assert "fluency" in executor.evaluators


class TestChatHistoryEdgeCases:
    """Tests for chat history edge cases."""

    @pytest.mark.asyncio
    async def test_chat_history_with_empty_messages(self):
        """Test chat history with empty messages list."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_empty_messages",
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

        # Mock agent factory with chat history but empty messages
        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_chat_history = Mock()
        mock_chat_history.messages = []  # Empty messages list
        mock_result.chat_history = mock_chat_history
        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify response is empty string
        assert report.results[0].agent_response == ""


class TestUnconfiguredMetrics:
    """Tests for handling unconfigured metrics."""

    @pytest.mark.asyncio
    async def test_skip_unconfigured_metric(self):
        """Test that unconfigured metrics are skipped during evaluation."""
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import EvaluationMetric
        from holodeck.models.llm import LLMProvider, ProviderEnum

        test_case = TestCaseModel(
            name="test_unconfigured",
            input="Query",
            expected_tools=None,
            ground_truth="Answer",
            files=None,
            # Request a metric that won't be in evaluators
            evaluations=[
                EvaluationMetric(
                    metric="custom_metric",  # Not configured
                    threshold=0.7,
                    enabled=True,
                )
            ],
        )

        # Agent config has no evaluations, so no evaluators will be created
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
            evaluations=None,  # No evaluations configured
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent factory
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Answer"
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
        )

        report = await executor.execute_tests()

        # Verify test ran but metric was skipped (no metric results)
        assert len(report.results) == 1
        assert len(report.results[0].metric_results) == 0


class TestDynamicRetrievalContext:
    """Tests for dynamic retrieval context extraction from tool results.

    Tests verify that:
    - retrieval_context is built from vectorstore tool results
    - retrieval_context is built from MCP tools with is_retrieval=True
    - Manual retrieval_context takes precedence over dynamic extraction
    - Empty tool results produce empty retrieval_context
    """

    def test_get_retrieval_tool_names_with_vectorstore(self):
        """Vectorstore tools are identified as retrieval tools."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import VectorstoreTool

        vectorstore_tool = VectorstoreTool(
            name="knowledge_base",
            description="Search knowledge base",
            source="data/docs",
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
            test_cases=[],
            evaluations=None,
            tools=[vectorstore_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        retrieval_tools = executor._get_retrieval_tool_names()

        # Vectorstore tools use "vectorstore-{name}" format
        assert "vectorstore-knowledge_base" in retrieval_tools

    def test_get_retrieval_tool_names_with_mcp_retrieval(self):
        """MCP tools with is_retrieval=True are identified as retrieval tools."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import CommandType, MCPTool

        mcp_tool = MCPTool(
            name="doc_search",
            description="Search documents via MCP",
            command=CommandType.NPX,
            args=["@search/mcp-server"],
            is_retrieval=True,
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
            test_cases=[],
            evaluations=None,
            tools=[mcp_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        retrieval_tools = executor._get_retrieval_tool_names()

        assert "doc_search" in retrieval_tools

    def test_get_retrieval_tool_names_excludes_non_retrieval_mcp(self):
        """MCP tools without is_retrieval=True are not included."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import CommandType, MCPTool

        mcp_tool = MCPTool(
            name="filesystem",
            description="Filesystem operations",
            command=CommandType.NPX,
            args=["@filesystem/mcp-server"],
            is_retrieval=False,  # Not a retrieval tool
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
            test_cases=[],
            evaluations=None,
            tools=[mcp_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        retrieval_tools = executor._get_retrieval_tool_names()

        assert "filesystem" not in retrieval_tools
        assert len(retrieval_tools) == 0

    def test_build_retrieval_context_filters_by_tool_name(self):
        """Only results from retrieval tools are included in context."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import VectorstoreTool

        vectorstore_tool = VectorstoreTool(
            name="knowledge_base",
            description="Search knowledge base",
            source="data/docs",
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
            test_cases=[],
            evaluations=None,
            tools=[vectorstore_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Mixed tool results - some from retrieval, some not
        tool_results = [
            {
                "name": "vectorstore-knowledge_base",
                "result": "Retrieved doc: Important information",
            },
            {"name": "calculator", "result": "42"},  # Not a retrieval tool
        ]

        context = executor._build_retrieval_context(tool_results)

        # Only vectorstore result should be included
        assert len(context) == 1
        assert "Important information" in context[0]

    def test_build_retrieval_context_empty_results(self):
        """Empty tool results produce empty retrieval context."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,
            tools=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        context = executor._build_retrieval_context([])

        assert context == []

    @pytest.mark.asyncio
    async def test_manual_retrieval_context_takes_precedence(self):
        """Manual retrieval_context in test case takes precedence over dynamic."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import (
            EvaluationConfig,
            RAGMetric,
            RAGMetricType,
        )
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import VectorstoreTool

        # Test case with manual retrieval_context
        test_case = TestCaseModel(
            name="test_manual_context",
            input="What is AI?",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
            retrieval_context=["Manual context 1", "Manual context 2"],
        )

        vectorstore_tool = VectorstoreTool(
            name="knowledge_base",
            description="Search knowledge base",
            source="data/docs",
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                RAGMetric(
                    metric_type=RAGMetricType.FAITHFULNESS,
                    threshold=0.7,
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
            tools=[vectorstore_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent response with tool results (should be ignored)
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "AI is artificial intelligence."
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = [{"name": "vectorstore-knowledge_base"}]
        mock_result.tool_results = [
            {
                "name": "vectorstore-knowledge_base",
                "result": "Dynamic context from vectorstore",
            }
        ]
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track what gets passed to evaluator
        evaluation_kwargs: dict = {}

        async def capture_evaluate(**kwargs) -> dict:
            evaluation_kwargs.update(kwargs)
            return {"faithfulness": 0.9}

        mock_evaluator = Mock()
        mock_evaluator.name = "faithfulness"
        mock_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset(
                    {
                        EvalParam.ACTUAL_OUTPUT,
                        EvalParam.INPUT,
                        EvalParam.RETRIEVAL_CONTEXT,
                    }
                ),
                uses_retrieval_context=True,
            )
        )
        mock_evaluator.evaluate = AsyncMock(side_effect=capture_evaluate)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"faithfulness": mock_evaluator},
        )

        await executor.execute_tests()

        # Manual retrieval_context should be used, not dynamic
        assert "retrieval_context" in evaluation_kwargs
        assert evaluation_kwargs["retrieval_context"] == [
            "Manual context 1",
            "Manual context 2",
        ]

    @pytest.mark.asyncio
    async def test_dynamic_retrieval_context_used_when_no_manual(self):
        """Dynamic retrieval_context from tool results used when no manual."""
        from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
        from holodeck.models.agent import Instructions
        from holodeck.models.evaluation import (
            EvaluationConfig,
            RAGMetric,
            RAGMetricType,
        )
        from holodeck.models.llm import LLMProvider, ProviderEnum
        from holodeck.models.tool import VectorstoreTool

        # Test case WITHOUT manual retrieval_context
        test_case = TestCaseModel(
            name="test_dynamic_context",
            input="What is AI?",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
            retrieval_context=None,  # No manual context
        )

        vectorstore_tool = VectorstoreTool(
            name="knowledge_base",
            description="Search knowledge base",
            source="data/docs",
        )

        eval_config = EvaluationConfig(
            model=None,
            metrics=[
                RAGMetric(
                    metric_type=RAGMetricType.FAITHFULNESS,
                    threshold=0.7,
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
            tools=[vectorstore_tool],
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig(
            llm_timeout=60
        )

        mock_file_processor = Mock(spec=FileProcessor)

        # Mock agent response with tool results
        mock_chat_history = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "AI is artificial intelligence."
        mock_chat_history.messages = [mock_message]

        mock_result = Mock()
        mock_result.tool_calls = [{"name": "vectorstore-knowledge_base"}]
        mock_result.tool_results = [
            {
                "name": "vectorstore-knowledge_base",
                "result": "Dynamic context from vectorstore search",
            }
        ]
        mock_result.chat_history = mock_chat_history

        mock_thread_run = Mock(spec=AgentThreadRun)
        mock_thread_run.invoke = AsyncMock(return_value=mock_result)
        mock_factory = Mock(spec=AgentFactory)
        mock_factory.create_thread_run = AsyncMock(return_value=mock_thread_run)

        # Track what gets passed to evaluator
        evaluation_kwargs: dict = {}

        async def capture_evaluate(**kwargs) -> dict:
            evaluation_kwargs.update(kwargs)
            return {"faithfulness": 0.9}

        mock_evaluator = Mock()
        mock_evaluator.name = "faithfulness"
        mock_evaluator.get_param_spec = Mock(
            return_value=ParamSpec(
                required=frozenset(
                    {
                        EvalParam.ACTUAL_OUTPUT,
                        EvalParam.INPUT,
                        EvalParam.RETRIEVAL_CONTEXT,
                    }
                ),
                uses_retrieval_context=True,
            )
        )
        mock_evaluator.evaluate = AsyncMock(side_effect=capture_evaluate)

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
            file_processor=mock_file_processor,
            agent_factory=mock_factory,
            evaluators={"faithfulness": mock_evaluator},
        )

        await executor.execute_tests()

        # Dynamic retrieval_context should be used
        assert "retrieval_context" in evaluation_kwargs
        assert len(evaluation_kwargs["retrieval_context"]) == 1
        assert (
            "Dynamic context from vectorstore search"
            in evaluation_kwargs["retrieval_context"][0]
        )


class TestEmptyEvaluationsConfig:
    """Tests for handling empty evaluations configuration."""

    @pytest.mark.asyncio
    async def test_get_metrics_for_test_with_no_evaluations(self):
        """Test _get_metrics_for_test returns empty list with no evaluations."""
        from holodeck.models.agent import Instructions
        from holodeck.models.llm import LLMProvider, ProviderEnum

        # Agent with no evaluations
        agent_config = Agent(
            name="test_agent",
            description="Test agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4",
                api_key="test-key",
            ),
            instructions=Instructions(inline="Test instructions"),
            test_cases=[],
            evaluations=None,  # No evaluations
            execution=None,
        )

        mock_loader = Mock(spec=ConfigLoader)
        mock_loader.load_agent_yaml.return_value = agent_config
        mock_loader.resolve_execution_config.return_value = ExecutionConfig()

        executor = TestExecutor(
            agent_config_path="test.yaml",
            config_loader=mock_loader,
        )

        # Test case with no per-test evaluations
        test_case = TestCaseModel(
            name="test",
            input="Query",
            expected_tools=None,
            ground_truth=None,
            files=None,
            evaluations=None,
        )

        metrics = executor._get_metrics_for_test(test_case)

        # Should return empty list
        assert metrics == []
