"""Test executor for running agent test cases with evaluation metrics.

This module orchestrates test execution by coordinating:
- Configuration resolution (CLI > YAML > env > defaults)
- File processing via FileProcessor
- Agent invocation via AgentFactory
- Metric evaluation via evaluators
- Report generation via TestReport models

Test execution follows a sequential flow:
1. Load agent configuration from YAML file
2. Resolve execution configuration (CLI > YAML > env > defaults)
3. Initialize components (FileProcessor, AgentFactory, Evaluators)
4. Execute each test case:
   a. Process files (if any)
   b. Invoke agent with test input + file context
   c. Validate tool calls against expected tools
   d. Run evaluation metrics
   e. Determine pass/fail status
5. Generate TestReport with summary statistics
"""

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from semantic_kernel.contents import ChatHistory

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.lib.evaluators.azure_ai import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)
from holodeck.lib.evaluators.base import BaseEvaluator
from holodeck.lib.evaluators.nlp_metrics import (
    BLEUEvaluator,
    METEOREvaluator,
    ROUGEEvaluator,
)
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_exception
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import EvaluationMetric
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
)

logger = get_logger(__name__)


def validate_tool_calls(
    actual: list[str],
    expected: list[str] | None,
) -> bool | None:
    """Validate actual tool calls against expected tools.

    Tool call validation compares the set of tools actually called by the agent
    against the set of tools expected by the test case. Validation is exact set
    matching - order doesn't matter, but all expected tools must be called and
    no extra tools should be called.

    Args:
        actual: List of tool names actually called by agent
        expected: List of expected tool names from test case (None = skip validation)

    Returns:
        True if actual matches expected exactly (set equality)
        False if actual doesn't match expected
        None if expected is None (validation skipped)
    """
    if expected is None:
        return None

    actual_set = set(actual)
    expected_set = set(expected)
    matched = actual_set == expected_set

    logger.debug(
        f"Tool validation: expected={expected_set}, actual={actual_set}, "
        f"matched={matched}"
    )

    return matched


class TestExecutor:
    """Executor for running agent test cases.

    Orchestrates the complete test execution flow:
    1. Loads agent configuration from YAML file
    2. Resolves execution configuration (CLI > YAML > env > defaults)
    3. Initializes components (FileProcessor, AgentFactory, Evaluators)
    4. Executes test cases sequentially
    5. Generates test report with results and summary

    Attributes:
        agent_config_path: Path to agent configuration YAML file
        cli_config: Execution config from CLI flags (optional)
        agent_config: Loaded agent configuration
        config: Resolved execution configuration
        file_processor: FileProcessor instance
        agent_factory: AgentFactory instance
        evaluators: Dictionary of evaluator instances by metric name
        config_loader: ConfigLoader instance
        progress_callback: Optional callback function for progress reporting
    """

    def __init__(
        self,
        agent_config_path: str,
        execution_config: ExecutionConfig | None = None,
        file_processor: FileProcessor | None = None,
        agent_factory: AgentFactory | None = None,
        evaluators: dict[str, BaseEvaluator] | None = None,
        config_loader: ConfigLoader | None = None,
        progress_callback: Callable[[TestResult], None] | None = None,
    ) -> None:
        """Initialize test executor with optional dependency injection.

        Follows dependency injection pattern for testability. Dependencies can be:
        - Injected explicitly (for testing with mocks)
        - Created automatically using factory methods (for normal usage)

        Args:
            agent_config_path: Path to agent configuration file
            execution_config: Optional execution config from CLI flags
            file_processor: Optional FileProcessor instance (auto-created if None)
            agent_factory: Optional AgentFactory instance (auto-created if None)
            evaluators: Optional dict of evaluator instances (auto-created if None)
            config_loader: Optional ConfigLoader instance (auto-created if None)
            progress_callback: Optional callback function called after each test.
                              Called with TestResult instance. Use for progress display.
        """
        self.agent_config_path = agent_config_path
        self.cli_config = execution_config
        self.config_loader = config_loader or ConfigLoader()
        self.progress_callback = progress_callback

        logger.debug(f"Initializing TestExecutor for config: {agent_config_path}")

        # Load agent config
        self.agent_config = self._load_agent_config()

        # Resolve execution config (CLI > YAML > env > defaults)
        self.config = self._resolve_execution_config()

        # Use injected dependencies or create defaults
        logger.debug("Initializing FileProcessor component")
        self.file_processor = file_processor or self._create_file_processor()

        logger.debug("Initializing AgentFactory component")
        self.agent_factory = agent_factory or self._create_agent_factory()

        logger.debug("Initializing Evaluators component")
        self.evaluators = evaluators or self._create_evaluators()

        logger.info(
            f"TestExecutor initialized: {len(self.evaluators)} evaluators, "
            f"timeout={self.config.llm_timeout}s"
        )

    def _load_agent_config(self) -> Agent:
        """Load and validate agent configuration.

        Returns:
            Loaded Agent configuration

        Raises:
            FileNotFoundError: If agent config file not found
            ValidationError: If agent config is invalid
        """
        return self.config_loader.load_agent_yaml(self.agent_config_path)

    def _resolve_execution_config(self) -> ExecutionConfig:
        """Resolve execution config with priority hierarchy.

        Returns:
            ExecutionConfig with all fields resolved
        """
        return self.config_loader.resolve_execution_config(
            cli_config=self.cli_config,
            yaml_config=self.agent_config.execution,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

    def _create_file_processor(self) -> FileProcessor:
        """Create file processor with resolved config.

        Returns:
            Initialized FileProcessor instance
        """
        # Convert download timeout from seconds to milliseconds
        download_timeout_ms = (self.config.download_timeout or 30) * 1000

        return FileProcessor(
            cache_dir=self.config.cache_dir or ".holodeck/cache",
            download_timeout_ms=download_timeout_ms,
        )

    def _create_agent_factory(self) -> AgentFactory:
        """Create agent factory with resolved config.

        Returns:
            Initialized AgentFactory instance
        """
        return AgentFactory(
            agent_config=self.agent_config,
            timeout=self.config.llm_timeout or 60.0,
        )

    def _create_evaluators(self) -> dict[str, BaseEvaluator]:
        """Create evaluator instances from evaluation config.

        Returns:
            Dictionary mapping metric names to evaluator instances
        """
        evaluators: dict[str, BaseEvaluator] = {}

        if not self.agent_config.evaluations:
            return evaluators

        # Create evaluators for configured metrics
        for metric_config in self.agent_config.evaluations.metrics:
            metric_name = metric_config.metric

            # Azure AI evaluators
            default_model = (
                self.agent_config.evaluations.model
                if self.agent_config.evaluations
                else None
            )

            if metric_name == "groundedness":
                evaluators[metric_name] = GroundednessEvaluator(
                    model_config=metric_config.model or default_model  # type: ignore
                )
            elif metric_name == "relevance":
                evaluators[metric_name] = RelevanceEvaluator(
                    model_config=metric_config.model or default_model  # type: ignore
                )
            elif metric_name == "coherence":
                evaluators[metric_name] = CoherenceEvaluator(
                    model_config=metric_config.model or default_model  # type: ignore
                )
            elif metric_name == "fluency":
                evaluators[metric_name] = FluencyEvaluator(
                    model_config=metric_config.model or default_model  # type: ignore
                )

            # NLP metrics
            elif metric_name == "bleu":
                evaluators[metric_name] = BLEUEvaluator()
            elif metric_name == "rouge":
                evaluators[metric_name] = ROUGEEvaluator()
            elif metric_name == "meteor":
                evaluators[metric_name] = METEOREvaluator()

        return evaluators

    async def execute_tests(self) -> TestReport:
        """Execute all test cases and generate report.

        Returns:
            TestReport with all results and summary statistics
        """
        test_results: list[TestResult] = []

        # Execute each test case sequentially
        test_cases = self.agent_config.test_cases or []
        logger.info(f"Starting test execution: {len(test_cases)} test cases")

        for idx, test_case in enumerate(test_cases, 1):
            logger.debug(f"Executing test {idx}/{len(test_cases)}: {test_case.name}")
            result = await self._execute_single_test(test_case)
            test_results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(
                f"Test {idx}/{len(test_cases)} {status}: {test_case.name} "
                f"({result.execution_time_ms}ms)"
            )

            # Invoke progress callback if provided
            if self.progress_callback:
                self.progress_callback(result)

        # Generate report with summary
        logger.debug("Generating test report")
        return self._generate_report(test_results)

    async def _execute_single_test(
        self,
        test_case: TestCaseModel,
    ) -> TestResult:
        """Execute a single test case.

        Args:
            test_case: Test case configuration

        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        errors: list[str] = []
        processed_files: list[ProcessedFileInput] = []

        logger.debug(f"Starting test execution: {test_case.name}")

        # Step 1: Process files (if any)
        if test_case.files:
            logger.debug(f"Processing {len(test_case.files)} files for test")
            for file_input in test_case.files:
                try:
                    processed = self.file_processor.process_file(file_input)
                    processed_files.append(processed)

                    if processed.error:
                        logger.warning(
                            f"File processing error: {processed.error} "
                            f"[file={file_input.path or file_input.url}]"
                        )
                        errors.append(f"File error: {processed.error}")
                except Exception as e:
                    log_exception(
                        logger,
                        "File processing failed",
                        e,
                        context={"file": file_input.path or file_input.url},
                    )
                    errors.append(f"File processing error: {str(e)}")

        # Step 2: Prepare agent input
        logger.debug(f"Preparing agent input for test: {test_case.name}")
        agent_input = self._prepare_agent_input(test_case, processed_files)

        # Step 3: Invoke agent
        agent_response = None
        tool_calls: list[str] = []

        logger.debug(f"Invoking agent for test: {test_case.name}")
        try:
            invoke_start = time.time()
            result = await self.agent_factory.invoke(agent_input)
            invoke_elapsed = time.time() - invoke_start

            agent_response = self._extract_response_text(result.chat_history)
            tool_calls = self._extract_tool_names(result.tool_calls)

            logger.debug(
                f"Agent invocation completed in {invoke_elapsed:.2f}s, "
                f"tools_called={len(tool_calls)}"
            )
        except TimeoutError:
            logger.error(
                f"Agent invocation timeout after {self.config.llm_timeout}s "
                f"[test={test_case.name}]"
            )
            errors.append(f"Agent invocation timeout after {self.config.llm_timeout}s")
        except Exception as e:
            log_exception(
                logger, "Agent invocation failed", e, context={"test": test_case.name}
            )
            errors.append(f"Agent invocation error: {str(e)}")

        # Step 4: Validate tool calls
        if test_case.expected_tools:
            logger.debug(
                f"Validating tool calls: expected={test_case.expected_tools}, "
                f"actual={tool_calls}"
            )
        tools_matched = validate_tool_calls(tool_calls, test_case.expected_tools)

        # Step 5: Run evaluations
        logger.debug(f"Running evaluations for test: {test_case.name}")
        metric_results = await self._run_evaluations(
            test_case, agent_response, processed_files
        )
        logger.debug(
            f"Completed {len(metric_results)} evaluations for test: {test_case.name}"
        )

        # Step 6: Determine pass/fail
        passed = self._determine_test_passed(metric_results, tools_matched, errors)
        metrics_passed = sum(1 for m in metric_results if m.passed)
        logger.debug(
            f"Test result determined: passed={passed}, "
            f"metrics_passed={metrics_passed}/{len(metric_results)}, "
            f"tools_matched={tools_matched}, errors={len(errors)}"
        )

        # Step 7: Build TestResult
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Test execution completed: {test_case.name} ({elapsed_ms}ms)")

        return TestResult(
            test_name=test_case.name,
            test_input=test_case.input,
            processed_files=processed_files,
            agent_response=agent_response,
            tool_calls=tool_calls,
            expected_tools=test_case.expected_tools,
            tools_matched=tools_matched,
            metric_results=metric_results,
            ground_truth=test_case.ground_truth,
            passed=passed,
            execution_time_ms=elapsed_ms,
            errors=errors,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _prepare_agent_input(
        self,
        test_case: TestCaseModel,
        processed_files: list[ProcessedFileInput],
    ) -> str:
        """Prepare agent input combining test input and file content.

        Args:
            test_case: Test case configuration
            processed_files: List of processed files

        Returns:
            Combined input string for agent
        """
        parts: list[str] = []

        # Add file contents if any
        if processed_files:
            for processed in processed_files:
                if processed.markdown_content:
                    file_name = (
                        processed.original.path or processed.original.url or "file"
                    )
                    parts.append(f"File: {file_name}\n{processed.markdown_content}")

        # Add test input
        parts.append(test_case.input)

        return "\n\n".join(parts)

    def _extract_response_text(self, chat_history: ChatHistory) -> str:
        """Extract agent's last response from chat history.

        Args:
            chat_history: Semantic Kernel ChatHistory object

        Returns:
            Agent's response text or empty string if not found
        """
        if not chat_history or not chat_history.messages:
            return ""

        # Get last assistant message (most recent response)
        for message in reversed(chat_history.messages):
            if message.role == "assistant":
                content = message.content
                return str(content) if content else ""

        return ""

    def _extract_tool_names(self, tool_calls: list[dict[str, Any]]) -> list[str]:
        """Extract tool names from tool calls list.

        Tool calls are represented as list of dicts with 'name' and 'arguments' keys.

        Args:
            tool_calls: List of tool call dicts from agent

        Returns:
            List of tool names that were called
        """
        return [call.get("name", "") for call in tool_calls if "name" in call]

    async def _run_evaluations(
        self,
        test_case: TestCaseModel,
        agent_response: str | None,
        processed_files: list[ProcessedFileInput],
    ) -> list[MetricResult]:
        """Run evaluation metrics for test case.

        Evaluations are run with graceful degradation - if a metric fails,
        the error is recorded but execution continues with other metrics.

        Args:
            test_case: Test case configuration
            agent_response: Agent's response text (can be None if agent failed)
            processed_files: Processed file inputs

        Returns:
            List of metric results
        """
        metric_results: list[MetricResult] = []

        if not self.agent_config.evaluations or not agent_response:
            return metric_results

        # Get metrics for this test (per-test override or global)
        metrics = self._get_metrics_for_test(test_case)

        # Run each metric
        for metric_config in metrics:
            metric_name = metric_config.metric

            if metric_name not in self.evaluators:
                # Metric not configured, skip
                logger.debug(f"Skipping unconfigured metric: {metric_name}")
                continue

            try:
                logger.debug(f"Running metric evaluation: {metric_name}")
                evaluator = self.evaluators[metric_name]
                start_time = time.time()

                # Prepare evaluation inputs
                eval_kwargs = {
                    "response": agent_response,
                }

                # Add optional inputs based on metric type
                if test_case.input:
                    eval_kwargs["query"] = test_case.input

                if test_case.ground_truth:
                    eval_kwargs["ground_truth"] = test_case.ground_truth

                # Combine file contents as context
                file_content = self._combine_file_contents(processed_files)
                if file_content and metric_name in ("groundedness", "relevance"):
                    eval_kwargs["context"] = file_content

                # Run evaluation
                result = await evaluator.evaluate(**eval_kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Extract score and passed status
                # NLP metrics return results with metric name as key
                # (e.g., "bleu", "meteor"). Azure AI metrics use "score".
                score = result.get(metric_name, result.get("score", 0.0))
                threshold = metric_config.threshold
                passed = score >= threshold if threshold else True

                logger.debug(
                    f"Metric evaluation completed: {metric_name}, "
                    f"score={score:.3f}, threshold={threshold}, "
                    f"passed={passed}, duration={elapsed_ms}ms"
                )

                metric_results.append(
                    MetricResult(
                        metric_name=metric_name,
                        score=score,
                        threshold=threshold,
                        passed=passed,
                        scale="0-1",
                        error=None,
                        retry_count=0,
                        evaluation_time_ms=elapsed_ms,
                        model_used=(
                            metric_config.model.name if metric_config.model else None
                        ),
                    )
                )

            except Exception as e:
                # Record error but continue with other metrics
                log_exception(
                    logger,
                    f"Metric evaluation failed: {metric_name}",
                    e,
                    level=logging.WARNING,
                )
                metric_results.append(
                    MetricResult(
                        metric_name=metric_name,
                        score=0.0,
                        threshold=metric_config.threshold,
                        passed=False,
                        scale="0-1",
                        error=str(e),
                        retry_count=0,
                        evaluation_time_ms=0,
                        model_used=None,
                    )
                )

        return metric_results

    def _get_metrics_for_test(
        self,
        _test_case: TestCaseModel,
    ) -> list[EvaluationMetric]:
        """Resolve metrics for a test case (per-test override or global).

        Args:
            _test_case: Test case configuration (reserved for per-test overrides)

        Returns:
            List of metrics to evaluate

        Note:
            Per-test metric overrides are planned for US3.
            Currently uses global metrics.
        """
        # TODO: Implement per-test metric override (US3)
        # For now, use global metrics
        if self.agent_config.evaluations:
            return self.agent_config.evaluations.metrics
        return []

    def _combine_file_contents(self, processed_files: list[ProcessedFileInput]) -> str:
        """Combine contents from all processed files.

        Args:
            processed_files: List of processed files

        Returns:
            Combined markdown content
        """
        contents: list[str] = []
        for processed in processed_files:
            if processed.markdown_content:
                contents.append(processed.markdown_content)
        return "\n\n".join(contents)

    def _determine_test_passed(
        self,
        metric_results: list[MetricResult],
        tools_matched: bool | None,
        errors: list[str],
    ) -> bool:
        """Determine if test passed based on metrics, tool validation, and errors.

        Test passes if:
        - No execution errors occurred
        - All metrics passed (or no metrics configured)
        - Tool calls matched (or no tool validation configured)

        Args:
            metric_results: Results from metric evaluations
            tools_matched: Tool validation result (None = skipped)
            errors: List of execution errors

        Returns:
            True if test passed, False otherwise
        """
        # Test fails if there were execution errors
        if errors:
            return False

        # Test fails if tool validation was performed and failed
        if tools_matched is False:
            return False

        # Test fails if any metric failed
        return not (metric_results and any(not m.passed for m in metric_results))

    def _generate_report(self, results: list[TestResult]) -> TestReport:
        """Generate test report with summary statistics.

        Args:
            results: List of test results

        Returns:
            Complete test report with summary
        """
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) if total_tests > 0 else 0.0

        # Calculate total duration
        total_duration_ms = sum(r.execution_time_ms for r in results)

        # Collect evaluated metrics and calculate average scores
        all_metrics: set[str] = set()
        metric_scores: dict[str, list[float]] = {}

        for result in results:
            for metric in result.metric_results:
                all_metrics.add(metric.metric_name)
                if metric.score:
                    if metric.metric_name not in metric_scores:
                        metric_scores[metric.metric_name] = []
                    metric_scores[metric.metric_name].append(metric.score)

        # Calculate average scores
        average_scores: dict[str, float] = {}
        for metric_name in metric_scores:
            scores = metric_scores[metric_name]
            average_scores[metric_name] = sum(scores) / len(scores) if scores else 0.0

        # Create summary - metrics_evaluated is count per metric
        metrics_evaluated: dict[str, int] = {
            metric_name: len(metric_scores.get(metric_name, []))
            for metric_name in all_metrics
        }

        summary = ReportSummary(
            total_tests=total_tests,
            passed=passed_tests,
            failed=failed_tests,
            pass_rate=pass_rate,
            total_duration_ms=total_duration_ms,
            metrics_evaluated=metrics_evaluated,
            average_scores=average_scores,
        )

        # Get holodeck version from package
        try:
            from holodeck import __version__

            version = __version__
        except (ImportError, AttributeError):
            version = "0.1.0"

        # Create report
        return TestReport(
            agent_name=self.agent_config.name,
            agent_config_path=self.agent_config_path,
            results=results,
            summary=summary,
            timestamp=datetime.now(UTC).isoformat(),
            holodeck_version=version,
            environment={"execution_config": str(self.config.model_dump())},
        )
