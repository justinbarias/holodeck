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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from semantic_kernel.contents import ChatHistory

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError
from holodeck.lib.evaluators.azure_ai import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)
from holodeck.lib.evaluators.base import BaseEvaluator
from holodeck.lib.evaluators.deepeval import (
    AnswerRelevancyEvaluator,
    ContextualPrecisionEvaluator,
    ContextualRecallEvaluator,
    ContextualRelevancyEvaluator,
    FaithfulnessEvaluator,
    GEvalEvaluator,
)
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.nlp_metrics import (
    BLEUEvaluator,
    METEOREvaluator,
    ROUGEEvaluator,
)
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_exception
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.lib.test_runner.eval_kwargs_builder import (
    EvalKwargsBuilder,
    build_retrieval_context_from_tools,
)
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import (
    EvaluationMetric,
    GEvalMetric,
    RAGMetric,
    RAGMetricType,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.observability import TracingConfig
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
)

logger = get_logger(__name__)


class RAGEvaluatorConstructor(Protocol):
    """Protocol for RAG evaluator constructors with full type safety.

    Defines the common constructor signature for all RAG evaluators.
    The actual evaluators may have additional parameters with defaults
    (timeout, retry_config) but this Protocol captures what we use.
    """

    def __call__(
        self,
        *,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        observability_config: TracingConfig | None = None,
    ) -> BaseEvaluator:
        """Construct a RAG evaluator with the given configuration."""
        ...


# Mapping of RAG metric types to their evaluator classes
# Used to eliminate repetitive if/elif chains in _create_evaluators
RAG_EVALUATOR_MAP: dict[RAGMetricType, RAGEvaluatorConstructor] = {
    RAGMetricType.FAITHFULNESS: FaithfulnessEvaluator,
    RAGMetricType.CONTEXTUAL_RELEVANCY: ContextualRelevancyEvaluator,
    RAGMetricType.CONTEXTUAL_PRECISION: ContextualPrecisionEvaluator,
    RAGMetricType.CONTEXTUAL_RECALL: ContextualRecallEvaluator,
    RAGMetricType.ANSWER_RELEVANCY: AnswerRelevancyEvaluator,
}


def validate_tool_calls(
    actual: list[str],
    expected: list[str] | None,
) -> bool | None:
    """Validate actual tool calls against expected tools.

    Tool call validation checks that each expected tool name is found within
    at least one actual tool call. This uses substring matching - if any actual
    tool name contains the expected tool name, it's considered a match.

    Args:
        actual: List of tool names actually called by agent
        expected: List of expected tool names from test case (None = skip validation)

    Returns:
        True if all expected tools are found (substring match) in actual
        False if any expected tool is not found in any actual tool
        None if expected is None (validation skipped)

    Examples:
        - expected=["search"], actual=["vectorstore-search"] -> True
        - expected=["search", "fetch"], actual=["search_tool", "fetch_data"] -> True
        - expected=["search"], actual=["fetch"] -> False
    """
    if expected is None:
        return None

    def is_expected_found(expected_tool: str) -> bool:
        """Check if expected tool name is found in any actual tool call."""
        return any(expected_tool in actual_tool for actual_tool in actual)

    matched = all(is_expected_found(exp) for exp in expected)

    logger.debug(
        f"Tool validation: expected={expected}, actual={actual}, " f"matched={matched}"
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
        on_test_start: Callable[[TestCaseModel], None] | None = None,
        force_ingest: bool = False,
        agent_config: Agent | None = None,
        resolved_execution_config: ExecutionConfig | None = None,
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
            force_ingest: Force re-ingestion of vector store source files.
            agent_config: Optional pre-loaded Agent config (auto-loaded if None)
            resolved_execution_config: Optional pre-resolved execution config
                                       (auto-resolved if None)
        """
        self.agent_config_path = agent_config_path
        self.cli_config = execution_config
        self.config_loader = config_loader or ConfigLoader()
        self.progress_callback = progress_callback
        self.on_test_start = on_test_start
        self._force_ingest = force_ingest

        logger.debug(f"Initializing TestExecutor for config: {agent_config_path}")

        # Use injected agent config or load from file
        self.agent_config = agent_config or self._load_agent_config()

        # Use injected resolved config or resolve from hierarchy
        self.config = resolved_execution_config or self._resolve_execution_config()

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
        # Load project-level config (same directory as agent.yaml)
        agent_dir = str(Path(self.agent_config_path).parent)
        project_config = self.config_loader.load_project_config(agent_dir)
        project_execution = project_config.execution if project_config else None

        # Load user-level config (~/.holodeck/)
        user_config = self.config_loader.load_global_config()
        user_execution = user_config.execution if user_config else None

        return self.config_loader.resolve_execution_config(
            cli_config=self.cli_config,
            yaml_config=self.agent_config.execution,
            project_config=project_execution,
            user_config=user_execution,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

    def _create_file_processor(self) -> FileProcessor:
        """Create file processor with resolved config.

        Returns:
            Initialized FileProcessor instance
        """
        return FileProcessor.from_execution_config(self.config)

    def _create_agent_factory(self) -> AgentFactory:
        """Create agent factory with resolved config.

        Returns:
            Initialized AgentFactory instance
        """
        return AgentFactory(
            agent_config=self.agent_config,
            force_ingest=self._force_ingest,
            execution_config=self.config,
        )

    def _build_deepeval_config(
        self, llm_provider: LLMProvider | None
    ) -> DeepEvalModelConfig | None:
        """Convert LLMProvider to DeepEvalModelConfig.

        Args:
            llm_provider: HoloDeck LLM provider configuration

        Returns:
            DeepEvalModelConfig instance or None if no provider
        """
        if not llm_provider:
            return None

        # Build config with fields available in LLMProvider
        # For Azure OpenAI, use model name as deployment name if not specified
        deployment_name = None
        if llm_provider.provider == ProviderEnum.AZURE_OPENAI:
            deployment_name = llm_provider.name  # Use model name as deployment name

        return DeepEvalModelConfig(
            provider=llm_provider.provider,
            model_name=llm_provider.name,
            api_key=llm_provider.api_key,
            endpoint=llm_provider.endpoint,
            deployment_name=deployment_name,
            temperature=0.0,  # Deterministic for evaluation
        )

    def _create_evaluators(self) -> dict[str, BaseEvaluator]:
        """Create evaluator instances from evaluation config.

        Supports standard EvaluationMetric, GEvalMetric, and RAGMetric types.

        Returns:
            Dictionary mapping metric names to evaluator instances
        """
        evaluators: dict[str, BaseEvaluator] = {}

        if not self.agent_config.evaluations:
            return evaluators

        # Get default model for all metrics
        default_model = self.agent_config.evaluations.model

        # Get observability config for span instrumentation
        # Only pass it if observability is enabled and traces are enabled
        observability_config = None
        if (
            self.agent_config.observability
            and self.agent_config.observability.enabled
            and self.agent_config.observability.traces
            and self.agent_config.observability.traces.enabled
        ):
            observability_config = self.agent_config.observability.traces

        # Create evaluators for configured metrics
        for metric_config in self.agent_config.evaluations.metrics:
            # Handle GEval custom criteria metrics
            if isinstance(metric_config, GEvalMetric):
                llm_model = metric_config.model or default_model
                deepeval_config = self._build_deepeval_config(llm_model)

                # Use metric name as the evaluator key
                evaluators[metric_config.name] = GEvalEvaluator(
                    name=metric_config.name,
                    criteria=metric_config.criteria,
                    evaluation_params=metric_config.evaluation_params,
                    evaluation_steps=metric_config.evaluation_steps,
                    strict_mode=metric_config.strict_mode,
                    model_config=deepeval_config,
                    threshold=metric_config.threshold or 0.5,
                    observability_config=observability_config,
                )
                logger.debug(
                    f"Created GEvalEvaluator: name={metric_config.name}, "
                    f"criteria_len={len(metric_config.criteria)}"
                )
                continue

            # Handle RAG evaluation metrics
            if isinstance(metric_config, RAGMetric):
                llm_model = metric_config.model or default_model
                deepeval_config = self._build_deepeval_config(llm_model)

                # Map RAGMetricType to evaluator class and create instance
                metric_name = metric_config.metric_type.value
                evaluator_class = RAG_EVALUATOR_MAP.get(metric_config.metric_type)
                if evaluator_class:
                    evaluators[metric_name] = evaluator_class(
                        model_config=deepeval_config,
                        threshold=metric_config.threshold,
                        include_reason=metric_config.include_reason,
                        observability_config=observability_config,
                    )
                    logger.debug(
                        f"Created RAG evaluator: type={metric_name}, "
                        f"threshold={metric_config.threshold}"
                    )
                continue

            # Handle standard EvaluationMetric types
            metric_name = metric_config.metric

            # Get model config (per-metric or default)
            llm_model = metric_config.model or default_model

            # Convert LLMProvider to ModelConfig for Azure evaluators
            azure_model_config = None
            if llm_model:
                from holodeck.lib.evaluators.azure_ai import ModelConfig

                # Validate required Azure config - fail fast with clear error message
                if not llm_model.endpoint or not llm_model.api_key:
                    raise ConfigError(
                        f"evaluations.metrics.{metric_name}",
                        f"Azure AI metrics require 'endpoint' and 'api_key' in LLM "
                        f"config for metric '{metric_name}'. Please configure these "
                        f"in your agent.yaml or set via environment variables.",
                    )

                azure_model_config = ModelConfig(
                    azure_endpoint=llm_model.endpoint,
                    api_key=llm_model.api_key,
                    azure_deployment=llm_model.name,
                )

            if metric_name == "groundedness":
                if azure_model_config:
                    evaluators[metric_name] = GroundednessEvaluator(
                        model_config=azure_model_config
                    )
            elif metric_name == "relevance":
                if azure_model_config:
                    evaluators[metric_name] = RelevanceEvaluator(
                        model_config=azure_model_config
                    )
            elif metric_name == "coherence":
                if azure_model_config:
                    evaluators[metric_name] = CoherenceEvaluator(
                        model_config=azure_model_config
                    )
            elif metric_name == "fluency":
                if azure_model_config:
                    evaluators[metric_name] = FluencyEvaluator(
                        model_config=azure_model_config
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

            if self.on_test_start:
                self.on_test_start(test_case)

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

        # Step 3: Invoke agent with isolated thread run
        agent_response = None
        tool_calls: list[str] = []
        tool_results: list[dict[str, Any]] = []

        logger.debug(f"Invoking agent for test: {test_case.name}")
        try:
            invoke_start = time.time()
            # Create isolated thread run for this test case
            thread_run = await self.agent_factory.create_thread_run()
            result = await thread_run.invoke(agent_input)
            invoke_elapsed = time.time() - invoke_start

            agent_response = self._extract_response_text(result.chat_history)
            tool_calls = self._extract_tool_names(result.tool_calls)
            tool_results = result.tool_results

            logger.debug(
                f"Agent invocation completed in {invoke_elapsed:.2f}s, "
                f"tools_called={len(tool_calls)}, tool_results={len(tool_results)}"
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
            test_case, agent_response, processed_files, tool_results
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
            timestamp=datetime.now(timezone.utc).isoformat(),
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
        tool_results: list[dict[str, Any]] | None = None,
    ) -> list[MetricResult]:
        """Run evaluation metrics for test case.

        Evaluations are run with graceful degradation - if a metric fails,
        the error is recorded but execution continues with other metrics.

        For RAG metrics, retrieval_context is resolved with priority:
        1. Manual override from test_case.retrieval_context (if provided)
        2. Dynamic extraction from retrieval tool results

        Args:
            test_case: Test case configuration
            agent_response: Agent's response text (can be None if agent failed)
            processed_files: Processed file inputs
            tool_results: List of tool result dicts with 'name' and 'result' keys

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
            # Get metric name based on metric type
            if isinstance(metric_config, GEvalMetric):
                metric_name = metric_config.name
            elif isinstance(metric_config, RAGMetric):
                metric_name = metric_config.metric_type.value
            else:
                metric_name = metric_config.metric

            if metric_name not in self.evaluators:
                # Metric not configured, skip
                logger.debug(f"Skipping unconfigured metric: {metric_name}")
                continue

            try:
                logger.debug(f"Running metric evaluation: {metric_name}")
                evaluator = self.evaluators[metric_name]
                start_time = time.time()

                # Prepare evaluation inputs using EvalKwargsBuilder
                # This handles the parameter name differences between evaluator types:
                # - Azure AI / NLP: response, query, ground_truth, context
                # - DeepEval: actual_output, input, expected_output, retrieval_context
                file_content = self._combine_file_contents(processed_files)

                # Resolve retrieval_context: manual override > dynamic from tools
                retrieval_context = test_case.retrieval_context
                if not retrieval_context and tool_results:
                    retrieval_context = build_retrieval_context_from_tools(
                        tool_results, self._get_retrieval_tool_names()
                    )

                kwargs_builder = EvalKwargsBuilder(
                    agent_response=agent_response,
                    input_query=test_case.input,
                    ground_truth=test_case.ground_truth,
                    file_content=file_content,
                    retrieval_context=retrieval_context,
                )
                eval_kwargs = kwargs_builder.build_for(evaluator)

                # Run evaluation
                result = await evaluator.evaluate(**eval_kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Extract score and passed status
                # NLP metrics return results with metric name as key
                # (e.g., "bleu", "meteor"). Azure AI metrics use "score".
                score = result.get(metric_name, result.get("score", 0.0))
                threshold = metric_config.threshold
                passed = score >= threshold if threshold else True
                # Extract reasoning (DeepEval metrics return this, NLP metrics don't)
                reasoning = result.get("reasoning")

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
                            metric_config.model.name
                            if metric_config.model and metric_config.model.name
                            else None
                        ),
                        reasoning=reasoning,
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
                        reasoning=None,
                    )
                )

        return metric_results

    def _get_metrics_for_test(
        self,
        test_case: TestCaseModel,
    ) -> list[EvaluationMetric | GEvalMetric | RAGMetric]:
        """Resolve metrics for a test case (per-test override or global).

        Args:
            test_case: Test case configuration with optional per-test metrics

        Returns:
            List of metrics to evaluate (standard, GEval, or RAG)

        Logic:
            - If test_case.evaluations is provided and non-empty, use those
              metrics directly (per-test override)
            - Otherwise, use all global metrics from agent_config.evaluations
            - If no evaluations are configured, return empty list
        """
        # If test case has per-test metrics specified, use those directly
        if test_case.evaluations:
            return test_case.evaluations

        # Fall back to global metrics
        if self.agent_config.evaluations:
            return list(self.agent_config.evaluations.metrics)
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

    def _get_retrieval_tool_names(self) -> set[str]:
        """Get names of tools that contribute to retrieval_context for RAG metrics.

        Retrieval tools are:
        - All vectorstore tools (type='vectorstore')
        - MCP tools with is_retrieval=True

        Returns:
            Set of tool names that are retrieval tools
        """
        from holodeck.models.tool import MCPTool, VectorstoreTool

        retrieval_tools: set[str] = set()

        if not self.agent_config.tools:
            return retrieval_tools

        for tool in self.agent_config.tools:
            if isinstance(tool, VectorstoreTool):
                # Vectorstore tools include plugin prefix in name
                retrieval_tools.add(f"vectorstore-{tool.name}")
            elif isinstance(tool, MCPTool) and tool.is_retrieval:
                # MCP tools use their configured name
                retrieval_tools.add(tool.name)

        return retrieval_tools

    def _build_retrieval_context(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[str]:
        """Build retrieval_context from retrieval tool results for RAG evaluation.

        Only results from retrieval tools (vectorstore, MCP with is_retrieval=True)
        are included. Non-retrieval tool results are excluded.

        Args:
            tool_results: List of tool result dicts with 'name' and 'result' keys

        Returns:
            List of retrieval context strings from retrieval tools only

        Note:
            This method delegates to build_retrieval_context_from_tools for
            the actual extraction logic.
        """
        retrieval_tool_names = self._get_retrieval_tool_names()
        return (
            build_retrieval_context_from_tools(tool_results, retrieval_tool_names) or []
        )

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

    async def shutdown(self) -> None:
        """Shutdown executor and cleanup resources.

        Must be called from the same task context where the executor was used
        to properly cleanup MCP plugins and other async resources.
        """
        try:
            logger.debug("TestExecutor shutting down")
            # Shutdown the agent factory (cleans up MCP plugins, vectorstores)
            await self.agent_factory.shutdown()
            logger.debug("TestExecutor shutdown complete")
        except Exception as e:
            logger.error(f"Error during TestExecutor shutdown: {e}")

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
            timestamp=datetime.now(timezone.utc).isoformat(),
            holodeck_version=version,
            environment={"execution_config": str(self.config.model_dump())},
        )
