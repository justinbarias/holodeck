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

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.lib.backends.base import (
    AgentBackend,
    BackendSessionError,
)
from holodeck.lib.backends.selector import BackendSelector
from holodeck.lib.chat_history_utils import extract_tool_names
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
from holodeck.lib.evaluators.deterministic import (
    EqualityEvaluator,
    NumericEvaluator,
)
from holodeck.lib.evaluators.nlp_metrics import (
    BLEUEvaluator,
    METEOREvaluator,
    ROUGEEvaluator,
)
from holodeck.lib.file_processor import FileProcessor
from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_exception
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.lib.test_runner.code_grader import (
    build_grader_context,
    invoke_grader,
)
from holodeck.lib.test_runner.eval_kwargs_builder import (
    EvalKwargsBuilder,
    build_retrieval_context_from_tools,
)
from holodeck.lib.test_runner.tool_arg_matcher import (
    evaluate_expected_tools,
    tool_name_matches,
)
from holodeck.lib.test_runner.tool_invocation_builder import pair_tool_calls
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.evaluation import (
    CodeMetric,
    EvaluationMetric,
    GEvalMetric,
    RAGMetric,
    RAGMetricType,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.observability import TracingConfig
from holodeck.models.test_case import ExpectedTool, TestCaseModel, Turn
from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
    ToolInvocation,
    TurnResult,
)
from holodeck.models.token_usage import TokenUsage

logger = get_logger(__name__)


class TestCaseFatal(Exception):  # noqa: N818
    # Semantic name preferred over an `Error` suffix — this is a control-flow
    # signal (test-case-scoped "stop" from a grader), not a typical error.
    """Raised when a ``CodeMetric`` grader with ``fail_on_error=True`` raises.

    Breaks the turn loop for **this test case only**. Other test cases still
    run. Caught at the per-test-case level in ``_execute_single_test`` /
    ``_run_multi_turn``. Executor-local by design — NOT part of the public API
    and not exported from this module.
    """

    def __init__(
        self,
        *,
        test_case_name: str | None,
        turn_index: int,
        grader_name: str,
        underlying: Exception,
    ) -> None:
        self.test_case_name = test_case_name
        self.turn_index = turn_index
        self.grader_name = grader_name
        self.underlying = underlying
        super().__init__(
            f"TestCaseFatal({grader_name}) at turn {turn_index} "
            f"of {test_case_name!r}: {type(underlying).__name__}: {underlying}"
        )


def _metric_kind(
    metric_config: EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric,
) -> Literal["standard", "rag", "geval", "code"]:
    """Map an evaluation metric config to its runtime kind discriminator.

    The config-side discriminator (`type`) is already a `Literal` on each
    variant (`models/evaluation.py`), so this is a one-to-one passthrough.
    """
    return metric_config.type


def _rollup_tools_matched(per_turn: list[bool | None]) -> bool | None:
    """Test-case-level tools_matched rollup per contracts §4.

    - `None` if every turn is `None` (no assertions anywhere).
    - `False` if any turn is explicitly `False`.
    - `True` otherwise (all `True` / `None`, with at least one assertion).
    """
    if all(v is None for v in per_turn):
        return None
    return not any(v is False for v in per_turn)


def _finalize_multi_turn_result(
    *,
    test_name: str | None,
    turns: list[TurnResult],
    start_ts: str,
) -> TestResult:
    """Roll up per-turn results into the test-case-level TestResult.

    Implements the contract in contracts/turn-result-schema.md §4 and
    data-model.md §9.
    """
    if not turns:
        raise ValueError("_finalize_multi_turn_result requires at least one turn")

    test_input = "\n---\n".join(t.input for t in turns)
    agent_response = turns[-1].response

    tool_calls: list[str] = []
    tool_invocations: list[ToolInvocation] = []
    for t in turns:
        tool_calls.extend(t.tool_calls)
        tool_invocations.extend(t.tool_invocations)

    errors: list[str] = []
    for t in turns:
        for msg in t.errors:
            errors.append(f"[turn {t.turn_index}] {msg}")

    # Token usage: element-wise sum, preserving None when no turn reported.
    summed: TokenUsage | None = None
    for t in turns:
        if t.token_usage is None:
            continue
        summed = t.token_usage if summed is None else summed + t.token_usage

    passed = all(t.passed for t in turns)
    execution_time_ms = sum(t.execution_time_ms for t in turns)

    # Per-metric rollup: each metric_name appears once; score = mean of
    # turns that ran it; passed = all(turn_passed for that metric).
    metric_rollup: dict[str, list[MetricResult]] = {}
    for t in turns:
        for m in t.metric_results:
            metric_rollup.setdefault(m.metric_name, []).append(m)
    rolled_metrics: list[MetricResult] = []
    for name, records in metric_rollup.items():
        if not records:
            continue
        scores = [r.score for r in records]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        all_passed = all((r.passed is not False) for r in records)
        first = records[0]
        rolled_metrics.append(
            MetricResult(
                metric_name=name,
                kind=first.kind,
                score=mean_score,
                threshold=first.threshold,
                passed=all_passed,
                scale=first.scale,
                error=None,
                retry_count=0,
                evaluation_time_ms=sum((r.evaluation_time_ms or 0) for r in records),
                model_used=first.model_used,
                reasoning=None,
            )
        )

    return TestResult(
        test_name=test_name,
        test_input=test_input,
        processed_files=[],
        agent_response=agent_response,
        tool_calls=tool_calls,
        tool_invocations=tool_invocations,
        expected_tools=None,
        tools_matched=_rollup_tools_matched([t.tools_matched for t in turns]),
        metric_results=rolled_metrics,
        ground_truth=None,
        passed=passed,
        execution_time_ms=execution_time_ms,
        errors=errors,
        timestamp=start_ts,
        token_usage=summed,
        turns=list(turns),
    )


def _detect_backend_kind(backend: AgentBackend) -> Literal["sk", "claude"]:
    """Return the pairing strategy key for the given backend instance.

    Claude's tool-call records carry `call_id` (from `ToolUseBlock.id`) and
    must be paired by id. Everything else (SK over OpenAI / Azure / Ollama)
    uses positional pairing on parallel lists. Detection is by class name so
    the import can stay lazy and avoid a cycle via `BackendSelector`.
    """
    return "claude" if type(backend).__name__ == "ClaudeBackend" else "sk"


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
        return any(
            tool_name_matches(expected_tool, actual_tool) for actual_tool in actual
        )

    matched = all(is_expected_found(exp) for exp in expected)

    logger.debug(
        f"Tool validation: expected={expected}, actual={actual}, " f"matched={matched}"
    )

    return matched


def _format_args_brief(args: dict[str, Any]) -> str:
    """Short human-readable rendering of `args_asserted` for error lines."""
    parts: list[str] = []
    for key, val in args.items():
        if isinstance(val, dict) and "fuzzy" in val:
            parts.append(f"{key}≈{val['fuzzy']}")
        elif isinstance(val, dict) and "regex" in val:
            parts.append(f"{key}~{val['regex']}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def _serialize_expected_tools_for_turn_result(
    expected: list[str | ExpectedTool] | None,
) -> list[Any] | None:
    """Dump `expected_tools` to TurnResult wire shape (str | dict).

    Mirrors the serialization TestCaseModel uses on dump so dashboards and
    reporters see the same normalized structure.
    """
    if expected is None:
        return None
    out: list[Any] = []
    for entry in expected:
        if isinstance(entry, str):
            out.append(entry)
        else:
            obj: dict[str, Any] = {"name": entry.name}
            if entry.args is not None:
                args_dump: dict[str, Any] = {}
                from holodeck.models.test_case import (
                    FuzzyMatcher,
                    LiteralMatcher,
                    RegexMatcher,
                )

                for k, v in entry.args.items():
                    if isinstance(v, LiteralMatcher):
                        args_dump[k] = v.value
                    elif isinstance(v, FuzzyMatcher):
                        args_dump[k] = {"fuzzy": v.pattern}
                    elif isinstance(v, RegexMatcher):
                        args_dump[k] = {"regex": v.compiled.pattern}
                    else:
                        args_dump[k] = v
                obj["args"] = args_dump
            if entry.count != 1:
                obj["count"] = entry.count
            out.append(obj)
    return out


def _partition_expected_tools(
    expected: list[str | ExpectedTool] | None,
) -> tuple[list[str], list[ExpectedTool]]:
    """Split an `expected_tools` list into fast-path names vs object entries.

    Per tasks-us3.md T027a: bare strings plus any `ExpectedTool(args=None,
    count==1)` go to the fast-path name-only validator; any `ExpectedTool`
    with `args is not None` or `count > 1` is routed through the arg-matcher.
    """
    if expected is None:
        return [], []
    names: list[str] = []
    objects: list[ExpectedTool] = []
    for entry in expected:
        if isinstance(entry, str):
            names.append(entry)
        elif entry.args is None and entry.count == 1:
            names.append(entry.name)
        else:
            objects.append(entry)
    return names, objects


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
        backend: AgentBackend | None = None,
        allow_side_effects: bool = False,
    ) -> None:
        """Initialize test executor with optional dependency injection.

        Follows dependency injection pattern for testability. Dependencies can be:
        - Injected explicitly (for testing with mocks)
        - Created automatically using factory methods (for normal usage)

        When ``backend`` is provided, the executor uses the provider-agnostic
        ``AgentBackend.invoke_once()`` path and skips ``AgentFactory`` creation.
        When neither ``backend`` nor ``agent_factory`` is provided, the executor
        can auto-select a backend via ``BackendSelector`` at execution time.

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
            backend: Optional AgentBackend instance. When provided, the executor
                     uses invoke_once() instead of AgentFactory.
            allow_side_effects: Allow bash/file_system.write in test mode
                                (passed to BackendSelector when auto-selecting).
        """
        self.agent_config_path = agent_config_path
        self.cli_config = execution_config
        self.config_loader = config_loader or ConfigLoader()
        self.progress_callback = progress_callback
        self.on_test_start = on_test_start
        self._force_ingest = force_ingest
        self._backend: AgentBackend | None = backend
        self._allow_side_effects = allow_side_effects

        logger.debug(f"Initializing TestExecutor for config: {agent_config_path}")

        # Use injected agent config or load from file
        self.agent_config = agent_config or self._load_agent_config()

        # Use injected resolved config or resolve from hierarchy
        self.config = resolved_execution_config or self._resolve_execution_config()

        # Use injected dependencies or create defaults
        logger.debug("Initializing FileProcessor component")
        self.file_processor = file_processor or self._create_file_processor()

        # Resolve agent invocation path:
        # 1. Injected backend  → use it, skip AgentFactory
        # 2. Injected factory  → use it (legacy / tests)
        # 3. Neither injected  → defer to _ensure_backend_initialized()
        if self._backend is not None:
            logger.debug("Using injected AgentBackend — skipping AgentFactory creation")
            self.agent_factory: AgentFactory | None = agent_factory
        elif agent_factory is not None:
            logger.debug("Using injected AgentFactory")
            self.agent_factory = agent_factory
        else:
            logger.debug(
                "No backend or agent_factory injected "
                "— will auto-select via BackendSelector at execution time"
            )
            self.agent_factory = None

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
            api_key=(
                llm_provider.api_key.get_secret_value()
                if llm_provider.api_key is not None
                else None
            ),
            endpoint=llm_provider.endpoint,
            deployment_name=deployment_name,
            temperature=0.0,  # Deterministic for evaluation
        )

    def _create_evaluators(self) -> dict[str, BaseEvaluator]:
        """Create evaluator instances from evaluation config.

        Supports standard EvaluationMetric, GEvalMetric, and RAGMetric types.

        Scans all three configuration rungs — agent-global, per-test-case, and
        per-turn — so that metrics introduced at the narrower rungs (FR-023)
        also register an evaluator. Without this the per-turn / per-test
        override path silently skipped metrics not declared globally.

        Returns:
            Dictionary mapping metric names to evaluator instances
        """
        evaluators: dict[str, BaseEvaluator] = {}

        # Collect every metric declared at any rung (agent / test-case / turn).
        # Later duplicates are ignored — the first occurrence wins, matching
        # the three-level resolver's precedence (turn > test_case > agent).
        all_metrics: list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric] = []
        if self.agent_config.evaluations:
            all_metrics.extend(self.agent_config.evaluations.metrics)
        for tc in self.agent_config.test_cases or []:
            if tc.evaluations:
                all_metrics.extend(tc.evaluations)
            for turn in tc.turns or []:
                if turn.evaluations:
                    all_metrics.extend(turn.evaluations)

        if not all_metrics:
            return evaluators

        # Get default model for all metrics
        default_model = (
            self.agent_config.evaluations.model
            if self.agent_config.evaluations
            else None
        )

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

        # Create evaluators for every configured metric across all rungs.
        for metric_config in all_metrics:
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

            # Handle code-grader metrics. CodeMetric does NOT register a
            # BaseEvaluator — the grader callable is invoked via
            # ``invoke_grader`` from ``code_grader.py`` inside
            # ``_run_evaluations`` (see T043 dispatch block below). We
            # explicitly ``continue`` here so an unknown metric_name branch
            # doesn't silently swallow it.
            if isinstance(metric_config, CodeMetric):
                logger.debug(f"CodeMetric configured — grader={metric_config.grader}")
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
                    api_key=llm_model.api_key.get_secret_value(),
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

            # US4 — deterministic evaluators (zero-LLM). Registered under the
            # metric-name key so ``_run_evaluations`` at
            # ``result.get(metric_name, ...)`` picks up the score.
            elif metric_name == "equality":
                evaluators[metric_name] = EqualityEvaluator(
                    case_insensitive=metric_config.case_insensitive,
                    strip_whitespace=metric_config.strip_whitespace,
                    strip_punctuation=metric_config.strip_punctuation,
                )
            elif metric_name == "numeric":
                evaluators[metric_name] = NumericEvaluator(
                    absolute_tolerance=metric_config.absolute_tolerance,
                    relative_tolerance=metric_config.relative_tolerance,
                    accept_percent=metric_config.accept_percent,
                    accept_thousands_separators=(
                        metric_config.accept_thousands_separators
                    ),
                )

        return evaluators

    async def _ensure_backend_initialized(self) -> None:
        """Auto-select a backend via BackendSelector if none was injected.

        Only triggers when both ``_backend`` and ``agent_factory`` are None,
        i.e. normal CLI usage where neither dependency was explicitly injected.
        """
        if self._backend is not None or self.agent_factory is not None:
            return

        logger.debug(
            "No backend or agent_factory injected — auto-selecting via BackendSelector"
        )
        self._backend = await BackendSelector.select(
            self.agent_config,
            tool_instances=None,
            mode="test",
            allow_side_effects=self._allow_side_effects,
        )

    async def execute_tests(self) -> TestReport:
        """Execute all test cases and generate report.

        Per-test-case concurrency is controlled by `parallel_test_cases`
        (feature 032 FR-009a). Turns within a single multi-turn case stay
        strictly sequential. Progress callbacks and reporter emission are
        serialised behind an asyncio.Lock so per-test-case output blocks
        don't interleave mid-record.
        """
        await self._ensure_backend_initialized()

        test_cases = self.agent_config.test_cases or []
        parallel = self.config.parallel_test_cases or 1
        logger.info(
            f"Starting test execution: {len(test_cases)} test cases "
            f"(parallel_test_cases={parallel})"
        )

        # Preserve input order in the output regardless of completion order.
        results: list[TestResult | None] = [None] * len(test_cases)
        semaphore = asyncio.Semaphore(parallel)
        emit_lock = asyncio.Lock()

        async def run_one(idx: int, test_case: TestCaseModel) -> None:
            async with semaphore:
                logger.debug(
                    f"Executing test {idx + 1}/{len(test_cases)}: {test_case.name}"
                )
                if self.on_test_start:
                    self.on_test_start(test_case)
                result = await self._execute_single_test(test_case)
                results[idx] = result
                async with emit_lock:
                    status = "PASS" if result.passed else "FAIL"
                    logger.info(
                        f"Test {idx + 1}/{len(test_cases)} {status}: "
                        f"{test_case.name} ({result.execution_time_ms}ms)"
                    )
                    if self.progress_callback:
                        self.progress_callback(result)

        if parallel <= 1:
            for idx, test_case in enumerate(test_cases):
                await run_one(idx, test_case)
        else:
            await asyncio.gather(*[run_one(i, tc) for i, tc in enumerate(test_cases)])

        # Collect non-None in order (all slots should be filled).
        ordered: list[TestResult] = [r for r in results if r is not None]

        logger.debug("Generating test report")
        return self._generate_report(ordered)

    async def _execute_single_test(
        self,
        test_case: TestCaseModel,
    ) -> TestResult:
        """Execute a single test case.

        Routes multi-turn test cases (`test_case.turns is not None`) through
        ``_run_multi_turn``; legacy single-turn cases use the existing path.
        """
        if test_case.turns is not None:
            return await self._run_multi_turn(test_case)

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
        raw_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        token_usage: TokenUsage | None = None
        backend_kind: Literal["sk", "claude"] = "sk"

        logger.debug(f"Invoking agent for test: {test_case.name}")
        try:
            invoke_start = time.time()

            if self._backend is not None:
                # New path: provider-agnostic AgentBackend
                exec_result = await self._backend.invoke_once(agent_input)
                if exec_result.is_error:
                    errors.append(f"Agent error: {exec_result.error_reason}")
                agent_response = exec_result.response
                raw_tool_calls = list(exec_result.tool_calls)
                tool_calls = extract_tool_names(exec_result.tool_calls)
                tool_results = exec_result.tool_results
                raw_usage_new: Any = exec_result.token_usage
                token_usage = (
                    raw_usage_new if isinstance(raw_usage_new, TokenUsage) else None
                )
                backend_kind = _detect_backend_kind(self._backend)
            elif self.agent_factory is not None:
                # Legacy path: AgentFactory / AgentThreadRun
                thread_run = await self.agent_factory.create_thread_run()
                legacy_result = await thread_run.invoke(agent_input)
                agent_response = legacy_result.response
                raw_tool_calls = list(legacy_result.tool_calls)
                tool_calls = extract_tool_names(legacy_result.tool_calls)
                tool_results = legacy_result.tool_results
                raw_usage_legacy: Any = legacy_result.token_usage
                token_usage = (
                    raw_usage_legacy
                    if isinstance(raw_usage_legacy, TokenUsage)
                    else None
                )
                backend_kind = "sk"
            else:
                errors.append("No backend or agent_factory available")

            invoke_elapsed = time.time() - invoke_start
            logger.debug(
                f"Agent invocation completed in {invoke_elapsed:.2f}s, "
                f"tools_called={len(tool_calls)}, tool_results={len(tool_results)}"
            )
        except BackendSessionError as e:
            log_exception(
                logger,
                "Backend session error",
                e,
                context={"test": test_case.name},
            )
            errors.append(f"Agent invocation error: {str(e)}")
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

        # Step 4: Validate tool calls. Widened to accept mixed str | ExpectedTool
        # lists: name-only + promoted-legacy go through the fast path; richer
        # ExpectedTool forms run through the arg matcher (US3 T027a).
        if test_case.expected_tools:
            logger.debug(
                f"Validating tool calls: expected={test_case.expected_tools}, "
                f"actual={tool_calls}"
            )
        fast_names_st, object_expected_st = _partition_expected_tools(
            test_case.expected_tools
        )
        try:
            paired_invocations_for_args = pair_tool_calls(
                raw_tool_calls, tool_results, backend_kind=backend_kind
            )
        except TypeError:
            paired_invocations_for_args = []
        fast_matched_st = validate_tool_calls(
            tool_calls, fast_names_st if fast_names_st else None
        )
        arg_matched_st, arg_details_st = evaluate_expected_tools(
            list(object_expected_st), paired_invocations_for_args
        )
        if test_case.expected_tools is None:
            tools_matched = None
        else:
            name_ok = fast_matched_st is not False
            tools_matched = bool(name_ok and arg_matched_st)

        for detail in arg_details_st or []:
            if detail["matched_call_index"] == -1:
                args_desc = _format_args_brief(detail["args_asserted"])
                tool_name = detail["expected_tool"]
                reason = detail["unmatched_reason"] or "no matching call"
                errors.append(f"expected {tool_name}({args_desc}): {reason}")

        # Step 5: Run evaluations
        logger.debug(f"Running evaluations for test: {test_case.name}")
        metric_results = await self._run_evaluations(
            metrics=self._get_metrics_for_test(test_case),
            agent_response=agent_response,
            input_query=test_case.input,
            ground_truth=test_case.ground_truth,
            retrieval_context=test_case.retrieval_context,
            processed_files=processed_files,
            tool_results=tool_results,
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

        try:
            tool_invocations = pair_tool_calls(
                raw_tool_calls, tool_results, backend_kind=backend_kind
            )
        except TypeError:
            # Defensive: mock-based tests may pass non-list sentinels through;
            # the on-disk contract still requires a list, so fall back to [].
            tool_invocations = []

        return TestResult(
            test_name=test_case.name,
            test_input=test_case.input or "",
            processed_files=processed_files,
            agent_response=agent_response,
            tool_calls=tool_calls,
            tool_invocations=tool_invocations,
            expected_tools=_serialize_expected_tools_for_turn_result(
                test_case.expected_tools
            ),
            tools_matched=tools_matched,
            metric_results=metric_results,
            ground_truth=test_case.ground_truth,
            passed=passed,
            execution_time_ms=elapsed_ms,
            errors=errors,
            timestamp=datetime.now(timezone.utc).isoformat(),
            token_usage=token_usage,
        )

    async def _run_multi_turn(self, test_case: TestCaseModel) -> TestResult:
        """Execute a multi-turn test case through a single AgentSession.

        Drives turns strictly sequentially (turn N+1 starts only after turn
        N resolves), closes the session in a ``finally`` block, maps
        ``asyncio.TimeoutError`` / backend errors to per-turn error entries,
        and marks remaining turns ``skipped=True`` after two consecutive
        ``BackendSessionError`` hits (research.md §2 heuristic).
        """
        if test_case.turns is None:
            raise ValueError("_run_multi_turn requires test_case.turns to be set")
        start_ts = datetime.now(timezone.utc).isoformat()
        start_perf = time.time()

        if self._backend is None:
            # Legacy / factory-only path doesn't support multi-turn — return an
            # explicit failure TurnResult so the roll-up still produces a
            # TestResult.
            skipped = [
                TurnResult(
                    turn_index=i,
                    input=t.input,
                    response=None,
                    ground_truth=t.ground_truth,
                    expected_tools=None,
                    tool_calls=[],
                    tool_invocations=[],
                    tools_matched=None,
                    arg_match_details=None,
                    metric_results=[],
                    passed=False,
                    execution_time_ms=0,
                    token_usage=None,
                    errors=[
                        "multi-turn dispatch requires an AgentBackend "
                        "(no backend injected)"
                    ],
                    skipped=True,
                    grader_details=None,
                )
                for i, t in enumerate(test_case.turns)
            ]
            result = _finalize_multi_turn_result(
                test_name=test_case.name, turns=skipped, start_ts=start_ts
            )
            return result

        backend_kind = _detect_backend_kind(self._backend)
        llm_timeout = self.config.llm_timeout or 60

        session = None
        turn_results: list[TurnResult] = []
        consecutive_session_errors = 0
        session_unrecoverable = False
        # Accumulates retrieval-tool chunks across turns so RAG metrics still
        # have grounding in later turns that don't re-retrieve.
        session_retrieval_context: list[str] = []

        try:
            session = await self._backend.create_session()

            for idx, turn in enumerate(test_case.turns):
                if session_unrecoverable:
                    turn_results.append(
                        TurnResult(
                            turn_index=idx,
                            input=turn.input,
                            response=None,
                            ground_truth=turn.ground_truth,
                            expected_tools=None,
                            tool_calls=[],
                            tool_invocations=[],
                            tools_matched=None,
                            arg_match_details=None,
                            metric_results=[],
                            passed=False,
                            execution_time_ms=0,
                            token_usage=None,
                            errors=[
                                "session became unrecoverable after two consecutive "
                                "BackendSessionError hits"
                            ],
                            skipped=True,
                            grader_details=None,
                        )
                    )
                    continue

                try:
                    turn_result = await self._run_single_turn(
                        test_case=test_case,
                        turn=turn,
                        turn_index=idx,
                        session=session,
                        llm_timeout=llm_timeout,
                        backend_kind=backend_kind,
                        session_retrieval_context=session_retrieval_context,
                    )
                except TestCaseFatal as fatal:
                    # fail_on_error=True grader raised — record a final
                    # failing turn for the current index then stop the turn
                    # loop. Remaining turns are skipped; other test cases
                    # continue unaffected.
                    logger.warning(
                        "TestCaseFatal: %s",
                        fatal,
                    )
                    turn_results.append(
                        TurnResult(
                            turn_index=idx,
                            input=turn.input,
                            response=None,
                            ground_truth=turn.ground_truth,
                            expected_tools=None,
                            tool_calls=[],
                            tool_invocations=[],
                            tools_matched=None,
                            arg_match_details=None,
                            metric_results=[],
                            passed=False,
                            execution_time_ms=0,
                            token_usage=None,
                            errors=[str(fatal)],
                            skipped=False,
                            grader_details=None,
                        )
                    )
                    session_unrecoverable = True
                    continue
                turn_results.append(turn_result)

                if turn_result.errors and any(
                    "BackendSessionError" in e for e in turn_result.errors
                ):
                    consecutive_session_errors += 1
                    if consecutive_session_errors >= 2:
                        session_unrecoverable = True
                else:
                    consecutive_session_errors = 0
        finally:
            if session is not None:
                try:
                    await session.close()
                except Exception as e:  # noqa: BLE001
                    log_exception(
                        logger,
                        "Error closing multi-turn session",
                        e,
                        level=logging.WARNING,
                    )

        # Re-issue the start timestamp in the rollup so the TestResult
        # records the moment the case *started*, matching the contract.
        result = _finalize_multi_turn_result(
            test_name=test_case.name, turns=turn_results, start_ts=start_ts
        )
        # Preserve the measured wall-clock: _finalize sums per-turn
        # execution_time_ms; use that as-is unless all turns were zero (empty
        # session path) — then fall back to the outer perf counter.
        if result.execution_time_ms == 0:
            result = result.model_copy(
                update={"execution_time_ms": int((time.time() - start_perf) * 1000)}
            )
        return result

    async def _run_single_turn(
        self,
        *,
        test_case: TestCaseModel,
        turn: Turn,
        turn_index: int,
        session: Any,
        llm_timeout: int,
        backend_kind: Literal["sk", "claude"],
        session_retrieval_context: list[str] | None = None,
    ) -> TurnResult:
        """Execute one turn against an open session and return its TurnResult.

        ``session_retrieval_context``, when provided, accumulates retrieval-tool
        output across turns of the same session. A later turn that answers
        from earlier-retrieved content still has grounding for RAG metrics
        (Faithfulness, ContextualRelevancy, ...). Extended in place with any
        new retrieval output produced by this turn.
        """
        # Per-turn file processing — residue must NOT leak into the next turn.
        processed_files: list[ProcessedFileInput] = []
        errors: list[str] = []

        if turn.files:
            for file_input in turn.files:
                try:
                    processed = self.file_processor.process_file(file_input)
                    processed_files.append(processed)
                    if processed.error:
                        errors.append(f"File error: {processed.error}")
                except Exception as e:  # noqa: BLE001
                    log_exception(
                        logger,
                        "Per-turn file processing failed",
                        e,
                        context={"file": file_input.path or file_input.url},
                    )
                    errors.append(f"File processing error: {str(e)}")

        message = self._compose_agent_input(turn.input, processed_files)

        response: str | None = None
        raw_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        tool_names: list[str] = []
        token_usage: TokenUsage | None = None

        start = time.perf_counter()
        try:
            exec_result = await asyncio.wait_for(
                session.send(message), timeout=llm_timeout
            )
            if exec_result.is_error:
                errors.append(f"Agent error: {exec_result.error_reason}")
            response = exec_result.response
            raw_tool_calls = list(exec_result.tool_calls)
            tool_results = list(exec_result.tool_results)
            tool_names = extract_tool_names(exec_result.tool_calls)
            raw_usage: Any = exec_result.token_usage
            token_usage = raw_usage if isinstance(raw_usage, TokenUsage) else None
        except asyncio.TimeoutError:
            errors.append("timeout")
        except BackendSessionError as e:
            errors.append(f"BackendSessionError: {e}")
        except Exception as e:  # noqa: BLE001
            log_exception(
                logger,
                "Per-turn agent invocation failed",
                e,
                context={"turn_index": turn_index},
            )
            errors.append(f"{type(e).__name__}: {e}")
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        try:
            tool_invocations = pair_tool_calls(
                raw_tool_calls, tool_results, backend_kind=backend_kind
            )
        except TypeError:
            tool_invocations = []

        # Per-turn tool-name assertion (US2 T020–T021, US3 T027a–T030).
        # Scoped to this turn's calls only — cross-turn credit is disallowed
        # (FR-011). Fast-path names + object-form matchers evaluated together.
        fast_names, object_expected = _partition_expected_tools(turn.expected_tools)
        fast_matched = validate_tool_calls(
            tool_names, fast_names if fast_names else None
        )
        arg_matched, arg_details = evaluate_expected_tools(
            object_expected,
            tool_invocations,
        )
        if turn.expected_tools is None:
            tools_matched: bool | None = None
        else:
            name_ok = fast_matched is not False
            tools_matched = bool(name_ok and arg_matched)
        arg_match_details_payload = arg_details if arg_details else None

        if fast_matched is False and fast_names:
            missing = [
                exp
                for exp in fast_names
                if not any(tool_name_matches(exp, actual) for actual in tool_names)
            ]
            if missing:
                missing_str = ", ".join(missing)
                errors.append(
                    f"expected tool(s) not called in this turn: {missing_str}"
                )

        # Per-assertion summary lines for the reporter (T030).
        for detail in arg_details or []:
            if detail["matched_call_index"] == -1:
                args_desc = _format_args_brief(detail["args_asserted"])
                tool_name = detail["expected_tool"]
                reason = detail["unmatched_reason"] or "no matching call"
                errors.append(f"expected {tool_name}({args_desc}): {reason}")

        # Extend session-scoped retrieval context with this turn's retrieval
        # tool output BEFORE evaluating. Later turns that respond from
        # earlier-retrieved content still have grounding for RAG metrics.
        if session_retrieval_context is not None and tool_results:
            new_chunks = build_retrieval_context_from_tools(
                tool_results, self._get_retrieval_tool_names()
            )
            if new_chunks:
                session_retrieval_context.extend(new_chunks)

        # Per-turn evaluations (US2 T009c). Skip if the turn errored before
        # producing a response — metric_results stays empty and the 4-conjunct
        # below catches the failure via `errors`.
        metric_results: list[MetricResult] = []
        grader_details_sink: dict[str, Any] = {}
        if response is not None:
            try:
                # Build turn_config for code graders — raw per-turn YAML-like
                # dict surfaced via ``Turn.model_dump`` (so grader-specific
                # keys like ``turn_program`` pass through untouched).
                turn_config_dict = turn.model_dump(mode="python")
                # Fall back to session-accumulated retrieval context when
                # the turn itself didn't invoke a retrieval tool (e.g.,
                # follow-up answers from prior chunks in the conversation).
                effective_turn_retrieval = turn.retrieval_context
                if not effective_turn_retrieval and session_retrieval_context:
                    effective_turn_retrieval = list(session_retrieval_context)
                metric_results = await self._run_evaluations(
                    metrics=self._resolve_turn_metrics(test_case, turn),
                    agent_response=response,
                    input_query=turn.input,
                    ground_truth=turn.ground_truth,
                    retrieval_context=effective_turn_retrieval,
                    processed_files=processed_files,
                    tool_results=tool_results,
                    skip_metrics_without_ground_truth=True,
                    tool_invocations=list(tool_invocations),
                    turn_index=turn_index,
                    test_case_name=test_case.name,
                    turn_config=turn_config_dict,
                    grader_details_sink=grader_details_sink,
                )
            except TestCaseFatal:
                # Re-raise so ``_run_multi_turn`` breaks the turn loop.
                raise
            except Exception as e:  # noqa: BLE001
                log_exception(
                    logger,
                    "Per-turn evaluation failed",
                    e,
                    context={"turn_index": turn_index},
                )
                errors.append(f"evaluation error: {type(e).__name__}: {e}")

        # Four-conjunct pass rule per contracts/turn-result-schema.md §3.
        passed = (
            errors == []
            and (tools_matched is None or tools_matched is True)
            and all((m.passed is not False) for m in metric_results)
        )

        return TurnResult(
            turn_index=turn_index,
            input=turn.input,
            response=response,
            ground_truth=turn.ground_truth,
            expected_tools=_serialize_expected_tools_for_turn_result(
                turn.expected_tools
            ),
            tool_calls=tool_names,
            tool_invocations=tool_invocations,
            tools_matched=tools_matched,
            arg_match_details=arg_match_details_payload,
            metric_results=metric_results,
            passed=passed,
            execution_time_ms=elapsed_ms,
            token_usage=token_usage,
            errors=errors,
            skipped=False,
            grader_details=grader_details_sink if grader_details_sink else None,
        )

    def _prepare_agent_input(
        self,
        test_case: TestCaseModel,
        processed_files: list[ProcessedFileInput],
    ) -> str:
        """Prepare agent input combining test input and file content."""
        input_text = test_case.input or ""
        return self._compose_agent_input(input_text, processed_files)

    def _compose_agent_input(
        self,
        input_text: str,
        processed_files: list[ProcessedFileInput],
    ) -> str:
        """Join processed file contents with a free-form input string."""
        parts: list[str] = []
        if processed_files:
            for processed in processed_files:
                if processed.markdown_content:
                    file_name = (
                        processed.original.path or processed.original.url or "file"
                    )
                    parts.append(f"File: {file_name}\n{processed.markdown_content}")
        parts.append(input_text)
        return "\n\n".join(parts)

    async def _run_evaluations(
        self,
        *,
        metrics: list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric],
        agent_response: str | None,
        input_query: str | None,
        ground_truth: str | None,
        retrieval_context: list[str] | None,
        processed_files: list[ProcessedFileInput],
        tool_results: list[dict[str, Any]] | None = None,
        skip_metrics_without_ground_truth: bool = False,
        # US4 — grader-context fields used only by ``CodeMetric`` dispatch.
        # Pulled into optional kwargs so legacy single-turn callers don't
        # have to thread them through.
        tool_invocations: list[Any] | None = None,
        turn_index: int = 0,
        test_case_name: str | None = None,
        turn_config: dict[str, Any] | None = None,
        grader_details_sink: dict[str, Any] | None = None,
    ) -> list[MetricResult]:
        """Run evaluation metrics against a single agent exchange.

        Evaluations are run with graceful degradation — if a metric fails,
        the error is recorded but execution continues with other metrics.

        For RAG metrics, the caller resolves `retrieval_context`; if it is
        `None`, this method falls back to extracting it from retrieval tool
        results.

        Text-comparison metrics (BLEU/ROUGE/METEOR/GEval/Azure AI) are
        skipped silently when `ground_truth` is `None` — the per-turn rollup
        (data-model.md §9) relies on this to distinguish "turn didn't run
        this metric" from "turn ran it and failed".

        Args:
            metrics: Pre-resolved metrics to evaluate (turn > test_case > agent).
            agent_response: Agent's response text (None if the agent failed).
            input_query: The turn / test-case input text.
            ground_truth: Expected output for this turn; `None` skips text metrics.
            retrieval_context: RAG context override; falls back to tool results.
            processed_files: Processed file inputs for this turn.
            tool_results: Per-call tool results (for RAG fallback extraction).

        Returns:
            List of metric results (one per metric that actually ran).
        """
        metric_results: list[MetricResult] = []

        if not agent_response:
            return metric_results

        # Run each metric
        for metric_config in metrics:
            # Get metric name based on metric type
            if isinstance(metric_config, GEvalMetric):
                metric_name = metric_config.name
            elif isinstance(metric_config, RAGMetric):
                metric_name = metric_config.metric_type.value
            elif isinstance(metric_config, CodeMetric):
                # US4 T043 — dispatch user-supplied grader via ``invoke_grader``.
                # ``CodeMetric`` graders do NOT use ``self.evaluators`` — the
                # callable is resolved at config load time and cached on the
                # model (see ``CodeMetric._resolve_grader``).
                if not metric_config.enabled:
                    continue
                grader_name = metric_config.display_name
                ctx = build_grader_context(
                    turn_input=input_query or "",
                    agent_response=agent_response,
                    ground_truth=ground_truth,
                    tool_invocations=tool_invocations or [],
                    retrieval_context=retrieval_context,
                    turn_index=turn_index,
                    test_case_name=test_case_name,
                    turn_config=turn_config or {},
                )
                mr, details, captured = invoke_grader(
                    metric_config.resolved_callable,
                    ctx,
                    metric_name=grader_name,
                    threshold=metric_config.threshold,
                )
                metric_results.append(mr)
                if details is not None and grader_details_sink is not None:
                    grader_details_sink[grader_name] = details
                # Escalate only when the grader raised AND fail_on_error=True.
                if captured is not None and metric_config.fail_on_error:
                    raise TestCaseFatal(
                        test_case_name=test_case_name,
                        turn_index=turn_index,
                        grader_name=grader_name,
                        underlying=captured,
                    )
                continue
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
                effective_retrieval_context = retrieval_context
                if not effective_retrieval_context and tool_results:
                    effective_retrieval_context = build_retrieval_context_from_tools(
                        tool_results, self._get_retrieval_tool_names()
                    )

                # Skip metrics that require ground_truth when the turn has
                # none — the per-turn rollup treats these as "metric not run"
                # for this turn (data-model.md §9, spec A6). Gated off for
                # legacy single-turn callers so they keep the prior semantics.
                from holodeck.lib.evaluators.param_spec import EvalParam

                spec = evaluator.get_param_spec()
                if skip_metrics_without_ground_truth and ground_truth is None:
                    needs_ground_truth = (
                        EvalParam.GROUND_TRUTH in spec.required
                        or EvalParam.EXPECTED_OUTPUT in spec.required
                    )
                    if needs_ground_truth:
                        logger.debug(
                            f"Skipping metric {metric_name}: requires ground_truth"
                        )
                        continue

                # Skip RAG metrics when retrieval_context is required but
                # nothing was retrieved in this turn or earlier in the session.
                # Avoids a hard DeepEval failure ("'retrieval_context' cannot
                # be None") on turns that legitimately answer without a
                # retrieval call.
                needs_retrieval = (
                    spec.uses_retrieval_context
                    or EvalParam.RETRIEVAL_CONTEXT in spec.required
                )
                if needs_retrieval and not effective_retrieval_context:
                    logger.debug(
                        f"Skipping metric {metric_name}: "
                        "requires retrieval_context but none available"
                    )
                    continue

                kwargs_builder = EvalKwargsBuilder(
                    agent_response=agent_response,
                    input_query=input_query,
                    ground_truth=ground_truth,
                    file_content=file_content,
                    retrieval_context=effective_retrieval_context,
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
                        kind=_metric_kind(metric_config),
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
                        kind=_metric_kind(metric_config),
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

    def _resolve_turn_metrics(
        self,
        test_case: TestCaseModel,
        turn: Turn,
    ) -> list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric]:
        """Resolve per-turn effective metrics with precedence turn > test_case > agent.

        Extends the two-level `_get_metrics_for_test` to add a turn-level rung
        (FR-023). Returns the most-specific non-empty list.
        """
        if turn.evaluations:
            return list(turn.evaluations)
        if test_case.evaluations:
            return list(test_case.evaluations)
        if self.agent_config.evaluations:
            return list(self.agent_config.evaluations.metrics)
        return []

    def _get_metrics_for_test(
        self,
        test_case: TestCaseModel,
    ) -> list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric]:
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
        - All hierarchical document tools (type='hierarchical_document')
        - MCP tools with is_retrieval=True

        Returns:
            Set of tool names that are retrieval tools
        """
        from holodeck.models.tool import (
            HierarchicalDocumentToolConfig,
            MCPTool,
            VectorstoreTool,
        )

        retrieval_tools: set[str] = set()

        if not self.agent_config.tools:
            return retrieval_tools

        for tool in self.agent_config.tools:
            if isinstance(tool, VectorstoreTool):
                # Semantic Kernel legacy naming
                retrieval_tools.add(f"vectorstore-{tool.name}")
                # Claude Agent SDK MCP naming
                retrieval_tools.add(f"mcp__holodeck_tools__{tool.name}_search")
            elif isinstance(tool, HierarchicalDocumentToolConfig):
                # Semantic Kernel legacy naming
                retrieval_tools.add(f"hierarchical_document-{tool.name}")
                # Claude Agent SDK MCP naming
                retrieval_tools.add(f"mcp__holodeck_tools__{tool.name}_search")
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
            if self._backend is not None:
                await self._backend.teardown()
            elif self.agent_factory is not None:
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
        pass_rate = (passed_tests / total_tests * 100.0) if total_tests > 0 else 0.0

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
