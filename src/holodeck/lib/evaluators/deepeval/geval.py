"""G-Eval evaluator for custom criteria evaluation.

This module provides the GEvalEvaluator class that wraps DeepEval's GEval metric
for evaluating LLM outputs against user-defined natural language criteria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.models.observability import TracingConfig

logger = get_logger(__name__)

# Valid evaluation parameter names
VALID_EVALUATION_PARAMS = frozenset(
    ["input", "actual_output", "expected_output", "context", "retrieval_context"]
)

# Mapping from string parameter names to LLMTestCaseParams enum
PARAM_MAPPING: dict[str, LLMTestCaseParams] = {
    "input": LLMTestCaseParams.INPUT,
    "actual_output": LLMTestCaseParams.ACTUAL_OUTPUT,
    "expected_output": LLMTestCaseParams.EXPECTED_OUTPUT,
    "context": LLMTestCaseParams.CONTEXT,
    "retrieval_context": LLMTestCaseParams.RETRIEVAL_CONTEXT,
}


class GEvalEvaluator(DeepEvalBaseEvaluator):
    """G-Eval custom criteria evaluator.

    Evaluates LLM outputs against user-defined criteria using the G-Eval algorithm,
    which combines chain-of-thought prompting with token probability scoring.

    G-Eval works in two phases:
    1. Step Generation: Auto-generates evaluation steps from the criteria
    2. Scoring: Uses the steps to score the test case on a 1-5 scale (normalized to 0-1)

    Attributes:
        _metric_name: Custom name for this evaluation metric
        _criteria: Natural language criteria for evaluation
        _evaluation_params: Test case fields to include in evaluation
        _evaluation_steps: Optional explicit evaluation steps
        _strict_mode: Whether to use binary scoring (1.0 or 0.0)

    Example:
        >>> evaluator = GEvalEvaluator(
        ...     name="Professionalism",
        ...     criteria="Evaluate if the response uses professional language",
        ...     threshold=0.7
        ... )
        >>> result = await evaluator.evaluate(
        ...     input="Write me an email",
        ...     actual_output="Dear Sir/Madam, ..."
        ... )
        >>> print(result["score"])  # 0.85
        >>> print(result["passed"])  # True
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: list[str] | None = None,
        evaluation_steps: list[str] | None = None,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        strict_mode: bool = False,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
        observability_config: TracingConfig | None = None,
    ) -> None:
        """Initialize G-Eval evaluator.

        Args:
            name: Metric identifier (e.g., "Correctness", "Helpfulness")
            criteria: Natural language evaluation criteria
            evaluation_params: Test case fields to include in evaluation.
                Valid options: ["input", "actual_output", "expected_output",
                              "context", "retrieval_context"]
                Default: ["actual_output"]
            evaluation_steps: Explicit evaluation steps. If None, G-Eval
                auto-generates steps from the criteria.
            model_config: LLM judge configuration. Defaults to Ollama gpt-oss:20b.
            threshold: Pass/fail score threshold (0.0-1.0). Default: 0.5.
            strict_mode: If True, scores are binary (1.0 or 0.0). Default: False.
            timeout: Evaluation timeout in seconds. Default: 60.0.
            retry_config: Retry configuration for transient failures.
            observability_config: Tracing configuration for span instrumentation.
                                 If None, no spans are created.

        Raises:
            ValueError: If invalid evaluation_params are provided.
        """
        # Validate and set evaluation params before calling super().__init__
        if evaluation_params is None:
            evaluation_params = ["actual_output"]

        # Validate evaluation params
        for param in evaluation_params:
            if param not in VALID_EVALUATION_PARAMS:
                raise ValueError(
                    f"Invalid evaluation_param: '{param}'. "
                    f"Valid options: {sorted(VALID_EVALUATION_PARAMS)}"
                )

        self._metric_name = name
        self._criteria = criteria
        self._evaluation_params = evaluation_params
        self._evaluation_steps = evaluation_steps
        self._strict_mode = strict_mode

        super().__init__(
            model_config=model_config,
            threshold=threshold,
            timeout=timeout,
            retry_config=retry_config,
            observability_config=observability_config,
        )

        logger.debug(
            f"GEvalEvaluator initialized: name={name}, "
            f"criteria_len={len(criteria)}, "
            f"evaluation_params={evaluation_params}, "
            f"strict_mode={strict_mode}"
        )

    @property
    def name(self) -> str:
        """Return the custom metric name."""
        return self._metric_name

    def _create_metric(self) -> GEval:
        """Create DeepEval GEval metric instance.

        Returns:
            Configured GEval metric instance.
        """
        # Convert string params to LLMTestCaseParams enum values
        enum_params = [PARAM_MAPPING[p] for p in self._evaluation_params]

        logger.debug(
            f"Creating GEval metric: name={self._metric_name}, "
            f"params={self._evaluation_params}, "
            f"steps_provided={self._evaluation_steps is not None}"
        )

        return GEval(
            name=self._metric_name,
            criteria=self._criteria,
            evaluation_params=enum_params,
            evaluation_steps=self._evaluation_steps,
            model=self._model,
            threshold=self._threshold,
            strict_mode=self._strict_mode,
            top_logprobs=5,  # OpenAI API limit is 5 (DeepEval defaults to 20)
        )
