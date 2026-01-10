"""Answer Relevancy evaluator for response quality evaluation.

This module provides the AnswerRelevancyEvaluator class that wraps DeepEval's
AnswerRelevancyMetric for measuring how relevant the response statements are
to the input query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from deepeval.metrics import AnswerRelevancyMetric

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.models.observability import TracingConfig

logger = get_logger(__name__)


class AnswerRelevancyEvaluator(DeepEvalBaseEvaluator):
    """Answer Relevancy evaluator - measures statement relevance to input.

    Evaluates how relevant the response statements are to the input query.
    Unlike other RAG metrics, this does NOT require retrieval_context.

    Required inputs:
        - input: User query
        - actual_output: Agent response

    Example:
        >>> evaluator = AnswerRelevancyEvaluator(threshold=0.7)
        >>> result = await evaluator.evaluate(
        ...     input="What is the return policy?",
        ...     actual_output="We offer a 30-day full refund at no extra cost."
        ... )
        >>> print(result["score"])  # High score if relevant

    Attributes:
        _include_reason: Whether to include reasoning in results.
        _strict_mode: Whether to use binary scoring (1.0 or 0.0).
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.ACTUAL_OUTPUT, EvalParam.INPUT}),
        uses_retrieval_context=False,
    )

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
        observability_config: TracingConfig | None = None,
    ) -> None:
        """Initialize Answer Relevancy evaluator.

        Args:
            model_config: LLM judge configuration. Defaults to Ollama gpt-oss:20b.
            threshold: Pass/fail score threshold (0.0-1.0). Default: 0.5.
            include_reason: Whether to include reasoning in results. Default: True.
            strict_mode: Binary scoring mode (1.0 or 0.0 only). Default: False.
            timeout: Evaluation timeout in seconds. Default: 60.0.
            retry_config: Retry configuration for transient failures.
            observability_config: Tracing configuration for span instrumentation.
                                 If None, no spans are created.
        """
        self._include_reason = include_reason
        self._strict_mode = strict_mode

        super().__init__(
            model_config=model_config,
            threshold=threshold,
            timeout=timeout,
            retry_config=retry_config,
            observability_config=observability_config,
        )

        logger.debug(
            f"AnswerRelevancyEvaluator initialized: "
            f"provider={self._model_config.provider.value}, "
            f"model={self._model_config.model_name}, "
            f"threshold={threshold}, include_reason={include_reason}, "
            f"strict_mode={strict_mode}"
        )

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "AnswerRelevancy"

    def _create_metric(self) -> AnswerRelevancyMetric:
        """Create DeepEval AnswerRelevancy metric instance.

        Returns:
            Configured AnswerRelevancyMetric instance.
        """
        logger.debug(
            f"Creating AnswerRelevancyMetric: "
            f"threshold={self._threshold}, include_reason={self._include_reason}, "
            f"strict_mode={self._strict_mode}"
        )

        return AnswerRelevancyMetric(
            model=self._model,
            threshold=self._threshold,
            include_reason=self._include_reason,
            strict_mode=self._strict_mode,
        )

    def _extract_result(self, metric: Any) -> dict[str, Any]:
        """Extract result from AnswerRelevancyMetric.

        Args:
            metric: The evaluated AnswerRelevancyMetric instance.

        Returns:
            Dictionary with score, passed, reasoning, metric_name, and threshold.
        """
        score = metric.score if metric.score is not None else 0.0
        return {
            "score": score,
            "passed": score >= self._threshold,
            "reasoning": metric.reason or "",
            "metric_name": self.name,
            "threshold": self._threshold,
        }
