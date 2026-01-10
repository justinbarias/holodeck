"""Faithfulness evaluator for RAG pipeline evaluation.

This module provides the FaithfulnessEvaluator class that wraps DeepEval's
FaithfulnessMetric for detecting hallucinations by comparing agent responses
to retrieval context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from deepeval.metrics import FaithfulnessMetric

from holodeck.lib.evaluators.base import RetryConfig
from holodeck.lib.evaluators.deepeval.base import DeepEvalBaseEvaluator
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.models.observability import TracingConfig

logger = get_logger(__name__)


class FaithfulnessEvaluator(DeepEvalBaseEvaluator):
    """Faithfulness evaluator for detecting hallucinations.

    Detects hallucinations by comparing agent response to retrieval context.
    Returns a low score if the response contains information not found in
    the retrieval context (hallucination detected).

    Required inputs:
        - input: User query
        - actual_output: Agent response
        - retrieval_context: List of retrieved text chunks

    Example:
        >>> evaluator = FaithfulnessEvaluator(threshold=0.8)
        >>> result = await evaluator.evaluate(
        ...     input="What are the store hours?",
        ...     actual_output="Store is open 24/7.",
        ...     retrieval_context=["Store hours: Mon-Fri 9am-5pm"]
        ... )
        >>> print(result["score"])  # Low score (hallucination detected)

    Attributes:
        _include_reason: Whether to include reasoning in results.
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset(
            {EvalParam.ACTUAL_OUTPUT, EvalParam.INPUT, EvalParam.RETRIEVAL_CONTEXT}
        ),
        uses_retrieval_context=True,
    )

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
        observability_config: TracingConfig | None = None,
    ) -> None:
        """Initialize Faithfulness evaluator.

        Args:
            model_config: LLM judge configuration. Defaults to Ollama gpt-oss:20b.
            threshold: Pass/fail score threshold (0.0-1.0). Default: 0.5.
            include_reason: Whether to include reasoning in results. Default: True.
            timeout: Evaluation timeout in seconds. Default: 60.0.
            retry_config: Retry configuration for transient failures.
            observability_config: Tracing configuration for span instrumentation.
                                 If None, no spans are created.
        """
        self._include_reason = include_reason

        super().__init__(
            model_config=model_config,
            threshold=threshold,
            timeout=timeout,
            retry_config=retry_config,
            observability_config=observability_config,
        )

        logger.debug(
            f"FaithfulnessEvaluator initialized: "
            f"provider={self._model_config.provider.value}, "
            f"model={self._model_config.model_name}, "
            f"threshold={threshold}, include_reason={include_reason}"
        )

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "Faithfulness"

    def _create_metric(self) -> FaithfulnessMetric:
        """Create DeepEval Faithfulness metric instance.

        Returns:
            Configured FaithfulnessMetric instance.
        """
        logger.debug(
            f"Creating FaithfulnessMetric: "
            f"threshold={self._threshold}, include_reason={self._include_reason}"
        )

        return FaithfulnessMetric(
            model=self._model,
            threshold=self._threshold,
            include_reason=self._include_reason,
        )

    def _extract_result(self, metric: Any) -> dict[str, Any]:
        """Extract result from FaithfulnessMetric.

        Args:
            metric: The evaluated FaithfulnessMetric instance.

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
