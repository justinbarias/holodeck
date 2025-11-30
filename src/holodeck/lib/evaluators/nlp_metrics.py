"""NLP metrics evaluators using Hugging Face evaluate library and SacreBLEU.

This module implements traditional NLP metrics (BLEU, ROUGE, METEOR) for
evaluating agent responses against ground truth text. BLEU uses SacreBLEU
directly for better smoothing support, while other metrics use the Hugging
Face evaluate library.

Supported Metrics:
- BLEU (SacreBLEU): Machine translation quality (precision-focused)
- ROUGE: Summarization quality (recall-focused)
- METEOR: Translation quality with synonym handling
- F1: Text similarity for classification tasks

References:
- Research: specs/006-agent-test-execution/research/
  test-execution-integration-research.md (Section 4: NLP Metrics Libraries)
- Hugging Face evaluate: https://huggingface.co/docs/evaluate/
- SacreBLEU: https://github.com/mjpost/sacrebleu
"""

from typing import Any, ClassVar

from holodeck.lib.evaluators.base import BaseEvaluator, EvaluationError
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid loading libraries if not needed
_evaluate = None
_sacrebleu = None


def _get_evaluate() -> Any:
    """Lazy load evaluate library."""
    global _evaluate
    if _evaluate is None:
        try:
            logger.debug("Loading evaluate library")
            import evaluate as ev

            _evaluate = ev
            logger.debug("evaluate library loaded successfully")
        except ImportError as e:
            logger.error("Failed to import evaluate library", exc_info=True)
            raise NLPMetricsError(
                "evaluate library is not installed. "
                "Install with: pip install evaluate"
            ) from e
    return _evaluate


def _get_sacrebleu() -> Any:
    """Lazy load sacrebleu library."""
    global _sacrebleu
    if _sacrebleu is None:
        try:
            logger.debug("Loading sacrebleu library")
            import sacrebleu as sb

            _sacrebleu = sb
            logger.debug("sacrebleu library loaded successfully")
        except ImportError as e:
            logger.error("Failed to import sacrebleu library", exc_info=True)
            raise NLPMetricsError(
                "sacrebleu library is not installed. "
                "Install with: pip install sacrebleu"
            ) from e
    return _sacrebleu


class NLPMetricsError(EvaluationError):
    """Exception raised when NLP metric computation fails."""

    pass


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluator using SacreBLEU with smoothing.

    BLEU (Bilingual Evaluation Understudy) measures precision of n-gram matches
    between prediction and reference text. Uses SacreBLEU with exponential
    smoothing to handle short sentences and avoid zero scores when there are
    no 4-gram matches.

    Score Range: 0.0-1.0 (normalized from SacreBLEU's 0-100 scale)
    Higher scores indicate better match to reference text.

    Attributes:
        threshold: Minimum passing score (0.0-1.0)
        timeout: Timeout in seconds for evaluation
        retry_config: Retry configuration for transient failures

    Example:
        >>> evaluator = BLEUEvaluator(threshold=0.5)
        >>> result = await evaluator.evaluate(
        ...     response="The cat sat on the mat",
        ...     ground_truth="The cat is on the mat"
        ... )
        >>> print(result["bleu"])  # 0.0-1.0
        >>> print(result["passed"])  # True if >= threshold
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
    )

    def __init__(
        self,
        threshold: float | None = None,
        timeout: float | None = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize BLEU evaluator.

        Args:
            threshold: Minimum passing score (0.0-1.0)
            timeout: Timeout in seconds
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(timeout=timeout, **kwargs)
        self.threshold = threshold

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement BLEU score calculation using SacreBLEU.

        Args:
            response: Agent's response text (required)
            ground_truth: Expected reference text (required)
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with:
                - bleu: float (0.0-1.0, normalized from SacreBLEU 0-100)
                - passed: bool (True if >= threshold, or True if no threshold)

        Raises:
            ValueError: If response or ground_truth is missing or invalid
            NLPMetricsError: If BLEU computation fails
        """
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")

        if response is None:
            raise ValueError("response is required for BLEU evaluation")
        if ground_truth is None:
            raise ValueError("ground_truth is required for BLEU evaluation")

        try:
            logger.debug("Computing BLEU score")
            sacrebleu = _get_sacrebleu()

            # Convert to string if needed
            response = str(response) if response else ""
            ground_truth = str(ground_truth) if ground_truth else ""

            logger.debug(
                f"BLEU: response length={len(response)}, "
                f"ground_truth length={len(ground_truth)}"
            )

            # SacreBLEU expects a hypothesis string and list of reference strings
            # Use exponential smoothing to avoid zero scores for short sentences
            result = sacrebleu.sentence_bleu(
                response, [ground_truth], smooth_method="exp"
            )

            # Normalize from 0-100 to 0-1 range
            score = result.score / 100.0

            # Check threshold (convert to native bool to avoid np.True_)
            passed = (
                bool(score >= self.threshold) if self.threshold is not None else True
            )

            logger.debug(
                f"BLEU computation completed: score={score:.3f}, "
                f"threshold={self.threshold}, passed={passed}"
            )

            return {"bleu": score, "passed": passed}

        except Exception as e:
            logger.error(f"BLEU computation failed: {e}", exc_info=True)
            raise NLPMetricsError(f"BLEU computation failed: {e}") from e


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score evaluator.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall
    of n-gram overlaps between prediction and reference. Commonly used for
    summarization evaluation.

    Variants:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence

    Score Range: 0.0-1.0 (F1 score)
    Higher scores indicate better recall of reference text.

    Attributes:
        threshold: Minimum passing score (0.0-1.0)
        variant: ROUGE variant to use for threshold check
            ("rouge1", "rouge2", "rougeL")
        timeout: Timeout in seconds for evaluation
        retry_config: Retry configuration for transient failures

    Example:
        >>> evaluator = ROUGEEvaluator(threshold=0.6, variant="rougeL")
        >>> result = await evaluator.evaluate(
        ...     response="The cat sat on the mat",
        ...     ground_truth="The cat is on the mat"
        ... )
        >>> print(result["rouge1"])  # 0.0-1.0
        >>> print(result["rouge2"])  # 0.0-1.0
        >>> print(result["rougeL"])  # 0.0-1.0
        >>> print(result["passed"])  # True if rougeL >= threshold
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
    )

    def __init__(
        self,
        threshold: float | None = None,
        variant: str = "rougeL",
        timeout: float | None = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize ROUGE evaluator.

        Args:
            threshold: Minimum passing score (0.0-1.0)
            variant: ROUGE variant for threshold check
                ("rouge1", "rouge2", "rougeL")
            timeout: Timeout in seconds
            **kwargs: Additional arguments passed to BaseEvaluator

        Raises:
            ValueError: If variant is not valid
        """
        super().__init__(timeout=timeout, **kwargs)
        self.threshold = threshold

        valid_variants = {"rouge1", "rouge2", "rougeL"}
        if variant not in valid_variants:
            raise ValueError(f"variant must be one of {valid_variants}, got: {variant}")
        self.variant = variant
        self._metric = None  # Lazy loaded

    def _get_metric(self) -> Any:
        """Lazy load ROUGE metric."""
        if self._metric is None:
            evaluate = _get_evaluate()
            self._metric = evaluate.load("rouge")
        return self._metric

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement ROUGE score calculation.

        Args:
            response: Agent's response text (required)
            ground_truth: Expected reference text (required)
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with:
                - rouge1: float (0.0-1.0)
                - rouge2: float (0.0-1.0)
                - rougeL: float (0.0-1.0)
                - passed: bool (True if variant score >= threshold)

        Raises:
            ValueError: If response or ground_truth is missing
            NLPMetricsError: If ROUGE computation fails
        """
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")

        if response is None:
            raise ValueError("response is required for ROUGE evaluation")
        if ground_truth is None:
            raise ValueError("ground_truth is required for ROUGE evaluation")

        try:
            metric = self._get_metric()

            # Convert to string if needed
            response = str(response) if response else ""
            ground_truth = str(ground_truth) if ground_truth else ""

            # ROUGE expects predictions as list of strings
            # and references as list of strings
            result = metric.compute(predictions=[response], references=[ground_truth])

            rouge1 = result["rouge1"]
            rouge2 = result["rouge2"]
            rouge_l = result["rougeL"]

            # Check threshold using selected variant (avoid np.True_)
            variant_score = result[self.variant]
            passed = (
                bool(variant_score >= self.threshold)
                if self.threshold is not None
                else True
            )

            return {
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rouge_l,
                "passed": passed,
            }

        except Exception as e:
            raise NLPMetricsError(f"ROUGE computation failed: {e}") from e


class METEOREvaluator(BaseEvaluator):
    """METEOR score evaluator.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    measures translation quality using synonym matching, stemming, and
    paraphrase detection. Provides better correlation with human judgment
    than BLEU.

    Score Range: 0.0-1.0
    Higher scores indicate better semantic match to reference text.

    Attributes:
        threshold: Minimum passing score (0.0-1.0)
        timeout: Timeout in seconds for evaluation
        retry_config: Retry configuration for transient failures

    Example:
        >>> evaluator = METEOREvaluator(threshold=0.7)
        >>> result = await evaluator.evaluate(
        ...     response="The automobile is red",
        ...     ground_truth="The car is red"
        ... )
        >>> print(result["meteor"])  # Higher than BLEU due to synonym handling
        >>> print(result["passed"])  # True if >= threshold
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
    )

    def __init__(
        self,
        threshold: float | None = None,
        timeout: float | None = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize METEOR evaluator.

        Args:
            threshold: Minimum passing score (0.0-1.0)
            timeout: Timeout in seconds
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(timeout=timeout, **kwargs)
        self.threshold = threshold
        self._metric = None  # Lazy loaded

    def _get_metric(self) -> Any:
        """Lazy load METEOR metric."""
        if self._metric is None:
            evaluate = _get_evaluate()
            self._metric = evaluate.load("meteor")
        return self._metric

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement METEOR score calculation.

        Args:
            response: Agent's response text (required)
            ground_truth: Expected reference text (required)
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary with:
                - meteor: float (0.0-1.0)
                - passed: bool (True if >= threshold)

        Raises:
            ValueError: If response or ground_truth is missing
            NLPMetricsError: If METEOR computation fails
        """
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")

        if response is None:
            raise ValueError("response is required for METEOR evaluation")
        if ground_truth is None:
            raise ValueError("ground_truth is required for METEOR evaluation")

        try:
            metric = self._get_metric()

            # Convert to string if needed
            response = str(response) if response else ""
            ground_truth = str(ground_truth) if ground_truth else ""

            # METEOR expects predictions as list of strings
            # and references as list of strings
            result = metric.compute(predictions=[response], references=[ground_truth])

            score = result["meteor"]

            # Check threshold (convert to native bool to avoid np.True_)
            passed = (
                bool(score >= self.threshold) if self.threshold is not None else True
            )

            return {"meteor": score, "passed": passed}

        except Exception as e:
            raise NLPMetricsError(f"METEOR computation failed: {e}") from e


# Export public API
__all__ = [
    "BLEUEvaluator",
    "ROUGEEvaluator",
    "METEOREvaluator",
    "NLPMetricsError",
]
