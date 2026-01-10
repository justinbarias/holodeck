"""Base evaluator for DeepEval metrics.

This module provides the abstract base class for all DeepEval-based evaluators.
It handles model configuration, test case construction, and result normalization.
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ClassVar

from deepeval.test_case import LLMTestCase

from holodeck.lib.evaluators.base import BaseEvaluator, RetryConfig
from holodeck.lib.evaluators.deepeval.config import DeepEvalModelConfig
from holodeck.lib.evaluators.deepeval.errors import DeepEvalError
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger

if TYPE_CHECKING:
    from holodeck.models.observability import TracingConfig

logger = get_logger(__name__)


class DeepEvalBaseEvaluator(BaseEvaluator):
    """Abstract base class for DeepEval-based evaluators.

    This class extends BaseEvaluator to provide DeepEval-specific functionality:
    - Model configuration and initialization
    - LLMTestCase construction from evaluation inputs
    - Result normalization and logging

    Subclasses must implement _create_metric() to return the specific
    DeepEval metric instance.

    Note: DeepEval uses different parameter names than Azure AI/NLP:
    - input (not query)
    - actual_output (not response)
    - expected_output (not ground_truth)

    Attributes:
        model_config: Configuration for the evaluation LLM
        threshold: Score threshold for pass/fail determination
        model: The initialized DeepEval model instance

    Example:
        >>> class MyMetricEvaluator(DeepEvalBaseEvaluator):
        ...     def _create_metric(self):
        ...         return SomeDeepEvalMetric(
        ...             threshold=self._threshold,
        ...             model=self._model
        ...         )
        >>>
        >>> evaluator = MyMetricEvaluator(threshold=0.7)
        >>> result = await evaluator.evaluate(
        ...     input="What is Python?",
        ...     actual_output="Python is a programming language."
        ... )
    """

    # Default PARAM_SPEC for DeepEval evaluators - uses DeepEval param names
    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.ACTUAL_OUTPUT}),
        optional=frozenset(
            {
                EvalParam.INPUT,
                EvalParam.EXPECTED_OUTPUT,
                EvalParam.CONTEXT,
                EvalParam.RETRIEVAL_CONTEXT,
            }
        ),
    )

    def __init__(
        self,
        model_config: DeepEvalModelConfig | None = None,
        threshold: float = 0.5,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
        observability_config: TracingConfig | None = None,
    ) -> None:
        """Initialize DeepEval base evaluator.

        Args:
            model_config: Configuration for the evaluation model.
                         Defaults to Ollama with gpt-oss:20b.
            threshold: Score threshold for pass/fail (0.0-1.0, default: 0.5)
            timeout: Evaluation timeout in seconds (default: 60.0)
            retry_config: Retry configuration for transient failures
            observability_config: Tracing configuration for span instrumentation.
                                 If None, no spans are created.
        """
        super().__init__(timeout=timeout, retry_config=retry_config)
        self._model_config = model_config or DeepEvalModelConfig()
        self._model = self._model_config.to_deepeval_model()
        self._threshold = threshold
        self._observability_config = observability_config

        logger.debug(
            f"DeepEval evaluator initialized: {self.name}, "
            f"provider={self._model_config.provider.value}, "
            f"model={self._model_config.model_name}, "
            f"threshold={threshold}"
        )

    def _build_test_case(self, **kwargs: Any) -> LLMTestCase:
        """Build DeepEval LLMTestCase from evaluation inputs.

        Supports both standard DeepEval parameter names and HoloDeck aliases:
        - input / query: The user's input query
        - actual_output / response: The agent's response
        - expected_output / ground_truth: The expected correct answer
        - context: General context information
        - retrieval_context: Retrieved chunks for RAG evaluation

        Args:
            **kwargs: Evaluation parameters with standard or aliased names

        Returns:
            LLMTestCase configured with the provided parameters
        """
        # Resolve input with standard name taking precedence over alias
        input_value: str = (
            str(kwargs.get("input"))
            if "input" in kwargs
            else str(kwargs.get("query", ""))
        )

        # Resolve actual_output with standard name taking precedence over alias
        actual_output_value: str = (
            str(kwargs.get("actual_output"))
            if "actual_output" in kwargs
            else str(kwargs.get("response", ""))
        )

        # Resolve expected_output with standard name taking precedence over alias
        expected_output_value: str | None = (
            str(kwargs.get("expected_output"))
            if "expected_output" in kwargs
            else (str(kwargs.get("ground_truth")) if "ground_truth" in kwargs else None)
        )

        return LLMTestCase(
            input=input_value,
            actual_output=actual_output_value,
            expected_output=expected_output_value,
            context=kwargs.get("context"),
            retrieval_context=kwargs.get("retrieval_context"),
        )

    def _summarize_test_case(self, test_case: LLMTestCase) -> dict[str, str]:
        """Create a truncated summary of test case for error reporting.

        Args:
            test_case: The LLMTestCase to summarize

        Returns:
            Dictionary with truncated field values (max 100 chars each)
        """
        max_len = 100

        def truncate(value: Any) -> str:
            if value is None:
                return ""
            s = str(value)
            return s[:max_len] + "..." if len(s) > max_len else s

        summary: dict[str, str] = {}
        if test_case.input:
            summary["input"] = truncate(test_case.input)
        if test_case.actual_output:
            summary["actual_output"] = truncate(test_case.actual_output)
        if test_case.expected_output:
            summary["expected_output"] = truncate(test_case.expected_output)
        if test_case.retrieval_context:
            summary["retrieval_context"] = truncate(test_case.retrieval_context)

        return summary

    def _validate_retrieval_context(self, **kwargs: Any) -> None:
        """Validate that retrieval_context is present for RAG metrics.

        RAG-specific metrics (Faithfulness, ContextualRelevancy, etc.) require
        retrieval_context to function properly. This method should be called
        by RAG evaluators to ensure required inputs are provided.

        Args:
            **kwargs: Evaluation parameters to validate

        Raises:
            ValueError: If retrieval_context is not provided
        """
        if "retrieval_context" not in kwargs or kwargs["retrieval_context"] is None:
            raise ValueError(
                f"{self.__class__.__name__} requires retrieval_context. "
                "Provide retrieval_context=[...] with retrieved text chunks."
            )

    def _create_span_context(self) -> Any:
        """Create span context for observability instrumentation.

        Returns a span context manager if observability is enabled,
        otherwise returns a nullcontext (no-op).

        Returns:
            Context manager that yields a span or None
        """
        if self._observability_config is None:
            return nullcontext()

        from holodeck.lib.observability import get_tracer

        tracer = get_tracer(__name__)
        return tracer.start_as_current_span(f"holodeck.evaluation.{self.name}")

    def _should_capture_content(self) -> bool:
        """Check if evaluation content capture is enabled.

        Returns:
            True if observability is configured and capture_evaluation_content is True
        """
        return (
            self._observability_config is not None
            and self._observability_config.capture_evaluation_content
        )

    def _set_span_attributes(
        self,
        span: Any,
        test_case: LLMTestCase,
        start_time: float,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Set span attributes for evaluation observability.

        Args:
            span: The OpenTelemetry span to set attributes on
            test_case: The LLMTestCase being evaluated
            start_time: Start time from time.perf_counter()
            result: Evaluation result dict (if evaluation completed)
        """
        if span is None:
            return

        # Standard attributes (always captured)
        span.set_attribute("evaluation.metric.name", self.name)
        span.set_attribute("evaluation.threshold", self._threshold)
        span.set_attribute(
            "evaluation.model.provider", self._model_config.provider.value
        )
        span.set_attribute("evaluation.model.name", self._model_config.model_name)

        # Content attributes (only when capture_evaluation_content=True)
        if self._should_capture_content():
            if test_case.input:
                span.set_attribute("evaluation.input", test_case.input[:1000])
            if test_case.actual_output:
                span.set_attribute(
                    "evaluation.actual_output", test_case.actual_output[:1000]
                )
            if test_case.expected_output:
                span.set_attribute(
                    "evaluation.expected_output", test_case.expected_output[:1000]
                )
            if test_case.retrieval_context:
                span.set_attribute(
                    "evaluation.retrieval_context",
                    json.dumps(test_case.retrieval_context)[:2000],
                )

        # Result attributes (if evaluation completed)
        if result is not None:
            span.set_attribute("evaluation.score", result.get("score", 0.0))
            span.set_attribute("evaluation.passed", result.get("passed", False))
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            span.set_attribute("evaluation.duration_ms", elapsed_ms)
            if self._should_capture_content() and result.get("reasoning"):
                span.set_attribute("evaluation.reasoning", result["reasoning"][:2000])

    @abstractmethod
    def _create_metric(self) -> Any:
        """Create the DeepEval metric instance.

        Subclasses must implement this method to return the specific
        DeepEval metric class configured with model and threshold.

        Returns:
            A DeepEval metric instance (e.g., GEval, AnswerRelevancyMetric)
        """
        pass

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Execute evaluation using the DeepEval metric with observability.

        This method:
        1. Builds an LLMTestCase from the input parameters
        2. Creates the metric instance via _create_metric()
        3. Wraps evaluation in a span (if observability enabled)
        4. Calls metric.measure() to perform evaluation
        5. Normalizes and returns the result

        Args:
            **kwargs: Evaluation parameters (input, actual_output, etc.)

        Returns:
            Dictionary containing:
                - score: Normalized score (0.0-1.0)
                - passed: Whether score meets threshold
                - reasoning: LLM-generated explanation
                - metric_name: Name of the metric
                - threshold: Configured threshold

        Raises:
            DeepEvalError: If the metric evaluation fails
        """
        start_time = time.perf_counter()
        test_case = self._build_test_case(**kwargs)
        metric = self._create_metric()

        logger.debug(
            f"Running DeepEval metric: {self.name}, "
            f"input_len={len(test_case.input or '')}, "
            f"output_len={len(test_case.actual_output or '')}"
        )

        span_ctx = self._create_span_context()

        with span_ctx as span:
            # Set pre-execution attributes
            self._set_span_attributes(span, test_case, start_time)

            try:
                metric.measure(test_case)
            except Exception as e:
                logger.error(f"DeepEval metric {self.name} failed: {e}")
                # Record error on span
                if span is not None:
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                raise DeepEvalError(
                    message=f"DeepEval metric '{self.name}' failed: {e}",
                    metric_name=self.name,
                    original_error=e,
                    test_case_summary=self._summarize_test_case(test_case),
                ) from e

            score = metric.score
            passed = bool(score >= self._threshold)
            reasoning = metric.reason or ""

            result = {
                "score": score,
                "passed": passed,
                "reasoning": reasoning,
                "metric_name": self.name,
                "threshold": self._threshold,
            }

            # Set post-execution attributes
            self._set_span_attributes(span, test_case, start_time, result)

            logger.debug(
                f"Evaluation complete: metric={self.name}, score={score:.3f}, "
                f"passed={passed}, threshold={self._threshold}"
            )

            return result
