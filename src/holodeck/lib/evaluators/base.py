"""Base evaluator interface for test case evaluation metrics.

This module provides the abstract base class for all evaluators used in the
HoloDeck test execution framework. Evaluators implement retry logic with
exponential backoff and timeout handling.

References:
- Research: specs/006-agent-test-execution/research.md
- Integration: specs/006-agent-test-execution/research/
  test-execution-integration-research.md
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_retry

logger = get_logger(__name__)


class EvaluationError(Exception):
    """Exception raised when evaluation fails after all retry attempts."""

    pass


class RetryConfig(BaseModel):
    """Configuration for retry logic with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exponential_base: Exponential base for backoff calculation (default: 2.0)

    Example:
        >>> config = RetryConfig(max_retries=3, base_delay=2.0)
        >>> # Delays will be: 2.0s, 4.0s, 8.0s
    """

    max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=2.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    exponential_base: float = Field(default=2.0, gt=1.0)

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """Validate max_retries is non-negative."""
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v

    @field_validator("base_delay", "max_delay")
    @classmethod
    def validate_delays(cls, v: float) -> float:
        """Validate delays are positive."""
        if v <= 0:
            raise ValueError("Delays must be positive")
        return v


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics.

    This class provides retry logic, timeout handling, and a common interface
    for all evaluators (AI-assisted and NLP metrics).

    Attributes:
        timeout: Timeout in seconds for evaluation (default: 60s, None for no timeout)
        retry_config: Configuration for retry logic with exponential backoff
        name: Evaluator name (defaults to class name)
        PARAM_SPEC: Class attribute declaring required/optional parameters

    Example:
        >>> class MyEvaluator(BaseEvaluator):
        ...     PARAM_SPEC = ParamSpec(
        ...         required=frozenset({EvalParam.RESPONSE, EvalParam.QUERY})
        ...     )
        ...     async def _evaluate_impl(self, **kwargs):
        ...         return {"score": 0.85, "passed": True}
        >>>
        >>> evaluator = MyEvaluator(timeout=30.0)
        >>> result = await evaluator.evaluate(query="test", response="answer")
    """

    # Default PARAM_SPEC - subclasses should override
    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE}),
    )

    def __init__(
        self,
        timeout: float | None = 60.0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize base evaluator.

        Args:
            timeout: Timeout in seconds (None for no timeout)
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

        logger.debug(
            f"Evaluator initialized: {self.name}, timeout={timeout}s, "
            f"max_retries={self.retry_config.max_retries}"
        )

    @property
    def name(self) -> str:
        """Return evaluator name (class name by default)."""
        return self.__class__.__name__

    @classmethod
    def get_param_spec(cls) -> ParamSpec:
        """Get the parameter specification for this evaluator.

        Returns:
            ParamSpec declaring required/optional parameters and context flags.
        """
        return cls.PARAM_SPEC

    @abstractmethod
    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement evaluation logic (must be overridden by subclasses).

        Args:
            **kwargs: Evaluation parameters
                (query, response, context, ground_truth, etc.)

        Returns:
            Dictionary containing evaluation results with at minimum:
                - score: float (0.0-1.0 or metric-specific scale)
                - passed: bool (if threshold-based)
                Additional metric-specific fields allowed

        Raises:
            Exception: Any evaluation-specific errors
        """
        pass

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error should trigger retry.

        Retryable errors (transient):
        - ConnectionError (network issues)
        - asyncio.TimeoutError (timeout during API call)
        - OSError (I/O errors)

        Non-retryable errors (permanent):
        - ValueError (invalid input)
        - TypeError (type mismatch)
        - Other logic errors

        Args:
            error: Exception to check

        Returns:
            True if error is retryable, False otherwise
        """
        retryable_types = (ConnectionError, asyncio.TimeoutError, OSError)
        return isinstance(error, retryable_types)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry attempt.

        Formula: min(base_delay * (exponential_base ** attempt), max_delay)

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds (capped at max_delay)

        Example:
            >>> evaluator = BaseEvaluator()
            >>> evaluator._calculate_delay(0)  # 2.0 * (2 ** 0) = 2.0s
            2.0
            >>> evaluator._calculate_delay(1)  # 2.0 * (2 ** 1) = 4.0s
            4.0
            >>> evaluator._calculate_delay(2)  # 2.0 * (2 ** 2) = 8.0s
            8.0
        """
        delay = self.retry_config.base_delay * (
            self.retry_config.exponential_base**attempt
        )
        return min(delay, self.retry_config.max_delay)

    async def _evaluate_with_retry(self, **kwargs: Any) -> dict[str, Any]:
        """Execute evaluation with retry logic.

        Implements exponential backoff retry for transient failures:
        - Retryable errors: ConnectionError, TimeoutError, OSError
        - Non-retryable errors: Fail immediately
        - Max retries: Configurable (default: 3)
        - Backoff: Exponential with configurable base and max delay

        Args:
            **kwargs: Parameters passed to _evaluate_impl()

        Returns:
            Evaluation result dictionary

        Raises:
            EvaluationError: If all retry attempts fail
        """
        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries):
            try:
                logger.debug(
                    f"Evaluation attempt {attempt + 1}/{self.retry_config.max_retries} "
                    f"for {self.name}"
                )
                return await self._evaluate_impl(**kwargs)
            except Exception as e:
                last_error = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    # Non-retryable error - fail immediately
                    logger.error(
                        f"Non-retryable error in {self.name}: {type(e).__name__}: {e}"
                    )
                    raise EvaluationError(
                        f"Evaluation failed with non-retryable error: {e}"
                    ) from e

                # Last attempt - don't delay, just raise
                if attempt == self.retry_config.max_retries - 1:
                    break

                # Calculate and apply exponential backoff delay
                delay = self._calculate_delay(attempt)
                log_retry(
                    logger,
                    f"Evaluation {self.name}",
                    attempt=attempt + 1,
                    max_attempts=self.retry_config.max_retries,
                    delay=delay,
                    error=e,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Evaluation {self.name} failed after {self.retry_config.max_retries} "
            f"attempts: {last_error}"
        )
        raise EvaluationError(
            f"Evaluation failed after {self.retry_config.max_retries} "
            f"attempts: {last_error}"
        ) from last_error

    async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        """Evaluate with timeout and retry logic.

        This is the main public interface for evaluation. It wraps the
        implementation with timeout and retry handling.

        Args:
            **kwargs: Evaluation parameters
                (query, response, context, ground_truth, etc.)

        Returns:
            Evaluation result dictionary

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            EvaluationError: If evaluation fails after retries

        Example:
            >>> evaluator = MyEvaluator(timeout=30.0)
            >>> result = await evaluator.evaluate(
            ...     query="What is the capital of France?",
            ...     response="The capital of France is Paris.",
            ...     context="France is a country in Europe.",
            ...     ground_truth="Paris"
            ... )
            >>> print(result["score"])
            0.95
        """
        logger.debug(f"Starting evaluation: {self.name} (timeout={self.timeout}s)")

        if self.timeout is None:
            # No timeout - evaluate directly with retry
            logger.debug(f"Evaluation {self.name}: no timeout")
            return await self._evaluate_with_retry(**kwargs)

        # Apply timeout using asyncio.wait_for
        try:
            logger.debug(f"Evaluation {self.name}: applying timeout of {self.timeout}s")
            return await asyncio.wait_for(
                self._evaluate_with_retry(**kwargs), timeout=self.timeout
            )
        except TimeoutError:
            logger.error(f"Evaluation {self.name} exceeded timeout of {self.timeout}s")
            raise  # Re-raise timeout error as-is


# Export public API
__all__ = ["BaseEvaluator", "EvaluationError", "RetryConfig", "EvalParam", "ParamSpec"]
