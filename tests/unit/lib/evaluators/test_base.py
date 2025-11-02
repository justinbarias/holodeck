"""Unit tests for base evaluator abstract interface.

Tests cover:
- Abstract evaluate() method enforcement
- Timeout handling
- Retry logic with exponential backoff
- Error handling and reporting
"""

import asyncio
from typing import Any

import pytest

from holodeck.lib.evaluators.base import BaseEvaluator, EvaluationError, RetryConfig


class TestRetryConfig:
    """Test RetryConfig model."""

    def test_default_values(self) -> None:
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self) -> None:
        """Test custom retry configuration values."""
        config = RetryConfig(
            max_retries=5, base_delay=1.0, max_delay=30.0, exponential_base=1.5
        )
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5

    def test_validation(self) -> None:
        """Test retry config validation."""
        with pytest.raises(ValueError):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValueError):
            RetryConfig(base_delay=0)

        with pytest.raises(ValueError):
            RetryConfig(max_delay=0)


class ConcreteEvaluator(BaseEvaluator):
    """Concrete evaluator implementation for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize concrete evaluator."""
        super().__init__(**kwargs)
        self.evaluate_call_count = 0

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Implement abstract evaluate method."""
        self.evaluate_call_count += 1
        return {"score": 0.85, "passed": True}


class FailingEvaluator(BaseEvaluator):
    """Evaluator that always fails for testing error handling."""

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Always raise an error."""
        raise ValueError("Simulated evaluation error")


class TimeoutEvaluator(BaseEvaluator):
    """Evaluator that simulates timeout."""

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Simulate long-running evaluation."""
        await asyncio.sleep(10)  # Longer than timeout
        return {"score": 0.5}


class TransientFailureEvaluator(BaseEvaluator):
    """Evaluator that fails on first attempts then succeeds."""

    def __init__(self, fail_count: int = 2, **kwargs: Any) -> None:
        """Initialize with number of failures before success."""
        super().__init__(**kwargs)
        self.attempt_count = 0
        self.fail_count = fail_count

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Fail for first N attempts, then succeed."""
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            raise ConnectionError(f"Transient error (attempt {self.attempt_count})")
        return {"score": 0.9, "passed": True}


class TestBaseEvaluator:
    """Test BaseEvaluator abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore

    @pytest.mark.asyncio
    async def test_concrete_evaluator_works(self) -> None:
        """Test that concrete implementation can be created and used."""
        evaluator = ConcreteEvaluator()
        result = await evaluator.evaluate(query="test", response="test response")

        assert result["score"] == 0.85
        assert result["passed"] is True
        assert evaluator.evaluate_call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_timeout(self) -> None:
        """Test that evaluation respects timeout setting."""
        evaluator = TimeoutEvaluator(timeout=1.0)  # 1 second timeout

        with pytest.raises(asyncio.TimeoutError):
            await evaluator.evaluate(query="test", response="test")

    @pytest.mark.asyncio
    async def test_evaluate_without_timeout(self) -> None:
        """Test evaluation without timeout (None)."""
        evaluator = ConcreteEvaluator(timeout=None)
        result = await evaluator.evaluate(query="test", response="test")

        assert result["score"] == 0.85

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry logic with transient failures."""
        retry_config = RetryConfig(max_retries=3, base_delay=0.1)
        evaluator = TransientFailureEvaluator(fail_count=2, retry_config=retry_config)

        result = await evaluator.evaluate(query="test", response="test")

        assert result["score"] == 0.9
        assert result["passed"] is True
        assert evaluator.attempt_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self) -> None:
        """Test that retries are exhausted and error is raised."""
        retry_config = RetryConfig(max_retries=2, base_delay=0.1)
        evaluator = TransientFailureEvaluator(
            fail_count=5, retry_config=retry_config
        )  # More failures than retries

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(query="test", response="test")

        assert "Transient error" in str(exc_info.value)
        assert evaluator.attempt_count == 2  # max_retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Test exponential backoff delay calculation."""
        retry_config = RetryConfig(
            max_retries=3, base_delay=1.0, exponential_base=2.0, max_delay=10.0
        )
        # Fail on first 2 attempts (attempt 1, 2), succeed on attempt 3
        evaluator = TransientFailureEvaluator(fail_count=2, retry_config=retry_config)

        start_time = asyncio.get_event_loop().time()
        result = await evaluator.evaluate(query="test", response="test")
        elapsed_time = asyncio.get_event_loop().time() - start_time

        # Expected delays: 1.0s (after attempt 0), 2.0s (after attempt 1) = 3.0s total
        # Allow some tolerance for execution time
        assert elapsed_time >= 2.5, f"Elapsed time {elapsed_time}s is too short"
        assert (
            elapsed_time <= 4.0
        ), f"Elapsed time {elapsed_time}s is too long"  # Allow overhead

        assert result["score"] == 0.9
        assert evaluator.attempt_count == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_max_delay_cap(self) -> None:
        """Test that delay is capped at max_delay."""
        retry_config = RetryConfig(
            max_retries=10,  # Many retries
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,  # Cap at 5 seconds
        )
        evaluator = TransientFailureEvaluator(fail_count=5, retry_config=retry_config)

        start_time = asyncio.get_event_loop().time()
        result = await evaluator.evaluate(query="test", response="test")
        elapsed_time = asyncio.get_event_loop().time() - start_time

        # Delays: 1.0, 2.0, 4.0, 5.0 (capped), 5.0 (capped) = 17.0s total
        # But we only need 5 failures, so: 1.0, 2.0, 4.0, 5.0, 5.0 = 17.0s
        assert (
            elapsed_time >= 16.0
        ), f"Elapsed time {elapsed_time}s should be ~17s (with cap)"

        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_non_retryable_error(self) -> None:
        """Test that non-retryable errors fail immediately."""
        evaluator = FailingEvaluator(
            retry_config=RetryConfig(max_retries=3, base_delay=0.1)
        )

        with pytest.raises(EvaluationError) as exc_info:
            await evaluator.evaluate(query="test", response="test")

        assert "Simulated evaluation error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self) -> None:
        """Test evaluation with context parameter."""
        evaluator = ConcreteEvaluator()
        result = await evaluator.evaluate(
            query="test", response="test response", context="additional context"
        )

        assert result["score"] == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_with_ground_truth(self) -> None:
        """Test evaluation with ground_truth parameter."""
        evaluator = ConcreteEvaluator()
        result = await evaluator.evaluate(
            query="test", response="test response", ground_truth="expected answer"
        )

        assert result["score"] == 0.85

    @pytest.mark.asyncio
    async def test_default_timeout_value(self) -> None:
        """Test default timeout is 60 seconds."""
        evaluator = ConcreteEvaluator()
        assert evaluator.timeout == 60.0

    @pytest.mark.asyncio
    async def test_default_retry_config(self) -> None:
        """Test default retry configuration."""
        evaluator = ConcreteEvaluator()
        assert evaluator.retry_config.max_retries == 3
        assert evaluator.retry_config.base_delay == 2.0
        assert evaluator.retry_config.exponential_base == 2.0

    def test_evaluator_name_property(self) -> None:
        """Test evaluator name property returns class name."""
        evaluator = ConcreteEvaluator()
        assert evaluator.name == "ConcreteEvaluator"

    @pytest.mark.asyncio
    async def test_result_includes_metadata(self) -> None:
        """Test that result includes evaluation metadata."""
        evaluator = ConcreteEvaluator()
        result = await evaluator.evaluate(query="test", response="test response")

        # Base implementation should return result from _evaluate_impl
        assert "score" in result
        assert "passed" in result


class TestRetryableErrors:
    """Test retry logic for different error types."""

    @pytest.mark.asyncio
    async def test_connection_error_is_retryable(self) -> None:
        """Test that ConnectionError triggers retry."""

        class ConnectionErrorEvaluator(BaseEvaluator):
            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
                self.call_count = 0

            async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
                self.call_count += 1
                if self.call_count < 2:
                    raise ConnectionError("Network error")
                return {"score": 0.5}

        evaluator = ConnectionErrorEvaluator(
            retry_config=RetryConfig(max_retries=3, base_delay=0.1)
        )
        result = await evaluator.evaluate(query="test", response="test")

        assert result["score"] == 0.5
        assert evaluator.call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_timeout_error_is_retryable(self) -> None:
        """Test that TimeoutError triggers retry."""

        class TimeoutErrorEvaluator(BaseEvaluator):
            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
                self.call_count = 0

            async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
                self.call_count += 1
                if self.call_count < 2:
                    raise TimeoutError("Request timeout")
                return {"score": 0.5}

        evaluator = TimeoutErrorEvaluator(
            retry_config=RetryConfig(max_retries=3, base_delay=0.1)
        )
        result = await evaluator.evaluate(query="test", response="test")

        assert result["score"] == 0.5
        assert evaluator.call_count == 2

    @pytest.mark.asyncio
    async def test_value_error_not_retryable(self) -> None:
        """Test that ValueError does not trigger retry (fails immediately)."""

        class ValueErrorEvaluator(BaseEvaluator):
            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
                self.call_count = 0

            async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
                self.call_count += 1
                raise ValueError("Invalid input")

        evaluator = ValueErrorEvaluator(
            retry_config=RetryConfig(max_retries=3, base_delay=0.1)
        )

        with pytest.raises(EvaluationError):
            await evaluator.evaluate(query="test", response="test")

        # Should not retry on ValueError
        assert evaluator.call_count == 1
