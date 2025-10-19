"""Tests for Evaluation models in holodeck.models.evaluation."""

import pytest
from pydantic import ValidationError

from holodeck.models.evaluation import EvaluationMetric
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestEvaluationMetric:
    """Tests for EvaluationMetric model."""

    def test_evaluation_metric_valid_creation(self) -> None:
        """Test creating a valid EvaluationMetric."""
        metric = EvaluationMetric(
            metric="groundedness",
        )
        assert metric.metric == "groundedness"
        assert metric.enabled is True
        assert metric.threshold is None

    def test_evaluation_metric_name_required(self) -> None:
        """Test that metric field is required."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationMetric()
        assert "metric" in str(exc_info.value).lower()

    def test_evaluation_metric_with_threshold(self) -> None:
        """Test EvaluationMetric with threshold."""
        metric = EvaluationMetric(
            metric="groundedness",
            threshold=4.0,
        )
        assert metric.threshold == 4.0

    def test_evaluation_metric_threshold_numeric(self) -> None:
        """Test that threshold must be numeric."""
        # Should accept float
        metric = EvaluationMetric(
            metric="groundedness",
            threshold=3.5,
        )
        assert metric.threshold == 3.5

    def test_evaluation_metric_threshold_optional(self) -> None:
        """Test that threshold is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.threshold is None

    def test_evaluation_metric_enabled_default_true(self) -> None:
        """Test that enabled defaults to true."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.enabled is True

    def test_evaluation_metric_enabled_can_be_false(self) -> None:
        """Test that enabled can be set to false."""
        metric = EvaluationMetric(
            metric="groundedness",
            enabled=False,
        )
        assert metric.enabled is False

    def test_evaluation_metric_with_model_override(self) -> None:
        """Test EvaluationMetric with per-metric model override."""
        model = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        metric = EvaluationMetric(
            metric="groundedness",
            model=model,
        )
        assert metric.model is not None
        assert metric.model.provider == ProviderEnum.OPENAI

    def test_evaluation_metric_model_optional(self) -> None:
        """Test that model is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.model is None

    def test_evaluation_metric_fail_on_error_default_false(self) -> None:
        """Test that fail_on_error defaults to false."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.fail_on_error is False

    def test_evaluation_metric_fail_on_error_can_be_true(self) -> None:
        """Test that fail_on_error can be set to true."""
        metric = EvaluationMetric(
            metric="groundedness",
            fail_on_error=True,
        )
        assert metric.fail_on_error is True

    def test_evaluation_metric_retry_on_failure_optional(self) -> None:
        """Test that retry_on_failure is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.retry_on_failure is None

    def test_evaluation_metric_retry_on_failure_valid(self) -> None:
        """Test that retry_on_failure can be set."""
        metric = EvaluationMetric(
            metric="groundedness",
            retry_on_failure=2,
        )
        assert metric.retry_on_failure == 2

    def test_evaluation_metric_retry_on_failure_bounds(self) -> None:
        """Test that retry_on_failure has reasonable bounds."""
        # Should accept 1-3 retries
        metric = EvaluationMetric(
            metric="groundedness",
            retry_on_failure=3,
        )
        assert metric.retry_on_failure == 3

    def test_evaluation_metric_timeout_ms_optional(self) -> None:
        """Test that timeout_ms is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.timeout_ms is None

    def test_evaluation_metric_timeout_ms_valid(self) -> None:
        """Test that timeout_ms can be set."""
        metric = EvaluationMetric(
            metric="groundedness",
            timeout_ms=5000,
        )
        assert metric.timeout_ms == 5000

    def test_evaluation_metric_timeout_ms_positive(self) -> None:
        """Test that timeout_ms must be positive."""
        with pytest.raises(ValidationError):
            EvaluationMetric(
                metric="groundedness",
                timeout_ms=0,
            )

    def test_evaluation_metric_scale_optional(self) -> None:
        """Test that scale is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.scale is None

    def test_evaluation_metric_scale_valid(self) -> None:
        """Test that scale can be set."""
        metric = EvaluationMetric(
            metric="groundedness",
            scale=5,
        )
        assert metric.scale == 5

    def test_evaluation_metric_custom_prompt_optional(self) -> None:
        """Test that custom_prompt is optional."""
        metric = EvaluationMetric(metric="groundedness")
        assert metric.custom_prompt is None

    def test_evaluation_metric_custom_prompt_valid(self) -> None:
        """Test that custom_prompt can be set."""
        prompt = "Evaluate the response for groundedness"
        metric = EvaluationMetric(
            metric="groundedness",
            custom_prompt=prompt,
        )
        assert metric.custom_prompt == prompt

    def test_evaluation_metric_all_fields(self) -> None:
        """Test EvaluationMetric with all optional fields."""
        model = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        metric = EvaluationMetric(
            metric="groundedness",
            threshold=4.0,
            enabled=True,
            scale=5,
            model=model,
            fail_on_error=False,
            retry_on_failure=2,
            timeout_ms=5000,
            custom_prompt="Custom evaluation prompt",
        )
        assert metric.metric == "groundedness"
        assert metric.threshold == 4.0
        assert metric.enabled is True
        assert metric.scale == 5
        assert metric.model is not None
        assert metric.fail_on_error is False
        assert metric.retry_on_failure == 2
        assert metric.timeout_ms == 5000
        assert metric.custom_prompt == "Custom evaluation prompt"

    def test_evaluation_metric_various_metric_names(self) -> None:
        """Test EvaluationMetric accepts various metric names."""
        for metric_name in [
            "groundedness",
            "relevance",
            "coherence",
            "safety",
            "f1_score",
            "bleu",
            "rouge",
        ]:
            metric = EvaluationMetric(metric=metric_name)
            assert metric.metric == metric_name
