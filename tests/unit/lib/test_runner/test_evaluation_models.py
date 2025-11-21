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

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("threshold", 3.5, 3.5),
            ("threshold", 4.0, 4.0),
            ("enabled", True, True),
            ("enabled", False, False),
            ("fail_on_error", True, True),
            ("fail_on_error", False, False),
            ("retry_on_failure", 2, 2),
            ("retry_on_failure", 3, 3),
            ("timeout_ms", 5000, 5000),
            ("scale", 5, 5),
            ("custom_prompt", "Evaluate the response", "Evaluate the response"),
        ],
        ids=[
            "threshold_float",
            "threshold_int",
            "enabled_true",
            "enabled_false",
            "fail_on_error_true",
            "fail_on_error_false",
            "retry_on_failure_2",
            "retry_on_failure_3",
            "timeout_ms_5000",
            "scale_5",
            "custom_prompt",
        ],
    )
    def test_evaluation_metric_optional_fields(
        self, field: str, value, expected
    ) -> None:
        """Test optional fields can be set with valid values."""
        kwargs = {"metric": "groundedness", field: value}
        metric = EvaluationMetric(**kwargs)
        assert getattr(metric, field) == expected

    @pytest.mark.parametrize(
        "field,default",
        [
            ("threshold", None),
            ("enabled", True),
            ("model", None),
            ("fail_on_error", False),
            ("retry_on_failure", None),
            ("timeout_ms", None),
            ("scale", None),
            ("custom_prompt", None),
        ],
        ids=[
            "threshold",
            "enabled",
            "model",
            "fail_on_error",
            "retry_on_failure",
            "timeout_ms",
            "scale",
            "custom_prompt",
        ],
    )
    def test_evaluation_metric_field_defaults(self, field: str, default) -> None:
        """Test that optional fields have correct defaults."""
        metric = EvaluationMetric(metric="groundedness")
        assert getattr(metric, field) == default

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

    def test_evaluation_metric_timeout_ms_positive(self) -> None:
        """Test that timeout_ms must be positive."""
        with pytest.raises(ValidationError):
            EvaluationMetric(
                metric="groundedness",
                timeout_ms=0,
            )

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
