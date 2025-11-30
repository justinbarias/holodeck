"""Unit tests for GEvalMetric model and discriminated union behavior.

Tests cover:
- Valid GEvalMetric instantiation with various configurations
- Validation: criteria non-empty
- Validation: evaluation_params valid values
- Validation: threshold 0-1 range
- Validation: name non-empty
- Discriminator works correctly in Union
- EvaluationMetric rejects metric='geval'
- Serialization/deserialization round-trip
"""

import pytest
from pydantic import ValidationError

from holodeck.models.evaluation import (
    VALID_EVALUATION_PARAMS,
    EvaluationConfig,
    EvaluationMetric,
    GEvalMetric,
)
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestGEvalMetricInstantiation:
    """Tests for GEvalMetric model creation."""

    def test_minimal_valid_geval_metric(self):
        """GEvalMetric with only required fields creates successfully."""
        metric = GEvalMetric(
            name="Professionalism",
            criteria="Evaluate if the response uses professional language",
        )

        assert metric.type == "geval"
        assert metric.name == "Professionalism"
        assert metric.criteria == "Evaluate if the response uses professional language"
        assert metric.evaluation_params == ["actual_output"]  # Default
        assert metric.evaluation_steps is None
        assert metric.strict_mode is False
        assert metric.threshold is None
        assert metric.model is None
        assert metric.enabled is True
        assert metric.fail_on_error is False

    def test_full_geval_metric_config(self):
        """GEvalMetric with all fields specified creates successfully."""
        llm_model = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
            api_key="test-key",
        )

        metric = GEvalMetric(
            name="Helpfulness",
            criteria="Evaluate if the response is helpful and addresses the query",
            evaluation_steps=[
                "Check if the response addresses the main question",
                "Verify that all requested information is provided",
                "Assess the clarity of the explanation",
            ],
            evaluation_params=["input", "actual_output", "expected_output"],
            strict_mode=True,
            threshold=0.8,
            model=llm_model,
            enabled=True,
            fail_on_error=True,
        )

        assert metric.type == "geval"
        assert metric.name == "Helpfulness"
        assert len(metric.evaluation_steps) == 3
        assert metric.evaluation_params == ["input", "actual_output", "expected_output"]
        assert metric.strict_mode is True
        assert metric.threshold == 0.8
        assert metric.model == llm_model
        assert metric.enabled is True
        assert metric.fail_on_error is True

    def test_geval_metric_with_all_valid_params(self):
        """GEvalMetric accepts all valid evaluation_params."""
        metric = GEvalMetric(
            name="Comprehensive",
            criteria="Full evaluation criteria",
            evaluation_params=list(VALID_EVALUATION_PARAMS),
        )

        assert set(metric.evaluation_params) == VALID_EVALUATION_PARAMS


class TestGEvalMetricValidation:
    """Tests for GEvalMetric field validation."""

    def test_empty_name_rejected(self):
        """GEvalMetric rejects empty name."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="",
                criteria="Some criteria",
            )
        assert "name must be a non-empty string" in str(exc_info.value)

    def test_whitespace_only_name_rejected(self):
        """GEvalMetric rejects whitespace-only name."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="   ",
                criteria="Some criteria",
            )
        assert "name must be a non-empty string" in str(exc_info.value)

    def test_empty_criteria_rejected(self):
        """GEvalMetric rejects empty criteria."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="",
            )
        assert "criteria must be a non-empty string" in str(exc_info.value)

    def test_whitespace_only_criteria_rejected(self):
        """GEvalMetric rejects whitespace-only criteria."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="   \n\t  ",
            )
        assert "criteria must be a non-empty string" in str(exc_info.value)

    def test_invalid_evaluation_param_rejected(self):
        """GEvalMetric rejects invalid evaluation_params values."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="Test criteria",
                evaluation_params=["actual_output", "invalid_param"],
            )
        error_str = str(exc_info.value)
        assert "Invalid evaluation_params" in error_str
        assert "invalid_param" in error_str

    def test_empty_evaluation_params_rejected(self):
        """GEvalMetric rejects empty evaluation_params list."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="Test criteria",
                evaluation_params=[],
            )
        assert "evaluation_params must not be empty" in str(exc_info.value)

    def test_threshold_below_zero_rejected(self):
        """GEvalMetric rejects threshold below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="Test criteria",
                threshold=-0.1,
            )
        assert "threshold must be between 0.0 and 1.0" in str(exc_info.value)

    def test_threshold_above_one_rejected(self):
        """GEvalMetric rejects threshold above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            GEvalMetric(
                name="TestMetric",
                criteria="Test criteria",
                threshold=1.5,
            )
        assert "threshold must be between 0.0 and 1.0" in str(exc_info.value)

    def test_threshold_at_boundaries_accepted(self):
        """GEvalMetric accepts threshold at 0.0 and 1.0 boundaries."""
        metric_zero = GEvalMetric(
            name="TestMetric",
            criteria="Test criteria",
            threshold=0.0,
        )
        assert metric_zero.threshold == 0.0

        metric_one = GEvalMetric(
            name="TestMetric",
            criteria="Test criteria",
            threshold=1.0,
        )
        assert metric_one.threshold == 1.0


class TestEvaluationMetricTypeDiscriminator:
    """Tests for EvaluationMetric type discriminator."""

    def test_evaluation_metric_default_type(self):
        """EvaluationMetric has type='standard' by default."""
        metric = EvaluationMetric(metric="groundedness", threshold=0.7)
        assert metric.type == "standard"

    def test_evaluation_metric_accepts_various_names(self):
        """EvaluationMetric accepts various metric names."""
        for metric_name in ["groundedness", "relevance", "bleu", "custom_metric"]:
            metric = EvaluationMetric(metric=metric_name, threshold=0.7)
            assert metric.metric == metric_name
            assert metric.type == "standard"


class TestDiscriminatedUnionBehavior:
    """Tests for discriminated union type detection in EvaluationConfig."""

    def test_config_with_standard_metric(self):
        """EvaluationConfig correctly parses standard EvaluationMetric."""
        config = EvaluationConfig(
            metrics=[
                {"type": "standard", "metric": "groundedness", "threshold": 0.7},
                {"type": "standard", "metric": "bleu", "threshold": 0.6},
            ]
        )

        assert len(config.metrics) == 2
        assert all(isinstance(m, EvaluationMetric) for m in config.metrics)
        assert config.metrics[0].metric == "groundedness"
        assert config.metrics[1].metric == "bleu"

    def test_config_with_standard_metric_requires_type(self):
        """EvaluationConfig requires 'type' field for discriminated union."""
        # For discriminated unions, the discriminator field must be present
        # when parsing from dict. The default only applies when creating
        # the model directly (not from dict).
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(
                metrics=[
                    {"metric": "bleu", "threshold": 0.6},
                ]
            )
        assert "union_tag_not_found" in str(exc_info.value)

    def test_config_model_instance_uses_default_type(self):
        """When creating model instances directly, type defaults to 'standard'."""
        metric = EvaluationMetric(metric="bleu", threshold=0.6)
        assert metric.type == "standard"

        # Can use model instance in config
        config = EvaluationConfig(metrics=[metric])
        assert len(config.metrics) == 1
        assert isinstance(config.metrics[0], EvaluationMetric)

    def test_config_with_geval_metric(self):
        """EvaluationConfig correctly parses GEvalMetric."""
        config = EvaluationConfig(
            metrics=[
                {
                    "type": "geval",
                    "name": "Professionalism",
                    "criteria": "Evaluate professional language",
                    "threshold": 0.8,
                },
            ]
        )

        assert len(config.metrics) == 1
        assert isinstance(config.metrics[0], GEvalMetric)
        assert config.metrics[0].name == "Professionalism"
        assert config.metrics[0].criteria == "Evaluate professional language"

    def test_config_with_mixed_metrics(self):
        """EvaluationConfig correctly handles mixed metric types."""
        config = EvaluationConfig(
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                api_key="test-key",
            ),
            metrics=[
                {"type": "standard", "metric": "bleu", "threshold": 0.6},
                {
                    "type": "geval",
                    "name": "Helpfulness",
                    "criteria": "Is the response helpful?",
                },
                {"type": "standard", "metric": "rouge", "threshold": 0.7},
                {
                    "type": "geval",
                    "name": "Accuracy",
                    "criteria": "Is the response accurate?",
                    "strict_mode": True,
                },
            ],
        )

        assert len(config.metrics) == 4

        # Check types are correctly identified
        assert isinstance(config.metrics[0], EvaluationMetric)
        assert isinstance(config.metrics[1], GEvalMetric)
        assert isinstance(config.metrics[2], EvaluationMetric)
        assert isinstance(config.metrics[3], GEvalMetric)

        # Check values
        assert config.metrics[0].metric == "bleu"
        assert config.metrics[1].name == "Helpfulness"
        assert config.metrics[2].metric == "rouge"
        assert config.metrics[3].name == "Accuracy"
        assert config.metrics[3].strict_mode is True


class TestSerializationRoundTrip:
    """Tests for serialization/deserialization behavior."""

    def test_geval_metric_dict_roundtrip(self):
        """GEvalMetric serializes and deserializes correctly."""
        original = GEvalMetric(
            name="Coherence",
            criteria="Evaluate response coherence",
            evaluation_steps=["Step 1", "Step 2"],
            evaluation_params=["input", "actual_output"],
            strict_mode=True,
            threshold=0.75,
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = GEvalMetric(**data)

        assert restored.name == original.name
        assert restored.criteria == original.criteria
        assert restored.evaluation_steps == original.evaluation_steps
        assert restored.evaluation_params == original.evaluation_params
        assert restored.strict_mode == original.strict_mode
        assert restored.threshold == original.threshold

    def test_evaluation_config_dict_roundtrip(self):
        """EvaluationConfig with mixed metrics serializes and deserializes."""
        original = EvaluationConfig(
            metrics=[
                EvaluationMetric(metric="bleu", threshold=0.6),
                GEvalMetric(
                    name="Test",
                    criteria="Test criteria",
                    threshold=0.7,
                ),
            ]
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = EvaluationConfig(**data)

        assert len(restored.metrics) == 2
        assert isinstance(restored.metrics[0], EvaluationMetric)
        assert isinstance(restored.metrics[1], GEvalMetric)
        assert restored.metrics[0].metric == "bleu"
        assert restored.metrics[1].name == "Test"

    def test_geval_metric_json_roundtrip(self):
        """GEvalMetric serializes to JSON and back correctly."""
        original = GEvalMetric(
            name="JSONTest",
            criteria="JSON serialization test",
            threshold=0.5,
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        restored = GEvalMetric.model_validate_json(json_str)

        assert restored.name == original.name
        assert restored.criteria == original.criteria
        assert restored.threshold == original.threshold


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_geval_with_multiline_criteria(self):
        """GEvalMetric accepts multiline criteria strings."""
        metric = GEvalMetric(
            name="Detailed",
            criteria="""
            Evaluate the response based on:
            1. Accuracy of information
            2. Clarity of explanation
            3. Completeness of answer
            """,
        )

        assert "Accuracy" in metric.criteria
        assert "Clarity" in metric.criteria
        assert "Completeness" in metric.criteria

    def test_geval_with_special_characters_in_name(self):
        """GEvalMetric accepts names with special characters."""
        metric = GEvalMetric(
            name="Test-Metric_2024 (v1)",
            criteria="Test criteria",
        )

        assert metric.name == "Test-Metric_2024 (v1)"

    def test_extra_fields_rejected(self):
        """GEvalMetric rejects extra fields (extra='forbid')."""
        with pytest.raises(ValidationError):
            GEvalMetric(
                name="Test",
                criteria="Test criteria",
                unknown_field="should fail",
            )

    def test_disabled_geval_metric(self):
        """GEvalMetric can be disabled."""
        metric = GEvalMetric(
            name="Disabled",
            criteria="This metric is disabled",
            enabled=False,
        )

        assert metric.enabled is False
