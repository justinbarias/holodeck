"""Tests for the legacy MetricResult.kind inference helper (T010d)."""

import logging

import pytest

from holodeck.lib.eval_run.legacy import infer_metric_kind
from holodeck.models.evaluation import RAGMetricType


@pytest.mark.unit
class TestInferMetricKind:
    """Lock the legacy classifier policy."""

    @pytest.mark.parametrize(
        "metric_name",
        ["bleu", "rouge", "meteor", "exact_match", "f1_score"],
    )
    def test_standard_nlp_names_classify_as_standard(self, metric_name: str) -> None:
        assert infer_metric_kind(metric_name) == "standard"

    def test_standard_classification_is_case_insensitive(self) -> None:
        assert infer_metric_kind("BLEU") == "standard"
        assert infer_metric_kind("Rouge") == "standard"

    @pytest.mark.parametrize("rag_type", list(RAGMetricType))
    def test_rag_metric_names_classify_as_rag(self, rag_type: RAGMetricType) -> None:
        assert infer_metric_kind(rag_type.value) == "rag"

    @pytest.mark.parametrize(
        "metric_name",
        ["Professionalism", "Helpfulness", "custom_criteria", "unknown_metric"],
    )
    def test_unknown_names_fall_back_to_geval(self, metric_name: str) -> None:
        assert infer_metric_kind(metric_name) == "geval"

    def test_inference_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Every inference call logs a WARNING — this is a fallback, not a
        happy path."""
        with caplog.at_level(logging.WARNING, logger="holodeck.lib.eval_run.legacy"):
            infer_metric_kind("bleu")
        assert any(
            "Inferred MetricResult.kind" in record.message for record in caplog.records
        )
