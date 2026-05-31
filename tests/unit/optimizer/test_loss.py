"""Unit tests for loss.scalarize (T2)."""

import pytest

from holodeck.lib.errors import OptimizerError
from holodeck.models.test_result import ReportSummary, TestReport, TestResult
from holodeck.optimizer.loss import excluded_metrics, scalarize


def _report(average_scores: dict[str, float], n_results: int = 2) -> TestReport:
    """Build a minimal TestReport carrying the given average_scores."""
    results = [
        TestResult(
            test_name=f"t{i}",
            test_input="q",
            agent_response="a",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-05-31T00:00:00Z",
        )
        for i in range(n_results)
    ]
    summary = ReportSummary(
        total_tests=n_results,
        passed=n_results,
        failed=0,
        pass_rate=100.0,
        total_duration_ms=n_results,
        metrics_evaluated=dict.fromkeys(average_scores, n_results),
        average_scores=average_scores,
    )
    return TestReport(
        agent_name="a",
        agent_config_path="agent.yaml",
        results=results,
        summary=summary,
        timestamp="2026-05-31T00:00:00Z",
        holodeck_version="test",
    )


class TestScalarize:
    """scalarize computes a weighted mean of summary.average_scores."""

    def test_matches_hand_computed_weighted_mean(self) -> None:
        report = _report({"groundedness": 0.8, "relevance": 0.6})
        weights = {"groundedness": 1.0, "relevance": 3.0}
        # (1.0*0.8 + 3.0*0.6) / (1.0 + 3.0) = 2.6 / 4 = 0.65
        assert scalarize(report, weights) == pytest.approx(0.65)

    def test_single_metric(self) -> None:
        report = _report({"groundedness": 0.42})
        assert scalarize(report, {"groundedness": 2.0}) == pytest.approx(0.42)

    def test_errored_metric_excluded_and_renormalized(self) -> None:
        # 'relevance' is absent from average_scores (its runs errored).
        report = _report({"groundedness": 0.8})
        weights = {"groundedness": 1.0, "relevance": 2.0}
        # Only groundedness contributes; weight renormalized to itself.
        assert scalarize(report, weights) == pytest.approx(0.8)

    def test_excluded_metrics_flags_missing(self) -> None:
        report = _report({"groundedness": 0.8})
        weights = {"groundedness": 1.0, "relevance": 2.0}
        assert excluded_metrics(report, weights) == ["relevance"]

    def test_no_scored_metric_raises(self) -> None:
        report = _report({})
        with pytest.raises(OptimizerError):
            scalarize(report, {"groundedness": 1.0})
