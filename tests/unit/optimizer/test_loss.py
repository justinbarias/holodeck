"""Unit tests for loss.scalarize (T2, revised in T9)."""

import pytest

from holodeck.lib.errors import OptimizerError
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
)
from holodeck.optimizer.loss import excluded_metrics, scalarize


def _report(metrics: list[tuple[str, float, str | None]]) -> TestReport:
    """Build a single-result report carrying the given metric results.

    Args:
        metrics: list of (metric_name, score, error) tuples.
    """
    metric_results = [
        MetricResult(metric_name=name, kind="standard", score=score, error=error)
        for name, score, error in metrics
    ]
    result = TestResult(
        test_name="t0",
        test_input="q",
        agent_response="a",
        passed=all(error is None for _, _, error in metrics),
        execution_time_ms=1,
        timestamp="2026-05-31T00:00:00Z",
        metric_results=metric_results,
    )
    summary = ReportSummary(
        total_tests=1,
        passed=1,
        failed=0,
        pass_rate=100.0,
        total_duration_ms=1,
        metrics_evaluated={},
        average_scores={},
    )
    return TestReport(
        agent_name="a",
        agent_config_path="agent.yaml",
        results=[result],
        summary=summary,
        timestamp="2026-05-31T00:00:00Z",
        holodeck_version="test",
    )


class TestScalarize:
    """scalarize returns 1 − weighted-mean of per-metric averages (a loss)."""

    def test_matches_hand_computed_weighted_mean(self) -> None:
        report = _report([("groundedness", 0.8, None), ("relevance", 0.6, None)])
        weights = {"groundedness": 1.0, "relevance": 3.0}
        # mean = (1.0*0.8 + 3.0*0.6) / (1.0 + 3.0) = 2.6 / 4 = 0.65 → loss 0.35
        assert scalarize(report, weights) == pytest.approx(0.35)

    def test_single_metric(self) -> None:
        report = _report([("groundedness", 0.42, None)])
        # loss = 1 − 0.42
        assert scalarize(report, {"groundedness": 2.0}) == pytest.approx(0.58)

    def test_perfect_score_is_zero_loss(self) -> None:
        # A fully-passing report has zero loss.
        report = _report([("equality", 1.0, None)])
        assert scalarize(report, {"equality": 1.0}) == pytest.approx(0.0)

    def test_zero_score_is_max_loss_not_dropped(self) -> None:
        # A legitimate 0.0 score must contribute (a fully-failing baseline) → loss 1.
        report = _report([("equality", 0.0, None)])
        assert scalarize(report, {"equality": 1.0}) == pytest.approx(1.0)

    def test_errored_metric_excluded_and_renormalized(self) -> None:
        report = _report([("groundedness", 0.8, None), ("relevance", 0.5, "boom")])
        weights = {"groundedness": 1.0, "relevance": 2.0}
        # 'relevance' errored → excluded; only groundedness contributes → loss 0.2.
        assert scalarize(report, weights) == pytest.approx(0.2)

    def test_excluded_metrics_flags_errored(self) -> None:
        report = _report([("groundedness", 0.8, None), ("relevance", 0.5, "boom")])
        weights = {"groundedness": 1.0, "relevance": 2.0}
        assert excluded_metrics(report, weights) == ["relevance"]

    def test_no_scored_metric_raises(self) -> None:
        report = _report([("groundedness", 0.5, "errored")])
        with pytest.raises(OptimizerError):
            scalarize(report, {"groundedness": 1.0})

    def test_metric_average_above_one_raises(self) -> None:
        # A metric outside [0, 1] makes 1 − mean an invalid loss → reject.
        report = _report([("tokens", 12.0, None)])
        with pytest.raises(OptimizerError, match="outside"):
            scalarize(report, {"tokens": 1.0})

    def test_metric_average_below_zero_raises(self) -> None:
        report = _report([("signed", -0.5, None)])
        with pytest.raises(OptimizerError, match="outside"):
            scalarize(report, {"signed": 1.0})
