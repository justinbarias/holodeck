"""Scalarized objective for the optimizer.

Collapses a ``TestReport`` into a single *loss* the optimizer minimizes:
``1 − weighted_mean`` of per-metric averages using the configured ``loss``
weights. Metric averages are required to lie in ``[0, 1]`` so the inverted
value is a well-defined loss; a metric that reports outside that range (e.g. a
raw token count) raises rather than silently producing a negative or >1 loss.

Per-metric averages are computed directly from the granular
``TestResult.metric_results`` rather than ``summary.average_scores``: the
executor's summary drops any score that equals ``0.0`` (a falsy-filter quirk),
which would make a fully-failing baseline unscorable. Errored metric runs
(``MetricResult.error`` set) are excluded from the average and flagged — not
scored as zero (spec decision 5) — while legitimate ``0.0`` scores are kept.
"""

import logging
from collections import defaultdict

from holodeck.lib.errors import OptimizerError
from holodeck.models.test_result import TestReport

logger = logging.getLogger(__name__)


def _metric_averages(report: TestReport) -> dict[str, float]:
    """Average each metric over its non-errored runs across all test cases."""
    scores: dict[str, list[float]] = defaultdict(list)
    for result in report.results:
        for metric in result.metric_results:
            if metric.error is None:
                scores[metric.metric_name].append(metric.score)
    return {name: sum(vals) / len(vals) for name, vals in scores.items() if vals}


def excluded_metrics(report: TestReport, loss_weights: dict[str, float]) -> list[str]:
    """Return weighted metrics that have no non-errored score in ``report``.

    Args:
        report: Completed test report.
        loss_weights: Per-metric weights for the scalarized objective.

    Returns:
        Names of weighted metrics with no usable score, preserving the order
        they appear in ``loss_weights``.
    """
    averages = _metric_averages(report)
    return [metric for metric in loss_weights if metric not in averages]


def scalarize(report: TestReport, loss_weights: dict[str, float]) -> float:
    """Compute the scalarized loss (lower is better) for a test report.

    Computes the renormalized weighted mean of the contributing metric averages
    and returns ``1 − mean`` so the optimizer can *minimize* it. Only metrics
    with at least one non-errored run contribute; the weights of contributing
    metrics are renormalized so excluding an errored metric does not bias the
    loss.

    Args:
        report: Completed test report.
        loss_weights: Per-metric weights (strictly positive).

    Returns:
        ``1 − weighted_mean`` of the contributing metric averages, in ``[0, 1]``.

    Raises:
        OptimizerError: If none of the weighted metrics produced a usable score,
            or a contributing metric average falls outside ``[0, 1]`` (so the
            inverted value would not be a valid loss).
    """
    averages = _metric_averages(report)
    missing = [metric for metric in loss_weights if metric not in averages]
    if missing:
        logger.warning(
            "Excluding metrics with no non-errored score from the objective: %s",
            ", ".join(missing),
        )

    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in loss_weights.items():
        if metric in averages:
            average = averages[metric]
            if not 0.0 <= average <= 1.0:
                raise OptimizerError(
                    f"Metric '{metric}' average {average} is outside [0, 1]; "
                    "the optimizer's loss (1 − weighted_mean) is only defined "
                    "for metrics normalized to [0, 1]."
                )
            weighted_sum += weight * average
            total_weight += weight

    if total_weight == 0.0:
        raise OptimizerError(
            "No weighted metric produced a usable score; "
            "cannot scalarize the objective. "
            f"Weighted metrics: {sorted(loss_weights)}; "
            f"scored metrics: {sorted(averages)}."
        )

    return 1.0 - weighted_sum / total_weight
