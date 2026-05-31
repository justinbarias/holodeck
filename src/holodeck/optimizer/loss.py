"""Scalarized objective for the optimizer.

Collapses a ``TestReport`` into a single number the optimizer maximizes: a
weighted mean of ``summary.average_scores`` using the configured ``loss``
weights. Metrics named in the weights but absent from ``average_scores`` (their
runs errored everywhere, or were not evaluated) are excluded from the mean and
flagged rather than scored as zero.
"""

import logging

from holodeck.lib.errors import OptimizerError
from holodeck.models.test_result import TestReport

logger = logging.getLogger(__name__)


def excluded_metrics(report: TestReport, loss_weights: dict[str, float]) -> list[str]:
    """Return weighted metrics that have no aggregate score in ``report``.

    Args:
        report: Completed test report.
        loss_weights: Per-metric weights for the scalarized objective.

    Returns:
        Names of weighted metrics absent from ``summary.average_scores``,
        preserving the order they appear in ``loss_weights``.
    """
    scores = report.summary.average_scores
    return [metric for metric in loss_weights if metric not in scores]


def scalarize(report: TestReport, loss_weights: dict[str, float]) -> float:
    """Compute the weighted-mean objective for a test report.

    Only metrics present in ``summary.average_scores`` contribute; the weights
    of contributing metrics are renormalized so excluding an errored metric does
    not deflate the score toward zero.

    Args:
        report: Completed test report.
        loss_weights: Per-metric weights (strictly positive).

    Returns:
        The renormalized weighted mean of the contributing metric scores.

    Raises:
        OptimizerError: If none of the weighted metrics have an aggregate score.
    """
    scores = report.summary.average_scores
    missing = excluded_metrics(report, loss_weights)
    if missing:
        logger.warning(
            "Excluding metrics with no aggregate score from the objective: %s",
            ", ".join(missing),
        )

    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in loss_weights.items():
        if metric in scores:
            weighted_sum += weight * scores[metric]
            total_weight += weight

    if total_weight == 0.0:
        raise OptimizerError(
            "No weighted metric produced an aggregate score; "
            "cannot scalarize the objective. "
            f"Weighted metrics: {sorted(loss_weights)}; "
            f"scored metrics: {sorted(scores)}."
        )

    return weighted_sum / total_weight
