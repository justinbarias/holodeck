"""Back-compat helpers for loading pre-Phase-2b EvalRun JSON files.

Older `EvalRun` artifacts produced before `MetricResult.kind` became required
are loadable if the reader imputes a `kind` before Pydantic validation.
`infer_metric_kind` is the one-line classifier the reader should use.
"""

from __future__ import annotations

import logging
from typing import Literal

from holodeck.models.evaluation import RAGMetricType

logger = logging.getLogger(__name__)

_STANDARD_METRIC_NAMES: frozenset[str] = frozenset(
    {"bleu", "rouge", "meteor", "exact_match", "f1_score"}
)

_RAG_METRIC_NAMES: frozenset[str] = frozenset(m.value for m in RAGMetricType)


def infer_metric_kind(
    metric_name: str,
) -> Literal["standard", "rag", "geval"]:
    """Infer a `MetricResult.kind` for a legacy run lacking the field.

    Policy:
    - Names in `{bleu, rouge, meteor, exact_match, f1_score}` → `"standard"`
    - Names matching `RAGMetricType` enum values → `"rag"`
    - Everything else (custom LLM-as-judge criteria) → `"geval"`

    A WARNING is logged on every call because inference is a fallback;
    callers should migrate to running `holodeck test` again to get a
    correctly-typed run.

    Args:
        metric_name: The `metric_name` field on a legacy `MetricResult`.

    Returns:
        The inferred kind literal.
    """
    key = metric_name.strip().lower()
    if key in _STANDARD_METRIC_NAMES:
        kind: Literal["standard", "rag", "geval"] = "standard"
    elif key in _RAG_METRIC_NAMES:
        kind = "rag"
    else:
        kind = "geval"

    logger.warning(
        "Inferred MetricResult.kind=%r for legacy metric_name=%r; "
        "re-run `holodeck test` to produce a correctly-typed EvalRun.",
        kind,
        metric_name,
    )
    return kind
