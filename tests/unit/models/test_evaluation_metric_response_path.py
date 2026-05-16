"""Validation tests for ``EvaluationMetric.response_path``.

Config-time validation should reject typo'd paths before any agent runs,
so authors don't lose a whole test pass to a syntax error in YAML.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from holodeck.models.evaluation import EvaluationMetric


@pytest.mark.unit
def test_response_path_default_is_none() -> None:
    metric = EvaluationMetric(metric="numeric")
    assert metric.response_path is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "answer",
        "result.value",
        "items[0].score",
        "data.items[12].nested.value",
    ],
)
def test_response_path_accepts_valid_grammar(path: str) -> None:
    metric = EvaluationMetric(metric="numeric", response_path=path)
    assert metric.response_path == path


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "",
        "   ",
        ".answer",
        "answer.",
        "answer..value",
        "1bad",
        "answer[a]",
        "answer[-1]",
        "$.answer",
    ],
)
def test_response_path_rejects_invalid_grammar(path: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        EvaluationMetric(metric="numeric", response_path=path)
    assert "response_path" in str(exc_info.value)


@pytest.mark.unit
def test_response_path_available_on_all_standard_metrics() -> None:
    """``response_path`` is generic across the standard family — not numeric-only."""
    for name in ("equality", "numeric", "bleu", "rouge", "meteor"):
        metric = EvaluationMetric(metric=name, response_path="answer")  # type: ignore[arg-type]
        assert metric.response_path == "answer"
