"""Tests for MetricResult.kind extended with "code" (T008).

data-model.md §7a extends kind to include the "code" literal so the
dashboard can bucket code-grader results separately.
"""

from typing import get_args

import pytest

from holodeck.lib.test_runner.executor import _metric_kind
from holodeck.models.evaluation import CodeMetric
from holodeck.models.test_result import MetricResult


@pytest.mark.unit
class TestMetricResultKindCode:
    def test_kind_annotation_includes_code(self) -> None:
        annotation = MetricResult.model_fields["kind"].annotation
        assert annotation is not None
        assert set(get_args(annotation)) == {"standard", "rag", "geval", "code"}

    def test_code_kind_round_trip(self) -> None:
        original = MetricResult(
            metric_name="x",
            kind="code",
            score=1.0,
            threshold=None,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=0,
            model_used=None,
            reasoning=None,
        )
        rehydrated = MetricResult.model_validate_json(original.model_dump_json())
        assert rehydrated == original

    def test_metric_kind_helper_returns_code(self) -> None:
        cm = CodeMetric(grader="some.module:fn")
        assert _metric_kind(cm) == "code"

    @pytest.mark.parametrize("kind", ["standard", "rag", "geval"])
    def test_legacy_kinds_still_parse(self, kind: str) -> None:
        original = MetricResult(
            metric_name="legacy",
            kind=kind,  # type: ignore[arg-type]
            score=0.5,
            threshold=0.5,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=1,
            model_used=None,
            reasoning=None,
        )
        rehydrated = MetricResult.model_validate_json(original.model_dump_json())
        assert rehydrated.kind == kind
