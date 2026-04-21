"""Tests for the MetricResult.kind discriminator (T010a).

The dashboard's Summary, Explorer, and Compare views partition metric results
by `kind` (standard / rag / geval). This test locks the field's shape so
future edits do not silently break the dashboard projection.
"""

from typing import Literal, get_args

import pytest
from pydantic import ValidationError

from holodeck.models.test_result import MetricResult


@pytest.mark.unit
class TestMetricResultKind:
    """Lock the MetricResult.kind contract consumed by US4/US5 dashboards."""

    def test_kind_field_annotation_is_literal_of_four_values(self) -> None:
        """Kind must be Literal["standard","rag","geval","code"] after feature 032."""
        annotation = MetricResult.model_fields["kind"].annotation
        assert annotation is not None
        assert set(get_args(annotation)) == {"standard", "rag", "geval", "code"}

    def test_kind_is_required(self) -> None:
        """Kind has no default — callers must supply it explicitly."""
        assert MetricResult.model_fields["kind"].is_required()

        with pytest.raises(ValidationError) as excinfo:
            MetricResult(
                metric_name="bleu",
                score=0.5,
                threshold=0.4,
                passed=True,
                scale="0-1",
                error=None,
                retry_count=0,
                evaluation_time_ms=10,
                model_used=None,
                reasoning=None,
            )
        assert "kind" in str(excinfo.value)

    @pytest.mark.parametrize("kind", ["standard", "rag", "geval"])
    def test_round_trip_preserves_kind(
        self, kind: Literal["standard", "rag", "geval"]
    ) -> None:
        """model_dump_json → model_validate_json round-trip preserves kind."""
        original = MetricResult(
            metric_name="example",
            score=0.9,
            threshold=0.8,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=5,
            model_used=None,
            reasoning=None,
            kind=kind,
        )
        rehydrated = MetricResult.model_validate_json(original.model_dump_json())
        assert rehydrated == original
        assert rehydrated.kind == kind

    def test_invalid_kind_rejected(self) -> None:
        """Values outside the literal set must fail validation."""
        with pytest.raises(ValidationError):
            MetricResult(
                metric_name="example",
                score=0.5,
                threshold=0.5,
                passed=True,
                scale="0-1",
                error=None,
                retry_count=0,
                evaluation_time_ms=1,
                model_used=None,
                reasoning=None,
                kind="unknown",  # type: ignore[arg-type]
            )
