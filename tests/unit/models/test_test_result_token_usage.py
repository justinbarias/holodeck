"""Tests for TestResult.token_usage (T010k).

Dashboard Compare view computes cost from persisted token counts. Locks the
default-None shape and the round-trip contract for legacy JSON files that
were written before this field existed.
"""

import pytest

from holodeck.models.test_result import TestResult
from holodeck.models.token_usage import TokenUsage


@pytest.mark.unit
class TestTestResultTokenUsage:
    """Lock the TestResult.token_usage contract."""

    def test_field_exists_with_default_none(self) -> None:
        assert "token_usage" in TestResult.model_fields
        field = TestResult.model_fields["token_usage"]
        assert field.default is None
        assert not field.is_required()

    def test_legacy_json_without_field_loads_with_none(self) -> None:
        """A JSON blob produced before this field existed must still load."""
        legacy = (
            '{"test_input":"hi","passed":true,"execution_time_ms":1,'
            '"timestamp":"2026-04-18T00:00:00Z"}'
        )
        result = TestResult.model_validate_json(legacy)
        assert result.token_usage is None

    def test_round_trip_preserves_populated_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        original = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-18T00:00:00Z",
            token_usage=usage,
        )
        rehydrated = TestResult.model_validate_json(original.model_dump_json())
        assert rehydrated.token_usage == usage

    def test_round_trip_preserves_none(self) -> None:
        original = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-18T00:00:00Z",
        )
        rehydrated = TestResult.model_validate_json(original.model_dump_json())
        assert rehydrated.token_usage is None
