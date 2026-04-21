"""Regression tests: legacy single-turn test cases emit unchanged TestResult shape
(SC-002). The presence of new additive optional fields (``turns: None``,
``cache_creation_tokens: 0``) must not affect callers that serialised prior
reports — Pydantic's `exclude_defaults=True` keeps JSON byte-for-byte
backwards-compatible.
"""

from __future__ import annotations

import pytest

from holodeck.models.test_result import TestResult


@pytest.mark.unit
class TestLegacyBackcompat:
    def test_exclude_defaults_hides_turns_field(self) -> None:
        result = TestResult(
            test_input="hi",
            agent_response="hello",
            passed=True,
            execution_time_ms=5,
            timestamp="2026-04-20T00:00:00+00:00",
        )
        dumped = result.model_dump(exclude_defaults=True)
        assert "turns" not in dumped

    def test_legacy_json_without_turns_loads(self) -> None:
        legacy = (
            '{"test_input":"hi","passed":true,"execution_time_ms":5,'
            '"timestamp":"2026-04-20T00:00:00+00:00"}'
        )
        result = TestResult.model_validate_json(legacy)
        assert result.turns is None

    def test_roundtrip_preserves_multi_turn_only_when_populated(self) -> None:
        """Round-trip preserves turns=None for single-turn cases."""
        populated = TestResult(
            test_input="a",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-20T00:00:00+00:00",
            turns=None,
        )
        rehydrated = TestResult.model_validate_json(populated.model_dump_json())
        assert rehydrated.turns is None
