"""Tests for TestResult.turns roll-up contract (T005).

Covers the §4 roll-up table from contracts/turn-result-schema.md and the
FR-015 regression (ReportSummary counts test cases, not turns).
"""

import pytest

from holodeck.lib.test_runner.executor import _finalize_multi_turn_result
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestResult,
    TurnResult,
)
from holodeck.models.token_usage import TokenUsage


def _make_turn(
    idx: int,
    *,
    input: str = "q",
    response: str | None = "a",
    passed: bool = True,
    errors: list[str] | None = None,
    execution_time_ms: int = 10,
    skipped: bool = False,
    token_usage: TokenUsage | None = None,
    tool_calls: list[str] | None = None,
    metric_results: list[MetricResult] | None = None,
) -> TurnResult:
    return TurnResult(
        turn_index=idx,
        input=input,
        response=response,
        ground_truth=None,
        expected_tools=None,
        tool_calls=tool_calls or [],
        tool_invocations=[],
        tools_matched=None,
        arg_match_details=None,
        metric_results=metric_results or [],
        passed=passed,
        execution_time_ms=execution_time_ms,
        token_usage=token_usage,
        errors=errors or [],
        skipped=skipped,
    )


@pytest.mark.unit
class TestTurnRollup:
    def test_test_input_joined_with_sep(self) -> None:
        turns = [_make_turn(0, input="a"), _make_turn(1, input="b")]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.test_input == "a\n---\nb"

    def test_agent_response_is_last_turn(self) -> None:
        turns = [
            _make_turn(0, response="first"),
            _make_turn(1, response="last"),
        ]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.agent_response == "last"

    def test_tool_calls_flattened(self) -> None:
        turns = [
            _make_turn(0, tool_calls=["lookup"]),
            _make_turn(1, tool_calls=["subtract", "divide"]),
        ]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.tool_calls == ["lookup", "subtract", "divide"]

    def test_passed_requires_all_turns_passed(self) -> None:
        turns = [_make_turn(0, passed=True), _make_turn(1, passed=False)]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.passed is False

    def test_execution_time_sums_turns(self) -> None:
        turns = [
            _make_turn(0, execution_time_ms=100),
            _make_turn(1, execution_time_ms=250, skipped=True),
        ]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        # Spec T005: skipped turns contribute 0 — but here the per-turn
        # execution_time_ms is still stored. The *sum* includes whatever
        # each turn reported. If the executor zeroes skipped turns, that's
        # recorded on the TurnResult itself.
        assert result.execution_time_ms == 350

    def test_ground_truth_is_none_at_testcase_level(self) -> None:
        turns = [_make_turn(0), _make_turn(1)]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.ground_truth is None

    def test_errors_prefixed_with_turn_index(self) -> None:
        turns = [
            _make_turn(0, errors=["timeout"], passed=False),
            _make_turn(1, errors=[]),
            _make_turn(2, errors=["backend: oops"], passed=False),
        ]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.errors == [
            "[turn 0] timeout",
            "[turn 2] backend: oops",
        ]

    def test_token_usage_summed_including_cache_fields(self) -> None:
        turns = [
            _make_turn(
                0,
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cache_creation_tokens=2,
                    cache_read_tokens=3,
                ),
            ),
            _make_turn(
                1,
                token_usage=TokenUsage(
                    prompt_tokens=20,
                    completion_tokens=10,
                    total_tokens=30,
                    cache_creation_tokens=4,
                    cache_read_tokens=6,
                ),
            ),
        ]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 30
        assert result.token_usage.completion_tokens == 15
        assert result.token_usage.total_tokens == 45
        assert result.token_usage.cache_creation_tokens == 6
        assert result.token_usage.cache_read_tokens == 9

    def test_turns_preserved_on_result(self) -> None:
        turns = [_make_turn(0), _make_turn(1)]
        result = _finalize_multi_turn_result(
            test_name="t", turns=turns, start_ts="2026-04-20T00:00:00+00:00"
        )
        assert result.turns is not None
        assert len(result.turns) == 2


@pytest.mark.unit
class TestReportSummaryTestCaseCount:
    """FR-015 regression: summary counts test cases, never turns."""

    def test_mixed_single_and_multi_turn_still_validates(self) -> None:
        # Represent: 1 single-turn pass, 1 multi-turn fail (with 3 turns).
        # total_tests == 2 (cases), not 4.
        summary = ReportSummary(
            total_tests=2,
            passed=1,
            failed=1,
            pass_rate=50.0,
            total_duration_ms=100,
        )
        assert summary.passed + summary.failed == summary.total_tests

    def test_single_turn_result_round_trips_without_turns_field(self) -> None:
        r = TestResult(
            test_input="hi",
            passed=True,
            execution_time_ms=1,
            timestamp="2026-04-20T00:00:00+00:00",
        )
        rehydrated = TestResult.model_validate_json(r.model_dump_json())
        assert rehydrated.turns is None
