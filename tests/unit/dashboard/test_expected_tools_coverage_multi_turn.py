"""Tests for multi-turn expected-tools coverage in the Explorer payload."""

from __future__ import annotations

import pytest

from holodeck.dashboard.explorer_data import (
    _build_expected_tools_coverage,
    _turn_view_from_result,
)
from holodeck.models.test_result import TestResult, TurnResult


def _turn(idx: int, expected: list[str], actual: list[str]) -> TurnResult:
    return TurnResult(
        turn_index=idx,
        input=f"q{idx}",
        response=f"r{idx}",
        tool_calls=actual,
        expected_tools=expected,
        passed=True,
        execution_time_ms=5,
    )


@pytest.mark.unit
def test_multi_turn_coverage_aggregates_per_turn_with_substring_match() -> None:
    """Multi-turn rolls up each turn's expected_tools using substring match."""
    turns = [
        _turn(
            0,
            expected=["legislation_search"],
            actual=["mcp__holodeck_tools__legislation_search_search"],
        ),
        _turn(1, expected=["legislation_search"], actual=["format_citation"]),
        _turn(
            2,
            expected=["format_citation"],
            actual=["mcp__holodeck_tools__format_citation"],
        ),
    ]
    case = TestResult(
        test_name="t",
        test_input="x",
        agent_response="r",
        passed=False,
        execution_time_ms=1,
        timestamp="2026-04-21T00:00:00+00:00",
        turns=turns,
    )

    cov = _build_expected_tools_coverage(case)

    assert cov.total == 3
    assert cov.matched == 2
    assert cov.missed == 1
    assert len(cov.per_turn) == 3
    assert cov.per_turn[0].expected == [("legislation_search", True)]
    assert cov.per_turn[1].expected == [("legislation_search", False)]
    assert cov.per_turn[2].expected == [("format_citation", True)]


@pytest.mark.unit
def test_single_turn_coverage_uses_substring_match() -> None:
    """Single-turn path also matches via substring so MCP names resolve."""
    case = TestResult(
        test_name="t",
        test_input="x",
        agent_response="r",
        passed=True,
        execution_time_ms=1,
        timestamp="2026-04-21T00:00:00+00:00",
        expected_tools=["legislation_search"],
        tool_calls=["mcp__holodeck_tools__legislation_search_search"],
    )

    cov = _build_expected_tools_coverage(case)

    assert cov.total == 1
    assert cov.matched == 1
    assert cov.rows == [("legislation_search", True)]
    assert cov.per_turn == []


@pytest.mark.unit
def test_turn_view_carries_expected_and_actual_tools() -> None:
    turn = _turn(
        0,
        expected=["legislation_search"],
        actual=["mcp__holodeck_tools__legislation_search_search"],
    )

    view = _turn_view_from_result(turn)

    assert view.expected_tools == ["legislation_search"]
    assert view.tool_calls == ["mcp__holodeck_tools__legislation_search_search"]
