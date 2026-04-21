"""Tests for tool_arg_matcher — SC-006 23-row acceptance matrix (US3 T015–T020).

Authoritative source: specs/032-multi-turn-test-cases/contracts/tool-arg-matchers.md §7.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from holodeck.lib.test_runner.tool_arg_matcher import (
    evaluate_expected_tools,
    find_matching_call,
    match_arg,
    match_fuzzy,
    match_literal,
    match_regex,
)
from holodeck.models.test_case import (
    ExpectedTool,
    FuzzyMatcher,
    LiteralMatcher,
    RegexMatcher,
)
from holodeck.models.test_result import ToolInvocation

# Matrix rows 1–4, 16–22 — literal + missing/extras


@pytest.mark.unit
@pytest.mark.parametrize(
    "expected, actual, should_match",
    [
        # Row 1
        (206588, 206588, True),
        # Row 2 (int ↔ float equivalence)
        (206588, 206588.0, True),
        # Row 3
        (206588, 206000, False),
        # Row 4
        (206588.0, 206588, True),
        # Row 18 bool strict
        (True, True, True),
        # Row 19 bool ≠ int
        (True, 1, False),
        # Row 20 list eq
        ([1, 2], [1, 2], True),
        # Row 21 list element numeric eq
        ([1, 2], [1.0, 2.0], True),
        # Row 22 dict eq
        ({"mode": "fast"}, {"mode": "fast"}, True),
    ],
)
def test_literal_matrix(expected: Any, actual: Any, should_match: bool) -> None:
    assert match_literal(expected, actual) is should_match


@pytest.mark.unit
def test_literal_none_vs_non_none() -> None:
    assert match_literal(None, None) is True
    assert match_literal(None, 0) is False
    assert match_literal(0, None) is False


# -------------------- Matrix rows 5–11 — fuzzy --------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern, actual, should_match",
    [
        # Row 5 separator
        ("206588", "206,588", True),
        # Row 6 whitespace
        ("206588", "206 588", True),
        # Row 7 numeric str ↔ float
        ("206588", 206588.0, True),
        # Row 8 no distance tolerance
        ("206588", 205000, False),
        # Row 9 percent
        ("14.14%", 0.1414, True),
        # Row 10 case
        ("YES", "yes", True),
        # Row 11 whitespace trim
        ("YES", " yes ", True),
    ],
)
def test_fuzzy_matrix(pattern: str, actual: Any, should_match: bool) -> None:
    assert match_fuzzy(pattern, actual) is should_match


# -------------------- Matrix rows 12–15 — regex --------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern, actual, should_match",
    [
        # Row 12 anchored fullmatch
        ("^2009.*$", "2009 cash flow", True),
        # Row 13 not anchored to start
        ("^2009.*$", "report 2009 cash flow", False),
        # Row 14 float suffix
        (r"^181001(\.0+)?$", "181001.0", True),
        # Row 15 mismatched decimal
        (r"^181001(\.0+)?$", "181001.5", False),
    ],
)
def test_regex_matrix(pattern: str, actual: Any, should_match: bool) -> None:
    compiled = re.compile(pattern)
    assert match_regex(compiled, actual) is should_match


# -------------------- Multi-call — row 23 --------------------


@pytest.mark.unit
def test_multi_call_any_wins() -> None:
    """Row 23 — two `subtract` calls in one turn, second matches."""
    invocations = [
        ToolInvocation(name="subtract", args={"a": 99}, bytes=0),
        ToolInvocation(name="subtract", args={"a": 206588.0}, bytes=0),
    ]
    expected = ExpectedTool(name="subtract", args={"a": {"fuzzy": "206588"}})
    idx, reason = find_matching_call(invocations, expected)
    assert idx == 1
    assert reason is None


# -------------------- None semantics (§5) --------------------


@pytest.mark.unit
def test_none_value_semantics() -> None:
    # Literal None matches actual None
    assert match_literal(None, None) is True
    # Fuzzy compares "none" (str(None).lower() == "none")
    assert match_fuzzy("none", None) is True
    # Regex compares str(None) == "None"
    assert match_regex(re.compile("None"), None) is True


# -------------------- Count threshold --------------------


@pytest.mark.unit
def test_count_threshold_one_satisfying_fails() -> None:
    invocations = [
        ToolInvocation(name="subtract", args={"a": 206588}, bytes=0),
        ToolInvocation(name="subtract", args={"a": 99}, bytes=0),
    ]
    expected = ExpectedTool(name="subtract", args={"a": {"fuzzy": "206588"}}, count=2)
    matched, details = evaluate_expected_tools([expected], invocations)
    assert matched is False


@pytest.mark.unit
def test_count_threshold_two_satisfying_passes() -> None:
    invocations = [
        ToolInvocation(name="subtract", args={"a": 206588}, bytes=0),
        ToolInvocation(name="subtract", args={"a": 206588.0}, bytes=0),
    ]
    expected = ExpectedTool(name="subtract", args={"a": {"fuzzy": "206588"}}, count=2)
    matched, details = evaluate_expected_tools([expected], invocations)
    assert matched is True


# -------------------- Missing / extras --------------------


@pytest.mark.unit
def test_missing_arg_reason() -> None:
    """Row 16 — asserted `a`, actual call missing `a`."""
    invocations = [ToolInvocation(name="subtract", args={"b": 1}, bytes=0)]
    expected = ExpectedTool(name="subtract", args={"a": 206588})
    matched, details = evaluate_expected_tools([expected], invocations)
    assert matched is False
    assert len(details) == 1
    assert details[0]["matched_call_index"] == -1
    assert "a" in details[0]["unmatched_reason"]
    assert "missing" in details[0]["unmatched_reason"]


@pytest.mark.unit
def test_extras_ignored_passes() -> None:
    """Row 17 — extra arg `c: 'y'` on actual does not cause failure."""
    invocations = [
        ToolInvocation(
            name="subtract",
            args={"a": 206588, "b": "x", "c": "y"},
            bytes=0,
        )
    ]
    expected = ExpectedTool(name="subtract", args={"a": 206588})
    matched, details = evaluate_expected_tools([expected], invocations)
    assert matched is True
    assert details[0]["matched_call_index"] == 0
    assert details[0]["unmatched_reason"] is None


# -------------------- match_arg tuple return --------------------


@pytest.mark.unit
def test_match_arg_literal_ok() -> None:
    ok, reason = match_arg(LiteralMatcher(206588), 206588)
    assert ok is True
    assert reason is None


@pytest.mark.unit
def test_match_arg_literal_mismatch() -> None:
    ok, reason = match_arg(LiteralMatcher(206588), 205000)
    assert ok is False
    assert reason is not None and "206588" in reason


@pytest.mark.unit
def test_match_arg_fuzzy_reason_has_pattern() -> None:
    ok, reason = match_arg(FuzzyMatcher("206588"), 205000)
    assert ok is False
    assert reason is not None and "206588" in reason


@pytest.mark.unit
def test_match_arg_regex_reason_has_pattern() -> None:
    ok, reason = match_arg(RegexMatcher(re.compile("^abc$")), "xyz")
    assert ok is False
    assert reason is not None and "abc" in reason
