"""Unit tests for deterministic evaluators (equality, numeric) — US4.

Covers T003–T013 in ``specs/032-multi-turn-test-cases/tasks-us4.md``:

- Equality: strict default, case-insensitive, whitespace/punctuation flags.
- Numeric: default tolerance (1e-6, boundary inclusive), absolute/relative,
  percent / thousands separators, parse-failure error surface.
- 15-row acceptance matrix (SC-008).
"""

from __future__ import annotations

from typing import Any

import pytest

from holodeck.lib.evaluators.deterministic import (
    EqualityEvaluator,
    NumericEvaluator,
)


@pytest.mark.unit
class TestEqualityEvaluator:
    @pytest.mark.asyncio
    async def test_equality_strict_default(self) -> None:
        ev = EqualityEvaluator()
        miss = await ev.evaluate(response="Yes", ground_truth="yes")
        assert miss["passed"] is False
        assert miss["equality"] == 0.0

        hit = await ev.evaluate(response="Yes", ground_truth="Yes")
        assert hit["passed"] is True
        assert hit["equality"] == 1.0

    @pytest.mark.asyncio
    async def test_equality_case_insensitive(self) -> None:
        ev = EqualityEvaluator(case_insensitive=True)
        still_miss = await ev.evaluate(response="Yes", ground_truth="yes.")
        assert still_miss["passed"] is False

        ev2 = EqualityEvaluator(case_insensitive=True, strip_punctuation=True)
        hit = await ev2.evaluate(response="Yes", ground_truth="yes.")
        assert hit["passed"] is True

    @pytest.mark.asyncio
    async def test_equality_strip_whitespace(self) -> None:
        ev = EqualityEvaluator(strip_whitespace=True)
        hit = await ev.evaluate(
            response=" hello  world ",
            ground_truth="hello world",
        )
        assert hit["passed"] is True


@pytest.mark.unit
class TestNumericEvaluator:
    @pytest.mark.asyncio
    async def test_numeric_default_tolerance(self) -> None:
        ev = NumericEvaluator()
        # diff = 1e-7 — within default 1e-6 tolerance
        r1 = await ev.evaluate(response="1.0000001", ground_truth="1")
        assert r1["passed"] is True

        # diff = 0.1 — outside default tolerance
        r2 = await ev.evaluate(response="1.1", ground_truth="1")
        assert r2["passed"] is False

        # boundary: diff == 1e-6 exactly; FR-018 semantics are inclusive
        r3 = await ev.evaluate(response="1.000001", ground_truth="1")
        assert r3["passed"] is True

    @pytest.mark.asyncio
    async def test_numeric_absolute_tolerance(self) -> None:
        # diff = 0.00864, abs_tol = 0.001 → fail; abs_tol = 0.01 → pass.
        ev_tight = NumericEvaluator(absolute_tolerance=0.001)
        r1 = await ev_tight.evaluate(response="0.15", ground_truth="0.14136")
        assert r1["passed"] is False

        ev_loose = NumericEvaluator(absolute_tolerance=0.01)
        r2 = await ev_loose.evaluate(response="0.142", ground_truth="0.14136")
        assert r2["passed"] is True

    @pytest.mark.asyncio
    async def test_numeric_relative_tolerance(self) -> None:
        ev = NumericEvaluator(relative_tolerance=0.01)
        # 100 * 1.005 = 100.5; |100.5-100|/|100| = 0.005 < 0.01
        r = await ev.evaluate(response="100.5", ground_truth="100")
        assert r["passed"] is True

    @pytest.mark.asyncio
    async def test_numeric_accept_percent(self) -> None:
        ev_off = NumericEvaluator(absolute_tolerance=0.01)
        bad = await ev_off.evaluate(response="14.14%", ground_truth="0.14136")
        assert bad["passed"] is False
        assert bad.get("error") is not None

        ev_on = NumericEvaluator(absolute_tolerance=0.01, accept_percent=True)
        good = await ev_on.evaluate(response="14.14%", ground_truth="0.14136")
        assert good["passed"] is True
        assert good.get("error") is None

    @pytest.mark.asyncio
    async def test_numeric_accept_thousands_separators(self) -> None:
        ev_off = NumericEvaluator()
        bad = await ev_off.evaluate(response="206,588", ground_truth="206588")
        assert bad["passed"] is False
        assert bad.get("error") is not None

        ev_on = NumericEvaluator(accept_thousands_separators=True)
        good = await ev_on.evaluate(response="206,588", ground_truth="206588")
        assert good["passed"] is True

    @pytest.mark.asyncio
    async def test_numeric_non_numeric_inputs(self) -> None:
        ev = NumericEvaluator()
        r = await ev.evaluate(response="abc", ground_truth="5")
        assert r["passed"] is False
        assert r.get("error") is not None
        assert "abc" in r["error"] or "parse" in r["error"].lower()


# -----------------------------------------------------------------------------
# 15-row acceptance matrix (SC-008). Each row is (evaluator, kwargs, response,
# ground_truth, expected_passed).
# -----------------------------------------------------------------------------

_MATRIX: list[tuple[str, dict[str, Any], str, str, bool]] = [
    # Equality
    ("equality", {}, "hello", "hello", True),
    ("equality", {}, "Hello", "hello", False),
    ("equality", {"case_insensitive": True}, "Hello", "hello", True),
    ("equality", {"strip_whitespace": True}, " a b ", "a b", True),
    ("equality", {"strip_punctuation": True}, "yes.", "yes", True),
    # Numeric — int vs float
    ("numeric", {}, "1", "1.0", True),
    # percent
    (
        "numeric",
        {"accept_percent": True, "absolute_tolerance": 1e-3},
        "50%",
        "0.5",
        True,
    ),
    # thousands
    ("numeric", {"accept_thousands_separators": True}, "1,000", "1000", True),
    # abs tolerance miss
    ("numeric", {"absolute_tolerance": 0.01}, "1.02", "1", False),
    # rel tolerance hit
    ("numeric", {"relative_tolerance": 0.1}, "110", "100", True),
    # abs tolerance hit
    ("numeric", {"absolute_tolerance": 0.5}, "1.4", "1", True),
    # percent off-by-default
    ("numeric", {}, "50%", "0.5", False),
    # boundary: diff == tolerance (inclusive). 0.5 - 0.25 = 0.25 exactly in IEEE 754.
    ("numeric", {"absolute_tolerance": 0.25}, "0.5", "0.25", True),
    # parse failure
    ("numeric", {}, "n/a", "5", False),
    # integer equality via numeric
    ("numeric", {}, "25587", "25587", True),
]


@pytest.mark.unit
@pytest.mark.parametrize("kind,kwargs,response,ground_truth,expected", _MATRIX)
@pytest.mark.asyncio
async def test_acceptance_matrix(
    kind: str,
    kwargs: dict[str, Any],
    response: str,
    ground_truth: str,
    expected: bool,
) -> None:
    if kind == "equality":
        ev: Any = EqualityEvaluator(**kwargs)
    else:
        ev = NumericEvaluator(**kwargs)
    result = await ev.evaluate(response=response, ground_truth=ground_truth)
    assert result["passed"] is expected, (
        f"{kind} kwargs={kwargs} response={response!r} gt={ground_truth!r} "
        f"expected passed={expected}, got {result}"
    )
