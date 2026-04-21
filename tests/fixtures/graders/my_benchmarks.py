"""Fixture graders for US4 tests (specs/032-multi-turn-test-cases/tasks-us4.md T002).

These graders exercise the contract surface described in
`specs/032-multi-turn-test-cases/contracts/code-grader-contract.md`:

- ``numeric_equal`` — returns bool; covers spec Independent Test for `type: code`.
- ``raises_value_error`` — always raises; covers T037 / T038 exception policy.
- ``returns_float`` — returns a bare float; covers T036 threshold derivation.
- ``returns_grader_result`` — returns a fully-formed ``GraderResult``; T034/T040.
- ``returns_dict`` — returns a non-standard shape; T041 grader-error path.
- ``program_equivalence`` — quickstart's canonical example (kept for parity).
"""

from __future__ import annotations

from typing import Any

from holodeck.lib.test_runner.code_grader import GraderContext, GraderResult


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def numeric_equal(ctx: GraderContext) -> bool:
    """Pass when ``agent_response`` parses to the same number as ``ground_truth``.

    Tolerates ``,`` thousands separators and ``.0`` fractional suffix.
    """
    if ctx.ground_truth is None:
        return False
    expected = _parse_number(ctx.ground_truth)
    actual = _parse_number(ctx.agent_response)
    if expected is None or actual is None:
        return False
    return expected == actual


def raises_value_error(ctx: GraderContext) -> GraderResult:
    """Always raise; exercises the grader exception policy (contracts §6)."""
    raise ValueError("intentional fixture failure")


def returns_float(ctx: GraderContext) -> float:
    """Return a bare float so the runner must derive ``passed`` from threshold."""
    return 0.75


def returns_grader_result(ctx: GraderContext) -> GraderResult:
    """Return a fully-formed ``GraderResult`` including ``details``."""
    return GraderResult(
        score=1.0,
        passed=True,
        reason="fixture result",
        details={"foo": "bar"},
    )


def returns_dict(ctx: GraderContext) -> dict[str, Any]:
    """Return a dict — not a supported return shape (contracts §3.2 last row)."""
    return {"score": 0.9, "passed": True}


def program_equivalence(ctx: GraderContext) -> GraderResult:
    """Canonical quickstart grader — compares ``turn_program`` string equality."""
    expected = ctx.turn_config.get("turn_program", "")
    actual = ctx.agent_response or ""
    matches = expected and actual and expected.strip() == actual.strip()
    return GraderResult(
        score=1.0 if matches else 0.0,
        passed=bool(matches),
        reason=f"expected {expected!r}, got {actual!r}",
    )
