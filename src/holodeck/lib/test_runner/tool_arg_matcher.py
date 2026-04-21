"""Tool-call argument matching primitives (US3, SC-006 acceptance matrix).

Authoritative contract: specs/032-multi-turn-test-cases/contracts/tool-arg-matchers.md.

This module is intentionally stateless and side-effect-free. It exposes:

- `match_literal(expected, actual)` — int↔float numeric equivalence + strict
  bool + deep equality otherwise.
- `match_fuzzy(pattern, actual)` — case/whitespace/separator/percent-tolerant
  numeric-aware string comparison.
- `match_regex(compiled, actual)` — `re.fullmatch` over `str(actual)`.
- `match_arg(matcher, actual)` — tuple-returning dispatcher used when building
  `arg_match_details` reasons.
- `find_matching_call(invocations, expected_tool)` — first-match-wins call
  selector for reporting; order-independent for pass/fail semantics.
- `evaluate_expected_tools(expected, invocations)` — returns
  `(matched, arg_match_details)` exactly matching
  contracts/turn-result-schema.md §2.1.

Number-normalization helpers (`_normalize`, `_try_parse_number`) are the first
canonical implementations in the repo; US4 `NumericEvaluator` should reuse
them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from holodeck.models.test_case import (
    ArgMatcher,
    ExpectedTool,
    FuzzyMatcher,
    LiteralMatcher,
    RegexMatcher,
)
from holodeck.models.test_result import ToolInvocation

# ---------------------------------------------------------------------------
# Literal matcher (contracts §2)
# ---------------------------------------------------------------------------


def match_literal(expected: Any, actual: Any) -> bool:
    """Literal equality with numeric int↔float equivalence + strict bool.

    Follows contracts/tool-arg-matchers.md §2. Performs element-wise literal
    matching inside lists so `[1, 2] == [1.0, 2.0]` (row 21).
    """
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(match_literal(e, a) for e, a in zip(expected, actual, strict=True))
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(match_literal(expected[k], actual[k]) for k in expected)
    # Bool strict — before numeric branch since bool is an int subclass.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    # None handling
    if expected is None or actual is None:
        return expected is None and actual is None
    # Numeric int ↔ float equivalence
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    # Strings / other — strict equality, no normalization.
    return bool(expected == actual)


# ---------------------------------------------------------------------------
# Normalization helpers (contracts §3) — exported for US4 reuse.
# ---------------------------------------------------------------------------


_SEPARATOR_RE = re.compile(r"[,_ ]")
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip thousands separators."""
    s = s.strip().lower()
    s = _SEPARATOR_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s


def _try_parse_number(s: str) -> float | None:
    """Parse a normalized numeric string, honoring a trailing `%`.

    Removes interior whitespace before attempting `float()` so
    "206 588"-style separator forms parse numerically.
    """
    pct = False
    if s.endswith("%"):
        s = s[:-1].strip()
        pct = True
    # Strip interior whitespace (e.g. "206 588" → "206588") so numeric
    # separator forms that survived normalization still parse.
    s = s.replace(" ", "")
    try:
        v = float(s)
        return v / 100 if pct else v
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Fuzzy matcher (contracts §3)
# ---------------------------------------------------------------------------


def match_fuzzy(pattern: str, actual: Any) -> bool:
    """Case/whitespace/separator/percent-tolerant, numeric-aware match."""
    s_expected = _normalize(pattern)
    s_actual = _normalize(str(actual))
    n_e = _try_parse_number(s_expected)
    n_a = _try_parse_number(s_actual)
    if n_e is not None and n_a is not None:
        return math.isclose(n_e, n_a, rel_tol=0.0, abs_tol=1e-9)
    return s_expected == s_actual


# ---------------------------------------------------------------------------
# Regex matcher (contracts §4)
# ---------------------------------------------------------------------------


def match_regex(compiled: re.Pattern[str], actual: Any) -> bool:
    """`re.fullmatch` over `str(actual)`."""
    return compiled.fullmatch(str(actual)) is not None


# ---------------------------------------------------------------------------
# Dispatch — match_arg returns (matched, reason_if_not)
# ---------------------------------------------------------------------------


def match_arg(matcher: ArgMatcher, actual: Any) -> tuple[bool, str | None]:
    """Dispatch a matcher against an actual value.

    Returns `(matched, reason)` where `reason` is `None` on match and a
    human-readable explanation otherwise.
    """
    if isinstance(matcher, LiteralMatcher):
        if match_literal(matcher.value, actual):
            return True, None
        return False, f"expected {matcher.value!r}, got {actual!r}"
    if isinstance(matcher, FuzzyMatcher):
        if match_fuzzy(matcher.pattern, actual):
            return True, None
        return False, f"expected ≈{matcher.pattern!r}, got {actual!r}"
    if isinstance(matcher, RegexMatcher):
        if match_regex(matcher.compiled, actual):
            return True, None
        return False, (f"expected regex {matcher.compiled.pattern!r}, got {actual!r}")
    raise TypeError(f"unknown matcher type: {type(matcher).__name__}")


# ---------------------------------------------------------------------------
# Tool name substring match — shared with `validate_tool_calls` fast path.
# ---------------------------------------------------------------------------


def _tool_name_matches(expected: str, actual: str) -> bool:
    """Case-sensitive substring match (preserves `validate_tool_calls` semantics)."""
    return expected in actual


# ---------------------------------------------------------------------------
# Call selection (contracts §1)
# ---------------------------------------------------------------------------


def _evaluate_call_against_args(
    invocation: ToolInvocation,
    args: dict[str, ArgMatcher],
) -> tuple[bool, str | None]:
    """Check one invocation against all asserted arg matchers.

    Returns `(matched, reason_if_not)`. The reason names the first failing
    arg so the reporter can point at it.
    """
    actual_args = invocation.args or {}
    for key, matcher in args.items():
        if key not in actual_args:
            return False, f"arg '{key}' missing"
        ok, reason = match_arg(matcher, actual_args[key])
        if not ok:
            return False, f"arg '{key}' mismatch: {reason}"
    return True, None


def find_matching_call(
    invocations: list[ToolInvocation],
    expected_tool: ExpectedTool,
) -> tuple[int, str | None]:
    """Find the first invocation satisfying the ExpectedTool.

    Filters invocations whose name contains `expected_tool.name` as a
    case-sensitive substring, then returns the index of the first whose args
    satisfy every asserted matcher. Returns `(-1, reason)` when no match.

    First-match-wins for reporting (the returned index); pass/fail semantics
    are order-independent because the caller counts ALL satisfying calls
    against the `count` threshold.
    """
    candidates = [
        (i, inv)
        for i, inv in enumerate(invocations)
        if _tool_name_matches(expected_tool.name, inv.name)
    ]
    if not candidates:
        return -1, f"no call to '{expected_tool.name}' found"

    args = expected_tool.args or {}
    if not args:
        # Name-only — first candidate wins.
        return candidates[0][0], None

    last_reason: str | None = None
    for idx, inv in candidates:
        ok, reason = _evaluate_call_against_args(inv, args)
        if ok:
            return idx, None
        last_reason = reason
    return -1, last_reason or f"no call to '{expected_tool.name}' satisfied all args"


# ---------------------------------------------------------------------------
# Top-level entry — evaluate_expected_tools
# ---------------------------------------------------------------------------


def evaluate_expected_tools(
    expected: Sequence[str | ExpectedTool],
    invocations: list[ToolInvocation],
) -> tuple[bool, list[dict[str, Any]]]:
    """Run every ExpectedTool against `invocations` and build arg_match_details.

    Only object-form entries (`ExpectedTool` with `args is not None` or
    `count > 1`) are consumed here; bare strings and simple name-only
    `ExpectedTool(args=None, count=1)` stay on the fast path and are ignored
    by this function (they produce no `arg_match_details` entries).

    Returns `(matched, details)` where `matched` is `all-pass` across every
    asserted object-form entry.
    """
    details: list[dict[str, Any]] = []
    overall = True
    for et in expected:
        if isinstance(et, str):
            # Fast path handles these; no arg_match_details entry.
            continue
        if et.args is None and et.count == 1:
            # Promoted legacy — fast path handles these too.
            continue
        args = et.args or {}
        # Count calls that satisfy every matcher.
        satisfying_indices: list[int] = []
        last_reason: str | None = None
        for idx, inv in enumerate(invocations):
            if not _tool_name_matches(et.name, inv.name):
                continue
            if args:
                ok, reason = _evaluate_call_against_args(inv, args)
                if ok:
                    satisfying_indices.append(idx)
                else:
                    last_reason = reason
            else:
                satisfying_indices.append(idx)

        args_asserted = _serialize_args_for_report(args)
        if len(satisfying_indices) >= et.count:
            details.append(
                {
                    "expected_tool": et.name,
                    "args_asserted": args_asserted,
                    "matched_call_index": satisfying_indices[0],
                    "unmatched_reason": None,
                }
            )
        else:
            overall = False
            # Reason selection: if no candidates at all, say so; if count
            # under-met, say how many we got; otherwise propagate arg reason.
            if not any(_tool_name_matches(et.name, inv.name) for inv in invocations):
                reason = f"no call to '{et.name}' found"
            elif not satisfying_indices and last_reason:
                reason = last_reason
            elif satisfying_indices:
                reason = (
                    f"expected {et.count} matching call(s), "
                    f"got {len(satisfying_indices)}"
                )
            else:
                reason = f"no call to '{et.name}' satisfied all args"
            details.append(
                {
                    "expected_tool": et.name,
                    "args_asserted": args_asserted,
                    "matched_call_index": -1,
                    "unmatched_reason": reason,
                }
            )
    return overall, details


def _serialize_args_for_report(args: dict[str, ArgMatcher]) -> dict[str, Any]:
    """Render args dict back to its wire form for `arg_match_details`."""
    out: dict[str, Any] = {}
    for key, val in args.items():
        if isinstance(val, LiteralMatcher):
            out[key] = val.value
        elif isinstance(val, FuzzyMatcher):
            out[key] = {"fuzzy": val.pattern}
        elif isinstance(val, RegexMatcher):
            out[key] = {"regex": val.compiled.pattern}
    return out
