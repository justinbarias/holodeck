"""Deterministic evaluators (equality, numeric) — feature 032 US4.

Implements the two built-in zero-LLM metrics described in
``specs/032-multi-turn-test-cases/data-model.md`` §5:

- ``EqualityEvaluator`` — case/whitespace/punctuation-aware string compare.
- ``NumericEvaluator`` — abs/rel tolerance + percent / thousands-separator parse.

Both mirror ``BLEUEvaluator`` (``nlp_metrics.py:78-167``): declare ``PARAM_SPEC``
with ``RESPONSE + GROUND_TRUTH`` required, implement async ``_evaluate_impl``
returning a dict keyed by the metric name so the executor's
``result.get(metric_name, result.get("score", 0.0))`` path picks up the score.

Neither evaluator raises from ``_evaluate_impl``; parse failures surface via
``error`` + ``passed=False`` per the contract the executor expects.
"""

from __future__ import annotations

import re
import string
from typing import Any, ClassVar

from holodeck.lib.evaluators.base import BaseEvaluator
from holodeck.lib.evaluators.param_spec import EvalParam, ParamSpec
from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Helpers — kept private until US3 lands its `tool_arg_matcher.py` shared
# module. TODO(US3): migrate to the shared numeric/normalization helpers.
# -----------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
# narrow NBSP, regular NBSP, underscore, comma — all treated as thousands separators.
_THOUSANDS_CHARS = ",_  "


def _normalize_text(
    value: str,
    *,
    case_insensitive: bool,
    strip_whitespace: bool,
    strip_punctuation: bool,
) -> str:
    """Normalize a string per the evaluator flags."""
    out = value
    if case_insensitive:
        out = out.casefold()
    if strip_punctuation:
        out = out.translate(_PUNCT_TABLE)
    if strip_whitespace:
        out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def _try_parse_number(
    value: str,
    *,
    accept_percent: bool,
    accept_thousands_separators: bool,
) -> tuple[float | None, str | None]:
    """Parse ``value`` as a number, optionally honouring % / thousands flags.

    Returns ``(parsed, error)``. On success ``error`` is ``None``; on failure
    ``parsed`` is ``None`` and ``error`` is a one-liner describing the problem.
    """
    raw = value.strip()
    if not raw:
        return None, "value is empty"
    is_percent = raw.endswith("%")
    if is_percent and not accept_percent:
        return None, f"percent sign in {raw!r} but accept_percent=False"
    if is_percent:
        raw = raw[:-1].strip()
    if accept_thousands_separators:
        for ch in _THOUSANDS_CHARS:
            raw = raw.replace(ch, "")
    else:
        # Reject when the input contains one of the thousands chars that would
        # otherwise be tolerated. ``float()`` itself rejects "," already.
        if any(ch in raw for ch in _THOUSANDS_CHARS):
            return None, f"thousands separator in {raw!r} but flag is off"
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        return None, f"could not parse {value!r} as number: {exc}"
    if is_percent:
        parsed /= 100.0
    return parsed, None


# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------


class EqualityEvaluator(BaseEvaluator):
    """Strict string-equality evaluator with optional normalization flags."""

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
    )

    def __init__(
        self,
        *,
        case_insensitive: bool = False,
        strip_whitespace: bool = False,
        strip_punctuation: bool = False,
        timeout: float | None = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the evaluator.

        Args:
            case_insensitive: Lowercase both sides before compare.
            strip_whitespace: Collapse whitespace runs + trim before compare.
            strip_punctuation: Remove ``string.punctuation`` before compare.
            timeout: Evaluator timeout in seconds (default 60).
            **kwargs: Forwarded to ``BaseEvaluator``.
        """
        super().__init__(timeout=timeout, **kwargs)
        self.case_insensitive = case_insensitive
        self.strip_whitespace = strip_whitespace
        self.strip_punctuation = strip_punctuation

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Compare ``response`` against ``ground_truth`` under the config flags.

        Returns:
            ``{"equality": 0.0 | 1.0, "passed": bool, "error": str | None}``.
        """
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")
        if response is None or ground_truth is None:
            return {
                "equality": 0.0,
                "passed": False,
                "error": "response and ground_truth are required",
            }
        left = _normalize_text(
            str(response),
            case_insensitive=self.case_insensitive,
            strip_whitespace=self.strip_whitespace,
            strip_punctuation=self.strip_punctuation,
        )
        right = _normalize_text(
            str(ground_truth),
            case_insensitive=self.case_insensitive,
            strip_whitespace=self.strip_whitespace,
            strip_punctuation=self.strip_punctuation,
        )
        equal = left == right
        return {
            "equality": 1.0 if equal else 0.0,
            "passed": equal,
            "error": None,
        }


class NumericEvaluator(BaseEvaluator):
    """Numeric-comparison evaluator with absolute + relative tolerance.

    Pass iff ``abs(actual - expected) <= absolute_tolerance`` (inclusive; FR-018)
    or ``abs(actual - expected) <= relative_tolerance * abs(expected)``.
    """

    PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(
        required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}),
    )

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 0.0,
        accept_percent: bool = False,
        accept_thousands_separators: bool = False,
        timeout: float | None = 60.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the evaluator.

        Args:
            absolute_tolerance: Maximum allowed absolute difference (≥ 0).
            relative_tolerance: Maximum allowed relative difference (≥ 0).
            accept_percent: Parse trailing ``%`` as ``/100``.
            accept_thousands_separators: Strip ``,`` / ``_`` / NBSP before parse.
            timeout: Evaluator timeout in seconds (default 60).
            **kwargs: Forwarded to ``BaseEvaluator``.
        """
        super().__init__(timeout=timeout, **kwargs)
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.accept_percent = accept_percent
        self.accept_thousands_separators = accept_thousands_separators

    async def _evaluate_impl(self, **kwargs: Any) -> dict[str, Any]:
        """Compare ``response`` and ``ground_truth`` numerically.

        Returns:
            ``{"numeric": 0.0 | 1.0, "passed": bool, "error": str | None}``.
        """
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")
        if response is None or ground_truth is None:
            return {
                "numeric": 0.0,
                "passed": False,
                "error": "response and ground_truth are required",
            }
        actual, err_a = _try_parse_number(
            str(response),
            accept_percent=self.accept_percent,
            accept_thousands_separators=self.accept_thousands_separators,
        )
        expected, err_e = _try_parse_number(
            str(ground_truth),
            accept_percent=self.accept_percent,
            accept_thousands_separators=self.accept_thousands_separators,
        )
        if err_a is not None or err_e is not None or actual is None or expected is None:
            return {
                "numeric": 0.0,
                "passed": False,
                "error": err_a or err_e,
            }
        diff = abs(actual - expected)
        abs_ok = diff <= self.absolute_tolerance
        rel_ok = self.relative_tolerance > 0 and diff <= self.relative_tolerance * abs(
            expected
        )
        passed = abs_ok or rel_ok
        return {
            "numeric": 1.0 if passed else 0.0,
            "passed": passed,
            "error": None,
        }


__all__ = ["EqualityEvaluator", "NumericEvaluator"]
