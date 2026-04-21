"""User-supplied Python grader invocation (feature 032 US4).

Provides the public ``GraderContext`` / ``GraderResult`` dataclasses that
user-land graders consume, plus ``invoke_grader`` — the runner-side helper
that normalizes shortcut returns (``bool``, ``float``), catches exceptions,
and produces a ``MetricResult`` envelope.

Contract: ``specs/032-multi-turn-test-cases/contracts/code-grader-contract.md``.

Principle I (No-Code-First) exception: this file and ``CodeMetric`` in
``holodeck.models.evaluation`` are the only places that dispatch user-supplied
Python. Only ``importlib.import_module`` is used for dynamic resolution; no
dynamic-code primitives beyond that. See ``spec.md §Complexity Tracking``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from holodeck.lib.logging_config import get_logger
from holodeck.models.test_result import MetricResult, ToolInvocation

logger = get_logger(__name__)


@dataclass(frozen=True)
class GraderContext:
    """Read-only per-turn context handed to a user-supplied grader.

    See ``contracts/code-grader-contract.md`` §3.1 for the field list. Both
    ordered collections are tuples rather than lists so graders cannot mutate
    them (FR-020).
    """

    turn_input: str
    agent_response: str
    ground_truth: str | None
    tool_invocations: tuple[ToolInvocation, ...]
    retrieval_context: tuple[str, ...] | None
    turn_index: int
    test_case_name: str | None
    turn_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderResult:
    """Structured return value for a grader (contracts §3.2).

    - ``score`` is the normalized ``[0.0, 1.0]`` score.
    - ``passed`` is optional; when ``None`` the runner derives it from the
      ``threshold`` (or defaults to ``>= 0.5``) — see contracts §5.
    - ``reason`` / ``details`` flow into the report / dashboard.
    """

    score: float
    passed: bool | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None


def _derive_passed(score: float, threshold: float | None) -> bool:
    """Per contracts §5."""
    if threshold is not None:
        return score >= threshold
    return score >= 0.5


def _normalize_return(raw: Any, threshold: float | None) -> GraderResult | None:
    """Normalize a grader's return value into a ``GraderResult``.

    Per contracts §3.2:
        - ``True`` → ``GraderResult(score=1.0, passed=True)``
        - ``False`` → ``GraderResult(score=0.0, passed=False)``
        - bare float → ``GraderResult(score=float, passed=None)`` — caller
          derives ``passed``.
        - ``GraderResult`` → passthrough.
        - anything else → ``None`` (caller records as grader error).
    """
    if isinstance(raw, GraderResult):
        return raw
    if isinstance(raw, bool):
        return GraderResult(
            score=1.0 if raw else 0.0,
            passed=raw,
        )
    if isinstance(raw, (int, float)):
        return GraderResult(score=float(raw), passed=None)
    return None


def build_grader_context(
    *,
    turn_input: str,
    agent_response: str,
    ground_truth: str | None,
    tool_invocations: list[ToolInvocation],
    retrieval_context: list[str] | None,
    turn_index: int,
    test_case_name: str | None,
    turn_config: dict[str, Any] | None = None,
) -> GraderContext:
    """Build a frozen ``GraderContext`` from per-turn runner state."""
    return GraderContext(
        turn_input=turn_input,
        agent_response=agent_response,
        ground_truth=ground_truth,
        tool_invocations=tuple(tool_invocations),
        retrieval_context=(
            tuple(retrieval_context) if retrieval_context is not None else None
        ),
        turn_index=turn_index,
        test_case_name=test_case_name,
        turn_config=turn_config or {},
    )


def invoke_grader(
    fn: Callable[[GraderContext], Any],
    ctx: GraderContext,
    *,
    metric_name: str,
    threshold: float | None,
) -> tuple[MetricResult, dict[str, Any] | None, Exception | None]:
    """Invoke ``fn`` on ``ctx`` and produce a ``MetricResult`` envelope.

    Wraps the call in ``try``/``except Exception`` per contracts §6. On
    failure returns a ``MetricResult`` with ``score=0.0, passed=False,
    error=...`` and surfaces the captured exception as the third element of
    the tuple so the runner can escalate when ``fail_on_error=True``.

    Args:
        fn: The grader callable (already resolved from ``module:callable``).
        ctx: The per-turn context to hand to the grader.
        metric_name: Metric display name for the ``MetricResult.metric_name``.
        threshold: Optional pass threshold (contracts §5).

    Returns:
        Tuple of ``(MetricResult, grader_details_or_None, captured_exception)``.
    """
    start = time.perf_counter()
    captured: Exception | None = None
    details: dict[str, Any] | None = None
    try:
        raw = fn(ctx)
        normalized = _normalize_return(raw, threshold)
        if normalized is None:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return (
                MetricResult(
                    metric_name=metric_name,
                    kind="code",
                    score=0.0,
                    threshold=threshold,
                    passed=False,
                    scale="0-1",
                    error=(
                        f"grader returned unsupported type {type(raw).__name__}; "
                        "expected bool, float, or GraderResult"
                    ),
                    retry_count=0,
                    evaluation_time_ms=elapsed_ms,
                    model_used=None,
                    reasoning=None,
                ),
                None,
                None,
            )
        # JSON-safety check on details (T040a).
        if normalized.details is not None:
            try:
                json.dumps(normalized.details, default=None)
            except TypeError as exc:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                return (
                    MetricResult(
                        metric_name=metric_name,
                        kind="code",
                        score=0.0,
                        threshold=threshold,
                        passed=False,
                        scale="0-1",
                        error=f"details not JSON-serializable: {exc}",
                        retry_count=0,
                        evaluation_time_ms=elapsed_ms,
                        model_used=None,
                        reasoning=None,
                    ),
                    None,
                    None,
                )
            details = dict(normalized.details)
        passed = (
            normalized.passed
            if normalized.passed is not None
            else _derive_passed(normalized.score, threshold)
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return (
            MetricResult(
                metric_name=metric_name,
                kind="code",
                score=float(normalized.score),
                threshold=threshold,
                passed=bool(passed),
                scale="0-1",
                error=None,
                retry_count=0,
                evaluation_time_ms=elapsed_ms,
                model_used=None,
                reasoning=normalized.reason,
            ),
            details,
            None,
        )
    except Exception as exc:  # noqa: BLE001 — contract: catch everything.
        logger.warning(
            "Grader %r raised %s: %s",
            metric_name,
            type(exc).__name__,
            exc,
        )
        captured = exc
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return (
            MetricResult(
                metric_name=metric_name,
                kind="code",
                score=0.0,
                threshold=threshold,
                passed=False,
                scale="0-1",
                error=f"{type(exc).__name__}: {exc}",
                retry_count=0,
                evaluation_time_ms=elapsed_ms,
                model_used=None,
                reasoning=None,
            ),
            None,
            captured,
        )


__all__ = [
    "GraderContext",
    "GraderResult",
    "build_grader_context",
    "invoke_grader",
]
