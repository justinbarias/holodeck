"""Compare data-assembly helpers (US5 Phase 5).

Projects :class:`EvalRun` data into flat dataclasses / DataFrames the three
Compare variants and shared case-matrix heatmap consume. All numeric
arithmetic here is deliberate — it matches the handoff's ``compare.js`` so
the Dash port stays within rounding distance of the HTML prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import TestResult

COMPARE_PALETTE: list[str] = ["#7bff5a", "#5ae0a6", "#ffcf5a"]
"""Slot colors — index 0 = baseline, 1 = run-1, 2 = run-2. Matches
handoff ``compare.js:5``."""


# PRICING_TABLE values are documented best-effort placeholders at the time
# of writing. Update when vendor pricing changes.
#
# Format: (input_usd_per_1M_tokens, output_usd_per_1M_tokens)
#   https://docs.anthropic.com/claude/docs/models-overview  (checked 2026-04)
#   https://platform.openai.com/docs/pricing                (checked 2026-04)
PRICING_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4.5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4.5": (0.80, 4.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
}


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunStats:
    pass_rate: float
    passed: int
    total: int
    duration_ms: int
    geval_avg: float
    rag_avg: float
    total_tokens: int | None
    est_cost: float
    cost_source: str  # "pricing_table" | "synthetic_duration"


@dataclass(frozen=True)
class StatRow:
    key: str
    label: str
    values: list[Any]  # one per run, same order as input `runs`
    formatted: list[str]
    deltas: list[float | None]  # None for baseline; signed diff vs baseline otherwise
    delta_polarity: str  # "normal" | "invert"


@dataclass(frozen=True)
class ConfigRow:
    label: str
    values: list[str]
    all_same: bool


@dataclass(frozen=True)
class Callout:
    run_id: str
    prompt_version: str
    regressions: list[str]
    regressions_extra: int
    improvements: list[str]
    improvements_extra: int


# --------------------------------------------------------------------------- #
# Per-run stats                                                               #
# --------------------------------------------------------------------------- #


def _iter_cases(run: EvalRun):
    return run.report.results


def _geval_scores(case: TestResult) -> list[float]:
    return [float(m.score) for m in case.metric_results if m.kind == "geval"]


def _rag_scores(case: TestResult) -> list[float]:
    return [float(m.score) for m in case.metric_results if m.kind == "rag"]


def _collect_tokens(run: EvalRun) -> int | None:
    """Sum of prompt + completion tokens across cases, or ``None`` when any
    case lacks ``token_usage``.
    """
    total = 0
    for case in _iter_cases(run):
        tu = case.token_usage
        if tu is None:
            return None
        total += int((tu.prompt_tokens or 0) + (tu.completion_tokens or 0))
    return total


def _synthetic_cost(model_name: str, duration_ms: int) -> float:
    rate_per_sec = 0.018 if "sonnet" in (model_name or "") else 0.012
    return (duration_ms / 1000) * rate_per_sec


def _pricing_cost(model_name: str, prompt_tok: int, out_tok: int) -> float | None:
    rates = PRICING_TABLE.get(model_name)
    if rates is None:
        return None
    p_in, p_out = rates
    return (prompt_tok * p_in + out_tok * p_out) / 1_000_000


def run_stats(run: EvalRun) -> RunStats:
    cases = _iter_cases(run)

    geval_all: list[float] = []
    rag_all: list[float] = []
    for c in cases:
        geval_all.extend(_geval_scores(c))
        rag_all.extend(_rag_scores(c))

    geval_avg = sum(geval_all) / len(geval_all) if geval_all else 0.0
    rag_avg = sum(rag_all) / len(rag_all) if rag_all else 0.0

    total_tokens = _collect_tokens(run)
    model_name = run.metadata.agent_config.model.name

    est_cost: float
    cost_source: str
    if total_tokens is not None:
        prompt_tok = sum(
            int(c.token_usage.prompt_tokens or 0) if c.token_usage else 0 for c in cases
        )
        out_tok = sum(
            int(c.token_usage.completion_tokens or 0) if c.token_usage else 0
            for c in cases
        )
        priced = _pricing_cost(model_name, prompt_tok, out_tok)
        if priced is not None:
            est_cost = priced
            cost_source = "pricing_table"
        else:
            est_cost = _synthetic_cost(model_name, run.report.summary.total_duration_ms)
            cost_source = "synthetic_duration"
    else:
        est_cost = _synthetic_cost(model_name, run.report.summary.total_duration_ms)
        cost_source = "synthetic_duration"

    total = run.report.summary.total_tests
    passed = run.report.summary.passed
    pass_rate = (passed / total) if total > 0 else 0.0
    return RunStats(
        pass_rate=pass_rate,
        passed=passed,
        total=total,
        duration_ms=run.report.summary.total_duration_ms,
        geval_avg=geval_avg,
        rag_avg=rag_avg,
        total_tokens=total_tokens,
        est_cost=est_cost,
        cost_source=cost_source,
    )


# --------------------------------------------------------------------------- #
# Case matrix                                                                 #
# --------------------------------------------------------------------------- #


def _case_score(case: TestResult) -> float | None:
    for m in case.metric_results:
        if m.kind == "geval":
            return float(m.score)
    rag = [float(m.score) for m in case.metric_results if m.kind == "rag"]
    if rag:
        return sum(rag) / len(rag)
    return 1.0 if case.passed else 0.0


def _case_by_name(run: EvalRun, name: str) -> TestResult | None:
    return next((c for c in run.report.results if c.test_name == name), None)


def compute_case_matrix(runs: list[EvalRun]) -> pd.DataFrame:
    """One row per (sorted) case union across runs.

    Columns per run (id = ``report.timestamp``):
        * ``score::<id>``        — ``float | None``
        * ``passed::<id>``       — ``bool``
        * ``regression::<id>``   — ``bool`` (baseline-passed but this failed)
        * ``improvement::<id>``  — ``bool`` (baseline-failed but this passed)
    """
    if not runs:
        return pd.DataFrame(columns=["case_name"])

    case_names = sorted(
        {c.test_name for r in runs for c in r.report.results if c.test_name}
    )

    baseline = runs[0]

    rows: list[dict[str, Any]] = []
    for name in case_names:
        row: dict[str, Any] = {"case_name": name}
        base_case = _case_by_name(baseline, name)
        base_passed = base_case.passed if base_case is not None else None

        for i, run in enumerate(runs):
            rid = run.report.timestamp
            case = _case_by_name(run, name)
            if case is None:
                row[f"score::{rid}"] = None
                row[f"passed::{rid}"] = False
                row[f"regression::{rid}"] = False
                row[f"improvement::{rid}"] = False
                continue

            row[f"score::{rid}"] = _case_score(case)
            row[f"passed::{rid}"] = case.passed
            if i == 0 or base_passed is None:
                row[f"regression::{rid}"] = False
                row[f"improvement::{rid}"] = False
            else:
                row[f"regression::{rid}"] = bool(base_passed and not case.passed)
                row[f"improvement::{rid}"] = bool((not base_passed) and case.passed)

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Config diff                                                                 #
# --------------------------------------------------------------------------- #


def _ext_thinking(run: EvalRun) -> str:
    cfg = run.metadata.agent_config
    if cfg.claude is None or cfg.claude.extended_thinking is None:
        return "False"
    return str(bool(getattr(cfg.claude.extended_thinking, "enabled", False)))


def compute_config_diff(runs: list[EvalRun]) -> list[ConfigRow]:
    getters: list[tuple[str, Any]] = [
        ("prompt_version", lambda r: r.metadata.prompt_version.version),
        ("model_name", lambda r: r.metadata.agent_config.model.name),
        (
            "temperature",
            lambda r: (
                f"{r.metadata.agent_config.model.temperature}"
                if r.metadata.agent_config.model.temperature is not None
                else "—"
            ),
        ),
        (
            "tags_joined",
            lambda r: " ".join(f"#{t}" for t in (r.metadata.prompt_version.tags or [])),
        ),
        ("git_commit", lambda r: (r.metadata.git_commit or "—")[:10]),
        ("extended_thinking", _ext_thinking),
    ]

    out: list[ConfigRow] = []
    for label, fn in getters:
        values = [str(fn(r)) for r in runs]
        out.append(
            ConfigRow(
                label=label,
                values=values,
                all_same=all(v == values[0] for v in values),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Callouts                                                                    #
# --------------------------------------------------------------------------- #


def compute_compare_callouts(runs: list[EvalRun]) -> list[Callout]:
    if len(runs) < 2:
        return []
    baseline = runs[0]
    base_case_map = {c.test_name: c for c in baseline.report.results}

    out: list[Callout] = []
    for run in runs[1:]:
        regressions: list[str] = []
        improvements: list[str] = []
        for case in run.report.results:
            base = base_case_map.get(case.test_name)
            if base is None:
                continue
            if base.passed and not case.passed:
                regressions.append(case.test_name or "")
            elif (not base.passed) and case.passed:
                improvements.append(case.test_name or "")
        out.append(
            Callout(
                run_id=run.report.timestamp,
                prompt_version=run.metadata.prompt_version.version,
                regressions=regressions[:3],
                regressions_extra=max(0, len(regressions) - 3),
                improvements=improvements[:3],
                improvements_extra=max(0, len(improvements) - 3),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Summary rows                                                                #
# --------------------------------------------------------------------------- #


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_dur(ms: int) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms}ms"


def _fmt_cost(v: float) -> str:
    return f"${v:.3f}"


def _fmt_int(v: Any) -> str:
    return str(int(v)) if v is not None else "—"


def _fmt_score(v: float) -> str:
    return f"{v:.2f}"


def _fmt_passed(st: RunStats) -> str:
    return f"{st.passed}/{st.total}"


def compute_summary_rows(runs: list[EvalRun]) -> list[StatRow]:
    stats = [run_stats(r) for r in runs]
    if not stats:
        return []
    all_tokens = all(s.total_tokens is not None for s in stats)

    definitions: list[tuple[str, str, Any, Any, str]] = [
        ("pass_rate", "Pass rate", lambda s: s.pass_rate, _fmt_pct, "normal"),
        (
            "passed_ratio",
            "Passed",
            lambda s: s.passed / s.total if s.total else 0.0,
            lambda _: "",
            "normal",
        ),
        ("geval_avg", "Avg G-Eval", lambda s: s.geval_avg, _fmt_score, "normal"),
        ("rag_avg", "Avg RAG", lambda s: s.rag_avg, _fmt_score, "normal"),
        ("duration_ms", "Duration", lambda s: s.duration_ms, _fmt_dur, "invert"),
    ]

    if all_tokens:
        definitions.append(
            (
                "total_tokens",
                "Total tokens",
                lambda s: s.total_tokens or 0,
                _fmt_int,
                "invert",
            )
        )

    definitions.append(
        ("est_cost", "Est. cost", lambda s: s.est_cost, _fmt_cost, "invert")
    )

    out: list[StatRow] = []
    for key, label, getter, fmt, polarity in definitions:
        values: list[Any] = [getter(s) for s in stats]
        base_val = values[0]
        if key == "passed_ratio":
            formatted = [_fmt_passed(s) for s in stats]
        else:
            formatted = [fmt(v) for v in values]
        deltas: list[float | None] = [None]
        for v in values[1:]:
            try:
                deltas.append(float(v) - float(base_val))
            except (TypeError, ValueError):
                deltas.append(None)
        out.append(
            StatRow(
                key=key,
                label=label,
                values=values,
                formatted=formatted,
                deltas=deltas,
                delta_polarity=polarity,
            )
        )

    return out


# --------------------------------------------------------------------------- #
# Delta styling                                                               #
# --------------------------------------------------------------------------- #


def delta_pill_class(value: float | None, *, invert: bool = False) -> str:
    if value is None or value == 0:
        return "hd-delta-neutral"
    positive = (value < 0) if invert else (value > 0)
    return "hd-delta-pos" if positive else "hd-delta-neg"
