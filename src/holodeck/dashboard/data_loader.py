"""Data layer for the eval-runs dashboard.

Responsibilities:
- Load EvalRun instances from disk (real mode) or the seed fixture.
- Normalize codebase ↔ handoff shape drift in one place:
  * `ReportSummary.pass_rate` is 0..100 in the canonical model; the dashboard
    exposes `pass_rate` on the 0..1 scale. To stay robust against legacy runs
    that persisted 0..1 by mistake, we compute fractions from
    ``passed / total_tests`` rather than trusting the stored number.
  * `ReportSummary.total_duration_ms` maps to the handoff's `duration_ms`.
- Derive per-chart DataFrames (summary, metric trend, breakdown).
- Detect regressions + prompt-version boundaries for chart annotations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from holodeck.dashboard.seed_data import build_seed_runs
from holodeck.models.eval_run import EvalRun

logger = logging.getLogger(__name__)

MetricKind = Literal["standard", "rag", "geval", "code"]


def _pass_rate_fraction(run: EvalRun) -> float:
    """Return pass_rate as a 0..1 fraction.

    The persisted ``ReportSummary.pass_rate`` has inconsistent units across
    producers — the test runner (executor.py) writes a 0..1 fraction while
    the seed-dataset fixture writes 0..100. Computing from
    ``passed/total_tests`` is authoritative and producer-agnostic.
    """
    s = run.report.summary
    return (s.passed / s.total_tests) if s.total_tests > 0 else 0.0


def load_all(results_dir: Path) -> list[EvalRun]:
    """Load every `*.json` file under `results_dir` as an EvalRun.

    Corrupt or schema-violating files are logged and skipped (FR-024, R6).
    Missing directory returns [] without error (FR-023).
    Results sorted by `report.timestamp` ascending.
    """
    if not results_dir.exists() or not results_dir.is_dir():
        return []

    runs: list[EvalRun] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(EvalRun.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skipping %s — failed to load EvalRun: %s", path.name, exc)
            continue
    runs.sort(key=lambda r: r.report.timestamp)
    return runs


def load_runs_for_app() -> list[EvalRun]:
    """Load runs according to env vars.

    - `HOLODECK_DASHBOARD_USE_SEED=1` → seed dataset.
    - Otherwise → `load_all(HOLODECK_DASHBOARD_RESULTS_DIR)`.
    """
    if os.environ.get("HOLODECK_DASHBOARD_USE_SEED") == "1":
        return build_seed_runs()
    results_dir = os.environ.get("HOLODECK_DASHBOARD_RESULTS_DIR")
    if not results_dir:
        return []
    return load_all(Path(results_dir))


def _pass_rate_tier(pass_rate: float) -> str:
    """Map 0..1 pass_rate to handoff tier thresholds (summary.js:360)."""
    if pass_rate >= 0.85:
        return "pass"
    if pass_rate >= 0.65:
        return "warn"
    return "fail"


def _prompt_version_str(run: EvalRun) -> str:
    return run.metadata.prompt_version.version


def _model_name_str(run: EvalRun) -> str:
    return run.metadata.agent_config.model.name


def _tags_list(run: EvalRun) -> list[str]:
    return list(run.metadata.prompt_version.tags or [])


def to_summary_dataframe(runs: list[EvalRun]) -> pd.DataFrame:
    """One row per run, sorted newest-first; pass_rate on 0..1 scale."""
    if not runs:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "pass_rate",
                "passed",
                "total",
                "duration_ms",
                "prompt_version",
                "model_name",
                "git_commit",
                "tags",
                "pass_rate_tier",
            ]
        )

    rows = []
    for run in runs:
        pr = _pass_rate_fraction(run)
        rows.append(
            {
                "id": run.report.timestamp,
                "timestamp": pd.to_datetime(run.report.timestamp),
                "pass_rate": pr,
                "passed": run.report.summary.passed,
                "total": run.report.summary.total_tests,
                "duration_ms": run.report.summary.total_duration_ms,
                "prompt_version": _prompt_version_str(run),
                "model_name": _model_name_str(run),
                "git_commit": run.metadata.git_commit or "",
                "tags": ",".join(_tags_list(run)),
                "pass_rate_tier": _pass_rate_tier(pr),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def _iter_metric_results(run: EvalRun, kind: MetricKind | None = None):
    for case in run.report.results:
        for m in case.metric_results:
            if kind is None or m.kind == kind:
                yield m


def to_metric_trend_dataframe(runs: list[EvalRun], kind: MetricKind) -> pd.DataFrame:
    """Wide DataFrame: index=timestamp, one column per metric name.

    Values are per-run averages of that metric's score within the given kind.
    """
    if not runs:
        return pd.DataFrame()

    per_run: list[dict] = []
    for run in runs:
        row: dict[str, object] = {"timestamp": pd.to_datetime(run.report.timestamp)}
        bucket: dict[str, list[float]] = {}
        for m in _iter_metric_results(run, kind):
            bucket.setdefault(m.metric_name, []).append(float(m.score))
        for name, scores in bucket.items():
            row[name] = sum(scores) / len(scores) if scores else None
        per_run.append(row)

    df = pd.DataFrame(per_run).sort_values("timestamp").reset_index(drop=True)
    return df


def to_breakdown_dataframe(
    runs: list[EvalRun], kind: MetricKind, recent_n: int = 6
) -> pd.DataFrame:
    """Per-metric aggregate over the last `recent_n` runs."""
    if not runs:
        return pd.DataFrame(columns=["metric_name", "avg_score", "pass_count", "total"])

    sorted_runs = sorted(runs, key=lambda r: r.report.timestamp)[-recent_n:]

    agg: dict[str, dict] = {}
    for run in sorted_runs:
        for m in _iter_metric_results(run, kind):
            entry = agg.setdefault(
                m.metric_name, {"scores": [], "pass_count": 0, "total": 0}
            )
            entry["scores"].append(float(m.score))
            entry["total"] += 1
            if m.passed:
                entry["pass_count"] += 1

    rows = []
    for name, e in agg.items():
        scores = e["scores"]
        rows.append(
            {
                "metric_name": name,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "pass_count": e["pass_count"],
                "total": e["total"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["metric_name", "avg_score", "pass_count", "total"])
    return (
        pd.DataFrame(rows)
        .sort_values("avg_score", ascending=False)
        .reset_index(drop=True)
    )


def detect_regressions(runs: list[EvalRun], drop_threshold: float = 0.04) -> list[int]:
    """Indices (chronological) where pass_rate drops more than `drop_threshold`."""
    if len(runs) < 2:
        return []
    sorted_runs = sorted(runs, key=lambda r: r.report.timestamp)
    rates = [_pass_rate_fraction(r) for r in sorted_runs]
    out = []
    for i in range(1, len(rates)):
        if rates[i] - rates[i - 1] < -drop_threshold:
            out.append(i)
    return out


def detect_version_boundaries(runs: list[EvalRun]) -> list[tuple[int, str]]:
    """Indices (chronological) where prompt_version changes, + the new version."""
    if not runs:
        return []
    sorted_runs = sorted(runs, key=lambda r: r.report.timestamp)
    out: list[tuple[int, str]] = []
    prev = None
    for i, run in enumerate(sorted_runs):
        v = _prompt_version_str(run)
        if prev is not None and v != prev:
            out.append((i, v))
        prev = v
    return out


def distinct_values(runs: list[EvalRun], field: str) -> list[str]:
    """Distinct values for filter dropdowns."""
    seen: list[str] = []
    for run in runs:
        if field == "prompt_version":
            v = _prompt_version_str(run)
            if v not in seen:
                seen.append(v)
        elif field == "model_name":
            v = _model_name_str(run)
            if v not in seen:
                seen.append(v)
        elif field == "tags":
            for t in _tags_list(run):
                if t not in seen:
                    seen.append(t)
        else:
            raise ValueError(f"unknown field for distinct_values: {field!r}")
    return sorted(seen)
