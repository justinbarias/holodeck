"""Filter dataclass + URL query-param serde.

`Filters` is the single source of truth for faceted filtering on the Summary
view. Applied before charts/tables render, round-trips to and from the URL
so refresh/share works (FR-028b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from holodeck.dashboard.loaded_run import LoadedRun
from holodeck.dashboard.revisions import run_time

MetricKind = Literal["standard", "rag", "geval", "code"]


@dataclass
class Filters:
    date_from: str | None = None  # ISO date string or None
    date_to: str | None = None
    prompt_versions: list[str] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)
    min_pass_rate: float = 0.0  # 0..1
    tags: list[str] = field(default_factory=list)
    metric_kind: MetricKind = "rag"
    search: str = ""  # substring match across prompt version / model / commit


def _bound(value: str, ts: pd.Timestamp, *, end: bool) -> pd.Timestamp:
    """Parse a filter date in the run timestamp's zone.

    A date-only ``date_to`` covers the whole day, so it ends at the next
    midnight (exclusive).
    """
    bound = pd.to_datetime(value)
    if bound.tzinfo is None and ts.tzinfo is not None:
        bound = bound.tz_localize(ts.tzinfo)
    elif bound.tzinfo is not None and ts.tzinfo is None:
        bound = bound.tz_convert(None)
    if end and len(value) == 10:
        bound = bound + pd.Timedelta(days=1)
    return bound


def _run_passes(run: LoadedRun, f: Filters) -> bool:
    # Parsed at the shared boundary: aware, naive read as UTC.
    ts = pd.Timestamp(run_time(run))
    if f.date_from and ts < _bound(f.date_from, ts, end=False):
        return False
    if f.date_to:
        to_bound = _bound(f.date_to, ts, end=True)
        if (ts >= to_bound) if len(f.date_to) == 10 else (ts > to_bound):
            return False
    if (
        f.prompt_versions
        and run.metadata.prompt_version.version not in f.prompt_versions
    ):
        return False
    if f.model_names and run.metadata.agent_config.model.name not in f.model_names:
        return False
    if f.min_pass_rate > 0:
        total = run.report.summary.total_tests
        pr = (run.report.summary.passed / total) if total > 0 else 0.0
        if pr < f.min_pass_rate:
            return False
    if f.tags:
        run_tags = set(run.metadata.prompt_version.tags or [])
        if not run_tags.intersection(f.tags):
            return False
    if f.search:
        needle = f.search.strip().lower()
        if needle:
            haystack = " ".join(
                [
                    run.metadata.prompt_version.version or "",
                    run.metadata.agent_config.model.name or "",
                    run.metadata.git_commit or "",
                    ts.strftime("%Y-%m-%d %H:%M"),
                    " ".join(run.metadata.prompt_version.tags or []),
                ]
            ).lower()
            if needle not in haystack:
                return False
    return True


def apply(filters: Filters, runs: list[LoadedRun]) -> list[LoadedRun]:
    return [r for r in runs if _run_passes(r, filters)]


def filters_to_query_params(f: Filters) -> dict[str, str]:
    """Serialize non-default fields to a flat dict suitable for URL query params."""
    out: dict[str, str] = {}
    if f.date_from:
        out["from"] = f.date_from
    if f.date_to:
        out["to"] = f.date_to
    if f.prompt_versions:
        out["versions"] = ",".join(f.prompt_versions)
    if f.model_names:
        out["models"] = ",".join(f.model_names)
    if f.min_pass_rate > 0:
        out["min"] = f"{f.min_pass_rate:.2f}"
    if f.tags:
        out["tags"] = ",".join(f.tags)
    if f.metric_kind != "rag":
        out["kind"] = f.metric_kind
    if f.search:
        out["q"] = f.search
    return out


def filters_from_query_params(params: dict[str, str]) -> Filters:
    f = Filters()
    if params.get("from"):
        f.date_from = params["from"]
    if params.get("to"):
        f.date_to = params["to"]
    if params.get("versions"):
        f.prompt_versions = [v for v in params["versions"].split(",") if v]
    if params.get("models"):
        f.model_names = [v for v in params["models"].split(",") if v]
    if params.get("min"):
        try:
            f.min_pass_rate = float(params["min"])
        except ValueError:
            f.min_pass_rate = 0.0
    if params.get("tags"):
        f.tags = [v for v in params["tags"].split(",") if v]
    kind = params.get("kind")
    if kind in ("standard", "rag", "geval", "code"):
        f.metric_kind = kind  # type: ignore[assignment]
    if params.get("q"):
        f.search = params["q"]
    return f
