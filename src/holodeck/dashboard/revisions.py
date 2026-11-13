"""Revision colours for prompt versions and cue numbers for runs.

Both helpers are derived from the full set of loaded runs, never from a
filtered subset, so a prompt version keeps its colour and a run keeps its
``R`` number on every view, chart and chip.

* Revision colours: the nth prompt version in sorted version order takes the
  nth colour of :data:`REVISION_PALETTE` (wrapping after eight).
* Run numbers: ``R1`` is the oldest loaded run, ``Rn`` the newest, ordered by
  the parsed run time (a timestamp without an offset is read as UTC). Runs
  with the same time keep their load order, and each still gets its own
  number. The timestamp stays the run id used by callbacks and URLs.

:func:`run_time` and :func:`chronological` are the one ordering every view,
chart, verdict and download uses.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from holodeck.models.eval_run import EvalRun


@dataclass(frozen=True)
class RevisionColour:
    """Tint for fills and same-hue ink for text on that tint."""

    name: str
    tint: str
    ink: str


REVISION_PALETTE: tuple[RevisionColour, ...] = (
    RevisionColour("blue", "#cfe0f8", "#234f99"),
    RevisionColour("pink", "#f9d6e4", "#93305e"),
    RevisionColour("yellow", "#f7e8a6", "#6b5200"),
    RevisionColour("green", "#d2ecd9", "#1d6337"),
    RevisionColour("goldenrod", "#f1d7a4", "#6e4700"),
    RevisionColour("buff", "#e8dcc4", "#5b4a2a"),
    RevisionColour("salmon", "#f6d3c6", "#8a3a1c"),
    RevisionColour("cherry", "#efc6cd", "#85202f"),
)

UNKNOWN_REVISION = RevisionColour("unknown", "#eceef3", "#555b6b")
"""Neutral chip for a version that is not in the loaded runs."""

_VERSION_TOKEN = re.compile(r"\d+|\D+")


def version_sort_key(version: str) -> tuple[tuple[int, int | str], ...]:
    """Natural sort key: ``v1.2`` < ``v1.2.1`` < ``v1.10``."""
    key: list[tuple[int, int | str]] = []
    for token in _VERSION_TOKEN.findall(version.strip().lower()):
        if token.isdigit():
            key.append((0, int(token)))
        else:
            key.append((1, token))
    return tuple(key)


def sorted_versions(versions: Iterable[str]) -> list[str]:
    """Distinct versions in revision order."""
    return sorted(set(versions), key=lambda v: (version_sort_key(v), v))


def revision_colour_map(runs: Iterable[EvalRun]) -> dict[str, RevisionColour]:
    """Map every prompt version in ``runs`` to its revision colour."""
    ordered = sorted_versions(r.metadata.prompt_version.version for r in runs)
    return {
        version: REVISION_PALETTE[i % len(REVISION_PALETTE)]
        for i, version in enumerate(ordered)
    }


def revision_colour(
    version: str, colour_map: dict[str, RevisionColour]
) -> RevisionColour:
    """Colour for ``version``; a neutral chip when the version is unknown."""
    return colour_map.get(version, UNKNOWN_REVISION)


def parse_run_time(value: str) -> datetime:
    """Parse a saved run timestamp as an aware datetime (naive → UTC)."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = pd.Timestamp(value).to_pydatetime()
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def run_time(run: EvalRun) -> datetime:
    """When ``run`` happened, as an aware datetime."""
    return parse_run_time(run.report.timestamp)


def chronological(runs: Iterable[EvalRun]) -> list[EvalRun]:
    """Oldest first by parsed time; equal times keep their input order."""
    return sorted(runs, key=run_time)


@dataclass(frozen=True)
class RunNumbers:
    """Cue numbers for the loaded runs.

    Keyed by run object so two saved runs with the same timestamp still get
    distinct numbers; ``by_timestamp`` serves callers that only hold a run id
    (it keeps the first number for a repeated timestamp).
    """

    by_run: dict[int, int] = field(default_factory=dict)
    by_timestamp: dict[str, int] = field(default_factory=dict)

    def number(self, run: EvalRun | str) -> int | None:
        if isinstance(run, str):
            return self.by_timestamp.get(run)
        found = self.by_run.get(id(run))
        if found is not None:
            return found
        return self.by_timestamp.get(run.report.timestamp)


def run_numbers(runs: Iterable[EvalRun]) -> RunNumbers:
    """1-based chronological number for every run in ``runs``."""
    by_run: dict[int, int] = {}
    by_timestamp: dict[str, int] = {}
    for i, run in enumerate(chronological(runs), start=1):
        by_run[id(run)] = i
        by_timestamp.setdefault(run.report.timestamp, i)
    return RunNumbers(by_run=by_run, by_timestamp=by_timestamp)


def run_label(run: EvalRun | str, numbers: RunNumbers) -> str:
    """``R<n>`` for a known run; the raw timestamp prefix otherwise."""
    n = numbers.number(run)
    if n is not None:
        return f"R{n}"
    return (run if isinstance(run, str) else run.report.timestamp)[:16]
