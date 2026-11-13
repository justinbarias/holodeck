"""Revision colours for prompt versions and cue numbers for runs.

Both helpers are derived from the full set of loaded runs, never from a
filtered subset, so a prompt version keeps its colour and a run keeps its
``R`` number on every view, chart and chip.

* Revision colours: the nth prompt version in sorted version order takes the
  nth colour of :data:`REVISION_PALETTE` (wrapping after eight).
* Run numbers: ``R1`` is the oldest loaded run, ``Rn`` the newest. The
  timestamp stays the run id used by callbacks and URLs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

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


def run_numbers(runs: Iterable[EvalRun]) -> dict[str, int]:
    """Map run id (``report.timestamp``) to its 1-based chronological number."""
    ordered = sorted(runs, key=lambda r: r.report.timestamp)
    return {run.report.timestamp: i for i, run in enumerate(ordered, start=1)}


def run_label(run_id: str, numbers: dict[str, int]) -> str:
    """``R<n>`` for a known run; the raw timestamp prefix otherwise."""
    n = numbers.get(run_id)
    return f"R{n}" if n is not None else run_id[:16]
