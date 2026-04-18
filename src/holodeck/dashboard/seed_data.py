"""Seed data for dashboard development.

Loads the golden fixture `tests/fixtures/dashboard/seed_runs.json` — a
snapshot of the handoff's `data.js` dataset (24 runs × 12 cases × 7 prompt
versions × trajectory 0.58→0.93) deserialised into `EvalRun` instances.

The fixture lets every UI task iterate without real `results/<slug>/` files
on disk; it also backs `AppTest`-style smoke tests.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from holodeck.models.eval_run import EvalRun

SEED_AGENT_DISPLAY_NAME = "customer-support"


def _fixture_path() -> Path:
    """Resolve the repo-committed seed fixture.

    Walks up from this file until we find `tests/fixtures/dashboard/seed_runs.json`.
    This works in both source checkouts and the wheel-installed layout (where
    `tests/` sits alongside `src/`).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "fixtures" / "dashboard" / "seed_runs.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "seed_runs.json not found; expected under tests/fixtures/dashboard/"
    )


@lru_cache(maxsize=1)
def build_seed_runs() -> list[EvalRun]:
    """Return the 24-run seed dataset as validated EvalRun instances."""
    raw = json.loads(_fixture_path().read_text(encoding="utf-8"))
    return [EvalRun.model_validate(item) for item in raw]


SEED_CONVERSATIONS: dict[str, dict] = {
    # Populated in US5 (Explorer view) — handoff data.js:160–178.
    # Left empty here; Summary view does not consume it.
}
