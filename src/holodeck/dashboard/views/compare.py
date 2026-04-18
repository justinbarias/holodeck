"""Compare view (US5 — stub in US4)."""

from __future__ import annotations

from typing import Any

from dash import html

from holodeck.models.eval_run import EvalRun


def render_compare(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    return html.Div(
        [
            html.Div("⧉", className="icon"),
            html.H2("Compare"),
            html.P(
                "Pick up to 3 runs; side-by-side, baseline+deltas, or "
                "matrix-first. Ships in US5."
            ),
        ],
        className="hd-stub",
    )
