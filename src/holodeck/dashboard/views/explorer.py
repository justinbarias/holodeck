"""Explorer view (US5 — stub in US4)."""

from __future__ import annotations

from typing import Any

from dash import html

from holodeck.models.eval_run import EvalRun


def render_explorer(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    return html.Div(
        [
            html.Div("◎", className="icon"),
            html.H2("Explorer"),
            html.P(
                "Three-column drilldown — runs → cases → detail with "
                "conversation thread + tool-call traces. Ships in US5."
            ),
        ],
        className="hd-stub",
    )
