"""Floating compare tray — visible across every view when the queue isn't empty.

Mounted as a sibling of ``#view-container`` in ``app.py`` so it survives
view switches. Chips carry the run's cue number and its revision colour.
"""

from __future__ import annotations

from typing import Any

from dash import html

from holodeck.dashboard.components.compare_add_button import MAX_QUEUE
from holodeck.dashboard.components.ui import icon
from holodeck.dashboard.revisions import (
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.models.eval_run import EvalRun


def render_tray(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    queue_ids = list(state.get("compare_queue") or [])
    runs_by_id = {r.report.timestamp: r for r in runs}
    items = [runs_by_id[rid] for rid in queue_ids if rid in runs_by_id]
    colours = revision_colour_map(runs)
    numbers = run_numbers(runs)
    on_compare = state.get("tab") == "compare"

    chips: list[Any] = []
    for i, run in enumerate(items):
        version = run.metadata.prompt_version.version
        colour = revision_colour(version, colours)
        label = run_label(run, numbers)
        chips.append(
            html.Span(
                [
                    html.Span(f"{label} · {version}", className="cue"),
                    *([html.Span("baseline", className="tray-base")] if i == 0 else []),
                    html.Button(
                        icon("x"),
                        id={
                            "type": "compare-slot-remove",
                            "run_id": run.report.timestamp,
                        },
                        className="tray-remove",
                        n_clicks=0,
                        **{"aria-label": f"Remove {label} from compare"},
                    ),
                ],
                className="tray-chip",
                style={"background": colour.tint, "color": colour.ink},
            )
        )

    hint = (
        "Add one more run to compare"
        if len(items) == 1
        else f"{len(items)} of {MAX_QUEUE}"
    )

    # Visibility follows the queue; the default style keeps the ids mounted.
    display = "flex" if items else "none"

    return html.Div(
        [
            html.Span("Compare", className="tray-title"),
            html.Div(chips, className="tray-chips"),
            html.Span(hint, className="tray-hint"),
            html.Button(
                "Clear",
                id="compare-tray-clear",
                className="btn-text btn-text-house",
                n_clicks=0,
            ),
            html.Button(
                "Open compare",
                id="compare-tray-open",
                className="btn btn-on-house",
                n_clicks=0,
                disabled=len(items) < 2,
                # On Compare the toolbar already shows these runs; keep the id
                # mounted for its callback but hide the duplicate action.
                hidden=on_compare,
                style={"display": "none"} if on_compare else {},
            ),
        ],
        id="compare-tray",
        className="cmp-tray",
        role="region",
        style={"display": display},
        **{"aria-label": "Compare queue"},
    )
