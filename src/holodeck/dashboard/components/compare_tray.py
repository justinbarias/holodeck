"""Floating compare tray — visible across every tab when the queue isn't empty.

Mounted as a sibling of ``#view-container`` in ``app.py`` so it survives
tab switches. Matches handoff ``compare.js::CompareTray``.
"""

from __future__ import annotations

from typing import Any

from dash import html

from holodeck.dashboard.compare_data import COMPARE_PALETTE
from holodeck.models.eval_run import EvalRun


def render_tray(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    queue_ids = list(state.get("compare_queue") or [])
    runs_by_id = {r.report.timestamp: r for r in runs}
    items = [runs_by_id[rid] for rid in queue_ids if rid in runs_by_id]

    slot_pills: list = []
    for i, run in enumerate(items):
        color = COMPARE_PALETTE[i] if i < len(COMPARE_PALETTE) else "var(--fg1)"
        slot_pills.append(
            html.Div(
                [
                    html.Span(
                        className="cmp-dot",
                        style={"background": color, "boxShadow": f"0 0 6px {color}"},
                    ),
                    *([html.Span("base", className="cmp-tray-base")] if i == 0 else []),
                    html.Span(
                        run.metadata.prompt_version.version,
                        className="mono fg",
                    ),
                    html.Span(
                        run.report.timestamp[:10],
                        className="mono",
                        style={"color": "var(--hd-muted)"},
                    ),
                    html.Button(
                        "×",
                        id={
                            "type": "compare-slot-remove",
                            "run_id": run.report.timestamp,
                        },
                        className="cmp-x",
                        n_clicks=0,
                    ),
                ],
                className="cmp-tray-item",
            )
        )

    for i in range(3 - len(items)):
        slot_pills.append(
            html.Div(
                html.Span(f"slot {len(items) + i + 1}", className="mono"),
                className="cmp-tray-item cmp-tray-empty",
            )
        )

    actions = html.Div(
        [
            html.Button(
                "Clear",
                id="compare-tray-clear",
                className="reset-btn",
                n_clicks=0,
            ),
            html.Button(
                "Open Compare →",
                id="compare-tray-open",
                className="hd-btn hd-btn-primary",
                n_clicks=0,
                disabled=len(items) < 2,
                style={"fontSize": "12px", "padding": "6px 14px"},
            ),
        ],
        className="cmp-tray-actions",
    )

    label = html.Div(
        [
            html.Span(
                "COMPARE QUEUE",
                className="eyebrow",
                style={"color": "var(--hd-accent-soft)"},
            ),
            html.Span(
                f"{len(items)}/3",
                className="mono",
                style={"color": "var(--hd-muted)"},
            ),
        ],
        className="cmp-tray-label",
    )

    # Visibility is toggled by the `sync_tray_visibility` callback in app.py;
    # the default style preserves the layout when the queue is empty.
    display = "flex" if items else "none"

    return html.Div(
        [label, html.Div(slot_pills, className="cmp-tray-items"), actions],
        id="compare-tray",
        className="cmp-tray",
        style={"display": display},
    )
