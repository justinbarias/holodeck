"""Shared `+`/slot-index compare-add button.

Mounted on every row that shows a run (Summary runs table, Explorer runs
column). A single pattern-match callback wired in ``app.py`` drives every
instance.

Behaviour matches handoff ``compare.js::CompareAddButton``:

* not in queue, queue has room → ``+`` (clickable)
* not in queue, queue full    → ``+`` disabled
* in queue                    → slot index ``1``/``2``/``3``
"""

from __future__ import annotations

from dash import html

MAX_QUEUE = 3


def render_add_button(run_id: str, queue: list[str]) -> html.Button:
    in_queue = run_id in queue
    idx = queue.index(run_id) + 1 if in_queue else None
    full = len(queue) >= MAX_QUEUE and not in_queue

    cls = "cmp-add cmp-add-sm"
    if in_queue:
        cls += " on"
    if full:
        cls += " disabled"

    if in_queue:
        label = html.Span(str(idx), className="cmp-add-num")
        title = f"In compare queue (slot {idx})"
    elif full:
        label = html.Span("+", className="cmp-add-plus")
        title = "Compare queue is full (3 max)"
    else:
        label = html.Span("+", className="cmp-add-plus")
        title = "Add to compare queue"

    return html.Button(
        label,
        id={"type": "compare-add", "run_id": run_id},
        className=cls,
        title=title,
        n_clicks=0,
        disabled=full,
    )
