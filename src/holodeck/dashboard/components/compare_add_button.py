"""Shared add-to-compare toggle.

Mounted on every row that shows a run (Summary runs table, Explorer runs
column). A single pattern-match callback wired in ``app.py`` drives every
instance.

* not in queue, queue has room → plus icon, "Add Rn to compare"
* not in queue, queue full    → plus icon, disabled
* in queue                    → check icon, pressed, "Remove Rn from compare"
"""

from __future__ import annotations

from dash import html

from holodeck.dashboard.components.ui import icon

MAX_QUEUE = 3


def render_add_button(
    run_id: str, queue: list[str], run_label: str | None = None
) -> html.Button:
    in_queue = run_id in queue
    full = len(queue) >= MAX_QUEUE and not in_queue
    name = run_label or run_id[:16]

    cls = "cmp-add"
    if in_queue:
        cls += " on"

    if in_queue:
        slot = queue.index(run_id) + 1
        label = f"Remove {name} from compare"
        title = "Baseline in compare" if slot == 1 else f"In compare, slot {slot}"
    elif full:
        label = f"Add {name} to compare (compare holds {MAX_QUEUE} runs)"
        title = f"Compare holds {MAX_QUEUE} runs; remove one first"
    else:
        label = f"Add {name} to compare"
        title = "Add to compare"

    return html.Button(
        icon("check" if in_queue else "plus"),
        id={"type": "compare-add", "run_id": run_id},
        className=cls,
        title=title,
        n_clicks=0,
        disabled=full,
        **{"aria-label": label, "aria-pressed": "true" if in_queue else "false"},
    )
