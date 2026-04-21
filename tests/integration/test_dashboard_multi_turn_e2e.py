"""End-to-end dashboard rendering against the multi-turn seed (US5 T023)."""

from __future__ import annotations

import json

import pytest
from dash import html

from holodeck.dashboard.seed_data import build_multi_turn_seed_case
from holodeck.dashboard.views.explorer import render_explorer


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for c in children:
            if isinstance(c, (str, int, float)) or c is None:
                continue
            yield from _walk(c)
    elif not isinstance(children, (str, int, float)):
        yield from _walk(children)


def _deep_plotly(node):
    if isinstance(node, (list, tuple)):
        return [_deep_plotly(c) for c in node]
    if hasattr(node, "to_plotly_json"):
        d = node.to_plotly_json()
        props = dict(d.get("props", {}) or {})
        if "children" in props:
            props["children"] = _deep_plotly(props["children"])
        d = dict(d)
        d["props"] = props
        return d
    return node


@pytest.mark.integration
def test_dashboard_renders_multi_turn_run() -> None:
    run = build_multi_turn_seed_case()
    # Select the failing 4-turn case so we can assert all three properties.
    state = {
        "explorer_run_id": run.report.timestamp,
        "explorer_case_name": "four_turn_math_failing",
        "explorer_runs_collapsed": False,
    }
    tree = render_explorer(state, [run])

    # (a) cases column has the turn-count chip for the 4-turn case.
    full_json = json.dumps(_deep_plotly(tree), default=str, ensure_ascii=False)
    assert "4 turns · 3/4 passed" in full_json
    assert "3 turns · 3/3 passed" in full_json

    # (b) detail pane's conversation section has 4 per-turn Details blocks.
    assert len(tree.children) == 3
    detail = tree.children[2]
    per_turn_details = [
        n
        for n in _walk(detail)
        if isinstance(n, html.Details)
        and "hd-explorer-turn" in (getattr(n, "className", "") or "")
    ]
    assert len(per_turn_details) == 4

    # (c) the failing turn-2 block surfaces error text via the existing helpers.
    detail_json = json.dumps(_deep_plotly(detail), default=str, ensure_ascii=False)
    assert "expected tool(s) not called in this turn: subtract" in detail_json
    # Failing metric reasoning routes through _metric_row_div.
    assert "Expected 25587, got 65587" in detail_json
