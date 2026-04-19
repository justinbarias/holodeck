"""Dash dashboard app-level smoke tests (US5 T445).

Validates that the Explorer + Compare view renderers return a well-formed
html.Div tree against the seed dataset. Visual fidelity checks are covered
separately via the Phase 8 Chrome MCP sweep.
"""

from __future__ import annotations

import pytest
from dash import dcc, html

from holodeck.dashboard.seed_data import build_seed_runs


def _walk(node):
    """Depth-first yield every Component descendant (and the root)."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list | tuple):
        for c in children:
            if isinstance(c, str | int | float) or c is None:
                continue
            yield from _walk(c)
    elif not isinstance(children, str | int | float):
        yield from _walk(children)


def _classes_contain(tree, token: str) -> bool:
    for node in _walk(tree):
        cls = getattr(node, "className", None)
        if isinstance(cls, str) and token in cls.split():
            return True
    return False


def _has_component(tree, type_) -> bool:
    return any(isinstance(n, type_) for n in _walk(tree))


def _has_id(tree, id_value: str) -> bool:
    return any(getattr(node, "id", None) == id_value for node in _walk(tree))


@pytest.fixture(scope="module")
def seed_runs():
    return build_seed_runs()


# --------------------------------------------------------------------------- #
# Explorer                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_render_explorer_produces_three_columns_and_five_sections(seed_runs):
    from holodeck.dashboard.views.explorer import render_explorer

    run = seed_runs[-1]
    state = {
        "explorer_run_id": run.report.timestamp,
        "explorer_case_name": run.report.results[0].test_name,
        "explorer_runs_collapsed": False,
    }
    tree = render_explorer(state, seed_runs)
    assert "hd-explorer-grid" in (tree.className or "")
    # Three direct children: runs col, cases col, detail
    assert isinstance(tree.children, list) and len(tree.children) == 3

    # Detail panel has exactly 5 sections: case header div + 4 <details>
    # (agent config, conversation, expected tools, evaluations)
    detail = tree.children[2]
    details_count = sum(1 for n in _walk(detail) if isinstance(n, html.Details))
    # html.Details appears also inside the agent-snapshot section's raw JSON
    # drawer, so >=4 is the contract for the top-level section collapsibles.
    assert details_count >= 4


# --------------------------------------------------------------------------- #
# Compare                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("variant,cls", [(1, "cmp-v1"), (2, "cmp-v2"), (3, "cmp-v3")])
def test_render_compare_variants(seed_runs, variant, cls):
    from holodeck.dashboard.views.compare import render_compare

    queue = [r.report.timestamp for r in seed_runs[-3:]]
    state = {"compare_queue": queue, "compare_variant": variant}
    tree = render_compare(state, seed_runs)
    assert _classes_contain(tree, cls)
    # Every variant renders the shared case-matrix Plotly graph.
    assert _has_id(tree, "chart-compare-matrix")
    assert _has_component(tree, dcc.Graph)


@pytest.mark.slow
def test_render_compare_empty_state(seed_runs):
    from holodeck.dashboard.views.compare import render_compare

    state = {"compare_queue": [], "compare_variant": 1}
    tree = render_compare(state, seed_runs)
    assert _classes_contain(tree, "cmp-empty")
    # Both quick-pick CTAs present
    assert _has_id(tree, "compare-quick-2")
    assert _has_id(tree, "compare-quick-3")
