"""Dash app — shell + navigation + callback registration.

Entry point: `python -m holodeck.dashboard --port 8501 --host 127.0.0.1`
from the CLI (`holodeck test view`) via subprocess boundary.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict
from pathlib import Path

import dash
from dash import Input, Output, State, callback, ctx, dcc, html

from holodeck.dashboard import state as state_mod
from holodeck.dashboard.compare_data import (
    quick_pick_best_worst,
    quick_pick_latest,
    quick_pick_latest_per_revision,
)
from holodeck.dashboard.components.compare_tray import render_tray
from holodeck.dashboard.components.rail import render_rail
from holodeck.dashboard.data_loader import load_runs_for_app, to_summary_dataframe
from holodeck.dashboard.filters import Filters
from holodeck.dashboard.filters import apply as apply_filters
from holodeck.dashboard.loaded_run import LoadedRun
from holodeck.dashboard.revisions import chronological
from holodeck.dashboard.views import render_compare, render_explorer, render_summary
from holodeck.dashboard.views.compare import render_compare_header
from holodeck.dashboard.views.summary import render_summary_header

AGENT_DISPLAY_NAME = os.environ.get(
    "HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME", "customer-support"
)
AGENT_NAME = os.environ.get("HOLODECK_DASHBOARD_AGENT_NAME", "customer-support")

app = dash.Dash(
    __name__,
    title="HoloDeck test view",
    update_title=None,
    suppress_callback_exceptions=True,
    assets_folder="assets",
)


# mtime-aware memo so newly-written results files surface without restart.
# Keyed on the env-var pair that determines the source; value is
# ``(fingerprint, runs)``. Seed mode pins the fingerprint to 0.0 so the cache
# never busts (deterministic fixture).
_runs_memo: dict[str, tuple[float, list[LoadedRun]]] = {}


def _cache_key() -> str:
    return f"{os.environ.get('HOLODECK_DASHBOARD_USE_SEED', '')}::{os.environ.get('HOLODECK_DASHBOARD_RESULTS_DIR', '')}"


def _results_dir_fingerprint() -> float:
    """Max mtime across the results dir and its ``*.json`` children.

    Returns 0.0 for seed mode, missing dir, or empty dir — callers treat any
    change from the previously-seen fingerprint as "reload needed".
    """
    if os.environ.get("HOLODECK_DASHBOARD_USE_SEED") == "1":
        return 0.0
    results_dir = os.environ.get("HOLODECK_DASHBOARD_RESULTS_DIR")
    if not results_dir:
        return 0.0
    path = Path(results_dir)
    if not path.is_dir():
        return 0.0
    candidates: list[Path] = [path, *path.glob("*.json")]
    return max((p.stat().st_mtime for p in candidates), default=0.0)


def get_runs() -> list[LoadedRun]:
    key = _cache_key()
    fp = _results_dir_fingerprint()
    cached = _runs_memo.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1]
    runs = load_runs_for_app()
    _runs_memo[key] = (fp, runs)
    return runs


def _seed_mode() -> bool:
    return os.environ.get("HOLODECK_DASHBOARD_USE_SEED") == "1"


def _rail(state: dict, runs: list[LoadedRun], tab: str, n_filtered: int) -> html.Aside:
    return render_rail(
        state,
        runs,
        active=tab,
        counts={
            "summary": n_filtered,
            "explorer": len(runs),
            "compare": len(state.get("compare_queue") or []),
        },
        agent_display_name=AGENT_DISPLAY_NAME,
        agent_name=AGENT_NAME,
        seed_mode=_seed_mode(),
    )


def _page_header(state: dict, runs: list[LoadedRun], tab: str) -> html.Header | None:
    if tab == "summary":
        return render_summary_header(state, runs)
    if tab == "compare":
        return render_compare_header(state, runs)
    # Explorer's heading is the selected case name, inside the detail panel.
    return None


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(
            id="app-state", storage_type="memory", data=state_mod.default_state()
        ),
        # Poll the results dir for new files every 5s. Fires n_intervals into
        # `render_all` which already calls get_runs() — cheap no-op when the
        # fingerprint is unchanged.
        dcc.Interval(id="runs-poll", interval=5000, n_intervals=0),
        html.A("Skip to content", href="#view-container", className="skip-link"),
        # The rail (brand, agent card, views, filters) renders into
        # #topbar-container; the page header renders into #tabbar-container.
        html.Div(id="topbar-container", className="rail-slot"),
        html.Div(id="tabbar-container", className="header-slot"),
        html.Main(id="view-container", tabIndex="-1"),
        # Compare tray floats across every view; sibling of #view-container so
        # it survives tab-switch re-renders (T442).
        html.Div(id="compare-tray-container"),
    ],
    className="app",
)


@callback(
    Output("topbar-container", "children"),
    Output("tabbar-container", "children"),
    Output("view-container", "children"),
    Output("compare-tray-container", "children"),
    Input("app-state", "data"),
    Input("runs-poll", "n_intervals"),
)
def render_all(state, _n_intervals):
    state = state or state_mod.default_state()
    runs = get_runs()
    tab = state.get("tab", "summary")
    filtered = apply_filters(state_mod.state_filters(state), runs)
    if tab == "explorer":
        view = render_explorer(state, runs)
    elif tab == "compare":
        view = render_compare(state, runs)
    else:
        view = render_summary(state, runs, agent_display_name=AGENT_DISPLAY_NAME)
    tray = render_tray(state, runs)
    return (
        _rail(state, runs, tab, len(filtered)),
        _page_header(state, runs, tab),
        view,
        tray,
    )


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "tab-btn", "value": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_tab_click(n_clicks_list, state):
    state = state or state_mod.default_state()
    if not any(n_clicks_list or []):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if trig and isinstance(trig, dict):
        return state_mod.set_tab(state, trig.get("value", "summary"))
    raise dash.exceptions.PreventUpdate


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "chip-version", "value": dash.ALL}, "n_clicks"),
    Input({"type": "chip-model", "value": dash.ALL}, "n_clicks"),
    Input({"type": "chip-tag", "value": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_chip_click(_v, _m, _t, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    kind = ctx.triggered_id.get("type", "")
    value = ctx.triggered_id.get("value", "")
    current = state.get("filters") or asdict(Filters())

    if kind == "chip-version":
        selected = list(current.get("prompt_versions") or [])
        field = "prompt_versions"
    elif kind == "chip-model":
        selected = list(current.get("model_names") or [])
        field = "model_names"
    elif kind == "chip-tag":
        selected = list(current.get("tags") or [])
        # tags have leading '#'
        value = value.lstrip("#")
        field = "tags"
    else:
        raise dash.exceptions.PreventUpdate

    if value in selected:
        selected.remove(value)
    else:
        selected.append(value)

    new_current = {**current, field: selected}
    return {**state, "filters": new_current}


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("filter-min-pass", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_slider(min_pct, state):
    state = state or state_mod.default_state()
    current = state.get("filters") or asdict(Filters())
    return {
        **state,
        "filters": {**current, "min_pass_rate": float(min_pct or 0) / 100.0},
    }


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("metric-kind", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_metric_kind(kind, state):
    state = state or state_mod.default_state()
    current = state.get("filters") or asdict(Filters())
    return {**state, "filters": {**current, "metric_kind": kind or "rag"}}


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("filter-reset", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_reset(_n, state):
    state = state or state_mod.default_state()
    return state_mod.set_filters(state, Filters())


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("runs-search", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_runs_search(value, state):
    state = state or state_mod.default_state()
    current = state.get("filters") or asdict(Filters())
    new_search = (value or "").strip()
    if new_search == (current.get("search") or ""):
        raise dash.exceptions.PreventUpdate
    return {**state, "filters": {**current, "search": new_search}}


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("url", "search"),
    State("app-state", "data"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_url(search, state):
    """Load the view a copied link describes (and follow back/forward).

    ``sync_url`` writes the address from the state, which fires this callback
    again; when the address already matches the state there is nothing to do,
    so the two never loop. Run references resolve to loaded run ids first.
    """
    state = state or state_mod.default_state()
    restored = state_mod.resolve_run_refs(
        state_mod.state_from_url_search(search or ""), get_runs()
    )
    if state_mod.url_search_from_state(restored) == state_mod.url_search_from_state(
        state
    ):
        raise dash.exceptions.PreventUpdate
    return {**state, **restored}


@callback(
    Output("url", "search"),
    Input("app-state", "data"),
    prevent_initial_call=True,
)
def sync_url(state):
    return state_mod.url_search_from_state(state or {})


@callback(
    Output("runs-csv-download", "data"),
    Input("runs-export", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_export_csv(_n, state):
    runs = get_runs()
    filtered = apply_filters(state_mod.state_filters(state or {}), runs)
    df = to_summary_dataframe(filtered)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return {"content": buf.getvalue(), "filename": "holodeck-runs.csv"}


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input(
        {"type": "run-row", "run_id": dash.ALL, "col": dash.ALL},
        "n_clicks",
    ),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_row_click(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    return state_mod.open_in_explorer(state, run_id)


# --------------------------------------------------------------------------- #
# Explorer navigation                                                          #
# --------------------------------------------------------------------------- #


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "explorer-run-row", "run_id": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_explorer_run_click(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    return {
        **state,
        "explorer_run_id": run_id,
        # Clear case to auto-select the first case of the new run.
        "explorer_case_name": None,
    }


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "explorer-case-row", "case_name": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_explorer_case_click(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    case_name = ctx.triggered_id.get("case_name")
    return {**state, "explorer_case_name": case_name}


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("explorer-runs-toggle", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_explorer_runs_toggle(n_clicks, state):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    state = state or state_mod.default_state()
    current = bool(state.get("explorer_runs_collapsed", False))
    return {**state, "explorer_runs_collapsed": not current}


# --------------------------------------------------------------------------- #
# Compare: add button, tray, variants                                          #
# --------------------------------------------------------------------------- #


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "compare-add", "run_id": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_add(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    queue = list(state.get("compare_queue") or [])
    if run_id in queue:
        return state_mod.remove_from_compare_queue(state, run_id)
    return state_mod.push_to_compare_queue(state, run_id)


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "compare-slot-remove", "run_id": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_slot_remove(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    return state_mod.remove_from_compare_queue(state, run_id)


def _clear_queue(n_clicks, state):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    state = state or state_mod.default_state()
    return {**state, "compare_queue": []}


# One callback per button: the tray is mounted on every view but the toolbar
# Clear exists only on Compare, and Dash rejects a callback whose inputs are
# not all in the layout.
@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("compare-tray-clear", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_tray_clear(n_clicks, state):
    return _clear_queue(n_clicks, state)


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("compare-clear", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_clear(n_clicks, state):
    return _clear_queue(n_clicks, state)


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("compare-tray-open", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_open(n_clicks, state):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    state = state or state_mod.default_state()
    return state_mod.set_tab(state, "compare")


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("compare-variant", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_variant(value, state):
    if value is None:
        raise dash.exceptions.PreventUpdate
    state = state or state_mod.default_state()
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise dash.exceptions.PreventUpdate from exc
    if v == int(state.get("compare_variant", 1) or 1):
        raise dash.exceptions.PreventUpdate
    return {**state, "compare_variant": v}


def quick_pick_queue(source: str, state: dict, runs: list[LoadedRun]) -> list[str]:
    """Run ids a quick-pick button queues; fewer than two means no pick."""
    if source == "compare-quick-best":
        # "In range" means the Summary filters; never fall back to all runs.
        in_range = apply_filters(state_mod.state_filters(state), runs)
        return quick_pick_best_worst(in_range) if len(in_range) >= 2 else []
    if source == "compare-quick-revisions":
        return quick_pick_latest_per_revision(runs)
    return quick_pick_latest(runs, 3 if source == "compare-quick-3" else 2)


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("compare-quick-2", "n_clicks"),
    Input("compare-quick-3", "n_clicks"),
    Input("compare-quick-best", "n_clicks"),
    Input("compare-quick-revisions", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_compare_quick_pick(_n2, _n3, _best, _revisions, state):
    trig = ctx.triggered[0] if ctx.triggered else {}
    if not trig.get("value"):
        raise dash.exceptions.PreventUpdate
    state = state or state_mod.default_state()
    source = (trig.get("prop_id") or "").split(".")[0]
    queue = quick_pick_queue(source, state, get_runs())
    if len(queue) < 2:
        raise dash.exceptions.PreventUpdate
    return {**state, "compare_queue": queue}


# --------------------------------------------------------------------------- #
# Summary: verdict links, date range, download, share link                    #
# --------------------------------------------------------------------------- #


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input({"type": "open-case", "run_id": dash.ALL, "case_name": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_open_case(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0] if ctx.triggered else {}
    if not trigger.get("value"):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    case_name = ctx.triggered_id.get("case_name") or None
    return {
        **state_mod.open_in_explorer(state, run_id),
        "explorer_case_name": case_name,
    }


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("filter-date-range", "start_date"),
    Input("filter-date-range", "end_date"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_date_range(start_date, end_date, state):
    state = state or state_mod.default_state()
    current = state.get("filters") or asdict(Filters())
    start = (start_date or "")[:10] or None
    end = (end_date or "")[:10] or None
    if start == current.get("date_from") and end == current.get("date_to"):
        raise dash.exceptions.PreventUpdate
    return {**state, "filters": {**current, "date_from": start, "date_to": end}}


@callback(
    Output("download-latest-run-file", "data"),
    Input("download-latest-run", "n_clicks"),
    prevent_initial_call=True,
)
def on_download_latest(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    runs = get_runs()
    if not runs:
        raise dash.exceptions.PreventUpdate
    latest = chronological(runs)[-1]
    return {
        "content": latest.model_dump_json(indent=2),
        "filename": f"holodeck-run-{latest.run_id}.json",
    }


# Copies the address bar (which `sync_url` keeps in step with the filters).
app.clientside_callback(
    """
    function (n) {
        if (!n) { return window.dash_clientside.no_update; }
        if (!navigator.clipboard) { return "Copy the address bar to share this view."; }
        return navigator.clipboard.writeText(window.location.href)
            .then(function () { return "Link copied."; })
            .catch(function () { return "Copy the address bar to share this view."; });
    }
    """,
    Output("filter-copy-status", "children"),
    Input("filter-copy-url", "n_clicks"),
    prevent_initial_call=True,
)


def main(host: str = "127.0.0.1", port: int = 8501, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
