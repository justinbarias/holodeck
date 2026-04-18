"""Dash app — shell + navigation + callback registration.

Entry point: `python -m holodeck.dashboard --port 8501 --host 127.0.0.1`
from the CLI (`holodeck test view`) via subprocess boundary.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict
from functools import lru_cache

import dash
from dash import Input, Output, State, callback, ctx, dcc, html

from holodeck.dashboard import state as state_mod
from holodeck.dashboard.data_loader import load_runs_for_app, to_summary_dataframe
from holodeck.dashboard.filters import Filters
from holodeck.dashboard.filters import apply as apply_filters
from holodeck.dashboard.views import render_compare, render_explorer, render_summary

AGENT_DISPLAY_NAME = os.environ.get(
    "HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME", "customer-support"
)
AGENT_NAME = os.environ.get("HOLODECK_DASHBOARD_AGENT_NAME", "customer-support")

app = dash.Dash(
    __name__,
    title="HoloDeck · Evaluation Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    assets_folder="assets",
)


@lru_cache(maxsize=1)
def _runs_cached(cache_key: str) -> list:
    return load_runs_for_app()


def _cache_key() -> str:
    return f"{os.environ.get('HOLODECK_DASHBOARD_USE_SEED', '')}::{os.environ.get('HOLODECK_DASHBOARD_RESULTS_DIR', '')}"


def get_runs() -> list:
    return _runs_cached(_cache_key())


def _topbar() -> html.Header:
    runs = get_runs()
    n = len(runs)
    mode = (
        "seed · data.js"
        if os.environ.get("HOLODECK_DASHBOARD_USE_SEED") == "1"
        else "live"
    )

    return html.Header(
        [
            html.Div(
                [
                    html.Div("▶", className="logo-mark"),
                    html.Div(
                        [
                            html.Div("HoloDeck", className="brand-name"),
                            html.Div("TEST VIEW", className="brand-sub"),
                        ],
                        className="brand",
                    ),
                ],
                className="topbar-left",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className="dot"),
                            html.Span("experiment", className="meta"),
                            html.Span(AGENT_DISPLAY_NAME, className="name"),
                            html.Span("▾", className="chev"),
                        ],
                        className="exp-picker",
                    ),
                    html.Span(
                        [
                            html.Span(f"{n}", className="mono fg"),
                            f" runs · results/{AGENT_NAME}/",
                        ],
                        className="mono",
                        style={"fontSize": "12px", "color": "var(--hd-muted)"},
                    ),
                ],
                className="topbar-center",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className="ind"),
                            html.Span(f"dash · {mode}"),
                        ],
                        className="warn",
                    ),
                    html.Button(["⤓ Download run"], className="hd-btn hd-btn-ghost"),
                ],
                className="topbar-right",
            ),
        ],
        className="topbar",
    )


def _tabbar(active: str, summary_count: int, compare_queue_n: int) -> html.Nav:
    def tab(label: str, value: str, count_text: str | None) -> html.Button:
        children: list = [label]
        if count_text is not None:
            children.append(html.Span(count_text, className="count"))
        cls = "tab active" if active == value else "tab"
        return html.Button(
            children, className=cls, id={"type": "tab-btn", "value": value}, n_clicks=0
        )

    compare_count = f"{compare_queue_n}/3" if compare_queue_n else "—"

    return html.Nav(
        [
            tab("Summary", "summary", str(summary_count)),
            tab("Explorer", "explorer", str(summary_count)),
            tab("Compare", "compare", compare_count),
            html.Div(
                [
                    html.Span(
                        "launched via",
                        style={"color": "var(--hd-muted)", "fontSize": "12px"},
                    ),
                    html.Span(
                        [
                            html.Span("$ ", className="p"),
                            html.Span("holodeck test view agent.yaml"),
                        ],
                        className="cmd-chip",
                    ),
                ],
                className="tabbar-right",
            ),
        ],
        className="tabbar",
    )


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(
            id="app-state", storage_type="memory", data=state_mod.default_state()
        ),
        html.Div(id="topbar-container"),
        html.Div(id="tabbar-container"),
        html.Main(id="view-container"),
    ],
    className="app",
)


@callback(
    Output("topbar-container", "children"),
    Output("tabbar-container", "children"),
    Output("view-container", "children"),
    Input("app-state", "data"),
)
def render_all(state):
    state = state or state_mod.default_state()
    runs = get_runs()
    tab = state.get("tab", "summary")
    compare_n = len(state.get("compare_queue") or [])
    filtered = apply_filters(state_mod.state_filters(state), runs)
    if tab == "explorer":
        view = render_explorer(state, runs)
    elif tab == "compare":
        view = render_compare(state, runs)
    else:
        view = render_summary(state, runs, agent_display_name=AGENT_DISPLAY_NAME)
    return _topbar(), _tabbar(tab, len(filtered), compare_n), view


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
    Input({"type": "queue-btn", "run_id": dash.ALL}, "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def on_queue_toggle(_clicks, state):
    state = state or state_mod.default_state()
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise dash.exceptions.PreventUpdate
    run_id = ctx.triggered_id.get("run_id")
    queue = list(state.get("compare_queue") or [])
    if run_id in queue:
        return state_mod.remove_from_compare_queue(state, run_id)
    return state_mod.push_to_compare_queue(state, run_id)


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


def main(host: str = "127.0.0.1", port: int = 8501, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
