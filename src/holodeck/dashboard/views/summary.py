"""Summary view — KPI strip, pass-rate panel, metric trend, breakdowns, runs table.

Layout matches the handoff's Summary tab (see `summary.js` + `Evaluation
Dashboard.html`). `render_summary` returns a tree of plain `html.*`/`dcc.*`
nodes; data-driven children come from `data_loader` + `charts`.

NOTE (T357 intentional deviation): the "Median duration" KPI card renders a
true median via `statistics.median`. The prototype's `summary.js:123,138`
label the card "Median" but compute a mean — that's a bug in the prototype.
We honor the label.
"""

from __future__ import annotations

import statistics
from typing import Any

from dash import dcc, html

from holodeck.dashboard import charts
from holodeck.dashboard.components.compare_add_button import render_add_button
from holodeck.dashboard.data_loader import (
    distinct_values,
    to_breakdown_dataframe,
    to_summary_dataframe,
)
from holodeck.dashboard.filters import apply as apply_filters
from holodeck.dashboard.state import state_filters
from holodeck.models.eval_run import EvalRun


def _empty_state() -> html.Div:
    return html.Div(
        [
            html.Div("∅", className="icon"),
            html.Div("No runs match your filters"),
            html.Div(
                [
                    "Clear filters or run ",
                    html.Code("holodeck test"),
                    " to produce a result.",
                ],
                style={"marginTop": "8px", "fontSize": "12px"},
            ),
        ],
        className="empty",
    )


# ----- Filter rail --------------------------------------------------------


def _chip_row(
    chip_id_prefix: str,
    options: list[str],
    selected: list[str],
) -> html.Div:
    chips = []
    for opt in options:
        cls = "chip on" if opt in selected else "chip"
        chips.append(
            html.Div(
                opt,
                className=cls,
                id={"type": chip_id_prefix, "value": opt},
                n_clicks=0,
            )
        )
    return html.Div(chips, className="chip-row")


def _filter_rail(runs: list[EvalRun], state: dict[str, Any]) -> html.Aside:
    filters = state_filters(state)
    versions = distinct_values(runs, "prompt_version")
    models = distinct_values(runs, "model_name")
    tags = distinct_values(runs, "tags")

    # Share URL (live)
    url_search = ""
    from holodeck.dashboard.state import url_search_from_state

    url_search = url_search_from_state(state) or "— (no active filters)"

    # Min pass rate slider (value display)
    pct = int(filters.min_pass_rate * 100)

    groups = [
        # Date range (static placeholder — populated from filters.date_from/to)
        html.Div(
            [
                html.Div("Date range", className="rail-label"),
                html.Div(
                    [
                        html.Span(
                            "Mar 7, 2026", id="filter-date-from", className="date-field"
                        ),
                        html.Span(
                            "Apr 18, 2026", id="filter-date-to", className="date-field"
                        ),
                    ],
                    style={"display": "flex", "gap": "6px", "flexDirection": "column"},
                ),
            ],
            className="rail-group",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Prompt version"),
                        html.Span(
                            (
                                ",".join(filters.prompt_versions)
                                if filters.prompt_versions
                                else "any"
                            ),
                            className="value",
                        ),
                    ],
                    className="rail-label",
                ),
                _chip_row("chip-version", versions, filters.prompt_versions),
            ],
            className="rail-group",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Model"),
                        html.Span(
                            (
                                ",".join(filters.model_names)
                                if filters.model_names
                                else "any"
                            ),
                            className="value",
                        ),
                    ],
                    className="rail-label",
                ),
                _chip_row("chip-model", models, filters.model_names),
            ],
            className="rail-group",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Pass-rate threshold"),
                        html.Span(f"≥ {pct}%", className="value"),
                    ],
                    className="rail-label",
                ),
                dcc.Slider(
                    id="filter-min-pass",
                    min=0,
                    max=100,
                    step=5,
                    value=pct,
                    marks={0: "0%", 100: "100%"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            className="rail-group",
        ),
    ]

    if tags:
        groups.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Frontmatter tags"),
                            html.Span(
                                ",".join(filters.tags) if filters.tags else "any",
                                className="value",
                            ),
                        ],
                        className="rail-label",
                    ),
                    _chip_row(
                        "chip-tag",
                        [f"#{t}" for t in tags],
                        [f"#{t}" for t in filters.tags],
                    ),
                ],
                className="rail-group",
            )
        )

    rail_filters = html.Div(
        [
            html.Div(
                [
                    html.H4("Filters"),
                    html.Button("Reset", id="filter-reset", className="reset-btn"),
                ],
                className="rail-footer",
            ),
            *groups,
        ],
        className="rail-card",
    )

    rail_share = html.Div(
        [
            html.H4("Share"),
            html.Div(
                url_search,
                id="filter-url-preview",
                className="mono",
                style={
                    "fontSize": "11px",
                    "padding": "8px 10px",
                    "background": "#050b09",
                    "border": "1px solid var(--hd-border)",
                    "borderRadius": "6px",
                    "color": "var(--hd-accent)",
                    "wordBreak": "break-all",
                    "marginBottom": "10px",
                },
            ),
            html.Button("Copy URL", id="filter-copy-url", className="reset-btn"),
            dcc.Clipboard(
                target_id="filter-url-preview",
                title="copy",
                style={"display": "inline-block", "marginLeft": "8px"},
            ),
        ],
        className="rail-card",
    )

    return html.Aside([rail_filters, rail_share], className="rail")


# ----- KPI strip ----------------------------------------------------------


def _fmt_delta(curr: float, prev: float | None) -> tuple[str, str]:
    if prev is None:
        return ("", "neutral")
    d = curr - prev
    sign = "▲" if d >= 0 else "▼"
    return (f"{sign} {abs(d) * 100:.1f} pp vs prior", "pos" if d >= 0 else "neg")


def _kpi_strip(runs: list[EvalRun]) -> html.Div:
    if not runs:
        return html.Div(className="kpi-strip")

    sorted_runs = sorted(runs, key=lambda r: r.report.timestamp)
    pr_series = [
        (
            (r.report.summary.passed / r.report.summary.total_tests)
            if r.report.summary.total_tests > 0
            else 0.0
        )
        for r in sorted_runs
    ]
    latest_pr = pr_series[-1]
    prev_pr = pr_series[-2] if len(pr_series) >= 2 else None
    delta_str, delta_cls = _fmt_delta(latest_pr, prev_pr)

    # G-Eval avg per run (use most recent)
    geval_per_run: list[float] = []
    for run in sorted_runs:
        scores = [
            float(m.score)
            for case in run.report.results
            for m in case.metric_results
            if m.kind == "geval"
        ]
        if scores:
            geval_per_run.append(sum(scores) / len(scores))
    latest_geval = geval_per_run[-1] if geval_per_run else 0.0
    geval_spark = geval_per_run[-8:] if geval_per_run else []

    # Median duration (honors label per T357)
    durations = [
        r.report.summary.total_duration_ms
        for r in sorted_runs
        if r.report.summary.total_duration_ms
    ]
    median_ms = statistics.median(durations) if durations else 0
    median_s = median_ms / 1000.0

    def _kpi_card(
        label: str, value_children, *, delta=None, caption=None, spark_fig=None
    ):
        body = [
            html.Div(label, className="kpi-label"),
            html.Div(value_children, className="kpi-value"),
        ]
        if delta is not None:
            body.append(delta)
        if caption is not None:
            body.append(
                html.Div(
                    caption,
                    className="mono",
                    style={"fontSize": "11px", "color": "var(--hd-muted)"},
                )
            )
        if spark_fig is not None:
            body.append(
                dcc.Graph(
                    figure=spark_fig,
                    config={"displayModeBar": False, "staticPlot": True},
                    className="kpi-spark",
                    style={"height": "34px", "width": "90px"},
                )
            )
        return html.Div(body, className="kpi")

    return html.Div(
        [
            _kpi_card(
                "Latest pass rate",
                [html.Span(f"{latest_pr * 100:.1f}%")],
                delta=(
                    html.Div(delta_str, className=f"kpi-delta {delta_cls}")
                    if delta_str
                    else None
                ),
                spark_fig=charts.sparkline(pr_series[-8:]),
            ),
            _kpi_card(
                "Runs (filtered)",
                [html.Span(str(len(runs))), html.Span("runs", className="kpi-unit")],
                caption="6 wks",
            ),
            _kpi_card(
                "Avg G-Eval score",
                [
                    html.Span(f"{latest_geval:.2f}"),
                    html.Span("/ 1.00", className="kpi-unit"),
                ],
                spark_fig=charts.sparkline(geval_spark) if geval_spark else None,
            ),
            _kpi_card(
                "Median duration",
                [html.Span(f"{median_s:.1f}s")],
                caption="per run",
            ),
        ],
        className="kpi-strip",
    )


# ----- Pass-rate + metric-trend panels ------------------------------------


def _pass_rate_panel(runs: list[EvalRun], agent_display_name: str) -> html.Div:
    n = len(runs)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("TRENDS", className="eyebrow"),
                            html.H3("Pass rate over time"),
                            html.P(
                                [
                                    f"{n} runs of ",
                                    html.Span(
                                        agent_display_name,
                                        style={"color": "var(--fg1)"},
                                    ),
                                    " · regressions flagged in coral · dashed lines mark prompt-version boundaries",
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                [
                                    html.Span(
                                        className="legend-swatch line",
                                        style={
                                            "background": "#7bff5a",
                                            "width": "12px",
                                        },
                                    ),
                                    "pass rate",
                                ],
                                className="legend-item",
                            ),
                            html.Span(
                                [
                                    html.Span(
                                        className="legend-swatch",
                                        style={"background": "#ff9d7e"},
                                    ),
                                    "regression",
                                ],
                                className="legend-item",
                            ),
                        ],
                        className="legend",
                    ),
                ],
                className="panel-head",
            ),
            dcc.Graph(
                id="chart-pass-rate",
                figure=charts.pass_rate_chart(runs),
                config={"displayModeBar": False},
                style={"height": "280px"},
            ),
        ],
        className="panel",
    )


def _metric_trend_panel(runs: list[EvalRun], kind: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("METRIC TRENDS", className="eyebrow"),
                            html.H3("Per-metric average scores"),
                            html.P(
                                [
                                    "Mean metric score per run, grouped by kind. Threshold line at ",
                                    html.Span("0.7", style={"color": "var(--fg1)"}),
                                    ".",
                                ]
                            ),
                        ]
                    ),
                    dcc.RadioItems(
                        id="metric-kind",
                        options=[
                            {"label": "rag", "value": "rag"},
                            {"label": "geval", "value": "geval"},
                            {"label": "standard", "value": "standard"},
                        ],
                        value=kind,
                        inline=True,
                        className="hd-segmented",
                    ),
                ],
                className="panel-head",
            ),
            dcc.Graph(
                id="chart-metric-trend",
                figure=charts.metric_trend_chart(runs, kind),  # type: ignore[arg-type]
                config={"displayModeBar": False},
                style={"height": "260px"},
            ),
        ],
        className="panel",
    )


# ----- Breakdown panels (name | bar | score rows) -------------------------


def _metric_row(name: str, avg: float, threshold: float = 0.7) -> html.Div:
    pct = max(0.0, min(avg, 1.0)) * 100
    fail = avg < threshold
    return html.Div(
        [
            html.Div(name, className="metric-name"),
            html.Div(
                [
                    html.Div(className="fill", style={"width": f"{pct}%"}),
                    html.Div(className="thresh", style={"left": f"{threshold * 100}%"}),
                ],
                className="metric-bar",
            ),
            html.Div(f"{avg:.2f}", className=f"metric-score{' fail' if fail else ''}"),
        ],
        className="metric-row",
    )


def _breakdown_panel(
    eyebrow: str, title: str, desc: str, kind: str, runs: list[EvalRun]
) -> html.Div:
    df = to_breakdown_dataframe(runs, kind, recent_n=6)  # type: ignore[arg-type]
    rows = [
        _metric_row(row["metric_name"], float(row["avg_score"]))
        for _, row in df.iterrows()
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(eyebrow, className="eyebrow"),
                    html.H3(title),
                    html.P(desc),
                ],
                className="panel-head",
            ),
            html.Div(rows) if rows else html.Div("no data", className="mono"),
        ],
        className="panel",
    )


def _breakdown_row(runs: list[EvalRun]) -> html.Div:
    return html.Div(
        [
            _breakdown_panel(
                "BREAKDOWN · STANDARD",
                "NLP metrics",
                "BLEU / ROUGE / METEOR — last 6 runs, avg across test cases.",
                "standard",
                runs,
            ),
            _breakdown_panel(
                "BREAKDOWN · RAG",
                "Retrieval & grounding",
                "Faithfulness, relevancy, precision, recall — averaged across recent runs.",
                "rag",
                runs,
            ),
            _breakdown_panel(
                "BREAKDOWN · G-EVAL",
                "Custom LLM judges",
                "Per-name custom G-Eval rubrics defined in agent.yaml.",
                "geval",
                runs,
            ),
        ],
        className="breakdowns",
    )


# ----- Runs table (html.Table with inline bar cell) -----------------------


def _pill_class(tier: str) -> str:
    return {"pass": "pill-pass", "warn": "pill-warn", "fail": "pill-fail"}.get(
        tier, "pill-neutral"
    )


def _runs_table(
    runs: list[EvalRun], compare_queue: list[str], search: str = ""
) -> html.Div:
    df = to_summary_dataframe(runs)
    if df.empty:
        return html.Div()

    header = html.Thead(
        html.Tr(
            [
                html.Th(""),
                html.Th("Timestamp"),
                html.Th("Pass rate"),
                html.Th("Tests"),
                html.Th("Prompt"),
                html.Th("Model"),
                html.Th("Duration"),
                html.Th("Commit"),
            ]
        )
    )

    body_rows = []
    for _, row in df.iterrows():
        pr = row["pass_rate"]
        tier = row["pass_rate_tier"]
        ts = row["timestamp"]
        run_id = row["id"]
        date_s = ts.strftime("%-d %b") if hasattr(ts, "strftime") else str(ts)
        time_s = ts.strftime("%H:%M") if hasattr(ts, "strftime") else ""

        def _nav_td(rid: str, col: str, children, **kwargs):
            return html.Td(
                children,
                id={"type": "run-row", "run_id": rid, "col": col},
                n_clicks=0,
                className="run-nav",
                **kwargs,
            )

        body_rows.append(
            html.Tr(
                [
                    html.Td(
                        render_add_button(run_id, list(compare_queue)),
                        style={"width": "44px", "textAlign": "center"},
                    ),
                    _nav_td(
                        run_id,
                        "ts",
                        html.Span(
                            [html.Span(date_s, className="date"), html.Span(time_s)],
                            className="ts",
                        ),
                    ),
                    _nav_td(
                        run_id,
                        "pr",
                        html.Div(
                            [
                                html.Span(
                                    f"{pr * 100:.1f}%",
                                    className=f"pill {_pill_class(tier)}",
                                ),
                                html.Div(
                                    html.Div(
                                        className="fill",
                                        style={"width": f"{pr * 100:.1f}%"},
                                    ),
                                    className="bar",
                                ),
                            ],
                            className="bar-inline",
                        ),
                    ),
                    _nav_td(
                        run_id,
                        "tests",
                        html.Span(
                            [
                                html.Span(str(int(row["passed"])), className="mono fg"),
                                "/",
                                html.Span(str(int(row["total"])), className="mono"),
                            ]
                        ),
                    ),
                    _nav_td(
                        run_id,
                        "prompt",
                        html.Span(
                            row["prompt_version"],
                            className="mono",
                            style={"color": "var(--hd-accent)"},
                        ),
                    ),
                    _nav_td(
                        run_id, "model", html.Span(row["model_name"], className="mono")
                    ),
                    _nav_td(
                        run_id,
                        "duration",
                        html.Span(
                            (
                                f"{row['duration_ms'] / 1000:.1f}s"
                                if row["duration_ms"]
                                else "—"
                            ),
                            className="mono",
                        ),
                    ),
                    _nav_td(
                        run_id,
                        "commit",
                        html.Span((row["git_commit"] or "")[:7], className="mono"),
                    ),
                ]
            )
        )

    table = html.Table([header, html.Tbody(body_rows)], className="runs")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("RUNS", className="eyebrow"),
                            html.H3(
                                [
                                    "All runs in view ",
                                    html.Span(
                                        str(len(runs)),
                                        className="mono",
                                        style={"color": "var(--hd-accent)"},
                                    ),
                                ]
                            ),
                        ],
                        className="table-title",
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="runs-search",
                                type="text",
                                placeholder="filter runs…",
                                value=search,
                                debounce=True,
                                className="table-search",
                            ),
                            html.Button(
                                "Export CSV",
                                id="runs-export",
                                className="hd-btn hd-btn-ghost",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "10px",
                            "alignItems": "center",
                        },
                    ),
                ],
                className="table-head",
            ),
            html.Div(table, style={"overflowX": "auto", "padding": "0 10px 10px"}),
            dcc.Download(id="runs-csv-download"),
        ],
        className="table-card",
    )


# ----- Entry point --------------------------------------------------------


def render_summary(
    state: dict[str, Any],
    runs: list[EvalRun],
    agent_display_name: str = "customer-support",
) -> html.Div:
    filters = state_filters(state)
    filtered = apply_filters(filters, runs)
    compare_queue = list(state.get("compare_queue") or [])

    if not runs:
        return html.Div(
            [
                _filter_rail(runs, state),
                html.Div(_empty_state(), className="main"),
            ],
            className="layout",
        )

    main_children: list[Any]
    if not filtered:
        main_children = [_empty_state()]
    else:
        main_children = [
            _kpi_strip(filtered),
            _pass_rate_panel(filtered, agent_display_name),
            _metric_trend_panel(filtered, filters.metric_kind),
            _breakdown_row(filtered),
            _runs_table(filtered, compare_queue, search=filters.search),
        ]

    return html.Div(
        [
            _filter_rail(runs, state),
            html.Div(main_children, className="main"),
        ],
        className="layout",
    )
