"""Summary view — verdict, regressions, figures, charts, breakdowns, runs.

The page opens with a sentence about the latest run in view against the run
before it and a numbered list of regressions (``verdict.py``). The numbers
match the red rings on the pass-rate chart. Runs carry cue numbers (R1…Rn)
and prompt versions carry revision colours (``revisions.py``); both are
derived from every loaded run so they never shift under a filter.

NOTE (T357): the "Median duration" figure is a true median over the runs in
view (``statistics.median``).
"""

from __future__ import annotations

import statistics
from typing import Any

from dash import dcc, html

from holodeck.dashboard import charts
from holodeck.dashboard.components.compare_add_button import render_add_button
from holodeck.dashboard.components.ui import (
    cue,
    delta_class,
    fmt_date,
    fmt_when,
    icon,
    parse_ts,
    revision_chip,
    signed,
    sr_only,
)
from holodeck.dashboard.data_loader import (
    metric_thresholds,
    shared_threshold,
    to_breakdown_dataframe,
)
from holodeck.dashboard.filters import apply as apply_filters
from holodeck.dashboard.revisions import (
    RevisionColour,
    RunNumbers,
    chronological,
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.state import state_filters
from holodeck.dashboard.verdict import (
    Regression,
    Verdict,
    build_verdict,
    find_regressions,
    pass_rate,
)
from holodeck.models.eval_run import EvalRun

KIND_LABELS: dict[str, str] = {
    "rag": "RAG",
    "geval": "G-Eval",
    "standard": "NLP",
    "code": "Code",
}

BREAKDOWN_RECENT_N = 6


def _chronological(runs: list[EvalRun]) -> list[EvalRun]:
    return chronological(runs)


def _geval_mean(run: EvalRun) -> float | None:
    scores = [
        float(m.score)
        for case in run.report.results
        for m in case.metric_results
        if m.kind == "geval"
    ]
    return sum(scores) / len(scores) if scores else None


def _open_case_button(
    run_id: str, case_name: str, children: Any, cls: str, label: str
) -> html.Button:
    return html.Button(
        children,
        id={"type": "open-case", "run_id": run_id, "case_name": case_name},
        className=cls,
        n_clicks=0,
        **{"aria-label": label},
    )


# --------------------------------------------------------------------------- #
# Header                                                                      #
# --------------------------------------------------------------------------- #


def render_summary_header(state: dict[str, Any], runs: list[EvalRun]) -> html.Header:
    filtered = apply_filters(state_filters(state), runs)
    if filtered:
        ordered = _chronological(filtered)
        first = parse_ts(ordered[0].report.timestamp)
        last = parse_ts(ordered[-1].report.timestamp)
        span = (
            fmt_date(first)
            if first.date() == last.date()
            else f"{fmt_date(first)} – {fmt_date(last)}"
        )
        sub = f"{len(filtered)} of {len(runs)} runs match · {span}"
    elif runs:
        sub = f"No runs of {len(runs)} match these filters"
    else:
        sub = "No saved runs yet"

    actions: list[Any] = [
        html.Span("holodeck test view", className="cue header-cmd"),
    ]
    if runs:
        actions.append(
            html.Button(
                [icon("download"), "Download latest run"],
                id="download-latest-run",
                className="btn",
                n_clicks=0,
            )
        )
        actions.append(dcc.Download(id="download-latest-run-file"))

    return html.Header(
        [
            html.Div([html.H1("Summary"), html.P(sub, className="page-sub")]),
            html.Div(actions, className="page-actions"),
        ],
        className="page-header",
    )


# --------------------------------------------------------------------------- #
# Verdict + regressions                                                       #
# --------------------------------------------------------------------------- #


def _verdict_block(verdict: Verdict) -> html.Div:
    detail: list[Any] = []
    for fc in verdict.failing[:2]:
        detail.append(
            html.Span(
                [
                    cue(fc.name),
                    fc.sentence_tail(verdict.label, verdict.previous_label),
                    " ",
                ]
            )
        )
    extra = len(verdict.failing) - 2
    if extra > 0:
        detail.append(f"{extra} more case{'s' if extra != 1 else ''} fail. ")
    duration = verdict.duration_sentence()
    if duration:
        detail.append(duration)

    children: list[Any] = [html.H2(verdict.headline(), className="verdict-title")]
    if detail:
        children.append(html.P(detail, className="verdict-detail"))
    if verdict.failing:
        name = verdict.failing[0].name
        children.append(
            _open_case_button(
                verdict.run_id,
                name,
                ["Open ", cue(name, strong=False), " in Explorer"],
                "link-btn",
                f"Open {name} from {verdict.label} in Explorer",
            )
        )
    else:
        children.append(
            _open_case_button(
                verdict.run_id,
                "",
                [f"Open {verdict.label} in Explorer"],
                "link-btn",
                f"Open {verdict.label} in Explorer",
            )
        )
    return html.Div(children, className="verdict-main")


def _regressions_block(
    regressions: list[Regression], colours: dict[str, RevisionColour]
) -> html.Div:
    title = html.H2(f"Regressions in range · {len(regressions)}", className="h-sm")
    if not regressions:
        return html.Div(
            [
                title,
                html.P(
                    "No run in view dropped more than four points below the run "
                    "before it.",
                    className="muted-note",
                ),
            ],
            className="verdict-regressions",
        )
    items = []
    for reg in regressions:
        first_case = reg.newly_failed[0] if reg.newly_failed else ""
        items.append(
            html.Li(
                _open_case_button(
                    reg.run_id,
                    first_case,
                    [
                        html.Span(str(reg.number), className="cue reg-num"),
                        cue(reg.label),
                        revision_chip(
                            reg.version, revision_colour(reg.version, colours)
                        ),
                        html.Span(
                            [
                                reg.rate_text(),
                                *(
                                    [" · ", cue(reg.case_text() or ""), " failed"]
                                    if reg.case_text()
                                    else []
                                ),
                            ],
                            className="reg-text",
                        ),
                    ],
                    "reg-row",
                    f"Regression {reg.number}: open {reg.label} in Explorer",
                )
            )
        )
    return html.Div(
        [title, html.Ol(items, className="reg-list")],
        className="verdict-regressions",
    )


def _verdict_section(
    verdict: Verdict,
    regressions: list[Regression],
    colours: dict[str, RevisionColour],
) -> html.Section:
    return html.Section(
        [_verdict_block(verdict), _regressions_block(regressions, colours)],
        className="sheet verdict span-all",
        **{"aria-label": "Latest run verdict"},
    )


# --------------------------------------------------------------------------- #
# Four figures                                                                #
# --------------------------------------------------------------------------- #


def _figure(term: Any, value: Any, note: Any) -> html.Div:
    return html.Div(
        [html.Dt(term), html.Dd([html.Span(value, className="fig-value"), note])],
        className="fig",
    )


def _delta_note(text: str, cls: str) -> html.Span:
    return html.Span(text, className=f"fig-note {cls}")


def _figures(
    filtered: list[EvalRun],
    total_runs: int,
    numbers: RunNumbers,
    colours: dict[str, RevisionColour],
) -> html.Dl:
    ordered = _chronological(filtered)
    latest = ordered[-1]
    previous = ordered[-2] if len(ordered) > 1 else None
    latest_label = run_label(latest, numbers)
    prev_label = run_label(previous, numbers) if previous else None
    version = latest.metadata.prompt_version.version

    # 1 · latest pass rate
    pr = pass_rate(latest)
    if previous is not None:
        d = (pr - pass_rate(previous)) * 100
        pr_note = _delta_note(f"{signed(d)} pp vs {prev_label}", delta_class(d))
    else:
        pr_note = _delta_note("first run in view", "delta-flat")
    fig_pass = _figure(
        [
            f"Latest pass rate · {latest_label} ",
            revision_chip(
                version, revision_colour(version, colours), extra_class="rv-sm"
            ),
        ],
        f"{pr * 100:.1f}%",
        pr_note,
    )

    # 2 · runs in view
    first_ts = parse_ts(ordered[0].report.timestamp)
    last_ts = parse_ts(latest.report.timestamp)
    days = (last_ts - first_ts).days
    span = f"{days // 7} weeks" if days >= 14 else f"{days} days"
    fig_runs = _figure(
        "Runs in view",
        str(len(filtered)),
        html.Span(f"of {total_runs} · {span}", className="fig-note delta-flat"),
    )

    # 3 · mean G-Eval of the latest run
    g_latest = _geval_mean(latest)
    g_prev = _geval_mean(previous) if previous is not None else None
    threshold = shared_threshold(filtered, "geval")
    if g_latest is None:
        fig_geval = _figure(
            "Mean G-Eval",
            "—",
            html.Span(
                f"no G-Eval metrics in {latest_label}", className="fig-note delta-flat"
            ),
        )
    else:
        notes: list[str] = []
        cls = "delta-flat"
        if g_prev is not None:
            d = g_latest - g_prev
            notes.append(f"{signed(d, '{:.2f}', zero='± 0.00')} vs {prev_label}")
            cls = delta_class(round(d, 2))
        if threshold is not None:
            notes.append(f"threshold {threshold:.2f}")
        fig_geval = _figure(
            f"Mean G-Eval · {latest_label}",
            f"{g_latest:.2f}",
            _delta_note(" · ".join(notes), cls) if notes else None,
        )

    # 4 · median duration across the view, plus the latest run's change
    durations = [r.report.summary.total_duration_ms for r in ordered]
    median_s = statistics.median(durations) / 1000 if durations else 0.0
    if previous is not None:
        dd = (
            latest.report.summary.total_duration_ms
            - previous.report.summary.total_duration_ms
        ) / 1000
        dur_note = _delta_note(
            f"{latest_label} {signed(dd)} s vs {prev_label}",
            delta_class(round(dd, 1), lower_is_better=True),
        )
    else:
        dur_note = _delta_note("per run", "delta-flat")
    fig_dur = _figure("Median duration", f"{median_s:.1f} s", dur_note)

    return html.Dl(
        [fig_pass, fig_runs, fig_geval, fig_dur],
        className="sheet figures span-all",
    )


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #


def _panel_head(title: str, sub: Any, right: Any = None) -> html.Div:
    left = html.Div(
        [html.H2(title, className="h-sm"), html.P(sub, className="panel-sub")]
    )
    return html.Div(
        [left, right] if right is not None else [left], className="panel-head"
    )


def _pass_rate_panel(
    filtered: list[EvalRun],
    numbers: RunNumbers,
    colours: dict[str, RevisionColour],
    regressions: list[Regression],
) -> html.Section:
    ordered = _chronological(filtered)
    first, last = ordered[0], ordered[-1]
    description = (
        f"Pass rate by run from {run_label(first, numbers)} "
        f"({pass_rate(first) * 100:.1f}%) to "
        f"{run_label(last, numbers)} "
        f"({pass_rate(last) * 100:.1f}%), "
        f"{len(regressions)} regression{'s' if len(regressions) != 1 else ''} marked."
    )
    legend = html.Div(
        [
            html.Span([html.Span(className="key-line"), "pass rate"], className="key"),
            html.Span([html.Span(className="key-ring"), "regression"], className="key"),
        ],
        className="chart-key",
    )
    return html.Section(
        [
            _panel_head(
                "Pass rate by run",
                "Bands are prompt versions in revision-colour order. Numbered red "
                "rings are the regressions listed above.",
                legend,
            ),
            html.Div(
                dcc.Graph(
                    id="chart-pass-rate",
                    figure=charts.pass_rate_chart(
                        filtered,
                        numbers=numbers,
                        colours=colours,
                        regressions=regressions,
                    ),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "280px"},
                ),
                role="img",
                className="chart-box",
                **{"aria-label": description},
            ),
        ],
        className="sheet panel span-all",
    )


def _metric_trend_panel(filtered: list[EvalRun], kind: str) -> html.Section:
    threshold = shared_threshold(filtered, kind)  # type: ignore[arg-type]
    sub = "Mean score per run for one metric kind."
    if threshold is not None:
        sub += f" Dashed line is the {threshold:.2f} threshold."
    elif metric_thresholds(filtered, kind):  # type: ignore[arg-type]
        sub += " Thresholds differ between these metrics, so none is drawn."

    switch = dcc.RadioItems(
        id="metric-kind",
        options=[
            {"label": KIND_LABELS[k], "value": k}
            for k in ("rag", "geval", "standard", "code")
        ],
        value=kind,
        inline=True,
        className="seg",
        inputClassName="seg-input",
        labelClassName="seg-option",
    )
    figure = charts.metric_trend_chart(filtered, kind, threshold)  # type: ignore[arg-type]
    has_data = bool(figure.data)
    body: Any
    if has_data:
        body = html.Div(
            dcc.Graph(
                id="chart-metric-trend",
                figure=figure,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "280px"},
            ),
            role="img",
            className="chart-box",
            **{"aria-label": f"{KIND_LABELS[kind]} metric trends by run"},
        )
    else:
        body = html.Div(
            f"No {KIND_LABELS[kind]} metrics in these runs.",
            className="empty-box",
        )
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Metric trends", className="h-sm"),
                            html.P(sub, className="panel-sub"),
                        ]
                    ),
                    html.Div(switch, role="group", **{"aria-label": "Metric kind"}),
                ],
                className="panel-head",
            ),
            body,
        ],
        className="sheet panel span-all",
    )


# --------------------------------------------------------------------------- #
# Breakdown panels                                                            #
# --------------------------------------------------------------------------- #


def _metric_bar(name: str, avg: float, threshold: float | None) -> html.Div:
    pct = max(0.0, min(avg, 1.0)) * 100
    fail = threshold is not None and avg < threshold
    track: list[Any] = [html.Span(className="bar-fill", style={"width": f"{pct:.1f}%"})]
    if threshold is not None:
        track.append(
            html.Span(className="bar-thresh", style={"left": f"{threshold * 100:.1f}%"})
        )
    score_children: list[Any] = [f"{avg:.2f}"]
    if fail:
        score_children.append(sr_only(f" below {threshold:.2f}"))
    return html.Div(
        [
            html.Span(name, className="cue bar-name", title=name),
            html.Span(track, className="bar-track", **{"aria-hidden": "true"}),
            html.Span(score_children, className="bar-score"),
        ],
        className="bar-row fail" if fail else "bar-row",
    )


_BREAKDOWNS: tuple[tuple[str, str, str], ...] = (
    (
        "rag",
        "RAG · retrieval and grounding",
        "Faithfulness, relevancy, precision and recall.",
    ),
    ("geval", "G-Eval · custom judges", "Rubrics defined in agent.yaml."),
    ("standard", "NLP · overlap metrics", "BLEU, ROUGE and METEOR; no model calls."),
    ("code", "Code graders", "Python graders from agent.yaml."),
)


def _breakdown_panel(
    kind: str, title: str, desc: str, runs: list[EvalRun]
) -> html.Section:
    df = to_breakdown_dataframe(runs, kind, recent_n=BREAKDOWN_RECENT_N)  # type: ignore[arg-type]
    n_recent = min(len(runs), BREAKDOWN_RECENT_N)
    sub = (
        f"{desc} Averaged across the last {n_recent} run{'s' if n_recent != 1 else ''}."
    )
    if df.empty:
        body: Any = html.Div(
            [
                f"No {KIND_LABELS[kind]} metrics in these runs. Add a ",
                html.Span(f"type: {kind}", className="cue"),
                " metric to agent.yaml to see results here.",
            ],
            className="empty-box",
        )
    else:
        recent = _chronological(runs)[-BREAKDOWN_RECENT_N:]
        thresholds = metric_thresholds(recent, kind)  # type: ignore[arg-type]
        body = html.Div(
            [
                _metric_bar(
                    str(row["metric_name"]),
                    float(row["avg_score"]),
                    thresholds.get(str(row["metric_name"])),
                )
                for _, row in df.iterrows()
            ],
            className="bars",
        )
    return html.Section(
        [html.H2(title, className="h-sm"), html.P(sub, className="panel-sub"), body],
        className="sheet panel",
    )


# --------------------------------------------------------------------------- #
# Runs table                                                                  #
# --------------------------------------------------------------------------- #


def _tier(pr: float) -> str:
    if pr >= 0.85:
        return "pass"
    if pr >= 0.65:
        return "warn"
    return "fail"


_TIER_ICON = {"pass": "check", "warn": "warn", "fail": "x"}  # nosec B105
_TIER_TEXT = {
    "pass": "at or above 85%",  # nosec B105
    "warn": "between 65% and 85%",
    "fail": "below 65%",
}


def _runs_table(
    filtered: list[EvalRun],
    compare_queue: list[str],
    numbers: RunNumbers,
    colours: dict[str, RevisionColour],
    regressions: list[Regression],
    search: str,
) -> html.Section:
    ordered = _chronological(filtered)
    previous_rates: list[float | None] = [None] + [pass_rate(r) for r in ordered[:-1]]
    reg_numbers = {r.label: r.number for r in regressions}
    max_ms = max((r.report.summary.total_duration_ms for r in ordered), default=0) or 1

    header = html.Thead(
        html.Tr(
            [
                html.Th(sr_only("Compare"), className="col-cmp", scope="col"),
                html.Th("Run", scope="col"),
                html.Th("When", scope="col"),
                html.Th("Pass rate", scope="col"),
                html.Th("Δ vs previous", scope="col", className="num"),
                html.Th("Cases", scope="col", className="num"),
                html.Th("Revision", scope="col"),
                html.Th("Model", scope="col"),
                html.Th("Duration", scope="col"),
                html.Th("Commit", scope="col"),
            ]
        )
    )

    rows = []
    for run, prev in reversed(list(zip(ordered, previous_rates, strict=True))):
        rid = run.report.timestamp
        label = run_label(run, numbers)
        pr = pass_rate(run)
        tier = _tier(pr)
        version = run.metadata.prompt_version.version
        in_queue = rid in compare_queue

        def nav(col: str, children: Any, cls: str = "", rid: str = rid) -> html.Td:
            return html.Td(
                children,
                id={"type": "run-row", "run_id": rid, "col": col},
                n_clicks=0,
                className=f"run-nav {cls}".strip(),
            )

        if label in reg_numbers:
            rate_cell: Any = html.Span(
                [icon("x"), f"{pr * 100:.1f}% · regression {reg_numbers[label]}"],
                className="state state-fail",
            )
        else:
            rate_cell = html.Span(
                [
                    icon(_TIER_ICON[tier]),
                    f"{pr * 100:.1f}%",
                    sr_only(f", {_TIER_TEXT[tier]}"),
                ],
                className=f"state state-{tier}",
            )

        if prev is None:
            delta_cell: Any = html.Span("—", className="delta-flat")
        else:
            d = (pr - prev) * 100
            delta_cell = html.Span(signed(d), className=delta_class(round(d, 1)))

        ms = run.report.summary.total_duration_ms
        duration_cell = html.Span(
            [
                html.Span(
                    html.Span(style={"width": f"{ms / max_ms * 100:.0f}%"}),
                    className="dur-bar",
                    **{"aria-hidden": "true"},
                ),
                f"{ms / 1000:.1f} s",
            ],
            className="dur",
        )

        rows.append(
            html.Tr(
                [
                    html.Td(
                        render_add_button(rid, compare_queue, label),
                        className="col-cmp",
                    ),
                    html.Td(
                        html.Button(
                            label,
                            id={"type": "run-row", "run_id": rid, "col": "run"},
                            n_clicks=0,
                            className="run-link cue cue-strong",
                            **{"aria-label": f"Open {label} in Explorer"},
                        )
                    ),
                    nav("ts", fmt_when(parse_ts(rid))),
                    nav("pr", rate_cell),
                    nav("delta", delta_cell, "num"),
                    nav(
                        "tests",
                        f"{run.report.summary.passed} / {run.report.summary.total_tests}",
                        "num",
                    ),
                    nav(
                        "prompt",
                        revision_chip(version, revision_colour(version, colours)),
                    ),
                    nav(
                        "model", cue(run.metadata.agent_config.model.name, strong=False)
                    ),
                    nav("duration", duration_cell),
                    nav(
                        "commit",
                        cue((run.metadata.git_commit or "—")[:7], strong=False),
                    ),
                ],
                className="row-on" if in_queue else "",
            )
        )

    return html.Section(
        [
            html.Div(
                [
                    html.H2("Runs", className="h-sm"),
                    html.Span(f"{len(filtered)} in view", className="panel-sub"),
                    html.Div(
                        [
                            html.Label(
                                [
                                    icon("search"),
                                    sr_only("Search runs"),
                                    dcc.Input(
                                        id="runs-search",
                                        type="search",
                                        placeholder="Search commit, model, tag",
                                        value=search,
                                        debounce=True,
                                        className="search-input",
                                    ),
                                ],
                                className="search",
                            ),
                            html.Button(
                                [icon("download"), "Export CSV"],
                                id="runs-export",
                                className="btn",
                                n_clicks=0,
                            ),
                        ],
                        className="table-tools",
                    ),
                ],
                className="table-head",
            ),
            html.Div(
                html.Table([header, html.Tbody(rows)], className="runs"),
                className="table-scroll",
                tabIndex="0",
                role="region",
                **{"aria-label": "Runs table"},
            ),
            dcc.Download(id="runs-csv-download"),
        ],
        className="sheet table-card span-all",
    )


# --------------------------------------------------------------------------- #
# Empty states + entry point                                                  #
# --------------------------------------------------------------------------- #


def _empty_state(title: str, body: Any) -> html.Section:
    return html.Section(
        [html.H2(title), html.P(body)],
        className="sheet empty span-all",
    )


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
            _empty_state(
                "No saved runs yet",
                [
                    "Run ",
                    html.Span("holodeck test agent.yaml", className="cue"),
                    f" to save a result for {agent_display_name}; it appears here "
                    "within five seconds.",
                ],
            ),
            className="page summary",
        )

    if not filtered:
        return html.Div(
            _empty_state(
                "No runs match these filters",
                "Reset the filters in the rail, or widen the date range or "
                "pass-rate threshold.",
            ),
            className="page summary",
        )

    numbers = run_numbers(runs)
    colours = revision_colour_map(runs)
    regressions = find_regressions(filtered, numbers)
    verdict = build_verdict(filtered, numbers, regressions)
    if verdict is None:  # unreachable: filtered is non-empty
        return html.Div(className="page summary")

    children: list[Any] = [
        _verdict_section(verdict, regressions, colours),
        _figures(filtered, len(runs), numbers, colours),
        _pass_rate_panel(filtered, numbers, colours, regressions),
        _metric_trend_panel(filtered, filters.metric_kind),
        *[_breakdown_panel(k, t, d, filtered) for k, t, d in _BREAKDOWNS],
        _runs_table(
            filtered, compare_queue, numbers, colours, regressions, filters.search
        ),
    ]
    return html.Div(children, className="page summary")
