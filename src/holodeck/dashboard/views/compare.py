"""Compare view — empty state, toolbar, three variants, shared case matrix.

All three variants reuse :func:`_case_matrix` (Plotly heatmap via
``dcc.Graph``) for the per-case matrix block. Layout tokens mirror the
handoff's ``compare.js`` and ``styles.js``.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from dash import dcc, html

from holodeck.dashboard.compare_data import (
    COMPARE_PALETTE,
    Callout,
    StatRow,
    compute_case_matrix,
    compute_compare_callouts,
    compute_config_diff,
    compute_summary_rows,
    delta_pill_class,
    run_stats,
)
from holodeck.models.eval_run import EvalRun

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _slot_label(i: int) -> str:
    return "BASELINE" if i == 0 else f"RUN {i}"


def _color(i: int) -> str:
    return COMPARE_PALETTE[i] if i < len(COMPARE_PALETTE) else "var(--fg1)"


def _dt(run: EvalRun) -> str:
    return run.report.timestamp.replace("T", " ")[:19]


def _delta_text(row: StatRow, idx: int) -> str | None:
    if idx == 0:
        return None
    d = row.deltas[idx]
    if d is None or d == 0:
        return "·"
    sign = "+" if d > 0 else ""
    if row.key == "pass_rate":
        return f"{sign}{d * 100:.1f}pp"
    if row.key == "duration_ms":
        return f"{sign}{int(d)}ms"
    if row.key == "est_cost":
        return f"{sign}{d:.3f}"
    if row.key == "total_tokens":
        return f"{sign}{int(d)}"
    return f"{sign}{d:.2f}"


def _delta_pill(row: StatRow, idx: int) -> Any:
    if idx == 0:
        return None
    d = row.deltas[idx]
    text = _delta_text(row, idx) or "·"
    cls = delta_pill_class(d, invert=(row.delta_polarity == "invert"))
    return html.Span(text, className=f"delta {cls}")


# --------------------------------------------------------------------------- #
# Empty state / toolbar                                                       #
# --------------------------------------------------------------------------- #


def _empty_state() -> html.Div:
    rect_stack = html.Div(
        [
            html.Div(
                style={
                    "width": "22px",
                    "height": "56px",
                    "position": "absolute",
                    "left": "2px",
                    "top": "8px",
                    "border": f"1.2px solid {COMPARE_PALETTE[0]}",
                    "borderRadius": "2px",
                    "opacity": 0.7,
                },
            ),
            html.Div(
                style={
                    "width": "22px",
                    "height": "50px",
                    "position": "absolute",
                    "left": "26px",
                    "top": "12px",
                    "border": f"1.2px solid {COMPARE_PALETTE[1]}",
                    "borderRadius": "2px",
                    "opacity": 0.55,
                },
            ),
            html.Div(
                style={
                    "width": "22px",
                    "height": "44px",
                    "position": "absolute",
                    "left": "50px",
                    "top": "16px",
                    "border": f"1.2px solid {COMPARE_PALETTE[2]}",
                    "borderRadius": "2px",
                    "opacity": 0.4,
                },
            ),
        ],
        style={"position": "relative", "width": "80px", "height": "68px"},
        className="cmp-empty-icon",
    )

    return html.Div(
        [
            rect_stack,
            html.H2("Pick runs to compare"),
            html.P(
                [
                    "Select up to 3 runs from the Explorer's Runs pane, the "
                    "Summary table, or quick-pick below. The first-selected "
                    "run becomes your ",
                    html.Span("baseline", style={"color": "var(--hd-accent)"}),
                    "; others show deltas against it.",
                ]
            ),
            html.Div(
                [
                    html.Button(
                        "Compare latest 2 runs",
                        id="compare-quick-2",
                        className="hd-btn hd-btn-primary",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Compare latest 3 runs",
                        id="compare-quick-3",
                        className="hd-btn hd-btn-ghost",
                        n_clicks=0,
                    ),
                ],
                className="cmp-empty-actions",
            ),
            html.Div(
                [
                    "Tip · click the ",
                    html.Span("+", className="cmp-empty-kbd"),
                    " icon on any run to add it to the compare queue.",
                ],
                className="cmp-empty-hint mono",
            ),
        ],
        className="cmp-empty",
    )


def _toolbar(runs: list[EvalRun], variant: int) -> html.Div:
    baseline = runs[0]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "COMPARE",
                        className="eyebrow",
                        style={"color": "var(--hd-accent-soft)"},
                    ),
                    html.H2(
                        [
                            f"{len(runs)} runs · baseline ",
                            html.Span(
                                baseline.metadata.prompt_version.version,
                                style={"color": COMPARE_PALETTE[0]},
                            ),
                        ],
                        style={
                            "margin": "2px 0 0",
                            "fontSize": "19px",
                            "fontWeight": 600,
                            "letterSpacing": "-.01em",
                        },
                    ),
                ],
                className="cmp-toolbar-left",
            ),
            html.Div(
                [
                    html.Span(
                        "layout",
                        className="mono",
                        style={"color": "var(--hd-muted)", "fontSize": "12px"},
                    ),
                    dcc.RadioItems(
                        id="compare-variant",
                        options=[
                            {"label": "side-by-side", "value": 1},
                            {"label": "baseline + deltas", "value": 2},
                            {"label": "matrix-first", "value": 3},
                        ],
                        value=variant,
                        inline=True,
                        className="seg",
                    ),
                    html.Button(
                        "Clear",
                        id="compare-clear",
                        className="reset-btn",
                        n_clicks=0,
                    ),
                ],
                className="cmp-toolbar-right",
            ),
        ],
        className="cmp-toolbar",
    )


# --------------------------------------------------------------------------- #
# Run-slot header + column utilities                                           #
# --------------------------------------------------------------------------- #


def _run_slot_header(run: EvalRun, i: int) -> html.Div:
    color = _color(i)
    cfg = run.metadata.agent_config
    pv = run.metadata.prompt_version
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        className="cmp-dot",
                        style={
                            "background": color,
                            "boxShadow": f"0 0 10px {color}90",
                        },
                    ),
                    html.Span(_slot_label(i), className="cmp-slot-label"),
                    html.Button(
                        "×",
                        id={
                            "type": "compare-slot-remove",
                            "run_id": run.report.timestamp,
                        },
                        className="cmp-x",
                        n_clicks=0,
                        title="Remove from comparison",
                    ),
                ],
                className="cmp-col-head-top",
            ),
            html.Div(_dt(run), className="cmp-col-head-ts mono"),
            html.Div(
                [
                    html.Span(pv.version, className="cmp-ver", style={"color": color}),
                    html.Span("·", className="mono"),
                    html.Span(cfg.model.name, className="mono"),
                ],
                className="cmp-col-head-meta",
            ),
            html.Div(
                (run.metadata.git_commit or "—")[:10],
                className="cmp-col-head-commit mono",
            ),
        ],
        className="cmp-col-head",
    )


# --------------------------------------------------------------------------- #
# Case matrix (shared across variants)                                        #
# --------------------------------------------------------------------------- #


def _case_matrix(runs: list[EvalRun]) -> html.Div:
    df = compute_case_matrix(runs)
    if df.empty:
        return html.Div(
            "No cases found across the selected runs.",
            className="mono",
            style={"color": "var(--hd-muted)", "padding": "16px"},
        )

    run_ids = [r.report.timestamp for r in runs]
    run_labels = ["base" if i == 0 else f"r{i}" for i in range(len(runs))]
    case_names = df["case_name"].tolist()

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    regression_cells: list[tuple[int, int]] = []
    improvement_cells: list[tuple[int, int]] = []

    for ri, _name in enumerate(case_names):
        z_row: list[float | None] = []
        text_row: list[str] = []
        for ci, rid in enumerate(run_ids):
            score = df.at[ri, f"score::{rid}"]
            passed = df.at[ri, f"passed::{rid}"]
            reg = df.at[ri, f"regression::{rid}"]
            imp = df.at[ri, f"improvement::{rid}"]
            if score is None:
                z_row.append(None)
                text_row.append("—")
            else:
                z_row.append(float(score))
                text_row.append(f"{'✓' if passed else '✕'} {float(score):.2f}")
            if reg:
                regression_cells.append((ri, ci))
            if imp:
                improvement_cells.append((ri, ci))
        z.append(z_row)
        text.append(text_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=run_labels,
            y=case_names,
            text=text,
            texttemplate="%{text}",
            textfont={"color": "#050b09", "family": "var(--font-mono)", "size": 11},
            colorscale=[
                (0.0, "#ff9d7e"),
                (0.5, "#1c2b25"),
                (1.0, "#7bff5a"),
            ],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate="<b>%{y}</b><br>%{x}<br>score: %{z:.2f}<extra></extra>",
        )
    )

    for ri, ci in regression_cells:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=ci - 0.5,
            x1=ci + 0.5,
            y0=ri - 0.5,
            y1=ri + 0.5,
            line={"color": "#ff9d7e", "dash": "dash", "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )
    for ri, ci in improvement_cells:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=ci - 0.5,
            x1=ci + 0.5,
            y0=ri - 0.5,
            y1=ri + 0.5,
            line={"color": "#7bff5a", "dash": "dash", "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        margin={"l": 140, "r": 20, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(360, 36 * len(case_names) + 40),
        xaxis={
            "side": "top",
            "tickfont": {"color": "#8a9a92", "family": "var(--font-mono)", "size": 11},
            "showgrid": False,
            "ticks": "",
        },
        yaxis={
            "autorange": "reversed",
            "tickfont": {"color": "#e5e7eb", "family": "var(--font-mono)", "size": 11},
            "showgrid": False,
            "ticks": "",
        },
    )

    legend = html.Div(
        [
            html.Span(
                [
                    html.Span(
                        className="legend-swatch",
                        style={"background": "rgba(123,255,90,.5)"},
                    ),
                    "pass",
                ],
                className="legend-item",
            ),
            html.Span(
                [
                    html.Span(
                        className="legend-swatch",
                        style={"background": "rgba(255,120,80,.5)"},
                    ),
                    "fail",
                ],
                className="legend-item",
            ),
            html.Span(
                [
                    html.Span(
                        className="legend-swatch",
                        style={
                            "background": "transparent",
                            "border": "1px dashed #ff9d7e",
                        },
                    ),
                    "regression vs. baseline",
                ],
                className="legend-item",
            ),
            html.Span(
                [
                    html.Span(
                        className="legend-swatch",
                        style={
                            "background": "transparent",
                            "border": "1px dashed #7bff5a",
                        },
                    ),
                    "improvement vs. baseline",
                ],
                className="legend-item",
            ),
        ],
        className="cmp-legend",
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("PER-CASE MATRIX", className="eyebrow"),
                    html.H3("Test-case pass/fail across runs"),
                    html.P(
                        "Heatmap of per-case scores. Cells show pass/fail plus "
                        "the primary metric (geval, else rag avg). Regressions "
                        "— passing in baseline but failing elsewhere — are "
                        "outlined."
                    ),
                ],
                className="cmp-block-head",
            ),
            dcc.Graph(
                id="chart-compare-matrix",
                figure=fig,
                config={"displayModeBar": False},
            ),
            legend,
        ],
        className="cmp-block",
    )


# --------------------------------------------------------------------------- #
# Variant 1 — side-by-side                                                    #
# --------------------------------------------------------------------------- #


def _v1_summary_block(runs: list[EvalRun]) -> html.Div:
    rows = compute_summary_rows(runs)
    cols_style = f"140px repeat({len(runs)}, minmax(0,1fr))"

    row_children: list = []
    for row in rows:
        row_children.append(html.Div(row.label, className="cmp-row-label"))
        for i, _run in enumerate(runs):
            cell_children: list = [
                html.Div(
                    row.formatted[i],
                    className=(
                        "cmp-cell-val mono"
                        + (" accent-big" if row.key == "pass_rate" else "")
                    ),
                )
            ]
            pill = _delta_pill(row, i)
            if pill is not None:
                cell_children.append(pill)
            row_children.append(html.Div(cell_children, className="cmp-cell"))

    return html.Div(
        [
            html.Div(
                [
                    html.Div("SUMMARY", className="eyebrow"),
                    html.H3("Headline stats"),
                    html.P(
                        [
                            "Deltas shown against the ",
                            html.Span(
                                "baseline",
                                style={"color": COMPARE_PALETTE[0]},
                            ),
                            ". Lower-is-better fields (duration, cost) invert "
                            "delta polarity.",
                        ]
                    ),
                ],
                className="cmp-block-head",
            ),
            html.Div(
                row_children,
                className="cmp-rows",
                style={"gridTemplateColumns": cols_style},
            ),
        ],
        className="cmp-block",
    )


def _v1_config_block(runs: list[EvalRun]) -> html.Div:
    rows = compute_config_diff(runs)
    cols_style = f"140px repeat({len(runs)}, minmax(0,1fr))"
    body: list = []
    for row in rows:
        body.append(html.Div(row.label, className="cmp-row-label"))
        for i, v in enumerate(row.values):
            different = (not row.all_same) and i > 0 and v != row.values[0]
            cls = "cmp-cell cmp-cfg-cell"
            if different:
                cls += " different"
            elif row.all_same:
                cls += " same"
            cell: list = [html.Span(v, className="mono fg")]
            if different:
                cell.append(html.Span("changed", className="cmp-diff-badge"))
            body.append(html.Div(cell, className=cls))

    return html.Div(
        [
            html.Div(
                [
                    html.Div("CONFIG DIFF", className="eyebrow"),
                    html.H3("What's different?"),
                    html.P("Rows where values differ across runs are highlighted."),
                ],
                className="cmp-block-head",
            ),
            html.Div(
                body,
                className="cmp-rows",
                style={"gridTemplateColumns": cols_style},
            ),
        ],
        className="cmp-block",
    )


def _variant_1(runs: list[EvalRun]) -> html.Div:
    cols_style = f"140px repeat({len(runs)}, minmax(0,1fr))"
    header_row = [html.Div()] + [_run_slot_header(r, i) for i, r in enumerate(runs)]
    return html.Div(
        [
            html.Div(
                header_row,
                className="cmp-cols",
                style={"gridTemplateColumns": cols_style},
            ),
            _v1_summary_block(runs),
            _v1_config_block(runs),
            _case_matrix(runs),
        ],
        className="cmp-v1",
    )


# --------------------------------------------------------------------------- #
# Variant 2 — baseline + deltas                                               #
# --------------------------------------------------------------------------- #


def _variant_2(runs: list[EvalRun]) -> html.Div:
    baseline = runs[0]
    bs = run_stats(baseline)

    baseline_card = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        className="cmp-dot",
                        style={"background": COMPARE_PALETTE[0]},
                    ),
                    html.Span("BASELINE", className="cmp-slot-label"),
                    html.Button(
                        "×",
                        id={
                            "type": "compare-slot-remove",
                            "run_id": baseline.report.timestamp,
                        },
                        className="cmp-x",
                        n_clicks=0,
                    ),
                ],
                className="cmp-v2-label",
            ),
            html.Div(
                [
                    html.Span(
                        baseline.metadata.prompt_version.version,
                        style={"color": COMPARE_PALETTE[0]},
                    ),
                    html.Span(
                        _dt(baseline),
                        className="mono",
                        style={"color": "var(--hd-muted)"},
                    ),
                ],
                className="cmp-v2-title",
            ),
            html.Div(
                [
                    f"{baseline.metadata.agent_config.model.name} · T ",
                    f"{baseline.metadata.agent_config.model.temperature} · ",
                    (baseline.metadata.git_commit or "—")[:10],
                ],
                className="cmp-v2-model mono",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"{bs.pass_rate * 100:.1f}%",
                                className="num mono",
                            ),
                            html.Span(
                                f"pass rate · {bs.passed}/{bs.total}",
                                className="cmp-sub mono",
                            ),
                        ],
                        className="cmp-v2-pass",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("geval", className="cmp-sub mono"),
                                    html.Span(
                                        f"{bs.geval_avg:.2f}", className="mono fg"
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("rag", className="cmp-sub mono"),
                                    html.Span(f"{bs.rag_avg:.2f}", className="mono fg"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("dur", className="cmp-sub mono"),
                                    html.Span(
                                        f"{bs.duration_ms / 1000:.1f}s",
                                        className="mono fg",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("cost", className="cmp-sub mono"),
                                    html.Span(
                                        f"${bs.est_cost:.3f}",
                                        className="mono fg",
                                    ),
                                ]
                            ),
                        ],
                        className="cmp-v2-stats",
                    ),
                ],
                className="cmp-v2-big",
            ),
            html.Div(
                [
                    html.Span(f"#{t}", className="chip", style={"cursor": "default"})
                    for t in (baseline.metadata.prompt_version.tags or [])
                ],
                className="cmp-v2-tags chip-row",
            ),
        ],
        className="cmp-v2-baseline",
    )

    delta_cards = []
    for i, run in enumerate(runs[1:], start=1):
        s = run_stats(run)
        model_changed = (
            run.metadata.agent_config.model.name
            != baseline.metadata.agent_config.model.name
        )
        model_line: list = [run.metadata.agent_config.model.name]
        if model_changed:
            model_line.append(html.Span("changed", className="cmp-diff-dot"))

        def row(label: str, value: str, delta: float | None, *, invert: bool = False):
            children: list = [
                html.Span(label, className="cmp-sub mono"),
                html.Span(value, className="mono fg", style={"fontSize": "15px"}),
            ]
            if delta is not None and delta != 0:
                sign = "+" if delta > 0 else ""
                if label == "pass rate":
                    text = f"{sign}{delta * 100:.1f}pp"
                elif label == "duration":
                    text = f"{sign}{int(delta)}ms"
                elif label == "cost":
                    text = f"{sign}{delta:.3f}"
                else:
                    text = f"{sign}{delta:.2f}"
                cls = delta_pill_class(delta, invert=invert)
                children.append(html.Span(text, className=f"delta {cls}"))
            else:
                children.append(html.Span("·", className="delta delta-neutral"))
            return html.Div(children, className="cmp-delta-row")

        delta_cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="cmp-dot",
                                style={"background": COMPARE_PALETTE[i]},
                            ),
                            html.Span(f"VS RUN {i}", className="cmp-slot-label"),
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
                        className="cmp-v2-label",
                    ),
                    html.Div(
                        [
                            html.Span(
                                run.metadata.prompt_version.version,
                                style={"color": COMPARE_PALETTE[i]},
                            ),
                            html.Span(
                                _dt(run),
                                className="mono",
                                style={"color": "var(--hd-muted)"},
                            ),
                        ],
                        className="cmp-v2-title",
                    ),
                    html.Div(model_line, className="cmp-v2-model mono"),
                    row(
                        "pass rate",
                        f"{s.pass_rate * 100:.1f}%",
                        s.pass_rate - bs.pass_rate,
                    ),
                    row("geval", f"{s.geval_avg:.2f}", s.geval_avg - bs.geval_avg),
                    row("rag", f"{s.rag_avg:.2f}", s.rag_avg - bs.rag_avg),
                    row(
                        "duration",
                        f"{s.duration_ms / 1000:.1f}s",
                        float(s.duration_ms - bs.duration_ms),
                        invert=True,
                    ),
                    row(
                        "cost",
                        f"${s.est_cost:.3f}",
                        s.est_cost - bs.est_cost,
                        invert=True,
                    ),
                ],
                className="cmp-v2-delta",
            )
        )

    grid = html.Div(
        [baseline_card, *delta_cards],
        className="cmp-v2-grid",
        style={
            "gridTemplateColumns": (
                f"minmax(0, 1.4fr) repeat({len(delta_cards)}, minmax(0, 1fr))"
            )
        },
    )

    return html.Div([grid, _case_matrix(runs)], className="cmp-v2")


# --------------------------------------------------------------------------- #
# Variant 3 — matrix first                                                    #
# --------------------------------------------------------------------------- #


def _v3_strip(runs: list[EvalRun]) -> html.Div:
    baseline_stats = run_stats(runs[0])
    cards = []
    for i, run in enumerate(runs):
        s = run_stats(run)
        is_base = i == 0
        diff = None if is_base else s.pass_rate - baseline_stats.pass_rate
        pass_pieces: list = [
            html.Span(f"{s.pass_rate * 100:.1f}%", className="num mono"),
        ]
        if diff is not None:
            sign = "+" if diff > 0 else ""
            pass_pieces.append(
                html.Span(
                    f"{sign}{diff * 100:.1f}pp",
                    className=f"delta {delta_pill_class(diff)}",
                )
            )
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="cmp-dot",
                                style={"background": _color(i)},
                            ),
                            html.Span(_slot_label(i), className="cmp-slot-label"),
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
                        className="cmp-v3-card-head",
                    ),
                    html.Div(
                        run.metadata.prompt_version.version,
                        className="cmp-v3-ver",
                        style={"color": _color(i)},
                    ),
                    html.Div(
                        run.metadata.agent_config.model.name,
                        className="cmp-v3-model mono",
                    ),
                    html.Div(pass_pieces, className="cmp-v3-pass"),
                    html.Div(
                        [
                            html.Span(
                                [
                                    "geval ",
                                    html.B(f"{s.geval_avg:.2f}"),
                                ],
                                className="mono",
                            ),
                            html.Span(
                                [
                                    "rag ",
                                    html.B(f"{s.rag_avg:.2f}"),
                                ],
                                className="mono",
                            ),
                            html.Span(f"{s.duration_ms / 1000:.1f}s", className="mono"),
                        ],
                        className="cmp-v3-mini",
                    ),
                ],
                className=f"cmp-v3-card {'baseline' if is_base else ''}",
            )
        )
    return html.Div(cards, className="cmp-v3-strip")


def _v3_callouts(callouts: list[Callout]) -> html.Div | None:
    any_rows = any(c.regressions or c.improvements for c in callouts)
    if not any_rows:
        return None

    cards = []
    for i, c in enumerate(callouts, start=1):
        body_rows: list = []
        if c.regressions:
            body_rows.append(
                html.Div(
                    [
                        html.Span(
                            f"{len(c.regressions) + c.regressions_extra} regression"
                            + (
                                "s"
                                if (len(c.regressions) + c.regressions_extra) != 1
                                else ""
                            ),
                            className="pill pill-fail",
                        ),
                        html.Span(
                            ", ".join(c.regressions)
                            + (
                                f" +{c.regressions_extra} more"
                                if c.regressions_extra
                                else ""
                            ),
                            className="mono",
                            style={"color": "var(--hd-muted)", "marginLeft": "8px"},
                        ),
                    ],
                    className="cmp-v3-callout-row",
                )
            )
        if c.improvements:
            body_rows.append(
                html.Div(
                    [
                        html.Span(
                            f"{len(c.improvements) + c.improvements_extra} improvement"
                            + (
                                "s"
                                if (len(c.improvements) + c.improvements_extra) != 1
                                else ""
                            ),
                            className="pill pill-pass",
                        ),
                        html.Span(
                            ", ".join(c.improvements)
                            + (
                                f" +{c.improvements_extra} more"
                                if c.improvements_extra
                                else ""
                            ),
                            className="mono",
                            style={"color": "var(--hd-muted)", "marginLeft": "8px"},
                        ),
                    ],
                    className="cmp-v3-callout-row",
                )
            )
        if not body_rows:
            body_rows.append(
                html.Span(
                    "No case-level changes from baseline.",
                    className="mono",
                    style={"color": "var(--hd-muted)"},
                )
            )

        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="cmp-dot",
                                style={"background": _color(i)},
                            ),
                            html.Span(
                                f"vs. baseline · {c.prompt_version}",
                                className="mono",
                            ),
                        ],
                        className="cmp-v3-callout-head",
                    ),
                    html.Div(body_rows, className="cmp-v3-callout-body"),
                ],
                className="cmp-v3-callout",
            )
        )

    return html.Div(cards, className="cmp-v3-callouts")


def _variant_3(runs: list[EvalRun]) -> html.Div:
    callouts = compute_compare_callouts(runs)
    pieces: list = [_v3_strip(runs)]
    callout_block = _v3_callouts(callouts)
    if callout_block is not None:
        pieces.append(callout_block)
    pieces.append(_case_matrix(runs))
    return html.Div(pieces, className="cmp-v3")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def render_compare(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    queue_ids = list(state.get("compare_queue") or [])
    runs_by_id = {r.report.timestamp: r for r in runs}
    queue_runs = [runs_by_id[rid] for rid in queue_ids if rid in runs_by_id]

    if len(queue_runs) < 2:
        return html.Div(_empty_state(), className="cmp-wrap")

    variant = int(state.get("compare_variant", 1) or 1)

    if variant == 2:
        variant_body: html.Div = _variant_2(queue_runs)
    elif variant == 3:
        variant_body = _variant_3(queue_runs)
    else:
        variant_body = _variant_1(queue_runs)

    return html.Div([_toolbar(queue_runs, variant), variant_body], className="cmp-wrap")
