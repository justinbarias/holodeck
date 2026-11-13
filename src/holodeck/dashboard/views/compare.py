"""Compare view — empty state, toolbar, three layouts, shared case matrix.

Runs appear by cue number with their revision chip. The first run in the
queue is the baseline; every change is measured against it. All three
layouts share :func:`_case_matrix`, a table that says Pass, Fail or Not run
in words in every cell (colour only repeats the word).
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

from holodeck.dashboard.compare_data import (
    ConfigRow,
    StatRow,
    compute_case_matrix,
    compute_config_diff,
    compute_summary_rows,
    run_stats,
)
from holodeck.dashboard.components.compare_add_button import MAX_QUEUE
from holodeck.dashboard.components.ui import (
    cue,
    delta_class,
    icon,
    revision_chip,
    signed,
    sr_only,
    state_word,
)
from holodeck.dashboard.filters import apply as apply_filters
from holodeck.dashboard.revisions import (
    RevisionColour,
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.state import state_filters
from holodeck.models.eval_run import EvalRun

_CONFIG_LABELS: dict[str, str] = {
    "prompt_version": "Prompt revision",
    "model_name": "Model",
    "temperature": "Temperature",
    "tags_joined": "Tags",
    "git_commit": "Commit",
    "extended_thinking": "Extended thinking",
}

_LOWER_IS_BETTER = {"duration_ms", "total_tokens", "est_cost"}


class _Ctx:
    """Cue numbers and revision colours for the runs on screen."""

    def __init__(self, all_runs: list[EvalRun]) -> None:
        self.numbers = run_numbers(all_runs)
        self.colours = revision_colour_map(all_runs)

    def label(self, run: EvalRun) -> str:
        return run_label(run, self.numbers)

    def colour(self, run: EvalRun) -> RevisionColour:
        return revision_colour(run.metadata.prompt_version.version, self.colours)

    def chip(self, run: EvalRun) -> html.Span:
        return revision_chip(run.metadata.prompt_version.version, self.colour(run))

    def run_tag(self, run: EvalRun) -> html.Span:
        return html.Span([cue(self.label(run)), self.chip(run)], className="run-tag")


# --------------------------------------------------------------------------- #
# Header, empty state, toolbar                                                #
# --------------------------------------------------------------------------- #


def render_compare_header(state: dict[str, Any], runs: list[EvalRun]) -> html.Header:
    queue = list(state.get("compare_queue") or [])
    numbers = run_numbers(runs)
    runs_by_id = {r.report.timestamp: r for r in runs}
    known = [runs_by_id[rid] for rid in queue if rid in runs_by_id]
    if len(known) >= 2:
        sub = (
            f"{len(known)} runs · baseline {run_label(known[0], numbers)} · "
            "change is measured against the baseline"
        )
    else:
        sub = f"Up to {MAX_QUEUE} saved runs side by side"
    return html.Header(
        [html.Div([html.H1("Compare"), html.P(sub, className="page-sub")])],
        className="page-header",
    )


def _empty_state(runs_in_range: int | None = None) -> html.Section:
    return html.Section(
        [
            html.H2("Pick runs to compare"),
            html.P(
                "Add runs with the + button in the Summary runs table or the "
                "Explorer runs list, or start from a quick pick. The first run "
                f"you add is the baseline; up to {MAX_QUEUE} runs fit side by side."
            ),
            html.Div(
                [
                    html.Button(
                        "Latest vs previous",
                        id="compare-quick-2",
                        className="btn btn-lg",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Best vs worst in range",
                        id="compare-quick-best",
                        className="btn btn-lg",
                        n_clicks=0,
                        disabled=runs_in_range is not None and runs_in_range < 2,
                    ),
                    html.Button(
                        "Latest run of the last three revisions",
                        id="compare-quick-revisions",
                        className="btn btn-lg",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Latest three runs",
                        id="compare-quick-3",
                        className="btn btn-lg",
                        n_clicks=0,
                    ),
                ],
                className="quick-picks",
            ),
            *(
                [_range_note(runs_in_range)]
                if runs_in_range is not None and runs_in_range < 2
                else []
            ),
        ],
        className="sheet cmp-empty",
    )


def _range_note(runs_in_range: int) -> html.P:
    if runs_in_range == 0:
        text = (
            "No runs match the Summary filters, so Best vs worst in range has "
            "nothing to pick. Widen the filters to use it."
        )
    else:
        text = (
            "Only one run matches the Summary filters, so Best vs worst in range "
            "has nothing to compare it with."
        )
    return html.P(text, className="muted-note range-note", role="status")


def _toolbar(runs: list[EvalRun], variant: int, ctx: _Ctx) -> html.Div:
    chips: list[Any] = []
    for i, run in enumerate(runs):
        label = ctx.label(run)
        chips.append(
            html.Span(
                [
                    cue(label),
                    ctx.chip(run),
                    *([html.Span("baseline", className="chip-note")] if i == 0 else []),
                    html.Button(
                        icon("x"),
                        id={
                            "type": "compare-slot-remove",
                            "run_id": run.report.timestamp,
                        },
                        className="chip-remove",
                        n_clicks=0,
                        **{"aria-label": f"Remove {label} from compare"},
                    ),
                ],
                className="run-chip",
            )
        )
    return html.Div(
        [
            html.Span(f"Comparing {len(runs)} runs", className="toolbar-title"),
            html.Div(chips, className="run-chips"),
            html.Div(
                [
                    html.Span(
                        "Layout", className="toolbar-label", id="compare-layout-label"
                    ),
                    dcc.RadioItems(
                        id="compare-variant",
                        options=[
                            {"label": "Side by side", "value": 1},
                            {"label": "Baseline + deltas", "value": 2},
                            {"label": "Matrix first", "value": 3},
                        ],
                        value=variant,
                        inline=True,
                        className="seg",
                        inputClassName="seg-input",
                        labelClassName="seg-option",
                    ),
                    html.Button(
                        "Clear",
                        id="compare-clear",
                        className="btn",
                        n_clicks=0,
                    ),
                ],
                className="toolbar-right",
                role="group",
                **{"aria-labelledby": "compare-layout-label"},
            ),
        ],
        className="sheet cmp-toolbar",
    )


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #


def _delta_text(row: StatRow, idx: int) -> str | None:
    d = row.deltas[idx]
    if idx == 0 or d is None:
        return None
    if row.key in ("pass_rate", "passed_ratio"):
        if row.key == "passed_ratio":
            return None
        return signed(d * 100, zero="± 0.0") + " pp"
    if row.key == "duration_ms":
        return signed(d / 1000, zero="± 0.0") + " s"
    if row.key == "est_cost":
        return signed(d, "{:.3f}", zero="± 0.000")
    if row.key == "total_tokens":
        return signed(d, "{:,.0f}", zero="± 0")
    return signed(d, "{:.2f}", zero="± 0.00")


def _delta_span(row: StatRow, idx: int) -> Any:
    text = _delta_text(row, idx)
    d = row.deltas[idx]
    if text is None or d is None:
        return None
    precision = 3 if row.key == "est_cost" else 2
    cls = delta_class(round(d, precision), lower_is_better=row.key in _LOWER_IS_BETTER)
    return html.Span(text, className=f"delta {cls}")


def _formatted(row: StatRow, idx: int) -> str:
    if row.key == "duration_ms":
        return f"{float(row.values[idx]) / 1000:.1f} s"
    if row.key == "passed_ratio":
        return row.formatted[idx].replace("/", " / ")
    if row.key == "total_tokens":
        return f"{int(row.values[idx]):,}"
    return row.formatted[idx]


_STAT_LABELS = {
    "pass_rate": "Pass rate",
    "passed_ratio": "Cases passed",
    "geval_avg": "Mean G-Eval",
    "rag_avg": "Mean RAG",
    "duration_ms": "Run duration",
    "total_tokens": "Total tokens",
    "est_cost": "Estimated cost",
}


def _stat_rows(runs: list[EvalRun]) -> list[StatRow]:
    rows = compute_summary_rows(runs)
    stats = [run_stats(r) for r in runs]
    # Cost is only shown when it comes from token counts and the pricing
    # table; a duration-based guess is not saved evidence.
    if not all(s.cost_source == "pricing_table" for s in stats):
        rows = [r for r in rows if r.key != "est_cost"]
    return rows


# --------------------------------------------------------------------------- #
# Case matrix (shared across layouts)                                         #
# --------------------------------------------------------------------------- #


def _present(score: Any) -> bool:
    """A case ran in a run when its matrix score is neither None nor NaN."""
    return score is not None and score == score


def _cell(
    passed: bool | None, score: Any, regression: bool, improvement: bool
) -> html.Td:
    if passed is None:
        body: Any = html.Span([icon("minus"), "Not run"], className="cell miss")
    elif passed:
        text = "Pass · fixed" if improvement else "Pass"
        body = html.Span([icon("check"), text], className="cell pass")
    else:
        text = "Fail · regression" if regression else "Fail"
        body = html.Span([icon("x"), text], className="cell fail")
    title = f"Score {float(score):.2f}" if score is not None else None
    return html.Td(body, title=title)


def _case_matrix(runs: list[EvalRun], ctx: _Ctx | None = None) -> html.Section:
    ctx = ctx or _Ctx(runs)
    df = compute_case_matrix(runs)
    head = html.Div(
        [
            html.H2("Test cases across runs", className="h-sm"),
            html.P(
                "Each cell says Pass, Fail or Not run; colour only repeats it. "
                "Fixed and regression compare each run with the baseline.",
                className="panel-sub",
            ),
        ],
        className="sheet-head",
    )
    if df.empty:
        return html.Section(
            [head, html.P("No test cases in these runs.", className="muted-note pad")],
            id="chart-compare-matrix",
            className="sheet cmp-block",
        )

    header = html.Thead(
        html.Tr(
            [html.Th("Case", scope="col")]
            + [
                html.Th(
                    [ctx.run_tag(run), *([sr_only(" (baseline)")] if i == 0 else [])],
                    scope="col",
                )
                for i, run in enumerate(runs)
            ]
        )
    )
    body_rows: list[Any] = []
    for ri in range(len(df)):
        cells: list[Any] = [
            html.Th(df.at[ri, "case_name"], scope="row", className="cue")
        ]
        for run in runs:
            rid = run.report.timestamp
            score = df.at[ri, f"score::{rid}"]
            present = _present(score)
            cells.append(
                _cell(
                    bool(df.at[ri, f"passed::{rid}"]) if present else None,
                    score if present else None,
                    bool(df.at[ri, f"regression::{rid}"]),
                    bool(df.at[ri, f"improvement::{rid}"]),
                )
            )
        body_rows.append(html.Tr(cells))

    return html.Section(
        [
            head,
            html.Div(
                html.Table([header, html.Tbody(body_rows)], className="matrix"),
                className="table-scroll",
                tabIndex="0",
                role="region",
                **{"aria-label": "Test cases across runs"},
            ),
        ],
        id="chart-compare-matrix",
        className="sheet cmp-block",
    )


# --------------------------------------------------------------------------- #
# Layout 1 — side by side                                                     #
# --------------------------------------------------------------------------- #


def _headline_stats(runs: list[EvalRun], ctx: _Ctx) -> html.Section:
    rows = _stat_rows(runs)
    header = html.Thead(
        html.Tr(
            [html.Th("Measure", scope="col")]
            + [html.Th(ctx.run_tag(run), scope="col") for run in runs]
        )
    )
    body: list[Any] = []
    for row in rows:
        cells: list[Any] = [html.Th(_STAT_LABELS.get(row.key, row.label), scope="row")]
        for i in range(len(runs)):
            delta = _delta_span(row, i)
            cells.append(
                html.Td(
                    [_formatted(row, i), *([" ", delta] if delta is not None else [])]
                )
            )
        body.append(html.Tr(cells))
    return html.Section(
        [
            html.Div(
                [
                    html.H2("Headline stats", className="h-sm"),
                    html.P(
                        f"Change is measured against the baseline, {ctx.label(runs[0])}.",
                        className="panel-sub",
                    ),
                ],
                className="sheet-head",
            ),
            html.Div(
                html.Table([header, html.Tbody(body)], className="stats"),
                className="table-scroll",
            ),
        ],
        className="sheet cmp-block",
    )


def _config_change(row: ConfigRow, runs: list[EvalRun], ctx: _Ctx) -> Any:
    if row.label == "prompt_version":
        pieces: list[Any] = []
        for i, run in enumerate(runs):
            if i > 0 and row.values[i] == row.values[i - 1]:
                continue
            if pieces:
                pieces.append(icon("arrow-right", "muted-ico"))
            pieces.append(ctx.chip(run))
        return html.Dd(pieces, className="chain")
    changes: list[Any] = []
    for i in range(1, len(runs)):
        if row.values[i] != row.values[i - 1]:
            changes.append(
                html.Span(
                    [
                        html.S(row.values[i - 1], className="was"),
                        " ",
                        row.values[i],
                        f" from {ctx.label(runs[i])}",
                    ],
                    className="change",
                )
            )
    return html.Dd(changes, className="cue changes")


def _config_block(runs: list[EvalRun], ctx: _Ctx) -> html.Section:
    rows = compute_config_diff(runs)
    changed = [r for r in rows if not r.all_same]
    same = [_CONFIG_LABELS.get(r.label, r.label).lower() for r in rows if r.all_same]
    items: list[Any] = [
        html.Div(
            [
                html.Dt(_CONFIG_LABELS.get(r.label, r.label)),
                _config_change(r, runs, ctx),
            ],
            className="diff-row",
        )
        for r in changed
    ]
    body: list[Any] = []
    if items:
        body.append(html.Dl(items, className="diff"))
    else:
        body.append(
            html.P(
                "No configuration differences between these runs.",
                className="muted-note pad",
            )
        )
    if same:
        body.append(
            html.P(f"Unchanged: {', '.join(same)}.", className="muted-note pad")
        )
    return html.Section(
        [
            html.Div(
                [
                    html.H2("What changed between runs", className="h-sm"),
                    html.P(
                        "Configuration differences from each run's saved snapshot.",
                        className="panel-sub",
                    ),
                ],
                className="sheet-head",
            ),
            *body,
        ],
        className="sheet cmp-block",
    )


def _variant_1(runs: list[EvalRun], ctx: _Ctx | None = None) -> html.Div:
    ctx = ctx or _Ctx(runs)
    return html.Div(
        [
            html.Div(
                [_headline_stats(runs, ctx), _config_block(runs, ctx)],
                className="cmp-split",
            ),
            _case_matrix(runs, ctx),
        ],
        className="cmp-v1 cmp-stack",
    )


# --------------------------------------------------------------------------- #
# Layout 2 — baseline + deltas                                                #
# --------------------------------------------------------------------------- #


def _variant_2(runs: list[EvalRun], ctx: _Ctx | None = None) -> html.Div:
    ctx = ctx or _Ctx(runs)
    baseline = runs[0]
    bs = run_stats(baseline)
    base_card = html.Div(
        [
            html.Span("Baseline", className="toolbar-label"),
            ctx.run_tag(baseline),
            html.Span(
                f"{bs.pass_rate * 100:.1f}% · {bs.passed} / {bs.total} · "
                f"{bs.duration_ms / 1000:.1f} s · "
                f"{baseline.metadata.agent_config.model.name}",
                className="base-stats",
            ),
        ],
        className="sheet base-card",
    )
    rows: list[Any] = [
        html.Div(
            [
                html.Span("Run"),
                html.Span("Pass rate"),
                html.Span("Mean G-Eval"),
                html.Span("Run duration"),
            ],
            className="delta-grid delta-head",
            **{"aria-hidden": "true"},
        )
    ]
    for run in runs[1:]:
        s = run_stats(run)
        dp = (s.pass_rate - bs.pass_rate) * 100
        dg = s.geval_avg - bs.geval_avg
        dd = (s.duration_ms - bs.duration_ms) / 1000
        rows.append(
            html.Div(
                [
                    ctx.run_tag(run),
                    html.Span(
                        [
                            sr_only("Pass rate "),
                            f"{s.pass_rate * 100:.1f}% ",
                            html.Span(
                                f"{signed(dp)} pp",
                                className=f"delta {delta_class(round(dp, 1))}",
                            ),
                        ]
                    ),
                    html.Span(
                        [
                            sr_only("Mean G-Eval "),
                            f"{s.geval_avg:.2f} ",
                            html.Span(
                                signed(dg, "{:.2f}", zero="± 0.00"),
                                className=f"delta {delta_class(round(dg, 2))}",
                            ),
                        ]
                    ),
                    html.Span(
                        [
                            sr_only("Run duration "),
                            f"{s.duration_ms / 1000:.1f} s ",
                            html.Span(
                                f"{signed(dd)} s",
                                className=f"delta {delta_class(round(dd, 1), lower_is_better=True)}",
                            ),
                        ]
                    ),
                ],
                className="delta-grid",
            )
        )
    return html.Div(
        [
            base_card,
            html.Div(rows, className="sheet delta-card"),
            _config_block(runs, ctx),
            _case_matrix(runs, ctx),
        ],
        className="cmp-v2 cmp-stack",
    )


# --------------------------------------------------------------------------- #
# Layout 3 — matrix first                                                     #
# --------------------------------------------------------------------------- #


def _strip(runs: list[EvalRun], ctx: _Ctx) -> html.Div:
    base = run_stats(runs[0])
    cards: list[Any] = []
    for i, run in enumerate(runs):
        s = run_stats(run)
        value: list[Any] = [f"{s.pass_rate * 100:.1f}%"]
        if i > 0:
            d = (s.pass_rate - base.pass_rate) * 100
            value += [
                " ",
                html.Span(signed(d), className=f"delta {delta_class(round(d, 1))}"),
            ]
        cards.append(
            html.Div(
                [
                    ctx.run_tag(run),
                    html.Div(value, className="strip-value"),
                    html.Div(
                        f"{s.passed} / {s.total} cases"
                        + (" · baseline" if i == 0 else ""),
                        className="panel-sub",
                    ),
                ],
                className="strip-card",
            )
        )
    return html.Div(cards, className="sheet strip")


_NARRATIVE_NAMES = 5


def _changes_vs_baseline(runs: list[EvalRun], ctx: _Ctx) -> tuple[list[str], list[str]]:
    """Every case fixed and every case broken against the baseline, each
    named once with the first run where it changed (not truncated)."""
    base = {c.test_name: c.passed for c in runs[0].report.results if c.test_name}
    fixed: list[str] = []
    broken: list[str] = []
    seen_fixed: set[str] = set()
    seen_broken: set[str] = set()
    for run in runs[1:]:
        for case in run.report.results:
            name = case.test_name or ""
            if name not in base:
                continue
            if not base[name] and case.passed and name not in seen_fixed:
                seen_fixed.add(name)
                fixed.append(f"{name} ({ctx.label(run)})")
            elif base[name] and not case.passed and name not in seen_broken:
                seen_broken.add(name)
                broken.append(f"{name} ({ctx.label(run)})")
    return fixed, broken


def _name_list(names: list[str]) -> str:
    shown = ", ".join(names[:_NARRATIVE_NAMES])
    extra = len(names) - _NARRATIVE_NAMES
    return f"{shown} and {extra} more" if extra > 0 else shown


def _narrative(runs: list[EvalRun], ctx: _Ctx) -> html.Div:
    df = compute_case_matrix(runs)
    lines: list[Any] = []
    fixed, broken = _changes_vs_baseline(runs, ctx)

    if fixed:
        n = len(fixed)
        lines.append(
            html.P(
                [
                    html.B(f"{n} case{'s' if n != 1 else ''} fixed"),
                    f" since the baseline: {_name_list(fixed)}.",
                ]
            )
        )
    if broken:
        n = len(broken)
        lines.append(
            html.P(
                [
                    state_word("regression", f"{n} regression{'s' if n != 1 else ''}"),
                    f" against the baseline: {_name_list(broken)}.",
                ]
            )
        )

    if not df.empty:
        always_fail = [
            str(df.at[ri, "case_name"])
            for ri in range(len(df))
            if all(
                _present(df.at[ri, f"score::{r.report.timestamp}"])
                and not bool(df.at[ri, f"passed::{r.report.timestamp}"])
                for r in runs
            )
        ]
        if always_fail:
            n = len(always_fail)
            lines.append(
                html.P(
                    [
                        state_word(
                            "fail", f"{n} case{'s' if n != 1 else ''} still fail"
                        ),
                        f" in every run: {', '.join(always_fail)}.",
                    ]
                )
            )

    base, last = runs[0], runs[-1]
    dd = (
        last.report.summary.total_duration_ms - base.report.summary.total_duration_ms
    ) / 1000
    if abs(dd) >= 0.05:
        lines.append(
            html.P(
                f"{ctx.label(last)} took {abs(dd):.1f} s "
                f"{'less' if dd < 0 else 'more'} than the baseline {ctx.label(base)}."
            )
        )
    if not lines:
        lines.append(html.P("No case changed result between these runs."))
    return html.Div(lines, className="sheet narrative")


def _variant_3(runs: list[EvalRun], ctx: _Ctx | None = None) -> html.Div:
    ctx = ctx or _Ctx(runs)
    return html.Div(
        [_strip(runs, ctx), _narrative(runs, ctx), _case_matrix(runs, ctx)],
        className="cmp-v3 cmp-stack",
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def render_compare(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    queue_ids = list(state.get("compare_queue") or [])
    runs_by_id = {r.report.timestamp: r for r in runs}
    queue_runs = [runs_by_id[rid] for rid in queue_ids if rid in runs_by_id]

    if len(queue_runs) < 2:
        in_range = len(apply_filters(state_filters(state), runs))
        return html.Div(_empty_state(in_range), className="page cmp-wrap")

    ctx = _Ctx(runs)
    variant = int(state.get("compare_variant", 1) or 1)
    if variant == 2:
        body: html.Div = _variant_2(queue_runs, ctx)
    elif variant == 3:
        body = _variant_3(queue_runs, ctx)
    else:
        body = _variant_1(queue_runs, ctx)

    return html.Div(
        [_toolbar(queue_runs, variant, ctx), body], className="page cmp-wrap"
    )
