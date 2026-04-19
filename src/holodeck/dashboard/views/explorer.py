"""Explorer view — 3-column drilldown (runs → cases → detail).

Layout + interaction mirror the design handoff's ``explorer.js``. Column
widths and collapse behavior come from ``Evaluation Dashboard.html``:

* expanded:   ``340px 340px 1fr``
* collapsed:  ``48px 340px 1fr``

State — ``state["explorer_run_id"]``, ``state["explorer_case_name"]``,
``state["explorer_runs_collapsed"]`` — lives in the app's ``dcc.Store``.
All navigation callbacks are registered in ``app.py`` so they share the
single ``Output("app-state", "data")`` sink.
"""

from __future__ import annotations

import json
from typing import Any

from dash import dcc, html

from holodeck.dashboard.components.compare_add_button import render_add_button
from holodeck.dashboard.explorer_data import (
    AgentSnapshot,
    CaseDetail,
    CaseHeader,
    CaseSummary,
    ConversationView,
    ExpectedToolsCoverage,
    MetricRow,
    ToolCallView,
    build_case_detail,
    list_case_summaries,
    select_run,
)
from holodeck.dashboard.seed_data import SEED_CONVERSATIONS
from holodeck.models.eval_run import EvalRun

# --------------------------------------------------------------------------- #
# Empty / not-found states                                                    #
# --------------------------------------------------------------------------- #


def _empty_no_runs() -> html.Div:
    return html.Div(
        [
            html.Span("∅", className="hd-empty-glyph"),
            html.H3("No runs found"),
            html.P(
                html.Code("Run holodeck test agent.yaml to generate one"),
                className="mono",
            ),
        ],
        className="hd-empty-state",
    )


def _placeholder_case_panel() -> html.Div:
    return html.Div(
        [
            html.Div("▸", style={"fontSize": "32px", "marginBottom": "8px"}),
            html.H3("Select a test case", style={"margin": "0 0 6px"}),
            html.P(
                "Pick a case on the left to inspect its config, conversation, "
                "tool calls, and evaluations.",
                className="mono",
            ),
        ],
        className="hd-explorer-detail hd-empty",
    )


# --------------------------------------------------------------------------- #
# Column builders                                                             #
# --------------------------------------------------------------------------- #


def _pass_rate_tier(pass_rate: float) -> str:
    if pass_rate >= 0.85:
        return "pass"
    if pass_rate >= 0.65:
        return "warn"
    return "fail"


def _model_suffix(model_name: str) -> str:
    parts = (model_name or "").split("-")
    return "-".join(parts[-2:]) if len(parts) >= 2 else model_name


def _runs_column_expanded(
    runs: list[EvalRun],
    active_run_id: str | None,
    queue: list[str],
) -> html.Div:
    # Newest first.
    sorted_runs = sorted(runs, key=lambda r: r.report.timestamp, reverse=True)

    rows: list = []
    for run in sorted_runs:
        rid = run.report.timestamp
        total = run.report.summary.total_tests
        pr = (run.report.summary.passed / total) if total > 0 else 0.0
        tier = _pass_rate_tier(pr)
        cls = "hd-run-row hd-run-row--active" if rid == active_run_id else "hd-run-row"
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                rid[:16].replace("T", " "),
                                className="mono",
                                style={"fontSize": "12px", "color": "var(--fg1)"},
                            ),
                            html.Span(
                                run.metadata.prompt_version.version,
                                className="mono",
                                style={"color": "var(--hd-accent)"},
                            ),
                            render_add_button(rid, queue),
                        ],
                        className="r1",
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "justifyContent": "space-between",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{pr * 100:.1f}%",
                                className=f"pill pill-{tier}",
                            ),
                            html.Span(
                                f"{run.report.summary.passed}/{run.report.summary.total_tests}",
                                className="mono",
                            ),
                            html.Span(
                                _model_suffix(run.metadata.agent_config.model.name),
                                className="mono",
                                style={"marginLeft": "auto"},
                            ),
                        ],
                        className="r2",
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                ],
                id={"type": "explorer-run-row", "run_id": rid},
                className=cls,
                n_clicks=0,
            )
        )

    header = html.Div(
        [
            html.Span("▸", style={"color": "var(--hd-accent)", "fontSize": "22px"}),
            html.H4(
                [
                    "Runs ",
                    html.Span(
                        str(len(sorted_runs)),
                        className="mono",
                        style={"color": "var(--hd-muted)", "fontWeight": 400},
                    ),
                ],
                style={"margin": 0, "fontSize": "13px"},
            ),
            html.Span(
                "newest first",
                className="mono",
                style={
                    "fontSize": "11px",
                    "color": "var(--hd-muted)",
                    "marginLeft": "auto",
                },
            ),
        ],
        id="explorer-runs-toggle",
        className="list-head",
        role="button",
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "cursor": "pointer",
        },
    )

    return html.Div(
        [
            header,
            html.Div(rows, className="list-scroll", style={"overflowY": "auto"}),
        ],
        className="list-card hd-explorer-runs",
    )


def _runs_column_collapsed(runs: list[EvalRun]) -> html.Div:
    return html.Div(
        [
            html.Span(
                "▸",
                style={
                    "color": "var(--hd-accent)",
                    "fontSize": "22px",
                    "marginTop": "16px",
                },
            ),
            html.Span(
                f"RUNS {len(runs)}",
                style={
                    "writingMode": "vertical-rl",
                    "transform": "rotate(180deg)",
                    "fontSize": "11px",
                    "letterSpacing": ".2em",
                    "textTransform": "uppercase",
                    "color": "var(--hd-muted)",
                    "fontFamily": "var(--font-mono)",
                    "marginTop": "12px",
                },
            ),
        ],
        id="explorer-runs-toggle",
        className="hd-explorer-runs-collapsed",
        n_clicks=0,
        role="button",
        title="Expand runs",
    )


def _cases_column(
    cases: list[CaseSummary],
    active_case_name: str | None,
) -> html.Div:
    pass_count = sum(1 for c in cases if c.passed)

    rows: list = []
    for c in cases:
        cls = "case-item active" if c.name == active_case_name else "case-item"
        metric_chips = []
        if c.geval_score is not None:
            metric_chips.append(
                html.Span(
                    f"geval {c.geval_score:.2f}",
                    className=f"mini-metric{' pass' if c.geval_score >= 0.7 else ' fail'}",
                )
            )
        if c.rag_avg_score is not None:
            metric_chips.append(
                html.Span(f"rag {c.rag_avg_score:.2f}", className="mini-metric")
            )
        if c.tools_called_count > 0:
            metric_chips.append(
                html.Span(f"{c.tools_called_count} tools", className="mini-metric")
            )

        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(c.name, className="nm", title=c.name),
                            html.Span(
                                "PASS" if c.passed else "FAIL",
                                className=f"pill {'pill-pass' if c.passed else 'pill-fail'}",
                            ),
                        ],
                        className="c1",
                    ),
                    html.Div(metric_chips, className="c2"),
                ],
                id={"type": "explorer-case-row", "case_name": c.name},
                className=cls,
                n_clicks=0,
            )
        )

    header = html.Div(
        [
            html.H4(
                [
                    "Test cases ",
                    html.Span(
                        str(len(cases)),
                        className="mono",
                        style={"color": "var(--hd-muted)", "fontWeight": 400},
                    ),
                ],
                style={"margin": 0, "fontSize": "13px"},
            ),
            html.Span(
                f"{pass_count} pass",
                className="mono",
                style={
                    "fontSize": "11px",
                    "color": "var(--hd-muted)",
                    "marginLeft": "auto",
                },
            ),
        ],
        className="list-head",
        style={"display": "flex", "alignItems": "center", "gap": "10px"},
    )

    return html.Div(
        [header, html.Div(rows, className="list-scroll")],
        className="list-card",
    )


# --------------------------------------------------------------------------- #
# Detail panel                                                                #
# --------------------------------------------------------------------------- #


def _case_header_section(h: CaseHeader) -> html.Div:
    def badge(label: str, value: Any, *, accent: bool = False) -> html.Span:
        return html.Span(
            [
                html.Span(
                    label,
                    className="mono",
                    style={"color": "var(--hd-muted)", "marginRight": "4px"},
                ),
                html.B(
                    value,
                    className="mono",
                    style={"color": "var(--hd-accent)" if accent else "var(--fg1)"},
                ),
            ],
            style={"display": "inline-flex", "gap": "4px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "✓ PASS" if h.passed else "✕ FAIL",
                        className=f"pill {'pill-pass' if h.passed else 'pill-fail'}",
                    ),
                    html.H2(h.case_name, style={"margin": 0}),
                ],
                className="row1",
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),
            html.Div(
                [
                    badge("run", h.run_timestamp.replace("T", " ")[:19]),
                    badge("prompt", h.prompt_version, accent=True),
                    badge("model", h.model_name),
                    badge(
                        "temp",
                        f"{h.temperature}" if h.temperature is not None else "—",
                    ),
                    badge("commit", (h.git_commit or "—")[:10]),
                ],
                className="row2",
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "12px",
                    "marginTop": "8px",
                },
            ),
        ],
        className="detail-head",
    )


def _agent_snapshot_section(snap: AgentSnapshot) -> html.Details:
    def item(k: str, v: Any, *, accent: bool = False) -> html.Div:
        return html.Div(
            [
                html.Span(k, className="k"),
                html.Span(
                    str(v),
                    className="v",
                    style=({"color": "var(--hd-accent)"} if accent else {}),
                ),
            ],
            className="cfg-item",
        )

    grid = html.Div(
        [
            item("model.provider", snap.model_provider),
            item("model.name", snap.model_name),
            item("model.temperature", snap.model_temperature),
            item("model.max_tokens", snap.model_max_tokens or "—"),
            item("embedding.provider", snap.embedding_provider or "—"),
            item("embedding.name", snap.embedding_name or "—"),
            item("claude.extended_thinking", snap.claude_extended_thinking),
            item("prompt.version", snap.prompt_version, accent=True),
            item("prompt.author", snap.prompt_author or "—"),
            item("prompt.file_path", snap.prompt_file_path or "—"),
            item("prompt.source", snap.prompt_source),
            item("prompt.tags", " ".join(f"#{t}" for t in snap.prompt_tags)),
        ],
        className="cfg-grid",
    )

    tools_row = html.Div(
        [
            html.Div(
                f"TOOLS ({len(snap.tools)})",
                className="eyebrow",
                style={
                    "fontSize": "10px",
                    "letterSpacing": ".15em",
                    "textTransform": "uppercase",
                    "color": "var(--hd-muted)",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                [
                    html.Span(
                        [
                            html.Span(
                                kind,
                                style={
                                    "color": "var(--hd-accent-soft)",
                                    "marginRight": "6px",
                                    "fontSize": "10px",
                                },
                            ),
                            html.Span(name),
                        ],
                        className="chip",
                        style={"cursor": "default"},
                    )
                    for kind, name in snap.tools
                ],
                className="chip-row",
            ),
        ],
        style={"marginTop": "12px"},
    )

    raw_drawer = html.Details(
        [
            html.Summary("View raw JSON", className="reset-btn"),
            html.Pre(
                json.dumps(snap.raw_config, indent=2, default=str),
                className="code",
                style={"marginTop": "8px"},
            ),
        ],
        style={"marginTop": "12px"},
    )

    summary = html.Summary(
        [
            html.Div(
                [
                    html.Div("AGENT CONFIG SNAPSHOT", className="eyebrow"),
                    html.H3("Configuration at run time"),
                    html.P(
                        [
                            "Captured from ",
                            html.Code("agent.yaml"),
                            " at run time · secret-bearing fields stripped before persist.",
                        ],
                        className="subtitle",
                    ),
                ],
                className="summary-text",
            ),
        ],
    )

    return html.Details(
        [summary, grid, tools_row, raw_drawer],
        className="panel hd-explorer-section",
        open=False,
    )


def _tool_call_panel(tc: ToolCallView) -> html.Div:
    header = html.Div(
        [
            html.Span("TOOL", className="badge"),
            html.Span(f"{tc.name}()", className="tname"),
            html.Span(f"{tc.result_size_bytes}B", className="bytes"),
        ],
        className="th",
    )

    pieces: list = [header]

    if tc.error:
        pieces.append(
            html.Div(
                f"Tool failed: {tc.error}",
                style={
                    "background": "rgba(255,120,80,.08)",
                    "borderLeft": "3px solid #ff9d7e",
                    "padding": "6px 10px",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "color": "#ff9d7e",
                },
            )
        )

    # args — always open
    pieces.append(
        html.Details(
            [
                html.Summary(
                    html.Span(
                        "args",
                        className="mono",
                        style={
                            "color": "var(--hd-muted)",
                            "textTransform": "uppercase",
                            "letterSpacing": ".1em",
                            "fontSize": "11px",
                        },
                    )
                ),
                html.Pre(json.dumps(tc.args, indent=2, default=str), className="code"),
            ],
            open=True,
        )
    )

    # result — collapsed by default when large
    result_summary = "result"
    if tc.large:
        result_summary = f"result — Expand ({tc.result_size_bytes}B)"
    pieces.append(
        html.Details(
            [
                html.Summary(
                    html.Span(
                        result_summary,
                        className="mono",
                        style={
                            "color": "var(--hd-muted)",
                            "textTransform": "uppercase",
                            "letterSpacing": ".1em",
                            "fontSize": "11px",
                        },
                    )
                ),
                html.Pre(
                    json.dumps(tc.result, indent=2, default=str), className="code"
                ),
            ],
            open=not tc.large,
        )
    )

    if tc.args == {} and tc.result is None and not tc.error:
        pieces.append(
            html.Div(
                "args/result not captured — re-run after upgrading to this HoloDeck version",
                className="mono",
                style={"color": "var(--hd-muted)", "fontSize": "11px"},
            )
        )

    return html.Div(pieces, className="tool-call")


def _render_assistant_body(text: str) -> Any:
    """Render an agent reply as prettified JSON if JSON-shaped, else as Markdown.

    JSON detection requires both a ``{`` or ``[`` prefix *and* a successful
    ``json.loads`` — a chatty reply that happens to embed JSON mid-sentence
    stays on the Markdown path.
    """
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if parsed is not None:
            return html.Pre(
                json.dumps(parsed, indent=2, ensure_ascii=False, default=str),
                className="code lang-json",
            )
    return dcc.Markdown(
        text,
        className="md-assistant",
        link_target="_blank",
    )


def _conversation_section(conv: ConversationView, model_name: str) -> html.Details:
    thread: list = []
    if conv.user:
        thread.append(
            html.Div(
                [html.Div("USER", className="who"), html.Div(conv.user)],
                className="bubble user",
            )
        )
    for tc in conv.tool_calls:
        thread.append(_tool_call_panel(tc))
    if conv.assistant:
        thread.append(
            html.Div(
                [
                    html.Div(f"AGENT · {model_name}", className="who"),
                    _render_assistant_body(conv.assistant),
                ],
                className="bubble assistant",
            )
        )

    summary = html.Summary(
        [
            html.Div(
                [
                    html.Div("CONVERSATION", className="eyebrow"),
                    html.H3("Thread with tool calls"),
                    html.P(
                        "User input, agent response, and every tool invocation that happened in between.",
                        className="subtitle",
                    ),
                ],
                className="summary-text",
            ),
        ],
    )

    return html.Details(
        [summary, html.Div(thread, className="thread")],
        className="panel hd-explorer-section",
        open=True,
    )


def _expected_tools_section(cov: ExpectedToolsCoverage) -> html.Details:
    match_pill_cls = (
        "pill-pass" if cov.total > 0 and cov.matched == cov.total else "pill-fail"
    )

    if cov.total == 0:
        body: Any = html.Div(
            "No expected tools configured for this case.",
            className="mono",
            style={"padding": "12px 0", "color": "var(--hd-muted)"},
        )
    else:
        body = html.Div(
            [
                html.Div(
                    [
                        html.Span("✓" if ok else "✕", className="ind"),
                        html.Span(name, className="nm mono"),
                        html.Span(
                            "called" if ok else "not invoked",
                            className="note",
                        ),
                    ],
                    className=f"expect-row {'ok' if ok else 'miss'}",
                )
                for name, ok in cov.rows
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "6px"},
        )

    summary = html.Summary(
        [
            html.Div(
                [
                    html.Div("EXPECTED TOOLS", className="eyebrow"),
                    html.H3("Tool-call coverage"),
                    html.P(
                        [
                            "Configured ",
                            html.Code("expected_tools"),
                            " vs. what the agent actually invoked.",
                        ],
                        className="subtitle",
                    ),
                ],
                className="summary-text",
            ),
            html.Div(
                html.Span(
                    f"{cov.matched}/{cov.total} matched",
                    className=f"pill {match_pill_cls}",
                ),
                className="summary-right",
            ),
        ],
    )

    return html.Details(
        [summary, body], className="panel hd-explorer-section", open=False
    )


def _metric_row_div(row: MetricRow) -> html.Div:
    score_cls = "mono fg" if row.passed else "mono"
    color = "var(--hd-accent)" if row.passed else "#ff9d7e"
    pieces = [
        html.Div(
            [
                html.Span(row.kind, className="kind"),
                html.Span(row.name, className="name"),
            ],
            className="nm",
        ),
        html.Div(
            [
                html.Span(
                    "score",
                    className="mono",
                    style={"fontSize": "11px", "color": "var(--hd-muted)"},
                ),
                html.Div(
                    f"{row.score:.2f}",
                    className=score_cls,
                    style={"fontSize": "14px", "fontWeight": 600, "color": color},
                ),
            ],
            style={"textAlign": "center"},
        ),
        html.Div(
            [
                html.Span(
                    "thresh",
                    className="mono",
                    style={"fontSize": "11px", "color": "var(--hd-muted)"},
                ),
                html.Div(
                    f"{row.threshold:.2f}" if row.threshold is not None else "—",
                    className="mono",
                    style={"fontSize": "13px"},
                ),
            ],
            style={"textAlign": "center"},
        ),
        html.Span(
            "PASS" if row.passed else "FAIL",
            className=f"pill {'pill-pass' if row.passed else 'pill-fail'}",
            style={"justifySelf": "end"},
        ),
    ]
    if row.reasoning:
        pieces.append(html.Div(row.reasoning, className="rsn"))

    return html.Div(pieces, className="eval-row")


def _evaluations_section(groups: dict[str, list[MetricRow]]) -> html.Details:
    blocks = []
    for kind, rows in groups.items():
        blocks.append(
            html.Div(
                [
                    html.Div(
                        kind.upper(),
                        className="eyebrow",
                        style={
                            "fontSize": "10px",
                            "letterSpacing": ".15em",
                            "textTransform": "uppercase",
                            "color": "var(--hd-accent-soft)",
                            "marginBottom": "8px",
                        },
                    ),
                    *(_metric_row_div(r) for r in rows),
                ],
                style={"marginBottom": "14px"},
            )
        )

    body: Any = blocks or html.Div(
        "No evaluations recorded for this case.",
        className="mono",
        style={"color": "var(--hd-muted)"},
    )

    summary = html.Summary(
        [
            html.Div(
                [
                    html.Div("EVALUATIONS", className="eyebrow"),
                    html.H3("Per-metric results"),
                    html.P(
                        "Score, threshold, and judge reasoning for every evaluation attached to this case.",
                        className="subtitle",
                    ),
                ],
                className="summary-text",
            ),
        ],
    )

    return html.Details(
        [summary, html.Div(body)],
        className="panel hd-explorer-section",
        open=False,
    )


def _detail_panel(run: EvalRun, detail: CaseDetail) -> html.Div:
    return html.Div(
        [
            _case_header_section(detail.header),
            _agent_snapshot_section(detail.agent_snapshot),
            _conversation_section(
                detail.conversation, detail.agent_snapshot.model_name
            ),
            _expected_tools_section(detail.expected_tools_coverage),
            _evaluations_section(detail.evaluations),
        ],
        className="detail hd-explorer-detail",
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def render_explorer(state: dict[str, Any], runs: list[EvalRun]) -> html.Div:
    if not runs:
        return html.Div(_empty_no_runs(), className="explorer")

    # Resolve selection with sensible defaults.
    run_id = state.get("explorer_run_id")
    run = select_run(runs, run_id) if run_id else None
    if run is None:
        run = sorted(runs, key=lambda r: r.report.timestamp)[-1]

    cases = list_case_summaries(run)
    case_name = state.get("explorer_case_name")
    if case_name is None or not any(c.name == case_name for c in cases):
        case_name = cases[0].name if cases else None

    runs_collapsed = bool(state.get("explorer_runs_collapsed", True))
    queue = list(state.get("compare_queue") or [])

    col_runs: html.Div
    if runs_collapsed:
        col_runs = _runs_column_collapsed(runs)
    else:
        col_runs = _runs_column_expanded(runs, run.report.timestamp, queue)

    col_cases = _cases_column(cases, case_name)

    if case_name is None:
        col_detail = _placeholder_case_panel()
    else:
        detail = build_case_detail(run, case_name, SEED_CONVERSATIONS)
        if detail is None:
            col_detail = _placeholder_case_panel()
        else:
            col_detail = _detail_panel(run, detail)

    grid_cls = "hd-explorer-grid"
    if runs_collapsed:
        grid_cls += " hd-explorer-grid--collapsed"

    return html.Div([col_runs, col_cases, col_detail], className=grid_cls)
