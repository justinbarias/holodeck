"""Explorer view — runs → test cases → case detail.

Three columns beside the compact rail: runs (cue-numbered, collapsible),
test cases for the selected run, and the case detail. The detail reads like
a prompt book: tool calls are numbered cues (Q1, Q2…) in a ruled margin
beside the conversation; expected tools refer back to those cues; unexpected
calls get the warning treatment; every evaluation shows its score against
its threshold with Pass or Fail in words.

State — ``state["explorer_run_id"]``, ``state["explorer_case_name"]``,
``state["explorer_runs_collapsed"]`` — lives in the app's ``dcc.Store``.
All navigation callbacks are registered in ``app.py`` so they share the
single ``Output("app-state", "data")`` sink.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from dash import dcc, html

from holodeck.dashboard.components.compare_add_button import render_add_button
from holodeck.dashboard.components.ui import (
    cue,
    fmt_when,
    icon,
    parse_ts,
    revision_chip,
    sr_only,
    state_word,
)
from holodeck.dashboard.explorer_data import (
    AgentSnapshot,
    CaseDetail,
    CaseHeader,
    CaseSummary,
    ConversationView,
    ExpectedToolsCoverage,
    MetricRow,
    ToolCallView,
    TurnCoverage,
    TurnView,
    build_case_detail,
    list_case_summaries,
    select_run,
)
from holodeck.dashboard.loaded_run import LoadedRun
from holodeck.dashboard.revisions import (
    RevisionColour,
    RunNumbers,
    chronological,
    revision_colour,
    revision_colour_map,
    run_label,
    run_numbers,
)
from holodeck.dashboard.seed_data import SEED_CONVERSATIONS
from holodeck.dashboard.verdict import pass_rate

KIND_LABELS: dict[str, str] = {
    "rag": "RAG",
    "geval": "G-Eval",
    "standard": "NLP",
    "code": "Code",
}

# --------------------------------------------------------------------------- #
# Empty / not-found states                                                    #
# --------------------------------------------------------------------------- #


def _empty_no_runs() -> html.Section:
    return html.Section(
        [
            html.H1("No saved runs yet"),
            html.P(
                [
                    "Run ",
                    html.Span("holodeck test agent.yaml", className="cue"),
                    " to save a result; the Explorer lists it here.",
                ]
            ),
        ],
        className="sheet empty",
    )


def _run_not_found_panel(ref: str) -> html.Div:
    return html.Div(
        html.Section(
            [
                html.H1("Run not found"),
                html.P(
                    [
                        "This link names run ",
                        html.Span(ref, className="cue"),
                        ", which is not in the loaded results. The file may "
                        "have been renamed or deleted. Pick a run from the list.",
                    ]
                ),
            ],
            className="sheet empty",
            role="status",
        ),
        className="detail hd-explorer-detail",
    )


def _placeholder_case_panel() -> html.Div:
    return html.Div(
        html.Section(
            [
                html.H1("Pick a test case"),
                html.P(
                    "Choose a case in the list to read its conversation, tool "
                    "cues and evaluations."
                ),
            ],
            className="sheet empty",
        ),
        className="detail hd-explorer-detail",
    )


# --------------------------------------------------------------------------- #
# Column builders                                                             #
# --------------------------------------------------------------------------- #


def _tier(pr: float) -> str:
    if pr >= 0.85:
        return "pass"
    if pr >= 0.65:
        return "warn"
    return "fail"


_TIER_ICON = {"pass": "check", "warn": "warn", "fail": "x"}  # nosec B105


def _runs_column_expanded(
    runs: list[LoadedRun],
    active_run_id: str | None,
    queue: list[str],
    numbers: RunNumbers | None = None,
    colours: dict[str, RevisionColour] | None = None,
) -> html.Div:
    numbers = numbers if numbers is not None else run_numbers(runs)
    colours = colours if colours is not None else revision_colour_map(runs)
    usual_model = Counter(r.metadata.agent_config.model.name for r in runs).most_common(
        1
    )[0][0]
    sorted_runs = list(reversed(chronological(runs)))

    rows: list[Any] = []
    for run in sorted_runs:
        rid = run.run_id
        label = run_label(run, numbers)
        pr = pass_rate(run)
        tier = _tier(pr)
        version = run.metadata.prompt_version.version
        model = run.metadata.agent_config.model.name
        when = fmt_when(parse_ts(run.report.timestamp))
        meta = when if model == usual_model else f"{when} · {model}"
        active = rid == active_run_id
        attrs: dict[str, Any] = {"aria-current": "true"} if active else {}
        rows.append(
            html.Li(
                [
                    html.Button(
                        [
                            html.Span(
                                [
                                    cue(label),
                                    revision_chip(
                                        version, revision_colour(version, colours)
                                    ),
                                ],
                                className="row-line",
                            ),
                            html.Span(
                                [
                                    html.Span(meta, className="row-meta"),
                                    html.Span(
                                        [icon(_TIER_ICON[tier]), f"{pr * 100:.1f}%"],
                                        className=f"state state-{tier}",
                                    ),
                                ],
                                className="row-line",
                            ),
                        ],
                        id={"type": "explorer-run-row", "run_id": rid},
                        className="list-row on" if active else "list-row",
                        n_clicks=0,
                        **attrs,
                    ),
                    render_add_button(rid, queue, label),
                ],
                className="list-item",
            )
        )

    header = html.Div(
        [
            html.H2(["Runs ", html.Span(str(len(sorted_runs)), className="cue count")]),
            html.Button(
                icon("chevron-left"),
                id="explorer-runs-toggle",
                className="icon-btn",
                n_clicks=0,
                title="Collapse runs",
                **{"aria-label": "Collapse runs", "aria-expanded": "true"},
            ),
        ],
        className="col-head",
    )

    return html.Div(
        [header, html.Ul(rows, className="list-scroll")],
        className="col hd-explorer-runs",
    )


def _runs_column_collapsed(runs: list[LoadedRun]) -> html.Div:
    return html.Div(
        html.Button(
            [
                icon("chevron-right"),
                html.Span(f"Runs {len(runs)}", className="vertical"),
            ],
            id="explorer-runs-toggle",
            className="collapsed-toggle",
            n_clicks=0,
            title="Show runs",
            **{"aria-label": f"Show runs ({len(runs)})", "aria-expanded": "false"},
        ),
        className="col hd-explorer-runs-collapsed",
    )


def _case_meta(c: CaseSummary) -> list[str]:
    parts: list[str] = []
    if c.geval_score is not None:
        parts.append(f"geval {c.geval_score:.2f}")
    if c.rag_avg_score is not None:
        parts.append(f"rag {c.rag_avg_score:.2f}")
    if c.tools_called_count > 0:
        n = c.tools_called_count
        parts.append(f"{n} tool{'s' if n != 1 else ''}")
    return parts


def _cases_column(
    cases: list[CaseSummary],
    active_case_name: str | None,
    run_text: str | None = None,
) -> html.Div:
    pass_count = sum(1 for c in cases if c.passed)

    rows: list[Any] = []
    for c in cases:
        active = c.name == active_case_name
        meta: list[Any] = [" · ".join(_case_meta(c))] if _case_meta(c) else []
        if c.turns_total is not None:
            chip_cls = "turn-chip fail" if c.turns_failed else "turn-chip"
            meta.append(
                html.Span(
                    f"{c.turns_total} turns · {c.turns_passed}/{c.turns_total} passed",
                    className=chip_cls,
                )
            )
        attrs: dict[str, Any] = {"aria-current": "true"} if active else {}
        rows.append(
            html.Li(
                html.Button(
                    [
                        html.Span(
                            [
                                html.Span(
                                    c.name,
                                    className="cue cue-strong case-name",
                                    title=c.name,
                                ),
                                state_word("pass" if c.passed else "fail"),
                            ],
                            className="row-line",
                        ),
                        html.Span(meta, className="row-meta"),
                    ],
                    id={"type": "explorer-case-row", "case_name": c.name},
                    className="list-row on" if active else "list-row",
                    n_clicks=0,
                    **attrs,
                ),
                className="list-item",
            )
        )

    sub = f"{pass_count} of {len(cases)} pass"
    if run_text:
        sub = f"{run_text} · {sub}"
    header = html.Div(
        [html.H2("Test cases"), html.Span(sub, className="col-sub")],
        className="col-head",
    )

    return html.Div(
        [header, html.Ul(rows, className="list-scroll")],
        className="col hd-explorer-cases",
    )


# --------------------------------------------------------------------------- #
# Detail panel                                                                #
# --------------------------------------------------------------------------- #


def _kv(label: str, value: Any, *, mono: bool = True) -> html.Div:
    return html.Div(
        [html.Dt(label), html.Dd(value, className="cue" if mono else "")],
        className="kv-item",
    )


def _case_header_section(
    h: CaseHeader,
    run_text: str | None = None,
    colour: RevisionColour | None = None,
) -> html.Section:
    when = fmt_when(parse_ts(h.run_timestamp))
    run_value = f"{run_text} · {when}" if run_text else h.run_timestamp
    version_value: Any = (
        revision_chip(h.prompt_version, colour)
        if colour is not None
        else h.prompt_version
    )
    temp = f"{h.temperature}" if h.temperature is not None else "—"
    return html.Section(
        [
            html.Div(
                [
                    state_word("pass" if h.passed else "fail", badge=True),
                    html.H1(h.case_name, className="cue case-title"),
                ],
                className="case-head",
            ),
            html.Dl(
                [
                    _kv("Run", run_value),
                    _kv("Revision", version_value, mono=False),
                    _kv("Model · temperature", f"{h.model_name} · {temp}"),
                    _kv("Commit", h.git_commit or "—"),
                ],
                className="kv kv-4",
            ),
        ],
        className="sheet detail-head",
    )


def _agent_snapshot_section(snap: AgentSnapshot) -> html.Details:
    tools = ", ".join(name for _, name in snap.tools) if snap.tools else "none"
    temp = f"{snap.model_temperature}" if snap.model_temperature is not None else "—"
    grid = html.Dl(
        [
            _kv("Backend", snap.model_provider),
            _kv("Model · temperature", f"{snap.model_name} · {temp}"),
            _kv("Max tokens", str(snap.model_max_tokens or "—")),
            _kv("Instructions", snap.prompt_file_path or snap.prompt_source),
            _kv("Tools", tools),
            _kv(
                "Embedding",
                (
                    f"{snap.embedding_provider} · {snap.embedding_name}"
                    if snap.embedding_provider
                    else "—"
                ),
            ),
            _kv("Prompt version", snap.prompt_version),
            _kv("Prompt author", snap.prompt_author or "—"),
            _kv("Tags", " ".join(f"#{t}" for t in snap.prompt_tags) or "—"),
            _kv("Extended thinking", "on" if snap.claude_extended_thinking else "off"),
        ],
        className="kv kv-3",
    )

    raw_drawer = html.Details(
        [
            html.Summary(
                "Show the saved configuration as JSON", className="sub-summary"
            ),
            html.Pre(
                json.dumps(snap.raw_config, indent=2, default=str),
                className="code",
            ),
        ],
        className="raw-drawer",
    )

    summary = html.Summary(
        [
            html.Span(icon("chevron-right"), className="disclosure"),
            html.Div(
                [
                    html.H2("Agent configuration at run time", className="h-sm"),
                    html.P(
                        "Captured from agent.yaml when the run started. Fields "
                        "holding secrets are removed before saving.",
                        className="panel-sub",
                    ),
                ],
            ),
        ],
        className="sheet-summary",
    )

    return html.Details(
        [summary, grid, raw_drawer],
        className="sheet hd-explorer-section config",
        open=False,
    )


def _args_summary(args: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in list(args.items())[:3]:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) > 32:
            text = text[:31] + "…"
        parts.append(f"{key} {text}")
    return " · ".join(parts)


def _tool_call_panel(
    tc: ToolCallView, cue_no: int | None = None, max_ms: int | None = None
) -> html.Div:
    failed = bool(tc.error)
    meta: list[str] = [f"{tc.result_size_bytes} B"]
    if tc.duration_ms is not None:
        meta.append(f"{tc.duration_ms / 1000:.2f} s")

    header: list[Any] = [
        html.Span(tc.name, className="cue cue-strong tool-name"),
        html.Span(_args_summary(tc.args), className="tool-args"),
        state_word("fail", "Failed") if failed else state_word("done"),
        html.Span(" · ".join(meta), className="tool-meta"),
    ]
    pieces: list[Any] = [html.Div(header, className="tool-head")]

    if tc.duration_ms is not None and max_ms:
        pieces.append(
            html.Div(
                html.Span(style={"width": f"{tc.duration_ms / max_ms * 100:.0f}%"}),
                className="dur-bar tool-dur",
                **{"aria-hidden": "true"},
            )
        )

    if tc.error:
        pieces.append(
            html.Div(
                [icon("x"), f"Tool failed: {tc.error}"],
                className="callout callout-fail",
            )
        )

    pieces.append(
        html.Details(
            [
                html.Summary("Arguments", className="sub-summary"),
                html.Pre(json.dumps(tc.args, indent=2, default=str), className="code"),
            ],
            open=False,
        )
    )
    result_label = "Result"
    if tc.large:
        result_label = f"Result ({tc.result_size_bytes} B)"
    pieces.append(
        html.Details(
            [
                html.Summary(result_label, className="sub-summary"),
                html.Pre(
                    json.dumps(tc.result, indent=2, default=str), className="code"
                ),
            ],
            open=not tc.large,
        )
    )

    if tc.args == {} and tc.result is None and not tc.error:
        pieces.append(
            html.P(
                "Arguments and result were not saved for this call; re-run with "
                "this HoloDeck version to capture them.",
                className="muted-note",
            )
        )

    card = html.Div(pieces, className="tool-call")
    if cue_no is None:
        return card
    return _cue_row(card, cue_no, failed)


def _cue_row(content: Any, cue_no: int | None, failed: bool = False) -> html.Div:
    """One row of the cue sheet: ruled margin (with an optional cue) + content."""
    margin: Any = None
    if cue_no is not None:
        margin = html.Span(
            [
                html.Span(className="cue-light fail" if failed else "cue-light"),
                f"Q{cue_no}",
                sr_only(", failed" if failed else ", done"),
            ],
            className="cue-mark",
        )
    return html.Div(
        [
            html.Div(margin, className="cue-margin"),
            html.Div(content, className="cue-body"),
        ],
        className="cue-row",
    )


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


def _user_bubble(text: str) -> html.Div:
    return html.Div(
        [html.Div("User", className="who sr-only-label"), html.Div(text)],
        className="bubble user",
    )


def _agent_reply(text: str, model_name: str | None) -> html.Div:
    who = f"Agent · {model_name}" if model_name else "Agent"
    return html.Div(
        [html.Div(who, className="who"), _render_assistant_body(text)],
        className="bubble assistant",
    )


def _max_duration(calls: list[ToolCallView]) -> int | None:
    values = [c.duration_ms for c in calls if c.duration_ms is not None]
    return max(values) if values else None


def _turn_thread_block(turn: TurnView, cue_start: int = 1) -> html.Div:
    """Render one turn inside a multi-turn conversation section.

    Tool calls keep counting cues across turns: pass ``cue_start`` for the
    first cue number in this turn. Skipped turns are marked in words.
    """
    is_skipped = turn.skipped

    header: list[Any] = [
        html.Span(f"Turn {turn.turn_index}", className="turn-title"),
    ]
    if is_skipped:
        header.append(state_word("miss", "Skipped"))
    else:
        header.append(
            html.Span(f"{turn.execution_time_ms / 1000:.2f} s", className="tool-meta")
        )
        if turn.tool_invocations:
            n = len(turn.tool_invocations)
            header.append(
                html.Span(f"{n} cue{'s' if n != 1 else ''}", className="tool-meta")
            )

    rows: list[Any] = []
    if turn.input:
        rows.append(_cue_row(_user_bubble(turn.input), None))
    max_ms = _max_duration(turn.tool_invocations)
    for offset, tc in enumerate(turn.tool_invocations):
        rows.append(_tool_call_panel(tc, cue_start + offset, max_ms))
    if turn.response is not None and turn.response != "":
        rows.append(_cue_row(_agent_reply(turn.response, None), None))

    extras: list[Any] = []
    if turn.metric_results:
        extras.append(
            html.Div(
                [_metric_row_div(m) for m in turn.metric_results],
                className="evals evals-turn",
            )
        )
    for msg in turn.errors:
        extras.append(html.Div([icon("x"), msg], className="callout callout-fail"))
    if turn.token_usage is not None:
        tu = turn.token_usage
        extras.append(
            html.P(
                f"Tokens: prompt {tu.get('prompt_tokens', 0)} · completion "
                f"{tu.get('completion_tokens', 0)} · total {tu.get('total_tokens', 0)}",
                className="muted-note cue",
            )
        )
    if extras:
        rows.append(_cue_row(html.Div(extras), None))

    details_cls = "hd-explorer-turn"
    if is_skipped:
        details_cls += " hd-explorer-turn--skipped"
    details = html.Details(
        [
            html.Summary(
                [html.Span(icon("chevron-right"), className="disclosure"), *header],
                className="turn-summary",
            ),
            html.Div(rows, className="cue-sheet"),
        ],
        className=details_cls,
        open=not is_skipped,
    )
    return html.Div([details], className="hd-explorer-turn-wrapper")


def _conversation_section(conv: ConversationView, model_name: str) -> html.Section:
    head = html.Div(
        [
            html.H2("Conversation", className="h-sm"),
            html.P(
                "User input, every tool cue in order, then the agent's answer.",
                className="panel-sub",
            ),
        ],
        className="sheet-head",
    )

    if conv.turns is not None:
        blocks: list[Any] = []
        next_cue = 1
        for t in conv.turns:
            blocks.append(_turn_thread_block(t, next_cue))
            next_cue += len(t.tool_invocations)
        return html.Section(
            [head, html.Div(blocks, className="turns hd-explorer-turns")],
            className="sheet hd-explorer-section conversation",
        )

    rows: list[Any] = []
    if conv.user:
        rows.append(_cue_row(_user_bubble(conv.user), None))
    max_ms = _max_duration(conv.tool_calls)
    for i, tc in enumerate(conv.tool_calls, start=1):
        rows.append(_tool_call_panel(tc, i, max_ms))
    if conv.assistant:
        rows.append(_cue_row(_agent_reply(conv.assistant, model_name), None))

    return html.Section(
        [head, html.Div(rows, className="cue-sheet")],
        className="sheet hd-explorer-section conversation",
    )


def _cue_for(name: str, cue_names: list[str], *, exact: bool) -> int | None:
    for i, called in enumerate(cue_names, start=1):
        if (called == name) if exact else (name in called):
            return i
    return None


def _unexpected_cues(unexpected: list[str], cue_names: list[str]) -> list[int | None]:
    """Cue number for each unexpected call, in order: a repeated tool name
    points at its own occurrence (Q1, then Q2), not the first one again."""
    used: set[int] = set()
    out: list[int | None] = []
    for name in unexpected:
        cue_no = next(
            (
                i
                for i, called in enumerate(cue_names, start=1)
                if called == name and i not in used
            ),
            None,
        )
        if cue_no is not None:
            used.add(cue_no)
        out.append(cue_no)
    return out


def _expected_tool_row(name: str, ok: bool, cue_no: int | None = None) -> html.Li:
    if ok:
        text = f"Called · Q{cue_no}" if cue_no is not None else "Called"
        status = state_word("pass", text)
    else:
        status = state_word("fail", "Not called")
    return html.Li(
        [html.Span(name, className="cue"), status],
        className="expect-row",
    )


def _unexpected_row(name: str, cue_no: int | None) -> html.Li:
    text = f"Unexpected call at Q{cue_no}" if cue_no is not None else "Unexpected call"
    return html.Li(
        [html.Span(name, className="cue"), state_word("warn", text)],
        className="expect-row warn",
    )


def _turn_coverage_block(tc: TurnCoverage) -> html.Div:
    expected_rows: list[Any] = [
        _expected_tool_row(name, ok) for name, ok in tc.expected
    ]
    if not expected_rows:
        expected_rows = [
            html.Li(
                "No tools asserted in this turn.", className="expect-row muted-note"
            )
        ]
    actual = ", ".join(tc.actual) if tc.actual else "no tool calls"
    return html.Div(
        [
            html.H3(f"Turn {tc.turn_index}", className="h-xs"),
            html.Ul(expected_rows, className="expect-list"),
            html.P(
                ["Called: ", html.Span(actual, className="cue")],
                className="muted-note",
            ),
        ],
        className="turn-coverage",
    )


def _expected_tools_section(
    cov: ExpectedToolsCoverage, cue_names: list[str] | None = None
) -> html.Section:
    cue_names = cue_names or []
    unexpected = cov.unexpected_names if cov.total > 0 else []

    status: list[Any] = []
    if cov.total > 0:
        all_matched = cov.matched == cov.total
        status.append(
            state_word(
                "pass" if all_matched else "fail",
                f"{cov.matched} of {cov.total} matched",
            )
        )
    if unexpected:
        status.append(state_word("warn", f"{len(unexpected)} unexpected"))

    if cov.total == 0 and not cov.per_turn:
        body: Any = html.P(
            "No expected tools are configured for this case.",
            className="muted-note pad",
        )
    elif cov.per_turn:
        body = html.Div(
            [_turn_coverage_block(tc) for tc in cov.per_turn],
            className="turn-coverages",
        )
    else:
        rows: list[Any] = [
            _expected_tool_row(name, ok, _cue_for(name, cue_names, exact=False))
            for name, ok in cov.rows
        ]
        rows += [
            _unexpected_row(name, cue_no)
            for name, cue_no in zip(
                unexpected, _unexpected_cues(unexpected, cue_names), strict=True
            )
        ]
        body = html.Ul(rows, className="expect-list")

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Expected tools", className="h-sm"),
                            html.P(
                                [
                                    html.Span("expected_tools", className="cue"),
                                    " from the test case against what the agent called.",
                                ],
                                className="panel-sub",
                            ),
                        ]
                    ),
                    html.Div(status, className="head-status"),
                ],
                className="sheet-head split",
            ),
            body,
        ],
        className="sheet hd-explorer-section",
    )


def _metric_row_div(row: MetricRow) -> html.Div:
    score = max(0.0, min(row.score, 1.0)) * 100
    track: list[Any] = [
        html.Span(
            className="bar-fill fail" if not row.passed else "bar-fill",
            style={"width": f"{score:.0f}%"},
        )
    ]
    if row.threshold is not None and 0 <= row.threshold <= 1:
        track.append(
            html.Span(
                className="bar-thresh", style={"left": f"{row.threshold * 100:.0f}%"}
            )
        )
    kind = KIND_LABELS.get(row.kind, row.kind)
    sub = f"{kind} · {row.model_used}" if row.model_used else kind
    pieces: list[Any] = [
        html.Div(
            [
                html.Span(row.name, className="cue cue-strong"),
                html.Span(sub, className="met-sub"),
            ],
            className="met-name",
        ),
        html.Div(
            [
                html.Span(
                    f"{row.score:.2f}",
                    className="met-score fail" if not row.passed else "met-score",
                ),
                html.Span(track, className="bar-track", **{"aria-hidden": "true"}),
            ],
            className="met-bar",
        ),
        html.Div(
            f"≥ {row.threshold:.2f}" if row.threshold is not None else "no threshold",
            className="met-thresh",
        ),
        html.Div(state_word("pass" if row.passed else "fail"), className="met-result"),
        html.Div(row.reasoning or "", className="met-reason"),
    ]
    return html.Div(pieces, className="eval-row")


def _code_average_header(avg_count: tuple[float, int]) -> html.P:
    avg, count = avg_count
    label = "score" if count == 1 else "scores"
    return html.P(
        [
            "Code grader average ",
            html.Span(f"{avg:.2f}", className="cue cue-strong"),
            f" across {count} {label}.",
        ],
        className="muted-note pad",
    )


def _evaluations_section(
    groups: dict[str, list[MetricRow]],
    code_metric_average: tuple[float, int] | None = None,
) -> html.Section:
    blocks: list[Any] = []
    if code_metric_average is not None:
        blocks.append(_code_average_header(code_metric_average))
    rows = [r for kind_rows in groups.values() for r in kind_rows]
    if rows:
        blocks.append(
            html.Div(
                [
                    html.Span("Metric"),
                    html.Span("Score"),
                    html.Span("Threshold"),
                    html.Span("Result"),
                    html.Span("Reasoning"),
                ],
                className="eval-row eval-head",
                **{"aria-hidden": "true"},
            )
        )
        blocks.extend(_metric_row_div(r) for r in rows)
    else:
        blocks.append(
            html.P(
                "No evaluations were saved for this case.", className="muted-note pad"
            )
        )

    return html.Section(
        [
            html.Div(
                [
                    html.H2("Evaluations", className="h-sm"),
                    html.P(
                        "Score, threshold and the judge's reasoning for every "
                        "metric on this case.",
                        className="panel-sub",
                    ),
                ],
                className="sheet-head",
            ),
            html.Div(blocks, className="evals"),
        ],
        className="sheet hd-explorer-section",
    )


def _detail_panel(
    run: LoadedRun,
    detail: CaseDetail,
    run_text: str | None = None,
    colour: RevisionColour | None = None,
) -> html.Div:
    if detail.conversation.turns is not None:
        cue_names = [
            tc.name for t in detail.conversation.turns for tc in t.tool_invocations
        ]
    else:
        cue_names = [tc.name for tc in detail.conversation.tool_calls]
    return html.Div(
        [
            _case_header_section(detail.header, run_text, colour),
            _agent_snapshot_section(detail.agent_snapshot),
            _conversation_section(
                detail.conversation, detail.agent_snapshot.model_name
            ),
            _expected_tools_section(detail.expected_tools_coverage, cue_names),
            _evaluations_section(detail.evaluations, detail.code_metric_average),
        ],
        className="detail hd-explorer-detail",
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def render_explorer(state: dict[str, Any], runs: list[LoadedRun]) -> html.Div:
    if not runs:
        return html.Div(_empty_no_runs(), className="page explorer-empty")

    numbers = run_numbers(runs)
    colours = revision_colour_map(runs)

    runs_collapsed = bool(state.get("explorer_runs_collapsed", False))
    queue = list(state.get("compare_queue") or [])
    grid_cls = "hd-explorer-grid"
    if runs_collapsed:
        grid_cls += " hd-explorer-grid--collapsed"

    def runs_column(active_run_id: str | None) -> html.Div:
        if runs_collapsed:
            return _runs_column_collapsed(runs)
        return _runs_column_expanded(runs, active_run_id, queue, numbers, colours)

    # No selection opens the latest run. A selection that names no loaded
    # run says so; it never shows a different run in its place.
    ref = state.get("explorer_run_id")
    if ref:
        run = select_run(runs, str(ref))
        if run is None:
            return html.Div(
                [
                    runs_column(None),
                    _cases_column([], None),
                    _run_not_found_panel(str(ref)),
                ],
                className=grid_cls,
            )
    else:
        run = chronological(runs)[-1]
    label = run_label(run, numbers)
    version = run.metadata.prompt_version.version

    cases = list_case_summaries(run)
    case_name = state.get("explorer_case_name")
    if case_name is None or not any(c.name == case_name for c in cases):
        case_name = cases[0].name if cases else None

    col_runs = runs_column(run.run_id)
    col_cases = _cases_column(cases, case_name, label)

    if case_name is None:
        col_detail: html.Div = _placeholder_case_panel()
    else:
        detail = build_case_detail(run, case_name, SEED_CONVERSATIONS)
        if detail is None:
            col_detail = _placeholder_case_panel()
        else:
            col_detail = _detail_panel(
                run, detail, label, revision_colour(version, colours)
            )

    return html.Div([col_runs, col_cases, col_detail], className=grid_cls)
