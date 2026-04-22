"""Rendering tests for multi-turn Explorer output (US5 T012–T015)."""

from __future__ import annotations

import json

import pytest
from dash import html

from holodeck.dashboard.explorer_data import (
    ConversationView,
    MetricRow,
    ToolCallView,
    TurnView,
)
from holodeck.dashboard.seed_data import build_multi_turn_seed_case
from holodeck.dashboard.views.explorer import (
    _cases_column,
    _conversation_section,
    _turn_thread_block,
)


def _deep_plotly(node):
    """Recursively convert a Dash tree to dicts via ``to_plotly_json``."""

    if isinstance(node, (list, tuple)):
        return [_deep_plotly(c) for c in node]
    if hasattr(node, "to_plotly_json"):
        d = node.to_plotly_json()
        props = d.get("props", {}) or {}
        new_props = dict(props)
        if "children" in new_props:
            new_props["children"] = _deep_plotly(new_props["children"])
        d = dict(d)
        d["props"] = new_props
        return d
    return node


def _find_all(node, predicate):
    """Walk a Dash component tree and yield nodes matching ``predicate``."""

    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if hasattr(current, "to_plotly_json"):
            if predicate(current):
                yield current
            children = getattr(current, "children", None)
            if children is not None:
                if isinstance(children, (list, tuple)):
                    stack.extend(children)
                else:
                    stack.append(children)


def _make_turn_view(i: int, skipped: bool = False) -> TurnView:
    return TurnView(
        turn_index=i,
        input=f"q{i}",
        response=None if skipped else f"r{i}",
        tool_invocations=[],
        metric_results=[],
        tools_matched=None,
        arg_match_details=None,
        errors=["oops"] if skipped else [],
        skipped=skipped,
        execution_time_ms=10 + i,
        token_usage=None,
        state="skipped" if skipped else "ran",
    )


# ---------- T012: per-turn Details blocks ---------------------------------


@pytest.mark.unit
def test_conversation_section_renders_per_turn_details_when_turns_present() -> None:
    turns = [_make_turn_view(0), _make_turn_view(1), _make_turn_view(2)]
    conv = ConversationView(
        user="",
        assistant="",
        tool_calls=[],
        turns=turns,
    )
    out = _conversation_section(conv, "gpt-x")
    # Find html.Details nodes whose className marks them as per-turn blocks.
    turn_details = list(
        _find_all(
            out,
            lambda n: isinstance(n, html.Details)
            and "hd-explorer-turn" in (getattr(n, "className", "") or ""),
        )
    )
    assert len(turn_details) == 3
    # Each turn-Details summary text mentions the turn index.
    rendered_json = json.dumps(_deep_plotly(out), default=str, ensure_ascii=False)
    assert "Turn 0" in rendered_json
    assert "Turn 1" in rendered_json
    assert "Turn 2" in rendered_json


# ---------- T013: single-turn conversation section byte-identity ----------

_SINGLE_TURN_GOLDEN = {
    "props": {
        "children": [
            {
                "props": {
                    "children": [
                        {
                            "props": {
                                "children": [
                                    {
                                        "props": {
                                            "children": "CONVERSATION",
                                            "className": "eyebrow",
                                        },
                                        "type": "Div",
                                        "namespace": "dash_html_components",
                                    },
                                    {
                                        "props": {"children": "Thread with tool calls"},
                                        "type": "H3",
                                        "namespace": "dash_html_components",
                                    },
                                    {
                                        "props": {
                                            "children": (
                                                "User input, agent response, "
                                                "and every tool invocation that "
                                                "happened in between."
                                            ),
                                            "className": "subtitle",
                                        },
                                        "type": "P",
                                        "namespace": "dash_html_components",
                                    },
                                ],
                                "className": "summary-text",
                            },
                            "type": "Div",
                            "namespace": "dash_html_components",
                        }
                    ]
                },
                "type": "Summary",
                "namespace": "dash_html_components",
            },
            {
                "props": {
                    "children": [
                        {
                            "props": {
                                "children": [
                                    {
                                        "props": {
                                            "children": "USER",
                                            "className": "who",
                                        },
                                        "type": "Div",
                                        "namespace": "dash_html_components",
                                    },
                                    {
                                        "props": {"children": "hi"},
                                        "type": "Div",
                                        "namespace": "dash_html_components",
                                    },
                                ],
                                "className": "bubble user",
                            },
                            "type": "Div",
                            "namespace": "dash_html_components",
                        },
                        {
                            "props": {
                                "children": [
                                    {
                                        "props": {
                                            "children": "AGENT · gpt-x",
                                            "className": "who",
                                        },
                                        "type": "Div",
                                        "namespace": "dash_html_components",
                                    },
                                    {
                                        "props": {
                                            "children": "hello",
                                            "className": "md-assistant",
                                            "link_target": "_blank",
                                        },
                                        "type": "Markdown",
                                        "namespace": "dash_core_components",
                                    },
                                ],
                                "className": "bubble assistant",
                            },
                            "type": "Div",
                            "namespace": "dash_html_components",
                        },
                    ],
                    "className": "thread",
                },
                "type": "Div",
                "namespace": "dash_html_components",
            },
        ],
        "className": "panel hd-explorer-section",
        "open": True,
    },
    "type": "Details",
    "namespace": "dash_html_components",
}


@pytest.mark.unit
def test_single_turn_conversation_section_unchanged() -> None:
    conv = ConversationView(
        user="hi",
        assistant="hello",
        tool_calls=[],
        turns=None,
    )
    out = _conversation_section(conv, "gpt-x")
    assert _deep_plotly(out) == _SINGLE_TURN_GOLDEN


# ---------- T014: cases column turn-count chip ----------------------------


@pytest.mark.unit
def test_cases_column_renders_turn_count_chip_when_present() -> None:
    from holodeck.dashboard.explorer_data import CaseSummary

    multi = CaseSummary(
        name="multi_case",
        passed=False,
        geval_score=None,
        rag_avg_score=None,
        tools_called_count=0,
        turns_total=4,
        turns_passed=3,
        turns_failed=1,
    )
    single = CaseSummary(
        name="single_case",
        passed=True,
        geval_score=None,
        rag_avg_score=None,
        tools_called_count=0,
    )
    col = _cases_column([multi, single], active_case_name=None)
    rendered = json.dumps(_deep_plotly(col), default=str, ensure_ascii=False)
    assert "4 turns · 3/4 passed" in rendered
    # single-turn row should have NO such chip text.
    # The phrase "1 turns" or "None" shouldn't appear for the single case.
    # Assert by counting the chip occurrences.
    assert rendered.count("turns · ") == 1


@pytest.mark.unit
def test_cases_column_no_chip_when_turns_total_none() -> None:
    from holodeck.dashboard.explorer_data import CaseSummary

    single = CaseSummary(
        name="single_only",
        passed=True,
        geval_score=None,
        rag_avg_score=None,
        tools_called_count=0,
    )
    col = _cases_column([single], active_case_name=None)
    rendered = json.dumps(_deep_plotly(col), default=str, ensure_ascii=False)
    assert "turns · " not in rendered


# ---------- T015: failing turn surfaces reason via existing helpers -------


@pytest.mark.unit
def test_failing_turn_surfaces_reason_via_existing_helpers() -> None:
    run = build_multi_turn_seed_case()
    failing_case = next(
        c for c in run.report.results if c.test_name == "four_turn_math_failing"
    )
    assert failing_case.turns is not None
    failing_turn = failing_case.turns[2]

    tv = TurnView(
        turn_index=failing_turn.turn_index,
        input=failing_turn.input,
        response=failing_turn.response,
        tool_invocations=[],
        metric_results=[
            MetricRow(
                kind="code",
                name="numeric",
                score=0.0,
                threshold=0.5,
                passed=False,
                reasoning="Expected 25587, got 65587",
            )
        ],
        tools_matched=False,
        arg_match_details=None,
        errors=list(failing_turn.errors),
        skipped=False,
        execution_time_ms=failing_turn.execution_time_ms,
        token_usage=None,
        state="ran",
    )
    block = _turn_thread_block(tv)
    rendered = json.dumps(_deep_plotly(block), default=str, ensure_ascii=False)
    # The failing metric reasoning is surfaced (via _metric_row_div).
    assert "Expected 25587, got 65587" in rendered
    # The error text is surfaced.
    assert "expected tool(s) not called" in rendered


@pytest.mark.unit
def test_turn_thread_block_uses_tool_call_panel_for_invocations() -> None:
    tool = ToolCallView(
        name="lookup",
        args={"k": 1},
        result={"v": 2},
        result_size_bytes=16,
        duration_ms=5,
        error=None,
    )
    tv = TurnView(
        turn_index=0,
        input="q",
        response="r",
        tool_invocations=[tool],
        metric_results=[],
        tools_matched=True,
        arg_match_details=None,
        errors=[],
        skipped=False,
        execution_time_ms=5,
        token_usage=None,
        state="ran",
    )
    block = _turn_thread_block(tv)
    tool_panels = list(
        _find_all(
            block,
            lambda n: isinstance(n, html.Div)
            and (getattr(n, "className", "") or "") == "tool-call",
        )
    )
    assert len(tool_panels) == 1
