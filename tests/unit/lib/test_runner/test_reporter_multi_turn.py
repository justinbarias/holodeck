"""Tests for multi-turn markdown/JSON reporter output (T034–T037, T023–T024)."""

from __future__ import annotations

import json

import pytest

from holodeck.lib.test_runner.reporter import generate_markdown_report
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
    TurnResult,
)


def _turn(idx: int, **kw) -> TurnResult:
    return TurnResult(
        turn_index=idx,
        input=kw.pop("input", f"q{idx}"),
        response=kw.pop("response", f"r{idx}"),
        ground_truth=None,
        metric_results=[],
        passed=kw.pop("passed", True),
        execution_time_ms=kw.pop("execution_time_ms", 10 + idx),
        errors=kw.pop("errors", []),
        skipped=kw.pop("skipped", False),
    )


def _multi_turn_report() -> TestReport:
    turns = [_turn(0, input="hi"), _turn(1, input="who?"), _turn(2, input="joke?")]
    result = TestResult(
        test_name="chit-chat",
        test_input="hi\n---\nwho?\n---\njoke?",
        agent_response="joke response",
        passed=True,
        execution_time_ms=30,
        timestamp="2026-04-20T00:00:00+00:00",
        turns=turns,
    )
    return TestReport(
        agent_name="agent-a",
        agent_config_path="agent.yaml",
        results=[result],
        summary=ReportSummary(
            total_tests=1, passed=1, failed=0, pass_rate=100.0, total_duration_ms=30
        ),
        timestamp="2026-04-20T00:00:01+00:00",
        holodeck_version="0.1.0",
    )


def _single_turn_report() -> TestReport:
    result = TestResult(
        test_name="legacy",
        test_input="hello",
        agent_response="hi",
        passed=True,
        execution_time_ms=5,
        timestamp="2026-04-20T00:00:00+00:00",
    )
    return TestReport(
        agent_name="agent-a",
        agent_config_path="agent.yaml",
        results=[result],
        summary=ReportSummary(
            total_tests=1, passed=1, failed=0, pass_rate=100.0, total_duration_ms=5
        ),
        timestamp="2026-04-20T00:00:01+00:00",
        holodeck_version="0.1.0",
    )


@pytest.mark.unit
def test_markdown_renders_per_turn_rows() -> None:
    report = _multi_turn_report()
    md = generate_markdown_report(report)
    # Each turn must appear as its own indented entry with the input snippet.
    for snippet in ("turn 0", "turn 1", "turn 2"):
        assert snippet.lower() in md.lower()
    assert "hi" in md and "who?" in md and "joke?" in md


@pytest.mark.unit
def test_markdown_hierarchy_parent_and_turns() -> None:
    report = _multi_turn_report()
    md = generate_markdown_report(report)
    # Parent test section is present AND a per-turn block follows.
    assert "### Test: chit-chat" in md
    assert "#### Turns" in md
    # Each turn block uses a deeper heading.
    assert md.count("##### Turn ") == 3


@pytest.mark.unit
def test_single_turn_markdown_no_turns_section() -> None:
    report = _single_turn_report()
    md = generate_markdown_report(report)
    assert "#### Turns" not in md
    assert "##### Turn " not in md


@pytest.mark.unit
def test_markdown_shows_per_turn_metric_scores() -> None:
    """Rendered markdown includes each turn's metric scores (T023)."""
    turns = [
        TurnResult(
            turn_index=0,
            input="q0",
            response="r0",
            ground_truth="gt0",
            metric_results=[
                MetricResult(
                    metric_name="bleu",
                    kind="standard",
                    score=0.8,
                    threshold=0.5,
                    passed=True,
                    scale="0-1",
                )
            ],
            passed=True,
            execution_time_ms=10,
            tools_matched=True,
            expected_tools=["subtract"],
            tool_calls=["subtract"],
        ),
    ]
    result = TestResult(
        test_name="metrics-per-turn",
        test_input="q0",
        agent_response="r0",
        passed=True,
        execution_time_ms=10,
        timestamp="2026-04-20T00:00:00+00:00",
        turns=turns,
    )
    report = TestReport(
        agent_name="a",
        agent_config_path="a.yaml",
        results=[result],
        summary=ReportSummary(
            total_tests=1, passed=1, failed=0, pass_rate=100.0, total_duration_ms=10
        ),
        timestamp="2026-04-20T00:00:01+00:00",
        holodeck_version="0.1.0",
    )
    md = generate_markdown_report(report)
    assert "bleu" in md
    assert "0.8" in md
    # Tool match glyph is present on the turn row.
    assert "✅" in md


@pytest.mark.unit
def test_failure_pinpoints_turn_index() -> None:
    """A failing turn's rendered line names `turn 2` and the failing metric (T024)."""
    turns = [
        TurnResult(
            turn_index=0,
            input="q0",
            response="r0",
            metric_results=[],
            passed=True,
            execution_time_ms=5,
        ),
        TurnResult(
            turn_index=1,
            input="q1",
            response="r1",
            metric_results=[],
            passed=True,
            execution_time_ms=5,
        ),
        TurnResult(
            turn_index=2,
            input="q2",
            response="wrong",
            ground_truth="25587",
            metric_results=[
                MetricResult(
                    metric_name="bleu",
                    kind="standard",
                    score=0.1,
                    threshold=0.5,
                    passed=False,
                    scale="0-1",
                )
            ],
            passed=False,
            execution_time_ms=8,
            errors=["expected tool(s) not called in this turn: subtract"],
            expected_tools=["subtract"],
            tool_calls=["lookup"],
            tools_matched=False,
        ),
    ]
    result = TestResult(
        test_name="pinpoint",
        test_input="q0\n---\nq1\n---\nq2",
        agent_response="wrong",
        passed=False,
        execution_time_ms=18,
        timestamp="2026-04-20T00:00:00+00:00",
        errors=["[turn 2] expected tool(s) not called in this turn: subtract"],
        turns=turns,
    )
    report = TestReport(
        agent_name="a",
        agent_config_path="a.yaml",
        results=[result],
        summary=ReportSummary(
            total_tests=1, passed=0, failed=1, pass_rate=0.0, total_duration_ms=18
        ),
        timestamp="2026-04-20T00:00:01+00:00",
        holodeck_version="0.1.0",
    )
    md = generate_markdown_report(report)
    # Failing turn is identified by index 2.
    assert "Turn 2" in md
    # Failing metric name appears.
    assert "bleu" in md
    # Missing tool name appears in the errors for the failing turn.
    assert "subtract" in md


@pytest.mark.unit
def test_json_reporter_includes_turns_field() -> None:
    multi = _multi_turn_report()
    single = _single_turn_report()
    multi_json = json.loads(multi.model_dump_json())
    single_json = json.loads(single.model_dump_json())
    assert multi_json["results"][0]["turns"] is not None
    assert len(multi_json["results"][0]["turns"]) == 3
    assert single_json["results"][0]["turns"] is None
