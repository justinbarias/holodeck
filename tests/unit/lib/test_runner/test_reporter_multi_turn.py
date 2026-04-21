"""Tests for multi-turn markdown/JSON reporter output (T034–T037)."""

from __future__ import annotations

import json

import pytest

from holodeck.lib.test_runner.reporter import generate_markdown_report
from holodeck.models.test_result import (
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
def test_json_reporter_includes_turns_field() -> None:
    multi = _multi_turn_report()
    single = _single_turn_report()
    multi_json = json.loads(multi.model_dump_json())
    single_json = json.loads(single.model_dump_json())
    assert multi_json["results"][0]["turns"] is not None
    assert len(multi_json["results"][0]["turns"]) == 3
    assert single_json["results"][0]["turns"] is None
