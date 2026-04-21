"""Multi-turn explorer_data tests (US5 T003–T008)."""

from __future__ import annotations

import pytest

from holodeck.dashboard.explorer_data import (
    CaseDetail,
    CaseSummary,
    MetricRow,
    ToolCallView,
    TurnView,
    build_case_detail,
    list_case_summaries,
)
from holodeck.dashboard.seed_data import build_multi_turn_seed_case


@pytest.fixture(scope="module")
def multi_turn_run():
    return build_multi_turn_seed_case()


# ---------- T003: single-turn case detail unchanged -----------------------


@pytest.mark.unit
def test_single_turn_case_detail_unchanged(multi_turn_run) -> None:
    detail = build_case_detail(multi_turn_run, "single_turn_legacy", {})
    assert isinstance(detail, CaseDetail)
    assert detail.conversation.turns is None


# ---------- T004: build_case_detail emits TurnViews for multi-turn --------


@pytest.mark.unit
def test_build_case_detail_emits_turn_views_when_turns_present(
    multi_turn_run,
) -> None:
    detail = build_case_detail(multi_turn_run, "three_turn_chit_chat", {})
    assert detail is not None
    assert detail.conversation.turns is not None
    assert len(detail.conversation.turns) == 3
    assert all(isinstance(t, TurnView) for t in detail.conversation.turns)


# ---------- T005: CaseSummary carries turn counts -------------------------


@pytest.mark.unit
def test_case_summary_carries_turn_counts(multi_turn_run) -> None:
    summaries = {s.name: s for s in list_case_summaries(multi_turn_run)}

    legacy = summaries["single_turn_legacy"]
    assert legacy.turns_total is None
    assert legacy.turns_passed is None
    assert legacy.turns_failed is None

    chit = summaries["three_turn_chit_chat"]
    assert chit.turns_total == 3
    assert chit.turns_passed == 3
    assert chit.turns_failed == 0

    failing = summaries["four_turn_math_failing"]
    assert failing.turns_total == 4
    assert failing.turns_passed == 3
    assert failing.turns_failed == 1
    assert isinstance(failing, CaseSummary)


# ---------- T006: TurnView field coverage ---------------------------------


@pytest.mark.unit
def test_turn_view_fields(multi_turn_run) -> None:
    detail = build_case_detail(multi_turn_run, "four_turn_math_failing", {})
    assert detail is not None
    assert detail.conversation.turns is not None
    tv = detail.conversation.turns[0]
    # Turn 0 has one tool invocation
    assert tv.turn_index == 0
    assert tv.input.startswith("Look up")
    assert tv.response == "Got it."
    assert len(tv.tool_invocations) == 1
    assert isinstance(tv.tool_invocations[0], ToolCallView)
    assert tv.tool_invocations[0].name == "lookup_order"
    assert tv.tools_matched is True
    assert tv.arg_match_details is None
    assert tv.errors == []
    assert tv.skipped is False
    assert tv.execution_time_ms == 120
    assert isinstance(tv.metric_results, list)


# ---------- T007: skipped turn marked distinctly --------------------------


@pytest.mark.unit
def test_skipped_turn_marked_in_view(multi_turn_run) -> None:
    # Swap in a skipped turn on the 3-turn case to assert the state marker.
    from holodeck.models.test_result import TurnResult

    base = multi_turn_run.model_copy(deep=True)
    three = next(
        c for c in base.report.results if c.test_name == "three_turn_chit_chat"
    )
    assert three.turns is not None
    three.turns[1] = TurnResult(
        turn_index=1,
        input=three.turns[1].input,
        response=None,
        metric_results=[],
        passed=False,
        execution_time_ms=0,
        errors=["session unrecoverable"],
        skipped=True,
    )
    detail = build_case_detail(base, "three_turn_chit_chat", {})
    assert detail is not None
    assert detail.conversation.turns is not None
    skipped = detail.conversation.turns[1]
    assert skipped.skipped is True
    assert skipped.state == "skipped"


# ---------- T008: metric kind "code" surfaced on TurnView -----------------


@pytest.mark.unit
def test_metric_kind_code_surfaced_on_turn_view(multi_turn_run) -> None:
    detail = build_case_detail(multi_turn_run, "four_turn_math_failing", {})
    assert detail is not None
    assert detail.conversation.turns is not None
    # turn 1 has a metric with kind="code"
    tv1 = detail.conversation.turns[1]
    assert tv1.metric_results, "expected at least one metric on turn 1"
    kinds = {m.kind for m in tv1.metric_results}
    assert "code" in kinds
    # turn 2 also has a failing code metric
    tv2 = detail.conversation.turns[2]
    assert any(
        m.kind == "code" and m.passed is False and isinstance(m, MetricRow)
        for m in tv2.metric_results
    )
