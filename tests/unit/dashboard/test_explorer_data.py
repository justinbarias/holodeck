"""Explorer data-assembly tests (US5 T401–T407)."""

from __future__ import annotations

import pytest

from holodeck.dashboard.explorer_data import (
    LARGE_TOOL_RESULT_BYTES,
    CaseDetail,
    CaseSummary,
    ToolCallView,
    build_case_detail,
    list_case_summaries,
    select_run,
)
from holodeck.dashboard.seed_data import build_seed_runs


@pytest.fixture(scope="module")
def seed_runs():
    return build_seed_runs()


# ---------- T401: select_run ----------------------------------------------


@pytest.mark.unit
def test_select_run_returns_match_by_timestamp_id(seed_runs):
    target = seed_runs[3]
    assert select_run(seed_runs, target.report.timestamp) is target


@pytest.mark.unit
def test_select_run_returns_none_for_missing(seed_runs):
    assert select_run(seed_runs, "no-such-id") is None


@pytest.mark.unit
def test_select_run_accepts_none(seed_runs):
    assert select_run(seed_runs, None) is None


# ---------- T402: list_case_summaries -------------------------------------


@pytest.mark.unit
def test_list_case_summaries_shape(seed_runs):
    run = seed_runs[-1]
    summaries = list_case_summaries(run)
    assert summaries, "expected non-empty list"
    assert all(isinstance(s, CaseSummary) for s in summaries)
    first = summaries[0]
    assert first.name == run.report.results[0].test_name
    assert first.passed == run.report.results[0].passed


@pytest.mark.unit
def test_list_case_summaries_geval_and_rag_avg(seed_runs):
    run = seed_runs[-1]
    summaries = list_case_summaries(run)
    # at least one geval / rag case should populate scores
    assert any(s.geval_score is not None for s in summaries)
    assert any(s.rag_avg_score is not None for s in summaries)


# ---------- T403: build_case_detail ---------------------------------------


@pytest.mark.unit
def test_build_case_detail_header_and_agent_snapshot(seed_runs):
    run = seed_runs[-1]
    case_name = run.report.results[0].test_name
    detail = build_case_detail(run, case_name, conversations_map={})
    assert isinstance(detail, CaseDetail)
    # header
    assert detail.header.run_timestamp == run.report.timestamp
    assert detail.header.prompt_version == run.metadata.prompt_version.version
    assert detail.header.model_name == run.metadata.agent_config.model.name
    assert detail.header.passed == run.report.results[0].passed
    # agent snapshot
    snap = detail.agent_snapshot
    assert snap.model_provider == str(run.metadata.agent_config.model.provider)
    assert snap.model_name == run.metadata.agent_config.model.name
    assert snap.prompt_version == run.metadata.prompt_version.version
    assert snap.prompt_source == run.metadata.prompt_version.source
    assert isinstance(snap.tools, list)


@pytest.mark.unit
def test_build_case_detail_missing_case_returns_none(seed_runs):
    run = seed_runs[-1]
    assert build_case_detail(run, "no_such_case", {}) is None


# ---------- T404: ToolCallView dataclass ----------------------------------


@pytest.mark.unit
def test_tool_call_view_large_flag_from_bytes():
    view = ToolCallView(
        name="lookup_order",
        args={"order_id": "A-1"},
        result={"status": "ok"},
        result_size_bytes=501,
        duration_ms=None,
        error=None,
    )
    assert view.large is True


@pytest.mark.unit
def test_tool_call_view_not_large_under_threshold():
    view = ToolCallView(
        name="noop",
        args={},
        result={"x": 1},
        result_size_bytes=LARGE_TOOL_RESULT_BYTES,
        duration_ms=None,
        error=None,
    )
    assert view.large is False
    assert LARGE_TOOL_RESULT_BYTES == 500


# ---------- T405: expected-tools coverage ---------------------------------


@pytest.mark.unit
def test_expected_tools_matched_and_missed(seed_runs):
    run = seed_runs[-1]
    # find a case with expected tools
    case = next(c for c in run.report.results if c.expected_tools)
    detail = build_case_detail(run, case.test_name, {})
    assert detail is not None
    coverage = detail.expected_tools_coverage
    assert coverage.total == len(case.expected_tools or [])
    # case-insensitive comparison
    called_lower = {t.lower() for t in case.tool_calls or []}
    expected_lower = {t.lower() for t in case.expected_tools or []}
    assert coverage.matched == len(called_lower & expected_lower)
    assert coverage.missed == len(expected_lower - called_lower)


@pytest.mark.unit
def test_expected_tools_empty_ok(seed_runs):
    run = seed_runs[-1]
    # find a case with no expected tools
    case = next((c for c in run.report.results if not c.expected_tools), None)
    if case is None:
        pytest.skip("seed lacks case with no expected tools")
    detail = build_case_detail(run, case.test_name, {})
    assert detail is not None
    assert detail.expected_tools_coverage.total == 0
    assert detail.expected_tools_coverage.matched == 0


# ---------- T406: conversations_map fallback ------------------------------


@pytest.mark.unit
def test_build_case_detail_uses_seed_default_fallback(seed_runs):
    run = seed_runs[-1]
    case_name = run.report.results[0].test_name
    # with an empty map and no legacy fallback entry, conversation is empty/user-only
    detail = build_case_detail(run, case_name, {})
    assert detail is not None
    # conversation exists even if empty
    assert detail.conversation is not None


@pytest.mark.unit
def test_build_case_detail_uses_conversations_map_when_present():
    # Synthetic run with structure minimal enough for detail assembly
    runs = build_seed_runs()
    run = runs[-1]
    case_name = run.report.results[0].test_name
    convo = {
        "user": "hi",
        "assistant": "hello",
        "tool_calls": [
            {
                "name": "t",
                "args": {"a": 1},
                "result": {"ok": True},
            }
        ],
    }
    detail = build_case_detail(run, case_name, {case_name: convo})
    assert detail is not None
    assert detail.conversation.user == "hi"
    assert detail.conversation.assistant == "hello"
    assert len(detail.conversation.tool_calls) == 1
    assert detail.conversation.tool_calls[0].name == "t"


# ---------- T407: evaluations ordering ------------------------------------


@pytest.mark.unit
def test_evaluations_ordering_and_omits_empty(seed_runs):
    run = seed_runs[-1]
    case_name = run.report.results[0].test_name
    detail = build_case_detail(run, case_name, {})
    assert detail is not None
    keys = list(detail.evaluations.keys())
    # geval → rag → standard → code order where present
    expected_order = [k for k in ("geval", "rag", "standard", "code") if k in keys]
    assert keys == expected_order
    # no empty groups
    assert all(len(v) > 0 for v in detail.evaluations.values())
