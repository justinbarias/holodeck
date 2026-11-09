"""Offline coverage for the live verify tier helpers (no Docker, no providers)."""

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

from holodeck.lib.test_runner.code_grader import GraderContext
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
    ToolInvocation,
    TurnResult,
)
from scripts.verify import infra
from scripts.verify.agui import check_turn, recall_problems, tool_call_names
from scripts.verify.live import (
    case_checks,
    leaked_credentials,
    load_report,
    no_pass_failures,
    prepare_config,
    service_name,
)
from scripts.verify.procs import Check

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "verify"


def _tool_run(name: str) -> list[dict[str, object]]:
    return [
        {"type": "RUN_STARTED"},
        {"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": name},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": "{}"},
        {"type": "TOOL_CALL_END", "toolCallId": "c1"},
        {"type": "TEXT_MESSAGE_START", "messageId": "m"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": "done"},
        {"type": "TEXT_MESSAGE_END", "messageId": "m"},
        {"type": "RUN_FINISHED"},
    ]


# ---------------------------------------------------------------------------
# AG-UI checks
# ---------------------------------------------------------------------------


def test_expect_tool_name_matches_prefixed_mcp_name() -> None:
    events = _tool_run("mcp__holodeck_tools__smoke_echo")
    assert tool_call_names(events) == ["mcp__holodeck_tools__smoke_echo"]
    assert check_turn(events, expect_tool_name="smoke_echo") == []


def test_expect_tool_name_fails_on_other_tool() -> None:
    problems = check_turn(_tool_run("ToolSearch"), expect_tool_name="smoke_echo")
    assert problems == ["no TOOL_CALL_START for 'smoke_echo': ['ToolSearch']"]


def test_expect_tool_name_fails_without_tool_events() -> None:
    events = [e for e in _tool_run("x") if not str(e["type"]).startswith("TOOL")]
    problems = check_turn(events, expect_tool_name="smoke_echo")
    assert problems[0].startswith("tool turn missing TOOL_CALL_START")
    assert "no TOOL_CALL_START for 'smoke_echo': []" in problems


def test_recall_requires_own_word_and_rejects_other_thread() -> None:
    assert (
        recall_problems(
            "Your word was bravo12.", expected="BRAVO12", forbidden="ALPHA9"
        )
        == []
    )
    assert recall_problems("I don't know.", expected="BRAVO12", forbidden="ALPHA9") == [
        "reply does not contain 'BRAVO12': \"I don't know.\""
    ]
    assert recall_problems(
        "bravo12 or alpha9", expected="BRAVO12", forbidden="ALPHA9"
    ) == ["reply contains 'ALPHA9' from another thread"]


# ---------------------------------------------------------------------------
# Agent config preparation
# ---------------------------------------------------------------------------


def _config() -> dict[str, object]:
    return {
        "embedding_provider": {"provider": "openai"},
        "tools": [
            {"name": "smoke_echo", "type": "function"},
            {"name": "verify_kb", "type": "vectorstore"},
            {"name": "verify_handbook", "type": "hierarchical_document"},
            {"name": "skillprobe", "type": "skill", "allowed_tools": ["verify_kb"]},
        ],
        "openai": {"agents": {"helper": {"tools": ["verify_kb", "smoke_echo"]}}},
        "observability": {"enabled": True, "service_name": "svc"},
        "test_cases": [
            {"name": "fn", "turns": [{"expected_tools": ["smoke_echo"]}]},
            {"name": "kb", "turns": [{"expected_tools": [{"name": "verify_kb"}]}]},
            {"name": "hd", "expected_tools": ["verify_handbook_search"]},
            {"name": "none", "turns": [{"input": "hi"}]},
        ],
    }


def _names(items: object) -> list[object]:
    assert isinstance(items, list)
    return [item["name"] for item in items if isinstance(item, dict)]


def test_prepare_config_drops_rag_tools_cases_and_references() -> None:
    original = _config()
    data, dropped = prepare_config(original, drop_rag=True, traces=True)
    assert dropped == ["kb", "hd"]
    assert data["tools"] == [
        {"name": "smoke_echo", "type": "function"},
        {"name": "skillprobe", "type": "skill", "allowed_tools": []},
    ]
    assert "embedding_provider" not in data
    assert data["openai"] == {"agents": {"helper": {"tools": ["smoke_echo"]}}}
    assert _names(data["test_cases"]) == ["fn", "none"]
    assert service_name(data) == "svc"
    assert original == _config()  # input left untouched


def test_prepare_config_keeps_everything_and_disables_traces() -> None:
    data, dropped = prepare_config(_config(), drop_rag=False, traces=False)
    assert dropped == []
    assert len(data["tools"]) == 4  # type: ignore[arg-type]
    assert data["observability"] == {"enabled": False, "service_name": "svc"}
    assert service_name(data) is None


@pytest.mark.parametrize("fixture", ["live-full-claude.yaml", "live-full-openai.yaml"])
def test_live_fixtures_without_rag_keep_non_rag_cases(fixture: str) -> None:
    import yaml

    raw = yaml.safe_load((FIXTURES / fixture).read_text(encoding="utf-8"))
    data, dropped = prepare_config(raw, drop_rag=True, traces=True)
    assert dropped == [
        "vectorstore tool",
        "chromadb vectorstore tool",
        "hierarchical document tool",
    ]
    kept = set(_names(data["tools"]))
    assert {"smoke_echo", "leak_probe", "verify_mcp"} <= kept
    assert "embedding_provider" not in data


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _result(name: str, passed: bool, turn: TurnResult) -> TestResult:
    return TestResult(
        test_name=name,
        test_input=turn.input,
        passed=passed,
        execution_time_ms=1,
        timestamp="2026-01-01T00:00:00Z",
        turns=[turn],
    )


def _turn(**kwargs: object) -> TurnResult:
    return TurnResult(
        turn_index=0, input="go", execution_time_ms=1, passed=True, **kwargs
    )


def _report(*results: TestResult) -> TestReport:
    return TestReport(
        agent_name="verify-live",
        agent_config_path="agent.yaml",
        results=list(results),
        summary=ReportSummary(
            total_tests=len(results),
            passed=sum(r.passed for r in results),
            failed=sum(not r.passed for r in results),
            pass_rate=0.0,
            total_duration_ms=1,
        ),
        timestamp="2026-01-01T00:00:00Z",
        holodeck_version="0.0.0",
    )


def test_case_checks_reports_pass_and_first_failure_reason() -> None:
    grader = MetricResult(
        metric_name="tool_result_contains", kind="code", score=0.0, passed=False
    )
    report = _report(
        _result("fn", True, _turn(tool_calls=["smoke_echo"])),
        _result("tools", False, _turn(tools_matched=False, expected_tools=["x"])),
        _result("grader", False, _turn(metric_results=[grader])),
        _result("error", False, _turn(errors=["boom"])),
    )
    rows = case_checks("claude", report)
    assert [(r.name, r.status) for r in rows] == [
        ("claude: case fn", "PASS"),
        ("claude: case tools", "FAIL"),
        ("claude: case grader", "FAIL"),
        ("claude: case error", "FAIL"),
    ]
    assert rows[0].detail == "tools: smoke_echo"
    assert rows[1].detail == "turn 0: expected tools ['x'] not called (called: [])"
    assert rows[2].detail == "turn 0: grader failed: tool_result_contains"
    assert rows[3].detail == "turn 0: boom"


def test_load_report_reads_newest_results_file(tmp_path: Path) -> None:
    assert load_report(tmp_path) is None
    folder = tmp_path / "results" / "verify-live"
    folder.mkdir(parents=True)
    report = _report(_result("fn", True, _turn()))
    (folder / "run.json").write_text(
        json.dumps({"report": report.model_dump(mode="json"), "metadata": {}})
    )
    loaded = load_report(tmp_path)
    assert loaded is not None
    assert [r.test_name for r in loaded.results] == ["fn"]
    broken = folder / "broken.json"
    broken.write_text("{not json")
    newer = (folder / "run.json").stat().st_mtime + 10
    os.utime(broken, (newer, newer))
    assert load_report(tmp_path) is None


def test_leaked_credentials_names_only_found_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERIFY_A", "secret-value-aaaa")
    monkeypatch.setenv("VERIFY_B", "secret-value-bbbb")
    monkeypatch.setenv("VERIFY_SHORT", "abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "log.txt").write_text("x secret-value-bbbb abc")
    names = ["VERIFY_A", "VERIFY_B", "VERIFY_SHORT", "VERIFY_UNSET"]
    assert leaked_credentials(tmp_path, names) == ["VERIFY_B"]


def test_no_pass_failures_ignores_non_case_rows() -> None:
    checks = [
        Check("claude: no credential in artifacts", "PASS"),
        Check("claude: case fn", "FAIL", "x"),
        Check("openai: agui thread A turn 1", "PASS"),
    ]
    rows = no_pass_failures(checks, ["claude", "openai"])
    assert [(r.name, r.status) for r in rows] == [("claude: overall", "FAIL")]


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def test_run_args_bind_loopback_and_pin_image() -> None:
    args = infra.run_args(infra.ASPIRE)
    assert args[:3] == ["run", "-d", "--name"]
    assert "127.0.0.1:4317:18889" in args
    assert "ASPIRE_DASHBOARD_UNSECURED_ALLOW_ANONYMOUS=true" in args
    assert args[-1] == "mcr.microsoft.com/dotnet/aspire-dashboard:13.5.2"
    assert not args[-1].endswith(":latest")
    assert infra.run_args(infra.QDRANT)[-1] == "qdrant/qdrant:v1.17.1"
    chroma = infra.run_args(infra.CHROMA)
    assert chroma[-1] == "chromadb/chroma:1.5.9"
    assert "127.0.0.1:8765:8000" in chroma


def test_ensure_without_docker_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(infra.shutil, "which", lambda _name: None)
    status = infra.ensure(infra.QDRANT, timeout=0.1)
    assert not status.ready
    assert not status.managed
    assert status.detail == "Docker unavailable: docker CLI not found"
    assert infra.stop_hint([status]) is None


def test_stop_hint_lists_only_managed_containers() -> None:
    managed = infra.InfraStatus(infra.QDRANT, True, "started", managed=True)
    reused = infra.InfraStatus(infra.ASPIRE, True, "reusing existing service")
    assert infra.stop_hint([managed, reused]) == "docker rm -f holodeck-verify-qdrant"


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def _graders() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_graders_under_test", FIXTURES / "verify_graders.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(
    name: str, result: str, args: dict[str, object] | None = None
) -> ToolInvocation:
    return ToolInvocation(name=name, args=args or {}, result=result, bytes=len(result))


def _ctx(
    config: Mapping[str, object], response: str, *calls: ToolInvocation
) -> GraderContext:
    return GraderContext(
        turn_input="go",
        agent_response=response,
        ground_truth=None,
        tool_invocations=calls,
        retrieval_context=None,
        turn_index=0,
        test_case_name="case",
        turn_config=dict(config),
    )


def test_tool_result_contains_needs_marker_in_named_tool_result() -> None:
    graders = _graders()
    config = {"tool": "verify_marker", "marker": "MCP-TOOL-RESULT: "}
    good = _call(
        name="mcp__verify_mcp__verify_marker", result='{"result":"MCP-TOOL-RESULT: x"}'
    )
    other = _call(name="smoke_echo", result="MCP-TOOL-RESULT: x")
    assert graders.tool_result_contains(_ctx(config, "", good)) is True
    assert graders.tool_result_contains(_ctx(config, "", other)) is False
    with pytest.raises(ValueError, match="turn_config.tool"):
        graders.tool_result_contains(_ctx({"marker": "m"}, "", good))


def test_response_contains_is_case_insensitive() -> None:
    graders = _graders()
    assert graders.response_contains(_ctx({"marker": "SKILL-ACK"}, "skill-ack")) is True
    assert graders.response_contains(_ctx({"marker": "SKILL-ACK"}, "nope")) is False


@pytest.mark.parametrize("tool", ["Task", "Agent"])
def test_claude_subagent_ran_accepts_task_and_agent(tool: str) -> None:
    graders = _graders()
    config = {"subagent": "verify_helper", "marker": "HELPER-ACK"}
    call = _call(
        name=tool, args={"subagent_type": "verify_helper"}, result="HELPER-ACK"
    )
    wrong = _call(
        name=tool, args={"subagent_type": "general-purpose"}, result="HELPER-ACK"
    )
    assert graders.claude_subagent_ran(_ctx(config, "", call)) is True
    assert graders.claude_subagent_ran(_ctx(config, "", wrong)) is False


def test_rag_skips_when_any_vector_store_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.verify import infra
    from scripts.verify.live import rag_skip_reason

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    up = infra.InfraStatus(infra.QDRANT, True, "started")
    down = infra.InfraStatus(infra.CHROMA, False, "port 8765 in use")
    assert rag_skip_reason([up, up]) is None
    assert rag_skip_reason([up, down]) == "chroma unavailable: port 8765 in use"
