"""Offline coverage for the live verify tier helpers (no Docker, no providers)."""

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest

from holodeck.lib.retrieval import CollectionSchema
from holodeck.lib.test_runner.code_grader import GraderContext
from holodeck.models.test_result import (
    MetricResult,
    ReportSummary,
    TestReport,
    TestResult,
    ToolInvocation,
    TurnResult,
)
from scripts.verify import infra, live
from scripts.verify.agui import check_turn, recall_problems, tool_call_names
from scripts.verify.live import (
    PROVIDERS,
    RagTarget,
    case_checks,
    drop_providers,
    embedding_skip_reason,
    exercised_targets,
    leaked_credentials,
    load_report,
    no_pass_failures,
    prepare_config,
    provider_cases_only,
    provider_checks,
    provider_skip_reasons,
    rag_targets,
    service_name,
    unique_collections,
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
        "postgres vectorstore tool",
        "weaviate vectorstore tool",
        "cosmos vectorstore tool",
        "pinecone vectorstore tool",
        "azure ai search vectorstore tool",
        "hierarchical document tool",
        "postgres hierarchical document tool",
        "weaviate hierarchical document tool",
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
    postgres = infra.run_args(infra.POSTGRES)
    assert postgres[-1] == "pgvector/pgvector:pg17"
    assert "127.0.0.1:54329:5432" in postgres
    assert "POSTGRES_DB=holodeck" in postgres
    assert infra.POSTGRES.container == "holodeck-verify-postgres"
    weaviate = infra.run_args(infra.WEAVIATE)
    assert weaviate[-1] == "cr.weaviate.io/semitechnologies/weaviate:1.40.4"
    assert "127.0.0.1:8089:8080" in weaviate
    assert "127.0.0.1:50059:50051" in weaviate
    assert "DEFAULT_VECTORIZER_MODULE=none" in weaviate
    cosmos = infra.run_args(infra.COSMOS)
    assert cosmos[-3:] == [
        "mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview",
        "--protocol",
        "http",
    ]
    assert "127.0.0.1:8081:8081" in cosmos
    assert infra.COSMOS.ready_timeout > infra.QDRANT.ready_timeout


def test_a_service_without_an_http_endpoint_is_ready_when_its_port_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(infra, "port_open", lambda port, host="127.0.0.1": False)
    assert infra.probe(infra.POSTGRES) == "port(s) [54329] not accepting connections"
    monkeypatch.setattr(infra, "port_open", lambda port, host="127.0.0.1": True)
    assert infra.probe(infra.POSTGRES) is None


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


def test_embedding_key_gates_every_vector_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert embedding_skip_reason() == "OPENAI_API_KEY not set (OpenAI embeddings)"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert embedding_skip_reason() is None


# ---------------------------------------------------------------------------
# Vector store providers
# ---------------------------------------------------------------------------

HOSTED = ("PINECONE_API_KEY", "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_API_KEY")


def _statuses(*down: str) -> dict[str, infra.InfraStatus]:
    return {
        service.container: infra.InfraStatus(
            service, service.label not in down, "port in use"
        )
        for service in live.LOCAL_SERVICES
    }


def test_provider_skip_reasons_name_the_missing_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in HOSTED:
        monkeypatch.delenv(name, raising=False)
    assert provider_skip_reasons(_statuses("chroma")) == {
        "chromadb": "chroma unavailable: port in use",
        "pinecone": ("PINECONE_API_KEY not set (hosted service, no local emulator)"),
        "azure-ai-search": (
            "AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_API_KEY not set "
            "(hosted service, no local emulator)"
        ),
    }
    assert provider_skip_reasons({})["qdrant"] == "qdrant unavailable: not started"


def test_hosted_provider_runs_only_with_every_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "https://x.search.windows.net")
    monkeypatch.delenv("AZURE_AI_SEARCH_API_KEY", raising=False)
    reasons = provider_skip_reasons(_statuses())
    assert "pinecone" not in reasons
    assert reasons == {
        "azure-ai-search": (
            "AZURE_AI_SEARCH_API_KEY not set (hosted service, no local emulator)"
        )
    }


def _live_agent() -> dict[str, object]:
    import yaml

    raw = yaml.safe_load((FIXTURES / "live-full-claude.yaml").read_text("utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_drop_providers_skips_only_that_providers_cases() -> None:
    data, skipped = drop_providers(
        _live_agent(), {"weaviate": "weaviate unavailable: x"}
    )
    assert skipped == {
        "weaviate vectorstore tool": "weaviate unavailable: x",
        "weaviate hierarchical document tool": "weaviate unavailable: x",
    }
    names = _names(data["tools"])
    assert "verify_beacon" not in names
    assert "verify_weaviate_handbook" not in names
    assert {"verify_kb", "verify_handbook", "verify_pg_handbook"} <= set(names)
    assert "embedding_provider" in data


def test_provider_cases_only_keeps_vector_store_cases() -> None:
    data = provider_cases_only(_live_agent())
    assert _names(data["test_cases"]) == [
        "vectorstore tool",
        "chromadb vectorstore tool",
        "postgres vectorstore tool",
        "weaviate vectorstore tool",
        "cosmos vectorstore tool",
        "pinecone vectorstore tool",
        "azure ai search vectorstore tool",
        "hierarchical document tool",
        "postgres hierarchical document tool",
        "weaviate hierarchical document tool",
    ]


def test_every_provider_has_a_vectorstore_tool_and_case_in_both_fixtures() -> None:
    import yaml

    for fixture in ("live-full-claude.yaml", "live-full-openai.yaml"):
        raw = yaml.safe_load((FIXTURES / fixture).read_text("utf-8"))
        tools = raw["tools"]
        vector = {
            t["database"]["provider"]
            for t in tools
            if t["type"] == "vectorstore" and "database" in t
        }
        assert {p.name for p in PROVIDERS} == vector, fixture
        hierarchical = {
            t["database"]["provider"]
            for t in tools
            if t["type"] == "hierarchical_document"
        }
        assert hierarchical == {"qdrant", "postgres", "weaviate"}, fixture
        # Each RAG case expects one tool and its marker is in that tool's corpus.
        by_name = {t["name"]: t for t in tools}
        rag_cases = 0
        for case in raw["test_cases"]:
            config = case["turns"][0]["turn_config"]
            tool = by_name.get(config.get("tool", ""))
            if tool is None or tool["type"] not in live.RAG_TOOL_TYPES:
                continue
            rag_cases += 1
            corpus = "".join(
                f.read_text("utf-8") for f in (FIXTURES / tool["source"]).glob("*.md")
            )
            assert config["marker"] in corpus, case["name"]
            assert case["turns"][0]["expected_tools"] == [tool["name"]]
        assert rag_cases == 10, fixture


def test_no_tool_name_contains_another_tool_name() -> None:
    # expected_tools and the graders match tool names by substring.
    names = _names(_live_agent()["tools"])
    clashes = [(a, b) for a in names for b in names if a != b and str(a) in str(b)]
    assert clashes == []


def test_unique_collections_give_every_remote_tool_its_own_name() -> None:
    original = _live_agent()
    data = unique_collections(original, "claudeab12")
    names: dict[str, str] = {}
    for tool in data["tools"]:  # type: ignore[attr-defined]
        database = tool.get("database")
        if database:
            names[tool["name"]] = database["collection_name"]
    assert names["verify_beacon"] == "verify_claudeab12_beacon"
    assert len(set(names.values())) == len(names) == 10
    assert all(n.startswith("verify_claudeab12_") for n in names.values())
    assert "collection_name" not in original["tools"][4]["database"]  # type: ignore[index]


def test_rag_targets_address_the_collection_the_tool_uses() -> None:
    data = unique_collections(_live_agent(), "t1")
    targets = {t.tool: t for t in rag_targets(data)}
    assert len(targets) == 10
    beacon = targets["verify_beacon"]
    assert beacon.provider == "weaviate"
    assert beacon.schema.name == "verify_t1_beacon"
    assert beacon.schema.dimensions == 1536
    assert beacon.connection["local_grpc_port"] == 50059
    hierarchical = targets["verify_weaviate_handbook"].schema
    assert hierarchical.name == "verify_t1_weaviate_handbook"
    assert "searchable_text" in [f.name for f in hierarchical.fields]
    cosmos = targets["verify_depot"]
    assert cosmos.connection["vector_index"] == "diskANN"
    assert targets["verify_harbor"].connection == {
        "collection_name": "verify_t1_harbor"
    }


def test_exercised_targets_follow_the_cases_that_ran() -> None:
    data = unique_collections(_live_agent(), "t1")
    targets = rag_targets(data)
    got = exercised_targets(data, targets, {"weaviate vectorstore tool"})
    assert [t.tool for t in got] == ["verify_beacon"]
    assert exercised_targets(data, targets, set()) == []


class _FakeAdapter:
    def __init__(self, *, exists: bool, count: int, fail_delete: bool = False) -> None:
        self.exists, self.records, self.fail_delete = exists, count, fail_delete
        self.deleted = self.closed = False

    async def collection_exists(self) -> bool:
        return self.exists

    async def count(self) -> int:
        return self.records

    async def delete_collection(self) -> None:
        if self.fail_delete:
            raise RuntimeError("boom")
        self.deleted = True

    async def close(self) -> None:
        self.closed = True


def _target() -> RagTarget:
    return RagTarget(
        "verify_beacon", "weaviate", CollectionSchema("verify_t1_beacon", 4, ()), {}
    )


def _probe_with(
    monkeypatch: pytest.MonkeyPatch,
    adapter: _FakeAdapter,
    checked: bool,
) -> list[Check]:
    monkeypatch.setattr(live, "create_adapter", lambda *_args: adapter)
    monkeypatch.setattr(live, "SETTLE_SECONDS", 0.0)
    target = _target()
    return provider_checks("claude", [target], [target] if checked else [])


def test_provider_check_passes_when_the_provider_holds_records_then_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(exists=True, count=3)
    [row] = _probe_with(monkeypatch, adapter, checked=True)
    assert (row.name, row.status) == (
        "claude: provider weaviate (verify_beacon) holds records",
        "PASS",
    )
    assert row.detail == "3 records in verify_t1_beacon; deleted"
    assert adapter.deleted and adapter.closed


def test_provider_check_fails_when_the_collection_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(exists=False, count=0)
    [row] = _probe_with(monkeypatch, adapter, checked=True)
    assert row.status == "FAIL"
    assert "fell back to in-memory" in row.detail
    assert adapter.closed and not adapter.deleted


def test_provider_check_fails_on_an_empty_collection_and_still_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(exists=True, count=0)
    [row] = _probe_with(monkeypatch, adapter, checked=True)
    assert row.status == "FAIL"
    assert row.detail == "collection verify_t1_beacon is empty; deleted"
    assert adapter.deleted


def test_unchecked_target_is_deleted_silently_and_a_failed_delete_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(exists=True, count=3)
    assert [r.status for r in _probe_with(monkeypatch, adapter, checked=False)] == [
        "PASS"
    ]
    assert adapter.deleted
    assert _probe_with(monkeypatch, _FakeAdapter(exists=False, count=0), False) == []
    broken = _FakeAdapter(exists=True, count=3, fail_delete=True)
    [row] = _probe_with(monkeypatch, broken, checked=False)
    assert row.status == "FAIL"
    assert row.name == "claude: provider weaviate (verify_beacon) cleanup"
    assert "RuntimeError: boom" in row.detail
    assert broken.closed


def _span_payload(
    service: str, name: str, start: int, result: str
) -> dict[str, object]:
    return {
        "data": {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": service}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": name,
                                    "startTimeUnixNano": str(start),
                                    "attributes": [
                                        {
                                            "key": "gen_ai.tool.call.result",
                                            "value": {"stringValue": result},
                                        }
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ]
        }
    }


def test_probe_spans_keeps_this_run_and_this_service() -> None:
    payload = _span_payload("svc", "execute_tool leak_probe", 200, "x")

    assert len(live.probe_spans(payload, "svc", 100)) == 1
    assert live.probe_spans(payload, "svc", 300) == []  # an earlier run
    assert live.probe_spans(payload, "other", 100) == []
    assert live.probe_spans({"title": "Resource not found"}, "svc", 100) == []


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("LEAK-PROBE-OK token=[REDACTED:github-token]", None),
        ("LEAK-PROBE-OK token=ghp_" + "x" * 36, "unredacted"),
        ("LEAK-PROBE-OK", "no leak_probe span shows"),
    ],
)
def test_trace_problem_requires_the_marker_and_no_token(
    result: str, expected: str | None
) -> None:
    payload = _span_payload("svc", "execute_tool leak_probe", 200, result)
    problem = live.trace_problem(live.probe_spans(payload, "svc", 100))

    assert (problem is None) if expected is None else (expected in (problem or ""))


def test_trace_problem_fails_when_no_span_arrived() -> None:
    assert "no leak_probe span" in (live.trace_problem([]) or "")
