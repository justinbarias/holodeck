"""Tests for ``TemporalMemoryTool`` and the memory handler layer (spec 043, T13).

The source tools run for real on the in-memory store with keyword search; the
backend is ``FakeBackend``. No network, database or model call.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import pytest_asyncio

from holodeck.lib import tool_initializer
from holodeck.lib.backends.base import BackendSessionError
from holodeck.lib.memory.backends.fake import ALL_OPERATIONS, FakeBackend
from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    TemporalMemoryError,
)
from holodeck.lib.memory.instants import format_instant, parse_instant
from holodeck.lib.memory.service import TemporalMemory
from holodeck.lib.memory.tool_api import (
    MEMORY_FUNCTION_NAMES,
    MEMORY_PROVENANCE,
    MEMORY_SEARCH,
    MEMORY_TIMELINE,
)
from holodeck.lib.memory.types import EpisodeIn, MemoryOperation
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.tools import temporal_memory as composition
from holodeck.tools.hierarchical_chunk_source import HierarchicalChunkSource
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.temporal_memory_tool import TemporalMemoryTool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

OPENAI = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o")
SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas" / "memory"
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
TIMESTAMP_KEYS = frozenset(
    {
        "valid_at",
        "known_at",
        "valid_from",
        "valid_to",
        "recorded_at",
        "expired_at",
        "reference_time",
    }
)
# Several sections per document: keyword search (BM25) scores a term that
# appears in half of a two-chunk corpus at 0 and returns nothing.
DOCS = {
    "alpha": (
        "---\neffective_date: 2020-01-01\n---\n# Rates\n\n"
        "## Pension\n\nThe pension rate is 100 dollars per fortnight.\n\n"
        "## Carer\n\nThe carer payment is paid weekly to a carer.\n\n"
        "## Youth\n\nYouth support is paid to students under age limits.\n"
    ),
    "beta": (
        "---\neffective_date: 2022-07-01\n---\n# Allowance\n\n"
        "## Rent\n\nThe rent allowance is 50 dollars per fortnight.\n\n"
        "## Travel\n\nTravel concessions apply to public transport.\n\n"
        "## Phone\n\nA phone allowance is paid each quarter.\n"
    ),
}


def _agent(**memory_fields: Any) -> Agent:
    tools: list[dict[str, Any]] = [
        {
            "name": name,
            "type": "hierarchical_document",
            "description": name,
            "source": f"{name}/",
            "search_mode": "keyword",
        }
        for name in DOCS
    ]
    tools.append(
        {
            "name": "law_memory",
            "type": "temporal_memory",
            "description": "Family law memory.",
            "source_tools": list(DOCS),
            "backend": {"driver": "kuzu"},
            **memory_fields,
        }
    )
    return Agent(name="a", model=OPENAI, instructions={"inline": "hi"}, tools=tools)


def _write_docs(root: Path) -> None:
    for name, text in DOCS.items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        (folder / "doc.md").write_text(text)


async def _doc_tool(root: Path, name: str) -> HierarchicalDocumentTool:
    tool = HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name=name,
            description=name,
            source=str(root / name),
            search_mode=SearchMode.KEYWORD,
        )
    )
    await tool.initialize()
    return tool


class _Rig:
    """Real source tools, a fake backend, and counters."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.backend = FakeBackend()
        self.tools: dict[str, HierarchicalDocumentTool] = {}
        self.ingest_runs = 0

    def memory_tool(self, **memory_fields: Any) -> TemporalMemoryTool:
        agent = _agent(**memory_fields)
        config = agent.tools[-1]  # type: ignore[index]
        return TemporalMemoryTool(agent, config, self.tools)  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    _write_docs(tmp_path)
    made = _Rig(tmp_path)
    made.tools = {name: await _doc_tool(tmp_path, name) for name in DOCS}

    real_ingest = TemporalMemory.ingest

    async def counting_ingest(self: TemporalMemory, **kwargs: Any) -> Any:
        made.ingest_runs += 1
        return await real_ingest(self, **kwargs)

    monkeypatch.setattr(TemporalMemory, "ingest", counting_ingest)
    monkeypatch.setattr(composition, "create_backend", lambda *args: made.backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    return made


@pytest_asyncio.fixture
async def started(rig: _Rig) -> Any:
    tool = rig.memory_tool(ingest_on_start=True)
    await tool.initialize()
    yield tool
    await tool.close()


def _schema(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SCHEMA_DIR / name).read_text())
    return loaded


def _timestamps(value: object) -> Iterator[str]:
    """Every non-null timestamp string in a result, at any depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TIMESTAMP_KEYS and item is not None:
                assert isinstance(item, str)
                yield item
            else:
                yield from _timestamps(item)
    elif isinstance(value, list):
        for item in value:
            yield from _timestamps(item)


def _assert_iso_utc(result: dict[str, Any]) -> None:
    stamps = list(_timestamps(result))
    assert stamps, "the result carries no timestamp to check"
    for stamp in stamps:
        assert ISO_UTC.match(stamp), stamp
        # The one formatter, round-tripped: no second format exists.
        assert format_instant(parse_instant(stamp)) == stamp


async def _search(tool: TemporalMemoryTool, **args: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(await tool.call(MEMORY_SEARCH, args))
    return result


async def _first_fact(tool: TemporalMemoryTool) -> dict[str, Any]:
    result = await _search(tool, query="pension rate")
    facts = [hit for hit in result["hits"] if hit["kind"] == "fact"]
    assert facts, result
    return facts[0]


# --- Lifecycle ---------------------------------------------------------------


async def _chunk_count(rig: _Rig) -> int:
    total = 0
    for tool in rig.tools.values():
        source = HierarchicalChunkSource(tool)
        for document in await source.documents():
            total += len(await source.chunks(document))
    return total


async def test_ing_07_ingest_on_start_runs_ingestion_once(rig: _Rig) -> None:
    chunks = await _chunk_count(rig)
    tool = rig.memory_tool(ingest_on_start=True)
    await tool.initialize()

    assert chunks > 0
    assert rig.ingest_runs == 1
    assert tool.ingest_report is not None
    assert len(tool.ingest_report.written) == chunks
    assert rig.backend.calls["ingest"] == chunks
    assert tool.is_initialized
    await tool.close()


async def test_ing_07_without_ingest_on_start_nothing_is_ingested(rig: _Rig) -> None:
    tool = rig.memory_tool(ingest_on_start=False)
    await tool.initialize()

    assert rig.ingest_runs == 0
    assert rig.backend.calls["ingest"] == 0
    assert tool.ingest_report is None
    assert tool.is_initialized
    await tool.close()


async def test_ing_07_unreachable_store_with_ingest_on_start_fails_start(
    rig: _Rig,
) -> None:
    rig.backend.available = False
    tool = rig.memory_tool(ingest_on_start=True)

    with pytest.raises(MemoryBackendUnavailableError):
        await tool.initialize()

    assert not tool.is_initialized
    assert rig.ingest_runs == 0
    assert rig.backend.calls["close"] == 1


async def test_ing_07_failed_episode_on_start_fails_start(rig: _Rig) -> None:
    def refuse(episode: EpisodeIn) -> Sequence[tuple[str, str]]:
        raise RuntimeError("secret document text")

    rig.backend = FakeBackend(extract=refuse)
    tool = rig.memory_tool(ingest_on_start=True)

    with pytest.raises(TemporalMemoryError, match="ingest_on_start stopped") as info:
        await tool.initialize()

    assert "secret document text" not in str(info.value)
    assert not tool.is_initialized
    assert rig.backend.calls["close"] == 1


async def test_ret_07_unreachable_store_starts_degraded(rig: _Rig) -> None:
    rig.backend.available = False
    tool = rig.memory_tool(ingest_on_start=False)
    await tool.initialize()

    assert tool.is_initialized
    assert tool.degraded_at_start
    result = await _search(tool, query="pension rate")
    assert result["degraded"] is True
    assert result["hits"], result
    assert {hit["kind"] for hit in result["hits"]} == {"chunk"}
    await tool.close()


async def test_reachable_store_does_not_start_degraded(rig: _Rig) -> None:
    tool = rig.memory_tool()
    await tool.initialize()

    assert not tool.degraded_at_start
    await tool.close()


async def test_close_closes_the_backend_not_the_sources(rig: _Rig) -> None:
    tool = rig.memory_tool()
    await tool.initialize()
    await tool.close()
    await tool.close()

    assert rig.backend.calls["close"] == 1
    assert all(source._initialized for source in rig.tools.values())
    assert not tool.is_initialized


async def test_call_before_initialize_raises_backend_session_error(
    rig: _Rig,
) -> None:
    tool = rig.memory_tool()

    with pytest.raises(BackendSessionError, match="law_memory"):
        await tool.call(MEMORY_SEARCH, {"query": "rate"})


# --- Success results: schema and timestamps (HST-01, HST-02) -----------------


async def test_hst_01_search_result_matches_schema(
    started: TemporalMemoryTool,
) -> None:
    result = await _search(started, query="pension rate")

    assert result["hits"], result
    assert {hit["kind"] for hit in result["hits"]} == {"fact", "chunk"}
    jsonschema.validate(result, _schema("memory_search.result.schema.json"))


async def test_hst_01_timeline_result_matches_schema(
    started: TemporalMemoryTool,
) -> None:
    fact = await _first_fact(started)
    result = json.loads(await started.call(MEMORY_TIMELINE, {"entity": fact["text"]}))

    assert result["entries"], result
    assert result["entity"] == fact["text"]
    jsonschema.validate(result, _schema("memory_timeline.result.schema.json"))


async def test_hst_01_provenance_result_matches_schema(
    started: TemporalMemoryTool,
) -> None:
    fact = await _first_fact(started)
    result = json.loads(
        await started.call(MEMORY_PROVENANCE, {"fact_id": fact["fact_id"]})
    )

    assert result["found"] is True
    assert result["sources"], result
    assert all(source["text"] for source in result["sources"])
    jsonschema.validate(result, _schema("memory_provenance.result.schema.json"))


async def test_hst_01_unknown_fact_id_is_found_false(
    started: TemporalMemoryTool,
) -> None:
    result = json.loads(
        await started.call(MEMORY_PROVENANCE, {"fact_id": "no-such-fact"})
    )

    assert result == {"found": False, "fact": None, "sources": []}
    jsonschema.validate(result, _schema("memory_provenance.result.schema.json"))


async def test_hst_02_every_timestamp_is_iso_8601_utc(
    started: TemporalMemoryTool,
) -> None:
    fact = await _first_fact(started)
    results = [
        await _search(started, query="pension rate"),
        json.loads(await started.call(MEMORY_TIMELINE, {"entity": fact["text"]})),
        json.loads(await started.call(MEMORY_PROVENANCE, {"fact_id": fact["fact_id"]})),
    ]

    for result in results:
        _assert_iso_utc(result)


@pytest.mark.parametrize(
    ("valid_at", "expected"),
    [
        ("2022-07-01", "2022-07-01T00:00:00Z"),
        ("2022-07-01T09:30:00+10:00", "2022-06-30T23:30:00Z"),
        ("2022-07-01T09:30:00", "2022-07-01T09:30:00Z"),
    ],
)
async def test_hst_02_date_arguments_are_echoed_as_utc(
    started: TemporalMemoryTool, valid_at: str, expected: str
) -> None:
    result = await _search(started, query="pension rate", valid_at=valid_at)

    assert result["valid_at"] == expected
    assert result["hits"], result
    jsonschema.validate(result, _schema("memory_search.result.schema.json"))


# --- Error results -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "field"),
    [
        (MEMORY_SEARCH, {"query": "rate", "valid_at": "last July"}, "valid_at"),
        (MEMORY_SEARCH, {"query": "rate", "known_at": "soon"}, "known_at"),
        (MEMORY_SEARCH, {"query": "rate", "hops": 3}, "hops"),
        (MEMORY_SEARCH, {"query": "rate", "hops": "2"}, "hops"),
        (MEMORY_SEARCH, {"query": "   "}, "query"),
        (MEMORY_SEARCH, {}, "query"),
        (MEMORY_SEARCH, {"query": "rate", "top_k": 5}, "unknown field"),
        (MEMORY_TIMELINE, {"entity": ""}, "entity"),
        (MEMORY_PROVENANCE, {}, "fact_id"),
    ],
)
async def test_invalid_argument_is_an_error_result_with_no_backend_call(
    rig: _Rig, name: str, arguments: dict[str, Any], field: str
) -> None:
    tool = rig.memory_tool()
    await tool.initialize()
    before = dict(rig.backend.calls)

    result = json.loads(await tool.call(name, arguments))

    assert result["error"]["code"] == "invalid_argument"
    assert field in result["error"]["message"]
    assert dict(rig.backend.calls) == before
    jsonschema.validate(result, _schema("error.schema.json"))
    await tool.close()


URL_WITH_KEY = "https://secret-host/?api_key=sk-example"


@pytest.mark.parametrize(
    ("arguments", "message", "absent"),
    [
        (
            {"query": "rate", "valid_at": "last July"},
            "valid_at must be an ISO 8601 date or datetime, for example "
            "2022-07-01 or 2022-07-01T09:30:00Z",
            "last July",
        ),
        (
            {"query": "rate", "valid_at": URL_WITH_KEY},
            "valid_at must be an ISO 8601 date or datetime, for example "
            "2022-07-01 or 2022-07-01T09:30:00Z",
            "sk-example",
        ),
        (
            {"query": "rate", "valid_at": "x" * 5000},
            "valid_at must be an ISO 8601 date or datetime, for example "
            "2022-07-01 or 2022-07-01T09:30:00Z",
            "xxxxxxxx",
        ),
        (
            {"query": "rate", "sk-secret-key": "v"},
            "an unknown field",
            "sk-secret-key",
        ),
        ({"query": "rate", "hops": 3}, "hops must be 1 or 2", None),
        ({"query": ""}, "query must be a non-empty string", None),
    ],
)
async def test_an_invalid_argument_message_never_echoes_the_input(
    rig: _Rig, arguments: dict[str, Any], message: str, absent: str | None
) -> None:
    tool = rig.memory_tool()
    await tool.initialize()

    result = json.loads(await tool.call(MEMORY_SEARCH, arguments))

    assert result == {"error": {"code": "invalid_argument", "message": message}}
    if absent is not None:
        assert absent not in result["error"]["message"]
    await tool.close()


async def test_bad_valid_at_message_does_not_name_the_value(rig: _Rig) -> None:
    tool = rig.memory_tool()
    await tool.initialize()

    result = json.loads(
        await tool.call(MEMORY_SEARCH, {"query": "rate", "valid_at": "last July"})
    )

    assert "last July" not in result["error"]["message"]
    await tool.close()


@pytest.mark.parametrize("name", [MEMORY_TIMELINE, MEMORY_PROVENANCE])
async def test_backend_down_is_backend_unavailable_for_timeline_and_provenance(
    started: TemporalMemoryTool, rig: _Rig, name: str
) -> None:
    rig.backend.available = False
    arguments = {"entity": "x"} if name == MEMORY_TIMELINE else {"fact_id": "x"}

    result = json.loads(await started.call(name, arguments))

    assert result["error"]["code"] == "backend_unavailable"
    # The backend's detail can name a host; it does not reach the model.
    assert "marked unavailable" not in result["error"]["message"]
    jsonschema.validate(result, _schema("error.schema.json"))


async def test_ret_07_backend_down_search_returns_degraded_chunk_hits(
    started: TemporalMemoryTool, rig: _Rig
) -> None:
    rig.backend.available = False

    result = await _search(started, query="pension rate")

    assert "error" not in result
    assert result["degraded"] is True
    assert result["hits"], result
    assert {hit["kind"] for hit in result["hits"]} == {"chunk"}
    assert all(hit["fact_id"] is None for hit in result["hits"])
    jsonschema.validate(result, _schema("memory_search.result.schema.json"))


@pytest.mark.parametrize(
    ("missing", "name", "arguments", "operation"),
    [
        (MemoryOperation.TIMELINE, MEMORY_TIMELINE, {"entity": "x"}, "timelines"),
        (MemoryOperation.PROVENANCE, MEMORY_PROVENANCE, {"fact_id": "x"}, "provenance"),
        (
            MemoryOperation.VALID_AT,
            MEMORY_SEARCH,
            {"query": "rate", "valid_at": "2022-07-01"},
            "valid-time",
        ),
    ],
)
async def test_unsupported_operation_is_not_supported(
    rig: _Rig,
    missing: MemoryOperation,
    name: str,
    arguments: dict[str, Any],
    operation: str,
) -> None:
    rig.backend = FakeBackend(supported=ALL_OPERATIONS - {missing})
    tool = rig.memory_tool()
    await tool.initialize()

    result = json.loads(await tool.call(name, arguments))

    assert result["error"]["code"] == "not_supported"
    assert operation in result["error"]["message"]
    jsonschema.validate(result, _schema("error.schema.json"))
    await tool.close()


async def test_unknown_function_name_raises(started: TemporalMemoryTool) -> None:
    with pytest.raises(KeyError, match="memory_forget"):
        await started.call("memory_forget", {})


async def test_result_json_is_compact_model_dump(
    started: TemporalMemoryTool,
) -> None:
    text = await started.call(MEMORY_PROVENANCE, {"fact_id": "no-such-fact"})

    assert text == '{"found":false,"fact":null,"sources":[]}'


# --- Tool initializer --------------------------------------------------------


@pytest_asyncio.fixture
async def initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeBackend, list[str]]:
    """Route ``initialize_tools`` to keyword tools and a fake backend."""
    _write_docs(tmp_path)
    backend = FakeBackend()
    closed: list[str] = []
    real_close = HierarchicalDocumentTool.close

    async def recording_close(self: HierarchicalDocumentTool) -> None:
        closed.append(self.config.name)
        await real_close(self)

    monkeypatch.setattr(HierarchicalDocumentTool, "close", recording_close)
    # No embedding service: keyword tools then make no embedding call.
    monkeypatch.setattr(
        tool_initializer, "create_embedding_service", lambda agent, **kw: None
    )
    monkeypatch.setattr(composition, "create_backend", lambda *args: backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: None)
    return tmp_path, backend, closed


async def test_initializer_creates_memory_tool_after_its_sources(
    initializer: tuple[Path, FakeBackend, list[str]],
) -> None:
    root, backend, closed = initializer

    instances = await tool_initializer.initialize_tools(
        _agent(ingest_on_start=True), base_dir=str(root)
    )

    assert list(instances) == ["alpha", "beta", "law_memory"]
    memory = instances["law_memory"]
    assert isinstance(memory, TemporalMemoryTool)
    assert memory.is_initialized
    assert memory.ingest_report is not None
    assert memory.ingest_report.written
    assert backend.calls["ingest"] == len(memory.ingest_report.written)
    await tool_initializer.close_tools(instances.values())
    assert backend.calls["close"] == 1
    assert sorted(closed) == ["alpha", "beta"]


async def test_initializer_can_skip_the_memory_tool(
    initializer: tuple[Path, FakeBackend, list[str]],
) -> None:
    root, backend, _ = initializer

    instances = await tool_initializer.initialize_tools(
        _agent(ingest_on_start=True), base_dir=str(root), initialize_memory=False
    )

    assert list(instances) == ["alpha", "beta"]
    assert sum(backend.calls.values()) == 0
    await tool_initializer.close_tools(instances.values())


async def test_ing_07_failed_memory_start_closes_the_source_tools(
    initializer: tuple[Path, FakeBackend, list[str]],
) -> None:
    root, backend, closed = initializer
    backend.available = False

    with pytest.raises(MemoryBackendUnavailableError):
        await tool_initializer.initialize_tools(
            _agent(ingest_on_start=True), base_dir=str(root)
        )

    assert sorted(closed) == ["alpha", "beta"]
    assert backend.calls["close"] == 1


def test_memory_function_names_are_the_three_fixed_names() -> None:
    assert MEMORY_FUNCTION_NAMES == (
        "memory_search",
        "memory_timeline",
        "memory_provenance",
    )


def test_agent_without_memory_tool_does_not_import_the_memory_stack(
    tmp_path: Path,
) -> None:
    # A fresh interpreter: other tests in this worker import the memory stack.
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "doc.md").write_text("# Rates\n\nThe rate is 100 dollars.\n")
    watched = (
        "graphiti_core",
        "holodeck.lib.memory.backends.graphiti",
        "holodeck.lib.memory.service",
        "holodeck.lib.memory.tool_api",
        "holodeck.tools.temporal_memory",
        "holodeck.tools.temporal_memory_tool",
    )
    code = f"""
import asyncio, json, sys
from holodeck.lib import tool_initializer
from holodeck.lib.backends.tool_adapters import (
    build_holodeck_sdk_server, create_tool_adapters,
)
from holodeck.models.agent import Agent

tool_initializer.create_embedding_service = lambda agent, **kw: None
agent = Agent(
    name="a",
    model={{"provider": "openai", "name": "gpt-4o"}},
    instructions={{"inline": "hi"}},
    tools=[{{
        "name": "docs", "type": "hierarchical_document", "description": "d",
        "source": {str(folder)!r}, "search_mode": "keyword",
    }}],
)

async def main():
    instances = await tool_initializer.initialize_tools(agent)
    adapters = create_tool_adapters(agent.tools, instances)
    _, names = build_holodeck_sdk_server(adapters)
    await tool_initializer.close_tools(instances.values())
    return names

names = asyncio.run(main())
print(json.dumps({{
    "names": names,
    "loaded": [m for m in {watched!r} if m in sys.modules],
}}))
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["names"] == ["mcp__holodeck_tools__docs_search"]
    assert report["loaded"] == []
