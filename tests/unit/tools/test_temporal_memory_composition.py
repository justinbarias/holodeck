"""Tests for ``open_memory``, the composition of a temporal memory (spec 043).

The source tools run for real on the in-memory store with keyword search; the
backend is ``FakeBackend``. No network, database or model call.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.errors import MemoryConfigError
from holodeck.lib.memory.service import TemporalMemory
from holodeck.lib.source_resolver import ResolvedSource
from holodeck.models.agent import Agent
from holodeck.models.claude_config import AuthProvider
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.tools import temporal_memory as composition
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.temporal_memory import (
    AGENT_SDK_SECONDS_PER_EPISODE,
    open_memory,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

OPENAI = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o")


def _anthropic(auth: AuthProvider | None) -> LLMProvider:
    return LLMProvider(
        provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-5", auth_provider=auth
    )


def _agent(
    model: LLMProvider = OPENAI,
    *,
    extraction_model: LLMProvider | None = None,
    source_tools: Sequence[str] = ("beta", "alpha"),
    memory: bool = True,
    **memory_fields: Any,
) -> Agent:
    tools: list[dict[str, Any]] = [
        {
            "name": name,
            "type": "hierarchical_document",
            "description": name,
            "source": f"{name}/",
        }
        for name in ("alpha", "beta")
    ]
    if memory:
        body: dict[str, Any] = {
            "name": "law_memory",
            "type": "temporal_memory",
            "description": "memory",
            "source_tools": list(source_tools),
            "backend": {"driver": "kuzu"},
            **memory_fields,
        }
        if extraction_model is not None:
            body["extraction_model"] = extraction_model.model_dump(mode="json")
        tools.append(body)
    return Agent(name="a", model=model, instructions={"inline": "hi"}, tools=tools)


async def _doc_tool(root: Path, name: str) -> HierarchicalDocumentTool:
    folder = root / name
    folder.mkdir()
    (folder / "doc.md").write_text(f"# {name}\n\nText of {name}.\n")
    tool = HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name=name,
            description=name,
            source=str(folder),
            search_mode=SearchMode.KEYWORD,
        )
    )
    await tool.initialize()
    return tool


class _Wiring:
    def __init__(self) -> None:
        self.backend = FakeBackend()
        self.tools: dict[str, HierarchicalDocumentTool] = {}
        self.service_kwargs: dict[str, Any] = {}
        self.backend_args: tuple[Any, ...] = ()
        self.initialize_kwargs: dict[str, Any] = {}


@pytest_asyncio.fixture
async def wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Wiring:
    wired = _Wiring()
    wired.tools = {
        "alpha": await _doc_tool(tmp_path, "alpha"),
        "beta": await _doc_tool(tmp_path, "beta"),
    }
    for tool in wired.tools.values():
        tool.close = AsyncMock()  # type: ignore[method-assign]

    async def fake_initialize(agent: Agent, **kwargs: Any) -> dict[str, Any]:
        wired.initialize_kwargs = kwargs
        return dict(wired.tools)

    def fake_create_backend(*args: Any) -> FakeBackend:
        wired.backend_args = args
        return wired.backend

    real_service = composition.TemporalMemory

    def spy_service(**kwargs: Any) -> Any:
        wired.service_kwargs = kwargs
        return real_service(**kwargs)

    monkeypatch.setattr(composition, "initialize_tools", fake_initialize)
    monkeypatch.setattr(composition, "create_backend", fake_create_backend)
    monkeypatch.setattr(composition, "create_embedding_service", lambda agent: "emb")
    monkeypatch.setattr(composition, "TemporalMemory", spy_service)
    return wired


async def test_sources_are_the_named_tools_in_source_tools_order(
    wiring: _Wiring,
) -> None:
    async with open_memory(_agent(source_tools=("beta", "alpha"))) as session:
        names = [source.name for source in wiring.service_kwargs["sources"]]
        assert session.backend is wiring.backend

    assert names == ["beta", "alpha"]


async def test_only_named_source_tools_are_wrapped(wiring: _Wiring) -> None:
    async with open_memory(_agent(source_tools=("alpha",))):
        names = [source.name for source in wiring.service_kwargs["sources"]]

    assert names == ["alpha"]


async def test_namespace_is_resolved_namespace(wiring: _Wiring) -> None:
    async with open_memory(_agent()) as default:
        assert default.config.resolved_namespace == "law_memory"
        assert wiring.service_kwargs["namespace"] == "law_memory"

    async with open_memory(_agent(namespace="family-law")):
        assert wiring.service_kwargs["namespace"] == "family-law"


async def test_service_settings_come_from_the_tool_config(wiring: _Wiring) -> None:
    agent = _agent(
        effective_date_key="valid_from",
        extraction_instructions="Extract rates.",
        schema_hints={"entity_types": ["Payment", "Person"]},
        top_k=7,
        hops=2,
        fusion_weights={"chunks": 0.25, "graph": 2.0},
    )
    async with open_memory(agent):
        pass

    assert wiring.service_kwargs["effective_date_key"] == "valid_from"
    assert wiring.service_kwargs["top_k"] == 7
    assert wiring.service_kwargs["hops"] == 2
    assert wiring.service_kwargs["weight_chunks"] == 0.25
    assert wiring.service_kwargs["weight_graph"] == 2.0
    assert wiring.service_kwargs["instructions"] == "Extract rates."
    config, model, embedding, entity_types = wiring.backend_args
    assert config.driver == "kuzu"
    assert model is agent.model
    assert embedding == "emb"
    assert list(entity_types) == ["Payment", "Person"]


async def test_extraction_model_overrides_the_agent_model(wiring: _Wiring) -> None:
    extraction = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini")
    async with open_memory(_agent(extraction_model=extraction)):
        pass

    assert wiring.backend_args[1].name == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_anthropic(AuthProvider.oauth_token), AGENT_SDK_SECONDS_PER_EPISODE),
        (_anthropic(AuthProvider.api_key), None),
        (_anthropic(None), None),
        (OPENAI, None),
    ],
)
async def test_seconds_per_episode_only_for_anthropic_oauth(
    wiring: _Wiring, model: LLMProvider, expected: float | None
) -> None:
    async with open_memory(_agent(model)):
        pass

    assert wiring.service_kwargs["seconds_per_episode"] == expected


async def test_extraction_model_decides_the_rate(wiring: _Wiring) -> None:
    oauth = _anthropic(AuthProvider.oauth_token)
    async with open_memory(_agent(OPENAI, extraction_model=oauth)):
        pass

    assert wiring.service_kwargs["seconds_per_episode"] == 11.0


async def test_backend_and_tools_close_on_normal_exit(wiring: _Wiring) -> None:
    async with open_memory(_agent()):
        assert wiring.backend.calls["close"] == 0

    assert wiring.backend.calls["close"] == 1
    for tool in wiring.tools.values():
        tool.close.assert_awaited_once()  # type: ignore[attr-defined]


async def test_backend_and_tools_close_when_the_body_raises(wiring: _Wiring) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with open_memory(_agent()):
            raise RuntimeError("boom")

    assert wiring.backend.calls["close"] == 1
    for tool in wiring.tools.values():
        tool.close.assert_awaited_once()  # type: ignore[attr-defined]


async def test_tools_close_when_the_backend_cannot_be_created(
    wiring: _Wiring, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any) -> FakeBackend:
        raise MemoryConfigError("memory backend 'x' is not supported")

    monkeypatch.setattr(composition, "create_backend", refuse)

    with pytest.raises(MemoryConfigError, match="not supported"):
        async with open_memory(_agent()):
            pytest.fail("the body must not run")

    for tool in wiring.tools.values():
        tool.close.assert_awaited_once()  # type: ignore[attr-defined]


async def test_agent_without_memory_tool_raises(wiring: _Wiring) -> None:
    with pytest.raises(MemoryConfigError, match="agent has no temporal_memory tool"):
        async with open_memory(_agent(memory=False)):
            pytest.fail("the body must not run")

    assert wiring.initialize_kwargs == {}


async def test_source_initialization_can_be_skipped(wiring: _Wiring) -> None:
    async with open_memory(_agent(), initialize_sources=False):
        pass

    assert wiring.initialize_kwargs == {}
    assert wiring.service_kwargs["sources"] == []
    assert wiring.backend.calls["close"] == 1


async def test_options_reach_initialize_tools(wiring: _Wiring) -> None:
    async with open_memory(_agent(), base_dir="/x", force_ingest_sources=True):
        pass

    assert wiring.initialize_kwargs["base_dir"] == "/x"
    assert wiring.initialize_kwargs["force_ingest"] is True


async def test_ing_08_preview_sources_initialize_no_tool(
    wiring: _Wiring, tmp_path: Path
) -> None:
    async with open_memory(
        _agent(), base_dir=str(tmp_path), preview_sources=True
    ) as session:
        sources = wiring.service_kwargs["sources"]
        report = await session.memory.ingest(dry_run=True)

    assert wiring.initialize_kwargs == {}
    assert [source.name for source in sources] == ["beta", "alpha"]
    assert all(not source._tool._initialized for source in sources)
    assert all(source._tool._adapter is None for source in sources)
    assert all(source._tool._embedding_service is None for source in sources)
    assert (report.documents, report.episodes_new) == (2, 2)
    assert wiring.backend.calls["ingest"] == 0
    assert wiring.backend.calls["close"] == 1


async def test_ing_08_preview_counts_what_a_full_run_writes(
    wiring: _Wiring, tmp_path: Path
) -> None:
    async with open_memory(
        _agent(), base_dir=str(tmp_path), preview_sources=True
    ) as session:
        preview = await session.memory.ingest(dry_run=True)
        preview_sources = wiring.service_kwargs["sources"]
    # The fake backend is emptied when it closes, so the second dry run
    # shares the full run's session.
    async with open_memory(_agent(), base_dir=str(tmp_path)) as session:
        full = await session.memory.ingest()
        again = await TemporalMemory(
            namespace=session.config.resolved_namespace,
            backend=session.backend,
            sources=preview_sources,
        ).ingest(dry_run=True)

    assert len(full.written) == preview.episodes_new == 2
    assert (again.episodes_new, again.episodes_skipped) == (0, 2)


async def test_skipping_sources_wins_over_preview(wiring: _Wiring) -> None:
    async with open_memory(_agent(), initialize_sources=False, preview_sources=True):
        pass

    assert wiring.service_kwargs["sources"] == []


async def test_ing_08_preview_downloads_a_remote_source_and_removes_it(
    wiring: _Wiring, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = tmp_path / "download"
    (download / "docs").mkdir(parents=True)
    (download / "docs" / "doc.md").write_text("# Remote\n\nText of remote.\n")
    resolved: list[str] = []

    async def fake_resolve(source: str, base_dir: str | None) -> ResolvedSource:
        resolved.append(source)
        return ResolvedSource(local_path=download, is_remote=True, temp_dir=download)

    monkeypatch.setattr(composition.SourceResolver, "resolve", fake_resolve)
    agent = _agent(source_tools=("alpha",))
    assert agent.tools is not None
    agent.tools[0].source = "https://example.test/corpus.zip"  # type: ignore[union-attr]

    async with open_memory(agent, preview_sources=True) as session:
        report = await session.memory.ingest(dry_run=True)
        documents = await wiring.service_kwargs["sources"][0].documents()

    assert resolved == ["https://example.test/corpus.zip"]
    assert [document.source_uri for document in documents] == ["docs/doc.md"]
    assert report.episodes_new == 1
    assert not download.exists()
