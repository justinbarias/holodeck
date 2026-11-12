"""Graphiti ingest against a real Neo4j with stub model clients (spec 043, T10).

No model call and no API key: the LLM client and the embedder are stubs. The
backend, its counting client, the Graphiti object and the Neo4j driver are the
real ones. The tests skip when Neo4j is not reachable (see ``conftest.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from holodeck.lib.memory.backends.graphiti import GraphitiBackend
from holodeck.lib.memory.chunk_source import ChunkMatch, ChunkRef, SourceDocument
from holodeck.lib.memory.service import TemporalMemory, episode_id_for
from holodeck.lib.memory.types import EpisodeIn
from tests.integration.memory.conftest import NamespaceFactory, Neo4jSettings
from tests.integration.memory.stub_clients import StubLLMClient

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.graphiti]

INSTRUCTIONS = "Extract amounts and the agency that administers them."
REFERENCE_TIME = datetime(2022, 7, 1, tzinfo=UTC)


def graph_size(settings: Neo4jSettings, namespace: str) -> tuple[int, int]:
    """Nodes and relationships of one namespace, read outside the backend."""
    from neo4j import GraphDatabase

    with GraphDatabase.driver(
        settings.uri, auth=(settings.user, settings.password)
    ) as driver:
        nodes = driver.execute_query(
            "MATCH (n) WHERE n.group_id = $g RETURN count(n) AS c", g=namespace
        ).records[0]["c"]
        edges = driver.execute_query(
            "MATCH ()-[r]->() WHERE r.group_id = $g RETURN count(r) AS c", g=namespace
        ).records[0]["c"]
    return int(nodes), int(edges)


def make_episode(
    namespace: str, name: str, *, context: Sequence[UUID] = ()
) -> EpisodeIn:
    """An episode whose id is derived the way the service derives it."""
    text = f"Rate {name} is administered by Agency {name}."
    return EpisodeIn(
        episode_id=episode_id_for(namespace, f"section-{name}", text),
        source_tool="docs",
        chunk_id=f"{name}-c0",
        source_uri=f"corpus/{name}.md",
        text=text,
        reference_time=REFERENCE_TIME,
        reference_time_source="front_matter",
        context_episode_ids=tuple(context),
    )


async def test_ing_01_ingest_writes_complete_episodes(
    backend: GraphitiBackend, stub_llm: StubLLMClient, new_namespace: NamespaceFactory
) -> None:
    namespace = new_namespace()
    first = make_episode(namespace, "one")
    second = make_episode(namespace, "two", context=[first.episode_id])
    assert first.episode_id != second.episode_id

    report = await backend.ingest(namespace, [first, second], INSTRUCTIONS)

    assert report.failed is None, report.error
    assert report.written == (first.episode_id, second.episode_id)
    assert stub_llm.calls > 0
    assert report.llm_calls == stub_llm.calls
    assert await backend.has_episode(namespace, first.episode_id) is True
    assert await backend.has_episode(namespace, second.episode_id) is True
    assert await backend.episode_count(namespace) == 2


async def test_ing_02_reingest_makes_no_llm_call(
    backend: GraphitiBackend, stub_llm: StubLLMClient, new_namespace: NamespaceFactory
) -> None:
    namespace = new_namespace()
    episodes = [make_episode(namespace, "one"), make_episode(namespace, "two")]
    first_run = await backend.ingest(namespace, episodes, INSTRUCTIONS)
    assert len(first_run.written) == 2
    calls_after_first_run = stub_llm.calls
    assert calls_after_first_run > 0

    second_run = await backend.ingest(namespace, episodes, INSTRUCTIONS)

    assert second_run.written == ()
    assert second_run.failed is None
    assert second_run.llm_calls == 0
    assert stub_llm.calls == calls_after_first_run
    assert await backend.episode_count(namespace) == 2


async def test_ing_09_a_pending_episode_is_processed_again(
    backend: GraphitiBackend, stub_llm: StubLLMClient, new_namespace: NamespaceFactory
) -> None:
    namespace = new_namespace()
    first = make_episode(namespace, "one")
    second = make_episode(namespace, "two")

    stub_llm.fail_with = RuntimeError("stub model refused the request")
    failed_run = await backend.ingest(namespace, [first, second], INSTRUCTIONS)

    assert failed_run.failed == first.episode_id
    assert failed_run.written == ()
    assert failed_run.error is not None
    assert "stub model refused the request" in failed_run.error
    assert stub_llm.calls > 0
    assert await backend.has_episode(namespace, first.episode_id) is False
    assert await backend.has_episode(namespace, second.episode_id) is False
    assert await backend.episode_count(namespace) == 0

    stub_llm.fail_with = None
    retry = await backend.ingest(namespace, [first, second], INSTRUCTIONS)

    assert retry.failed is None, retry.error
    assert retry.written == (first.episode_id, second.episode_id)
    assert await backend.has_episode(namespace, first.episode_id) is True
    assert await backend.episode_count(namespace) == 2


async def test_mb_04_purge_clears_one_namespace_only(
    backend: GraphitiBackend,
    new_namespace: NamespaceFactory,
    neo4j_settings: Neo4jSettings,
) -> None:
    kept = new_namespace()
    purged = new_namespace()
    kept_episodes = [make_episode(kept, "one"), make_episode(kept, "two")]
    purged_episodes = [
        make_episode(purged, "one"),
        make_episode(purged, "two"),
        make_episode(purged, "three"),
    ]
    assert len((await backend.ingest(kept, kept_episodes, INSTRUCTIONS)).written) == 2
    assert (
        len((await backend.ingest(purged, purged_episodes, INSTRUCTIONS)).written) == 3
    )
    assert await backend.episode_count(kept) == 2
    assert await backend.episode_count(purged) == 3
    # Entities and facts too, not only episode nodes.
    kept_before = graph_size(neo4j_settings, kept)
    assert kept_before[0] > 2 and kept_before[1] > 0
    assert graph_size(neo4j_settings, purged)[0] > 3

    await backend.purge(purged)

    assert graph_size(neo4j_settings, purged) == (0, 0)
    assert graph_size(neo4j_settings, kept) == kept_before

    assert await backend.episode_count(purged) == 0
    assert await backend.has_episode(purged, purged_episodes[0].episode_id) is False
    assert await backend.episode_count(kept) == 2
    assert await backend.has_episode(kept, kept_episodes[0].episode_id) is True
    assert await backend.has_episode(kept, kept_episodes[1].episode_id) is True


async def test_mb_02_has_episode_is_scoped_to_the_namespace(
    backend: GraphitiBackend, new_namespace: NamespaceFactory
) -> None:
    namespace_a = new_namespace()
    namespace_b = new_namespace()
    episode = make_episode(namespace_a, "one")
    report = await backend.ingest(namespace_a, [episode], INSTRUCTIONS)
    assert report.written == (episode.episode_id,)

    assert await backend.has_episode(namespace_a, episode.episode_id) is True
    assert await backend.has_episode(namespace_b, episode.episode_id) is False
    assert await backend.episode_count(namespace_a) == 1
    assert await backend.episode_count(namespace_b) == 0


class FileSource:
    """A ``ChunkSource`` over one real file and chunks held in memory."""

    def __init__(self, name: str, path: Path, texts: Sequence[str]) -> None:
        self.name = name
        self._document = SourceDocument(
            source_tool=name,
            source_uri=f"{name}/{path.name}",
            path=path,
            mtime_reliable=False,
        )
        self._chunks = [
            ChunkRef(
                source_tool=name,
                chunk_id=f"{name}-{ordinal}",
                source_uri=self._document.source_uri,
                section_id=f"{name}-section-{ordinal}",
                ordinal=ordinal,
                text=text,
            )
            for ordinal, text in enumerate(texts)
        ]

    async def documents(self) -> Sequence[SourceDocument]:
        return [self._document]

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        assert document == self._document
        return list(self._chunks)

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        raise AssertionError("ingestion does not search")


def write_dated(path: Path, effective_date: str) -> Path:
    path.write_text(
        f"---\neffective_date: {effective_date}\n---\nPayment rates.\n",
        encoding="utf-8",
    )
    return path


async def test_ing_04_service_run_through_the_real_backend(
    backend: GraphitiBackend,
    stub_llm: StubLLMClient,
    new_namespace: NamespaceFactory,
    tmp_path: Path,
) -> None:
    namespace = new_namespace()
    current = FileSource(
        "rates-2022",
        write_dated(tmp_path / "rates-2022.md", "2022-07-01"),
        ["Rate A is 500 dollars a fortnight.", "Rate B is 300 dollars a fortnight."],
    )
    older = FileSource(
        "rates-2019",
        write_dated(tmp_path / "rates-2019.md", "2019-03-01"),
        [
            "Rate A was 450 dollars a fortnight.",
            "Rate B was 280 dollars a fortnight.",
            "Rate C was 90 dollars a fortnight.",
        ],
    )
    service = TemporalMemory(
        namespace=namespace, backend=backend, sources=[current, older]
    )
    chunk_count = 5

    first = await service.ingest()

    assert first.failed is None, first.error
    assert first.documents == 2
    assert first.episodes_new == chunk_count
    assert first.episodes_skipped == 0
    assert len(first.written) == chunk_count
    assert first.llm_calls == stub_llm.calls > 0
    assert await backend.episode_count(namespace) == chunk_count
    # The 2019 document is older, so its chunks are submitted first.
    older_ids = [
        episode_id_for(namespace, f"rates-2019-section-{ordinal}", text)
        for ordinal, text in enumerate(
            [
                "Rate A was 450 dollars a fortnight.",
                "Rate B was 280 dollars a fortnight.",
                "Rate C was 90 dollars a fortnight.",
            ]
        )
    ]
    assert list(first.written[:3]) == older_ids

    calls_after_first_run = stub_llm.calls
    second = await service.ingest()

    assert second.failed is None
    assert second.episodes_new == 0
    assert second.episodes_skipped == chunk_count
    assert second.written == ()
    assert second.llm_calls == 0
    assert stub_llm.calls == calls_after_first_run
    assert await backend.episode_count(namespace) == chunk_count
