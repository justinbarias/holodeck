"""Graphiti backend, write side (spec 043, design seam 3, task T7).

Graphiti is replaced by an in-memory fake: no database, no model, no network.
The fake keeps episode nodes the way Graphiti stores them (one ``name``
field), so the tests read the lifecycle state the backend writes.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance import graph_data_operations
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, SecretStr

from holodeck.lib.memory import (
    EpisodeIn,
    IngestReport,
    MemoryBackend,
    MemoryBackendUnavailableError,
    MemoryConfigError,
    TemporalMemoryError,
)
from holodeck.lib.memory.backends import graphiti_clients
from holodeck.lib.memory.backends.graphiti import (
    BACKFILL_FACT_START_QUERY,
    CLOSE_FACTS_QUERY,
    CLOSED_SINCE_QUERY,
    COMPLETE_PREFIX,
    DATE_EPISODE_FACTS_QUERY,
    EPISODE_OPEN_FACTS_QUERY,
    KEEP_EPISODE_EDGES_QUERY,
    OLD_SECTION_EPISODES_QUERY,
    REOPEN_FACTS_QUERY,
    TEXT_CLOSURE_CANDIDATES_QUERY,
    GraphitiBackend,
    decode_episode_name,
    encode_episode_name,
    is_complete,
    source_sentence,
)
from holodeck.lib.memory.backends.graphiti_closure import SectionKey
from holodeck.lib.memory.registry import create_backend
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import (
    GraphBackendConfig,
    MemorySchemaHints,
    ReasoningEffort,
)

pytestmark = [pytest.mark.unit]

NS = "family-assistance"
OTHER_NS = "other-namespace"
PASSWORD = "db-pass-not-real-7781"  # noqa: S105 - test value
API_KEY = "sk-unit-not-real-5512"
WHEN = datetime(2022, 7, 1, tzinfo=UTC)


def episode(name: str, *, context: tuple[UUID, ...] = ()) -> EpisodeIn:
    return EpisodeIn(
        episode_id=uuid5(NAMESPACE_URL, name),
        source_tool="family_law",
        chunk_id=f"{name}_chunk_0",
        source_uri=f"corpus/{name}.md",
        text=f"The rate in {name} is 120 dollars.",
        reference_time=WHEN,
        reference_time_source="front_matter",
        context_episode_ids=context,
    )


E1 = episode("c1")
E2 = episode("c2", context=(E1.episode_id,))
E3 = episode("c3", context=(E1.episode_id, E2.episode_id))


# --- fakes ------------------------------------------------------------------


@dataclass
class FakeLLM:
    calls: int = 0
    token_tracker: TokenUsageTracker = field(default_factory=TokenUsageTracker)


@dataclass
class FakeEdge:
    """The ``RELATES_TO`` fields the dating and closure queries use."""

    uuid: str
    group_id: str
    episodes: list[str]
    valid_at: datetime | None = None
    fact: str = ""
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    source: str = "source-entity"
    target: str = "target-entity"
    name: str = "RELATES_TO"


@dataclass
class FakeStore:
    nodes: dict[str, EpisodicNode] = field(default_factory=dict)
    edges: list[FakeEdge] = field(default_factory=list)
    provider: GraphProvider = GraphProvider.NEO4J
    events: list[tuple[str, str, str]] = field(default_factory=list)
    queries: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail_load: BaseException | None = None
    fail_save_complete: BaseException | None = None
    fail_date: BaseException | None = None
    cleared: list[list[str] | None] = field(default_factory=list)


class FakeDriver:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.clones: list[str] = []

    def clone(self, database: str) -> FakeDriver:
        self.clones.append(database)
        return self

    @property
    def provider(self) -> GraphProvider:
        return self.store.provider

    async def execute_query(self, query: str, **params: Any) -> tuple[Any, ...]:
        self.store.queries.append((query, params))
        if query == DATE_EPISODE_FACTS_QUERY:
            return self._date_facts(params, only=params["uuid"])
        if query == BACKFILL_FACT_START_QUERY:
            return self._date_facts(params, only=None)
        if query == CLOSED_SINCE_QUERY:
            return self._closed_since(params)
        if query == REOPEN_FACTS_QUERY:
            return self._reopen(params)
        if query == EPISODE_OPEN_FACTS_QUERY:
            return self._episode_open_facts(params)
        if query == OLD_SECTION_EPISODES_QUERY:
            return self._old_section_episodes(params)
        if query == TEXT_CLOSURE_CANDIDATES_QUERY:
            return self._text_closure_candidates(params)
        if query == CLOSE_FACTS_QUERY:
            return self._close_facts(params)
        if query == KEEP_EPISODE_EDGES_QUERY:
            return self._keep_edges(params)
        if "uuid" in params:
            return self._mark_complete(params)
        count = sum(
            1
            for node in self.store.nodes.values()
            if node.group_id == params["group_id"]
            and node.name.startswith(params["prefix"])
        )
        return ([{"episode_count": count}], None, None)

    def _date_facts(self, params: dict[str, Any], only: str | None) -> tuple[Any, ...]:
        """What the two dating queries do, in Python.

        An undated edge of the group (only those naming ``only``, when
        given) gets the earliest ``valid_at`` of its episodes in the group;
        the ingest query falls back to ``$reference_time``. An edge whose
        ``invalid_at`` is not after that start stays undated.
        """
        if self.store.fail_date is not None:
            raise self.store.fail_date
        group = params["group_id"]
        dated = 0
        inverted = 0
        for edge in self.store.edges:
            if edge.group_id != group or edge.valid_at is not None:
                continue
            if only is not None and only not in edge.episodes:
                continue
            starts = [
                node.valid_at
                for uuid in edge.episodes
                if (node := self.store.nodes.get(uuid)) is not None
                and node.group_id == group
            ]
            first_seen = min(starts) if starts else params.get("reference_time")
            if first_seen is None:
                continue
            if edge.invalid_at is not None and edge.invalid_at <= first_seen:
                inverted += 1
                continue
            edge.valid_at = first_seen
            dated += 1
        return ([{"dated": dated, "inverted": inverted}], None, None)

    def _keep_edges(self, params: dict[str, Any]) -> tuple[Any, ...]:
        node = self.store.nodes.get(params["uuid"])
        if node is None or node.group_id != params["group_id"]:
            return ([], None, None)
        node.entity_edges = node.entity_edges + [
            u for u in params["edges"] if u not in node.entity_edges
        ]
        return ([{"edges": len(node.entity_edges)}], None, None)

    def _closed_since(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Closed facts of the group listed on the episode node and stamped
        at or after ``$since``."""
        node = self.store.nodes.get(params["uuid"])
        if node is None or node.group_id != params["group_id"]:
            return ([], None, None)
        listed = set(node.entity_edges)
        rows = [
            {
                "fact_uuid": e.uuid,
                "fact": e.fact,
                "episodes": list(e.episodes),
                "name": e.name,
                "source_uuid": e.source,
                "target_uuid": e.target,
            }
            for e in self.store.edges
            if e.group_id == params["group_id"]
            and e.uuid in listed
            and e.invalid_at is not None
            and e.expired_at is not None
            and e.expired_at >= params["since"]
        ]
        return (rows, None, None)

    def _episode_open_facts(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Open facts of the group that name the episode."""
        rows = [
            {
                "fact_uuid": e.uuid,
                "name": e.name,
                "source_uuid": e.source,
                "target_uuid": e.target,
            }
            for e in self.store.edges
            if e.group_id == params["group_id"]
            and params["uuid"] in e.episodes
            and e.expired_at is None
        ]
        return (rows, None, None)

    def _old_section_episodes(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Complete episodes of the group, dated before the reference time,
        whose name holds the section id or number."""
        keys = (params["section_id_key"], params["section_number_key"])
        rows = [
            {"uuid": node.uuid, "name": node.name}
            for node in self.store.nodes.values()
            if node.group_id == params["group_id"]
            and node.valid_at < params["reference_time"]
            and node.name.startswith(params["prefix"])
            and any(key in node.name for key in keys)
        ]
        return (rows, None, None)

    def _text_closure_candidates(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Open, dated facts of the group stated by an old episode, by no
        episode of the new section and by no episode dated after it."""

        def later(uuid: str) -> bool:
            node = self.store.nodes.get(uuid)
            return (
                node is not None
                and node.group_id == params["group_id"]
                and node.valid_at > params["reference_time"]
            )

        rows = [
            {"fact_uuid": e.uuid, "fact": e.fact}
            for e in self.store.edges
            if e.group_id == params["group_id"]
            and e.invalid_at is None
            and e.expired_at is None
            and e.valid_at is not None
            and e.valid_at < params["reference_time"]
            and any(u in params["old_uuids"] for u in e.episodes)
            and not any(u in params["new_uuids"] for u in e.episodes)
            and not any(later(u) for u in e.episodes)
        ]
        return (rows, None, None)

    def _close_facts(self, params: dict[str, Any]) -> tuple[Any, ...]:
        closed = 0
        for e in self.store.edges:
            if (
                e.group_id == params["group_id"]
                and e.uuid in params["fact_uuids"]
                and e.invalid_at is None
                and e.expired_at is None
            ):
                e.invalid_at = params["reference_time"]
                e.expired_at = params["now"]
                closed += 1
        return ([{"closed": closed}], None, None)

    def _reopen(self, params: dict[str, Any]) -> tuple[Any, ...]:
        reopened = 0
        for e in self.store.edges:
            if e.group_id == params["group_id"] and e.uuid in params["fact_uuids"]:
                e.invalid_at = None
                e.expired_at = None
                reopened += 1
        return ([{"reopened": reopened}], None, None)

    def _mark_complete(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """The conditional update: it changes a node that exists, never adds one."""
        if self.store.fail_save_complete is not None:
            raise self.store.fail_save_complete
        node = self.store.nodes.get(params["uuid"])
        if node is None or node.group_id != params["group_id"]:
            return ([{"updated": 0}], None, None)
        node.name = params["name"]
        self.store.events.append(("save", node.uuid, node.name))
        return ([{"updated": 1}], None, None)


class FakeGraphiti:
    """Stands in for ``graphiti_core.Graphiti`` on the calls the backend makes."""

    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.driver = FakeDriver(store)
        self.llm_client = FakeLLM()
        self.add_calls: list[dict[str, Any]] = []
        self.fail_on: dict[str, BaseException] = {}
        self.indices_built = 0
        self.closed = 0
        self.record_tokens = True
        self.new_edges: dict[str, list[FakeEdge]] = {}
        # Facts the model marks contradicted, by the episode that does it.
        self.closes: dict[str, list[str]] = {}
        # Facts another writer closes while an episode is extracted: in the
        # time window, not on the episode's ``entity_edges``.
        self.concurrent_closes: dict[str, list[str]] = {}

    async def build_indices_and_constraints(self) -> None:
        self.indices_built += 1

    async def add_episode(self, **kwargs: Any) -> None:
        uuid = kwargs["uuid"]
        node = self.store.nodes.get(uuid)
        self.store.events.append(("add_episode", uuid, node.name if node else ""))
        self.add_calls.append(kwargs)
        self.llm_client.calls += 2
        if self.record_tokens:
            self.llm_client.token_tracker.record("extract", 100, 10)
            self.llm_client.token_tracker.record("edges", 50, 5)
        if uuid in self.fail_on:
            raise self.fail_on[uuid]
        if node is None:
            raise NodeNotFoundError(uuid)
        added = self.new_edges.get(uuid, [])
        self.store.edges.extend(added)
        # Graphiti lists the edges it resolved or invalidated for the episode,
        # replacing the list a failed attempt left.
        node.entity_edges = [e.uuid for e in added] + list(self.closes.get(uuid, []))
        for closed in self.store.edges:
            if closed.uuid in self.closes.get(uuid, []):
                # What Graphiti does to a contradicted fact.
                closed.invalid_at = kwargs["reference_time"]
                closed.expired_at = datetime.now(UTC)
            elif closed.uuid in self.concurrent_closes.get(uuid, []):
                closed.invalid_at = kwargs["reference_time"]
                closed.expired_at = datetime.now(UTC)

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    store = FakeStore()

    async def save(self: EpisodicNode, driver: object) -> None:
        if store.fail_save_complete is not None and is_complete(self.name):
            raise store.fail_save_complete
        store.events.append(("save", self.uuid, self.name))
        store.nodes[self.uuid] = self.model_copy(deep=True)

    async def get_by_uuids(
        cls: type[EpisodicNode], driver: object, uuids: list[str]
    ) -> list[EpisodicNode]:
        if store.fail_load is not None:
            raise store.fail_load
        return [store.nodes[u].model_copy(deep=True) for u in uuids if u in store.nodes]

    async def get_by_uuid(
        cls: type[EpisodicNode], driver: object, uuid: str
    ) -> EpisodicNode:
        if uuid not in store.nodes:
            raise NodeNotFoundError(uuid)
        return store.nodes[uuid].model_copy(deep=True)

    async def clear_data(driver: object, group_ids: list[str] | None = None) -> None:
        store.cleared.append(group_ids)
        if group_ids is None:
            store.nodes.clear()
            return
        for uuid in [u for u, n in store.nodes.items() if n.group_id in group_ids]:
            del store.nodes[uuid]

    monkeypatch.setattr(EpisodicNode, "save", save)
    monkeypatch.setattr(EpisodicNode, "get_by_uuids", classmethod(get_by_uuids))
    monkeypatch.setattr(EpisodicNode, "get_by_uuid", classmethod(get_by_uuid))
    monkeypatch.setattr(graph_data_operations, "clear_data", clear_data)
    return store


@pytest.fixture
def fake(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> FakeGraphiti:
    graphiti = FakeGraphiti(store)
    builds: list[int] = []

    def build_graphiti(*args: object) -> graphiti_clients.GraphitiRuntime:
        builds.append(1)
        return graphiti_clients.GraphitiRuntime(
            graphiti=graphiti,  # type: ignore[arg-type]
            llm=graphiti.llm_client,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(graphiti_clients, "build_graphiti", build_graphiti)
    graphiti.builds = builds  # type: ignore[attr-defined]
    return graphiti


def make_backend(
    entity_types: tuple[str, ...] = (),
    hints: MemorySchemaHints | None = None,
    *,
    close_on_text_change: bool = False,
) -> GraphitiBackend:
    config = GraphBackendConfig(
        driver="neo4j",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=SecretStr(PASSWORD),
    )
    model = LLMProvider(
        provider=ProviderEnum.OPENAI, name="gpt-test", api_key=SecretStr(API_KEY)
    )
    if hints is None and entity_types:
        hints = MemorySchemaHints(entity_types=list(entity_types))
    return GraphitiBackend(
        config,
        model,
        _NoEmbeddings(),
        hints,
        close_on_text_change=close_on_text_change,
    )


class _NoEmbeddings:
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the fake Graphiti never embeds")


def state(store: FakeStore, ep: EpisodeIn) -> str | None:
    node = store.nodes.get(str(ep.episode_id))
    if node is None:
        return None
    decoded = decode_episode_name(node.name)
    return decoded.state if decoded else "foreign"


# --- name codec -------------------------------------------------------------


def test_d21_episode_name_round_trips() -> None:
    for name_state in ("pending", "complete"):
        name = encode_episode_name(name_state, E1)
        assert name.startswith(f"hd1:{name_state}:")
        decoded = decode_episode_name(name)
        assert decoded is not None
        assert decoded.state == name_state
        assert decoded.source_tool == "family_law"
        assert decoded.chunk_id == "c1_chunk_0"
        assert decoded.source_uri == "corpus/c1.md"
        assert decoded.reference_time_source == "front_matter"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Rate notice 2022",
        'hd2:complete:{"source_tool":"t","chunk_id":"c","source_uri":"u","date_source":"mtime"}',
        'hd1:done:{"source_tool":"t","chunk_id":"c","source_uri":"u","date_source":"mtime"}',
        "hd1:complete:not json",
        "hd1:complete:[1, 2]",
        'hd1:complete:{"source_tool":"t","chunk_id":"c","source_uri":"u"}',
        'hd1:complete:{"source_tool":"t","chunk_id":"c","source_uri":"u","date_source":"x"}',
    ],
)
def test_ing_09_a_name_that_does_not_decode_is_not_ours(name: str) -> None:
    assert decode_episode_name(name) is None
    # Complete is decided by the prefix alone, the rule the count query uses.
    assert is_complete(name) is name.startswith(COMPLETE_PREFIX)


def test_d21_complete_prefix_matches_the_encoder() -> None:
    assert encode_episode_name("complete", E1).startswith(COMPLETE_PREFIX)
    assert not encode_episode_name("pending", E1).startswith(COMPLETE_PREFIX)


def test_d21_source_description_is_a_sentence_with_uri_and_date() -> None:
    sentence = source_sentence(E1)
    assert sentence == "Passage from corpus/c1.md, in effect from 2022-07-01."
    with pytest.raises(json.JSONDecodeError):
        json.loads(sentence)


# --- lifecycle --------------------------------------------------------------


@pytest.mark.asyncio
async def test_d21_absent_episode_is_saved_pending_before_add_episode(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    report = await make_backend().ingest(NS, [E1], "extract rates")

    uid = str(E1.episode_id)
    assert [(kind, u) for kind, u, _ in store.events] == [
        ("save", uid),
        ("add_episode", uid),
        ("save", uid),
    ]
    assert store.events[0][2] == encode_episode_name("pending", E1)
    # add_episode found the pre-saved node, still pending
    assert store.events[1][2] == encode_episode_name("pending", E1)
    assert store.events[2][2] == encode_episode_name("complete", E1)
    assert report.written == (E1.episode_id,)
    assert report.failed is None


@pytest.mark.asyncio
async def test_d21_pending_node_holds_the_episode_fields(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E1.episode_id)] = RuntimeError("stop after the pending save")
    await make_backend().ingest(NS, [E1], "x")

    node = store.nodes[str(E1.episode_id)]
    assert node.uuid == str(E1.episode_id)
    assert node.group_id == NS
    assert node.source == EpisodeType.text
    assert node.content == E1.text
    assert node.valid_at == WHEN
    assert node.source_description == source_sentence(E1)
    assert node.name == encode_episode_name("pending", E1)


@pytest.mark.asyncio
async def test_d21_success_marks_complete_and_keeps_graphiti_fields(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    fake.new_edges[str(E1.episode_id)] = [FakeEdge("edge-1", NS, [str(E1.episode_id)])]
    await backend.ingest(NS, [E1], "x")

    assert state(store, E1) == "complete"
    # The marker save reloads the node, so add_episode's own writes survive.
    assert store.nodes[str(E1.episode_id)].entity_edges == ["edge-1"]
    assert await backend.has_episode(NS, E1.episode_id) is True


@pytest.mark.asyncio
async def test_ing_05_failed_add_episode_stops_and_reports_without_raising(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E2.episode_id)] = TimeoutError("model timed out")
    backend = make_backend()

    report = await backend.ingest(NS, [E1, E2, E3], "x")

    assert report.written == (E1.episode_id,)
    assert report.failed == E2.episode_id
    assert report.error == "model timed out"
    assert [c["uuid"] for c in fake.add_calls] == [
        str(E1.episode_id),
        str(E2.episode_id),
    ]
    assert state(store, E2) == "pending"
    assert state(store, E3) is None
    assert await backend.has_episode(NS, E2.episode_id) is False


@pytest.mark.asyncio
async def test_ing_09_pending_episode_is_processed_again(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E1.episode_id)] = RuntimeError("first run fails")
    backend = make_backend()
    first = await backend.ingest(NS, [E1], "x")
    assert first.failed == E1.episode_id

    del fake.fail_on[str(E1.episode_id)]
    store.events.clear()
    second = await backend.ingest(NS, [E1], "x")

    uid = str(E1.episode_id)
    # No second pending save: the existing pending node is processed again.
    assert [(kind, u) for kind, u, _ in store.events] == [
        ("add_episode", uid),
        ("save", uid),
    ]
    assert second.written == (E1.episode_id,)
    assert state(store, E1) == "complete"


@pytest.mark.asyncio
async def test_ing_09_has_episode_is_false_for_pending_and_absent(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E1.episode_id)] = RuntimeError("leave it pending")
    backend = make_backend()
    await backend.ingest(NS, [E1], "x")

    assert state(store, E1) == "pending"
    assert await backend.has_episode(NS, E1.episode_id) is False
    assert await backend.has_episode(NS, E2.episode_id) is False
    assert fake.add_calls and all(
        c["uuid"] != str(E2.episode_id) for c in fake.add_calls
    )


@pytest.mark.asyncio
async def test_ing_09_a_foreign_node_under_our_uuid_is_refused(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.nodes[str(E1.episode_id)] = EpisodicNode(
        uuid=str(E1.episode_id),
        name="user title",
        group_id=NS,
        source=EpisodeType.text,
        source_description="",
        content=E1.text,
        valid_at=WHEN,
    )
    backend = make_backend()
    assert await backend.has_episode(NS, E1.episode_id) is False

    report = await backend.ingest(NS, [E1], "x")

    assert report.written == ()
    assert report.failed == E1.episode_id
    assert "holds other content" in (report.error or "")
    assert fake.add_calls == []
    assert store.nodes[str(E1.episode_id)].name == "user title"


@pytest.mark.asyncio
async def test_ing_09_a_pending_node_with_another_body_is_refused(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.nodes[str(E1.episode_id)] = EpisodicNode(
        uuid=str(E1.episode_id),
        name=encode_episode_name("pending", E1),
        group_id=NS,
        source=EpisodeType.text,
        source_description="",
        content="old wrong fact",
        valid_at=WHEN,
    )

    report = await make_backend().ingest(NS, [E1], "x")

    assert report.failed == E1.episode_id
    assert fake.add_calls == []


@pytest.mark.asyncio
async def test_d21_an_episode_purged_during_extraction_is_not_brought_back(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    original = fake.add_episode

    async def add_then_lose_the_node(**kwargs: Any) -> None:
        await original(**kwargs)
        del store.nodes[kwargs["uuid"]]

    fake.add_episode = add_then_lose_the_node  # type: ignore[method-assign]
    backend = make_backend()

    report = await backend.ingest(NS, [E1], "x")

    assert report.written == ()
    assert report.failed == E1.episode_id
    assert "removed while it was being extracted" in (report.error or "")
    assert str(E1.episode_id) not in store.nodes
    assert await backend.has_episode(NS, E1.episode_id) is False


@pytest.mark.asyncio
async def test_mb_02_context_from_another_namespace_is_left_out(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    foreign = episode("foreign")
    await backend.ingest(OTHER_NS, [foreign], "x")
    await backend.ingest(NS, [E1], "x")
    with_context = episode("ctx", context=(foreign.episode_id, E1.episode_id))
    fake.add_calls.clear()

    await backend.ingest(NS, [with_context], "x")

    assert fake.add_calls[0]["previous_episode_uuids"] == [str(E1.episode_id)]


@pytest.mark.asyncio
async def test_a_database_that_went_offline_raises_unavailable(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    from neo4j.exceptions import DatabaseUnavailable

    fake.fail_on[str(E1.episode_id)] = DatabaseUnavailable("database is offline")

    with pytest.raises(MemoryBackendUnavailableError, match="DatabaseUnavailable"):
        await make_backend().ingest(NS, [E1], "x")


@pytest.mark.asyncio
async def test_gph_01_a_key_from_the_environment_is_redacted(
    store: FakeStore, fake: FakeGraphiti, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-canary-7781")
    fake.fail_on[str(E1.episode_id)] = RuntimeError(
        "Incorrect API key provided: sk-env-canary-7781"
    )
    backend = GraphitiBackend(
        GraphBackendConfig(driver="falkordb", uri="redis://u:db-canary-33@host:6379"),
        LLMProvider(provider=ProviderEnum.OPENAI, name="m"),
        _NoEmbeddings(),
    )

    report = await backend.ingest(NS, [E1], "x")

    assert report.error == "Incorrect API key provided: ***"
    assert backend._redact("redis://u:db-canary-33@host") == "redis://u:***@host"


@pytest.mark.asyncio
async def test_ing_02_complete_episode_is_skipped_with_no_add_episode(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [E1], "x")
    fake.add_calls.clear()
    store.events.clear()

    report = await backend.ingest(NS, [E1, E2], "x")

    assert [c["uuid"] for c in fake.add_calls] == [str(E2.episode_id)]
    assert report.written == (E2.episode_id,)
    assert report.llm_calls == 2
    assert all(u != str(E1.episode_id) for _, u, _ in store.events)


@pytest.mark.asyncio
async def test_d21_marker_save_failure_is_reported_and_leaves_pending(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.fail_save_complete = RuntimeError("marker write rejected")

    report = await make_backend().ingest(NS, [E1, E2], "x")

    assert report.written == ()
    assert report.failed == E1.episode_id
    assert report.error == "marker write rejected"
    assert state(store, E1) == "pending"


@pytest.mark.asyncio
async def test_mb_02_episode_of_another_namespace_is_not_touched(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(OTHER_NS, [E1], "x")
    fake.add_calls.clear()

    assert await backend.has_episode(NS, E1.episode_id) is False
    report = await backend.ingest(NS, [E1], "x")

    assert report.failed == E1.episode_id
    assert report.error is not None and OTHER_NS in report.error
    assert fake.add_calls == []
    assert store.nodes[str(E1.episode_id)].group_id == OTHER_NS


# --- what reaches add_episode -----------------------------------------------


@pytest.mark.asyncio
async def test_ing_10_instructions_and_context_reach_add_episode(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    await make_backend().ingest(NS, [E1, E2, E3], "Extract amounts as entities.")

    first, _second, third = fake.add_calls
    assert first["custom_extraction_instructions"] == "Extract amounts as entities."
    assert third["custom_extraction_instructions"] == "Extract amounts as entities."
    assert first["previous_episode_uuids"] == []
    assert third["previous_episode_uuids"] == [
        str(E1.episode_id),
        str(E2.episode_id),
    ]
    assert third["uuid"] == str(E3.episode_id)
    assert third["group_id"] == NS
    assert third["source"] == EpisodeType.text
    assert third["reference_time"] == WHEN
    assert third["episode_body"] == E3.text
    assert third["source_description"] == source_sentence(E3)
    assert third["entity_types"] is None


@pytest.mark.asyncio
async def test_d21_entity_types_become_fieldless_models(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    await make_backend(("Payment", "Agency")).ingest(NS, [E1], "x")

    entity_types = fake.add_calls[0]["entity_types"]
    assert sorted(entity_types) == ["Agency", "Payment"]
    for name, model in entity_types.items():
        assert issubclass(model, BaseModel)
        assert model.__name__ == name
        assert model.model_fields == {}


# --- fact start dates (Fix A) -----------------------------------------------

JULY_2020 = datetime(2020, 7, 1, tzinfo=UTC)
JULY_2026 = datetime(2026, 7, 1, tzinfo=UTC)


def dated_episode(name: str, when: datetime) -> EpisodeIn:
    return EpisodeIn(
        episode_id=uuid5(NAMESPACE_URL, name),
        source_tool="family_law",
        chunk_id=f"{name}_chunk_0",
        source_uri=f"corpus/{name}.md",
        text=f"Section text of {name}.",
        reference_time=when,
        reference_time_source="front_matter",
    )


C2020 = dated_episode("compilation-2020", JULY_2020)
C2026 = dated_episode("compilation-2026", JULY_2026)


def edge(store: FakeStore, uuid: str) -> FakeEdge:
    return next(e for e in store.edges if e.uuid == uuid)


@pytest.mark.asyncio
async def test_fix_a_an_undated_fact_gets_the_episode_reference_time(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    uid = str(C2026.episode_id)
    fake.new_edges[uid] = [FakeEdge("new", NS, [uid])]

    report = await make_backend().ingest(NS, [C2026], "x")

    assert report.written == (C2026.episode_id,)
    assert edge(store, "new").valid_at == datetime(2026, 7, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_fix_a_a_dated_fact_keeps_its_start(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    uid = str(C2026.episode_id)
    own = datetime(2024, 1, 1, tzinfo=UTC)
    fake.new_edges[uid] = [FakeEdge("dated", NS, [uid], valid_at=own)]

    await make_backend().ingest(NS, [C2026], "x")

    assert edge(store, "dated").valid_at == own


@pytest.mark.asyncio
async def test_fix_a_facts_of_another_group_or_episode_are_not_touched(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    uid = str(C2026.episode_id)
    store.edges.append(FakeEdge("other-group", OTHER_NS, [uid]))
    store.edges.append(FakeEdge("other-episode", NS, ["some-other-episode"]))

    await make_backend().ingest(NS, [C2026], "x")

    assert edge(store, "other-group").valid_at is None
    assert edge(store, "other-episode").valid_at is None


@pytest.mark.asyncio
async def test_fix_a_a_restated_undated_fact_keeps_the_earliest_start(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [C2020], "x")
    old, new = str(C2020.episode_id), str(C2026.episode_id)
    # Undated before the fix; Graphiti adds the new episode to a duplicate.
    store.edges.append(FakeEdge("restated", NS, [old, new]))

    await backend.ingest(NS, [C2026], "x")

    assert edge(store, "restated").valid_at == JULY_2020


@pytest.mark.asyncio
async def test_fix_a_dating_is_one_scoped_write_before_the_marker(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    await make_backend().ingest(NS, [C2026], "x")

    writes = [(q, p) for q, p in store.queries if "uuid" in p]
    assert [q for q, _ in writes] == [DATE_EPISODE_FACTS_QUERY, writes[1][0]]
    assert "SET e.name" in writes[1][0]
    assert writes[0][1] == {
        "uuid": str(C2026.episode_id),
        "group_id": NS,
        "reference_time": JULY_2026,
    }
    assert "r.valid_at IS NULL" in DATE_EPISODE_FACTS_QUERY
    assert "r.group_id = $group_id" in DATE_EPISODE_FACTS_QUERY


@pytest.mark.asyncio
async def test_fix_a_a_failed_dating_is_reported_and_leaves_pending(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.fail_date = RuntimeError("dating rejected")

    report = await make_backend().ingest(NS, [C2026], "x")

    assert report.written == ()
    assert report.failed == C2026.episode_id
    assert report.error == "dating rejected"
    assert state(store, C2026) == "pending"


@pytest.mark.asyncio
async def test_fix_a_other_stores_skip_dating_and_still_complete(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.provider = GraphProvider.FALKORDB
    uid = str(C2026.episode_id)
    fake.new_edges[uid] = [FakeEdge("new", NS, [uid])]

    report = await make_backend().ingest(NS, [C2026], "x")

    assert report.written == (C2026.episode_id,)
    assert edge(store, "new").valid_at is None
    assert all(q != DATE_EPISODE_FACTS_QUERY for q, _ in store.queries)


@pytest.mark.asyncio
async def test_fix_a_backfill_dates_by_the_earliest_episode_and_is_idempotent(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [C2020, C2026], "x")
    old, new = str(C2020.episode_id), str(C2026.episode_id)
    kept = datetime(2023, 3, 1, tzinfo=UTC)
    store.edges.extend(
        [
            FakeEdge("both", NS, [new, old]),
            FakeEdge("only-2026", NS, [new]),
            FakeEdge("dated", NS, [old], valid_at=kept),
            FakeEdge("no-episode-node", NS, ["missing"]),
            FakeEdge("other-group", OTHER_NS, [old]),
        ]
    )

    assert await backend.backfill_fact_start_dates(NS) == 2

    assert edge(store, "both").valid_at == JULY_2020
    assert edge(store, "only-2026").valid_at == JULY_2026
    assert edge(store, "dated").valid_at == kept
    assert edge(store, "no-episode-node").valid_at is None
    assert edge(store, "other-group").valid_at is None
    assert await backend.backfill_fact_start_dates(NS) == 0
    assert store.queries[-1] == (BACKFILL_FACT_START_QUERY, {"group_id": NS})


@pytest.mark.asyncio
async def test_fix_a_backfill_refuses_a_store_it_cannot_date(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.provider = GraphProvider.KUZU

    with pytest.raises(TemporalMemoryError, match="neo4j only"):
        await make_backend().backfill_fact_start_dates(NS)


# --- closure check -----------------------------------------------------------

DEFINITIONS = ("Part 2—Interpretation", "Division 1—Definitions")
RATE_PATH = ("Part 3—Family tax benefit", "Division 2—Rate")
ACT_2020 = "corpus/family-assistance-2020.md"
ACT_2026 = "corpus/family-assistance-2026.md"


def sectioned(
    name: str,
    text: str,
    section: SectionKey,
    *,
    when: datetime = JULY_2026,
    uri: str = ACT_2026,
    section_ids: tuple[UUID, ...] = (),
    tool: str = "family_law",
    key: str = "",
) -> EpisodeIn:
    episode_id = uuid5(NAMESPACE_URL, name)
    return EpisodeIn(
        episode_id=episode_id,
        source_tool=tool,
        chunk_id=f"{name}_chunk",
        source_uri=uri,
        text=text,
        reference_time=when,
        reference_time_source="front_matter",
        section_id=section.section_id,
        section_number=section.number,
        section_path=section.path,
        section_episode_ids=section_ids or (episode_id,),
        document_key=key,
    )


def old(name: str, text: str, section: SectionKey) -> EpisodeIn:
    return sectioned(name, text, section, when=JULY_2020, uri=ACT_2020)


async def close_with(
    store: FakeStore,
    fake: FakeGraphiti,
    old_episodes: list[EpisodeIn],
    facts: dict[str, tuple[str, EpisodeIn]],
    new: EpisodeIn,
    backend: GraphitiBackend | None = None,
    *,
    replaced: bool = True,
) -> IngestReport:
    """Ingest the old episodes, add their facts, then let ``new`` close them.

    Each fact joins its own two entities. With ``replaced``, ``new`` also
    writes a fact between the same two entities, so the replacement guard
    lets the closure stand and the text rules decide.
    """
    backend = backend or make_backend()
    for episode_in in old_episodes:
        await backend.ingest(NS, [episode_in], "x")
    new_id = str(new.episode_id)
    for uuid, (text, origin) in facts.items():
        source, target = f"{uuid}-source", f"{uuid}-target"
        store.edges.append(
            FakeEdge(
                uuid,
                NS,
                [str(origin.episode_id)],
                JULY_2020,
                fact=text,
                source=source,
                target=target,
            )
        )
        if replaced:
            fake.new_edges.setdefault(new_id, []).append(
                FakeEdge(
                    f"{uuid}-next",
                    NS,
                    [new_id],
                    fact=text,
                    source=source,
                    target=target,
                )
            )
    fake.closes.setdefault(new_id, []).extend(facts)
    return await backend.ingest(NS, [new], "x")


def is_open(store: FakeStore, uuid: str) -> bool:
    found = edge(store, uuid)
    return found.invalid_at is None and found.expired_at is None


S61AA = SectionKey("h4_part_3_61aa_nil_rate", "61AA", RATE_PATH)
OLD_61AA = """#### 61AA Nil Part A rate for certain visa holders
(1) Section 61AA applies to an individual on a day if the individual holds a \
permanent visa other than:
(a) a Subclass 117 (Orphan Relative) visa;
(b) a Subclass 837 (Orphan Relative) visa.
(2) The individual's Part A rate of family tax benefit is nil on that day unless:
(c) parental leave pay or dad and partner pay under the Paid Parental Leave Act \
2010 is payable to the individual on that day."""
NEW_61AA = OLD_61AA.replace(
    "parental leave pay or dad and partner pay under", "parental leave pay under"
)
VISA_FACT = (
    "A Subclass 117 (Orphan Relative) visa is excluded from the permanent visa "
    "category for section 61AA"
)
DAPP_FACT = (
    "Section 61AA's nil Part A family tax benefit rate does not apply to an "
    "individual if parental leave pay or dad and partner pay under the Paid "
    "Parental Leave Act 2010 is payable"
)


@pytest.mark.asyncio
async def test_closure_reopens_a_fact_on_a_kept_line_not_one_on_a_changed_line(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("61aa-2020", OLD_61AA, S61AA)

    report = await close_with(
        store,
        fake,
        [o],
        {"visa": (VISA_FACT, o), "dapp": (DAPP_FACT, o)},
        sectioned("61aa-2026", NEW_61AA, S61AA),
    )

    assert is_open(store, "visa")
    assert not is_open(store, "dapp")
    assert edge(store, "dapp").invalid_at == JULY_2026
    counts = (report.facts_closed, report.facts_reopened, report.facts_kept_closed)
    assert counts == (2, 1, 1)


@pytest.mark.asyncio
async def test_closure_a_fact_closed_by_another_section_is_reopened(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    s3aa = SectionKey("h4_part_2_3aa_paid_parental_leave", "3AA", DEFINITIONS)
    s3 = SectionKey("h4_part_2_3_definitions", "3", DEFINITIONS)
    o = old(
        "3aa-2020",
        "#### 3AA Paid parental leave\n(1) An individual is not eligible for "
        "family tax benefit while parental leave pay is payable to the individual.",
        s3aa,
    )
    new = sectioned(
        "3-2026",
        "#### 3 Definitions\n(1) In this Act:\nparental leave pay has the same "
        "meaning as in the Paid Parental Leave Act 2010.",
        s3,
    )
    fact = "An individual is not eligible while parental leave pay is payable"

    await close_with(store, fake, [o], {"ppl": (fact, o)}, new)

    assert is_open(store, "ppl")


@pytest.mark.asyncio
async def test_closure_an_unchanged_section_reopens_every_fact(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    s85cb = SectionKey("h4_part_8_85cb_notifying_support_agency", "85CB", RATE_PATH)
    line = (
        "When an approved provider gives a certificate under section 85CB, it must "
        "notify an appropriate State or Territory support agency in accordance "
        "with section 204K of the Family Assistance Administration Act."
    )
    heading = "#### 85CB Notifying support agency"
    o = old("85cb-2020", f"{heading}\n{line}", s85cb)
    # The same words, wrapped differently: another chunk text, another episode.
    reflowed = line.replace("State or Territory", "State or\nTerritory")
    new = sectioned("85cb-2026", f"{heading}\n\n{reflowed}\n", s85cb)
    fact = "An approved provider must notify a support agency"

    await close_with(store, fake, [o], {"notify": (fact, o)}, new)

    assert is_open(store, "notify")


S3 = SectionKey(
    "h4_part_2_interpretation_division_1_definitions_3_definitions", "3", DEFINITIONS
)
HSS_OLD = (
    "Human Services Secretary means the Secretary of the Department administered "
    "by the Minister administering the Human Services (Centrelink) Act 1997."
)
HSS_NEW = (
    "Human Services Secretary means the Chief Executive Officer of Services "
    "Australia."
)
PPL = "parental leave pay has the same meaning as in the Paid Parental Leave Act 2010."


@pytest.mark.asyncio
async def test_closure_a_changed_definition_stays_closed(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("3-2020", f"#### 3 Definitions\n(1) In this Act:\n{HSS_OLD}\n{PPL}", S3)
    new = sectioned(
        "3-2026", f"#### 3 Definitions\n(1) In this Act:\n{HSS_NEW}\n{PPL}", S3
    )

    report = await close_with(
        store,
        fake,
        [o],
        {"hss": (HSS_OLD.rstrip("."), o), "ppl": (PPL.rstrip("."), o)},
        new,
    )

    assert not is_open(store, "hss")
    assert is_open(store, "ppl")
    assert (report.facts_reopened, report.facts_kept_closed) == (1, 1)


@pytest.mark.asyncio
async def test_closure_a_renamed_heading_with_its_number_is_the_same_section(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    path = ("Part 8—Security",)
    o = old(
        "57gj-2020",
        "#### 57GJ Security notice from Home Affairs Minister\n"
        "(1) The Home Affairs Minister may give the Secretary a security notice "
        "about an individual.\n(2) The notice must be in writing.",
        SectionKey("h4_part_8_57gj_security_notice_from_home_affairs", "57GJ", path),
    )
    new = sectioned(
        "57gj-2026",
        "#### 57GJ Benefit restriction notice from ASIO Minister\n"
        "(1) The ASIO Minister may give the Secretary a benefit restriction "
        "notice about an individual.\n(2) The notice must be in writing.",
        SectionKey("h4_part_8_57gj_benefit_restriction_notice_from_asio", "57GJ", path),
    )
    giver = (
        "The Home Affairs Minister may give the Secretary a security notice "
        "about an individual"
    )

    await close_with(
        store,
        fake,
        [o],
        {"giver": (giver, o), "writing": ("A security notice must be in writing", o)},
        new,
    )

    # Without the number rule the ids differ and both would be reopened.
    assert not is_open(store, "giver")
    assert is_open(store, "writing")


@pytest.mark.asyncio
async def test_closure_a_split_section_counts_as_another_section_a_known_limit(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old(
        "s3-2020",
        "#### 3 Applicable percentage\n(1) The applicable percentage for an "
        "individual is 85%.",
        SectionKey("h4_part_1_3_applicable_percentage", "3", DEFINITIONS),
    )
    new = sectioned(
        "s3a-2026",
        "#### 3A Applicable percentage—other cases\n(1) The applicable "
        "percentage for an individual in other cases is 50%.",
        SectionKey("h4_part_1_3a_applicable_percentage_other", "3A", DEFINITIONS),
    )
    fact = "The applicable percentage for an individual is 85%"

    await close_with(store, fake, [o], {"pct": (fact, o)}, new)

    # "3" and "3A" are different sections, so the closure is undone even if
    # the percentage moved to 3A.
    assert is_open(store, "pct")


S61A = SectionKey("h4_part_3_61a_rate_reduction", "61A", RATE_PATH)
OLD_61A = (
    "#### 61A Rate reduction\n"
    "(1) The rate is reduced by 20 cents for each dollar above the threshold.\n"
    "(2) A reduction does not apply to a newly arrived resident.\n"
    "(3) The threshold is $5,000 a year."
)


@pytest.mark.parametrize("whole_section", [True, False])
@pytest.mark.asyncio
async def test_closure_reads_every_chunk_of_the_new_section(
    store: FakeStore, fake: FakeGraphiti, whole_section: bool
) -> None:
    backend = make_backend()
    o = old("61a-2020", OLD_61A, S61A)
    # 2026: the section is two chunks. The resident line is in the first,
    # unchanged; the second holds the changed threshold and closes the facts.
    first = sectioned(
        "61a-2026-0",
        "#### 61A Rate reduction\n"
        "(1) The rate is reduced by 20 cents for each dollar above the threshold.\n"
        "(2) A reduction does not apply to a newly arrived resident.",
        S61A,
    )
    await backend.ingest(NS, [first], "x")
    second_id = uuid5(NAMESPACE_URL, "61a-2026-1")
    second = sectioned(
        "61a-2026-1",
        "(3) The threshold is $6,000 a year.",
        S61A,
        section_ids=(first.episode_id, second_id) if whole_section else (second_id,),
    )
    facts = {
        "resident": ("A reduction does not apply to a newly arrived resident", o),
        "threshold": ("The threshold is $5,000 a year", o),
    }

    await close_with(store, fake, [o], facts, second, backend)

    assert is_open(store, "resident") is whole_section
    assert not is_open(store, "threshold")


@pytest.mark.asyncio
async def test_closure_check_is_skipped_for_an_episode_with_no_section(
    store: FakeStore, fake: FakeGraphiti, caplog: pytest.LogCaptureFixture
) -> None:
    # Neither episode has a section id: no window query is made at all.
    o = dated_episode("3-2020-unsectioned", JULY_2020)
    backend = make_backend()

    with caplog.at_level("INFO", logger="holodeck.lib.memory.backends.graphiti"):
        report = await close_with(
            store,
            fake,
            [o],
            {"ppl": (PPL, o)},
            dated_episode("3-2026-unsectioned", JULY_2026),
            backend,
        )
        await backend.ingest(NS, [dated_episode("another", JULY_2026)], "x")

    assert not is_open(store, "ppl")
    assert report.facts_closed == 0
    assert all(q != CLOSED_SINCE_QUERY for q, _ in store.queries)
    skipped = [r for r in caplog.records if "closure check skipped" in r.message]
    assert len(skipped) == 1


@pytest.mark.asyncio
async def test_closure_leaves_other_groups_and_earlier_closures_alone(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("3-2020", f"#### 3 Definitions\n{PPL}", S3)
    earlier = datetime(2021, 1, 1, tzinfo=UTC)
    store.edges.append(
        FakeEdge(
            "closed-before",
            NS,
            [str(o.episode_id)],
            JULY_2020,
            fact=PPL,
            invalid_at=earlier,
            expired_at=earlier,
        )
    )
    store.edges.append(FakeEdge("other-group", OTHER_NS, [str(o.episode_id)], fact=PPL))
    new = sectioned("3-2026", f"#### 3 Definitions\n{PPL}\n{HSS_NEW}", S3)
    fake.closes[str(new.episode_id)] = ["other-group"]

    report = await close_with(store, fake, [o], {"ppl": (PPL, o)}, new)

    assert is_open(store, "ppl")
    assert edge(store, "closed-before").expired_at == earlier
    assert edge(store, "other-group").expired_at is not None
    assert report.facts_closed == 1
    reopen = [p for q, p in store.queries if q == REOPEN_FACTS_QUERY]
    assert reopen == [{"group_id": NS, "fact_uuids": ["ppl"]}]


@pytest.mark.asyncio
async def test_closure_an_old_episode_without_section_keys_falls_through_to_text(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # Written before section metadata: its section is unknown, so rule (a)
    # proves nothing and the unchanged text decides.
    o = dated_episode("3-2020-unsectioned", JULY_2020)
    new = sectioned("3-2026", f"#### 3 Definitions\n{o.text}\n{HSS_NEW}", S3)

    await close_with(store, fake, [o], {"text": ("Section text of 3-2020", o)}, new)

    assert is_open(store, "text")


@pytest.mark.asyncio
async def test_closure_a_new_fact_closed_on_arrival_keeps_its_closure(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    new = sectioned("3-2026", f"#### 3 Definitions\n{PPL}", S3)
    uid = str(new.episode_id)
    fake.new_edges[uid] = [FakeEdge("arrived", NS, [uid], JULY_2026, fact=PPL)]
    fake.closes[uid] = ["arrived"]

    report = await make_backend().ingest(NS, [new], "x")

    assert not is_open(store, "arrived")
    assert (report.facts_closed, report.facts_kept_closed) == (1, 1)


@pytest.mark.asyncio
async def test_closure_guard_reopens_a_fact_with_no_replacement(
    store: FakeStore, fake: FakeGraphiti, caplog: pytest.LogCaptureFixture
) -> None:
    # The DAPP line changed, so the text rules would keep the closure; but
    # the new episode wrote no fact that could replace it.
    o = old("61aa-2020", OLD_61AA, S61AA)

    with caplog.at_level("INFO", logger="holodeck.lib.memory.backends.graphiti"):
        report = await close_with(
            store,
            fake,
            [o],
            {"dapp": (DAPP_FACT, o)},
            sectioned("61aa-2026", NEW_61AA, S61AA),
            replaced=False,
        )

    assert is_open(store, "dapp")
    assert report.facts_reopened == 1
    assert report.facts_reopened_no_replacement == 1
    assert report.facts_kept_closed == 0
    logged = [r.getMessage() for r in caplog.records if "closure check" in r.message]
    assert any("1 reopened (1 with no replacement)" in m for m in logged)


@pytest.mark.parametrize(
    ("source", "target", "name", "stays_closed"),
    [
        # A -uses-> B is not replaced by B -uses-> C (getzep/graphiti#1729).
        ("B", "C", "USES", False),
        # The same two entities, reversed.
        ("B", "A", "USED_BY", True),
        # The same source and the same name, folded.
        ("A", "D", "uses", True),
        # The same source, another name.
        ("A", "D", "OWNS", False),
    ],
)
@pytest.mark.asyncio
async def test_closure_guard_needs_a_fact_that_could_replace_the_closed_one(
    store: FakeStore,
    fake: FakeGraphiti,
    source: str,
    target: str,
    name: str,
    stays_closed: bool,
) -> None:
    backend = make_backend()
    o = old("61aa-2020", OLD_61AA, S61AA)
    await backend.ingest(NS, [o], "x")
    store.edges.append(
        FakeEdge(
            "dapp",
            NS,
            [str(o.episode_id)],
            JULY_2020,
            fact=DAPP_FACT,
            source="A",
            target="B",
            name="USES",
        )
    )
    new = sectioned("61aa-2026", NEW_61AA, S61AA)
    uid = str(new.episode_id)
    fake.new_edges[uid] = [
        FakeEdge("next", NS, [uid], source=source, target=target, name=name)
    ]
    fake.closes[uid] = ["dapp"]

    report = await backend.ingest(NS, [new], "x")

    assert is_open(store, "dapp") is not stays_closed
    assert report.facts_reopened_no_replacement == (0 if stays_closed else 1)


@pytest.mark.asyncio
async def test_edge_types_and_map_reach_add_episode(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    hints = MemorySchemaHints.model_validate(
        {
            "entity_types": ["Payment", {"name": "Agency", "description": "A body"}],
            "edge_types": [
                {
                    "name": "HAS_RATE",
                    "description": "A payment's rate.",
                    "attributes": [{"name": "amount", "type": "number"}],
                }
            ],
            "edge_type_map": [
                {"source": "Payment", "target": "Entity", "edges": ["HAS_RATE"]}
            ],
        }
    )

    await make_backend(hints=hints).ingest(NS, [E1], "x")

    call = fake.add_calls[0]
    assert call["entity_types"]["Agency"].__doc__ == "A body"
    assert sorted(call["edge_types"]) == ["HAS_RATE"]
    assert set(call["edge_types"]["HAS_RATE"].model_fields) == {"amount"}
    assert call["edge_type_map"] == {("Payment", "Entity"): ["HAS_RATE"]}


@pytest.mark.asyncio
async def test_no_edge_types_pass_none(store: FakeStore, fake: FakeGraphiti) -> None:
    await make_backend().ingest(NS, [E1], "x")

    assert fake.add_calls[0]["edge_types"] is None
    assert fake.add_calls[0]["edge_type_map"] is None


@pytest.mark.parametrize("effort", ["high", None])
@pytest.mark.asyncio
async def test_reasoning_effort_reaches_build_graphiti(
    store: FakeStore,
    fake: FakeGraphiti,
    monkeypatch: pytest.MonkeyPatch,
    effort: ReasoningEffort | None,
) -> None:
    seen: list[tuple[object, ...]] = []

    def build_graphiti(*args: object) -> graphiti_clients.GraphitiRuntime:
        seen.append(args)
        return graphiti_clients.GraphitiRuntime(
            graphiti=fake,  # type: ignore[arg-type]
            llm=fake.llm_client,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(graphiti_clients, "build_graphiti", build_graphiti)
    backend = GraphitiBackend(
        GraphBackendConfig(driver="neo4j", uri="bolt://127.0.0.1:7687"),
        LLMProvider(provider=ProviderEnum.OPENAI, name="m", api_key=SecretStr("k")),
        _NoEmbeddings(),
        None,
        effort,
    )

    await backend.ingest(NS, [E1], "x")

    assert seen[0][3] == effort


def test_closure_section_keys_round_trip_through_the_name() -> None:
    sectioned_in = sectioned("57gj", "text", SectionKey("s57gj", "57GJ", ("Part 8",)))

    decoded = decode_episode_name(encode_episode_name("complete", sectioned_in))

    assert decoded is not None
    assert decoded.section == SectionKey("s57gj", "57GJ", ("Part 8",))
    unsectioned = decode_episode_name(encode_episode_name("complete", E1))
    assert unsectioned is not None
    assert unsectioned.section is None
    assert "section_id" not in encode_episode_name("complete", E1)


# --- counts, tokens, indices, purge, close ----------------------------------


@pytest.mark.asyncio
async def test_d21_episode_count_counts_complete_episodes_in_one_query(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E3.episode_id)] = RuntimeError("leave pending")
    backend = make_backend()
    await backend.ingest(NS, [E1, E2, E3], "x")
    await backend.ingest(OTHER_NS, [episode("other")], "x")
    store.queries.clear()

    assert await backend.episode_count(NS) == 2

    assert len(store.queries) == 1
    query, params = store.queries[0]
    assert "count(" in query
    assert params == {"group_id": NS, "prefix": "hd1:complete:"}


@pytest.mark.asyncio
async def test_ing_06_llm_calls_and_tokens_are_the_delta_of_one_call(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [E1], "x")

    report = await backend.ingest(NS, [E2, E3], "x")

    assert report.llm_calls == 4
    assert report.input_tokens == 300
    assert report.output_tokens == 30


@pytest.mark.asyncio
async def test_ing_06_tokens_are_none_when_the_client_records_none(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.record_tokens = False

    report = await make_backend().ingest(NS, [E1], "x")

    assert report.llm_calls == 2
    assert report.input_tokens is None
    assert report.output_tokens is None


@pytest.mark.asyncio
async def test_d21_indices_are_built_once_before_the_first_write(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    assert await backend.has_episode(NS, E1.episode_id) is False
    assert fake.indices_built == 0

    await backend.ingest(NS, [E1], "x")
    await backend.ingest(NS, [E2], "x")

    assert fake.indices_built == 1
    assert fake.builds == [1]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_d21_namespace_uses_the_driver_graphiti_clones_for_the_group(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [E1], "x")
    await backend.episode_count(OTHER_NS)

    assert fake.driver.clones == [NS, OTHER_NS]


@pytest.mark.asyncio
async def test_mb_04_purge_clears_exactly_one_namespace(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [E1], "x")
    other = episode("other")
    await backend.ingest(OTHER_NS, [other], "x")

    await backend.purge(NS)

    assert store.cleared == [[NS]]
    assert str(E1.episode_id) not in store.nodes
    assert str(other.episode_id) in store.nodes


@pytest.mark.asyncio
async def test_close_is_safe_before_use_and_closes_the_driver_once(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    unused = make_backend()
    await unused.close()
    assert fake.closed == 0

    backend = make_backend()
    await backend.has_episode(NS, E1.episode_id)
    await backend.close()
    await backend.close()
    assert fake.closed == 1


@dataclass
class HeldBuild:
    """A ``build_graphiti`` that waits in its thread until it is released."""

    graphiti: FakeGraphiti
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def __call__(self, *args: object) -> graphiti_clients.GraphitiRuntime:
        self.entered.set()
        assert self.release.wait(timeout=10), "the test never released the build"
        return graphiti_clients.GraphitiRuntime(
            graphiti=self.graphiti,  # type: ignore[arg-type]
            llm=self.graphiti.llm_client,  # type: ignore[arg-type]
        )

    async def wait_entered(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, 10)


@pytest.fixture
def held(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> HeldBuild:
    build = HeldBuild(FakeGraphiti(store))
    monkeypatch.setattr(graphiti_clients, "build_graphiti", build)
    return build


async def _cancelled_during_build(backend: GraphitiBackend, held: HeldBuild) -> None:
    task = asyncio.create_task(backend.has_episode(NS, E1.episode_id))
    await held.wait_entered()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_runtime_built_after_its_caller_was_cancelled_is_closed_once(
    held: HeldBuild,
) -> None:
    backend = make_backend()
    await _cancelled_during_build(backend, held)
    assert held.graphiti.closed == 0

    held.release.set()
    for _ in range(500):
        if held.graphiti.closed:
            break
        await asyncio.sleep(0.01)

    assert held.graphiti.closed == 1
    await backend.close()
    assert held.graphiti.closed == 1


@pytest.mark.asyncio
async def test_close_after_a_cancelled_build_waits_for_it_and_closes_it(
    held: HeldBuild,
) -> None:
    backend = make_backend()
    await _cancelled_during_build(backend, held)

    closing = asyncio.create_task(backend.close())
    await asyncio.sleep(0.05)
    assert not closing.done()  # the runtime does not exist yet
    held.release.set()
    await closing

    assert held.graphiti.closed == 1


@pytest.mark.asyncio
async def test_close_during_a_build_closes_the_runtime_it_produces(
    held: HeldBuild,
) -> None:
    backend = make_backend()
    using = asyncio.create_task(backend.has_episode(NS, E1.episode_id))
    await held.wait_entered()

    closing = asyncio.create_task(backend.close())
    await asyncio.sleep(0.05)
    held.release.set()
    await asyncio.gather(using, closing)

    # close() waited for the build, so the runtime it produced is closed.
    assert held.graphiti.closed == 1
    await backend.close()
    assert held.graphiti.closed == 1


# --- errors -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreachable_store_raises_unavailable_without_secrets(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.fail_load = ConnectionRefusedError(f"connect refused (auth {PASSWORD})")
    backend = make_backend()

    with pytest.raises(MemoryBackendUnavailableError) as raised:
        await backend.has_episode(NS, E1.episode_id)
    with pytest.raises(MemoryBackendUnavailableError):
        await backend.ingest(NS, [E1], "x")

    assert raised.value.backend == "graphiti"
    assert PASSWORD not in str(raised.value)
    assert "ConnectionRefusedError" in str(raised.value)


@pytest.mark.asyncio
async def test_a_store_error_that_is_not_a_lost_store_is_not_unavailable(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    store.fail_load = ValueError(f"bad query parameter (auth {PASSWORD})")
    backend = make_backend()

    with pytest.raises(TemporalMemoryError) as checked:
        await backend.has_episode(NS, E1.episode_id)
    with pytest.raises(TemporalMemoryError) as ingested:
        await backend.ingest(NS, [E1], "x")

    for raised in (checked, ingested):
        assert not isinstance(raised.value, MemoryBackendUnavailableError)
        assert "ValueError" in str(raised.value)
        assert PASSWORD not in str(raised.value)


@pytest.mark.asyncio
async def test_store_lost_during_add_episode_raises_unavailable(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E1.episode_id)] = ServiceUnavailable("database went away")

    with pytest.raises(MemoryBackendUnavailableError) as raised:
        await make_backend().ingest(NS, [E1], "x")

    assert type(raised.value) is MemoryBackendUnavailableError
    assert "database went away" in str(raised.value)
    assert state(store, E1) == "pending"


@pytest.mark.asyncio
async def test_gph_01_api_key_is_redacted_from_a_reported_error(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    fake.fail_on[str(E1.episode_id)] = RuntimeError(f"401 for key {API_KEY}")

    report = await make_backend().ingest(NS, [E1], "x")

    assert report.error is not None
    assert API_KEY not in report.error
    assert report.error == "401 for key ***"


@pytest.mark.asyncio
async def test_cfg_05_missing_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "graphiti_core", None)
    backend = make_backend()

    with pytest.raises(MemoryConfigError) as raised:
        await backend.has_episode(NS, E1.episode_id)

    assert "pip install 'holodeck-ai[graphiti]'" in str(raised.value)


# --- registry and import boundary -------------------------------------------


def test_mb_01_registry_creates_a_graphiti_backend_that_satisfies_the_port() -> None:
    config = GraphBackendConfig(driver="neo4j", uri="bolt://127.0.0.1:7687")
    model = LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-test")

    backend = create_backend(
        config, model, _NoEmbeddings(), MemorySchemaHints(entity_types=["Payment"])
    )

    assert isinstance(backend, GraphitiBackend)
    assert isinstance(backend, MemoryBackend)
    assert backend.name == "graphiti"


def test_cfg_05_importing_the_registry_does_not_import_graphiti() -> None:
    script = (
        "import sys\n"
        "from holodeck.lib.memory.registry import create_backend\n"
        "from holodeck.models.llm import LLMProvider\n"
        "from holodeck.models.tool import GraphBackendConfig\n"
        "class E:\n"
        "    async def generate_embeddings(self, texts):\n"
        "        return []\n"
        "create_backend(GraphBackendConfig(driver='neo4j', uri='bolt://x:1'),\n"
        "    LLMProvider(provider='openai', name='m'), E())\n"
        "loaded = sorted(m for m in sys.modules\n"
        "    if m.split('.')[0] in ('graphiti_core', 'neo4j', 'falkordb', 'kuzu'))\n"
        "print('LOADED=' + ','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == "LOADED="


# --- text closing (close_on_text_change) -------------------------------------

TEXT_QUERIES = (
    OLD_SECTION_EPISODES_QUERY,
    TEXT_CLOSURE_CANDIDATES_QUERY,
    CLOSE_FACTS_QUERY,
)
S61AA_WAITING = SectionKey("h4_part_3_61aa_waiting_period", "61AA", RATE_PATH)
OLD_61AA_WAITING = """#### 61AA Nil Part A rate during the waiting period
(1) The nil Part A rate rule applies to an individual for a day in the waiting \
period unless:
(a) the individual is the holder of a permanent visa covered by subsection (2); or
(c) parental leave pay, or dad and partner pay, under the Paid Parental Leave Act \
2010 is payable to the individual.
(2) The permanent visas are the following:
(a) a Subclass 117 (Orphan Relative) visa;
(b) a Subclass 837 (Orphan Relative) visa."""
NEW_61AA_WAITING = OLD_61AA_WAITING.replace(
    "(c) parental leave pay, or dad and partner pay, under",
    "(c) parental leave pay under",
)
DAPP_WAITING_FACT = (
    "The nil Part A rate rule does not apply for a day in the waiting period if "
    "dad and partner pay under the Paid Parental Leave Act 2010 is payable"
)
VISA_WAITING_FACT = (
    "A Subclass 117 (Orphan Relative) visa is excluded from the permanent visa "
    "category for the waiting period"
)
S57GJ = SectionKey("h4_part_8_57gj_notice", "57GJ", ("Part 8—Security",))
OLD_57GJ = """#### 57GJ Notice by Home Affairs Minister
(1) The Home Affairs Minister may give the Secretary a written notice requiring \
this Division to apply to a specified individual.
(2) The Home Affairs Minister must give a copy of the notice to the individual."""
NEW_57GJ = OLD_57GJ.replace("Home Affairs Minister", "ASIO Minister")
HOME_AFFAIRS_FACT = (
    "The Home Affairs Minister may give a written notice requiring this Division "
    "to apply to a specified individual"
)


def open_fact(
    uuid: str,
    text: str,
    episodes: list[EpisodeIn],
    *,
    group: str = NS,
    valid_at: datetime | None = JULY_2020,
) -> FakeEdge:
    return FakeEdge(
        uuid, group, [str(e.episode_id) for e in episodes], valid_at, fact=text
    )


async def text_close(
    store: FakeStore,
    before: list[EpisodeIn],
    facts: list[FakeEdge],
    new: EpisodeIn,
    *,
    close_on_text_change: bool = True,
) -> IngestReport:
    """Ingest ``before``, add open ``facts``, then ingest ``new``.

    Graphiti closes nothing here: only text closing can.
    """
    backend = make_backend(close_on_text_change=close_on_text_change)
    for episode_in in before:
        await backend.ingest(NS, [episode_in], "x")
    store.edges.extend(facts)
    return await backend.ingest(NS, [new], "x")


@pytest.mark.asyncio
async def test_text_closing_closes_a_fact_on_a_changed_line_not_one_on_a_kept_line(
    store: FakeStore, fake: FakeGraphiti, caplog: pytest.LogCaptureFixture
) -> None:
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    new = sectioned("61aa-2026", NEW_61AA_WAITING, S61AA_WAITING)

    with caplog.at_level("INFO", logger="holodeck.lib.memory.backends.graphiti"):
        report = await text_close(
            store,
            [o],
            [
                open_fact("dapp", DAPP_WAITING_FACT, [o]),
                open_fact("visa", VISA_WAITING_FACT, [o]),
            ],
            new,
        )

    assert not is_open(store, "dapp")
    assert edge(store, "dapp").invalid_at == JULY_2026
    assert edge(store, "dapp").expired_at is not None
    assert is_open(store, "visa")
    assert report.facts_closed_by_text == 1
    assert report.facts_closed == 0
    assert any("1 closed by text change" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_text_closing_closes_a_fact_of_a_renamed_minister(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("57gj-2020", OLD_57GJ, S57GJ)

    report = await text_close(
        store,
        [o],
        [open_fact("home-affairs", HOME_AFFAIRS_FACT, [o])],
        sectioned("57gj-2026", NEW_57GJ, S57GJ),
    )

    assert not is_open(store, "home-affairs")
    assert report.facts_closed_by_text == 1


@pytest.mark.asyncio
async def test_text_closing_never_closes_a_fact_the_new_section_restated(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    head_id = uuid5(NAMESPACE_URL, "61aa-2026-head")
    new_id = uuid5(NAMESPACE_URL, "61aa-2026")
    head = sectioned(
        "61aa-2026-head",
        "#### 61AA Nil Part A rate during the waiting period",
        S61AA_WAITING,
        section_ids=(head_id, new_id),
    )
    new = sectioned(
        "61aa-2026", NEW_61AA_WAITING, S61AA_WAITING, section_ids=(head_id, new_id)
    )

    report = await text_close(
        store,
        [o, head],
        [
            open_fact("by-new", DAPP_WAITING_FACT, [o, new]),
            open_fact("by-sibling", DAPP_WAITING_FACT, [o, head]),
        ],
        new,
    )

    assert is_open(store, "by-new")
    assert is_open(store, "by-sibling")
    assert report.facts_closed_by_text == 0


@pytest.mark.asyncio
async def test_text_closing_leaves_other_sections_groups_closed_and_later_facts(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    other = old(
        "3aa-2020",
        "#### 3AA Paid parental leave\n(1) dad and partner pay under the Paid "
        "Parental Leave Act 2010 is payable to the individual.",
        SectionKey("h4_part_2_3aa", "3AA", DEFINITIONS),
    )
    earlier = datetime(2021, 1, 1, tzinfo=UTC)
    already = open_fact("already-closed", DAPP_WAITING_FACT, [o])
    already.invalid_at = earlier
    already.expired_at = earlier

    report = await text_close(
        store,
        [o, other],
        [
            open_fact("dapp", DAPP_WAITING_FACT, [o]),
            open_fact("other-section", DAPP_WAITING_FACT, [other]),
            open_fact("other-group", DAPP_WAITING_FACT, [o], group=OTHER_NS),
            already,
            open_fact("from-2026", DAPP_WAITING_FACT, [o], valid_at=JULY_2026),
        ],
        sectioned("61aa-2026", NEW_61AA_WAITING, S61AA_WAITING),
    )

    assert not is_open(store, "dapp")
    assert is_open(store, "other-section")
    assert is_open(store, "other-group")
    assert is_open(store, "from-2026")
    assert edge(store, "already-closed").expired_at == earlier
    assert report.facts_closed_by_text == 1
    close = [p for q, p in store.queries if q == CLOSE_FACTS_QUERY]
    assert [(p["group_id"], p["fact_uuids"]) for p in close] == [(NS, ["dapp"])]
    assert close[0]["reference_time"] == JULY_2026


@pytest.mark.asyncio
async def test_text_closing_closes_nothing_in_an_unchanged_section(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)

    report = await text_close(
        store,
        [o],
        [
            open_fact("dapp", DAPP_WAITING_FACT, [o]),
            open_fact("visa", VISA_WAITING_FACT, [o]),
        ],
        sectioned("61aa-2026", f"{OLD_61AA_WAITING}\n", S61AA_WAITING),
    )

    assert is_open(store, "dapp")
    assert is_open(store, "visa")
    assert report.facts_closed_by_text == 0
    assert not [q for q, _ in store.queries if q == CLOSE_FACTS_QUERY]


@pytest.mark.asyncio
async def test_text_closing_keeps_a_subsection_fact_whose_line_was_reworded(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # "subsection"/"does" become "subsections"/"do", so the changed line
    # matches the fact at 3/11 against 0.625 for the old line: beyond the
    # 0.1 margin. The new line still holds every anchored word (subsection,
    # 4, not, apply) and adds no restrictive word: the fact stays open.
    section = SectionKey("h4_part_3_61ab_waiting", "61AB", RATE_PATH)
    old_text = (
        "#### 61AB Waiting period\n(1) For the waiting period:\n"
        "(a) the nil Part A rate rule applies; and\n"
        "(c) subsection (4) does not apply;"
    )
    new_text = old_text.replace(
        "(c) subsection (4) does not apply;",
        "(c) subsections (4) and (4A) do not apply;",
    )
    o = old("61ab-2020", old_text, section)

    report = await text_close(
        store,
        [o],
        [open_fact("sub4", "Subsection (4) does not apply to the waiting period", [o])],
        sectioned("61ab-2026", new_text, section),
    )

    assert is_open(store, "sub4")
    assert report.facts_closed_by_text == 0
    assert not [q for q, _ in store.queries if q == CLOSE_FACTS_QUERY]


@pytest.mark.asyncio
async def test_text_closing_off_closes_nothing_and_runs_no_query(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)

    report = await text_close(
        store,
        [o],
        [open_fact("dapp", DAPP_WAITING_FACT, [o])],
        sectioned("61aa-2026", NEW_61AA_WAITING, S61AA_WAITING),
        close_on_text_change=False,
    )

    assert is_open(store, "dapp")
    assert report.facts_closed_by_text == 0
    assert not [q for q, _ in store.queries if q in TEXT_QUERIES]


@pytest.mark.asyncio
async def test_text_closing_waits_until_every_chunk_of_the_new_section_is_written(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # The rest of the 2026 section is not in the graph yet, so a missing
    # old line may only have moved to a later chunk.
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    new_id = uuid5(NAMESPACE_URL, "61aa-2026")
    later_id = uuid5(NAMESPACE_URL, "61aa-2026-later")

    report = await text_close(
        store,
        [o],
        [open_fact("dapp", DAPP_WAITING_FACT, [o])],
        sectioned(
            "61aa-2026",
            NEW_61AA_WAITING,
            S61AA_WAITING,
            section_ids=(new_id, later_id),
        ),
    )

    assert is_open(store, "dapp")
    assert report.facts_closed_by_text == 0


@pytest.mark.asyncio
async def test_text_closing_compares_with_every_older_version(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # 2020: chunks c1 and c2. 2023 changes c1 only, so c2 keeps its 2020
    # episode. 2026 changes c2. The newest older date (2023) holds only c1;
    # the 2020 threshold fact must still be compared with 2026.
    section = SectionKey("h4_part_3_61a_rate_reduction", "61A", RATE_PATH)
    ids = {
        name: uuid5(NAMESPACE_URL, name)
        for name in ("c1-2020", "c2-2020", "c1-2023", "c2-2026")
    }
    c2020 = (ids["c1-2020"], ids["c2-2020"])
    c1_2020 = sectioned(
        "c1-2020",
        "(1) The rate is 100 dollars.",
        section,
        when=JULY_2020,
        uri=ACT_2020,
        section_ids=c2020,
    )
    c2_2020 = sectioned(
        "c2-2020",
        "(2) The threshold is 5,000 dollars.",
        section,
        when=JULY_2020,
        uri=ACT_2020,
        section_ids=c2020,
    )
    c1_2023 = sectioned(
        "c1-2023",
        "(1) The rate is 120 dollars.",
        section,
        when=datetime(2023, 7, 1, tzinfo=UTC),
        uri="corpus/act-2023.md",
        section_ids=(ids["c1-2023"], ids["c2-2020"]),
    )
    c2_2026 = sectioned(
        "c2-2026",
        "(2) The threshold is 6,000 dollars.",
        section,
        section_ids=(ids["c1-2023"], ids["c2-2026"]),
    )

    report = await text_close(
        store,
        [c1_2020, c2_2020, c1_2023],
        [
            open_fact("threshold", "The threshold is 5,000 dollars", [c2_2020]),
            open_fact(
                "rate-120",
                "The rate is 120 dollars",
                [c1_2023],
                valid_at=datetime(2023, 7, 1, tzinfo=UTC),
            ),
        ],
        c2_2026,
    )

    assert not is_open(store, "threshold")
    assert edge(store, "threshold").invalid_at == JULY_2026
    assert is_open(store, "rate-120")
    assert report.facts_closed_by_text == 1
    candidates = [p for q, p in store.queries if q == TEXT_CLOSURE_CANDIDATES_QUERY]
    assert sorted(candidates[-1]["old_uuids"]) == sorted(
        [str(ids["c1-2020"]), str(ids["c2-2020"])]
    )


@pytest.mark.parametrize(
    ("tool", "old_key", "new_key"),
    [
        # Same tool, a series_id on each: different documents.
        ("family_law", "employees-handbook", "supplier-terms"),
        # Another tool, no series_id.
        ("supplier_docs", "", ""),
    ],
)
@pytest.mark.asyncio
async def test_text_closing_skips_an_unrelated_document_with_the_same_heading(
    store: FakeStore, fake: FakeGraphiti, tool: str, old_key: str, new_key: str
) -> None:
    # employees.md and suppliers.md both have "# 3 Benefits", so the
    # chunker gives both the same section id.
    benefits = SectionKey("h1_3_benefits", "3", ())
    employees = sectioned(
        "employees-3",
        "# 3 Benefits\nEmployees receive 20 days of annual leave.",
        benefits,
        when=JULY_2020,
        uri="corpus/employees.md",
        key=old_key,
    )
    suppliers = sectioned(
        "suppliers-3",
        "# 3 Benefits\nSuppliers receive payment within 30 days.",
        benefits,
        uri="corpus/suppliers.md",
        tool=tool,
        key=new_key,
    )

    report = await text_close(
        store,
        [employees],
        [open_fact("leave", "Employees receive 20 days of annual leave", [employees])],
        suppliers,
    )

    assert is_open(store, "leave")
    assert report.facts_closed_by_text == 0


@pytest.mark.asyncio
async def test_text_closing_compares_two_tools_that_share_a_series_id(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # One tool per compilation: the series_id says they are versions.
    o = sectioned(
        "61aa-2020",
        OLD_61AA_WAITING,
        S61AA_WAITING,
        when=JULY_2020,
        uri=ACT_2020,
        tool="act_2020",
        key="family-assistance-act",
    )
    new = sectioned(
        "61aa-2026",
        NEW_61AA_WAITING,
        S61AA_WAITING,
        tool="act_2026",
        key="family-assistance-act",
    )

    report = await text_close(
        store, [o], [open_fact("dapp", DAPP_WAITING_FACT, [o])], new
    )

    assert not is_open(store, "dapp")
    assert report.facts_closed_by_text == 1


@pytest.mark.asyncio
async def test_text_closing_leaves_a_fact_that_a_later_version_states(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # 2026 was ingested before 2023. The 2023 text drops the line, but the
    # 2026 text states the fact again: it holds after 2023.
    o2020 = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    o2026 = sectioned("61aa-2026", OLD_61AA_WAITING + "\n", S61AA_WAITING)
    late = sectioned(
        "61aa-2023",
        NEW_61AA_WAITING,
        S61AA_WAITING,
        when=datetime(2023, 7, 1, tzinfo=UTC),
        uri="corpus/family-assistance-2023.md",
    )

    report = await text_close(
        store,
        [o2020, o2026],
        [open_fact("dapp", DAPP_WAITING_FACT, [o2020, o2026])],
        late,
    )

    assert is_open(store, "dapp")
    assert report.facts_closed_by_text == 0


@pytest.mark.asyncio
async def test_text_closing_waits_while_a_sibling_chunk_is_pending(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # The other chunk of the 2026 section was saved pending and failed: its
    # facts may not be in the graph, so nothing is closed yet.
    o = old("61aa-2020", OLD_61AA_WAITING, S61AA_WAITING)
    sibling_id = uuid5(NAMESPACE_URL, "61aa-2026-sibling")
    new_id = uuid5(NAMESPACE_URL, "61aa-2026")
    sibling = sectioned(
        "61aa-2026-sibling",
        "(3) Another paragraph.",
        S61AA_WAITING,
        section_ids=(sibling_id, new_id),
    )
    backend = make_backend(close_on_text_change=True)
    await backend.ingest(NS, [o], "x")
    fake.fail_on[str(sibling_id)] = RuntimeError("extraction failed")
    await backend.ingest(NS, [sibling], "x")
    assert state(store, sibling) == "pending"
    store.edges.append(open_fact("dapp", DAPP_WAITING_FACT, [o]))

    report = await backend.ingest(
        NS,
        [
            sectioned(
                "61aa-2026",
                NEW_61AA_WAITING,
                S61AA_WAITING,
                section_ids=(sibling_id, new_id),
            )
        ],
        "x",
    )

    assert is_open(store, "dapp")
    assert report.facts_closed_by_text == 0


# --- review fixes: attribution, retries, dating guard, order, lineage --------


@pytest.mark.asyncio
async def test_closure_ignores_a_closure_the_episode_did_not_make(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # Another writer closes a fact of another section during this episode's
    # window. It is not on this episode's entity_edges, so it is neither
    # counted nor reopened (rule a would reopen it as "other section").
    s3aa = SectionKey("h4_part_2_3aa_paid_parental_leave", "3AA", DEFINITIONS)
    elsewhere = old("3aa-2020", "#### 3AA Paid parental leave\n(1) Old text.", s3aa)
    o = old("61aa-2020", OLD_61AA, S61AA)
    backend = make_backend()
    await backend.ingest(NS, [elsewhere], "x")
    store.edges.append(
        FakeEdge(
            "concurrent",
            NS,
            [str(elsewhere.episode_id)],
            JULY_2020,
            fact="Old text",
        )
    )
    new = sectioned("61aa-2026", NEW_61AA, S61AA)
    fake.concurrent_closes[str(new.episode_id)] = ["concurrent"]

    report = await close_with(store, fake, [o], {"dapp": (DAPP_FACT, o)}, new, backend)

    assert not is_open(store, "concurrent")
    assert not is_open(store, "dapp")
    assert (report.facts_closed, report.facts_reopened) == (1, 0)


@pytest.mark.asyncio
async def test_closure_a_retry_checks_the_closures_of_the_failed_attempt(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # Attempt 1 closes both facts, then dating fails: the episode stays
    # pending. Attempt 2 closes nothing new and replaces entity_edges; the
    # check still covers attempt 1, so the fact on the kept line reopens.
    backend = make_backend()
    o = old("61aa-2020", OLD_61AA, S61AA)
    new = sectioned("61aa-2026", NEW_61AA, S61AA)
    store.fail_date = RuntimeError("dating rejected")

    first = await close_with(
        store,
        fake,
        [o],
        {"visa": (VISA_FACT, o), "dapp": (DAPP_FACT, o)},
        new,
        backend,
    )
    assert first.failed == new.episode_id
    assert not is_open(store, "visa")
    pending = store.nodes[str(new.episode_id)]
    created = pending.created_at
    store.fail_date = None
    fake.closes.clear()
    fake.new_edges.clear()

    retry = await backend.ingest(NS, [new], "x")

    assert retry.written == (new.episode_id,)
    assert is_open(store, "visa")
    assert not is_open(store, "dapp")
    counts = (retry.facts_closed, retry.facts_reopened, retry.facts_kept_closed)
    assert counts == (2, 1, 1)
    windows = [p for q, p in store.queries if q == CLOSED_SINCE_QUERY]
    assert windows[-1]["since"] == created
    kept = [p for q, p in store.queries if q == KEEP_EPISODE_EDGES_QUERY]
    assert set(kept[-1]["edges"]) >= {"visa", "dapp"}


@pytest.mark.parametrize("restated_in_2026", [True, False])
@pytest.mark.asyncio
async def test_closure_reopens_a_fact_that_a_later_episode_states(
    store: FakeStore, fake: FakeGraphiti, restated_in_2026: bool
) -> None:
    # 2026 was ingested before 2023. The 2023 text changed the line and the
    # model closed the fact at 2023. When 2026 states the fact again, the
    # fact still holds after 2023: the closure is undone.
    backend = make_backend()
    o2020 = old("61aa-2020", OLD_61AA, S61AA)
    o2026 = sectioned("61aa-2026", OLD_61AA + "\n", S61AA)
    for episode_in in (o2020, o2026):
        await backend.ingest(NS, [episode_in], "x")
    sources = [str(o2020.episode_id)]
    if restated_in_2026:
        sources.append(str(o2026.episode_id))
    store.edges.append(
        FakeEdge("dapp", NS, sources, JULY_2020, fact=DAPP_FACT, source="A", target="B")
    )
    late = sectioned(
        "61aa-2023",
        NEW_61AA,
        S61AA,
        when=datetime(2023, 7, 1, tzinfo=UTC),
        uri="corpus/family-assistance-2023.md",
    )
    late_id = str(late.episode_id)
    fake.new_edges[late_id] = [
        FakeEdge("dapp-next", NS, [late_id], fact=DAPP_FACT, source="A", target="B")
    ]
    fake.closes[late_id] = ["dapp"]

    report = await backend.ingest(NS, [late], "x")

    assert is_open(store, "dapp") is restated_in_2026
    assert report.facts_reopened == (1 if restated_in_2026 else 0)


@pytest.mark.parametrize(
    ("tool", "old_key", "new_key", "reopened"),
    [
        ("family_law", "", "", False),
        ("family_law", "act-a", "act-a", False),
        ("other_tool", "act-a", "act-a", False),
        ("family_law", "act-a", "act-b", True),
        ("other_tool", "", "act-a", True),
    ],
)
@pytest.mark.asyncio
async def test_closure_reopens_a_fact_from_another_document_lineage(
    store: FakeStore,
    fake: FakeGraphiti,
    tool: str,
    old_key: str,
    new_key: str,
    reopened: bool,
) -> None:
    o = sectioned(
        "61aa-2020", OLD_61AA, S61AA, when=JULY_2020, uri=ACT_2020, key=old_key
    )
    new = sectioned("61aa-2026", NEW_61AA, S61AA, tool=tool, key=new_key)

    await close_with(store, fake, [o], {"dapp": (DAPP_FACT, o)}, new)

    assert is_open(store, "dapp") is reopened


@pytest.mark.asyncio
async def test_fix_a_a_fact_that_ends_before_its_start_stays_undated(
    store: FakeStore, fake: FakeGraphiti, caplog: pytest.LogCaptureFixture
) -> None:
    uid = str(C2026.episode_id)
    ends = datetime(2025, 1, 1, tzinfo=UTC)
    fake.new_edges[uid] = [
        FakeEdge("ends-before", NS, [uid], invalid_at=ends),
        FakeEdge("ends-on-start", NS, [uid], invalid_at=JULY_2026),
        FakeEdge("ends-after", NS, [uid], invalid_at=datetime(2027, 1, 1, tzinfo=UTC)),
    ]

    with caplog.at_level("WARNING", logger="holodeck.lib.memory.backends.graphiti"):
        report = await make_backend().ingest(NS, [C2026], "x")

    assert report.written == (C2026.episode_id,)
    assert edge(store, "ends-before").valid_at is None
    assert edge(store, "ends-on-start").valid_at is None
    assert edge(store, "ends-after").valid_at == JULY_2026
    assert any("2 facts left undated" in r.getMessage() for r in caplog.records)
    assert "r.invalid_at > first_seen" in DATE_EPISODE_FACTS_QUERY


@pytest.mark.asyncio
async def test_fix_a_backfill_leaves_a_fact_that_ends_before_its_start_undated(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    backend = make_backend()
    await backend.ingest(NS, [C2020, C2026], "x")
    new = str(C2026.episode_id)
    store.edges.extend(
        [
            FakeEdge("inverted", NS, [new], invalid_at=JULY_2020),
            FakeEdge("fits", NS, [new], invalid_at=datetime(2027, 1, 1, tzinfo=UTC)),
        ]
    )

    assert await backend.backfill_fact_start_dates(NS) == 1

    assert edge(store, "inverted").valid_at is None
    assert edge(store, "fits").valid_at == JULY_2026
    assert await backend.backfill_fact_start_dates(NS) == 0
    assert "r.invalid_at > first_seen" in BACKFILL_FACT_START_QUERY


def test_document_key_round_trips_through_the_name_and_is_optional() -> None:
    section = SectionKey("h1_3_benefits", "3", ())
    keyed = sectioned("keyed", "text", section, key="employees-handbook")
    plain = sectioned("plain", "text", section)

    decoded = decode_episode_name(encode_episode_name("complete", keyed))
    assert decoded is not None
    assert decoded.document_key == "employees-handbook"
    assert "document_key" not in encode_episode_name("complete", plain)
    plain_name = decode_episode_name(encode_episode_name("complete", plain))
    assert plain_name is not None
    assert plain_name.document_key == ""


@pytest.mark.asyncio
async def test_closure_equal_positional_ids_of_other_sections_are_other_sections(
    store: FakeStore, fake: FakeGraphiti
) -> None:
    # Both chunks are the 12th of their document, so the chunker gave both
    # "chunk_12"; their numbers differ, so they are different sections.
    o = old("61aa-2020", OLD_61AA, SectionKey("chunk_12", "61AA", RATE_PATH))
    new = sectioned("61ab-2026", NEW_61AA, SectionKey("chunk_12", "61AB", RATE_PATH))

    await close_with(store, fake, [o], {"dapp": (DAPP_FACT, o)}, new)

    assert is_open(store, "dapp")
