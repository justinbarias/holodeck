"""Graphiti backend, write side (spec 043, design seam 3, task T7).

Graphiti is replaced by an in-memory fake: no database, no model, no network.
The fake keeps episode nodes the way Graphiti stores them (one ``name``
field), so the tests read the lifecycle state the backend writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance import graph_data_operations
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, SecretStr

from holodeck.lib.memory import (
    EpisodeIn,
    MemoryBackend,
    MemoryBackendUnavailableError,
    MemoryConfigError,
    MemoryNotSupportedError,
    MemoryOperation,
    MemoryQuery,
)
from holodeck.lib.memory.backends import graphiti_clients
from holodeck.lib.memory.backends.graphiti import (
    COMPLETE_PREFIX,
    GraphitiBackend,
    decode_episode_name,
    encode_episode_name,
    is_complete,
    source_sentence,
)
from holodeck.lib.memory.registry import create_backend
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig

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
class FakeStore:
    nodes: dict[str, EpisodicNode] = field(default_factory=dict)
    events: list[tuple[str, str, str]] = field(default_factory=list)
    queries: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail_load: BaseException | None = None
    fail_save_complete: BaseException | None = None
    cleared: list[list[str] | None] = field(default_factory=list)


class FakeDriver:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.clones: list[str] = []

    def clone(self, database: str) -> FakeDriver:
        self.clones.append(database)
        return self

    async def execute_query(self, query: str, **params: Any) -> tuple[Any, ...]:
        self.store.queries.append((query, params))
        if "uuid" in params:
            return self._mark_complete(params)
        count = sum(
            1
            for node in self.store.nodes.values()
            if node.group_id == params["group_id"]
            and node.name.startswith(params["prefix"])
        )
        return ([{"episode_count": count}], None, None)

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
        node.entity_edges = ["edge-1"]

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


def make_backend(entity_types: tuple[str, ...] = ()) -> GraphitiBackend:
    config = GraphBackendConfig(
        driver="neo4j",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=SecretStr(PASSWORD),
    )
    model = LLMProvider(
        provider=ProviderEnum.OPENAI, name="gpt-test", api_key=SecretStr(API_KEY)
    )
    return GraphitiBackend(config, model, _NoEmbeddings(), entity_types)


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


# --- errors -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_mb_03_read_side_is_not_supported_yet(fake: FakeGraphiti) -> None:
    backend = make_backend()
    query = MemoryQuery(text="rate", valid_at=WHEN, known_at=WHEN)

    assert backend.capabilities().supported == frozenset(
        {MemoryOperation.INGEST, MemoryOperation.PURGE}
    )
    with pytest.raises(MemoryNotSupportedError) as search:
        await backend.search(NS, query)
    with pytest.raises(MemoryNotSupportedError) as timeline:
        await backend.timeline(NS, "rate")
    with pytest.raises(MemoryNotSupportedError) as provenance:
        await backend.provenance(NS, "fact-1")

    assert search.value.operation == MemoryOperation.SEARCH.value
    assert timeline.value.operation == MemoryOperation.TIMELINE.value
    assert provenance.value.operation == MemoryOperation.PROVENANCE.value
    assert fake.builds == []  # type: ignore[attr-defined]


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

    backend = create_backend(config, model, _NoEmbeddings(), ("Payment",))

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
