"""Graphiti backend, read side (spec 043, design seam 5, task T11).

Graphiti is replaced by an in-memory fake. Its ``search_`` evaluates the
``SearchFilters`` the backend builds the way Graphiti's Cypher does: fields
joined by AND, groups inside a field joined by OR, a dated comparison never
matching null. It also fails on two dated groups in one field, which
collide on one Cypher parameter name in Graphiti 0.31.2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search import search_utils
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from neo4j.exceptions import ServiceUnavailable
from pydantic import SecretStr

from holodeck.lib.memory import (
    EpisodeIn,
    MemoryBackendUnavailableError,
    MemoryNotSupportedError,
    MemoryOperation,
    MemoryQuery,
    TemporalMemoryError,
)
from holodeck.lib.memory.backends import graphiti_clients
from holodeck.lib.memory.backends.graphiti import (
    GraphitiBackend,
    encode_episode_name,
)
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import GraphBackendConfig
from tests.unit.lib.memory.test_entity_match import CASES as ENTITY_CASES
from tests.unit.lib.memory.test_graphiti_backend_write import (
    API_KEY,
    NS,
    OTHER_NS,
    PASSWORD,
    FakeDriver,
    FakeGraphiti,
    FakeStore,
    _NoEmbeddings,
    store,
)

pytestmark = [pytest.mark.unit]

__all__ = ["store"]  # the write tests' fixture, reused

NOW = datetime(2025, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 1, tzinfo=UTC)  # first compilation ingested
T2 = datetime(2024, 6, 1, tzinfo=UTC)  # second compilation ingested
BETWEEN = datetime(2024, 3, 1, tzinfo=UTC)
A_FROM = datetime(2019, 7, 1, tzinfo=UTC)
A_TO = datetime(2020, 7, 1, tzinfo=UTC)
V_2020 = datetime(2020, 1, 1, tzinfo=UTC)
V_2021 = datetime(2021, 1, 1, tzinfo=UTC)


def make_episode(name: str, when: datetime) -> EpisodeIn:
    return EpisodeIn(
        episode_id=uuid5(NAMESPACE_URL, f"read-{name}"),
        source_tool="family_law",
        chunk_id=f"{name}_chunk_0",
        source_uri=f"corpus/{name}.md",
        text=f"Passage {name}.",
        reference_time=when,
        reference_time_source="front_matter",
    )


EP_A = make_episode("a", A_FROM)
EP_B = make_episode("b", A_TO)
EP_PENDING = make_episode("pending", A_TO)
EP_OTHER = make_episode("other", A_TO)
FOREIGN_EPISODE = "foreign-episode"

ENTITY_RATE = "entity-rate"
ENTITY_AGENCY = "entity-agency"
ENTITY_MINISTER = "entity-minister"
ENTITY_OFFICE = "entity-office"
ENTITY_ELSEWHERE = "entity-elsewhere"


# --- fakes ------------------------------------------------------------------


@dataclass
class Graph:
    """Entities, facts and mentions behind the fake Graphiti."""

    edges: dict[str, EntityEdge] = field(default_factory=dict)
    entities: dict[str, EntityNode] = field(default_factory=dict)
    mentions: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[tuple[str, str], float] = field(default_factory=dict)
    searches: list[dict[str, Any]] = field(default_factory=list)
    entity_queries: list[dict[str, Any]] = field(default_factory=list)
    fail_search: BaseException | None = None
    fail_edges: BaseException | None = None


def _compare(value: datetime | None, date_filter: DateFilter) -> bool:
    op = date_filter.comparison_operator
    if op is ComparisonOperator.is_null:
        return value is None
    if op is ComparisonOperator.is_not_null:
        return value is not None
    instant = date_filter.date
    if value is None or instant is None:
        return False  # Cypher: a comparison with null is never true
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)  # a naive stored value is UTC
    return {
        ComparisonOperator.equals: value == instant,
        ComparisonOperator.not_equals: value != instant,
        ComparisonOperator.greater_than: value > instant,
        ComparisonOperator.less_than: value < instant,
        ComparisonOperator.greater_than_equal: value >= instant,
        ComparisonOperator.less_than_equal: value <= instant,
    }[op]


def _field_passes(
    value: datetime | None, groups: list[list[DateFilter]] | None
) -> bool:
    if groups is None:
        return True
    dated_per_position: Counter[int] = Counter()
    for group in groups:
        for position, date_filter in enumerate(group):
            if date_filter.comparison_operator not in (
                ComparisonOperator.is_null,
                ComparisonOperator.is_not_null,
            ):
                dated_per_position[position] += 1
    # Graphiti names the parameter after the position in the group only.
    assert all(
        count <= 1 for count in dated_per_position.values()
    ), "two dated groups in one field share a Cypher parameter"
    return any(all(_compare(value, f) for f in group) for group in groups)


def passes(edge: EntityEdge, filters: SearchFilters) -> bool:
    return all(
        _field_passes(value, groups)
        for value, groups in (
            (edge.valid_at, filters.valid_at),
            (edge.invalid_at, filters.invalid_at),
            (edge.created_at, filters.created_at),
            (edge.expired_at, filters.expired_at),
        )
    )


class ReadDriver(FakeDriver):
    def __init__(self, store: FakeStore, graph: Graph) -> None:
        super().__init__(store)
        self.graph = graph
        self.fail_query: BaseException | None = None

    def clone(self, database: str) -> ReadDriver:
        self.clones.append(database)
        return self

    async def execute_query(self, query: str, **params: Any) -> tuple[Any, ...]:
        if "entity_ref" not in params:
            return await super().execute_query(query, **params)
        if self.fail_query is not None:
            raise self.fail_query
        self.graph.entity_queries.append(params)
        ref = params["entity_ref"]
        # FIND_ENTITIES_QUERY: a uuid, or a name the reference contains in
        # any case. Only a pre-filter; the backend decides.
        records = [
            {"uuid": node.uuid, "name": node.name}
            for node in self.graph.entities.values()
            if node.group_id == params["group_id"]
            and (node.uuid == ref or (node.name and node.name.lower() in ref.lower()))
        ]
        return (records, None, None)


class ReadGraphiti(FakeGraphiti):
    def __init__(self, store: FakeStore, graph: Graph) -> None:
        super().__init__(store)
        self.graph = graph
        self.read_driver = ReadDriver(store, graph)
        self.driver = self.read_driver

    async def search_(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
        driver: object | None = None,
        query_vector: list[float] | None = None,
    ) -> SearchResults:
        assert search_filter is not None
        assert config.edge_config is not None
        self.graph.searches.append(
            {
                "query": query,
                "config": config,
                "group_ids": group_ids,
                "origins": bfs_origin_node_uuids,
                "filter": search_filter,
                "driver": driver,
            }
        )
        if self.graph.fail_search is not None:
            raise self.graph.fail_search
        in_group = [
            e
            for e in self.graph.edges.values()
            if group_ids is None or e.group_id in group_ids
        ]
        words = set(query.lower().split())
        text_hits = [e for e in in_group if words & set(e.fact.lower().split())]
        origins = (
            bfs_origin_node_uuids
            if bfs_origin_node_uuids is not None
            else [e.source_node_uuid for e in text_hits]
        )
        reached = self._bfs(in_group, origins, config.edge_config.bfs_max_depth)
        candidates = {e.uuid: e for e in [*text_hits, *reached]}
        kind = "closed" if search_filter.expired_at is not None else "open"
        kept = [e for e in candidates.values() if passes(e, search_filter)]
        scored = sorted(
            ((e, self.graph.scores.get((e.uuid, kind), 1.0)) for e in kept),
            key=lambda pair: (-pair[1], pair[0].uuid),
        )[: config.limit]
        return SearchResults(
            edges=[e for e, _ in scored], edge_reranker_scores=[s for _, s in scored]
        )

    @staticmethod
    def _bfs(
        edges: list[EntityEdge], origins: list[str], depth: int
    ) -> list[EntityEdge]:
        """Outgoing edges only, up to ``depth`` entity-to-entity edges."""
        reached: list[EntityEdge] = []
        frontier = set(origins)
        for _ in range(depth):
            step = [e for e in edges if e.source_node_uuid in frontier]
            reached.extend(step)
            frontier = {e.target_node_uuid for e in step}
        return reached


@pytest.fixture
def graph(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> Graph:
    data = Graph()

    async def get_by_uuids(
        cls: type[EntityEdge], driver: object, uuids: list[str]
    ) -> list[EntityEdge]:
        if data.fail_edges is not None:
            raise data.fail_edges
        return [data.edges[u] for u in uuids if u in data.edges]

    async def get_by_node_uuid(
        cls: type[EntityEdge], driver: object, node_uuid: str
    ) -> list[EntityEdge]:
        if data.fail_edges is not None:
            raise data.fail_edges
        # Undirected, as Graphiti's Cypher; a self-loop would come twice.
        return [
            e
            for e in data.edges.values()
            if node_uuid in (e.source_node_uuid, e.target_node_uuid)
        ]

    async def get_mentioned_nodes(
        driver: object, episodes: list[EpisodicNode]
    ) -> list[EntityNode]:
        return [
            data.entities[uuid]
            for episode in episodes
            for uuid in data.mentions.get(episode.uuid, [])
        ]

    monkeypatch.setattr(EntityEdge, "get_by_uuids", classmethod(get_by_uuids))
    monkeypatch.setattr(EntityEdge, "get_by_node_uuid", classmethod(get_by_node_uuid))
    monkeypatch.setattr(search_utils, "get_mentioned_nodes", get_mentioned_nodes)
    seed(store, data)
    return data


@pytest.fixture
def fake(
    store: FakeStore, graph: Graph, monkeypatch: pytest.MonkeyPatch
) -> ReadGraphiti:
    graphiti = ReadGraphiti(store, graph)

    def build_graphiti(*args: object) -> graphiti_clients.GraphitiRuntime:
        return graphiti_clients.GraphitiRuntime(
            graphiti=graphiti,  # type: ignore[arg-type]
            llm=graphiti.llm_client,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(graphiti_clients, "build_graphiti", build_graphiti)
    return graphiti


def make_backend() -> GraphitiBackend:
    config = GraphBackendConfig(
        driver="neo4j",
        uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password=SecretStr(PASSWORD),
    )
    model = LLMProvider(
        provider=ProviderEnum.OPENAI,
        name="gpt-test",
        api_key=SecretStr(API_KEY),
        endpoint=None,
    )
    return GraphitiBackend(config, model, _NoEmbeddings())


# --- data -------------------------------------------------------------------


def put_episode(
    store: FakeStore,
    ep: EpisodeIn,
    *,
    namespace: str = NS,
    complete: bool = True,
) -> None:
    store.nodes[str(ep.episode_id)] = EpisodicNode(
        uuid=str(ep.episode_id),
        name=encode_episode_name("complete" if complete else "pending", ep),
        group_id=namespace,
        labels=[],
        source=EpisodeType.text,
        source_description="Passage.",
        content=ep.text,
        created_at=T1,
        valid_at=ep.reference_time,
    )


def put_edge(
    graph: Graph,
    uuid: str,
    fact: str,
    *,
    source: str = ENTITY_RATE,
    target: str = ENTITY_AGENCY,
    valid_at: datetime | None,
    invalid_at: datetime | None = None,
    created_at: datetime = T1,
    expired_at: datetime | None = None,
    episodes: tuple[str, ...] = (),
    namespace: str = NS,
) -> None:
    graph.edges[uuid] = EntityEdge(
        uuid=uuid,
        group_id=namespace,
        source_node_uuid=source,
        target_node_uuid=target,
        name="RELATES",
        fact=fact,
        episodes=list(episodes),
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


def put_entity(graph: Graph, uuid: str, name: str, namespace: str = NS) -> None:
    graph.entities[uuid] = EntityNode(uuid=uuid, name=name, group_id=namespace)


def seed(store: FakeStore, graph: Graph) -> None:
    """Two compilations: rate A (100) closed at 2020-07-01 by B (120) at T2."""
    put_episode(store, EP_A)
    put_episode(store, EP_B)
    put_episode(store, EP_PENDING, complete=False)
    put_episode(store, EP_OTHER, namespace=OTHER_NS)
    store.nodes[FOREIGN_EPISODE] = EpisodicNode(
        uuid=FOREIGN_EPISODE,
        name="not ours",
        group_id=NS,
        labels=[],
        source=EpisodeType.text,
        source_description="x",
        content="x",
        created_at=T1,
        valid_at=T1,
    )
    put_entity(graph, ENTITY_RATE, "Rate A")
    put_entity(graph, ENTITY_AGENCY, "Agency B")
    put_entity(graph, ENTITY_MINISTER, "Minister")
    put_entity(graph, ENTITY_OFFICE, "Office")
    put_entity(graph, ENTITY_ELSEWHERE, "Rate A", namespace=OTHER_NS)
    put_edge(
        graph,
        "fact-a",
        "Rate A is 100 dollars",
        valid_at=A_FROM,
        invalid_at=A_TO,
        created_at=T1,
        expired_at=T2,
        episodes=(
            str(EP_A.episode_id),
            str(EP_PENDING.episode_id),
            str(EP_OTHER.episode_id),
            FOREIGN_EPISODE,
            str(EP_A.episode_id),
        ),
    )
    put_edge(
        graph,
        "fact-b",
        "Rate A is 120 dollars",
        valid_at=A_TO,
        created_at=T2,
        episodes=(str(EP_B.episode_id),),
    )
    put_edge(
        graph,
        "fact-null",
        "Agency B is administered by the Minister",
        source=ENTITY_AGENCY,
        target=ENTITY_MINISTER,
        valid_at=None,
        created_at=T1,
        episodes=(str(EP_A.episode_id),),
    )
    put_edge(
        graph,
        "fact-far",
        "The Minister reports to the Office",
        source=ENTITY_MINISTER,
        target=ENTITY_OFFICE,
        valid_at=A_FROM,
        created_at=T1,
    )
    put_edge(
        graph,
        "fact-elsewhere",
        "Rate A is 100 dollars",
        source=ENTITY_ELSEWHERE,
        target=ENTITY_ELSEWHERE,
        valid_at=A_FROM,
        created_at=T1,
        episodes=(str(EP_OTHER.episode_id),),
        namespace=OTHER_NS,
    )


def ids(hits: list[Any]) -> list[str]:
    return [hit.fact_id for hit in hits]


# --- RET-03: valid_at and known_at -----------------------------------------


@pytest.mark.asyncio
async def test_ret_03_valid_at_before_the_change_returns_the_older_fact_only(
    fake: ReadGraphiti, graph: Graph
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2020, known_at=NOW)
    )

    assert ids(hits) == ["fact-a"]
    assert hits[0].valid_from == A_FROM
    assert hits[0].valid_to == A_TO
    assert hits[0].recorded_at == T1
    assert hits[0].expired_at == T2
    assert hits[0].superseded_by is None
    # K is the present, and both passes still run: OPEN, then CLOSED_LATER.
    assert [s["filter"].expired_at is not None for s in graph.searches] == [
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_ret_03_valid_at_after_the_change_returns_the_newer_fact_only(
    fake: ReadGraphiti, graph: Graph
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2021, known_at=NOW)
    )

    assert ids(hits) == ["fact-b"]
    assert hits[0].valid_from == A_TO
    assert hits[0].valid_to is None
    assert hits[0].expired_at is None


@pytest.mark.asyncio
async def test_ret_03_between_two_compilations_the_older_fact_is_open(
    fake: ReadGraphiti, graph: Graph
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2021, known_at=BETWEEN)
    )

    assert ids(hits) == ["fact-a"]
    assert hits[0].valid_from == A_FROM
    assert hits[0].valid_to is None
    assert hits[0].expired_at is None
    assert hits[0].recorded_at == T1
    assert len(graph.searches) == 2
    open_filter, closed_filter = (s["filter"] for s in graph.searches)
    assert open_filter.expired_at is None
    assert closed_filter.invalid_at is None


@pytest.mark.asyncio
async def test_ret_03_a_known_at_before_the_first_ingest_returns_nothing(
    fake: ReadGraphiti,
) -> None:
    hits = await make_backend().search(
        NS,
        MemoryQuery(text="dollars", valid_at=V_2021, known_at=T1 - timedelta(days=1)),
    )

    assert hits == []


@pytest.mark.asyncio
async def test_ret_03_a_fact_closed_just_after_known_at_is_returned_open(
    fake: ReadGraphiti, graph: Graph
) -> None:
    # Closed 10 seconds before "now", and K is 30 seconds before "now": the
    # rule admits the fact as open however close K is to the present.
    put_edge(
        graph,
        "fact-just-closed",
        "Rate Z is 5 dollars",
        valid_at=A_FROM,
        invalid_at=A_TO,
        created_at=T1,
        expired_at=NOW - timedelta(seconds=10),
    )

    hits = await make_backend().search(
        NS,
        MemoryQuery(
            text="dollars", valid_at=V_2021, known_at=NOW - timedelta(seconds=30)
        ),
    )

    just_closed = [hit for hit in hits if hit.fact_id == "fact-just-closed"]
    assert len(just_closed) == 1
    assert just_closed[0].valid_to is None
    assert just_closed[0].expired_at is None
    assert len(graph.searches) == 2


@pytest.mark.asyncio
async def test_ret_03_a_fact_with_no_valid_at_is_returned_by_the_null_group(
    fake: ReadGraphiti,
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="administered", valid_at=V_2021, known_at=NOW)
    )

    assert ids(hits) == ["fact-null"]
    assert hits[0].valid_from is None


@pytest.mark.asyncio
async def test_ret_03_filters_hold_the_known_at_rule_in_each_field(
    fake: ReadGraphiti, graph: Graph
) -> None:
    await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2021, known_at=BETWEEN)
    )
    open_filter, closed_filter = (s["filter"] for s in graph.searches)
    le, gt, null = (
        ComparisonOperator.less_than_equal,
        ComparisonOperator.greater_than,
        ComparisonOperator.is_null,
    )

    def shape(groups: list[list[DateFilter]] | None) -> list[list[tuple[Any, ...]]]:
        assert groups is not None
        return [[(f.comparison_operator, f.date) for f in group] for group in groups]

    assert shape(open_filter.created_at) == [[(le, BETWEEN)]]
    assert shape(open_filter.valid_at) == [[(le, V_2021)], [(null, None)]]
    assert shape(open_filter.invalid_at) == [[(gt, V_2021)], [(null, None)]]
    assert shape(closed_filter.created_at) == [[(le, BETWEEN)]]
    assert shape(closed_filter.valid_at) == [[(le, V_2021)], [(null, None)]]
    assert shape(closed_filter.expired_at) == [[(gt, BETWEEN)]]


@pytest.mark.asyncio
async def test_ret_03_an_edge_in_both_searches_is_returned_once_with_its_best_score(
    fake: ReadGraphiti, graph: Graph
) -> None:
    # V before A closed and K before the closing: A matches OPEN and
    # CLOSED_LATER. CLOSED_LATER scores it higher here.
    graph.scores[("fact-a", "open")] = 0.5
    graph.scores[("fact-a", "closed")] = 2.0

    hits = await make_backend().search(
        NS, MemoryQuery(text="rate", valid_at=V_2020, known_at=BETWEEN)
    )

    assert len(graph.searches) == 2
    assert ids(hits) == ["fact-a"]
    assert [hit.score for hit in hits] == [2.0]
    assert hits[0].valid_to is None and hits[0].expired_at is None


@pytest.mark.asyncio
async def test_ret_03_a_returned_edge_that_breaks_the_rule_is_dropped(
    fake: ReadGraphiti, monkeypatch: pytest.MonkeyPatch, graph: Graph
) -> None:
    # A driver with its own search interface may ignore the filters.
    async def unfiltered(*args: Any, **kwargs: Any) -> SearchResults:
        edges = list(graph.edges.values())
        return SearchResults(edges=edges, edge_reranker_scores=[1.0] * len(edges))

    monkeypatch.setattr(fake, "search_", unfiltered)

    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2021, known_at=NOW)
    )

    assert sorted(ids(hits)) == ["fact-b", "fact-far", "fact-null"]


# The known-at rule over every null / non-null combination. V is 2021-01-01;
# K is BETWEEN (a past instant) or NOW (the present). Each row is one edge
# and what the search shows for it at each K: None when it is not returned,
# else the (valid_to, expired_at) pair the hit carries. (None, None) is the
# fact shown open.
LATE_START = datetime(2022, 1, 1, tzinfo=UTC)  # starts after V
LATE_END = datetime(2023, 1, 1, tzinfo=UTC)  # ends after V
EARLY_EXP = datetime(2024, 2, 1, tzinfo=UTC)  # closed before BETWEEN
LATE_EXP = datetime(2024, 6, 1, tzinfo=UTC)  # closed after BETWEEN, before NOW
AFTER_BETWEEN = datetime(2024, 4, 1, tzinfo=UTC)
OPEN = (None, None)


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None)


Shown = tuple[datetime | None, datetime | None] | None
KNOWN_AT_ROWS: list[
    tuple[
        str, datetime | None, datetime | None, datetime | None, datetime, Shown, Shown
    ]
] = [
    # name, valid_at, invalid_at, expired_at, created_at, at BETWEEN, at NOW
    ("undated-open", None, None, None, T1, OPEN, OPEN),
    ("dated-open", A_FROM, None, None, T1, OPEN, OPEN),
    ("undated-expired-early", None, None, EARLY_EXP, T1, (None, EARLY_EXP), (None, EARLY_EXP)),  # noqa: E501
    ("dated-expired-early", A_FROM, None, EARLY_EXP, T1, (None, EARLY_EXP), (None, EARLY_EXP)),  # noqa: E501
    ("undated-expired-late", None, None, LATE_EXP, T1, OPEN, (None, LATE_EXP)),
    ("dated-expired-late", A_FROM, None, LATE_EXP, T1, OPEN, (None, LATE_EXP)),
    ("undated-ended", None, A_TO, None, T1, None, None),
    ("dated-ended", A_FROM, A_TO, None, T1, None, None),
    ("undated-ended-closed-early", None, A_TO, EARLY_EXP, T1, None, None),
    ("dated-ended-closed-early", A_FROM, A_TO, EARLY_EXP, T1, None, None),
    ("undated-ended-closed-late", None, A_TO, LATE_EXP, T1, OPEN, None),
    ("dated-ended-closed-late", A_FROM, A_TO, LATE_EXP, T1, OPEN, None),
    ("undated-ends-later", None, LATE_END, None, T1, (LATE_END, None), (LATE_END, None)),  # noqa: E501
    ("dated-ends-later", A_FROM, LATE_END, None, T1, (LATE_END, None), (LATE_END, None)),  # noqa: E501
    ("ends-later-closed-early", A_FROM, LATE_END, EARLY_EXP, T1, (LATE_END, EARLY_EXP), (LATE_END, EARLY_EXP)),  # noqa: E501
    ("ends-later-closed-late", A_FROM, LATE_END, LATE_EXP, T1, OPEN, (LATE_END, LATE_EXP)),  # noqa: E501
    ("not-started", LATE_START, None, None, T1, None, None),
    ("not-started-closed-late", LATE_START, A_TO, LATE_EXP, T1, None, None),
    ("recorded-between-ks", A_FROM, None, None, AFTER_BETWEEN, None, OPEN),
    ("recorded-at-k", A_FROM, None, None, BETWEEN, OPEN, OPEN),
    ("starts-at-v", V_2021, None, None, T1, OPEN, OPEN),
    ("ends-at-v", A_FROM, V_2021, None, T1, None, None),
    ("closed-at-k", A_FROM, A_TO, BETWEEN, T1, None, None),
    # Timezone-naive store values are read as UTC.
    ("naive-closed-late", _naive(A_FROM), _naive(A_TO), _naive(LATE_EXP), _naive(T1), OPEN, None),  # noqa: E501
    ("naive-ends-later", _naive(A_FROM), _naive(LATE_END), None, _naive(T1), (LATE_END, None), (LATE_END, None)),  # noqa: E501
]  # fmt: skip


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "valid_at", "invalid_at", "expired_at", "created_at", "at_k", "at_now"),
    KNOWN_AT_ROWS,
    ids=[row[0] for row in KNOWN_AT_ROWS],
)
async def test_ret_03_known_at_rule_over_null_and_non_null_timestamps(
    fake: ReadGraphiti,
    graph: Graph,
    name: str,
    valid_at: datetime | None,
    invalid_at: datetime | None,
    expired_at: datetime | None,
    created_at: datetime,
    at_k: Shown,
    at_now: Shown,
) -> None:
    put_entity(graph, "entity-zebra", "Zebra")
    put_edge(
        graph,
        name,
        "Zebra fact",
        source="entity-zebra",
        target="entity-zebra",
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        created_at=created_at,
    )
    backend = make_backend()

    for known_at, expected in ((BETWEEN, at_k), (NOW, at_now)):
        hits = await backend.search(
            NS, MemoryQuery(text="zebra", valid_at=V_2021, known_at=known_at)
        )
        shown = [(hit.valid_to, hit.expired_at) for hit in hits]
        assert shown == ([] if expected is None else [expected]), known_at
        for hit in hits:
            assert hit.fact_id == name
            assert hit.recorded_at == created_at.replace(tzinfo=UTC)
            for value in (hit.valid_from, hit.valid_to, hit.expired_at):
                assert value is None or value.tzinfo is not None


# --- error translation ------------------------------------------------------


def _fail(fake: ReadGraphiti, graph: Graph, error: BaseException) -> None:
    graph.fail_search = error
    fake.read_driver.fail_query = error
    graph.fail_edges = error


async def _read(backend: GraphitiBackend, operation: str) -> object:
    if operation == "search":
        return await backend.search(
            NS, MemoryQuery(text="dollars", valid_at=NOW, known_at=NOW)
        )
    if operation == "timeline":
        return await backend.timeline(NS, "Rate A")
    return await backend.provenance(NS, "fact-a")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["search", "timeline", "provenance"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError(f"embedding dimension mismatch (auth {PASSWORD})"),
        RuntimeError(f"query failed for key {API_KEY}"),
    ],
    ids=["value-error", "runtime-error"],
)
async def test_ret_03_a_store_failure_that_is_not_a_lost_store_is_not_unavailable(
    fake: ReadGraphiti, graph: Graph, operation: str, error: BaseException
) -> None:
    _fail(fake, graph, error)

    with pytest.raises(TemporalMemoryError) as raised:
        await _read(make_backend(), operation)

    # Unavailable would let a search degrade quietly; this must surface.
    assert not isinstance(raised.value, MemoryBackendUnavailableError)
    assert type(error).__name__ in str(raised.value)
    assert PASSWORD not in str(raised.value)
    assert API_KEY not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["search", "timeline", "provenance"])
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError(f"connection reset (auth {PASSWORD})"),
        ServiceUnavailable(f"bolt refused for key {API_KEY}"),
    ],
    ids=["connection-error", "service-unavailable"],
)
async def test_ret_03_a_lost_store_is_unavailable_and_redacted(
    fake: ReadGraphiti, graph: Graph, operation: str, error: BaseException
) -> None:
    _fail(fake, graph, error)

    with pytest.raises(MemoryBackendUnavailableError) as raised:
        await _read(make_backend(), operation)

    assert raised.value.backend == "graphiti"
    assert type(error).__name__ in str(raised.value)
    assert PASSWORD not in str(raised.value)
    assert API_KEY not in str(raised.value)


# --- MB-02: one namespace ---------------------------------------------------


@pytest.mark.asyncio
async def test_ret_03_search_reads_one_namespace_only(
    fake: ReadGraphiti, graph: Graph
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2020, known_at=NOW)
    )

    assert "fact-elsewhere" not in ids(hits)
    assert graph.searches[0]["group_ids"] == [NS]
    assert graph.searches[0]["driver"] is fake.driver
    assert fake.driver.clones == [NS]


# --- RET-04 and RET-09: expansion ------------------------------------------


@pytest.mark.asyncio
async def test_ret_04_config_uses_rrf_hops_and_limit_without_cross_encoder(
    fake: ReadGraphiti, graph: Graph
) -> None:
    await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=NOW, known_at=NOW, hops=2, limit=3)
    )

    config: SearchConfig = graph.searches[0]["config"]
    assert config.edge_config is not None
    assert config.edge_config.reranker is EdgeReranker.rrf
    assert config.edge_config.search_methods == [
        EdgeSearchMethod.bm25,
        EdgeSearchMethod.cosine_similarity,
        EdgeSearchMethod.bfs,
    ]
    assert config.edge_config.bfs_max_depth == 2
    assert config.limit == 3
    assert config.node_config is None
    assert config.episode_config is None
    assert config.community_config is None


@pytest.mark.asyncio
async def test_ret_04_hops_bound_the_expansion_and_limit_bounds_the_result(
    fake: ReadGraphiti, graph: Graph
) -> None:
    graph.mentions[str(EP_A.episode_id)] = [ENTITY_RATE]
    backend = make_backend()
    one = await backend.search(
        NS,
        MemoryQuery(
            text="nothing-matches",
            valid_at=V_2021,
            known_at=NOW,
            hops=1,
            seed_episode_ids=(EP_A.episode_id,),
        ),
    )
    two = await backend.search(
        NS,
        MemoryQuery(
            text="nothing-matches",
            valid_at=V_2021,
            known_at=NOW,
            hops=2,
            seed_episode_ids=(EP_A.episode_id,),
        ),
    )
    limited = await backend.search(
        NS,
        MemoryQuery(
            text="nothing-matches",
            valid_at=V_2021,
            known_at=NOW,
            hops=2,
            limit=1,
            seed_episode_ids=(EP_A.episode_id,),
        ),
    )

    # Rate -> Agency is one hop; Agency -> Minister is the second.
    assert ids(one) == ["fact-b"]
    assert sorted(ids(two)) == ["fact-b", "fact-null"]
    assert len(limited) == 1


@pytest.mark.asyncio
async def test_ret_09_seeds_are_the_entities_complete_episodes_mention(
    fake: ReadGraphiti, graph: Graph
) -> None:
    graph.mentions[str(EP_A.episode_id)] = [ENTITY_RATE, ENTITY_AGENCY]
    graph.mentions[str(EP_B.episode_id)] = [ENTITY_RATE]
    graph.mentions[str(EP_PENDING.episode_id)] = [ENTITY_MINISTER]
    graph.mentions[str(EP_OTHER.episode_id)] = [ENTITY_ELSEWHERE]

    await make_backend().search(
        NS,
        MemoryQuery(
            text="dollars",
            valid_at=NOW,
            known_at=NOW,
            seed_episode_ids=(
                EP_A.episode_id,
                EP_B.episode_id,
                EP_PENDING.episode_id,
                EP_OTHER.episode_id,
            ),
        ),
    )

    assert graph.searches[0]["origins"] == [ENTITY_RATE, ENTITY_AGENCY]


@pytest.mark.asyncio
async def test_ret_09_no_seed_lets_graphiti_expand_from_its_own_results(
    fake: ReadGraphiti, graph: Graph
) -> None:
    # "120" matches fact-b only (Rate A -> Agency B). fact-null (Agency B ->
    # Minister) shares no word with the query: it is reached only when
    # Graphiti expands from the source of its own first result.
    backend = make_backend()
    unseeded = await backend.search(
        NS, MemoryQuery(text="120", valid_at=V_2021, known_at=NOW, hops=2)
    )
    # Seeds whose episodes mention nothing are no seeds either.
    unknown_seed = await backend.search(
        NS,
        MemoryQuery(
            text="120",
            valid_at=V_2021,
            known_at=NOW,
            hops=2,
            seed_episode_ids=(uuid5(NAMESPACE_URL, "unknown"),),
        ),
    )

    assert sorted(ids(unseeded)) == ["fact-b", "fact-null"]
    assert sorted(ids(unknown_seed)) == ["fact-b", "fact-null"]
    assert all(s["origins"] is None for s in graph.searches)


# --- RET-06: sources --------------------------------------------------------


@pytest.mark.asyncio
async def test_ret_06_hit_sources_are_complete_holodeck_episodes_only(
    fake: ReadGraphiti,
) -> None:
    hits = await make_backend().search(
        NS, MemoryQuery(text="dollars", valid_at=V_2020, known_at=NOW)
    )

    sources = hits[0].sources
    assert [s.episode_id for s in sources] == [EP_A.episode_id]
    assert sources[0].source_tool == "family_law"
    assert sources[0].chunk_id == "a_chunk_0"
    assert sources[0].source_uri == "corpus/a.md"
    assert sources[0].reference_time == A_FROM
    assert sources[0].reference_time_source == "front_matter"
    assert sources[0].text is None


@pytest.mark.asyncio
async def test_ret_06_provenance_resolves_a_fact_to_its_chunks(
    fake: ReadGraphiti,
) -> None:
    provenance = await make_backend().provenance(NS, "fact-a")

    assert provenance is not None
    assert provenance.fact_id == "fact-a"
    assert provenance.text == "Rate A is 100 dollars"
    assert provenance.valid_from == A_FROM
    assert provenance.valid_to == A_TO
    assert provenance.recorded_at == T1
    assert provenance.expired_at == T2
    assert [(s.source_tool, s.chunk_id, s.text) for s in provenance.sources] == [
        ("family_law", "a_chunk_0", "Passage a.")
    ]


@pytest.mark.asyncio
async def test_ret_06_provenance_of_an_unknown_or_foreign_fact_is_none(
    fake: ReadGraphiti,
) -> None:
    backend = make_backend()

    assert await backend.provenance(NS, "no-such-fact") is None
    assert await backend.provenance(NS, "fact-elsewhere") is None
    assert await backend.provenance(OTHER_NS, "fact-a") is None


# --- RET-08: timeline -------------------------------------------------------


@pytest.mark.asyncio
async def test_ret_08_timeline_lists_open_and_closed_facts_by_valid_from(
    fake: ReadGraphiti, graph: Graph
) -> None:
    put_edge(graph, "fact-tie", "Rate A tie", valid_at=A_FROM, created_at=T1)
    put_edge(graph, "fact-undated", "Rate A undated", valid_at=None, created_at=T1)

    entries = await make_backend().timeline(NS, "rate a")

    assert ids(entries) == ["fact-undated", "fact-a", "fact-tie", "fact-b"]
    closed = entries[1]
    assert closed.valid_to == A_TO
    assert closed.expired_at == T2
    assert [s.chunk_id for s in closed.sources] == ["a_chunk_0"]
    assert entries[3].valid_to is None
    assert graph.entity_queries == [{"group_id": NS, "entity_ref": "rate a"}]


@pytest.mark.asyncio
async def test_ret_08_timeline_by_uuid_and_unknown_entity(fake: ReadGraphiti) -> None:
    backend = make_backend()

    by_uuid = await backend.timeline(NS, ENTITY_MINISTER)
    unknown = await backend.timeline(NS, "Nobody")
    elsewhere = await backend.timeline(OTHER_NS, ENTITY_RATE)

    assert ids(by_uuid) == ["fact-null", "fact-far"]
    assert unknown == []
    assert elsewhere == []


@pytest.mark.asyncio
async def test_ret_08_timeline_uses_an_entity_named_inside_the_reference(
    fake: ReadGraphiti,
) -> None:
    backend = make_backend()

    exact = await backend.timeline(NS, "rate a")
    topic = await backend.timeline(NS, "The Rate A amount over time")

    assert ids(exact) == ["fact-a", "fact-b"]
    assert ids(topic) == ids(exact)


@pytest.mark.asyncio
async def test_ret_08_an_exact_name_is_not_widened_by_contained_names(
    fake: ReadGraphiti, graph: Graph
) -> None:
    put_entity(graph, "entity-longer", "Rate A supplement")
    put_edge(
        graph,
        "fact-longer",
        "Rate A supplement is 5",
        valid_at=A_FROM,
        created_at=T1,
        source="entity-longer",
        target="entity-longer",
    )

    entries = await make_backend().timeline(NS, "Rate A supplement")

    assert ids(entries) == ["fact-longer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("reference", "names", "expected"), ENTITY_CASES)
async def test_ret_08_timeline_resolves_names_at_word_boundaries(
    fake: ReadGraphiti,
    graph: Graph,
    reference: str,
    names: tuple[str, ...],
    expected: list[str],
) -> None:
    # A namespace of its own, so the seeded "Rate A" plays no part.
    namespace = "entity-match"
    for index, name in enumerate(names):
        put_entity(graph, f"entity-{name}", name, namespace=namespace)
        put_edge(
            graph,
            f"fact-{name}",
            f"{name} fact",
            source=f"entity-{name}",
            target=f"entity-{name}",
            valid_at=A_FROM + timedelta(days=index),
            namespace=namespace,
        )

    entries = await make_backend().timeline(namespace, reference)

    assert ids(entries) == [f"fact-{name}" for name in expected]


# --- MB-03: capabilities ----------------------------------------------------


def test_mb_03_read_operations_are_supported() -> None:
    supported = make_backend().capabilities().supported

    assert supported == frozenset(
        {
            MemoryOperation.INGEST,
            MemoryOperation.PURGE,
            MemoryOperation.SEARCH,
            MemoryOperation.VALID_AT,
            MemoryOperation.KNOWN_AT,
            MemoryOperation.EXPAND,
            MemoryOperation.TIMELINE,
            MemoryOperation.PROVENANCE,
            MemoryOperation.ENTITY_TYPE_HINTS,
        }
    )


@pytest.mark.parametrize(
    "operation",
    [
        MemoryOperation.SINGLE_VALUED_PREDICATES,
        MemoryOperation.SUPERSESSION_AUDIT,
        MemoryOperation.RETRACTION,
        MemoryOperation.REVIEW_QUEUE,
    ],
)
def test_mb_03_unsupported_rows_raise_naming_the_operation(
    operation: MemoryOperation,
) -> None:
    capabilities = make_backend().capabilities()

    with pytest.raises(MemoryNotSupportedError) as raised:
        capabilities.require(operation)

    assert raised.value.operation == operation.value
    assert raised.value.backend == "graphiti"
    assert operation.value in str(raised.value)


# --- unavailable store ------------------------------------------------------


@pytest.mark.asyncio
async def test_ret_03_search_on_a_lost_store_raises_unavailable_redacted(
    fake: ReadGraphiti, graph: Graph
) -> None:
    graph.fail_search = ServiceUnavailable(f"bolt refused (auth {PASSWORD})")

    with pytest.raises(MemoryBackendUnavailableError) as raised:
        await make_backend().search(
            NS, MemoryQuery(text="dollars", valid_at=NOW, known_at=NOW)
        )

    assert raised.value.backend == "graphiti"
    assert PASSWORD not in str(raised.value)
    assert "ServiceUnavailable" in str(raised.value)


@pytest.mark.asyncio
async def test_ret_08_timeline_and_provenance_on_a_lost_store_raise_unavailable(
    fake: ReadGraphiti, graph: Graph
) -> None:
    backend = make_backend()
    fake.read_driver.fail_query = ConnectionRefusedError(f"refused {API_KEY}")
    graph.fail_edges = ConnectionRefusedError(f"refused {API_KEY}")

    with pytest.raises(MemoryBackendUnavailableError) as timeline:
        await backend.timeline(NS, "Rate A")
    with pytest.raises(MemoryBackendUnavailableError) as provenance:
        await backend.provenance(NS, "fact-a")

    assert API_KEY not in str(timeline.value)
    assert API_KEY not in str(provenance.value)
