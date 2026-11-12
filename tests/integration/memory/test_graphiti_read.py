"""Graphiti read side against a real Neo4j (spec 043, design seam 5, T11).

Most facts are written with Graphiti's own ``save`` methods, so every
timestamp is exact: two "compilations" ingested at T1 and T2, the older
rate closed by the newer one. One test runs a real ingest with the stub LLM
to show what extraction stores. No model or embedding endpoint is called.

Every seeded fact has the embedding ``[0, 0, 0, 1]``. The stub embedder maps
a query whose length is a multiple of 7 to ``[1, 0.5, 0.25, 0]``, which is
orthogonal to it, so similarity search finds nothing and each hit comes from
full-text search or breadth-first expansion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from holodeck.lib.memory.backends.graphiti import (
    GraphitiBackend,
    encode_episode_name,
)
from holodeck.lib.memory.service import episode_id_for
from holodeck.lib.memory.types import EpisodeIn, FactHit, MemoryQuery
from tests.integration.memory.conftest import NamespaceFactory, Neo4jSettings
from tests.integration.memory.stub_clients import EDGE_FACT, StubLLMClient

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.graphiti]

T1 = datetime(2024, 1, 1, tzinfo=UTC)  # first compilation ingested
T2 = datetime(2024, 6, 1, tzinfo=UTC)  # second compilation ingested
BETWEEN = datetime(2024, 3, 1, tzinfo=UTC)
A_FROM = datetime(2019, 7, 1, tzinfo=UTC)
A_TO = datetime(2020, 7, 1, tzinfo=UTC)
V_2020 = datetime(2020, 1, 1, tzinfo=UTC)
V_2021 = datetime(2021, 1, 1, tzinfo=UTC)
FAR_AWAY = [0.0, 0.0, 0.0, 1.0]

# Queries of 7 characters: the stub embedder gives them no similarity hit.
DOLLARS = "dollars"
REVIEWS = "reviews"
NOTHING = "nothing"
assert len(DOLLARS) == len(REVIEWS) == len(NOTHING) == 7


def now() -> datetime:
    return datetime.now(UTC)


def make_episode(namespace: str, name: str, when: datetime) -> EpisodeIn:
    text = f"Passage {name}."
    return EpisodeIn(
        episode_id=episode_id_for(namespace, f"section-{name}", text),
        source_tool="rates",
        chunk_id=f"rates-{name}-c0",
        source_uri=f"corpus/rates-{name}.md",
        text=text,
        reference_time=when,
        reference_time_source="front_matter",
    )


@dataclass(frozen=True, slots=True)
class Seeded:
    """Uuids of the facts and episodes one namespace was seeded with."""

    namespace: str
    fact_a: str  # 100 dollars, valid 2019-07-01 to 2020-07-01, closed at T2
    fact_b: str  # 120 dollars, valid from 2020-07-01, recorded at T2
    fact_c_old: str  # Rate C, 50 euro, valid 2019-07-01 to 2020-07-01
    fact_c_new: str  # Rate C, 60 euro, valid from 2020-07-01
    fact_reviews: str  # Rate C, open since 2019, the only full-text hit for REVIEWS
    fact_appoints: str  # Minister -> Office, open
    fact_inspects: str  # Office -> Inspector, open: two hops from Minister
    episode_a: EpisodeIn
    episode_b: EpisodeIn
    episode_minister: EpisodeIn


@pytest_asyncio.fixture
async def seed(
    neo4j_settings: Neo4jSettings,
) -> AsyncIterator[object]:
    """A function that seeds one namespace through Graphiti's own models."""
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    driver = Neo4jDriver(
        neo4j_settings.uri, neo4j_settings.user, neo4j_settings.password
    )

    async def run(namespace: str) -> Seeded:
        return await _seed(driver, namespace)

    try:
        yield run
    finally:
        await driver.close()


async def _seed(driver: object, namespace: str) -> Seeded:
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    assert isinstance(driver, GraphDriver)

    async def entity(name: str) -> str:
        node = EntityNode(
            name=name, group_id=namespace, created_at=T1, name_embedding=FAR_AWAY
        )
        await node.save(driver)
        return node.uuid

    async def episode(ep: EpisodeIn, mentions: Sequence[str]) -> str:
        node = EpisodicNode(
            uuid=str(ep.episode_id),
            name=encode_episode_name("complete", ep),
            group_id=namespace,
            labels=[],
            source=EpisodeType.text,
            source_description="Passage.",
            content=ep.text,
            created_at=T1,
            valid_at=ep.reference_time,
        )
        await node.save(driver)
        for target in mentions:
            await EpisodicEdge(
                group_id=namespace,
                source_node_uuid=node.uuid,
                target_node_uuid=target,
                created_at=T1,
            ).save(driver)
        return node.uuid

    async def fact(
        source: str,
        target: str,
        text: str,
        *,
        valid_at: datetime | None,
        invalid_at: datetime | None = None,
        created_at: datetime = T1,
        expired_at: datetime | None = None,
        episodes: Sequence[str] = (),
    ) -> str:
        edge = EntityEdge(
            group_id=namespace,
            source_node_uuid=source,
            target_node_uuid=target,
            name="RELATES",
            fact=text,
            fact_embedding=FAR_AWAY,
            episodes=list(episodes),
            created_at=created_at,
            valid_at=valid_at,
            invalid_at=invalid_at,
            expired_at=expired_at,
        )
        await edge.save(driver)
        return edge.uuid

    rate = await entity("Rate A")
    agency = await entity("Agency B")
    rate_c = await entity("Rate C")
    minister = await entity("Minister")
    office = await entity("Office")
    inspector = await entity("Inspector")
    ep_a = make_episode(namespace, "2019", A_FROM)
    ep_b = make_episode(namespace, "2020", A_TO)
    ep_minister = make_episode(namespace, "minister", A_FROM)
    await episode(ep_a, [rate, agency])
    await episode(ep_b, [rate, agency])
    await episode(ep_minister, [minister])
    return Seeded(
        namespace=namespace,
        fact_a=await fact(
            rate,
            agency,
            "Rate A is 100 dollars a fortnight",
            valid_at=A_FROM,
            invalid_at=A_TO,
            created_at=T1,
            expired_at=T2,
            episodes=[str(ep_a.episode_id)],
        ),
        fact_b=await fact(
            rate,
            agency,
            "Rate A is 120 dollars a fortnight",
            valid_at=A_TO,
            created_at=T2,
            episodes=[str(ep_b.episode_id)],
        ),
        fact_c_old=await fact(
            rate_c,
            agency,
            "Rate C is 50 euro",
            valid_at=A_FROM,
            invalid_at=A_TO,
            created_at=T1,
            expired_at=T2,
        ),
        fact_c_new=await fact(
            rate_c, agency, "Rate C is 60 euro", valid_at=A_TO, created_at=T2
        ),
        fact_reviews=await fact(
            rate_c, agency, "Rate C has annual reviews", valid_at=A_FROM
        ),
        fact_appoints=await fact(
            minister, office, "The Minister appoints the Office", valid_at=A_FROM
        ),
        fact_inspects=await fact(
            office, inspector, "The Office employs an Inspector", valid_at=A_FROM
        ),
        episode_a=ep_a,
        episode_b=ep_b,
        episode_minister=ep_minister,
    )


async def seeded(seed: object, namespace: str) -> Seeded:
    assert callable(seed)
    result = await seed(namespace)
    assert isinstance(result, Seeded)
    return result


def ids(hits: Sequence[FactHit]) -> set[str]:
    return {hit.fact_id for hit in hits}


# --- RET-03 -----------------------------------------------------------------


async def test_ret_03_between_two_compilations(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())
    ns = data.namespace

    before = await backend.search(
        ns, MemoryQuery(text=DOLLARS, valid_at=V_2020, known_at=now())
    )
    after = await backend.search(
        ns, MemoryQuery(text=DOLLARS, valid_at=V_2021, known_at=now())
    )
    as_known_between = await backend.search(
        ns, MemoryQuery(text=DOLLARS, valid_at=V_2021, known_at=BETWEEN)
    )

    assert [hit.fact_id for hit in before] == [data.fact_a]
    assert before[0].valid_from == A_FROM
    assert before[0].valid_to == A_TO
    assert before[0].recorded_at == T1
    assert before[0].expired_at == T2

    assert [hit.fact_id for hit in after] == [data.fact_b]
    assert after[0].valid_from == A_TO
    assert after[0].valid_to is None
    assert after[0].recorded_at == T2

    # The older fact, as the store believed it at BETWEEN: still open.
    assert [hit.fact_id for hit in as_known_between] == [data.fact_a]
    assert as_known_between[0].valid_from == A_FROM
    assert as_known_between[0].valid_to is None
    assert as_known_between[0].expired_at is None
    assert as_known_between[0].recorded_at == T1
    assert as_known_between[0].score > 0
    assert [s.chunk_id for s in as_known_between[0].sources] == ["rates-2019-c0"]


async def test_ret_03_known_at_before_the_first_ingest_returns_nothing(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())

    hits = await backend.search(
        data.namespace,
        MemoryQuery(
            text=DOLLARS, valid_at=V_2021, known_at=datetime(2023, 1, 1, tzinfo=UTC)
        ),
    )

    assert hits == []


async def test_ret_03_expansion_with_no_seed_respects_the_time_filter(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())
    ns = data.namespace

    # Only the reviews fact matches the text. Graphiti then expands from its
    # source (Rate C) and finds the two rates; the time filter keeps one.
    in_2021 = await backend.search(
        ns, MemoryQuery(text=REVIEWS, valid_at=V_2021, known_at=now())
    )
    in_2020 = await backend.search(
        ns, MemoryQuery(text=REVIEWS, valid_at=V_2020, known_at=now())
    )

    assert ids(in_2021) == {data.fact_reviews, data.fact_c_new}
    assert ids(in_2020) == {data.fact_reviews, data.fact_c_old}


async def test_ret_03_an_undated_fact_from_a_real_ingest_gets_the_episode_date(
    backend: GraphitiBackend,
    new_namespace: NamespaceFactory,
    stub_llm: StubLLMClient,
    neo4j_settings: Neo4jSettings,
) -> None:
    ns = new_namespace()
    episode = make_episode(ns, "ingested", A_TO)
    report = await backend.ingest(ns, [episode], "Extract who administers what.")
    assert report.failed is None, report.error

    from neo4j import GraphDatabase

    with GraphDatabase.driver(
        neo4j_settings.uri, auth=(neo4j_settings.user, neo4j_settings.password)
    ) as driver:
        rows = driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $g "
            "RETURN e.uuid AS uuid, e.valid_at AS valid_at",
            g=ns,
        ).records
    # The stub model gives no date. Graphiti stores none, and the backend
    # then dates the fact with its episode's effective date (Fix A).
    assert [row["valid_at"].to_native() for row in rows] == [A_TO]

    hits = await backend.search(
        ns, MemoryQuery(text="administered", valid_at=V_2021, known_at=now())
    )

    assert [hit.text for hit in hits] == [EDGE_FACT]
    assert hits[0].valid_from == A_TO
    assert hits[0].fact_id == rows[0]["uuid"]
    assert [s.episode_id for s in hits[0].sources] == [episode.episode_id]

    provenance = await backend.provenance(ns, hits[0].fact_id)

    assert provenance is not None
    assert [
        (s.source_tool, s.chunk_id, s.source_uri, s.text) for s in provenance.sources
    ] == [("rates", "rates-ingested-c0", "corpus/rates-ingested.md", episode.text)]
    assert provenance.sources[0].reference_time == A_TO


# --- MB-02: another namespace -----------------------------------------------


async def test_ret_03_a_fact_in_another_namespace_is_never_returned(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    mine = await seeded(seed, new_namespace())
    theirs = await seeded(seed, new_namespace())
    theirs_ids = {
        theirs.fact_a,
        theirs.fact_b,
        theirs.fact_c_old,
        theirs.fact_c_new,
        theirs.fact_reviews,
        theirs.fact_appoints,
        theirs.fact_inspects,
    }

    searches = [
        await backend.search(
            mine.namespace, MemoryQuery(text=text, valid_at=v, known_at=k, hops=2)
        )
        for text in (DOLLARS, REVIEWS, "Rate")
        for v in (V_2020, V_2021)
        for k in (BETWEEN, now())
    ]
    seeded_search = await backend.search(
        mine.namespace,
        MemoryQuery(
            text=NOTHING,
            valid_at=V_2021,
            known_at=now(),
            hops=2,
            # Episode ids from the other namespace give no seed.
            seed_episode_ids=(theirs.episode_minister.episode_id,),
        ),
    )
    timeline = await backend.timeline(mine.namespace, "Rate A")

    returned = set().union(*(ids(hits) for hits in searches))
    assert returned
    assert returned.isdisjoint(theirs_ids)
    assert seeded_search == []
    assert {entry.fact_id for entry in timeline}.isdisjoint(theirs_ids)
    assert await backend.provenance(mine.namespace, theirs.fact_a) is None


# --- RET-04 and RET-09: seeds and hops --------------------------------------


async def test_ret_09_seed_episodes_expand_to_the_hop_limit(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())
    ns = data.namespace
    seeds = (data.episode_minister.episode_id,)

    unseeded = await backend.search(
        ns, MemoryQuery(text=NOTHING, valid_at=V_2021, known_at=now(), hops=2)
    )
    one_hop = await backend.search(
        ns,
        MemoryQuery(
            text=NOTHING,
            valid_at=V_2021,
            known_at=now(),
            hops=1,
            seed_episode_ids=seeds,
        ),
    )
    two_hops = await backend.search(
        ns,
        MemoryQuery(
            text=NOTHING,
            valid_at=V_2021,
            known_at=now(),
            hops=2,
            seed_episode_ids=seeds,
        ),
    )
    limited = await backend.search(
        ns,
        MemoryQuery(
            text=NOTHING,
            valid_at=V_2021,
            known_at=now(),
            hops=2,
            limit=1,
            seed_episode_ids=seeds,
        ),
    )

    assert unseeded == []
    assert ids(one_hop) == {data.fact_appoints}
    assert ids(two_hops) == {data.fact_appoints, data.fact_inspects}
    assert len(limited) == 1


# --- RET-08: timeline -------------------------------------------------------


async def test_ret_08_timeline_is_ordered_by_valid_from_with_closed_facts(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())

    entries = await backend.timeline(data.namespace, "rate a")

    assert [entry.fact_id for entry in entries] == [data.fact_a, data.fact_b]
    assert [entry.valid_from for entry in entries] == [A_FROM, A_TO]
    closed = entries[0]
    assert closed.valid_to == A_TO
    assert closed.expired_at == T2
    assert closed.recorded_at == T1
    assert [s.chunk_id for s in closed.sources] == ["rates-2019-c0"]
    assert entries[1].valid_to is None
    assert await backend.timeline(data.namespace, "Nobody") == []
    topic = await backend.timeline(data.namespace, "Rate A amounts over time")
    assert [entry.fact_id for entry in topic] == [data.fact_a, data.fact_b]


async def test_ret_08_timeline_resolves_names_at_word_boundaries(
    backend: GraphitiBackend,
    new_namespace: NamespaceFactory,
    neo4j_settings: Neo4jSettings,
) -> None:
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    ns = new_namespace()
    driver = Neo4jDriver(
        neo4j_settings.uri, neo4j_settings.user, neo4j_settings.password
    )
    facts: dict[str, str] = {}
    try:
        for index, name in enumerate(("s. 1", "s. 10", "Part I", "Part III")):
            node = EntityNode(
                name=name, group_id=ns, created_at=T1, name_embedding=FAR_AWAY
            )
            await node.save(driver)
            edge = EntityEdge(
                group_id=ns,
                source_node_uuid=node.uuid,
                target_node_uuid=node.uuid,
                name="RELATES",
                fact=f"{name} sets the amount",
                fact_embedding=FAR_AWAY,
                episodes=[],
                created_at=T1,
                valid_at=datetime(2019, 1, 1 + index, tzinfo=UTC),
            )
            await edge.save(driver)
            facts[name] = edge.uuid
    finally:
        await driver.close()

    async def timeline_of(reference: str) -> list[str]:
        return [e.fact_id for e in await backend.timeline(ns, reference)]

    assert await timeline_of("history of s. 10") == [facts["s. 10"]]
    assert await timeline_of("history of s. 1") == [facts["s. 1"]]
    assert await timeline_of("S. 10") == [facts["s. 10"]]
    assert await timeline_of("history of Part III") == [facts["Part III"]]
    assert await timeline_of("s. 1, s. 10 and Part I") == [
        facts["s. 1"],
        facts["s. 10"],
        facts["Part I"],
    ]
    assert await timeline_of("s. 100") == []


async def test_ret_08_timeline_breaks_a_valid_from_tie_by_fact_id(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())

    entries = await backend.timeline(data.namespace, "Rate C")

    assert [entry.valid_from for entry in entries] == [A_FROM, A_FROM, A_TO]
    assert [entry.fact_id for entry in entries[:2]] == sorted(
        [data.fact_c_old, data.fact_reviews]
    )
    assert entries[2].fact_id == data.fact_c_new


# --- RET-06: provenance -----------------------------------------------------


async def test_ret_06_provenance_resolves_a_fact_to_its_chunks(
    backend: GraphitiBackend, new_namespace: NamespaceFactory, seed: object
) -> None:
    data = await seeded(seed, new_namespace())

    provenance = await backend.provenance(data.namespace, data.fact_b)

    assert provenance is not None
    assert provenance.text == "Rate A is 120 dollars a fortnight"
    assert provenance.valid_from == A_TO
    assert provenance.recorded_at == T2
    assert [
        (s.episode_id, s.source_tool, s.chunk_id, s.source_uri, s.text)
        for s in provenance.sources
    ] == [
        (
            data.episode_b.episode_id,
            "rates",
            "rates-2020-c0",
            "corpus/rates-2020.md",
            data.episode_b.text,
        )
    ]
    assert await backend.provenance(data.namespace, str(uuid4())) is None
