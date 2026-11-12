"""The dating, closure-check and text-closing Cypher on a real Neo4j (spec 043).

The unit tests in ``tests/unit/lib/memory/test_graphiti_backend_write.py``
run these rules on a fake driver that re-implements each query in Python.
Here each query constant runs on Neo4j against hand-built episodes, entities
and facts, and two tests run the backend's closure check and text closing
end to end on the same kind of graph. Episodes are saved with Graphiti's own
``EpisodicNode.save``, so their fields have Graphiti's layout. No model and
no embedding is called.

The connection comes from ``conftest.py`` (``HOLODECK_TEST_NEO4J_URI``,
``HOLODECK_TEST_NEO4J_USER``, ``HOLODECK_TEST_NEO4J_PASSWORD``); the tests
skip when Neo4j is not reachable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
import pytest_asyncio
from neo4j.time import DateTime as Neo4jDateTime
from pydantic import SecretStr

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
    EpisodeState,
    GraphitiBackend,
    encode_episode_name,
)
from holodeck.lib.memory.backends.graphiti_closure import SectionKey
from holodeck.lib.memory.types import EpisodeIn
from tests.integration.memory.conftest import Neo4jSettings

if TYPE_CHECKING:
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.graphiti]

Y2020 = datetime(2020, 7, 1, tzinfo=UTC)
Y2023 = datetime(2023, 7, 1, tzinfo=UTC)
Y2025 = datetime(2025, 1, 1, tzinfo=UTC)
Y2026 = datetime(2026, 7, 1, tzinfo=UTC)
Y2027 = datetime(2027, 1, 1, tzinfo=UTC)
PATH = ("Part 3—Family tax benefit", "Division 2—Rate")
S61A = SectionKey("h4_part_3_61a_rate_reduction", "61A", PATH)


@dataclass(frozen=True, slots=True)
class Graph:
    """A driver and a namespace unique to one test.

    Every uuid a test writes (episodes, facts, entities) is derived from the
    namespace, so parallel tests and leftovers of earlier runs never share
    a node or an edge.
    """

    driver: Neo4jDriver
    namespace: str

    def id(self, name: str, namespace: str | None = None) -> str:
        """A fact, entity or placeholder uuid unique to this test."""
        return f"{namespace or self.namespace}-{name}"

    @property
    def other(self) -> str:
        """A second namespace of this test, removed with the first."""
        return f"{self.namespace}-x"


@pytest_asyncio.fixture
async def graph(neo4j_settings: Neo4jSettings) -> AsyncIterator[Graph]:
    """A Graphiti Neo4j driver and a fresh namespace, removed afterwards."""
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    driver = Neo4jDriver(
        neo4j_settings.uri, neo4j_settings.user, neo4j_settings.password
    )
    namespace = f"it-{uuid4().hex[:12]}"
    try:
        yield Graph(driver, namespace)
    finally:
        try:
            # The side namespace ("<namespace>-x") starts with the namespace.
            await driver.execute_query(
                "MATCH (n) WHERE n.group_id STARTS WITH $g DETACH DELETE n",
                g=namespace,
            )
        finally:
            await driver.close()


def backend() -> GraphitiBackend:
    """A backend whose closure methods are called with ``graph.driver``."""
    from holodeck.models.llm import LLMProvider, ProviderEnum
    from holodeck.models.tool import GraphBackendConfig

    config = GraphBackendConfig(driver="neo4j", uri="bolt://unused:7687")
    model = LLMProvider(
        provider=ProviderEnum.OPENAI, name="stub-model", api_key=SecretStr("unused")
    )
    return GraphitiBackend(config, model, _NoEmbeddings(), close_on_text_change=True)


class _NoEmbeddings:
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the closure queries never embed")


def episode_uuid(graph: Graph, name: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"{graph.namespace}/{name}")


def episode_in(
    graph: Graph,
    name: str,
    when: datetime,
    *,
    text: str = "",
    section: SectionKey | None = None,
    section_ids: Sequence[UUID] = (),
    tool: str = "acts",
    key: str = "",
) -> EpisodeIn:
    episode_id = episode_uuid(graph, name)
    return EpisodeIn(
        episode_id=episode_id,
        source_tool=tool,
        chunk_id=f"{name}_chunk",
        source_uri=f"corpus/{name}.md",
        text=text or f"Text of {name}.",
        reference_time=when,
        reference_time_source="front_matter",
        section_id=section.section_id if section else "",
        section_number=section.number if section else "",
        section_path=section.path if section else (),
        section_episode_ids=tuple(section_ids) or (episode_id,),
        document_key=key,
    )


async def save_episode(
    graph: Graph,
    episode: EpisodeIn,
    *,
    state: EpisodeState = "complete",
    created_at: datetime | None = None,
    entity_edges: Sequence[str] = (),
    namespace: str | None = None,
) -> str:
    """Save ``episode`` with Graphiti's ``EpisodicNode.save``; return its uuid."""
    from graphiti_core.nodes import EpisodeType, EpisodicNode

    node = EpisodicNode(
        uuid=str(episode.episode_id),
        name=encode_episode_name(state, episode),
        group_id=namespace or graph.namespace,
        labels=[],
        source=EpisodeType.text,
        source_description="test passage",
        content=episode.text,
        created_at=created_at or datetime.now(UTC),
        valid_at=episode.reference_time,
        entity_edges=list(entity_edges),
    )
    await node.save(graph.driver)
    return node.uuid


async def save_fact(
    graph: Graph,
    uuid: str,
    episodes: Sequence[str],
    *,
    fact: str = "",
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
    source: str = "A",
    target: str = "B",
    name: str = "RELATES_TO",
    namespace: str | None = None,
) -> None:
    """One ``RELATES_TO`` edge in Graphiti's layout, between two entities.

    The stored uuid is ``graph.id(uuid, namespace)``.
    """
    group = namespace or graph.namespace
    await graph.driver.execute_query(
        """
        MERGE (s:Entity {uuid: $source}) SET s.group_id = $g, s.name = $source
        MERGE (t:Entity {uuid: $target}) SET t.group_id = $g, t.name = $target
        CREATE (s)-[:RELATES_TO {
            uuid: $uuid, group_id: $g, name: $name, fact: $fact,
            episodes: $episodes, created_at: $created_at, valid_at: $valid_at,
            invalid_at: $invalid_at, expired_at: $expired_at
        }]->(t)
        """,
        source=graph.id(source, group),
        target=graph.id(target, group),
        g=group,
        uuid=graph.id(uuid, group),
        name=name,
        fact=fact or uuid,
        episodes=list(episodes),
        created_at=datetime.now(UTC),
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


@dataclass(frozen=True, slots=True)
class FactState:
    valid_at: datetime | None
    invalid_at: datetime | None
    expired_at: datetime | None

    @property
    def open(self) -> bool:
        return self.invalid_at is None and self.expired_at is None


async def fact_state(
    graph: Graph, uuid: str, namespace: str | None = None
) -> FactState:
    group = namespace or graph.namespace
    result = await graph.driver.execute_query(
        """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        WHERE r.group_id = $g
        RETURN r.valid_at AS valid_at, r.invalid_at AS invalid_at,
               r.expired_at AS expired_at
        """,
        uuid=graph.id(uuid, group),
        g=group,
    )
    (record,) = result.records

    def native(value: object) -> datetime | None:
        if value is None:
            return None
        assert isinstance(value, Neo4jDateTime)
        return value.to_native()

    return FactState(
        native(record["valid_at"]),
        native(record["invalid_at"]),
        native(record["expired_at"]),
    )


async def rows(graph: Graph, query: str, **params: object) -> list[dict[str, object]]:
    """The query's records as dicts, run the way the backend runs it."""
    result = await graph.driver.execute_query(query, **params)
    return [dict(record) for record in result.records]


# --- dating -----------------------------------------------------------------


async def test_dating_query_dates_by_the_earliest_episode_and_skips_inverted(
    graph: Graph,
) -> None:
    old = await save_episode(graph, episode_in(graph, "v2020", Y2020))
    new = await save_episode(graph, episode_in(graph, "v2026", Y2026))
    await save_fact(graph, "restated", [old, new])
    await save_fact(graph, "new-only", [new])
    await save_fact(graph, "ends-before", [new], invalid_at=Y2025)
    await save_fact(graph, "ends-on-start", [new], invalid_at=Y2026)
    await save_fact(graph, "ends-after", [new], invalid_at=Y2027)
    await save_fact(graph, "dated", [new], valid_at=Y2023)
    await save_fact(graph, "not-this-episode", [old])
    await save_fact(graph, "other-group", [new], namespace=graph.other)

    (counts,) = await rows(
        graph,
        DATE_EPISODE_FACTS_QUERY,
        uuid=new,
        group_id=graph.namespace,
        reference_time=Y2026,
    )

    assert counts == {"dated": 3, "inverted": 2}
    assert (await fact_state(graph, "restated")).valid_at == Y2020
    assert (await fact_state(graph, "new-only")).valid_at == Y2026
    assert (await fact_state(graph, "ends-before")).valid_at is None
    assert (await fact_state(graph, "ends-on-start")).valid_at is None
    assert (await fact_state(graph, "ends-after")).valid_at == Y2026
    assert (await fact_state(graph, "dated")).valid_at == Y2023
    assert (await fact_state(graph, "not-this-episode")).valid_at is None
    assert (await fact_state(graph, "other-group", graph.other)).valid_at is None


async def test_dating_query_falls_back_to_the_reference_time(graph: Graph) -> None:
    # No episode node: the start is the reference time.
    await save_fact(graph, "orphan", [graph.id("missing-episode")])

    (counts,) = await rows(
        graph,
        DATE_EPISODE_FACTS_QUERY,
        uuid=graph.id("missing-episode"),
        group_id=graph.namespace,
        reference_time=Y2026,
    )

    assert counts == {"dated": 1, "inverted": 0}
    assert (await fact_state(graph, "orphan")).valid_at == Y2026


async def test_backfill_query_dates_every_undated_fact_once(graph: Graph) -> None:
    old = await save_episode(graph, episode_in(graph, "v2020", Y2020))
    new = await save_episode(graph, episode_in(graph, "v2026", Y2026))
    await save_fact(graph, "both", [new, old])
    await save_fact(graph, "inverted", [new], invalid_at=Y2020)
    await save_fact(graph, "no-node", [graph.id("missing-episode")])

    (first,) = await rows(graph, BACKFILL_FACT_START_QUERY, group_id=graph.namespace)
    (second,) = await rows(graph, BACKFILL_FACT_START_QUERY, group_id=graph.namespace)

    assert first == {"dated": 1, "inverted": 1}
    assert second == {"dated": 0, "inverted": 1}
    assert (await fact_state(graph, "both")).valid_at == Y2020
    assert (await fact_state(graph, "inverted")).valid_at is None
    assert (await fact_state(graph, "no-node")).valid_at is None


# --- closure check ------------------------------------------------------------


async def test_closed_since_query_finds_only_this_episodes_closures(
    graph: Graph,
) -> None:
    since = datetime.now(UTC)
    before = since - timedelta(days=1)
    after = since + timedelta(seconds=1)
    own = await save_episode(
        graph,
        episode_in(graph, "v2026", Y2026),
        state="pending",
        entity_edges=[
            graph.id(n) for n in ("mine-closed", "mine-open", "mine-closed-before")
        ],
    )
    await save_fact(
        graph,
        "mine-closed",
        [graph.id("o")],
        invalid_at=Y2026,
        expired_at=after,
        name="USES",
    )
    await save_fact(graph, "mine-open", [own])
    await save_fact(
        graph,
        "mine-closed-before",
        [graph.id("o")],
        invalid_at=Y2023,
        expired_at=before,
    )
    # A concurrent writer's closure, in the window but not on the list.
    await save_fact(
        graph, "concurrent", [graph.id("o")], invalid_at=Y2026, expired_at=after
    )

    found = await rows(
        graph, CLOSED_SINCE_QUERY, group_id=graph.namespace, uuid=own, since=since
    )

    assert [row["fact_uuid"] for row in found] == [graph.id("mine-closed")]
    (row,) = found
    assert row["episodes"] == [graph.id("o")]
    assert row["name"] == "USES"
    assert row["source_uuid"] == graph.id("A")
    assert row["target_uuid"] == graph.id("B")


async def test_keep_episode_edges_query_adds_earlier_attempts_edges(
    graph: Graph,
) -> None:
    own = await save_episode(
        graph,
        episode_in(graph, "v2026", Y2026),
        state="pending",
        entity_edges=["retry-1", "both"],
    )

    (count,) = await rows(
        graph,
        KEEP_EPISODE_EDGES_QUERY,
        uuid=own,
        group_id=graph.namespace,
        edges=["first-1", "both"],
    )

    assert count == {"edges": 3}
    (stored,) = await rows(
        graph,
        "MATCH (e:Episodic {uuid: $uuid}) RETURN e.entity_edges AS edges",
        uuid=own,
    )
    assert stored["edges"] == ["retry-1", "both", "first-1"]


async def test_episode_open_facts_and_reopen_queries(graph: Graph) -> None:
    own = str(episode_uuid(graph, "v2026"))
    await save_fact(graph, "open", [own], source="C", target="D", name="OWNS")
    await save_fact(graph, "closed", [own], invalid_at=Y2026, expired_at=Y2026)
    await save_fact(
        graph, "elsewhere-closed", [graph.id("o")], invalid_at=Y2026, expired_at=Y2026
    )

    found = await rows(
        graph, EPISODE_OPEN_FACTS_QUERY, group_id=graph.namespace, uuid=own
    )
    (reopened,) = await rows(
        graph,
        REOPEN_FACTS_QUERY,
        group_id=graph.namespace,
        fact_uuids=[graph.id("elsewhere-closed")],
    )

    assert found == [
        {
            "fact_uuid": graph.id("open"),
            "name": "OWNS",
            "source_uuid": graph.id("C"),
            "target_uuid": graph.id("D"),
        }
    ]
    assert reopened == {"reopened": 1}
    assert (await fact_state(graph, "elsewhere-closed")).open
    assert not (await fact_state(graph, "closed")).open


# --- text closing ---------------------------------------------------------------


async def test_old_section_episodes_query_prefilters_older_complete_episodes(
    graph: Graph,
) -> None:
    same = await save_episode(graph, episode_in(graph, "s2020", Y2020, section=S61A))
    renamed = await save_episode(
        graph,
        episode_in(
            graph,
            "renamed2023",
            Y2023,
            section=SectionKey("h4_renamed_61a", "61A", PATH),
        ),
    )
    await save_episode(
        graph, episode_in(graph, "pending2020", Y2020, section=S61A), state="pending"
    )
    await save_episode(graph, episode_in(graph, "later2027", Y2027, section=S61A))
    await save_episode(
        graph,
        episode_in(
            graph, "other2020", Y2020, section=SectionKey("h4_61b", "61B", PATH)
        ),
    )
    await save_episode(
        graph,
        episode_in(graph, "group2020", Y2020, section=S61A),
        namespace=graph.other,
    )

    found = await rows(
        graph,
        OLD_SECTION_EPISODES_QUERY,
        group_id=graph.namespace,
        reference_time=Y2026,
        prefix=COMPLETE_PREFIX,
        section_id_key=f'"section_id":"{S61A.section_id}"',
        section_number_key='"section_number":"61A"',
    )

    assert sorted(str(row["uuid"]) for row in found) == sorted([same, renamed])


async def test_text_closure_candidates_query_excludes_restated_and_later_facts(
    graph: Graph,
) -> None:
    old = await save_episode(graph, episode_in(graph, "v2020", Y2020))
    new = await save_episode(graph, episode_in(graph, "v2023", Y2023))
    later = await save_episode(graph, episode_in(graph, "v2026", Y2026))
    await save_fact(graph, "candidate", [old], fact="The rate is $10", valid_at=Y2020)
    await save_fact(graph, "restated", [old, new], valid_at=Y2020)
    await save_fact(graph, "later-source", [old, later], valid_at=Y2020)
    await save_fact(
        graph, "closed", [old], valid_at=Y2020, invalid_at=Y2023, expired_at=Y2023
    )
    await save_fact(graph, "undated", [old])
    await save_fact(graph, "starts-at-reference", [old], valid_at=Y2023)
    await save_fact(graph, "not-old", [graph.id("unrelated")], valid_at=Y2020)

    found = await rows(
        graph,
        TEXT_CLOSURE_CANDIDATES_QUERY,
        group_id=graph.namespace,
        reference_time=Y2023,
        old_uuids=[old],
        new_uuids=[new],
    )

    assert found == [{"fact_uuid": graph.id("candidate"), "fact": "The rate is $10"}]


async def test_close_facts_query_closes_open_facts_only(graph: Graph) -> None:
    await save_fact(graph, "open", [graph.id("o")], valid_at=Y2020)
    await save_fact(
        graph,
        "closed",
        [graph.id("o")],
        valid_at=Y2020,
        invalid_at=Y2023,
        expired_at=Y2023,
    )
    now = datetime.now(UTC)

    (closed,) = await rows(
        graph,
        CLOSE_FACTS_QUERY,
        group_id=graph.namespace,
        fact_uuids=[graph.id("open"), graph.id("closed")],
        reference_time=Y2026,
        now=now,
    )

    assert closed == {"closed": 1}
    assert await fact_state(graph, "open") == FactState(Y2020, Y2026, now)
    assert (await fact_state(graph, "closed")).invalid_at == Y2023


# --- the backend methods, end to end ----------------------------------------------


async def test_check_closures_reopens_a_kept_line_and_ignores_a_concurrent_closure(
    graph: Graph,
) -> None:
    old_text = (
        "#### 61A Rate reduction\n(1) The threshold is $5,000 a year.\n"
        "(2) A reduction does not apply to a newly arrived resident."
    )
    new_text = old_text.replace("$5,000", "$6,000")
    old = await save_episode(
        graph, episode_in(graph, "61a-2020", Y2020, text=old_text, section=S61A)
    )
    other_section = SectionKey("h4_61b", "61B", PATH)
    elsewhere = await save_episode(
        graph,
        episode_in(
            graph, "61b-2020", Y2020, text="(1) Other text.", section=other_section
        ),
    )
    new = episode_in(graph, "61a-2026", Y2026, text=new_text, section=S61A)
    since = datetime.now(UTC) - timedelta(seconds=5)
    own = await save_episode(
        graph,
        new,
        state="pending",
        created_at=since,
        entity_edges=[
            graph.id(n)
            for n in ("threshold", "resident", "threshold-next", "resident-next")
        ],
    )
    stamped = datetime.now(UTC)
    for uuid, fact in (
        ("threshold", "The threshold is $5,000 a year"),
        ("resident", "A reduction does not apply to a newly arrived resident"),
    ):
        await save_fact(
            graph,
            uuid,
            [old],
            fact=fact,
            valid_at=Y2020,
            invalid_at=Y2026,
            expired_at=stamped,
            source=f"{uuid}-s",
            target=f"{uuid}-t",
        )
        await save_fact(
            graph,
            f"{uuid}-next",
            [own],
            valid_at=Y2026,
            source=f"{uuid}-s",
            target=f"{uuid}-t",
        )
    await save_fact(
        graph,
        "concurrent",
        [elsewhere],
        fact="Other text",
        valid_at=Y2020,
        invalid_at=Y2026,
        expired_at=stamped,
    )

    tally = await backend()._check_closures(graph.driver, graph.namespace, new, since)

    assert (tally.closed, tally.reopened, tally.kept) == (2, 1, 1)
    assert (await fact_state(graph, "resident")).open
    assert not (await fact_state(graph, "threshold")).open
    assert not (await fact_state(graph, "concurrent")).open


async def test_close_changed_lines_compares_every_older_version(graph: Graph) -> None:
    # 2020: c1 and c2. 2023 changes c1, so c2 keeps its 2020 episode. 2026
    # changes c2: the 2020 threshold fact is closed although the newest
    # older date (2023) holds only c1.
    ids = {
        n: episode_uuid(graph, n) for n in ("c1-2020", "c2-2020", "c1-2023", "c2-2026")
    }
    c1_2020 = episode_in(
        graph,
        "c1-2020",
        Y2020,
        text="(1) The rate is 100 dollars.",
        section=S61A,
        section_ids=(ids["c1-2020"], ids["c2-2020"]),
    )
    c2_2020 = episode_in(
        graph,
        "c2-2020",
        Y2020,
        text="(2) The threshold is 5,000 dollars.",
        section=S61A,
        section_ids=(ids["c1-2020"], ids["c2-2020"]),
    )
    c1_2023 = episode_in(
        graph,
        "c1-2023",
        Y2023,
        text="(1) The rate is 120 dollars.",
        section=S61A,
        section_ids=(ids["c1-2023"], ids["c2-2020"]),
    )
    c2_2026 = episode_in(
        graph,
        "c2-2026",
        Y2026,
        text="(2) The threshold is 6,000 dollars.",
        section=S61A,
        section_ids=(ids["c1-2023"], ids["c2-2026"]),
    )
    unrelated = episode_in(
        graph,
        "suppliers-2020",
        Y2020,
        text="(2) The threshold is 5,000 dollars.",
        section=S61A,
        tool="supplier_docs",
    )
    for item in (c1_2020, c2_2020, c1_2023, unrelated):
        await save_episode(graph, item)
    await save_episode(graph, c2_2026, state="pending")
    await save_fact(
        graph,
        "threshold",
        [str(ids["c2-2020"])],
        fact="The threshold is 5,000 dollars",
        valid_at=Y2020,
    )
    await save_fact(
        graph,
        "supplier-threshold",
        [str(unrelated.episode_id)],
        fact="The threshold is 5,000 dollars",
        valid_at=Y2020,
    )
    await save_fact(
        graph,
        "rate-120",
        [str(ids["c1-2023"])],
        fact="The rate is 120 dollars",
        valid_at=Y2023,
    )

    closed = await backend()._close_changed_lines(
        graph.driver, graph.namespace, c2_2026
    )

    assert closed == 1
    state = await fact_state(graph, "threshold")
    assert (state.valid_at, state.invalid_at) == (Y2020, Y2026)
    assert (await fact_state(graph, "supplier-threshold")).open
    assert (await fact_state(graph, "rate-120")).open
