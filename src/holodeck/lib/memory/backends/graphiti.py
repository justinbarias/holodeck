"""The Graphiti memory backend (spec 043, design seams 3 and 5).

Write side. Each ``EpisodeIn`` becomes one Graphiti episode, exactly once, under the
content-hash uuid the service chose:

1. Load the episode node by uuid. If it is absent, save it as *pending*.
   If it is *complete*, skip it.
2. Run ``add_episode`` on that uuid.
3. Give each of the episode's facts that has no start date the start of the
   earliest episode that states it (Neo4j). An undated fact would count as
   true at every instant and could never be closed. A fact whose end is not
   after that start is left undated: dating it would invert its interval.
4. Check each fact ``add_episode`` closed against the source text, and
   reopen it when its section, or its line, did not change, when no edge
   the episode wrote could replace it, when it comes from another document,
   or when a later-dated episode states it (Neo4j; see
   ``graphiti_closure``). Skipped for an episode with no section id.
5. With ``close_on_text_change``, close each open fact of the older versions
   of the episode's section whose source line changed and that the new
   section text no longer states (Neo4j; ``decide_text_closure``). Runs
   once every other chunk of the new section is complete.
6. Load the node again and save it as *complete*.

One namespace must be ingested by one writer at a time. Nothing in HoloDeck
serialises it (the CLI, ``ingest_on_start`` and the Temporal activity each
call ``TemporalMemory.ingest`` with no lock). The closure check finds the
facts this episode closed through the episode node's ``entity_edges`` (the
edges Graphiti resolved or invalidated for it), so a concurrent writer's
closures are not taken for this episode's. Text closing and the reads of
older versions still assume no other writer changes the namespace meanwhile.

``backfill_fact_start_dates`` applies step 3 to a graph ingested before it.

The state lives in the node's ``name`` (``hd1:<state>:<json>``), because an
episode node stores only Graphiti's own nine fields. A crash at any point
leaves the episode pending, and the next run processes it again.

Read side. ``search`` applies the known-at rule (design seam 5) as two
filtered Graphiti searches merged by edge uuid, OPEN and CLOSED_LATER, for
every ``known_at``. ``timeline`` resolves its reference with
``entity_match.resolve_entities``; it and ``provenance`` read edges and the
episodes they name. Sources come from the episode ``name``, never from
``source_description``.

This module does not import ``graphiti_core`` at import time. Graphiti, its
driver and its clients are built on first use, once, so constructing the
backend needs neither the ``graphiti`` extra nor a reachable store.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from uuid import UUID

from holodeck.lib.extras import MissingExtraError, require_extra
from holodeck.lib.memory.backends.graphiti_closure import (
    EdgeShape,
    SectionKey,
    could_replace,
    decide_closure,
    decide_text_closure,
    same_section,
    section_lines,
)
from holodeck.lib.memory.backends.graphiti_secrets import credential_values, redact
from holodeck.lib.memory.entity_match import EntityCandidate, resolve_entities
from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryConfigError,
    TemporalMemoryError,
)
from holodeck.lib.memory.types import (
    EpisodeIn,
    EpisodeRef,
    FactHit,
    IngestReport,
    MemoryCapabilities,
    MemoryOperation,
    MemoryQuery,
    Provenance,
    ReferenceTimeSource,
    TimelineEntry,
    validate_namespace,
)
from holodeck.models.llm import LLMProvider
from holodeck.models.tool import (
    GraphBackendConfig,
    MemorySchemaHints,
    ReasoningEffort,
)

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EpisodicNode
    from graphiti_core.search.search_config import SearchConfig
    from graphiti_core.search.search_filters import SearchFilters
    from pydantic import BaseModel

    from holodeck.lib.memory.backends.graphiti_clients import (
        CountingLLMClient,
        GraphitiRuntime,
    )

logger = logging.getLogger(__name__)

NAME = "graphiti"
GRAPHITI_EXTRA = "graphiti"
GRAPHITI_FEATURE = "The graphiti temporal memory backend"

# The rows the spec's Graphiti capability table marks "Not supported" (MB-03).
UNSUPPORTED = frozenset(
    {
        MemoryOperation.SINGLE_VALUED_PREDICATES,
        MemoryOperation.SUPERSESSION_AUDIT,
        MemoryOperation.RETRACTION,
        MemoryOperation.REVIEW_QUEUE,
    }
)
SUPPORTED = frozenset(MemoryOperation) - UNSUPPORTED

# Before every date the store can hold, so a fact with no start sorts first.
_NO_START = datetime.min.replace(tzinfo=UTC)

EpisodeState = Literal["pending", "complete"]
NAME_VERSION = "hd1"
COMPLETE_PREFIX = f"{NAME_VERSION}:complete:"

COUNT_COMPLETE_QUERY = """
MATCH (e:Episodic)
WHERE e.group_id = $group_id AND e.name STARTS WITH $prefix
RETURN count(e) AS episode_count
"""

# A conditional update. It cannot create a node, so an episode that a purge
# removed during extraction is not brought back as complete with no facts.
MARK_COMPLETE_QUERY = """
MATCH (e:Episodic {uuid: $uuid})
WHERE e.group_id = $group_id
SET e.name = $name
RETURN count(e) AS updated
"""

# A fact with no start date counts as true at every instant (seam 5) and can
# never be closed (seam 3). Graphiti leaves ``valid_at`` null when the
# extraction model gives no date, so the backend dates such a fact by the
# document it came from: the earliest start of the episodes that state it.
# The episode just written is one of them, so a new fact gets that episode's
# reference time; ``$reference_time`` is the fallback when no episode node is
# found. A non-null ``valid_at`` is never changed. A fact whose ``invalid_at``
# is not after that start stays undated (``inverted``): its interval would
# end before it began.
DATE_EPISODE_FACTS_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id = $group_id AND r.valid_at IS NULL AND $uuid IN r.episodes
UNWIND r.episodes AS episode_uuid
OPTIONAL MATCH (ep:Episodic {uuid: episode_uuid})
WHERE ep.group_id = $group_id
WITH r, coalesce(min(ep.valid_at), $reference_time) AS first_seen
WITH r, first_seen, (r.invalid_at IS NULL OR r.invalid_at > first_seen) AS fits
FOREACH (ignored IN CASE WHEN fits THEN [1] ELSE [] END | SET r.valid_at = first_seen)
RETURN count(CASE WHEN fits THEN 1 END) AS dated,
       count(CASE WHEN NOT fits THEN 1 END) AS inverted
"""

# The same dating for a graph ingested before the rule above existed. Each
# undated fact gets the earliest start of its episodes in the namespace,
# under the same interval guard. A second run finds no undated fact it can
# date and changes nothing.
BACKFILL_FACT_START_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id = $group_id AND r.valid_at IS NULL
UNWIND r.episodes AS episode_uuid
MATCH (ep:Episodic {uuid: episode_uuid})
WHERE ep.group_id = $group_id AND ep.valid_at IS NOT NULL
WITH r, min(ep.valid_at) AS first_seen
WITH r, first_seen, (r.invalid_at IS NULL OR r.invalid_at > first_seen) AS fits
FOREACH (ignored IN CASE WHEN fits THEN [1] ELSE [] END | SET r.valid_at = first_seen)
RETURN count(CASE WHEN fits THEN 1 END) AS dated,
       count(CASE WHEN NOT fits THEN 1 END) AS inverted
"""

# A retry of a pending episode: ``add_episode`` replaces the node's
# ``entity_edges`` with the retry's edges. The earlier attempts' edges are
# added back, so the closures an attempt made before it failed are checked.
KEEP_EPISODE_EDGES_QUERY = """
MATCH (e:Episodic {uuid: $uuid})
WHERE e.group_id = $group_id
WITH e, coalesce(e.entity_edges, []) AS current
SET e.entity_edges = current + [u IN $edges WHERE NOT u IN current]
RETURN size(e.entity_edges) AS edges
"""

# The facts ``add_episode`` closed for this episode. Graphiti records no
# "closed by" link on a fact, but it lists every edge it resolved or
# invalidated for an episode in the episode node's ``entity_edges``. So a
# closure counts when the fact is in that list and was stamped at or after
# ``$since``: the first attempt's start, read with Graphiti's clock (it
# stamps ``expired_at`` with the same clock). The list excludes a concurrent
# writer's closures; the window excludes an old closed fact the episode
# restated.
CLOSED_SINCE_QUERY = """
MATCH (e:Episodic {uuid: $uuid})
WHERE e.group_id = $group_id
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.group_id = $group_id AND r.uuid IN coalesce(e.entity_edges, [])
  AND r.invalid_at IS NOT NULL
  AND r.expired_at IS NOT NULL AND r.expired_at >= $since
RETURN r.uuid AS fact_uuid, r.fact AS fact, r.episodes AS episodes,
       r.name AS name, s.uuid AS source_uuid, t.uuid AS target_uuid
"""

# The open facts an episode wrote or touched: the facts that could replace
# one it closed (the replacement guard in ``graphiti_closure``).
EPISODE_OPEN_FACTS_QUERY = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.group_id = $group_id AND $uuid IN r.episodes AND r.expired_at IS NULL
RETURN r.uuid AS fact_uuid, r.name AS name, s.uuid AS source_uuid,
       t.uuid AS target_uuid
"""

# Reopen facts the closure check found still stated by their source.
REOPEN_FACTS_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id = $group_id AND r.uuid IN $fact_uuids
SET r.invalid_at = null, r.expired_at = null
RETURN count(r) AS reopened
"""

# Text closing (``decide_text_closure``). The older complete episodes of a
# section: a pre-filter on the name, which holds the section keys as JSON
# (``encode_episode_name``); ``same_section`` and the document lineage
# (``_same_lineage``) decide in Python.
OLD_SECTION_EPISODES_QUERY = """
MATCH (e:Episodic)
WHERE e.group_id = $group_id AND e.valid_at < $reference_time
  AND e.name STARTS WITH $prefix
  AND (e.name CONTAINS $section_id_key OR e.name CONTAINS $section_number_key)
RETURN e.uuid AS uuid, e.name AS name
"""

# The open, dated facts the old episodes state and the new section does not.
# A fact that any chunk of the new section restated is still true, and so is
# a fact an episode dated after the new one states: an older version
# ingested after a newer one does not end what the newer one says.
TEXT_CLOSURE_CANDIDATES_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id = $group_id AND r.invalid_at IS NULL AND r.expired_at IS NULL
  AND r.valid_at IS NOT NULL AND r.valid_at < $reference_time
  AND any(u IN r.episodes WHERE u IN $old_uuids)
  AND none(u IN r.episodes WHERE u IN $new_uuids)
OPTIONAL MATCH (later:Episodic)
WHERE later.uuid IN r.episodes AND later.group_id = $group_id
  AND later.valid_at > $reference_time
WITH r, count(later) AS later_sources
WHERE later_sources = 0
RETURN r.uuid AS fact_uuid, r.fact AS fact
"""

# Close facts whose source line changed. An already-closed fact is not touched.
CLOSE_FACTS_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.group_id = $group_id AND r.uuid IN $fact_uuids
  AND r.invalid_at IS NULL AND r.expired_at IS NULL
SET r.invalid_at = $reference_time, r.expired_at = $now
RETURN count(r) AS closed
"""

# The candidate entities of a timeline: an exact uuid, or a name the reference
# contains in any case (which covers an equal name). This is a pre-filter
# only; ``entity_match.resolve_entities`` decides, so "s. 1" is not taken for
# "s. 10".
FIND_ENTITIES_QUERY = """
MATCH (n:Entity)
WHERE n.group_id = $group_id
  AND (n.uuid = $entity_ref
       OR (size(n.name) > 0 AND toLower($entity_ref) CONTAINS toLower(n.name)))
RETURN n.uuid AS uuid, n.name AS name
"""

T = TypeVar("T")


class EmbeddingService(Protocol):
    """What the backend needs from HoloDeck's embedding service."""

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in order."""
        ...


# --- Episode name codec ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeName:
    """The HoloDeck fields and the state kept in an episode node's ``name``.

    ``section`` is None for an episode written without a section id,
    including every episode written before section metadata existed.
    ``document_key`` is the document's lineage (``EpisodeIn.document_key``),
    empty when it has none.
    """

    state: EpisodeState
    source_tool: str
    chunk_id: str
    source_uri: str
    reference_time_source: ReferenceTimeSource
    section: SectionKey | None = None
    document_key: str = ""


def encode_episode_name(state: EpisodeState, episode: EpisodeIn) -> str:
    """Return ``hd1:<state>:<json>`` for ``episode``.

    The section keys are written only when the episode has a section id,
    and the document key only when it is set, so the name of an episode
    without them is what it was before they existed.
    """
    payload: dict[str, str | list[str]] = {
        "source_tool": episode.source_tool,
        "chunk_id": episode.chunk_id,
        "source_uri": episode.source_uri,
        "date_source": episode.reference_time_source,
    }
    if episode.section_id:
        payload["section_id"] = episode.section_id
        payload["section_number"] = episode.section_number
        payload["section_path"] = list(episode.section_path)
    if episode.document_key:
        payload["document_key"] = episode.document_key
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"{NAME_VERSION}:{state}:{body}"


def decode_episode_name(name: str) -> EpisodeName | None:
    """Read an episode name back. None means "not ours, not complete"."""
    parts = name.split(":", 2)
    if len(parts) != 3 or parts[0] != NAME_VERSION:
        return None
    state, body = parts[1], parts[2]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    source_tool = payload.get("source_tool")
    chunk_id = payload.get("chunk_id")
    source_uri = payload.get("source_uri")
    date_source = payload.get("date_source")
    if not (
        isinstance(source_tool, str)
        and isinstance(chunk_id, str)
        and isinstance(source_uri, str)
        and isinstance(date_source, str)
    ):
        return None
    if state == "pending":
        typed_state: EpisodeState = "pending"
    elif state == "complete":
        typed_state = "complete"
    else:
        return None
    if date_source == "front_matter":
        typed_source: ReferenceTimeSource = "front_matter"
    elif date_source == "mtime":
        typed_source = "mtime"
    elif date_source == "ingestion":
        typed_source = "ingestion"
    else:
        return None
    return EpisodeName(
        state=typed_state,
        source_tool=source_tool,
        chunk_id=chunk_id,
        source_uri=source_uri,
        reference_time_source=typed_source,
        section=_decode_section(payload),
        document_key=_decode_text(payload, "document_key"),
    )


def _decode_text(payload: dict[str, object], key: str) -> str:
    """An optional text field of a decoded name; "" when absent or malformed."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _decode_section(payload: dict[str, object]) -> SectionKey | None:
    """The section keys of a decoded name; None when absent or malformed.

    A malformed section does not make the episode foreign: it only makes
    its section unknown to the closure check.
    """
    section_id = payload.get("section_id")
    if not isinstance(section_id, str) or not section_id:
        return None
    number = payload.get("section_number")
    path = payload.get("section_path")
    path_items = path if isinstance(path, list) else []
    return SectionKey(
        section_id=section_id,
        number=number if isinstance(number, str) else "",
        path=tuple(item for item in path_items if isinstance(item, str)),
    )


def is_complete(name: str) -> bool:
    """True when ``name`` carries the complete marker.

    The prefix alone decides, which is the rule the count query uses, so
    ``has_episode`` and ``episode_count`` cannot disagree.
    """
    return name.startswith(COMPLETE_PREFIX)


def source_sentence(episode: EpisodeIn) -> str:
    """The plain sentence stored in ``source_description``.

    It is shown to the extraction model, so it is prose and not JSON.
    Provenance never parses it; it reads ``name``.
    """
    effective = episode.reference_time.date().isoformat()
    return f"Passage from {episode.source_uri}, in effect from {effective}."


# --- Errors ---------------------------------------------------------------


def require_graphiti() -> None:
    """Raise ``MemoryConfigError`` naming the install command (CFG-05).

    Also covers a partial install: a package the ``graphiti`` extra brings
    (for example ``anthropic``) that is missing when the clients are imported.
    """
    try:
        require_extra("graphiti_core", GRAPHITI_EXTRA, GRAPHITI_FEATURE)
        importlib.import_module("holodeck.lib.memory.backends.graphiti_clients")
    except MissingExtraError as error:
        raise MemoryConfigError(str(error)) from error
    except ModuleNotFoundError as error:
        missing = MissingExtraError(
            GRAPHITI_FEATURE, GRAPHITI_EXTRA, f"missing module '{error.name}'"
        )
        raise MemoryConfigError(str(missing)) from error


def _unreachable_error_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean the graph store cannot be reached."""
    from neo4j.exceptions import DatabaseUnavailable, DriverError

    # DriverError covers ServiceUnavailable and SessionExpired. A database
    # that went offline answers with DatabaseUnavailable.
    return (ConnectionError, DriverError, DatabaseUnavailable)


def _is_unreachable(error: BaseException) -> bool:
    return isinstance(error, _unreachable_error_types())


def _dates_facts(driver: GraphDriver) -> bool:
    """True for a store where the fact-dating and closure queries run.

    The queries match ``RELATES_TO`` relationships with a list of episodes,
    which is Neo4j's layout. Kuzu keeps a fact as a node and Neptune keeps
    the episodes as one string; FalkorDB is not verified.
    """
    from graphiti_core.driver.driver import GraphProvider

    return bool(driver.provider == GraphProvider.NEO4J)


def _describe(error: BaseException) -> str:
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


# --- Read-side helpers ----------------------------------------------------


def _utc(value: datetime) -> datetime:
    """A stored instant as aware UTC. A naive value is taken as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _utc_or_none(value: datetime | None) -> datetime | None:
    return None if value is None else _utc(value)


def _search_config(query: MemoryQuery) -> SearchConfig:
    """Edges only: full-text, similarity and breadth-first, fused by RRF.

    Built per call: Graphiti's recipe objects are shared and ``search``
    changes them. RRF needs no model; the cross-encoder refuses every call.
    """
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )

    return SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[
                EdgeSearchMethod.bm25,
                EdgeSearchMethod.cosine_similarity,
                EdgeSearchMethod.bfs,
            ],
            reranker=EdgeReranker.rrf,
            bfs_max_depth=query.hops,
        ),
        limit=query.limit,
    )


def _known_at_filters(
    valid_at: datetime, known_at: datetime, *, closed_later: bool
) -> SearchFilters:
    """The OPEN or CLOSED_LATER filter of design seam 5.

    Fields are joined by AND and groups inside a field by OR. A dated
    comparison never matches null, so each nullable field has an
    ``IS NULL`` group, and no field has more than one dated group.
    """
    from graphiti_core.search.search_filters import (
        ComparisonOperator,
        DateFilter,
        SearchFilters,
    )

    def dated(op: ComparisonOperator, instant: datetime) -> list[DateFilter]:
        return [DateFilter(date=instant, comparison_operator=op)]

    is_null = [DateFilter(comparison_operator=ComparisonOperator.is_null)]
    recorded = [dated(ComparisonOperator.less_than_equal, known_at)]
    started = [dated(ComparisonOperator.less_than_equal, valid_at), is_null]
    if closed_later:
        return SearchFilters(
            created_at=recorded,
            valid_at=started,
            expired_at=[dated(ComparisonOperator.greater_than, known_at)],
        )
    return SearchFilters(
        created_at=recorded,
        valid_at=started,
        invalid_at=[dated(ComparisonOperator.greater_than, valid_at), is_null],
    )


def _known_at_rule(edge: EntityEdge, valid_at: datetime, known_at: datetime) -> bool:
    """The known-at rule of design seam 5, on one edge."""
    if _utc(edge.created_at) > known_at:
        return False
    start = _utc_or_none(edge.valid_at)
    if start is not None and start > valid_at:
        return False
    end = _utc_or_none(edge.invalid_at)
    expired = _utc_or_none(edge.expired_at)
    return end is None or end > valid_at or (expired is not None and expired > known_at)


def _as_believed_at(
    edge: EntityEdge, known_at: datetime
) -> tuple[datetime | None, datetime | None]:
    """``valid_to`` and ``expired_at`` as the store held them at ``known_at``.

    A closing recorded after ``known_at`` was not known then: the fact is
    returned open, with neither an end nor an expiry.
    """
    expired = _utc_or_none(edge.expired_at)
    if expired is not None and expired > known_at:
        return None, None
    return _utc_or_none(edge.invalid_at), expired


def _episode_ref(
    node: EpisodicNode, namespace: str, *, with_text: bool
) -> EpisodeRef | None:
    """A complete HoloDeck episode as an ``EpisodeRef``; None for any other."""
    if node.group_id != namespace or not is_complete(node.name):
        return None
    decoded = decode_episode_name(node.name)
    if decoded is None:
        return None
    try:
        episode_id = UUID(node.uuid)
    except ValueError:
        return None
    return EpisodeRef(
        episode_id=episode_id,
        source_tool=decoded.source_tool,
        chunk_id=decoded.chunk_id,
        source_uri=decoded.source_uri,
        reference_time=_utc(node.valid_at),
        reference_time_source=decoded.reference_time_source,
        text=node.content if with_text else None,
    )


def _sources(edge: EntityEdge, refs: dict[str, EpisodeRef]) -> tuple[EpisodeRef, ...]:
    """The edge's episodes that are known, in the edge's order, once each."""
    return tuple(refs[uuid] for uuid in dict.fromkeys(edge.episodes) if uuid in refs)


# --- Backend --------------------------------------------------------------


class GraphitiBackend:
    """``MemoryBackend`` over Graphiti."""

    def __init__(
        self,
        config: GraphBackendConfig,
        extraction_model: LLMProvider,
        embedding_service: EmbeddingService,
        schema_hints: MemorySchemaHints | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        close_on_text_change: bool = False,
    ) -> None:
        self._config = config
        self._extraction_model = extraction_model
        self._embedding_service = embedding_service
        self._schema_hints = schema_hints or MemorySchemaHints()
        self._reasoning_effort = reasoning_effort
        self._close_on_text_change = close_on_text_change
        self._capabilities = MemoryCapabilities(backend=NAME, supported=SUPPORTED)
        self._graphiti: Graphiti | None = None
        self._llm: CountingLLMClient | None = None
        self._entity_models: dict[str, type[BaseModel]] | None = None
        self._edge_models: dict[str, type[BaseModel]] | None = None
        self._edge_type_map: dict[tuple[str, str], list[str]] | None = None
        self._scoped: dict[str, GraphDriver] = {}
        self._indices_built = False
        self._lock = asyncio.Lock()
        self._orphans: set[asyncio.Task[None]] = set()
        self._undated_warned = False
        self._unsectioned_logged = False

    @property
    def name(self) -> str:
        return NAME

    def capabilities(self) -> MemoryCapabilities:
        return self._capabilities

    # --- lifecycle ------------------------------------------------------

    async def _graph(self) -> Graphiti:
        """Build Graphiti, its driver and its clients once."""
        async with self._lock:
            if self._graphiti is None:
                require_graphiti()
                from holodeck.lib.memory.backends import graphiti_clients

                # In a thread: the Kuzu driver opens its database and runs
                # schema queries in its constructor. A thread cannot be
                # stopped, so a cancelled caller hands the build to a task
                # that closes the runtime when it is ready.
                build = asyncio.ensure_future(
                    asyncio.to_thread(
                        graphiti_clients.build_graphiti,
                        self._config,
                        self._extraction_model,
                        self._embedding_service,
                        self._reasoning_effort,
                    )
                )
                try:
                    runtime = await asyncio.shield(build)
                except asyncio.CancelledError:
                    self._close_when_built(build)
                    raise
                except TemporalMemoryError:
                    raise
                except Exception as error:  # noqa: BLE001 - no driver type leaves
                    detail = self._redact(_describe(error))
                    raise MemoryConfigError(
                        f"the graphiti backend could not be built: {detail}"
                    ) from error
                hints = self._schema_hints
                self._entity_models = (
                    graphiti_clients.entity_type_models(hints.entity_type_specs)
                    if hints.entity_types
                    else None
                )
                self._edge_models = (
                    graphiti_clients.edge_type_models(hints.edge_types)
                    if hints.edge_types
                    else None
                )
                # Unset: Graphiti allows every edge type between any entities.
                self._edge_type_map = (
                    graphiti_clients.edge_type_map(hints.edge_type_map)
                    if hints.edge_type_map
                    else None
                )
                self._graphiti = runtime.graphiti
                self._llm = runtime.llm
            return self._graphiti

    async def _ready(self) -> Graphiti:
        """Graphiti with indices and constraints built, once per instance.

        Writes need the constraints; search needs the full-text index.
        """
        graphiti = await self._graph()
        async with self._lock:
            if not self._indices_built:
                await self._store(
                    "build indices", graphiti.build_indices_and_constraints()
                )
                self._indices_built = True
        return graphiti

    def _driver(self, graphiti: Graphiti, namespace: str) -> GraphDriver:
        """The driver Graphiti uses for ``namespace``.

        Graphiti clones the driver per group: the same driver on Neo4j, one
        graph per group on FalkorDB. Pending nodes, counts and purges must
        use the same graph that ``add_episode`` writes to.
        """
        driver = self._scoped.get(namespace)
        if driver is None:
            driver = graphiti.driver.clone(database=namespace)
            self._scoped[namespace] = driver
        return driver

    def _close_when_built(self, build: asyncio.Future[GraphitiRuntime]) -> None:
        """Close the runtime of a build whose caller was cancelled.

        The task is kept in ``_orphans`` so ``close`` can wait for it.
        """

        async def close_runtime() -> None:
            try:
                runtime = await build
            except Exception:  # noqa: BLE001 - nothing was built, nothing to close
                return
            try:
                await runtime.graphiti.close()
            except Exception as error:  # noqa: BLE001 - logged; the caller is gone
                logger.warning(
                    "graphiti runtime built after cancellation did not close: %s",
                    type(error).__name__,
                )

        task = asyncio.get_running_loop().create_task(close_runtime())
        self._orphans.add(task)
        task.add_done_callback(self._orphans.discard)

    async def close(self) -> None:
        # The lock waits for a build in progress, so its runtime is closed here.
        async with self._lock:
            graphiti = self._graphiti
            self._graphiti = None
            self._llm = None
            self._scoped.clear()
            self._indices_built = False
        orphans = list(self._orphans)
        if orphans:
            await asyncio.gather(*orphans)
        if graphiti is not None:
            await self._store("close", graphiti.close())

    # --- store access ---------------------------------------------------

    def _redact(self, text: str) -> str:
        """Remove credential values from text that leaves the backend."""
        return redact(text, credential_values(self._extraction_model, self._config))

    async def _store(self, operation: str, call: Awaitable[T]) -> T:
        """Await a store call; no driver exception type leaves the backend.

        Only a store that cannot be reached is "unavailable", which a search
        answers with a degraded result. Any other failure (a query error, a
        dimension mismatch, a bug) is a ``TemporalMemoryError``: it must not
        be hidden as a degraded search.
        """
        try:
            return await call
        except TemporalMemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - no driver type leaves
            detail = f"{operation} failed: {self._redact(_describe(error))}"
            if _is_unreachable(error):
                raise MemoryBackendUnavailableError(NAME, detail) from error
            raise TemporalMemoryError(f"Memory backend '{NAME}': {detail}") from error

    async def _load(self, driver: GraphDriver, episode_id: UUID) -> EpisodicNode | None:
        from graphiti_core.nodes import EpisodicNode

        nodes = await self._store(
            "load episode", EpisodicNode.get_by_uuids(driver, [str(episode_id)])
        )
        return nodes[0] if nodes else None

    # --- port: write side -----------------------------------------------

    async def has_episode(self, namespace: str, episode_id: UUID) -> bool:
        validate_namespace(namespace)
        graphiti = await self._graph()
        node = await self._load(self._driver(graphiti, namespace), episode_id)
        return (
            node is not None and node.group_id == namespace and is_complete(node.name)
        )

    async def episode_count(self, namespace: str) -> int:
        validate_namespace(namespace)
        graphiti = await self._graph()
        driver = self._driver(graphiti, namespace)
        result = await self._store(
            "count episodes",
            driver.execute_query(
                COUNT_COMPLETE_QUERY, group_id=namespace, prefix=COMPLETE_PREFIX
            ),
        )
        records = result[0] if result else []
        if not records:
            return 0
        value = records[0]["episode_count"]
        if not isinstance(value, int):
            raise MemoryBackendUnavailableError(
                NAME, f"count episodes returned {type(value).__name__}, not int"
            )
        return value

    async def ingest(
        self,
        namespace: str,
        episodes: Sequence[EpisodeIn],
        instructions: str,
    ) -> IngestReport:
        self._capabilities.require(MemoryOperation.INGEST)
        validate_namespace(namespace)
        graphiti = await self._ready()
        driver = self._driver(graphiti, namespace)
        meter = _Meter.start(self._llm, graphiti)
        tally = _ClosureTally()
        written: list[UUID] = []
        for episode in episodes:
            outcome = await self._ingest_one(
                graphiti, driver, namespace, episode, instructions, tally
            )
            if isinstance(outcome, _Failure):
                return meter.report(
                    written, tally, failed=episode.episode_id, error=outcome.error
                )
            if outcome == "written":
                written.append(episode.episode_id)
        return meter.report(written, tally)

    async def _ingest_one(
        self,
        graphiti: Graphiti,
        driver: GraphDriver,
        namespace: str,
        episode: EpisodeIn,
        instructions: str,
        tally: _ClosureTally,
    ) -> Literal["written", "skipped"] | _Failure:
        """Ingest one episode, or say why it failed.

        Closure-check counts are added to ``tally``.
        """
        from graphiti_core.nodes import EpisodeType, EpisodicNode
        from graphiti_core.utils.datetime_utils import utc_now

        node = await self._load(driver, episode.episode_id)
        if node is not None and node.group_id != namespace:
            return _Failure(
                f"episode {episode.episode_id} exists in another namespace "
                f"('{node.group_id}')"
            )
        if node is not None and is_complete(node.name):
            return "skipped"
        if node is not None and (
            decode_episode_name(node.name) is None or node.content != episode.text
        ):
            # The id is a hash of the text. A node under it with another
            # name or body was not written by this backend, and Graphiti
            # would extract that node's body, not the submitted text.
            return _Failure(
                f"episode {episode.episode_id} is taken by a node that holds "
                "other content"
            )
        description = source_sentence(episode)
        context = await self._context(driver, namespace, episode)
        # The closure check covers every attempt: ``since`` is the first
        # attempt's start (the pending node's ``created_at``, which Graphiti
        # keeps), and the edges earlier attempts recorded are kept.
        earlier_edges: list[str] = []
        if node is not None:
            since = _utc(node.created_at)
            earlier_edges = list(node.entity_edges)
        else:
            pending = EpisodicNode(
                uuid=str(episode.episode_id),
                name=encode_episode_name("pending", episode),
                group_id=namespace,
                labels=[],
                source=EpisodeType.text,
                source_description=description,
                content=episode.text,
                # Graphiti's clock: it stamps the facts it closes with it.
                created_at=utc_now(),
                valid_at=episode.reference_time,
            )
            since = pending.created_at
            await self._store("save pending episode", pending.save(driver))

        try:
            await graphiti.add_episode(
                name=encode_episode_name("pending", episode),
                episode_body=episode.text,
                source_description=description,
                reference_time=episode.reference_time,
                source=EpisodeType.text,
                group_id=namespace,
                uuid=str(episode.episode_id),
                entity_types=self._entity_models,
                edge_types=self._edge_models,
                edge_type_map=self._edge_type_map,
                previous_episode_uuids=context,
                custom_extraction_instructions=instructions,
            )
            if earlier_edges and _dates_facts(driver):
                await driver.execute_query(
                    KEEP_EPISODE_EDGES_QUERY,
                    uuid=str(episode.episode_id),
                    group_id=namespace,
                    edges=earlier_edges,
                )
            # Before the marker: a failure here leaves the episode pending,
            # so the next run dates its facts again.
            await self._date_episode_facts(driver, namespace, episode)
            checked = await self._check_closures(driver, namespace, episode, since)
            if self._close_on_text_change and _dates_facts(driver):
                checked.closed_by_text = await self._close_changed_lines(
                    driver, namespace, episode
                )
            if _dates_facts(driver) and episode.section_id:
                self._log_closures(episode, checked)
            tally.add(checked)
            result = await driver.execute_query(
                MARK_COMPLETE_QUERY,
                uuid=str(episode.episode_id),
                group_id=namespace,
                name=encode_episode_name("complete", episode),
            )
            records = result[0] if result else []
            if not records or records[0]["updated"] != 1:
                return _Failure(
                    f"episode {episode.episode_id} was removed while it was "
                    "being extracted"
                )
        except Exception as error:  # noqa: BLE001 - reported, not raised (ING-05)
            if _is_unreachable(error):
                raise MemoryBackendUnavailableError(
                    NAME, self._redact(_describe(error))
                ) from error
            logger.warning(
                "graphiti episode %s failed: %s",
                episode.episode_id,
                type(error).__name__,
            )
            return _Failure(self._redact(str(error) or type(error).__name__))
        return "written"

    async def _date_episode_facts(
        self, driver: GraphDriver, namespace: str, episode: EpisodeIn
    ) -> None:
        """Give the episode's undated facts a start date (one Cypher write).

        Raises like ``add_episode``: the caller reports the failure.
        """
        if not _dates_facts(driver):
            if not self._undated_warned:
                self._undated_warned = True
                logger.warning(
                    "graphiti facts with no start date are left undated on %s; "
                    "dating is implemented for neo4j only",
                    driver.provider.value,
                )
            return
        result = await driver.execute_query(
            DATE_EPISODE_FACTS_QUERY,
            uuid=str(episode.episode_id),
            group_id=namespace,
            reference_time=episode.reference_time,
        )
        inverted = _count(result, "inverted")
        if inverted:
            logger.warning(
                "graphiti episode %s: %d facts left undated; their end is not "
                "after the start their episodes give",
                episode.episode_id,
                inverted,
            )

    async def _check_closures(
        self,
        driver: GraphDriver,
        namespace: str,
        episode: EpisodeIn,
        since: datetime,
    ) -> _ClosureTally:
        """Reopen the facts ``add_episode`` closed whose source is unchanged.

        Rules in ``graphiti_closure``. Neo4j only, like the dating; an
        episode with no section id is not checked. Raises like
        ``add_episode``: the caller reports the failure, and the episode
        stays pending.
        """
        if not _dates_facts(driver):
            return _ClosureTally()
        if not episode.section_id:
            if not self._unsectioned_logged:
                self._unsectioned_logged = True
                logger.info(
                    "graphiti closure check skipped: episodes carry no section "
                    "id, so closed facts are left as the model closed them"
                )
            return _ClosureTally()
        own = str(episode.episode_id)
        result = await driver.execute_query(
            CLOSED_SINCE_QUERY, group_id=namespace, uuid=own, since=since
        )
        closed = _closed_facts(result[0] if result else [])
        if not closed:
            return _ClosureTally()
        from graphiti_core.nodes import EpisodicNode

        wanted = [str(i) for i in episode.section_episode_ids if str(i) != own]
        wanted += [uuid for fact in closed for uuid in fact.episodes if uuid != own]
        nodes = await EpisodicNode.get_by_uuids(driver, list(dict.fromkeys(wanted)))
        by_uuid = {node.uuid: node for node in nodes if node.group_id == namespace}
        new_section = SectionKey(
            episode.section_id, episode.section_number, episode.section_path
        )
        new_texts = _section_texts(episode, by_uuid)
        written = await driver.execute_query(
            EPISODE_OPEN_FACTS_QUERY, group_id=namespace, uuid=own
        )
        replacements = _edge_shapes(written[0] if written else [])
        reopen: list[str] = []
        no_replacement = 0
        for fact in closed:
            sources = [
                by_uuid[uuid]
                for uuid in dict.fromkeys(fact.episodes)
                if uuid != own and uuid in by_uuid
            ]
            if not sources:
                # A new fact closed on arrival, or a source outside the
                # namespace: nothing to compare, so the closure stands.
                continue
            origin = min(sources, key=lambda node: (_utc(node.valid_at), node.uuid))
            origin_name = decode_episode_name(origin.name)
            old_section = origin_name.section if origin_name else None
            decision = decide_closure(
                fact=fact.fact,
                old_section=old_section,
                old_texts=_old_texts(origin, origin_name, sources),
                new_section=new_section,
                new_texts=new_texts,
                replaced=any(could_replace(fact.shape, e) for e in replacements),
                restated_later=any(
                    _utc(node.valid_at) > _utc(episode.reference_time)
                    for node in sources
                ),
                other_document=(
                    origin_name is not None and not _same_lineage(origin_name, episode)
                ),
            )
            if decision.reopen:
                reopen.append(fact.uuid)
                if decision.reason == "no_replacement":
                    no_replacement += 1
        if reopen:
            await driver.execute_query(
                REOPEN_FACTS_QUERY, group_id=namespace, fact_uuids=reopen
            )
        tally = _ClosureTally(
            closed=len(closed),
            reopened=len(reopen),
            kept=len(closed) - len(reopen),
            reopened_no_replacement=no_replacement,
        )
        return tally

    async def _close_changed_lines(
        self, driver: GraphDriver, namespace: str, episode: EpisodeIn
    ) -> int:
        """Close open facts of the section's older versions whose line changed.

        Rules in ``graphiti_closure.decide_text_closure``. The older versions
        are every complete episode dated before the episode's, of the same
        section (``same_section``) and the same document lineage
        (``_same_lineage``). All earlier dates count, not only the latest:
        a chunk left unchanged by an intermediate version keeps its first
        episode and date. A fact from a line that an intermediate version
        removed was closed then, so taking the union is safe. Runs only when
        every other chunk of the new section is complete, so a line that
        moved to a later chunk is not taken for a removed line. Returns how
        many facts were closed. Raises like ``add_episode``: the caller
        reports the failure.
        """
        from graphiti_core.nodes import EpisodicNode
        from graphiti_core.utils.datetime_utils import utc_now

        if not episode.section_id:
            return 0
        own = str(episode.episode_id)
        new_ids = list(
            dict.fromkeys([own, *(str(i) for i in episode.section_episode_ids)])
        )
        others = [uuid for uuid in new_ids if uuid != own]
        siblings = await EpisodicNode.get_by_uuids(driver, others) if others else []
        by_uuid = {node.uuid: node for node in siblings if node.group_id == namespace}
        # A pending sibling has not finished; its facts may not be written.
        if any(
            uuid not in by_uuid or not is_complete(by_uuid[uuid].name)
            for uuid in others
        ):
            return 0
        new_section = SectionKey(
            episode.section_id, episode.section_number, episode.section_path
        )
        result = await driver.execute_query(
            OLD_SECTION_EPISODES_QUERY,
            group_id=namespace,
            reference_time=episode.reference_time,
            prefix=COMPLETE_PREFIX,
            section_id_key=_name_key("section_id", episode.section_id),
            section_number_key=(
                _name_key("section_number", episode.section_number)
                if episode.section_number
                else _name_key("section_id", episode.section_id)
            ),
        )
        candidates = [
            uuid
            for uuid, name in _named_rows(result[0] if result else [])
            if uuid not in new_ids
            and name.section is not None
            and same_section(name.section, new_section)
            and _same_lineage(name, episode)
        ]
        if not candidates:
            return 0
        old_nodes = sorted(
            (
                node
                for node in await EpisodicNode.get_by_uuids(driver, candidates)
                if node.group_id == namespace
                and _utc(node.valid_at) < _utc(episode.reference_time)
            ),
            key=_chunk_order,
        )
        if not old_nodes:
            return 0
        found = await driver.execute_query(
            TEXT_CLOSURE_CANDIDATES_QUERY,
            group_id=namespace,
            reference_time=episode.reference_time,
            old_uuids=[node.uuid for node in old_nodes],
            new_uuids=new_ids,
        )
        facts = _open_facts(found[0] if found else [])
        if not facts:
            return 0
        old_lines = section_lines([node.content for node in old_nodes])
        new_text = "\n".join(_section_texts(episode, by_uuid))
        close = [
            uuid
            for uuid, fact in facts
            if decide_text_closure(
                fact=fact, old_lines=old_lines, new_text=new_text
            ).close
        ]
        if not close:
            return 0
        written = await driver.execute_query(
            CLOSE_FACTS_QUERY,
            group_id=namespace,
            fact_uuids=close,
            reference_time=episode.reference_time,
            now=utc_now(),
        )
        records = written[0] if written else []
        value = records[0]["closed"] if records else 0
        return value if isinstance(value, int) else 0

    @staticmethod
    def _log_closures(episode: EpisodeIn, tally: _ClosureTally) -> None:
        logger.info(
            "graphiti closure check, episode %s: %d closed by extraction, "
            "%d reopened (%d with no replacement), %d kept closed, "
            "%d closed by text change",
            episode.episode_id,
            tally.closed,
            tally.reopened,
            tally.reopened_no_replacement,
            tally.kept,
            tally.closed_by_text,
        )

    async def backfill_fact_start_dates(self, namespace: str) -> int:
        """Date every undated fact in ``namespace``; return how many changed.

        For a graph ingested before ingestion dated its facts. Each fact
        with a null ``valid_at`` gets the earliest ``valid_at`` of the
        episodes that state it. Idempotent: a second call returns 0.
        Not part of the ``MemoryBackend`` port.
        """
        validate_namespace(namespace)
        graphiti = await self._graph()
        driver = self._driver(graphiti, namespace)
        if not _dates_facts(driver):
            raise TemporalMemoryError(
                f"Memory backend '{NAME}': dating facts is implemented for "
                f"neo4j only, not {driver.provider.value}"
            )
        result = await self._store(
            "backfill fact start dates",
            driver.execute_query(BACKFILL_FACT_START_QUERY, group_id=namespace),
        )
        records = result[0] if result else []
        if not records:
            return 0
        value = records[0]["dated"]
        if not isinstance(value, int):
            raise TemporalMemoryError(
                f"Memory backend '{NAME}': backfill returned "
                f"{type(value).__name__}, not int"
            )
        inverted = _count(result, "inverted")
        if inverted:
            logger.warning(
                "graphiti backfill in %s: %d facts left undated; their end is "
                "not after the start their episodes give",
                namespace,
                inverted,
            )
        return value

    async def _context(
        self, driver: GraphDriver, namespace: str, episode: EpisodeIn
    ) -> list[str]:
        """The context episodes that exist in ``namespace``, in order.

        Graphiti loads context by uuid with no group filter, so an id from
        another namespace would put that namespace's text in the prompt.
        """
        wanted = [str(i) for i in episode.context_episode_ids]
        if not wanted:
            return []
        from graphiti_core.nodes import EpisodicNode

        nodes = await self._store(
            "load context episodes", EpisodicNode.get_by_uuids(driver, wanted)
        )
        allowed = {node.uuid for node in nodes if node.group_id == namespace}
        return [uuid for uuid in wanted if uuid in allowed]

    async def purge(self, namespace: str) -> None:
        self._capabilities.require(MemoryOperation.PURGE)
        validate_namespace(namespace)
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        graphiti = await self._graph()
        driver = self._driver(graphiti, namespace)
        # group_ids must never be None here: clear_data then deletes every node.
        await self._store("purge", clear_data(driver, group_ids=[namespace]))

    # --- port: read side -----------------------------------------------

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        self._capabilities.require(MemoryOperation.SEARCH)
        if query.seed_episode_ids:
            self._capabilities.require(MemoryOperation.EXPAND)
        validate_namespace(namespace)
        graphiti = await self._ready()
        driver = self._driver(graphiti, namespace)
        seeds = await self._seed_entities(driver, namespace, query.seed_episode_ids)
        config = _search_config(query)
        valid_at, known_at = query.valid_at, query.known_at
        # Two searches: facts open at V, and closed facts that were still open
        # at K. The second runs for every K, the present included: ingestion
        # can close a fact between the caller reading its clock and this call.
        passes = [False, True]
        results = await asyncio.gather(
            *(
                self._store(
                    "search",
                    graphiti.search_(
                        query.text,
                        config=config,
                        group_ids=[namespace],
                        # None, not []: with no seed Graphiti expands from
                        # the sources of its own first results.
                        bfs_origin_node_uuids=seeds or None,
                        search_filter=_known_at_filters(
                            valid_at, known_at, closed_later=closed_later
                        ),
                        driver=driver,
                    ),
                )
                for closed_later in passes
            )
        )
        best: dict[str, tuple[EntityEdge, float]] = {}
        for result in results:
            for edge, score in zip(
                result.edges, result.edge_reranker_scores, strict=False
            ):
                # The filters already hold the rule; a driver with its own
                # search interface may not, so the rule is checked again.
                if edge.group_id != namespace or not _known_at_rule(
                    edge, valid_at, known_at
                ):
                    continue
                kept = best.get(edge.uuid)
                if kept is None or score > kept[1]:
                    best[edge.uuid] = (edge, score)
        ranked = sorted(best.values(), key=lambda pair: (-pair[1], pair[0].uuid))
        ranked = ranked[: query.limit]
        refs = await self._episodes(
            driver,
            namespace,
            (uuid for edge, _ in ranked for uuid in edge.episodes),
            with_text=False,
        )
        hits: list[FactHit] = []
        for edge, score in ranked:
            valid_to, expired_at = _as_believed_at(edge, known_at)
            hits.append(
                FactHit(
                    fact_id=edge.uuid,
                    text=edge.fact,
                    valid_from=_utc_or_none(edge.valid_at),
                    valid_to=valid_to,
                    recorded_at=_utc(edge.created_at),
                    expired_at=expired_at,
                    sources=_sources(edge, refs),
                    score=float(score),
                    # Graphiti does not record which edge closed which.
                    superseded_by=None,
                )
            )
        return hits

    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]:
        self._capabilities.require(MemoryOperation.TIMELINE)
        validate_namespace(namespace)
        from graphiti_core.edges import EntityEdge

        graphiti = await self._graph()
        driver = self._driver(graphiti, namespace)
        result = await self._store(
            "find entity",
            driver.execute_query(
                FIND_ENTITIES_QUERY, group_id=namespace, entity_ref=entity_ref
            ),
        )
        records = result[0] if result else []
        candidates = [
            EntityCandidate(entity_id=uuid, name=name)
            for uuid, name in ((r["uuid"], r["name"]) for r in records)
            if isinstance(uuid, str) and isinstance(name, str)
        ]
        entity_uuids = resolve_entities(entity_ref, candidates)
        edges: dict[str, EntityEdge] = {}
        for entity_uuid in entity_uuids:
            found = await self._store(
                "load entity facts", EntityEdge.get_by_node_uuid(driver, entity_uuid)
            )
            for edge in found:
                if edge.group_id == namespace:
                    edges.setdefault(edge.uuid, edge)
        refs = await self._episodes(
            driver,
            namespace,
            (uuid for edge in edges.values() for uuid in edge.episodes),
            with_text=False,
        )
        entries = [
            TimelineEntry(
                fact_id=edge.uuid,
                text=edge.fact,
                valid_from=_utc_or_none(edge.valid_at),
                valid_to=_utc_or_none(edge.invalid_at),
                recorded_at=_utc(edge.created_at),
                expired_at=_utc_or_none(edge.expired_at),
                sources=_sources(edge, refs),
            )
            for edge in edges.values()
        ]
        entries.sort(key=lambda entry: (entry.valid_from or _NO_START, entry.fact_id))
        return entries

    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None:
        self._capabilities.require(MemoryOperation.PROVENANCE)
        validate_namespace(namespace)
        from graphiti_core.edges import EntityEdge

        graphiti = await self._graph()
        driver = self._driver(graphiti, namespace)
        found = await self._store(
            "load fact", EntityEdge.get_by_uuids(driver, [fact_id])
        )
        edge = next(
            (e for e in found if e.uuid == fact_id and e.group_id == namespace), None
        )
        if edge is None:
            return None
        refs = await self._episodes(driver, namespace, edge.episodes, with_text=True)
        return Provenance(
            fact_id=edge.uuid,
            text=edge.fact,
            valid_from=_utc_or_none(edge.valid_at),
            valid_to=_utc_or_none(edge.invalid_at),
            recorded_at=_utc(edge.created_at),
            expired_at=_utc_or_none(edge.expired_at),
            sources=_sources(edge, refs),
        )

    async def _seed_entities(
        self, driver: GraphDriver, namespace: str, episode_ids: Sequence[UUID]
    ) -> list[str]:
        """Uuids of the entities the complete seed episodes mention (RET-09).

        One episode lookup and one mention lookup; no model call.
        """
        if not episode_ids:
            return []
        from graphiti_core.nodes import EpisodicNode
        from graphiti_core.search import search_utils

        nodes = await self._store(
            "load seed episodes",
            EpisodicNode.get_by_uuids(driver, [str(i) for i in episode_ids]),
        )
        episodes = [
            node
            for node in nodes
            if node.group_id == namespace and is_complete(node.name)
        ]
        if not episodes:
            return []
        entities = await self._store(
            "load seed entities", search_utils.get_mentioned_nodes(driver, episodes)
        )
        seeds: list[str] = []
        for entity in entities:
            if entity.group_id == namespace and entity.uuid not in seeds:
                seeds.append(entity.uuid)
        return seeds

    async def _episodes(
        self,
        driver: GraphDriver,
        namespace: str,
        uuids: Iterable[str],
        *,
        with_text: bool,
    ) -> dict[str, EpisodeRef]:
        """The complete HoloDeck episodes among ``uuids``, by uuid, in one lookup."""
        wanted = list(dict.fromkeys(uuids))
        if not wanted:
            return {}
        from graphiti_core.nodes import EpisodicNode

        nodes = await self._store(
            "load episodes", EpisodicNode.get_by_uuids(driver, wanted)
        )
        refs: dict[str, EpisodeRef] = {}
        for node in nodes:
            ref = _episode_ref(node, namespace, with_text=with_text)
            if ref is not None:
                refs[node.uuid] = ref
        return refs


@dataclass(frozen=True, slots=True)
class _ClosedFact:
    """A fact ``add_episode`` closed, as the window query returns it."""

    uuid: str
    fact: str
    episodes: tuple[str, ...]
    shape: EdgeShape


def _edge_shape(record: Mapping[str, object]) -> EdgeShape | None:
    """The endpoints and name of a row, or None when one is missing."""
    source = record["source_uuid"]
    target = record["target_uuid"]
    name = record["name"]
    if not (
        isinstance(source, str) and isinstance(target, str) and isinstance(name, str)
    ):
        return None
    return EdgeShape(source_uuid=source, target_uuid=target, name=name)


def _edge_shapes(records: Sequence[Mapping[str, object]]) -> list[EdgeShape]:
    """Typed rows of ``EPISODE_OPEN_FACTS_QUERY``; a malformed row is left out."""
    return [shape for record in records if (shape := _edge_shape(record)) is not None]


def _closed_facts(records: Sequence[Mapping[str, object]]) -> list[_ClosedFact]:
    """Typed rows of ``CLOSED_SINCE_QUERY``; a malformed row is left out."""
    facts: list[_ClosedFact] = []
    for record in records:
        uuid = record["fact_uuid"]
        fact = record["fact"]
        episodes = record["episodes"]
        shape = _edge_shape(record)
        if not (
            isinstance(uuid, str)
            and isinstance(fact, str)
            and isinstance(episodes, list)
            and shape is not None
        ):
            continue
        facts.append(
            _ClosedFact(
                uuid=uuid,
                fact=fact,
                episodes=tuple(e for e in episodes if isinstance(e, str)),
                shape=shape,
            )
        )
    return facts


def _section_texts(
    episode: EpisodeIn, by_uuid: Mapping[str, EpisodicNode]
) -> list[str]:
    """The new section's chunk texts in document order.

    The episode's own text comes from the episode; the other chunks from
    their nodes. A chunk with no node yet (later in the document) is absent.
    """
    ids = episode.section_episode_ids or (episode.episode_id,)
    if episode.episode_id not in ids:
        ids = (*ids, episode.episode_id)
    texts: list[str] = []
    for episode_id in ids:
        if episode_id == episode.episode_id:
            texts.append(episode.text)
        elif (node := by_uuid.get(str(episode_id))) is not None:
            texts.append(node.content)
    return texts


def _name_key(key: str, value: str) -> str:
    """``"key":"value"`` as ``encode_episode_name`` writes it."""
    return f"{json.dumps(key)}:{json.dumps(value, ensure_ascii=False)}"


def _named_rows(
    records: Sequence[Mapping[str, object]],
) -> list[tuple[str, EpisodeName]]:
    """Rows of ``OLD_SECTION_EPISODES_QUERY`` with a name that decodes."""
    rows: list[tuple[str, EpisodeName]] = []
    for record in records:
        uuid = record["uuid"]
        name = record["name"]
        if not (isinstance(uuid, str) and isinstance(name, str)):
            continue
        decoded = decode_episode_name(name)
        if decoded is not None:
            rows.append((uuid, decoded))
    return rows


def _open_facts(records: Sequence[Mapping[str, object]]) -> list[tuple[str, str]]:
    """Rows of ``TEXT_CLOSURE_CANDIDATES_QUERY`` as (uuid, fact)."""
    return [
        (uuid, fact)
        for record in records
        if isinstance(uuid := record["fact_uuid"], str)
        and isinstance(fact := record["fact"], str)
    ]


def _chunk_order(node: EpisodicNode) -> tuple[datetime, str, str]:
    """A stable order for old chunks: by date, then chunk id.

    The decision does not depend on it.
    """
    name = decode_episode_name(node.name)
    return (_utc(node.valid_at), name.chunk_id if name is not None else "", node.uuid)


def _same_lineage(old: EpisodeName, new: EpisodeIn) -> bool:
    """True when ``old`` may be an earlier version of ``new``'s document.

    When both documents have a ``series_id`` (``document_key``), the keys
    decide. Otherwise the two must come from the same source tool. Two
    documents in one tool with no ``series_id`` cannot be told apart, so a
    heading they share counts as one section: give them a ``series_id``.
    """
    if old.document_key and new.document_key:
        return old.document_key == new.document_key
    return old.source_tool == new.source_tool


def _count(result: Sequence[object] | None, key: str) -> int:
    """An integer column of a one-row aggregate result; 0 when absent."""
    if not result:
        return 0
    records = result[0]
    if not isinstance(records, Sequence) or not records:
        return 0
    record = records[0]
    if not isinstance(record, Mapping):
        return 0
    value = record.get(key)
    return value if isinstance(value, int) else 0


def _old_texts(
    origin: EpisodicNode,
    origin_name: EpisodeName | None,
    sources: Sequence[EpisodicNode],
) -> list[str]:
    """The origin's text, then the fact's other sources in its section.

    Another source counts when it is in the same document and the same
    section as the origin (``same_section``, so the chunks of a section
    split under positional ids count); with an unknown section, the origin
    stands alone.
    """
    texts = [origin.content]
    if origin_name is None or origin_name.section is None:
        return texts
    for node in sources:
        if node is origin:
            continue
        name = decode_episode_name(node.name)
        if (
            name is not None
            and name.section is not None
            and name.source_uri == origin_name.source_uri
            and same_section(name.section, origin_name.section)
        ):
            texts.append(node.content)
    return texts


@dataclass(slots=True)
class _ClosureTally:
    """Closure-check counts over one ``ingest`` call."""

    closed: int = 0
    reopened: int = 0
    kept: int = 0
    reopened_no_replacement: int = 0
    closed_by_text: int = 0

    def add(self, other: _ClosureTally) -> None:
        self.closed += other.closed
        self.reopened += other.reopened
        self.kept += other.kept
        self.reopened_no_replacement += other.reopened_no_replacement
        self.closed_by_text += other.closed_by_text


@dataclass(frozen=True, slots=True)
class _Failure:
    """Why one episode failed. Reported in ``IngestReport``, not raised."""

    error: str


@dataclass(slots=True)
class _Meter:
    """LLM calls and tokens over one ``ingest`` call.

    Calls are counted by ``CountingLLMClient``. Tokens are the delta of
    Graphiti's token tracker; they are None when calls were made and the
    client recorded none (``OpenAIGenericClient`` does not record).
    """

    llm: CountingLLMClient | None
    graphiti: Graphiti
    calls: int
    tracked_calls: int
    input_tokens: int
    output_tokens: int

    @classmethod
    def start(cls, llm: CountingLLMClient | None, graphiti: Graphiti) -> _Meter:
        tracked, inp, out = cls._tracker(graphiti)
        return cls(
            llm=llm,
            graphiti=graphiti,
            calls=llm.calls if llm is not None else 0,
            tracked_calls=tracked,
            input_tokens=inp,
            output_tokens=out,
        )

    @staticmethod
    def _tracker(graphiti: Graphiti) -> tuple[int, int, int]:
        usage = graphiti.llm_client.token_tracker.get_usage()
        return (
            sum(u.call_count for u in usage.values()),
            sum(u.total_input_tokens for u in usage.values()),
            sum(u.total_output_tokens for u in usage.values()),
        )

    def report(
        self,
        written: list[UUID],
        tally: _ClosureTally,
        *,
        failed: UUID | None = None,
        error: str | None = None,
    ) -> IngestReport:
        calls = (self.llm.calls if self.llm is not None else 0) - self.calls
        tracked, inp, out = self._tracker(self.graphiti)
        recorded = tracked - self.tracked_calls
        input_tokens: int | None = inp - self.input_tokens
        output_tokens: int | None = out - self.output_tokens
        if calls > 0 and recorded == 0:
            input_tokens = None
            output_tokens = None
        return IngestReport(
            written=tuple(written),
            failed=failed,
            error=error,
            llm_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            facts_closed=tally.closed,
            facts_reopened=tally.reopened,
            facts_kept_closed=tally.kept,
            facts_reopened_no_replacement=tally.reopened_no_replacement,
            facts_closed_by_text=tally.closed_by_text,
        )
