"""The Graphiti memory backend (spec 043, design seams 3 and 5).

Write side. Each ``EpisodeIn`` becomes one Graphiti episode, exactly once, under the
content-hash uuid the service chose:

1. Load the episode node by uuid. If it is absent, save it as *pending*.
   If it is *complete*, skip it.
2. Run ``add_episode`` on that uuid.
3. Load the node again and save it as *complete*.

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
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar
from uuid import UUID

from holodeck.lib.extras import MissingExtraError, require_extra
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
from holodeck.models.tool import GraphBackendConfig

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
    """The HoloDeck fields and the state kept in an episode node's ``name``."""

    state: EpisodeState
    source_tool: str
    chunk_id: str
    source_uri: str
    reference_time_source: ReferenceTimeSource


def encode_episode_name(state: EpisodeState, episode: EpisodeIn) -> str:
    """Return ``hd1:<state>:<json>`` for ``episode``."""
    payload = {
        "source_tool": episode.source_tool,
        "chunk_id": episode.chunk_id,
        "source_uri": episode.source_uri,
        "date_source": episode.reference_time_source,
    }
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
        entity_types: Sequence[str] = (),
    ) -> None:
        self._config = config
        self._extraction_model = extraction_model
        self._embedding_service = embedding_service
        self._entity_types = tuple(entity_types)
        self._capabilities = MemoryCapabilities(backend=NAME, supported=SUPPORTED)
        self._graphiti: Graphiti | None = None
        self._llm: CountingLLMClient | None = None
        self._entity_models: dict[str, type[BaseModel]] | None = None
        self._scoped: dict[str, GraphDriver] = {}
        self._indices_built = False
        self._lock = asyncio.Lock()
        self._orphans: set[asyncio.Task[None]] = set()

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
                self._entity_models = (
                    graphiti_clients.entity_type_models(self._entity_types)
                    if self._entity_types
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
        written: list[UUID] = []
        for episode in episodes:
            outcome = await self._ingest_one(
                graphiti, driver, namespace, episode, instructions
            )
            if isinstance(outcome, _Failure):
                return meter.report(
                    written, failed=episode.episode_id, error=outcome.error
                )
            if outcome == "written":
                written.append(episode.episode_id)
        return meter.report(written)

    async def _ingest_one(
        self,
        graphiti: Graphiti,
        driver: GraphDriver,
        namespace: str,
        episode: EpisodeIn,
        instructions: str,
    ) -> Literal["written", "skipped"] | _Failure:
        """Ingest one episode, or say why it failed."""
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
        if node is None:
            pending = EpisodicNode(
                uuid=str(episode.episode_id),
                name=encode_episode_name("pending", episode),
                group_id=namespace,
                labels=[],
                source=EpisodeType.text,
                source_description=description,
                content=episode.text,
                created_at=utc_now(),
                valid_at=episode.reference_time,
            )
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
                previous_episode_uuids=context,
                custom_extraction_instructions=instructions,
            )
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
        )
