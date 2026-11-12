"""The Graphiti memory backend: write side (spec 043, design seam 3).

Each ``EpisodeIn`` becomes one Graphiti episode, exactly once, under the
content-hash uuid the service chose:

1. Load the episode node by uuid. If it is absent, save it as *pending*.
   If it is *complete*, skip it.
2. Run ``add_episode`` on that uuid.
3. Load the node again and save it as *complete*.

The state lives in the node's ``name`` (``hd1:<state>:<json>``), because an
episode node stores only Graphiti's own nine fields. A crash at any point
leaves the episode pending, and the next run processes it again.

This module does not import ``graphiti_core`` at import time. Graphiti, its
driver and its clients are built on first use, once, so constructing the
backend needs neither the ``graphiti`` extra nor a reachable store.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol, TypeVar
from uuid import UUID

from holodeck.lib.extras import MissingExtraError, require_extra
from holodeck.lib.memory.backends.graphiti_secrets import credential_values, redact
from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryConfigError,
    MemoryNotSupportedError,
    TemporalMemoryError,
)
from holodeck.lib.memory.types import (
    EpisodeIn,
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
    from graphiti_core.nodes import EpisodicNode
    from pydantic import BaseModel

    from holodeck.lib.memory.backends.graphiti_clients import CountingLLMClient

logger = logging.getLogger(__name__)

NAME = "graphiti"
GRAPHITI_EXTRA = "graphiti"
GRAPHITI_FEATURE = "The graphiti temporal memory backend"

SUPPORTED = frozenset({MemoryOperation.INGEST, MemoryOperation.PURGE})

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


# --- Backend --------------------------------------------------------------


class GraphitiBackend:
    """``MemoryBackend`` over Graphiti. This task builds the write side only."""

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

                try:
                    # In a thread: the Kuzu driver opens its database and
                    # runs schema queries in its constructor.
                    runtime = await asyncio.to_thread(
                        graphiti_clients.build_graphiti,
                        self._config,
                        self._extraction_model,
                        self._embedding_service,
                    )
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

    async def _ready_for_write(self) -> Graphiti:
        """Graphiti with indices and constraints built, once per instance."""
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

    async def close(self) -> None:
        graphiti = self._graphiti
        self._graphiti = None
        self._llm = None
        self._scoped.clear()
        self._indices_built = False
        if graphiti is not None:
            await self._store("close", graphiti.close())

    # --- store access ---------------------------------------------------

    def _redact(self, text: str) -> str:
        """Remove credential values from text that leaves the backend."""
        return redact(text, credential_values(self._extraction_model, self._config))

    async def _store(self, operation: str, call: Awaitable[T]) -> T:
        """Await a store call; any store failure leaves as unavailable."""
        try:
            return await call
        except TemporalMemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - no driver type leaves
            raise MemoryBackendUnavailableError(
                NAME, f"{operation} failed: {self._redact(_describe(error))}"
            ) from error

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
        graphiti = await self._ready_for_write()
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

    # --- port: read side (a later task) ---------------------------------

    def _not_built(self, operation: MemoryOperation) -> NoReturn:
        self._capabilities.require(operation)
        raise MemoryNotSupportedError(NAME, operation.value)

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        self._not_built(MemoryOperation.SEARCH)

    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]:
        self._not_built(MemoryOperation.TIMELINE)

    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None:
        self._not_built(MemoryOperation.PROVENANCE)


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
