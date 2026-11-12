"""The memory ingestion activity (spec 043, HST-03 and ING-07).

An agent node is one activity (:mod:`holodeck.temporal.activity`), so the
memory functions the agent calls already run inside that activity. Ingestion
is different: its length grows with the corpus, so it gets its own activity,
registered once per worker under :data:`MEMORY_INGEST_ACTIVITY` and scheduled
by a workflow with a :class:`~holodeck.temporal.models.MemoryIngestInput`
naming the node whose agent owns the ``temporal_memory`` tool.

The activity:

* builds the agent's source tools with ``initialize_tools(...,
  initialize_memory=False)``, composes the memory with
  :func:`~holodeck.tools.temporal_memory.compose_memory`, and runs
  ``TemporalMemory.ingest``;
* heartbeats after each episode, through a backend wrapper that heartbeats
  after every completion check (``has_episode``) and every write
  (``ingest``), the two backend calls the service makes per episode;
* is safe to retry: completed episodes have a content-hash id and are skipped
  by the service, so a retry submits only the episodes left;
* closes the backend and the source tools on every exit, cancellation
  included.

Failures cross the activity boundary as an ``ApplicationError`` built from
fixed text, the exception's type name, the node id and episode ids only. The
text of an ingestion error can quote a source document (a bad front-matter
value, a YAML parser's excerpt, an extraction error), a credential or a host
name, and Temporal stores the failure in the workflow history. So no
exception's ``str()`` is forwarded, the original is not chained
(``raise ... from None``), heartbeats carry counts only, and the error text
is logged only at DEBUG. ``holodeck memory ingest`` on the worker host shows
the full error.

Classification:

* **Retryable:** ``MemoryBackendUnavailableError`` (the store could not be
  reached) and a failed episode, type ``MemoryEpisodeFailedError``, whose
  details carry the :class:`~holodeck.temporal.models.MemoryIngestResult`.
  A retry submits only the episodes left.
* **Non-retryable:** every other exception, typed by its class name: an
  unknown node id, ``ConfigError``, ``FileNotFoundError``,
  ``BackendInitError``, ``MemoryConfigError``, ``MemoryValueError``,
  ``MemoryNotSupportedError``, and any unexpected error. Retrying cannot fix
  a bad document or configuration, and a bug must not retry forever. This is
  the reverse of the agent activity's default, where a plain exception is
  retryable evidence about the model.
* **Cancellation** propagates unchanged as ``CancelledError``.

Workflow code must not import this module: it schedules the activity by name.
The memory modules are still imported inside the function bodies, so importing
this module pulls in neither ``holodeck.lib.memory`` nor ``graphiti_core``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

from temporalio import activity
from temporalio.exceptions import ApplicationError

from holodeck.config.context import agent_base_dir
from holodeck.config.loader import ConfigLoader
from holodeck.lib.workflow.edge import resolve_agent_path
from holodeck.models.agent import Agent
from holodeck.models.tool import TemporalMemoryToolConfig
from holodeck.models.workflow import EdgeNode
from holodeck.temporal.models import (
    MEMORY_INGEST_ACTIVITY,
    MemoryIngestInput,
    MemoryIngestResult,
)

if TYPE_CHECKING:
    from holodeck.lib.memory.ports import MemoryBackend
    from holodeck.lib.memory.service import IngestRunReport
    from holodeck.lib.memory.types import (
        EpisodeIn,
        FactHit,
        IngestReport,
        MemoryCapabilities,
        MemoryQuery,
        Provenance,
        TimelineEntry,
    )
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.vectorstore_tool import VectorStoreTool

logger = logging.getLogger(__name__)

EPISODE_FAILED_ERROR_TYPE = "MemoryEpisodeFailedError"
"""``ApplicationError.type`` of a run that stopped on a failed episode."""

_MemoryIngestActivity = Callable[[MemoryIngestInput], Awaitable[MemoryIngestResult]]


class HeartbeatingBackend:
    """A ``MemoryBackend`` that calls ``beat`` after each per-episode call.

    The service checks each episode with ``has_episode`` and writes each new
    one with ``ingest``, one episode per call. A beat after both gives at
    least one heartbeat per episode with no change to the service.

    Args:
        inner: The backend every call is delegated to.
        beat: Called with the number of episodes checked and submitted so far.
    """

    def __init__(self, inner: MemoryBackend, beat: Callable[[int, int], None]) -> None:
        self._inner = inner
        self._beat = beat
        self.checked = 0
        self.submitted = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def capabilities(self) -> MemoryCapabilities:
        return self._inner.capabilities()

    async def has_episode(self, namespace: str, episode_id: UUID) -> bool:
        found = await self._inner.has_episode(namespace, episode_id)
        self.checked += 1
        self._beat(self.checked, self.submitted)
        return found

    async def episode_count(self, namespace: str) -> int:
        return await self._inner.episode_count(namespace)

    async def ingest(
        self,
        namespace: str,
        episodes: Sequence[EpisodeIn],
        instructions: str,
    ) -> IngestReport:
        report = await self._inner.ingest(namespace, episodes, instructions)
        self.submitted += len(episodes)
        self._beat(self.checked, self.submitted)
        return report

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        return await self._inner.search(namespace, query)

    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]:
        return await self._inner.timeline(namespace, entity_ref)

    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None:
        return await self._inner.provenance(namespace, fact_id)

    async def purge(self, namespace: str) -> None:
        await self._inner.purge(namespace)

    async def close(self) -> None:
        await self._inner.close()


def _heartbeat(checked: int, submitted: int) -> None:
    """Record progress with the Temporal server. Counts only, no text."""
    activity.heartbeat(checked, submitted)


def _result(node_id: str, report: IngestRunReport) -> MemoryIngestResult:
    """Copy the counts and ids of ``report``. ``report.error`` is dropped."""
    return MemoryIngestResult(
        node_id=node_id,
        documents=report.documents,
        episodes_new=report.episodes_new,
        episodes_skipped=report.episodes_skipped,
        episodes_written=len(report.written),
        written=[str(episode_id) for episode_id in report.written],
        failed=None if report.failed is None else str(report.failed),
        llm_calls=report.llm_calls,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
    )


async def _run_ingest(
    node_id: str, agent: Agent, config: TemporalMemoryToolConfig, agent_path: Path
) -> MemoryIngestResult:
    """Build the sources and the memory, ingest, and close everything.

    Returns:
        The counts of the run. ``failed`` is set when it stopped on an
        episode; the caller raises for it.
    """
    from holodeck.lib.tool_initializer import close_tools, initialize_tools
    from holodeck.tools.temporal_memory import chunk_sources, compose_memory

    base_dir = str(agent_path.parent)
    token = agent_base_dir.set(base_dir)
    tools: dict[str, VectorStoreTool | HierarchicalDocumentTool] = {}
    backend: MemoryBackend | None = None
    try:
        tools = dict(
            await initialize_tools(agent, base_dir=base_dir, initialize_memory=False)
        )
        session = await compose_memory(
            agent,
            config,
            chunk_sources(config, tools),
            wrap_backend=lambda inner: HeartbeatingBackend(inner, _heartbeat),
        )
        backend = session.backend
        report = await session.memory.ingest()
    finally:
        # A finally, not an except: cancellation arrives as CancelledError,
        # a BaseException, and the backend and tools must close on it too.
        try:
            if backend is not None:
                await backend.close()
        finally:
            try:
                await close_tools(tools.values())
            finally:
                agent_base_dir.reset(token)

    # report.error is dropped: it can quote the document.
    return _result(node_id, report)


def _is_retryable(exc: Exception) -> bool:
    """Whether a retry can succeed: only an unreachable store qualifies."""
    from holodeck.lib.memory.errors import MemoryBackendUnavailableError

    return isinstance(exc, MemoryBackendUnavailableError)


def _sanitized_failure(node_id: str, exc: Exception) -> ApplicationError:
    """Return the ``ApplicationError`` that stands for ``exc`` in history.

    Built from fixed text, the node id and the exception's type name only.
    ``str(exc)`` is never used: it can quote a document, a credential or a
    host name.
    """
    name = type(exc).__name__
    if _is_retryable(exc):
        return ApplicationError(
            f"memory ingestion for node '{node_id}' could not reach the memory "
            f"backend ({name}); the activity is retried and completed episodes "
            "are skipped.",
            type=name,
        )
    return ApplicationError(
        f"memory ingestion for node '{node_id}' failed with {name}. The error "
        "text is not recorded because it can quote a source document or a "
        "credential. Run `holodeck memory ingest` on the worker host to see it.",
        type=name,
        non_retryable=True,
    )


def _memory_config(agent: Agent) -> TemporalMemoryToolConfig | None:
    for tool in agent.tools or []:
        if isinstance(tool, TemporalMemoryToolConfig):
            return tool
    return None


def memory_ingest_activity(
    nodes: Sequence[EdgeNode], base_dir: Path
) -> _MemoryIngestActivity | None:
    """Build the memory ingestion activity for the nodes that have a memory.

    Agent paths are resolved and agents loaded here, once, before the worker
    starts, as :func:`~holodeck.temporal.activity.agent_activity` does.

    Args:
        nodes: The worker's edge nodes.
        base_dir: Directory the nodes' ``edge.agent`` paths resolve against.

    Returns:
        An activity named :data:`MEMORY_INGEST_ACTIVITY`, or ``None`` when no
        node's agent has a ``temporal_memory`` tool.

    Raises:
        ConfigError: If an agent path escapes ``base_dir`` or an agent is
            invalid.
        FileNotFoundError: If a node's ``agent.yaml`` does not exist.
    """
    memories: dict[str, tuple[Agent, TemporalMemoryToolConfig, Path]] = {}
    for node in nodes:
        agent_path = resolve_agent_path(node, base_dir)
        agent = ConfigLoader().load_agent_yaml(str(agent_path))
        config = _memory_config(agent)
        if config is not None:
            memories[node.id] = (agent, config, agent_path)
    if not memories:
        return None

    @activity.defn(name=MEMORY_INGEST_ACTIVITY)
    async def ingest_memory(activity_input: MemoryIngestInput) -> MemoryIngestResult:
        """Ingest the memory of one node's agent.

        Args:
            activity_input: Names the node.

        Returns:
            The counts of the run.
        """
        node_id = activity_input.node_id
        entry = memories.get(node_id)
        if entry is None:
            raise ApplicationError(
                f"node '{node_id}' has no temporal_memory tool on "
                f"this worker; nodes with one: {sorted(memories)}",
                type="ConfigError",
                non_retryable=True,
            )
        agent, config, agent_path = entry
        try:
            result = await _run_ingest(node_id, agent, config, agent_path)
        except Exception as exc:
            # Exception, not BaseException: CancelledError passes unchanged.
            logger.warning(
                "memory ingestion for node '%s' failed with %s",
                node_id,
                type(exc).__name__,
            )
            # The error text only at DEBUG: it can quote a document.
            logger.debug("memory ingestion error for node '%s'", node_id, exc_info=True)
            raise _sanitized_failure(node_id, exc) from None
        if result.failed is not None:
            raise ApplicationError(
                f"memory ingestion for node '{node_id}' stopped at episode "
                f"{result.failed}; a retry submits only the episodes left. Run "
                "`holodeck memory ingest` to see the error.",
                result.model_dump(mode="json"),
                type=EPISODE_FAILED_ERROR_TYPE,
            )
        logger.info(
            "memory ingestion for node '%s': wrote %d episodes, skipped %d",
            node_id,
            result.episodes_written,
            result.episodes_skipped,
        )
        return result

    return cast("_MemoryIngestActivity", ingest_memory)


__all__ = [
    "EPISODE_FAILED_ERROR_TYPE",
    "MEMORY_INGEST_ACTIVITY",
    "HeartbeatingBackend",
    "memory_ingest_activity",
]
