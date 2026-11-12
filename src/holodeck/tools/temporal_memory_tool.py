"""The agent-facing ``temporal_memory`` tool (spec 043, design section 6).

``initialize_tools`` creates this tool after its source tools, because it
wraps them as chunk sources. The tool owns the memory backend and the
``MemoryToolHandlers`` that every host binding calls. It holds no binding
code: the Claude and OpenAI Agents bindings register the three function
names and pass each call to ``TemporalMemoryTool.call``.

Lifecycle (design 6):

* The graph store is down at start and ``ingest_on_start`` is false: the
  tool starts, logs a warning, and each search returns chunk hits with
  ``degraded: true`` until the store is back.
* The graph store is down and ``ingest_on_start`` is true: start fails with
  ``MemoryBackendUnavailableError`` (ING-07).
* ``ingest_on_start`` is true: ingestion runs once during ``initialize``;
  an episode that fails fails the start.
* ``close`` closes the backend only. ``close_tools`` closes the source tools.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    TemporalMemoryError,
)
from holodeck.lib.memory.service import IngestRunReport
from holodeck.lib.memory.tool_api import MemoryToolHandlers
from holodeck.models.agent import Agent
from holodeck.models.tool import TemporalMemoryToolConfig
from holodeck.tools.temporal_memory import (
    MemorySession,
    chunk_sources,
    compose_memory,
)

if TYPE_CHECKING:
    from holodeck.lib.litellm_support import LiteLLMEmbeddingService

logger = logging.getLogger(__name__)


class TemporalMemoryTool:
    """A started temporal memory and its three agent functions.

    Args:
        agent: The agent that owns the tool.
        config: The ``temporal_memory`` tool configuration.
        source_tools: Initialized tool instances keyed by tool name. Each
            name in ``config.source_tools`` must be an initialized
            ``HierarchicalDocumentTool``.
        embedding_service: The agent's shared embedding service; a new one
            is created from the agent when None.
    """

    def __init__(
        self,
        agent: Agent,
        config: TemporalMemoryToolConfig,
        source_tools: Mapping[str, object],
        *,
        embedding_service: LiteLLMEmbeddingService | None = None,
    ) -> None:
        self.config = config
        self._agent = agent
        self._source_tools = dict(source_tools)
        self._embedding_service = embedding_service
        self._session: MemorySession | None = None
        self._handlers: MemoryToolHandlers | None = None
        self.degraded_at_start = False
        """True when the graph store could not be reached at start."""
        self.ingest_report: IngestRunReport | None = None
        """The ``ingest_on_start`` run, or None when none ran."""

    @property
    def is_initialized(self) -> bool:
        """True after ``initialize`` succeeded and before ``close``."""
        return self._handlers is not None

    async def initialize(self) -> None:
        """Build the backend and service, probe the store, ingest if asked.

        Raises:
            MemoryConfigError: A source tool is not an initialized
                ``hierarchical_document`` tool, the backend is not supported,
                or its optional dependency is missing (CFG-05).
            MemoryBackendUnavailableError: The store is down and
                ``ingest_on_start`` is true.
            TemporalMemoryError: ``ingest_on_start`` ran and an episode
                failed.
        """
        if self.is_initialized:
            return
        sources = chunk_sources(self.config, self._source_tools)
        session = await compose_memory(
            self._agent,
            self.config,
            sources,
            embedding_service=self._embedding_service,
        )
        try:
            await self._probe(session)
            if self.config.ingest_on_start:
                self.ingest_report = await self._ingest(session)
        except BaseException:
            await session.backend.close()
            raise
        self._session = session
        self._handlers = MemoryToolHandlers(session.memory)

    async def _probe(self, session: MemorySession) -> None:
        """Reach the store once. Down at start is degraded, unless ingesting."""
        try:
            await session.backend.episode_count(self.config.resolved_namespace)
        except MemoryBackendUnavailableError:
            if self.config.ingest_on_start:
                raise
            self.degraded_at_start = True
            logger.warning(
                "temporal_memory tool '%s': memory backend '%s' is unavailable; "
                "searches return source passages only (degraded) until it is back",
                self.config.name,
                session.backend.name,
            )

    async def _ingest(self, session: MemorySession) -> IngestRunReport:
        report = await session.memory.ingest()
        if report.failed is not None:
            # report.error is not repeated: it can quote the document.
            raise TemporalMemoryError(
                f"temporal_memory tool '{self.config.name}': ingest_on_start "
                f"stopped at episode {report.failed}. Run `holodeck memory "
                "ingest` to see the error."
            )
        logger.info(
            "temporal_memory tool '%s': ingest_on_start wrote %d episodes, "
            "skipped %d",
            self.config.name,
            len(report.written),
            report.episodes_skipped,
        )
        return report

    @property
    def handlers(self) -> MemoryToolHandlers:
        """The handler layer.

        Raises:
            BackendSessionError: The tool is not initialized, as for the
                other tools. A host fault, not a result for the model.
        """
        if self._handlers is None:
            from holodeck.lib.backends.base import BackendSessionError

            raise BackendSessionError(f"Tool '{self.config.name}' is not initialized")
        return self._handlers

    async def call(self, name: str, arguments: Mapping[str, object]) -> str:
        """Run memory function ``name`` and return its result as JSON text.

        Raises:
            BackendSessionError: The tool is not initialized.
            KeyError: ``name`` is not one of ``MEMORY_FUNCTION_NAMES``.
        """
        return await self.handlers.call(name, arguments)

    async def close(self) -> None:
        """Close the backend. Safe to call more than once."""
        session = self._session
        self._session = None
        self._handlers = None
        if session is not None:
            await session.backend.close()

    async def cleanup(self) -> None:
        """Release resources; the hook backends call on teardown."""
        await self.close()
