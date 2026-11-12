"""The contract every temporal memory backend implements (spec 043, MB-01)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from holodeck.lib.memory.types import (
    EpisodeIn,
    FactHit,
    IngestReport,
    MemoryCapabilities,
    MemoryQuery,
    Provenance,
    TimelineEntry,
)


@runtime_checkable
class MemoryBackend(Protocol):
    """A bi-temporal fact store for one or more namespaces.

    Rules every backend follows:

    * Every read and write is scoped to the ``namespace`` argument. A
      backend never returns or changes data in another namespace.
    * An operation the backend cannot do raises ``MemoryNotSupportedError``
      before it touches the store. It never returns a partial result.
    * A store that cannot be reached raises
      ``MemoryBackendUnavailableError``.
    * No backend client type appears in an argument or a return value.
    """

    @property
    def name(self) -> str:
        """Backend name as written in agent YAML."""
        ...

    def capabilities(self) -> MemoryCapabilities:
        """Return the operations this backend supports."""
        ...

    async def has_episode(self, namespace: str, episode_id: UUID) -> bool:
        """Return True when the episode exists and its ingestion completed.

        Makes no LLM call. An episode that was started and not completed
        returns False, so the caller submits it again.
        """
        ...

    async def episode_count(self, namespace: str) -> int:
        """Return the number of completed episodes in the namespace."""
        ...

    async def ingest(
        self,
        namespace: str,
        episodes: Sequence[EpisodeIn],
        instructions: str,
    ) -> IngestReport:
        """Extract facts from ``episodes``, in the order given.

        The caller submits only episodes for which ``has_episode`` is False,
        ordered by reference time. The backend stops at the first episode
        that fails and reports it; it does not raise for a failed episode.

        Args:
            namespace: The namespace to write to.
            episodes: Episodes to ingest, oldest reference time first.
            instructions: Extraction guidance for the backend's model.
        """
        ...

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        """Return facts valid at ``query.valid_at`` and known at ``query.known_at``.

        Best first, at most ``query.limit`` facts.
        """
        ...

    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]:
        """Return every fact about an entity, ordered by ``valid_from``.

        Includes closed and expired facts. An unknown entity returns ``[]``.
        """
        ...

    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None:
        """Return a fact and its supporting episodes, or None when unknown."""
        ...

    async def purge(self, namespace: str) -> None:
        """Remove every episode, entity and fact in the namespace."""
        ...

    async def close(self) -> None:
        """Release the backend's connections."""
        ...
