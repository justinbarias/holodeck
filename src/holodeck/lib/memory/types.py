"""Types that cross the temporal memory boundary (spec 043).

These are the only shapes a ``MemoryBackend`` accepts and returns. Backend
client types (Graphiti nodes, edges, filters) stay inside the backend module.

Time convention
---------------
Every ``datetime`` is timezone-aware UTC. A naive or offset datetime raises
``MemoryValueError`` at construction.

* Valid time: when a fact held in the world (``valid_from``, ``valid_to``).
* System time: when the store believed it (``recorded_at``, ``expired_at``).

Periods are half-open: ``[from, to)``. ``None`` as the end means "still open".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal
from uuid import UUID

from holodeck.lib.memory.errors import MemoryNotSupportedError, MemoryValueError

ReferenceTimeSource = Literal["front_matter", "mtime", "ingestion"]
"""Where an episode's reference time came from (spec D4)."""

HitKind = Literal["fact", "chunk"]

MIN_HOPS = 1
MAX_HOPS = 2

_NAMESPACE = re.compile(r"^[a-zA-Z0-9_-]+$")


def require_utc(value: datetime, field: str) -> datetime:
    """Return ``value`` when it is timezone-aware UTC; raise otherwise."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise MemoryValueError(f"{field} must be a timezone-aware UTC datetime")
    return value


def _require_utc_or_none(value: datetime | None, field: str) -> None:
    if value is not None:
        require_utc(value, field)


def validate_namespace(namespace: str) -> str:
    """Return ``namespace`` when every backend can store it; raise otherwise.

    Letters, digits, underscore and hyphen only. This is the rule of the
    strictest backend (Graphiti's ``group_id``), applied to all of them so a
    namespace stays valid when the backend changes.
    """
    if not _NAMESPACE.fullmatch(namespace):
        raise MemoryValueError(
            f"namespace '{namespace}' is not valid: use letters, digits, "
            "'_' and '-' only"
        )
    return namespace


class MemoryOperation(str, Enum):
    """An operation that a memory backend may or may not support."""

    INGEST = "ingestion"
    SEARCH = "search"
    VALID_AT = "valid-time queries"
    KNOWN_AT = "system-time (known_at) queries"
    EXPAND = "graph expansion from seed episodes"
    TIMELINE = "entity timelines"
    PROVENANCE = "fact provenance"
    PURGE = "namespace purge"
    ENTITY_TYPE_HINTS = "entity type hints"
    SINGLE_VALUED_PREDICATES = "single-valued predicates"
    SUPERSESSION_AUDIT = "supersession rationale and audit trail"
    REVIEW_QUEUE = "pending-merge and low-confidence review"
    RETRACTION = "retraction of a replaced source version"


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """The operations one backend supports."""

    backend: str
    supported: frozenset[MemoryOperation]

    def supports(self, operation: MemoryOperation) -> bool:
        """Return True when the backend supports ``operation``."""
        return operation in self.supported

    def require(self, operation: MemoryOperation, hint: str = "") -> None:
        """Raise ``MemoryNotSupportedError`` unless ``operation`` is supported."""
        if operation not in self.supported:
            raise MemoryNotSupportedError(self.backend, operation.value, hint)


@dataclass(frozen=True, slots=True)
class EpisodeIn:
    """One chunk submitted to a backend as an episode.

    Attributes:
        episode_id: Content-derived key. See ``service.episode_id_for``.
        source_tool: Name of the ``hierarchical_document`` tool the chunk
            came from.
        chunk_id: The chunk's id in that tool.
        source_uri: Path or URI of the source document.
        text: Chunk text handed to extraction.
        reference_time: The document's effective date.
        reference_time_source: Where ``reference_time`` came from.
        context_episode_ids: Episodes of the chunks that precede this one in
            the same document, nearest last. A backend may give them to its
            model as context. Empty for the first chunk of a document.
    """

    episode_id: UUID
    source_tool: str
    chunk_id: str
    source_uri: str
    text: str
    reference_time: datetime
    reference_time_source: ReferenceTimeSource
    context_episode_ids: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        require_utc(self.reference_time, "reference_time")
        if self.episode_id in self.context_episode_ids:
            raise MemoryValueError("an episode cannot be its own context")


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    """An ingested episode, as a backend reports it back.

    Attributes:
        text: The chunk text. Set only on the sources that ``provenance``
            returns; None everywhere else, so search results stay small.
    """

    episode_id: UUID
    source_tool: str
    chunk_id: str
    source_uri: str
    reference_time: datetime
    reference_time_source: ReferenceTimeSource
    text: str | None = None

    def __post_init__(self) -> None:
        require_utc(self.reference_time, "reference_time")


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ``MemoryBackend.ingest`` call did.

    A backend stops at the first episode that fails. ``written`` then holds
    the episodes completed before it, in submission order.

    Attributes:
        written: Episodes completed by this call.
        failed: The episode the call stopped on, or None.
        error: Why it failed, or None.
        llm_calls: LLM calls the backend made.
        input_tokens: Prompt tokens, or None when the backend cannot count.
        output_tokens: Completion tokens, or None when it cannot count.
    """

    written: tuple[UUID, ...]
    failed: UUID | None = None
    error: str | None = None
    llm_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """A point-in-time search.

    Attributes:
        text: The search text.
        valid_at: Return facts that held in the world at this instant.
        known_at: Return facts the store believed at this instant.
        hops: Expansion depth in entity-to-entity edges, 1 or 2.
        limit: Maximum number of facts to return.
        seed_episode_ids: Episodes whose entities seed the expansion. Empty
            means search without seeds.
    """

    text: str
    valid_at: datetime
    known_at: datetime
    hops: int = MIN_HOPS
    limit: int = 10
    seed_episode_ids: tuple[UUID, ...] = ()

    def __post_init__(self) -> None:
        require_utc(self.valid_at, "valid_at")
        require_utc(self.known_at, "known_at")
        if not MIN_HOPS <= self.hops <= MAX_HOPS:
            raise MemoryValueError(f"hops must be {MIN_HOPS} or {MAX_HOPS}")
        if self.limit < 1:
            raise MemoryValueError("limit must be at least 1")


@dataclass(frozen=True, slots=True)
class _Fact:
    """Fields shared by every view of a fact."""

    fact_id: str
    text: str
    valid_from: datetime | None
    valid_to: datetime | None
    recorded_at: datetime
    expired_at: datetime | None
    sources: tuple[EpisodeRef, ...]

    def __post_init__(self) -> None:
        _require_utc_or_none(self.valid_from, "valid_from")
        _require_utc_or_none(self.valid_to, "valid_to")
        require_utc(self.recorded_at, "recorded_at")
        _require_utc_or_none(self.expired_at, "expired_at")


@dataclass(frozen=True, slots=True)
class FactHit(_Fact):
    """One fact returned by ``MemoryBackend.search``. Higher score is better.

    ``superseded_by`` is the id of the fact that closed this one, when the
    backend records it.
    """

    score: float = 0.0
    superseded_by: str | None = None


@dataclass(frozen=True, slots=True)
class TimelineEntry(_Fact):
    """One fact about an entity, open or closed."""


@dataclass(frozen=True, slots=True)
class Provenance(_Fact):
    """A fact and every episode that supports it."""


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One result of ``memory_search`` after fusion with chunk retrieval.

    A ``fact`` hit carries ``fact_id`` and a valid period. A ``chunk`` hit is
    a passage from a source tool and carries neither.
    """

    kind: HitKind
    text: str
    score: float
    source_tool: str
    chunk_id: str
    source_uri: str
    fact_id: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    degraded: bool = False

    def __post_init__(self) -> None:
        _require_utc_or_none(self.valid_from, "valid_from")
        _require_utc_or_none(self.valid_to, "valid_to")
