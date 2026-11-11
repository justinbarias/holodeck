"""Argument and result models of the three memory tools (spec 043, design 6).

These models are host-neutral. Each backend binding (Claude, OpenAI Agents,
Temporal) uses them to build its tool definitions and to validate arguments.
JSON Schemas under ``schemas/memory`` are generated from them by
``scripts/generate_memory_schemas.py``.

Dates in arguments stay text, because models send dates as text. Dates in
results are ISO 8601 UTC text with a ``Z`` suffix.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.lib.memory.instants import format_instant, parse_instant
from holodeck.lib.memory.types import (
    MAX_HOPS,
    MIN_HOPS,
    EpisodeRef,
    HitKind,
    MemoryHit,
    Provenance,
    ReferenceTimeSource,
    TimelineEntry,
)

MEMORY_SEARCH = "memory_search"
MEMORY_TIMELINE = "memory_timeline"
MEMORY_PROVENANCE = "memory_provenance"
MEMORY_FUNCTION_NAMES = (MEMORY_SEARCH, MEMORY_TIMELINE, MEMORY_PROVENANCE)

_NOT_BLANK = r"\S"


def _fmt(value: datetime | None) -> str | None:
    return None if value is None else format_instant(value)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Arguments -------------------------------------------------------------


class MemorySearchArgs(_Model):
    """Arguments of ``memory_search``."""

    query: str = Field(
        min_length=1,
        pattern=_NOT_BLANK,
        description="What to look for.",
    )
    valid_at: str | None = Field(
        default=None,
        description="What was true on this date (ISO 8601). Default: now.",
    )
    known_at: str | None = Field(
        default=None,
        description="What the memory had recorded by this date. Default: now.",
    )
    hops: int | None = Field(
        default=None,
        ge=MIN_HOPS,
        le=MAX_HOPS,
        strict=True,
        description="Graph expansion depth in entity-to-entity edges.",
    )

    @field_validator("valid_at")
    @classmethod
    def _check_valid_at(cls, value: str | None) -> str | None:
        if value is not None:
            parse_instant(value, "valid_at")
        return value

    @field_validator("known_at")
    @classmethod
    def _check_known_at(cls, value: str | None) -> str | None:
        if value is not None:
            parse_instant(value, "known_at")
        return value

    def resolved_valid_at(self, now: datetime) -> datetime:
        """Return ``valid_at`` as a UTC instant, or ``now`` when unset."""
        if self.valid_at is None:
            return now
        return parse_instant(self.valid_at, "valid_at")

    def resolved_known_at(self, now: datetime) -> datetime:
        """Return ``known_at`` as a UTC instant, or ``now`` when unset."""
        if self.known_at is None:
            return now
        return parse_instant(self.known_at, "known_at")


class MemoryTimelineArgs(_Model):
    """Arguments of ``memory_timeline``."""

    entity: str = Field(
        min_length=1,
        pattern=_NOT_BLANK,
        description="The entity whose facts to list.",
    )


class MemoryProvenanceArgs(_Model):
    """Arguments of ``memory_provenance``."""

    fact_id: str = Field(
        min_length=1,
        pattern=_NOT_BLANK,
        description="The fact_id of a fact returned by another memory tool.",
    )


# --- Results ---------------------------------------------------------------


class SourceOut(_Model):
    """One episode that supports a fact."""

    episode_id: str
    source_tool: str
    chunk_id: str
    source_uri: str
    reference_time: str
    reference_time_source: ReferenceTimeSource
    text: str | None = None

    @classmethod
    def from_ref(cls, ref: EpisodeRef) -> SourceOut:
        return cls(
            episode_id=str(ref.episode_id),
            source_tool=ref.source_tool,
            chunk_id=ref.chunk_id,
            source_uri=ref.source_uri,
            reference_time=format_instant(ref.reference_time),
            reference_time_source=ref.reference_time_source,
            text=ref.text,
        )


class MemoryHitOut(_Model):
    """One search hit: a fact or a source passage."""

    kind: HitKind
    text: str
    score: float
    source_tool: str
    chunk_id: str
    source_uri: str
    fact_id: str | None
    valid_from: str | None
    valid_to: str | None
    superseded_by: str | None

    @classmethod
    def from_hit(cls, hit: MemoryHit) -> MemoryHitOut:
        return cls(
            kind=hit.kind,
            text=hit.text,
            score=hit.score,
            source_tool=hit.source_tool,
            chunk_id=hit.chunk_id,
            source_uri=hit.source_uri,
            fact_id=hit.fact_id,
            valid_from=_fmt(hit.valid_from),
            valid_to=_fmt(hit.valid_to),
            superseded_by=hit.superseded_by,
        )


class MemorySearchResult(_Model):
    """Result of ``memory_search``."""

    valid_at: str
    known_at: str
    degraded: bool
    hits: list[MemoryHitOut]

    @classmethod
    def build(
        cls,
        hits: Sequence[MemoryHit],
        valid_at: datetime,
        known_at: datetime,
        *,
        degraded: bool = False,
    ) -> MemorySearchResult:
        """Build the result.

        Pass ``degraded=True`` when the backend was unavailable. A degraded
        search can have no hits, so the hits alone cannot carry the flag.
        """
        return cls(
            valid_at=format_instant(valid_at),
            known_at=format_instant(known_at),
            degraded=degraded or any(hit.degraded for hit in hits),
            hits=[MemoryHitOut.from_hit(hit) for hit in hits],
        )


class TimelineEntryOut(_Model):
    """One fact about an entity, open or closed."""

    fact_id: str
    text: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str | None
    expired_at: str | None
    sources: list[SourceOut]

    @classmethod
    def from_entry(cls, entry: TimelineEntry | Provenance) -> TimelineEntryOut:
        """Convert a fact view. Source text is left out, so a timeline stays small."""
        sources = [
            SourceOut.from_ref(ref).model_copy(update={"text": None})
            for ref in entry.sources
        ]
        return cls(
            fact_id=entry.fact_id,
            text=entry.text,
            valid_from=_fmt(entry.valid_from),
            valid_to=_fmt(entry.valid_to),
            recorded_at=_fmt(entry.recorded_at),
            expired_at=_fmt(entry.expired_at),
            sources=sources,
        )


class MemoryTimelineResult(_Model):
    """Result of ``memory_timeline``."""

    entity: str
    entries: list[TimelineEntryOut]


class MemoryProvenanceResult(_Model):
    """Result of ``memory_provenance``."""

    found: bool
    fact: TimelineEntryOut | None
    sources: list[SourceOut]

    @classmethod
    def build(cls, provenance: Provenance | None) -> MemoryProvenanceResult:
        if provenance is None:
            return cls(found=False, fact=None, sources=[])
        return cls(
            found=True,
            fact=TimelineEntryOut.from_entry(provenance),
            sources=[SourceOut.from_ref(ref) for ref in provenance.sources],
        )


class MemoryToolErrorBody(_Model):
    """Why a tool call failed."""

    code: Literal["invalid_argument", "not_supported", "backend_unavailable"]
    message: str


class MemoryToolError(_Model):
    """Result of a failed tool call."""

    error: MemoryToolErrorBody


MEMORY_TOOL_MODELS: dict[str, tuple[type[_Model], type[_Model]]] = {
    MEMORY_SEARCH: (MemorySearchArgs, MemorySearchResult),
    MEMORY_TIMELINE: (MemoryTimelineArgs, MemoryTimelineResult),
    MEMORY_PROVENANCE: (MemoryProvenanceArgs, MemoryProvenanceResult),
}
"""Each function name with its ``(args model, result model)`` pair."""
