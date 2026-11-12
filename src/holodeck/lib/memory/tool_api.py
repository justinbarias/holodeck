"""Argument and result models of the three memory tools (spec 043, design 6).

These models are host-neutral. Each backend binding (Claude, OpenAI Agents,
Temporal) uses them to build its tool definitions and to validate arguments.
JSON Schemas under ``schemas/memory`` are generated from them by
``scripts/generate_memory_schemas.py``.

Dates in arguments stay text, because models send dates as text. Dates in
results are ISO 8601 UTC text with a ``Z`` suffix.

``MemoryToolHandlers`` is the handler layer: one async method per function,
plus ``call(name, arguments)``, which validates the arguments, runs the
handler and returns the result as JSON text. A cause the model can correct
comes back as a ``MemoryToolError`` result, never as an exception. Bindings
register the three names and pass calls through; they add nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryNotSupportedError,
    MemoryValueError,
)
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

if TYPE_CHECKING:
    from holodeck.lib.memory.service import TemporalMemory

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
        description=(
            "The name of the entity whose facts to list, as facts state it: a "
            "payment, body, person or provision, for example 'Family Benefit'. "
            "Not a question and not a description."
        ),
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


# --- What the agent is told -------------------------------------------------

MEMORY_FUNCTION_DESCRIPTIONS: dict[str, str] = {
    MEMORY_SEARCH: (
        "Search the knowledge base for facts and passages. Pass `valid_at` (a "
        "date) to ask what was true on that date; it defaults to today. A hit "
        "of kind `fact` has `valid_from` and `valid_to` and was true on "
        "`valid_at`. A hit of kind `chunk` is a source passage with a start "
        "date only, and it may come from a superseded version; prefer facts "
        "when they disagree. Use `memory_timeline` for the history of one "
        "entity and `memory_provenance` for the source of a fact."
    ),
    MEMORY_TIMELINE: (
        "List every fact about one entity, ordered by `valid_from`, including "
        "facts that no longer hold. `valid_from` and `valid_to` say when a fact "
        "was true; `recorded_at` and `expired_at` say when the memory believed "
        "it. Use `memory_provenance` with a `fact_id` to read its source."
    ),
    MEMORY_PROVENANCE: (
        "Return a fact and the source passages that support it, each with its "
        "source tool, chunk id, document and effective date. Pass a `fact_id` "
        "from `memory_search` or `memory_timeline`. `found` is false when the "
        "`fact_id` is unknown."
    ),
}
"""The built-in text of each function (design 6, "What the agent is told")."""


def function_description(name: str, tool_description: str = "") -> str:
    """Return what the agent is told about function ``name``.

    The YAML ``description`` of the memory tool goes in front of the built-in
    text. It does not replace it: the built-in text states rules the agent
    needs.

    Raises:
        KeyError: ``name`` is not one of ``MEMORY_FUNCTION_NAMES``.
    """
    built_in = MEMORY_FUNCTION_DESCRIPTIONS[name]
    prefix = tool_description.strip()
    return f"{prefix}\n\n{built_in}" if prefix else built_in


# --- Handlers ---------------------------------------------------------------

ErrorCode = Literal["invalid_argument", "not_supported", "backend_unavailable"]

_ArgsT = TypeVar("_ArgsT", bound=_Model)


def tool_error(code: ErrorCode, message: str) -> MemoryToolError:
    """Return the error result for ``code``."""
    return MemoryToolError(error=MemoryToolErrorBody(code=code, message=message))


def _not_supported(error: MemoryNotSupportedError) -> MemoryToolError:
    # Built from the fields, not str(error): the hint is backend text.
    return tool_error(
        "not_supported",
        f"memory backend '{error.backend}' does not support {error.operation}",
    )


def _unavailable(error: MemoryBackendUnavailableError) -> MemoryToolError:
    # The detail is left out: it can name a host or quote a driver message.
    return tool_error(
        "backend_unavailable",
        f"memory backend '{error.backend}' is unavailable; try again later",
    )


_DATE_EXPECTED = (
    "an ISO 8601 date or datetime, for example 2022-07-01 or 2022-07-01T09:30:00Z"
)
_NON_EMPTY = "a non-empty string"
_EXPECTED: dict[str, str] = {
    "query": _NON_EMPTY,
    "entity": _NON_EMPTY,
    "fact_id": _NON_EMPTY,
    "valid_at": _DATE_EXPECTED,
    "known_at": _DATE_EXPECTED,
    "hops": f"{MIN_HOPS} or {MAX_HOPS}",
}
"""What each argument must be. Fixed text: an error message never repeats
what the model sent."""


def _validation_message(model: type[_Model], error: ValidationError) -> str:
    """One fixed phrase per failed field.

    The message holds the name of a field of ``model`` and a fixed description
    of what it must be. It never holds text the model sent: not a value, and
    not the name of a field the model made up.
    """
    parts: list[str] = []
    for item in error.errors(include_url=False, include_input=False):
        loc = item["loc"]
        field = str(loc[0]) if loc else ""
        if field not in model.model_fields:
            part = "an unknown field"
        else:
            part = f"{field} must be {_EXPECTED.get(field, 'valid')}"
        if part not in parts:
            parts.append(part)
    return "; ".join(parts) or "the arguments are not valid"


def parse_arguments(
    model: type[_ArgsT], arguments: Mapping[str, object]
) -> _ArgsT | MemoryToolError:
    """Validate ``arguments`` with ``model``; an invalid one is an error result."""
    try:
        return model.model_validate(dict(arguments))
    except ValidationError as error:
        return tool_error("invalid_argument", _validation_message(model, error))


class MemoryToolHandlers:
    """The three memory functions over one ``TemporalMemory``. Host-neutral.

    Each handler takes the validated argument model and returns the result
    model, or a ``MemoryToolError`` for a cause the model can correct:

    * ``invalid_argument``: an argument fails validation.
    * ``not_supported``: the backend lacks the operation (named).
    * ``backend_unavailable``: the backend is down, for ``memory_timeline``
      and ``memory_provenance``. ``memory_search`` then returns chunk hits
      with ``degraded: true`` instead.

    Any other exception propagates: it is a host fault.
    """

    def __init__(self, memory: TemporalMemory) -> None:
        self._memory = memory

    async def memory_search(
        self, args: MemorySearchArgs
    ) -> MemorySearchResult | MemoryToolError:
        """Run ``memory_search``. Unset instants default to the service clock."""
        try:
            valid_at = (
                None
                if args.valid_at is None
                else parse_instant(args.valid_at, "valid_at")
            )
            known_at = (
                None
                if args.known_at is None
                else parse_instant(args.known_at, "known_at")
            )
            outcome = await self._memory.search(
                args.query, valid_at=valid_at, known_at=known_at, hops=args.hops
            )
        except MemoryValueError:
            # Fixed text: the error can quote a date or a backend detail.
            return tool_error(
                "invalid_argument",
                "valid_at, known_at or hops was rejected; check each "
                "against its format",
            )
        except MemoryNotSupportedError as error:
            return _not_supported(error)
        except MemoryBackendUnavailableError as error:
            # The service degrades on the backend search itself; this is any
            # other store call that could not be reached.
            return _unavailable(error)
        return MemorySearchResult.build(
            outcome.hits,
            outcome.valid_at,
            outcome.known_at,
            degraded=outcome.degraded,
        )

    async def memory_timeline(
        self, args: MemoryTimelineArgs
    ) -> MemoryTimelineResult | MemoryToolError:
        """Run ``memory_timeline``."""
        try:
            entries = await self._memory.timeline(args.entity)
        except MemoryNotSupportedError as error:
            return _not_supported(error)
        except MemoryBackendUnavailableError as error:
            return _unavailable(error)
        return MemoryTimelineResult(
            entity=args.entity,
            entries=[TimelineEntryOut.from_entry(entry) for entry in entries],
        )

    async def memory_provenance(
        self, args: MemoryProvenanceArgs
    ) -> MemoryProvenanceResult | MemoryToolError:
        """Run ``memory_provenance``. An unknown ``fact_id`` is ``found: false``."""
        try:
            provenance = await self._memory.provenance(args.fact_id)
        except MemoryNotSupportedError as error:
            return _not_supported(error)
        except MemoryBackendUnavailableError as error:
            return _unavailable(error)
        return MemoryProvenanceResult.build(provenance)

    async def run(self, name: str, arguments: Mapping[str, object]) -> _Model:
        """Validate ``arguments`` for function ``name`` and run its handler.

        Returns:
            The result model, or a ``MemoryToolError``.

        Raises:
            KeyError: ``name`` is not one of ``MEMORY_FUNCTION_NAMES``.
        """
        if name == MEMORY_SEARCH:
            search = parse_arguments(MemorySearchArgs, arguments)
            if isinstance(search, MemoryToolError):
                return search
            return await self.memory_search(search)
        if name == MEMORY_TIMELINE:
            timeline = parse_arguments(MemoryTimelineArgs, arguments)
            if isinstance(timeline, MemoryToolError):
                return timeline
            return await self.memory_timeline(timeline)
        if name == MEMORY_PROVENANCE:
            provenance = parse_arguments(MemoryProvenanceArgs, arguments)
            if isinstance(provenance, MemoryToolError):
                return provenance
            return await self.memory_provenance(provenance)
        raise KeyError(f"unknown memory function '{name}'")

    async def call(self, name: str, arguments: Mapping[str, object]) -> str:
        """Run function ``name`` and return its result as JSON text.

        This is the one entry point every binding passes calls to, so the
        same call returns the same text on every host.

        Raises:
            KeyError: ``name`` is not one of ``MEMORY_FUNCTION_NAMES``.
        """
        result = await self.run(name, arguments)
        return result.model_dump_json()
