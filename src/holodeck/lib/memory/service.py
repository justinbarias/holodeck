"""The temporal memory service: ingestion between ``ChunkSource`` and a backend.

The service holds the rules that belong to neither side: episode identity,
effective dates and the order episodes are submitted in (spec 043, design
section 2). Only the ingest path is here.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from opentelemetry.trace import Span, Status, StatusCode

from holodeck.lib.memory.chunk_source import ChunkRef, ChunkSource, SourceDocument
from holodeck.lib.memory.effective_date import effective_date
from holodeck.lib.memory.errors import MemoryConfigError, TemporalMemoryError
from holodeck.lib.memory.instants import format_instant
from holodeck.lib.memory.ports import MemoryBackend
from holodeck.lib.memory.types import (
    EpisodeIn,
    IngestReport,
    ReferenceTimeSource,
    validate_namespace,
)
from holodeck.lib.observability import get_tracer

DEFAULT_EXTRACTION_INSTRUCTIONS = (
    "Extract amounts, rates, percentages, thresholds and dates as entities, "
    "and state each one as a fact about the payment, person or provision it "
    "applies to. Keep the unit and the period, for example dollars per "
    "fortnight. When the text gives a start or end date for a value, use it "
    "as the fact's valid period."
)

CONTEXT_CHUNKS = 2
"""Preceding chunks of the same document given to extraction as context."""

RUN_SPAN = "holodeck.memory.ingest"
EPISODE_SPAN = "holodeck.memory.ingest_episode"
EPISODE_FAILED_STATUS = "episode failed"
"""Span status never carries error text: it can quote a document or a secret.
The text is in the run report. An exception is named by its type only."""


def episode_id_for(namespace: str, section_id: str, chunk_text: str) -> UUID:
    """Return the episode id of a chunk.

    The id depends on the namespace, the section id and the chunk text only.
    It does not depend on the chunk id, which shifts when a section is
    inserted, or on the source tool, so identical text in two tools is one
    episode (spec assumption A8).
    """
    digest = hashlib.sha256(f"{section_id}\n{chunk_text}".encode()).hexdigest()
    return uuid5(uuid5(NAMESPACE_URL, f"holodeck/memory/{namespace}"), digest)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class IngestRunReport:
    """What one ``TemporalMemory.ingest`` run did.

    A run that stops on a failed episode does not look at the episodes after
    it, so ``episodes_new`` and ``episodes_skipped`` then count the episodes
    examined up to and including the failed one.

    Attributes:
        documents: Documents considered.
        episodes_new: Episodes that were not complete in the backend before
            this run.
        episodes_skipped: Episodes already complete, plus repeats of an
            episode seen earlier in the same run.
        written: Episodes completed by this run, in submission order.
        failed: The episode the run stopped on, or None.
        error: Why it failed, or None.
        llm_calls: LLM calls the backend made.
        input_tokens: Prompt tokens, or None when no backend report had a count.
        output_tokens: Completion tokens, or None when none had a count.
        duration_seconds: Wall time of the run, from ``time.monotonic``.
        dry_run: True when nothing was submitted to the backend.
        estimated_seconds: Dry run only: ``episodes_new`` times
            ``seconds_per_episode``; None when no rate is known.
    """

    documents: int
    episodes_new: int
    episodes_skipped: int
    written: tuple[UUID, ...]
    failed: UUID | None
    error: str | None
    llm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float
    dry_run: bool
    estimated_seconds: float | None


@dataclass(frozen=True, slots=True)
class _DatedDocument:
    source: ChunkSource
    position: int
    document: SourceDocument
    date: datetime
    date_source: ReferenceTimeSource


def _set_tokens(
    span: Span, input_tokens: int | None, output_tokens: int | None
) -> None:
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)


def _add_optional(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


class TemporalMemory:
    """Reads chunks from source tools and writes episodes to a backend."""

    def __init__(
        self,
        *,
        namespace: str,
        backend: MemoryBackend,
        sources: Sequence[ChunkSource],
        effective_date_key: str = "effective_date",
        instructions: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
        seconds_per_episode: float | None = None,
    ) -> None:
        """Create the service.

        Args:
            namespace: Namespace every backend call uses.
            backend: The fact store.
            sources: The source tools, with unique names.
            effective_date_key: Front-matter key holding a document's date.
            instructions: Extraction guidance; the built-in default when None.
            clock: Returns the current UTC time. Read once per ingest run.
            seconds_per_episode: Rate used for the dry-run estimate.

        Raises:
            MemoryValueError: If ``namespace`` is not valid.
            MemoryConfigError: If two sources share a name.
        """
        self._namespace = validate_namespace(namespace)
        names = [source.name for source in sources]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise MemoryConfigError(
                "source tool names must be unique; repeated: " + ", ".join(duplicates)
            )
        self._backend = backend
        self._sources = tuple(sources)
        self._effective_date_key = effective_date_key
        self._instructions = (
            DEFAULT_EXTRACTION_INSTRUCTIONS if instructions is None else instructions
        )
        self._clock = clock
        self._seconds_per_episode = seconds_per_episode

    def _select(self, source_tools: Sequence[str] | None) -> tuple[ChunkSource, ...]:
        if source_tools is None:
            return self._sources
        known = {source.name for source in self._sources}
        for name in source_tools:
            if name not in known:
                raise MemoryConfigError(
                    f"unknown source tool '{name}'; known: {', '.join(sorted(known))}"
                )
        wanted = set(source_tools)
        return tuple(source for source in self._sources if source.name in wanted)

    async def _dated_documents(
        self, sources: Sequence[ChunkSource], now: datetime
    ) -> list[_DatedDocument]:
        """Read every document and its effective date, in one ascending order."""
        dated: list[_DatedDocument] = []
        # Position in the configured list breaks a date tie, so the first
        # tool in source_tools owns a chunk that two tools share (A8).
        positions = {source.name: index for index, source in enumerate(self._sources)}
        for source in sources:
            position = positions[source.name]
            documents = await source.documents()
            if not documents:
                # A tool's store loader hides a store error as an empty result,
                # so an empty source is never "nothing to do".
                raise TemporalMemoryError(
                    f"source tool '{source.name}' returned no documents. "
                    "Check that the tool initialized and that its store is "
                    "reachable."
                )
            for document in documents:
                date, date_source = await asyncio.to_thread(
                    effective_date, document, self._effective_date_key, now
                )
                dated.append(
                    _DatedDocument(source, position, document, date, date_source)
                )
        dated.sort(
            key=lambda item: (item.date, item.position, item.document.source_uri)
        )
        return dated

    async def ingest(
        self,
        *,
        dry_run: bool = False,
        source_tools: Sequence[str] | None = None,
    ) -> IngestRunReport:
        """Submit every chunk without a complete episode to the backend.

        Episodes go one at a time, oldest document first (ING-04). The run
        stops at the first episode the backend fails on (ING-05). With
        ``dry_run`` it only asks the backend which episodes exist (ING-08).

        Args:
            dry_run: Count the new episodes; make no write and no LLM call.
            source_tools: Limit the run to these source tools.

        Raises:
            MemoryConfigError: If a name in ``source_tools`` is unknown.
            TemporalMemoryError: If a source has no documents.
            MemoryBackendUnavailableError: If the backend cannot be reached.
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span(
            RUN_SPAN, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("holodeck.memory.namespace", self._namespace)
            span.set_attribute("holodeck.memory.dry_run", dry_run)
            try:
                report = await self._ingest(dry_run, source_tools)
            except BaseException as exc:
                span.set_attribute("holodeck.memory.failed", True)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            span.set_attribute("holodeck.memory.documents", report.documents)
            span.set_attribute("holodeck.memory.episodes_new", report.episodes_new)
            span.set_attribute(
                "holodeck.memory.episodes_skipped", report.episodes_skipped
            )
            span.set_attribute("holodeck.memory.episodes_written", len(report.written))
            span.set_attribute("holodeck.memory.llm_calls", report.llm_calls)
            _set_tokens(span, report.input_tokens, report.output_tokens)
            failed = report.failed is not None
            span.set_attribute("holodeck.memory.failed", failed)
            if failed:
                span.set_status(Status(StatusCode.ERROR, EPISODE_FAILED_STATUS))
            return report

    async def _ingest(
        self, dry_run: bool, source_tools: Sequence[str] | None
    ) -> IngestRunReport:
        started = time.monotonic()
        now = self._clock()
        sources = self._select(source_tools)
        documents = await self._dated_documents(sources, now)

        seen: set[UUID] = set()
        new = 0
        skipped = 0
        written: list[UUID] = []
        failed: UUID | None = None
        error: str | None = None
        llm_calls = 0
        input_tokens: int | None = None
        output_tokens: int | None = None

        for item in documents:
            chunks = sorted(await item.source.chunks(item.document), key=_ordinal_of)
            ids = [
                episode_id_for(self._namespace, chunk.section_id, chunk.text)
                for chunk in chunks
            ]
            for index, chunk in enumerate(chunks):
                episode_id = ids[index]
                if episode_id in seen:
                    skipped += 1
                    continue
                seen.add(episode_id)
                if await self._backend.has_episode(self._namespace, episode_id):
                    skipped += 1
                    continue
                new += 1
                if dry_run:
                    continue
                report = await self._submit(item, chunk, ids, index)
                written.extend(report.written)
                llm_calls += report.llm_calls
                input_tokens = _add_optional(input_tokens, report.input_tokens)
                output_tokens = _add_optional(output_tokens, report.output_tokens)
                if report.failed is not None:
                    failed = report.failed
                    error = report.error
                    break
            if failed is not None:
                break

        estimate = (
            new * self._seconds_per_episode
            if dry_run and self._seconds_per_episode is not None
            else None
        )
        return IngestRunReport(
            documents=len(documents),
            episodes_new=new,
            episodes_skipped=skipped,
            written=tuple(written),
            failed=failed,
            error=error,
            llm_calls=llm_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=time.monotonic() - started,
            dry_run=dry_run,
            estimated_seconds=estimate,
        )

    async def _submit(
        self,
        item: _DatedDocument,
        chunk: ChunkRef,
        ids: Sequence[UUID],
        index: int,
    ) -> IngestReport:
        own = ids[index]
        # The same text twice in one document gives one id. An episode cannot
        # be its own context, and an id is listed once.
        preceding = ids[max(0, index - CONTEXT_CHUNKS) : index]
        context = tuple(dict.fromkeys(i for i in preceding if i != own))
        episode = EpisodeIn(
            episode_id=own,
            source_tool=chunk.source_tool,
            chunk_id=chunk.chunk_id,
            source_uri=chunk.source_uri,
            text=chunk.text,
            reference_time=item.date,
            reference_time_source=item.date_source,
            context_episode_ids=context,
        )
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span(
            EPISODE_SPAN, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("holodeck.memory.episode_id", str(own))
            span.set_attribute("holodeck.memory.source_tool", chunk.source_tool)
            span.set_attribute("holodeck.memory.chunk_id", chunk.chunk_id)
            span.set_attribute("holodeck.memory.source_uri", chunk.source_uri)
            span.set_attribute(
                "holodeck.memory.reference_time", format_instant(item.date)
            )
            span.set_attribute(
                "holodeck.memory.reference_time_source", item.date_source
            )
            try:
                report = await self._backend.ingest(
                    self._namespace, [episode], self._instructions
                )
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            span.set_attribute("holodeck.memory.llm_calls", report.llm_calls)
            _set_tokens(span, report.input_tokens, report.output_tokens)
            if report.failed is not None:
                span.set_status(Status(StatusCode.ERROR, EPISODE_FAILED_STATUS))
            return report


def _ordinal_of(chunk: ChunkRef) -> int:
    return chunk.ordinal
