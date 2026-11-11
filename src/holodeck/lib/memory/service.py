"""The temporal memory service: ingestion, search and fusion (spec 043).

The service sits between ``ChunkSource`` and a backend and holds the rules that
belong to neither side: episode identity, effective dates, the order episodes
are submitted in (design section 2) and the fusion of chunk and fact hits
(design section 5).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from opentelemetry.trace import Span, Status, StatusCode

from holodeck.lib.memory.chunk_source import (
    ChunkMatch,
    ChunkRef,
    ChunkSource,
    SourceDocument,
)
from holodeck.lib.memory.effective_date import effective_date
from holodeck.lib.memory.errors import (
    MemoryBackendUnavailableError,
    MemoryConfigError,
    MemoryValueError,
    TemporalMemoryError,
)
from holodeck.lib.memory.instants import format_instant
from holodeck.lib.memory.ports import MemoryBackend
from holodeck.lib.memory.types import (
    MAX_HOPS,
    MIN_HOPS,
    EpisodeIn,
    EpisodeRef,
    FactHit,
    IngestReport,
    MemoryHit,
    MemoryOperation,
    MemoryQuery,
    Provenance,
    ReferenceTimeSource,
    TimelineEntry,
    require_utc,
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
SEARCH_SPAN = "holodeck.memory.search"
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


RRF_K = 60
"""Reciprocal rank fusion constant. Fixed: the weights are the control."""


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
class SearchOutcome:
    """What one ``TemporalMemory.search`` call returned.

    ``MemorySearchResult.build(outcome.hits, outcome.valid_at,
    outcome.known_at, degraded=outcome.degraded)`` turns it into the tool
    result.

    Attributes:
        hits: Fused hits, best first, at most ``top_k``.
        valid_at: The valid-time instant used (the query time by default).
        known_at: The system-time instant used (the query time by default).
        degraded: True when the backend was unavailable and the hits are
            chunk hits only. A degraded search can have no hits.
    """

    hits: tuple[MemoryHit, ...]
    valid_at: datetime
    known_at: datetime
    degraded: bool


@dataclass(frozen=True, slots=True)
class _ChunkHit:
    """A chunk hit that survived the date filter."""

    match: ChunkMatch
    rank: int
    """Position in its tool's list after chunks dated after ``valid_at`` are
    dropped. Starts at 1."""
    episode_id: UUID
    valid_from: datetime | None


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
        top_k: int = 10,
        hops: int = MIN_HOPS,
        weight_chunks: float = 1.0,
        weight_graph: float = 1.0,
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
            top_k: Hits a search returns, and the number of chunks asked of
                each source tool.
            hops: Default expansion depth of a search, 1 or 2.
            weight_chunks: Weight of chunk hits in rank fusion.
            weight_graph: Weight of fact hits in rank fusion.

        Raises:
            MemoryValueError: If ``namespace`` is not valid, ``top_k`` is
                below 1, ``hops`` is not 1 or 2, a weight is negative or not
                finite, or both weights are 0.
            MemoryConfigError: If two sources share a name.
        """
        self._namespace = validate_namespace(namespace)
        if top_k < 1:
            raise MemoryValueError("top_k must be at least 1")
        _check_hops(hops)
        for label, weight in (
            ("weight_chunks", weight_chunks),
            ("weight_graph", weight_graph),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise MemoryValueError(f"{label} must be a finite number, 0 or more")
        if weight_chunks == 0 and weight_graph == 0:
            raise MemoryValueError("weight_chunks and weight_graph cannot both be 0")
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
        self._top_k = top_k
        self._hops = hops
        self._weight_chunks = weight_chunks
        self._weight_graph = weight_graph

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

    # --- read side ---------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        hops: int | None = None,
    ) -> SearchOutcome:
        """Return chunk and fact hits for ``query``, fused into one ranking.

        Every source tool's ``find`` and the backend's ``search`` are called
        (RET-02). Both instants default to the same clock reading (RET-01).

        Fusion (design section 5) uses ranks only, with ``k = 60`` and the
        configured weights; ranks start at 1::

            score(chunk) = w_chunks / (60 + rank in its tool's list)
            score(fact)  = w_graph / (60 + rank in the fact list)
                         + w_chunks / (60 + best rank among the chunk hits
                                       that are a source of the fact)

        The second term of a fact is 0 when none of its sources was hit.

        * Chunk rank is the position in the hit's own tool list, counted after
          chunks dated after ``valid_at`` are dropped. A fact takes the best
          (lowest) rank over all tools, so the same text from two tools
          credits a fact once. The design gives no rule for several tools;
          this is the reading of "rank in its tool's list" and "best rank".
        * A chunk takes the valid period ``[effective date, open)``. A chunk
          whose document takes effect after ``valid_at`` is dropped and gives
          no seed. A document with no known date (none in front matter, no
          reliable mtime) is not dated by this search: its chunks are kept
          with no start date.
        * Ties in score break by kind (facts before chunks), then by id: the
          fact id, or the tool name then the chunk id. The order does not
          depend on the order of the sources or of equal-rank hits. This is
          the rule of the design.
        * The result holds the best ``top_k`` hits.
        * If the backend raises ``MemoryBackendUnavailableError`` the result
          holds chunk hits only, scored ``1 / (60 + rank)`` (one retriever
          left, so the weights no longer matter), each marked ``degraded``.
          Every other error propagates, including one from a source tool.

        Args:
            query: The search text.
            valid_at: Return facts that held at this instant. Default: now.
            known_at: Return facts the store believed at this instant.
                Default: now.
            hops: Expansion depth, 1 or 2. Default: the configured depth.

        Raises:
            MemoryValueError: If an instant is not UTC or ``hops`` is invalid.
            MemoryNotSupportedError: If ``valid_at`` or ``known_at`` is set
                and the backend lacks that capability.
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span(
            SEARCH_SPAN, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("holodeck.memory.namespace", self._namespace)
            try:
                outcome = await self._search(query, valid_at, known_at, hops)
            except BaseException as exc:
                # Never the message: it can quote the query or a document.
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            facts = sum(1 for hit in outcome.hits if hit.kind == "fact")
            span.set_attribute("holodeck.memory.hits", len(outcome.hits))
            span.set_attribute("holodeck.memory.fact_hits", facts)
            span.set_attribute("holodeck.memory.chunk_hits", len(outcome.hits) - facts)
            span.set_attribute("holodeck.memory.degraded", outcome.degraded)
            return outcome

    async def _search(
        self,
        query: str,
        valid_at: datetime | None,
        known_at: datetime | None,
        hops: int | None,
    ) -> SearchOutcome:
        now = self._clock()
        capabilities = self._backend.capabilities()
        if valid_at is not None:
            require_utc(valid_at, "valid_at")
            capabilities.require(MemoryOperation.VALID_AT)
        if known_at is not None:
            require_utc(known_at, "known_at")
            capabilities.require(MemoryOperation.KNOWN_AT)
        resolved_valid = now if valid_at is None else valid_at
        resolved_known = now if known_at is None else known_at
        depth = self._hops if hops is None else hops
        _check_hops(depth)

        chunk_hits = await self._find_chunks(query, resolved_valid, now)
        seeds = tuple(dict.fromkeys(hit.episode_id for hit in chunk_hits))
        if seeds and not capabilities.supports(MemoryOperation.EXPAND):
            # Seeds are optional: without them the backend searches by text.
            seeds = ()
        try:
            facts = await self._backend.search(
                self._namespace,
                MemoryQuery(
                    text=query,
                    valid_at=resolved_valid,
                    known_at=resolved_known,
                    hops=depth,
                    limit=self._top_k,
                    seed_episode_ids=seeds,
                ),
            )
        except MemoryBackendUnavailableError:
            hits = self._degraded_hits(chunk_hits)
            return SearchOutcome(hits, resolved_valid, resolved_known, True)
        hits = self._fuse(chunk_hits, facts)
        return SearchOutcome(hits, resolved_valid, resolved_known, False)

    async def _find_chunks(
        self, query: str, valid_at: datetime, now: datetime
    ) -> list[_ChunkHit]:
        """Ask every source tool for chunks; drop those dated after ``valid_at``."""
        found = await asyncio.gather(
            *(source.find(query, self._top_k) for source in self._sources)
        )
        matches = [
            (source, list(result)[: self._top_k])
            for source, result in zip(self._sources, found, strict=True)
        ]
        dates = await self._dates_of(matches, now)
        hits: list[_ChunkHit] = []
        for source, source_matches in matches:
            rank = 0
            for match in source_matches:
                date = dates[(source.name, match.chunk.source_uri)]
                if date is not None and date > valid_at:
                    continue
                rank += 1
                hits.append(
                    _ChunkHit(
                        match=match,
                        rank=rank,
                        episode_id=episode_id_for(
                            self._namespace,
                            match.chunk.section_id,
                            match.chunk.text,
                        ),
                        valid_from=date,
                    )
                )
        return hits

    async def _dates_of(
        self,
        matches: Sequence[tuple[ChunkSource, Sequence[ChunkMatch]]],
        now: datetime,
    ) -> dict[tuple[str, str], datetime | None]:
        """Return the effective date of every document a match came from.

        One search reads each distinct document once, from the source's
        ``document()``. Nothing is kept between searches, so a date corrected
        on disk is seen by the next one.
        """
        dates: dict[tuple[str, str], datetime | None] = {}
        for source, source_matches in matches:
            for uri in sorted({match.chunk.source_uri for match in source_matches}):
                dates[(source.name, uri)] = await self._date_of(
                    await source.document(uri), now
                )
        return dates

    async def _date_of(
        self, document: SourceDocument | None, now: datetime
    ) -> datetime | None:
        if document is None:
            return None
        try:
            date, origin = await asyncio.to_thread(
                effective_date, document, self._effective_date_key, now
            )
        except MemoryValueError:
            # Ingestion reports a bad date. A search must not fail on it.
            return None
        return None if origin == "ingestion" else date

    def _fuse(
        self, chunk_hits: Sequence[_ChunkHit], facts: Sequence[FactHit]
    ) -> tuple[MemoryHit, ...]:
        best: dict[UUID, int] = {}
        for hit in chunk_hits:
            best[hit.episode_id] = min(hit.rank, best.get(hit.episode_id, hit.rank))
        fused: list[MemoryHit] = []
        for hit in chunk_hits:
            fused.append(
                self._chunk_to_hit(hit, _rrf(self._weight_chunks, hit.rank), False)
            )
        seen: set[str] = set()
        rank = 0
        for fact in facts:
            if fact.fact_id in seen:
                continue
            seen.add(fact.fact_id)
            rank += 1
            score = _rrf(self._weight_graph, rank)
            credited = [
                best[s.episode_id] for s in fact.sources if s.episode_id in best
            ]
            if credited:
                score += _rrf(self._weight_chunks, min(credited))
            source = _source_of_fact(fact, best)
            fused.append(
                MemoryHit(
                    kind="fact",
                    text=fact.text,
                    score=score,
                    source_tool="" if source is None else source.source_tool,
                    chunk_id="" if source is None else source.chunk_id,
                    source_uri="" if source is None else source.source_uri,
                    fact_id=fact.fact_id,
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    superseded_by=fact.superseded_by,
                )
            )
        fused.sort(key=_chunk_hit_key)
        return tuple(fused[: self._top_k])

    def _degraded_hits(self, chunk_hits: Sequence[_ChunkHit]) -> tuple[MemoryHit, ...]:
        hits = [
            self._chunk_to_hit(hit, _rrf(1.0, hit.rank), True) for hit in chunk_hits
        ]
        hits.sort(key=_chunk_hit_key)
        return tuple(hits[: self._top_k])

    @staticmethod
    def _chunk_to_hit(hit: _ChunkHit, score: float, degraded: bool) -> MemoryHit:
        chunk = hit.match.chunk
        return MemoryHit(
            kind="chunk",
            text=chunk.text,
            score=score,
            source_tool=chunk.source_tool,
            chunk_id=chunk.chunk_id,
            source_uri=chunk.source_uri,
            valid_from=hit.valid_from,
            degraded=degraded,
        )

    async def timeline(self, entity_ref: str) -> list[TimelineEntry]:
        """Return every fact about an entity, open or closed, by ``valid_from``.

        A pass-through to the backend in this service's namespace. An unknown
        entity returns an empty list.

        Raises:
            MemoryNotSupportedError: If the backend has no timelines.
            MemoryBackendUnavailableError: If the backend cannot be reached.
        """
        return await self._backend.timeline(self._namespace, entity_ref)

    async def provenance(self, fact_id: str) -> Provenance | None:
        """Return a fact and the episodes that support it, or None when unknown.

        A pass-through to the backend in this service's namespace. The
        backend fills the chunk text of each source from its episode record
        (design section 6), so no source tool is asked for it.

        Raises:
            MemoryNotSupportedError: If the backend has no provenance.
            MemoryBackendUnavailableError: If the backend cannot be reached.
        """
        return await self._backend.provenance(self._namespace, fact_id)

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


def _check_hops(hops: int) -> None:
    if not MIN_HOPS <= hops <= MAX_HOPS:
        raise MemoryValueError(f"hops must be {MIN_HOPS} or {MAX_HOPS}")


def _rrf(weight: float, rank: int) -> float:
    return weight / (RRF_K + rank)


def _chunk_hit_key(hit: MemoryHit) -> tuple[float, int, str, str]:
    """Order of fused hits: score, then kind (facts first), then id."""
    if hit.kind == "fact":
        return (-hit.score, 0, "", hit.fact_id or "")
    return (-hit.score, 1, hit.source_tool, hit.chunk_id)


def _source_of_fact(fact: FactHit, best: dict[UUID, int]) -> EpisodeRef | None:
    """Pick the source a fact hit names.

    The source chunk that was itself hit with the best rank when there is
    one, else the earliest source by reference time. Ties break on tool and
    chunk id, so the choice does not depend on the backend's source order.
    """
    credited = [source for source in fact.sources if source.episode_id in best]
    if credited:
        return min(
            credited,
            key=lambda s: (best[s.episode_id], s.source_tool, s.chunk_id),
        )
    if fact.sources:
        return min(
            fact.sources,
            key=lambda s: (s.reference_time, s.source_tool, s.chunk_id),
        )
    return None
