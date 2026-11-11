"""A deterministic in-process memory backend for tests. Data is lost on close.

It makes no LLM or network call. "Extraction" is a rule the test controls:

* By default each episode yields one fact whose subject and text are the
  episode text.
* Pass ``extract`` to return ``(subject, text)`` pairs per episode.

Invalidation follows the rule the real backends are expected to follow: two
facts with the same subject cannot both be open at one instant. A newer fact
closes the older one at the newer fact's ``valid_from``. A fact that arrives
late but is older is closed at the next fact's ``valid_from``.

The store is bi-temporal. Closing a fact keeps the version the store
believed before, so a search with an earlier ``known_at`` still sees the
fact as open. This is the behaviour a real backend must reproduce.

System time comes from ``clock``. The default clock is a counter, so two
runs of the same test record the same instants.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid5

from holodeck.lib.memory.errors import MemoryBackendUnavailableError
from holodeck.lib.memory.types import (
    EpisodeIn,
    EpisodeRef,
    FactHit,
    IngestReport,
    MemoryCapabilities,
    MemoryOperation,
    MemoryQuery,
    Provenance,
    TimelineEntry,
)

NAME = "fake"

ALL_OPERATIONS = frozenset(MemoryOperation)

Extract = Callable[[EpisodeIn], Sequence[tuple[str, str]]]
Clock = Callable[[], datetime]

_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def _one_fact_per_episode(episode: EpisodeIn) -> Sequence[tuple[str, str]]:
    return [(episode.text, episode.text)]


def counter_clock(start: datetime = _EPOCH) -> Clock:
    """Return a clock that advances one second on every call."""
    ticks = 0

    def now() -> datetime:
        nonlocal ticks
        ticks += 1
        return start + timedelta(seconds=ticks)

    return now


@dataclass(frozen=True, slots=True)
class _StoredFact:
    fact_id: str
    subject: str
    text: str
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    expired_at: datetime | None
    episode_ids: tuple[UUID, ...]
    superseded_by: str | None = None

    def valid_at(self, instant: datetime) -> bool:
        return self.valid_from <= instant and (
            self.valid_to is None or instant < self.valid_to
        )

    def known_at(self, instant: datetime) -> bool:
        return self.recorded_at <= instant and (
            self.expired_at is None or instant < self.expired_at
        )


class _Namespace:
    def __init__(self) -> None:
        self.episodes: dict[UUID, EpisodeRef] = {}
        self.facts: dict[str, _StoredFact] = {}
        self.history: list[_StoredFact] = []
        self.texts: dict[UUID, str] = {}


class FakeBackend:
    """In-memory ``MemoryBackend``.

    Attributes:
        calls: How many times each port method was called.
        llm_calls: One per episode ingested, standing in for extraction.
        available: Set False to make every store operation raise
            ``MemoryBackendUnavailableError``.
        fail_on: Episode ids whose ingestion fails.
        instructions: The instructions passed to each ``ingest`` call.
    """

    def __init__(
        self,
        *,
        extract: Extract = _one_fact_per_episode,
        clock: Clock | None = None,
        supported: frozenset[MemoryOperation] = ALL_OPERATIONS,
    ) -> None:
        self._extract = extract
        self._clock = clock or counter_clock()
        self._capabilities = MemoryCapabilities(backend=NAME, supported=supported)
        self._namespaces: dict[str, _Namespace] = {}
        self.calls: Counter[str] = Counter()
        self.llm_calls = 0
        self.available = True
        self.fail_on: set[UUID] = set()
        self.instructions: list[str] = []

    @property
    def name(self) -> str:
        return NAME

    def capabilities(self) -> MemoryCapabilities:
        return self._capabilities

    def _enter(self, method: str, operation: MemoryOperation | None = None) -> None:
        self.calls[method] += 1
        if operation is not None:
            self._capabilities.require(operation)
        if not self.available:
            raise MemoryBackendUnavailableError(NAME, "marked unavailable by the test")

    def _space(self, namespace: str) -> _Namespace:
        return self._namespaces.setdefault(namespace, _Namespace())

    async def has_episode(self, namespace: str, episode_id: UUID) -> bool:
        self._enter("has_episode")
        return episode_id in self._space(namespace).episodes

    async def episode_count(self, namespace: str) -> int:
        self._enter("episode_count")
        return len(self._space(namespace).episodes)

    async def ingest(
        self,
        namespace: str,
        episodes: Sequence[EpisodeIn],
        instructions: str,
    ) -> IngestReport:
        self._enter("ingest", MemoryOperation.INGEST)
        self.instructions.append(instructions)
        space = self._space(namespace)
        written: list[UUID] = []
        calls_before = self.llm_calls
        for episode in episodes:
            if episode.episode_id in self.fail_on:
                return IngestReport(
                    written=tuple(written),
                    failed=episode.episode_id,
                    error="extraction failed (set by the test)",
                    llm_calls=self.llm_calls - calls_before,
                )
            self.llm_calls += 1
            for index, (subject, text) in enumerate(self._extract(episode)):
                self._add_fact(space, episode, index, subject, text)
            space.episodes[episode.episode_id] = EpisodeRef(
                episode_id=episode.episode_id,
                source_tool=episode.source_tool,
                chunk_id=episode.chunk_id,
                source_uri=episode.source_uri,
                reference_time=episode.reference_time,
                reference_time_source=episode.reference_time_source,
            )
            space.texts[episode.episode_id] = episode.text
            written.append(episode.episode_id)
        return IngestReport(
            written=tuple(written), llm_calls=self.llm_calls - calls_before
        )

    def _add_fact(
        self,
        space: _Namespace,
        episode: EpisodeIn,
        index: int,
        subject: str,
        text: str,
    ) -> None:
        now = self._clock()
        fact_id = str(uuid5(episode.episode_id, f"fact-{index}"))
        valid_from = episode.reference_time
        same_subject = [f for f in space.facts.values() if f.subject == subject]

        duplicate = next((f for f in same_subject if f.text == text), None)
        if duplicate is not None:
            space.facts[duplicate.fact_id] = replace(
                duplicate, episode_ids=(*duplicate.episode_ids, episode.episode_id)
            )
            return

        later = sorted(
            (f for f in same_subject if f.valid_from > valid_from),
            key=lambda f: f.valid_from,
        )
        valid_to = later[0].valid_from if later else None
        superseded_by = later[0].fact_id if later else None

        for older in same_subject:
            if older.valid_from <= valid_from and older.valid_at(valid_from):
                # Keep what the store believed until now, then record the
                # closed version as what it believes from now on.
                space.history.append(replace(older, expired_at=now))
                space.facts[older.fact_id] = replace(
                    older, valid_to=valid_from, recorded_at=now, superseded_by=fact_id
                )

        space.facts[fact_id] = _StoredFact(
            fact_id=fact_id,
            subject=subject,
            text=text,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_at=now,
            expired_at=None,
            episode_ids=(episode.episode_id,),
            superseded_by=superseded_by,
        )

    def _sources(self, space: _Namespace, fact: _StoredFact) -> tuple[EpisodeRef, ...]:
        return tuple(space.episodes[i] for i in fact.episode_ids if i in space.episodes)

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        self._enter("search", MemoryOperation.SEARCH)
        space = self._space(namespace)
        words = set(query.text.lower().split())
        seeds = set(query.seed_episode_ids)
        hits: list[FactHit] = []
        for fact in (*space.facts.values(), *space.history):
            if not (fact.valid_at(query.valid_at) and fact.known_at(query.known_at)):
                continue
            overlap = len(words & set(fact.text.lower().split()))
            seeded = bool(seeds & set(fact.episode_ids))
            if overlap == 0 and not seeded:
                continue
            hits.append(
                FactHit(
                    fact_id=fact.fact_id,
                    text=fact.text,
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    recorded_at=fact.recorded_at,
                    expired_at=fact.expired_at,
                    sources=self._sources(space, fact),
                    score=float(overlap) + (1.0 if seeded else 0.0),
                    superseded_by=fact.superseded_by,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.fact_id))
        return hits[: query.limit]

    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]:
        self._enter("timeline", MemoryOperation.TIMELINE)
        space = self._space(namespace)
        entries = [
            TimelineEntry(
                fact_id=fact.fact_id,
                text=fact.text,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                recorded_at=fact.recorded_at,
                expired_at=fact.expired_at,
                sources=self._sources(space, fact),
            )
            for fact in space.facts.values()
            if fact.subject == entity_ref
        ]
        entries.sort(key=lambda entry: (entry.valid_from or _EPOCH, entry.fact_id))
        return entries

    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None:
        self._enter("provenance", MemoryOperation.PROVENANCE)
        space = self._space(namespace)
        fact = space.facts.get(fact_id)
        if fact is None:
            return None
        return Provenance(
            fact_id=fact.fact_id,
            text=fact.text,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            recorded_at=fact.recorded_at,
            expired_at=fact.expired_at,
            sources=tuple(
                replace(source, text=space.texts[source.episode_id])
                for source in self._sources(space, fact)
            ),
        )

    async def purge(self, namespace: str) -> None:
        self._enter("purge", MemoryOperation.PURGE)
        self._namespaces.pop(namespace, None)

    async def close(self) -> None:
        self.calls["close"] += 1
        self._namespaces.clear()
