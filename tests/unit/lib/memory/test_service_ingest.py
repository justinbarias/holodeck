"""TemporalMemory ingests chunks in date order, once each (spec 043, T6)."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from holodeck.lib.memory import (
    EpisodeIn,
    IngestReport,
    MemoryBackendUnavailableError,
    MemoryConfigError,
    MemoryValueError,
    TemporalMemoryError,
)
from holodeck.lib.memory import service as service_module
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.chunk_source import (
    ChunkMatch,
    ChunkRef,
    SourceDocument,
)
from holodeck.lib.memory.service import (
    DEFAULT_EXTRACTION_INSTRUCTIONS,
    IngestRunReport,
    TemporalMemory,
    episode_id_for,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NS = "family-assistance"
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def at(year: int, month: int = 7, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# --- test doubles -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocSpec:
    """One document: its uri, front-matter date (or None) and chunk texts."""

    uri: str
    date: str | None
    texts: tuple[str, ...]
    mtime_reliable: bool = False


class FakeSource:
    """A ChunkSource over real files, with chunks the test can change."""

    def __init__(self, name: str, directory: Path, specs: Sequence[DocSpec]) -> None:
        self._name = name
        self._documents: list[SourceDocument] = []
        self.chunk_lists: dict[str, list[ChunkRef]] = {}
        self.chunk_calls = 0
        for spec in specs:
            path = directory / f"{name}_{spec.uri.replace('/', '_')}"
            front = f"---\neffective_date: {spec.date}\n---\n" if spec.date else ""
            path.write_text(f"{front}body\n", encoding="utf-8")
            self._documents.append(
                SourceDocument(
                    source_tool=name,
                    source_uri=spec.uri,
                    path=path,
                    mtime_reliable=spec.mtime_reliable,
                )
            )
            self.chunk_lists[spec.uri] = [
                ChunkRef(
                    source_tool=name,
                    chunk_id=f"{spec.uri}_chunk_{index}",
                    source_uri=spec.uri,
                    section_id=f"{spec.uri}#{index}",
                    ordinal=index,
                    text=text,
                )
                for index, text in enumerate(spec.texts)
            ]

    @property
    def name(self) -> str:
        return self._name

    async def documents(self) -> Sequence[SourceDocument]:
        return list(self._documents)

    async def document(self, source_uri: str) -> SourceDocument | None:
        for document in self._documents:
            if document.source_uri == source_uri:
                return document
        return None

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        self.chunk_calls += 1
        return list(self.chunk_lists[document.source_uri])

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        return []

    def set_text(self, uri: str, index: int, text: str) -> None:
        old = self.chunk_lists[uri][index]
        self.chunk_lists[uri][index] = replace(old, text=text)


class EmptySource(FakeSource):
    """A source whose store loader returned nothing."""

    async def documents(self) -> Sequence[SourceDocument]:
        return []


class SpyBackend(FakeBackend):
    """FakeBackend that records submissions and can add token counts."""

    def __init__(
        self,
        tokens: Callable[[int], tuple[int | None, int | None]] | None = None,
    ) -> None:
        super().__init__()
        self.submitted: list[tuple[EpisodeIn, ...]] = []
        self.unavailable_on_ingest = False
        self._tokens = tokens
        self._ingest_calls = 0

    async def ingest(
        self, namespace: str, episodes: Sequence[EpisodeIn], instructions: str
    ) -> IngestReport:
        self.submitted.append(tuple(episodes))
        if self.unavailable_on_ingest:
            raise MemoryBackendUnavailableError("spy", "down during ingest")
        report = await super().ingest(namespace, episodes, instructions)
        self._ingest_calls += 1
        if self._tokens is None:
            return report
        input_tokens, output_tokens = self._tokens(self._ingest_calls)
        return replace(report, input_tokens=input_tokens, output_tokens=output_tokens)

    @property
    def episodes(self) -> list[EpisodeIn]:
        """Every submitted episode, in submission order."""
        flat = [episode for batch in self.submitted for episode in batch]
        assert len(flat) == len(self.submitted), "one episode per ingest call"
        return flat


def memory(
    backend: FakeBackend,
    sources: Sequence[FakeSource],
    **kwargs: object,
) -> TemporalMemory:
    options: dict[str, object] = {"clock": lambda: NOW, **kwargs}
    return TemporalMemory(
        namespace=NS,
        backend=backend,
        sources=list(sources),
        **options,  # type: ignore[arg-type]
    )


def id_of(uri: str, index: int, text: str) -> UUID:
    return episode_id_for(NS, f"{uri}#{index}", text)


# --- episode identity -------------------------------------------------------


async def test_episode_id_for_follows_the_contract_formula() -> None:
    digest = hashlib.sha256(b"s1\ntext").hexdigest()
    expected = uuid5(uuid5(NAMESPACE_URL, "holodeck/memory/ns"), digest)

    assert episode_id_for("ns", "s1", "text") == expected


async def test_episode_id_for_changes_with_namespace_section_and_text() -> None:
    base = episode_id_for("ns", "s1", "text")

    assert episode_id_for("other", "s1", "text") != base
    assert episode_id_for("ns", "s2", "text") != base
    assert episode_id_for("ns", "s1", "other text") != base
    assert episode_id_for("ns", "s1", "text") == base


# --- construction and selection ---------------------------------------------


async def test_namespace_is_validated(tmp_path: Path) -> None:
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("x",))])

    with pytest.raises(MemoryValueError, match="bad namespace"):
        TemporalMemory(
            namespace="bad namespace", backend=FakeBackend(), sources=[source]
        )


async def test_source_names_must_be_unique(tmp_path: Path) -> None:
    first = FakeSource("a", tmp_path, [DocSpec("d1", "2020-07-01", ("x",))])
    second = FakeSource("a", tmp_path, [DocSpec("d2", "2020-07-01", ("y",))])

    with pytest.raises(MemoryConfigError, match="unique.*a"):
        TemporalMemory(namespace=NS, backend=FakeBackend(), sources=[first, second])


async def test_unknown_source_tool_names_the_tool_and_writes_nothing(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("x",))])

    with pytest.raises(MemoryConfigError, match="nope"):
        await memory(backend, [source]).ingest(source_tools=["a", "nope"])

    assert backend.calls["ingest"] == 0
    assert backend.calls["has_episode"] == 0


async def test_source_tools_limits_the_run_to_those_names(tmp_path: Path) -> None:
    backend = SpyBackend()
    a = FakeSource("a", tmp_path, [DocSpec("da", "2020-07-01", ("from a",))])
    b = FakeSource("b", tmp_path, [DocSpec("db", "2021-07-01", ("from b",))])

    report = await memory(backend, [a, b]).ingest(source_tools=["b"])

    assert report.documents == 1
    assert [episode.source_tool for episode in backend.episodes] == ["b"]
    assert [episode.text for episode in backend.episodes] == ["from b"]


async def test_a_source_with_no_documents_is_an_error_before_any_write(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    good = FakeSource("good", tmp_path, [DocSpec("d", "2020-07-01", ("x",))])
    empty = EmptySource("hollow", tmp_path, [])

    with pytest.raises(TemporalMemoryError, match="hollow.*returned no documents"):
        await memory(backend, [good, empty]).ingest()

    assert backend.calls["ingest"] == 0
    assert backend.calls["has_episode"] == 0


# --- ING-04: one date order -------------------------------------------------


async def test_ing_04_documents_of_all_sources_go_in_ascending_date_order(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    a = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("a/2022", "2022-07-01", ("rate 2022",)),
            DocSpec("a/2020", "2020-07-01", ("rate 2020",)),
        ],
    )
    b = FakeSource("b", tmp_path, [DocSpec("b/2019", "2019-07-01", ("rate 2019",))])

    report = await memory(backend, [a, b]).ingest()

    assert [e.reference_time for e in backend.episodes] == [
        at(2019),
        at(2020),
        at(2022),
    ]
    assert [e.text for e in backend.episodes] == ["rate 2019", "rate 2020", "rate 2022"]
    assert report.documents == 3
    assert report.episodes_new == 3


async def test_ing_04_equal_dates_break_ties_by_source_position_then_uri(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    b = FakeSource("b", tmp_path, [DocSpec("x", "2020-07-01", ("b x",))])
    a = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("z", "2020-07-01", ("a z",)),
            DocSpec("m", "2020-07-01", ("a m",)),
        ],
    )

    await memory(backend, [b, a]).ingest()

    # "b" is first in the configured list, so it goes first on a date tie.
    assert [e.text for e in backend.episodes] == ["b x", "a m", "a z"]


async def test_ing_04_chunks_of_a_document_go_in_ascending_ordinal(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource(
        "a", tmp_path, [DocSpec("d", "2020-07-01", ("zero", "one", "two"))]
    )
    source.chunk_lists["d"].reverse()

    await memory(backend, [source]).ingest()

    assert [e.text for e in backend.episodes] == ["zero", "one", "two"]


# --- episode contents -------------------------------------------------------


async def test_episode_carries_the_chunk_and_the_document_date(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("law/doc.md", "2020-07-01", ("t0",))])

    await memory(backend, [source]).ingest()

    (episode,) = backend.episodes
    assert episode.episode_id == id_of("law/doc.md", 0, "t0")
    assert episode.source_tool == "a"
    assert episode.chunk_id == "law/doc.md_chunk_0"
    assert episode.source_uri == "law/doc.md"
    assert episode.text == "t0"
    assert episode.reference_time == at(2020)
    assert episode.reference_time_source == "front_matter"
    assert episode.context_episode_ids == ()


async def test_ing_03_date_falls_back_to_mtime_then_to_the_clock(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("local", None, ("local text",), mtime_reliable=True),
            DocSpec("remote", None, ("remote text",), mtime_reliable=False),
        ],
    )
    local_path = next(
        d.path for d in await source.documents() if d.source_uri == "local"
    )
    mtime = datetime(2015, 3, 4, tzinfo=UTC)
    os.utime(local_path, (mtime.timestamp(), mtime.timestamp()))

    await memory(backend, [source]).ingest()

    by_text = {e.text: e for e in backend.episodes}
    assert by_text["local text"].reference_time == mtime
    assert by_text["local text"].reference_time_source == "mtime"
    assert by_text["remote text"].reference_time == NOW
    assert by_text["remote text"].reference_time_source == "ingestion"


async def test_the_clock_is_read_once_per_run(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource(
        "a",
        tmp_path,
        [DocSpec(f"d{i}", None, (f"t{i}",), mtime_reliable=False) for i in range(3)],
    )
    reads: list[datetime] = []

    def clock() -> datetime:
        reads.append(NOW.replace(second=len(reads) + 1))
        return reads[-1]

    await memory(backend, [source], clock=clock).ingest()

    assert len(reads) == 1
    assert {e.reference_time for e in backend.episodes} == {reads[0]}
    assert len(backend.episodes) == 3


async def test_context_is_the_two_preceding_chunks_of_the_same_document(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    texts = ("c0", "c1", "c2", "c3")
    source = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("first", "2019-07-01", ("f0", "f1")),
            DocSpec("second", "2020-07-01", texts),
        ],
    )

    await memory(backend, [source]).ingest()

    second = [e for e in backend.episodes if e.source_uri == "second"]
    ids = [id_of("second", i, t) for i, t in enumerate(texts)]
    assert [e.context_episode_ids for e in second] == [
        (),
        (ids[0],),
        (ids[0], ids[1]),
        (ids[1], ids[2]),
    ]
    first = [e for e in backend.episodes if e.source_uri == "first"]
    assert first[0].context_episode_ids == ()
    assert first[1].context_episode_ids == (id_of("first", 0, "f0"),)


async def test_context_still_names_chunks_that_were_skipped(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("c0", "c1", "c2"))])
    service = memory(backend, [source])
    await service.ingest()
    backend.submitted.clear()
    source.set_text("d", 2, "c2 changed")

    report = await service.ingest()

    assert report.episodes_new == 1
    (episode,) = backend.episodes
    assert episode.text == "c2 changed"
    assert episode.context_episode_ids == (id_of("d", 0, "c0"), id_of("d", 1, "c1"))


async def test_a_repeated_chunk_in_one_document_is_not_its_own_context(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("x", "x", "y"))])
    same_section = source.chunk_lists["d"]
    same_section[1] = replace(same_section[1], section_id=same_section[0].section_id)

    report = await memory(backend, [source]).ingest()

    assert [e.text for e in backend.episodes] == ["x", "y"]
    assert report.episodes_new == 2
    assert report.episodes_skipped == 1
    assert backend.episodes[1].context_episode_ids == (episode_id_for(NS, "d#0", "x"),)


# --- A8: one episode when two tools index the same file ---------------------


async def test_a8_the_same_chunk_from_two_tools_is_submitted_once(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    spec = DocSpec("shared.md", "2020-07-01", ("t0", "t1"))
    a = FakeSource("a", tmp_path, [spec])
    b = FakeSource("b", tmp_path, [spec])

    report = await memory(backend, [b, a]).ingest()

    # The first tool in the configured list owns the shared chunks.
    assert [(e.source_tool, e.text) for e in backend.episodes] == [
        ("b", "t0"),
        ("b", "t1"),
    ]
    assert report.documents == 2
    assert report.episodes_new == 2
    assert report.episodes_skipped == 2
    assert len(report.written) == 2


# --- ING-01, ING-02: skipping complete episodes -----------------------------


async def test_ing_01_ing_02_a_second_run_makes_no_write_and_no_llm_call(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("d1", "2019-07-01", ("a", "b")),
            DocSpec("d2", "2020-07-01", ("c", "d", "e")),
        ],
    )
    service = memory(backend, [source])
    first = await service.ingest()
    ingest_calls = backend.calls["ingest"]
    llm_calls = backend.llm_calls
    assert first.episodes_new == 5
    assert ingest_calls == 5
    assert llm_calls == 5

    second = await service.ingest()

    assert second.episodes_new == 0
    assert second.episodes_skipped == 5
    assert second.written == ()
    assert second.llm_calls == 0
    assert backend.calls["ingest"] == ingest_calls
    assert backend.llm_calls == llm_calls


async def test_ing_01_only_a_changed_chunk_is_submitted_again(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])
    service = memory(backend, [source])
    await service.ingest()
    backend.submitted.clear()
    source.set_text("d", 1, "b changed")

    report = await service.ingest()

    assert report.episodes_new == 1
    assert report.episodes_skipped == 2
    assert [e.text for e in backend.episodes] == ["b changed"]
    assert report.written == (id_of("d", 1, "b changed"),)


async def test_each_backend_call_carries_exactly_one_episode(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])

    await memory(backend, [source]).ingest()

    assert [len(batch) for batch in backend.submitted] == [1, 1, 1]


# --- ING-10: instructions ---------------------------------------------------


async def test_ing_10_the_default_instructions_reach_the_backend(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b"))])

    await memory(backend, [source]).ingest()

    assert DEFAULT_EXTRACTION_INSTRUCTIONS.startswith(
        "Extract amounts, rates, percentages, thresholds and dates as entities"
    )
    assert DEFAULT_EXTRACTION_INSTRUCTIONS.endswith(
        "use it as the fact's valid period."
    )
    assert backend.instructions == [DEFAULT_EXTRACTION_INSTRUCTIONS] * 2


async def test_ing_10_configured_instructions_replace_the_default(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    await memory(backend, [source], instructions="only dates").ingest()

    assert backend.instructions == ["only dates"]


# --- ING-05: stop on the first failure --------------------------------------


async def test_ing_05_a_failed_episode_stops_the_run(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])
    first, second, third = (id_of("d", i, t) for i, t in enumerate("abc"))
    backend.fail_on = {second}

    report = await memory(backend, [source]).ingest()

    assert report.written == (first,)
    assert report.failed == second
    assert report.error is not None
    assert "extraction failed" in report.error
    assert backend.calls["ingest"] == 2
    assert third not in {e.episode_id for e in backend.episodes}
    assert await backend.has_episode(NS, first) is True
    assert await backend.has_episode(NS, third) is False


async def test_ing_05_a_failure_in_an_early_document_skips_later_documents(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource(
        "a",
        tmp_path,
        [
            DocSpec("old", "2019-07-01", ("o0",)),
            DocSpec("new", "2020-07-01", ("n0",)),
        ],
    )
    backend.fail_on = {id_of("old", 0, "o0")}

    report = await memory(backend, [source]).ingest()

    assert report.written == ()
    assert report.failed == id_of("old", 0, "o0")
    assert [e.text for e in backend.episodes] == ["o0"]


async def test_ing_05_the_next_run_resumes_at_the_failed_episode(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])
    service = memory(backend, [source])
    backend.fail_on = {id_of("d", 1, "b")}
    await service.ingest()
    backend.fail_on = set()
    backend.submitted.clear()

    report = await service.ingest()

    assert report.failed is None
    assert [e.text for e in backend.episodes] == ["b", "c"]
    assert report.written == (id_of("d", 1, "b"), id_of("d", 2, "c"))
    assert report.episodes_skipped == 1


async def test_ing_05_backend_unavailable_propagates_from_has_episode(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    backend.available = False
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    with pytest.raises(MemoryBackendUnavailableError):
        await memory(backend, [source]).ingest()


async def test_ing_05_backend_unavailable_propagates_from_ingest(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    backend.unavailable_on_ingest = True
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    with pytest.raises(MemoryBackendUnavailableError):
        await memory(backend, [source]).ingest()

    assert len(backend.submitted) == 1


# --- ING-08: dry run --------------------------------------------------------


async def test_ing_08_dry_run_counts_new_episodes_and_estimates_the_time(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])
    service = memory(backend, [source], seconds_per_episode=2.5)
    await service.ingest(source_tools=["a"])  # leaves nothing new
    source.set_text("d", 0, "a changed")
    source.set_text("d", 2, "c changed")
    ingest_calls = backend.calls["ingest"]
    llm_calls = backend.llm_calls

    report = await service.ingest(dry_run=True)

    assert report.dry_run is True
    assert report.episodes_new == 2
    assert report.episodes_skipped == 1
    assert report.written == ()
    assert report.estimated_seconds == 5.0
    assert report.llm_calls == 0
    assert backend.calls["ingest"] == ingest_calls
    assert backend.llm_calls == llm_calls
    assert await backend.episode_count(NS) == 3


async def test_ing_08_dry_run_without_a_rate_gives_no_estimate(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b"))])

    report = await memory(backend, [source]).ingest(dry_run=True)

    assert report.episodes_new == 2
    assert report.estimated_seconds is None
    assert backend.calls["ingest"] == 0
    assert backend.calls["has_episode"] == 2


async def test_a_real_run_has_no_estimate(tmp_path: Path) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    report = await memory(backend, [source], seconds_per_episode=2.0).ingest()

    assert report.dry_run is False
    assert report.estimated_seconds is None


# --- report totals ----------------------------------------------------------


async def test_token_totals_sum_the_backend_reports(tmp_path: Path) -> None:
    backend = SpyBackend(tokens=lambda n: (100 * n, 10 * n))
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])

    report = await memory(backend, [source]).ingest()

    assert report.input_tokens == 600
    assert report.output_tokens == 60
    assert report.llm_calls == 3


async def test_token_totals_skip_reports_without_a_count(tmp_path: Path) -> None:
    backend = SpyBackend(tokens=lambda n: (50, None) if n == 2 else (None, None))
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b", "c"))])

    report = await memory(backend, [source]).ingest()

    assert report.input_tokens == 50
    assert report.output_tokens is None


async def test_token_totals_are_none_when_no_report_has_a_count(
    tmp_path: Path,
) -> None:
    backend = SpyBackend()
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a", "b"))])

    report = await memory(backend, [source]).ingest()

    assert report.llm_calls == 2
    assert report.input_tokens is None
    assert report.output_tokens is None


async def test_duration_is_measured_with_time_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([100.0, 107.5])
    monkeypatch.setattr(
        service_module, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    report = await memory(SpyBackend(), [source]).ingest()

    assert report.duration_seconds == 7.5


async def test_the_report_is_a_frozen_ingest_run_report(tmp_path: Path) -> None:
    source = FakeSource("a", tmp_path, [DocSpec("d", "2020-07-01", ("a",))])

    report = await memory(SpyBackend(), [source]).ingest()

    assert isinstance(report, IngestRunReport)
    assert report.failed is None
    assert report.error is None
    with pytest.raises(AttributeError):
        report.documents = 9  # type: ignore[misc]


# --- property tests (seeded, no extra dependency) ---------------------------

SEEDS = range(25)


@dataclass(frozen=True, slots=True)
class Corpus:
    sources: list[FakeSource]
    chunk_count: int

    def pick_chunk(self, rng: random.Random) -> tuple[FakeSource, str, int]:
        source = rng.choice(self.sources)
        uri = rng.choice(sorted(source.chunk_lists))
        index = rng.randrange(len(source.chunk_lists[uri]))
        return source, uri, index


def random_corpus(seed: int, directory: Path) -> Corpus:
    rng = random.Random(seed)  # noqa: S311 - reproducible test input
    sources: list[FakeSource] = []
    chunk_count = 0
    for s in range(rng.randint(1, 3)):
        specs: list[DocSpec] = []
        for d in range(rng.randint(1, 4)):
            when = at(rng.randint(2000, 2030), rng.randint(1, 12), rng.randint(1, 28))
            texts = tuple(
                f"seed{seed} s{s} d{d} c{c} n{rng.randint(0, 10**6)}"
                for c in range(rng.randint(1, 5))
            )
            chunk_count += len(texts)
            specs.append(DocSpec(f"s{s}/d{d}.md", when.strftime("%Y-%m-%d"), texts))
        sources.append(FakeSource(f"s{s}", directory, specs))
    return Corpus(sources, chunk_count)


@pytest.mark.parametrize("seed", SEEDS)
async def test_ing_01_property_reingesting_an_unchanged_corpus_is_a_no_op(
    seed: int, tmp_path: Path
) -> None:
    corpus = random_corpus(seed, tmp_path)
    backend = SpyBackend()
    service = memory(backend, corpus.sources)

    first = await service.ingest()
    ingest_calls = backend.calls["ingest"]
    llm_calls = backend.llm_calls
    second = await service.ingest()

    assert corpus.chunk_count >= 1
    assert first.episodes_new == corpus.chunk_count
    assert len(first.written) == corpus.chunk_count
    assert second.episodes_new == 0
    assert second.episodes_skipped == corpus.chunk_count
    assert second.written == ()
    assert backend.calls["ingest"] == ingest_calls == corpus.chunk_count
    assert backend.llm_calls == llm_calls == corpus.chunk_count


@pytest.mark.parametrize("seed", SEEDS)
async def test_ing_01_property_changing_one_chunk_yields_one_new_episode(
    seed: int, tmp_path: Path
) -> None:
    corpus = random_corpus(seed, tmp_path)
    backend = SpyBackend()
    service = memory(backend, corpus.sources)
    await service.ingest()
    backend.submitted.clear()
    source, uri, index = corpus.pick_chunk(
        random.Random(seed + 1000)  # noqa: S311 - reproducible test input
    )
    source.set_text(uri, index, f"edited in seed {seed}")
    ingest_calls = backend.calls["ingest"]

    report = await service.ingest()

    assert report.episodes_new == 1
    assert report.episodes_skipped == corpus.chunk_count - 1
    assert backend.calls["ingest"] == ingest_calls + 1
    assert [e.text for e in backend.episodes] == [f"edited in seed {seed}"]
    assert report.written == (id_of(uri, index, f"edited in seed {seed}"),)


@pytest.mark.parametrize("seed", SEEDS)
async def test_ing_04_property_the_backend_sees_non_decreasing_reference_times(
    seed: int, tmp_path: Path
) -> None:
    corpus = random_corpus(seed, tmp_path)
    backend = SpyBackend()

    await memory(backend, corpus.sources).ingest()

    times = [e.reference_time for e in backend.episodes]
    assert len(times) == corpus.chunk_count
    assert times == sorted(times)
    # Each document's chunks stay together and in ordinal order.
    for source in corpus.sources:
        for uri, chunks in source.chunk_lists.items():
            submitted = [e.text for e in backend.episodes if e.source_uri == uri]
            assert submitted == [chunk.text for chunk in chunks]
