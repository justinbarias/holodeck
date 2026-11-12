"""TemporalMemory searches chunks and facts and fuses them (spec 043, T12)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from holodeck.lib.memory import (
    MemoryBackendUnavailableError,
    MemoryNotSupportedError,
    MemoryOperation,
    MemoryValueError,
)
from holodeck.lib.memory import service as service_module
from holodeck.lib.memory.backends.fake import ALL_OPERATIONS, FakeBackend
from holodeck.lib.memory.chunk_source import ChunkMatch
from holodeck.lib.memory.service import SEARCH_SPAN, TemporalMemory, episode_id_for
from holodeck.lib.memory.tool_api import MemorySearchResult
from holodeck.lib.memory.types import FactHit, MemoryHit, MemoryQuery
from tests.unit.lib.memory.test_service_ingest import DocSpec, FakeSource

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NS = "family-assistance"
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
QUERY = "rate"
DOC_DATE = datetime(2020, 7, 1, tzinfo=UTC)


# --- test doubles -----------------------------------------------------------


class RankedSource(FakeSource):
    """A FakeSource whose ``find`` returns a ranked list the test sets."""

    def __init__(self, name: str, directory: Path, specs: Sequence[DocSpec]) -> None:
        super().__init__(name, directory, specs)
        self.ranked: list[tuple[str, int]] = []
        self.find_calls: list[tuple[str, int]] = []
        self.find_error: Exception | None = None
        self.document_calls: list[str] = []
        self.listing_forbidden = False

    async def documents(self):  # type: ignore[no-untyped-def]
        if self.listing_forbidden:
            raise AssertionError("a search must not list the store's documents")
        return await super().documents()

    async def document(self, source_uri: str):  # type: ignore[no-untyped-def]
        self.document_calls.append(source_uri)
        return await super().document(source_uri)

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        self.find_calls.append((query, top_k))
        if self.find_error is not None:
            raise self.find_error
        return [
            ChunkMatch(chunk=self.chunk_lists[uri][index], score=100.0 - rank)
            for rank, (uri, index) in enumerate(self.ranked)
        ]


class OrderedBackend(FakeBackend):
    """FakeBackend that returns facts in an order the test chooses.

    The fake ranks by word overlap, and a seed changes that rank. The tests
    fix the graph rank instead, by fact text, so the fusion maths is checked
    against known ranks.
    """

    def __init__(
        self, *, supported: frozenset[MemoryOperation] = ALL_OPERATIONS
    ) -> None:
        super().__init__(supported=supported)
        self.order: list[str] = []
        self.queries: list[MemoryQuery] = []
        self.search_error: Exception | None = None

    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]:
        self.queries.append(query)
        if self.search_error is not None:
            raise self.search_error
        # The fake cuts to ``limit`` by its own rank; cut after the test's order.
        hits = await super().search(namespace, replace(query, limit=10_000))
        position = {text: index for index, text in enumerate(self.order)}
        hits.sort(key=lambda hit: position.get(hit.text, len(position)))
        return hits[: query.limit]


def one_doc(
    directory: Path,
    texts: Sequence[str],
    *,
    name: str = "rules",
    date: str | None = "2020-07-01",
) -> RankedSource:
    return RankedSource(name, directory, [DocSpec("a.md", date, tuple(texts))])


def memory(
    backend: FakeBackend, sources: Sequence[FakeSource], **kwargs: object
) -> TemporalMemory:
    options: dict[str, object] = {"clock": lambda: NOW, **kwargs}
    return TemporalMemory(
        namespace=NS,
        backend=backend,
        sources=list(sources),
        **options,  # type: ignore[arg-type]
    )


async def ingested(
    backend: FakeBackend, sources: Sequence[FakeSource], **kwargs: object
) -> TemporalMemory:
    service = memory(backend, sources, **kwargs)
    report = await service.ingest()
    assert report.failed is None
    assert report.episodes_new > 0
    return service


def fact_ids(backend_hits: Sequence[FactHit]) -> dict[str, str]:
    return {hit.text: hit.fact_id for hit in backend_hits}


async def facts_by_text(backend: FakeBackend) -> dict[str, str]:
    everything = await backend.search(
        NS, MemoryQuery(text=QUERY, valid_at=NOW, known_at=NOW, limit=100)
    )
    assert everything
    return fact_ids(everything)


def kinds(hits: Sequence[MemoryHit]) -> list[str]:
    return [hit.kind for hit in hits]


# The three-chunk document of the worked example. Every text holds the query
# word, so the fake backend returns every fact.
F1 = "rate one"
F2 = "rate two"
F3 = "rate three"


async def worked_example(
    tmp_path: Path, **kwargs: object
) -> tuple[TemporalMemory, OrderedBackend, RankedSource]:
    source = one_doc(tmp_path, [F1, F2, F3])
    backend = OrderedBackend()
    backend.order = [F1, F2, F3]  # graph rank 1, 2, 3
    service = await ingested(backend, [source], **kwargs)
    return service, backend, source


# --- RET-01 -----------------------------------------------------------------


async def test_ret_01_both_instants_default_to_the_clock(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)

    outcome = await service.search(QUERY)

    assert outcome.hits
    assert outcome.valid_at == NOW
    assert outcome.known_at == NOW
    assert backend.queries[-1].valid_at == NOW
    assert backend.queries[-1].known_at == NOW


async def test_ret_01_explicit_instants_are_used_and_returned(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)
    valid_at = datetime(2021, 3, 1, tzinfo=UTC)
    known_at = datetime(2030, 1, 1, tzinfo=UTC)

    outcome = await service.search(QUERY, valid_at=valid_at, known_at=known_at)

    assert outcome.valid_at == valid_at
    assert outcome.known_at == known_at
    assert backend.queries[-1].valid_at == valid_at
    assert backend.queries[-1].known_at == known_at
    assert outcome.hits


async def test_ret_01_an_instant_must_be_utc(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)

    with pytest.raises(MemoryValueError, match="valid_at"):
        await service.search(QUERY, valid_at=datetime(2021, 1, 1))
    with pytest.raises(MemoryValueError, match="known_at"):
        await service.search(QUERY, known_at=datetime(2021, 1, 1))
    assert backend.queries == []


async def test_a_explicit_valid_at_needs_the_valid_at_capability(
    tmp_path: Path,
) -> None:
    source = one_doc(tmp_path, [F1])
    backend = OrderedBackend(supported=ALL_OPERATIONS - {MemoryOperation.VALID_AT})
    service = await ingested(backend, [source])

    # The default query asks for nothing the backend lacks.
    assert (await service.search(QUERY)).hits
    with pytest.raises(MemoryNotSupportedError) as raised:
        await service.search(QUERY, valid_at=NOW)

    assert raised.value.operation == MemoryOperation.VALID_AT.value
    assert source.find_calls == [(QUERY, 10)]
    assert len(backend.queries) == 1


async def test_an_explicit_known_at_needs_the_known_at_capability(
    tmp_path: Path,
) -> None:
    source = one_doc(tmp_path, [F1])
    backend = OrderedBackend(supported=ALL_OPERATIONS - {MemoryOperation.KNOWN_AT})
    service = await ingested(backend, [source])

    assert (await service.search(QUERY, valid_at=NOW)).hits
    with pytest.raises(MemoryNotSupportedError) as raised:
        await service.search(QUERY, known_at=NOW)

    assert raised.value.operation == MemoryOperation.KNOWN_AT.value
    assert len(backend.queries) == 1


# --- RET-02 -----------------------------------------------------------------


async def test_ret_02_every_source_and_the_backend_are_searched(
    tmp_path: Path,
) -> None:
    first = one_doc(tmp_path, [F1, F2], name="first")
    second = one_doc(tmp_path, ["rate other"], name="second")
    backend = OrderedBackend()
    service = await ingested(backend, [first, second], top_k=7)

    await service.search(QUERY)

    assert first.find_calls == [(QUERY, 7)]
    assert second.find_calls == [(QUERY, 7)]
    assert len(backend.queries) == 1
    assert backend.queries[0].text == QUERY
    assert backend.queries[0].limit == 7


async def test_ret_02_hit_chunks_seed_the_expansion_by_episode_id(
    tmp_path: Path,
) -> None:
    service, backend, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 2), ("a.md", 0)]

    await service.search(QUERY)

    expected = (
        episode_id_for(NS, "a.md#2", F3),
        episode_id_for(NS, "a.md#0", F1),
    )
    assert backend.queries[-1].seed_episode_ids == expected


async def test_ret_02_a_chunk_shared_by_two_tools_seeds_once(tmp_path: Path) -> None:
    first = one_doc(tmp_path, [F1], name="first")
    second = one_doc(tmp_path, [F1], name="second")
    backend = OrderedBackend()
    service = await ingested(backend, [first, second])
    first.ranked = [("a.md", 0)]
    second.ranked = [("a.md", 0)]

    await service.search(QUERY)

    assert backend.queries[-1].seed_episode_ids == (episode_id_for(NS, "a.md#0", F1),)


async def test_ret_02_no_chunk_hits_means_no_seeds(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)

    outcome = await service.search(QUERY)

    assert backend.queries[-1].seed_episode_ids == ()
    assert outcome.hits
    assert set(kinds(outcome.hits)) == {"fact"}


async def test_ret_02_a_backend_without_expansion_is_searched_without_seeds(
    tmp_path: Path,
) -> None:
    source = one_doc(tmp_path, [F1, F2])
    backend = OrderedBackend(supported=ALL_OPERATIONS - {MemoryOperation.EXPAND})
    service = await ingested(backend, [source])
    source.ranked = [("a.md", 0)]

    outcome = await service.search(QUERY)

    assert backend.queries[-1].seed_episode_ids == ()
    assert "chunk" in kinds(outcome.hits)
    assert "fact" in kinds(outcome.hits)
    assert outcome.degraded is False


async def test_ret_02_a_failing_source_tool_raises_and_is_not_degraded(
    tmp_path: Path,
) -> None:
    good = one_doc(tmp_path, [F1], name="good")
    bad = one_doc(tmp_path, ["rate bad"], name="bad")
    backend = OrderedBackend()
    service = await ingested(backend, [good, bad])
    bad.find_error = RuntimeError("index offline")

    with pytest.raises(RuntimeError, match="index offline"):
        await service.search(QUERY)

    assert backend.queries == []


async def test_ret_02_hops_default_to_the_configured_depth_and_can_be_set(
    tmp_path: Path,
) -> None:
    service, backend, _ = await worked_example(tmp_path, hops=2)

    await service.search(QUERY)
    await service.search(QUERY, hops=1)

    assert [query.hops for query in backend.queries] == [2, 1]
    with pytest.raises(MemoryValueError, match="hops"):
        await service.search(QUERY, hops=3)


# --- RET-05: fusion ---------------------------------------------------------


async def test_ret_05_worked_values_with_equal_weights(tmp_path: Path) -> None:
    service, backend, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]  # chunk rank 1 is the source of the rank-2 fact
    ids = await facts_by_text(backend)

    outcome = await service.search(QUERY)

    by_key = {(hit.kind, hit.text): hit for hit in outcome.hits}
    rank1_fact = by_key[("fact", F1)]
    rank2_fact = by_key[("fact", F2)]
    rank1_chunk = by_key[("chunk", F2)]
    assert rank1_fact.score == pytest.approx(1 / 61, abs=1e-6)
    assert rank1_fact.score == pytest.approx(0.016393, abs=1e-6)
    assert rank1_chunk.score == pytest.approx(1 / 61, abs=1e-6)
    assert rank2_fact.score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert rank2_fact.score == pytest.approx(0.032522, abs=1e-6)
    assert rank2_fact.fact_id == ids[F2]
    # Agreement of both retrievers lifts the rank-2 fact above the rank-1 fact.
    order = [(hit.kind, hit.text) for hit in outcome.hits]
    assert order[0] == ("fact", F2)
    assert order.index(("fact", F2)) < order.index(("fact", F1))


async def test_ret_05_a_tie_puts_facts_before_chunks(tmp_path: Path) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]

    outcome = await service.search(QUERY)

    order = [(hit.kind, hit.text) for hit in outcome.hits]
    # The rank-1 fact and the rank-1 chunk both score 1/61.
    assert order.index(("fact", F1)) + 1 == order.index(("chunk", F2))


async def test_ret_05_zero_chunk_weight_orders_facts_by_graph_rank(
    tmp_path: Path,
) -> None:
    service, backend, source = await worked_example(tmp_path, weight_chunks=0.0)
    source.ranked = [("a.md", 2), ("a.md", 1), ("a.md", 0)]
    backend.order = [F2, F3, F1]

    outcome = await service.search(QUERY)

    facts = [hit for hit in outcome.hits if hit.kind == "fact"]
    chunks = [hit for hit in outcome.hits if hit.kind == "chunk"]
    assert [hit.text for hit in facts] == [F2, F3, F1]
    assert [hit.score for hit in facts] == pytest.approx(
        [1 / 61, 1 / 62, 1 / 63], abs=1e-6
    )
    assert len(chunks) == 3
    assert all(hit.score == 0.0 for hit in chunks)
    assert kinds(outcome.hits) == ["fact"] * 3 + ["chunk"] * 3


async def test_ret_05_zero_graph_weight_leaves_only_the_chunk_term(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path, weight_graph=0.0)
    source.ranked = [("a.md", 2)]

    outcome = await service.search(QUERY)

    fact = next(hit for hit in outcome.hits if hit.kind == "fact" and hit.text == F3)
    assert fact.score == pytest.approx(1 / 61, abs=1e-6)
    other = next(hit for hit in outcome.hits if hit.kind == "fact" and hit.text == F1)
    assert other.score == 0.0


async def test_ret_05_the_weights_scale_each_term(tmp_path: Path) -> None:
    service, _, source = await worked_example(
        tmp_path, weight_graph=2.0, weight_chunks=0.5
    )
    source.ranked = [("a.md", 1), ("a.md", 0)]

    outcome = await service.search(QUERY)

    by_key = {(hit.kind, hit.text): hit.score for hit in outcome.hits}
    assert by_key[("fact", F1)] == pytest.approx(2 / 61 + 0.5 / 62, abs=1e-9)
    assert by_key[("fact", F2)] == pytest.approx(2 / 62 + 0.5 / 61, abs=1e-9)
    assert by_key[("fact", F3)] == pytest.approx(2 / 63, abs=1e-9)
    assert by_key[("chunk", F2)] == pytest.approx(0.5 / 61, abs=1e-9)
    assert by_key[("chunk", F1)] == pytest.approx(0.5 / 62, abs=1e-9)


async def test_ret_05_a_fact_takes_the_best_rank_over_tools(tmp_path: Path) -> None:
    first = one_doc(tmp_path, [F1, F2], name="first")
    second = one_doc(tmp_path, [F1, F2], name="second")
    backend = OrderedBackend()
    backend.order = [F2, F1]
    service = await ingested(backend, [first, second])
    # F1 is rank 2 in the first tool and rank 1 in the second.
    first.ranked = [("a.md", 1), ("a.md", 0)]
    second.ranked = [("a.md", 0)]

    outcome = await service.search(QUERY)

    fact = next(hit for hit in outcome.hits if hit.kind == "fact" and hit.text == F1)
    assert fact.score == pytest.approx(1 / 62 + 1 / 61, abs=1e-9)
    # A chunk scores by the rank in its own tool's list.
    scores = {
        (hit.source_tool, hit.text): hit.score
        for hit in outcome.hits
        if hit.kind == "chunk"
    }
    assert scores[("first", F2)] == pytest.approx(1 / 61, abs=1e-9)
    assert scores[("first", F1)] == pytest.approx(1 / 62, abs=1e-9)
    assert scores[("second", F1)] == pytest.approx(1 / 61, abs=1e-9)


async def test_ret_05_a_fact_with_two_hit_sources_takes_the_best_rank(
    tmp_path: Path,
) -> None:
    # Two chunks state the same fact, so the fact has two source episodes.
    source = one_doc(tmp_path, ["rate x", "rate y"])
    backend = FakeBackend(extract=lambda episode: [("base", "rate shared fact")])
    service = await ingested(backend, [source])
    source.ranked = [("a.md", 1), ("a.md", 0)]

    outcome = await service.search("shared")

    [fact] = [hit for hit in outcome.hits if hit.kind == "fact"]
    assert fact.text == "rate shared fact"
    # Graph rank 1, and the best source chunk is at chunk rank 1.
    assert fact.score == pytest.approx(1 / 61 + 1 / 61, abs=1e-9)
    # The hit names the source chunk that ranked best.
    assert fact.chunk_id == "a.md_chunk_1"


async def test_ret_05_chunk_ties_across_tools_break_by_tool_then_chunk_id(
    tmp_path: Path,
) -> None:
    alpha = one_doc(tmp_path, ["rate a1", "rate a2"], name="alpha")
    beta = one_doc(tmp_path, ["rate b1", "rate b2"], name="beta")
    backend = OrderedBackend()
    service = await ingested(backend, [beta, alpha], weight_graph=0.0)
    alpha.ranked = [("a.md", 1), ("a.md", 0)]
    beta.ranked = [("a.md", 0), ("a.md", 1)]

    outcome = await service.search(QUERY)

    chunks = [hit for hit in outcome.hits if hit.kind == "chunk"]
    assert [(hit.source_tool, hit.text) for hit in chunks] == [
        ("alpha", "rate a2"),
        ("beta", "rate b1"),
        ("alpha", "rate a1"),
        ("beta", "rate b2"),
    ]


async def test_ret_05_top_k_keeps_the_best_hits(tmp_path: Path) -> None:
    service, _, source = await worked_example(tmp_path, top_k=2)
    source.ranked = [("a.md", 1)]

    outcome = await service.search(QUERY)

    assert [(hit.kind, hit.text) for hit in outcome.hits] == [
        ("fact", F2),
        ("fact", F1),
    ]


async def test_ret_05_weights_are_validated(tmp_path: Path) -> None:
    backend = OrderedBackend()
    with pytest.raises(MemoryValueError, match="both be 0"):
        memory(backend, [], weight_chunks=0.0, weight_graph=0.0)
    with pytest.raises(MemoryValueError, match="weight_chunks"):
        memory(backend, [], weight_chunks=-1.0)
    with pytest.raises(MemoryValueError, match="weight_graph"):
        memory(backend, [], weight_graph=math.inf)
    with pytest.raises(MemoryValueError, match="top_k"):
        memory(backend, [], top_k=0)
    with pytest.raises(MemoryValueError, match="hops"):
        memory(backend, [], hops=0)


# --- RET-05: property -------------------------------------------------------

PROPERTY_SEEDS = range(200)
WEIGHTS = (0.0, 0.5, 1.0, 2.0)


async def test_ret_05_property_fusion_is_stable_and_follows_the_formula(
    tmp_path: Path,
) -> None:
    """200 seeded corpora: source order never changes the result, and every
    score and the whole order equal an independent computation of the design
    formula. Chunks at equal ranks in different tools tie on purpose."""
    ties_seen = 0
    for seed in PROPERTY_SEEDS:
        rng = random.Random(seed)  # noqa: S311 - reproducible test input
        directory = tmp_path / f"case{seed}"
        directory.mkdir()
        texts = [f"{QUERY} seed{seed} c{i}" for i in range(rng.randint(2, 6))]
        names = ["t0", "t1", "t2"][: rng.randint(1, 3)]
        sources = [one_doc(directory, texts, name=name) for name in names]
        w_chunks = rng.choice(WEIGHTS)
        w_graph = rng.choice([w for w in WEIGHTS if (w, w_chunks) != (0.0, 0.0)])
        top_k = rng.randint(1, 12)
        order = texts[:]
        rng.shuffle(order)
        backend = OrderedBackend()
        backend.order = order
        await ingested(backend, sources)
        ids = await facts_by_text(backend)
        for source in sources:
            picked = rng.sample(range(len(texts)), rng.randint(0, len(texts)))
            source.ranked = [("a.md", index) for index in picked]

        outcomes = []
        for _ in range(3):
            shuffled = sources[:]
            rng.shuffle(shuffled)
            service = memory(
                backend,
                shuffled,
                top_k=top_k,
                weight_chunks=w_chunks,
                weight_graph=w_graph,
            )
            outcomes.append(await service.search(QUERY))
        reference = [
            (hit.kind, hit.fact_id or hit.source_tool + hit.chunk_id, hit.score)
            for hit in outcomes[0].hits
        ]
        assert outcomes[0].hits, f"seed {seed}"
        for other in outcomes[1:]:
            assert [
                (hit.kind, hit.fact_id or hit.source_tool + hit.chunk_id, hit.score)
                for hit in other.hits
            ] == reference, f"seed {seed}: order depends on the source order"

        # An independent computation of the design formula.
        expected: list[tuple[float, int, str, str, str]] = []
        best: dict[str, int] = {}
        for source in sources:
            for rank, (_, index) in enumerate(source.ranked[:top_k], start=1):
                expected.append(
                    (-w_chunks / (60 + rank), 1, source.name, f"a.md_chunk_{index}", "")
                )
                best[texts[index]] = min(rank, best.get(texts[index], rank))
        for rank, text in enumerate(order[:top_k], start=1):
            score = w_graph / (60 + rank)
            if text in best:
                score += w_chunks / (60 + best[text])
            expected.append((-score, 0, "", ids[text], text))
        expected.sort(key=lambda item: item[:4])
        ties_seen += len({item[0] for item in expected}) < len(expected)
        expected = expected[:top_k]
        assert len(outcomes[0].hits) == len(expected) >= 1
        for hit, want in zip(outcomes[0].hits, expected, strict=True):
            assert hit.score == pytest.approx(-want[0], abs=1e-9), f"seed {seed}"
            assert hit.kind == ("fact" if want[1] == 0 else "chunk")
            if hit.kind == "fact":
                assert hit.fact_id == want[3], f"seed {seed}"
            else:
                assert (hit.source_tool, hit.chunk_id) == want[2:4], f"seed {seed}"
    # The corpora must exercise equal scores, or the property proves little.
    assert ties_seen > 20


# --- RET-06: provenance fields, valid period, chunk dating ------------------


async def test_ret_06_every_hit_carries_source_and_valid_period(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]

    outcome = await service.search(QUERY)

    assert {"fact", "chunk"} <= set(kinds(outcome.hits))
    for hit in outcome.hits:
        assert hit.source_tool == "rules"
        assert hit.source_uri == "a.md"
        assert hit.chunk_id.startswith("a.md_chunk_")
        assert hit.valid_from == DOC_DATE
        assert hit.valid_to is None
    fact = next(hit for hit in outcome.hits if hit.kind == "fact" and hit.text == F2)
    assert fact.chunk_id == "a.md_chunk_1"
    assert fact.fact_id is not None
    chunk = next(hit for hit in outcome.hits if hit.kind == "chunk")
    assert chunk.fact_id is None
    assert chunk.chunk_id == "a.md_chunk_1"


async def test_ret_06_a_fact_carries_its_own_valid_period(tmp_path: Path) -> None:
    older = RankedSource(
        "rules",
        tmp_path,
        [
            DocSpec("old.md", "2020-07-01", ("rate is 100",)),
            DocSpec("new.md", "2022-07-01", ("rate is 120",)),
        ],
    )
    backend = FakeBackend(
        extract=lambda episode: [("base rate", episode.text)],
    )
    service = await ingested(backend, [older])

    outcome = await service.search("rate", valid_at=datetime(2021, 1, 1, tzinfo=UTC))
    later = await service.search("rate", valid_at=datetime(2023, 1, 1, tzinfo=UTC))

    [fact] = [hit for hit in outcome.hits if hit.kind == "fact"]
    assert fact.text == "rate is 100"
    assert fact.valid_from == datetime(2020, 7, 1, tzinfo=UTC)
    assert fact.valid_to == datetime(2022, 7, 1, tzinfo=UTC)
    assert fact.superseded_by is not None
    [current] = [hit for hit in later.hits if hit.kind == "fact"]
    assert current.text == "rate is 120"
    assert current.valid_to is None
    assert current.source_uri == "new.md"


async def test_ret_06_a_chunk_from_a_later_document_is_dropped_and_not_a_seed(
    tmp_path: Path,
) -> None:
    source = RankedSource(
        "rules",
        tmp_path,
        [
            DocSpec("old.md", "2020-07-01", ("rate old",)),
            DocSpec("new.md", "2022-07-01", ("rate new",)),
        ],
    )
    backend = OrderedBackend()
    service = await ingested(backend, [source])
    source.ranked = [("new.md", 0), ("old.md", 0)]

    before = await service.search(QUERY, valid_at=datetime(2021, 1, 1, tzinfo=UTC))
    after = await service.search(QUERY, valid_at=datetime(2023, 1, 1, tzinfo=UTC))

    chunks_before = [hit for hit in before.hits if hit.kind == "chunk"]
    assert [hit.source_uri for hit in chunks_before] == ["old.md"]
    assert chunks_before[0].valid_from == DOC_DATE
    assert chunks_before[0].valid_to is None
    # Ranks count what is left: the old chunk is now rank 1.
    assert chunks_before[0].score == pytest.approx(1 / 61, abs=1e-9)
    assert backend.queries[0].seed_episode_ids == (
        episode_id_for(NS, "old.md#0", "rate old"),
    )
    assert [hit.source_uri for hit in after.hits if hit.kind == "chunk"] == [
        "new.md",
        "old.md",
    ]
    assert len(backend.queries[1].seed_episode_ids) == 2


async def test_ret_06_a_chunk_dated_exactly_at_valid_at_is_kept(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]

    outcome = await service.search(QUERY, valid_at=DOC_DATE)

    assert [hit.kind for hit in outcome.hits].count("chunk") == 1


async def test_ret_06_a_chunk_with_no_known_date_is_kept_without_a_start(
    tmp_path: Path,
) -> None:
    source = one_doc(tmp_path, ["rate undated"], date=None)
    backend = OrderedBackend()
    service = await ingested(backend, [source])
    source.ranked = [("a.md", 0)]

    outcome = await service.search(QUERY, valid_at=datetime(2001, 1, 1, tzinfo=UTC))

    [chunk] = [hit for hit in outcome.hits if hit.kind == "chunk"]
    assert chunk.valid_from is None
    assert chunk.valid_to is None


async def test_ret_06_a_search_reads_each_document_once_and_never_lists(
    tmp_path: Path,
) -> None:
    source = RankedSource(
        "rules",
        tmp_path,
        [
            DocSpec("a.md", "2020-07-01", (F1, F2)),
            DocSpec("b.md", "2020-08-01", ("rate b",)),
        ],
    )
    service = await ingested(OrderedBackend(), [source])
    source.ranked = [("a.md", 0), ("b.md", 0), ("a.md", 1)]
    source.listing_forbidden = True
    source.document_calls.clear()

    await service.search(QUERY)

    assert sorted(source.document_calls) == ["a.md", "b.md"]


async def test_ret_06_a_date_corrected_on_disk_is_seen_by_the_next_search(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]
    path = source._documents[0].path  # noqa: SLF001 - the file the source reads
    valid_at = datetime(2025, 1, 1, tzinfo=UTC)

    path.write_text("---\neffective_date: 2030-01-01\n---\nbody\n", encoding="utf-8")
    first = await service.search(QUERY, valid_at=valid_at)
    path.write_text("---\neffective_date: 2020-01-01\n---\nbody\n", encoding="utf-8")
    second = await service.search(QUERY, valid_at=valid_at)

    assert [hit.kind for hit in first.hits].count("chunk") == 0
    [chunk] = [hit for hit in second.hits if hit.kind == "chunk"]
    assert chunk.valid_from == datetime(2020, 1, 1, tzinfo=UTC)


async def test_ret_06_a_bad_front_matter_date_gives_an_undated_chunk(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]
    path = source._documents[0].path  # noqa: SLF001 - the file the source reads
    path.write_text("---\neffective_date: not-a-date\n---\nbody\n", encoding="utf-8")

    outcome = await service.search(QUERY, valid_at=datetime(2001, 1, 1, tzinfo=UTC))

    [chunk] = [hit for hit in outcome.hits if hit.kind == "chunk"]
    assert chunk.valid_from is None


async def test_ret_06_chunks_of_a_deleted_document_are_kept_without_a_start(
    tmp_path: Path,
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]
    source._documents.clear()  # noqa: SLF001 - the store lost the document

    outcome = await service.search(QUERY)

    [chunk] = [hit for hit in outcome.hits if hit.kind == "chunk"]
    assert chunk.valid_from is None


# --- RET-07: degraded -------------------------------------------------------


async def test_ret_07_an_unavailable_backend_gives_chunk_hits_only(
    tmp_path: Path,
) -> None:
    service, backend, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 2), ("a.md", 0)]
    backend.available = False

    outcome = await service.search(QUERY)

    assert outcome.degraded is True
    assert [hit.text for hit in outcome.hits] == [F3, F1]
    assert set(kinds(outcome.hits)) == {"chunk"}
    assert all(hit.degraded for hit in outcome.hits)
    assert [hit.score for hit in outcome.hits] == pytest.approx(
        [1 / 61, 1 / 62], abs=1e-9
    )
    for hit in outcome.hits:
        assert (hit.source_tool, hit.source_uri) == ("rules", "a.md")
        assert hit.valid_from == DOC_DATE
    # The weights are not applied when only one retriever answers.
    zero = await memory(backend, [source], weight_chunks=0.0).search(QUERY)
    assert [hit.score for hit in zero.hits] == pytest.approx([1 / 61, 1 / 62], abs=1e-9)


async def test_ret_07_a_degraded_search_can_have_no_hits(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)
    backend.available = False

    outcome = await service.search(QUERY)

    assert outcome.degraded is True
    assert outcome.hits == ()
    result = MemorySearchResult.build(
        outcome.hits, outcome.valid_at, outcome.known_at, degraded=outcome.degraded
    )
    assert result.degraded is True
    assert result.hits == []


async def test_ret_07_a_healthy_search_is_not_degraded(tmp_path: Path) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]

    outcome = await service.search(QUERY)

    assert outcome.degraded is False
    assert outcome.hits
    assert not any(hit.degraded for hit in outcome.hits)
    result = MemorySearchResult.build(
        outcome.hits, outcome.valid_at, outcome.known_at, degraded=outcome.degraded
    )
    assert result.degraded is False
    assert len(result.hits) == len(outcome.hits)


async def test_ret_07_only_an_unavailable_backend_degrades(tmp_path: Path) -> None:
    service, backend, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 0)]

    backend.search_error = MemoryNotSupportedError("fake", "search")
    with pytest.raises(MemoryNotSupportedError):
        await service.search(QUERY)

    backend.search_error = RuntimeError("driver bug")
    with pytest.raises(RuntimeError, match="driver bug"):
        await service.search(QUERY)

    backend.search_error = MemoryBackendUnavailableError("fake", "down")
    assert (await service.search(QUERY)).degraded is True


# --- timeline and provenance ------------------------------------------------


async def test_timeline_is_the_backends_timeline_in_the_namespace(
    tmp_path: Path,
) -> None:
    source = RankedSource(
        "rules",
        tmp_path,
        [
            DocSpec("new.md", "2022-07-01", ("rate is 120",)),
            DocSpec("old.md", "2020-07-01", ("rate is 100",)),
        ],
    )
    backend = FakeBackend(extract=lambda episode: [("base rate", episode.text)])
    service = await ingested(backend, [source])

    entries = await service.timeline("base rate")

    assert [entry.text for entry in entries] == ["rate is 100", "rate is 120"]
    assert entries[0].valid_to == datetime(2022, 7, 1, tzinfo=UTC)
    assert await service.timeline("unknown entity") == []
    other = TemporalMemory(namespace="other", backend=backend, sources=[source])
    assert await other.timeline("base rate") == []


async def test_provenance_returns_the_fact_and_its_source_text(tmp_path: Path) -> None:
    service, backend, _ = await worked_example(tmp_path)
    ids = await facts_by_text(backend)

    provenance = await service.provenance(ids[F2])

    assert provenance is not None
    assert provenance.fact_id == ids[F2]
    assert [source.text for source in provenance.sources] == [F2]
    assert provenance.sources[0].chunk_id == "a.md_chunk_1"
    assert await service.provenance("no-such-fact") is None
    other = TemporalMemory(namespace="other", backend=backend, sources=[])
    assert await other.provenance(ids[F2]) is None


async def test_timeline_and_provenance_name_an_unsupported_operation(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        supported=ALL_OPERATIONS
        - {MemoryOperation.TIMELINE, MemoryOperation.PROVENANCE}
    )
    service = memory(backend, [])

    with pytest.raises(MemoryNotSupportedError) as timeline_error:
        await service.timeline("x")
    with pytest.raises(MemoryNotSupportedError) as provenance_error:
        await service.provenance("x")

    assert timeline_error.value.operation == MemoryOperation.TIMELINE.value
    assert provenance_error.value.operation == MemoryOperation.PROVENANCE.value


# --- spans ------------------------------------------------------------------


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        service_module, "get_tracer", lambda name: provider.get_tracer(name)
    )
    yield exporter
    provider.shutdown()


def search_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == SEARCH_SPAN]


async def test_one_span_per_search_with_counts(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]
    exporter.clear()

    outcome = await service.search(QUERY)

    [span] = search_spans(exporter)
    attributes = dict(span.attributes or {})
    assert attributes["holodeck.memory.namespace"] == NS
    assert attributes["holodeck.memory.hits"] == len(outcome.hits) > 0
    assert attributes["holodeck.memory.fact_hits"] == 3
    assert attributes["holodeck.memory.chunk_hits"] == 1
    assert attributes["holodeck.memory.degraded"] is False
    assert span.status.status_code is not StatusCode.ERROR


async def test_a_degraded_search_is_marked_on_its_span(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    service, backend, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]
    backend.available = False
    exporter.clear()

    await service.search(QUERY)

    [span] = search_spans(exporter)
    assert span.attributes is not None
    assert span.attributes["holodeck.memory.degraded"] is True
    assert span.attributes["holodeck.memory.fact_hits"] == 0
    assert span.attributes["holodeck.memory.chunk_hits"] == 1


async def test_a_failed_search_span_names_the_error_type_only(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    source = one_doc(tmp_path, [F1])
    backend = OrderedBackend()
    service = await ingested(backend, [source])
    source.find_error = RuntimeError("SECRET-TEXT-IN-AN-ERROR")
    exporter.clear()

    with pytest.raises(RuntimeError):
        await service.search("QUERY-TEXT-MARKER")

    [span] = search_spans(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "RuntimeError"
    dumped = repr(span.attributes) + repr(span.status) + repr(span.events)
    assert "SECRET-TEXT-IN-AN-ERROR" not in dumped
    assert "QUERY-TEXT-MARKER" not in dumped


async def test_a_search_span_never_carries_query_or_chunk_text(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    service, _, source = await worked_example(tmp_path)
    source.ranked = [("a.md", 1)]
    exporter.clear()

    await service.search(QUERY + " QUERY-TEXT-MARKER")

    [span] = search_spans(exporter)
    dumped = repr(span.attributes) + repr(span.events) + repr(span.status)
    assert "QUERY-TEXT-MARKER" not in dumped
    assert F2 not in dumped
