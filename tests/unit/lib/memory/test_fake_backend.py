"""FakeBackend behaves the way every memory backend must (spec 043, T2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from holodeck.lib.memory import (
    EpisodeIn,
    MemoryBackendUnavailableError,
    MemoryNotSupportedError,
    MemoryOperation,
    MemoryQuery,
)
from holodeck.lib.memory.backends.fake import ALL_OPERATIONS, FakeBackend
from tests.unit.lib.memory.test_entity_match import CASES as ENTITY_CASES

pytestmark = pytest.mark.asyncio

NS = "family-assistance"
EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
LATER = EPOCH + timedelta(days=1)
Y2019 = datetime(2019, 7, 1, tzinfo=UTC)
Y2020 = datetime(2020, 7, 1, tzinfo=UTC)
Y2022 = datetime(2022, 7, 1, tzinfo=UTC)


def episode(name: str, text: str, when: datetime) -> EpisodeIn:
    return EpisodeIn(
        episode_id=uuid5(NAMESPACE_URL, name),
        source_tool="family_law",
        chunk_id=f"{name}_chunk_0",
        source_uri=f"corpus/{name}.md",
        text=text,
        reference_time=when,
        reference_time_source="front_matter",
    )


E0 = episode("c0", "rate is 90", Y2019)
E1 = episode("c1", "rate is 100", Y2020)
E2 = episode("c2", "rate is 120", Y2022)


def rate_backend(**kwargs: frozenset[MemoryOperation]) -> FakeBackend:
    return FakeBackend(extract=lambda ep: [("rate", ep.text)], **kwargs)


def query(valid_at: datetime, known_at: datetime = LATER) -> MemoryQuery:
    return MemoryQuery(text="rate", valid_at=valid_at, known_at=known_at)


async def texts(backend: FakeBackend, q: MemoryQuery, ns: str = NS) -> list[str]:
    return [hit.text for hit in await backend.search(ns, q)]


async def test_ingest_reports_what_it_wrote_and_counts_one_llm_call_each() -> None:
    backend = rate_backend()

    report = await backend.ingest(NS, [E1, E2], "extract amounts")

    assert report.written == (E1.episode_id, E2.episode_id)
    assert report.failed is None
    assert report.llm_calls == 2
    assert backend.instructions == ["extract amounts"]
    assert await backend.episode_count(NS) == 2
    assert await backend.has_episode(NS, E1.episode_id) is True
    assert await backend.has_episode(NS, E0.episode_id) is False


async def test_a_newer_fact_closes_the_older_one_at_its_valid_from() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1, E2], "")

    in_2021 = await backend.search(NS, query(datetime(2021, 1, 1, tzinfo=UTC)))
    in_2023 = await backend.search(NS, query(datetime(2023, 1, 1, tzinfo=UTC)))

    assert [hit.text for hit in in_2021] == ["rate is 100"]
    assert in_2021[0].valid_from == Y2020
    assert in_2021[0].valid_to == Y2022
    assert in_2021[0].superseded_by == in_2023[0].fact_id
    assert [hit.text for hit in in_2023] == ["rate is 120"]
    assert in_2023[0].valid_to is None


async def test_known_at_before_the_second_ingest_still_sees_the_old_fact_open() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1], "")
    between = EPOCH + timedelta(seconds=1)  # the counter clock's first tick
    await backend.ingest(NS, [E2], "")

    hits = await backend.search(
        NS, query(datetime(2023, 1, 1, tzinfo=UTC), known_at=between)
    )

    assert [hit.text for hit in hits] == ["rate is 100"]
    assert hits[0].valid_to is None


async def test_a_late_older_fact_is_closed_and_changes_nothing_else() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1, E2], "")
    await backend.ingest(NS, [E0], "")

    assert await texts(backend, query(datetime(2019, 12, 1, tzinfo=UTC))) == [
        "rate is 90"
    ]
    assert await texts(backend, query(datetime(2021, 1, 1, tzinfo=UTC))) == [
        "rate is 100"
    ]
    assert await texts(backend, query(datetime(2023, 1, 1, tzinfo=UTC))) == [
        "rate is 120"
    ]
    timeline = await backend.timeline(NS, "rate")
    assert [(e.text, e.valid_from, e.valid_to) for e in timeline] == [
        ("rate is 90", Y2019, Y2020),
        ("rate is 100", Y2020, Y2022),
        ("rate is 120", Y2022, None),
    ]


async def test_the_same_fact_from_a_second_episode_adds_a_source() -> None:
    backend = rate_backend()
    again = episode("c1-copy", "rate is 100", Y2020)
    await backend.ingest(NS, [E1, again], "")

    hits = await backend.search(NS, query(datetime(2021, 1, 1, tzinfo=UTC)))
    provenance = await backend.provenance(NS, hits[0].fact_id)

    assert len(hits) == 1
    assert provenance is not None
    assert [(s.source_tool, s.chunk_id) for s in provenance.sources] == [
        ("family_law", "c1_chunk_0"),
        ("family_law", "c1-copy_chunk_0"),
    ]
    assert all(source.text for source in provenance.sources)
    assert all(source.text is None for source in hits[0].sources)
    assert await backend.provenance(NS, "no-such-fact") is None


async def test_seed_episodes_return_their_facts_without_a_text_match() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E2], "")
    q = MemoryQuery(
        text="unrelated words",
        valid_at=datetime(2023, 1, 1, tzinfo=UTC),
        known_at=LATER,
        seed_episode_ids=(E2.episode_id,),
    )

    assert await texts(backend, q) == ["rate is 120"]
    assert (
        await texts(
            backend, MemoryQuery(text="unrelated", valid_at=q.valid_at, known_at=LATER)
        )
        == []
    )


async def test_ing_05_ingest_stops_at_the_failed_episode_and_reports_it() -> None:
    backend = rate_backend()
    backend.fail_on = {E1.episode_id}

    report = await backend.ingest(NS, [E0, E1, E2], "")

    assert report.written == (E0.episode_id,)
    assert report.failed == E1.episode_id
    assert report.error == "extraction failed (set by the test)"
    assert report.llm_calls == 1
    assert await backend.has_episode(NS, E1.episode_id) is False
    assert await backend.has_episode(NS, E2.episode_id) is False


async def test_mb_02_a_namespace_never_sees_another_namespaces_data() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1], "")

    assert await texts(backend, query(datetime(2021, 1, 1, tzinfo=UTC)), "other") == []
    assert await backend.has_episode("other", E1.episode_id) is False
    assert await backend.timeline("other", "rate") == []
    assert await backend.episode_count("other") == 0


async def test_mb_04_purge_removes_one_namespace_and_no_other() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1, E2], "")
    await backend.ingest("other", [E1], "")

    await backend.purge(NS)

    assert await backend.episode_count(NS) == 0
    assert await backend.has_episode(NS, E1.episode_id) is False
    assert await texts(backend, query(datetime(2021, 1, 1, tzinfo=UTC))) == []
    assert await backend.episode_count("other") == 1


async def test_mb_03_an_unsupported_operation_raises_and_returns_nothing() -> None:
    backend = rate_backend(supported=ALL_OPERATIONS - {MemoryOperation.TIMELINE})
    await backend.ingest(NS, [E1], "")

    with pytest.raises(MemoryNotSupportedError) as raised:
        await backend.timeline(NS, "rate")

    assert (
        str(raised.value) == "Memory backend 'fake' does not support entity timelines."
    )
    assert raised.value.operation == "entity timelines"


async def test_an_unavailable_store_raises_the_typed_error() -> None:
    backend = rate_backend()
    backend.available = False

    with pytest.raises(MemoryBackendUnavailableError, match="'fake' is unavailable"):
        await backend.search(NS, query(Y2020))
    with pytest.raises(MemoryBackendUnavailableError):
        await backend.has_episode(NS, UUID(int=0))


async def test_calls_are_counted_per_method() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1], "")
    await backend.has_episode(NS, E1.episode_id)
    await backend.has_episode(NS, E2.episode_id)

    assert backend.calls["ingest"] == 1
    assert backend.calls["has_episode"] == 2
    assert backend.llm_calls == 1


Y2024 = datetime(2024, 7, 1, tzinfo=UTC)


async def test_a_fact_that_returns_after_a_change_starts_a_new_period() -> None:
    backend = rate_backend()
    again = episode("c3", "rate is 100", Y2024)
    await backend.ingest(NS, [E1, E2, again], "")

    assert await texts(backend, query(datetime(2025, 1, 1, tzinfo=UTC))) == [
        "rate is 100"
    ]
    assert await texts(backend, query(datetime(2023, 1, 1, tzinfo=UTC))) == [
        "rate is 120"
    ]
    periods = [
        (e.text, e.valid_from, e.valid_to) for e in await backend.timeline(NS, "rate")
    ]
    assert periods == [
        ("rate is 100", Y2020, Y2022),
        ("rate is 120", Y2022, Y2024),
        ("rate is 100", Y2024, None),
    ]


async def test_ing_05_an_extraction_error_is_reported_not_raised() -> None:
    def extract(ep: EpisodeIn) -> list[tuple[str, str]]:
        if ep is E1:
            raise ValueError("bad episode")
        return [("rate", ep.text)]

    backend = FakeBackend(extract=extract)

    report = await backend.ingest(NS, [E0, E1, E2], "")

    assert report.written == (E0.episode_id,)
    assert report.failed == E1.episode_id
    assert report.error == "bad episode"
    assert await backend.has_episode(NS, E1.episode_id) is False


async def test_a_fact_seen_as_open_in_the_past_shows_no_later_expiry() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1], "")
    before_second = backend._clock()
    await backend.ingest(NS, [E2], "")

    hits = await backend.search(
        NS, query(datetime(2023, 1, 1, tzinfo=UTC), known_at=before_second)
    )

    assert [(h.text, h.valid_to, h.expired_at) for h in hits] == [
        ("rate is 100", None, None)
    ]


async def test_mb_03_seeds_need_the_expand_capability() -> None:
    backend = rate_backend(supported=ALL_OPERATIONS - {MemoryOperation.EXPAND})
    await backend.ingest(NS, [E2], "")
    seeded = MemoryQuery(
        text="rate",
        valid_at=datetime(2023, 1, 1, tzinfo=UTC),
        known_at=LATER,
        seed_episode_ids=(E2.episode_id,),
    )

    with pytest.raises(MemoryNotSupportedError, match="graph expansion from seed"):
        await backend.search(NS, seeded)
    assert await texts(backend, query(datetime(2023, 1, 1, tzinfo=UTC))) == [
        "rate is 120"
    ]


async def test_ret_08_timeline_uses_a_subject_named_inside_the_reference() -> None:
    backend = rate_backend()
    await backend.ingest(NS, [E1, E2], "")
    exact = await backend.timeline(NS, "rate")

    topic = await backend.timeline(NS, "the rate over time")
    unknown = await backend.timeline(NS, "the amount over time")

    assert len(exact) == 2
    assert topic == exact
    assert unknown == []


@pytest.mark.parametrize(("reference", "subjects", "expected"), ENTITY_CASES)
async def test_ret_08_timeline_resolves_subjects_like_every_backend(
    reference: str, subjects: tuple[str, ...], expected: list[str]
) -> None:
    # One fact per subject; the default extractor makes the text the subject.
    backend = FakeBackend()
    await backend.ingest(
        NS,
        [
            episode(f"subject-{index}", subject, Y2019 + timedelta(days=index))
            for index, subject in enumerate(subjects)
        ],
        "",
    )

    entries = await backend.timeline(NS, reference)

    assert [entry.text for entry in entries] == expected
