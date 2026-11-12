"""Ingestion emits one trace per run and one span per episode (spec 043, ING-06)."""

from __future__ import annotations

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

from holodeck.lib.memory import IngestReport
from holodeck.lib.memory import service as service_module
from holodeck.lib.memory.backends.fake import FakeBackend
from holodeck.lib.memory.service import (
    EPISODE_SPAN,
    RUN_SPAN,
    TemporalMemory,
    episode_id_for,
)
from holodeck.lib.memory.types import EpisodeIn
from tests.unit.lib.memory.test_service_ingest import DocSpec, FakeSource

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NS = "family-assistance"
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
TEXT_MARK = "ZZ-CHUNK-TEXT-MARKER"
INSTR_MARK = "ZZ-INSTRUCTION-MARKER"


class TokenBackend(FakeBackend):
    """FakeBackend whose reports carry token counts."""

    def __init__(self, tokens: tuple[int | None, int | None]) -> None:
        super().__init__()
        self._tokens = tokens

    async def ingest(
        self, namespace: str, episodes: Sequence[EpisodeIn], instructions: str
    ) -> IngestReport:
        report = await super().ingest(namespace, episodes, instructions)
        return replace(
            report, input_tokens=self._tokens[0], output_tokens=self._tokens[1]
        )


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


def _source(tmp_path: Path) -> FakeSource:
    texts = tuple(f"{TEXT_MARK} {word}" for word in ("one", "two", "three"))
    return FakeSource("rules", tmp_path, [DocSpec("a.md", "2020-07-01", texts)])


def _memory(backend: FakeBackend, source: FakeSource) -> TemporalMemory:
    return TemporalMemory(
        namespace=NS,
        backend=backend,
        sources=[source],
        clock=lambda: NOW,
        instructions=INSTR_MARK,
    )


def _named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


async def test_ing_06_one_run_span_and_one_episode_span_per_new_episode(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    await _memory(FakeBackend(), _source(tmp_path)).ingest()

    runs = _named(exporter, RUN_SPAN)
    episodes = _named(exporter, EPISODE_SPAN)
    assert len(runs) == 1
    assert len(episodes) == 3
    run = runs[0]
    assert run.context is not None
    for span in episodes:
        assert span.context is not None and span.parent is not None
        assert span.context.trace_id == run.context.trace_id
        assert span.parent.span_id == run.context.span_id
    assert run.attributes is not None
    assert run.attributes["holodeck.memory.namespace"] == NS
    assert run.attributes["holodeck.memory.dry_run"] is False
    assert run.attributes["holodeck.memory.documents"] == 1
    assert run.attributes["holodeck.memory.episodes_new"] == 3
    assert run.attributes["holodeck.memory.episodes_skipped"] == 0
    assert run.attributes["holodeck.memory.episodes_written"] == 3
    assert run.attributes["holodeck.memory.llm_calls"] == 3
    assert run.attributes["holodeck.memory.failed"] is False
    assert run.status.status_code != StatusCode.ERROR


async def test_ing_06_episode_span_attributes(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    await _memory(FakeBackend(), _source(tmp_path)).ingest()

    expected_ids = {
        str(episode_id_for(NS, f"a.md#{i}", f"{TEXT_MARK} {w}"))
        for i, w in enumerate(["one", "two", "three"])
    }
    seen = set()
    for span in _named(exporter, EPISODE_SPAN):
        attrs = span.attributes
        assert attrs is not None
        seen.add(attrs["holodeck.memory.episode_id"])
        assert attrs["holodeck.memory.source_tool"] == "rules"
        assert str(attrs["holodeck.memory.chunk_id"]).startswith("a.md_chunk_")
        assert attrs["holodeck.memory.source_uri"] == "a.md"
        assert attrs["holodeck.memory.reference_time"] == "2020-07-01T00:00:00Z"
        assert attrs["holodeck.memory.reference_time_source"] == "front_matter"
        assert attrs["holodeck.memory.llm_calls"] == 1
    assert seen == expected_ids


async def test_ing_06_second_run_has_no_episode_spans(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    memory = _memory(FakeBackend(), _source(tmp_path))
    await memory.ingest()
    exporter.clear()

    await memory.ingest()

    runs = _named(exporter, RUN_SPAN)
    assert len(runs) == 1
    assert len(_named(exporter, EPISODE_SPAN)) == 0
    assert runs[0].attributes is not None
    assert runs[0].attributes["holodeck.memory.episodes_skipped"] == 3
    assert runs[0].attributes["holodeck.memory.episodes_new"] == 0


async def test_ing_06_dry_run_has_no_episode_spans(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    await _memory(FakeBackend(), _source(tmp_path)).ingest(dry_run=True)

    runs = _named(exporter, RUN_SPAN)
    assert len(runs) == 1
    assert len(_named(exporter, EPISODE_SPAN)) == 0
    assert runs[0].attributes is not None
    assert runs[0].attributes["holodeck.memory.dry_run"] is True
    assert runs[0].attributes["holodeck.memory.episodes_new"] == 3


async def test_ing_06_failed_episode_marks_its_span_and_the_run(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    backend = FakeBackend()
    backend.fail_on.add(episode_id_for(NS, "a.md#1", f"{TEXT_MARK} two"))

    report = await _memory(backend, _source(tmp_path)).ingest()

    assert report.failed is not None
    episodes = sorted(_named(exporter, EPISODE_SPAN), key=lambda s: s.start_time or 0)
    assert len(episodes) == 2
    assert episodes[0].status.status_code != StatusCode.ERROR
    assert episodes[1].status.status_code == StatusCode.ERROR
    # The status says that it failed, never why: error text can quote a
    # document or a credential. The reason is in the run report.
    assert episodes[1].status.description == "episode failed"
    assert report.error == "extraction failed (set by the test)"
    assert all(
        "set by the test" not in (span.status.description or "")
        for span in exporter.get_finished_spans()
    )
    run = _named(exporter, RUN_SPAN)[0]
    assert run.attributes is not None
    assert run.attributes["holodeck.memory.failed"] is True
    assert run.attributes["holodeck.memory.episodes_written"] == 1
    assert run.status.status_code == StatusCode.ERROR


async def test_ing_06_token_attributes_match_the_backend_report(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    await _memory(TokenBackend((100, 20)), _source(tmp_path)).ingest()

    episodes = _named(exporter, EPISODE_SPAN)
    assert len(episodes) == 3
    for span in episodes:
        assert span.attributes is not None
        assert span.attributes["gen_ai.usage.input_tokens"] == 100
        assert span.attributes["gen_ai.usage.output_tokens"] == 20
    run = _named(exporter, RUN_SPAN)[0]
    assert run.attributes is not None
    assert run.attributes["gen_ai.usage.input_tokens"] == 300
    assert run.attributes["gen_ai.usage.output_tokens"] == 60


async def test_ing_06_token_attributes_absent_when_backend_gives_none(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    await _memory(TokenBackend((None, None)), _source(tmp_path)).ingest()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.attributes is not None
        assert "gen_ai.usage.input_tokens" not in span.attributes
        assert "gen_ai.usage.output_tokens" not in span.attributes


async def test_ing_06_no_span_carries_chunk_text_or_instructions(
    exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    backend = FakeBackend()
    backend.fail_on.add(episode_id_for(NS, "a.md#2", f"{TEXT_MARK} three"))
    await _memory(backend, _source(tmp_path)).ingest()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        values = [str(v) for v in (span.attributes or {}).values()]
        values.append(span.status.description or "")
        values.extend(str(e.attributes) for e in span.events)
        for value in values:
            assert TEXT_MARK not in value
            assert INSTR_MARK not in value


async def test_ing_06_without_a_provider_the_report_is_unchanged(
    tmp_path: Path,
) -> None:
    report = await _memory(FakeBackend(), _source(tmp_path)).ingest()

    assert report.episodes_new == 3
    assert len(report.written) == 3
    assert report.failed is None
