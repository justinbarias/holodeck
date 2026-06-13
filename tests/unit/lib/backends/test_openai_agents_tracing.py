"""Unit tests for holodeck.lib.backends.openai_agents_tracing.

The OpenAI Agents SDK trace ``TracingProcessor`` mirror is exercised with fake
SDK span objects (no SDK run is driven). OTel spans are captured with an
in-memory ``TracerProvider`` so the mirror's attribute mapping — and the
existing ``RedactingSpanProcessor`` scrubbing of credential-shaped tool output —
are asserted end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from holodeck.lib.backends.openai_agents_tracing import build_tracing_mirror

# ---------------------------------------------------------------------------
# Fakes mirroring the SDK Span / SpanData shapes
# ---------------------------------------------------------------------------


def _fake_span(
    span_data: Any,
    *,
    trace_id: str = "trace_abc",
    span_id: str = "span_1",
    parent_id: str | None = None,
    started_at: str | None = "2026-06-13T01:00:00.000000+00:00",
    ended_at: str | None = "2026-06-13T01:00:01.000000+00:00",
    error: dict[str, Any] | None = None,
) -> Any:
    """Build a duck-typed stand-in for an SDK ``Span``."""
    return SimpleNamespace(
        span_data=span_data,
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
    )


class _FunctionSpanData:
    """Stand-in for ``agents.tracing.span_data.FunctionSpanData``."""

    def __init__(self, name: str, input: str | None, output: Any | None) -> None:
        self.name = name
        self.input = input
        self.output = output
        self.mcp_data: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "function"


class _GenerationSpanData:
    """Stand-in for ``agents.tracing.span_data.GenerationSpanData``."""

    def __init__(self, model: str | None, usage: dict[str, Any] | None) -> None:
        self.model = model
        self.usage = usage
        self.input = None
        self.output = None
        self.model_config: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "generation"


class _AgentSpanData:
    """Stand-in for ``agents.tracing.span_data.AgentSpanData``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handoffs: list[str] | None = None
        self.tools: list[str] | None = None
        self.output_type: str | None = None

    @property
    def type(self) -> str:
        return "agent"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemorySpanExporter:
    """Install an isolated TracerProvider with redaction + in-memory export.

    The mirror resolves its tracer via ``holodeck...get_tracer`` (which reads the
    global OTel provider). Point that provider at an in-memory exporter behind
    the real ``RedactingSpanProcessor`` so scrubbing is exercised on export.
    """
    from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # The mirror calls get_tracer(name) -> trace.get_tracer(name); patch the
    # module-global provider lookup used by that helper.
    monkeypatch.setattr(
        "holodeck.lib.backends.openai_agents_tracing.get_tracer",
        lambda name: provider.get_tracer(name),
    )
    return exporter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTracingMirror:
    """The mirror creates OTel spans from SDK span-end events."""

    def test_function_span_maps_to_otel_span(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _FunctionSpanData(
                name="search_docs", input='{"q": "x"}', output="found 3 rows"
            )
        )
        mirror.on_span_start(span)
        mirror.on_span_end(span)
        mirror.force_flush()

        spans = in_memory_tracing.get_finished_spans()
        assert len(spans) == 1
        otel = spans[0]
        assert otel.name == "openai_agents.function search_docs"
        attrs = otel.attributes or {}
        assert attrs["gen_ai.openai.span_type"] == "function"
        assert attrs["gen_ai.tool.name"] == "search_docs"
        assert attrs["tool.input"] == '{"q": "x"}'
        assert attrs["tool.output"] == "found 3 rows"

    def test_generation_span_records_model_and_usage(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _GenerationSpanData(
                model="gpt-4o-mini",
                usage={"input_tokens": 12, "output_tokens": 5},
            ),
            span_id="span_gen",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        attrs = otel.attributes or {}
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.usage.input_tokens"] == 12
        assert attrs["gen_ai.usage.output_tokens"] == 5

    def test_credential_tool_output_is_redacted(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """A credential-shaped tool.output is scrubbed before export (FR-088)."""
        mirror = build_tracing_mirror("test-agent")
        key = "sk-ant-api03-" + "A" * 95
        leaked = f"token {key} done"
        span = _fake_span(
            _FunctionSpanData(name="leaky", input=None, output=leaked),
            span_id="span_leak",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        output_attr = (otel.attributes or {})["tool.output"]
        assert key not in output_attr
        assert "REDACTED" in output_attr

    def test_hierarchy_ids_recorded_as_attributes(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """SDK trace/span/parent ids are attached as attributes (best-effort)."""
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _AgentSpanData(name="root"),
            trace_id="trace_xyz",
            span_id="span_root",
            parent_id="span_parent",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        attrs = in_memory_tracing.get_finished_spans()[0].attributes or {}
        assert attrs["gen_ai.openai.trace_id"] == "trace_xyz"
        assert attrs["gen_ai.openai.span_id"] == "span_root"
        assert attrs["gen_ai.openai.parent_id"] == "span_parent"

    def test_uses_sdk_timestamps(self, in_memory_tracing: InMemorySpanExporter) -> None:
        """Explicit SDK start/end ISO timestamps drive the OTel span window."""
        mirror = build_tracing_mirror("test-agent")
        span = _fake_span(
            _AgentSpanData(name="root"),
            started_at="2026-06-13T01:00:00.000000+00:00",
            ended_at="2026-06-13T01:00:02.500000+00:00",
        )
        mirror.on_span_end(span)
        mirror.force_flush()

        otel = in_memory_tracing.get_finished_spans()[0]
        # 2.5 seconds in nanoseconds.
        assert otel.end_time - otel.start_time == 2_500_000_000

    def test_on_span_end_never_raises_on_malformed_span(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """A malformed span must not break the agent run (errors swallowed)."""
        mirror = build_tracing_mirror("test-agent")
        broken = SimpleNamespace()  # missing every expected attribute
        # Must not raise.
        mirror.on_span_end(broken)
        mirror.force_flush()

    def test_trace_lifecycle_methods_are_noops(
        self, in_memory_tracing: InMemorySpanExporter
    ) -> None:
        """Trace start/end + shutdown must not emit spans or raise."""
        mirror = build_tracing_mirror("test-agent")
        fake_trace = SimpleNamespace(trace_id="t", name="wf")
        mirror.on_trace_start(fake_trace)
        mirror.on_trace_end(fake_trace)
        mirror.shutdown()
        assert in_memory_tracing.get_finished_spans() == ()
