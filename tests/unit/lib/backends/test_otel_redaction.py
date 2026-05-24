"""Trace attributes are scrubbed by RedactingSpanProcessor (spec 034 P2b)."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor


def _new_provider_with_redaction() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # RedactingSpanProcessor must run BEFORE the exporting processor in the
    # chain so the exporter sees redacted attributes.
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.mark.unit
def test_redacting_processor_scrubs_tool_output():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.output", "key=ghp_" + "x" * 36)
    span_data = exporter.get_finished_spans()[0]
    assert "[REDACTED:github-token]" in span_data.attributes["tool.output"]


@pytest.mark.unit
def test_redacting_processor_leaves_unrelated_attributes_alone():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.name", "Bash")
        span.set_attribute("rows.returned", 42)
    span_data = exporter.get_finished_spans()[0]
    assert span_data.attributes["tool.name"] == "Bash"
    assert span_data.attributes["rows.returned"] == 42


@pytest.mark.unit
def test_redacting_processor_handles_tool_input_namespace():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.input.headers", "Authorization: Bearer abc.def-1")
    span_data = exporter.get_finished_spans()[0]
    assert "Bearer [REDACTED]" in span_data.attributes["tool.input.headers"]
