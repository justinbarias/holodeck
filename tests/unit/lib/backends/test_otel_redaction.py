"""Trace attributes are scrubbed by RedactingSpanProcessor (spec 034 P2b)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import holodeck.lib.backends.otel_redaction as _otel_mod
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
        span.set_attribute(
            "tool.input.headers",
            "Authorization: Bearer abc.def-1_longtoken_xxxx",
        )
    span_data = exporter.get_finished_spans()[0]
    assert "Bearer [REDACTED]" in span_data.attributes["tool.input.headers"]


@pytest.mark.unit
def test_redacting_processor_scrubs_litellm_input_messages():
    """LiteLLM emits content under gen_ai.input.messages (GenAI semconv)."""
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    payload = '[{"role": "user", "content": "key is ghp_' + "x" * 36 + '"}]'
    with tracer.start_as_current_span("litellm_request") as span:
        span.set_attribute("gen_ai.input.messages", payload)
    span_data = exporter.get_finished_spans()[0]
    attr = span_data.attributes["gen_ai.input.messages"]
    assert "ghp_" not in attr
    assert "[REDACTED:github-token]" in attr


@pytest.mark.unit
def test_redacting_processor_scrubs_litellm_output_messages():
    """LiteLLM emits completions under gen_ai.output.messages."""
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    payload = '[{"role": "assistant", "content": "token AKIA' + "A" * 16 + '"}]'
    with tracer.start_as_current_span("litellm_request") as span:
        span.set_attribute("gen_ai.output.messages", payload)
    span_data = exporter.get_finished_spans()[0]
    attr = span_data.attributes["gen_ai.output.messages"]
    assert "AKIA" not in attr


@pytest.mark.unit
def test_redacting_processor_scrubs_litellm_system_instructions():
    """LiteLLM emits the system prompt under gen_ai.system_instructions."""
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("litellm_request") as span:
        span.set_attribute("gen_ai.system_instructions", "use ghp_" + "x" * 36)
    span_data = exporter.get_finished_spans()[0]
    assert "ghp_" not in span_data.attributes["gen_ai.system_instructions"]


@pytest.mark.unit
def test_redacting_processor_keeps_legacy_genai_prefixes():
    """Legacy gen_ai.prompt / gen_ai.completion remain covered (back-compat)."""
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("chat") as span:
        span.set_attribute("gen_ai.prompt.0.content", "ghp_" + "x" * 36)
        span.set_attribute("gen_ai.completion.0.content", "ghp_" + "y" * 36)
    span_data = exporter.get_finished_spans()[0]
    assert "ghp_" not in span_data.attributes["gen_ai.prompt.0.content"]
    assert "ghp_" not in span_data.attributes["gen_ai.completion.0.content"]


@pytest.mark.unit
def test_redacting_processor_logs_error_when_attributes_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit one-shot ERROR when span._attributes is absent (SDK shape drift)."""
    # Reset the module-level guard so this test is isolated.
    _otel_mod._warned_missing_attributes_attr = False

    processor = RedactingSpanProcessor()
    fake_span = MagicMock()
    # Make getattr(span, "_attributes", None) return None.
    del fake_span._attributes
    fake_span.__class__.__name__ = "Span"

    with patch.object(
        type(fake_span),
        "_attributes",
        new_callable=lambda: property(lambda self: None),
        create=True,
    ):
        pass  # use a simpler approach below

    # Simplest way: make getattr return None via spec
    span_no_attrs = MagicMock(spec=[])  # spec=[] means no attrs defined
    with caplog.at_level(logging.ERROR):
        processor.on_end(span_no_attrs)  # type: ignore[arg-type]
        processor.on_end(span_no_attrs)  # type: ignore[arg-type]

    error_records = [
        r for r in caplog.records if "OTel SDK may have changed" in r.message
    ]
    assert len(error_records) == 1, "Warning should fire exactly once (one-shot guard)"

    # Reset guard after test so other tests are unaffected.
    _otel_mod._warned_missing_attributes_attr = False


@pytest.mark.unit
def test_redacting_processor_continues_when_redact_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exception inside redact_credentials is caught; span still exports."""
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)

    with (
        patch(
            "holodeck.lib.backends.otel_redaction.redact_credentials",
            side_effect=RuntimeError("boom"),
        ),
        tracer.start_as_current_span("execute_tool") as span,
    ):
        span.set_attribute("tool.output", "ghp_" + "x" * 36)

    # Span should still export despite the redaction failure.
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    # A warning should have been logged.
    assert any("failed to redact" in r.message for r in caplog.records)


@pytest.mark.unit
def test_set_up_tracing_deduplicates_redacting_processor() -> None:
    """Calling set_up_tracing twice must not add a second RedactingSpanProcessor."""
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

    from holodeck.lib.observability.providers import (
        _get_span_processors,
        set_up_tracing,
    )
    from holodeck.models.observability import ObservabilityConfig

    config = ObservabilityConfig(enabled=True)
    resource_mock = MagicMock()

    provider = SdkTracerProvider()
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=provider),
        patch("opentelemetry.trace.set_tracer_provider"),
    ):
        set_up_tracing(config, resource_mock, [])
        set_up_tracing(config, resource_mock, [])

    processors = _get_span_processors(provider)
    redacting = [p for p in processors if isinstance(p, RedactingSpanProcessor)]
    assert len(redacting) == 1, "RedactingSpanProcessor must appear exactly once"
