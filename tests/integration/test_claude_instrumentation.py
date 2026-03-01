"""Integration tests for Claude GenAI instrumentation with real OTel providers.

Validates that the ``otel-instrumentation-claude-agent-sdk`` package
integrates correctly with HoloDeck's observability infrastructure.
Tests use ``InMemorySpanExporter`` to capture and assert spans without
requiring an external collector.

The module-level ``importorskip`` ensures these tests are silently skipped
when the optional ``claude-otel`` extras group is not installed.
"""

import pytest

otel_claude = pytest.importorskip(
    "opentelemetry.instrumentation.claude_agent_sdk",
    reason="otel-instrumentation-claude-agent-sdk not installed",
)

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    """Create a fresh in-memory span exporter for capturing test spans."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    """Create a TracerProvider wired to the in-memory exporter."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.mark.integration
class TestClaudeInstrumentationIntegration:
    """Integration tests for Claude Agent SDK OTel instrumentation.

    Tests validate that the instrumentation package produces spans
    conforming to GenAI semantic conventions when invoked through
    HoloDeck's ClaudeBackend.
    """

    # T023 tests will be added here
