"""End-to-end: tracer provider has RedactingSpanProcessor registered first.

Spec 034 P2b — wiring verification.

The test calls set_up_tracing() directly (the HoloDeck factory that owns
TracerProvider construction) and asserts that a RedactingSpanProcessor is
present in the processor chain *before* any exporting processor.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider

from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor


def _unwrap_processors(provider: TracerProvider) -> list[object]:
    """Extract the list of SpanProcessor instances from a TracerProvider.

    The SDK (>=1.20) wraps all processors in a SynchronousMultiSpanProcessor
    or ConcurrentMultiSpanProcessor stored at ``_active_span_processor``.
    The actual list lives at ``_span_processors`` on that wrapper.
    """
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        return []
    inner = getattr(active, "_span_processors", None)
    if inner is not None:
        return list(inner)
    # Fallback: single processor attached directly
    return [active]


@pytest.mark.integration
def test_tracer_provider_has_redaction_processor() -> None:
    """RedactingSpanProcessor is wired into the provider returned by set_up_tracing."""
    from opentelemetry.sdk.resources import Resource

    from holodeck.lib.observability.providers import set_up_tracing
    from holodeck.models.observability import ObservabilityConfig

    resource = Resource.create({"service.name": "test-redaction"})
    config = ObservabilityConfig(enabled=True)

    # Call set_up_tracing with no exporters — the returned provider must still
    # carry a RedactingSpanProcessor regardless.
    provider = set_up_tracing(config, resource, span_exporters=[])

    processors = _unwrap_processors(provider)
    assert any(
        isinstance(p, RedactingSpanProcessor) for p in processors
    ), f"RedactingSpanProcessor not found in processor chain: {processors}"


@pytest.mark.integration
def test_redaction_processor_registered_before_exporters() -> None:
    """RedactingSpanProcessor appears at index 0, before any exporting processors.

    Uses a freshly-constructed TracerProvider (not the global) to avoid
    cross-test bleed from other set_up_tracing calls in the same session.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from holodeck.lib.observability.providers import set_up_tracing
    from holodeck.models.observability import ObservabilityConfig

    resource = Resource.create({"service.name": "test-redaction-order"})
    config = ObservabilityConfig(enabled=True)

    # Use a fresh provider injected via the "existing provider" path so that
    # set_up_tracing adds processors to it rather than the polluted global one.
    fresh_provider = TracerProvider(resource=resource)
    original = trace._TRACER_PROVIDER
    try:
        trace._TRACER_PROVIDER = fresh_provider
        exporter = InMemorySpanExporter()
        provider = set_up_tracing(config, resource, span_exporters=[exporter])
    finally:
        trace._TRACER_PROVIDER = original

    assert provider is fresh_provider, "set_up_tracing must reuse the injected provider"

    processors = _unwrap_processors(provider)
    assert processors, "Expected at least one processor"

    redaction_index = next(
        (i for i, p in enumerate(processors) if isinstance(p, RedactingSpanProcessor)),
        None,
    )
    assert redaction_index is not None, "RedactingSpanProcessor not found"
    assert redaction_index == 0, (
        f"RedactingSpanProcessor must be first (index 0), "
        f"but found at index {redaction_index}. Chain: {processors}"
    )


@pytest.mark.integration
def test_redaction_processor_not_duplicated_on_existing_provider() -> None:
    """Reusing an existing TracerProvider does not add a duplicate processor."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource

    from holodeck.lib.observability.providers import set_up_tracing
    from holodeck.models.observability import ObservabilityConfig

    resource = Resource.create({"service.name": "test-redaction-dedup"})
    config = ObservabilityConfig(enabled=True)

    # Simulate an existing provider that already has a RedactingSpanProcessor
    # (as set_up_tracing would leave it after a first call).
    existing_provider = TracerProvider(resource=resource)
    existing_provider.add_span_processor(RedactingSpanProcessor())

    # Monkey-patch the global so set_up_tracing sees this provider as "already set"
    original = trace._TRACER_PROVIDER
    try:
        trace._TRACER_PROVIDER = existing_provider

        # This call should detect the existing SDK provider and NOT add another
        # RedactingSpanProcessor.
        set_up_tracing(config, resource, span_exporters=[])
    finally:
        trace._TRACER_PROVIDER = original

    processors = _unwrap_processors(existing_provider)
    redaction_count = sum(
        1 for p in processors if isinstance(p, RedactingSpanProcessor)
    )
    assert redaction_count == 1, (
        f"Expected exactly 1 RedactingSpanProcessor, found {redaction_count}. "
        f"Chain: {processors}"
    )
