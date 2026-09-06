"""Semantic Kernel telemetry instrumentation for HoloDeck.

Enables Semantic Kernel's native OpenTelemetry instrumentation via
environment variables. SK provides comprehensive GenAI semantic convention
support out of the box.

Task: T061 - Implement enable_semantic_kernel_telemetry()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.models.observability import ObservabilityConfig

# Environment variable names used by Semantic Kernel for telemetry
SK_OTEL_DIAGNOSTICS_ENV = "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"
SK_OTEL_SENSITIVE_ENV = (
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"
)


def enable_semantic_kernel_telemetry(config: ObservabilityConfig) -> None:
    """Enable Semantic Kernel's native OpenTelemetry instrumentation.

    Sets environment variables that Semantic Kernel reads to enable telemetry.
    SK provides comprehensive GenAI semantic convention support, automatically
    capturing attributes like:

    - gen_ai.operation.name (e.g., "chat.completions")
    - gen_ai.system (e.g., "openai")
    - gen_ai.request.model (e.g., "gpt-4o")
    - gen_ai.response.id, gen_ai.response.finish_reason
    - gen_ai.usage.prompt_tokens, gen_ai.usage.completion_tokens

    When sensitive diagnostics is enabled, SK also captures:
    - gen_ai.content.prompt (via span events)
    - gen_ai.content.completion (via span events)

    Args:
        config: ObservabilityConfig with traces settings

    Note:
        This function must be called BEFORE any Semantic Kernel operations.
        SK reads these environment variables at initialization time.

    Example:
        >>> from holodeck.models.observability import ObservabilityConfig
        >>> config = ObservabilityConfig(enabled=True)
        >>> enable_semantic_kernel_telemetry(config)
        >>> # SK will now emit GenAI semantic convention spans
    """
    # Always enable basic GenAI diagnostics when observability is on
    os.environ[SK_OTEL_DIAGNOSTICS_ENV] = "true"

    # Enable sensitive content capture if explicitly configured
    # This captures prompts and completions in span events
    if config.traces.capture_content:
        os.environ[SK_OTEL_SENSITIVE_ENV] = "true"


def enable_litellm_telemetry(config: ObservabilityConfig) -> None:
    """Register LiteLLM's OpenTelemetry callback for GenAI spans.

    LiteLLM backs the RAG layer (embeddings + contextual-retrieval chat).
    Its OTel callback emits GenAI semantic-convention spans
    (``gen_ai.operation.name``, ``gen_ai.request.model``, token usage) and,
    when content capture is on, message content under
    ``gen_ai.input.messages`` / ``gen_ai.output.messages`` — both prefixes
    are scrubbed by ``RedactingSpanProcessor``.

    Must be called AFTER the global tracer provider is configured: the
    callback reuses an existing SDK ``TracerProvider``, so LiteLLM spans
    flow through HoloDeck's processor chain (including redaction) and
    exporters. Idempotent — a second call is a no-op.

    Semantic-convention shape (pinned against LiteLLM 1.88): HoloDeck does
    not set ``OTEL_SEMCONV_STABILITY_OPT_IN``. With the default (legacy)
    shape every call is one ``litellm_request`` span whose operation rides on
    ``llm.request.type`` (``aembedding`` / ``acompletion``) and whose provider
    rides on ``gen_ai.system``. Operators who export
    ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` before
    startup get the current GenAI shape instead (span name
    ``embeddings <model>`` / ``chat <model>``, ``gen_ai.provider.name``).
    In both shapes ``gen_ai.operation.name`` and the finish-reason attribute
    are written only together with captured content. The model, token-usage,
    and content attribute names are identical in both shapes, so redaction
    and the no-content mode hold either way.

    Content capture: with ``SPAN_ONLY`` LiteLLM also emits a
    ``raw_gen_ai_request`` child span carrying the provider payload under
    ``llm.openai.stringified_raw_response`` (skipped under the semconv
    opt-in); that key is in the redaction prefix list. With ``NO_CONTENT``
    neither the message attributes nor the raw-response span are emitted.
    Failure spans carry ``error.*`` attributes and an ``exception`` event in
    every mode; ``RedactingSpanProcessor`` scrubs those as well. The env
    opt-in is read once when the callback is constructed.

    Args:
        config: ObservabilityConfig with traces settings.
    """
    import litellm
    from litellm.integrations.opentelemetry import (
        OpenTelemetry,
        OpenTelemetryConfig,
    )

    if any(isinstance(callback, OpenTelemetry) for callback in litellm.callbacks):
        return

    # SPAN_ONLY (not SPAN_AND_EVENT): RedactingSpanProcessor scrubs span
    # attributes, not span events, so content must never ride on events.
    capture_mode = "SPAN_ONLY" if config.traces.capture_content else "NO_CONTENT"
    otel_callback = OpenTelemetry(
        config=OpenTelemetryConfig(capture_message_content=capture_mode)
    )
    litellm.callbacks.append(otel_callback)


__all__ = [
    "enable_semantic_kernel_telemetry",
    "enable_litellm_telemetry",
    "SK_OTEL_DIAGNOSTICS_ENV",
    "SK_OTEL_SENSITIVE_ENV",
]
