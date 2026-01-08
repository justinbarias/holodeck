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


__all__ = [
    "enable_semantic_kernel_telemetry",
    "SK_OTEL_DIAGNOSTICS_ENV",
    "SK_OTEL_SENSITIVE_ENV",
]
