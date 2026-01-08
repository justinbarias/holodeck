"""OpenTelemetry observability module for HoloDeck.

This module provides telemetry instrumentation for HoloDeck agents,
following OpenTelemetry GenAI semantic conventions.

Public API:
    initialize_observability: Initialize all telemetry providers
    shutdown_observability: Gracefully shutdown providers
    get_tracer: Get a tracer for creating spans
    get_meter: Get a meter for creating metrics
    enable_semantic_kernel_telemetry: Enable SK's native GenAI instrumentation
    ObservabilityContext: Container for initialized providers

Errors:
    ObservabilityError: Base exception for observability errors
    ObservabilityConfigError: Invalid configuration error

Example:
    >>> from holodeck.lib.observability import initialize_observability
    >>> from holodeck.lib.observability import shutdown_observability
    >>> from holodeck.models.observability import ObservabilityConfig
    >>>
    >>> config = ObservabilityConfig(enabled=True)
    >>> context = initialize_observability(config, agent_name="my-agent")
    >>> try:
    ...     # Run agent
    ...     pass
    ... finally:
    ...     shutdown_observability(context)

Task: T053 - Export public API from __init__.py
"""

from holodeck.lib.observability.errors import (
    ObservabilityConfigError,
    ObservabilityError,
)
from holodeck.lib.observability.instrumentation import (
    enable_semantic_kernel_telemetry,
)
from holodeck.lib.observability.providers import (
    ObservabilityContext,
    get_meter,
    get_tracer,
    initialize_observability,
    shutdown_observability,
)

__all__ = [
    # Core functions
    "initialize_observability",
    "shutdown_observability",
    "get_tracer",
    "get_meter",
    # Instrumentation
    "enable_semantic_kernel_telemetry",
    # Context
    "ObservabilityContext",
    # Errors
    "ObservabilityError",
    "ObservabilityConfigError",
]
