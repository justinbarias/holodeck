"""OpenTelemetry provider setup and management for HoloDeck.

Provides functions to initialize, configure, and shutdown telemetry providers
following OpenTelemetry best practices and GenAI semantic conventions.

Tasks:
    T040 - Implement create_resource(config, agent_name) function
    T041 - Implement ObservabilityContext dataclass
    T042 - Implement set_up_logging() function
    T043 - Implement set_up_tracing() function
    T044 - Implement set_up_metrics() function
    T045 - Implement initialize_observability(config, agent_name) function
    T046 - Implement shutdown_observability() function
    T047 - Implement get_tracer() and get_meter() helper functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

    from holodeck.models.observability import ObservabilityConfig


# Module-level state for global provider access
_observability_context: ObservabilityContext | None = None


@dataclass
class ObservabilityContext:
    """Container for initialized observability components.

    Holds references to all telemetry providers and tracks which
    exporters are active. Used for lifecycle management and
    provider access.

    Attributes:
        tracer_provider: OpenTelemetry TracerProvider instance
        meter_provider: OpenTelemetry MeterProvider instance
        logger_provider: OpenTelemetry LoggerProvider instance
        exporters: List of enabled exporter names (e.g., ["console", "otlp"])
        resource: Shared OpenTelemetry Resource
    """

    tracer_provider: TracerProvider | None
    meter_provider: MeterProvider | None
    logger_provider: Any  # LoggerProvider, using Any to avoid import issues
    exporters: list[str] = field(default_factory=list)
    resource: Resource = field(default_factory=Resource.create)

    def is_enabled(self) -> bool:
        """Check if observability is active.

        Returns:
            True if all providers are initialized, False otherwise.
        """
        return (
            self.tracer_provider is not None
            and self.meter_provider is not None
            and self.logger_provider is not None
        )

    def get_resource(self) -> Resource:
        """Get the shared OpenTelemetry resource.

        Returns:
            The Resource instance shared by all providers.
        """
        return self.resource


def create_resource(config: ObservabilityConfig, agent_name: str) -> Resource:
    """Create OpenTelemetry resource with service name and attributes.

    Service name resolution order:
    1. config.service_name (if provided)
    2. f"holodeck-{agent_name}" (default)

    Args:
        config: Observability configuration from agent.yaml
        agent_name: Agent name from agent.yaml (used for default service name)

    Returns:
        OpenTelemetry Resource with service name and custom attributes

    Example:
        >>> config = ObservabilityConfig(enabled=True)
        >>> resource = create_resource(config, agent_name="customer-support")
        >>> # Service name is "holodeck-customer-support"
    """
    service_name = config.service_name or f"holodeck-{agent_name}"

    attributes: dict[str, Any] = {
        "service.name": service_name,
        **config.resource_attributes,
    }

    return Resource.create(attributes)


def set_up_logging(
    config: ObservabilityConfig,
    resource: Resource,
    log_exporters: list[Any],
    verbose: bool = False,
    quiet: bool = False,
) -> Any:
    """Set up OpenTelemetry LoggerProvider and bridge Python logging.

    Must be called FIRST before tracing and metrics per OTel Python docs.

    Args:
        config: Observability configuration
        resource: Shared OpenTelemetry Resource
        log_exporters: List of log exporters to add
        verbose: If True, set log level to DEBUG
        quiet: If True, set log level to WARNING (overrides verbose)

    Returns:
        Configured LoggerProvider instance
    """
    import logging

    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    # Determine log level based on flags (same logic as setup_logging)
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger_provider = LoggerProvider(resource=resource)

    # Add all log exporters
    for exporter in log_exporters:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    set_logger_provider(logger_provider)

    # Bridge Python's logging module to OTel
    # This handler captures Python log records and sends them to OTel
    otel_handler = LoggingHandler(
        level=log_level,
        logger_provider=logger_provider,
    )

    # Add OTel handler to root logger so all Python logs are captured
    # Note: Child loggers (like "holodeck") propagate to root by default,
    # so we only need to add the handler here to avoid duplicate records.
    logging.getLogger().addHandler(otel_handler)
    logging.getLogger().setLevel(log_level)

    # Configure third-party loggers to respect verbosity settings
    from holodeck.lib.logging_config import configure_third_party_loggers

    configure_third_party_loggers(log_level)

    return logger_provider


def set_up_tracing(
    config: ObservabilityConfig,
    resource: Resource,
    span_exporters: list[Any],
) -> TracerProvider:
    """Set up OpenTelemetry TracerProvider with span exporters.

    Creates a TracerProvider with the resource and registers it globally.
    If a TracerProvider was already set by another library, we use that
    existing provider and add our span processors to it.

    Args:
        config: Observability configuration
        resource: Shared OpenTelemetry Resource
        span_exporters: List of span exporters to add

    Returns:
        Configured TracerProvider instance

    Note:
        This must be called before any code that creates spans. The
        SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS env var
        should be set in main.py before any SK imports.
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Check if a real TracerProvider was already set by another library
    existing_provider = trace.get_tracer_provider()

    if isinstance(existing_provider, TracerProvider):
        # Another library already set a TracerProvider - use it
        # We can still add our span processors to capture telemetry
        tracer_provider = existing_provider
    else:
        # No real provider set yet - create ours with the resource
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

    # Configure batch processor settings from config
    for exporter in span_exporters:
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=config.traces.max_queue_size,
            max_export_batch_size=config.traces.max_export_batch_size,
        )
        tracer_provider.add_span_processor(processor)

    return tracer_provider


def set_up_metrics(
    config: ObservabilityConfig,
    resource: Resource,
    metric_readers: list[Any],
) -> MeterProvider:
    """Set up OpenTelemetry MeterProvider with metric readers.

    Args:
        config: Observability configuration
        resource: Shared OpenTelemetry Resource
        metric_readers: List of metric readers to add

    Returns:
        Configured MeterProvider instance
    """
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers,
    )

    metrics.set_meter_provider(meter_provider)

    return meter_provider


def initialize_observability(
    config: ObservabilityConfig,
    agent_name: str,
    verbose: bool = False,
    quiet: bool = False,
) -> ObservabilityContext:
    """Initialize all telemetry providers based on configuration.

    Args:
        config: Observability configuration from agent.yaml
        agent_name: Agent name from agent.yaml (used for default service name)
        verbose: If True, set log level to DEBUG
        quiet: If True, set log level to WARNING (overrides verbose)

    Returns:
        ObservabilityContext with initialized providers

    Raises:
        ObservabilityConfigError: If configuration is invalid

    Note:
        Initialization order is critical:
        1. Configure logging first (prevents double logging)
        2. Set up logging provider
        3. Set up tracing provider
        4. Set up metrics provider

    Example:
        >>> from holodeck.lib.observability import initialize_observability
        >>> from holodeck.models.observability import ObservabilityConfig
        >>>
        >>> config = ObservabilityConfig(enabled=True)
        >>> context = initialize_observability(config, agent_name="my-agent")
    """
    global _observability_context

    from holodeck.lib.observability.config import configure_exporters, configure_logging

    # 1. Create shared resource
    resource = create_resource(config, agent_name)

    # 2. Configure exporters (returns span, metric, log exporters)
    span_exporters, metric_readers, log_exporters, exporter_names = configure_exporters(
        config
    )

    # 3. Configure logging (prevents double logging with console exporter)
    configure_logging(config)

    # 4. Set up logging (must be first per OTel docs)
    logger_provider = set_up_logging(config, resource, log_exporters, verbose, quiet)

    # 5. Set up tracing
    tracer_provider = set_up_tracing(config, resource, span_exporters)

    # 6. Set up metrics
    meter_provider = set_up_metrics(config, resource, metric_readers)

    # 7. Create and store context
    _observability_context = ObservabilityContext(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        exporters=exporter_names,
        resource=resource,
    )

    # 8. Enable Semantic Kernel telemetry (GenAI semantic conventions)
    from holodeck.lib.observability.instrumentation import (
        enable_semantic_kernel_telemetry,
    )

    enable_semantic_kernel_telemetry(config)

    return _observability_context


def shutdown_observability(context: ObservabilityContext) -> None:
    """Flush pending telemetry and shutdown providers.

    Args:
        context: ObservabilityContext from initialize_observability

    Note:
        Should be called during application shutdown.
        Blocks until all pending spans/metrics are flushed.
    """
    global _observability_context

    # Force flush all pending data before shutdown
    if context.tracer_provider:
        context.tracer_provider.force_flush()

    if context.meter_provider:
        context.meter_provider.force_flush()

    if context.logger_provider:
        context.logger_provider.force_flush()

    # Shutdown in reverse order of initialization
    if context.meter_provider:
        context.meter_provider.shutdown()

    if context.tracer_provider:
        context.tracer_provider.shutdown()

    if context.logger_provider:
        context.logger_provider.shutdown()

    _observability_context = None


def get_tracer(name: str) -> Tracer:
    """Get an OpenTelemetry tracer instance.

    Args:
        name: Tracer name (typically __name__)

    Returns:
        OpenTelemetry Tracer instance

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation"):
        ...     # do work
    """
    return trace.get_tracer(name)


def get_meter(name: str) -> Meter:
    """Get an OpenTelemetry meter instance.

    Args:
        name: Meter name (typically __name__)

    Returns:
        OpenTelemetry Meter instance

    Example:
        >>> meter = get_meter(__name__)
        >>> counter = meter.create_counter("requests")
        >>> counter.add(1)
    """
    return metrics.get_meter(name)


__all__ = [
    "ObservabilityContext",
    "create_resource",
    "set_up_logging",
    "set_up_tracing",
    "set_up_metrics",
    "initialize_observability",
    "shutdown_observability",
    "get_tracer",
    "get_meter",
]
