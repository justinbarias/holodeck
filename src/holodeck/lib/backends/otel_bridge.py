"""OTel Bridge for Claude Agent SDK.

Translates HoloDeck ObservabilityConfig into environment variable dicts
that configure OpenTelemetry for the Claude subprocess.
"""

import logging

from holodeck.models.observability import ObservabilityConfig, OTLPProtocol

logger = logging.getLogger(__name__)

# Maps OTLPProtocol enum to the OTEL_EXPORTER_OTLP_PROTOCOL env var value.
_PROTOCOL_MAP: dict[OTLPProtocol, str] = {
    OTLPProtocol.GRPC: "grpc",
    OTLPProtocol.HTTP: "http/protobuf",
}

# Default filter_namespaces value from LogsConfig model.
_DEFAULT_FILTER_NAMESPACES = ["semantic_kernel"]


def _collect_unsupported_fields(config: ObservabilityConfig) -> list[str]:
    """Collect names of unsupported fields that have non-default values.

    These fields exist in HoloDeck's ObservabilityConfig but cannot be
    translated to Claude subprocess environment variables.

    Args:
        config: Observability configuration to inspect.

    Returns:
        List of human-readable field descriptions for unsupported features.
    """
    unsupported: list[str] = []

    if config.exporters.azure_monitor and config.exporters.azure_monitor.enabled:
        unsupported.append("exporters.azure_monitor")

    if config.exporters.prometheus and config.exporters.prometheus.enabled:
        unsupported.append("exporters.prometheus")

    if config.traces.redaction_patterns:
        unsupported.append("traces.redaction_patterns")

    if config.traces.sample_rate != 1.0:
        unsupported.append("traces.sample_rate")

    if config.logs.filter_namespaces != _DEFAULT_FILTER_NAMESPACES:
        unsupported.append("logs.filter_namespaces")

    return unsupported


def translate_observability(config: ObservabilityConfig) -> dict[str, str]:
    """Translate ObservabilityConfig to env vars for the Claude subprocess.

    Produces a dict of environment variable key-value pairs that configure
    OpenTelemetry in the Claude subprocess. All values are strings.

    Args:
        config: HoloDeck observability configuration from agent YAML.

    Returns:
        Dictionary of environment variable names to string values.
        Empty dict if observability is disabled.
    """
    if not config.enabled:
        return {}

    env: dict[str, str] = {}

    # Core telemetry enablement
    env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "1"

    # OTLP exporter configuration
    otlp = config.exporters.otlp
    otlp_enabled = otlp is not None and otlp.enabled

    # Metrics exporter
    if otlp_enabled and config.metrics.enabled:
        env["OTEL_METRICS_EXPORTER"] = "otlp"
    else:
        env["OTEL_METRICS_EXPORTER"] = "none"

    # Logs exporter
    if otlp_enabled and config.logs.enabled:
        env["OTEL_LOGS_EXPORTER"] = "otlp"
    else:
        env["OTEL_LOGS_EXPORTER"] = "none"

    # Protocol and endpoint (only if OTLP is enabled)
    if otlp_enabled and otlp is not None:
        env["OTEL_EXPORTER_OTLP_PROTOCOL"] = _PROTOCOL_MAP[otlp.protocol]
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp.endpoint

    # Export intervals
    interval = str(config.metrics.export_interval_ms)
    env["OTEL_METRIC_EXPORT_INTERVAL"] = interval
    env["OTEL_LOGS_EXPORT_INTERVAL"] = interval

    # Privacy controls (FR-038: default off)
    if config.traces.capture_content:
        env["OTEL_LOG_USER_PROMPTS"] = "true"
        env["OTEL_LOG_TOOL_DETAILS"] = "true"

    # Warn about unsupported fields
    unsupported = _collect_unsupported_fields(config)
    if unsupported:
        logger.warning(
            "Claude subprocess does not support these observability settings "
            "(they will be ignored): %s",
            ", ".join(unsupported),
        )

    return env
