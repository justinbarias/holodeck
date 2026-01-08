"""Telemetry exporters for HoloDeck observability.

Supports OTLP, Prometheus, Azure Monitor, and Console exporters.
"""

from holodeck.lib.observability.exporters.console import (
    create_console_exporters,
    create_console_log_exporter,
    create_console_metric_reader,
    create_console_span_exporter,
)
from holodeck.lib.observability.exporters.otlp import (
    adjust_endpoint_for_protocol,
    create_otlp_exporters,
    create_otlp_log_exporter,
    create_otlp_metric_reader,
    create_otlp_span_exporter,
    resolve_headers,
)

__all__ = [
    # Console exporters
    "create_console_exporters",
    "create_console_span_exporter",
    "create_console_metric_reader",
    "create_console_log_exporter",
    # OTLP exporters
    "create_otlp_exporters",
    "create_otlp_span_exporter",
    "create_otlp_metric_reader",
    "create_otlp_log_exporter",
    "resolve_headers",
    "adjust_endpoint_for_protocol",
]
