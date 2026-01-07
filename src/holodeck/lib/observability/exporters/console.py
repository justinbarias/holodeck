"""Console exporter for OpenTelemetry telemetry.

Provides console output for development/debugging when no other
exporters are configured. Uses OpenTelemetry's built-in console exporters.

Tasks:
    T048 - Implement ConsoleSpanExporter wrapper
    T049 - Implement ConsoleMetricExporter wrapper
    T050 - Implement ConsoleLogExporter wrapper
    T051 - Implement create_console_exporters() factory function
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

if TYPE_CHECKING:
    from holodeck.models.observability import ConsoleExporterConfig


def create_console_span_exporter(config: ConsoleExporterConfig) -> ConsoleSpanExporter:
    """Create a console span exporter.

    Args:
        config: Console exporter configuration

    Returns:
        Configured ConsoleSpanExporter instance
    """
    return ConsoleSpanExporter()


def create_console_metric_reader(
    config: ConsoleExporterConfig,
) -> PeriodicExportingMetricReader:
    """Create a console metric reader.

    Args:
        config: Console exporter configuration

    Returns:
        PeriodicExportingMetricReader with ConsoleMetricExporter
    """
    exporter = ConsoleMetricExporter()
    return PeriodicExportingMetricReader(exporter)


def create_console_log_exporter(
    config: ConsoleExporterConfig,
) -> ConsoleLogRecordExporter:
    """Create a console log exporter.

    Args:
        config: Console exporter configuration

    Returns:
        Configured ConsoleLogRecordExporter instance
    """
    return ConsoleLogRecordExporter()


def create_console_exporters(
    config: ConsoleExporterConfig,
) -> tuple[
    ConsoleSpanExporter, PeriodicExportingMetricReader, ConsoleLogRecordExporter
]:
    """Create all console exporters (spans, metrics, logs).

    Factory function that creates all three exporter types for
    the console exporter configuration.

    Args:
        config: Console exporter configuration

    Returns:
        Tuple of (span_exporter, metric_reader, log_exporter)

    Example:
        >>> from holodeck.models.observability import ConsoleExporterConfig
        >>> config = ConsoleExporterConfig()
        >>> span_exp, metric_reader, log_exp = create_console_exporters(config)
    """
    span_exporter = create_console_span_exporter(config)
    metric_reader = create_console_metric_reader(config)
    log_exporter = create_console_log_exporter(config)

    return span_exporter, metric_reader, log_exporter


__all__ = [
    "create_console_span_exporter",
    "create_console_metric_reader",
    "create_console_log_exporter",
    "create_console_exporters",
]
