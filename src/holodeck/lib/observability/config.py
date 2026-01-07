"""Observability configuration utilities.

Handles exporter configuration and logging coordination to prevent
double logging when console exporter is active.

Task: T052 - Implement configure_exporters(), is_console_exporter_active()
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.models.observability import ObservabilityConfig


def _any_exporter_enabled(config: ObservabilityConfig) -> bool:
    """Check if any exporter (other than console) is enabled.

    Args:
        config: Observability configuration

    Returns:
        True if any non-console exporter is enabled, False otherwise.
    """
    exporters = config.exporters
    return any(
        [
            exporters.otlp and exporters.otlp.enabled,
            exporters.prometheus and exporters.prometheus.enabled,
            exporters.azure_monitor and exporters.azure_monitor.enabled,
        ]
    )


def is_console_exporter_active(config: ObservabilityConfig) -> bool:
    """Check if console exporter is active.

    Console exporter is active when:
    - Explicitly enabled in configuration, OR
    - No other exporters are enabled (console is the default fallback)

    Args:
        config: Observability configuration

    Returns:
        True if console exporter is active, False otherwise.
    """
    # Explicitly enabled
    if config.exporters.console and config.exporters.console.enabled:
        return True

    # Default to console when no other exporters are configured
    return not _any_exporter_enabled(config)


def configure_logging(config: ObservabilityConfig) -> None:
    """Configure logging to prevent duplicates with console exporter.

    When console exporter is active, removes default StreamHandlers
    from the holodeck logger to prevent duplicate output.

    Args:
        config: Observability configuration

    Note:
        Called automatically by initialize_observability().
    """
    if not is_console_exporter_active(config):
        return

    holodeck_logger = logging.getLogger("holodeck")

    # Remove console handlers to prevent double logging
    for handler in holodeck_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (
            sys.stdout,
            sys.stderr,
        ):
            holodeck_logger.removeHandler(handler)


def configure_exporters(
    config: ObservabilityConfig,
) -> tuple[list[Any], list[Any], list[Any], list[str]]:
    """Configure all explicitly enabled exporters.

    Args:
        config: Observability configuration

    Returns:
        Tuple of (span_exporters, metric_readers, log_exporters, exporter_names)

    Note:
        Only exporters that are explicitly enabled in configuration are added.
        The serve command enables console exporter by default for server logging.
    """
    from holodeck.lib.observability.exporters.console import create_console_exporters

    span_exporters: list[Any] = []
    metric_readers: list[Any] = []
    log_exporters: list[Any] = []
    exporter_names: list[str] = []

    # OTLP exporter (Phase 5 - US2)
    if config.exporters.otlp and config.exporters.otlp.enabled:
        from holodeck.lib.observability.exporters.otlp import create_otlp_exporters

        otlp_span, otlp_metric_reader, otlp_log = create_otlp_exporters(
            config.exporters.otlp
        )
        span_exporters.append(otlp_span)
        metric_readers.append(otlp_metric_reader)
        log_exporters.append(otlp_log)
        exporter_names.append("otlp")

    # Prometheus exporter (Phase 6 - US3)
    if config.exporters.prometheus and config.exporters.prometheus.enabled:
        # TODO: Will be implemented in Phase 6 (US3)
        exporter_names.append("prometheus")

    # Azure Monitor exporter (Phase 7 - US4)
    if config.exporters.azure_monitor and config.exporters.azure_monitor.enabled:
        # TODO: Will be implemented in Phase 7 (US4)
        exporter_names.append("azure_monitor")

    # Console exporter (explicitly enabled or default when no others configured)
    console_explicitly_enabled = (
        config.exporters.console and config.exporters.console.enabled
    )
    console_as_default = not _any_exporter_enabled(config)

    if console_explicitly_enabled or console_as_default:
        from holodeck.models.observability import ConsoleExporterConfig

        console_config = config.exporters.console or ConsoleExporterConfig()
        console_span, console_metric_reader, console_log = create_console_exporters(
            console_config
        )
        span_exporters.append(console_span)
        metric_readers.append(console_metric_reader)
        log_exporters.append(console_log)
        if "console" not in exporter_names:
            exporter_names.append("console")

    return span_exporters, metric_readers, log_exporters, exporter_names


__all__ = [
    "is_console_exporter_active",
    "configure_logging",
    "configure_exporters",
]
