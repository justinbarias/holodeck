"""Unit tests for the OTel bridge between Python logging and the LoggerProvider.

The CLI decision between OTel and ``setup_logging`` is owned by the
per-command ``test_*_observability.py`` suites.

Task: T120 - Tests for logging coordination
"""

import logging
from unittest.mock import patch

import pytest
from opentelemetry._logs import get_logger_provider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from holodeck.models.observability import ObservabilityConfig


@pytest.mark.unit
@pytest.mark.usefixtures("isolated_otel")
class TestOTelLoggingCapture:
    """Tests for OTel LoggerProvider capturing Python logging."""

    def test_otel_logger_provider_initialized(self) -> None:
        """The LoggerProvider is installed globally and bridged to root logging."""
        from holodeck.lib.observability import initialize_observability

        config = ObservabilityConfig(enabled=True)

        context = initialize_observability(config, agent_name="test-agent")

        assert context.logger_provider is not None
        assert get_logger_provider() is context.logger_provider
        bridges = [
            h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler)
        ]
        assert len(bridges) == 1

    def test_otel_shutdown_flushes_logs(self) -> None:
        """shutdown_observability exports log records still waiting in the batch."""
        from holodeck.lib.observability import (
            initialize_observability,
            shutdown_observability,
        )

        exporter = InMemoryLogRecordExporter()
        with patch(
            "holodeck.lib.observability.config.configure_exporters",
            return_value=([], [], [exporter], ["memory"]),
        ):
            context = initialize_observability(
                ObservabilityConfig(enabled=True), agent_name="test-agent"
            )

        logging.getLogger("holodeck.flush_check").warning("flush me")
        # The batch processor holds the record until its schedule delay.
        assert exporter.get_finished_logs() == ()

        shutdown_observability(context)

        # Other libraries may log during initialization; count only ours.
        bodies = [log.log_record.body for log in exporter.get_finished_logs()]
        assert bodies.count("flush me") == 1
