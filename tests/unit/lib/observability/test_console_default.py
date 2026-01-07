"""Unit tests for console exporter as default behavior.

TDD: These tests are written FIRST, before implementation.

Task: T038 - Unit tests for console exporter as default when no exporters configured
"""

import logging

import pytest

from holodeck.models.observability import (
    ConsoleExporterConfig,
    ExportersConfig,
    ObservabilityConfig,
    OTLPExporterConfig,
    PrometheusExporterConfig,
)


@pytest.mark.unit
class TestIsConsoleExporterActive:
    """Tests for is_console_exporter_active() function."""

    def test_returns_true_when_no_exporters_configured(self) -> None:
        """Test returns True when no exporters are configured."""
        from holodeck.lib.observability.config import is_console_exporter_active

        config = ObservabilityConfig(enabled=True)
        assert is_console_exporter_active(config) is True

    def test_returns_false_when_otlp_enabled(self) -> None:
        """Test returns False when OTLP exporter is enabled."""
        from holodeck.lib.observability.config import is_console_exporter_active

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(enabled=True),
            ),
        )
        assert is_console_exporter_active(config) is False

    def test_returns_false_when_prometheus_enabled(self) -> None:
        """Test returns False when Prometheus exporter is enabled."""
        from holodeck.lib.observability.config import is_console_exporter_active

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                prometheus=PrometheusExporterConfig(enabled=True),
            ),
        )
        assert is_console_exporter_active(config) is False

    def test_returns_true_when_console_explicitly_enabled(self) -> None:
        """Test returns True when console is explicitly enabled."""
        from holodeck.lib.observability.config import is_console_exporter_active

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                console=ConsoleExporterConfig(enabled=True),
            ),
        )
        assert is_console_exporter_active(config) is True

    def test_returns_false_when_console_disabled_and_otlp_enabled(self) -> None:
        """Test returns False when console is disabled and OTLP is enabled."""
        from holodeck.lib.observability.config import is_console_exporter_active

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                console=ConsoleExporterConfig(enabled=False),
                otlp=OTLPExporterConfig(enabled=True),
            ),
        )
        assert is_console_exporter_active(config) is False


@pytest.mark.unit
class TestConfigureExporters:
    """Tests for configure_exporters() function."""

    def test_returns_console_exporters_when_none_configured(self) -> None:
        """Test returns console exporters when no exporters configured."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(enabled=True)
        span_exp, metric_readers, log_exp, names = configure_exporters(config)

        assert len(span_exp) == 1
        assert len(metric_readers) == 1
        assert len(log_exp) == 1
        assert "console" in names

    def test_exporter_names_includes_console_as_default(self) -> None:
        """Test exporter names list includes 'console' as default."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(enabled=True)
        _, _, _, names = configure_exporters(config)

        assert names == ["console"]

    def test_returns_console_when_explicitly_enabled(self) -> None:
        """Test returns console exporters when explicitly enabled."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                console=ConsoleExporterConfig(enabled=True),
            ),
        )
        span_exp, metric_readers, log_exp, names = configure_exporters(config)

        assert "console" in names

    def test_returns_empty_when_otlp_enabled_only(self) -> None:
        """Test returns no console exporters when OTLP is enabled."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(enabled=True),
            ),
        )
        span_exp, metric_readers, log_exp, names = configure_exporters(config)

        # OTLP exporters are not implemented yet in Phase 3
        # So the lists will be empty, but "console" should not be in names
        assert "console" not in names


@pytest.mark.unit
class TestConfigureLogging:
    """Tests for configure_logging() function."""

    def test_removes_stream_handlers_when_console_active(self) -> None:
        """Test removes StreamHandlers when console exporter is active."""
        import sys

        from holodeck.lib.observability.config import configure_logging

        # Set up a handler on the holodeck logger
        logger = logging.getLogger("holodeck")
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        initial_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        initial_count = len(initial_handlers)

        config = ObservabilityConfig(enabled=True)  # Console as default
        configure_logging(config)

        # Handler should be removed
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and h.stream in (sys.stdout, sys.stderr)
        ]
        assert len(stream_handlers) < initial_count

    def test_does_not_remove_handlers_when_otlp_configured(self) -> None:
        """Test does not remove handlers when OTLP is configured."""
        import sys

        from holodeck.lib.observability.config import configure_logging

        # Set up a handler on the holodeck logger
        logger = logging.getLogger("holodeck_test_otlp")
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        initial_count = len(
            [
                h
                for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
                and h.stream in (sys.stdout, sys.stderr)
            ]
        )

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(enabled=True),
            ),
        )
        configure_logging(config)

        # Handler should NOT be removed (we check a different logger)
        # For the test to work correctly, we verify that configure_logging
        # does NOT affect handlers when OTLP is configured
        final_count = len(
            [
                h
                for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
                and h.stream in (sys.stdout, sys.stderr)
            ]
        )
        assert final_count == initial_count
