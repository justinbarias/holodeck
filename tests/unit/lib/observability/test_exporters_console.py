"""Unit tests for console exporter module.

TDD: These tests are written FIRST, before implementation.
All tests should FAIL until the console exporters are implemented.

Task: T037 - Unit tests for console exporter creation
"""

import pytest

from holodeck.models.observability import ConsoleExporterConfig


@pytest.mark.unit
class TestConsoleSpanExporter:
    """Tests for console span exporter creation."""

    def test_create_returns_console_span_exporter(self) -> None:
        """Test returns ConsoleSpanExporter instance."""
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        from holodeck.lib.observability.exporters.console import (
            create_console_span_exporter,
        )

        config = ConsoleExporterConfig()
        exporter = create_console_span_exporter(config)

        assert isinstance(exporter, ConsoleSpanExporter)

    def test_accepts_console_config(self) -> None:
        """Test function accepts ConsoleExporterConfig."""
        from holodeck.lib.observability.exporters.console import (
            create_console_span_exporter,
        )

        config = ConsoleExporterConfig(pretty_print=True, include_timestamps=False)
        exporter = create_console_span_exporter(config)

        assert exporter is not None


@pytest.mark.unit
class TestConsoleMetricReader:
    """Tests for console metric reader creation."""

    def test_create_returns_periodic_reader(self) -> None:
        """Test returns PeriodicExportingMetricReader."""
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.console import (
            create_console_metric_reader,
        )

        config = ConsoleExporterConfig()
        reader = create_console_metric_reader(config)

        assert isinstance(reader, PeriodicExportingMetricReader)

    def test_accepts_console_config(self) -> None:
        """Test function accepts ConsoleExporterConfig."""
        from holodeck.lib.observability.exporters.console import (
            create_console_metric_reader,
        )

        config = ConsoleExporterConfig(pretty_print=False)
        reader = create_console_metric_reader(config)

        assert reader is not None


@pytest.mark.unit
class TestConsoleLogExporter:
    """Tests for console log exporter creation."""

    def test_create_returns_console_log_exporter(self) -> None:
        """Test returns ConsoleLogRecordExporter instance."""
        from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter

        from holodeck.lib.observability.exporters.console import (
            create_console_log_exporter,
        )

        config = ConsoleExporterConfig()
        exporter = create_console_log_exporter(config)

        assert isinstance(exporter, ConsoleLogRecordExporter)

    def test_accepts_console_config(self) -> None:
        """Test function accepts ConsoleExporterConfig."""
        from holodeck.lib.observability.exporters.console import (
            create_console_log_exporter,
        )

        config = ConsoleExporterConfig(include_timestamps=True)
        exporter = create_console_log_exporter(config)

        assert exporter is not None


@pytest.mark.unit
class TestCreateConsoleExporters:
    """Tests for create_console_exporters() factory function."""

    def test_returns_tuple_of_three(self) -> None:
        """Test returns tuple with span, metric, and log exporters."""
        from holodeck.lib.observability.exporters.console import (
            create_console_exporters,
        )

        config = ConsoleExporterConfig()
        result = create_console_exporters(config)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_first_element_is_span_exporter(self) -> None:
        """Test first element is ConsoleSpanExporter."""
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        from holodeck.lib.observability.exporters.console import (
            create_console_exporters,
        )

        config = ConsoleExporterConfig()
        span_exp, _, _ = create_console_exporters(config)

        assert isinstance(span_exp, ConsoleSpanExporter)

    def test_second_element_is_metric_reader(self) -> None:
        """Test second element is PeriodicExportingMetricReader."""
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.console import (
            create_console_exporters,
        )

        config = ConsoleExporterConfig()
        _, metric_reader, _ = create_console_exporters(config)

        assert isinstance(metric_reader, PeriodicExportingMetricReader)

    def test_third_element_is_log_exporter(self) -> None:
        """Test third element is ConsoleLogRecordExporter."""
        from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter

        from holodeck.lib.observability.exporters.console import (
            create_console_exporters,
        )

        config = ConsoleExporterConfig()
        _, _, log_exp = create_console_exporters(config)

        assert isinstance(log_exp, ConsoleLogRecordExporter)

    def test_all_exporters_are_not_none(self) -> None:
        """Test all returned exporters are not None."""
        from holodeck.lib.observability.exporters.console import (
            create_console_exporters,
        )

        config = ConsoleExporterConfig()
        span_exp, metric_reader, log_exp = create_console_exporters(config)

        assert span_exp is not None
        assert metric_reader is not None
        assert log_exp is not None
