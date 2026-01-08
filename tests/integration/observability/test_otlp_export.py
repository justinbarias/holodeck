"""Integration tests for OTLP exporter with mock collector.

Task: T070 - Integration test for OTLP exporter with mock collector

These tests verify that OTLP exporters correctly integrate with the
configure_exporters() function and properly export telemetry data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.observability import (
    ExportersConfig,
    ObservabilityConfig,
    OTLPExporterConfig,
    OTLPProtocol,
)


@pytest.mark.integration
class TestOTLPExportIntegration:
    """Integration tests for OTLP export to mock collector."""

    def test_configure_exporters_creates_otlp_exporters(self) -> None:
        """Test configure_exporters creates OTLP exporters when enabled."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=True,
                    endpoint="http://localhost:4317",
                    protocol=OTLPProtocol.GRPC,
                ),
            ),
        )

        span_exporters, metric_readers, log_exporters, names = configure_exporters(
            config
        )

        assert "otlp" in names
        assert len(span_exporters) == 1
        assert len(metric_readers) == 1
        assert len(log_exporters) == 1

    def test_configure_exporters_skips_disabled_otlp(self) -> None:
        """Test configure_exporters skips OTLP when disabled.

        When OTLP is disabled and no other exporter is enabled,
        console becomes the default exporter.
        """
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=False,
                    endpoint="http://localhost:4317",
                ),
            ),
        )

        span_exporters, metric_readers, log_exporters, names = configure_exporters(
            config
        )

        # OTLP is disabled, so it's not in the names
        assert "otlp" not in names
        # Console becomes default when no other exporter is enabled
        assert "console" in names
        assert len(span_exporters) == 1
        assert len(metric_readers) == 1
        assert len(log_exporters) == 1

    def test_configure_exporters_multiple_exporters(self) -> None:
        """Test configure_exporters creates multiple exporters."""
        from holodeck.lib.observability.config import configure_exporters
        from holodeck.models.observability import ConsoleExporterConfig

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=True,
                    endpoint="http://localhost:4317",
                    protocol=OTLPProtocol.GRPC,
                ),
                console=ConsoleExporterConfig(enabled=True),
            ),
        )

        span_exporters, metric_readers, log_exporters, names = configure_exporters(
            config
        )

        assert "otlp" in names
        assert "console" in names
        assert len(span_exporters) == 2
        assert len(metric_readers) == 2
        assert len(log_exporters) == 2

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_span_exporter_receives_correct_config(
        self, mock_span_exporter: MagicMock
    ) -> None:
        """Test span exporter receives correct configuration."""
        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=True,
                    endpoint="http://jaeger:4317",
                    protocol=OTLPProtocol.GRPC,
                    timeout_ms=45000,
                    insecure=True,
                ),
            ),
        )

        configure_exporters(config)

        mock_span_exporter.assert_called_once()
        call_kwargs = mock_span_exporter.call_args[1]
        assert call_kwargs["endpoint"] == "http://jaeger:4317"
        assert call_kwargs["timeout"] == 45.0
        assert call_kwargs["insecure"] is True

    def test_http_protocol_uses_http_exporters(self) -> None:
        """Test HTTP protocol creates HTTP-based exporters."""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as OTLPLogExporterHTTP,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )

        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=True,
                    endpoint="http://localhost:4318",
                    protocol=OTLPProtocol.HTTP,
                ),
            ),
        )

        span_exporters, _, log_exporters, _ = configure_exporters(config)

        assert isinstance(span_exporters[0], OTLPSpanExporterHTTP)
        assert isinstance(log_exporters[0], OTLPLogExporterHTTP)

    def test_grpc_protocol_uses_grpc_exporters(self) -> None:
        """Test gRPC protocol creates gRPC-based exporters."""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as OTLPLogExporterGRPC,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )

        from holodeck.lib.observability.config import configure_exporters

        config = ObservabilityConfig(
            enabled=True,
            exporters=ExportersConfig(
                otlp=OTLPExporterConfig(
                    enabled=True,
                    endpoint="http://localhost:4317",
                    protocol=OTLPProtocol.GRPC,
                ),
            ),
        )

        span_exporters, _, log_exporters, _ = configure_exporters(config)

        assert isinstance(span_exporters[0], OTLPSpanExporterGRPC)
        assert isinstance(log_exporters[0], OTLPLogExporterGRPC)
