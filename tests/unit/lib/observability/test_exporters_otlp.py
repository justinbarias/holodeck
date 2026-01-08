"""Unit tests for OTLP exporter module.

TDD: These tests are written FIRST, before implementation.
All tests should FAIL until the OTLP exporters are implemented.

Tasks:
    T064 - Unit tests for OTLP gRPC span exporter creation
    T065 - Unit tests for OTLP HTTP span exporter creation
    T066 - Unit tests for OTLP metric exporter creation
    T067 - Unit tests for OTLP log exporter creation
    T068 - Unit tests for custom headers injection (env var substitution)
    T069 - Unit tests for endpoint port adjustment based on protocol
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol

if TYPE_CHECKING:
    pass


# T068: Unit tests for custom headers injection (env var substitution)
@pytest.mark.unit
class TestResolveHeaders:
    """Tests for header resolution with env var substitution."""

    def test_resolves_env_var_in_value(self) -> None:
        """Test ${VAR} pattern is substituted with env value."""
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        os.environ["TEST_API_KEY"] = "secret123"
        try:
            headers = {"Authorization": "Bearer ${TEST_API_KEY}"}
            result = resolve_headers(headers)
            assert result["Authorization"] == "Bearer secret123"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_resolves_multiple_env_vars(self) -> None:
        """Test multiple ${VAR} patterns are resolved."""
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        os.environ["TEST_USER"] = "admin"
        os.environ["TEST_PASS"] = "password123"  # noqa: S105
        try:
            headers = {"X-Auth": "${TEST_USER}:${TEST_PASS}"}
            result = resolve_headers(headers)
            assert result["X-Auth"] == "admin:password123"
        finally:
            del os.environ["TEST_USER"]
            del os.environ["TEST_PASS"]

    def test_raises_on_missing_env_var(self) -> None:
        """Test ConfigError raised for missing env var."""
        from holodeck.lib.errors import ConfigError
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        # Ensure the variable doesn't exist
        os.environ.pop("NONEXISTENT_VAR_12345", None)

        headers = {"Authorization": "Bearer ${NONEXISTENT_VAR_12345}"}
        with pytest.raises(ConfigError):
            resolve_headers(headers)

    def test_passes_through_static_headers(self) -> None:
        """Test headers without ${} are passed through unchanged."""
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        headers = {"Content-Type": "application/json", "X-Custom": "static-value"}
        result = resolve_headers(headers)
        assert result == headers

    def test_mixed_static_and_env_headers(self) -> None:
        """Test mix of static and env var headers."""
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        os.environ["TEST_TOKEN"] = "token123"  # noqa: S105
        try:
            headers = {
                "Authorization": "Bearer ${TEST_TOKEN}",
                "Content-Type": "application/json",
            }
            result = resolve_headers(headers)
            assert result["Authorization"] == "Bearer token123"
            assert result["Content-Type"] == "application/json"
        finally:
            del os.environ["TEST_TOKEN"]

    def test_empty_headers_returns_empty(self) -> None:
        """Test empty headers dict returns empty dict."""
        from holodeck.lib.observability.exporters.otlp import resolve_headers

        result = resolve_headers({})
        assert result == {}


# T069: Unit tests for endpoint port adjustment based on protocol
@pytest.mark.unit
class TestAdjustEndpointForProtocol:
    """Tests for endpoint port adjustment based on protocol."""

    def test_grpc_keeps_4317_port(self) -> None:
        """Test gRPC protocol keeps port 4317."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://localhost:4317"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.GRPC)
        assert result == "http://localhost:4317"

    def test_http_adjusts_4317_to_4318(self) -> None:
        """Test HTTP protocol changes 4317 to 4318."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://localhost:4317"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.HTTP)
        assert result == "http://localhost:4318"

    def test_grpc_adjusts_4318_to_4317(self) -> None:
        """Test gRPC protocol changes 4318 to 4317."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://localhost:4318"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.GRPC)
        assert result == "http://localhost:4317"

    def test_http_keeps_4318_port(self) -> None:
        """Test HTTP protocol keeps port 4318."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://localhost:4318"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.HTTP)
        assert result == "http://localhost:4318"

    def test_non_localhost_unchanged(self) -> None:
        """Test non-localhost endpoints are unchanged."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "https://otel-collector.example.com:4317"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.HTTP)
        assert result == endpoint

    def test_custom_port_unchanged(self) -> None:
        """Test custom ports are not adjusted."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://localhost:9999"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.HTTP)
        assert result == "http://localhost:9999"

    def test_127_0_0_1_treated_as_localhost(self) -> None:
        """Test 127.0.0.1 is treated the same as localhost."""
        from holodeck.lib.observability.exporters.otlp import (
            adjust_endpoint_for_protocol,
        )

        endpoint = "http://127.0.0.1:4317"
        result = adjust_endpoint_for_protocol(endpoint, OTLPProtocol.HTTP)
        assert result == "http://127.0.0.1:4318"


# T064: Unit tests for OTLP gRPC span exporter creation
@pytest.mark.unit
class TestOTLPSpanExporterGRPC:
    """Tests for OTLP gRPC span exporter creation."""

    def test_create_returns_grpc_span_exporter(self) -> None:
        """Test returns OTLPSpanExporter (gRPC) instance."""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        exporter = create_otlp_span_exporter_grpc(config)
        assert isinstance(exporter, OTLPSpanExporterGRPC)

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_uses_configured_endpoint(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter uses configured endpoint."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://custom-collector:4317",
            protocol=OTLPProtocol.GRPC,
        )
        create_otlp_span_exporter_grpc(config)

        mock_exporter_cls.assert_called_once()
        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["endpoint"] == "http://custom-collector:4317"

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_uses_configured_timeout(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter converts timeout_ms to seconds."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
            timeout_ms=60000,  # 60 seconds
        )
        create_otlp_span_exporter_grpc(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["timeout"] == 60.0

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_uses_configured_compression_gzip(
        self, mock_exporter_cls: MagicMock
    ) -> None:
        """Test exporter uses gzip compression when configured."""
        import grpc

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
            compression="gzip",
        )
        create_otlp_span_exporter_grpc(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["compression"] == grpc.Compression.Gzip

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_uses_insecure_setting(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter uses insecure flag."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
            insecure=False,
        )
        create_otlp_span_exporter_grpc(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["insecure"] is False

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterGRPC",
    )
    def test_uses_resolved_headers(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter uses resolved headers."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_grpc,
        )

        os.environ["TEST_GRPC_KEY"] = "grpc-secret"
        try:
            config = OTLPExporterConfig(
                endpoint="http://localhost:4317",
                protocol=OTLPProtocol.GRPC,
                headers={"Authorization": "Bearer ${TEST_GRPC_KEY}"},
            )
            create_otlp_span_exporter_grpc(config)

            call_kwargs = mock_exporter_cls.call_args[1]
            assert call_kwargs["headers"] == (("authorization", "Bearer grpc-secret"),)
        finally:
            del os.environ["TEST_GRPC_KEY"]


# T065: Unit tests for OTLP HTTP span exporter creation
@pytest.mark.unit
class TestOTLPSpanExporterHTTP:
    """Tests for OTLP HTTP span exporter creation."""

    def test_create_returns_http_span_exporter(self) -> None:
        """Test returns OTLPSpanExporter (HTTP) instance."""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        exporter = create_otlp_span_exporter_http(config)
        assert isinstance(exporter, OTLPSpanExporterHTTP)

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterHTTP",
    )
    def test_appends_v1_traces_to_endpoint(self, mock_exporter_cls: MagicMock) -> None:
        """Test HTTP endpoint gets /v1/traces suffix."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        create_otlp_span_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:4318/v1/traces"

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterHTTP",
    )
    def test_does_not_duplicate_v1_traces(self, mock_exporter_cls: MagicMock) -> None:
        """Test doesn't add /v1/traces if already present."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318/v1/traces",
            protocol=OTLPProtocol.HTTP,
        )
        create_otlp_span_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:4318/v1/traces"

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterHTTP",
    )
    def test_uses_configured_headers(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter uses configured headers."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
            headers={"X-Custom-Header": "custom-value"},
        )
        create_otlp_span_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["headers"] == {"X-Custom-Header": "custom-value"}

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPSpanExporterHTTP",
    )
    def test_uses_http_compression(self, mock_exporter_cls: MagicMock) -> None:
        """Test exporter uses HTTP compression enum."""
        from opentelemetry.exporter.otlp.proto.http import Compression

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_span_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
            compression="gzip",
        )
        create_otlp_span_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["compression"] == Compression.Gzip


# T066: Unit tests for OTLP metric exporter creation
@pytest.mark.unit
class TestOTLPMetricExporter:
    """Tests for OTLP metric exporter creation."""

    def test_create_grpc_returns_periodic_reader(self) -> None:
        """Test returns PeriodicExportingMetricReader for gRPC."""
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.otlp import create_otlp_metric_reader

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        reader = create_otlp_metric_reader(config)
        assert isinstance(reader, PeriodicExportingMetricReader)

    def test_create_http_returns_periodic_reader(self) -> None:
        """Test returns PeriodicExportingMetricReader for HTTP."""
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.otlp import create_otlp_metric_reader

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        reader = create_otlp_metric_reader(config)
        assert isinstance(reader, PeriodicExportingMetricReader)

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPMetricExporterHTTP",
    )
    def test_http_metric_endpoint_has_v1_metrics(
        self, mock_exporter_cls: MagicMock
    ) -> None:
        """Test HTTP metric endpoint gets /v1/metrics suffix."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_metric_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        create_otlp_metric_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:4318/v1/metrics"


# T067: Unit tests for OTLP log exporter creation
@pytest.mark.unit
class TestOTLPLogExporter:
    """Tests for OTLP log exporter creation."""

    def test_create_grpc_returns_log_exporter(self) -> None:
        """Test returns OTLPLogExporter (gRPC) instance."""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as OTLPLogExporterGRPC,
        )

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_log_exporter_grpc,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        exporter = create_otlp_log_exporter_grpc(config)
        assert isinstance(exporter, OTLPLogExporterGRPC)

    def test_create_http_returns_log_exporter(self) -> None:
        """Test returns OTLPLogExporter (HTTP) instance."""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as OTLPLogExporterHTTP,
        )

        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_log_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        exporter = create_otlp_log_exporter_http(config)
        assert isinstance(exporter, OTLPLogExporterHTTP)

    @patch(
        "holodeck.lib.observability.exporters.otlp.OTLPLogExporterHTTP",
    )
    def test_http_log_endpoint_has_v1_logs(self, mock_exporter_cls: MagicMock) -> None:
        """Test HTTP log endpoint gets /v1/logs suffix."""
        from holodeck.lib.observability.exporters.otlp import (
            create_otlp_log_exporter_http,
        )

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        create_otlp_log_exporter_http(config)

        call_kwargs = mock_exporter_cls.call_args[1]
        assert call_kwargs["endpoint"] == "http://localhost:4318/v1/logs"

    def test_dispatcher_uses_grpc_for_grpc_protocol(self) -> None:
        """Test create_otlp_log_exporter dispatches to gRPC."""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as OTLPLogExporterGRPC,
        )

        from holodeck.lib.observability.exporters.otlp import create_otlp_log_exporter

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        exporter = create_otlp_log_exporter(config)
        assert isinstance(exporter, OTLPLogExporterGRPC)

    def test_dispatcher_uses_http_for_http_protocol(self) -> None:
        """Test create_otlp_log_exporter dispatches to HTTP."""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as OTLPLogExporterHTTP,
        )

        from holodeck.lib.observability.exporters.otlp import create_otlp_log_exporter

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        exporter = create_otlp_log_exporter(config)
        assert isinstance(exporter, OTLPLogExporterHTTP)


# Factory function tests
@pytest.mark.unit
class TestCreateOTLPExporters:
    """Tests for create_otlp_exporters() factory function."""

    def test_returns_tuple_of_three(self) -> None:
        """Test returns tuple with span, metric, and log exporters."""
        from holodeck.lib.observability.exporters.otlp import create_otlp_exporters

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        result = create_otlp_exporters(config)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_grpc_protocol_creates_grpc_exporters(self) -> None:
        """Test gRPC protocol creates all gRPC exporters."""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as OTLPLogExporterGRPC,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.otlp import create_otlp_exporters

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        span_exp, metric_reader, log_exp = create_otlp_exporters(config)

        assert isinstance(span_exp, OTLPSpanExporterGRPC)
        assert isinstance(metric_reader, PeriodicExportingMetricReader)
        assert isinstance(log_exp, OTLPLogExporterGRPC)

    def test_http_protocol_creates_http_exporters(self) -> None:
        """Test HTTP protocol creates all HTTP exporters."""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as OTLPLogExporterHTTP,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from holodeck.lib.observability.exporters.otlp import create_otlp_exporters

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        span_exp, metric_reader, log_exp = create_otlp_exporters(config)

        assert isinstance(span_exp, OTLPSpanExporterHTTP)
        assert isinstance(metric_reader, PeriodicExportingMetricReader)
        assert isinstance(log_exp, OTLPLogExporterHTTP)


# Compression helper tests
@pytest.mark.unit
class TestCompressionHelpers:
    """Tests for compression conversion helpers."""

    def test_get_compression_grpc_gzip(self) -> None:
        """Test gzip maps to grpc.Compression.Gzip."""
        import grpc

        from holodeck.lib.observability.exporters.otlp import get_compression_grpc

        result = get_compression_grpc("gzip")
        assert result == grpc.Compression.Gzip

    def test_get_compression_grpc_deflate(self) -> None:
        """Test deflate maps to grpc.Compression.Deflate."""
        import grpc

        from holodeck.lib.observability.exporters.otlp import get_compression_grpc

        result = get_compression_grpc("deflate")
        assert result == grpc.Compression.Deflate

    def test_get_compression_grpc_none(self) -> None:
        """Test None returns None."""
        from holodeck.lib.observability.exporters.otlp import get_compression_grpc

        result = get_compression_grpc(None)
        assert result is None

    def test_get_compression_http_gzip(self) -> None:
        """Test gzip maps to HTTP Compression.Gzip."""
        from opentelemetry.exporter.otlp.proto.http import Compression

        from holodeck.lib.observability.exporters.otlp import get_compression_http

        result = get_compression_http("gzip")
        assert result == Compression.Gzip

    def test_get_compression_http_deflate(self) -> None:
        """Test deflate maps to HTTP Compression.Deflate."""
        from opentelemetry.exporter.otlp.proto.http import Compression

        from holodeck.lib.observability.exporters.otlp import get_compression_http

        result = get_compression_http("deflate")
        assert result == Compression.Deflate

    def test_get_compression_http_none(self) -> None:
        """Test None returns None."""
        from holodeck.lib.observability.exporters.otlp import get_compression_http

        result = get_compression_http(None)
        assert result is None


# Span exporter dispatcher tests
@pytest.mark.unit
class TestCreateOTLPSpanExporter:
    """Tests for create_otlp_span_exporter() dispatcher function."""

    def test_dispatches_to_grpc_for_grpc_protocol(self) -> None:
        """Test dispatcher creates gRPC exporter for gRPC protocol."""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )

        from holodeck.lib.observability.exporters.otlp import create_otlp_span_exporter

        config = OTLPExporterConfig(
            endpoint="http://localhost:4317",
            protocol=OTLPProtocol.GRPC,
        )
        exporter = create_otlp_span_exporter(config)
        assert isinstance(exporter, OTLPSpanExporterGRPC)

    def test_dispatches_to_http_for_http_protocol(self) -> None:
        """Test dispatcher creates HTTP exporter for HTTP protocol."""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )

        from holodeck.lib.observability.exporters.otlp import create_otlp_span_exporter

        config = OTLPExporterConfig(
            endpoint="http://localhost:4318",
            protocol=OTLPProtocol.HTTP,
        )
        exporter = create_otlp_span_exporter(config)
        assert isinstance(exporter, OTLPSpanExporterHTTP)
