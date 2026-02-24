"""Tests for Phase 7 OTel Bridge (T010-T016).

Tests for translate_observability() which translates HoloDeck
ObservabilityConfig into environment variable dicts for the Claude subprocess.
"""

from __future__ import annotations

import logging

import pytest

from holodeck.lib.backends.otel_bridge import translate_observability
from holodeck.models.observability import (
    AzureMonitorExporterConfig,
    ExportersConfig,
    LogsConfig,
    MetricsConfig,
    ObservabilityConfig,
    OTLPExporterConfig,
    OTLPProtocol,
    PrometheusExporterConfig,
    TracingConfig,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_config(
    enabled: bool = True,
    otlp_enabled: bool = True,
    otlp_protocol: OTLPProtocol = OTLPProtocol.GRPC,
    otlp_endpoint: str = "http://localhost:4317",
    metrics_enabled: bool = True,
    metrics_interval_ms: int = 5000,
    logs_enabled: bool = True,
    traces_enabled: bool = True,
    capture_content: bool = False,
    sample_rate: float = 1.0,
    redaction_patterns: list[str] | None = None,
    filter_namespaces: list[str] | None = None,
    azure_monitor_enabled: bool = False,
    prometheus_enabled: bool = False,
) -> ObservabilityConfig:
    """Build an ObservabilityConfig with convenient defaults for testing."""
    traces = TracingConfig(
        enabled=traces_enabled,
        capture_content=capture_content,
        sample_rate=sample_rate,
        redaction_patterns=redaction_patterns or [],
    )
    metrics = MetricsConfig(
        enabled=metrics_enabled,
        export_interval_ms=metrics_interval_ms,
    )
    logs_kwargs: dict = {"enabled": logs_enabled}
    if filter_namespaces is not None:
        logs_kwargs["filter_namespaces"] = filter_namespaces
    logs = LogsConfig(**logs_kwargs)

    exporters_kwargs: dict = {}
    if otlp_enabled:
        exporters_kwargs["otlp"] = OTLPExporterConfig(
            enabled=True,
            endpoint=otlp_endpoint,
            protocol=otlp_protocol,
        )
    if azure_monitor_enabled:
        exporters_kwargs["azure_monitor"] = AzureMonitorExporterConfig(
            enabled=True,
            connection_string="InstrumentationKey=fake-key",
        )
    if prometheus_enabled:
        exporters_kwargs["prometheus"] = PrometheusExporterConfig(enabled=True)

    exporters = ExportersConfig(**exporters_kwargs)

    return ObservabilityConfig(
        enabled=enabled,
        traces=traces,
        metrics=metrics,
        logs=logs,
        exporters=exporters,
    )


# ---------------------------------------------------------------------------
# T010: TestTranslateObservabilityOtlpEnabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityOtlpEnabled:
    """Core env vars with OTLP+GRPC enabled."""

    def test_core_env_vars_present(self) -> None:
        """All core env vars are present when OTLP is enabled."""
        config = _make_config()
        result = translate_observability(config)

        assert result["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in result
        assert "OTEL_EXPORTER_OTLP_PROTOCOL" in result

    def test_all_values_are_strings(self) -> None:
        """Every value in the result dict is a string."""
        config = _make_config()
        result = translate_observability(config)

        for key, value in result.items():
            assert isinstance(value, str), f"{key} is {type(value)}, expected str"

    def test_endpoint_from_config(self) -> None:
        """Endpoint matches the OTLP exporter config."""
        config = _make_config(otlp_endpoint="http://collector:4317")
        result = translate_observability(config)

        assert result["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"

    def test_grpc_protocol_mapping(self) -> None:
        """GRPC protocol maps to 'grpc'."""
        config = _make_config(otlp_protocol=OTLPProtocol.GRPC)
        result = translate_observability(config)

        assert result["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"


# ---------------------------------------------------------------------------
# T011: TestTranslateObservabilityExportInterval
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityExportInterval:
    """Export interval configuration."""

    def test_custom_interval(self) -> None:
        """Custom metrics interval is reflected in env vars."""
        config = _make_config(metrics_interval_ms=10000)
        result = translate_observability(config)

        assert result["OTEL_METRIC_EXPORT_INTERVAL"] == "10000"

    def test_logs_reuses_metrics_interval(self) -> None:
        """Logs export interval reuses metrics interval."""
        config = _make_config(metrics_interval_ms=7500)
        result = translate_observability(config)

        assert result["OTEL_LOGS_EXPORT_INTERVAL"] == "7500"

    def test_default_interval_is_5000(self) -> None:
        """Default interval is 5000ms."""
        config = _make_config()
        result = translate_observability(config)

        assert result["OTEL_METRIC_EXPORT_INTERVAL"] == "5000"


# ---------------------------------------------------------------------------
# T012: TestTranslateObservabilityPrivacy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityPrivacy:
    """Privacy controls (FR-038): capture_content opt-in."""

    def test_default_no_capture(self) -> None:
        """By default, capture env vars are not set."""
        config = _make_config(capture_content=False)
        result = translate_observability(config)

        assert result.get("OTEL_LOG_USER_PROMPTS") != "true"
        assert result.get("OTEL_LOG_TOOL_DETAILS") != "true"

    def test_capture_content_sets_both_env_vars(self) -> None:
        """capture_content=True sets both prompt and tool detail vars."""
        config = _make_config(capture_content=True)
        result = translate_observability(config)

        assert result["OTEL_LOG_USER_PROMPTS"] == "true"
        assert result["OTEL_LOG_TOOL_DETAILS"] == "true"


# ---------------------------------------------------------------------------
# T013: TestTranslateObservabilityDisabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityDisabled:
    """Disabled observability returns empty dict."""

    def test_disabled_returns_empty(self) -> None:
        """enabled=False returns empty dict."""
        config = _make_config(enabled=False)
        result = translate_observability(config)

        assert result == {}

    def test_disabled_with_otlp_config_still_empty(self) -> None:
        """Disabled with OTLP config still returns empty dict."""
        config = _make_config(
            enabled=False,
            otlp_enabled=True,
            otlp_endpoint="http://collector:4317",
        )
        result = translate_observability(config)

        assert result == {}


# ---------------------------------------------------------------------------
# T014: TestTranslateObservabilityWarnings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityWarnings:
    """Unsupported field warnings."""

    def test_azure_monitor_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Azure Monitor triggers a warning."""
        config = _make_config(azure_monitor_enabled=True)
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "azure_monitor" in caplog.text.lower()

    def test_prometheus_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Prometheus triggers a warning."""
        config = _make_config(prometheus_enabled=True)
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "prometheus" in caplog.text.lower()

    def test_redaction_patterns_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-empty redaction_patterns triggers a warning."""
        config = _make_config(redaction_patterns=[r"\d{4}-\d{4}"])
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "redaction_patterns" in caplog.text

    def test_sample_rate_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-default sample_rate triggers a warning."""
        config = _make_config(sample_rate=0.5)
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "sample_rate" in caplog.text

    def test_filter_namespaces_non_default_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-default filter_namespaces triggers a warning."""
        config = _make_config(filter_namespaces=["custom_logger"])
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "filter_namespaces" in caplog.text

    def test_default_filter_namespaces_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default filter_namespaces (['semantic_kernel']) does NOT trigger warning."""
        config = _make_config(filter_namespaces=["semantic_kernel"])
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        assert "filter_namespaces" not in caplog.text

    def test_consolidated_single_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multiple unsupported fields produce a single consolidated warning."""
        config = _make_config(
            azure_monitor_enabled=True,
            prometheus_enabled=True,
            sample_rate=0.5,
        )
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        # Count warning records from our logger
        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "holodeck.lib.backends.otel_bridge" in r.name
        ]
        assert len(warning_records) == 1

    def test_no_unsupported_fields_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No unsupported fields means no warning emitted."""
        config = _make_config()
        with caplog.at_level(logging.WARNING):
            translate_observability(config)

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "holodeck.lib.backends.otel_bridge" in r.name
        ]
        assert len(warning_records) == 0


# ---------------------------------------------------------------------------
# T015: TestTranslateObservabilityHttpProtocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityHttpProtocol:
    """Protocol mapping: HTTP and GRPC."""

    def test_http_maps_to_http_protobuf(self) -> None:
        """HTTP protocol maps to 'http/protobuf', not 'http'."""
        config = _make_config(otlp_protocol=OTLPProtocol.HTTP)
        result = translate_observability(config)

        assert result["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"

    def test_grpc_maps_to_grpc(self) -> None:
        """GRPC protocol maps to 'grpc'."""
        config = _make_config(otlp_protocol=OTLPProtocol.GRPC)
        result = translate_observability(config)

        assert result["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"


# ---------------------------------------------------------------------------
# T016: TestTranslateObservabilityLogsExporter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateObservabilityLogsExporter:
    """Logs and metrics exporter env vars."""

    def test_logs_enabled_sets_otlp(self) -> None:
        """Logs enabled with OTLP sets OTEL_LOGS_EXPORTER to 'otlp'."""
        config = _make_config(logs_enabled=True)
        result = translate_observability(config)

        assert result["OTEL_LOGS_EXPORTER"] == "otlp"

    def test_logs_disabled_sets_none(self) -> None:
        """Logs disabled sets OTEL_LOGS_EXPORTER to 'none'."""
        config = _make_config(logs_enabled=False)
        result = translate_observability(config)

        assert result["OTEL_LOGS_EXPORTER"] == "none"

    def test_metrics_enabled_sets_otlp(self) -> None:
        """Metrics enabled with OTLP sets OTEL_METRICS_EXPORTER to 'otlp'."""
        config = _make_config(metrics_enabled=True)
        result = translate_observability(config)

        assert result["OTEL_METRICS_EXPORTER"] == "otlp"

    def test_metrics_disabled_sets_none(self) -> None:
        """Metrics disabled sets OTEL_METRICS_EXPORTER to 'none'."""
        config = _make_config(metrics_enabled=False)
        result = translate_observability(config)

        assert result["OTEL_METRICS_EXPORTER"] == "none"
