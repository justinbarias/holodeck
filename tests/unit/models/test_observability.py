"""Unit tests for observability configuration models.

TDD: These tests are written FIRST, before implementation.
All tests should FAIL until the models are implemented.
"""

import pytest
from pydantic import ValidationError


# T007: Unit tests for LogLevel enum
@pytest.mark.unit
class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_valid_levels(self) -> None:
        """Test all valid log levels."""
        from holodeck.models.observability import LogLevel

        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_string_enum(self) -> None:
        """Test LogLevel is a string enum."""
        from holodeck.models.observability import LogLevel

        assert isinstance(LogLevel.INFO, str)
        assert LogLevel.INFO.value == "INFO"

    def test_all_levels_defined(self) -> None:
        """Test that all expected log levels are defined."""
        from holodeck.models.observability import LogLevel

        expected_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        actual_levels = {level.value for level in LogLevel}
        assert actual_levels == expected_levels


# T008: Unit tests for TracingConfig
@pytest.mark.unit
class TestTracingConfig:
    """Tests for TracingConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import TracingConfig

        config = TracingConfig()
        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.capture_content is False
        assert config.capture_evaluation_content is False
        assert config.redaction_patterns == []
        assert config.max_queue_size == 2048
        assert config.max_export_batch_size == 512

    def test_capture_evaluation_content_default_false(self) -> None:
        """Test capture_evaluation_content defaults to False."""
        from holodeck.models.observability import TracingConfig

        config = TracingConfig()
        assert config.capture_evaluation_content is False

    def test_capture_evaluation_content_can_be_enabled(self) -> None:
        """Test capture_evaluation_content can be set to True."""
        from holodeck.models.observability import TracingConfig

        config = TracingConfig(capture_evaluation_content=True)
        assert config.capture_evaluation_content is True

    def test_capture_evaluation_content_independent_of_capture_content(self) -> None:
        """Test capture_evaluation_content is independent of capture_content."""
        from holodeck.models.observability import TracingConfig

        # Only capture_content enabled
        config1 = TracingConfig(capture_content=True, capture_evaluation_content=False)
        assert config1.capture_content is True
        assert config1.capture_evaluation_content is False

        # Only capture_evaluation_content enabled
        config2 = TracingConfig(capture_content=False, capture_evaluation_content=True)
        assert config2.capture_content is False
        assert config2.capture_evaluation_content is True

        # Both enabled
        config3 = TracingConfig(capture_content=True, capture_evaluation_content=True)
        assert config3.capture_content is True
        assert config3.capture_evaluation_content is True

    def test_sample_rate_validation_lower_bound(self) -> None:
        """Test sample_rate rejects values below 0.0."""
        from holodeck.models.observability import TracingConfig

        with pytest.raises(ValidationError):
            TracingConfig(sample_rate=-0.1)

    def test_sample_rate_validation_upper_bound(self) -> None:
        """Test sample_rate rejects values above 1.0."""
        from holodeck.models.observability import TracingConfig

        with pytest.raises(ValidationError):
            TracingConfig(sample_rate=1.1)

    def test_sample_rate_valid_edge_cases(self) -> None:
        """Test sample_rate accepts edge values 0.0 and 1.0."""
        from holodeck.models.observability import TracingConfig

        config_zero = TracingConfig(sample_rate=0.0)
        assert config_zero.sample_rate == 0.0

        config_one = TracingConfig(sample_rate=1.0)
        assert config_one.sample_rate == 1.0

    def test_redaction_patterns_valid_regex(self) -> None:
        """Test redaction patterns with valid regex."""
        from holodeck.models.observability import TracingConfig

        # SSN pattern
        config = TracingConfig(redaction_patterns=[r"\d{3}-\d{2}-\d{4}"])
        assert len(config.redaction_patterns) == 1

        # Email pattern
        config2 = TracingConfig(
            redaction_patterns=[r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"]
        )
        assert len(config2.redaction_patterns) == 1

    def test_redaction_patterns_invalid_regex(self) -> None:
        """Test redaction patterns reject invalid regex."""
        from holodeck.models.observability import TracingConfig

        with pytest.raises(ValidationError) as exc_info:
            TracingConfig(redaction_patterns=["[invalid"])
        assert "Invalid regex pattern" in str(exc_info.value)

    def test_max_queue_size_minimum(self) -> None:
        """Test max_queue_size must be at least 1."""
        from holodeck.models.observability import TracingConfig

        with pytest.raises(ValidationError):
            TracingConfig(max_queue_size=0, max_export_batch_size=1)

        # Note: max_export_batch_size must be <= max_queue_size
        config = TracingConfig(max_queue_size=1, max_export_batch_size=1)
        assert config.max_queue_size == 1

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import TracingConfig

        with pytest.raises(ValidationError):
            TracingConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_max_queue_size_must_be_gte_max_export_batch_size(self) -> None:
        """Test max_queue_size must be >= max_export_batch_size."""
        from holodeck.models.observability import TracingConfig

        # Valid: queue_size > batch_size
        config = TracingConfig(max_queue_size=1024, max_export_batch_size=512)
        assert config.max_queue_size == 1024
        assert config.max_export_batch_size == 512

        # Valid: queue_size == batch_size
        config_equal = TracingConfig(max_queue_size=512, max_export_batch_size=512)
        assert config_equal.max_queue_size == 512
        assert config_equal.max_export_batch_size == 512

        # Invalid: queue_size < batch_size
        with pytest.raises(ValidationError) as exc_info:
            TracingConfig(max_queue_size=256, max_export_batch_size=512)
        assert "max_queue_size" in str(exc_info.value)
        assert "max_export_batch_size" in str(exc_info.value)


# T009: Unit tests for MetricsConfig
@pytest.mark.unit
class TestMetricsConfig:
    """Tests for MetricsConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import MetricsConfig

        config = MetricsConfig()
        assert config.enabled is True
        assert config.export_interval_ms == 5000
        assert config.include_semantic_kernel_metrics is True

    def test_export_interval_minimum_rejected(self) -> None:
        """Test export_interval_ms rejects values below 1000ms."""
        from holodeck.models.observability import MetricsConfig

        with pytest.raises(ValidationError):
            MetricsConfig(export_interval_ms=999)

    def test_export_interval_minimum_accepted(self) -> None:
        """Test export_interval_ms accepts minimum 1000ms."""
        from holodeck.models.observability import MetricsConfig

        config = MetricsConfig(export_interval_ms=1000)
        assert config.export_interval_ms == 1000

    def test_export_interval_custom_value(self) -> None:
        """Test export_interval_ms accepts custom values."""
        from holodeck.models.observability import MetricsConfig

        config = MetricsConfig(export_interval_ms=10000)
        assert config.export_interval_ms == 10000

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import MetricsConfig

        with pytest.raises(ValidationError):
            MetricsConfig(unknown_field="value")  # type: ignore[call-arg]


# T010: Unit tests for LogsConfig
@pytest.mark.unit
class TestLogsConfig:
    """Tests for LogsConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import LogLevel, LogsConfig

        config = LogsConfig()
        assert config.enabled is True
        assert config.level == LogLevel.INFO
        assert config.include_trace_context is True
        assert config.filter_namespaces == ["semantic_kernel"]

    def test_level_enum_validation(self) -> None:
        """Test level must be valid LogLevel enum."""
        from holodeck.models.observability import LogLevel, LogsConfig

        config = LogsConfig(level=LogLevel.DEBUG)
        assert config.level == LogLevel.DEBUG

        config2 = LogsConfig(level=LogLevel.ERROR)
        assert config2.level == LogLevel.ERROR

    def test_level_accepts_string(self) -> None:
        """Test level accepts string values that match enum."""
        from holodeck.models.observability import LogLevel, LogsConfig

        config = LogsConfig(level="WARNING")  # type: ignore[arg-type]
        assert config.level == LogLevel.WARNING

    def test_filter_namespaces_custom(self) -> None:
        """Test filter_namespaces accepts custom values."""
        from holodeck.models.observability import LogsConfig

        config = LogsConfig(filter_namespaces=["holodeck", "custom"])
        assert config.filter_namespaces == ["holodeck", "custom"]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import LogsConfig

        with pytest.raises(ValidationError):
            LogsConfig(unknown_field="value")  # type: ignore[call-arg]


# T011: Unit tests for ConsoleExporterConfig
@pytest.mark.unit
class TestConsoleExporterConfig:
    """Tests for ConsoleExporterConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import ConsoleExporterConfig

        config = ConsoleExporterConfig()
        assert config.enabled is True
        assert config.pretty_print is True
        assert config.include_timestamps is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from holodeck.models.observability import ConsoleExporterConfig

        config = ConsoleExporterConfig(
            enabled=False, pretty_print=False, include_timestamps=False
        )
        assert config.enabled is False
        assert config.pretty_print is False
        assert config.include_timestamps is False

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import ConsoleExporterConfig

        with pytest.raises(ValidationError):
            ConsoleExporterConfig(unknown_field="value")  # type: ignore[call-arg]


# T012: Unit tests for OTLPExporterConfig
@pytest.mark.unit
class TestOTLPExporterConfig:
    """Tests for OTLPExporterConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol

        config = OTLPExporterConfig()
        assert config.enabled is True
        assert config.endpoint == "http://localhost:4317"
        assert config.protocol == OTLPProtocol.GRPC
        assert config.headers == {}
        assert config.timeout_ms == 30000
        assert config.compression is None
        assert config.insecure is True

    def test_protocol_enum_grpc(self) -> None:
        """Test protocol with GRPC."""
        from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol

        config = OTLPExporterConfig(protocol=OTLPProtocol.GRPC)
        assert config.protocol == OTLPProtocol.GRPC

    def test_protocol_enum_http(self) -> None:
        """Test protocol with HTTP."""
        from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol

        config = OTLPExporterConfig(protocol=OTLPProtocol.HTTP)
        assert config.protocol == OTLPProtocol.HTTP

    def test_protocol_accepts_string(self) -> None:
        """Test protocol accepts string values that match enum."""
        from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol

        config = OTLPExporterConfig(protocol="http")  # type: ignore[arg-type]
        assert config.protocol == OTLPProtocol.HTTP

    def test_endpoint_custom(self) -> None:
        """Test endpoint with custom values."""
        from holodeck.models.observability import OTLPExporterConfig

        config = OTLPExporterConfig(endpoint="http://collector:4317")
        assert config.endpoint == "http://collector:4317"

        config2 = OTLPExporterConfig(endpoint="https://otel.example.com:443")
        assert config2.endpoint == "https://otel.example.com:443"

    def test_headers_custom(self) -> None:
        """Test headers with custom values."""
        from holodeck.models.observability import OTLPExporterConfig

        config = OTLPExporterConfig(
            headers={"authorization": "Bearer token", "x-custom": "value"}
        )
        assert config.headers == {"authorization": "Bearer token", "x-custom": "value"}

    def test_timeout_ms_minimum(self) -> None:
        """Test timeout_ms minimum of 1000ms."""
        from holodeck.models.observability import OTLPExporterConfig

        with pytest.raises(ValidationError):
            OTLPExporterConfig(timeout_ms=999)

        config = OTLPExporterConfig(timeout_ms=1000)
        assert config.timeout_ms == 1000

    def test_compression_values(self) -> None:
        """Test compression with various values."""
        from holodeck.models.observability import OTLPExporterConfig

        config_none = OTLPExporterConfig(compression=None)
        assert config_none.compression is None

        config_gzip = OTLPExporterConfig(compression="gzip")
        assert config_gzip.compression == "gzip"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import OTLPExporterConfig

        with pytest.raises(ValidationError):
            OTLPExporterConfig(unknown_field="value")  # type: ignore[call-arg]


# T013: Unit tests for PrometheusExporterConfig
@pytest.mark.unit
class TestPrometheusExporterConfig:
    """Tests for PrometheusExporterConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import PrometheusExporterConfig

        config = PrometheusExporterConfig()
        assert config.enabled is False
        assert config.port == 8889
        assert config.host == "0.0.0.0"  # noqa: S104
        assert config.path == "/metrics"

    def test_port_range_lower_bound_rejected(self) -> None:
        """Test port rejects values below 1024."""
        from holodeck.models.observability import PrometheusExporterConfig

        with pytest.raises(ValidationError):
            PrometheusExporterConfig(port=1023)

    def test_port_range_upper_bound_rejected(self) -> None:
        """Test port rejects values above 65535."""
        from holodeck.models.observability import PrometheusExporterConfig

        with pytest.raises(ValidationError):
            PrometheusExporterConfig(port=65536)

    def test_port_range_edge_cases_accepted(self) -> None:
        """Test port accepts edge values 1024 and 65535."""
        from holodeck.models.observability import PrometheusExporterConfig

        config_low = PrometheusExporterConfig(port=1024)
        assert config_low.port == 1024

        config_high = PrometheusExporterConfig(port=65535)
        assert config_high.port == 65535

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from holodeck.models.observability import PrometheusExporterConfig

        config = PrometheusExporterConfig(
            enabled=True, port=9090, host="127.0.0.1", path="/custom-metrics"
        )
        assert config.enabled is True
        assert config.port == 9090
        assert config.host == "127.0.0.1"
        assert config.path == "/custom-metrics"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import PrometheusExporterConfig

        with pytest.raises(ValidationError):
            PrometheusExporterConfig(unknown_field="value")  # type: ignore[call-arg]


# T014: Unit tests for AzureMonitorExporterConfig
@pytest.mark.unit
class TestAzureMonitorExporterConfig:
    """Tests for AzureMonitorExporterConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        config = AzureMonitorExporterConfig()
        assert config.enabled is False
        assert config.connection_string is None
        assert config.disable_offline_storage is False
        assert config.storage_directory is None

    def test_connection_string_required_when_enabled(self) -> None:
        """Test connection_string is required when enabled=True."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        with pytest.raises(ValidationError) as exc_info:
            AzureMonitorExporterConfig(enabled=True)
        assert "connection string is required" in str(exc_info.value).lower()

    def test_disabled_without_connection_string(self) -> None:
        """Test disabled config accepts None connection_string."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        config = AzureMonitorExporterConfig(enabled=False)
        assert config.enabled is False
        assert config.connection_string is None

    def test_enabled_with_connection_string(self) -> None:
        """Test enabled config with valid connection string."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        conn_str = "InstrumentationKey=xxx;IngestionEndpoint=https://example.com"
        config = AzureMonitorExporterConfig(enabled=True, connection_string=conn_str)
        assert config.enabled is True
        assert config.connection_string == conn_str

    def test_custom_storage_settings(self) -> None:
        """Test custom storage configuration."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        conn_str = "InstrumentationKey=xxx;IngestionEndpoint=https://example.com"
        config = AzureMonitorExporterConfig(
            enabled=True,
            connection_string=conn_str,
            disable_offline_storage=True,
            storage_directory="/tmp/azmon",  # noqa: S108
        )
        assert config.disable_offline_storage is True
        assert config.storage_directory == "/tmp/azmon"  # noqa: S108

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import AzureMonitorExporterConfig

        with pytest.raises(ValidationError):
            AzureMonitorExporterConfig(unknown_field="value")  # type: ignore[call-arg]


# T015: Unit tests for ExportersConfig
@pytest.mark.unit
class TestExportersConfig:
    """Tests for ExportersConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import ExportersConfig

        config = ExportersConfig()
        assert config.console is None
        assert config.otlp is None
        assert config.prometheus is None
        assert config.azure_monitor is None

    def test_get_enabled_exporters_empty(self) -> None:
        """Test get_enabled_exporters with no exporters."""
        from holodeck.models.observability import ExportersConfig

        config = ExportersConfig()
        assert config.get_enabled_exporters() == []

    def test_get_enabled_exporters_with_otlp(self) -> None:
        """Test get_enabled_exporters with OTLP enabled."""
        from holodeck.models.observability import ExportersConfig, OTLPExporterConfig

        config = ExportersConfig(otlp=OTLPExporterConfig(enabled=True))
        assert "otlp" in config.get_enabled_exporters()

    def test_get_enabled_exporters_with_console(self) -> None:
        """Test get_enabled_exporters with console enabled."""
        from holodeck.models.observability import ConsoleExporterConfig, ExportersConfig

        config = ExportersConfig(console=ConsoleExporterConfig(enabled=True))
        assert "console" in config.get_enabled_exporters()

    def test_get_enabled_exporters_with_prometheus(self) -> None:
        """Test get_enabled_exporters with Prometheus enabled."""
        from holodeck.models.observability import (
            ExportersConfig,
            PrometheusExporterConfig,
        )

        config = ExportersConfig(prometheus=PrometheusExporterConfig(enabled=True))
        assert "prometheus" in config.get_enabled_exporters()

    def test_get_enabled_exporters_multiple(self) -> None:
        """Test get_enabled_exporters with multiple exporters."""
        from holodeck.models.observability import (
            ConsoleExporterConfig,
            ExportersConfig,
            OTLPExporterConfig,
        )

        config = ExportersConfig(
            console=ConsoleExporterConfig(enabled=True),
            otlp=OTLPExporterConfig(enabled=True),
        )
        enabled = config.get_enabled_exporters()
        assert "console" in enabled
        assert "otlp" in enabled
        assert len(enabled) == 2

    def test_get_enabled_exporters_disabled(self) -> None:
        """Test get_enabled_exporters with disabled exporter."""
        from holodeck.models.observability import ExportersConfig, OTLPExporterConfig

        config = ExportersConfig(otlp=OTLPExporterConfig(enabled=False))
        assert config.get_enabled_exporters() == []

    def test_uses_console_as_default_no_exporters(self) -> None:
        """Test uses_console_as_default with no exporters."""
        from holodeck.models.observability import ExportersConfig

        config = ExportersConfig()
        assert config.uses_console_as_default() is True

    def test_uses_console_as_default_with_exporter(self) -> None:
        """Test uses_console_as_default with OTLP enabled."""
        from holodeck.models.observability import ExportersConfig, OTLPExporterConfig

        config = ExportersConfig(otlp=OTLPExporterConfig(enabled=True))
        assert config.uses_console_as_default() is False

    def test_uses_console_as_default_with_disabled_exporter(self) -> None:
        """Test uses_console_as_default with disabled exporter."""
        from holodeck.models.observability import ExportersConfig, OTLPExporterConfig

        config = ExportersConfig(otlp=OTLPExporterConfig(enabled=False))
        assert config.uses_console_as_default() is True

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import ExportersConfig

        with pytest.raises(ValidationError):
            ExportersConfig(unknown_field="value")  # type: ignore[call-arg]


# T016: Unit tests for ObservabilityConfig
@pytest.mark.unit
class TestObservabilityConfig:
    """Tests for ObservabilityConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from holodeck.models.observability import ObservabilityConfig

        config = ObservabilityConfig()
        assert config.enabled is False
        assert config.service_name is None
        assert config.resource_attributes == {}

    def test_service_name_optional(self) -> None:
        """Test service_name can be None (defaults to holodeck-{agent_name})."""
        from holodeck.models.observability import ObservabilityConfig

        config = ObservabilityConfig(enabled=True)
        assert config.service_name is None

    def test_service_name_override(self) -> None:
        """Test service_name can be explicitly set."""
        from holodeck.models.observability import ObservabilityConfig

        config = ObservabilityConfig(enabled=True, service_name="custom-service")
        assert config.service_name == "custom-service"

    def test_nested_configs_created(self) -> None:
        """Test nested configs have default values."""
        from holodeck.models.observability import (
            ExportersConfig,
            LogsConfig,
            MetricsConfig,
            ObservabilityConfig,
            TracingConfig,
        )

        config = ObservabilityConfig()
        assert isinstance(config.traces, TracingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.logs, LogsConfig)
        assert isinstance(config.exporters, ExportersConfig)

    def test_nested_configs_with_defaults(self) -> None:
        """Test nested configs have correct default values."""
        from holodeck.models.observability import LogLevel, ObservabilityConfig

        config = ObservabilityConfig()
        # TracingConfig defaults
        assert config.traces.enabled is True
        assert config.traces.sample_rate == 1.0
        # MetricsConfig defaults
        assert config.metrics.enabled is True
        assert config.metrics.export_interval_ms == 5000
        # LogsConfig defaults
        assert config.logs.enabled is True
        assert config.logs.level == LogLevel.INFO
        # ExportersConfig defaults
        assert config.exporters.console is None
        assert config.exporters.otlp is None

    def test_resource_attributes_custom(self) -> None:
        """Test resource_attributes with custom values."""
        from holodeck.models.observability import ObservabilityConfig

        config = ObservabilityConfig(
            enabled=True,
            resource_attributes={"environment": "production", "version": "1.0.0"},
        )
        assert config.resource_attributes == {
            "environment": "production",
            "version": "1.0.0",
        }

    def test_full_configuration(self) -> None:
        """Test full configuration with all fields."""
        from holodeck.models.observability import (
            ExportersConfig,
            LogLevel,
            LogsConfig,
            MetricsConfig,
            ObservabilityConfig,
            OTLPExporterConfig,
            TracingConfig,
        )

        config = ObservabilityConfig(
            enabled=True,
            service_name="my-service",
            traces=TracingConfig(sample_rate=0.5, capture_content=True),
            metrics=MetricsConfig(export_interval_ms=10000),
            logs=LogsConfig(level=LogLevel.DEBUG),
            exporters=ExportersConfig(otlp=OTLPExporterConfig(enabled=True)),
            resource_attributes={"team": "platform"},
        )
        assert config.enabled is True
        assert config.service_name == "my-service"
        assert config.traces.sample_rate == 0.5
        assert config.traces.capture_content is True
        assert config.metrics.export_interval_ms == 10000
        assert config.logs.level == LogLevel.DEBUG
        assert config.exporters.otlp is not None
        assert config.exporters.otlp.enabled is True
        assert config.resource_attributes == {"team": "platform"}

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        from holodeck.models.observability import ObservabilityConfig

        with pytest.raises(ValidationError):
            ObservabilityConfig(unknown_field="value")  # type: ignore[call-arg]
