"""Observability configuration models for HoloDeck.

Defines Pydantic models for OpenTelemetry observability configuration,
following the no-code-first principle with YAML-based configuration.
"""

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LogLevel(str, Enum):
    """Supported log levels for observability."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OTLPProtocol(str, Enum):
    """OTLP transport protocol."""

    GRPC = "grpc"
    HTTP = "http"


class TracingConfig(BaseModel):
    """Tracing-specific configuration.

    Controls distributed tracing settings including sampling rate,
    content capture, and redaction patterns for sensitive data.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable trace collection")
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 to 1.0)",
    )
    capture_content: bool = Field(
        default=False,
        description="Capture prompts and completions in spans (sensitive data)",
    )
    capture_evaluation_content: bool = Field(
        default=False,
        description=(
            "Capture evaluation inputs (input, actual_output, expected_output) "
            "in spans (sensitive data)"
        ),
    )
    redaction_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for redacting sensitive data",
    )
    max_queue_size: int = Field(
        default=2048,
        ge=1,
        description="Maximum spans in export queue",
    )
    max_export_batch_size: int = Field(
        default=512,
        ge=1,
        description="Maximum spans per export batch",
    )

    @field_validator("redaction_patterns")
    @classmethod
    def validate_redaction_patterns(cls, v: list[str]) -> list[str]:
        """Validate each pattern is valid regex."""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        return v

    @model_validator(mode="after")
    def validate_queue_size_batch_size(self) -> "TracingConfig":
        """Validate max_queue_size >= max_export_batch_size."""
        if self.max_queue_size < self.max_export_batch_size:
            raise ValueError(
                f"max_queue_size ({self.max_queue_size}) must be >= "
                f"max_export_batch_size ({self.max_export_batch_size})"
            )
        return self


class MetricsConfig(BaseModel):
    """Metrics-specific configuration.

    Controls metrics collection settings including export interval
    and Semantic Kernel metrics inclusion.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable metrics collection")
    export_interval_ms: int = Field(
        default=5000,
        ge=1000,
        description="Metrics export interval in milliseconds (minimum 1000ms)",
    )
    include_semantic_kernel_metrics: bool = Field(
        default=True,
        description="Include Semantic Kernel internal metrics",
    )


class LogsConfig(BaseModel):
    """Logging-specific configuration.

    Controls structured logging settings including log level,
    trace context inclusion, and namespace filtering.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable structured logging export")
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level to export",
    )
    include_trace_context: bool = Field(
        default=True,
        description="Include trace/span IDs in log records",
    )
    filter_namespaces: list[str] = Field(
        default_factory=lambda: ["semantic_kernel"],
        description="Logger namespaces to include",
    )


class ConsoleExporterConfig(BaseModel):
    """Console exporter configuration.

    Used as the default exporter when no other exporters are explicitly enabled.
    Outputs telemetry to stdout for development/debugging purposes.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable console exporter")
    pretty_print: bool = Field(default=True, description="Format for readability")
    include_timestamps: bool = Field(default=True, description="Include timestamps")


class OTLPExporterConfig(BaseModel):
    """OTLP exporter configuration.

    Configures export to OTLP collectors (Jaeger, Honeycomb, Datadog, etc.)
    via gRPC or HTTP protocol.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Enable OTLP exporter")
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint",
    )
    protocol: OTLPProtocol = Field(
        default=OTLPProtocol.GRPC,
        description="Transport protocol (grpc or http)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers for authentication",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        description="Export timeout in milliseconds",
    )
    compression: str | None = Field(
        default=None,
        description="Compression algorithm (gzip, deflate, or None)",
    )
    insecure: bool = Field(
        default=True,
        description="Use insecure connection (no TLS)",
    )


class PrometheusExporterConfig(BaseModel):
    """Prometheus exporter configuration.

    Exposes metrics via HTTP endpoint for Prometheus scraping.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable Prometheus endpoint")
    port: int = Field(
        default=8889,
        ge=1024,
        le=65535,
        description="Port for metrics endpoint (1024-65535)",
    )
    host: str = Field(
        default="0.0.0.0",  # noqa: S104  # nosec B104  # Intentional for Prometheus
        description="Host to bind metrics endpoint",
    )
    path: str = Field(
        default="/metrics",
        description="Path for metrics endpoint",
    )


class AzureMonitorExporterConfig(BaseModel):
    """Azure Monitor exporter configuration.

    Exports telemetry directly to Azure Monitor / Application Insights.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable Azure Monitor exporter")
    connection_string: str | None = Field(
        default=None,
        description="Application Insights connection string",
    )
    disable_offline_storage: bool = Field(
        default=False,
        description="Disable offline storage for retry",
    )
    storage_directory: str | None = Field(
        default=None,
        description="Custom directory for offline storage",
    )

    @model_validator(mode="after")
    def validate_connection_string_required(self) -> "AzureMonitorExporterConfig":
        """Ensure connection_string is provided when enabled."""
        if self.enabled and not self.connection_string:
            raise ValueError(
                "Azure Monitor connection string is required when enabled. "
                "Set via config or APPLICATIONINSIGHTS_CONNECTION_STRING env var."
            )
        return self


class ExportersConfig(BaseModel):
    """Container for exporter configurations.

    Note: If no exporters are explicitly enabled, console exporter is used
    as the default for development/debugging purposes.
    """

    model_config = ConfigDict(extra="forbid")

    console: ConsoleExporterConfig | None = Field(
        default=None,
        description="Console exporter (default if no other exporters)",
    )
    otlp: OTLPExporterConfig | None = Field(
        default=None,
        description="OTLP exporter configuration",
    )
    prometheus: PrometheusExporterConfig | None = Field(
        default=None,
        description="Prometheus exporter configuration",
    )
    azure_monitor: AzureMonitorExporterConfig | None = Field(
        default=None,
        description="Azure Monitor exporter configuration",
    )

    def get_enabled_exporters(self) -> list[str]:
        """Return list of enabled exporter names."""
        enabled = []
        if self.console and self.console.enabled:
            enabled.append("console")
        if self.otlp and self.otlp.enabled:
            enabled.append("otlp")
        if self.prometheus and self.prometheus.enabled:
            enabled.append("prometheus")
        if self.azure_monitor and self.azure_monitor.enabled:
            enabled.append("azure_monitor")
        return enabled

    def uses_console_as_default(self) -> bool:
        """Check if console exporter will be used as default.

        Returns True if no other exporters are explicitly enabled,
        indicating console exporter should be used as fallback.
        """
        return len(self.get_enabled_exporters()) == 0


class ObservabilityConfig(BaseModel):
    """Top-level observability configuration.

    Root configuration for OpenTelemetry observability features.
    Service name defaults to 'holodeck-{agent.name}' if not explicitly set.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable observability features")
    service_name: str | None = Field(
        default=None,
        description="Override service name (defaults to 'holodeck-{agent.name}')",
    )
    traces: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Tracing configuration",
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration",
    )
    logs: LogsConfig = Field(
        default_factory=LogsConfig,
        description="Logging configuration",
    )
    exporters: ExportersConfig = Field(
        default_factory=ExportersConfig,
        description="Exporter configurations",
    )
    resource_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Additional OpenTelemetry resource attributes",
    )
