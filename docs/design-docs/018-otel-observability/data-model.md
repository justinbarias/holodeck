# Data Model: OpenTelemetry Observability

**Feature**: 018-otel-observability
**Date**: 2026-01-04

## Overview

This document defines the Pydantic models for observability configuration in HoloDeck. All models follow HoloDeck's no-code-first principle, enabling YAML-based configuration.

---

## Entity Relationship Diagram

```text
AgentConfig (existing)
    │
    └── observability: ObservabilityConfig
            │
            ├── service_name: str | None  # Optional override, defaults to "holodeck-{agent.name}"
            ├── enabled: bool
            │
            ├── traces: TracingConfig
            │       ├── enabled: bool
            │       ├── sample_rate: float
            │       ├── capture_content: bool
            │       └── redaction_patterns: list[str]
            │
            ├── metrics: MetricsConfig
            │       ├── enabled: bool
            │       └── export_interval_ms: int
            │
            ├── logs: LogsConfig
            │       ├── enabled: bool
            │       ├── level: LogLevel
            │       └── include_trace_context: bool
            │
            └── exporters: ExportersConfig
                    ├── console: ConsoleExporterConfig | None (default if none enabled)
                    ├── otlp: OTLPExporterConfig | None
                    ├── prometheus: PrometheusExporterConfig | None
                    └── azure_monitor: AzureMonitorExporterConfig | None
```

---

## Core Models

### ObservabilityConfig

Top-level configuration for observability features.

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional

class ObservabilityConfig(BaseModel):
    """Top-level observability configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable observability features"
    )

    service_name: str | None = Field(
        default=None,
        description="Override service name (defaults to 'holodeck-{agent.name}' if not specified)"
    )

    traces: "TracingConfig" = Field(
        default_factory=lambda: TracingConfig(),
        description="Tracing configuration"
    )

    metrics: "MetricsConfig" = Field(
        default_factory=lambda: MetricsConfig(),
        description="Metrics configuration"
    )

    logs: "LogsConfig" = Field(
        default_factory=lambda: LogsConfig(),
        description="Logging configuration"
    )

    exporters: "ExportersConfig" = Field(
        default_factory=lambda: ExportersConfig(),
        description="Exporter configurations"
    )

    resource_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Additional OpenTelemetry resource attributes"
    )
```

**Validation Rules**:
- `service_name`: If provided, must be non-empty string; if None, defaults to `"holodeck-{agent.name}"` at runtime
- If `enabled=True` and no exporters configured, emit warning

---

### TracingConfig

Configuration for distributed tracing.

```python
class TracingConfig(BaseModel):
    """Tracing-specific configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable trace collection"
    )

    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 to 1.0)"
    )

    capture_content: bool = Field(
        default=False,
        description="Capture prompts and completions in spans (sensitive data)"
    )

    redaction_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for redacting sensitive data in captured content"
    )

    max_queue_size: int = Field(
        default=2048,
        ge=1,
        description="Maximum spans in export queue"
    )

    max_export_batch_size: int = Field(
        default=512,
        ge=1,
        description="Maximum spans per export batch"
    )
```

**Validation Rules**:
- `sample_rate`: Must be between 0.0 and 1.0
- `redaction_patterns`: Each pattern must be valid regex
- `max_queue_size` >= `max_export_batch_size`

---

### MetricsConfig

Configuration for metrics collection.

```python
class MetricsConfig(BaseModel):
    """Metrics-specific configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )

    export_interval_ms: int = Field(
        default=5000,
        ge=1000,
        description="Metrics export interval in milliseconds"
    )

    include_semantic_kernel_metrics: bool = Field(
        default=True,
        description="Include Semantic Kernel internal metrics"
    )
```

**Validation Rules**:
- `export_interval_ms`: Minimum 1000ms to prevent excessive exports

---

### LogsConfig

Configuration for structured logging with trace context.

```python
from enum import Enum

class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogsConfig(BaseModel):
    """Logging-specific configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable structured logging export"
    )

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level to export"
    )

    include_trace_context: bool = Field(
        default=True,
        description="Include trace/span IDs in log records"
    )

    filter_namespaces: list[str] = Field(
        default_factory=lambda: ["semantic_kernel"],
        description="Logger namespaces to include"
    )
```

**Validation Rules**:
- `level`: Must be valid LogLevel enum value

---

## Exporter Models

### ExportersConfig

Container for all exporter configurations.

```python
class ExportersConfig(BaseModel):
    """Container for exporter configurations.

    Note: If no exporters are explicitly enabled, console exporter is used
    as the default for development/debugging purposes.
    """

    console: Optional["ConsoleExporterConfig"] = Field(
        default=None,
        description="Console exporter configuration (default if no other exporters)"
    )

    otlp: Optional["OTLPExporterConfig"] = Field(
        default=None,
        description="OTLP exporter configuration"
    )

    prometheus: Optional["PrometheusExporterConfig"] = Field(
        default=None,
        description="Prometheus exporter configuration"
    )

    azure_monitor: Optional["AzureMonitorExporterConfig"] = Field(
        default=None,
        description="Azure Monitor exporter configuration"
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
        """Check if console exporter will be used as default (no explicit exporters)."""
        return len(self.get_enabled_exporters()) == 0
```

---

### ConsoleExporterConfig

Console exporter for development and debugging. Used as default when no other exporters are configured.

```python
class ConsoleExporterConfig(BaseModel):
    """Console exporter configuration.

    Used as the default exporter when no other exporters are explicitly enabled.
    Outputs telemetry to stdout for development/debugging purposes.
    """

    enabled: bool = Field(
        default=True,
        description="Enable console exporter"
    )

    pretty_print: bool = Field(
        default=True,
        description="Format output for human readability"
    )

    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in console output"
    )
```

**Validation Rules**:
- No special validation required
- When console exporter is active (explicitly or as default), the default HoloDeck logger's console handlers are suppressed to prevent double logging

**Default Behavior**:
- If `observability.enabled=True` and no exporters are explicitly configured, the console exporter is automatically enabled
- This provides immediate feedback for development without requiring external infrastructure

---

### OTLPExporterConfig

OTLP exporter for traces, metrics, and logs.

```python
from enum import Enum

class OTLPProtocol(str, Enum):
    """OTLP transport protocol."""
    GRPC = "grpc"
    HTTP = "http"

class OTLPExporterConfig(BaseModel):
    """OTLP exporter configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable OTLP exporter"
    )

    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint"
    )

    protocol: OTLPProtocol = Field(
        default=OTLPProtocol.GRPC,
        description="Transport protocol (grpc or http)"
    )

    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers for authentication (supports env var substitution)"
    )

    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        description="Export timeout in milliseconds"
    )

    compression: Optional[str] = Field(
        default=None,
        description="Compression algorithm (gzip, deflate, or None)"
    )

    insecure: bool = Field(
        default=True,
        description="Use insecure connection (no TLS)"
    )

    @validator("endpoint")
    def validate_endpoint(cls, v: str, values: dict) -> str:
        """Adjust default port based on protocol."""
        if v == "http://localhost:4317" and values.get("protocol") == OTLPProtocol.HTTP:
            return "http://localhost:4318"
        return v
```

**Validation Rules**:
- `endpoint`: Must be valid URL
- `headers`: Values support `${ENV_VAR}` substitution
- `compression`: Must be `gzip`, `deflate`, or `None`

---

### PrometheusExporterConfig

Prometheus metrics endpoint configuration.

```python
class PrometheusExporterConfig(BaseModel):
    """Prometheus exporter configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint"
    )

    port: int = Field(
        default=8889,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics endpoint"
    )

    host: str = Field(
        default="0.0.0.0",
        description="Host to bind metrics endpoint"
    )

    path: str = Field(
        default="/metrics",
        description="Path for metrics endpoint"
    )
```

**Validation Rules**:
- `port`: Must be valid port number (1024-65535 for non-root)
- `host`: Must be valid IP address or hostname

---

### AzureMonitorExporterConfig

Azure Monitor / Application Insights exporter.

```python
class AzureMonitorExporterConfig(BaseModel):
    """Azure Monitor exporter configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable Azure Monitor exporter"
    )

    connection_string: Optional[str] = Field(
        default=None,
        description="Application Insights connection string (supports env var substitution)"
    )

    disable_offline_storage: bool = Field(
        default=False,
        description="Disable offline storage for retry"
    )

    storage_directory: Optional[str] = Field(
        default=None,
        description="Custom directory for offline storage"
    )

    @field_validator("connection_string", mode="before")
    @classmethod
    def resolve_connection_string(cls, v: Optional[str]) -> Optional[str]:
        """Resolve from environment if not provided."""
        if v is None:
            import os
            return os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
        return v

    @model_validator(mode="after")
    def validate_connection_string_required(self) -> "AzureMonitorExporterConfig":
        """Ensure connection_string is provided when enabled."""
        if self.enabled and not self.connection_string:
            raise ValueError(
                "Azure Monitor connection string is required when enabled. "
                "Set via config or APPLICATIONINSIGHTS_CONNECTION_STRING env var."
            )
        return self
```

**Validation Rules**:
- `connection_string`: Must be valid Application Insights connection string format
- If `enabled=True` and `connection_string` is None, raise validation error (enforced by `model_validator`)

---

## Integration with AgentConfig

The `ObservabilityConfig` integrates into the existing `AgentConfig` model:

```python
# In src/holodeck/models/agent.py

class AgentConfig(BaseModel):
    """Agent configuration model."""

    name: str
    description: Optional[str] = None
    model: LLMProviderConfig
    instructions: InstructionsConfig
    tools: list[ToolUnion] = []
    evaluations: Optional[EvaluationConfig] = None
    test_cases: list[TestCaseModel] = []

    # NEW: Observability configuration
    observability: Optional[ObservabilityConfig] = Field(
        default=None,
        description="Observability configuration for telemetry"
    )
```

---

## YAML Configuration Example

### Basic (service name auto-derived from agent name)

```yaml
name: customer-support-agent
description: Handles customer inquiries

model:
  provider: openai
  name: gpt-4o-mini

instructions:
  file: instructions/system-prompt.md

observability:
  enabled: true
  # service_name defaults to "holodeck-customer-support-agent"

  traces:
    enabled: true
    sample_rate: 1.0
    capture_content: false
    redaction_patterns:
      - '\b\d{3}-\d{2}-\d{4}\b'  # SSN pattern

  metrics:
    enabled: true
    export_interval_ms: 5000

  logs:
    enabled: true
    level: INFO
    include_trace_context: true

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc
      headers:
        authorization: Bearer ${OTEL_API_KEY}

    prometheus:
      enabled: true
      port: 8889

    azure_monitor:
      enabled: false
      connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

### With custom service name override

```yaml
name: customer-support-agent
description: Handles customer inquiries

model:
  provider: openai
  name: gpt-4o-mini

instructions:
  file: instructions/system-prompt.md

observability:
  enabled: true
  service_name: prod-cs-agent  # Custom override (instead of "holodeck-customer-support-agent")

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
```

---

## State Transitions

Observability has simple lifecycle states:

```text
┌─────────────┐
│  DISABLED   │  (observability.enabled = false)
└─────────────┘
       │
       │ enable
       ▼
┌─────────────┐
│ INITIALIZING│  (providers being set up)
└─────────────┘
       │
       │ success
       ▼
┌─────────────┐
│   ACTIVE    │  (telemetry being collected/exported)
└─────────────┘
       │
       │ shutdown
       ▼
┌─────────────┐
│  SHUTDOWN   │  (providers flushed and closed)
└─────────────┘
```

---

## Validation Summary

| Field | Validation | Error Message |
|-------|------------|---------------|
| `sample_rate` | 0.0 <= value <= 1.0 | "Sample rate must be between 0.0 and 1.0" |
| `redaction_patterns` | Valid regex | "Invalid regex pattern: {pattern}" |
| `export_interval_ms` | >= 1000 | "Export interval must be at least 1000ms" |
| `port` | 1024-65535 | "Port must be between 1024 and 65535" |
| `connection_string` | Required if Azure Monitor enabled | "Azure Monitor connection string is required" |
| `endpoint` | Valid URL | "Invalid OTLP endpoint URL" |
