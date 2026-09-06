# Research: OpenTelemetry Observability with Semantic Conventions

**Feature**: 018-otel-observability
**Date**: 2026-01-04
**Status**: Complete

## Research Summary

This document consolidates research findings for implementing OpenTelemetry observability in HoloDeck, leveraging Semantic Kernel's native instrumentation.

---

## 1. Semantic Kernel OpenTelemetry Integration

### Decision: Leverage Semantic Kernel's Native Instrumentation

**Rationale**: Semantic Kernel Python SDK already generates comprehensive telemetry (traces, metrics, logs) for function execution and model invocation. No need to implement custom instrumentation.

**Alternatives Considered**:
- Custom instrumentation from scratch: Rejected - duplicates existing capability
- Third-party instrumentation library: Rejected - SK's native support is comprehensive

### Key Findings from Semantic Kernel Telemetry Demo

**Source**: [microsoft/semantic-kernel/python/samples/demos/telemetry](https://github.com/microsoft/semantic-kernel/tree/dc7c1c048488a8b611b5382d0d7efe05608a9fa0/python/samples/demos/telemetry)

#### Environment Variables

| Variable | Purpose |
|----------|---------|
| `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS` | Enable GenAI diagnostics spans |
| `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE` | Enable prompt/completion capture |

#### Telemetry Scenarios

1. **AI Service**: Direct model calls generate spans with operation metadata
2. **Kernel Functions**: Wrapped functions create parent spans containing model invocation spans
3. **Auto Function Invocation**: Loop iteration spans encompass nested function calls

#### Code Pattern: Provider Setup

```python
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Shared resource for all providers
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: "holodeck-agent"
})
```

---

## 2. OTLP Exporter Configuration

### Decision: Support Both gRPC and HTTP/Protobuf Protocols

**Rationale**: gRPC is default (port 4317), HTTP/protobuf (port 4318) needed for environments where gRPC is blocked.

**Alternatives Considered**:
- gRPC only: Rejected - limits deployment flexibility
- HTTP/JSON: Rejected - less efficient than protobuf

### Implementation Pattern

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# gRPC endpoint (default)
exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

# HTTP endpoint alternative
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
http_exporter = HTTPSpanExporter(endpoint="http://localhost:4318/v1/traces")
```

### Required Packages

```text
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp-proto-grpc>=1.20.0
opentelemetry-exporter-otlp-proto-http>=1.20.0  # For HTTP protocol option
```

---

## 3. Prometheus Exporter Configuration

### Decision: Use PrometheusMetricReader with HTTP Server

**Rationale**: Standard pattern for Prometheus scraping; metrics exposed on configurable port.

**Alternatives Considered**:
- Push gateway: Rejected - Prometheus is pull-based by design
- OTLP-to-Prometheus bridge: Rejected - adds unnecessary complexity

### Implementation Pattern

```python
from prometheus_client import start_http_server
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider

# Start HTTP server for Prometheus scraping
start_http_server(port=8889, addr="0.0.0.0")

# Configure metric reader
reader = PrometheusMetricReader()
meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
```

### Required Packages

```text
opentelemetry-exporter-prometheus>=0.60b1
prometheus-client>=0.17.0
```

### Limitations

- **No multiprocessing support**: Prometheus exporter doesn't work in multiprocessing environments
- **Metrics only**: Prometheus is metrics-focused; traces require separate OTLP exporter

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OTEL_EXPORTER_PROMETHEUS_HOST` | localhost | Server binding address |
| `OTEL_EXPORTER_PROMETHEUS_PORT` | 9464 | Server binding port |

---

## 4. Azure Monitor Exporter Configuration

### Decision: Use azure-monitor-opentelemetry-exporter Package

**Rationale**: Official Microsoft package with full trace/metric/log support for Application Insights.

**Alternatives Considered**:
- OpenCensus Azure exporter: Rejected - deprecated in favor of OpenTelemetry
- OTLP to Azure: Rejected - direct exporter is simpler

### Implementation Pattern

```python
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorTraceExporter,
    AzureMonitorMetricExporter,
    AzureMonitorLogExporter,
)

# Connection string from environment
connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")

# Trace exporter
trace_exporter = AzureMonitorTraceExporter(connection_string=connection_string)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))

# Metric exporter
metric_exporter = AzureMonitorMetricExporter(connection_string=connection_string)
metric_reader = PeriodicExportingMetricReader(metric_exporter)

# Log exporter (experimental)
log_exporter = AzureMonitorLogExporter(connection_string=connection_string)
```

### Required Packages

```text
azure-monitor-opentelemetry-exporter>=1.0.0b24
```

### Configuration Options

| Parameter | Purpose |
|-----------|---------|
| `connection_string` | Application Insights connection string |
| `disable_offline_storage` | Disable retry file storage (default: False) |
| `storage_directory` | Custom directory for retry files |
| `credential` | Azure AD token credential |

---

## 5. GenAI Semantic Conventions

### Decision: Follow OpenTelemetry GenAI Semantic Conventions

**Rationale**: Industry standard ensures compatibility with observability tools that understand GenAI telemetry.

**Alternatives Considered**:
- Custom attribute naming: Rejected - breaks tool compatibility
- LangSmith conventions: Rejected - proprietary, not open standard

### Core Attributes

#### Required Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.operation.name` | Operation type | `chat`, `embeddings`, `text_completion` |
| `gen_ai.system` | Provider identifier | `openai`, `anthropic`, `azure_openai` |

#### Conditionally Required

| Attribute | Condition | Description |
|-----------|-----------|-------------|
| `gen_ai.request.model` | When available | Model name requested |
| `error.type` | On failure | Error classification |

#### Recommended Attributes

| Attribute | Description |
|-----------|-------------|
| `gen_ai.request.max_tokens` | Max tokens requested |
| `gen_ai.request.temperature` | Sampling temperature |
| `gen_ai.request.top_p` | Nucleus sampling parameter |
| `gen_ai.usage.input_tokens` | Prompt tokens consumed |
| `gen_ai.usage.output_tokens` | Completion tokens generated |
| `gen_ai.response.id` | Unique completion ID |
| `gen_ai.response.model` | Actual model used |
| `gen_ai.response.finish_reasons` | Why generation stopped |

#### Optional (Sensitive - Opt-In)

| Attribute | Description |
|-----------|-------------|
| `gen_ai.prompt.0.role` | Message role (system/user/assistant) |
| `gen_ai.prompt.0.content` | Message content |
| `gen_ai.completion.0.role` | Response role |
| `gen_ai.completion.0.content` | Response content |

### Span Naming Convention

```text
{gen_ai.operation.name} {gen_ai.request.model}
```

Examples:
- `chat gpt-4o`
- `embeddings text-embedding-3-small`
- `execute_tool search_knowledge_base`

---

## 6. Provider Initialization Order

### Decision: Initialize in Order - Logging, Tracing, Metrics

**Rationale**: Logging must be first per OpenTelemetry Python documentation to capture all events.

### Implementation Pattern

```python
def initialize_observability(config: ObservabilityConfig) -> None:
    """Initialize all telemetry providers in correct order."""
    resource = create_resource(config)

    # 1. Logging (must be first)
    set_up_logging(config, resource)

    # 2. Tracing
    set_up_tracing(config, resource)

    # 3. Metrics
    set_up_metrics(config, resource)
```

---

## 7. Batch Processing and Buffer Configuration

### Decision: Use BatchSpanProcessor with Configurable Limits

**Rationale**: Batch processing reduces overhead; bounded buffer prevents memory exhaustion per clarification session.

### Configuration

| Parameter | Default | HoloDeck Setting |
|-----------|---------|------------------|
| `max_queue_size` | 2048 | 2048 spans |
| `max_export_batch_size` | 512 | 512 spans |
| `export_timeout_millis` | 30000 | 30000ms |
| `schedule_delay_millis` | 5000 | 5000ms |

### Memory Limit Implementation

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,  # ~5MB estimated
    max_export_batch_size=512,
)
```

---

## 8. Sensitive Data Handling

### Decision: Opt-In Content Capture with Redaction Support

**Rationale**: Privacy by default; users explicitly enable content capture for debugging.

### Implementation Pattern

```python
import os
import re

# Enable via environment variable (Semantic Kernel pattern)
if config.traces.capture_content:
    os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"] = "true"

# Redaction processor
class RedactingSpanProcessor(SpanProcessor):
    def __init__(self, patterns: list[str]):
        self.patterns = [re.compile(p) for p in patterns]

    def on_end(self, span: ReadableSpan) -> None:
        for attr in ["gen_ai.prompt.0.content", "gen_ai.completion.0.content"]:
            if attr in span.attributes:
                value = span.attributes[attr]
                for pattern in self.patterns:
                    value = pattern.sub("[REDACTED]", value)
                # Note: Span attributes are immutable; need custom approach
```

---

## 9. Multiple Exporter Support

### Decision: Configure Exporters as List, All Receive Same Telemetry

**Rationale**: Users may need both local debugging (OTLP) and production monitoring (Azure Monitor).

### Implementation Pattern

```python
def configure_exporters(config: ObservabilityConfig) -> list[SpanExporter]:
    """Configure all enabled exporters."""
    exporters = []

    if config.otlp and config.otlp.enabled:
        exporters.append(create_otlp_exporter(config.otlp))

    if config.azure_monitor and config.azure_monitor.enabled:
        exporters.append(create_azure_monitor_exporter(config.azure_monitor))

    return exporters

# Add all exporters to provider
for exporter in exporters:
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
```

---

## 10. Console Exporter as Default

### Decision: Use Console Exporter When No Exporters Configured

**Rationale**: Provides immediate feedback for development without requiring external infrastructure. Follows Semantic Kernel's pattern.

**Alternatives Considered**:
- Warning only (no output): Rejected - poor developer experience
- Require explicit exporter: Rejected - increases setup friction

### Implementation Pattern

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
from opentelemetry.sdk._logs.export import ConsoleLogExporter

def configure_exporters(config: ObservabilityConfig) -> list[SpanExporter]:
    """Configure exporters, defaulting to console if none specified."""
    exporters = []

    if config.exporters.otlp and config.exporters.otlp.enabled:
        exporters.append(create_otlp_exporter(config.exporters.otlp))

    if config.exporters.prometheus and config.exporters.prometheus.enabled:
        # Prometheus handled separately (metric reader, not exporter)
        pass

    if config.exporters.azure_monitor and config.exporters.azure_monitor.enabled:
        exporters.append(create_azure_monitor_exporter(config.exporters.azure_monitor))

    # Default to console if no exporters configured
    if not exporters and not (config.exporters.prometheus and config.exporters.prometheus.enabled):
        exporters.append(ConsoleSpanExporter())

    return exporters
```

---

## 11. Double Logging Prevention

### Decision: Suppress Default Logger When OTel Console Exporter Active

**Rationale**: Prevents duplicate log entries when both HoloDeck's default logger and OpenTelemetry's console exporter write to stdout.

**Alternatives Considered**:
- Let users handle it: Rejected - confusing default behavior
- Always use OTel logging: Rejected - may break existing non-observability users

### Implementation Pattern

```python
import logging

def configure_logging(config: ObservabilityConfig) -> None:
    """Configure logging, preventing duplicates with console exporter."""
    holodeck_logger = logging.getLogger("holodeck")

    # Check if console exporter will be used
    using_console_exporter = is_console_exporter_active(config)

    if using_console_exporter:
        # Remove default console handlers to prevent double logging
        for handler in holodeck_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream in (sys.stdout, sys.stderr):
                    holodeck_logger.removeHandler(handler)

    # Set up OTel logging handler
    if config.logs.enabled:
        otel_handler = LoggingHandler()
        otel_handler.addFilter(logging.Filter("holodeck"))
        otel_handler.addFilter(logging.Filter("semantic_kernel"))
        holodeck_logger.addHandler(otel_handler)

def is_console_exporter_active(config: ObservabilityConfig) -> bool:
    """Check if console exporter will be used (explicitly or as default)."""
    has_explicit_exporter = any([
        config.exporters.otlp and config.exporters.otlp.enabled,
        config.exporters.prometheus and config.exporters.prometheus.enabled,
        config.exporters.azure_monitor and config.exporters.azure_monitor.enabled,
    ])
    return not has_explicit_exporter
```

### Behavior Matrix

| Scenario | Default Logger | OTel Console | Result |
|----------|---------------|--------------|--------|
| No observability | Active | N/A | Normal HoloDeck logging |
| Observability + OTLP | Active | Inactive | Normal logging + OTel to OTLP |
| Observability + no exporter | Suppressed | Active | OTel console only |
| Observability + console explicit | Suppressed | Active | OTel console only |

---

## 12. YAML Configuration Schema

### Decision: Nested Structure Under `observability` Key

**Rationale**: Consistent with HoloDeck's YAML-first approach; clear hierarchy for exporters.

### Example Configuration

```yaml
observability:
  enabled: true
  service_name: my-agent

  traces:
    enabled: true
    sample_rate: 1.0
    capture_content: false
    redaction_patterns:
      - '\b\d{3}-\d{2}-\d{4}\b'  # SSN
      - '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email

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
      protocol: grpc  # or http
      headers:
        api-key: ${OTEL_API_KEY}

    prometheus:
      enabled: false
      port: 8889
      host: 0.0.0.0

    azure_monitor:
      enabled: false
      connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

---

## References

1. [Semantic Kernel Telemetry Demo](https://github.com/microsoft/semantic-kernel/tree/dc7c1c048488a8b611b5382d0d7efe05608a9fa0/python/samples/demos/telemetry)
2. [Microsoft Learn: Telemetry with Aspire Dashboard](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/telemetry-with-aspire-dashboard?tabs=Powershell&pivots=programming-language-python)
3. [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/)
4. [OpenTelemetry GenAI Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
5. [Azure Monitor OpenTelemetry Exporter](https://pypi.org/project/azure-monitor-opentelemetry-exporter/)
6. [OpenTelemetry Prometheus Exporter](https://pypi.org/project/opentelemetry-exporter-prometheus/)
