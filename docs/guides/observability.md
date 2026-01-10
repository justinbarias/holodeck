# Observability Guide

This guide explains HoloDeck's OpenTelemetry-based observability for tracing, metrics, and logging.

## Overview

HoloDeck provides built-in observability using OpenTelemetry, following the **GenAI semantic conventions** for AI/LLM instrumentation. All configuration is done through YAML—no code required.

Key features:

- **Distributed Tracing** - Track requests across LLM calls and tool executions
- **GenAI Attributes** - Token usage, model info, and completion details
- **Multiple Exporters** - Console, OTLP (gRPC/HTTP), Prometheus, Azure Monitor
- **Sensitive Data Control** - Capture or redact prompts/completions
- **Zero-Code Setup** - Configure entirely in agent.yaml

## Quick Start

Add observability to your agent configuration:

```yaml
# agent.yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o

observability:
  enabled: true
  # Console output by default - great for development
```

Run with observability enabled:

```bash
holodeck chat agent.yaml
# or
holodeck serve agent.yaml
```

You'll see trace and span information in the console output.

### Send to OTLP Endpoint

```yaml
observability:
  enabled: true
  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc
      insecure: true
```

---

## Configuration Reference

### Full Schema

```yaml
observability:
  enabled: true                    # Enable/disable observability
  service_name: "custom-service"   # Optional override (default: "holodeck-{agent.name}")

  traces:
    enabled: true                  # Trace collection
    sample_rate: 1.0               # 0.0 to 1.0 (default: 100% sampling)
    capture_content: false         # Capture prompts/completions (default: false)
    redaction_patterns:            # Regex patterns for sensitive data
      - '\b\d{3}-\d{2}-\d{4}\b'   # Example: SSN pattern
    max_queue_size: 2048           # Max spans in buffer
    max_export_batch_size: 512     # Spans per batch

  metrics:
    enabled: true                  # Metrics collection
    export_interval_ms: 5000       # Export every 5 seconds
    include_semantic_kernel_metrics: true

  logs:
    enabled: true                  # Structured logging
    level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    include_trace_context: true    # Include trace/span IDs
    filter_namespaces:
      - semantic_kernel            # Which loggers to capture

  resource_attributes:             # Custom OTel resource attributes
    environment: production
    version: 1.0

  exporters:
    console:
      enabled: true                # Console output (default)
      pretty_print: true
      include_timestamps: true

    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc               # grpc or http
      headers:
        authorization: "Bearer ${OTEL_API_KEY}"
      timeout_ms: 30000
      compression: gzip
      insecure: true

    prometheus:
      enabled: false
      port: 8889
      host: 0.0.0.0
      path: /metrics

    azure_monitor:
      enabled: false
      connection_string: "${APPLICATIONINSIGHTS_CONNECTION_STRING}"
```

### Service Name

By default, the service name is `holodeck-{agent.name}`:

- Agent named `research` → service name `holodeck-research`

Override with `service_name`:

```yaml
observability:
  enabled: true
  service_name: "my-custom-service"
```

---

## Traces Configuration

### Sample Rate

Control what percentage of requests are traced:

```yaml
traces:
  sample_rate: 1.0    # 100% sampling (default, good for development)
  sample_rate: 0.1    # 10% sampling (production with high traffic)
  sample_rate: 0.0    # Disable tracing
```

### Capturing Prompts and Completions

By default, prompts and completions are NOT captured (privacy). Enable for debugging:

```yaml
traces:
  capture_content: true  # Captures full prompt/completion in span events
```

### Redacting Sensitive Data

Redact patterns before export:

```yaml
traces:
  capture_content: true
  redaction_patterns:
    - '\b\d{3}-\d{2}-\d{4}\b'     # SSN
    - '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
    - '\b\d{16}\b'                 # Credit card (16 digits)
```

### Buffer Settings

Configure span buffering for batch export:

```yaml
traces:
  max_queue_size: 2048         # Max spans in memory
  max_export_batch_size: 512   # Spans per export batch
```

---

## GenAI Semantic Conventions

HoloDeck uses Semantic Kernel's native OpenTelemetry instrumentation, which follows [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

### Captured Attributes

Every LLM invocation captures:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.system` | Provider name | `openai`, `anthropic` |
| `gen_ai.request.model` | Model identifier | `gpt-4o`, `claude-3-sonnet` |
| `gen_ai.request.temperature` | Temperature setting | `0.7` |
| `gen_ai.usage.input_tokens` | Prompt token count | `150` |
| `gen_ai.usage.output_tokens` | Completion token count | `75` |
| `gen_ai.usage.total_tokens` | Total tokens | `225` |
| `gen_ai.response.finish_reason` | Why generation stopped | `stop`, `length` |
| `gen_ai.response.id` | Provider's completion ID | `chatcmpl-...` |

### Span Events (when capture_content enabled)

- `gen_ai.content.prompt` - Full prompt text
- `gen_ai.content.completion` - Full completion text

---

## Evaluation Tracing (DeepEval)

HoloDeck creates OpenTelemetry spans for DeepEval metric evaluations during test runs. Enable `capture_evaluation_content` to capture evaluation inputs/outputs:

```yaml
observability:
  enabled: true
  traces:
    capture_evaluation_content: true  # Capture inputs, outputs, reasoning
```

### Captured Span Attributes

Every DeepEval evaluation creates a span named `holodeck.evaluation.{metric_name}` with these attributes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `evaluation.metric.name` | Metric identifier | `geval`, `faithfulness` |
| `evaluation.threshold` | Pass/fail threshold | `0.7` |
| `evaluation.model.provider` | Evaluation LLM provider | `openai`, `ollama` |
| `evaluation.model.name` | Evaluation model name | `gpt-4o`, `llama3.2` |
| `evaluation.score` | Evaluation score (0.0-1.0) | `0.85` |
| `evaluation.passed` | Whether score met threshold | `true` |
| `evaluation.duration_ms` | Evaluation duration | `1523` |

### Content Attributes (when capture_evaluation_content enabled)

| Attribute | Description | Max Length |
|-----------|-------------|------------|
| `evaluation.input` | User query being evaluated | 1000 chars |
| `evaluation.actual_output` | Agent response being evaluated | 1000 chars |
| `evaluation.expected_output` | Ground truth (if provided) | 1000 chars |
| `evaluation.retrieval_context` | RAG context chunks (JSON) | 2000 chars |
| `evaluation.reasoning` | LLM-generated evaluation reasoning | 2000 chars |

---

## Exporters

### Console Exporter (Default)

The console exporter outputs to stdout—ideal for development:

```yaml
exporters:
  console:
    enabled: true
    pretty_print: true       # Human-readable format
    include_timestamps: true
```

When no exporters are explicitly enabled, console is used automatically.

### OTLP Exporter

Export to any OpenTelemetry-compatible backend (Jaeger, Zipkin, Grafana Tempo, etc.):

#### gRPC (Default)

```yaml
exporters:
  otlp:
    enabled: true
    endpoint: http://localhost:4317
    protocol: grpc
    insecure: true  # No TLS (development)
```

#### HTTP/Protobuf

```yaml
exporters:
  otlp:
    enabled: true
    endpoint: http://localhost:4318
    protocol: http
```

#### With Authentication

```yaml
exporters:
  otlp:
    enabled: true
    endpoint: https://otel-collector.example.com:4317
    protocol: grpc
    headers:
      authorization: "Bearer ${OTEL_API_KEY}"
    compression: gzip
    timeout_ms: 30000
```

### Prometheus Exporter (Planned)

```yaml
exporters:
  prometheus:
    enabled: true
    port: 8889
    host: 0.0.0.0
    path: /metrics
```

Metrics available at `http://localhost:8889/metrics`.

### Azure Monitor Exporter (Planned)

```yaml
exporters:
  azure_monitor:
    enabled: true
    connection_string: "${APPLICATIONINSIGHTS_CONNECTION_STRING}"
```

### Multiple Exporters

Enable multiple exporters simultaneously:

```yaml
exporters:
  console:
    enabled: true      # Local debugging
  otlp:
    enabled: true      # Send to backend
    endpoint: http://localhost:4317
```

---

## Setting Up an OTLP Sink

### .NET Aspire Dashboard

The [Aspire Dashboard](https://learn.microsoft.com/en-us/dotnet/aspire/fundamentals/dashboard) provides a free, local OpenTelemetry UI.

#### Quick Start with Docker

```bash
# Run Aspire Dashboard
docker run --rm -d \
  --name aspire-dashboard \
  -p 18888:18888 \
  -p 4317:18889 \
  mcr.microsoft.com/dotnet/aspire-dashboard:9.0
```

- **Dashboard UI**: http://localhost:18888
- **OTLP gRPC Endpoint**: http://localhost:4317

#### Configure HoloDeck

```yaml
observability:
  enabled: true
  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc
      insecure: true
```

#### Run Your Agent

```bash
holodeck chat agent.yaml
# or
holodeck serve agent.yaml
```

Open http://localhost:18888 to view traces, metrics, and logs.

---

## Environment Variables

### User Configuration

Set sensitive values in environment:

```bash
# OTLP authentication
export OTEL_API_KEY="your-api-key"

# Azure Monitor
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
```

Use in YAML with `${VAR_NAME}`:

```yaml
exporters:
  otlp:
    headers:
      authorization: "Bearer ${OTEL_API_KEY}"
```

### Auto-Enabled by HoloDeck

HoloDeck automatically sets these environment variables when observability is enabled:

```bash
# Enables Semantic Kernel GenAI telemetry
SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true

# Enables prompt/completion capture (only if capture_content: true)
SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE=true
```

---

## Examples

### Development Setup

```yaml
# agent.yaml
name: dev-agent
model:
  provider: ollama
  name: llama3.2:latest

observability:
  enabled: true
  traces:
    capture_content: true  # See prompts for debugging
  exporters:
    console:
      enabled: true
      pretty_print: true
```

### Production with OTLP

```yaml
# agent.yaml
name: prod-agent
model:
  provider: openai
  name: gpt-4o

observability:
  enabled: true
  service_name: "prod-research-agent"

  traces:
    sample_rate: 0.1       # 10% sampling
    capture_content: false # Privacy

  metrics:
    enabled: true
    export_interval_ms: 10000

  logs:
    enabled: true
    level: WARNING

  resource_attributes:
    environment: production
    version: "1.2.0"
    team: research

  exporters:
    otlp:
      enabled: true
      endpoint: https://otel.example.com:4317
      protocol: grpc
      headers:
        authorization: "Bearer ${OTEL_API_KEY}"
      compression: gzip
```

### Multi-Exporter Setup

```yaml
observability:
  enabled: true

  exporters:
    # Local debugging
    console:
      enabled: true

    # Central observability platform
    otlp:
      enabled: true
      endpoint: http://tempo:4317
      protocol: grpc

    # Metrics for alerting
    prometheus:
      enabled: true
      port: 8889
```

### Minimal Tracing Only

```yaml
observability:
  enabled: true
  metrics:
    enabled: false
  logs:
    enabled: false
  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
```

---

## Integration with Commands

Observability is available in all HoloDeck commands:

### Chat

```bash
holodeck chat agent.yaml --verbose
# Traces each message exchange
```

### Test

```bash
holodeck test agent.yaml
# Traces each test case execution
```

### Serve

```bash
holodeck serve agent.yaml
# Traces each API request
```

---

## Performance

Observability is designed for minimal overhead:

| Metric | Value |
|--------|-------|
| Overhead | < 5% of response time |
| Scale | ~100 requests/min, ~10K spans/hour |
| Batch size | 512 spans (default) |
| Buffer | 2048 spans max |
| Drop policy | Oldest-first when buffer full |

---

## Troubleshooting

### No traces appearing

1. Check `observability.enabled: true`
2. Verify exporter endpoint is reachable
3. Check for firewall/network issues
4. Enable verbose mode: `holodeck chat agent.yaml --verbose`

### Missing token counts

Some providers don't return token usage in streaming mode. Use non-streaming for complete metrics.

### High memory usage

Reduce buffer size for high-volume scenarios:

```yaml
traces:
  max_queue_size: 512
  max_export_batch_size: 128
```

### OTLP connection refused

Ensure your OTLP endpoint is running and accessible:

```bash
# Test gRPC endpoint
grpcurl -plaintext localhost:4317 list

# Test HTTP endpoint
curl http://localhost:4318/v1/traces
```

---

## Best Practices

1. **Development**: Use console exporter with `capture_content: true`
2. **Production**: Use OTLP with sampling and no content capture
3. **Security**: Never capture content in production without redaction
4. **Sampling**: Use 10-25% sampling for high-traffic services
5. **Retention**: Configure your backend's retention policy appropriately

---

## Next Steps

- See [Agent Server Guide](serve.md) for deploying agents
- See [Global Configuration](global-config.md) for shared settings
- See [OpenTelemetry Documentation](https://opentelemetry.io/docs/) for backend setup
- See [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for attribute details
