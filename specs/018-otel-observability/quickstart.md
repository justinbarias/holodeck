# Quickstart: OpenTelemetry Observability

**Feature**: 018-otel-observability
**Date**: 2026-01-04

## Overview

This guide shows how to enable observability for your HoloDeck agents in 3 steps.

---

## Prerequisites

- HoloDeck installed (`pip install holodeck-ai`)
- An observability backend (optional for getting started):
  - **OTLP**: Jaeger, Aspire Dashboard, or any OTLP collector
  - **Prometheus**: Prometheus server with Grafana
  - **Azure Monitor**: Application Insights resource

---

## Step 1: Add Observability Configuration

Add the `observability` section to your `agent.yaml`:

```yaml
name: my-agent
description: My AI agent with observability

model:
  provider: openai
  name: gpt-4o-mini

instructions:
  inline: You are a helpful assistant.

# Enable observability
observability:
  enabled: true
  service_name: my-agent

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
```

That's it! Your agent now exports telemetry to the default OTLP endpoint.

---

## Step 2: Start an OTLP Collector (Optional)

For local development, use the Aspire Dashboard:

```bash
# Start Aspire Dashboard with Docker
docker run --rm -it \
  -p 18888:18888 \
  -p 4317:4317 \
  mcr.microsoft.com/dotnet/aspire-dashboard:latest
```

Access the dashboard at `http://localhost:18888`.

---

## Step 3: Run Your Agent

```bash
# Interactive chat
holodeck chat agent.yaml

# Run tests
holodeck test agent.yaml

# Start REST API server
holodeck serve agent.yaml
```

View traces, metrics, and logs in your observability backend.

---

## Configuration Examples

### Basic OTLP (Recommended for Development)

```yaml
observability:
  enabled: true
  service_name: my-agent

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
```

### Prometheus Metrics

```yaml
observability:
  enabled: true
  service_name: my-agent

  exporters:
    prometheus:
      enabled: true
      port: 8889
```

Scrape metrics at `http://localhost:8889/metrics`.

### Azure Monitor

```yaml
observability:
  enabled: true
  service_name: my-agent

  exporters:
    azure_monitor:
      enabled: true
      connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

Set the environment variable:

```bash
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
```

### Multiple Exporters

```yaml
observability:
  enabled: true
  service_name: my-agent

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317

    prometheus:
      enabled: true
      port: 8889

    azure_monitor:
      enabled: true
      connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

### With Sensitive Data Capture

```yaml
observability:
  enabled: true
  service_name: my-agent

  traces:
    capture_content: true  # Capture prompts/completions
    redaction_patterns:
      - '\b\d{3}-\d{2}-\d{4}\b'  # Redact SSNs
      - '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Redact emails

  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
```

### With Custom Headers (Authentication)

```yaml
observability:
  enabled: true
  service_name: my-agent

  exporters:
    otlp:
      enabled: true
      endpoint: https://otel-collector.example.com:4317
      headers:
        authorization: Bearer ${OTEL_API_KEY}
        x-custom-header: my-value
```

---

## What Gets Captured

### Traces

Every LLM invocation creates spans with:

- `gen_ai.system`: Provider (openai, anthropic, etc.)
- `gen_ai.request.model`: Model name
- `gen_ai.usage.input_tokens`: Prompt tokens
- `gen_ai.usage.output_tokens`: Completion tokens
- `gen_ai.response.finish_reasons`: Why generation stopped

### Metrics

- Token usage counters
- Request duration histograms
- Error counts

### Logs

Structured logs with trace context for correlation.

---

## Viewing Telemetry

### Aspire Dashboard

1. Open `http://localhost:18888`
2. Click **Traces** to see request flows
3. Click individual spans to see attributes

### Jaeger

1. Open Jaeger UI (default: `http://localhost:16686`)
2. Select service: `my-agent`
3. Click **Find Traces**

### Prometheus + Grafana

1. Add Prometheus data source in Grafana
2. Query `holodeck_*` metrics
3. Create dashboards for token usage, latency, errors

### Azure Monitor

1. Open Application Insights in Azure Portal
2. Click **Transaction search**
3. Filter by operation name

---

## Troubleshooting

### No telemetry appearing?

1. Verify `observability.enabled: true`
2. Check exporter endpoint is reachable
3. Ensure at least one exporter is enabled

### Connection refused errors?

1. Verify collector is running
2. Check port matches configuration
3. For Docker, ensure ports are exposed

### Missing spans?

1. Check `sample_rate` is > 0 (default: 1.0)
2. Verify `traces.enabled: true`

---

## Next Steps

- [Full Configuration Reference](./data-model.md)
- [Research Notes](./research.md)
- [API Contracts](./contracts/observability-api.md)
