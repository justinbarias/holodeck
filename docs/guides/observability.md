# Observability Guide

OpenTelemetry-based tracing, metrics, and logging for HoloDeck agents — configured entirely in YAML.

## Quick start

Run a local trace UI (the [.NET Aspire dashboard](https://learn.microsoft.com/en-us/dotnet/aspire/fundamentals/dashboard)) and point your agent at it.

```bash
docker run --rm -d --name aspire-dashboard \
  -p 18888:18888 -p 4317:18889 \
  mcr.microsoft.com/dotnet/aspire-dashboard:latest
# Dashboard UI: http://localhost:18888 · OTLP gRPC ingest: localhost:4317
```

```yaml
# agent.yaml
observability:
  enabled: true
  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc
      insecure: true
```

```bash
holodeck chat agent.yaml      # or: holodeck serve agent.yaml
```

Open http://localhost:18888 to see traces, metrics, and logs stream in as the agent runs.

## How it works

When `observability.enabled` is set, HoloDeck stands up an OpenTelemetry pipeline following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for LLM and tool spans. The instrumentation layer emits spans for each LLM call and tool execution; you choose where they go via the `exporters` block (console, OTLP, Prometheus, Azure Monitor). Content capture is off by default for privacy and can be enabled with redaction. Everything is config-driven — no code changes. See the [OpenAI Backend guide](openai-backend.md#tracing) for backend-specific tracing behavior.

---

## Configuration reference

```yaml
observability:
  enabled: true
  service_name: "custom-service"   # default: "holodeck-{agent.name}"

  traces:
    enabled: true
    sample_rate: 1.0               # 0.0–1.0 (default 100%)
    capture_content: false         # capture prompts/completions (default false)
    redaction_patterns:
      - '\b\d{3}-\d{2}-\d{4}\b'    # e.g. SSN
    max_queue_size: 2048
    max_export_batch_size: 512

  metrics:
    enabled: true
    export_interval_ms: 5000

  logs:
    enabled: true
    level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    include_trace_context: true
    filter_namespaces:
      - holodeck                   # which loggers to capture

  resource_attributes:
    environment: production
    version: 1.0

  exporters:
    console:
      enabled: true
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

The default `service_name` is `holodeck-{agent.name}` (agent `research` → `holodeck-research`); override with `service_name`.

---

## Traces

- **Sampling** — `sample_rate: 1.0` traces everything (good for dev); `0.1` samples 10% (high-traffic production); `0.0` disables tracing.
- **Content capture** — `capture_content: false` (default) keeps prompts/completions out of spans. Set `true` to record them as span events for debugging.
- **Redaction** — with capture on, `redaction_patterns` are scrubbed before export:

```yaml
traces:
  capture_content: true
  redaction_patterns:
    - '\b\d{3}-\d{2}-\d{4}\b'                                   # SSN
    - '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'    # Email
    - '\b\d{16}\b'                                              # Credit card
```

- **Buffering** — `max_queue_size` (default 2048) and `max_export_batch_size` (default 512) tune batch export.

### GenAI attributes

Every LLM invocation captures:

| Attribute                       | Description           | Example                     |
| ------------------------------- | --------------------- | --------------------------- |
| `gen_ai.system`                 | Provider name         | `openai`, `anthropic`       |
| `gen_ai.request.model`          | Model identifier      | `gpt-4o`, `claude-sonnet-4-5` |
| `gen_ai.request.temperature`    | Temperature           | `0.7`                       |
| `gen_ai.usage.input_tokens`     | Prompt tokens         | `150`                       |
| `gen_ai.usage.output_tokens`    | Completion tokens     | `75`                        |
| `gen_ai.usage.total_tokens`     | Total tokens          | `225`                       |
| `gen_ai.response.finish_reason` | Why generation stopped | `stop`, `length`           |
| `gen_ai.response.id`            | Provider completion ID | `chatcmpl-...`             |

When `capture_content` is enabled, spans also carry `gen_ai.content.prompt` and `gen_ai.content.completion` events.

### OpenAI Agents backend tracing

On the OpenAI Agents backend the SDK runs its own tracing pipeline; HoloDeck installs a `TracingProcessor` that **mirrors** each finished SDK span into an OTel span on HoloDeck's tracer (carrying your redaction and exporters). The mirror is active only when both `observability.enabled` and `observability.traces.enabled` are true.

| Configuration                                       | platform.openai.com upload | OTel mirror |
| --------------------------------------------------- | -------------------------- | ----------- |
| `provider: openai`                                  | ✓                          | ✓           |
| `provider: azure_openai`                            | none (mirror only)         | ✓           |
| `observability.disable_provider_tracing: true`      | none (mirror only)         | ✓           |

`capture_content` (default `false`) controls whether sensitive tool input/output is included in uploaded spans. See the [OpenAI Backend guide](openai-backend.md#tracing) for full detail.

### Evaluation tracing (DeepEval)

HoloDeck creates a `holodeck.evaluation.{metric_name}` span for each DeepEval metric during test runs, with attributes like `evaluation.metric.name`, `evaluation.threshold`, `evaluation.model.provider`, `evaluation.score` (0.0–1.0), `evaluation.passed`, and `evaluation.duration_ms`. Enable input/output capture with:

```yaml
observability:
  traces:
    capture_evaluation_content: true   # input, actual/expected output, retrieval context, reasoning
```

Content attributes (`evaluation.input`, `evaluation.actual_output`, `evaluation.expected_output`, `evaluation.reasoning`) are truncated to 1000–2000 chars.

---

## Exporters

### Console (default)

```yaml
exporters:
  console:
    enabled: true
    pretty_print: true
    include_timestamps: true
```

Used automatically when no other exporter is enabled.

### OTLP

Export to any OpenTelemetry backend (Aspire, Jaeger, Grafana Tempo, …):

```yaml
exporters:
  otlp:
    enabled: true
    endpoint: http://localhost:4317     # 4318 for protocol: http
    protocol: grpc                      # grpc or http
    insecure: true                      # no TLS (development)
    headers:
      authorization: "Bearer ${OTEL_API_KEY}"   # when authenticating
    compression: gzip
    timeout_ms: 30000
```

### Prometheus (planned)

```yaml
exporters:
  prometheus:
    enabled: true
    port: 8889
    host: 0.0.0.0
    path: /metrics
```

Metrics at `http://localhost:8889/metrics`.

### Azure Monitor (planned)

```yaml
exporters:
  azure_monitor:
    enabled: true
    connection_string: "${APPLICATIONINSIGHTS_CONNECTION_STRING}"
```

### Multiple exporters

```yaml
exporters:
  console:
    enabled: true                       # local debugging
  otlp:
    enabled: true                       # central backend
    endpoint: http://localhost:4317
```

---

## Environment variables

Set sensitive values in the environment and reference them with `${VAR_NAME}`:

```bash
export OTEL_API_KEY="your-api-key"
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
```

```yaml
exporters:
  otlp:
    headers:
      authorization: "Bearer ${OTEL_API_KEY}"
```

---

## Examples

### Development

```yaml
observability:
  enabled: true
  traces:
    capture_content: true               # see prompts while debugging
  exporters:
    console:
      enabled: true
      pretty_print: true
```

### Production with OTLP

```yaml
observability:
  enabled: true
  service_name: "prod-research-agent"
  traces:
    sample_rate: 0.1
    capture_content: false
  metrics:
    enabled: true
    export_interval_ms: 10000
  logs:
    enabled: true
    level: WARNING
  resource_attributes:
    environment: production
    version: "1.2.0"
  exporters:
    otlp:
      enabled: true
      endpoint: https://otel.example.com:4317
      protocol: grpc
      headers:
        authorization: "Bearer ${OTEL_API_KEY}"
      compression: gzip
```

Observability is available across `holodeck chat`, `holodeck test`, and `holodeck serve` — each traces message exchanges, test cases, and API requests respectively. Overhead is under ~5% of response time at ~100 req/min; the buffer holds 2048 spans (oldest dropped first when full).

---

## Troubleshooting

### No traces appearing

1. Confirm `observability.enabled: true`.
2. Verify the exporter endpoint is reachable (no firewall/network block).
3. Run with `--verbose`: `holodeck chat agent.yaml --verbose`.

### Missing token counts

Some providers omit token usage in streaming mode. Use non-streaming for complete metrics.

### High memory usage

```yaml
traces:
  max_queue_size: 512
  max_export_batch_size: 128
```

### OTLP connection refused

```bash
grpcurl -plaintext localhost:4317 list      # test gRPC
curl http://localhost:4318/v1/traces        # test HTTP
```

---

## Next steps

- [Agent Server Guide](serve.md) — deploying agents as servers
- [OpenAI Backend guide](openai-backend.md#tracing) — backend-specific tracing
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — backend setup
- [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — attribute details
