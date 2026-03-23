# Observability

The observability subsystem provides OpenTelemetry instrumentation for HoloDeck
agents, following GenAI semantic conventions. It manages traces, metrics, and
logs through a unified initialization lifecycle and supports multiple exporters
(console, OTLP, with Prometheus and Azure Monitor planned).

## Package entry point

::: holodeck.lib.observability
    options:
      docstring_style: google
      show_source: true
      members:
        - initialize_observability
        - shutdown_observability
        - get_tracer
        - get_meter
        - get_observability_context
        - enable_semantic_kernel_telemetry
        - ObservabilityContext
        - ObservabilityError
        - ObservabilityConfigError

---

## Providers (`providers`)

Core provider setup and lifecycle management. Creates the OpenTelemetry
`TracerProvider`, `MeterProvider`, and `LoggerProvider`, and exposes helper
accessors.

::: holodeck.lib.observability.providers.ObservabilityContext
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.create_resource
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.set_up_logging
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.set_up_tracing
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.set_up_metrics
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.initialize_observability
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.shutdown_observability
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.get_tracer
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.get_meter
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.providers.get_observability_context
    options:
      docstring_style: google
      show_source: true

---

## Configuration (`config`)

Exporter configuration and logging coordination. Prevents double logging when
the console exporter is active, and builds the exporter lists consumed by the
provider setup functions.

::: holodeck.lib.observability.config.is_console_exporter_active
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.config.configure_logging
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.config.configure_exporters
    options:
      docstring_style: google
      show_source: true

---

## Instrumentation (`instrumentation`)

Semantic Kernel telemetry integration. Sets the environment variables that
Semantic Kernel reads at startup to emit GenAI semantic convention spans.

::: holodeck.lib.observability.instrumentation.enable_semantic_kernel_telemetry
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.instrumentation.SK_OTEL_DIAGNOSTICS_ENV
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.instrumentation.SK_OTEL_SENSITIVE_ENV
    options:
      docstring_style: google
      show_source: true

---

## Errors (`errors`)

Custom exception hierarchy for observability failures.

::: holodeck.lib.observability.errors.ObservabilityError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.errors.ObservabilityConfigError
    options:
      docstring_style: google
      show_source: true

---

## Exporters

### Console exporter (`exporters.console`)

Development/debugging exporter that writes telemetry to stdout. Used as the
default fallback when no other exporters are configured.

::: holodeck.lib.observability.exporters.console.create_console_exporters
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.console.create_console_span_exporter
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.console.create_console_metric_reader
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.console.create_console_log_exporter
    options:
      docstring_style: google
      show_source: true

### OTLP exporter (`exporters.otlp`)

Exports telemetry via OTLP (gRPC or HTTP) to any compatible collector such as
Jaeger, Honeycomb, or Datadog.

::: holodeck.lib.observability.exporters.otlp.create_otlp_exporters
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_span_exporter
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_span_exporter_grpc
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_span_exporter_http
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_metric_reader
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_metric_exporter_grpc
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_metric_exporter_http
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_log_exporter
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_log_exporter_grpc
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.create_otlp_log_exporter_http
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.resolve_headers
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.adjust_endpoint_for_protocol
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.get_compression_grpc
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.observability.exporters.otlp.get_compression_http
    options:
      docstring_style: google
      show_source: true
