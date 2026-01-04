# Observability Internal API Contract

**Feature**: 018-otel-observability
**Date**: 2026-01-04

## Overview

This document defines the internal Python API contract for the observability module. These are internal APIs used by HoloDeck components, not exposed to end users.

---

## Module: `holodeck.lib.observability`

### Public Functions

#### `initialize_observability`

Initialize all telemetry providers based on configuration.

```python
def initialize_observability(config: ObservabilityConfig) -> ObservabilityContext:
    """Initialize OpenTelemetry providers for traces, metrics, and logs.

    Args:
        config: Observability configuration from agent.yaml

    Returns:
        ObservabilityContext with initialized providers

    Raises:
        ObservabilityConfigError: If configuration is invalid
        ExporterConnectionError: If exporter fails to connect

    Example:
        >>> from holodeck.lib.observability import initialize_observability
        >>> from holodeck.models.observability import ObservabilityConfig
        >>>
        >>> config = ObservabilityConfig(enabled=True)
        >>> context = initialize_observability(config)
        >>> # Telemetry now active
    """
```

#### `shutdown_observability`

Gracefully shutdown all telemetry providers.

```python
def shutdown_observability(context: ObservabilityContext) -> None:
    """Flush pending telemetry and shutdown providers.

    Args:
        context: ObservabilityContext from initialize_observability

    Note:
        Should be called during application shutdown.
        Blocks until all pending spans/metrics are exported.
    """
```

#### `get_tracer`

Get a tracer for creating spans.

```python
def get_tracer(name: str) -> Tracer:
    """Get an OpenTelemetry tracer instance.

    Args:
        name: Tracer name (typically __name__)

    Returns:
        OpenTelemetry Tracer instance

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation"):
        ...     # do work
    """
```

#### `get_meter`

Get a meter for creating metrics.

```python
def get_meter(name: str) -> Meter:
    """Get an OpenTelemetry meter instance.

    Args:
        name: Meter name (typically __name__)

    Returns:
        OpenTelemetry Meter instance

    Example:
        >>> meter = get_meter(__name__)
        >>> counter = meter.create_counter("requests")
        >>> counter.add(1)
    """
```

---

### ObservabilityContext

Context object returned by `initialize_observability`.

```python
@dataclass
class ObservabilityContext:
    """Container for initialized observability components."""

    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    logger_provider: LoggerProvider
    exporters: list[str]  # Names of enabled exporters

    def is_enabled(self) -> bool:
        """Check if observability is active."""
        ...

    def get_resource(self) -> Resource:
        """Get the shared OpenTelemetry resource."""
        ...
```

---

## Module: `holodeck.lib.observability.exporters`

### Exporter Factory Functions

#### `create_otlp_exporter`

```python
def create_otlp_exporter(
    config: OTLPExporterConfig,
    signal: Literal["traces", "metrics", "logs"]
) -> SpanExporter | MetricExporter | LogExporter:
    """Create OTLP exporter for the specified signal type.

    Args:
        config: OTLP exporter configuration
        signal: Type of telemetry signal

    Returns:
        Configured exporter instance

    Raises:
        ExporterConfigError: If configuration is invalid
    """
```

#### `create_prometheus_exporter`

```python
def create_prometheus_exporter(
    config: PrometheusExporterConfig
) -> PrometheusMetricReader:
    """Create Prometheus metric reader with HTTP server.

    Args:
        config: Prometheus exporter configuration

    Returns:
        PrometheusMetricReader instance

    Raises:
        PortInUseError: If configured port is already bound

    Note:
        Starts HTTP server on configured host:port
    """
```

#### `create_azure_monitor_exporter`

```python
def create_azure_monitor_exporter(
    config: AzureMonitorExporterConfig,
    signal: Literal["traces", "metrics", "logs"]
) -> AzureMonitorTraceExporter | AzureMonitorMetricExporter | AzureMonitorLogExporter:
    """Create Azure Monitor exporter for the specified signal type.

    Args:
        config: Azure Monitor exporter configuration
        signal: Type of telemetry signal

    Returns:
        Configured Azure Monitor exporter

    Raises:
        ConnectionStringError: If connection string is invalid
    """
```

---

## Module: `holodeck.lib.observability.semantic_conventions`

### GenAI Attribute Constants

```python
# Operation names
GENAI_OPERATION_CHAT = "chat"
GENAI_OPERATION_EMBEDDINGS = "embeddings"
GENAI_OPERATION_TEXT_COMPLETION = "text_completion"

# Attribute keys
GENAI_SYSTEM = "gen_ai.system"
GENAI_OPERATION_NAME = "gen_ai.operation.name"
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GENAI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GENAI_REQUEST_TOP_P = "gen_ai.request.top_p"

GENAI_RESPONSE_ID = "gen_ai.response.id"
GENAI_RESPONSE_MODEL = "gen_ai.response.model"
GENAI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

GENAI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GENAI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Tool execution
GENAI_TOOL_NAME = "gen_ai.tool.name"
GENAI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Sensitive (opt-in)
GENAI_PROMPT_ROLE = "gen_ai.prompt.{index}.role"
GENAI_PROMPT_CONTENT = "gen_ai.prompt.{index}.content"
GENAI_COMPLETION_ROLE = "gen_ai.completion.{index}.role"
GENAI_COMPLETION_CONTENT = "gen_ai.completion.{index}.content"
```

---

## Error Types

```python
class ObservabilityError(HoloDeckError):
    """Base exception for observability errors."""
    pass

class ObservabilityConfigError(ObservabilityError):
    """Invalid observability configuration."""
    pass

class ExporterConnectionError(ObservabilityError):
    """Exporter failed to connect to backend."""
    pass

class PortInUseError(ObservabilityError):
    """Port already in use (Prometheus exporter)."""
    pass

class ConnectionStringError(ObservabilityError):
    """Invalid Azure Monitor connection string."""
    pass
```

---

## Usage in CLI Commands

### Integration with `holodeck chat`

```python
# In holodeck/cli/commands/chat.py

async def chat_command(agent_path: str) -> None:
    config = load_agent_config(agent_path)

    context = None
    if config.observability and config.observability.enabled:
        context = initialize_observability(config.observability)

    try:
        await run_chat_session(config)
    finally:
        if context:
            shutdown_observability(context)
```

### Integration with `holodeck test`

```python
# In holodeck/cli/commands/test.py

async def test_command(agent_path: str) -> None:
    config = load_agent_config(agent_path)

    context = None
    if config.observability and config.observability.enabled:
        context = initialize_observability(config.observability)

    try:
        results = await run_tests(config)
        report_results(results)
    finally:
        if context:
            shutdown_observability(context)
```

### Integration with `holodeck serve`

```python
# In holodeck/serve/app.py

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for observability setup/teardown."""
    config = app.state.agent_config

    context = None
    if config.observability and config.observability.enabled:
        context = initialize_observability(config.observability)
        app.state.observability_context = context

    yield  # Server runs here

    if context:
        shutdown_observability(context)

def create_app(agent_path: str) -> FastAPI:
    config = load_agent_config(agent_path)
    app = FastAPI(lifespan=lifespan)
    app.state.agent_config = config
    return app
```

**Note**: The serve command uses FastAPI's lifespan context manager to ensure observability is initialized before the server starts accepting requests and properly shutdown when the server stops. Each HTTP request automatically inherits the trace context.

---

## Thread Safety

All public functions are thread-safe:

- `initialize_observability`: Uses thread-safe global provider registration
- `shutdown_observability`: Synchronizes flush operations
- `get_tracer` / `get_meter`: Return thread-safe instances

The observability module follows OpenTelemetry SDK threading guarantees.
