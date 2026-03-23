# Quickstart: OTel GenAI Instrumentation Integration

**Feature**: 022-otel-genai-semconv
**Date**: 2026-03-01

## Prerequisites

- HoloDeck installed with dev dependencies (`make install-dev`)
- Python 3.10+
- Virtual environment activated (`source .venv/bin/activate`)

## Installation

Install the optional instrumentation package:

```bash
# Via extras group
uv pip install "holodeck-ai[claude-otel]"

# Or directly
uv add otel-instrumentation-claude-agent-sdk>=0.0.3
```

## Usage

### Enable GenAI Instrumentation

Add the `observability` block to your `agent.yaml`:

```yaml
name: my-claude-agent
model:
  provider: anthropic
  name: claude-sonnet-4-20250514

instructions:
  inline: "You are a helpful assistant."

observability:
  enabled: true
  traces:
    enabled: true
    capture_content: false  # Set to true to capture prompts/completions
  metrics:
    enabled: true
  exporters:
    otlp:
      enabled: true
      endpoint: "http://localhost:4317"
      protocol: grpc
```

Run tests or chat:

```bash
holodeck test agent.yaml -v
holodeck chat agent.yaml
```

### Verify Spans

With an OTLP collector (e.g., Jaeger) running at `localhost:4317`, you should see:

```
holodeck.cli.test
  └── invoke_agent my-claude-agent
        ├── execute_tool search_knowledge_base
        └── execute_tool get_user_context
```

### Without the Instrumentation Package

If the package is not installed, HoloDeck logs a warning and continues normally:

```
WARNING: otel-instrumentation-claude-agent-sdk not installed;
Claude GenAI instrumentation disabled.
Install with: pip install holodeck-ai[claude-otel]
```

All other observability (HoloDeck spans, subprocess env vars, SK telemetry) continues to work.

## Development

### Running Tests

```bash
# Run all tests (parallel)
make test

# Run just the Claude backend unit tests
pytest tests/unit/lib/backends/test_claude_backend.py -n auto -v

# Run the instrumentation integration test (span hierarchy verification)
pytest tests/integration/test_claude_instrumentation.py -n auto -v

# Run with coverage
make test-coverage
```

### Code Quality

```bash
make format     # Format with Black + Ruff
make lint       # Run Ruff + Bandit
make type-check # MyPy
```

## Key Files

| File | Purpose |
|------|---------|
| `src/holodeck/lib/backends/claude_backend.py` | Integration point — `_activate_instrumentation()`, `uninstrument()` in teardown |
| `src/holodeck/lib/observability/providers.py` | `get_observability_context()` accessor |
| `tests/unit/lib/backends/test_claude_backend.py` | Instrumentation lifecycle unit tests |
| `tests/integration/test_claude_instrumentation.py` | Span hierarchy integration test (InMemorySpanExporter) |
| `pyproject.toml` | `[claude-otel]` optional dependency group |
