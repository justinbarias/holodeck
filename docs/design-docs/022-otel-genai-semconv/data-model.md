# Data Model: OTel GenAI Instrumentation Integration

**Feature**: 022-otel-genai-semconv
**Date**: 2026-03-01

## Overview

This feature introduces no new Pydantic models, database entities, or YAML schema changes. It extends the existing `ClaudeBackend` class with two instance attributes for lifecycle management and adds one accessor function to the observability module.

## Modified Entities

### ClaudeBackend (class — `claude_backend.py`)

**New Instance Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `_instrumentor` | `ClaudeAgentSdkInstrumentor \| None` | `None` | Reference to the active instrumentor instance. Set during `_activate_instrumentation()` (called from `initialize()`). Cleared during `teardown()`. |

**State Transitions:**

```
__init__()     → _instrumentor=None
initialize()   → _instrumentor=instance  (if activated)
                  _instrumentor=None      (if skipped)
teardown()     → _instrumentor=None
```

**Global Scope Constraint:** `instrument()` monkey-patches process-global functions. Only one active instrumented `ClaudeBackend` should exist at a time. Current usage is sequential (BackendSelector creates one backend per invocation).

**New Private Method:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `_activate_instrumentation` | `(self) -> None` | Activates GenAI instrumentation if the package is installed and observability is enabled. Called from `initialize()`. Isolated from the main init flow so that `return` on ImportError or disabled observability only exits this helper. |

### ObservabilityContext (dataclass — `providers.py`)

**No changes to the dataclass itself.** A new module-level accessor is added:

```python
def get_observability_context() -> ObservabilityContext | None:
    """Return the current ObservabilityContext, or None if not initialized.

    Thread-safety note: This accessor reads module-level state that is set
    by ``initialize_observability()`` in the CLI layer's main thread, before
    ``asyncio.run()`` is called. All async tasks (including
    ``ClaudeBackend.initialize()``) run in the same thread, so no
    synchronization is needed. If future code introduces background task
    spawning that accesses this state, thread synchronization will be
    required.
    """
    return _observability_context
```

## Existing Entities (Unchanged)

| Entity | Location | Why Unchanged |
|--------|----------|---------------|
| `ObservabilityConfig` | `models/observability.py` | Existing `traces.enabled`, `traces.capture_content`, `metrics.enabled` fields already carry all information needed by the instrumentor. |
| `Agent` | `models/agent.py` | `agent.observability` and `agent.name` already provide config and agent name. |
| `ClaudeAgentOptions` | Claude Agent SDK | No changes — hooks are injected by the instrumentation package wrapping `query()`, not through options. |
| `translate_observability()` | `otel_bridge.py` | Subprocess env vars unchanged — coexists with main-process instrumentation. |

## External Entities (Owned by Instrumentation Package)

These are documented for reference only — HoloDeck does NOT define or modify them:

| Entity | Owner | Description |
|--------|-------|-------------|
| `ClaudeAgentSdkInstrumentor` | `otel-instrumentation-claude-agent-sdk` | OTel Instrumentor class. HoloDeck calls `instrument()`/`uninstrument()`. |
| `invoke_agent` span | Instrumentation package | GenAI parent span created per `query()` call. |
| `execute_tool` span | Instrumentation package | Child span per tool invocation (via SDK hooks). |
| `gen_ai.client.token.usage` | Instrumentation package | Token usage histogram metric. |
| `gen_ai.client.operation.duration` | Instrumentation package | Operation duration histogram metric. |

## Dependency Graph

```
pyproject.toml
  └── [claude-otel] optional group
        └── otel-instrumentation-claude-agent-sdk >= 0.0.3, < 0.1.0
              ├── opentelemetry-api ~= 1.12          (satisfied by existing deps)
              ├── opentelemetry-instrumentation       (new transitive)
              ├── opentelemetry-semantic-conventions   (new transitive)
              ├── wrapt >= 1.0.0                      (new transitive)
              └── claude-agent-sdk >= 0.1.44          (satisfied: pinned at 0.1.44)
```
