# Research: OTel GenAI Instrumentation Integration

**Feature**: 022-otel-genai-semconv
**Date**: 2026-03-01

## R-001: How to Use `otel-instrumentation-claude-agent-sdk` v0.0.3

### Decision

Use the standard OTel Instrumentor pattern: create a `ClaudeAgentSdkInstrumentor()` instance, call `instrument()` with explicit provider references, and call `uninstrument()` on teardown.

### Findings

**Import Path:**
```python
from opentelemetry.instrumentation.claude_agent_sdk import ClaudeAgentSdkInstrumentor
```

**`instrument()` Signature:**
```python
def instrument(
    self,
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    agent_name: str | None = None,
    capture_content: bool = False,
    **kwargs: Any,
) -> None
```

**`uninstrument()` Signature:**
```python
def uninstrument(self, **kwargs: Any) -> None
```

**What `instrument()` Does:**

1. Wraps four targets via `wrapt`:
   - Module-level `query()` function (standalone invocations)
   - `ClaudeSDKClient.__init__()` (client construction)
   - `ClaudeSDKClient.query()` (client-based invocations)
   - `ClaudeSDKClient.receive_response()` (response processing)

2. Builds instrumentation hooks via `build_instrumentation_hooks()`:
   - `PreToolUse` → creates `execute_tool` child span
   - `PostToolUse` → finalizes span with results
   - `PostToolUseFailure` → records error attributes on span
   - `Stop` → no-op placeholder

3. Uses `merge_hooks()` to append instrumentation hooks AFTER any user-provided hooks (user hooks always execute first).

4. Creates `invoke_agent` parent span (CLIENT kind) per `query()` call with GenAI semconv attributes:
   - `gen_ai.operation.name`
   - `gen_ai.system` = `"claude_agent_sdk"`
   - `gen_ai.request.model` / `gen_ai.response.model`
   - `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
   - `gen_ai.usage.cache_creation_input_tokens` / `gen_ai.usage.cache_read_input_tokens`
   - `gen_ai.response.finish_reason`
   - `conversation_id`

5. Records two histogram metrics:
   - `gen_ai.client.token.usage` (by type: input/output/cache)
   - `gen_ai.client.operation.duration` (wall-clock seconds)

**What `uninstrument()` Does:**

Restores original functions by reading `__wrapped__` attributes set by `wrapt` on the four instrumented targets.

**Idempotency:** Standard `BaseInstrumentor` pattern — calling `instrument()` when already instrumented is a no-op. Calling `uninstrument()` when not instrumented is also safe.

**Dependencies:**
- `opentelemetry-api ~= 1.12`
- `opentelemetry-instrumentation >= 0.50b0`
- `opentelemetry-semantic-conventions >= 0.50b0`
- `wrapt >= 1.0.0, < 2.0.0`
- `claude-agent-sdk >= 0.1.44` (extras group `[instruments]`)

### Rationale

The standard OTel Instrumentor pattern provides:
- Familiar API for OTel users
- Built-in idempotency guards
- Clean lifecycle via `instrument()`/`uninstrument()`
- Hook merging that preserves user hooks

### Alternatives Considered

None — the package already provides the standard pattern. No wrapper or adapter needed.

---

## R-002: How ClaudeBackend Accesses OTel Providers

### Decision

Use the existing module-level `_observability_context` in `providers.py`, exposed via a new `get_observability_context()` accessor function.

### Findings

**Current Architecture:**

1. CLI layer (`test.py`/`chat.py`) calls `initialize_observability(config, agent_name)` → returns `ObservabilityContext` and stores it in module-level `_observability_context`.
2. `ObservabilityContext` holds `tracer_provider: TracerProvider | None` and `meter_provider: MeterProvider | None`.
3. `ClaudeBackend.initialize()` currently only accesses `agent.observability` (the config model) — it does NOT have a reference to the initialized providers.
4. The `_observability_context` variable exists but has no public getter.

**Solution:** Add `get_observability_context() -> ObservabilityContext | None` to `providers.py` and export it from `holodeck.lib.observability`. The `ClaudeBackend` calls this during `initialize()` to get provider references.

**Why not pass providers to `ClaudeBackend` constructor?** The `BackendSelector.select()` API is protocol-driven and doesn't know about observability. Adding an `obs_context` parameter would require modifying the protocol, selector, and all callers. The module-level accessor is simpler and matches how `get_tracer()`/`get_meter()` already work.

**Why not use the global OTel providers?** FR-004 requires passing HoloDeck's specific `TracerProvider`, not the global one. HoloDeck may configure exporters (OTLP, console) that differ from whatever is globally set. Using explicit provider references ensures spans flow to HoloDeck's configured exporters.

### Rationale

Module-level accessor follows existing patterns (`get_tracer()`, `get_meter()`). Minimal API change. No protocol modifications needed.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Pass `ObservabilityContext` to `ClaudeBackend.__init__()` | Requires `AgentBackend` protocol change, `BackendSelector` change, and all caller changes. Over-engineering for this use case. |
| Use global `trace.get_tracer_provider()` | May not be HoloDeck's configured provider. Violates FR-004 (explicit provider passing). |
| Read `_observability_context` directly | Private variable access across modules is a code smell. Accessor is cleaner. |

---

## R-003: Optional Dependency Handling Pattern

### Decision

Use try/except `ImportError` at usage site (inside `initialize()`), not at module level. Log warning and continue.

### Findings

**Pattern:**
```python
# Inside ClaudeBackend._activate_instrumentation():
def _activate_instrumentation(self) -> None:
    """Activate GenAI instrumentation if package is installed and observability is enabled."""
    agent = self._agent
    obs = agent.observability
    if not obs or not obs.enabled or not obs.traces or not obs.traces.enabled:
        return

    try:
        from opentelemetry.instrumentation.claude_agent_sdk import (
            ClaudeAgentSdkInstrumentor,
        )
    except ImportError:
        logger.warning(
            "otel-instrumentation-claude-agent-sdk not installed; "
            "Claude GenAI instrumentation disabled. "
            "Install with: pip install holodeck-ai[claude-otel]"
        )
        return  # return only exits this helper, not initialize()

    # ... instrument() call ...
```

**Why a private method (not inline in `initialize()`)?**
- A `return` on ImportError inside `initialize()` would short-circuit the entire method, skipping options building and validation. Extracting to `_activate_instrumentation()` isolates the early-return to the helper only.

**Why deferred import (not module-level)?**
- FR-005/SC-003: Zero overhead when observability disabled. Module-level import would execute even when observability is off.
- FR-002: Package not installed must not crash the backend. A module-level `ImportError` would prevent the entire `claude_backend.py` from loading.

**Optional Dependency Group:** `[claude-otel]` in `pyproject.toml`:
```toml
claude-otel = ["otel-instrumentation-claude-agent-sdk>=0.0.3,<0.1.0"]
```

After modifying `pyproject.toml`, run `uv lock` to regenerate the lock file.

### Rationale

Deferred import is the standard Python pattern for optional dependencies. Matches how HoloDeck already handles optional vector store providers.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Module-level try/except | Still imports when observability is disabled (violates SC-003). |
| `importlib.util.find_spec()` check | Extra step — try/except is simpler and more Pythonic. |
| Make it a required dependency | Violates FR-001 (optional dependency requirement). |

---

## R-004: Instrumentation Lifecycle in ClaudeBackend

### Decision

Store the `ClaudeAgentSdkInstrumentor` instance on `self._instrumentor`. Rely on `BaseInstrumentor`'s internal idempotency for `uninstrument()` safety — no separate `_instrumented` flag needed.

### Findings

**Activation (in `_activate_instrumentation()`, called from `initialize()`):**
1. Check `agent.observability` is not None and `enabled` is True and `traces.enabled` is True. If not, return (no-op).
2. Try importing `ClaudeAgentSdkInstrumentor`. If `ImportError`, log warning, return.
3. Get `ObservabilityContext` via `get_observability_context()`.
4. If context is None (providers not initialized), return.
5. Create instrumentor instance, call `instrument()` with:
   - `tracer_provider=obs_context.tracer_provider`
   - `meter_provider=obs_context.meter_provider` (only if `metrics.enabled`)
   - `agent_name=agent.name`
   - `capture_content=agent.observability.traces.capture_content`
6. Store instrumentor on `self._instrumentor`.

**Deactivation (in `teardown()`):**
1. If `self._instrumentor` is not None:
   - Call `self._instrumentor.uninstrument()` — safe even if already uninstrumented (`BaseInstrumentor` handles idempotently)
   - Set `self._instrumentor = None`

**Error Handling:**
- Wrap the entire `_activate_instrumentation()` body in try/except. If `instrument()` raises (e.g., version incompatibility), log warning and continue without instrumentation. Never fail backend initialization due to instrumentation errors.

**Global Scope Constraint:**
- `instrument()` monkey-patches process-global functions (`query()`, `ClaudeSDKClient.__init__`, etc.). Only one active instrumented `ClaudeBackend` should exist at a time. Current usage is always sequential (BackendSelector creates one backend at a time). Document this constraint in the docstring.

### Rationale

Storing the instrumentor instance enables clean teardown. `BaseInstrumentor` handles idempotent `uninstrument()` internally, so no external `_instrumented` flag is needed — simpler code.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Global instrumentor (module-level) | Multiple `ClaudeBackend` instances would conflict. Instance-scoped is safer. |
| No explicit teardown (rely on GC) | `uninstrument()` restores original functions — must be called explicitly for clean lifecycle (FR-007). |
| Add `_instrumented` flag | Redundant with `BaseInstrumentor._is_instrumented_by_opentelemetry`. Checking `self._instrumentor is not None` is sufficient. |

---

## R-005: Span Hierarchy Alignment

### Decision

No special action needed. OTel context propagation handles hierarchy automatically.

### Findings

**Current span hierarchy:**
```
holodeck.cli.test (or .chat, .serve)   ← created by CLI layer
  └── invoke_agent <agent-name>                  ← created by instrumentation package
        ├── execute_tool <tool-name>             ← created by PreToolUse hook
        ├── execute_tool <tool-name>             ← created by PreToolUse hook
        └── ...
```

**How it works:**
1. CLI layer creates parent span (`holodeck.cli.test`) via `tracer.start_as_current_span()`.
2. `ClaudeBackend.invoke_once()` runs within that span's context.
3. The instrumentation package's wrapped `query()` creates `invoke_agent` span using the tracer obtained from the passed `tracer_provider`. Since it runs within the active context, the `invoke_agent` span automatically becomes a child of the parent span.
4. Tool hooks create `execute_tool` child spans under `invoke_agent`.

**No suppression needed:** `ClaudeBackend` does NOT use `agent_factory.py` (SK-only), so there is no competing `holodeck.agent.invoke` span. The instrumentation package's `invoke_agent` span is the only invocation-level span for Claude.

**Retries:** Each retry in `invoke_once()` calls `query()` separately, producing a separate `invoke_agent` span per attempt — all children of the parent CLI span. This is correct behavior (operators see each retry as a distinct invocation).

### Rationale

OTel's context propagation model makes this zero-effort. The key requirement is passing HoloDeck's `TracerProvider` (not global) so spans flow to the correct exporters.

---

## R-006: Coexistence with otel_bridge.py

### Decision

No changes to `otel_bridge.py`. Both mechanisms coexist.

### Findings

**Two complementary telemetry paths:**

| Mechanism | Scope | Purpose |
|-----------|-------|---------|
| `otel_bridge.py` (env vars) | Claude subprocess | Internal subprocess telemetry (Claude Code's own spans/metrics) |
| Instrumentation package | HoloDeck main process | Agent invocation traces (GenAI semconv spans wrapping `query()`) |

The env vars from `otel_bridge.py` go into `ClaudeAgentOptions.env` and are read by the Claude subprocess at startup. The instrumentation package wraps the `query()` function in the main process. These operate at different layers and produce complementary trace data.

**No conflict:** The instrumentation package's hooks are injected into the SDK's `query()` options, which are passed to the subprocess. The subprocess doesn't know about the main-process instrumentation. The main process doesn't know about the subprocess's internal OTel setup.

### Rationale

Preserving both paths gives operators the richest possible observability: HoloDeck-level GenAI spans (from the package) plus subprocess-internal telemetry (from env vars).

---

## R-007: Current `claude-agent-sdk` Version

### Decision

No version change needed. HoloDeck already pins `claude-agent-sdk==0.1.44`.

### Findings

- `pyproject.toml` line 50: `"claude-agent-sdk==0.1.44"`
- Instrumentation package requires: `claude-agent-sdk >= 0.1.44`
- Current pin satisfies the requirement.

### Rationale

No action needed.
