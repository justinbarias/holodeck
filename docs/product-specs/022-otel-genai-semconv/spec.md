# Feature Specification: Integrate OTel GenAI Instrumentation into Claude Backend

**Feature Branch**: `022-otel-genai-semconv`
**Created**: 2026-02-28
**Updated**: 2026-03-01 (v0.0.3 + claude-agent-sdk 0.1.44)
**Status**: Draft
**Input**: User description: "Integrate the external `otel-instrumentation-claude-agent-sdk` package (https://pypi.org/project/otel-instrumentation-claude-agent-sdk/, https://github.com/justinbarias/opentelemetry-instrumentation-claude-agent-sdk) into HoloDeck's Claude backend so that all Claude Agent SDK invocations automatically produce GenAI semantic convention spans and metrics, properly nested under HoloDeck's existing span hierarchy."

## Clarifications

### Session 2026-02-28

- Q: Where does the instrumentation package live (monorepo vs separate repo)? → A: Separate repository (https://github.com/justinbarias/opentelemetry-instrumentation-claude-agent-sdk). Published to PyPI as `otel-instrumentation-claude-agent-sdk` (prerelease v0.0.2). HoloDeck consumes it as an external dependency.
- Q: How does the package intercept Claude Agent SDK calls (monkey-patch vs hook injection vs wrapper)? → A: Monkey-patch with auto-injected hooks (standard OTel Instrumentor pattern). `instrument()` wraps `query()` and `ClaudeSDKClient.__init__()` to automatically merge instrumentation hooks into user-provided `ClaudeAgentOptions`.
- Q: What is the hook merge strategy when users already have hooks for the same event type? → A: Append after user hooks. Instrumentation hooks run last, observing final state after user modifications, never interfering with user permission/security decisions.

### Session 2026-03-01

- Spec scope revised: the instrumentation package now exists as an independent external package with its own spec. This spec focuses exclusively on HoloDeck's integration of that package. All GenAI semconv compliance, span attribute population, hook registration, tool/subagent tracing, metrics emission, and content capture gating are owned by the external package.
- Version bump: `otel-instrumentation-claude-agent-sdk` v0.0.3 released — adds hook-driven tool execution tracing (PreToolUse, PostToolUse, PostToolUseFailure). Requires `claude-agent-sdk>=0.1.44` (up from 0.1.37). Spec and `pyproject.toml` updated accordingly.
- Q: Does `holodeck.agent.invoke` span apply to Claude backend invocations? → A: No. `holodeck.agent.invoke` is created in `agent_factory.py`, which is only used by `SKBackend`. `ClaudeBackend` has its own invocation path and never creates this span. FR-010 (span suppression) removed as unnecessary.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Activate Instrumentation During Backend Initialization (Priority: P1)

A platform operator configures a Claude agent with observability enabled. When `holodeck test`, `holodeck chat`, or the serve API processes a request, HoloDeck activates the external instrumentation package during `ClaudeBackend.initialize()`, passing HoloDeck's `TracerProvider`, `MeterProvider`, agent name, and content capture preference. All subsequent Claude Agent SDK calls produce GenAI semantic convention spans that are proper children of HoloDeck's existing parent spans.

**Why this priority**: This is the core integration point. Without it, the external instrumentation package is never activated and no GenAI spans or metrics are produced within HoloDeck.

**Independent Test**: Can be tested by initializing a `ClaudeBackend` with observability enabled and an in-memory OTel exporter, running an invocation, and verifying that (a) `instrument()` was called with the correct parameters and (b) spans produced by the instrumentation package appear as children of the active HoloDeck parent span.

**Acceptance Scenarios**:

1. **Given** a Claude agent with `observability.enabled: true` and `observability.traces.enabled: true`, **When** `ClaudeBackend.initialize()` runs, **Then** `ClaudeAgentSdkInstrumentor().instrument()` is called with `tracer_provider` from HoloDeck's observability context, `agent_name` from the agent config, and `capture_content` from `observability.traces.capture_content`.
2. **Given** a Claude agent with `observability.metrics.enabled: true`, **When** `ClaudeBackend.initialize()` runs, **Then** `instrument()` is called with `meter_provider` from HoloDeck's observability context.
3. **Given** a Claude agent with `observability.metrics.enabled: false`, **When** `ClaudeBackend.initialize()` runs, **Then** `instrument()` is called without a `meter_provider` (or with `None`).
4. **Given** observability is disabled (`observability.enabled: false` or `observability` is `None`), **When** `ClaudeBackend.initialize()` runs, **Then** `instrument()` is NOT called and zero OTel overhead is added.
5. **Given** traces are disabled but observability is enabled (`observability.traces.enabled: false`), **When** `ClaudeBackend.initialize()` runs, **Then** `instrument()` is NOT called (metrics-only mode does not trigger span instrumentation).

---

### User Story 2 - Graceful Degradation When Package Not Installed (Priority: P1)

A HoloDeck user who has not installed the optional `otel-instrumentation-claude-agent-sdk` package runs `holodeck test` with a Claude agent. The test executes normally without crashing. If observability is enabled, a warning is logged indicating that GenAI instrumentation is unavailable, but all other functionality — including existing HoloDeck spans and subprocess OTel env vars — continues to work.

**Why this priority**: HoloDeck must never crash due to a missing optional dependency. Graceful degradation is a prerequisite for shipping the integration as an optional extra.

**Independent Test**: Can be tested by mocking the import of the instrumentation package to raise `ImportError`, running `ClaudeBackend.initialize()` with observability enabled, and verifying that (a) no exception is raised, (b) a warning is logged, and (c) the backend remains functional.

**Acceptance Scenarios**:

1. **Given** the `otel-instrumentation-claude-agent-sdk` package is NOT installed, **When** `ClaudeBackend.initialize()` runs with observability enabled, **Then** a warning is logged and the backend initializes successfully without GenAI instrumentation.
2. **Given** the package is NOT installed, **When** a Claude invocation executes, **Then** it completes successfully with no GenAI spans (but existing HoloDeck spans and `otel_bridge.py` env vars still function).
3. **Given** the package is installed but `instrument()` raises an unexpected error (e.g., version incompatibility), **When** `ClaudeBackend.initialize()` runs, **Then** the error is caught, a warning is logged, and the backend initializes without GenAI instrumentation.

---

### User Story 3 - Clean Deactivation on Teardown (Priority: P2)

When a HoloDeck backend is torn down (e.g., test run complete, chat session ended, serve process stopped), the instrumentation is cleanly deactivated so that subsequent Claude Agent SDK usage is not unexpectedly instrumented, and no OTel resources leak.

**Why this priority**: Clean lifecycle management prevents state leakage between test runs and ensures predictable behavior in long-running processes.

**Independent Test**: Can be tested by initializing a `ClaudeBackend`, tearing it down, and verifying that `uninstrument()` was called and subsequent `query()` calls do NOT produce instrumented spans.

**Acceptance Scenarios**:

1. **Given** a `ClaudeBackend` has been initialized with instrumentation active, **When** `teardown()` is called, **Then** `uninstrument()` is called on the instrumentor.
2. **Given** a `ClaudeBackend` was initialized WITHOUT instrumentation (observability disabled or package not installed), **When** `teardown()` is called, **Then** no error is raised (uninstrument is skipped or is a safe no-op).
3. **Given** multiple sequential test runs with different agents, **When** each run initializes and tears down its backend, **Then** instrumentation is scoped to each run's lifetime with no span leakage between runs.

---

### User Story 4 - Span Hierarchy Alignment (Priority: P2)

When the instrumentation package is active, the GenAI `invoke_agent` spans produced by the package nest correctly under HoloDeck's existing parent spans across all entry points (`holodeck test`, `holodeck chat`, serve API). Note: `ClaudeBackend` has its own invocation path and does not use `agent_factory.py`, so there is no `holodeck.agent.invoke` span to suppress — the instrumentation package's spans are the only invocation-level spans for Claude.

**Why this priority**: Correct span hierarchy is what makes the traces navigable. Without it, operators see orphaned spans or confusing nesting.

**Independent Test**: Can be tested by running a Claude invocation under each HoloDeck entry point and verifying that the `invoke_agent` span is a child of the correct parent span.

**Acceptance Scenarios**:

1. **Given** `holodeck test` is running with observability enabled, **When** a Claude invocation occurs, **Then** the `invoke_agent` span is a child of the `holodeck.cli.test` span (via OTel context propagation).
2. **Given** `holodeck chat` is running with observability enabled, **When** a Claude invocation occurs, **Then** the `invoke_agent` span is a child of the `holodeck.cli.chat` span.
3. **Given** the serve API is handling a request with observability enabled, **When** a Claude invocation occurs, **Then** the `invoke_agent` span is a child of the `holodeck.cli.serve` span.

---

### Edge Cases

- What happens when the `otel-instrumentation-claude-agent-sdk` package is not installed? HoloDeck logs a warning and continues without GenAI instrumentation — it does not crash.
- What happens when the package version is incompatible with the installed Claude Agent SDK version? HoloDeck catches the initialization error, logs a warning, and continues uninstrumented.
- What happens when both `otel_bridge.py` env vars and the instrumentation package are active? They coexist — subprocess-internal telemetry (from env vars) and HoloDeck-level instrumentation (from the package) produce complementary traces.
- What happens when observability is enabled but no `TracerProvider` or `MeterProvider` is configured in HoloDeck's observability context? The instrumentation package uses OTel API no-op behavior — zero overhead, no errors.
- What happens during retries in `invoke_once()`? Each retry calls `query()`, and the instrumentation wraps each `query()` call, so operators see multiple `invoke_agent` spans per `invoke_once()` when retries occur.
- What happens when `instrument()` is called multiple times without `uninstrument()`? The instrumentation package is idempotent — subsequent calls are no-ops.

## Requirements *(mandatory)*

### Functional Requirements

**Dependency Management**

- **FR-001**: HoloDeck MUST add `otel-instrumentation-claude-agent-sdk>=0.0.3,<0.1.0` as an optional dependency (e.g., in a `[claude-otel]` or `[observability]` extras group), so users who do not need Claude GenAI instrumentation are not burdened with the transitive dependency. *(Note: `claude-agent-sdk` is already pinned at `==0.1.44`, which satisfies the instrumentation package's `>=0.1.44` requirement — no upgrade needed.)*
- **FR-002**: HoloDeck MUST handle the case where `otel-instrumentation-claude-agent-sdk` is not installed by logging a warning and continuing without GenAI instrumentation. The Claude backend MUST remain fully functional without the instrumentation package.

**Instrumentation Activation**

- **FR-003**: HoloDeck MUST activate the instrumentation package during `ClaudeBackend.initialize()` by calling `ClaudeAgentSdkInstrumentor().instrument()` when observability is enabled and traces are enabled.
- **FR-004**: HoloDeck MUST pass the following configuration to `instrument()`:
  - `tracer_provider`: The `TracerProvider` from HoloDeck's observability context (not the global provider), to ensure spans flow to HoloDeck's configured exporters.
  - `meter_provider`: The `MeterProvider` from HoloDeck's observability context, when metrics are enabled. Omitted (or `None`) when metrics are disabled.
  - `agent_name`: The agent's name from `agent.name`.
  - `capture_content`: The value of `observability.traces.capture_content` from the agent's configuration.
- **FR-005**: HoloDeck MUST NOT activate the instrumentation package when observability is disabled (`observability.enabled: false` or `observability` is `None`), ensuring zero overhead in the non-observability path.
- **FR-006**: HoloDeck MUST NOT activate the instrumentation package when traces are disabled (`observability.traces.enabled: false`), even if observability is globally enabled.

**Instrumentation Deactivation**

- **FR-007**: HoloDeck MUST call `uninstrument()` during `ClaudeBackend.teardown()` to cleanly remove instrumentation hooks and prevent state leakage between backend instances.
- **FR-008**: The deactivation MUST be safe to call even if `instrument()` was never called (no-op), and MUST NOT raise exceptions.

**Span Hierarchy**

- **FR-009**: Claude backend spans produced by the instrumentation package MUST be proper children of existing HoloDeck parent spans (`holodeck.cli.test`, `holodeck.cli.chat`, `holodeck.cli.serve`) by virtue of OTel context propagation — the instrumentation package creates spans within the active context, which HoloDeck's parent spans establish. Note: `ClaudeBackend` does not use `agent_factory.py` (SK-only), so no `holodeck.agent.invoke` span exists to suppress.

**Coexistence with otel_bridge.py**

- **FR-011**: HoloDeck MUST continue passing OTel env vars to the Claude subprocess via `otel_bridge.py` alongside the instrumentation package activation. These serve complementary purposes: subprocess-internal telemetry (env vars) vs. HoloDeck-level agent invocation traces (instrumentation package).

**No GenAI Logic in HoloDeck**

- **FR-012**: HoloDeck MUST NOT contain any GenAI semantic convention attribute names, span name patterns, or metric instrument definitions in its own codebase. All GenAI semconv compliance is owned by the external instrumentation package.

### Key Entities

- **ClaudeAgentSdkInstrumentor**: The public entry point of the external instrumentation package (`otel-instrumentation-claude-agent-sdk`). Provides `instrument()` / `uninstrument()` methods. HoloDeck calls these during backend lifecycle.
- **ObservabilityContext**: HoloDeck's existing dataclass holding `tracer_provider`, `meter_provider`, `logger_provider`, `exporters`, and `resource`. The `tracer_provider` and `meter_provider` are passed to the instrumentor.
- **ObservabilityConfig**: HoloDeck's existing Pydantic model for observability settings. The `traces.capture_content` flag controls whether the instrumentation package captures sensitive content in span attributes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: When observability is enabled and the package is installed, Claude backend invocations produce GenAI spans visible in any OTel-compatible tracing backend.
- **SC-002**: Claude backend spans nest correctly under HoloDeck parent spans (`holodeck.cli.test`, `holodeck.cli.chat`, `holodeck.cli.serve`) with no orphaned spans.
- **SC-003**: When observability is disabled, zero OTel overhead is added — no spans, no metrics, no import of the instrumentation package at module load time.
- **SC-004**: HoloDeck remains fully functional when the `otel-instrumentation-claude-agent-sdk` package is not installed — graceful degradation with a logged warning.
- **SC-005**: Sequential test runs with different agents show clean instrumentation lifecycle — no span leakage between runs.
- **SC-006**: The `instrument()` call receives the correct `tracer_provider`, `meter_provider`, `agent_name`, and `capture_content` values derived from HoloDeck's configuration.

## Assumptions

- The external `otel-instrumentation-claude-agent-sdk` package (v0.0.3+) provides `ClaudeAgentSdkInstrumentor` with `instrument(tracer_provider, meter_provider, agent_name, capture_content)` and `uninstrument()` methods. v0.0.3 adds hook-driven tool execution tracing (PreToolUse, PostToolUse, PostToolUseFailure spans).
- The instrumentation package requires `claude-agent-sdk>=0.1.44`. HoloDeck already pins `claude-agent-sdk==0.1.44` in `pyproject.toml`, satisfying this requirement — no upgrade needed.
- The instrumentation package handles all GenAI semantic convention compliance internally — span naming, attribute population, hook registration, tool/subagent tracing, metrics emission, content capture gating, and crash cleanup. HoloDeck is not responsible for any of these.
- HoloDeck's existing `ObservabilityContext` (from `initialize_observability()`) provides `tracer_provider` and `meter_provider` instances that can be passed directly to the instrumentation package.
- The instrumentation package uses OTel context propagation, so spans automatically nest under any active parent span in the calling context.
- The `otel_bridge.py` env-var approach for subprocess telemetry will be preserved alongside the instrumentation package — they are complementary, not competing.
- The instrumentation package's `instrument()` is idempotent — calling it multiple times is safe.
- The instrumentation package is at alpha maturity (v0.0.3). HoloDeck's integration should be resilient to minor API changes in the package.
