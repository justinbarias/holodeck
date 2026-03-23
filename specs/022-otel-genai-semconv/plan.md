# Implementation Plan: Integrate OTel GenAI Instrumentation into Claude Backend

**Branch**: `022-otel-genai-semconv` | **Date**: 2026-03-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/022-otel-genai-semconv/spec.md`

## Summary

Integrate the external `otel-instrumentation-claude-agent-sdk` package (v0.0.3) into HoloDeck's Claude backend so that all Claude Agent SDK invocations automatically produce GenAI semantic convention spans and metrics. The integration is thin: HoloDeck calls `instrument()` during `ClaudeBackend.initialize()` with the already-initialized `TracerProvider` and `MeterProvider`, and calls `uninstrument()` during `teardown()`. All GenAI semconv compliance is owned by the external package.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- `otel-instrumentation-claude-agent-sdk>=0.0.3,<0.1.0` (new — optional extra, capped at 0.1.0 due to alpha instability)
- `claude-agent-sdk==0.1.44` (already at required version)
- `opentelemetry-sdk>=1.20.0,<2.0.0` (existing)
- `opentelemetry-api>=1.12` (transitive via instrumentation package)

**Storage**: N/A
**Testing**: pytest with `@pytest.mark.unit`, `@pytest.mark.asyncio`, `-n auto`
**Target Platform**: Linux/macOS (Python 3.10+)
**Project Type**: Single project (existing holodeck monorepo)
**Performance Goals**: Zero overhead when observability disabled; no measurable latency impact when enabled (instrumentation adds only OTel span creation per `query()` call)
**Constraints**: Optional dependency — HoloDeck must never crash if package is not installed
**Scale/Scope**: ~5 files modified, ~1 new test file, ~120 LOC of integration code + ~250 LOC of tests (unit + integration)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | PASS | No YAML schema changes needed. Existing `observability` config block drives activation. |
| II. MCP for API Integrations | N/A | No API integration; this is an OTel instrumentation integration. |
| III. Test-First with Multimodal Support | PASS | Tests written for all activation/deactivation/degradation scenarios. |
| IV. OpenTelemetry-Native Observability | PASS | Core alignment — this feature extends OTel coverage to Claude backend with GenAI semantic conventions. |
| V. Evaluation Flexibility with Model Overrides | N/A | No evaluation changes. |

**Architecture Constraints**: PASS — changes stay within the Agent Engine (backend layer). No cross-engine coupling introduced.

## Project Structure

### Documentation (this feature)

```text
specs/022-otel-genai-semconv/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/lib/backends/
├── claude_backend.py       # MODIFIED — add instrument()/uninstrument() calls
├── otel_bridge.py          # UNCHANGED — subprocess env vars coexist
└── __init__.py             # UNCHANGED — no new public exports needed

src/holodeck/lib/observability/
├── providers.py            # MODIFIED — add get_observability_context() accessor
└── __init__.py             # MODIFIED — export get_observability_context

tests/unit/lib/backends/
└── test_claude_backend.py  # MODIFIED — add instrumentation lifecycle tests

tests/integration/
└── test_claude_instrumentation.py  # NEW — span hierarchy integration test

pyproject.toml              # MODIFIED — add optional dependency group + uv lock
```

**Structure Decision**: Single project, minimal surface area. All changes stay within the existing `lib/backends/` and `lib/observability/` modules.

## Constitution Check — Post-Design Re-evaluation

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | PASS | No new YAML fields. Existing `observability` block is sufficient. Users enable GenAI instrumentation by setting `observability.enabled: true` + `observability.traces.enabled: true` — pure YAML, no Python. |
| II. MCP for API Integrations | N/A | No external API integration introduced. |
| III. Test-First with Multimodal Support | PASS | Test plan covers: activation (US1), graceful degradation (US2), teardown (US3), span hierarchy (US4), all edge cases. |
| IV. OpenTelemetry-Native Observability | PASS | Direct implementation of this principle — adds GenAI semantic convention compliance to Claude backend via external package. |
| V. Evaluation Flexibility with Model Overrides | N/A | No evaluation changes. |

**Architecture Constraints**: PASS — All changes within Agent Engine. `get_observability_context()` accessor is a read-only view of existing state, not a new coupling path.

**No gate violations.** Design proceeds.

## Design Decisions Summary

| Decision | Choice | Reference |
|----------|--------|-----------|
| Import strategy | Deferred import inside `_activate_instrumentation()` (try/except ImportError) | R-003 |
| Provider access | `get_observability_context()` accessor on module-level state (with thread-safety docstring) | R-002 |
| Instrumentor lifecycle | Instance-scoped `self._instrumentor` only — no `_instrumented` flag; rely on `BaseInstrumentor` idempotency for `uninstrument()` safety | R-004 |
| Activation isolation | Extract to `_activate_instrumentation()` private method — prevents `return` from short-circuiting `initialize()` | Review #2 |
| Span hierarchy | Automatic via OTel context propagation — no special code needed | R-005 |
| otel_bridge coexistence | No changes — complementary mechanisms at different layers | R-006 |
| Dependency version | `claude-agent-sdk==0.1.44` already satisfies `>=0.1.44` requirement (pre-satisfied) | R-007 |
| Version ceiling | `>=0.0.3,<0.1.0` — cap at 0.1.0 to limit alpha-stage breaking changes | Review R1 |
| Optional dependency group | `[claude-otel]` extras group in `pyproject.toml` + `uv lock` | R-003 |
| Error handling | Catch all exceptions from `instrument()`, log warning, continue | R-004 |
| Global scope constraint | Only one active instrumented ClaudeBackend at a time (process-global monkey-patch) | Review #4 |
| Integration testing | One `InMemorySpanExporter` test verifying span parent-child hierarchy | Review R3 |

## Complexity Tracking

No violations — this is a thin integration layer with no new abstractions.
