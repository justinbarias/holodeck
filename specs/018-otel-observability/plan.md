# Implementation Plan: OpenTelemetry Observability with Semantic Conventions

**Branch**: `018-otel-observability` | **Date**: 2026-01-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/018-otel-observability/spec.md`

## Summary

Enable OpenTelemetry-based observability for HoloDeck agents with support for OTLP, Prometheus, and Azure Monitor exporters. The implementation leverages Semantic Kernel's native telemetry instrumentation and follows OpenTelemetry GenAI semantic conventions. Users configure observability through YAML without writing code, consistent with HoloDeck's no-code-first principle.

**Service Name Convention**: The OpenTelemetry service name defaults to `"holodeck-{agent.name}"` (e.g., `"holodeck-customer-support"`), derived from the agent's `name` field. Users can optionally override this with `observability.service_name`.

## Technical Context

**Language/Version**: Python 3.10+ (per constitution and CLAUDE.md)
**Primary Dependencies**:
- `semantic-kernel` (existing) - provides native OpenTelemetry instrumentation
- `opentelemetry-sdk` - core OpenTelemetry SDK
- `opentelemetry-exporter-otlp-proto-grpc` - OTLP gRPC exporter
- `opentelemetry-exporter-prometheus` - Prometheus metrics exporter
- `azure-monitor-opentelemetry-exporter` - Azure Monitor exporter
- `prometheus-client` - HTTP server for Prometheus scraping

**Storage**: N/A (telemetry is exported, not stored locally)
**Testing**: pytest with async support, mocked exporters for unit tests
**Target Platform**: Linux/macOS/Windows (CLI tool)
**Project Type**: Single project (extends existing HoloDeck CLI)
**Performance Goals**: <5% overhead on agent response time, ~100 req/min, ~10K spans/hour
**Constraints**: Bounded buffer (2048 spans/5MB max), oldest-first drop policy
**Scale/Scope**: Medium scale deployment, header-based auth only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. No-Code-First Agent Definition** | PASS | Observability configured via YAML `observability:` section |
| **II. MCP for API Integrations** | N/A | No external API integrations; uses standard OTel exporters |
| **III. Test-First with Multimodal Support** | PASS | Test cases defined in spec; observability applies to all agent types |
| **IV. OpenTelemetry-Native Observability** | PASS | This feature implements the principle directly |
| **V. Evaluation Flexibility with Model Overrides** | N/A | Not an evaluation feature |

**Architecture Constraints Check**:
- Agent Engine: Enhanced with observability instrumentation
- Evaluation Framework: Unaffected
- Deployment Engine: Unaffected
- Cross-engine communication: Via well-defined telemetry contracts

**Code Quality Check**:
- Python 3.10+: PASS
- Google Style Guide: Enforced via Black/Ruff
- MyPy strict mode: Required for new code
- pytest markers: Will use @pytest.mark.unit and @pytest.mark.integration
- 80% minimum coverage: Required

## Project Structure

### Documentation (this feature)

```text
specs/018-otel-observability/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── observability.py           # NEW: Pydantic models for observability config
├── lib/
│   └── observability/             # NEW: Observability module
│       ├── __init__.py
│       ├── config.py              # Configuration loader/validator
│       ├── providers.py           # Tracer/Meter/Logger provider setup
│       ├── exporters/
│       │   ├── __init__.py
│       │   ├── console.py         # Console exporter (default when no others configured)
│       │   ├── otlp.py            # OTLP exporter setup
│       │   ├── prometheus.py      # Prometheus exporter setup
│       │   └── azure_monitor.py   # Azure Monitor exporter setup
│       ├── instrumentation.py     # Semantic Kernel instrumentation hooks
│       └── semantic_conventions.py # GenAI attribute constants
├── cli/
│   └── commands/
│       └── (existing commands)    # MODIFY: Add observability initialization
├── chat/
│   └── (existing files)           # MODIFY: Integrate telemetry context
└── serve/
    └── (existing files)           # MODIFY: Integrate telemetry for REST API requests

tests/
├── unit/
│   └── lib/
│       └── observability/         # NEW: Unit tests
│           ├── test_config.py
│           ├── test_providers.py
│           ├── test_exporters.py
│           └── test_console_default.py  # Console as default behavior
└── integration/
    └── observability/             # NEW: Integration tests
        └── test_telemetry_flow.py
```

**Structure Decision**: Single project structure, extending existing HoloDeck codebase with new `lib/observability/` module.

## Complexity Tracking

> No violations - all constitution principles satisfied.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | - | - |
