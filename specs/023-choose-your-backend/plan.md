# Implementation Plan: Choose Your Backend

**Branch**: `023-choose-your-backend` | **Date**: 2026-03-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/023-choose-your-backend/spec.md`

## Summary

Add Google ADK and Microsoft Agent Framework as two new execution backends alongside existing Semantic Kernel and Claude Agent SDK. This requires: extending the `ProviderEnum` with `google_adk` and `agent_framework` values, creating backend-specific Pydantic config models, implementing `AgentBackend`/`AgentSession` protocols for each new backend, building tool adapters for all 5 tool types, abstracting the embedding service behind a protocol to decouple `tool_initializer.py` from Semantic Kernel, and updating the `BackendSelector` routing logic. The existing chat executor and test executor require zero changes due to proper protocol abstraction.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: google-adk (pinned RC), agent-framework-core (pinned v1.0.0rc4), semantic-kernel (existing), claude-agent-sdk (existing)
**Storage**: N/A (in-memory session management for both new backends)
**Testing**: pytest with pytest-asyncio, pytest-mock, pytest-xdist (`-n auto`)
**Target Platform**: Linux/macOS (CLI tool)
**Project Type**: Single Python package with optional dependency groups
**Performance Goals**: Backend initialization < 5s, single-turn invocation latency dominated by upstream LLM, not adapter overhead
**Constraints**: New backends MUST NOT add import-time dependencies; lazy imports only. `semantic-kernel` remains a core dependency (embedding default adapter).
**Scale/Scope**: 2 new backends, ~12 new files, ~20 modified files, ~3000 lines of new code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First | PASS | All backend selection via YAML `model.provider` and backend-specific YAML sections. No Python code required from users. |
| II. MCP for API Integrations | PASS | Both new backends support MCP tools natively (ADK via `McpToolset`, AF via `MCPStdioTool`/`MCPStreamableHTTPTool`). No custom API tool types introduced. |
| III. Test-First with Multimodal | PASS | Existing test framework unchanged. New backends return `ExecutionResult` compatible with all evaluation metrics. |
| IV. OTel-Native Observability | DEFERRED | OTel instrumentation for new backends is out of scope for initial release. Both frameworks have varying OTel support. Will be addressed in a follow-up feature. |
| V. Evaluation Flexibility | PASS | New backends return `ExecutionResult` with token_usage, tool_calls, tool_results — fully compatible with existing 3-level evaluation model. |

**Gate Result**: PASS (OTel deferred is acceptable — it's additive, not a violation)

### Post-Phase 1 Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First | PASS | Data model confirms all new config is YAML-driven (`GoogleADKConfig`, `AgentFrameworkConfig` as optional Pydantic sections). Quickstart shows pure-YAML examples. |
| II. MCP for API Integrations | PASS | Tool adapter design (data-model §5-7) maps all MCP transports to native backend APIs. No custom API tool types introduced. |
| III. Test-First with Multimodal | PASS | Project structure includes unit tests for all new modules + integration tests per backend. `ExecutionResult` unchanged. |
| IV. OTel-Native Observability | DEFERRED | No change from pre-design. |
| V. Evaluation Flexibility | PASS | No change — `ExecutionResult` contract unchanged. |

**Post-Design Gate**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/023-choose-your-backend/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   ├── llm.py                          # MODIFY: Add GOOGLE_ADK, AGENT_FRAMEWORK to ProviderEnum
│   ├── agent.py                        # MODIFY: Add google_adk, agent_framework optional fields
│   ├── google_adk_config.py            # NEW: GoogleADKConfig Pydantic model
│   └── af_config.py                    # NEW: AgentFrameworkConfig Pydantic model
│
├── lib/
│   ├── tool_initializer.py             # MODIFY: Use EmbeddingService protocol instead of SK classes
│   ├── embedding_protocol.py           # NEW: EmbeddingService protocol + SK adapter
│   └── backends/
│       ├── __init__.py                 # MODIFY: Export new types
│       ├── base.py                     # NO CHANGE
│       ├── selector.py                 # MODIFY: Add routing for google_adk, agent_framework
│       ├── sk_backend.py               # NO CHANGE
│       ├── claude_backend.py           # NO CHANGE
│       ├── adk_backend.py              # NEW: ADKBackend + ADKSession
│       ├── adk_tool_adapters.py        # NEW: HoloDeck → ADK tool conversion
│       ├── af_backend.py               # NEW: AFBackend + AFSession
│       ├── af_tool_adapters.py         # NEW: HoloDeck → AF tool conversion
│       └── af_embedding_adapter.py     # NEW: AF embedding adapter behind protocol
│
├── chat/
│   ├── session.py                      # NO CHANGE
│   └── executor.py                     # NO CHANGE
│
└── cli/
    └── utils/
        └── wizard.py                   # MODIFY: Add google_adk, agent_framework to provider choices

tests/
├── unit/
│   ├── lib/
│   │   ├── backends/
│   │   │   ├── test_adk_backend.py     # NEW
│   │   │   ├── test_af_backend.py      # NEW
│   │   │   ├── test_adk_tool_adapters.py # NEW
│   │   │   ├── test_af_tool_adapters.py  # NEW
│   │   │   └── test_selector.py        # MODIFY: Add new provider routing tests
│   │   └── test_embedding_protocol.py  # NEW
│   └── models/
│       ├── test_google_adk_config.py   # NEW
│       └── test_af_config.py           # NEW
└── integration/
    ├── test_adk_integration.py         # NEW (requires google-adk installed)
    └── test_af_integration.py          # NEW (requires agent-framework installed)
```

**Structure Decision**: Follows existing single-project structure. New backend files are added under `lib/backends/` matching the established pattern (cf. `sk_backend.py`, `claude_backend.py`). New config models follow the `claude_config.py` pattern.

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |
