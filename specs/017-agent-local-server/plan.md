# Implementation Plan: Agent Local Server

**Branch**: `017-agent-local-server` | **Date**: 2025-12-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/017-agent-local-server/spec.md`

## Summary

Implement a local HTTP server for HoloDeck agents supporting two protocols:
1. **AG-UI Protocol** (default) - Event-driven protocol for real-time agent UIs using `ag-ui-protocol` SDK
2. **REST Protocol** - Simple JSON/SSE API at `/agent/<agent-name>/chat` endpoints

The server exposes a single agent via `holodeck serve` CLI command with configurable port, protocol, CORS, and debugging options. Sessions are managed in-memory with 30-minute TTL expiration.

**Multimodal Support**: Both protocols support file uploads (images, PDFs, Office docs) via:
- AG-UI: Native `BinaryInputContent` with base64 or URL references
- REST: JSON with base64-encoded files or multipart form-data uploads

Files are processed by HoloDeck's existing `FileProcessor` (OCR, text extraction) before being sent to the agent.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI 0.115+, Uvicorn 0.34+, ag-ui-protocol 0.1.10+
**Storage**: In-memory (session store with TTL cleanup)
**Testing**: pytest with pytest-asyncio for async tests
**Target Platform**: Local development (Linux, macOS, Windows)
**Project Type**: Single project (extends existing HoloDeck structure)
**Performance Goals**: <2s first token latency, 100 concurrent sessions
**Constraints**: <200ms health check response, graceful shutdown <5s
**Scale/Scope**: Single agent per server instance (multi-agent via orchestration in future)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. No-Code-First Agent Definition
- ✅ **PASS**: Server exposes existing YAML-configured agents. No new code required from users.

### II. MCP for API Integrations
- ✅ **PASS**: Server does not introduce new API integrations. Uses existing MCP tool infrastructure.

### III. Test-First with Multimodal Support
- ✅ **PASS**: Feature includes contract tests (OpenAPI), integration tests for endpoints, and unit tests for session management.

### IV. OpenTelemetry-Native Observability
- ⚠️ **DEFERRED**: Basic logging implemented. Full OpenTelemetry instrumentation deferred to observability feature.
- **Justification**: Observability is explicitly out of scope per spec assumptions ("production deployment features addressed in separate specification").

### V. Evaluation Flexibility with Model Overrides
- ✅ **N/A**: This feature does not add evaluation metrics.

### Architecture Constraints
- ✅ **PASS**: Server is part of Deployment Engine, decoupled from Agent Engine and Evaluation Framework.
- Server uses AgentExecutor (Agent Engine) through clean interface.

### Code Quality & Testing Discipline
- ✅ **PASS**: Python 3.10+, pytest, MyPy strict, 80% coverage target.

## Project Structure

### Documentation (this feature)

```text
specs/017-agent-local-server/
├── plan.md              # This file
├── research.md          # Phase 0 output (complete)
├── data-model.md        # Phase 1 output (complete)
├── quickstart.md        # Phase 1 output (complete)
├── contracts/
│   ├── openapi.yaml     # REST API contract (complete)
│   └── sse-events.md    # SSE event contract (complete)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── serve/                          # NEW: Server module
│   ├── __init__.py
│   ├── server.py                   # AgentServer class (FastAPI app factory)
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── base.py                 # Protocol ABC
│   │   ├── agui.py                 # AG-UI protocol adapter
│   │   └── rest.py                 # REST protocol adapter
│   ├── session_store.py            # In-memory session management
│   ├── middleware.py               # Logging, CORS, error handling
│   └── models.py                   # Request/Response Pydantic models
├── cli/
│   └── commands/
│       └── serve.py                # NEW: CLI command
├── chat/                           # EXISTING: Reuse AgentExecutor
│   ├── executor.py
│   └── session.py
└── models/                         # EXISTING: Extend as needed

tests/
├── unit/
│   └── serve/
│       ├── test_session_store.py
│       ├── test_models.py
│       └── test_protocols.py
├── integration/
│   └── serve/
│       ├── test_server_rest.py
│       ├── test_server_agui.py
│       └── test_health_endpoints.py
└── contract/
    └── serve/
        └── test_openapi_contract.py
```

**Structure Decision**: Follows existing HoloDeck single-project structure with new `serve/` module. Reuses existing `chat/executor.py` for agent execution.

## Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
    # ... existing ...
    "fastapi>=0.115.0,<1.0.0",
    "uvicorn[standard]>=0.34.0,<1.0.0",
    "ag-ui-protocol>=0.1.10,<1.0.0",
]
```

## Key Integration Points

| Component | Interface | Notes |
|-----------|-----------|-------|
| `AgentExecutor` | `execute_turn(message) -> AgentResponse` | Existing chat executor |
| `Agent` | Pydantic model from `agent.yaml` | Existing config loader |
| `ConfigLoader` | `load(path) -> dict` | Existing YAML loader |
| `logging_config` | `get_logger(__name__)` | Existing structured logging |

## Complexity Tracking

> No constitution violations require justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Phase 1 Artifacts Generated

- [x] `research.md` - Technology decisions and patterns
- [x] `data-model.md` - Entity definitions and relationships
- [x] `contracts/openapi.yaml` - REST API specification
- [x] `contracts/sse-events.md` - Streaming event contract
- [x] `quickstart.md` - Developer usage guide

## Next Steps

Run `/speckit.tasks` to generate implementation tasks from this plan.
