# Implementation Plan: Async Tool Initialization Endpoints

**Branch**: `025-tool-init-endpoints` | **Date**: 2026-03-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/025-tool-init-endpoints/spec.md`

## Summary

Extend `holodeck serve` with three protocol-agnostic REST endpoints (`POST /tools/{tool-name}/init`, `GET /tools/{tool-name}/init`, `GET /tools`) that allow operators to asynchronously pre-warm vectorstore and hierarchical_document tools before the first chat request. A new in-memory job orchestration layer manages initialization lifecycle (pending → in_progress → completed/failed), enforces configurable concurrency limits, tracks progress, and instruments all events with OpenTelemetry. The existing `initialize_tools()` pipeline is reused for the actual ingestion/indexing work.

Additionally, the tool config `source` field is extended to accept remote URIs (`s3://`, `az://`, `https://`) for container-based production deployments where data sources are not co-located with the agent. A `SourceResolver` transparently downloads remote data to a temp directory before passing it to the existing ingestion pipeline. See [design-remote-sources.md](./design-remote-sources.md) for full details.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI (existing), Pydantic v2 (existing), asyncio (existing), OpenTelemetry (existing), httpx>=0.27 (new core dep), boto3>=1.42.0 (optional extra `[s3]`, aligns with existing deploy-aws), azure-storage-blob>=12.19 (optional extra `[azure-blob]`)
**Storage**: In-memory (dict-based job registry, consistent with existing SessionStore pattern)
**Testing**: pytest with `-n auto`, markers: `@pytest.mark.unit`, `@pytest.mark.integration`
**Target Platform**: Linux/macOS server (uvicorn)
**Project Type**: Single project (existing `src/holodeck/` structure)
**Performance Goals**: POST endpoint response within 500ms (SC-001); status polling reflects state changes within 2s (SC-002)
**Constraints**: Max 3 concurrent init jobs (configurable, FR-013); in-memory state only (no persistence across restarts)
**Scale/Scope**: Single-server deployment; job count bounded by number of vectorstore/hierarchical_document tools in agent config

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Assessment |
|-----------|--------|------------|
| **I. No-Code-First** | ✅ PASS | Endpoints are server-managed infrastructure — no user Python code required. Tool configs remain YAML-only. |
| **II. MCP for API Integrations** | ✅ N/A | These are internal server endpoints, not external API integrations. No MCP server needed. |
| **III. Test-First with Multimodal Support** | ✅ PASS | Unit + integration tests will cover all endpoints, status transitions, edge cases, and error conditions. |
| **IV. OpenTelemetry-Native** | ✅ PASS | FR-014 explicitly requires OTel instrumentation for init job lifecycle events and embedding model calls. |
| **V. Evaluation Flexibility** | ✅ N/A | Feature does not involve evaluation metrics. |
| **Architecture Constraints** | ✅ PASS | Endpoints live in the Deployment Engine (serve layer), not Agent Engine. Tool initialization reuses Agent Engine via existing `initialize_tools()` — clean cross-engine boundary. |
| **Code Quality** | ✅ PASS | Python 3.10+, type hints, Google style, pytest markers, 80%+ coverage target. |

**Gate Result: PASS** — No violations. Proceed to Phase 0.

**Post-Design Re-check (Phase 1)**: ✅ PASS — Design artifacts (data-model.md, contracts/openapi.yaml, quickstart.md) introduce no new constitution concerns. Two new modules (`tool_init_manager.py`, `tool_init_routes.py`) stay within the Deployment Engine boundary. The `ToolInitManager` reuses `initialize_tools()` from the Agent Engine — no tight coupling. OTel instrumentation follows established codebase patterns (research R5). FR-015 adds a Pydantic model validator for tool name uniqueness — consistent with existing validation patterns.

## Project Structure

### Documentation (this feature)

```text
specs/025-tool-init-endpoints/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── openapi.yaml     # Tool init endpoint contracts
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── serve/
│   ├── server.py              # [MODIFY] Register tool init routes on main app, wire shutdown
│   ├── models.py              # [MODIFY] Add ToolInitStatus, ToolInfo, ProblemDetail response models
│   ├── tool_init_manager.py   # [NEW] Job orchestration: state machine, concurrency, progress, temp dir cleanup
│   └── tool_init_routes.py    # [NEW] FastAPI router with 3 endpoints
├── lib/
│   ├── source_resolver.py     # [NEW] URI-aware source resolution (local, S3, Azure Blob, HTTP)
│   └── tool_initializer.py    # [MODIFY] Add initialize_single_tool() wrapper for per-tool init
├── tools/
│   ├── common.py              # [MODIFY] resolve_source_path() delegates to SourceResolver
│   └── vectorstore_tool.py    # [MODIFY] Add progress_callback parameter to initialize()
└── models/
    └── tool.py                # [MODIFY] source field validation accepts URIs (s3://, az://, https://)

tests/
├── unit/
│   ├── lib/
│   │   └── test_source_resolver.py   # [NEW] Resolution, download mocking, scheme detection
│   └── serve/
│       ├── test_tool_init_manager.py  # [NEW] Job lifecycle, concurrency, cancellation
│       └── test_tool_init_routes.py   # [NEW] Endpoint responses, error codes, validation
└── integration/serve/
    └── test_tool_init.py              # [NEW] End-to-end init flow with real tools
```

**Structure Decision**: Single project, extending existing packages. Three key new modules: `source_resolver.py` (data access), `tool_init_manager.py` (orchestration), `tool_init_routes.py` (HTTP layer). Cloud SDK dependencies are optional extras with lazy imports.

## Verification Notes

*Added after speckit.verify pass on 2026-03-23.*

### Codebase Gaps Requiring Implementation Changes

1. **Per-tool initialization**: `initialize_tools()` in `tool_initializer.py:148` initializes ALL tools at once. A new `initialize_single_tool(agent, tool_name, force_ingest, progress_callback)` wrapper must be added to extract and initialize a single tool by name.
2. **Progress callback**: `VectorStoreTool.initialize()` and `HierarchicalDocumentTool.initialize()` have no progress reporting mechanism. An optional `progress_callback: Callable[[int, int], None] | None` parameter must be added to both, called after each file is processed with `(processed, total)`.
3. **Shutdown integration**: The server does NOT use FastAPI lifespan events. `ToolInitManager.shutdown()` must be called from `AgentServer.stop()`, following the existing `SessionStore.stop_cleanup_task()` pattern.

### Error Response Format

All error responses from init endpoints must use the existing RFC 7807 ProblemDetail format (`application/problem+json`), not the simple `{"detail": "..."}` format. The OpenAPI contract has been updated accordingly.

### Security Notes

- **Authentication/authorization is not a HoloDeck concern.** Init endpoints inherit the server's auth posture (currently none). Following container ecosystem patterns, auth should be implemented outside HoloDeck via reverse proxy, service mesh, or API gateway. Operators deploying to untrusted networks must secure access externally.
- **Error detail sanitization**: `error_detail` in InitJobResponse MUST be sanitized before inclusion. Strip file system paths to relative paths, redact API keys/tokens from embedding provider errors, never include stack traces. Full details are logged server-side only.

### Performance Notes

- **SC-004 limitation**: The pre-warming latency benefit (SC-004) requires a persistent vector store backend (Qdrant, PostgreSQL, Azure AI Search). `_needs_reingest()` queries the external collection to detect already-indexed data. For in-memory vector stores, pre-warming has no effect since data isn't shared across tool instances — each chat session creates its own in-memory store.
- **Embedding provider rate limits**: Concurrent init jobs share the same embedding provider. Operators should set `max_concurrent` appropriately for their provider's rate limits. The default of 3 is conservative.
- **Remote source size limits**: Default `max_source_size_bytes` is 1 GB per init job. S3/Azure resolvers compute total size from object listing before download; HTTP resolver checks Content-Length header or enforces limit via streaming byte counter. Exceeding the limit fails the job before or during download with a descriptive error.

### State Machine Clarification

- `POST /tools/{tool-name}/init` returns 409 Conflict for BOTH `pending` and `in_progress` states (not just `in_progress`). A job that's been accepted but hasn't started is still "in progress" from the client's perspective.
- Tools with no init job return `init_status: null` in the `GET /tools` listing (not `not_started` as mentioned in spec US3 scenario 2). The spec should be aligned during implementation.

### Miscellaneous

- CLAUDE.md claims "Tool system supports 6 types" but the actual `ToolUnion` in `models/tool.py:884-891` has 5 types. Separate housekeeping fix needed.

## Complexity Tracking

> No constitution violations — table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
