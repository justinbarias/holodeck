# Tasks: US1 - Trigger Tool Initialization

**Feature Branch**: `025-tool-init-endpoints`
**User Story**: US1 (P1) - Trigger Tool Initialization
**Date**: 2026-03-24
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md)

**User Story**: A HoloDeck user running `holodeck serve` triggers the initialization (ingestion/indexing) of a vector-store-based tool via `POST /tools/{tool-name}/init`. The server immediately returns 201 Created with a `Location` header and response body containing an href to a status resource. Initialization proceeds in the background.

**Acceptance Scenarios**:
1. POST /tools/{tool-name}/init for vectorstore tool returns 201 Created with Location header and status href.
2. POST /tools/{tool-name}/init for hierarchical_document tool returns 201 Created.
3. POST /tools/{tool-name}/init for non-initializable tool type returns 400 Bad Request.
4. POST /tools/{tool-name}/init for nonexistent tool returns 404 Not Found.

**FR Coverage**: FR-001, FR-002, FR-005, FR-006, FR-007, FR-009, FR-010, FR-011, FR-012, FR-013, FR-014, FR-015, FR-016, FR-017, FR-018, FR-019, FR-020, FR-021

---

## Phase 1: Setup — Dependencies

These tasks add new project dependencies required by the feature. No story label — they are foundational.

- [x] T001 [P] Add `httpx>=0.27` to the core `dependencies` list in `pyproject.toml`. Place it after the existing `aiohttp` entry. This is the async HTTP client used by `HttpResolver` for remote source downloads.

- [x] T002 [P] Add optional dependency extras for cloud source resolvers in `pyproject.toml`. Add `s3 = ["boto3>=1.42.0"]` (reuses the same version constraint as `deploy-aws`), `azure-blob = ["azure-storage-blob>=12.19"]`, and `all-sources = ["boto3>=1.42.0", "azure-storage-blob>=12.19"]` to `[project.optional-dependencies]`. Place them after the existing `deploy-aws` entry.

- [x] T003 Run `uv sync` to install the new `httpx` core dependency into the virtual environment and verify no dependency conflicts.

---

## Phase 2: Foundational — Response Models & Data Types

These tasks create the shared data types and response models that all subsequent phases depend on.

- [x] T004 [P] Add `InitJobState` enum to `src/holodeck/serve/models.py`. Define `class InitJobState(str, Enum)` with values: `PENDING = "pending"`, `IN_PROGRESS = "in_progress"`, `COMPLETED = "completed"`, `FAILED = "failed"`. Place it after the existing `ServerState` enum.

- [x] T005 [P] Add `InitJobProgress` Pydantic model to `src/holodeck/serve/models.py`. Fields: `documents_processed: int`, `total_documents: int | None = None`. Place it after `InitJobState`.

- [x] T006 [P] Add `InitJobResponse` Pydantic model to `src/holodeck/serve/models.py`. Fields: `tool_name: str`, `state: InitJobState`, `href: str`, `created_at: datetime`, `started_at: datetime | None = None`, `completed_at: datetime | None = None`, `message: str | None = None`, `error_detail: str | None = None`, `progress: InitJobProgress | None = None`, `force: bool`. Import `datetime` from the `datetime` module.

- [x] T007 [P] Verify existing `ProblemDetail` model in `src/holodeck/serve/models.py` matches the RFC 7807 schema required by init endpoints (fields: `type: str = "about:blank"`, `title: str`, `status: int`, `detail: str | None`, `instance: str | None`). The model already exists (lines ~220-239). Confirm no changes needed; if any field is missing, add it. All error responses from init endpoints will use this format with `application/problem+json` content type.

- [x] T008 [P] Add `ToolInfoResponse` and `ToolListResponse` Pydantic models to `src/holodeck/serve/models.py`. `ToolInfoResponse` fields: `name: str`, `type: str`, `supports_init: bool`, `init_status: InitJobState | None = None`. `ToolListResponse` fields: `tools: list[ToolInfoResponse]`, `total: int`. These are used by US3 but defined now as foundational types.

---

## Phase 3: Foundational — Core Infrastructure & Source Resolution

These tasks build the core orchestration, tool initialization, and source resolution infrastructure. Source resolution tasks are included here because T011 (_run_init_job) depends on SourceResolver.

- [x] T009 Add `InitJob` dataclass to `src/holodeck/serve/tool_init_manager.py`. Fields: `tool_name: str`, `state: InitJobState`, `created_at: datetime`, `started_at: datetime | None = None`, `completed_at: datetime | None = None`, `message: str | None = None`, `error_detail: str | None = None`, `progress: InitJobProgress | None = None`, `force: bool = False`. Import `InitJobState` and `InitJobProgress` from `holodeck.serve.models`. Use `@dataclass` (not Pydantic) since this is mutable internal state mutated by background tasks (R4).

- [x] T010 Implement `ToolInitManager` class in `src/holodeck/serve/tool_init_manager.py` (FR-013). Constructor accepts `agent: Agent`, `max_concurrent: int = 3`. Internal state: `_jobs: dict[str, InitJob]`, `_tasks: dict[str, asyncio.Task]`, `_active_count: int = 0`, `_shutting_down: bool = False`. Implement `start_init_job(tool_name: str, force: bool = False) -> InitJob` method that: (a) validates tool exists in agent config and supports init (vectorstore or hierarchical_document), (b) checks `_shutting_down` flag, (c) rejects with 409-semantic if an active job exists for this tool (pending or in_progress), (d) rejects with 429-semantic if `_active_count >= max_concurrent`, (e) creates an `InitJob` in `pending` state, (f) creates an `asyncio.Task` wrapping `_run_init_job()`, (g) registers a done callback that decrements `_active_count` and transitions state. Return distinct exception types for 400/404/409/429 cases so the route handler can map them to HTTP status codes.

- [x] T011 Implement `_run_init_job()` async method on `ToolInitManager` in `src/holodeck/serve/tool_init_manager.py`. This method: (a) transitions job state to `in_progress`, sets `started_at`, increments `_active_count`, (b) resolves the tool source via `SourceResolver.resolve_context()` (context manager for guaranteed cleanup), (c) calls `initialize_single_tool()` with the agent, tool name, force flag, and a progress callback that updates `job.progress`, (d) on success, transitions to `completed`, sets `completed_at` and summary message, (e) on exception, transitions to `failed`, sets `error_detail` via `sanitize_error_detail()`, (f) on `asyncio.CancelledError`, transitions to `failed` with message "Cancelled due to server shutdown". Ensure `_active_count` is decremented in all code paths (finally block).

- [x] T012 Implement `get_job(tool_name: str) -> InitJob | None` and `shutdown()` methods on `ToolInitManager` in `src/holodeck/serve/tool_init_manager.py` (FR-009). `get_job()` returns the job for a tool or None. `shutdown()` sets `_shutting_down = True`, cancels all in-progress tasks via `task.cancel()`, awaits them with `asyncio.gather(*tasks, return_exceptions=True)`, marks interrupted jobs as `failed`, and cleans up temp directories. Follow the `SessionStore.stop_cleanup_task()` pattern (R3).

- [x] T013 [P] Add `initialize_single_tool()` function to `src/holodeck/lib/tool_initializer.py` (R7). Signature: `async def initialize_single_tool(agent: Agent, tool_name: str, force_ingest: bool = False, progress_callback: Callable[[int, int | None], None] | None = None, source_override: Path | None = None) -> None`. This wrapper: (a) finds the tool config by name from `agent.tools`, (b) raises `ToolInitializerError` if not found or not a vectorstore/hierarchical_document type, (c) creates an embedding service via the existing `_create_embedding_service()` logic, (d) instantiates and initializes the single tool, passing `force_ingest` and `progress_callback`. **Source override mechanism**: When `source_override` is provided, create a shallow copy of the tool config and set `copy.source = str(source_override)` before constructing the tool instance — do NOT mutate the original agent config. Reuse existing patterns from `initialize_tools()` but scoped to a single tool.

- [x] T014 [P] Add optional `progress_callback: Callable[[int, int | None], None] | None = None` parameter to `VectorStoreTool.initialize()` in `src/holodeck/tools/vectorstore_tool.py`. After each file is processed in the ingestion loop, call `progress_callback(processed_count, total_count)` if the callback is not None. The `total_count` is determined from the file discovery step before processing begins. Default is None (no-op, backward compatible).

- [x] T015 [P] Add optional `progress_callback: Callable[[int, int | None], None] | None = None` parameter to `HierarchicalDocumentTool.initialize()` in `src/holodeck/tools/hierarchical_document_tool.py`. Same pattern as T014 — call `progress_callback(processed_count, total_count)` after each document is processed. Default is None (backward compatible).

- [x] T016 [P] Add `@model_validator(mode="after")` named `validate_tool_name_uniqueness` to the `Agent` class in `src/holodeck/models/agent.py` (FR-015, R6). Note: the existing `validate_tools()` is a `@field_validator("tools")`, not a model validator — the new validator is a separate `@model_validator(mode="after")` method. If duplicate tool names are detected across all tool types, raise `ValueError` with a message listing the duplicates. This runs automatically at config load time for all code paths (CLI, serve, tests). Place it after the existing `validate_tools()` field validator.

- [x] T017 [P] Update source field validation in `src/holodeck/models/tool.py` to accept remote URIs (FR-016). The `source` field on `VectorstoreTool` and `HierarchicalDocumentToolConfig` must accept `s3://`, `az://`, `https://`, and `http://` scheme prefixes in addition to local paths. Add a field validator that checks for supported schemes (`s3://`, `az://`, `https://`, `http://`, `file://`, or no scheme for local paths) and rejects unsupported schemes with a `ValidationError`.

### Source Resolution (T018–T026)

- [x] T018 Create `src/holodeck/lib/source_resolver.py` with the `ResolvedSource` dataclass and `SourceResolver` class skeleton. `ResolvedSource` fields: `local_path: Path`, `is_remote: bool`, `temp_dir: Path | None`. `SourceResolver` exposes: `resolve()` (static async), `resolve_context()` (async context manager with guaranteed cleanup), `cleanup()` (static async), and `cleanup_orphans()` (static async). Implement `_detect_scheme(source: str) -> str` helper that returns `"local"`, `"s3"`, `"az"`, `"http"`, or raises `ValueError` for unsupported schemes.

- [x] T019 Implement `LocalResolver` in `src/holodeck/lib/source_resolver.py`. Wraps the existing `resolve_source_path()` logic from `src/holodeck/tools/common.py`: absolute paths used directly, relative paths resolved against `base_dir`. Returns `ResolvedSource(local_path=resolved, is_remote=False, temp_dir=None)`.

- [x] T020 [P] Implement `S3Resolver` in `src/holodeck/lib/source_resolver.py` (FR-016, FR-018, FR-019, FR-020). Lazy-imports `boto3` with a clear error if missing (`SourceError("S3 source requires boto3. Install with: pip install holodeck[s3]")`). Parses bucket and prefix from `s3://bucket/prefix/` URI. Lists objects via `list_objects_v2`, filters by supported extensions, validates total size against `max_source_size_bytes` (pre-download), validates each key with `validate_object_path()` for path traversal prevention, downloads to a temp directory preserving relative structure. Retries on transient failures (3 attempts with exponential backoff). Auth failures fail immediately (FR-021).

- [x] T021 [P] Implement `AzureBlobResolver` in `src/holodeck/lib/source_resolver.py` (FR-016, FR-018, FR-019, FR-020). Lazy-imports `azure-storage-blob` with a clear error if missing. Parses container and prefix from `az://container/prefix/` URI. Uses `AZURE_STORAGE_CONNECTION_STRING` env var or `DefaultAzureCredential`. Lists blobs via `list_blobs(name_starts_with=prefix)`, filters by supported extensions, validates total size, validates paths, downloads to temp directory. Same retry and error handling pattern as S3Resolver (FR-021).

- [x] T022 [P] Implement `HttpResolver` in `src/holodeck/lib/source_resolver.py` (FR-016, FR-020, FR-021). Uses `httpx` (core dependency) for async downloads. Single file downloads only (no archive extraction). Infers filename from URL path. Reads `Content-Length` header for pre-download size check; if absent, enforces `max_source_size_bytes` via streaming byte counter. Supports optional `HOLODECK_HTTP_AUTH_HEADER` env var for Authorization header. Retries HTTP 5xx and network errors (3 attempts, exponential backoff 1s/2s/4s). HTTP 4xx fails immediately with sanitized error.

- [x] T023 Implement `sanitize_error_detail()` utility function in `src/holodeck/lib/source_resolver.py`. Redacts: AWS access key IDs (`AKIA[A-Z0-9]{16}`), Azure connection strings (`AccountKey=[^;]+`), bearer tokens (`Bearer [A-Za-z0-9._\-]+`), generic API keys (`(api[_-]?key|token|secret|password)\s*[=:]\s*\S+`), and absolute temp dir paths (`/tmp/holodeck-init-[^/\s]+`). Returns the sanitized string. Full unredacted errors are logged server-side at ERROR level.

- [x] T024 Implement `validate_object_path()` utility function in `src/holodeck/lib/source_resolver.py`. Validates that an S3 object key or Azure blob name does not contain `..` and that the resolved download path stays within the temp directory (`Path.resolve().is_relative_to(temp_dir.resolve())`). Returns the validated `Path` or raises `ValueError`. Invalid keys are logged and skipped (do not fail the entire init job).

- [x] T025 Implement `cleanup_orphans()` static async method on `SourceResolver` in `src/holodeck/lib/source_resolver.py`. Scans `tempfile.gettempdir()` for directories matching `holodeck-init-*` prefix older than `max_age_hours` (default 1 hour). Removes them via `asyncio.to_thread(shutil.rmtree, ...)`. Returns count of removed directories. Called on server startup.

- [x] T026 Update `resolve_source_path()` in `src/holodeck/tools/common.py` to delegate to `SourceResolver` for URI-scheme sources. If the source string starts with `s3://`, `az://`, `https://`, or `http://`, raise an informational error directing callers to use `SourceResolver.resolve_context()` instead (since remote resolution requires async context management). Local path resolution continues unchanged.

---

## Phase 4: US1 — Endpoint & Server Integration

These tasks implement the POST endpoint, route registration, shutdown wiring, and OTel instrumentation.

- [ ] T027 [US1] Create `src/holodeck/serve/tool_init_routes.py` with a FastAPI `APIRouter` (R8). Define `POST /tools/{tool_name}/init` route handler that: (a) accepts `tool_name` path parameter and optional `force: bool = False` query parameter, (b) calls `ToolInitManager.start_init_job(tool_name, force)`, (c) on success, returns 201 Created with `Location` header set to `/tools/{tool_name}/init` and `InitJobResponse` body, (d) catches distinct exception types from `ToolInitManager` and returns `ProblemDetail` responses: 400 for non-initializable tool type, 404 for nonexistent tool, 409 for already-active job, 429 for concurrency limit reached. All error responses use `application/problem+json` content type.

- [ ] T028 [US1] Register the tool init router in `src/holodeck/serve/server.py`. In `AgentServer.create_app()`, import the router from `tool_init_routes.py` and call `app.include_router(router)` alongside the existing health/ready route registrations (FR-012 — protocol-agnostic, not inside protocol-specific routers). Pass the `ToolInitManager` instance to the router via `app.state.tool_init_manager`.

- [ ] T029 [US1] Create and wire `ToolInitManager` instance in `AgentServer.__init__()` at `src/holodeck/serve/server.py`. Instantiate `ToolInitManager(agent=self.agent_config)` and store as `self._tool_init_manager`. Optionally accept `max_concurrent_init_jobs: int = 3` constructor parameter on `AgentServer` for configurability.

- [ ] T030 [US1] Wire `ToolInitManager.shutdown()` into `AgentServer.stop()` at `src/holodeck/serve/server.py` (FR-009, R3). Add `await self._tool_init_manager.shutdown()` call before the existing session cleanup. Follow the same pattern as `await self.sessions.stop_cleanup_task()`. Also call `await SourceResolver.cleanup_orphans()` during server startup in `AgentServer.start()` to clean up stale temp directories from previous runs.

- [ ] T031 [US1] Add OTel instrumentation for init job lifecycle in `src/holodeck/serve/tool_init_manager.py` (FR-014, R5). Import `get_tracer` from `holodeck.lib.observability`. Create spans: `holodeck.serve.tool_init.start` (when job is created), `holodeck.serve.tool_init.progress` (on progress updates, if frequent enough), `holodeck.serve.tool_init.complete` (on success), `holodeck.serve.tool_init.failed` (on failure with `StatusCode.ERROR`). Add span attributes: `tool_init.job.tool_name`, `tool_init.job.state`, `tool_init.job.documents_processed`, `tool_init.job.duration_ms`, `tool_init.job.force`. Use the conditional `nullcontext()` pattern when observability is disabled.

---

## Phase 5: Polish — Quality & Validation

- [ ] T032 [US1] Run `make format && make lint-fix` to ensure all new and modified files pass Black formatting and Ruff linting.

- [ ] T033 [US1] Run `make type-check` and fix any MyPy errors in new and modified files. Ensure all public functions have complete type hints and all imports resolve correctly.

- [ ] T034 [US1] Run `make test-unit -n auto` to verify no existing tests are broken by the changes. Fix any regressions.

---

## Dependencies & Execution Order

```
Phase 1 (parallel):
  T001 (httpx dep) | T002 (cloud extras) — independent
  T003 (uv sync) — depends on T001, T002

Phase 2 (parallel, after Phase 1):
  T004 (InitJobState) | T005 (InitJobProgress) | T006 (InitJobResponse)
  | T007 (ProblemDetail) | T008 (ToolInfoResponse/ToolListResponse)
  — all independent additions to models.py

Phase 3 (core infrastructure + source resolution — merged):
  Core infrastructure:
    T009 (InitJob dataclass) — depends on T004, T005
    T010 (ToolInitManager core) — depends on T009
    T012 (get_job + shutdown) — depends on T010
    T013 | T014 | T015 | T016 | T017 — all independent [P], can start immediately
    T009 + T010 — sequential chain

  Source resolution (T018–T026):
    T018 (SourceResolver skeleton) — no dependencies
    T019 (LocalResolver) — depends on T018
    T020 [P] | T021 [P] | T022 [P] — independent resolvers, parallel after T018
    T023 | T024 | T025 | T026 — independent utilities, parallel

  Cross-group dependency:
    T011 (_run_init_job) — depends on T010, T013, T018 (all within Phase 3)

Phase 4 (after Phase 3):
  T027 (POST endpoint) — depends on T010, T012, T006, T007
  T028 (route registration) — depends on T027
  T029 (ToolInitManager wiring) — depends on T010
  T030 (shutdown + orphan cleanup wiring) — depends on T012, T025, T028, T029
  T031 (OTel instrumentation) — depends on T010, T011

  T028 + T029 can run in parallel (different sections of server.py)
  T030 depends on both T028 and T029
  T031 can run in parallel with T027-T030

Phase 5 (sequential, after all previous):
  T032 (format + lint) → T033 (type-check) → T034 (test suite)
```

## Key Files Reference

| File | Action | Tasks | FR Coverage |
|------|--------|-------|-------------|
| `pyproject.toml` | MODIFY | T001, T002 | FR-018 |
| `src/holodeck/serve/models.py` | MODIFY | T004, T005, T006, T007, T008 | FR-002, FR-004, FR-011 |
| `src/holodeck/serve/tool_init_manager.py` | NEW | T009, T010, T011, T012, T031 | FR-001, FR-007, FR-009, FR-013, FR-014 |
| `src/holodeck/serve/tool_init_routes.py` | NEW | T027 | FR-001, FR-002, FR-005, FR-006, FR-007, FR-010, FR-012, FR-013 |
| `src/holodeck/serve/server.py` | MODIFY | T028, T029, T030 | FR-009, FR-012 |
| `src/holodeck/lib/tool_initializer.py` | MODIFY | T013 | FR-001, FR-011 |
| `src/holodeck/tools/vectorstore_tool.py` | MODIFY | T014 | FR-011 |
| `src/holodeck/tools/hierarchical_document_tool.py` | MODIFY | T015 | FR-011 |
| `src/holodeck/models/agent.py` | MODIFY | T016 | FR-015 |
| `src/holodeck/models/tool.py` | MODIFY | T017 | FR-016 |
| `src/holodeck/lib/source_resolver.py` | NEW | T018, T019, T020, T021, T022, T023, T024, T025 | FR-016, FR-017, FR-018, FR-019, FR-020, FR-021 |
| `src/holodeck/tools/common.py` | MODIFY | T026 | FR-016 |

## Implementation Strategy

### Guiding Principles

1. **Bottom-up construction**: Response models and data types first, then orchestration layer, then source resolution, then HTTP endpoint, then server wiring. Each layer is testable in isolation before integration.

2. **Backward compatibility**: All changes to existing files are additive. `progress_callback=None` defaults preserve existing behavior. Source field validation is extended, not replaced. `initialize_tools()` continues to work unchanged alongside the new `initialize_single_tool()`.

3. **Error responses as contracts**: All error responses use RFC 7807 `ProblemDetail` format with `application/problem+json` content type. Error details are sanitized via `sanitize_error_detail()` — no file paths, API keys, or stack traces in client-facing responses.

4. **Concurrency without locks**: The `ToolInitManager` uses a manual int counter for concurrency limiting (R2). Check-then-increment is atomic in asyncio's single-threaded cooperative model. No `await` between the check and the mutation.

5. **Guaranteed cleanup**: Remote source temp directories are cleaned up via `async with SourceResolver.resolve_context()` (finally block). Server shutdown cleans up any remaining temp dirs. Orphan recovery on startup handles SIGKILL scenarios.

6. **Optional cloud dependencies**: boto3 and azure-storage-blob are optional extras with lazy imports. Missing SDK produces a clear, actionable error message. httpx is a core dependency since it is lightweight and async-native.

7. **OTel follows existing patterns**: Span naming uses `holodeck.serve.tool_init.*` namespace. Attributes use dot-namespaced keys. Conditional `nullcontext()` when observability is disabled. No new OTel infrastructure — reuses `get_tracer(__name__)`.
