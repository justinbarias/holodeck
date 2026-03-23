# Research: Async Tool Initialization Endpoints

**Feature Branch**: `025-tool-init-endpoints`
**Date**: 2026-03-23

## R1: Background Job Execution Pattern

**Decision**: Use `asyncio.create_task()` with a custom `ToolInitManager` class that tracks task handles and job state.

**Rationale**: FastAPI `BackgroundTasks` is fire-and-forget — no handle for cancellation, no state tracking, no concurrency limiting. `asyncio.Queue` adds complexity for a producer-consumer pattern when the requirement is to reject (429), not queue. `asyncio.create_task()` gives us handles for cancellation, done callbacks for state transitions, and integrates naturally with the existing codebase pattern (see `SessionStore._cleanup_task`).

**Alternatives considered**:
- FastAPI `BackgroundTasks`: No task handle, no cancellation, no concurrency control — rejected
- `asyncio.Queue` with workers: Queues work rather than rejecting — wrong semantics for FR-013 (429 on limit)
- Celery / external task queue: Overkill for in-memory single-server scope — rejected

## R2: Concurrency Limiting (FR-013)

**Decision**: Manual `int` counter on the manager class. Check-then-increment is atomic in asyncio's single-threaded cooperative model as long as no `await` appears between the check and increment.

**Rationale**: `asyncio.Semaphore` blocks (waits) until a slot is free — we need immediate rejection with 429 instead. A manual counter with synchronous check-and-increment is race-free in asyncio without locks, provided no suspension point (`await`) exists between the check and the mutation.

**Alternatives considered**:
- `asyncio.Semaphore`: Blocks instead of rejecting — wrong semantics
- `asyncio.BoundedSemaphore` with `_value` check: Accesses private attribute — fragile
- `asyncio.Lock` + counter: Unnecessary complexity — single-threaded event loop makes this redundant

## R3: Graceful Shutdown (FR-009)

**Decision**: Wire `ToolInitManager.shutdown()` into `AgentServer.stop()`, following the existing explicit start/stop lifecycle pattern. On shutdown, cancel all in-progress tasks via `task.cancel()`, await them with `asyncio.gather(*tasks, return_exceptions=True)`, and mark interrupted jobs as `failed`.

**Rationale**: The existing `AgentServer` does NOT use FastAPI lifespan events. It uses explicit `start()` and `stop()` methods called externally by the CLI/uvicorn runner. `SessionStore.stop_cleanup_task()` follows this same pattern — cancel the task, await it, handle `CancelledError`. The `ToolInitManager` integrates identically: `AgentServer.stop()` calls `await self._tool_init_manager.shutdown()` alongside the existing session cleanup.

**Alternatives considered**:
- FastAPI lifespan context manager: AgentServer doesn't use lifespan events — would require architectural change — rejected
- Custom `signal.signal()` handlers: Conflict with uvicorn's signal handling — rejected
- `atexit` handlers: Don't work with async code — rejected

## R4: Progress Reporting (FR-011)

**Decision**: Direct mutation of a shared `InitJob` dataclass (or Pydantic model) from the background task. HTTP GET handlers read the same object. No locks needed.

**Rationale**: asyncio runs on a single thread. Dict/attribute assignment is not an `await` point, so a background task updating `job.documents_processed = 42` and an HTTP handler reading it cannot race. This is the idiomatic asyncio pattern. The existing `SessionStore` uses the same approach — `dict[str, ServerSession]` mutated from cleanup tasks and read from HTTP handlers.

**Alternatives considered**:
- `asyncio.Lock`: Unnecessary — single-threaded event loop guarantees no concurrent access
- `asyncio.Queue` for progress events: Over-engineered — direct mutation is simpler and sufficient
- Thread-safe structures (`threading.Lock`): Only needed if `run_in_executor()` is used — not applicable here

## R5: OpenTelemetry Instrumentation (FR-014)

**Decision**: Follow existing codebase patterns exactly. Use `get_tracer(__name__)` for span creation, `holodeck.serve.tool_init` as the span namespace, and the conditional `nullcontext()` pattern when observability is disabled.

**Rationale**: The codebase has established OTel patterns in `middleware.py`, `deepeval/base.py`, and CLI commands. Span naming follows `holodeck.<layer>.<operation>`. Attributes use dot-namespaced keys. Error handling sets `StatusCode.ERROR` on spans. The `if span:` guard prevents failures when observability is off.

**Key patterns to follow**:
- Span names: `holodeck.serve.tool_init.start`, `holodeck.serve.tool_init.progress`, `holodeck.serve.tool_init.complete`
- Attributes: `tool_init.job.tool_name`, `tool_init.job.state`, `tool_init.job.documents_processed`, `tool_init.job.duration_ms`
- Import pattern: `from holodeck.lib.observability import get_tracer`
- Conditional: `nullcontext()` when `observability_enabled` is False

## R6: Tool Name Uniqueness Validation (FR-015)

**Decision**: Add a `@model_validator(mode="after")` to the `Agent` class in `src/holodeck/models/agent.py` that checks tool name uniqueness across all tool types at config load time.

**Rationale**: Currently, the `validate_tools()` validator only checks max tool count (50). MCP duplicate detection exists (`_check_mcp_duplicate()` in `config/loader.py`) but only for MCP tools added via CLI commands — not at config load time and not across all tool types. The Agent model validator is the right place because it runs automatically when the Pydantic model is instantiated, which happens at config load time for all code paths (CLI, serve, tests).

**Alternatives considered**:
- Config validator (`config/validator.py`): Only contains error formatting utilities — wrong location
- Config loader (`config/loader.py`): Would need to be called explicitly — model validators run automatically
- Separate validation pass: Redundant when Pydantic validators exist for this exact purpose

## R7: Reusing `initialize_tools()` Pipeline

**Decision**: Call `initialize_tools()` from `tool_initializer.py` within the background job, passing `force_ingest` from the `?force` query parameter. The init endpoint initializes tools in isolation — it does NOT share instances with the chat flow.

**Rationale**: The spec clarifies that "pre-warming validates config and ingests/indexes documents, but the chat flow still initializes its own tool instances. The latency benefit comes from data already being ingested [in the vector store], not from reusing the init endpoint's tool instance." The existing `_needs_reingest()` mtime check in `VectorStoreTool.initialize()` will detect already-indexed data and skip re-ingestion, delivering the SC-004 latency benefit automatically.

**Key considerations**:
- `initialize_tools()` currently initializes ALL tools at once. A new `initialize_single_tool()` wrapper is needed to target a single tool by name — see Verification Notes in plan.md.
- `VectorStoreTool.initialize()` and `HierarchicalDocumentTool.initialize()` lack progress callbacks. An optional `progress_callback: Callable[[int, int], None] | None` parameter must be added to both.
- `force_ingest=True` maps to the `?force` query parameter (same behavior as `--force-ingest` CLI flag)
- Each init job creates its own embedding service — no shared state concerns
- The Claude backend has a TODO for `force_ingest` passthrough — this feature doesn't depend on fixing that since init endpoints call `initialize_tools()` directly, not through the backend
- **SC-004 limitation**: The pre-warming latency benefit only applies when using a persistent vector store backend (Qdrant, PostgreSQL, Azure AI Search). For in-memory stores, `_needs_reingest()` queries the collection which doesn't persist across tool instances.

## R8: Endpoint Registration Strategy (FR-012)

**Decision**: Create a FastAPI `APIRouter` in `tool_init_routes.py` and include it in the main app inside `AgentServer.create_app()`, alongside the existing health/ready routes. The router is protocol-agnostic.

**Rationale**: Health/ready endpoints are already registered directly on the main app (not inside protocol-specific routers). Tool init endpoints follow the same pattern — they are operational infrastructure, not chat protocol. Using an `APIRouter` keeps the code organized while allowing the server module to include it with a single `app.include_router()` call.

**Alternatives considered**:
- Register routes directly in `server.py`: Increases file size; harder to test in isolation — rejected
- Protocol-specific registration: Contradicts FR-012 requirement for protocol-agnostic endpoints — rejected

## R9: Remote Source Resolution (FR-016 – FR-021)

**Decision**: Extend the `source` field to accept remote URIs (`s3://`, `az://`, `https://`). A new `SourceResolver` module transparently downloads remote data to a temp directory. The existing ingestion pipeline operates on the local copy unchanged.

**Rationale**: Container-based production deployments store data in cloud storage, not packaged with the agent image. Transparent resolution keeps the existing pipeline unchanged — everything downstream of `resolve_source_path()` sees a local path. The `source` field becomes a URI-aware string rather than adding a separate `data_source` config block, maintaining the simple YAML experience.

**Architecture**: `SourceResolver.resolve(source, base_dir) → ResolvedSource(local_path, is_remote, temp_dir)`. Provider-specific resolvers (S3, Azure Blob, HTTP) handle download. Temp dir cleaned up on job completion/failure/shutdown.

**Alternatives considered**:
- Explicit `data_source` config block: More structured but creates two ways to specify data, violates simplicity — rejected
- POST body source override: Breaks "YAML is single source of truth", security risk (URI injection on unauthenticated endpoint) — rejected
- Sidecar/init-container pattern: K8s-specific, pushes complexity to deployment manifest — rejected as the only approach (but still valid as a user deployment pattern)

**Dependencies**:
- `httpx`: Added to core deps (lightweight async HTTP client for HTTP resolver)
- `boto3`: Optional extra `[s3]` — lazy import with clear error if missing
- `azure-storage-blob`: Optional extra `[azure-blob]` — lazy import with clear error if missing

**Credentials**: Follow cloud-native environment variable conventions. No secrets in YAML config. AWS uses standard credential chain; Azure uses connection string or managed identity; HTTP uses optional auth header env var.

**Temp dir lifecycle**: Created per-init-job, cleaned up on completion/failure/shutdown. No persistent cache (future optimization).
