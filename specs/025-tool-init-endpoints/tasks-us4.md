# Tasks — US4: Re-initialize a Tool

**Feature**: 025-tool-init-endpoints
**Story**: US4 — Re-initialize a Tool (P3)
**Depends on**: US1 must be complete (POST endpoint, ToolInitManager, InitJob state machine)

---

## Prerequisites from US1

US4 extends the POST handler and ToolInitManager created in US1. The following must already exist:

- `src/holodeck/serve/tool_init_routes.py` — POST `/tools/{tool-name}/init` endpoint
- `src/holodeck/serve/tool_init_manager.py` — `ToolInitManager` with `start_init_job()`, `_jobs` dict, `_tasks` dict, `_active_count`, state machine
- `src/holodeck/serve/models.py` — `InitJob`, `InitJobState`, `InitJobResponse`
- Shutdown wiring in `server.py` calling `ToolInitManager.shutdown()`

---

## Tasks

- [ ] T001 [US4] Extend `ToolInitManager.start_init_job()` to detect re-init scenarios — `src/holodeck/serve/tool_init_manager.py`
  - When `_jobs[tool_name]` exists and state is `pending` or `in_progress`, raise a domain error that maps to 409 Conflict. **Note**: `force=true` does NOT override 409 for pending/in_progress jobs. Force only controls whether re-ingestion skips the index-exists check when replacing a completed/failed job.
  - When `_jobs[tool_name]` exists and state is `completed` or `failed`, discard the previous `InitJob` and create a fresh one (state=pending, new `created_at`)
  - Remove the stale entry from `_tasks` if present (for failed jobs where the task is done)
  - Ensure `_active_count` is not decremented for the replaced completed/failed job (it should already be 0 for terminal states)

- [ ] T002 [US4] **Verify** that `force: bool = Query(False)` query parameter exists on POST endpoint — `src/holodeck/serve/tool_init_routes.py`. US1 T027 already adds this parameter and threads it through `ToolInitManager.start_init_job(tool_name, force)` → `InitJob.force` → `initialize_single_tool(force_ingest=force)`. Confirm the full chain works; if any link is missing, add it.

- [ ] T003 [US4] **Verify** that `start_init_job()` accepts `force: bool` and propagates it to the background task — `src/holodeck/serve/tool_init_manager.py`. US1 T010/T011 already implement this. Confirm `job.force` is stored on the `InitJob` and `initialize_single_tool(force_ingest=job.force)` is called in `_run_init_job()`. If missing, add it.

- [ ] T004 [US4] Return 409 Conflict with RFC 7807 ProblemDetail body for active jobs — `src/holodeck/serve/tool_init_routes.py`
  - Catch the domain error raised by T001 in the POST handler
  - Return 409 with `application/problem+json` body: type, title ("Initialization already active"), status 409, detail including tool name and current state
  - Ensure the `Location` header is NOT set on 409 responses (only on 201)

- [ ] T005 [US4] Verify graceful shutdown cancels re-init jobs — `src/holodeck/serve/tool_init_manager.py`
  - No new code expected: US1 shutdown wiring iterates `_tasks` and cancels all pending/in-progress tasks
  - Verify that re-init jobs (replacement jobs created by T001) are properly tracked in `_tasks` so shutdown can find and cancel them
  - If the replaced job's asyncio.Task handle was not cleaned up in T001, ensure it is removed from `_tasks` before inserting the new task handle
