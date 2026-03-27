# Tasks: Poll Initialization Status (US2)

**Input**: Design documents from `/specs/025-tool-init-endpoints/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md, contracts/openapi.yaml, quickstart.md
**Tests**: NOT included in this task file. Tests will be addressed separately.
**Dependency**: US2 depends on US1 foundational work being complete. Specifically: `ToolInitManager` class with `_jobs` dict and job lifecycle management in `src/holodeck/serve/tool_init_manager.py`, response models (`InitJobResponse`, `ProblemDetail`, `InitJobState`, `InitJobProgress`) in `src/holodeck/serve/models.py`, `tool_init_routes.py` router registered on the main app in `src/holodeck/serve/server.py`, and the POST `/tools/{tool_name}/init` endpoint already functional. If any US1 artifacts are missing, STOP and complete US1 first.

**Organization**: Tasks are grouped by phase to enable incremental delivery. All tasks carry the [US2] label for traceability.

## Format: `[ID] [P?] [US2] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US2]**: Required on every task
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Verify US1 Prerequisites

**Purpose**: Confirm that US1 foundational infrastructure is in place before adding the GET endpoint.

- [x] T001 [US2] Verify US1 prerequisites are merged: confirm `ToolInitManager` class exists in `src/holodeck/serve/tool_init_manager.py` with `_jobs: dict[str, InitJob]` attribute, confirm `InitJobResponse` and `ProblemDetail` models exist in `src/holodeck/serve/models.py`, confirm `tool_init_routes.py` router is registered in `src/holodeck/serve/server.py`, and confirm POST `/tools/{tool_name}/init` handler exists in `src/holodeck/serve/tool_init_routes.py`. If any are missing, STOP and complete US1 first

**Checkpoint**: US1 infrastructure confirmed. Ready to add GET endpoint.

---

## Phase 2: Manager Layer — Job Lookup

**Purpose**: Ensure `ToolInitManager` exposes a method to retrieve a job by tool name, returning `None` when no job exists. US1 may already provide this; if so, this phase is a no-op verification.

- [x] T002 [US2] Check whether `ToolInitManager` in `src/holodeck/serve/tool_init_manager.py` already has a `get_job(tool_name: str) -> InitJob | None` method (or equivalent). If it exists, verify it returns the `InitJob` from `_jobs` dict by key, returning `None` when the key is absent. Mark this task complete and skip T003
- [x] T003 [US2] If `get_job()` does not exist, add it to `ToolInitManager` in `src/holodeck/serve/tool_init_manager.py`. Implementation: `return self._jobs.get(tool_name)`. Add Google-style docstring. Type hint the return as `InitJob | None`. No locking needed — single-threaded asyncio (design decision R4)

**Checkpoint**: `ToolInitManager.get_job()` available for the route handler.

---

## Phase 3: Route Handler — GET Endpoint

**Purpose**: Implement the `GET /tools/{tool_name}/init` endpoint in the existing routes module. This is the core of US2.

- [x] T004 [US2] Add the `get_tool_init_status` route handler to `src/holodeck/serve/tool_init_routes.py`. Implementation details:
    - Route: `@router.get("/tools/{tool_name}/init", response_model=InitJobResponse, responses={404: {"model": ProblemDetail, "content": {"application/problem+json": {}}}})`
    - Path parameter: `tool_name: str`
    - Access `ToolInitManager` via `request.app.state.tool_init_manager` (same DI pattern as the POST handler from US1 T027)
    - Call `manager.get_job(tool_name)` to retrieve the `InitJob`
    - If job is `None`, return a `JSONResponse` with status 404, `media_type="application/problem+json"`, and body matching `ProblemDetail` schema: `type="about:blank"`, `title="Not Found"`, `status=404`, `detail="No initialization job found for tool '{tool_name}'."`, `instance=f"/tools/{tool_name}/init"`
    - If job exists, convert `InitJob` to `InitJobResponse` (reuse the same conversion logic from US1's POST handler — extract to a helper if not already shared) and return with status 200
    - Add Google-style docstring referencing FR-003, FR-004

- [x] T005 [US2] Verify that the `InitJob` to `InitJobResponse` conversion is shared between the POST handler (US1) and the new GET handler. If US1 inlined the conversion in the POST handler, extract it to a private helper function `_job_to_response(job: InitJob) -> InitJobResponse` in `src/holodeck/serve/tool_init_routes.py` and update both handlers to use it. The helper must populate: `tool_name`, `state`, `href=f"/tools/{job.tool_name}/init"`, `created_at`, `started_at`, `completed_at`, `message`, `error_detail`, `progress`, `force`

**Checkpoint**: GET endpoint returns 200 with `InitJobResponse` or 404 with `ProblemDetail`. Manually verifiable via `curl http://localhost:8000/tools/{tool_name}/init`.

---

## Phase 4: Polish

**Purpose**: Code quality, formatting, and validation.

- [x] T006 [P] [US2] Run `make format` to format `src/holodeck/serve/tool_init_manager.py` and `src/holodeck/serve/tool_init_routes.py` with Black + Ruff
- [x] T007 [P] [US2] Run `make lint` and fix any Ruff + Bandit violations in `src/holodeck/serve/tool_init_manager.py` and `src/holodeck/serve/tool_init_routes.py`
- [x] T008 [US2] Run `make type-check` and fix any MyPy errors in `src/holodeck/serve/tool_init_manager.py` and `src/holodeck/serve/tool_init_routes.py` — ensure `get_job()` return type and route handler return type annotations are correct
- [x] T009 [US2] Run full test suite `make test` to verify no regressions across entire codebase

**Checkpoint**: All quality checks pass. US2 complete.

---

## Dependencies & Execution Order

### Cross-Story Dependencies

- **US1 (REQUIRED BEFORE US2)**: US2 depends on US1 for:
  - `ToolInitManager` class with `_jobs` dict in `src/holodeck/serve/tool_init_manager.py`
  - Response models (`InitJobResponse`, `InitJobState`, `InitJobProgress`, `ProblemDetail`) in `src/holodeck/serve/models.py`
  - Router registration in `src/holodeck/serve/server.py`
  - POST handler in `src/holodeck/serve/tool_init_routes.py` (the GET handler lives alongside it)
- **US2 is independent of US3** (tool listing) and US4 (re-initialization)

### Phase Dependencies

- **Phase 1 (Verify)**: No dependencies beyond US1 being merged
- **Phase 2 (Manager Layer)**: Depends on Phase 1 confirmation
- **Phase 3 (Route Handler)**: Depends on Phase 2 (`get_job()` available)
  - T005 depends on T004 (refactor after handler is written)
- **Phase 4 (Polish)**: Depends on Phase 3 completion
  - T006 and T007 can run in parallel
  - T008 depends on T006/T007 (format/lint first, then type-check)
  - T009 depends on T008

---

## Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/serve/tool_init_manager.py` | Phase 2 | Add `get_job()` method (if not already present from US1) |
| `src/holodeck/serve/tool_init_routes.py` | Phase 3 | Add GET `/tools/{tool_name}/init` handler, extract shared conversion helper |

---

## Notes

- [P] tasks = different files or independent checks, no dependencies between them
- [US2] label on every task for traceability
- This is a small, focused story: one new route handler + one manager method
- No OTel instrumentation needed for the GET endpoint (design decision R5: read-only, no extra spans)
- The GET handler reuses the same `InitJobResponse` model from US1 — no new response models needed
- The 404 response uses `application/problem+json` content type per RFC 7807 (see plan.md Verification Notes)
