# Tasks: US3 - List All Tools and Their Init Status

**Feature Branch**: `025-tool-init-endpoints`
**User Story**: US3 (P2) - List All Tools and Their Init Status
**Date**: 2026-03-24

## Dependencies

> US1 foundational work MUST be complete before starting US3. Specifically:
> - `ToolInfoResponse`, `ToolListResponse` response models in `src/holodeck/serve/models.py`
> - `ToolInitManager` class in `src/holodeck/serve/tool_init_manager.py`
> - Tool init router registered in `src/holodeck/serve/server.py`
> - `InitJobState` enum in `src/holodeck/serve/models.py`

## Tasks

- [ ] T001 [US3] Add parameterless `get_all_tool_statuses() -> list[ToolInfoResponse]` method to `ToolInitManager` in `src/holodeck/serve/tool_init_manager.py`. Iterates `self._agent.tools` internally, derives `supports_init = type in ("vectorstore", "hierarchical_document")` for each tool, and cross-references `self._jobs` for `init_status`. Returns `init_status: null` when no `InitJob` exists for a tool (per data-model.md and openapi.yaml — NOT the spec's `not_started`). Import `ToolInfoResponse` from `holodeck.serve.models`.
- [ ] T002 [P] [US3] Add `GET /tools` route handler in `src/holodeck/serve/tool_init_routes.py`. Access `ToolInitManager` via `request.app.state.tool_init_manager` (same DI pattern as US1/US2 handlers). Call `manager.get_all_tool_statuses()` and return `ToolListResponse(tools=statuses, total=len(statuses))` with 200 OK. Ensure the `total` field is set to the length of the tools list.
- [ ] T003 [P] [US3] Run `make format && make lint-fix` to format and lint all modified files in `src/holodeck/serve/`.
- [ ] T004 [US3] Run `make type-check` and fix any MyPy errors in `src/holodeck/serve/tool_init_manager.py` and `src/holodeck/serve/tool_init_routes.py`.
- [ ] T005 [US3] Run `make test` to verify no regressions across the entire codebase.
