# Tasks: US4 — Backend-Aware Health Checks

**Input**: Design documents from `/specs/024-claude-serve-deploy/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md (required), contracts/health-endpoint.md (required)
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.

**User Story**: An infrastructure operator monitors deployed Claude agents. The health check endpoint validates not just that the server process is running, but that Claude backend prerequisites are satisfied: Node.js available, credentials accessible, SDK subprocess can be spawned. Enables load balancers/orchestrators to route traffic only to genuinely ready instances.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US4]**: All user story tasks belong to User Story 4
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: Verify US1 dependencies are in place and understand the current health endpoint implementation.

- [ ] T001 Verify US1 has added `backend_ready: bool = True` and `backend_diagnostics: list[str] = []` fields to `HealthResponse` in `src/holodeck/serve/models.py`. If fields are missing, add them per data-model.md contract. Confirm the model serializes correctly with default values (existing consumers see no change)
- [ ] T002 Verify `validate_nodejs()` in `src/holodeck/lib/backends/validators.py` exists and is importable. Verify `validate_credentials()` in the same file exists and is importable. These are the validators that health checks will reuse
- [ ] T003 Verify the current health endpoint in `src/holodeck/serve/server.py` returns `HealthResponse` with `status`, `agent_name`, `agent_ready`, `active_sessions`, `uptime_seconds`. Understand the `_register_health_endpoints()` method structure

---

## Phase 2: Foundational — US1 Dependencies

**Purpose**: Ensure the HealthResponse model changes from US1 are in place. US1 adds the fields; US4 populates them with live checks.

- [ ] T004 If US1 has not yet added `backend_ready` and `backend_diagnostics` to `HealthResponse` in `src/holodeck/serve/models.py`, add them now:
  - `backend_ready: bool = Field(default=True, description="Whether backend prerequisites are satisfied")`
  - `backend_diagnostics: list[str] = Field(default_factory=list, description="Diagnostic messages (empty when healthy)")`
  - Confirm existing serialization tests still pass (new fields have defaults, so backward-compatible)

- [ ] T005 Verify that `ProviderEnum.ANTHROPIC` is accessible from `src/holodeck/models/llm.py` for provider detection in health check logic. Confirm `self.agent_config.model.provider` is the correct path to read the provider from `AgentServer`

**Checkpoint**: HealthResponse model has backend_ready and backend_diagnostics fields. Validators are importable. Provider detection path is confirmed.

---

## Phase 3: US4 Implementation — Backend Health Check Logic

**Purpose**: Implement `check_backend_health()` on `AgentServer`, wire it into the health endpoint, and add TTL-based caching to avoid per-request overhead.

### 3A: Core Health Check Method

> Implement `check_backend_health()` that reuses existing validators and returns backend readiness status.

#### Tests (TDD — write FIRST, verify they FAIL)

- [ ] T006 [P] [US4] Write unit test `test_check_backend_health_healthy_when_anthropic_prerequisites_met` in `tests/unit/serve/test_server.py` — create an `AgentServer` with `agent_config.model.provider == ProviderEnum.ANTHROPIC`. Mock `validate_nodejs()` to succeed (no exception). Mock `validate_credentials()` to return a valid env dict. Call `check_backend_health()`. Assert returns `(True, [])` — backend_ready is True, diagnostics list is empty

- [ ] T007 [P] [US4] Write unit test `test_check_backend_health_unhealthy_when_nodejs_missing` in `tests/unit/serve/test_server.py` — create an `AgentServer` with `agent_config.model.provider == ProviderEnum.ANTHROPIC`. Mock `validate_nodejs()` to raise `ConfigError`. Call `check_backend_health()`. Assert returns `(False, ["Node.js not found on PATH (required for Claude Agent SDK)"])` — backend_ready is False, diagnostics contains Node.js error message

- [ ] T008 [P] [US4] Write unit test `test_check_backend_health_degraded_when_credentials_invalid` in `tests/unit/serve/test_server.py` — create an `AgentServer` with `agent_config.model.provider == ProviderEnum.ANTHROPIC`. Mock `validate_nodejs()` to succeed. Mock `validate_credentials()` to raise `ConfigError` with credential message. Call `check_backend_health()`. Assert returns `(False, ["...credential error message..."])` — backend_ready is False, diagnostics contains credential error

- [ ] T009 [P] [US4] Write unit test `test_check_backend_health_skipped_for_non_anthropic_provider` in `tests/unit/serve/test_server.py` — create an `AgentServer` with `agent_config.model.provider == ProviderEnum.OPENAI`. Call `check_backend_health()`. Assert returns `(True, [])` — non-Anthropic providers skip backend validation entirely, always report healthy

- [ ] T010 [P] [US4] Write unit test `test_check_backend_health_collects_multiple_failures` in `tests/unit/serve/test_server.py` — create an `AgentServer` with `agent_config.model.provider == ProviderEnum.ANTHROPIC`. Mock `validate_nodejs()` to raise `ConfigError`. Mock `validate_credentials()` to also raise `ConfigError`. Call `check_backend_health()`. Assert returns `(False, [node_error_msg, credential_error_msg])` — both diagnostics collected, not short-circuited on first failure

#### Implementation

- [ ] T011 [US4] Implement `check_backend_health()` method on `AgentServer` in `src/holodeck/serve/server.py`:
  - Signature: `async def check_backend_health(self) -> tuple[bool, list[str]]`
  - If `self.agent_config.model.provider != ProviderEnum.ANTHROPIC`, return `(True, [])`
  - For Anthropic provider, call `validate_nodejs()` and `validate_credentials(self.agent_config.model)` inside try/except blocks
  - Collect ALL diagnostic messages (do not short-circuit on first failure)
  - Return `(backend_ready, diagnostics)` where `backend_ready = len(diagnostics) == 0`
  - Import `validate_nodejs` and `validate_credentials` from `holodeck.lib.backends.validators`
  - Import `ConfigError` from `holodeck.lib.errors` for the try/except blocks around validator calls
  - Import `ProviderEnum` from `holodeck.models.llm`
  - Run tests T006–T010 and verify they PASS

### 3B: TTL-Based Caching

> Cache health check results for 30 seconds to avoid per-request subprocess overhead from `node --version` calls.

#### Tests (TDD)

- [ ] T012 [P] [US4] Write unit test `test_check_backend_health_caches_result_for_30_seconds` in `tests/unit/serve/test_server.py` — create an `AgentServer` with Anthropic provider. Mock validators to succeed. Call `check_backend_health()` twice within 30 seconds (mock `time.monotonic()`). Assert validators are called only once (cached result returned on second call)

- [ ] T013 [P] [US4] Write unit test `test_check_backend_health_cache_expires_after_30_seconds` in `tests/unit/serve/test_server.py` — create an `AgentServer` with Anthropic provider. Mock validators to succeed. Call `check_backend_health()`. Advance mocked time by 31 seconds. Call `check_backend_health()` again. Assert validators are called twice (cache expired, re-checked)

- [ ] T014 [P] [US4] Write unit test `test_check_backend_health_cache_updates_on_state_change` in `tests/unit/serve/test_server.py` — call `check_backend_health()` with validators succeeding (cached as healthy). Advance time past TTL. Change mock so `validate_nodejs()` now raises `ConfigError`. Call `check_backend_health()` again. Assert result changes from healthy to unhealthy with diagnostics

#### Implementation

- [ ] T015 [US4] Add TTL-based caching to `check_backend_health()` in `src/holodeck/serve/server.py`:
  - Add instance attributes in `__init__()`: `self._backend_health_cache: tuple[bool, list[str]] | None = None` and `self._backend_health_cache_time: float = 0.0`
  - Add class constant: `_BACKEND_HEALTH_TTL: float = 30.0`
  - In `check_backend_health()`, check `time.monotonic() - self._backend_health_cache_time < self._BACKEND_HEALTH_TTL` — if within TTL, return cached result
  - Otherwise, run validators, cache result with current timestamp, return result
  - Import `time` at module level
  - Run tests T012–T014 and verify they PASS

### 3C: Health Endpoint Wiring

> Wire `check_backend_health()` into the `/health` and `/health/agent` endpoints. Map results to status values: healthy, degraded, unhealthy.

#### Tests (TDD)

- [ ] T016 [P] [US4] Write unit test `test_health_endpoint_returns_backend_ready_true_when_healthy` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider, all prerequisites met. Use `TestClient` to call `GET /health`. Assert response JSON contains `backend_ready: true`, `backend_diagnostics: []`, `status: "healthy"`

- [ ] T017 [P] [US4] Write unit test `test_health_endpoint_returns_unhealthy_when_nodejs_missing` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider, `validate_nodejs()` mocked to fail. Use `TestClient` to call `GET /health`. Assert response JSON contains `backend_ready: false`, `backend_diagnostics: ["...node error..."]`, `status: "unhealthy"`

- [ ] T018 [P] [US4] Write unit test `test_health_endpoint_returns_degraded_when_credentials_invalid` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider, Node.js available but credentials invalid. Use `TestClient` to call `GET /health`. Assert response JSON contains `backend_ready: false`, `status: "degraded"`, `backend_diagnostics` includes credential error

- [ ] T019 [P] [US4] Write unit test `test_health_endpoint_backward_compatible_for_non_anthropic` in `tests/unit/serve/test_server.py` — create `AgentServer` with OpenAI provider. Use `TestClient` to call `GET /health`. Assert response JSON contains `backend_ready: true`, `backend_diagnostics: []`. Verify existing fields (`status`, `agent_name`, `agent_ready`, `active_sessions`, `uptime_seconds`) are still present and correct

- [ ] T020 [P] [US4] Write unit test `test_health_agent_endpoint_also_includes_backend_fields` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider. Use `TestClient` to call `GET /health/agent`. Assert response JSON contains `backend_ready` and `backend_diagnostics` fields (same behavior as `/health`)

#### Implementation

- [ ] T021 [US4] Modify `_register_health_endpoints()` in `src/holodeck/serve/server.py`:
  - In the `health()` handler, call `await self.check_backend_health()` to get `(backend_ready, diagnostics)`
  - Determine status: if `not self.is_ready` or `not backend_ready` and diagnostics indicate hard failure (Node.js missing) → `"unhealthy"`; if `not backend_ready` and diagnostics indicate soft failure (credentials) → `"degraded"`; otherwise → `"healthy"`
  - Pass `backend_ready=backend_ready` and `backend_diagnostics=diagnostics` to `HealthResponse`
  - Apply same logic to `health_agent()` handler
  - Status determination logic: `"unhealthy"` when Node.js is missing (critical dependency); `"degraded"` when credentials are invalid (server can accept requests but will fail on invocation); `"healthy"` when all checks pass
  - Run tests T016–T020 and verify they PASS

### 3D: Status Determination Logic

> Define clear rules for mapping diagnostic types to status values.

#### Tests (TDD)

- [ ] T022 [P] [US4] Write unit test `test_status_unhealthy_when_server_not_ready_regardless_of_backend` in `tests/unit/serve/test_server.py` — create `AgentServer` where `is_ready` is False. Call `GET /health`. Assert `status: "unhealthy"` even if backend checks pass (server-level unready overrides backend health)

- [ ] T023 [P] [US4] Write unit test `test_status_unhealthy_when_both_nodejs_and_credentials_fail` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider. Mock both `validate_nodejs()` and `validate_credentials()` to fail. Call `GET /health`. Assert `status: "unhealthy"` (Node.js failure escalates to unhealthy)

- [ ] T024 [P] [US4] Write unit test `test_status_degraded_only_when_credentials_fail_but_nodejs_present` in `tests/unit/serve/test_server.py` — create `AgentServer` with Anthropic provider. Mock `validate_nodejs()` to succeed, `validate_credentials()` to fail. Call `GET /health`. Assert `status: "degraded"` (not unhealthy — server can still accept requests for potential credential refresh scenarios)

#### Implementation

- [ ] T025 [US4] Extract status determination into a private helper `_determine_health_status()` in `src/holodeck/serve/server.py`:
  - Signature: `def _determine_health_status(self, backend_ready: bool, diagnostics: list[str]) -> str`
  - If `not self.is_ready` → return `"unhealthy"`
  - If `backend_ready` → return `"healthy"`
  - If any diagnostic contains "Node.js" (critical runtime dependency missing) → return `"unhealthy"`
  - Otherwise (credential issues only) → return `"degraded"`
  - Update `_register_health_endpoints()` to use this helper
  - Run tests T022–T024 and verify they PASS

**Checkpoint**: Health endpoint returns backend_ready, backend_diagnostics, and correct status values. Caching prevents per-request overhead. Backward-compatible for non-Anthropic providers.

---

## Phase 4: Polish

**Purpose**: Code quality and final validation.

- [ ] T026 Run `make format` to format all modified/new files with Black + Ruff
- [ ] T027 Run `make lint` and fix any Ruff + Bandit violations in `src/holodeck/serve/server.py`, `src/holodeck/serve/models.py`, and test files
- [ ] T028 Run `make type-check` and fix any MyPy errors in `src/holodeck/serve/server.py`, `src/holodeck/serve/models.py`
- [ ] T029 Run full test suite `make test` to verify no regressions
- [ ] T030 Verify acceptance scenarios from user story:
  - Scenario 1: Claude-backed serve instance with all prerequisites met — health endpoint returns `{"status": "healthy", "backend_ready": true, "backend_diagnostics": []}`
  - Scenario 2: Claude-backed serve instance where Node.js becomes unavailable — health endpoint returns `{"status": "unhealthy", "backend_ready": false, "backend_diagnostics": ["Node.js not found..."]}`
  - Scenario 3: Claude-backed serve instance where credentials are invalid/expired — health endpoint returns `{"status": "degraded", "backend_ready": false, "backend_diagnostics": ["...credential error..."]}`

---

## Dependencies & Execution Order

### External Dependencies (MUST be complete before US4 begins)

- **US1**: `HealthResponse` model changes in `src/holodeck/serve/models.py` (adds `backend_ready` and `backend_diagnostics` fields). If US1 is not complete, Phase 2 (T004) handles adding the fields as a prerequisite
- **US1**: `validators.py` enhancements in `src/holodeck/lib/backends/validators.py` (enhanced `validate_nodejs()` with version check). US4 reuses these validators — they must exist and be importable

### Phase Dependencies

```
Phase 1 (Setup: T001–T003)
    ↓
Phase 2 (Foundational: T004–T005)
    ↓
Phase 3A (Core health check: T006–T011)
    ↓
Phase 3B (TTL caching: T012–T015) ─── depends on 3A
    ↓
Phase 3C (Endpoint wiring: T016–T021) ─── depends on 3A, 3B
    ↓
Phase 3D (Status logic: T022–T025) ─── depends on 3C
    ↓
Phase 4 (Polish: T026–T030)
```

### Parallel Execution Opportunities

Within each phase, [P]-tagged tasks can run in parallel:

- **Phase 3A tests**: T006, T007, T008, T009, T010 can run in parallel (same file, independent tests)
- **Phase 3B tests**: T012, T013, T014 can run in parallel (same file, independent tests)
- **Phase 3C tests**: T016, T017, T018, T019, T020 can run in parallel (same file, independent tests)
- **Phase 3D tests**: T022, T023, T024 can run in parallel (same file, independent tests)
- **Phase 4 quality checks**: T026, T027, T028 can run in parallel (independent tools)

### Cross-Phase Parallelism

- 3A and 3B tests can be written in parallel (both are in `test_server.py` but test different methods)
- 3C and 3D tests can be written in parallel (test different aspects of the endpoint)

---

## Implementation Strategy

1. **Start with HealthResponse model verification** (Phase 1–2) — confirm US1 has laid the groundwork or add fields if needed
2. **Build check_backend_health() first** (Phase 3A) — this is the core logic that all other phases depend on
3. **Add caching immediately** (Phase 3B) — health endpoints are polled frequently; caching prevents per-request subprocess calls to `node --version`
4. **Wire into endpoints** (Phase 3C) — connect the method to actual HTTP responses
5. **Refine status determination** (Phase 3D) — extract and test the healthy/degraded/unhealthy classification logic

### Key Design Decisions

- **Reuse existing validators**: `validate_nodejs()` and `validate_credentials()` from `validators.py` are called at both startup (US1) and runtime health checks (US4). No duplication.
- **Collect all failures**: Health check does NOT short-circuit on first failure. All validators run, and all diagnostics are returned so operators see the full picture in one request.
- **30-second TTL cache**: Prevents `node --version` subprocess overhead on every health poll. Cache is per-instance, invalidates naturally after TTL.
- **Three-tier status**: `healthy` (all good), `degraded` (functional but credential issues), `unhealthy` (critical dependency missing or server not ready). Consumers checking `status == "healthy"` correctly treat both degraded and unhealthy as not-healthy.
- **Backward-compatible**: New fields have defaults. Non-Anthropic providers always return `backend_ready: true, backend_diagnostics: []`.

---

## Key Files Modified/Created

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/serve/models.py` | Phase 2 | Verify/add `backend_ready`, `backend_diagnostics` fields on `HealthResponse` |
| `src/holodeck/serve/server.py` | Phase 3A–3D | Add `check_backend_health()`, `_determine_health_status()`, TTL cache attributes, modify `_register_health_endpoints()` |
| `tests/unit/serve/test_server.py` | Phase 3A–3D | Add 19 new unit tests for backend health check logic, caching, endpoint wiring, and status determination |

---

## Notes

- [P] tasks = independent tests within the same phase, can run in parallel
- [US4] label maps all user story tasks to User Story 4
- TDD: Verify tests fail before implementing
- Commit after each sub-phase (3A, 3B, 3C, 3D) for clean git history
- Stop at any checkpoint to validate progress
- The health check method is async to support potential future async validators, even though current validators are synchronous
- Status determination is extracted into a helper for testability and to keep endpoint handlers clean
