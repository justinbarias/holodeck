# Tasks: User Story 1 — Serve a Claude-Backed Agent via REST/AG-UI

**Feature**: 024-claude-serve-deploy | **Story**: US1 | **Date**: 2026-03-20
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md)

**User Story**: A platform operator configures an agent with `provider: anthropic` and runs `holodeck serve agent.yaml`. The server starts, performs pre-flight validation (Node.js availability, API credentials), and begins accepting requests. Clients interact with the agent through REST or AG-UI protocol endpoints identically to how they would with an OpenAI-backed agent.

**Acceptance Scenarios**:
1. Given agent config with `provider: anthropic` and Node.js installed, When operator runs `holodeck serve agent.yaml`, Then server starts successfully and health endpoint reports ready.
2. Given running Claude-backed server, When client sends message to REST endpoint, Then server returns ExecutionResult-compatible response.
3. Given running Claude-backed server, When client opens streaming session, Then text chunks delivered progressively.
4. Given agent config with `provider: anthropic` but no Node.js, When operator runs `holodeck serve`, Then server fails fast with clear error.
5. Given agent config with `provider: anthropic` but no credentials, When operator runs `holodeck serve`, Then server fails fast with clear error.

**FR Coverage**: FR-001, FR-006, FR-008, FR-012, FR-013, FR-014

---

## Phase 1: Setup — Shared Infrastructure

These tasks create the data model extensions and shared constants that all US1 tasks depend on. No story label — they are foundational.

- [x] T001 Add `max_concurrent_sessions` field to `ClaudeConfig` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/claude_config.py`. Add `max_concurrent_sessions: int | None = Field(default=10, ge=1, le=100, description="Maximum concurrent Claude SDK subprocesses per serve instance")`. Place it after the existing `max_turns` field. This enables the session cap configured via YAML (`claude.max_concurrent_sessions`).

- [x] T002 [P] Add `backend_ready` and `backend_diagnostics` fields to `HealthResponse` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/models.py`. Add: `backend_ready: bool = Field(default=True, description="Whether backend prerequisites are satisfied")` and `backend_diagnostics: list[str] = Field(default_factory=list, description="Diagnostic messages when backend is degraded/unhealthy")`. These are additive, backward-compatible fields. Update the `status` field description to include `"degraded"` as a valid value.

- [x] T003 [P] Define `NODEJS_MIN_VERSION` constant in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/validators.py`. Add `NODEJS_MIN_VERSION: int = 18` at module level after the existing env var candidate tuples. This constant is used by the enhanced `validate_nodejs()` (T005) and health check validators.

---

## Phase 2: Foundational — Blocking Prerequisites

These tasks implement the core validation logic, error mapping, and server lifecycle changes that must exist before the serve path works end-to-end.

- [x] T004 [US1] Enhance `validate_nodejs()` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/validators.py` to verify Node.js version >= 18 (FR-012). After the existing `shutil.which("node")` check passes, run `subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)`. Parse the output (format: `v22.1.0`) — strip the `v` prefix, split on `.`, extract the major version as int. If major < `NODEJS_MIN_VERSION`, raise `ConfigError("nodejs", f"Node.js version {version} found but >= {NODEJS_MIN_VERSION} is required by Claude Agent SDK. ...")`. Handle `subprocess.TimeoutExpired` and `subprocess.CalledProcessError` by raising `ConfigError` with an actionable message. Import `subprocess` at the top of the file.

- [x] T005 [US1] Add `_validate_backend_prerequisites()` method to `AgentServer` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/server.py` (FR-001, FR-008). Implement `async def _validate_backend_prerequisites(self) -> None` that: (a) checks `self.agent_config.model.provider == ProviderEnum.ANTHROPIC`, (b) if Anthropic, calls `validate_nodejs()` and `validate_credentials(self.agent_config.model)` from `holodeck.lib.backends.validators`, (c) on `ConfigError`, raises `BackendInitError` with the error message. The CLI layer (serve command) catches this and exits. (d) for non-Anthropic providers, returns immediately (no-op). Import `BackendInitError` from `holodeck.lib.backends`, `ProviderEnum` from `holodeck.models.llm`, and the validator functions.

- [x] T006 [US1] Integrate `_validate_backend_prerequisites()` into the server startup lifecycle in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/server.py`. Locate the point where the server transitions to `RUNNING` state (currently in `create_app()` which sets `READY`, then uvicorn starts). Add a call to `await self._validate_backend_prerequisites()` in the `start()` method (or the appropriate startup hook) BEFORE the state transitions to `RUNNING`. The startup lifecycle catches `BackendInitError` from `_validate_backend_prerequisites()`, logs the error with `logger.error()`, and exits. If validation fails, the server never reaches `RUNNING` state. Log a success message on success: `"Backend prerequisites validated for {provider} provider"`.

- [x] T007 [US1] Pass Claude session cap to `SessionStore` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/server.py` (FR-014). In `AgentServer.__init__()`, detect if `self.agent_config.model.provider == ProviderEnum.ANTHROPIC` and `self.agent_config.claude` is not None. If so, read `self.agent_config.claude.max_concurrent_sessions` and pass it to `SessionStore(max_sessions=cap)` instead of the default 1000. Non-Anthropic providers keep the existing 1000 default. Import `ProviderEnum` from `holodeck.models.llm`.

- [x] T008 [US1] Map `BackendSessionError` to HTTP 502 in REST protocol at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/protocols/rest.py` (FR-014, crash handling). In the chat endpoint handler(s), wrap the `AgentExecutor` invocation with a try/except that catches `BackendSessionError` (import from `holodeck.lib.backends`). When caught, return a `JSONResponse` with status 502 and body: `{"error": "backend_error", "message": "Claude Agent SDK subprocess terminated unexpectedly. Start a new session to retry.", "retriable": true}`. Remove the crashed session from the session store.

- [x] T009 [US1] Map capacity exceeded to HTTP 503 in REST protocol at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/protocols/rest.py` (FR-014, session cap). When `SessionStore.create()` raises `RuntimeError` due to max sessions exceeded, catch it and return a `JSONResponse` with status 503, `Retry-After: 5` header, and body matching the session-cap contract: `{"error": "capacity_exceeded", "message": "Maximum concurrent Claude sessions ({max}) reached. Retry after existing sessions complete.", "active_sessions": N, "max_sessions": N}`.

- [x] T010 [US1] Map `BackendSessionError` to SSE error event in AG-UI protocol at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/protocols/agui.py` (FR-014, crash handling). In the AG-UI endpoint handler, wrap the `AgentExecutor` invocation with a try/except that catches `BackendSessionError` (import from `holodeck.lib.backends`). When caught, emit an SSE error event: `event: error\ndata: {"type": "backend_error", "message": "Claude Agent SDK subprocess terminated unexpectedly.", "retriable": true}`. Remove the crashed session from the session store.

- [x] T011 [US1] Map capacity exceeded to SSE error event in AG-UI protocol at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/serve/protocols/agui.py` (FR-014, session cap). When `SessionStore.create()` raises `RuntimeError` due to max sessions exceeded, catch it and emit an SSE error event: `event: error\ndata: {"type": "capacity_exceeded", "message": "Maximum concurrent Claude sessions ({max}) reached."}`.

---

## Phase 3: User Story 1 — Unit Tests

These tasks verify all US1 production code with comprehensive unit tests.

- [x] T012 [US1] Write unit tests for `ClaudeConfig.max_concurrent_sessions` in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/models/test_claude_config.py`. Add tests: (a) default value is 10, (b) `max_concurrent_sessions: 50` is valid, (c) `max_concurrent_sessions: 0` raises `ValidationError` (ge=1), (d) `max_concurrent_sessions: 101` raises `ValidationError` (le=100), (e) `max_concurrent_sessions: null` is valid (None). Run with `pytest tests/unit/models/test_claude_config.py -n auto`.

- [x] T013 [US1] Write unit tests for enhanced `validate_nodejs()` version check in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/lib/backends/test_validators.py`. Add tests: (a) Node.js v22.1.0 passes validation (mock `subprocess.run` returning `v22.1.0`), (b) Node.js v18.0.0 passes (boundary), (c) Node.js v16.20.0 raises `ConfigError` with version message, (d) `node --version` times out raises `ConfigError`, (e) `node --version` returns non-zero exit code raises `ConfigError`, (f) node not found on PATH raises `ConfigError` (existing behavior preserved), (g) unparseable version output (e.g., garbage string) raises `ConfigError`. Mock `shutil.which` and `subprocess.run` appropriately. Run with `pytest tests/unit/lib/backends/test_validators.py -n auto`.

- [x] T014 [US1] Write unit tests for `_validate_backend_prerequisites()` in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/serve/test_server.py`. Add tests: (a) Anthropic provider with valid Node.js + credentials — no exception raised, (b) Anthropic provider with missing Node.js — raises `BackendInitError`, (c) Anthropic provider with missing credentials — raises `BackendInitError`, (d) non-Anthropic provider (e.g., openai) — skips validation entirely, (e) Node.js version too old — raises `BackendInitError`. Mock `validate_nodejs` and `validate_credentials`. Run with `pytest tests/unit/serve/test_server.py -n auto`.

- [x] T015 [US1] Write unit tests for session cap enforcement in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/serve/test_server.py`. Add tests: (a) Anthropic provider with `max_concurrent_sessions: 5` — SessionStore created with `max_sessions=5`, (b) non-Anthropic provider — SessionStore created with default `max_sessions=1000`, (c) Anthropic provider without `claude` config section — SessionStore uses default 10. Run with `pytest tests/unit/serve/test_server.py -n auto`.

- [x] T016 [US1] Write unit tests for REST protocol error mapping in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/serve/test_protocols_rest.py` (or existing test file for REST protocol). Add tests: (a) `BackendSessionError` during chat returns 502 with `{"error": "backend_error", "retriable": true}`, (b) capacity exceeded (RuntimeError from SessionStore) returns 503 with `{"error": "capacity_exceeded"}` and `Retry-After` header, (c) crashed session is removed from session store after 502 response. Mock `AgentExecutor` to raise the respective exceptions. Run with `pytest tests/unit/serve/ -n auto`.

- [x] T017 [US1] Write unit tests for AG-UI protocol error mapping in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/serve/test_protocols_agui.py` (or existing test file for AG-UI protocol). Add tests: (a) `BackendSessionError` during AG-UI streaming emits SSE error event with `"type": "backend_error"`, (b) capacity exceeded emits SSE error event with `"type": "capacity_exceeded"`, (c) crashed session is removed from session store after error event. Mock `AgentExecutor` to raise the respective exceptions. Run with `pytest tests/unit/serve/ -n auto`.

- [x] T018 [US1] Write unit test verifying env var pass-through for proxy credential injection (FR-006) in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/lib/backends/test_claude_backend.py`. Verify that when `ANTHROPIC_BASE_URL`, `HTTP_PROXY`, and `HTTPS_PROXY` are set in the process environment, the `build_options()` method on `ClaudeBackend` produces `ClaudeAgentOptions.env` that **merges with** (not replaces) the inherited environment. Set env vars via `monkeypatch.setenv()`, call `build_options()`, and assert the env dict contains the proxy vars. This confirms FR-006 without any new production code. Run with `pytest tests/unit/lib/backends/test_claude_backend.py -n auto`.

---

## Phase 4: User Story 1 — Integration Tests & OTel Verification

- [x] T019 [US1] Create Claude agent fixture for serve integration tests at `/Users/justinbarias/Documents/Git/python/agentlab/tests/fixtures/claude_agent/agent.yaml`. Define a minimal Claude agent config: `name: claude-serve-test`, `model: {provider: anthropic, name: claude-sonnet-4-20250514}`, `instructions: {inline: "You are a helpful test assistant."}`, `claude: {permission_mode: acceptAll, max_concurrent_sessions: 3}`.

- [x] T020 [US1] Write integration test for Claude serve startup and health endpoint in new file `/Users/justinbarias/Documents/Git/python/agentlab/tests/integration/serve/test_server_claude.py` (FR-001, FR-005, FR-010). Test: (a) `AgentServer` with Claude agent fixture starts and health endpoint returns `backend_ready: true` (requires Node.js + credentials in CI), (b) health endpoint returns `backend_diagnostics: []` when all prerequisites met, (c) server transitions to RUNNING state after successful pre-flight validation. Mark with `@pytest.mark.integration`. Use `httpx.AsyncClient` with the FastAPI test client pattern. Run with `pytest tests/integration/serve/test_server_claude.py -n auto`.

- [x] T021 [US1] Write integration test for Claude serve request handling in `/Users/justinbarias/Documents/Git/python/agentlab/tests/integration/serve/test_server_claude.py` (FR-010). Test: (a) POST to REST chat endpoint with Claude agent returns 200 with `ExecutionResult`-compatible response (has `content`, `session_id`, `tool_calls` fields), (b) streaming endpoint returns SSE events with progressive text chunks, (c) session is created and tracked in SessionStore. Mark with `@pytest.mark.integration`. Run with `pytest tests/integration/serve/test_server_claude.py -n auto`.

- [x] T022 [US1] Write integration test verifying OTel instrumentation in serve mode in `/Users/justinbarias/Documents/Git/python/agentlab/tests/integration/serve/test_server_claude.py` (FR-013). Test: (a) configure an `InMemorySpanExporter`, (b) start Claude serve, send a request, (c) assert at least one span with `gen_ai.system` attribute is emitted, (d) verify span has `gen_ai.request.model` attribute matching the configured model. This confirms FR-013 — GenAI semantic convention instrumentation works in serve mode without new code. Mark with `@pytest.mark.integration`.

---

## Phase 5: Polish — Validation & Quality

- [x] T023 [US1] Validate quickstart Scenario 1 (Serve a Claude Agent Locally) from `/Users/justinbarias/Documents/Git/python/agentlab/specs/024-claude-serve-deploy/quickstart.md`. Create a unit test in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/serve/test_serve_quickstart.py` that: (a) parses the quickstart YAML config through the `Agent` Pydantic model and asserts validation passes, (b) verifies `agent.claude.max_concurrent_sessions == 5`, (c) verifies `agent.model.provider == ProviderEnum.ANTHROPIC`. This does NOT require a running server — config parsing only.

- [x] T024 [US1] Run full test suite and code quality checks. Execute `make test-unit`, `make lint`, `make format-check`, and `make type-check` from `/Users/justinbarias/Documents/Git/python/agentlab`. Fix any failures introduced by US1 changes. Ensure all existing tests still pass (zero regressions per FR-009). All new files must pass Black formatting, Ruff linting, and MyPy type checking.

---

## Dependencies & Execution Order

```
T001 (ClaudeConfig field)
  └─ no dependencies — start immediately

T002 (HealthResponse fields)
  └─ no dependencies — start immediately

T003 (NODEJS_MIN_VERSION constant)
  └─ no dependencies — start immediately

T001 + T002 + T003 can all run in parallel (Phase 1).

T004 (validate_nodejs enhancement) ── depends on T003
T005 (_validate_backend_prerequisites) ── depends on T004
T006 (startup lifecycle integration) ── depends on T005
T007 (session cap in __init__) ── depends on T001

T008 (REST 502 mapping) ── depends on T006
T009 (REST 503 mapping) ── depends on T007
T010 (AG-UI 502 mapping) ── depends on T006
T011 (AG-UI 503 mapping) ── depends on T007

T008 + T009 can run in parallel (REST protocol changes).
T010 + T011 can run in parallel (AG-UI protocol changes).
REST and AG-UI groups can run in parallel with each other.

T012 (ClaudeConfig tests) ── depends on T001
T013 (validator tests) ── depends on T004
T014 (prereq tests) ── depends on T005, T006
T015 (session cap tests) ── depends on T007
T016 (REST error tests) ── depends on T008, T009
T017 (AG-UI error tests) ── depends on T010, T011
T018 (env var pass-through test) ── no production deps, can run after Phase 1

T012 + T013 + T015 + T018 can run in parallel (independent test files).
T014 can run in parallel with T016 + T017 (independent test files).

T019 (fixture) ── no dependencies — can start in Phase 1
T020 (integration: startup) ── depends on T006, T019
T021 (integration: request handling) ── depends on T008, T009, T019
T022 (integration: OTel) ── depends on T020

T023 (quickstart validation) ── depends on T001
T024 (full suite) ── depends on ALL previous tasks
```

## Parallel Execution Examples

The following groups can be executed in parallel within each phase:

**Phase 1 parallel group:**
- T001 (ClaudeConfig field) | T002 (HealthResponse fields) | T003 (NODEJS_MIN_VERSION) — all independent, no cross-file dependencies

**Phase 2 first wave:**
- T004 (validate_nodejs) | T007 (session cap) — T004 depends on T003, T007 depends on T001, no overlap

**Phase 2 second wave (after T004, T005):**
- T008 (REST 502) + T009 (REST 503) | T010 (AG-UI 502) + T011 (AG-UI 503) — all independent protocol/endpoint changes

**Phase 3 first wave:**
- T012 (config tests) | T013 (validator tests) | T015 (session cap tests) | T018 (env var tests) — independent test files

**Phase 3 second wave:**
- T014 (server tests) | T016 (REST tests) | T017 (AG-UI tests) — independent test files

**Phase 4:**
- T019 (fixture, can start early) | T020, T021 (integration, after Phase 2) | T022 (OTel, after T020)

**Phase 5 sequential:**
- T023 → T024

## Implementation Strategy

### Guiding Principles

1. **Fail-fast validation**: All Claude backend prerequisites (Node.js >= 18, API credentials) MUST be checked at server startup in `AgentServer.start()`, before transitioning to RUNNING state. The server must never appear healthy while lacking prerequisites.

2. **Backward compatibility**: All changes to `HealthResponse` and health endpoints are additive. Existing consumers that check `status == "healthy"` continue to work. The new `"degraded"` status is only relevant to Claude-aware consumers.

3. **Contract compliance**: REST error responses (502 for crashes, 503 for capacity) and AG-UI SSE error events MUST match the contracts defined in `/Users/justinbarias/Documents/Git/python/agentlab/specs/024-claude-serve-deploy/contracts/session-cap.md` exactly.

4. **No new backend code**: The `ClaudeBackend` and `ClaudeSession` already implement the `AgentBackend`/`AgentSession` protocols. US1 only adds server-level validation, error mapping, and lifecycle management around the existing backend.

5. **OTel for free**: GenAI semantic convention instrumentation activates automatically via `ClaudeBackend.initialize()` in serve mode. No new instrumentation code is needed — only an integration test to verify (T022).

6. **Error handling**: `BackendInitError` is raised by `_validate_backend_prerequisites()` on pre-flight failure; the CLI layer (serve command) catches it and exits. `BackendSessionError` maps to 502 (retriable), capacity exceeded maps to 503 (with `Retry-After`). Crashed sessions are removed from `SessionStore` to free the slot immediately.

### Key Files Reference

| File | Action | Tasks | FR Coverage |
|------|--------|-------|-------------|
| `src/holodeck/models/claude_config.py` | MODIFY | T001, T012 | FR-014 |
| `src/holodeck/serve/models.py` | MODIFY | T002 | FR-005 |
| `src/holodeck/lib/backends/validators.py` | MODIFY | T003, T004, T013 | FR-012 |
| `src/holodeck/serve/server.py` | MODIFY | T005, T006, T007, T014, T015 | FR-001, FR-008, FR-014 |
| `src/holodeck/serve/protocols/rest.py` | MODIFY | T008, T009, T016 | FR-014 |
| `src/holodeck/serve/protocols/agui.py` | MODIFY | T010, T011, T017 | FR-014 |
| `tests/unit/models/test_claude_config.py` | MODIFY | T012 | FR-014 |
| `tests/unit/lib/backends/test_validators.py` | MODIFY | T013 | FR-012 |
| `tests/unit/serve/test_server.py` | MODIFY | T014, T015 | FR-001, FR-014 |
| `tests/unit/serve/test_protocols_rest.py` | MODIFY | T016 | FR-014 |
| `tests/unit/serve/test_protocols_agui.py` | MODIFY | T017 | FR-014 |
| `tests/unit/lib/backends/test_claude_backend.py` | MODIFY | T018 | FR-006 |
| `tests/fixtures/claude_agent/agent.yaml` | NEW | T019 | FR-010 |
| `tests/integration/serve/test_server_claude.py` | NEW | T020, T021, T022 | FR-010, FR-013 |
| `tests/unit/serve/test_serve_quickstart.py` | NEW | T023 | — |
