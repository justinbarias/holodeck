# Tasks: US5 - Integration Test Coverage for Claude Serve/Deploy

**User Story**: A developer contributing to HoloDeck runs the test suite and gets confidence that Claude backend support for serve and deploy works correctly. US5 covers ONLY the integration test gaps not addressed by US1-US4 unit/integration tests. Shared test fixtures, end-to-end scenarios (streaming, graceful shutdown, OTel), and final regression validation.

**Dependencies**: US1 (serve infrastructure), US2 (deploy infrastructure), US3 (secure container), US4 (health checks) must be implemented before US5 tests can validate their behavior.

**Spec References**: FR-009 (no regressions), FR-010 (Claude integration tests), SC-005 (CI coverage), SC-006 (zero regressions)

---

## Phase 1: Setup — Test Fixtures

- [ ] T001 [P] CREATE Claude agent fixture file at `tests/fixtures/claude_agent/agent.yaml` with `provider: anthropic`, `name: claude-sonnet-4-20250514`, inline instructions, `claude.permission_mode: acceptAll`, and `claude.max_concurrent_sessions: 5` — this fixture is shared by both serve and deploy integration tests
- [ ] T002 [P] CREATE Claude agent instruction file at `tests/fixtures/claude_agent/instructions.md` with a minimal system prompt for test scenarios (e.g., "You are a test assistant. Respond concisely.")
- [ ] T003 [P] VERIFY existing SK agent fixture at `tests/fixtures/deploy/sample_agent/agent.yaml` is unchanged and still valid — this is the regression baseline for FR-009

## Phase 2: Foundational — Test Infrastructure

- [ ] T004 ADD shared pytest fixtures in `tests/integration/conftest.py` (or create if not present): `claude_agent_config` fixture that loads `tests/fixtures/claude_agent/agent.yaml` through `ConfigLoader` and returns a validated `Agent` model; `mock_nodejs_available` fixture that patches `shutil.which("node")` to return a path and `subprocess.run` for `node --version` to return `v22.0.0`; `mock_nodejs_unavailable` fixture that patches `shutil.which("node")` to return `None`
- [ ] T005 ADD shared pytest fixture `mock_anthropic_credentials` in `tests/integration/conftest.py` that sets `ANTHROPIC_API_KEY=test-key-xxxx` in the environment via `monkeypatch` for tests that need credential validation to pass
- [ ] T006 ADD `tests/integration/serve/__init__.py` if not present (already exists per glob) and `tests/integration/deploy/__init__.py` if not present (already exists per glob) — verify both exist to ensure test discovery works

## Phase 3: US5 — Serve Integration Tests

- [ ] T007 [US5] CREATE `tests/integration/serve/test_server_claude.py` with test class `TestClaudeServeIntegration` — import `AgentServer`, `ConfigLoader`, `BackendSelector`, and `httpx.AsyncClient` for ASGI testing
- [ ] T008 [US5] ADD integration test `test_claude_streaming_delivers_progressive_chunks` in `tests/integration/serve/test_server_claude.py` — open streaming session via AG-UI or REST streaming endpoint (with mocked `ClaudeSession.send_streaming()`), assert multiple text chunks arrive progressively before final response (acceptance scenario 1)
- [ ] T009 [US5] ADD integration test `test_claude_session_cap_returns_503` in `tests/integration/serve/test_server_claude.py` — start server with `max_concurrent_sessions: 2`, create 2 active sessions (mocked), send a 3rd request, assert HTTP 503 with `"error": "capacity_exceeded"` and `Retry-After` header (FR-014, session-cap contract)
- [ ] T010 [US5] ADD integration test `test_claude_subprocess_crash_returns_502` in `tests/integration/serve/test_server_claude.py` — mock `ClaudeSession.send()` to raise `BackendSessionError`, assert HTTP 502 with `"error": "backend_error"` and `"retriable": true` (edge case EC-2, session-cap contract)
- [ ] T011 [US5] ADD integration test `test_claude_serve_emits_otel_spans` in `tests/integration/serve/test_server_claude.py` — start Claude-backed server with in-memory OTel exporter, send a request (mocked backend), assert at least one span is emitted with GenAI semantic convention attributes (FR-013)
- [ ] T012 [US5] ADD integration test `test_claude_server_graceful_shutdown` in `tests/integration/serve/test_server_claude.py` — start server, create an active session, trigger shutdown, assert all sessions are closed and subprocesses terminated before server exits

## Phase 3: US5 — Deploy Integration Tests

- [ ] T013 [US5] CREATE `tests/integration/deploy/test_build_claude.py` with test class `TestClaudeDeployIntegration` — import deploy build functions, `ConfigLoader`, and `Agent` model
- [ ] T014 [US5] ADD integration test `test_deploy_entrypoint_validates_claude_prerequisites` in `tests/integration/deploy/test_build_claude.py` — inspect generated entrypoint script content, assert it contains `node --version` check and `ANTHROPIC_API_KEY` validation before `holodeck serve` invocation (acceptance scenario 3, FR-004)
- [ ] T015 [US5] ADD integration test `test_deploy_entrypoint_uses_positional_arg` in `tests/integration/deploy/test_build_claude.py` — inspect generated entrypoint, assert it uses `holodeck serve /app/agent.yaml` (positional) NOT `holodeck serve --config /app/agent.yaml` (FR-011 entrypoint bug fix)
- [ ] T016 [US5] ADD integration test `test_deploy_build_sk_agent_unchanged` in `tests/integration/deploy/test_build_claude.py` — run `_generate_dockerfile_content()` with existing SK agent fixture (`tests/fixtures/deploy/sample_agent/agent.yaml`), assert generated Dockerfile does NOT contain Node.js commands (FR-009 regression guard)
- [ ] T017 [P] [US5] ADD integration test `test_deploy_dockerfile_nonroot_user` in `tests/integration/deploy/test_build_claude.py` — assert Claude agent Dockerfile contains `USER` directive with non-root UID and `RUN groupadd`/`useradd` or equivalent (FR-003)
- [ ] T018 [P] [US5] ADD integration test `test_deploy_env_passthrough_configured` in `tests/integration/deploy/test_build_claude.py` — assert generated Dockerfile or entrypoint does not filter or override `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `HTTP_PROXY`, `HTTPS_PROXY` environment variables (FR-006)
- [ ] T019 [US5] ADD conditional integration test `test_deploy_built_container_serves_requests` in `tests/integration/deploy/test_build_claude.py` — marked with `@pytest.mark.skipif(not docker_available())` — build actual Docker image from Claude agent fixture, run container with mocked credentials, send health check request, assert healthy response (acceptance scenario 2, requires Docker daemon)

## Phase 4: Polish

- [ ] T020 RUN `make format && make lint && make type-check` — fix any formatting, linting, or type errors introduced by US5 test files
- [ ] T021 RUN `make test-unit -n auto` — verify all new unit tests pass and no existing unit tests regress
- [ ] T022 RUN `make test-integration -n auto` — verify all new integration tests pass (with appropriate mocks) and no existing integration tests regress (FR-009)
- [ ] T023 VERIFY test markers: all new unit tests have `@pytest.mark.unit`, all new integration tests have `@pytest.mark.integration`, Docker-dependent test has `@pytest.mark.slow`
- [ ] T024 REVIEW test names for consistency — all test method names should follow `test_{what}_{expected_outcome}` pattern matching existing test conventions in the repo

## Phase 5: Validation

- [ ] T025 RUN all tests across US1-US5 (`make test -n auto`) and verify FR-009 (zero SK regressions) and FR-010 (Claude integration coverage) are met. Confirm no existing SK-backend tests fail, and that new Claude serve/deploy integration tests provide end-to-end coverage for streaming, session cap, OTel, graceful shutdown, entrypoint validation, and Docker build scenarios.

---

## Dependencies

```
T001, T002, T003 ─── (no dependencies, can run in parallel)
       │
       ▼
T004, T005, T006 ─── (depend on T001 for fixture path)
       │
       ▼
T007 (serve file creation)    T013 (deploy file creation)
       │                              │
       ▼                              ▼
T008-T012 (serve integration)  T014-T019 (deploy integration)
       │                              │
       └──────────┬───────────────────┘
                  ▼
            T020-T024 (polish)
                  │
                  ▼
            T025 (validation)
```

## Parallel Execution Examples

**Round 1** (fixtures — all independent):
```
T001 + T002 + T003  →  parallel
```

**Round 2** (infrastructure — depend on fixtures):
```
T004 + T005 + T006  →  parallel
```

**Round 3** (integration test file creation):
```
T007 + T013  →  parallel (create test files)
```

**Round 4** (integration tests — serve and deploy can run in parallel):
```
T008-T012 (serve integration)  ||  T014-T019 (deploy integration)
```

**Round 5** (polish — sequential):
```
T020 → T021 → T022 → T023 → T024
```

**Round 6** (validation):
```
T025
```

## Implementation Strategy

1. **Fixture-first**: Create the Claude agent YAML fixture (T001-T003) before any test code. This fixture drives both serve and deploy test suites and must match the schema defined in `data-model.md`.

2. **Shared infrastructure**: Build reusable pytest fixtures (T004-T006) that mock Node.js and credentials. These prevent integration tests from requiring real Node.js or Anthropic API access, enabling CI execution without external dependencies.

3. **Integration tests only**: US5 does not duplicate unit tests already covered in US1-US4. Validator unit tests are in US1 T014, server unit tests in US1 T015/T017 and US4, Dockerfile unit tests in US2 T005/T008, and protocol error mapping tests in US1 T018/T019. US5 focuses on end-to-end integration scenarios that exercise the full code path.

4. **Mock boundaries**: Integration tests mock at the backend boundary (`ClaudeBackend`, `ClaudeSession`) rather than at HTTP or subprocess level. This tests the full serve/deploy code path while avoiding dependency on the Claude Agent SDK subprocess.

5. **Regression as first-class concern**: Tests T003 and T016 explicitly verify SK backend paths remain unchanged. These are not afterthoughts but core acceptance criteria (FR-009, SC-006).

6. **Conditional Docker tests**: T019 uses `pytest.mark.skipif` to gracefully skip when Docker daemon is unavailable, preventing CI failures in environments without Docker. Mark with `@pytest.mark.slow` for optional exclusion.

7. **Final validation**: T025 runs the full test suite across all user stories, ensuring no regressions and confirming end-to-end coverage meets FR-009 and FR-010 requirements.

---

## Summary

| Phase | Task Count | Parallelizable | Sequential |
|-------|-----------|----------------|------------|
| Phase 1: Setup (fixtures) | 3 | 3 | 0 |
| Phase 2: Foundational (infrastructure) | 3 | 3 | 0 |
| Phase 3: Serve integration [US5] | 6 | 0 | 6 |
| Phase 3: Deploy integration [US5] | 7 | 2 | 5 |
| Phase 4: Polish | 5 | 0 | 5 |
| Phase 5: Validation | 1 | 0 | 1 |
| **Total** | **25** | **8** | **17** |
