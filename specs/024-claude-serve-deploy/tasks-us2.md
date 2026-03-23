# Tasks: Deploy a Claude-Backed Agent as a Container (US2)

**Input**: Design documents from `/specs/024-claude-serve-deploy/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.
**Dependency**: US2 depends on US1's serve infrastructure (pre-flight validation, health endpoint enhancements, `validators.py` Node.js version check). Tasks below assume US1 foundational work is already merged. US2 depends on US3 for entrypoint fixes and base image Node.js layer. Container images won't run correctly until US3 entrypoint work is complete.

**Organization**: Tasks are grouped by phase to enable incremental delivery. User Story 2 tasks carry the [US2] label; shared setup/polish phases do not.

## Format: `[ID] [P?] [US2] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US2]**: User Story 2 tasks
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify that deploy-related dependencies (Docker SDK, Jinja2) are already available and that US1 foundational work is in place before starting US2. No new dependencies to add — FastAPI, Docker SDK, and Jinja2 were added in US1.

**CRITICAL**: US1 must be complete before starting. Specifically: `validate_nodejs()` with version check in `src/holodeck/lib/backends/validators.py`, `max_concurrent_sessions` on `ClaudeConfig` in `src/holodeck/models/claude_config.py`, and serve pre-flight validation in `src/holodeck/serve/server.py` must already exist.

- [ ] T001 Verify US1 prerequisites are merged: confirm `validate_nodejs()` in `src/holodeck/lib/backends/validators.py` includes Node.js version check (>= 18), `ClaudeConfig.max_concurrent_sessions` field exists in `src/holodeck/models/claude_config.py`, and `_validate_backend_prerequisites()` exists in `src/holodeck/serve/server.py`. If any are missing, STOP and complete US1 first
- [ ] T002 Verify `jinja2` and `docker` packages are available in the virtualenv by running `python -c "import jinja2; import docker"`. If missing, add to `pyproject.toml` and run `uv lock`

**Checkpoint**: All shared infrastructure from US1 confirmed present. Deploy dependencies available.

---

## Phase 2: Foundational (Deploy-Specific Prerequisites)

**Purpose**: Create the Claude agent fixture for deploy tests and establish the test infrastructure needed by US2 implementation tasks.

- [ ] T003 Create test fixture `tests/fixtures/claude_agent/agent.yaml` with a minimal Claude agent config: `name: claude-deploy-test`, `model.provider: anthropic`, `model.name: claude-sonnet-4-20250514`, `instructions.inline: "You are a test assistant."`, `deployment.port: 8080`, `deployment.protocol: rest`. Include `claude.max_concurrent_sessions: 5` to exercise the new config field
- [ ] T004 [P] Verify the fixture loads correctly by writing a smoke test in `tests/unit/deploy/test_dockerfile.py` (append to existing file): `test_claude_agent_fixture_loads` — load `tests/fixtures/claude_agent/agent.yaml` via `ConfigLoader`, assert `agent.model.provider == ProviderEnum.ANTHROPIC`, assert `agent.claude.max_concurrent_sessions == 5`

**Checkpoint**: Claude deploy test fixture ready. Existing test infrastructure confirmed working.

---

## Phase 3: User Story 2 Implementation

**Purpose**: Implement Dockerfile generation with Node.js and dry-run support. Entrypoint fixes and base image work are deferred to US3.

### 3a: Dockerfile Template — Node.js Conditional Block (FR-002, FR-003)

- [ ] T005 [P] [US2] Write unit tests FIRST in `tests/unit/deploy/test_dockerfile.py` (append to existing file). Tests MUST FAIL before implementation. Include: (1) `test_generate_dockerfile_without_nodejs` — call `generate_dockerfile(agent_name="test", port=8080, protocol="rest")` without `needs_nodejs`, assert output does NOT contain "nodejs" or "nodesource", (2) `test_generate_dockerfile_with_nodejs` — call `generate_dockerfile(agent_name="test", port=8080, protocol="rest", needs_nodejs=True)`, assert output contains `nodesource/setup_22.x` and `apt-get install -y --no-install-recommends nodejs`, (3) `test_generate_dockerfile_nodejs_cleanup` — call with `needs_nodejs=True`, assert output contains `rm -rf /var/lib/apt/lists/*` after Node.js install, (4) `test_generate_dockerfile_nodejs_before_user_switch` — call with `needs_nodejs=True`, assert the Node.js `RUN` block appears before the `USER holodeck` line (security: install as root), (5) `test_generate_dockerfile_default_needs_nodejs_false` — call without `needs_nodejs` param, assert no Node.js block in output (backward-compatible default)
- [ ] T006 [US2] Add `needs_nodejs: bool = False` parameter to `generate_dockerfile()` in `src/holodeck/deploy/dockerfile.py`. Pass it to the Jinja2 template context. Add conditional Node.js installation block to `HOLODECK_DOCKERFILE_TEMPLATE` after the `USER root` / `WORKDIR /app` section and before `COPY entrypoint.sh`:
    ```
    {% if needs_nodejs %}
    # Install Node.js (required for Claude Agent SDK)
    RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
        && apt-get install -y --no-install-recommends nodejs \
        && rm -rf /var/lib/apt/lists/*
    {% endif %}
    ```
- [ ] T007 [US2] Run tests in `tests/unit/deploy/test_dockerfile.py` with `pytest tests/unit/deploy/test_dockerfile.py -n auto -v` and verify all pass (both new and existing tests)

### 3b: Deploy Command — Provider Detection (FR-002, FR-007)

- [ ] T008 [P] [US2] Write unit tests FIRST in `tests/unit/deploy/test_dockerfile.py` (or a new `tests/unit/cli/commands/test_deploy_claude.py` if the file grows too large). Tests MUST FAIL before implementation. Include: (1) `test_generate_dockerfile_content_detects_anthropic_provider` — create a mock Agent with `model.provider=ProviderEnum.ANTHROPIC` and a mock DeploymentConfig, call `_generate_dockerfile_content(agent, config, "1.0.0")`, assert output contains Node.js installation block, (2) `test_generate_dockerfile_content_skips_nodejs_for_openai` — create mock Agent with `model.provider=ProviderEnum.OPENAI`, call `_generate_dockerfile_content()`, assert output does NOT contain Node.js block, (3) `test_generate_dockerfile_content_skips_nodejs_for_ollama` — same for `ProviderEnum.OLLAMA`
- [ ] T009 [US2] Modify `_generate_dockerfile_content()` in `src/holodeck/cli/commands/deploy.py` to detect the agent's provider. Import `ProviderEnum` from `holodeck.models.llm`. Before the `return generate_dockerfile(...)` call, add: `needs_nodejs = agent.model.provider == ProviderEnum.ANTHROPIC`. Pass `needs_nodejs=needs_nodejs` to `generate_dockerfile()` call
- [ ] T010 [US2] Run tests for the deploy command with `pytest tests/unit/deploy/ -n auto -v` and verify all pass

### 3c: Dry-Run Output Verification (FR-007)

- [ ] T011 [P] [US2] Write unit test in `tests/unit/deploy/test_dockerfile.py`: `test_dry_run_shows_nodejs_for_claude_agent` — simulate the dry-run path by calling `_generate_dockerfile_content()` with an anthropic-provider agent, capture the returned Dockerfile string, assert it contains: (1) "Node.js" or "nodejs" comment, (2) `nodesource/setup_22.x`, (3) `apt-get install -y --no-install-recommends nodejs`, (4) non-root user (`USER holodeck`), (5) `HEALTHCHECK` directive. This validates that --dry-run output (which just prints the Dockerfile) shows Claude-specific additions
- [ ] T012 [P] [US2] Write unit test: `test_dry_run_skips_nodejs_for_non_claude_agent` — same as above but with `provider=openai`, assert output does NOT contain Node.js installation block

### 3d: Integration Tests (FR-009, FR-010)

- [ ] T013 [US2] Create integration test file `tests/integration/deploy/test_build_claude.py`. Write tests that exercise the full deploy build path for Claude agents. Include: (1) `test_build_claude_agent_generates_dockerfile_with_nodejs` — load the fixture `tests/fixtures/claude_agent/agent.yaml`, call the build pipeline (mock Docker SDK to avoid actual image build), capture generated Dockerfile, assert Node.js block present, (2) `test_build_openai_agent_no_nodejs` — use an existing OpenAI fixture, call build pipeline, assert no Node.js in Dockerfile
- [ ] T014 [US2] Run integration tests with `pytest tests/integration/deploy/ -n auto -v` and verify all pass. Also run `pytest tests/integration/ -n auto -v` to verify no regressions in existing deploy integration tests (FR-009)

### 3e: Acceptance Scenario Verification Tests

- [ ] T015 [P] [US2] Write acceptance test `test_anthropic_deploy_build_includes_nodejs` in `tests/integration/deploy/test_build_claude.py` — corresponds to acceptance scenario 1. Load Claude agent fixture, trigger `_generate_dockerfile_content()`, assert Dockerfile includes `nodejs` installation alongside Python runtime (verify both `python` base image and `nodejs` install are present)
- [ ] T016 [P] [US2] Write acceptance test `test_dry_run_shows_claude_dockerfile_additions` in `tests/integration/deploy/test_build_claude.py` — corresponds to acceptance scenario 5. Simulate dry-run by calling `_generate_dockerfile_content()` with Claude agent, assert output contains: Node.js installation, non-root `USER holodeck`, `HEALTHCHECK`, and security-related patterns (`--no-install-recommends`, `rm -rf /var/lib/apt/lists/*`)

---

## Phase 4: Polish

**Purpose**: Code quality, regression checks, and final validation.

- [ ] T017 [P] Run `make format` to format all new and modified files with Black + Ruff
- [ ] T018 [P] Run `make lint` and fix any Ruff + Bandit violations in `src/holodeck/deploy/dockerfile.py`, `src/holodeck/cli/commands/deploy.py`
- [ ] T019 Run `make type-check` and fix any MyPy errors in modified files — ensure `needs_nodejs` parameter has proper type annotation
- [ ] T020 Run full test suite `make test` to verify no regressions across entire codebase (FR-009). Pay special attention to existing deploy tests passing unchanged
- [ ] T021 Run `make security` to verify no new security issues introduced by the Node.js installation pattern
- [ ] T022 Verify that the generated Dockerfile follows secure deployment practices per FR-003: non-root user, `--no-install-recommends`, cache cleanup, `HEALTHCHECK` directive present

---

## Dependencies & Execution Order

### Cross-Story Dependencies

- **US1 (REQUIRED BEFORE US2)**: US2 depends on US1's serve infrastructure for the container to actually work. Specifically:
  - `validate_nodejs()` with version check in `src/holodeck/lib/backends/validators.py` — reused in entrypoint validation logic
  - `ClaudeConfig.max_concurrent_sessions` in `src/holodeck/models/claude_config.py` — used in fixture and container config
  - Serve pre-flight validation in `src/holodeck/serve/server.py` — the container's `holodeck serve` command must work
  - Health endpoint enhancements in `src/holodeck/serve/models.py` — container health checks depend on backend-aware health
- **US2 depends on US3** for entrypoint fixes and base image Node.js layer. Container images won't run correctly until US3 entrypoint work is complete. Specifically: inline entrypoint Claude validation (`_prepare_build_context()`), base image entrypoint fix (`docker/entrypoint.sh` `--config` bug), `validate_claude_requirements()` in entrypoint, and base Dockerfile Node.js conditional layer are all US3 responsibilities.
- **US2 is independent of US4** (backend-aware health checks) — though US4's health endpoint enhancements make the container's HEALTHCHECK more useful

### Phase Dependencies

- **Phase 1 (Setup)**: Depends on US1 completion — verify prerequisites
- **Phase 2 (Foundational)**: Depends on Phase 1 — create test fixtures
- **Phase 3 (Implementation)**: Depends on Phase 2 — all deploy changes
  - Phase 3a (Dockerfile template) can start independently
  - Phase 3b (deploy command provider detection) depends on 3a (calls `generate_dockerfile()` with `needs_nodejs`)
  - Phase 3c (dry-run verification) depends on 3a + 3b (needs working Dockerfile generation)
  - Phase 3d (integration tests) depends on 3a + 3b (needs full pipeline working)
  - Phase 3e (acceptance tests) depends on 3d (builds on integration test infrastructure)
- **Phase 4 (Polish)**: Depends on all Phase 3 sub-phases being complete

### Within Each Sub-Phase

- Tests MUST be written and FAIL before implementation (TDD)
- Implementation after tests
- Verify tests PASS after implementation

---

## Parallel Execution Examples

### Parallel Group 1: Phase 3a (Dockerfile template tests)

```bash
# These tasks can run in parallel (different test functions, no dependencies):
Task T005: "Write Dockerfile template Node.js tests in tests/unit/deploy/test_dockerfile.py"
Task T008: "Write deploy command provider detection tests"
```

### Parallel Group 2: Acceptance Scenarios

```bash
# These tasks can run in parallel (independent test functions, same file):
Task T015: "Write test_anthropic_deploy_build_includes_nodejs"
Task T016: "Write test_dry_run_shows_claude_dockerfile_additions"
```

### Parallel Group 3: Polish (independent checks)

```bash
# These tasks can run in parallel (independent quality checks):
Task T017: "Run make format"
Task T018: "Run make lint"
```

---

## Implementation Strategy

### MVP First (Phases 1-3b Only)

1. Complete Phase 1: Setup (T001-T002) — prerequisites verified
2. Complete Phase 2: Foundational (T003-T004) — test fixture ready
3. Complete Phase 3a: Dockerfile template (T005-T007) — `needs_nodejs` works
4. Complete Phase 3b: Deploy command detection (T008-T010) — `holodeck deploy build` detects Claude
5. **STOP and VALIDATE**: Run `make test` — `deploy build` generates correct Dockerfile for Claude agents
6. Test manually with `holodeck deploy build tests/fixtures/claude_agent/agent.yaml --dry-run`

### Incremental Delivery

1. Phase 1 -> Prerequisites confirmed
2. Phase 2 -> Test fixture ready
3. Phase 3a -> Dockerfile template supports Node.js conditional block
4. Phase 3b -> Deploy command auto-detects provider, passes `needs_nodejs` (MVP!)
5. Phase 3c -> Dry-run output verified
6. Phase 3d -> Integration tests pass
7. Phase 3e -> All acceptance scenarios verified
8. Phase 4 -> Code quality validated, no regressions

### Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/deploy/dockerfile.py` | Phase 3a | Add `needs_nodejs` param + conditional Jinja2 Node.js block |
| `src/holodeck/cli/commands/deploy.py` | Phase 3b | Detect provider in `_generate_dockerfile_content()` |
| `tests/unit/deploy/test_dockerfile.py` | Phase 2, 3a, 3b, 3c | Claude fixture smoke test, Node.js template tests, dry-run tests |
| `tests/integration/deploy/test_build_claude.py` | Phase 3d, 3e | **NEW**: Claude deploy build integration + acceptance tests |
| `tests/fixtures/claude_agent/agent.yaml` | Phase 2 | **NEW**: Claude agent fixture for deploy tests |

---

## Notes

- [P] tasks = different files, no dependencies between them
- [US2] label on every user story task for traceability
- All Dockerfile changes must maintain backward compatibility — `needs_nodejs=False` (default) produces identical output to current behavior
- Node.js version 22.x is used (matching plan.md nodesource setup URL)
- Entrypoint fixes (inline and base image), base image Node.js layer, and Claude validation in entrypoints are all deferred to US3
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
