# Tasks: Secure Container Deployment with Credential Isolation (US3)

**Input**: Design documents from `/specs/024-claude-serve-deploy/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/health-endpoint.md
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.

**Organization**: Tasks are grouped by phase. US3 focuses on Dockerfile security hardening (non-root user, capability drops, read-only root filesystem, tmpfs support) and environment variable pass-through for the proxy credential injection pattern (`ANTHROPIC_BASE_URL`).

## Format: `[ID] [P?] [US3] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US3]**: All user story tasks belong to User Story 3
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root
- **Docker assets**: `docker/Dockerfile`, `docker/entrypoint.sh`
- **Deploy module**: `src/holodeck/deploy/dockerfile.py`

---

## Phase 1: Setup — Understand Current State

**Purpose**: Audit the existing Dockerfile, entrypoint, and deploy template to identify gaps against Anthropic secure deployment guidelines.

- [ ] T001 Audit `docker/Dockerfile` for existing security posture: verify non-root user (`holodeck`, uid 1000) is already created, `/app` ownership is set, `USER holodeck` is the final layer. Document any gaps (missing tmpfs-compatible directory structure, missing `/tmp` permissions, no capability-awareness notes).
- [ ] T002 Audit `docker/entrypoint.sh` for non-root compatibility: verify all file operations use paths owned by `holodeck` user, no `apt-get` or root-only operations at runtime, no hardcoded absolute paths outside `/app` and `/tmp`. Document the `--config` bug on line 111 (uses `--config` flag instead of positional argument).
- [ ] T003 Audit `src/holodeck/deploy/dockerfile.py` Jinja2 template (`HOLODECK_DOCKERFILE_TEMPLATE`) for security: verify non-root user switch happens after all `COPY`/`RUN` operations, `chown` covers all copied files, no writable paths outside `/app` are assumed.

**Checkpoint**: Current security posture documented — gaps identified for US3 implementation.

---

## Phase 2: Foundational — US1/US2 Dependency Verification

**Purpose**: Verify that US1 (serve infrastructure) and US2 (base Dockerfile with Node.js) provide the foundation US3 builds upon.

- [ ] T004 Verify US2 has added conditional Node.js installation block (`{% if needs_nodejs %}`) to `src/holodeck/deploy/dockerfile.py` template. US3 security hardening layers build on top of this Node.js block.
- [ ] T005 Verify US2 has added Node.js installation to `docker/Dockerfile` base image. US3 non-root and tmpfs changes must not break the Node.js layer or prevent the `node` binary from being executable by the `holodeck` user.
- [ ] T006 Verify US1 serve infrastructure passes environment variables through to `ClaudeBackend`. Confirm that `build_options()` in `src/holodeck/lib/backends/claude_backend.py` (line 272) constructs `env` dict by merging auth + otel vars but does NOT replace `os.environ` — inherited env vars like `ANTHROPIC_BASE_URL` must reach the SDK subprocess.

**Checkpoint**: US1/US2 foundations confirmed — US3 implementation can proceed safely.

---

## Phase 3: User Story 3 — Secure Container Deployment

**Purpose**: Implement and test all acceptance scenarios for secure container deployment with credential isolation.

### 3a. Non-Root User and Directory Permissions

- [ ] T007 [P] [US3] Write unit test `test_dockerfile_template_non_root_user_final_layer` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile via `generate_dockerfile()` with `needs_nodejs=True`, assert the last `USER` directive is `USER holodeck` (not root), and that it appears after all `COPY` and `RUN` commands. Verify the test FAILS if the USER directive is removed.
- [ ] T008 [P] [US3] Write unit test `test_dockerfile_template_chown_covers_all_copied_paths` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile with `instruction_files=["instructions.md"]` and `data_directories=["data/"]`, assert `chown -R holodeck:holodeck /app` appears after all COPY operations, ensuring all files are owned by the non-root user.
- [ ] T009 [P] [US3] Write unit test `test_dockerfile_template_tmpfs_compatible_dirs` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile with `needs_nodejs=True`, assert the template creates a `/tmp` directory with holodeck ownership OR does not write to any path outside `/app` and `/tmp`. Verify the container can function when `/` is read-only and only `/tmp` is writable.
- [ ] T010 [P] [US3] Modify `HOLODECK_DOCKERFILE_TEMPLATE` in `src/holodeck/deploy/dockerfile.py` to add a tmpfs-compatible working directory block: after the `WORKDIR /app` line, add `RUN mkdir -p /tmp/holodeck && chown holodeck:holodeck /tmp/holodeck` so the Claude SDK subprocess has a writable scratch space when root filesystem is read-only. Add `ENV TMPDIR=/tmp/holodeck` to direct temp file operations to the owned directory.
- [ ] T011 [P] [US3] Modify `docker/Dockerfile` base image to add tmpfs-compatible directory: after the `WORKDIR /app` line and before `USER holodeck`, add `RUN mkdir -p /tmp/holodeck && chown holodeck:holodeck /tmp/holodeck`. Set `ENV TMPDIR=/tmp/holodeck`.

### 3b. Capability Drop Compatibility

- [ ] T012 [P] [US3] Write unit test `test_entrypoint_no_privileged_operations` in `tests/unit/deploy/test_entrypoint.py` (new file) — parse `docker/entrypoint.sh` as text, assert it contains no `apt-get`, `yum`, `apk`, `mount`, `chmod` (except on `/app/entrypoint.sh` during build), `chown`, or other operations requiring elevated capabilities. This ensures the entrypoint works with `--cap-drop ALL`.
- [ ] T013 [P] [US3] Write unit test `test_dockerfile_no_runtime_root_operations` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile via `generate_dockerfile()`, parse the output, assert that no `RUN` commands appear after the final `USER holodeck` directive. All root operations must complete before the user switch.
- [ ] T014 [P] [US3] Write integration test `test_generated_dockerfile_cap_drop_compatible` in `tests/unit/deploy/test_dockerfile.py` — generate a full Dockerfile with `needs_nodejs=True`, parse it line by line, build a list of operations that occur after `USER holodeck`, assert none of them require root capabilities (no `apt-get`, `pip install --system`, `chmod` on system paths, etc.).

### 3c. Read-Only Root Filesystem Support

- [ ] T015 [P] [US3] Write unit test `test_dockerfile_template_no_writable_root_assumptions` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile, scan all `ENV`, `RUN`, and `COPY` directives, assert no environment variable points to a path outside `/app`, `/tmp`, or `/home/holodeck` as a writable location. This validates read-only root filesystem compatibility.
- [ ] T016 [P] [US3] Write unit test `test_entrypoint_uses_only_tmpfs_safe_paths` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, extract all file paths referenced (via regex for absolute paths), assert all writable paths are under `/app`, `/tmp`, or `/home/holodeck`. No writes to `/var`, `/etc`, `/usr`, etc.
- [ ] T017 [US3] Modify `docker/entrypoint.sh` to set `TMPDIR=/tmp/holodeck` if not already set, ensuring Claude SDK subprocess temp files go to the tmpfs-mounted directory. Add after the default configuration block (line 19): `HOLODECK_TMPDIR="${TMPDIR:-/tmp/holodeck}"` and `export TMPDIR="${HOLODECK_TMPDIR}"`.

### 3d. Environment Variable Pass-Through (Proxy Pattern)

- [ ] T018 [P] [US3] Write unit test `test_env_passthrough_anthropic_base_url` in `tests/unit/lib/backends/test_claude_backend.py` — mock `os.environ` to include `ANTHROPIC_BASE_URL=http://localhost:8081`, call the standalone `build_options()` function from `holodeck.lib.backends.claude_backend` (module-level function at line 237 of `claude_backend.py`, not a method on `ClaudeBackend`) with a mocked `Agent` config, assert the resulting `ClaudeAgentOptions.env` either contains `ANTHROPIC_BASE_URL` or (by design) does not filter it from the inherited environment. The key assertion: the SDK subprocess must receive this variable.
- [ ] T019 [P] [US3] Write unit test `test_env_passthrough_http_proxy_vars` in `tests/unit/lib/backends/test_claude_backend.py` — mock `os.environ` to include `HTTP_PROXY=http://proxy:3128` and `HTTPS_PROXY=http://proxy:3128`, call the standalone `build_options()` function from `holodeck.lib.backends.claude_backend` (module-level function at line 237 of `claude_backend.py`, not a method on `ClaudeBackend`) with a mocked `Agent` config, assert proxy variables are not stripped or overridden by the `env` dict construction at `claude_backend.py` line 272. Verify the `env` dict merges with (not replaces) inherited environment.
- [ ] T020 [P] [US3] Write unit test `test_env_passthrough_anthropic_api_key` in `tests/unit/lib/backends/test_claude_backend.py` — mock `os.environ` to include `ANTHROPIC_API_KEY=sk-test-key`, call the standalone `build_options()` function from `holodeck.lib.backends.claude_backend` (module-level function at line 237 of `claude_backend.py`, not a method on `ClaudeBackend`) with a mocked `Agent` config, assert the API key reaches the subprocess environment (either via explicit `env` inclusion or inherited passthrough).
- [ ] T021 [US3] If the standalone `build_options()` function in `src/holodeck/lib/backends/claude_backend.py` (module-level function at line 237, not a method on `ClaudeBackend`) constructs the `env` dict in a way that replaces (not merges with) the inherited environment, fix it. The `env` kwarg on `ClaudeAgentOptions` must be additive — it adds/overrides specific keys but does not prevent other inherited env vars from reaching the subprocess. Verify by reading Claude Agent SDK docs for `ClaudeAgentOptions.env` behavior.
- [ ] T022 [P] [US3] Write unit test `test_dockerfile_env_passthrough_vars_declared` in `tests/unit/deploy/test_dockerfile.py` — generate a Dockerfile with `needs_nodejs=True`, assert it does NOT hardcode `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `HTTP_PROXY`, or `HTTPS_PROXY` to any value. These must be injected at `docker run` time, not baked into the image.

### 3e. Entrypoint Fixes for Non-Root Execution

- [ ] T023 [P] [US3] Write unit test `test_entrypoint_positional_arg_not_config_flag` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, extract the `holodeck serve` invocation line, assert it uses positional argument format (`holodeck serve ${HOLODECK_AGENT_CONFIG}`) NOT `--config` flag format (`holodeck serve --config ${HOLODECK_AGENT_CONFIG}`). This validates the FR-011 fix.
- [ ] T024 [US3] Fix `docker/entrypoint.sh` line 111: change `CMD_ARGS="--config ${HOLODECK_AGENT_CONFIG}"` to build the command with the config as a positional argument. The corrected invocation should be: `exec holodeck serve "${HOLODECK_AGENT_CONFIG}" --port "${HOLODECK_PORT}" --protocol "${HOLODECK_PROTOCOL}"`. Remove the `CMD_ARGS` variable pattern in favor of direct argument passing for clarity.
- [ ] T025 [P] [US3] Write unit test `test_entrypoint_exec_replaces_shell` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, assert the `holodeck serve` invocation uses `exec` prefix, ensuring the shell process is replaced and the Python process receives signals directly (critical for non-root graceful shutdown with `--cap-drop ALL`).

### 3f. Claude-Specific Entrypoint Validation

- [ ] T026 [P] [US3] Write unit test `test_entrypoint_claude_validation_function_exists` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, assert a `validate_claude_requirements` function (or equivalent) is defined that checks for `node` binary availability and `ANTHROPIC_API_KEY` (or `ANTHROPIC_BASE_URL`) presence.
- [ ] T027 [US3] Add `validate_claude_requirements()` function to `docker/entrypoint.sh` — check if the agent config YAML contains `provider: anthropic` (using `grep`), and if so: (1) verify `node --version` succeeds, (2) verify `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL` is set. On failure, call `log_error` with actionable message and `exit 1`. Call this function from `main()` after `validate_config` and before launching serve.
- [ ] T028 [P] [US3] Write unit test `test_entrypoint_claude_validation_checks_node` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, assert the `validate_claude_requirements` function body contains a check for `node` (e.g., `command -v node` or `which node` or `node --version`).
- [ ] T029 [P] [US3] Write unit test `test_entrypoint_claude_validation_checks_credentials` in `tests/unit/deploy/test_entrypoint.py` — parse `docker/entrypoint.sh`, assert the `validate_claude_requirements` function body checks for `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL` environment variable presence.

### 3g. Inline Entrypoint Security (deploy build output)

- [ ] T030 [P] [US3] Write unit test `test_inline_entrypoint_non_root_compatible` in `tests/unit/cli/commands/test_deploy.py` (following existing CLI command test naming pattern in `tests/unit/cli/commands/`) — locate the inline entrypoint string in `src/holodeck/cli/commands/deploy.py` `_prepare_build_context()`, assert it contains no root-only operations (`apt-get`, `chmod` on system paths, etc.) and uses `exec` for the `holodeck serve` invocation.
- [ ] T031 [P] [US3] Write unit test `test_inline_entrypoint_claude_validation` in `tests/unit/cli/commands/test_deploy.py` (following existing CLI command test naming pattern in `tests/unit/cli/commands/`) — for a Claude agent config (`provider: anthropic`), verify the inline entrypoint generated by `_prepare_build_context()` includes Node.js validation (`node --version` check) before launching `holodeck serve`.
- [ ] T032 [US3] Modify the inline entrypoint in `src/holodeck/cli/commands/deploy.py` `_prepare_build_context()` to add Claude-specific validation when the agent provider is anthropic: insert a `node --version` check and `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL` check before the `exec holodeck serve` line.

**Checkpoint**: All acceptance scenarios have corresponding tests and implementations.

---

## Phase 4: Polish — Code Quality and Regression Verification

**Purpose**: Ensure all changes pass code quality checks and introduce no regressions.

- [ ] T033 [US3] Run `make format` to format all new and modified files with Black + Ruff.
- [ ] T034 [US3] Run `make lint` and fix any Ruff + Bandit violations in US3 files: `src/holodeck/deploy/dockerfile.py`, `docker/entrypoint.sh`, `src/holodeck/cli/commands/deploy.py`, and all new test files.
- [ ] T035 [US3] Run `make type-check` and fix any MyPy errors in US3 modified Python files.
- [ ] T036 [US3] Run `pytest tests/unit/deploy/ -n auto -v` to verify all US3 unit tests pass.
- [ ] T037 [US3] Run `pytest tests/unit/lib/backends/test_claude_backend.py -n auto -v -k "env_passthrough"` to verify all environment pass-through tests pass.
- [ ] T038 [US3] Run `make test` to verify no regressions across the full test suite (FR-009).

**Checkpoint**: US3 complete — secure container deployment validated, code quality verified, no regressions.

---

## Dependencies & Execution Order

### External Dependencies (BLOCKING)

- **US1 (Serve Infrastructure)**: Must be complete before US3. US3 depends on serve pre-flight validation infrastructure and `build_options()` env handling.
- **US2 (Container Build with Node.js)**: Must be complete before US3. US3 security hardening layers on top of the Node.js Dockerfile additions from US2.

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — audit tasks can start immediately
- **Phase 2 (Foundational)**: Depends on US1 + US2 completion
- **Phase 3 (US3 Tasks)**: Depends on Phase 2 verification
- **Phase 4 (Polish)**: Depends on Phase 3 completion

### Within Phase 3

- **3a (Non-Root User)**, **3b (Capability Drop)**, **3c (Read-Only FS)**, **3d (Env Pass-Through)** can run in parallel — they modify different files
- **3e (Entrypoint Fixes)** can run in parallel with 3a-3d — modifies `docker/entrypoint.sh` (shared with 3c T017, sequence T017 before T024)
- **3f (Claude Validation)** depends on 3e (entrypoint structure must be fixed first)
- **3g (Inline Entrypoint)** can run in parallel with 3e/3f — modifies `deploy.py` not `entrypoint.sh`

---

## Parallel Execution Examples

### Example 1: Phase 3a + 3b + 3c + 3d in parallel

```
Worker 1 (3a): T007, T008, T009, T010, T011 — Dockerfile non-root + tmpfs dirs
Worker 2 (3b): T012, T013, T014 — capability drop compatibility tests
Worker 3 (3c): T015, T016, T017 — read-only FS tests + TMPDIR fix
Worker 4 (3d): T018, T019, T020, T021, T022 — env var pass-through
```

### Example 2: Phase 3e + 3g in parallel

```
Worker 1 (3e): T023, T024, T025 — entrypoint positional arg fix
Worker 2 (3g): T030, T031, T032 — inline entrypoint security
```

### Example 3: All [P] tests from Phase 3

```bash
# Run all parallelizable US3 tests:
pytest tests/unit/deploy/test_dockerfile.py \
       tests/unit/deploy/test_entrypoint.py \
       tests/unit/cli/commands/test_deploy.py \
       tests/unit/lib/backends/test_claude_backend.py \
       -n auto -v -k "non_root or cap_drop or tmpfs or env_passthrough or positional_arg or read_only"
```

---

## Implementation Strategy

### Defense-in-Depth Security Layers

US3 implements Anthropic's secure deployment guidelines through four complementary layers:

1. **Non-root execution** (3a, 3e): The container runs as `holodeck` (uid 1000). All files are `chown`ed before the `USER` switch. The entrypoint uses `exec` for proper signal handling.

2. **Capability drop compatibility** (3b): No runtime operations require elevated Linux capabilities. All privileged operations (`apt-get`, `chown`, `chmod`) happen during `docker build`, not at `docker run` time.

3. **Read-only root filesystem** (3c): Writable paths are limited to `/tmp/holodeck` (tmpfs-mounted at runtime) and `/app` (if not read-only). The `TMPDIR` env var directs all temp file creation to the tmpfs path.

4. **Credential isolation** (3d, 3f): API keys and proxy URLs are injected at runtime via `docker run -e`, never baked into the image. The `ANTHROPIC_BASE_URL` env var enables the proxy pattern where a local credential-injecting proxy handles authentication.

### Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/deploy/dockerfile.py` | 3a | Add tmpfs-compatible dir, TMPDIR env var |
| `docker/Dockerfile` | 3a | Add tmpfs-compatible dir, TMPDIR env var |
| `docker/entrypoint.sh` | 3c, 3e, 3f | Fix positional arg bug, add TMPDIR, add Claude validation |
| `src/holodeck/cli/commands/deploy.py` | 3g | Add Claude validation to inline entrypoint |
| `tests/unit/deploy/test_dockerfile.py` | 3a, 3b, 3c, 3d | Add security-focused Dockerfile tests |
| `tests/unit/deploy/test_entrypoint.py` | 3b, 3c, 3e, 3f | NEW: Entrypoint security and fix tests |
| `tests/unit/cli/commands/test_deploy.py` | 3g | Add inline entrypoint security tests |
| `tests/unit/lib/backends/test_claude_backend.py` | 3d | Add env var pass-through tests |

### Acceptance Scenario Traceability

| Acceptance Scenario | Validating Tasks |
|---------------------|-----------------|
| AS-1: `ANTHROPIC_BASE_URL` routes through proxy | T018, T021, T022 |
| AS-2: `--cap-drop ALL --security-opt no-new-privileges` works | T012, T013, T014, T025 |
| AS-3: `--read-only --tmpfs /tmp` works | T009, T010, T011, T015, T016, T017 |
| AS-4: `--user 1000:1000` non-root works | T007, T008, T023, T024, T030 |

### FR Traceability

| Requirement | Validating Tasks |
|-------------|-----------------|
| FR-003: Secure Dockerfile (non-root, capability-aware, read-only FS) | T007-T017 |
| FR-006: Env var pass-through to Claude SDK subprocess | T018-T022 |
| FR-011: Entrypoint positional arg fix | T023, T024 |

---

## Notes

- [P] tasks = different files or independent test functions, no dependencies
- [US3] label maps all tasks to User Story 3 for traceability
- Entrypoint tests parse shell scripts as text (not execute them) for unit-level verification
- Environment pass-through tests mock `os.environ` to verify subprocess env construction
- The `--config` bug fix (T024) is a prerequisite for any container deployment — it must be verified before US3 is considered complete
- The `TMPDIR=/tmp/holodeck` pattern ensures Claude SDK subprocess temp files land in tmpfs without requiring the SDK to be aware of the deployment topology
- Commit after each sub-phase (3a, 3b, etc.) or logical group of tasks
- Stop at any checkpoint to validate progress independently
