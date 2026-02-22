# Tasks: Phase 9 — Test Runner Updates — Claude Agent SDK

**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Prerequisites**: Phases 1–8 complete (base.py, validators.py, sk_backend.py, claude_backend.py, selector.py, tool_adapters.py, mcp_bridge.py, otel_bridge.py)

**Tests**: TDD approach — write failing tests first, then implement.

**Organization**: Phase 9 updates the test runner and CLI to consume the new `BackendSelector` / `ExecutionResult` interface. Tasks are organized into three workstreams:
- (A) Test Runner `executor.py` refactor to use `BackendSelector` + `ExecutionResult`
- (B) `agent_factory.py` thin facade for backward compatibility
- (C) `--allow-side-effects` CLI flag on `holodeck test`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US4)
- Include exact file paths in descriptions

---

## Phase 9A: Test Runner Executor — BackendSelector Integration

**Goal**: Update `executor.py` to use `BackendSelector.select()` instead of directly constructing `AgentFactory`, and consume `ExecutionResult` instead of `AgentExecutionResult`. Remove all SK-specific type imports from the test runner layer.

**Plan Reference**: `plan.md` lines 477–498 (Phase 9: Test Runner Updates)
**Spec Reference**: `spec.md` lines 77–89 (US4 — Run Evaluations Against Claude-Native Agents), lines 218–239 (FR-008, FR-012b, FR-013)
**Research Reference**: `research.md` lines 234–243 (§8 — Retry Behaviour)
**Data Model Reference**: `data-model.md` lines 202–257 (§4 — ExecutionResult), lines 362–376 (§7 — Entity Relationships)
**Contract Reference**: `contracts/execution-result.md` lines 17–72 (ExecutionResult, Error Conditions), lines 100–128 (AgentBackend Lifecycle)

**Independent Test**: Run `holodeck test` against a non-Anthropic agent fixture. All existing test cases execute and produce evaluation scores. No SK types leak into the executor layer.

### Tests for Phase 9A (TDD — write first, verify they FAIL)

- [x] T001 [P] [US4] Write unit test for `TestExecutor` using `BackendSelector` instead of `AgentFactory` in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `plan.md:481–484` ("Replace AgentFactory(config).create_thread_run() with BackendSelector.select(agent, tools)")
  - Mock `BackendSelector.select()` to return a mock `AgentBackend`.
  - Mock `backend.invoke_once()` to return an `ExecutionResult(response="test response", tool_calls=[], tool_results=[])`.
  - Execute a single test case through `TestExecutor._execute_single_test()`.
  - Assert the agent response is correctly extracted from `ExecutionResult.response`.
  - Assert no imports of `ChatHistory`, `FunctionCallContent`, or `FunctionResultContent` exist in `executor.py`.

- [x] T002 [P] [US4] Write unit test for `ExecutionResult` error handling in test executor in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `plan.md:485–486` ("max_turns exceeded → mark test as failed, not evaluation error"), `contracts/execution-result.md:66–72`
  - Mock `backend.invoke_once()` to return `ExecutionResult(response="partial", is_error=True, error_reason="max_turns limit reached")`.
  - Assert test result has `passed=False`.
  - Assert errors list contains "max_turns limit reached".
  - Assert agent_response is preserved ("partial") for possible evaluation.

- [x] T003 [P] [US4] Write unit test for subprocess crash error handling in test executor in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `plan.md:486` ("Subprocess crash → mark as execution error, continue test suite"), `contracts/execution-result.md:69`
  - Mock `backend.invoke_once()` to return `ExecutionResult(response="", is_error=True, error_reason="subprocess terminated unexpectedly")`.
  - Assert test result has `passed=False`.
  - Assert errors list contains "subprocess terminated unexpectedly".
  - Assert the test suite continues to execute subsequent test cases (not aborted).

- [x] T004 [P] [US4] Write unit test for `BackendSessionError` exception handling in test executor in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `plan.md:486`, `contracts/execution-result.md:134–146` (BackendSessionError)
  - Mock `backend.invoke_once()` to raise `BackendSessionError("subprocess crashed")`.
  - Assert test result has `passed=False`.
  - Assert errors list contains the exception message.
  - Assert subsequent test cases still execute.

- [x] T005 [P] [US4] Write unit test for tool call extraction from `ExecutionResult` in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `plan.md:483` ("Remove all ChatHistory, FunctionCallContent imports"), `contracts/execution-result.md:36–52`
  - Mock `backend.invoke_once()` to return `ExecutionResult(response="found it", tool_calls=[{"name": "kb_search", "arguments": {"query": "refund"}, "call_id": "t01"}], tool_results=[{"call_id": "t01", "result": "30-day policy", "is_error": False}])`.
  - Assert `extract_tool_names(result.tool_calls)` returns `["kb_search"]`.
  - Note: `extract_tool_names()` in `chat_history_utils.py` already accepts `list[dict[str, Any]]` — confirmed compatible with `ExecutionResult.tool_calls` format.
  - Assert tool results are correctly passed to evaluation.
  - Assert `validate_tool_calls()` works with `expected_tools=["kb_search"]`.

- [x] T006 [P] [US4] Write unit test for backend lifecycle (initialize + invoke + teardown) in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `contracts/execution-result.md:114–128` (AgentBackend Lifecycle)
  - Mock `BackendSelector.select()` to return a mock backend.
  - Run `execute_tests()` with 2 test cases.
  - Assert `backend.invoke_once()` is called exactly 2 times (one per test case).
  - Assert `backend.teardown()` is called during `executor.shutdown()`.

- [x] T007 [P] [US4] Write unit test for token usage extraction from `ExecutionResult` in `tests/unit/lib/test_runner/test_executor.py`
  - **Ref**: `data-model.md:231–238` (token_usage field), `contracts/execution-result.md:54–60`
  - Mock `backend.invoke_once()` returning `ExecutionResult` with `token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)`.
  - Assert `result.token_usage` is accessible and correctly populated on `ExecutionResult`.
  - Note: Wiring `token_usage` into `TestResult` is deferred to a future phase. This test only verifies `ExecutionResult` carries the data.

### Implementation for Phase 9A

- [x] T008 [US4] Refactor `TestExecutor.__init__()` to accept `AgentBackend` instead of `AgentFactory` in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:481` ("Replace AgentFactory(config) with BackendSelector.select(agent, tools)")
  - Add optional `backend: AgentBackend | None = None` parameter to `__init__()`.
  - Keep `agent_factory` parameter for backward compatibility (deprecated, emit warning if used).
  - Add `_backend: AgentBackend | None` instance variable.
  - When `_backend` is set, skip `_create_agent_factory()` entirely — `BackendSelector.select()` → `backend.initialize()` handles all tool init (vectorstore, MCP, hierarchical docs) internally.
  - Note: `_create_agent_factory()` is kept for the legacy `agent_factory` injection path only. It will be removed entirely after chat migration in Phase 10.
  - The actual switch to `BackendSelector` happens in T010 (the `_execute_single_test` method).

- [x] T009 [US4] Add `BackendSelector` import and remove SK-specific type imports from `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:483–484` ("Remove all ChatHistory, FunctionCallContent, FunctionResultContent imports")
  - Add: `from holodeck.lib.backends import BackendSelector, ExecutionResult, BackendSessionError`
  - Remove: Any direct imports of `ChatHistory`, `FunctionCallContent`, `FunctionResultContent` (verify none exist in current executor.py — they may be indirect via `agent_factory`).
  - Keep: `from holodeck.lib.chat_history_utils import extract_tool_names` (this is SK-free).
  - Keep: `from holodeck.lib.test_runner.agent_factory import AgentFactory` (for backward compat in T012).

- [x] T010 [US4] Refactor `TestExecutor._execute_single_test()` to use `BackendSelector` + `ExecutionResult` in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:481–486`, `contracts/execution-result.md:114–128`
  - Replace the current invocation block (lines 543–551):
    ```python
    # OLD:
    thread_run = await self.agent_factory.create_thread_run()
    result = await thread_run.invoke(agent_input)
    agent_response = result.response
    tool_calls = extract_tool_names(result.tool_calls)
    tool_results = result.tool_results
    ```
    With backend-based invocation:
    ```python
    # NEW:
    result = await self._backend.invoke_once(agent_input)
    if result.is_error:
        errors.append(f"Agent error: {result.error_reason}")
        agent_response = result.response  # preserve partial response
    else:
        agent_response = result.response
    tool_calls = extract_tool_names(result.tool_calls)
    tool_results = result.tool_results
    ```
  - Add `max_turns` exceeded detection: if `result.is_error and result.error_reason == "max_turns limit reached"` → append to errors, mark test as failed (NOT evaluation error).
  - Add subprocess crash detection: if `result.is_error and "subprocess terminated" in (result.error_reason or "")` → append to errors as execution error, continue test suite.
  - Wrap invocation in `try/except BackendSessionError` for unexpected backend failures.

- [x] T011 [US4] Add async backend initialization to `TestExecutor` in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `contracts/execution-result.md:118–120` ("backend.initialize() → Tool index setup, credential validation")
  - Add `async _ensure_backend_initialized()` method that:
    1. If `_backend` is already set, return immediately.
    2. Otherwise, call `BackendSelector.select(self.agent_config, tool_instances=None, mode="test", allow_side_effects=self._allow_side_effects)`.
    3. Store the returned backend in `self._backend`.
  - Call `_ensure_backend_initialized()` at the start of `execute_tests()`.
  - Update `shutdown()` to call `self._backend.teardown()` if backend exists.
  - **Tool init clarification**: `tool_instances=None` is correct for all paths:
    - SK path: `BackendSelector.select()` → `SKBackend` wraps `AgentFactory` which builds tools (vectorstore, MCP, hierarchical docs) internally during `initialize()`.
    - Claude path: `BackendSelector.select()` → `ClaudeBackend` builds tool adapters via `build_holodeck_sdk_server()` internally during `initialize()`.
    - Executor never needs to know about provider-specific tool init.

- [x] T012 [US4] Update `TestExecutor._create_agent_factory()` for backward compatibility in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:488–491` ("Thin facade: AgentFactory delegates to BackendSelector, preserves public API")
  - Add private method `_invoke_via_legacy_factory(self, agent_input: str) -> ExecutionResult` that:
    1. Calls `self.agent_factory.create_thread_run()`.
    2. Calls `thread_run.invoke(agent_input)`.
    3. Converts `AgentExecutionResult` → `ExecutionResult` (map response, tool_calls, tool_results, token_usage).
    4. Returns the `ExecutionResult`.
  - In `_execute_single_test()`: if `self._backend` is set, use `self._backend.invoke_once()`; else if `self.agent_factory` is set, use `self._invoke_via_legacy_factory()`.
  - This ensures existing unit tests that mock `AgentFactory` continue to work during transition.

- [x] T013 [US4] Add `_allow_side_effects` instance variable to `TestExecutor` in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:493–496` ("--allow-side-effects flag")
  - Add `allow_side_effects: bool = False` parameter to `__init__()`.
  - Store as `self._allow_side_effects`.
  - Pass to `BackendSelector.select()` in T011.
  - This variable is populated from the CLI flag added in Phase 9C (T020).

**Checkpoint**: At this point, `TestExecutor` should use `BackendSelector` for all invocations. Existing non-Anthropic tests must still pass. Run: `pytest tests/unit/lib/test_runner/test_executor.py -n auto -v`

---

## Phase 9B: AgentFactory Backward Compatibility Facade

**Goal**: Ensure `AgentFactory` continues to work as a public API entry point for any code that imports it directly, while internally delegating to the backend abstraction.

**Plan Reference**: `plan.md` lines 488–491 (Phase 9.2 — agent_factory.py update)
**Spec Reference**: `spec.md` lines 232–233 (FR-010 — non-Anthropic agents work without modification)

**Independent Test**: Any existing code that calls `AgentFactory(config).create_thread_run()` continues to work. `AgentExecutionResult` is still importable from `agent_factory`.

### Tests for Phase 9B (TDD — write first, verify they FAIL)

- [x] T014 [P] [US1] Write unit test verifying `AgentExecutionResult` is re-exported from `agent_factory.py` in `tests/unit/lib/test_runner/test_agent_factory.py`
  - **Ref**: `plan.md:490` ("AgentExecutionResult re-exported from backends.base.ExecutionResult for backward compat")
  - Assert `from holodeck.lib.test_runner.agent_factory import AgentExecutionResult` still works.
  - Assert `AgentExecutionResult` fields are a superset of `ExecutionResult` fields (response, tool_calls, tool_results, token_usage).

- [x] T015 [P] [US1] Write unit test verifying existing `AgentFactory.create_thread_run()` still works for non-Anthropic providers in `tests/unit/lib/test_runner/test_agent_factory.py`
  - **Ref**: `plan.md:489` ("Preserves public API for any code that imports AgentFactory directly")
  - Use existing test fixtures for OpenAI/Azure/Ollama agents.
  - Assert `AgentFactory(config).create_thread_run()` returns an `AgentThreadRun`.
  - Assert `thread_run.invoke("Hello")` returns an `AgentExecutionResult`.
  - **This test should ALREADY pass** — the goal is to confirm it does NOT break during Phase 9.

### Implementation for Phase 9B

- [x] T016 [US1] Ensure `AgentExecutionResult` backward compatibility in `src/holodeck/lib/test_runner/agent_factory.py`
  - **Ref**: `plan.md:490`
  - `AgentExecutionResult` already exists as a separate dataclass in `agent_factory.py`. It stays as-is (NOT re-exported from `ExecutionResult` — they are separate classes with different fields; `AgentExecutionResult` has `chat_history`).
  - Add a deprecation comment noting it is kept for backward compatibility and new code should use `ExecutionResult` from `backends.base`.
  - **No functional changes needed** — `AgentExecutionResult` is already the return type of `AgentThreadRun.invoke()`. The SK backend path is unchanged.

- [x] T017 [US1] Verify all existing `AgentFactory` tests pass without modification in `tests/unit/lib/test_runner/test_agent_factory.py`
  - **Ref**: `plan.md:498` ("Run full existing test suite; zero regressions on non-Anthropic agents")
  - Run: `pytest tests/unit/lib/test_runner/test_agent_factory.py -n auto -v`
  - No test modifications expected. If any fail, fix the regression before proceeding.

**Checkpoint**: `AgentFactory` is backward compatible. All existing tests pass. Run: `pytest tests/unit/lib/test_runner/ -n auto -v`

---

## Phase 9C: `--allow-side-effects` CLI Flag

**Goal**: Add `--allow-side-effects` flag to the `holodeck test` CLI command. When absent, `ClaudeBackend` force-disables bash and file_system.write for Anthropic-provider test runs. When present, configured settings are respected with a warning.

**Plan Reference**: `plan.md` lines 493–496 (Phase 9.3 — --allow-side-effects flag)
**Spec Reference**: `spec.md` lines 125–141 (US7 — File System and Execution Access), lines 143–157 (US8 — Permission and Safety Governance)
**Research Reference**: `research.md` lines 106–118 (§3 — Permission Mode Mapping, test run override)

**Independent Test**: Run `holodeck test agent.yaml` with an Anthropic agent that has `bash.enabled: true`. Without `--allow-side-effects`, bash is force-disabled. With `--allow-side-effects`, bash is respected and a warning is emitted.

### Tests for Phase 9C (TDD — write first, verify they FAIL)

- [x] T018 [P] [US8] Write unit test for `--allow-side-effects` flag parsing in `tests/unit/cli/test_test_cmd.py`
  - **Ref**: `plan.md:493–496`
  - **Note**: `tests/unit/cli/test_test_cmd.py` is a new file — create it with standard pytest imports, Click CliRunner setup, and necessary mocks.
  - Use Click's `CliRunner` to invoke the test command.
  - Assert `--allow-side-effects` is parsed as `True` when present.
  - Assert default is `False` when absent.

- [x] T019 [P] [US8] Write unit test verifying `allow_side_effects` is passed to `TestExecutor` in `tests/unit/cli/test_test_cmd.py`
  - **Ref**: `plan.md:494` ("When absent (default): ClaudeBackend disables bash and file_system access")
  - Mock `TestExecutor.__init__` to capture the `allow_side_effects` argument.
  - Invoke `holodeck test agent.yaml --allow-side-effects` → assert `allow_side_effects=True` passed.
  - Invoke `holodeck test agent.yaml` → assert `allow_side_effects=False` passed.

### Implementation for Phase 9C

- [x] T020 [US8] Add `--allow-side-effects` option to `holodeck test` command in `src/holodeck/cli/commands/test.py`
  - **Ref**: `plan.md:493–496`, `research.md:106–118`
  - Add Click option:
    ```python
    @click.option(
        "--allow-side-effects",
        is_flag=True,
        default=False,
        help="Allow bash execution and file system writes during test runs for Anthropic agents. "
             "By default, these are force-disabled for safety.",
    )
    ```
  - Pass `allow_side_effects` to `TestExecutor.__init__()`.
  - When `allow_side_effects=True`, emit a startup warning:
    ```
    "Warning: --allow-side-effects enabled. Test run may modify files or execute shell commands."
    ```

- [x] T021 [US8] Pass `allow_side_effects` through `TestExecutor` to `BackendSelector.select()` in `src/holodeck/lib/test_runner/executor.py`
  - **Ref**: `plan.md:494–496`, `selector.py:24` (allow_side_effects parameter)
  - Ensure `_ensure_backend_initialized()` (from T011) passes `self._allow_side_effects` to `BackendSelector.select()`.
  - The `ClaudeBackend` already handles the force-disable logic (implemented in Phase 8, T018/T020).

**Checkpoint**: `--allow-side-effects` flag works end-to-end. Run: `pytest tests/unit/cli/test_test_cmd.py -n auto -v`

---

## Phase 9D: Code Quality & Verification

**Goal**: Run all code quality checks and verify zero regressions across the full test suite.

**Plan Reference**: `plan.md` lines 551–561 (Phase 12: Code Quality — run after each phase)

- [x] T022 Run `make format` to apply Black + Ruff formatting to all new/modified files
  - **Ref**: `plan.md:556`
  - Files: `src/holodeck/lib/test_runner/executor.py`, `src/holodeck/lib/test_runner/agent_factory.py`, `src/holodeck/cli/commands/test.py`, test files.

- [x] T023 Run `make lint-fix` to auto-fix linting issues in new/modified files
  - **Ref**: `plan.md:557`

- [x] T024 Run `make type-check` to verify MyPy strict mode passes for all modified code
  - **Ref**: `plan.md:558`
  - Ensure `executor.py` type hints reference `AgentBackend` and `ExecutionResult` (not SK types).

- [x] T025 Run `make test-unit` to verify zero regressions across the full unit test suite
  - **Ref**: `plan.md:559`, `plan.md:498` ("Run full existing test suite; zero regressions")
  - Run: `pytest tests/unit/ -n auto -v`
  - All existing tests from Phases 1–8 must still pass.
  - All new Phase 9 tests must pass.

- [x] T026 Run `make security` to verify Bandit + Safety + detect-secrets pass
  - **Ref**: `plan.md:560`
  - No new security warnings allowed in modified files.

**Checkpoint**: Full CI pipeline green. Ready for Phase 10 (Streaming Chat Refactor).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 9A Tests (T001–T007)**: No dependencies on each other — all [P] parallelizable. All depend on Phases 1–8 being complete.
- **Phase 9A Implementation (T008–T013)**: T009 must come first (imports). T008, T010, T011 are sequential (each builds on prior). T012 depends on T008+T010. T013 depends on T008.
- **Phase 9B Tests (T014–T015)**: Depend on Phase 9A implementation being complete. Both [P] parallelizable.
- **Phase 9B Implementation (T016–T017)**: T016 is mostly a no-op (verify existing code). T017 is a verification step.
- **Phase 9C Tests (T018–T019)**: Depend on Phase 9A implementation being complete. Both [P] parallelizable.
- **Phase 9C Implementation (T020–T021)**: T020 depends on T018–T019 (tests written). T021 depends on T013 (allow_side_effects variable) and T020 (CLI flag).
- **Phase 9D Quality (T022–T026)**: Depends on all Phase 9A + 9B + 9C being complete. T022–T024 are parallelizable. T025 depends on T022–T024. T026 depends on T025.

### Execution Order (Critical Path)

```
T001–T007 (parallel)  ─── all Phase 9A tests
      │
T009                  ─── add imports, remove SK types
      │
T008, T013            ─── __init__ refactor + allow_side_effects variable
      │
T010                  ─── _execute_single_test refactor (core change)
      │
T011                  ─── async backend initialization
      │
T012                  ─── backward compatibility adapter
      │
T014–T015 (parallel)  ─── Phase 9B tests
      │
T016, T017            ─── Phase 9B implementation + verification
      │
T018–T019 (parallel)  ─── Phase 9C tests
      │
T020, T021            ─── Phase 9C implementation (CLI flag + passthrough)
      │
T022–T024 (parallel)  ─── formatting, linting, type-check
      │
T025                  ─── full test suite
      │
T026                  ─── security scan
```

### Within Phase 9A Implementation

- T009 (imports) → T008 (__init__ refactor) → T013 (allow_side_effects) → T010 (_execute_single_test) → T011 (backend init) → T012 (backward compat)
- T008 and T013 can be done in either order before T010.

### Parallel Opportunities

```bash
# Launch all Phase 9A tests in parallel:
Task: T001 — BackendSelector integration test
Task: T002 — max_turns error handling test
Task: T003 — subprocess crash error handling test
Task: T004 — BackendSessionError exception test
Task: T005 — tool call extraction test
Task: T006 — backend lifecycle test
Task: T007 — token usage extraction test

# Launch all Phase 9B tests in parallel:
Task: T014 — AgentExecutionResult re-export test
Task: T015 — existing AgentFactory test

# Launch all Phase 9C tests in parallel:
Task: T018 — --allow-side-effects flag parsing test
Task: T019 — allow_side_effects passthrough test

# Launch quality checks in parallel:
Task: T022 — format
Task: T023 — lint-fix
Task: T024 — type-check
```

---

## Implementation Strategy

### MVP First (Phase 9A Only)

1. Write all T001–T007 tests (TDD — verify they FAIL)
2. Implement T008–T013 (executor.py refactor)
3. Verify all Phase 9A tests pass
4. **STOP and VALIDATE**: `TestExecutor` works with `BackendSelector` for both SK and Claude backends
5. Run existing test suite to confirm zero regressions

### Full Phase 9

1. Complete Phase 9A (MVP above)
2. Write + implement Phase 9B (T014–T017 — backward compatibility)
3. Write + implement Phase 9C (T018–T021 — CLI flag)
4. Run quality checks (T022–T026)
5. **STOP and VALIDATE**: Full `holodeck test` command works with both backends

### User Story Coverage

| User Story | Tasks | Key Deliverable |
|---|---|---|
| US1 (Run Claude-Native Agent) | T014–T017 | AgentFactory backward compatibility |
| US4 (Run Evaluations) | T001–T013 | Test runner uses BackendSelector + ExecutionResult |
| US7 (File System Access) | T020–T021 | --allow-side-effects flag |
| US8 (Permission Governance) | T018–T021 | Safe defaults for test runs |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- TDD: write tests first, verify they fail, then implement
- All `claude_agent_sdk` calls remain mocked in unit tests — no API key needed
- Integration tests (requiring real API keys) are deferred to Phase 11 per plan.md:541–548
- The key architectural change: `TestExecutor` no longer creates `AgentFactory` directly — it uses `BackendSelector.select()` which returns an `AgentBackend` (either `SKBackend` or `ClaudeBackend`)
- `AgentFactory` and `AgentExecutionResult` remain importable for backward compatibility but new code should use `BackendSelector` and `ExecutionResult`
- Commit after each logical group (Phase 9A tests → Phase 9A impl → Phase 9B → Phase 9C → quality)
- Stop at any checkpoint to validate independently
