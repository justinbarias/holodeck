# Tasks: Phase 10 — Chat Layer Decoupling + Streaming

**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Prerequisites**: Phases 1–9 complete (backend abstraction, SK refactor, test runner decoupled)

**Tests**: TDD approach requested — write tests **first**, ensure they **fail**, then implement.

**Organization**: Tasks map to Phase 10 sub-steps (10.1–10.6) from plan.md lines 503–618.
Each task references the source document and line number that defines it.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[US5]**: User Story 5 — Streaming Chat with Claude-Native Agents (spec.md lines 93–108)
- **[US1]**: User Story 1 — Run a Claude-Native Agent (spec.md lines 27–41)
- Include exact file paths in descriptions

---

## Phase 1: Setup (TDD Scaffolding)

**Purpose**: Write failing tests first that define the target API surface before any implementation changes.

> **Ref**: plan.md lines 605–616 (10.5 — Update tests); TDD approach per user request.

- [ ] T001 [P] [US1] Write test `test_execute_turn_uses_session_send` in `tests/unit/agent/test_executor.py` — inject mock `AgentBackend`/`AgentSession` via `backend=` constructor param; assert `_session.send(message)` is called and `AgentResponse` is returned. Do NOT mock `BackendSelector.select` — the injected backend bypasses the selector entirely. **(Ref: plan.md:605–607, data-model.md:416–421)**

- [ ] T002 [P] [US5] Write test `test_execute_turn_streaming_yields_chunks` in `tests/unit/agent/test_executor.py` — mock `AgentSession.send_streaming()` to yield `["Hello", " world"]`; assert `execute_turn_streaming()` yields the same chunks and history is updated after stream completes. **(Ref: plan.md:559–579, data-model.md:440–457)**

- [ ] T003 [P] [US1] Write test `test_auto_select_backend` in `tests/unit/agent/test_executor.py` — do NOT inject `backend=`; mock `BackendSelector.select()` to return a mock `AgentBackend`; call `execute_turn("hello")`; assert `BackendSelector.select` was called with `mode="chat"`. **(Ref: plan.md:529–539, research.md:356–390)**

- [ ] T004 [P] [US1] Write test `test_get_history_tracks_messages` in `tests/unit/agent/test_executor.py` — call `execute_turn()` twice; assert `get_history()` returns a `list[dict]` with 4 entries (2 user + 2 assistant) in order. **(Ref: plan.md:544–551, data-model.md:424–436)**

- [ ] T005 [P] [US1] Write test `test_clear_history_closes_session` in `tests/unit/agent/test_executor.py` — call `execute_turn()`, then `clear_history()`; assert `_session.close()` was called, `_session` is `None`, and `_history` is empty. **(Ref: plan.md:553–556)**

- [ ] T006 [P] [US1] Write test `test_shutdown_closes_session_and_backend` in `tests/unit/agent/test_executor.py` — call `execute_turn()`, then `shutdown()`; assert both `session.close()` and `backend.teardown()` are called. **(Ref: plan.md:557)**

- [ ] T007 [P] [US1] Write test `test_backend_error_wrapped` in `tests/unit/agent/test_executor.py` — mock `_session.send()` to raise `BackendSessionError`; assert `execute_turn()` wraps it in `RuntimeError` for backward compat. **(Ref: plan.md:548)**

- [ ] T008 [P] [US5] Write test `test_process_message_streaming_yields_chunks` in `tests/unit/agent/test_session.py` — mock `AgentExecutor.execute_turn_streaming()` to yield `["Hi", " there"]`; assert `ChatSessionManager.process_message_streaming()` yields same chunks and increments `session.message_count`. **(Ref: plan.md:583–586)**

- [ ] T009 [P] [US5] Write test `test_process_message_streaming_validates_message` in `tests/unit/agent/test_session.py` — call `process_message_streaming("")` (empty string); assert validation error is raised before any streaming begins. **(Ref: plan.md:583–586)**

**Checkpoint**: All 9 tests written and **failing** (implementation not yet done). Run `pytest tests/unit/agent/ -n auto` to confirm failures.

---

## Phase 2: Foundational — Executor Refactor (Blocking)

**Purpose**: Core `AgentExecutor` refactor that MUST be complete before streaming or session manager changes.

> **Ref**: plan.md lines 517–558 (10.1 — Refactor chat/executor.py); contracts/execution-result.md lines 79–110.

**⚠️ CRITICAL**: No streaming or session manager work can begin until this phase is complete.

- [ ] T010 [US1] Update imports in `src/holodeck/chat/executor.py`: **Remove** `AgentFactory`, `AgentThreadRun` (line 11), `ExecutionConfig` (line 13). **Keep** `TokenUsage` (line 14), `ToolExecution`, `ToolStatus` (line 15) — still needed by `_convert_tool_calls()`. **Add** `AgentBackend`, `AgentSession`, `ExecutionResult`, `BackendSessionError`, `BackendInitError` from `holodeck.lib.backends.base`; `BackendSelector` from `holodeck.lib.backends.selector`; `AsyncGenerator` from `collections.abc` (for streaming). **(Ref: plan.md:517–521, contracts/execution-result.md:100–110)**

- [ ] T011 [US1] Refactor `AgentExecutor.__init__()` in `src/holodeck/chat/executor.py` (currently lines 41–86). **Keep params**: `agent_config: Agent`, `on_execution_start`, `on_execution_complete`. **Add param**: `backend: AgentBackend | None = None`. **Remove params**: `enable_observability`, `timeout`, `max_retries`, `force_ingest` (these were passed to `AgentFactory` which is being removed — backend handles them via `BackendSelector`). **Replace internals**: remove `self._factory = AgentFactory(...)` and `self._thread_run: AgentThreadRun | None`; add `self._backend = backend`, `self._session: AgentSession | None = None`, `self._history: list[dict[str, Any]] = []`. Constructor stores config only — no I/O, no factory creation. Remove the `try/except` block wrapping factory init (lines 70–86). **(Ref: plan.md:523–527, data-model.md:416–421)**

- [ ] T012 [US1] Implement `_ensure_backend_and_session()` private async method in `src/holodeck/chat/executor.py` — if `_session` is not None, return early; if `_backend` is None, call `await BackendSelector.select(self.agent_config, tool_instances=None, mode="chat", allow_side_effects=False)` and store result (note: full 4-arg signature; `tool_instances` is `dict[str, Any] | None`, `None` is correct for chat — backend initializes tools lazily); then call `await self._backend.create_session()` and store in `_session`. **(Ref: plan.md:529–539, research.md:356–390, selector.py:20–25)**

- [ ] T013 [US1] Refactor `execute_turn()` in `src/holodeck/chat/executor.py` (currently lines 88–149). Preserve the full response construction pipeline:
  1. Keep `start_time = time.time()` and compute `elapsed` for `AgentResponse.execution_time`.
  2. Keep `on_execution_start(message)` callback invocation (if set).
  3. Call `await self._ensure_backend_and_session()`.
  4. Replace `_thread_run.invoke(message)` with `result = await self._session.send(message)`.
  5. Call `self._convert_tool_calls(result.tool_calls)` to produce `list[ToolExecution]` for `AgentResponse.tool_executions`.
  6. Map `result.token_usage` → `AgentResponse.tokens_used` (note: `ExecutionResult.token_usage` defaults to `TokenUsage.zero()`, not `None`).
  7. Append `{"role": "user", "content": message}` and `{"role": "assistant", "content": result.response}` to `self._history`.
  8. Keep `on_execution_complete(response)` callback invocation (if set).
  9. Catch `BackendSessionError` / `BackendInitError` → wrap in `RuntimeError` for backward compat.
  **(Ref: plan.md:544–548, executor.py:103–149)**

- [ ] T014 [US1] Refactor `get_history()` in `src/holodeck/chat/executor.py` (currently lines 151–167) — return `list(self._history)` instead of reading from `_thread_run.chat_history.messages`. Remove any SK `ChatHistory` type references. **(Ref: plan.md:550–551, data-model.md:424–436)**

- [ ] T015 [US1] Refactor `clear_history()` in `src/holodeck/chat/executor.py` (currently line 169) — change from sync to `async def`. Call `await self._session.close()` if session exists, set `self._session = None`, clear `self._history = []`. Next `execute_turn()` creates a fresh session. **(Ref: plan.md:553–556)**

- [ ] T016 [US1] Refactor `shutdown()` in `src/holodeck/chat/executor.py` (currently lines 178–191) — replace `await self._factory.shutdown()` with `await self._session.close()` (if session exists) + `await self._backend.teardown()` (if backend exists). **(Ref: plan.md:557)**

**Checkpoint**: T001, T003, T004, T005, T006, T007 tests should now **pass**. Run `pytest tests/unit/agent/test_executor.py -n auto -k "not streaming"`.

---

## Phase 3: User Story 5 — Streaming Chat (Priority: P5)

**Goal**: Add real-time token-by-token streaming from Claude-native agents to the `holodeck chat` terminal.

**Independent Test**: Run `holodeck chat` with a Claude-native agent. Text streams progressively, not all at once.

> **Ref**: spec.md lines 93–108 (US5); plan.md lines 559–601 (10.2–10.4).

### 3a: Streaming in Executor

- [ ] T017 [US5] Implement `execute_turn_streaming()` async generator in `src/holodeck/chat/executor.py` — call `_ensure_backend_and_session()`, iterate `async for chunk in self._session.send_streaming(message)`, collect chunks, yield each chunk, after stream completes append user and assistant messages to `self._history`. **(Ref: plan.md:559–579, data-model.md:440–457, quickstart.md:414–425)**

**Checkpoint**: T002 test should now **pass**. Run `pytest tests/unit/agent/test_executor.py -n auto -k "streaming"`.

### 3b: Streaming in Session Manager

- [ ] T018 [US5] Implement `process_message_streaming()` async generator in `src/holodeck/chat/session.py` (currently lines 82–131 have `process_message()`) — validate message (reuse existing validation logic), call `self._executor.execute_turn_streaming(message)`, yield chunks, increment `session.message_count` after stream completes. **(Ref: plan.md:583–586)**

- [ ] T019 [US5] Update `start()` in `src/holodeck/chat/session.py` (currently lines 48–80). Two changes: (1) **Update `AgentExecutor()` constructor call** (currently lines 62–67) to match refactored signature from T011 — remove `enable_observability`, `force_ingest`, `timeout` params; keep only `agent_config` and `on_execution_start`/`on_execution_complete` if used. (2) **Simplify error handling**: remove `try/except` wrapping the constructor since it no longer does I/O. Initialization errors now surface on first turn (lazy-init pattern). Keep `ChatSession()` creation (lines 70–74) as-is. **(Ref: plan.md:587–588, session.py:57–80)**

**Checkpoint**: T008, T009 tests should now **pass**. Run `pytest tests/unit/agent/test_session.py -n auto -k "streaming"`.

### 3c: Streaming in CLI

- [ ] T020 [US5] Update REPL loop in `src/holodeck/cli/commands/chat.py` (currently lines 320–392) — replace spinner-based blocking pattern with streaming text output:
  1. Write `"Agent: "` prefix, then `sys.stdout.write(chunk)` + `sys.stdout.flush()` for each chunk from `session_manager.process_message_streaming(message)`.
  2. Track elapsed time with `time.time()` around the stream.
  3. After stream completes, construct minimal `AgentResponse(content="".join(chunks), tool_executions=[], tokens_used=None, execution_time=elapsed)`. Token usage and tool details are NOT available from the streaming path (per plan.md:577–578) — add a comment noting this limitation.
  4. Call `progress.update(response)` with the minimal response for message count / inline status.
  5. In verbose mode, skip the status panel (no tool data); show response text only.
  6. Keep `ChatSpinnerThread` class definition for potential fallback use.
  Both backends work: Claude gets progressive text, SK gets single-chunk display (identical to current). **(Ref: plan.md:592–601, quickstart.md:431–434, data-model.md:440–457)**

**Checkpoint**: Full streaming pipeline functional. Both Claude (real streaming) and SK (single-chunk) backends work through the same code path.

---

## Phase 4: Test Refactoring (Existing Tests)

**Purpose**: Rewrite existing tests to use mock backends instead of `AgentFactory` mocks.

> **Ref**: plan.md lines 603–616 (10.5 — Update tests).

- [ ] T021 [P] [US1] Rewrite `TestAgentExecutorInitialization` in `tests/unit/agent/test_executor.py` (currently line 18) — remove all `@mock.patch("holodeck.chat.executor.AgentFactory")` decorators; inject mock `AgentBackend` via `backend=` constructor param. Remove `AgentExecutionResult` and `AgentFactory` imports from test file. **(Ref: plan.md:603–607)**

- [ ] T022 [P] [US1] Rewrite `TestAgentExecutorExecution` in `tests/unit/agent/test_executor.py` (currently line 67) — replace `AgentFactory` mock with mock `AgentBackend`/`AgentSession`; mock `session.send()` to return `ExecutionResult`. Verify `AgentResponse` dataclass is correctly populated from `ExecutionResult` fields. **(Ref: plan.md:603–607, contracts/execution-result.md:17–32)**

- [ ] T023 [P] [US1] Rewrite `TestAgentExecutorHistory` in `tests/unit/agent/test_executor.py` (currently line 175) — remove SK `ChatHistory` mock; verify history is tracked via local `list[dict]` with `{"role": "user"|"assistant", "content": str}` entries. **(Ref: plan.md:550–551, data-model.md:424–436)**

- [ ] T024 [P] [US1] Rewrite `TestAgentResponseStructure` in `tests/unit/agent/test_executor.py` (currently line 236) — verify the full `ExecutionResult` → `AgentResponse` conversion: (1) `result.response` → `response.content`; (2) `result.tool_calls` (`list[dict]`) → `response.tool_executions` (`list[ToolExecution]`) via `_convert_tool_calls()` — test with a tool call dict containing `{"name": "search", "arguments": {"q": "test"}}` and verify `ToolExecution.tool_name == "search"`; (3) `result.token_usage` (`TokenUsage`, defaults to `.zero()`, never `None`) → `response.tokens_used` (`TokenUsage | None`); (4) `response.execution_time` is a positive float. **(Ref: contracts/execution-result.md:17–32, executor.py:193–216)**

- [ ] T025 [P] [US1] Update `tests/unit/agent/test_session.py` — replace all `@mock.patch("holodeck.chat.session.AgentExecutor")` (currently lines 50, 63, 75, 118) with updated mock paths if AgentExecutor import path changes; ensure existing `process_message()` tests still pass with refactored executor. **(Ref: plan.md:612–614)**

**Checkpoint**: All existing tests pass with new mock patterns. Run `pytest tests/unit/agent/ -n auto` — zero failures.

---

## Phase 5: Verification & Spec Updates

**Purpose**: Final verification and documentation.

> **Ref**: plan.md lines 615–618 (10.6 — Update spec documents, verify zero SK imports in chat/).

- [ ] T026 [P] Verify zero imports of `AgentFactory`, `AgentThreadRun`, `ChatHistory` in `src/holodeck/chat/` — run `grep -r "AgentFactory\|AgentThreadRun\|ChatHistory" src/holodeck/chat/` and confirm no results. **(Ref: plan.md:617)**

- [ ] T027 [P] Run full test suite `pytest tests/ -n auto` and confirm zero regressions across all modules (unit + integration). **(Ref: plan.md:617)**

- [ ] T028 [P] Run code quality checks: `make format && make lint-fix && make type-check` — fix any issues introduced by the refactor. **(Ref: plan.md:648–656)**

- [ ] T029 Update spec documents — add Phase 10 completion notes to `specs/021-claude-agent-sdk/plan.md`, `data-model.md`, `quickstart.md`, and `research.md` as needed to reflect implemented state. **(Ref: plan.md:615)**

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup (TDD Scaffolding)     ← Write tests FIRST (T001–T009, all [P])
    │
Phase 2: Foundational (Executor)     ← BLOCKS all streaming work (T010–T016, sequential)
    │
    ├── Phase 3a: Streaming Executor  (T017, depends on Phase 2)
    │       │
    │       └── Phase 3b: Streaming Session Manager  (T018–T019, depends on T017)
    │               │
    │               └── Phase 3c: Streaming CLI  (T020, depends on T018)
    │
    └── Phase 4: Test Refactoring     ← Can start in parallel with Phase 3 (T021–T025, all [P])
            │
            └── Phase 5: Verification  (T026–T029, after Phases 3+4 complete)
```

### Critical Path

**T001–T009 → T010 → T011 → T012 → T013 → T014–T016 → T017 → T018 → T019 → T020 → T026–T029**

### Parallel Opportunities

| Parallel Group | Tasks | Condition |
|---|---|---|
| TDD scaffolding | T001–T009 | All independent — different test classes/methods |
| Executor internals | T014, T015, T016 | All independent — different methods in same file (after T013) |
| Streaming pipeline | T017 then T018 then T020 | Sequential — each depends on prior |
| Test rewrites | T021–T025 | All independent — different test classes |
| Verification | T026–T028 | All independent — different checks |

### Within Phase 2 (Executor Refactor)

```
T010 (remove/add imports)
  └── T011 (refactor constructor)     ← depends on T010
        └── T012 (implement _ensure_backend_and_session)  ← depends on T011
              └── T013 (refactor execute_turn)  ← depends on T012
                    ├── T014 (refactor get_history)      ← parallel after T013
                    ├── T015 (refactor clear_history)     ← parallel after T013
                    └── T016 (refactor shutdown)           ← parallel after T013
```

---

## Parallel Example: Phase 1 (TDD Scaffolding)

```bash
# Launch all 9 test stubs in parallel (all different test files/classes):
Task T001: "Write test_execute_turn_uses_session_send in tests/unit/agent/test_executor.py"
Task T002: "Write test_execute_turn_streaming_yields_chunks in tests/unit/agent/test_executor.py"
Task T003: "Write test_auto_select_backend in tests/unit/agent/test_executor.py"
Task T004: "Write test_get_history_tracks_messages in tests/unit/agent/test_executor.py"
Task T005: "Write test_clear_history_closes_session in tests/unit/agent/test_executor.py"
Task T006: "Write test_shutdown_closes_session_and_backend in tests/unit/agent/test_executor.py"
Task T007: "Write test_backend_error_wrapped in tests/unit/agent/test_executor.py"
Task T008: "Write test_process_message_streaming_yields_chunks in tests/unit/agent/test_session.py"
Task T009: "Write test_process_message_streaming_validates_message in tests/unit/agent/test_session.py"
```

---

## Parallel Example: Phase 4 (Test Refactoring)

```bash
# Launch all 5 test rewrites in parallel (all different test classes):
Task T021: "Rewrite TestAgentExecutorInitialization"
Task T022: "Rewrite TestAgentExecutorExecution"
Task T023: "Rewrite TestAgentExecutorHistory"
Task T024: "Rewrite TestAgentResponseStructure"
Task T025: "Update tests/unit/agent/test_session.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 — Non-Streaming Decoupling)

1. Complete Phase 1: Write all failing tests (T001–T009)
2. Complete Phase 2: Executor refactor (T010–T016) — **CRITICAL, blocks everything**
3. **STOP and VALIDATE**: Run `pytest tests/unit/agent/ -n auto -k "not streaming"` — all non-streaming tests pass
4. This alone delivers a fully decoupled chat executor that works with both backends

### Incremental Delivery

1. Phase 1 → All tests written and failing → TDD foundation set
2. Phase 2 → Non-streaming decoupling complete → Chat works with both backends via `session.send()`
3. Phase 3 → Streaming added → Progressive text display in `holodeck chat` for Claude agents
4. Phase 4 → Old tests rewritten → Clean test suite using mock backends
5. Phase 5 → Verification → Zero SK imports in chat layer, all checks pass

### Key Invariant

After Phase 10, **no module in `src/holodeck/chat/` imports `AgentFactory`, `AgentThreadRun`, or any Semantic Kernel type**. The chat layer is completely backend-agnostic.

---

## Notes

- [P] tasks = different files or independent methods, no dependencies
- [US1/US5] labels map tasks to user stories from spec.md
- All Phase 2 tasks (T010–T016) modify `src/holodeck/chat/executor.py` — execute sequentially
- Phase 3 tasks span 3 files — execute in sub-phase order (3a → 3b → 3c)
- Phase 4 tasks are all [P] — rewrite tests independently
- Commit after each phase checkpoint
- `_ensure_backend_and_session()` is the lynchpin — get this right first (T012)
