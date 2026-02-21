# Tasks: Native Claude Agent SDK Integration — Phase 4: SK Backend Refactor

**Feature**: `021-claude-agent-sdk`
**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Scope**: Phase 4 only (SK Backend Refactor). Continues from `tasks-foundations.md` (T023b).
**Approach**: TDD — test tasks are written and **must fail** before implementation tasks begin
**References**: Each task includes a document reference `(doc:L<line>)` pointing to the relevant design decision
**FRs**: FR-010, FR-011, FR-012b (plan.md:L258-259)

---

## Format: `[ID] [P?] Description (doc:Lline)`

- **[P]**: Task is parallelizable (different files, no unresolved dependencies)
- No `[US?]` labels: Phase 4 is a foundational refactor prerequisite for all user stories
- **TDD rule**: Test tasks MUST precede their implementation tasks

---

## Phase 4 Overview

**Purpose**: Extract SK-specific code from `agent_factory.py` into the `lib/backends/` package. The existing code is not deleted — it is moved and wrapped to implement the `AgentBackend` Protocol. After Phase 4, no *downstream consumer* (test runner executor, chat executor, chat session, models) imports `ChatHistory` or other SK types. Two modules retain internal SK usage until Phase 9: `agent_factory.py` (legacy execution path) and `llm_context_generator.py` (LLM prompt construction).

**Files created**:
- `src/holodeck/lib/backends/sk_backend.py` (new — core SK backend implementation)
- `src/holodeck/lib/backends/selector.py` (new — backend routing by provider)

**Files modified (source)**:
- `src/holodeck/lib/test_runner/agent_factory.py` (thin facade preserving public API)
- `src/holodeck/lib/chat_history_utils.py` (remove SK-specific function)
- `src/holodeck/models/chat.py` (remove `ChatHistory` type dependency)
- `src/holodeck/chat/executor.py` (remove `ChatHistory` import, use `result.response`)
- `src/holodeck/chat/session.py` (remove `ChatHistory` import, use `list[dict]`)
- `src/holodeck/lib/test_runner/executor.py` (remove `extract_last_assistant_content`, use `result.response`)

**Files modified (tests)**:
- `tests/unit/lib/test_chat_history_utils.py` (remove tests for moved function)
- `tests/unit/models/test_chat_models.py` (update `ChatSession` constructions to `list[dict]`)
- `tests/unit/agent/test_executor.py` (add `response` to mocked `AgentExecutionResult`)
- `tests/unit/lib/test_runner/test_executor.py` (remove stale `extract_last_assistant_content` import)

**Key invariant**: Existing unit tests MUST pass after documented test updates (T032b, T033b, T035b, T037b). The refactor is behaviour-preserving — production code paths produce identical results through new interfaces.

**Key goal**: After Phase 4, **no downstream consumer** (test runner executor, chat executor, chat session, models) **imports `ChatHistory` from `semantic_kernel`** (plan.md:L277). `agent_factory.py` and `llm_context_generator.py` retain internal SK usage until Phase 9.

---

## Phase 4A: Tests (write these first — they MUST fail before implementation)

> **NOTE**: Write ALL test tasks before implementing. Run `pytest tests/unit/lib/backends/test_sk_backend.py tests/unit/lib/backends/test_selector.py -n auto` to confirm all fail with `ImportError` or `ModuleNotFoundError` before starting Phase 4B.

### Tests for `sk_backend.py`

- [ ] T024 [P] Write unit tests for `_extract_response()` private function in `tests/unit/lib/backends/test_sk_backend.py`: (1) extracts last assistant message content from a `ChatHistory` with multiple messages; (2) returns empty string for empty `ChatHistory`; (3) returns empty string for `ChatHistory` with no assistant messages; (4) returns empty string for `ChatHistory` where last assistant message has `None` content. Mock `ChatHistory` using `semantic_kernel.contents.ChatHistory` directly in the test (this is the one place outside `sk_backend.py` where ChatHistory is used — in tests that verify the SK-specific implementation). (plan.md:L276, research.md:L295-306, quickstart.md:L362-366)

- [ ] T025 [P] Write unit tests for `SKBackend` implementing `AgentBackend` Protocol in `tests/unit/lib/backends/test_sk_backend.py`: (1) `SKBackend` passes `isinstance(backend, AgentBackend)` runtime check; (2) `initialize()` calls `_ensure_tools_initialized()` on the underlying factory; (3) `invoke_once(message)` returns `ExecutionResult` with non-empty `response` field (mock the underlying SK agent invocation); (4) `invoke_once()` correctly maps `tool_calls` from `AgentExecutionResult` to `ExecutionResult.tool_calls`; (5) `invoke_once()` correctly maps `tool_results` from `AgentExecutionResult` to `ExecutionResult.tool_results`; (6) `invoke_once()` correctly maps `TokenUsage` from `AgentExecutionResult` to `ExecutionResult.token_usage`; (7) `invoke_once()` populates `response` via `_extract_response()` — never returns empty string when ChatHistory has an assistant message; (8) `teardown()` calls `shutdown()` on the underlying factory. Use mocks/patches for the SK Kernel, Agent, and ChatHistory — do NOT make real LLM calls. (plan.md:L269-277, data-model.md:L297-338, contracts/execution-result.md:L100-110)

- [ ] T026 [P] Write unit tests for `SKSession` implementing `AgentSession` Protocol in `tests/unit/lib/backends/test_sk_backend.py`: (1) `SKSession` passes `isinstance(session, AgentSession)` runtime check; (2) `send(message)` returns `ExecutionResult` with populated `response`; (3) `send(message)` preserves conversation state across multiple calls (multi-turn); (4) `send_streaming(message)` is an async generator that yields the complete response as a single chunk (no-op streaming — plan.md:L274); (5) `close()` succeeds without error. Mock the underlying `AgentThreadRun` — do NOT make real LLM calls. (plan.md:L271-275, data-model.md:L260-293, contracts/execution-result.md:L86-96)

### Tests for `selector.py`

- [ ] T027 [P] Write unit tests for `BackendSelector.create()` routing logic in `tests/unit/lib/backends/test_selector.py`: (1) `provider=openai` returns `SKBackend` instance; (2) `provider=azure_openai` returns `SKBackend` instance; (3) `provider=ollama` returns `SKBackend` instance; (4) `provider=anthropic` raises `BackendInitError` with message indicating Claude backend is not yet implemented (placeholder until Phase 8); (5) returned backend has `initialize()` called automatically by `create()`. Mock the SKBackend constructor to avoid real kernel creation. Create `tests/unit/lib/backends/test_selector.py` (new file). (plan.md:L448-464, research.md:L295-306)

### Tests for `models/chat.py` update

- [ ] T028 [P] Write unit tests for updated `ChatSession.history` type in `tests/unit/models/test_chat_models.py` (existing file — add new test class `TestChatSessionHistoryType`): (1) `ChatSession` construction with default `history=[]` succeeds; (2) `ChatSession` construction with pre-populated `history=[{"role": "user", "content": "hi"}]` succeeds; (3) other `ChatSession` fields (`session_id`, `agent_config`, `started_at`, `message_count`, `state`, `metadata`) are unchanged; (4) `ChatSession` no longer requires `arbitrary_types_allowed` in `model_config` (since `list[dict]` is a native Pydantic type). Note: these tests will fail initially because `ChatSession.history` is still typed as `ChatHistory`. (plan.md:L285-288, data-model.md:L179-198)

### Tests for caller updates

- [ ] T029 [P] Write unit tests for `AgentExecutionResult.response` field in `tests/unit/lib/test_runner/test_agent_factory.py` (existing file — add new test class or functions): (1) `AgentExecutionResult` accepts a `response: str` field with default `""`; (2) when `AgentThreadRun.invoke()` completes, `result.response` is populated with the last assistant message content (not empty); (3) backward compat: `result.chat_history` is still accessible (not removed in Phase 4 — removed in Phase 9). This tests the transition strategy: callers can use `result.response` immediately and Phase 9 removes `chat_history`. (plan.md:L276-277, contracts/execution-result.md:L150-156)

**Checkpoint**: All test tasks written. Run `pytest tests/unit/lib/backends/test_sk_backend.py tests/unit/lib/backends/test_selector.py -n auto` — all MUST fail.

---

## Phase 4B: Implementation

### Core: `sk_backend.py`

- [ ] T030 Create `src/holodeck/lib/backends/sk_backend.py` implementing: (plan.md:L269-277, data-model.md:L297-338, contracts/execution-result.md:L86-128, quickstart.md:L362-378)

  **Private function: `_extract_response(history: ChatHistory) -> str`**
  - Moved from `chat_history_utils.py:extract_last_assistant_content()`
  - Searches reversed `history.messages` for last assistant message
  - Returns `str(content)` or empty string if not found
  - All `ChatHistory` usage is confined to this module

  **Class: `SKSession(AgentSession)`**
  - `__init__(self, thread_run: AgentThreadRun)` — wraps existing AgentThreadRun
  - `send(self, message: str) -> ExecutionResult` — calls `thread_run.invoke(message)`, converts `AgentExecutionResult` to `ExecutionResult` using `_extract_response()` for the `response` field
  - `send_streaming(self, message: str) -> AsyncGenerator[str, None]` — calls `send()` and yields the complete response as a single chunk (no-op streaming per plan.md:L274)
  - `close(self) -> None` — no-op (SK thread runs have no explicit cleanup)
  - Keep all SK imports (`ChatHistory`, `ChatHistoryAgentThread`, etc.) internal to this module

  **Class: `SKBackend(AgentBackend)`**
  - `__init__(self, agent_config: Agent, execution_config: ExecutionConfig | None, force_ingest: bool, mode: Literal["test", "chat"], max_retries: int = DEFAULT_MAX_RETRIES, retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS)` — stores config only; creates `AgentFactory` instance synchronously (same as current `AgentFactory.__init__()`), passing `max_retries` and `retry_delay_seconds` through
  - `initialize(self) -> None` — calls `self._factory._ensure_tools_initialized()`
  - `invoke_once(self, message: str, context: list[dict] | None) -> ExecutionResult` — creates isolated thread run, invokes, converts result. Response is populated via `_extract_response(result.chat_history)`. Token usage is mapped from `result.token_usage`. Tool calls and tool results are passed through.
  - `create_session(self) -> SKSession` — creates thread run via factory, wraps in `SKSession`
  - `teardown(self) -> None` — calls `self._factory.shutdown()`

### Core: `selector.py`

- [ ] T031 Create `src/holodeck/lib/backends/selector.py` implementing: (plan.md:L448-464, research.md:L295-306)

  **Class: `BackendSelector`**
  - `@staticmethod async def create(agent: Agent, tool_instances: list | None = None, mode: Literal["test", "chat"] = "test", execution_config: ExecutionConfig | None = None, force_ingest: bool = False) -> AgentBackend`
  - If `agent.model.provider == ProviderEnum.ANTHROPIC` → raise `BackendInitError("Claude-native backend not yet implemented. Use a non-Anthropic provider or wait for Phase 8.")`
  - Otherwise → create `SKBackend(agent, execution_config, force_ingest, mode)`, call `await backend.initialize()`, return backend
  - Import `SKBackend` from `holodeck.lib.backends.sk_backend`
  - Import `BackendInitError` from `holodeck.lib.backends.base`

### Refactor: `chat_history_utils.py`

- [ ] T032 Update `src/holodeck/lib/chat_history_utils.py`: (plan.md:L279-283)
  - **Remove** `extract_last_assistant_content()` function entirely — it has been moved to `sk_backend.py` as `_extract_response()`
  - **Remove** `from semantic_kernel.contents import ChatHistory` import — no SK types remain in this module
  - **Keep** `extract_tool_names(tool_calls: list[dict[str, Any]]) -> list[str]` — this function is SK-free and used by the test runner
  - Update module docstring to reflect the remaining function only

- [ ] T032b Update `tests/unit/lib/test_chat_history_utils.py` to reflect the removed function: (plan.md:L279-283)
  - **Remove** the entire `TestExtractLastAssistantContent` class (7 tests) — the function it tests has been moved to `sk_backend.py` and is now covered by T024's tests for `_extract_response()`
  - **Remove** `from holodeck.lib.chat_history_utils import extract_last_assistant_content` import
  - **Remove** `from semantic_kernel.contents import ChatHistory` import (if present)
  - **Keep** `TestExtractToolNames` class — `extract_tool_names()` remains in `chat_history_utils.py`
  - Update module docstring to reflect the remaining test class only

### Update: `models/chat.py`

- [ ] T033 Update `src/holodeck/models/chat.py`: (plan.md:L285-288, data-model.md:L179-198)
  - **Change** `ChatSession.history: ChatHistory` → `history: list[dict[str, Any]] = Field(default_factory=list)`
  - **Remove** `from semantic_kernel.contents import ChatHistory` import
  - **Remove** `arbitrary_types_allowed=True` from `ChatSession.model_config` (no longer needed — `list[dict]` is a native Pydantic type; change to `model_config = ConfigDict()` or remove the line if no other config is needed)
  - Add docstring to `history` field: `"""Conversation history as provider-agnostic dicts. Format: {"role": "user"|"assistant", "content": str}. SK backend serializes ChatHistory internally. Claude backend leaves empty (state in subprocess)."""`
  - Verify all other `ChatSession` fields remain unchanged

- [ ] T033b Update `tests/unit/models/test_chat_models.py` to match the updated `ChatSession.history` type: (plan.md:L285-288)
  - **Remove** `from semantic_kernel.contents import ChatHistory` import
  - **Replace** all `ChatSession(history=ChatHistory(), ...)` constructions with `ChatSession(history=[], ...)` (2 occurrences: `test_session_started_at_in_future` at L60-63 and `test_session_defaults` at L68)
  - **Verify** all existing `ChatSession` assertions still pass with the `list[dict]` type

### Update: `agent_factory.py` (Thin Facade)

- [ ] T034 Update `src/holodeck/lib/test_runner/agent_factory.py`: (plan.md:L290-294, contracts/execution-result.md:L150-156)
  - **Add** `response: str = ""` field to `AgentExecutionResult` dataclass **after** `token_usage` (last field, with default — all existing positional constructions continue to work). This provides a transition path so callers can use `result.response` instead of `extract_last_assistant_content(result.chat_history)`
  - **Update** `AgentThreadRun._invoke_agent_impl()` to populate `response` field: after extracting tool calls and before returning, call `_extract_response_from_history(self.chat_history)` (a new private method on `AgentThreadRun` that calls the same logic as `_extract_response` from `sk_backend.py`) and set `response=extracted_content` in the returned `AgentExecutionResult`
  - **Add** private method `AgentThreadRun._extract_response_from_history(self, history: ChatHistory) -> str` that extracts the last assistant message — identical logic to `sk_backend._extract_response()`. This avoids importing from sk_backend (which would create a bidirectional dependency). The duplication is intentional and temporary — Phase 9 removes `AgentExecutionResult` entirely in favour of `ExecutionResult`.
  - **Remove** the import of `extract_last_assistant_content` from `chat_history_utils` if present (it's not directly imported in agent_factory.py, but verify)
  - **Preserve** all existing public API: `AgentFactory`, `AgentThreadRun`, `AgentExecutionResult` remain importable
  - **Preserve** `AgentFactory.__init__()` synchronous kernel creation (no behaviour change)
  - **Note**: `AgentFactory` does NOT delegate to `BackendSelector` in Phase 4. The delegation is deferred to Phase 9 when `TestExecutor` switches to using `BackendSelector.create()` directly. In Phase 4, `AgentFactory` remains the canonical entry point with its existing implementation, and `SKBackend` is a parallel implementation of the same logic behind the `AgentBackend` Protocol.

### Update: `chat/executor.py` (Remove ChatHistory)

- [ ] T035 Update `src/holodeck/chat/executor.py` to remove all `ChatHistory` imports and usage: (plan.md:L277)
  - **Remove** `from semantic_kernel.contents import ChatHistory` (L10)
  - **Remove** `from holodeck.lib.chat_history_utils import extract_last_assistant_content` (L12)
  - **Update** `execute_turn()` (L124): replace `content = extract_last_assistant_content(result.chat_history)` with `content = result.response` — uses the new `AgentExecutionResult.response` field populated by T034
  - **Update** `get_history() -> ChatHistory` (L154): change return type to `list[dict[str, Any]]`. Internally, if `self._thread_run` exists, serialize its `chat_history` to `list[dict]` via a helper: `[{"role": m.role.value, "content": str(m.content)} for m in self._thread_run.chat_history.messages]` (use `.value` — SK's `AuthorRole` is a string enum; `str()` produces `"AuthorRole.ASSISTANT"` not `"assistant"`). If no thread run, return `[]`. This preserves the `get_history()` public API while removing the SK type from the return signature.
  - **Update** `clear_history()` (L164): no changes needed (already just sets `_thread_run = None`)
  - **Verify** no other ChatHistory references remain in the file

- [ ] T035b Update `tests/unit/agent/test_executor.py` to match the updated `AgentExecutionResult` and `chat/executor.py` changes: (plan.md:L277)
  - **Remove** `from semantic_kernel.contents import ChatHistory` import (L9)
  - **Update** all mocked `AgentExecutionResult` constructions (4+ occurrences) to include `response="<expected content>"` — e.g., `AgentExecutionResult(tool_calls=[], tool_results=[], chat_history=mock_history, response="Hi there!")`. Without this, `result.response` will be `""` after T035 changes executor to read `result.response` instead of extracting from `chat_history`
  - **Update** `test_get_history_returns_chat_history` (L196): rename to `test_get_history_returns_list` and update assertions to expect `list[dict]` return type instead of `ChatHistory`
  - **Verify** all existing test assertions still pass

### Update: `chat/session.py` (Remove ChatHistory)

- [ ] T036 Update `src/holodeck/chat/session.py` to remove all `ChatHistory` imports and usage: (plan.md:L277, data-model.md:L179-198)
  - **Remove** `from semantic_kernel.contents import ChatHistory` (L8)
  - **Update** `start()` (L72): replace `history = ChatHistory()` with `history: list[dict[str, Any]] = []` — matches the updated `ChatSession.history` type from T033
  - **Update** `ChatSession` construction (L73-77): pass `history=history` (which is now `[]`)
  - **Verify** no other ChatHistory references remain in the file

### Update: `test_runner/executor.py` (Remove `extract_last_assistant_content`)

- [ ] T037 Update `src/holodeck/lib/test_runner/executor.py` to use `result.response` instead of `extract_last_assistant_content`: (plan.md:L277)
  - **Update** import (L32-35): remove `extract_last_assistant_content` from `from holodeck.lib.chat_history_utils import (...)`. Keep `extract_tool_names`.
  - **Update** `_execute_single_test()` (L551): replace `agent_response = extract_last_assistant_content(result.chat_history)` with `agent_response = result.response` — uses the new `AgentExecutionResult.response` field from T034
  - **Verify** no other `extract_last_assistant_content` or `ChatHistory` references remain in the file
  - **Note**: `extract_tool_names(result.tool_calls)` (L552) remains unchanged — `extract_tool_names` is SK-free and stays in `chat_history_utils.py`

- [ ] T037b Update `tests/unit/lib/test_runner/test_executor.py` to remove stale `extract_last_assistant_content` references: (plan.md:L277)
  - **Remove** the local import of `extract_last_assistant_content` from `chat_history_utils` at L2297 (inside `test_extract_response_text_empty_history`)
  - **Remove or rewrite** the `test_extract_response_text_empty_history` test — the function it tests no longer exists in `chat_history_utils.py` (moved to `sk_backend.py` as `_extract_response()`, covered by T024)
  - **Update** any mocked `AgentExecutionResult` constructions (L261-264) to include `response="<expected content>"` — without this, `result.response` will be `""` after T037 changes executor to read `result.response`
  - **Verify** no other `extract_last_assistant_content` or `ChatHistory` references remain in the test file

### Update: `backends/__init__.py` exports

- [ ] T038 Update `src/holodeck/lib/backends/__init__.py` to export key types for convenient imports: (plan.md:L185)
  - Export from `base`: `ExecutionResult`, `AgentSession`, `AgentBackend`, `BackendError`, `BackendInitError`, `BackendSessionError`, `BackendTimeoutError`
  - Export from `selector`: `BackendSelector`
  - Export from `sk_backend`: `SKBackend`, `SKSession`
  - Use `__all__` list for explicit public API

**Checkpoint**: All implementation and test-update tasks complete. Run `pytest tests/unit/ -n auto` — ALL tests MUST pass (existing tests pass after documented updates in T032b, T033b, T035b, T037b). The refactor is behaviour-preserving.

---

## Phase 4C: Verification & Quality

- [ ] T039 Run full existing unit test suite: `make test-unit` — zero regressions. If any test fails, the failure is in the refactor (not the test) and MUST be fixed before proceeding. Pay special attention to: (plan.md:L299)
  - `tests/unit/lib/test_runner/` — agent factory tests
  - `tests/unit/models/` — chat model tests
  - `tests/unit/chat/` — chat executor/session tests (if they exist)

- [ ] T040 Run code quality checks — all MUST pass: (plan.md:L553-560)
  ```bash
  make format         # Black + Ruff formatting
  make lint-fix       # Auto-fix linting issues
  make type-check     # MyPy strict mode — all new code must pass
  make security       # Bandit + Safety + detect-secrets
  ```

- [ ] T041 **Integration Gate** (plan.md:L303): Before starting Phase 5, run a full `holodeck test` workflow against a real non-Anthropic agent (e.g., an OpenAI agent with at least one vectorstore tool). All test cases must execute successfully. This prevents asyncio lifecycle regressions from the SK refactor from being buried under Claude backend code in later phases. Document the test command and result in a comment on the PR or in the commit message.

---

## Dependencies & Execution Order

### Within Phase 4: TDD Order

```
1. Write all test tasks (T024–T029) — confirm they FAIL
2. Implement core modules (T030–T031) — sk_backend.py, selector.py
3. Refactor existing modules (T032–T037) — chat_history_utils, models/chat, agent_factory, callers
4. Update existing tests (T032b, T033b, T035b, T037b) — align mocks and imports with refactored code
5. Update exports (T038)
6. Verify (T039–T041) — all tests pass, quality checks, integration gate
```

### Task Dependencies (strict order)

```
T024, T025, T026 ─────┐
T027 ─────────────────┤ (all test tasks can be written in parallel)
T028 ─────────────────┤
T029 ─────────────────┘
         │
         ▼ (confirm all tests fail)
T030 ◄─── sk_backend.py (depends on: T024–T026 tests exist and fail)
         │
T031 ◄─── selector.py (depends on: T027 test exists; T030 for SKBackend import)
         │
         ├── T032 chat_history_utils.py (depends on: T030 — function moved there)
         │    └── T032b test_chat_history_utils.py (depends on: T032 — function removed)
         ├── T033 models/chat.py (depends on: T028 test exists)
         │    └── T033b test_chat_models.py (depends on: T033 — history type changed)
         │
T034 ◄─── agent_factory.py (depends on: T029 test exists; T030 for understanding)
         │
         ├── T035 chat/executor.py (depends on: T032, T033, T034 — uses result.response)
         │    └── T035b test_executor.py [agent] (depends on: T035 — mocks need response field)
         ├── T036 chat/session.py (depends on: T033 — uses list[dict])
         └── T037 test_runner/executor.py (depends on: T032, T034 — uses result.response)
              └── T037b test_executor.py [test_runner] (depends on: T037 — stale import removed)
              │
T038 ◄─── __init__.py exports (depends on: T030, T031)
              │
T039 ◄─── test suite (depends on: all implementation + test-update tasks)
T040 ◄─── quality checks (depends on: T039 passing)
T041 ◄─── integration gate (depends on: T039, T040)
```

### Parallel Opportunities

**Test writing** (T024–T029): All six test tasks target different files or distinct test classes — they CAN all be written in parallel:
- Developer A: `test_sk_backend.py` (T024, T025, T026 — same file, sequential within)
- Developer B: `test_selector.py` (T027)
- Developer C: `test_chat_models.py` additions (T028) + `test_agent_factory.py` additions (T029)

**Implementation** (T030–T038):
- T030 and T031 are sequential (T031 imports from T030)
- T032, T033 can run in parallel with T034 (different files)
- T035, T036, T037 can run in parallel (different files, all depend on T032–T034)
- T038 can run in parallel with T035–T037

**Test updates** (T032b, T033b, T035b, T037b):
- T032b depends on T032 (same module's tests)
- T033b depends on T033 (same model's tests)
- T035b depends on T035 (same executor's tests)
- T037b depends on T037 (same executor's tests)
- T032b∥T033b∥T035b∥T037b can run in parallel (different test files)

### Parallel Example

```bash
# Phase 4A — Write all tests in parallel:
# Dev A: tests/unit/lib/backends/test_sk_backend.py (T024, T025, T026)
# Dev B: tests/unit/lib/backends/test_selector.py (T027)
# Dev C: tests/unit/models/test_chat_models.py additions (T028) + test_agent_factory.py (T029)

# Confirm new backend tests fail:
pytest tests/unit/lib/backends/test_sk_backend.py tests/unit/lib/backends/test_selector.py -n auto

# Phase 4B — Sequential core, then parallel callers, then parallel test updates:
# T030 → T031 (sequential — selector imports sk_backend)
# T032 + T033 + T034 (parallel — different files)
# T035 + T036 + T037 (parallel — different files, all depend on T032–T034)
# T032b + T033b + T035b + T037b (parallel — different test files, align with refactored code)
# T038 (after T030, T031)

# Phase 4C — Sequential verification:
# T039 → T040 → T041
```

---

## Task Summary

| Group | Tasks | Tests | Implementations | Parallel |
|-------|-------|-------|-----------------|---------|
| Phase 4A (Tests) | T024–T029 | 6 | 0 | T024–T029 (across files) |
| Phase 4B (Implementation) | T030–T038 | 0 | 9 | T032∥T033∥T034, T035∥T036∥T037 |
| Phase 4B (Test Updates) | T032b, T033b, T035b, T037b | 4 | 0 | T032b∥T033b∥T035b∥T037b |
| Phase 4C (Verification) | T039–T041 | 0 | 0 | Sequential |
| **Total** | **22 tasks** | **10** | **9** | |

---

## Continuity from Phase 3

Phase 3 ended with T023b. Phase 4 starts at T024.

**Prerequisites from earlier phases**:
- `src/holodeck/lib/backends/__init__.py` exists (T007, Phase 1)
- `tests/unit/lib/backends/__init__.py` exists (T007b, Phase 1)
- `src/holodeck/lib/backends/base.py` exists with `ExecutionResult`, `AgentSession`, `AgentBackend`, exception hierarchy (T008, Phase 1)
- `src/holodeck/models/claude_config.py` exists (T014, Phase 2)
- `src/holodeck/lib/backends/validators.py` exists (T023, Phase 3)
- `AgentFactoryError` consolidated to `lib/errors.py` only (T023b, Phase 3)

---

## Design Notes

### Why `AgentExecutionResult.response` (transition strategy)

The plan states that `SKBackend.invoke_once()` MUST populate `ExecutionResult.response` via `_extract_response()` (plan.md:L276). However, in Phase 4, callers (`test_runner/executor.py`, `chat/executor.py`) still use `AgentFactory` → `AgentThreadRun` → `AgentExecutionResult`. To achieve the "no ChatHistory imports outside sk_backend.py" goal without prematurely switching callers to the new backend interface (that's Phase 9/10), we add a `response` field to `AgentExecutionResult` and populate it in `AgentThreadRun._invoke_agent_impl()`.

This duplication of response extraction logic is **intentional and temporary**:
- `sk_backend.py:_extract_response()` — used by `SKBackend.invoke_once()` and `SKSession.send()`
- `agent_factory.py:AgentThreadRun._extract_response_from_history()` — used by the legacy path

Phase 9 removes `AgentExecutionResult` entirely and switches all callers to `ExecutionResult` via `BackendSelector`. The duplication is eliminated at that point.

### Why `AgentFactory` does NOT delegate to `BackendSelector` yet

The plan (plan.md:L290-294) states `AgentFactory` "internally delegates to `BackendSelector.create()`." However, `BackendSelector` is fully wired with `ClaudeBackend` routing in Phase 8 (plan.md:L448-464). In Phase 4, `BackendSelector` only supports `SKBackend` and raises `BackendInitError` for Anthropic providers.

Wiring `AgentFactory` through `BackendSelector` in Phase 4 would be premature — the `TestExecutor` and `ChatSessionManager` still expect `AgentFactory`'s synchronous init pattern and `AgentThreadRun`'s `invoke()` API. The delegation happens in Phase 9 when callers are switched to `BackendSelector.create()` directly.

In Phase 4, `AgentFactory` retains its existing implementation. `SKBackend` exists as a **parallel implementation** of the same logic behind the `AgentBackend` Protocol — ready for Phase 9 to wire in.

### Why `ChatHistory` is still present in `agent_factory.py` and `llm_context_generator.py`

After Phase 4, two modules outside `sk_backend.py` still import `ChatHistory` from `semantic_kernel`:

1. **`agent_factory.py`**: `AgentFactory` and `AgentThreadRun` use `ChatHistory` internally for the legacy execution path. Phase 9 removes this entirely (it becomes a thin re-export facade).
2. **`llm_context_generator.py`** (L200-203): Uses `ChatHistory` for LLM prompt construction — unrelated to agent execution. Deferred to Phase 9 for cleanup.

These are accepted because the important goal — that **no downstream consumer** (test runner executor, chat executor, chat session, models) imports ChatHistory — IS achieved in Phase 4.

### `chat/executor.py:get_history()` return type change

Phase 10 (plan.md:L510-513) replaces `get_history()` with two methods: `get_message_count()` and `get_history() -> list[dict]`. Phase 4 changes the return type to `list[dict[str, Any]]` as a prerequisite (removing the ChatHistory return type). The internal serialization (`ChatHistory.messages → list[dict]`) bridges the gap until Phase 10 implements the full replacement.

---

## Code Quality Gates

Run after Phase 4 implementation:

```bash
make format         # Black + Ruff formatting
make lint-fix       # Auto-fix linting issues
make type-check     # MyPy strict mode (all new code must pass)
make test-unit      # pytest tests/unit/ -n auto
make security       # Bandit + Safety + detect-secrets
```

---

## What Follows Phase 4

Phase 4 is the **Integration Gate** before Phases 5–7 can proceed in parallel:

```
Phase 4 (SK backend refactor — this file) ← MUST be stable + tested first
    │
    ├── Phase 5 (tool adapters)       ─┐
    ├── Phase 6 (MCP bridge)           ├── parallel after Phase 4 confirmed green
    └── Phase 7 (OTel bridge)         ─┘
```

Phases 5, 6, and 7 can progress in parallel ONLY after T041 (Integration Gate) passes.
