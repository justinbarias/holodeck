# Tasks: US4 — Backend-Specific Configuration

**Input**: Design documents from `/specs/023-choose-your-backend/`
**Prerequisites**: plan.md (required), spec.md (required), data-model.md (required), US1 (ADK config model + backend), US2 (AF config model + backend)
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.

**User Story**: A platform user wants to tune backend-specific features. For Google ADK: streaming mode, code execution. For Agent Framework: message compaction strategy, client type. These settings are in optional YAML sections (`google_adk:` or `agent_framework:`) that are ignored when using other backends.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US4]**: All tasks belong to User Story 4
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: N/A — config models (`GoogleADKConfig`, `AgentFrameworkConfig`) and backend files (`adk_backend.py`, `af_backend.py`) are built in US1 and US2. No new files or dependencies needed for US4.

---

## Phase 2: Foundational — Config Wiring Infrastructure

**Purpose**: Ensure the backend `initialize()` methods read their respective config sections from the `Agent` model and store them for runtime use. This is the prerequisite for all US4 feature wiring.

### Tests (TDD — write FIRST, verify they FAIL)

> **NOTE**: Write these tests FIRST, ensure they FAIL before implementation.
> All tests calling `initialize()` MUST mock external SDK imports and validators to isolate config wiring logic.

- [ ] T001 [P] [US4] Write unit test `test_adk_backend_reads_google_adk_config` in `tests/unit/lib/backends/test_adk_backend.py` — create an `Agent` with `backend: google_adk`, `model.provider: google`, and `google_adk: { streaming_mode: sse, code_execution: true, max_iterations: 5, output_key: result }`. Mock ADK SDK imports. Call `ADKBackend.initialize()`. Assert `self._config` is a `GoogleADKConfig` instance with `streaming_mode == StreamingMode.SSE`, `code_execution == True`, `max_iterations == 5`, `output_key == "result"`

- [ ] T002 [P] [US4] Write unit test `test_adk_backend_uses_defaults_when_no_config` in `tests/unit/lib/backends/test_adk_backend.py` — create an `Agent` with `backend: google_adk`, `model.provider: google`, and NO `google_adk` section. Call `ADKBackend.initialize()`. Assert `self._config` is `None` or defaults are applied (streaming_mode=none, code_execution=False, max_iterations=None, output_key="output")

- [ ] T003 [P] [US4] Write unit test `test_af_backend_reads_agent_framework_config` in `tests/unit/lib/backends/test_af_backend.py` — create an `Agent` with `backend: agent_framework`, `model.provider: openai`, and `agent_framework: { compaction_strategy: sliding_window, max_tool_rounds: 10 }`. Mock AF SDK imports. Call `AFBackend.initialize()`. Assert `self._config` is an `AgentFrameworkConfig` instance with `compaction_strategy == CompactionStrategy.SLIDING_WINDOW`, `max_tool_rounds == 10`

- [ ] T004 [P] [US4] Write unit test `test_af_backend_uses_defaults_when_no_config` in `tests/unit/lib/backends/test_af_backend.py` — create an `Agent` with `backend: agent_framework`, `model.provider: openai`, and NO `agent_framework` section. Call `AFBackend.initialize()`. Assert `self._config` is `None` or defaults are applied (compaction_strategy=none, max_tool_rounds=None)

### Implementation

- [ ] T005 [US4] Ensure `ADKBackend.__init__()` in `src/holodeck/lib/backends/adk_backend.py` stores `self._config = agent.google_adk` (the optional `GoogleADKConfig` from the Agent model). If US1 already does this, verify and mark as done

- [ ] T006 [US4] Ensure `AFBackend.__init__()` in `src/holodeck/lib/backends/af_backend.py` stores `self._config = agent.agent_framework` (the optional `AgentFrameworkConfig` from the Agent model). If US2 already does this, verify and mark as done. Run tests T001–T004 and verify they PASS

**Checkpoint**: Config wiring infrastructure confirmed — backends read their config sections correctly.

---

## Phase 3: US4 Implementation — Feature Wiring and Ignore Behavior

**Purpose**: Wire backend-specific config values to actual runtime behavior (streaming, code execution, compaction) and validate that mismatched config sections are silently ignored.

### 3A: ADK Streaming Mode Wiring

> Wire `GoogleADKConfig.streaming_mode` to ADK runner invocation behavior.

#### Tests (TDD)

- [ ] T008 [P] [US4] Write unit test `test_adk_streaming_mode_sse_delivers_progressively` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with `google_adk: { streaming_mode: sse }`. Mock ADK `Runner`. Call `ADKBackend.invoke_once()` or `ADKSession.send_streaming()`. Assert the runner is invoked with SSE streaming enabled (verify the ADK `run_async` or equivalent is called with streaming parameters). Verify `ExecutionResult` is returned successfully

- [ ] T009 [P] [US4] Write unit test `test_adk_streaming_mode_none_returns_complete_response` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with `google_adk: { streaming_mode: none }` (or no config). Call `ADKBackend.invoke_once()`. Assert the runner is invoked WITHOUT streaming. Verify `ExecutionResult.response` contains the full response

#### Implementation

- [ ] T010 [US4] Wire `streaming_mode` in `ADKBackend.invoke_once()` and `ADKSession.send()` / `ADKSession.send_streaming()` in `src/holodeck/lib/backends/adk_backend.py` — when `self._config` is not None and `self._config.streaming_mode == StreamingMode.SSE`, use ADK's streaming invocation path (e.g., `runner.run_async()` with streaming events). When `none`, use standard synchronous invocation. Default to `none` when `self._config` is `None`. Run tests T008–T009 and verify they PASS

### 3B: ADK Code Execution Wiring

> Wire `GoogleADKConfig.code_execution` to ADK agent construction.

#### Tests (TDD)

- [ ] T012 [P] [US4] Write unit test `test_adk_code_execution_enabled` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with `google_adk: { code_execution: true }`. Call `ADKBackend.initialize()`. Assert the ADK `LlmAgent` is constructed with code execution capability enabled (verify the constructor args or tools list includes code execution)

- [ ] T013 [P] [US4] Write unit test `test_adk_code_execution_disabled_by_default` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with no `google_adk` section or `code_execution: false`. Call `ADKBackend.initialize()`. Assert the ADK `LlmAgent` does NOT have code execution capability

#### Implementation

- [ ] T014 [US4] Wire `code_execution` in `ADKBackend.initialize()` in `src/holodeck/lib/backends/adk_backend.py` — when `self._config` is not None and `self._config.code_execution == True`, add ADK's code execution tool/capability to the agent construction. When False or config is None, omit it. Run tests T012–T013 and verify they PASS

### 3C: ADK Max Iterations and Output Key Wiring

> Wire `GoogleADKConfig.max_iterations` and `output_key` to ADK runner/session behavior.

#### Tests (TDD)

- [ ] T016 [P] [US4] Write unit test `test_adk_max_iterations_limits_agent_loop` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with `google_adk: { max_iterations: 3 }`. Mock ADK runner. Call `ADKBackend.invoke_once()`. Assert the runner is configured with max iterations of 3 (verify constructor arg or run parameter)

- [ ] T017 [P] [US4] Write unit test `test_adk_output_key_extracts_correct_value` in `tests/unit/lib/backends/test_adk_backend.py` — create `Agent` with `google_adk: { output_key: "final_answer" }`. Mock ADK session state to contain `{ "final_answer": "42" }`. Call `ADKBackend.invoke_once()`. Assert `ExecutionResult.response` is `"42"` (extracted from session state using the configured output key)

#### Implementation

- [ ] T018 [US4] Wire `max_iterations` in `ADKBackend.initialize()` or `invoke_once()` in `src/holodeck/lib/backends/adk_backend.py` — pass `max_iterations` to the ADK `Runner` or `LlmAgent` construction when configured. When `None`, use ADK's default behavior

- [ ] T019 [US4] Wire `output_key` in `ADKBackend.invoke_once()` and `ADKSession.send()` in `src/holodeck/lib/backends/adk_backend.py` — extract the final response from ADK session state using `self._config.output_key` (default: `"output"`). Fall back to `"output"` when `self._config` is `None`. Run tests T016–T017 and verify they PASS

### 3D: AF Compaction Strategy Wiring

> Wire `AgentFrameworkConfig.compaction_strategy` to AF session/agent behavior.

#### Tests (TDD)

- [ ] T021 [P] [US4] Write unit test `test_af_compaction_sliding_window` in `tests/unit/lib/backends/test_af_backend.py` — create `Agent` with `agent_framework: { compaction_strategy: sliding_window }`. Mock AF SDK. Call `AFBackend.create_session()` and then `AFSession.send()` multiple times. Assert the AF agent or session is configured with sliding window compaction (verify constructor args or session setup includes the compaction strategy)

- [ ] T022 [P] [US4] Write unit test `test_af_compaction_summarization` in `tests/unit/lib/backends/test_af_backend.py` — create `Agent` with `agent_framework: { compaction_strategy: summarization }`. Mock AF SDK. Call `AFBackend.create_session()`. Assert the AF session is configured with summarization compaction strategy

- [ ] T023 [P] [US4] Write unit test `test_af_compaction_none_by_default` in `tests/unit/lib/backends/test_af_backend.py` — create `Agent` with no `agent_framework` section. Call `AFBackend.create_session()`. Assert no compaction is applied (default behavior, no compaction strategy set on the AF session)

#### Implementation

- [ ] T024 [US4] Wire `compaction_strategy` in `AFBackend.create_session()` or `AFSession.__init__()` in `src/holodeck/lib/backends/af_backend.py` — when `self._config` is not None and `self._config.compaction_strategy != CompactionStrategy.NONE`, configure the AF session with the appropriate compaction strategy (sliding_window or summarization). When `none` or config is None, leave session with default (no compaction). Run tests T021–T023 and verify they PASS

### 3E: AF Max Tool Rounds Wiring

> Wire `AgentFrameworkConfig.max_tool_rounds` to AF agent invocation behavior.

#### Tests (TDD)

- [ ] T026 [P] [US4] Write unit test `test_af_max_tool_rounds_limits_invocations` in `tests/unit/lib/backends/test_af_backend.py` — create `Agent` with `agent_framework: { max_tool_rounds: 5 }`. Mock AF SDK. Call `AFBackend.invoke_once()`. Assert the AF agent's invocation is configured with max_tool_rounds=5 (verify constructor arg or invocation parameter)

- [ ] T027 [P] [US4] Write unit test `test_af_max_tool_rounds_unlimited_by_default` in `tests/unit/lib/backends/test_af_backend.py` — create `Agent` with no `agent_framework` section. Call `AFBackend.invoke_once()`. Assert no max_tool_rounds limit is applied (default behavior)

#### Implementation

- [ ] T028 [US4] Wire `max_tool_rounds` in `AFBackend.invoke_once()` and `AFSession.send()` in `src/holodeck/lib/backends/af_backend.py` — when `self._config` is not None and `self._config.max_tool_rounds` is set, pass it to the AF agent invocation. When `None` or config is None, use AF's default (unlimited). Run tests T026–T027 and verify they PASS

### 3F: Ignore-When-Mismatched Behavior

> Validate that backend-specific config sections are silently ignored when a different backend is active.

#### Tests (TDD)

- [ ] T030 [P] [US4] Write unit test `test_google_adk_section_ignored_when_backend_is_openai` in `tests/unit/models/test_agent_config_ignore.py` (new file) — create `Agent` with `model.provider: openai`, NO explicit `backend` field, and `google_adk: { streaming_mode: sse, code_execution: true }`. Assert the Agent model is created successfully without `ValidationError`. Assert `agent.google_adk` is a valid `GoogleADKConfig` (Pydantic parses it), but it is not used at runtime. Verify `BackendSelector.select()` routes to `AFBackend` (not ADKBackend)

- [ ] T031 [P] [US4] Write unit test `test_agent_framework_section_ignored_when_backend_is_claude` in `tests/unit/models/test_agent_config_ignore.py` — create `Agent` with `model.provider: anthropic`, NO explicit `backend` field, and `agent_framework: { compaction_strategy: sliding_window }`. Assert model creation succeeds. Assert `BackendSelector.select()` routes to `ClaudeBackend`. Assert the `agent_framework` config section is parsed but not consumed

- [ ] T032 [P] [US4] Write unit test `test_google_adk_section_ignored_when_backend_is_agent_framework` in `tests/unit/models/test_agent_config_ignore.py` — create `Agent` with `backend: agent_framework`, `model.provider: openai`, and `google_adk: { streaming_mode: sse }`. Assert model creation succeeds. Assert `BackendSelector.select()` routes to `AFBackend`

- [ ] T033 [P] [US4] Write unit test `test_agent_framework_section_ignored_when_backend_is_google_adk` in `tests/unit/models/test_agent_config_ignore.py` — create `Agent` with `backend: google_adk`, `model.provider: google`, and `agent_framework: { compaction_strategy: summarization }`. Assert model creation succeeds. Assert `BackendSelector.select()` routes to `ADKBackend`

- [ ] T034 [P] [US4] Write unit test `test_both_config_sections_present_only_active_one_used` in `tests/unit/models/test_agent_config_ignore.py` — create `Agent` with `backend: google_adk`, `model.provider: google`, `google_adk: { streaming_mode: sse }`, AND `agent_framework: { compaction_strategy: sliding_window }`. Assert model creation succeeds. Assert `BackendSelector.select()` routes to `ADKBackend`. Assert ADKBackend reads only `google_adk` config, not `agent_framework`

#### Implementation

- [ ] T035 [US4] Verify that `BackendSelector.select()` in `src/holodeck/lib/backends/selector.py` does NOT validate or reject non-matching config sections. The selector should only read the `backend` field (or auto-detect from `model.provider`) and instantiate the matching backend. Non-matching config sections remain on the Agent model but are never accessed by the selected backend. If the selector currently rejects them, fix it to ignore them. Run tests T030–T034 and verify they PASS

**Note**: Strict model validation (`extra='forbid'`, invalid enum values) is tested in US1 (`test_google_adk_config.py`) and US2 (`test_af_config.py`). Not duplicated here.

**Checkpoint**: All US4 features wired — streaming, code execution, compaction, max tool rounds, and ignore behavior all verified.

---

## Phase 4: Polish

**Purpose**: Code quality and final validation.

- [ ] T036 Run `make format` to format all modified/new files with Black + Ruff
- [ ] T037 Run `make lint` and fix any Ruff + Bandit violations in files touched by US4
- [ ] T038 Run `make type-check` and fix any MyPy errors in `adk_backend.py`, `af_backend.py`, `google_adk_config.py`, `af_config.py`, and new test files
- [ ] T039 Run full test suite `make test` to verify no regressions
- [ ] T040 Verify acceptance scenarios from user story:
  - Scenario 1: agent.yaml with `backend: google_adk` and `google_adk: { streaming_mode: sse }` results in streaming responses delivered progressively
  - Scenario 2: agent.yaml with `backend: agent_framework` and `agent_framework: { compaction_strategy: sliding_window }` results in message history compacted
  - Scenario 3: agent.yaml with `provider: openai` and a `google_adk:` section present results in section ignored without errors, auto-detected backend used

---

## Dependencies & Execution Order

### External Dependencies (MUST be complete before US4 begins)

- **US1**: `GoogleADKConfig` model (`src/holodeck/models/google_adk_config.py`), `ADKBackend` + `ADKSession` (`src/holodeck/lib/backends/adk_backend.py`), `google_adk` field on Agent model (`src/holodeck/models/agent.py`), unit tests (`tests/unit/models/test_google_adk_config.py`)
- **US2**: `AgentFrameworkConfig` model (`src/holodeck/models/af_config.py`), `AFBackend` + `AFSession` (`src/holodeck/lib/backends/af_backend.py`), `agent_framework` field on Agent model (`src/holodeck/models/agent.py`), unit tests (`tests/unit/models/test_af_config.py`)

### Phase Dependencies

- **Foundational (Phase 2)**: No external dependencies beyond US1/US2 completion — can start immediately after US1/US2
- **Implementation (Phase 3)**: Depends on Phase 2 completion
  - 3A (ADK streaming), 3B (ADK code exec), 3C (ADK iterations/output key) can proceed in parallel
  - 3D (AF compaction), 3E (AF max tool rounds) can proceed in parallel
  - 3F (ignore behavior) can proceed in parallel with 3A–3E (tests different files)
- **Polish (Phase 4)**: Depends on all Phase 3 sub-phases being complete

### Parallel Opportunities

- T001, T002, T003, T004 can run in parallel (different test files)
- T008, T009, T012, T013, T016, T017 can run in parallel (same file, independent tests)
- T021, T022, T023, T026, T027 can run in parallel (same file, independent tests)
- T030, T031, T032, T033, T034 can run in parallel (same file, independent tests)
- T036, T037, T038 can run in parallel (independent quality checks)

---

## Key Files Modified/Created

| File | Phase | Changes |
|------|-------|---------|
| `src/holodeck/lib/backends/adk_backend.py` | Phase 2, 3A–3C | Config wiring, streaming mode, code execution, max iterations, output key |
| `src/holodeck/lib/backends/af_backend.py` | Phase 2, 3D–3E | Config wiring, compaction strategy, max tool rounds |
| `src/holodeck/lib/backends/selector.py` | Phase 3F | Verify ignore-when-mismatched behavior (likely no change needed) |
| `tests/unit/lib/backends/test_adk_backend.py` | Phase 2, 3A–3C | Config wiring + feature wiring tests |
| `tests/unit/lib/backends/test_af_backend.py` | Phase 2, 3D–3E | Config wiring + feature wiring tests |
| `tests/unit/models/test_agent_config_ignore.py` | Phase 3F | **NEW**: Ignore-when-mismatched behavior tests |

---

## Notes

- [P] tasks = different files, no dependencies
- [US4] label maps all tasks to User Story 4
- TDD: Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate progress
- US4 is primarily a wiring story — most config models and backend skeletons already exist from US1/US2
- The "ignore behavior" tests (Phase 3F) are the most important acceptance tests for this story — they validate that non-matching config sections never cause errors
- Strict model validation (`extra='forbid'`, invalid enum values) is covered by US1/US2 tests — not duplicated here
