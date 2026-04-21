---
description: "Tasks for User Story 1 — Multi-turn conversation execution (P1 MVP)"
---

# Tasks: User Story 1 — Run a Multi-Turn Conversation Against an Agent (P1)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md, plan.md, research.md, data-model.md, contracts/test-case-schema.md, contracts/turn-result-schema.md
**Approach**: Test-Driven Development — every behavior change starts with a failing unit or integration test.

**Story Goal**: Extend the test runner so a `test_case` can declare an ordered `turns` list; the executor drives them through a single `AgentSession` (turn N+1 only after turn N), captures per-turn input/response/tool-invocations/token-usage, and reports `TestResult.turns`. No per-turn assertions yet (US2), no arg matchers (US3), no programmatic evaluators (US4). Per-turn multimodal `files` are flattened into the prompt string via the existing `_prepare_agent_input` helper (no `AgentSession.send()` protocol widening — native multimodal per-turn is deferred to a later feature).

**Independent Test (from spec)**: Define a 3-turn test case with no ground truths. Run `holodeck test`. Verify: (a) turn-2 resolves turn-1 anaphora; (b) the report lists one row per turn; (c) a single failing turn fails the whole test case. Matches SC-003 (100% of turns appear in the report with input/response/tool-invocations/pass-fail) and the dual-backend half of SC-010.

**Scope boundary**: This file covers the foundational model/schema work that US2–US5 depend on (Turn, TurnResult, `TestResult.turns`, `TokenUsage` cache fields, `parallel_test_cases`, `MetricResult.kind` widening, `grader_details`, multi-turn markdown hierarchy). Complete this story first — it's blocking.

---

## Phase 1: Setup

- [ ] T001 Read `specs/032-multi-turn-test-cases/spec.md`, `plan.md`, `data-model.md`, and `contracts/test-case-schema.md` + `contracts/turn-result-schema.md`; confirm Python 3.10+ venv is active and `pytest -n auto` works from repo root.
- [ ] T002 [P] Create `tests/fixtures/multi_turn/` (mkdir -p) and add a curated ConvFinQA-shaped fixture at `tests/fixtures/multi_turn/convfinqa_sample.yaml` (4 turns, no ground truths yet) that later phases can reuse.

---

## Phase 2: Foundational (blocks US2–US5)

**⚠️ CRITICAL**: No subsequent user story can start until Phase 2 is green.

### Foundational tests (TDD — write first, expect FAIL)

- [ ] T003 [P] Write `tests/unit/models/test_turn_model.py` asserting `Turn(input="hi")` round-trips, `Turn(input="")` raises `ValidationError`, unknown fields are forbidden (`extra="forbid"`), and `ground_truth=""` is rejected.
- [ ] T004 [P] Write `tests/unit/lib/test_testcase_models_multi_turn.py` covering: `turns` + `input` on the same test case raises `ValidationError`; `turns: []` raises; legacy `TestCaseModel(input="x")` still parses; top-level `files` / `retrieval_context` alongside `turns` raises with a hint message per contracts/test-case-schema.md §2.
- [ ] T005 [P] Write `tests/unit/models/test_test_result_turn_rollup.py` covering the §4 roll-up table: `test_input` joined with `\n---\n`, `agent_response` = last turn's response, `tool_calls`/`tool_invocations` flattened in turn order, **`processed_files` flattened in turn order** (contracts §4), `passed = all(turn.passed)`, `ground_truth=None` at test-case level, `execution_time_ms = sum(turns)` (skipped turns contribute 0), `errors` prefixed `[turn N]`. Also assert `ReportSummary.validate_test_counts` still satisfies `passed + failed == total_tests` counting test cases (not turns) for a mixed single-turn + multi-turn run (FR-015 regression).
- [ ] T006 [P] Write `tests/unit/models/test_token_usage_cache_fields.py` covering new `cache_creation_tokens` / `cache_read_tokens` defaults (0), `__add__` element-wise over all four countable fields, and the relaxed `total_tokens >= prompt + completion` validator (data-model.md §10a).
- [ ] T007 [P] Write `tests/unit/models/test_execution_config_parallel.py` asserting `ExecutionConfig.parallel_test_cases` defaults to `1`, rejects `< 1`, and that `DEFAULT_EXECUTION_CONFIG["parallel_test_cases"] == 1`.
- [ ] T008 [P] Write `tests/unit/models/test_metric_result_kind_extended.py` asserting `MetricResult(kind="code", metric_name="x", score=1.0, passed=True, ...)` round-trips, legacy `kind` values still parse, and `_metric_kind(CodeMetric(...))` returns `"code"` (data-model.md §7a — prerequisite for US4).

### Foundational implementation (only after T003–T008 fail)

- [ ] T009 [US1] Add `Turn` model in `src/holodeck/models/test_case.py` per data-model.md §1 (input required non-empty, `ground_truth` optional non-empty, `expected_tools` typed as `list[str] | None` for now — US3 widens it, `files`, `retrieval_context`, `evaluations` optional). `ConfigDict(extra="forbid")`.
- [ ] T010 [US1] Add stub `ExpectedTool` placeholder alias `ExpectedTool = str` in `src/holodeck/models/test_case.py` with a `# TODO(US3): widen to object form` marker so US3 can extend without churn. Do not widen yet.
- [ ] T011 [US1] Extend `TestCaseModel` in `src/holodeck/models/test_case.py`: make `input` optional, add `turns: list[Turn] | None`, and add a `model_validator(mode="after")` enforcing the mutual-exclusion + non-empty-turns + top-level-files-forbidden rules from contracts/test-case-schema.md §2.
- [ ] T012 [P] [US1] Add `TurnResult` model in `src/holodeck/models/test_result.py` per data-model.md §8. Required fields include `execution_time_ms` and `metric_results`; defaults include `skipped=False`; **reserve `grader_details: dict[str, Any] | None = None`** for US4 (contract/code-grader-contract.md §7).
- [ ] T013 [US1] Add optional `turns: list[TurnResult] | None = None` to `TestResult` in `src/holodeck/models/test_result.py`; confirm `ReportSummary.validate_test_counts` still counts test cases (regression asserted in T005).
- [ ] T014 [P] [US1] Extend `TokenUsage` in `src/holodeck/models/token_usage.py` with `cache_creation_tokens` / `cache_read_tokens` (ge=0, default 0), update `__add__` and `zero()`, and relax the `total_tokens` validator to `>=`. **Audit + update `tests/unit/models/test_token_usage.py` and `tests/unit/models/test_test_result_token_usage.py`** for the relaxed validator (existing tests assert strict equality; update each to `>=` or add separate strict-equality cases for non-cached runs). Call out the behavior change in the eventual PR description.
- [ ] T015 [P] [US1] Add `parallel_test_cases: int = Field(1, ge=1)` to `ExecutionConfig` in `src/holodeck/models/config.py` and bump `DEFAULT_EXECUTION_CONFIG` in `src/holodeck/config/defaults.py`.
- [ ] T016 [P] [US1] Extend `MetricResult.kind` in `src/holodeck/models/test_result.py` from `Literal["standard","rag","geval"]` to `Literal["standard","rag","geval","code"]` (additive; legacy JSON still parses). Update `_metric_kind()` return annotation in `src/holodeck/lib/test_runner/executor.py` to match (data-model.md §7a — unblocks US4 before it ships).
- [ ] T017 [US1] Run `pytest tests/unit/models -n auto` — all foundational tests green.

**Checkpoint**: Data-model foundation complete. US2/US3/US4/US5 are unblocked, but US1's executor work continues below.

---

## Phase 3: Executor multi-turn dispatch (US1 core)

### Tests first

- [ ] T018 [P] [US1] Write `tests/unit/lib/test_runner/test_executor_multi_turn.py::test_dispatch_detects_turns` that stubs an `AgentBackend`/`AgentSession` and asserts a `TestCaseModel` with `turns` routes through `backend.create_session()` + `session.send()` — not `invoke_once()`.
- [ ] T019 [P] [US1] Add `test_executor_multi_turn.py::test_legacy_single_turn_unchanged` asserting a legacy `TestCaseModel(input="x")` still routes through `invoke_once()` and leaves `TestResult.turns = None` (SC-002 regression).
- [ ] T020 [P] [US1] Add `test_executor_multi_turn.py::test_turns_strictly_sequential` asserting turn N+1's `send()` is only awaited after turn N's response resolves (use an `asyncio.Event` trace).
- [ ] T021 [P] [US1] Add `test_executor_multi_turn.py::test_session_closed_on_completion` asserting `session.close()` is called on success, failure, and unrecoverable-session paths (use a context spy).
- [ ] T022 [P] [US1] Add `test_executor_multi_turn.py::test_turn_timeout_fails_turn_continues_next` — a per-turn `asyncio.TimeoutError` marks that turn failed, session stays open, next turn runs (FR-008, spec Scenario 4).
- [ ] T023 [P] [US1] Add `test_executor_multi_turn.py::test_two_consecutive_session_errors_mark_remaining_skipped` — backend raises `BackendSessionError` on turn 2 and turn 3; the executor treats the session as unrecoverable after the second consecutive error and marks turn 4 (and beyond) as `skipped=True` with an explanatory `errors` entry (research.md §2 heuristic; spec Edge Case "Session errors mid-conversation"). A single isolated `BackendSessionError` must NOT trigger skipping.
- [ ] T024 [P] [US1] Add `test_executor_multi_turn.py::test_per_turn_tool_invocations_partitioned` asserting each `TurnResult.tool_invocations` is populated directly from that turn's `ExecutionResult.tool_calls` / `tool_results` — no cross-turn bleed, no shared buffer.
- [ ] T025 [P] [US1] Add `test_executor_multi_turn.py::test_token_usage_rollup_sum` — per-turn `TokenUsage` objects sum element-wise (including `cache_creation_tokens` / `cache_read_tokens`) into the test-case-level `TestResult.token_usage` (FR-007 + T006 contract).
- [ ] T026 [P] [US1] Add `test_executor_multi_turn.py::test_files_flattened_into_turn_prompt_not_replayed` — turn 1 has `files: [{path: chart.png, type: image}]`, turn 2 has no files. Assert: (a) `_prepare_agent_input` is called with turn 1's `files` and its output is passed to `session.send(message=...)`; (b) turn 2's `send()` receives only the plain input string, no residual file content (FR-009). Per-turn `processed_files` collected for rollup.
- [ ] T027 [P] [US1] Add `test_executor_multi_turn.py::test_test_case_passed_requires_all_turns_passed` — one failing turn flips test-case `passed` to `False` even if the others pass (FR-016).

### Implementation (only after T018–T027 FAIL)

- [ ] T028 [US1] In `src/holodeck/lib/test_runner/executor.py`, add a `_is_multi_turn(test_case)` branch that checks `test_case.turns is not None` and dispatches to a new private `_run_multi_turn(test_case, backend, ...)` coroutine; legacy path is untouched.
- [ ] T029 [US1] Implement `_run_multi_turn` in `src/holodeck/lib/test_runner/executor.py` to: call `backend.create_session()` once, iterate turns sequentially calling `session.send(_prepare_agent_input(turn.input, turn.files))` under `asyncio.wait_for(..., timeout=llm_timeout)`, collect per-turn `ExecutionResult`s, and always `await session.close()` in a `finally`. No `AgentSession.send()` protocol widening.
- [ ] T030 [US1] Capture per-turn timing (`time.perf_counter` around each `send()`) and populate `TurnResult.execution_time_ms`. Populate `TurnResult.tool_invocations` directly from `exec_result.tool_calls` / `exec_result.tool_results` via the existing `pair_tool_calls` helper — one `ExecutionResult` per turn, no cross-turn sharing.
- [ ] T031 [US1] Map per-turn errors: `asyncio.TimeoutError` → `errors=["timeout"]`; other backend exceptions → `errors=[f"{type(e).__name__}: {e}"]`. Track a counter of consecutive `BackendSessionError` occurrences; on the second consecutive hit, break the loop and mark remaining turns `skipped=True` with explanatory `errors` (research.md §2 heuristic). Reset the counter on any successful turn.
- [ ] T032 [US1] Implement the `TestResult` roll-up in `_finalize_multi_turn_result` per contracts/turn-result-schema.md §4: `test_input` join, `agent_response` = last turn response, `tool_calls` / `tool_invocations` flattened, **`processed_files` flattened in turn order**, `token_usage` summed (including cache fields), `passed = all(t.passed)`, `errors` prefixed `[turn N]`, `ground_truth=None`. Leave `metric_results` empty here — US2 fills it.
- [ ] T033 [US1] Run `pytest tests/unit/lib/test_runner -n auto` — T018–T027 now green.

---

## Phase 4: Reporter + CLI wiring (markdown hierarchy + parallel flag)

### Tests first

- [ ] T034 [P] [US1] Write `tests/unit/lib/test_runner/test_reporter_multi_turn.py::test_markdown_renders_per_turn_rows` asserting the markdown reporter emits one child row per turn under the test-case row, with `turn_index`, truncated `input`, pass/fail glyph, and `execution_time_ms`.
- [ ] T035 [P] [US1] Add `test_reporter_multi_turn.py::test_markdown_hierarchy_parent_and_turns` — markdown output has a parent heading per multi-turn test case and an indented block per turn with input/response/tools/metrics (absorbs former US5 T019).
- [ ] T036 [P] [US1] Add `test_reporter_multi_turn.py::test_single_turn_markdown_unchanged` — a legacy test case renders exactly as the pre-feature golden, preserving SC-002 at the markdown layer (absorbs former US5 T020).
- [ ] T037 [P] [US1] Add `test_reporter_multi_turn.py::test_json_reporter_includes_turns_field` — `TestResult.turns` present on serialized JSON for multi-turn cases and absent for legacy (absorbs former US5 T021).
- [ ] T038 [P] [US1] Write `tests/unit/cli/test_test_command_parallel_flag.py` asserting `holodeck test --parallel-test-cases 4` pushes `parallel_test_cases=4` through `ExecutionConfig` and that CLI > YAML > env > defaults precedence holds.

### Implementation

- [ ] T039 [US1] In `src/holodeck/lib/test_runner/reporter.py`, extend the markdown renderer to produce hierarchical output (parent + per-turn block) when `TestResult.turns` is present; single-turn path untouched. JSON path is automatic via Pydantic serialization.
- [ ] T040 [US1] In `src/holodeck/cli/commands/test.py`, add `--parallel-test-cases INTEGER` Click option plumbed through the existing resolve chain (`CLI > YAML > env > defaults`). Validate `>= 1`.
- [ ] T041 [P] [US1] Write `tests/unit/config/test_execution_config_parallel_resolve.py::test_parallel_propagates_through_resolve_chain` (pure TDD; fails first).
- [ ] T042 [US1] In `src/holodeck/config/loader.py`, extend `_resolve_execution_config` (or equivalent) to propagate `parallel_test_cases` without breaking existing keys. T041 goes green.

---

## Phase 5: Concurrency orchestration (FR-009a)

### Tests first

- [ ] T043 [P] [US1] Write `tests/unit/lib/test_runner/test_executor_parallel.py::test_parallel_respects_semaphore` — with `parallel_test_cases=2` and 4 test cases, at most 2 sessions are open concurrently (trace via `asyncio.Semaphore` spy).
- [ ] T044 [P] [US1] Add `test_executor_parallel.py::test_turns_within_case_stay_sequential` — even under `parallel_test_cases=4`, a single multi-turn case still sends turns strictly one-at-a-time (FR-009a).
- [ ] T045 [P] [US1] Add `test_executor_parallel.py::test_reporter_writes_serialized` — concurrent test cases do not interleave reporter output mid-record (write lock or single-producer queue).

### Implementation

- [ ] T046 [US1] Wrap the per-test-case coroutine in `asyncio.Semaphore(execution.parallel_test_cases)` in `src/holodeck/lib/test_runner/executor.py`; each acquired slot holds its own `AgentSession`.
- [ ] T047 [US1] Guard reporter emission with an `asyncio.Lock` (or funnel through a single producer coroutine) so per-test-case output blocks don't interleave.

---

## Phase 6: Dual-backend integration (SC-010 smoke)

- [ ] T048 [P] [US1] Write `tests/integration/test_multi_turn_dual_backend.py::test_sk_backend_three_turn_passes` using a mocked SK `AgentSession` (or the real Ollama provider when `HOLODECK_IT_OLLAMA=1`). Asserts 3 turns stream, `TestResult.turns` has 3 entries with non-empty responses and summed token usage.
- [ ] T049 [P] [US1] Add `test_multi_turn_dual_backend.py::test_claude_backend_three_turn_passes` gated on `HOLODECK_IT_ANTHROPIC=1` (or a stubbed Claude `AgentSession`). Same assertions; identical pass/fail as the SK case (SC-010).
- [ ] T050 [US1] Document how to run the gated integration tests (`make test-integration HOLODECK_IT_OLLAMA=1 HOLODECK_IT_ANTHROPIC=1`) in the test file's module docstring — no standalone doc file.

---

## Phase 7: Back-compat + Polish

- [ ] T051 [P] [US1] Add `tests/unit/lib/test_runner/test_legacy_backcompat.py` running every existing fixture under `tests/fixtures/` through the new executor path and asserting structural equality with the pre-feature golden JSON (use `DeepDiff` or explicit dict comparison with `exclude_defaults=True` on both sides — avoids false failures from additive optional fields like `turns: null`, `cache_creation_tokens: 0`). Preserves SC-002.
- [ ] T052 [P] [US1] Update `CLAUDE.md` "Recent Changes" block via `.specify/scripts/bash/update-agent-context.sh claude` so future sessions see the new `parallel_test_cases` / `turns` surface.
- [ ] T053 [US1] Run `make format && make lint && make type-check && pytest -n auto` — all green.

**Checkpoint**: US1 fully functional. Any `turns:`-shaped test case runs end-to-end with per-turn reporting on both backends. Foundational models (`Turn`, `TurnResult`, `TestResult.turns`, `TokenUsage` cache fields, `MetricResult.kind='code'`, `TurnResult.grader_details`) are in place for US2–US5.

---

## Dependencies

- Phase 1 (T001–T002) → Phase 2 (T003–T017) → Phase 3 (T018–T033) → Phase 4 (T034–T042) → Phase 5 (T043–T047) → Phase 6 (T048–T050) → Phase 7 (T051–T053).
- T003–T008 run in parallel; T009–T016 mostly parallelizable except T011 (awaits T009/T010) and T013 (awaits T012).
- T018–T027 all parallel (new test file additions).
- T028–T032 serial (same file: `executor.py`).

## Parallel examples

```
# After T002: launch foundational tests together
pytest tests/unit/models/test_turn_model.py \
       tests/unit/lib/test_testcase_models_multi_turn.py \
       tests/unit/models/test_test_result_turn_rollup.py \
       tests/unit/models/test_token_usage_cache_fields.py \
       tests/unit/models/test_execution_config_parallel.py \
       tests/unit/models/test_metric_result_kind_extended.py -n auto
```

## MVP definition for US1

Delivers a runnable multi-turn conversation with per-turn recording. Shippable on its own even without per-turn assertions — gives benchmark authors immediate visibility into conversational state. Also lands the foundational schema pieces (`grader_details`, `MetricResult.kind='code'`, `processed_files` rollup, reporter hierarchy) that US2/US4/US5 depend on, so those stories become pure additive work rather than schema-widening work.
