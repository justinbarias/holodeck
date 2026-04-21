---
description: "Tasks for User Story 2 — Per-turn ground truths and expected tools (P1)"
---

# Tasks: User Story 2 — Assert Per-Turn Ground Truths and Expected Tools (P1)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US2), data-model.md §8–§9, contracts/turn-result-schema.md §3–§4.
**Approach**: Test-Driven Development.

**Story Goal**: For each turn, run the configured metric stack against that turn's `ground_truth` and verify `expected_tools: list[str]` (legacy name-only form) were called *in that turn*. Roll up per-metric scores to the test-case level using `score = mean(turn_scores)` and `passed = all(turn_passed)`, skipping turns that had no ground truth for a given metric. `TurnResult.passed` composes four conjuncts (errors, skipped, tools_matched, metrics) per contract §3.

**Independent Test**: Author a test case where turn 3 has `ground_truth: "25587"` and `expected_tools: ["subtract"]`. Run `holodeck test`. Verify turn 3 passes on metric match and tool-name match; fails if the agent answers `"25000"` or never calls `subtract`. Failure pinpoints turn index in the report (SC-004).

**Depends on**: US1 Phase 2 foundational models (Turn, TurnResult, `TestResult.turns`, `MetricResult.kind='code'` widening, `TurnResult.grader_details`) and US1 T002 fixture. If those aren't landed, start there first.

---

## Phase 1: Setup

- [ ] T001 Re-read `specs/032-multi-turn-test-cases/spec.md` §US2 and data-model.md §7a (MetricResult.kind) + §9 (roll-up rules). Confirm US1 foundational models are merged.

---

## Phase 2: Per-turn ground-truth evaluation (TDD)

### Tests first

- [ ] T002 [P] [US2] Write `tests/unit/lib/test_runner/test_executor_per_turn_metrics.py::test_turn_with_ground_truth_runs_metrics` — a turn with `ground_truth` and a BLEU-configured agent produces a `MetricResult(metric_name="bleu")` on that `TurnResult.metric_results`, with `passed` derived from the configured threshold.
- [ ] T003 [P] [US2] Add `test_executor_per_turn_metrics.py::test_turn_without_ground_truth_skips_text_metrics` — a turn with no `ground_truth` has no BLEU/ROUGE/METEOR entries in its `metric_results` (A6).
- [ ] T004 [P] [US2] Add `test_executor_per_turn_metrics.py::test_per_turn_evaluations_override_agent_defaults` — when `turns[i].evaluations` is set, it overrides agent-level metrics for that turn only (FR-023).
- [ ] T004a [P] [US2] Add `test_executor_per_turn_metrics.py::test_test_case_evaluations_override_agent_when_turn_unset` — agent has BLEU only, test_case has ROUGE only, turn has no `evaluations` → turn evaluates ROUGE only. Covers the middle rung of the three-level resolver.
- [ ] T005 [P] [US2] Add `test_executor_per_turn_metrics.py::test_rollup_mean_across_turns` — agent has BLEU configured; 3 turns score 0.8, 0.4, 0.6 → `TestResult.metric_results[bleu].score == 0.6` and `passed = all(per-turn passed)`.
- [ ] T006 [P] [US2] Add `test_executor_per_turn_metrics.py::test_rollup_skips_turns_without_metric` — turn 2 has no `ground_truth`; the metric averages only over turns 1 and 3 (data-model.md §9).
- [ ] T007 [P] [US2] Add `test_executor_per_turn_metrics.py::test_metric_omitted_when_every_turn_skipped` — no turn has a `ground_truth`; the metric does not appear in `TestResult.metric_results` at all.
- [ ] T008 [P] [US2] Add `test_executor_per_turn_metrics.py::test_failing_turn_flips_rolled_up_passed` — 2/3 turns pass BLEU, 1 fails → rolled-up `passed=False` even though mean score might still exceed threshold.

### Implementation

- [ ] T009a [US2] Refactor `_run_evaluations` in `src/holodeck/lib/test_runner/executor.py` (lines 772-910) to accept `(agent_response, ground_truth, retrieval_context, processed_files, tool_results)` as parameters instead of reading from `test_case.input` / `test_case.ground_truth`. Single-turn legacy call site at `executor.py:724-740` must pass equivalent values — verified by the existing single-turn test suite staying green (SC-002).
- [ ] T009b [US2] Add `_resolve_turn_metrics(agent_config, test_case, turn) -> list[MetricType]` in `executor.py` returning per-turn effective metrics with precedence **turn > test_case > agent**. Extends the existing two-level `_get_metrics_for_test` (executor.py:912-937). Unit-tested via T004/T004a.
- [ ] T009c [US2] In `_run_multi_turn`, after each `session.send()`, call `_resolve_turn_metrics(...)` then `_run_evaluations(...)` with the per-turn inputs. Store the resulting `MetricResult` list on `TurnResult.metric_results`.
- [ ] T009d [US2] Run `pytest tests/unit/lib/test_runner -n auto` — confirm no legacy regressions after the T009a refactor (SC-002 gate before proceeding to Phase 3).
- [ ] T010 [US2] Implement `_rollup_metric_results(turns: list[TurnResult]) -> list[MetricResult]` per data-model.md §9: group by `metric_name`, skip turns where the metric is absent, compute `score = mean(turn_scores)`, `passed = all(turn_passed)`, drop metrics that every turn skipped.
- [ ] T011 [US2] Wire `_rollup_metric_results` into `_finalize_multi_turn_result` so `TestResult.metric_results` is populated from the per-turn data (replacing the empty-list stub left by US1 T032).
- [ ] T012 [US2] Implement `TurnResult.passed` per contracts/turn-result-schema.md §3: `errors == [] and skipped is False and tools_matched is not False and all(m.passed is not False for m in metric_results)`. All four conjuncts required.
- [ ] T012a [P] [US2] Add `test_executor_per_turn_metrics.py::test_turn_with_errors_only_fails` — a turn with `errors=['timeout']`, no `expected_tools`, no `ground_truth`, no `metric_results` has `passed=False`. Locks in the first conjunct (`errors == []`). Also `test_skipped_turn_fails` and `test_tools_matched_false_fails` for completeness.
- [ ] T013 [US2] Run `pytest tests/unit/lib/test_runner/test_executor_per_turn_metrics.py -n auto` — T002–T008 + T012a green.

---

## Phase 3: Per-turn tool-name assertion (TDD)

### Tests first

- [ ] T014 [P] [US2] Write `tests/unit/lib/test_runner/test_executor_per_turn_tools.py::test_tool_name_match_substring_scoped_to_turn` — turn 3 asserts `expected_tools: ["subtract"]`; a prior turn also called `subtract`. Only in-turn calls count → turn 3 fails if it never called `subtract` that turn (spec Scenario US2.3).
- [ ] T015 [P] [US2] Add `test_executor_per_turn_tools.py::test_per_turn_tool_match_inherits_substring_contract` — regression guard that `validate_tool_calls` substring semantics (case-sensitive, `expected in actual`) still hold when called from the multi-turn path. Covers SK plugin-name prefixing (`"Math-subtract"` accepts expected `"subtract"`).
- [ ] T016 [P] [US2] Add `test_executor_per_turn_tools.py::test_missing_expected_tool_fails_turn` — expected `["divide"]`, agent calls nothing → `tools_matched=False`, `turn.passed=False`, `errors` mentions the missing name.
- [ ] T017 [P] [US2] Add `test_executor_per_turn_tools.py::test_extra_tools_do_not_fail_turn` — expected `["subtract"]`, agent also calls `lookup` → `tools_matched=True` (FR — `expected_tools` is a lower bound, not upper).
- [ ] T018 [P] [US2] Add `test_executor_per_turn_tools.py::test_no_expected_tools_leaves_matched_none` — turn with no `expected_tools` → `TurnResult.tools_matched is None`; the test-case rollup treats `None` as neutral.
- [ ] T019 [P] [US2] Add `test_executor_per_turn_tools.py::test_rollup_tools_matched_all_turns` — covers three cases: (a) turn 1 `matched=True`, turn 2 `matched=None`, turn 3 `matched=True` → test-case `True`; (b) all turns `None` → test-case `None`; (c) any turn explicitly `False` → test-case `False`.

### Implementation

- [ ] T020 [US2] Call `validate_tool_calls([inv.name for inv in turn.tool_invocations], turn.expected_tools)` from `src/holodeck/lib/test_runner/executor.py`. Do NOT modify `validate_tool_calls` itself — its contract is shared with the legacy single-turn path and covered by existing tests (SC-002). Set `TurnResult.tools_matched` per contracts/turn-result-schema.md §2.
- [ ] T021 [US2] Populate `TurnResult.errors` with a human-readable missing-tool message when `tools_matched=False`. Preserve the existing legacy error-message shape for back-compat on single-turn cases.
- [ ] T022 [US2] Update `_finalize_multi_turn_result` so `TestResult.tools_matched` aggregates per contracts/turn-result-schema.md §4: `None` if every turn is `None`, `True` if all turns are `True`/`None`, `False` if any turn is `False`.

---

## Phase 4: Reporter updates for per-turn metric/tool detail

### Tests first

- [ ] T023 [P] [US2] Add `tests/unit/lib/test_runner/test_reporter_multi_turn.py::test_markdown_shows_per_turn_metric_scores` — rendered markdown includes each turn's metric scores (e.g., `bleu=0.8`) and tool match glyph.
- [ ] T024 [P] [US2] Add `test_reporter_multi_turn.py::test_failure_pinpoints_turn_index` — a failing turn's rendered line names `turn 2` and the failing metric or missing tool (SC-004).

### Implementation

- [ ] T025 [US2] Extend the US1 markdown hierarchy (from US1 T039) in `src/holodeck/lib/test_runner/reporter.py` to print per-turn metric scores and tool-match status under each turn row. JSON path stays automatic.

---

## Phase 5: End-to-end happy path

- [ ] T026 [US2] Add `tests/integration/test_multi_turn_us2_e2e.py::test_4turn_per_turn_ground_truth_tools` — 4-turn fixture (use `tests/fixtures/multi_turn/convfinqa_sample.yaml` created in US1 T002, extended here with ground truths + `["subtract"]` / `["divide"]`), stubbed backend with scripted responses and tool calls, asserts per-turn metric results appear, rolled-up `TestResult.passed=True`, per-turn breakdown in markdown report.
- [ ] T027 [US2] Run `make format && make lint && pytest -n auto`. All green.

**Checkpoint**: US1 + US2 together deliver a shippable P1 slice — multi-turn execution with per-turn success criteria.

---

## Dependencies

- Phase 1 → Phase 2 → Phase 3 (parallel with Phase 2 implementation is OK, both touch `executor.py` but different helpers) → Phase 4 → Phase 5.
- Within Phase 2: T002–T008, T012a parallel; T009a–T013 serial (all `executor.py`).
- Within Phase 3: T014–T019 parallel; T020–T022 serial.

## Independent test criteria (recap)

- Per-turn `ground_truth` mismatch fails that turn → whole test case fails (FR-016).
- Turn tool assertion is scoped to its own turn (FR-011 — no cross-turn credit).
- Roll-up `score = mean` and `passed = all` matches data-model.md §9 — verified by T005–T008.
- `TurnResult.passed` 4-conjunct rule — verified by T012a.
- Back-compat: every legacy single-turn test stays byte-identical (piggy-backs on US1 T051, gated here at T009d).
