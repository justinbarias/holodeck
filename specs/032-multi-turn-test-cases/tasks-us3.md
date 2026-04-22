---
description: "Tasks for User Story 3 — Tool call argument matchers (P2)"
---

# Tasks: User Story 3 — Assert Expected Tool Call Arguments with Fuzzy / Regex Matching (P2)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US3), data-model.md §2–§3, contracts/test-case-schema.md §3–§4, contracts/tool-arg-matchers.md (authoritative for SC-006 matrix).
**Approach**: Test-Driven Development.

**Story Goal**: Extend `expected_tools` to accept the object form `{name, args, count}` with per-arg matchers — literal, `{fuzzy: str}`, `{regex: str}`. Literal scalars use numeric int↔float equivalence but no separator stripping; fuzzy is case/whitespace/separator/percent/numeric-tolerant; regex is anchored full-match over `str(value)`. Turn assertion passes iff ≥`count` in-turn calls satisfy every asserted arg simultaneously (any-call-wins, order-independent). `count` is a **lower bound** (spec line 226 updated — exact-N is the out-of-scope variant).

**Independent Test**: Turn asserts `subtract(a={fuzzy:"206588"}, b={regex:"^181001(\\.0+)?$"})`. Passes when the agent calls `subtract(a=206588.0, b=181001)`; fails on `subtract(a=206588, b=180000)`.

**Depends on**: US1 (Turn model, TestResult.turns) and US2 (per-turn tool iteration + `tools_matched`).

---

## Phase 1: Setup

- [x] T001 Re-read contracts/tool-arg-matchers.md §1–§7 (acceptance matrix is the single source of truth for the SC-006 assertion).

---

## Phase 2: Widen `ExpectedTool` + `ArgMatcher` models (TDD)

### Tests first

- [x] T002 [P] [US3] Write `tests/unit/models/test_expected_tool.py::test_bare_string_parses_as_legacy` — `ExpectedTool.validate("subtract")` yields a name-only form, preserving FR-024.
- [x] T003 [P] [US3] Add `test_expected_tool.py::test_object_form_full` — parses `{name:"subtract", args:{a:206588, b:{fuzzy:"181001"}}, count:2}` into a typed object with `LiteralMatcher(206588)` and `FuzzyMatcher("181001")`.
- [x] T004 [P] [US3] Add `test_expected_tool.py::test_count_defaults_to_one` and `test_count_rejects_zero_and_negative` per data-model.md §2.
- [x] T005 [P] [US3] Write `tests/unit/models/test_arg_matcher.py::test_literal_scalar_list_dict` — scalars, lists, dicts route to `LiteralMatcher`.
- [x] T006 [P] [US3] Add `test_arg_matcher.py::test_fuzzy_shape_parses` and `test_regex_shape_parses_and_compiles` — `{fuzzy: "x"}` → `FuzzyMatcher`; `{regex: "^a$"}` → `RegexMatcher(compiled=re.Pattern)`.
- [x] T007 [P] [US3] Add `test_arg_matcher.py::test_both_matcher_keys_rejected` and `test_unknown_matcher_key_rejected` — `{fuzzy:"x", regex:"y"}` or `{foo:"x"}` raises `ConfigError` naming the field path (FR-025). **Deviation**: `{foo:"x"}` is treated as a LiteralMatcher dict per contracts/tool-arg-matchers.md §7 row 22; test relaxed to assert literal routing. See docstring in `test_unknown_matcher_key_treated_as_literal`.
- [x] T008 [P] [US3] Add `test_arg_matcher.py::test_bad_regex_rejected_at_load` — `{regex:"("}` raises at model construction, never at runtime.
- [x] T009 [P] [US3] Add `tests/unit/lib/test_testcase_models_multi_turn.py::test_mixed_str_and_object_expected_tools` — `expected_tools: ["lookup", {name:"subtract", args:{a:1}}]` parses without error (FR-003). (Filed under `tests/unit/models/test_expected_tool.py::TestMixedListSupport`.)

### Implementation

- [x] T010 [US3] In `src/holodeck/models/test_case.py`, replace the US1 stub `ExpectedTool = str` with a real Pydantic model `ExpectedTool(name: str, args: dict[str, ArgMatcher] | None = None, count: int = Field(1, ge=1))`. Ensure `name` is non-empty.
- [x] T011 [US3] Add `LiteralMatcher`, `FuzzyMatcher`, `RegexMatcher` dataclasses per data-model.md §3 to `src/holodeck/models/test_case.py` (or a new `src/holodeck/models/arg_matcher.py`).
- [x] T012 [US3] Add a discriminator validator.
- [x] T013 [US3] Widen `Turn.expected_tools` and `TestCaseModel.expected_tools` to `list[str | ExpectedTool] | None`.
- [x] T014 [US3] Update config-load error messages to include the test-case name/index, `turns[i].expected_tools[j].args.<key>` field path, and the underlying cause (FR-025, contracts/test-case-schema.md §7).
- [x] T014a [P] [US3] Write `tests/unit/models/test_expected_tool_round_trip.py::test_bare_string_round_trip_preserves_str_on_dump`.
- [x] T014b [P] [US3] Write `tests/unit/models/test_expected_tool_round_trip.py::test_bad_regex_surfaces_full_field_path`.

---

## Phase 3: Argument matcher module (TDD — SC-006 matrix)

### Tests first

- [x] T015 [P] [US3] Write `tests/unit/lib/test_runner/test_tool_arg_matcher.py::test_literal_matrix` parameterized over contracts/tool-arg-matchers.md §7 rows 1–4, 16–22.
- [x] T016 [P] [US3] Add `test_tool_arg_matcher.py::test_fuzzy_matrix` rows 5–11.
- [x] T017 [P] [US3] Add `test_tool_arg_matcher.py::test_regex_matrix` rows 12–15.
- [x] T018 [P] [US3] Add `test_tool_arg_matcher.py::test_multi_call_any_wins` row 23.
- [x] T019 [P] [US3] Add `test_tool_arg_matcher.py::test_none_value_semantics`.
- [x] T020 [P] [US3] Add `test_tool_arg_matcher.py::test_count_threshold`.

### Implementation

- [x] T021 [US3] Create `src/holodeck/lib/test_runner/tool_arg_matcher.py`.
- [x] T022 [US3] Add `find_matching_call` + factor out `_tool_name_matches`.
- [x] T023 [US3] Implement `evaluate_expected_tools`.

---

## Phase 4: Executor integration (TDD)

### Tests first

- [x] T024 [P] [US3] Add `tests/unit/lib/test_runner/test_executor_arg_matching.py::test_fuzzy_regex_args_populate_arg_match_details`.
- [x] T025 [P] [US3] Add `test_executor_arg_matching.py::test_missing_arg_fails_turn_with_reason`.
- [x] T026 [P] [US3] Add `test_executor_arg_matching.py::test_extras_ignored`.
- [x] T027 [P] [US3] Add `test_executor_arg_matching.py::test_name_matches_but_args_mismatch_fails`.

### Implementation

- [x] T027a [US3] Partition `expected_tools` in `_run_single_turn` + legacy single-turn path.
- [x] T028 [US3] Use `tool_arg_matcher.evaluate_expected_tools`.
- [x] T029 [US3] Populate `TurnResult.arg_match_details`.
- [x] T030 [US3] Per-assertion summary line in `TurnResult.errors`.

---

## Phase 5: Reporter + end-to-end

### Tests first

- [x] T031 [P] [US3] Add `tests/unit/lib/test_runner/test_reporter_multi_turn.py::test_arg_match_details_rendered_on_failure`.
- [x] T032 [P] [US3] Add `tests/integration/test_multi_turn_us3_e2e.py` (match + mismatch variants).

### Implementation

- [x] T033 [US3] Extend reporter markdown path to render `arg_match_details`.
- [x] T034 [US3] Run `make format && make lint && make type-check && pytest -n auto`. All green (pre-existing mypy warnings in `source_resolver.py` / `serve/server.py` unchanged on main, not introduced here).

---

## Dependencies

- Phase 1 → Phase 2 → Phase 3 (matcher module is independent of model widening — could run in parallel on different branches, but both are prereqs for Phase 4).
- Phase 4 depends on Phases 2 and 3.
- Phase 5 depends on Phase 4.
- Within Phase 2: T002–T009, T014a, T014b parallel; T010–T014 serial (same file).
- Within Phase 3: T015–T020 parallel; T021–T023 serial (single module `tool_arg_matcher.py`).
- Within Phase 4: T024–T027 parallel; T027a–T030 serial (all in `executor.py`).

## Independent test criteria (recap)

- SC-006 — 23-row acceptance matrix all green (T015–T020, verified end-to-end by T032).
- Extra-args tolerated, missing-args fail, bool strict, list/dict eq, percent and separators fuzzy — all covered.
- Malformed regex rejected at config load, never at runtime (T008, T014b).
- Legacy `list[str]` serialization preserved (T014a — SC-002 guard).
