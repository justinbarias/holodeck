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

- [ ] T001 Re-read contracts/tool-arg-matchers.md §1–§7 (acceptance matrix is the single source of truth for the SC-006 assertion).

---

## Phase 2: Widen `ExpectedTool` + `ArgMatcher` models (TDD)

### Tests first

- [ ] T002 [P] [US3] Write `tests/unit/models/test_expected_tool.py::test_bare_string_parses_as_legacy` — `ExpectedTool.validate("subtract")` yields a name-only form, preserving FR-024.
- [ ] T003 [P] [US3] Add `test_expected_tool.py::test_object_form_full` — parses `{name:"subtract", args:{a:206588, b:{fuzzy:"181001"}}, count:2}` into a typed object with `LiteralMatcher(206588)` and `FuzzyMatcher("181001")`.
- [ ] T004 [P] [US3] Add `test_expected_tool.py::test_count_defaults_to_one` and `test_count_rejects_zero_and_negative` per data-model.md §2.
- [ ] T005 [P] [US3] Write `tests/unit/models/test_arg_matcher.py::test_literal_scalar_list_dict` — scalars, lists, dicts route to `LiteralMatcher`.
- [ ] T006 [P] [US3] Add `test_arg_matcher.py::test_fuzzy_shape_parses` and `test_regex_shape_parses_and_compiles` — `{fuzzy: "x"}` → `FuzzyMatcher`; `{regex: "^a$"}` → `RegexMatcher(compiled=re.Pattern)`.
- [ ] T007 [P] [US3] Add `test_arg_matcher.py::test_both_matcher_keys_rejected` and `test_unknown_matcher_key_rejected` — `{fuzzy:"x", regex:"y"}` or `{foo:"x"}` raises `ConfigError` naming the field path (FR-025).
- [ ] T008 [P] [US3] Add `test_arg_matcher.py::test_bad_regex_rejected_at_load` — `{regex:"("}` raises at model construction, never at runtime.
- [ ] T009 [P] [US3] Add `tests/unit/lib/test_testcase_models_multi_turn.py::test_mixed_str_and_object_expected_tools` — `expected_tools: ["lookup", {name:"subtract", args:{a:1}}]` parses without error (FR-003).

### Implementation

- [ ] T010 [US3] In `src/holodeck/models/test_case.py`, replace the US1 stub `ExpectedTool = str` with a real Pydantic model `ExpectedTool(name: str, args: dict[str, ArgMatcher] | None = None, count: int = Field(1, ge=1))`. Ensure `name` is non-empty.
- [ ] T011 [US3] Add `LiteralMatcher`, `FuzzyMatcher`, `RegexMatcher` dataclasses per data-model.md §3 to `src/holodeck/models/test_case.py` (or a new `src/holodeck/models/arg_matcher.py`).
- [ ] T012 [US3] Add a discriminator validator (follow the shape-based validator idiom from `FileInput` at test_case.py:38–67 — not the `Field(discriminator="type")` idiom used by `MetricType`, since `ArgMatcher` is shape-discriminated):
  - Accepts dicts with exactly one of `fuzzy` / `regex` → coerces to `FuzzyMatcher` / `RegexMatcher` (compile regex eagerly).
  - Rejects dicts with both keys or any unknown key, naming the full field path.
  - Treats every other value as `LiteralMatcher(value)`.
- [ ] T013 [US3] Widen `Turn.expected_tools` and `TestCaseModel.expected_tools` to `list[str | ExpectedTool] | None`, with a pre-validator that promotes bare strings to `ExpectedTool(name=..)` internally. Add a `@field_serializer` on `expected_tools` that emits bare strings when `args is None and count == 1` (the promoted-legacy shape), preserving the list[str] wire format for existing consumers (FR-024).
- [ ] T014 [US3] Update config-load error messages to include the test-case name/index, `turns[i].expected_tools[j].args.<key>` field path, and the underlying cause (FR-025, contracts/test-case-schema.md §7).
- [ ] T014a [P] [US3] Write `tests/unit/models/test_expected_tool_round_trip.py::test_bare_string_round_trip_preserves_str_on_dump` — `TestCaseModel(expected_tools=["lookup"]).model_dump()["expected_tools"] == ["lookup"]`. Protects SC-002 byte-identity for legacy single-turn JSON output.
- [ ] T014b [P] [US3] Write `tests/unit/models/test_expected_tool_round_trip.py::test_bad_regex_surfaces_full_field_path` — loading `TestCaseModel(name="tc1", turns=[{"input":"x", "expected_tools":[{"name":"x","args":{"a":{"regex":"("}}}]}])` raises `ConfigError` whose message contains `"tc1"` AND `"turns[0].expected_tools[0].args.a.regex"`.

---

## Phase 3: Argument matcher module (TDD — SC-006 matrix)

### Tests first

- [ ] T015 [P] [US3] Write `tests/unit/lib/test_runner/test_tool_arg_matcher.py::test_literal_matrix` parameterized over contracts/tool-arg-matchers.md §7 rows 1–4, 16–22 (literal, int↔float, missing arg, extras ignored, bool strict, list/dict eq).
- [ ] T016 [P] [US3] Add `test_tool_arg_matcher.py::test_fuzzy_matrix` parameterized over rows 5–11 (thousands separators, whitespace, percent, case).
- [ ] T017 [P] [US3] Add `test_tool_arg_matcher.py::test_regex_matrix` parameterized over rows 12–15 (`fullmatch` anchoring, backslash escaping).
- [ ] T018 [P] [US3] Add `test_tool_arg_matcher.py::test_multi_call_any_wins` row 23 — two `subtract` calls in one turn; second matches → assertion passes and `matched_call_index=1`.
- [ ] T019 [P] [US3] Add `test_tool_arg_matcher.py::test_none_value_semantics` — asserted literal `None` matches actual `None`; fuzzy compares `"none"`; regex compares `"None"` (contracts §5).
- [ ] T020 [P] [US3] Add `test_tool_arg_matcher.py::test_count_threshold` — `count=2`, only one satisfying call → fail; two satisfying calls → pass (lower-bound semantics per spec line 226).

### Implementation

- [ ] T021 [US3] Create `src/holodeck/lib/test_runner/tool_arg_matcher.py` with:
  - `match_literal(expected, actual) -> bool` per contracts/tool-arg-matchers.md §2.
  - `_normalize(s)` + `_try_parse_number(s)` helpers per §3. (First canonical number-parse helpers in the repo; US4 `NumericEvaluator` should reuse them.)
  - `match_fuzzy(pattern, actual) -> bool` per §3.
  - `match_regex(compiled, actual) -> bool` via `re.fullmatch(str(actual))` per §4.
  - `match_arg(matcher, actual) -> tuple[bool, str | None]` returning a `(matched, reason_if_not)` tuple for `arg_match_details`.
- [ ] T022 [US3] Add `find_matching_call(invocations, expected_tool) -> int` that iterates turn invocations, filters by case-sensitive substring name match (factor out `_tool_name_matches(expected: str, actual: str) -> bool` from `validate_tool_calls` so both code paths share one substring implementation), and returns the first index whose args satisfy every matcher (first-match-wins for reporting, order-independent for pass/fail). Returns `-1` when no match.
- [ ] T023 [US3] Implement `evaluate_expected_tools(expected: list[str | ExpectedTool], invocations: list[ToolInvocation]) -> tuple[bool, list[dict]]` returning `(matched, arg_match_details)` where each `details` entry has keys `expected_tool`, `args_asserted`, `matched_call_index`, `unmatched_reason` — exactly matching contracts/turn-result-schema.md §2.1. No new Pydantic types introduced; plain dicts flow directly into `TurnResult.arg_match_details`.

---

## Phase 4: Executor integration (TDD)

### Tests first

- [ ] T024 [P] [US3] Add `tests/unit/lib/test_runner/test_executor_arg_matching.py::test_fuzzy_regex_args_populate_arg_match_details` — a turn asserts a mixed literal+fuzzy+regex arg set; `TurnResult.arg_match_details` contains one entry per asserted tool with `matched_call_index` and `unmatched_reason`.
- [ ] T025 [P] [US3] Add `test_executor_arg_matching.py::test_missing_arg_fails_turn_with_reason` — asserted `a:206588`, agent omits `a` → `unmatched_reason="arg 'a' missing"`, `tools_matched=False`.
- [ ] T026 [P] [US3] Add `test_executor_arg_matching.py::test_extras_ignored` — agent passes an extra `rounding_mode="half-up"` not asserted → still passes (A4).
- [ ] T027 [P] [US3] Add `test_executor_arg_matching.py::test_name_matches_but_args_mismatch_fails` — `subtract` was called, but args differ → turn still fails (FR-011 "name AND args simultaneously").

### Implementation

- [ ] T027a [US3] In `src/holodeck/lib/test_runner/executor.py`, in both `_run_multi_turn` (US2 T020 call site) AND the legacy single-turn call site (`executor.py:691`), normalize `expected_tools` before dispatch: collect `list[str]` names from bare-string entries plus every `ExpectedTool(args=None, count==1)`, and pass to the existing `validate_tool_calls(tool_names, name_only_expected)` for the fast path. Route any `ExpectedTool` whose `args is not None` OR `count > 1` through `tool_arg_matcher.evaluate_expected_tools`. This resolves the US2↔US3 type-handoff: both call sites keep working after `expected_tools` widens to `list[str | ExpectedTool]`.
- [ ] T028 [US3] Use `tool_arg_matcher.evaluate_expected_tools` from T027a. Its return tuple `(matched, details)` feeds both `TurnResult.tools_matched` (combined with the name-only fast-path result via `matched and fast_path_matched`) and `TurnResult.arg_match_details`.
- [ ] T029 [US3] Populate `TurnResult.arg_match_details` with one entry per `ExpectedTool` that has `args` (contracts/turn-result-schema.md §2.1). `matched_call_index=-1` and a non-null `unmatched_reason` when no call satisfies all asserted args.
- [ ] T030 [US3] Ensure `TurnResult.errors` includes a concise summary line per failed assertion (e.g. `"expected subtract(a≈206588, b≈181001): no matching call"`), so the markdown reporter can surface it without inspecting `arg_match_details`.

---

## Phase 5: Reporter + end-to-end

### Tests first

- [ ] T031 [P] [US3] Add `tests/unit/lib/test_runner/test_reporter_multi_turn.py::test_arg_match_details_rendered_on_failure` — a failing turn's markdown row shows `expected subtract(a≈206588, b≈181001)` with the mismatch reason.
- [ ] T032 [P] [US3] Add `tests/integration/test_multi_turn_us3_e2e.py::test_convfinqa_arg_match_passes_and_fails` — uses `tests/fixtures/multi_turn/convfinqa_sample.yaml` (created in US1 T002, extended with ground truths in US2 T026), further extended with object-form `expected_tools` carrying fuzzy+regex args; scripted backend with two variants (match / mismatch); asserts full SC-006 acceptance end-to-end.

### Implementation

- [ ] T033 [US3] Extend `src/holodeck/lib/test_runner/reporter.py` markdown path to render `arg_match_details` under each failing turn.
- [ ] T034 [US3] Run `make format && make lint && make type-check && pytest -n auto`. All green. Confirm each of rows 1–23 in contracts §7 maps to a parameterized case in T015/T016/T017/T018 (no gaps); T019/T020 are additive coverage beyond the 23-row matrix.

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
