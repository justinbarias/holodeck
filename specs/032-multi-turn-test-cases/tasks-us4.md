---
description: "Tasks for User Story 4 — Custom and code-based evaluators (P2)"
---

# Tasks: User Story 4 — Built-in Deterministic + User-Supplied Code Evaluators (P2)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US4, §Complexity Tracking), data-model.md §5 (EvaluationMetric extensions) + §6 (CodeMetric) + §7a (MetricResult.kind), contracts/code-grader-contract.md.
**Approach**: Test-Driven Development.

**Story Goal**: Add two built-in deterministic evaluators (`equality`, `numeric`) implemented against the existing `BaseEvaluator` protocol (async `_evaluate_impl` returning `dict`, with `PARAM_SPEC`), plus a new top-level `CodeMetric` union variant for user-supplied Python graders referenced by import path `module:callable`. Graders are resolved at config load time (never runtime), receive a frozen `GraderContext`, and return a `GraderResult` (or shortcut). Exceptions in a grader fail only that turn's metric unless `fail_on_error=true`, in which case the executor raises a local `TestCaseFatal` that breaks the turn loop for that one test case.

**Independent Test**:
- `type: standard, metric: numeric, absolute_tolerance: 0.01` on `ground_truth: "0.14136"` vs agent response `"14.14%"` → passes.
- `type: code, grader: "my_benchmarks:numeric_equal"` → fixture grader passes on `"25587"`/`"25,587"`/`"25587.0"`, fails on `"25000"`.

**Depends on**: US1 foundational models — `Turn`, `TurnResult`, `TestResult.turns`, `MetricResult.kind='code'` Literal widening (already landed by US1 T016), and `TurnResult.grader_details` (already reserved by US1 T012). Independent of US2/US3, but naturally composes with them.

---

## Phase 1: Setup

- [ ] T001 Re-read contracts/code-grader-contract.md (all sections) and data-model.md §5–§7a. Confirm US1 T012 reserved `TurnResult.grader_details` and US1 T016 widened `MetricResult.kind` to include `"code"` (no schema widening needed in US4).
- [ ] T002 [P] Add `tests/fixtures/graders/` directory with a module `my_benchmarks.py` (matching the spec Independent Test path `my_benchmarks:numeric_equal` — do NOT use `convfinqa_fixture_graders` naming) exposing:
  - `numeric_equal(ctx) -> bool`
  - `raises_value_error(ctx) -> GraderResult` (always raises)
  - `returns_float(ctx) -> float` (returns `0.75`)
  - `returns_grader_result(ctx) -> GraderResult` (fully-formed result)
  - `returns_dict(ctx) -> dict` (returns a non-standard shape; covers T041)

  Use pytest's `pythonpath` in `pyproject.toml` (or `tests/conftest.py`) rather than a per-subdir `conftest.py` — this matches the existing repo convention (no `tests/fixtures/*/conftest.py` pattern exists today).

---

## Phase 2: Built-in deterministic evaluators (TDD)

### Tests first

- [ ] T003 [P] [US4] Write `tests/unit/lib/evaluators/test_deterministic.py::test_equality_strict_default` — `EqualityEvaluator` with all flags off: `"Yes"` vs `"yes"` → fail; `"Yes"` vs `"Yes"` → pass.
- [ ] T004 [P] [US4] Add `test_deterministic.py::test_equality_case_insensitive` — flag on → `"Yes"` vs `"yes."` still fails (trailing punctuation); add `strip_punctuation=true` → passes.
- [ ] T005 [P] [US4] Add `test_deterministic.py::test_equality_strip_whitespace` — `" hello  world "` vs `"hello world"` passes with `strip_whitespace=true`.
- [ ] T006 [P] [US4] Add `test_deterministic.py::test_numeric_default_tolerance` — no tolerances set → `abs_tol=1e-6`, `rel_tol=0.0`. Test three cases: diff = 1e-7 (`"1.0000001"` vs `"1"`) → pass (within tolerance); diff = 0.1 (`"1.1"` vs `"1"`) → fail; **boundary diff = 1e-6 (`"1.000001"` vs `"1"`) → pass** (FR-018 semantics are `abs(actual - expected) <= absolute_tolerance`; inclusive on the boundary).
- [ ] T007 [P] [US4] Add `test_deterministic.py::test_numeric_absolute_tolerance` — `abs_tol=0.01`, `"0.145"` vs `"0.14136"` → fail; `"0.142"` vs `"0.14136"` → pass.
- [ ] T008 [P] [US4] Add `test_deterministic.py::test_numeric_relative_tolerance` — `rel_tol=0.01`, actual within 1% of expected → pass.
- [ ] T009 [P] [US4] Add `test_deterministic.py::test_numeric_accept_percent` — `"14.14%"` parsed as `0.1414`; without flag → evaluator fails to parse and returns a dict with `passed=False` and an `error` key.
- [ ] T010 [P] [US4] Add `test_deterministic.py::test_numeric_accept_thousands_separators` — `"206,588"` parses only when flag is on.
- [ ] T011 [P] [US4] Add `test_deterministic.py::test_numeric_non_numeric_inputs` — `"abc"` vs `"5"` → dict with `passed=False` and `error` naming the parse failure; does not raise.
- [ ] T012 [P] [US4] Write `tests/unit/lib/test_runner/test_executor_deterministic_kind.py::test_equality_and_numeric_produce_metric_result_with_kind_standard` — run the executor over a minimal agent config with one `type: standard, metric: equality` metric; assert the produced `MetricResult.kind == "standard"` (data-model.md §7a) and `score in [0.0, 1.0]`. This is an **executor-level test**, not an evaluator-unit test (evaluators return dicts; the executor builds the `MetricResult` envelope at `executor.py:866-884`).
- [ ] T013 [P] [US4] Add 15-row acceptance matrix (SC-008) as a parameterized test in `test_deterministic.py` covering: exact strings, case/whitespace/punct, int vs float, percent, thousands, abs vs rel tolerance, **negative tolerance boundary** (rejected at model level — see T015), and at least one `diff == tolerance` boundary case → all produce expected pass/fail.

### Implementation

- [ ] T014 [US4] Create `src/holodeck/lib/evaluators/deterministic.py` with `EqualityEvaluator(BaseEvaluator)` and `NumericEvaluator(BaseEvaluator)`. Each class:
  - Declares `PARAM_SPEC: ClassVar[ParamSpec] = ParamSpec(required=frozenset({EvalParam.RESPONSE, EvalParam.GROUND_TRUTH}))` (mirrors `BLEUEvaluator` at `nlp_metrics.py:78-167`).
  - Implements `async def _evaluate_impl(self, **kwargs) -> dict[str, Any]` returning `{"equality": 0.0 | 1.0, "passed": bool, "error": str | None}` (or `{"numeric": ...}`) — keyed by metric name so `result.get(metric_name, ...)` at `executor.py:854` reads it correctly.
  - Never raises from `_evaluate_impl`; parse failures surface via the `error` key and `passed=False`.
  - Reuses `_normalize` / `_try_parse_number` helpers from US3's `tool_arg_matcher.py` when US3 is landed; if US3 is not yet merged, define private helpers in `deterministic.py` and leave a `# TODO(US3): migrate to shared helpers` marker. The `MetricResult` envelope is built by the executor, not by the evaluator.
- [ ] T015 [US4] Extend `src/holodeck/models/evaluation.py` `EvaluationMetric`:
  - Allow `metric: Literal["bleu","rouge","meteor","equality","numeric"]`. **Breaking-change caveat**: `EvaluationMetric.metric` is currently `str` (free-form). Narrowing to a `Literal` rejects any pre-existing non-listed name; audit existing fixtures/user configs for stragglers before merging. Add a migration note to the PR description.
  - Add `case_insensitive`, `strip_whitespace`, `strip_punctuation`, `absolute_tolerance`, `relative_tolerance`, `accept_percent`, `accept_thousands_separators` as optional fields with data-model.md §5 defaults.
  - Add a validator that: (a) logs a warning (does not error) if `model` is set on `equality`/`numeric` (no LLM is used); (b) rejects negative tolerances.
- [ ] T016 [US4] Extend `_create_evaluators` in `src/holodeck/lib/test_runner/executor.py` (currently lines ~386–513) with two new branches:
  ```python
  elif metric_name == "equality":
      self.evaluators["equality"] = EqualityEvaluator(
          case_insensitive=metric_config.case_insensitive,
          strip_whitespace=metric_config.strip_whitespace,
          strip_punctuation=metric_config.strip_punctuation,
      )
  elif metric_name == "numeric":
      self.evaluators["numeric"] = NumericEvaluator(
          absolute_tolerance=metric_config.absolute_tolerance,
          relative_tolerance=metric_config.relative_tolerance,
          accept_percent=metric_config.accept_percent,
          accept_thousands_separators=metric_config.accept_thousands_separators,
      )
  ```
  Evaluators register under their metric-name key so `result.get(metric_name, result.get("score", 0.0))` at `executor.py:854` finds the score. No new dispatch helper needed.

---

## Phase 3: (removed — `MetricResult.kind` widening moved to US1 T016)

Retained as a placeholder for task-ID continuity with previous revisions.

- [ ] T017 [US4] Verify-only — confirm US1 T016 landed the `MetricResult.kind` Literal extension and the `_metric_kind()` signature update (data-model.md §7a). No code change here; read `src/holodeck/models/test_result.py:46` and `executor.py:92-100` to confirm.

---

## Phase 4: `CodeMetric` + grader contract (TDD)

### Tests first

- [ ] T021 [P] [US4] Write `tests/unit/models/test_code_metric.py::test_grader_path_format` — `grader: "my_pkg.mod:fn"` parses; `"bad-path"` (missing `:`) raises `ConfigError`; double-colon raises.
- [ ] T022 [P] [US4] Add `test_code_metric.py::test_load_time_import_failure_is_config_error` — `grader: "nonexistent.module:fn"` raises `ConfigError` at `CodeMetric` construction with the test-case / turn-index / underlying `ModuleNotFoundError` message.
- [ ] T023 [P] [US4] Add `test_code_metric.py::test_load_time_attribute_error` — module exists but the callable doesn't → `ConfigError`.
- [ ] T024 [P] [US4] Add `test_code_metric.py::test_non_callable_rejected` — `grader` points to a module-level constant → `ConfigError`.
- [ ] T025 [P] [US4] Add `test_code_metric.py::test_resolved_callable_cached_on_instance` — use `mock.patch.object(importlib, "import_module", wraps=importlib.import_module)` to count invocations; assert `import_module` is called **exactly once** across 10 accesses of `metric.resolved_callable`. Avoids the `sys.modules` cache being the only thing exercised.
- [ ] T026 [P] [US4] Add `test_code_metric.py::test_defaults` — `enabled=True`, `fail_on_error=False`, `threshold=None`, `name` falls back to callable name.
- [ ] T027 [P] [US4] Add `test_code_metric.py::test_metric_type_union_accepts_code_variant` — `MetricType` discriminated union parses `{type: "code", grader: "mod:fn"}` as `CodeMetric`.

### Implementation

- [ ] T028 [US4] Add `CodeMetric` to `src/holodeck/models/evaluation.py` per data-model.md §6: discriminator `type: Literal["code"]`, `grader: str` validated against `^[\w.]+:[\w_]+$`, optional `threshold`, `enabled`, `fail_on_error`, `name`.
- [ ] T029 [US4] In `CodeMetric.model_validator(mode="after")`, call `importlib.import_module(module)`, `getattr(module, callable_name)`, assert callable, and cache the callable on a private attr. Raise `ConfigError` with the full context (test-case name, turn index if available, grader path, underlying exception class+msg) on any failure.
- [ ] T030 [US4] Update the `MetricType` union in `src/holodeck/models/evaluation.py` to `Annotated[Union[EvaluationMetric, GEvalMetric, RAGMetric, CodeMetric], Field(discriminator="type")]`.
- [ ] T031 [US4] Propagate the wider `MetricType` through `TestCaseModel.evaluations`, `Turn.evaluations`, agent-level `EvaluationConfig.metrics`, and `_metric_kind()` so every reader admits `CodeMetric` (no isinstance narrowing that drops it).
- [ ] T031a [US4] In `src/holodeck/lib/test_runner/executor.py::_create_evaluators` (~line 386-513), add an explicit `elif isinstance(metric_config, CodeMetric): continue` branch. Rationale: code graders are NOT part of `self.evaluators` (they run through `invoke_grader` at turn time, not through the evaluator pipeline). Inline-comment the branch pointing to T043 so future readers find the grader dispatch point. Without this branch, a `CodeMetric` flowing into `_create_evaluators` silently does nothing and the grader never runs — T047 would still surface it but closer-to-failure coverage is better.

---

## Phase 5: `GraderContext` / `GraderResult` + invocation harness (TDD)

### Tests first

- [ ] T032 [P] [US4] Write `tests/unit/lib/test_runner/test_code_grader.py::test_grader_context_is_frozen` — `GraderContext` cannot be mutated (`FrozenInstanceError` on assignment).
- [ ] T033 [P] [US4] Add `test_code_grader.py::test_grader_context_tuples_immutable` — `tool_invocations` is a tuple, `retrieval_context` is a tuple or `None`, not lists.
- [ ] T034 [P] [US4] Add `test_code_grader.py::test_invoke_with_grader_result` — `returns_grader_result` fixture → `MetricResult` carries score/passed/reason from the `GraderResult`.
- [ ] T035 [P] [US4] Add `test_code_grader.py::test_return_true_false_shortcuts` — `True` → `score=1.0, passed=True`; `False` → `score=0.0, passed=False` (contracts §3.2).
- [ ] T036 [P] [US4] Add `test_code_grader.py::test_return_float_threshold_derivation` — `returns_float=0.75` with `threshold=0.7` → `passed=True`; with `threshold=0.8` → `passed=False`; with `threshold=None` → default gate `>= 0.5` (contracts §5).
- [ ] T037 [P] [US4] Add `test_code_grader.py::test_exception_captured_only_that_turn` — `raises_value_error` grader runs mid-turn, that turn's `MetricResult.error` captures `"ValueError: ..."` and `passed=False`; other turns still execute.
- [ ] T038 [P] [US4] Add `test_code_grader.py::test_fail_on_error_halts_test_case` — `fail_on_error=true` on the raising grader → remaining turns not executed for this test case (via `TestCaseFatal`); other test cases continue.
- [ ] T039 [P] [US4] Add `test_code_grader.py::test_grader_receives_expected_context_fields` — asserts `ctx.turn_input`, `agent_response`, `ground_truth`, `tool_invocations`, `retrieval_context`, `turn_index`, `test_case_name`, `turn_config` are all populated.
- [ ] T040 [P] [US4] Add `test_code_grader.py::test_details_preserved_on_turn_result` — grader returns `GraderResult(details={"foo":"bar"})`; `TurnResult.grader_details["<metric_name>"] == {"foo":"bar"}` (contracts §7).
- [ ] T040a [P] [US4] Add `test_code_grader.py::test_details_must_be_json_safe` — grader returns `GraderResult(details={"bad": object()})`; the resulting `MetricResult.error` names `"details not JSON-serializable"` and the grader is recorded as failing before the report is written. Fails fast at grader-return time rather than at report-emission time.
- [ ] T041 [P] [US4] Add `test_code_grader.py::test_non_standard_return_treated_as_error` — `returns_dict` fixture grader → recorded as grader error, not as success (contracts §3.2 last row).

### Implementation

- [ ] T042 [US4] Create `src/holodeck/lib/test_runner/code_grader.py` with:
  - `@dataclass(frozen=True) class GraderContext` per contracts §3.1 (tuples for ordered collections).
  - `@dataclass(frozen=True) class GraderResult` per §3.2.
  - `invoke_grader(fn, ctx, threshold) -> MetricResult` that normalizes shortcut returns, wraps the call in `try/except Exception`, times with `time.perf_counter` (matches DeepEval evaluators at `deepeval/base.py:243,320`; emits int milliseconds via `int((perf_counter() - start) * 1000)`), validates `details` is JSON-serializable via `json.dumps(..., default=None)` catching `TypeError`, and returns `MetricResult(kind="code", metric_name=..., score, threshold, passed, error, evaluation_time_ms, model_used=None, reasoning=result.reason)`.
  - An exported `build_grader_context(turn_config, turn_state) -> GraderContext` helper the executor calls per turn.
- [ ] T042a [US4] Define `class TestCaseFatal(Exception)` inside `src/holodeck/lib/test_runner/executor.py` (executor-local, not exported). Carry attributes `test_case_name`, `turn_index`, `grader_name`. Docstring: "Raised when a CodeMetric grader with `fail_on_error=True` raises; breaks the turn loop for this test case only. Other test cases continue. Caught at the per-test-case level in `_execute_single_test`." One-line unit test asserting it inherits from `Exception`.
- [ ] T043 [US4] In `src/holodeck/lib/test_runner/executor.py`, during per-turn metric evaluation, detect `CodeMetric` entries and route to `invoke_grader`. Respect `fail_on_error=True` by raising `TestCaseFatal(test_case_name, turn_index, grader_name)` to break the turn loop for that case only; other test cases still run. Catch it at the per-test-case boundary, mark the test case failed with an explanatory error, and move on.
- [ ] T044 [US4] Confirm `TurnResult.grader_details` is already reserved from US1 T012 (per data-model.md §8). In `src/holodeck/lib/test_runner/executor.py`, populate it from `GraderResult.details` keyed by `metric_name` after each grader invocation. Add a regression test that `grader_details is None` when no code graders ran.

---

## Phase 6: Three-level configuration + end-to-end

### Tests first

- [ ] T045 [P] [US4] Write `tests/unit/config/test_evaluator_three_level.py::test_code_metric_usable_at_all_three_levels` — same `CodeMetric` attached at agent `evaluations.metrics`, at test-case `evaluations`, or at per-turn `evaluations` — all produce identical behavior (FR-023).
- [ ] T046 [P] [US4] Add `test_evaluator_three_level.py::test_equality_numeric_usable_at_all_three_levels` — same for the two new deterministic metrics.
- [ ] T047 [P] [US4] Add `tests/integration/test_multi_turn_us4_e2e.py::test_convfinqa_numeric_plus_code_grader` — 1-turn multi-turn case with `numeric` built-in + fixture `code` grader (`my_benchmarks:numeric_equal`); stubbed agent returns `"25587"`; assert both `MetricResult`s land in `TurnResult.metric_results` with correct kinds (`"standard"` and `"code"`), and roll-up applies.

### Implementation

- [ ] T048 [US4] Confirm the three-level resolution in `src/holodeck/lib/test_runner/executor.py` passes new metric types through unchanged (no evaluator-kind gating). Add a small regression guard if gating exists.
- [ ] T049 [US4] Update `src/holodeck/lib/test_runner/reporter.py` markdown to render `kind: code` metrics with their reason (if present) and their `grader` callable name.
- [ ] T050 [US4] Run `make format && make lint && make type-check && pytest -n auto`. All green.

---

## Phase 7: Docs + constitution-exception verification

- [ ] T051 [P] [US4] Update `specs/032-multi-turn-test-cases/quickstart.md` §5 to pin the grader module name to `my_benchmarks:program_equivalence` (matches the fixture path in T002). Quickstart currently already uses that shape — verify no drift.
- [ ] T052 [P] [US4] Confirm the `type: code` section of `contracts/code-grader-contract.md` carries the trust-boundary note (FR-026, A9). PR description MUST link to `spec.md §Complexity Tracking` per reviewer sign-off requirement.
- [ ] T052a [US4] Run `grep -rn "exec\\|eval(\\|compile(" src/holodeck/lib/test_runner/code_grader.py src/holodeck/models/evaluation.py` and confirm the output is empty. Locks in that the Principle I exception does not leak dynamic-code execution beyond `importlib.import_module`.

---

## Dependencies

- Phase 1 → Phase 2 and Phase 4 in parallel.
- Phase 5 depends on Phase 4.
- Phase 6 depends on Phases 2, 4, 5.
- Phase 7 is docs/verification, runs any time after Phase 5.
- Within Phase 2: T003–T013 all parallel; T014–T016 serial (single evaluator module + `executor.py`).
- Within Phase 4: T021–T027 parallel; T028–T031a serial (same model file + `executor.py`).
- Within Phase 5: T032–T041 parallel; T042–T044 serial.

## Independent test criteria (recap)

- SC-007 — grader author can wire up a working code grader in <30 min: validated by `tests/fixtures/graders/my_benchmarks.py` + T034–T040.
- SC-008 — 15-pair acceptance matrix for `equality` + `numeric` is T013 (includes boundary `diff == tolerance`).
- SC-009 — grader exception only fails that turn: T037/T038.
- Constitution exception stays scoped to `CodeMetric` + `code_grader.py`: T052a grep-verified.
