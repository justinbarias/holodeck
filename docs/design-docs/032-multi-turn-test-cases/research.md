# Phase 0 Research: Multi-Turn Test Cases

**Feature**: 032-multi-turn-test-cases
**Date**: 2026-04-20

All five spec-level clarifications (Session 2026-04-20) are already resolved in `spec.md`. This document records the *implementation-level* decisions that flow from them, plus research into the reference dataset and the backend APIs we depend on.

---

## 1. ConvFinQA dialogue shape (reference corpus)

**Decision**: The reference dataset confirms the spec's multi-turn shape. No dataset-specific built-ins are added to HoloDeck; the built-in set stays generic (`equality`, `numeric`).

**Investigated**: `/Users/justinbarias/Documents/Git/python/justinbarias/data/convfinqa_dataset.json` (3037 train + 421 dev examples).

Each record has:

```json
{
  "id": "Single_JKHY/2009/page_28.pdf-3",
  "doc": { "pre_text": "...", "post_text": "...", "table_ori": [...] },
  "dialogue": {
    "conv_questions": ["what is the net cash from operating activities in 2009?", "what about in 2008?", "what is the difference?", "what percentage change does this represent?"],
    "conv_answers":   ["206588", "181001", "25587", "14.1%"],
    "turn_program":   ["206588", "181001", "subtract(206588, 181001)", "subtract(206588, 181001), divide(#0, 181001)"],
    "executed_answers": [206588.0, 181001.0, 25587.0, 0.14136]
  },
  "features": { "num_dialogue_turns": 4 }
}
```

**Implications** (all already satisfied by the spec):

- `conv_questions` → `turns[*].input` (ordered, stateful — turn 2's "what about in 2008?" is anaphoric on turn 1).
- `conv_answers` → `turns[*].ground_truth` (scalar, not prose — BLEU/ROUGE are bad proxies; `numeric` is the natural evaluator).
- `turn_program` → grading target for a user-supplied `type: code` grader. `#0` back-references make this a DAG, which is why no built-in can express it (spec §Complexity Tracking).
- `executed_answers` → numeric ground-truth target for the `numeric` built-in evaluator.

**Alternatives considered**: Ship a ConvFinQA importer or `turn_program` grader. **Rejected** per Assumption A10 — keeps HoloDeck dataset-agnostic; the grader lives in the user's benchmark repo (code-grader contract + quickstart show how).

---

## 2. Backend multi-turn API

**Decision**: Drive all multi-turn turns through `AgentBackend.create_session()` + `AgentSession.send()` (per FR-006). The existing protocols already support this; no backend changes are required.

**Rationale**:
- `AgentSession.send()` is defined in `src/holodeck/lib/backends/base.py` and implemented by both `SKSession` (`src/holodeck/lib/backends/sk_backend.py`) and `ClaudeSession` (`src/holodeck/lib/backends/claude_backend.py`). Both maintain conversation state between calls — `SKSession` via the underlying SK `AgentThreadRun` / `ChatHistory`; `ClaudeSession` via the SDK's conversation id.
- `ExecutionResult.token_usage` is already populated per call on both sessions (feature 031 contract), so per-turn accounting is automatic.
- Single-turn legacy test cases continue to use `invoke_once()` — keeps the session cost off tests that don't need it.

**Alternatives considered**:
- Concatenate all turns into one prompt and call `invoke_once()`. **Rejected**: would lose stateful context semantics; the agent would see the whole dialogue as a single user message rather than a back-and-forth (breaks Acceptance Scenario 1.2 — "turn 2's prompt is sent after turn 1's response is returned").
- Build a thin wrapper that manages history manually per turn. **Rejected**: reinvents what `AgentSession` already provides, risks divergence between SK and Claude history conventions, and duplicates token accounting.

**Error recovery**: Per FR-008, a per-turn `BackendSessionError` / `BackendTimeoutError` / generic exception fails *that* turn; the executor attempts the next `send()` on the same session. If the session is unrecoverable (e.g. the SDK subprocess died), remaining turns are marked **skipped** and the test case fails with a clear reason. Detection: if two consecutive `send()` calls raise `BackendSessionError`, treat the session as unrecoverable.

---

## 3. Argument matcher design

**Decision**: Three matcher kinds backed by Python stdlib only — no `fuzzywuzzy`. Primary implementation uses normalization-based comparison (`re`, `math`); `difflib.SequenceMatcher` is retained as a documented fallback should a future tolerance profile need edit-distance (not used in v1).

| Matcher | Implementation | Pre-comparison normalization |
|---------|----------------|------------------------------|
| **exact (literal)** | `actual == expected` after numeric coercion | `int` ↔ `float` equivalence (`206588 == 206588.0` passes); type-aware for bool/None; no string stripping |
| **fuzzy** | `_fuzzy_eq(actual, expected)` | Both coerced to string, lowercased, whitespace-collapsed, thousands separators (`,`, `_`, Unicode narrow NBSP) stripped, trailing percent suffix recognized (`"14.14%"` ↔ `0.1414`), trailing `.0+` trimmed; then compared as numbers if both parse, else as strings |
| **regex** | `re.fullmatch(pattern, str(actual))` | Pattern compiled at load time (FR-025); full match only (anchored) |

**Rationale**:
- Stdlib-only keeps the dependency footprint flat (principle: no new runtime deps without reason).
- Explicit normalization rules are easier to document (FR-013) and test (SC-006) than a black-box library.
- `re.fullmatch` gives anchored semantics without requiring users to write `^…$` (FR-014).

**Alternatives considered**:
- `rapidfuzz` for Levenshtein-based fuzzy matching. **Rejected**: introduces a compiled C dep for a feature (thousands-separator tolerance, numeric equivalence) that doesn't actually need edit-distance.
- Use `difflib.SequenceMatcher`. **Not used in v1** for the same reason — the spec's fuzzy semantics (FR-013) are normalization-based, not distance-based. Kept as a documented stdlib fallback should a future benchmark require true edit-distance tolerance; this is an additive extension, not a v1 code path.

---

## 4. Evaluator surface (`equality`, `numeric`, `code`)

**Decision** (matches Session 2026-04-20 clarification #1): `equality` and `numeric` extend `EvaluationMetric.metric`; `CodeMetric` is a new top-level variant in the `MetricType` discriminated union.

**Rationale**:
- `equality` and `numeric` operate on the same `(actual_output, expected_output)` pair as BLEU/ROUGE/METEOR — same shape, same dispatch site in `_create_evaluators`, same three-level model-override semantics (none of the three actually need an LLM; model override is simply ignored). Adding them as new values of `EvaluationMetric.metric` is the minimum-invasive change.
- `code` is shape-incompatible: it carries a grader import path instead of a metric name, and it receives the *full per-turn context* (tool invocations, retrieval context, turn index, test-case name), not just two strings. Forcing it onto `EvaluationMetric` would pollute that model with a bunch of optional-and-mutually-exclusive fields. A new top-level variant keeps each shape cohesive.

**`EvaluationMetric.metric` extension** (FR-018):

```yaml
- type: standard
  metric: equality
  case_insensitive: false        # default
  strip_whitespace: false        # default
  strip_punctuation: false       # default
  threshold: 1.0                 # equality is binary; threshold 1.0 effectively means "must match"

- type: standard
  metric: numeric
  absolute_tolerance: 1e-6       # default (per clarification #5)
  relative_tolerance: 0.0        # default
  accept_percent: false          # default — when true, "14%" parses as 0.14
  accept_thousands_separators: false
```

**`CodeMetric` new variant** (FR-019):

```yaml
- type: code
  grader: "my_benchmarks.convfinqa:program_equivalence"
  threshold: 1.0                 # optional; defaults to grader-reported passed/failed
  enabled: true
```

The grader callable is resolved at configuration **load time** via `importlib.import_module` + `getattr` — not first use — so typos surface before any agent runs (FR-025).

**Alternatives considered**:
- Make `code` a metric-name on `EvaluationMetric`. **Rejected**: `EvaluationMetric.metric` is a plain string — there's no place to put `grader: "path:func"` without bolting an orthogonal field on.
- Subtype `EvaluationMetric` for `equality`/`numeric` as separate discriminator variants (`type: equality`, `type: numeric`). **Rejected**: adds two new dispatch branches where one metric-name switch already exists; also makes the three-level model-override logic (which is keyed on `type`) rewire itself.

---

## 5. Code-grader contract

**Decision**: Graders are plain sync callables (Python functions or any callable class). The grader receives a single immutable `GraderContext` dataclass and returns a `GraderResult` dataclass. Exceptions fail *that* turn only and are captured as `MetricResult.error`.

**`GraderContext` fields** (read-only via `@dataclass(frozen=True)`):

```python
turn_input: str
agent_response: str
ground_truth: str | None
tool_invocations: list[ToolInvocation]    # ordered, as captured on the turn
retrieval_context: list[str] | None
turn_index: int                            # 0-based
test_case_name: str | None
turn_config: dict[str, Any]               # raw per-turn config, for grader-specific keys
```

**`GraderResult` shape** (FR-021):

```python
score: float           # in [0.0, 1.0]
passed: bool | None = None    # if None, derived from score >= threshold (default 0.5 if no threshold)
reason: str | None = None
details: dict[str, Any] | None = None
```

Graders may alternatively return a bare `bool` (auto-lifted to `score=1.0, passed=True` or `score=0.0, passed=False`) or a bare `float` (auto-lifted to `score=float`, `passed=score >= threshold`).

**Load-time resolution** (FR-019):
- `grader: "module.submodule:callable_name"` format.
- `module.submodule` imported via `importlib.import_module`.
- `callable_name` looked up via `getattr`; must be callable.
- Non-import, non-getattr, or non-callable → `ConfigError` at load time with path `test_cases[i].evaluations[j].grader` and the reason.

**Trust model** (A9 + FR-026): graders run in-process, no sandbox. Documentation (FR-026) states this clearly and warns against loading from untrusted sources.

**Alternatives considered**:
- Async graders. **Rejected for v1**: none of the built-in deterministic evaluators are async; forcing users to write `async def` for deterministic checks is friction. We can add async support if a future benchmark needs network I/O inside the grader.
- Sandbox via subprocess. **Rejected**: spec §Out of Scope; HoloDeck already runs user Python (tool plugins, MCP servers) in-process at the same trust level.

---

## 6. Parallel test-case orchestration

**Decision** (per clarification #4): Gate the outer test-case loop behind an `asyncio.Semaphore(parallel_test_cases)`. Within a test case, turns remain strictly sequential.

**Rationale**:
- Each multi-turn test case uses its own `AgentSession` (FR-006) — sessions don't share state, so across-test-case concurrency is safe.
- `asyncio.Semaphore` is the stdlib-native primitive, already used elsewhere (context generator batching).
- Preserving turn-sequentiality inside a test case is required for conversational state to make sense (turn 2 depends on turn 1's response).

**Reporter writes**: The reporter already collects `TestResult` instances in order via a progress callback; when concurrent, results come in out-of-order. Each concurrent task carries its **original list position** (0-based index into `self.agent_config.test_cases`) in its scheduling tuple; the final report sorts by this numeric index before emitting. This does **not** depend on test-case `name` — duplicate or missing names are tolerated (same as today's contract) and the report is deterministic regardless of completion order.

**CLI surface**: `holodeck test --parallel-test-cases N` (with `-p N` short alias) — resolved through the existing CLI > YAML > env > defaults chain per FR-009a.

**Alternatives considered**:
- Thread pool. **Rejected**: the whole runtime is async; mixing threads with async backends invites deadlocks.
- Process pool. **Rejected**: each worker would need its own backend initialization (subprocess spawn per worker for Claude), eating any parallelism win.

---

## 7. Per-turn metric roll-up

**Decision** (per clarification #2): Each unique `metric_name` appears exactly once at the test-case level with `score = mean(turn_scores)` and `passed = all(turn_passed)`. Turns that skipped a metric (no `ground_truth`, or metric N/A) do not contribute to the mean.

**Reporter contract preservation**:
- `TestResult.metric_results` is still a flat list of `MetricResult` — existing readers (dashboard Compare view, CLI summary) see no shape change.
- Each `MetricResult` at the rolled-up level carries the element-wise mean of per-turn scores and the conjunction of per-turn pass flags.
- Per-turn scores also appear on `TurnResult.metric_results` for users who want the per-turn breakdown.

**Edge cases**:
- Every turn skipped a metric → the metric does not appear in `TestResult.metric_results` at all (prevents `mean([])` = 0 from masquerading as a real 0-score).
- Multi-turn test case where no turn has a `ground_truth` → `TestResult.metric_results == []`, `passed` depends solely on tool assertions and errors.

---

## 8. Dashboard rendering (feature 031)

**Decision**: The JSON report schema gains one optional field (`TestResult.turns`), and two concrete dashboard files are modified so the new shape renders as expandable rows.

**JSON contract**: `TestResult.turns: list[TurnResult] | None` — `None` or absent for single-turn legacy runs; populated for multi-turn. Dashboard detects "multi-turn" by `len(turns) > 0` and shows a disclosure row per turn. The existing Explorer view for `tool_invocations` works turn-by-turn because each `TurnResult.tool_invocations` is the same shape as `TestResult.tool_invocations`.

**Touchpoints** (tracked in plan.md §Project Structure):
- `src/holodeck/dashboard/explorer_data.py` — detect `TestResult.turns` and emit per-turn rows in the explorer payload.
- `src/holodeck/dashboard/components/` — add a per-turn disclosure component; single-turn rendering path untouched.

**Token-usage contract update (feature 031 cross-cut)**: `TokenUsage` gains two additive fields — `cache_creation_tokens: int = 0` and `cache_read_tokens: int = 0` — with the `total_tokens` validator updated to allow `total >= prompt + completion` (cache reads do not double-count prompt tokens at all providers). The element-wise sum across turns then includes these fields. This is an additive schema change; legacy JSON without `cache_*` deserializes with zeroes. The PR must link to spec §Complexity Tracking *and* note this expansion for reviewer acknowledgement.

**Back-compat**: Existing runs that don't have `turns` or `cache_*` in their JSON continue to load into the new models (all new fields optional with safe defaults). Dashboard conditional on presence of `turns`.

---

## Exit Criteria

- [x] All spec NEEDS CLARIFICATION resolved (none exist — pre-resolved in spec §Clarifications).
- [x] Each FR-001…FR-026 has a clearly identified implementation home (mapped in plan.md §Project Structure).
- [x] No new runtime dependencies introduced.
- [x] Constitution re-check: PASS (code-grader exception scoped and justified).
