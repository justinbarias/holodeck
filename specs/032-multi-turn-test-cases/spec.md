# Feature Specification: Multi-Turn Test Cases, Per-Turn Assertions, Tool-Arg Matchers, and Programmatic Evaluators

**Feature Branch**: `032-multi-turn-test-cases`
**Created**: 2026-04-19
**Status**: Draft
**Input**: User description: "Analyze the convfinqa dataset and extend the test executor to support multi-turn test cases, per-turn ground truths, per-turn expected tools, and expected tool call args with fuzzy and/or regex matching."

## Context

The user pointed to `convfinqa_dataset.json` as the motivating corpus. Each example in that corpus is a short conversation between a user and an analyst agent over a single financial document (SEC-style pre-text, post-text, and a numeric table). Every example contains:

- An ordered list of **turn questions** (typically 3–5 turns).
- A corresponding list of **ground-truth answers** (one per turn).
- A **turn program** per turn — the symbolic tool calls that derive each answer (e.g., `subtract(206588, 181001)` on turn 3, `subtract(206588, 181001), divide(#0, 181001)` on turn 4). Later turns reference earlier ones via `#N` placeholders, so the conversation is **stateful**.

HoloDeck's current `TestCaseModel` (`src/holodeck/models/test_case.py`) supports a single `input`, a single `ground_truth`, and a flat `expected_tools: list[str]` that only checks names. It cannot express the ConvFinQA shape or any other benchmark where each turn has its own success criteria.

This feature extends HoloDeck so users can define multi-turn test cases, with assertions per turn on answers, on which tools the agent must call, and on the arguments those tools are called with (using exact, fuzzy, or regex matching).

## Clarifications

### Session 2026-04-20

- Q: Where should the new evaluator types (`equality`, `numeric`, `code`) live in the `MetricType` discriminated union in `src/holodeck/models/evaluation.py`? → A: Hybrid — `equality` and `numeric` extend `EvaluationMetric.metric` (they operate on two strings like NLP metrics); `code` becomes a new top-level `CodeMetric` variant with its own `type` literal (it has a distinct `grader` field and bypasses the `metric`-name pattern).
- Q: How should `TestResult.metric_results` roll up per-turn metric scores for multi-turn test cases? → A: Per-metric average across turns — each `metric_name` appears once at the test-case level with `score = mean(turn_scores)` and `passed = all(turn_passed)`.
- Q: How does the existing `llm_timeout` config apply to multi-turn test cases? → A: Per-turn — each `AgentSession.send()` is bounded by `llm_timeout` independently; exceeding it fails that turn and execution continues with the next turn (per FR-008). No whole-test-case wall-clock timeout is introduced.
- Q: Can multiple multi-turn test cases run in parallel? → A: Introduce a new `parallel_test_cases: int` field on `ExecutionConfig` (default 1 — sequential). When >1, the executor runs up to N test cases concurrently, each in its own `AgentSession`. Within a single multi-turn test case, turns MUST remain strictly sequential (turn N+1 only after turn N completes) to preserve conversational state.
- Q: What are the default tolerances for the `numeric` evaluator when the user does not set either `absolute_tolerance` or `relative_tolerance`? → A: `absolute_tolerance = 1e-6`, `relative_tolerance = 0.0`. Pass when `abs(actual - expected) <= absolute_tolerance`. Catches float-roundtrip noise while failing on substantively wrong answers.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Multi-Turn Conversation Against an Agent (Priority: P1)

A benchmark author loads the ConvFinQA corpus into an `agent.yaml` as test cases. Each test case has an ordered list of turns. `holodeck test` feeds the turns to the agent as a **single stateful conversation** (one session, one shared context), captures the agent response at each turn, and reports the outcome **per turn** as well as a rolled-up pass/fail for the whole test case.

**Why this priority**: Without stateful multi-turn execution, none of the other user stories have a place to plug in. This is the foundation that unlocks conversational benchmarks for HoloDeck. It is also the smallest viable slice: even without tool-argument assertions, a user gets immediate value by being able to run any conversational benchmark.

**Independent Test**: Define a test case with 3 turns and no tool assertions. Run `holodeck test`. Verify: (a) the agent's turn-2 response demonstrates awareness of turn-1 (e.g., resolves an anaphoric reference like "and in 2008?"); (b) the report lists one row per turn; (c) a single failing turn fails the test case as a whole.

**Acceptance Scenarios**:

1. **Given** a test case with `turns: [{input: A}, {input: B}, {input: C}]` and no ground truths, **When** `holodeck test` runs, **Then** the agent receives all three prompts in the same session, and the report includes the response for each turn.
2. **Given** a multi-turn test case where turn 2 asks "and what about 2008?" (relying on turn 1's context), **When** the agent is invoked, **Then** turn 2's prompt is sent after turn 1's response is returned (not concatenated, not independently).
3. **Given** a traditional single-turn test case (uses the legacy flat `input` / `ground_truth` shape), **When** `holodeck test` runs, **Then** the test case executes unchanged and produces the same report it would have before this feature — no migration required.
4. **Given** any turn within a multi-turn test case errors (timeout, backend error), **When** the executor catches the error, **Then** the failing turn is marked failed and the remaining turns still execute and report against the surviving session state, unless the backend session itself is unrecoverable.

---

### User Story 2 - Assert Per-Turn Ground Truths and Expected Tools (Priority: P1)

A benchmark author wants to know, for each turn, whether the agent produced the right answer and called the right tools. The author attaches a `ground_truth` and an `expected_tools` list **to each turn individually**. The executor evaluates the turn response against the turn's ground truth using the same metric stack (BLEU, ROUGE, G-Eval, etc.) already configured for the test case, and verifies the agent called every tool named in `expected_tools` on that turn.

**Why this priority**: This is what makes the multi-turn runs *measurable*. Without per-turn ground truth and per-turn tool expectations, turn-level assertions collapse back to a single aggregate judgement, defeating the purpose of running a conversational benchmark. Ships together with Story 1 as the P1 slice.

**Independent Test**: Author a test case where turn 3 expects ground truth `"25587"` and tools `["subtract"]`. Run `holodeck test`. Verify the report shows turn 3 passing on metric match and on tool-name match, and fails if the agent answers `"25000"` or doesn't call `subtract`.

**Acceptance Scenarios**:

1. **Given** a turn with `ground_truth: "25587"` and a configured BLEU metric threshold, **When** the agent's turn response is evaluated, **Then** the metric result is recorded against that specific turn in the report.
2. **Given** a turn with `expected_tools: ["subtract", "divide"]`, **When** the agent calls `subtract` and `divide` (in any order, possibly alongside other tools), **Then** the turn passes the tool-name assertion.
3. **Given** a turn with `expected_tools: ["divide"]` where the agent never calls `divide` on that turn, **When** the turn is evaluated, **Then** the turn fails the tool-name assertion regardless of whether `divide` was called on an earlier or later turn.
4. **Given** a test case where turns 1 and 2 pass but turn 3 fails on ground-truth comparison, **When** the report is generated, **Then** the test case is marked failed and the per-turn breakdown pinpoints turn 3 as the failing turn.

---

### User Story 3 - Assert Expected Tool Call Arguments with Fuzzy or Regex Matching (Priority: P2)

A benchmark author needs to verify not just *which* tool was called but *how* it was called. For the ConvFinQA example whose turn 3 program is `subtract(206588, 181001)`, the author wants to assert that `subtract` was called with the two specific operand values. Because LLMs often produce slight variations (`206588.0` vs `206588`, `"206,588"` vs `"206588"`, extra whitespace, numeric precision drift), the author can specify each expected argument as one of:

- A literal value (exact match after light normalization),
- A **fuzzy** matcher (case-insensitive substring, numeric tolerance, or whitespace-insensitive equality),
- A **regex** matcher (full-match against the argument rendered as a string).

Matchers are specified per argument name. Extra, unasserted arguments do not cause failure. A missing asserted argument does.

**Why this priority**: Tool-argument assertions add meaningful rigor for benchmarks that care about *how* the agent reasons, not just which functions it named. Defers behind P1 because a benchmark is still runnable and broadly informative without it — but without this, the ConvFinQA use case specifically loses the ability to verify the agent is operating on the right numbers.

**Independent Test**: Author a turn with expected tool `subtract` and expected args `{a: {fuzzy: "206588"}, b: {regex: "^181001(\\.0+)?$"}}`. Verify the turn passes when the agent calls `subtract(a=206588.0, b=181001)` and fails when the agent calls `subtract(a=206588, b=180000)`.

**Acceptance Scenarios**:

1. **Given** a turn with `expected_tools: [{name: "subtract", args: {a: 206588, b: 181001}}]` (literal values), **When** the agent calls `subtract(a=206588.0, b=181001)`, **Then** the turn passes (numeric equivalence normalization applies).
2. **Given** a turn with `expected_tools: [{name: "lookup", args: {cell: {regex: "^2009.*cash.*$"}}}]`, **When** the agent calls `lookup(cell="2009 net cash from operating activities")`, **Then** the turn passes because the argument string satisfies the regex.
3. **Given** a turn with `expected_tools: [{name: "subtract", args: {a: {fuzzy: "206588"}}}]`, **When** the agent calls `subtract(a="206,588", b=181001)`, **Then** the turn passes (fuzzy matcher tolerates thousands separators and stringification).
4. **Given** a turn asserting `args: {a: 206588}`, **When** the agent calls `subtract(a=206588, b=181001, rounding_mode="half-up")`, **Then** the turn passes — extra unasserted arguments do not fail the assertion.
5. **Given** a turn asserting an argument value, **When** the agent calls the tool but omits that argument entirely, **Then** the turn fails with a clear message identifying the missing argument.
6. **Given** a turn that asserts tool `subtract` was called N times (default N=1) with specific args, **When** the agent calls `subtract` multiple times in the turn, **Then** the turn passes if **any** call in that turn matches the asserted args (order-independent, first-match wins).

---

### User Story 4 - Custom and Code-Based Evaluators (Priority: P2)

Standard NLP metrics (BLEU, ROUGE, METEOR) and LLM-as-judge (G-Eval) are the wrong tool when the success criterion is deterministic — e.g., "the agent's final answer must equal `25587` within numeric tolerance" or "the agent's tool-call graph must be equivalent to the `turn_program` `subtract(206588, 181001), divide(#0, 181001)`". BLEU on a number is a bad proxy; LLM judges are expensive and non-deterministic. For these cases, a benchmark author wants to plug in **programmatic evaluators** — either a handful of built-in deterministic checks or a user-supplied Python function that receives the full turn context and returns a score and pass/fail.

Built-in programmatic evaluators include at minimum:

- `equality` — string equality after configurable normalization (case, whitespace, punctuation).
- `numeric` — parse actual vs expected as numbers and compare with absolute / relative tolerance; optionally tolerant of thousands separators and percent suffixes.

User-supplied evaluators are referenced by import path (e.g., `type: code`, `grader: my_benchmarks.convfinqa:grade_program`). The callable receives a well-defined context object (turn input, agent response, ground truth, ordered tool invocations, retrieval context, test-case metadata) and returns a standard result shape (score, passed, optional reason, optional details). The author controls what the grader checks — equality, program-tree equivalence, domain-specific rubrics, whatever.

**Why this priority**: For ConvFinQA and any other benchmark whose success criteria are deterministic, this unblocks useful measurement that the current metric stack cannot express honestly. It ships after P1 because multi-turn execution is the prerequisite — there's nothing to grade until turns exist. Paired with Story 3 (P2) because together they give a benchmark author full control over what "pass" means per turn.

**Independent Test**: Register a code grader `my_benchmarks:numeric_equal` that returns `passed=True` when `abs(float(actual) - float(expected)) < 0.01`. Attach it to a turn with `ground_truth: "25587"`. Verify the turn passes when the agent answers `"25587"`, `"25,587"`, or `"25587.0"`, and fails when it answers `"25000"`.

**Acceptance Scenarios**:

1. **Given** a turn with `evaluations: [{type: equality, case_insensitive: true}]` and `ground_truth: "Yes"`, **When** the agent responds `"yes."`, **Then** the turn passes the equality evaluator (after normalization) and fails without `case_insensitive`.
2. **Given** a turn with `evaluations: [{type: numeric, absolute_tolerance: 0.01}]` and `ground_truth: "0.14136"`, **When** the agent responds `"14.14%"`, **Then** the numeric evaluator parses both (with percent handling), compares within tolerance, and passes.
3. **Given** a turn with `evaluations: [{type: code, grader: "convfinqa.graders:program_equivalence"}]`, **When** the executor invokes the turn, **Then** the grader is imported once, called with the full turn context including `tool_invocations`, and its returned score/passed are recorded as a regular metric result in the report.
4. **Given** a grader that raises an exception, **When** the turn is evaluated, **Then** the turn is marked failed with an error-reason attribution that names the grader and its exception, and subsequent turns continue to execute.
5. **Given** a grader reference that cannot be imported (bad path, missing module), **When** the configuration is loaded, **Then** loading fails with a clear error naming the test case, turn index, and grader path — never at runtime.
6. **Given** a code grader, **When** it runs, **Then** it has no ability to mutate the agent's session or future turns — its return value is the only channel back into the report.

---

### User Story 5 - Report Per-Turn Results in the Test Dashboard (Priority: P3)

The eval-run dashboard (feature 031) and the CLI report both render test-case results. After this feature ships, each multi-turn test case appears as a parent row with expandable per-turn children showing input, response, matched/missed tools, matched/failed arg assertions, and metric scores for that turn.

**Why this priority**: The report is already valuable even as a flat list of turns; a dedicated visual hierarchy is polish rather than core correctness. Lands after P1/P2 because its value depends on the data they produce.

**Independent Test**: After Stories 1 and 2 ship, open the dashboard for a run that contains a multi-turn test case. Verify each turn is individually inspectable and the parent row summarizes the count of passing / failing turns.

**Acceptance Scenarios**:

1. **Given** a completed test run with multi-turn test cases, **When** the markdown/JSON report is generated, **Then** each turn has its own entry under the test case with input, response, tool invocations for that turn, and per-turn metric scores.
2. **Given** a completed test run, **When** the run is opened in the dashboard, **Then** multi-turn test cases render as expandable rows with per-turn detail, and single-turn test cases render as they do today.

---

### Edge Cases

- **Empty turns list**: A test case with `turns: []` is a configuration error and must be rejected at load time with a clear message.
- **Mixing legacy and turns fields**: If a test case specifies *both* the legacy flat `input`/`ground_truth` and the new `turns` field, it is a configuration error — the author must pick one shape.
- **Session errors mid-conversation**: If the backend session becomes unrecoverable mid-conversation (e.g., connection lost, max_turns exhausted at backend level), the remaining turns are marked skipped (not failed) and the test case is marked failed with an explanatory reason.
- **Tool called but no matching arg set**: If `expected_tools` asserts tool `subtract` with specific args and the agent calls `subtract` with different args, the turn fails on arg assertion — it does not pass merely because a tool of that name was called.
- **Tool called with extra args**: Extra arguments beyond those asserted are allowed and do not cause failure (permissive matching on unasserted keys).
- **Tool called zero times when expected once**: Fails with a message naming the missing tool.
- **Tool called multiple times**: A single satisfying call anywhere within the turn passes the assertion (see Story 3, scenario 6).
- **Regex compile errors**: A malformed regex in `expected_tools[*].args[*].regex` is a configuration error caught at load time, not a runtime test failure.
- **Numeric argument fuzziness**: Integer vs float equivalence (`206588` vs `206588.0`) and stringified numeric arguments with thousands separators (`"206,588"`) are treated as matching under fuzzy matching. Exact (literal) matching normalizes numeric types before comparison but does not strip separators.
- **Tools called outside the expected set**: An agent calling extra tools not listed in `expected_tools` does not fail the turn — `expected_tools` is a lower bound, not an upper bound, consistent with current behavior.

## Requirements *(mandatory)*

### Functional Requirements

**Configuration schema**

- **FR-001**: The test case schema MUST accept an optional ordered `turns` list. When `turns` is present, the legacy flat `input`, `ground_truth`, and `expected_tools` fields MUST NOT also be set on the same test case; this combination MUST be rejected at configuration load time with a clear error.
- **FR-002**: Each element of `turns` MUST support `input` (required, non-empty string), `ground_truth` (optional string), `expected_tools` (optional list), `files` (optional multimodal inputs — same schema as today's test case), and `retrieval_context` (optional list of strings — for RAG metrics on that turn).
- **FR-003**: The `expected_tools` field (both at the legacy test-case level and at the new turn level) MUST accept two shapes: a list of strings (legacy, name-only) and a list of objects `{name: str, args?: dict, count?: int}`. Both shapes MUST be supported simultaneously within one list.
- **FR-004**: Within an `expected_tools` object's `args` map, each value MUST accept one of: a literal scalar/list/object (exact match after normalization), `{fuzzy: str}` (fuzzy match), `{regex: str}` (regex match, anchored full-match against the stringified argument value). Invalid shapes MUST be rejected at load time.
- **FR-005**: Existing single-turn test cases MUST continue to run unchanged with no edits required. The executor MUST auto-detect whether a test case is single-turn (legacy) or multi-turn (`turns` present) and dispatch accordingly.

**Execution**

- **FR-006**: For multi-turn test cases, the executor MUST drive execution through the provider-agnostic `AgentBackend.create_session()` / `AgentSession.send()` path (see `src/holodeck/lib/backends/base.py`), sending turn N+1 only after turn N's response has been received, and MUST close the session on test-case completion (success, failure, or unrecoverable error). Single-turn legacy test cases MAY continue to use the existing `AgentBackend.invoke_once()` path.
- **FR-007**: The executor MUST capture, per turn: the turn input, the agent's response text, the list of tool calls made during that turn (name + args + result), and the token usage for that turn. The test-case-level `TestResult.token_usage` roll-up MUST equal the element-wise sum of per-turn token usages for multi-turn cases, and MUST equal the single invocation's usage for single-turn legacy cases — preserving the existing dashboard cost contract (feature 031).
- **FR-008**: The existing `llm_timeout` config MUST apply per turn — each `AgentSession.send()` invocation is bounded by `llm_timeout` independently. If a turn errors at invocation time (timeout, backend error), the turn MUST be marked failed with the error reason, and the executor MUST attempt to continue with subsequent turns using the existing session. If the session itself is unrecoverable, remaining turns MUST be marked skipped with an explanatory reason. No whole-test-case wall-clock timeout is introduced by this feature.
- **FR-009**: Per-turn `files` MUST be processed and included with that turn's input only, not replayed on subsequent turns.
- **FR-009a**: `ExecutionConfig` MUST gain a new field `parallel_test_cases: int` (default 1). When >1, the executor MUST run up to N test cases concurrently, each in its own `AgentSession`. Within a single multi-turn test case, turns MUST remain strictly sequential (turn N+1 only after turn N completes) to preserve conversational state. `parallel_test_cases` MUST be exposed through the existing CLI > YAML > env > defaults resolution chain.

**Assertions**

- **FR-010**: For each turn, the executor MUST evaluate the turn response against the turn's `ground_truth` (if provided) using the metrics configured at the test-case or agent level, producing per-turn metric results.
- **FR-011**: For each turn with `expected_tools`, the executor MUST verify that every named tool was called at least the required number of times during that turn (default required count is 1). Tool calls from earlier or later turns MUST NOT satisfy a turn's tool assertion. Tool-name matching MUST inherit the existing legacy contract in `validate_tool_calls` (`src/holodeck/lib/test_runner/executor.py`): **case-sensitive substring match** — an expected name `subtract` passes if any actual tool name contains `subtract` as a substring (accommodates SK plugin name-prefixing). When an `expected_tools` entry also specifies `args`, the turn passes the assertion only if a single actual call satisfies both the name substring match AND the arg matchers from FR-012 simultaneously.
- **FR-012**: For each expected tool that specifies `args`, the executor MUST check for at least one actual call to that tool in the turn where every asserted arg matches by its specified matcher (exact, fuzzy, or regex). Unasserted arg keys on the actual call MUST NOT cause failure.
- **FR-013**: Fuzzy matching MUST be case-insensitive, whitespace-normalized, tolerate numeric equivalence (int vs float, trailing zeros), and tolerate common thousands separators (e.g., `"206,588"` ↔ `206588`). The exact tolerance model MUST be documented alongside the feature.
- **FR-014**: Regex matching MUST be full-match (anchored) against the argument value rendered via `str(value)`, using Python regex semantics.

**Reporting**

- **FR-015**: The `TestResult` model (`src/holodeck/models/test_result.py`) MUST gain an optional `turns: list[TurnResult]` field. Each `TurnResult` contains input, response, tool invocations, tool/arg assertion outcomes, per-turn metric scores, pass/fail, token usage, and turn duration. For multi-turn cases, `TestResult`'s existing top-level fields MUST be populated with test-case-level roll-ups: `test_input` = concatenated turn inputs, `agent_response` = final turn's response, `tool_calls` / `tool_invocations` = merged list across turns (in turn order), `token_usage` = element-wise sum across turns (per FR-007), `passed` = `all(turn.passed)`, and `metric_results` = per-metric roll-up where each unique `metric_name` appears exactly once with `score = mean(turn_scores for that metric)` and `passed = all(turn_passed for that metric)`. Turns that skipped a metric (no `ground_truth` or metric not applicable) MUST NOT contribute to that metric's mean. Single-turn legacy cases MUST leave `turns` empty or unset. `ReportSummary.validate_test_counts` (which enforces `passed + failed == total_tests`) MUST continue counting test cases, not turns.
- **FR-016**: A multi-turn test case passes overall only if every turn passes. A single failed turn MUST fail the test case.
- **FR-017**: The existing eval-run dashboard (feature 031) MUST render multi-turn test cases with expandable per-turn detail without regressing single-turn display.

**Programmatic evaluators**

- **FR-018**: The `EvaluationMetric` model (discriminator `type: standard`) MUST accept two new values of its `metric` field: `equality` (with optional `case_insensitive`, `strip_whitespace`, `strip_punctuation` normalization flags surfaced through this model) and `numeric` (with optional `absolute_tolerance`, `relative_tolerance`, `accept_percent`, `accept_thousands_separators` flags surfaced through this model). These deterministic evaluators operate on `actual_output` / `expected_output` strings, matching the existing NLP-metric shape. When the user does not set either tolerance on `numeric`, defaults MUST be `absolute_tolerance = 1e-6` and `relative_tolerance = 0.0`, with pass semantics `abs(actual - expected) <= absolute_tolerance`. `equality` MUST default all normalization flags to `false` (strict string equality). All defaults MUST be documented in the feature docs.
- **FR-019**: The `MetricType` discriminated union in `src/holodeck/models/evaluation.py` MUST gain a new top-level variant `CodeMetric` with discriminator `type: code`, carrying a required `grader: "module.path:function_name"` field (and not inheriting the NLP-metric fields). The grader callable MUST be resolved and imported at configuration load time, not at first use.
- **FR-020**: Code graders MUST receive a stable, documented context object containing: `turn_input`, `agent_response`, `ground_truth` (if present), `tool_invocations` (ordered list with names, args, results), `retrieval_context` (if present), `turn_index`, `test_case_name`, and the full per-turn config. The context MUST be read-only from the grader's perspective.
- **FR-021**: Code graders MUST return a result that conforms to a documented shape (score in `[0, 1]` or boolean, optional `passed`, optional `reason`, optional `details`). Returned values MUST be recorded in the report alongside built-in metric results.
- **FR-022**: A code grader that raises an exception MUST fail only that turn's evaluator with the exception captured as the failure reason, and MUST NOT halt the test run.
- **FR-023**: Built-in `equality` and `numeric` evaluators and user-supplied `code` evaluators MUST be usable at all three configuration levels where metrics are already supported (agent-level default, test-case-level override, per-turn override).

**Back-compat and safety**

- **FR-024**: The legacy flat `expected_tools: list[str]` contract MUST remain accepted without modification for existing configs.
- **FR-025**: Configuration errors (empty turns, mixed legacy/new shapes, malformed regex, invalid matcher shape, unresolvable grader path, invalid tolerance values) MUST be caught at config load time with error messages that name the offending test case and field path — never surfaced as opaque runtime failures.
- **FR-026**: Code graders run in-process as trusted Python code — the same trust level as any other code in the user's agent project. The documentation MUST state this clearly so users do not load grader code from untrusted sources.

### Key Entities

- **Turn**: One exchange within a multi-turn test case. Holds the user input, optional ground truth, optional expected-tools list (with optional arg matchers), optional file inputs, optional retrieval context, and (at runtime) the captured agent response, tool invocations, metric scores, and pass/fail status for that turn.
- **Test Case**: Either single-turn (legacy shape) or multi-turn (ordered list of Turns plus test-case-level `name` and metric overrides). A multi-turn test case owns the session lifecycle for its turns.
- **Expected Tool Call**: An assertion about a tool the agent is expected to call during a specific turn. Carries the tool name, an optional argument-matcher map, and an optional minimum call count.
- **Argument Matcher**: The rule used to compare one expected argument value against the actual value the agent passed. Kinds: `exact` (literal, normalized), `fuzzy` (case- and whitespace-tolerant, numeric-aware), `regex` (anchored Python regex over the stringified value).
- **Turn Result**: The per-turn outcome recorded in the test report — input, response, tool invocations, assertion outcomes, metric scores, token usage, duration, pass/fail.
- **Programmatic Evaluator**: A deterministic grader that produces a score/pass-fail without invoking an LLM. Either built-in (`equality`, `numeric`) or user-supplied (`code`, referenced by import path). Input: a documented per-turn context object. Output: a standard metric-result shape.
- **Grader Context**: The read-only record passed to a code grader — turn input, agent response, ground truth, tool invocations, retrieval context, turn index, test-case metadata.

## Assumptions

- **A1 — Benchmark pluralism**: Although ConvFinQA is the motivating dataset, the feature is not scoped to it. A generic ConvFinQA importer may be added later as a separate utility; this spec only covers HoloDeck's native schema.
- **A2 — Continue-on-turn-failure**: When a turn fails (on ground-truth mismatch or tool assertion), subsequent turns still execute. The default is informative ("tell me every turn's outcome") rather than halt-on-first-fail. Authors who want halt semantics can add an explicit follow-up feature later.
- **A3 — Tool-arg ordering is name-based**: Tool calls are matched by argument *name*, not position. This aligns with how Semantic Kernel and Claude Agent SDK already represent tool calls (keyword-arg dicts).
- **A4 — Extra arguments are permitted**: Unasserted arguments on an actual tool call do not fail the assertion. This preserves robustness against legitimate tool-signature evolution.
- **A5 — Default minimum call count is 1**: If `expected_tools` names a tool without a `count`, the turn passes when that tool is called at least once with matching args in the turn. Exact-count assertions can be a future extension.
- **A6 — Per-turn metric evaluation reuses the existing metric stack**: No new metric types are introduced. Metrics configured at the test-case or agent level are applied to each turn with `ground_truth` present. Turns without `ground_truth` skip text-comparison metrics (as today).
- **A7 — Numeric normalization for literal matching**: Literal (exact) arg matching normalizes `int` vs `float` equivalence (`206588 == 206588.0`) but does not strip separators. Separator tolerance is a `fuzzy` feature.
- **A8 — Session per test case**: Each multi-turn test case runs in its own fresh backend session. No cross-test-case state bleed.
- **A9 — Code graders are trusted**: Code graders execute in the test runner process with no sandbox. This is acceptable because HoloDeck already runs the user's tool plugins and MCP servers in-process; grader code is held to the same trust boundary. Documentation must make this explicit.
- **A10 — No dataset-specific built-ins**: ConvFinQA-specific graders (e.g., `turn_program` equivalence) are *not* built into HoloDeck. They belong in the user's benchmark repo as a code grader. The built-in set stays small and generic (`equality`, `numeric`).

## Out of Scope

- Importing ConvFinQA or any other corpus directly. A one-off conversion utility can be added later but is not part of this feature.
- Sandboxing or resource-limiting code graders. They run in-process with full Python privileges (see A9). Sandboxing can be a future hardening feature if demand exists.
- Shipping a built-in ConvFinQA `turn_program` equivalence grader. Reference implementations may be provided as examples under `docs/` or a user-land `examples/` repo, but the HoloDeck package itself stays dataset-agnostic.
- Automatic generation of `turns` from prior single-turn runs.
- Assertions on the **order** of tool calls within a turn. Order-sensitive assertions can be a future extension.
- Halt-on-first-failing-turn execution mode.
- Cross-turn assertions (e.g., "turn 4 must reference the answer from turn 3"). The session already preserves context; evaluating cross-turn derivation is a future research question.
- Exact-count tool-call assertions (`count: 2`). Assumed default is "at least one" (A5).

## Complexity Tracking

### Constitution Principle I — Justified Exception (`type: code` graders)

**Principle cited**: Core Principle I — *No-Code-First Agent Definition*: "Every agent configuration MUST be defined through declarative YAML files. Users SHOULD NOT write Python code to define agents, tools, evaluations, or test cases."

**Exception requested**: User Story 4 introduces a `type: code` evaluator that requires a user-supplied Python callable referenced by import path (e.g., `grader: "my_benchmarks.convfinqa:grade_program"`). This is, strictly, Python code written to define an evaluation.

**Why current constraints are insufficient**: Benchmarks such as ConvFinQA define success as **symbolic program equivalence** between the agent's tool-call graph and a reference `turn_program` (e.g., `subtract(206588, 181001), divide(#0, 181001)` with `#N` back-references to prior turn results). No combination of existing HoloDeck evaluators (NLP overlap metrics, LLM-as-judge G-Eval, or the new deterministic `equality` / `numeric` evaluators introduced by this feature) can express this check faithfully:

- NLP metrics operate on text and cannot reason over tool-call graphs.
- G-Eval introduces LLM non-determinism and cost on a check that is deterministic by nature.
- `equality` / `numeric` compare scalar answers and cannot traverse a DAG with back-references.

A YAML-embedded expression DSL was considered (see the batch-1 verification question); it would reproduce, in YAML, a Turing-adjacent expression language just to avoid calling Python — a net-negative on simplicity and maintenance burden.

**Scope of the exception**: The exception is scoped *exclusively* to the evaluator surface. All other no-code guarantees remain intact:

- Agents, tools, MCP servers, and test cases — still YAML-only.
- Built-in evaluators (`equality`, `numeric`, plus existing `standard`/`geval`/`rag`) — still YAML-only.
- `type: code` is an **opt-in escape hatch** for benchmarks whose success criteria are deterministic but non-expressible in YAML.

**Mitigations**:

1. The built-in `equality` and `numeric` evaluators introduced by this feature satisfy 80%+ of deterministic grading needs without any Python.
2. Documentation MUST lead with the YAML-native evaluators and position `type: code` as the escape hatch, not the default.
3. Assumption A9 (trusted, in-process execution) and FR-026 require documentation to warn against loading grader code from untrusted sources.
4. Constitution amendment is not required — this is a **Justified Exception** per the governance clause ("Justified exceptions (Complexity Tracking table) MUST reference specific principles and explain why constraints cannot be met"), logged here for reviewer acknowledgment at PR time.

**Reviewer sign-off requirement**: The PR implementing this feature MUST link to this Complexity Tracking section in its description so reviewers explicitly accept the exception before merge.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A benchmark author can author, run, and interpret a 4-turn ConvFinQA-shaped test case end-to-end from a fresh `agent.yaml` in under 15 minutes, including reading the docs.
- **SC-002**: 100% of existing single-turn test cases in the HoloDeck test suite and in user-facing templates continue to pass with no config edits after the feature lands.
- **SC-003**: For a 100-example ConvFinQA-shaped test run (4 turns each, 400 turns total), 100% of turns appear in the final report with the four fields authors most need: turn input, agent response, per-turn tool invocations, per-turn pass/fail.
- **SC-004**: Turn-level failures pinpoint the failing turn index in the report in 100% of cases — no overall "test case failed" without a per-turn breakdown.
- **SC-005**: Given a malformed regex or a `turns: []` list, `holodeck test` exits with a configuration error code before any agent invocation, and the error message names the offending test-case `name` and field path, in 100% of cases.
- **SC-006**: Tool-argument matching (exact + fuzzy + regex combined) achieves the expected pass/fail outcome on an acceptance test matrix of at least 20 pairs of (expected args, actual args) that together cover: int/float equivalence, thousands separators, case/whitespace variation, anchored regex, missing argument, extra unasserted argument, multi-call fuzzy match.
- **SC-007**: A benchmark author can author, register, and run a working `code` grader (e.g., ConvFinQA `turn_program` equivalence) in under 30 minutes of work starting from the feature's documentation, and it integrates into the per-turn report with no changes to the reporter.
- **SC-008**: Built-in `equality` and `numeric` evaluators achieve the expected pass/fail on an acceptance test matrix of at least 15 pairs covering: exact strings, case/whitespace/punctuation variation, integer vs float, percent suffix handling, thousands separators, absolute vs relative tolerance, negative tolerance boundary.
- **SC-009**: A grader that raises an exception fails its single turn, records the exception type and message in the report, and never halts the overall test run — verified across 100% of acceptance tests for error handling.
- **SC-010**: A representative multi-turn test case (3+ turns, at least one `ground_truth` and one `expected_tools` assertion per turn) passes end-to-end on **both** the Semantic Kernel backend (OpenAI or Ollama provider) and the Claude Agent SDK backend, with identical pass/fail outcomes and consistent per-turn `TokenUsage` capture on each. Feature is not considered shippable until this dual-backend smoke integration is green.
