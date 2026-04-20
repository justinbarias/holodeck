# Implementation Plan: Multi-Turn Test Cases, Per-Turn Assertions, Tool-Arg Matchers, and Programmatic Evaluators

**Branch**: `032-multi-turn-test-cases` | **Date**: 2026-04-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/032-multi-turn-test-cases/spec.md`

## Summary

Extend the HoloDeck test runner so an `agent.yaml` test case can declare an ordered list of `turns`, each with its own `ground_truth`, `expected_tools` (optionally with per-arg matchers), `files`, and `retrieval_context`. Execution routes through `AgentBackend.create_session()` so turns share stateful conversation context. Each turn is evaluated independently, then rolled up per-metric at the test-case level. Add built-in deterministic evaluators (`equality`, `numeric`) on the existing `EvaluationMetric` shape and a new top-level `CodeMetric` variant for user-supplied Python graders referenced by import path. Legacy single-turn test cases remain unchanged.

**Reference corpus investigated**: `/Users/justinbarias/Documents/Git/python/justinbarias/data/convfinqa_dataset.json` (3,037 train + 421 dev ConvFinQA examples). Each example has `dialogue.conv_questions`, `dialogue.conv_answers`, `dialogue.turn_program`, and `dialogue.executed_answers` lists of equal length. Later turns reference earlier turn results via `#N` placeholders (e.g. turn 4: `subtract(206588, 181001), divide(#0, 181001)`). This confirms (a) turns are strictly ordered and stateful, (b) ground truths are scalar answers best graded by `numeric` or `code` (not BLEU/ROUGE), (c) tool-arg assertions must accept literal, fuzzy, and regex shapes to handle `206588` vs `206588.0` vs `"206,588"` drift, and (d) program-equivalence grading (turn_program with `#N` back-refs) must live in a user-supplied `code` grader — the built-in set stays dataset-agnostic per Assumption A10.

## Technical Context

**Language/Version**: Python 3.10+ (matches existing HoloDeck target; see constitution §Code Quality).
**Primary Dependencies**: Existing — Pydantic v2, Click, `semantic-kernel`, `claude-agent-sdk`, `deepeval`. No new runtime dependencies; all matchers and evaluators built on Python stdlib (`re`, `difflib`, `importlib`, `decimal`).
**Storage**: No new persistence. Test reports continue to land at `results/<slugified-agent-name>/<ISO-timestamp>.json` (feature 031). `TestResult.turns` is an additive JSON field.
**Testing**: pytest with `-n auto` parallel execution; markers `@pytest.mark.unit`, `@pytest.mark.integration`. Integration tests hit both SK (Ollama or OpenAI) and Claude SDK backends per SC-010.
**Target Platform**: CLI on macOS / Linux (no platform-specific behavior).
**Project Type**: Single-project Python package (src/holodeck, existing layout).
**Performance Goals**: Turn-sequential within a test case (conversational state preserved); optional `parallel_test_cases` (FR-009a) unlocks across-test-case concurrency. No per-turn latency targets beyond the existing `llm_timeout` per `AgentSession.send()`.
**Constraints**:
- Backwards compatibility: every legacy single-turn test case must continue to run with identical output shape (SC-002).
- `ReportSummary.validate_test_counts` continues counting **test cases**, never turns (FR-015).
- `TestResult.token_usage` roll-up = element-wise sum across turns (FR-007), preserving the feature 031 cost contract.
- Dual-backend green: SK (OpenAI/Ollama) + Claude SDK, identical pass/fail on the representative smoke test (SC-010).
- Code graders run in-process, unsandboxed, resolved at load time (FR-019, FR-022, A9).
**Scale/Scope**: Designed to handle ConvFinQA-shaped runs: 100+ test cases × 3–5 turns each = 300–500 turns per run (SC-003).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Compliance | Notes |
|-----------|------------|-------|
| I. No-Code-First | **Justified exception** (logged in spec §Complexity Tracking) for `type: code` graders only. All other new surfaces (`turns`, `expected_tools.args`, `equality`, `numeric`) remain pure YAML. |
| II. MCP for APIs | N/A — no new external API integrations. Existing MCP tool contract unchanged. |
| III. Test-First + Multimodal | Per-turn `files` preserves multimodal support (FR-009). Every new matcher/evaluator/orchestration code path ships with unit tests; integration tests cover dual-backend multi-turn (SC-010). Spec mandates `ground_truth` or evaluators per turn (aligns with principle). |
| IV. OpenTelemetry-Native | Each turn is already an `AgentSession.send()` call which flows through the existing SK/Claude tracing; no new spans required beyond what the backends already emit. Reporter additions are pure data; no new instrumented sites. |
| V. Evaluation Flexibility | New `equality` + `numeric` add deterministic metrics that require **no LLM call** (aligns with principle's "NLP metrics MUST not require LLM calls" extension). Code graders inherit whatever model the user chooses inside their function — HoloDeck imposes no model at the evaluator boundary. Three-level model override (agent / test-case / per-metric) continues to apply unchanged for standard / geval / rag. |

**Initial gate: PASS** — the single no-code-first exception is pre-justified in spec §Complexity Tracking (scoped to evaluator surface, mitigated by built-in YAML-native `equality`/`numeric`, documented trust model). No further gate entries needed.

Post-design re-check (end of Phase 1): **PASS** — the design keeps the exception surface to exactly one config type (`CodeMetric`) and one resolution site (load-time import). See §Complexity Tracking at the bottom of this plan.

## Project Structure

### Documentation (this feature)

```text
specs/032-multi-turn-test-cases/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── test-case-schema.md        # YAML schema (turns, expected_tools objects, matchers)
│   ├── turn-result-schema.md      # TestResult.turns + TurnResult JSON contract
│   ├── code-grader-contract.md    # Grader callable signature + context + result shape
│   └── tool-arg-matchers.md       # exact/fuzzy/regex matcher semantics
├── checklists/
│   └── requirements.md  # Pre-existing quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

This feature lands inside the existing single-project layout. New files are green; modified files are yellow.

```text
src/holodeck/
├── models/
│   ├── test_case.py          # MODIFY — add Turn, make `input`/`ground_truth`/`expected_tools` mutually exclusive with new `turns` field; add ExpectedTool + ArgMatcher variants
│   ├── evaluation.py         # MODIFY — allow `metric: equality | numeric` with normalization / tolerance flags; add `CodeMetric` variant to MetricType union
│   ├── test_result.py        # MODIFY — add TurnResult + optional `turns: list[TurnResult]` on TestResult + roll-up contract; extend `MetricResult.kind` Literal to include `"code"`
│   ├── token_usage.py        # MODIFY — add `cache_creation_tokens` and `cache_read_tokens` fields; update `total_tokens` validator to include them (cross-cuts feature 031 cost contract — flag in PR)
│   └── config.py             # MODIFY — ExecutionConfig.parallel_test_cases: int = 1
├── lib/
│   ├── evaluators/
│   │   └── deterministic.py  # NEW — EqualityEvaluator, NumericEvaluator (no LLM)
│   └── test_runner/
│       ├── executor.py             # MODIFY — (a) dispatch single-turn vs multi-turn; (b) integrate matchers/evaluators; (c) roll-up per-metric aggregation; (d) optional parallel test-case orchestration; (e) extend `_metric_kind()` return annotation to `Literal["standard","rag","geval","code"]` to match the new `CodeMetric` variant
│       ├── tool_arg_matcher.py     # NEW — match_arg(actual, expected_matcher) → bool + rationale
│       ├── code_grader.py          # NEW — import resolver + grader invocation harness with per-turn context dataclass
│       └── reporter.py             # MODIFY — render turns in markdown; JSON is automatic
├── cli/
│   └── commands/
│       └── test.py            # MODIFY — expose `--parallel-test-cases N`
├── config/
│   ├── defaults.py            # MODIFY — add `parallel_test_cases: 1` to `DEFAULT_EXECUTION_CONFIG`
│   └── loader.py              # MODIFY (small) — plumb parallel_test_cases through the resolve chain
└── dashboard/                 # P3 — render multi-turn as expandable rows (FR-017, SC-010)
    ├── explorer_data.py       # MODIFY — detect `TestResult.turns` and surface per-turn rows in the explorer payload
    └── components/            # MODIFY — add the per-turn disclosure component; single-turn rendering path untouched
tests/
├── unit/
│   └── lib/
│       ├── test_runner/
│       │   ├── test_executor_multi_turn.py        # NEW
│       │   ├── test_tool_arg_matcher.py           # NEW
│       │   └── test_code_grader.py                # NEW
│       ├── evaluators/
│       │   └── test_deterministic.py              # NEW
│       └── test_testcase_models_multi_turn.py     # NEW
│   └── models/
│       └── test_test_result_turn_rollup.py        # NEW
└── integration/
    └── test_multi_turn_dual_backend.py            # NEW — SC-010 smoke: SK + Claude
```

**Structure Decision**: **Single project** (existing). No new top-level directories. New matcher/evaluator/grader modules slot cleanly under `src/holodeck/lib/test_runner/` and `src/holodeck/lib/evaluators/` next to their peers. Modified modules stay within their current packages — no module moves, no breaking import paths.

## Phase 0 — Research (output: `research.md`)

Areas researched (details and decisions in `research.md`):

1. **ConvFinQA dialogue shape** — confirm turn ordering, ground-truth scalar nature, turn_program back-ref semantics, and the required matcher surface.
2. **Multi-turn backend API** — verify both `SKSession` and `ClaudeSession` implement `AgentSession` with shared state across `send()` calls, per-call error recovery, and token accounting per turn.
3. **Argument matcher design** — selection among `difflib`, custom normalization, and regex for the three matcher kinds; tolerance model (int/float equivalence, thousands separators, percent suffix).
4. **Programmatic evaluator surface** — how to make `equality` and `numeric` fit on `EvaluationMetric.metric` without destabilizing the existing discriminator; how `CodeMetric` slots into the `MetricType` union without affecting GEval/RAG dispatch.
5. **Code grader contract** — import-path resolution at config load time, read-only context dataclass, exception → per-turn failure.
6. **Parallel test-case orchestration** — structured concurrency with `asyncio.Semaphore(parallel_test_cases)`; independence of sessions; safe interleaving of reporter writes.
7. **Roll-up semantics** — per-metric average and `all(turn_passed)` aggregation; skipped turns (no ground_truth) do not contribute to the mean.
8. **Dashboard rendering (feature 031)** — minimal shape required so the dashboard can expand turns without a new table; contract via `TestResult.turns` JSON.

**Exit criterion**: zero NEEDS CLARIFICATION markers in this plan. All five spec clarifications (Session 2026-04-20) are already resolved in-spec; research.md records the implementation-level decisions that flow from them.

## Phase 1 — Design & Contracts

**Prerequisites**: research.md complete.

1. **Data model (`data-model.md`)** — document the new Pydantic shapes end-to-end:
   - `Turn` (input, ground_truth?, expected_tools?, files?, retrieval_context?, evaluations?).
   - `ExpectedTool` (name, args?, count?) as a union with bare `str` for legacy compatibility.
   - `ArgMatcher` union: literal | `{fuzzy: str}` | `{regex: str}` (regex compiled eagerly at load).
   - `TestCaseModel` — add `turns: list[Turn] | None`, make legacy flat fields + `turns` mutually exclusive.
   - `EvaluationMetric.metric` — document `equality` | `numeric` with normalization / tolerance flags.
   - `CodeMetric` — new top-level variant, discriminator `type: code`, carries `grader: str` (module path:callable); resolved at load time.
   - `TurnResult` + `TestResult.turns: list[TurnResult] | None`.
   - `ExecutionConfig.parallel_test_cases: int = 1`.

2. **Contracts (`contracts/`)**:
   - **`test-case-schema.md`** — YAML grammar for new shapes with validation rules and three exemplar test cases (legacy single-turn untouched; multi-turn with per-turn `ground_truth`; multi-turn with tool-arg matchers).
   - **`turn-result-schema.md`** — JSON schema for `TestResult.turns` + roll-up rules for `test_input`, `agent_response`, `tool_calls`, `tool_invocations`, `token_usage`, `passed`, `metric_results`.
   - **`code-grader-contract.md`** — grader callable signature `(ctx: GraderContext) -> GraderResult`, `GraderContext` fields (turn_input, agent_response, ground_truth?, tool_invocations, retrieval_context?, turn_index, test_case_name, turn_config), `GraderResult` shape (`score: float in [0,1] | bool`, `passed?: bool`, `reason?: str`, `details?: dict`), and exception semantics.
   - **`tool-arg-matchers.md`** — exact matcher normalization (int↔float equivalence, no separator stripping); fuzzy matcher (case-insensitive, whitespace-normalized, numeric-aware with separator stripping and percent suffix handling); regex matcher (anchored full-match via `re.fullmatch` against `str(value)`); N-call semantics (default count=1, first-match-wins, order-independent); missing vs extra argument behavior.

3. **Quickstart (`quickstart.md`)** — complete end-to-end ConvFinQA-shaped example: convert one dialogue into an `agent.yaml` test case, run `holodeck test`, inspect the per-turn report; shows all three evaluator styles (BLEU for a text turn, `numeric` for the scalar answer, user-supplied `code` grader for turn_program equivalence).

4. **Agent context update** — run `.specify/scripts/bash/update-agent-context.sh claude` so CLAUDE.md's "Recent Changes" block records the new feature branch (`032-multi-turn-test-cases: ExecutionConfig.parallel_test_cases, Turn/TurnResult models, built-in deterministic evaluators, code-grader contract`).

**Exit criterion**: data-model.md, contracts/*.md, quickstart.md exist, all functional requirements FR-001…FR-026 mapped to a concrete model / contract / module, and constitution re-check still PASS.

## Phase 2 — Task Breakdown (handled by `/speckit.tasks`, not this command)

High-level ordering this plan assumes /speckit.tasks will honor:

1. **P1 vertical slice** (models + executor multi-turn dispatch + per-turn recording, no arg matchers, no code graders): schema for `turns`, `TurnResult`, executor dispatch, reporter prints per-turn, dual-backend smoke passes.
2. **P1 polish**: per-turn metric evaluation + per-metric roll-up rule + back-compat contract tests for every legacy test in `tests/`.
3. **P2 — tool-arg matchers**: `ExpectedTool` object shape, `ArgMatcher` union, `tool_arg_matcher.py`, integration into per-turn tool assertion.
4. **P2 — programmatic evaluators**: `equality` + `numeric` on `EvaluationMetric`, deterministic evaluators, then `CodeMetric` + `code_grader.py`.
5. **P3 — dashboard + reporter turns rendering** (feature 031 render path). Concretely: update `src/holodeck/dashboard/explorer_data.py` to detect `TestResult.turns` and emit per-turn rows; add a disclosure component under `src/holodeck/dashboard/components/` without altering the single-turn path.
6. **Concurrency opt-in**: `parallel_test_cases` orchestration + CLI flag.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| `type: code` grader (breaches Principle I, No-Code-First) | User-defined graders are the only way to express symbolic / graph-equivalence checks (e.g. ConvFinQA `turn_program` with `#N` back-refs) that no NLP / LLM-judge / equality / numeric evaluator can express | A YAML-embedded expression DSL would reproduce a Turing-adjacent language just to avoid calling Python — net-negative on simplicity. LLM-judge graders add cost and non-determinism on deterministic checks. See spec §Complexity Tracking for the full justification; PR description MUST link to that section per spec requirement. |
