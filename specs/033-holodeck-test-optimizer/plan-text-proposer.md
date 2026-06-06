# Implementation Plan: Iterative Textual Refinement (multi-step TGD)

**Spec:** `specs/033-holodeck-test-optimizer/textual-iterative-refinement.md`
**Status:** Draft for review (2026-06-02) — no code written yet.
**Companion task list:** `tasks-text-proposer.md`

## Overview

Today the textual phase runs exactly one Critic→Applier rewrite per declared axis
per cycle: `TextualProposer.ask()` consumes the axis and then returns `None`
(`proposers/textual.py:160-165`), and `tell()` is a no-op (`:220`). With one
declared axis — the common case, including the `financial-assistant` sample —
the phase emits a single trial and `textual_phase.max_trials`/`patience` never
bind (the run audit showed `numeric: 14, textual: 2`).

This work turns the textual phase into a true multi-step TextGrad loop on a
**single** axis: each step sees the previous attempt, its loss, and the
failing-case context re-derived from that attempt's own report, and proposes the
next refinement. Iteration is bounded by the *existing* `max_trials`/`patience`
budget and the loop's *existing* accept/revert gate — so a wandering chain can
never become the output. No new config, no schema change, numeric phase
byte-identical.

## Architecture Decisions

These are locked by the spec (§Decisions, confirmed 2026-06-02); restated here
because they shape the task slicing:

1. **Gradient = attempt + loss + failing cases.** Step N+1's Critic sees attempt
   N's text, its loss, and `build_failing_context(attempt-N report)` — already
   computed by the scorer, so iteration adds no eval cost.
2. **Momentum in the proposer, revert in the loop.** Each step chains from the
   *last attempt* (in-place TextGrad `step()`); the loop's `best_agent`-only-
   advances-on-accept is the validation revert. `rebase-on-best` is a documented
   fallback, **not** the default.
3. **Budget = existing `textual_phase.max_trials`/`patience`.** Their semantics
   shift from "across axes" to "refinement steps on the axis." No new knob.
4. **Single-axis scope.** `>1` declared textual axis falls back to today's
   one-pass-per-axis behaviour and logs that multi-axis ordering is unsupported.

### Key implementation facts established during planning

- **The `tell` extension is backward-compatible.** Both call sites
  (`loop.py:135`, `loop.py:157`) and both existing test call sites
  (`test_numeric.py:74,125`) pass exactly three positional args, so adding
  `report: TestReport | None = None` as a 4th optional param breaks nothing.
- **`build_failing_context` already exists** (`textual.py:96`) and takes a
  `TestReport | None`, returning a bounded, secret-free summary. Reuse as-is for
  the per-attempt gradient.
- **Stop conditions stay in `loop._run_phase`** (`loop.py:111-113`): the
  `for _ in range(max_trials)` + `no_improve >= patience` loop already bounds the
  phase. The proposer just has to keep returning proposals instead of `None`.
- **`begin()` is called once per phase** (`loop.py:107`) with the current
  `best_agent`/`best_report` — the natural place to seed per-phase iteration
  state and reset the momentum chain.

## Dependency Graph

```
Task 1: tell(report=...) protocol  ── foundation, additive/optional
    │
    ├── (numeric.py tell ignores report)   ← same task, keeps numeric identical
    ├── (loop.py passes report at :157)    ← same task, unlocks the data flow
    │
    └── Task 2: TextualProposer iteration state machine
            │   (+ critic.yaml prior-attempt prompt — coupled to _critique call)
            │
            └── Task 3: loop integration + e2e + docs
```

Bottom-up: the protocol/data-flow change (Task 1) must land first because the
proposer can only iterate once it receives the per-attempt `report`. The Critic
prompt change is coupled to the proposer's `_critique` signature, so it rides
with Task 2. Integration/e2e/docs (Task 3) verify the assembled behaviour.

## Task List

### Phase 1: Contract & data flow
- **Task 1** — Extend `Proposer.tell` with an optional `report`; thread it from
  the loop's scored-trial call site; numeric proposer accepts and ignores it.

#### Checkpoint: Contract
- [ ] `make test-unit` green (numeric + loop tests unchanged in behaviour).
- [ ] `make type-check` clean (protocol + both implementers + call sites).
- [ ] Numeric phase output byte-identical on the `financial-assistant` sample
      config (no behavioural diff — the arg is inert for numeric).

### Phase 2: Iterative textual core
- **Task 2** — Rewrite `TextualProposer` into a single-axis iteration state
  machine (momentum source, per-attempt gradient, history-aware Critic prompt,
  `>1 axis` fallback); enrich `critic.yaml` to accept prior-attempt/loss context;
  unit tests for momentum, context source, multi-step continuation, and fallback.

#### Checkpoint: Core
- [ ] New unit tests in `test_textual.py` prove: step 2 chains from the **last
      attempt** (not best), uses the **last attempt's report** for context, and
      `ask()` keeps returning proposals across iterations.
- [ ] `>1 axis` path logs the fallback line and behaves as today (one pass/axis).
- [ ] All existing `test_textual.py` cases still pass unchanged.

### Phase 3: Integration, e2e, docs
- **Task 3** — Loop-level integration test (multi-step run, `patience` halt,
  `best_agent` never regresses below single-step), an e2e stub-backend case where
  iterative steps reduce loss, and the docs/schema-description note on the
  `max_trials`/`patience` semantics shift.

#### Checkpoint: Complete
- [ ] `make test` (full, parallel) green.
- [ ] `make format && make lint && make type-check` clean.
- [ ] `docs/guides/evaluations.md` (and the schema field descriptions) state the
      shifted `textual_phase` semantics.
- [ ] Manual: run `holodeck test optimize` on a 1-axis stub and confirm
      `report.md` shows ≥2 `phase: textual` trials in a cycle when budget allows.
- [ ] Human review before merge.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| k=1 eval noise makes the momentum chain wander and waste the budget | Med | Loop revert means a drifted chain is never the output; `patience` kills a flat chain early. `rebase-on-best` fallback documented in spec if observed in practice. |
| Semantics shift on `max_trials`/`patience` silently changes behaviour for existing configs | Med | No code regression (defaults unchanged), but Task 3 *must* land the docs + schema-description note so the meaning change is discoverable. |
| Enriched Critic prompt accidentally carries secrets | High | Gradient is built only from instruction text + `build_failing_context` (test_input, truncated response, metric names/scores). Task 2 adds prior-attempt text + loss only — assert in tests that no credential-bearing field is interpolated. |
| Numeric phase behaviour drifts via the shared `tell` change | Med | `report` is additive/optional and ignored by `NumericProposer`; Phase-1 checkpoint asserts byte-identical numeric output. |
| Proposer's internal chain leaks into `best_agent` outside the accept gate | High | Proposer never mutates `best_agent`; it only returns `Proposal`s. Loop's `_apply` builds fresh candidates and advances `best_agent` only on accept. Integration test asserts no regression. |

## Open Questions

- **Convergence signal (optional nicety):** spec §state-machine allows the Critic
  to emit an explicit "no further improvement" to let `ask()` return `None` early.
  Treat as *optional* in Task 2 — implement only if the Critic JSON already has a
  clean place for it; otherwise rely on `patience`. Flag for reviewer.
- **History bound:** spec says `_history` is "bounded length" but not the bound.
  Proposing last 3 attempts (summary, loss, accepted) to keep the prompt small —
  confirm with reviewer.
