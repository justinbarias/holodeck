# Tasks: Iterative Textual Refinement (multi-step TGD)

**Plan:** `plan-text-proposer.md`
**Spec:** `textual-iterative-refinement.md`
**Convention:** after each task run `make format`, `make lint`, `make type-check`,
`make test-unit` (per CLAUDE.md workflow). Run pytest with `-n auto`.

---

## Task 1: Extend the `tell` protocol with an optional scored report

**Description:** Add `report: TestReport | None = None` to `Proposer.tell` so a
proposer can build the next gradient from the candidate it was just scored on.
Thread the report the loop already has into the scored-trial call site; pass
`None` on the skipped-trial path. The numeric proposer accepts and ignores the
new arg so its behaviour is byte-identical. This is the additive foundation the
textual iteration (Task 2) depends on.

**Acceptance criteria:**
- [ ] `Proposer.tell` (`proposers/base.py`) signature is
      `tell(self, proposal, loss, accepted, report: TestReport | None = None)`
      with the contract documented (when `report` is `None`; that numeric ignores it).
- [ ] `loop.py:157` passes the `report` from `await self.scorer(candidate)`;
      `loop.py:135` (skip path) passes `report=None` (or omits it).
- [ ] `NumericProposer.tell` (`numeric.py`) accepts `report` and ignores it.

**Verification:**
- [ ] `pytest tests/unit/optimizer/test_numeric.py tests/unit/optimizer/test_loop.py -n auto`
      passes with no changes to their assertions (the 3-arg call sites still type-check).
- [ ] `make type-check` clean — protocol, both implementers, and both call sites agree.
- [ ] Manual diff check: numeric-only run on the sample config produces the same
      trials as before (arg is inert for numeric).

**Dependencies:** None.

**Files likely touched:**
- `src/holodeck/optimizer/proposers/base.py`
- `src/holodeck/optimizer/proposers/numeric.py`
- `src/holodeck/optimizer/loop.py`

**Estimated scope:** Small (3 files, mechanical).

---

## Task 2: Single-axis iteration state machine in `TextualProposer`

**Description:** Replace the one-proposal-per-axis logic with a per-phase
single-axis iteration loop. On `begin()`, seed `_axis = axes[0]`, capture
`_best_text`/`_best_report`, and reset the momentum chain
(`_last_text/_last_loss/_last_report = None`, `_history = []`). Each `ask()`
sources from the **last attempt** when present (momentum), else `_best_text`;
builds the gradient context from `_last_report or _best_report`; passes the
attempt index, last loss, and bounded history into the Critic; and returns the
rewritten proposal. `tell()` stores `proposal.new_text`, `loss`, `report`, and
appends `(summary, loss, accepted)` to `_history`. With `>1` declared textual
axis, fall back to today's one-pass-per-axis behaviour and log once that
multi-axis iterative ordering is unsupported. Enrich `critic.yaml` so its prompt
accepts the prior-attempt/loss context. Stop conditions stay in the loop —
`ask()` returns `None` only on the (optional) Critic convergence signal or, in
fallback mode, when axes are consumed.

**Acceptance criteria:**
- [ ] On a 1-axis agent, `ask()` returns a proposal on every call (it does **not**
      return `None` after the first), bounded only by the loop's budget.
- [ ] Step 2's source text is attempt 1's `new_text` (momentum), and its failing
      context is derived from attempt 1's `report` — not from `_best_*`.
- [ ] `tell(proposal, loss, accepted, report)` updates `_last_*` and appends to a
      length-bounded `_history`.
- [ ] `>1` textual axis → one-pass-per-axis fallback with a single log line; no crash.
- [ ] `critic.yaml` prompt consumes prior-attempt text + last loss; **no secret-
      bearing field** is ever interpolated (only instruction text + failing context).

**Verification:**
- [ ] New `tests/unit/optimizer/test_textual.py` cases (fake invoker, scripted JSON):
      momentum-from-last-attempt, context-from-last-report, multi-step continuation
      (≥2 proposals), and the `>1 axis` fallback + log assertion.
- [ ] All existing `test_textual.py` cases pass unchanged.
- [ ] `make type-check` clean.

**Dependencies:** Task 1.

**Files likely touched:**
- `src/holodeck/optimizer/proposers/textual.py`
- `src/holodeck/optimizer/agents/critic.yaml`
- `tests/unit/optimizer/test_textual.py`

**Estimated scope:** Medium (3 files; the core change).

---

## Task 3: Loop integration, e2e, and the semantics-shift docs

**Description:** Prove the assembled behaviour end-to-end and document the
budget-semantics change. Add a loop-level test driving the iterative textual
proposer against a scorer with a known gradient: assert the textual phase runs
multiple steps up to `max_trials`, that `patience` halts a flat (non-improving)
chain, and that the final `best_agent` is never worse than the single-step
baseline (loop revert holds). Extend the e2e stub-backend case so iterative
textual steps measurably reduce loss across a phase. Finally, record in
`docs/guides/evaluations.md` and the relevant schema field descriptions that
`textual_phase.max_trials`/`patience` now mean "refinement steps on the axis"
rather than "across axes."

**Acceptance criteria:**
- [ ] Integration test: iterative textual phase runs ≥2 steps when budget allows;
      `patience` stops a flat chain; `best_agent` never regresses below the
      single-step result.
- [ ] e2e stub case shows loss decreasing across iterative textual steps.
- [ ] `docs/guides/evaluations.md` documents the shifted `textual_phase` semantics.
- [ ] Schema field descriptions for `textual_phase.{max_trials,patience}` reflect
      the new meaning (no schema *shape* change — description text only).

**Verification:**
- [ ] `pytest tests/unit/optimizer/test_loop.py tests/integration/test_optimize_e2e.py -n auto`
      passes.
- [ ] `make test` (full, parallel) green; `make format && make lint && make type-check` clean.
- [ ] Manual: `holodeck test optimize` on a 1-axis stub shows ≥2 `phase: textual`
      trials in a cycle in `report.md`.

**Dependencies:** Task 2.

**Files likely touched:**
- `tests/unit/optimizer/test_loop.py`
- `tests/integration/test_optimize_e2e.py`
- `docs/guides/evaluations.md`
- `schemas/agent.schema.json` (description text only, if the field is described there)

**Estimated scope:** Medium (3–4 files).

---

## Checkpoints

### Checkpoint A — after Task 1 (Contract)
- [ ] `make test-unit` green; numeric + loop behaviour unchanged.
- [ ] `make type-check` clean across protocol, implementers, call sites.

### Checkpoint B — after Task 2 (Core)
- [ ] Momentum, context-source, and multi-step continuation proven by unit tests.
- [ ] `>1 axis` fallback verified; existing textual tests still green.
- [ ] No secret-bearing field reaches the Critic/Applier prompt (asserted).

### Checkpoint C — after Task 3 (Complete)
- [ ] Full `make test` green; format/lint/type-check clean.
- [ ] Budget-semantics shift documented in guide + schema descriptions.
- [ ] Manual 1-axis run shows multi-step textual trials.
- [ ] Human review before merge.

## Out of scope (do not implement here)

- Multi-axis iterative ordering (decision 4 defers it — fallback only).
- Repeats (`k>1`), train/holdout split, variance-aware acceptance.
- Few-shot demonstration optimization (named #1 v2 item in `optimizer.md`).
- `rebase-on-best` as the default (documented fallback only).
- Any new config knob or schema *field* (decision 3: reuse existing).
