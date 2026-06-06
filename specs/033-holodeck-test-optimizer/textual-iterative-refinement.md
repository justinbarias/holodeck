# Spec: Iterative Textual Refinement (multi-step TGD)

**Status:** Draft (interview, 2026-06-02)
**Feeds:** `specs/033-holodeck-test-optimizer/` (extends the textual phase of the MVP)
**Depends on:** the shipped MVP (`plan.md`, `tasks.md`) and the minimized-loss objective.
**Why this file exists:** The MVP's textual phase implements a *single* TextGrad step —
one Critic→Applier rewrite per declared axis per cycle — then `ask()` returns `None`
(`proposers/textual.py:160-165`) and `tell()` is a no-op (`:220`). With one declared
axis (the common case, incl. the `financial-assistant` sample) the textual phase runs
exactly **one trial per cycle**, and `textual_phase.max_trials`/`patience` never bind.
TextGrad/OPRO's entire value is in *iterating* the step with feedback. This spec adds
that iteration.

## Objective

- **Outcome:** Within a textual phase, refine a single instruction axis over multiple
  Critic→Applier steps, each step seeing the previous attempt and *why* it failed, until
  the phase's existing budget (`max_trials`/`patience`) stops it.
- **User:** HoloDeck agent authors optimizing a Claude/SK agent whose headroom is in the
  prompt, not the numeric knobs (e.g. the `financial-assistant` sample, 1 textual axis).
- **Why now:** The MVP proved the single-step Critic/Applier works; the trial audit
  (`numeric: 14, textual: 2`) made the truncation visible. The scorer already produces a
  fresh `TestReport` per candidate, so the failing-case "gradient" for the next step is
  **already computed** — iteration is nearly free in eval cost.
- **Success:** On a 1-axis agent with prompt-fixable failures, a textual phase takes
  ≥2 refinement steps when budget allows, `patience` halts a non-improving chain, and the
  run's final `best_agent` is never worse than the single-step baseline (loop revert).
- **Constraint:** No new config or schema; reuse `TestExecutor` and the loop's accept/
  revert unchanged; numeric proposer behaviour byte-identical.

## Decisions (confirmed 2026-06-02)

1. **Gradient = attempt + loss + failing cases.** Step N+1's Critic sees attempt N's
   rewritten text, its loss delta vs. best, and the **failing-case context re-derived
   from attempt N's own report** (`build_failing_context`, already in `textual.py:96`).
   This is the TextGrad-faithful natural-language gradient and costs no extra eval.

2. **Momentum in the proposer, revert in the loop.** Each step chains from the **last
   attempt** (TextGrad updates the variable in place; it does not re-base on a frozen
   best each step). Drift is bounded by the loop's existing mechanism: `best_agent` only
   advances on an accepted (loss-improving) trial and the run returns `best_agent`, so a
   wandering chain can never become the output, and `patience` kills it early. This is
   the exact split TextGrad uses (in-place `step()` + validation-gated keep-best).
   - *Documented fallback:* if k=1 eval noise makes the chain wander wastefully, a
     `rebase-on-best` variant (each step starts from `best_agent`'s text, rejected
     attempts as "avoid this" context only) is the conservative alternative. Not the
     default; revisit if observed.

3. **Budget = existing `textual_phase.max_trials`/`patience`.** No new knob. These fields
   finally bind: `max_trials` caps refinement steps for the axis; `patience` stops the
   axis after K consecutive non-improving steps. Their *semantics* shift from "across
   axes" to "refinement steps on the axis" — documented, no schema change.

4. **Single-axis scope.** This spec defines iteration for **one** textual axis. With >1
   declared textual axis the proposer falls back to today's one-pass-per-axis behaviour
   and logs that multi-axis iterative ordering is not yet supported. Multi-axis ordering
   (axis-by-axis-to-convergence vs. round-robin) is explicitly deferred.

## Mechanism

### Protocol change

`Proposer.tell` gains the scored report so the proposer can build the next gradient:

```python
def tell(self, proposal: Proposal, loss: float, accepted: bool,
         report: TestReport | None = None) -> None: ...
```

`report=None` covers the skipped-trial path (`loop.py:135`, where no candidate was
scored). `NumericProposer.tell` accepts and ignores it (behaviour unchanged). The loop's
scored-trial call site (`loop.py:157`) passes the `report` it already has from the scorer.

### `TextualProposer` state machine (single axis)

Per phase (`begin`):
- `_axis = axes[0]`; `_best_text = get_path(best_agent, axis.path)`; `_best_report`.
- `_last_text = None`, `_last_loss = None`, `_last_report = None`, `_history = []`.

`ask()` (returns a refinement each call; `None` only on Critic-signalled convergence):
- `source = _last_text if _last_text is not None else _best_text`  *(momentum)*
- `context = build_failing_context(_last_report or _best_report)`
- `gradient = critique(source, context, history=_history, last_loss=_last_loss)`
  — prompt names that this is refinement step *i*, shows the prior attempt's loss and
  that it was rejected, and asks for the single most impactful next change.
- `new_text, summary = apply_edit(source, gradient, axis.max_chars)`
- stash pending attempt; return `Proposal(textual_axis, new_text, edit_summary)`.

`tell(proposal, loss, accepted, report)`:
- `_last_text = proposal.new_text`; `_last_loss = loss`; `_last_report = report`.
- `_history.append((summary, loss, accepted))` (bounded length).

Stop conditions are owned by `loop._run_phase` (unchanged): `max_trials` reached or
`patience` consecutive non-accepts. The proposer only returns `None` early if the Critic
emits an explicit "no further improvement" signal (optional nicety).

### Loop interaction (unchanged except the `tell` arg)

`_run_phase` already loops `ask → apply → score → accept? → tell` under `max_trials`/
`patience`. The only edit is passing `report` into `tell`. `best_agent` advancement,
`min_delta` gate, and final-result selection are untouched — they are the revert gate.

## Commands / surface

No new CLI flags; `holodeck test optimize <agent.yaml>` is unchanged. The behavioural
change is internal to the textual phase. Config reuses
`evaluations.optimizer.textual_phase.{max_trials,patience}` with the shifted semantics
above. `report.md`/`trials.jsonl` already record every textual trial; iterations appear
as ordinary `phase: textual` trials with their `edit_summary` and `loss`.

## Code touchpoints

- `src/holodeck/optimizer/proposers/base.py` — extend `tell` signature; doc the contract.
- `src/holodeck/optimizer/proposers/textual.py` — iteration state, momentum source,
  history-aware Critic prompt, `tell` stores report; `>1 axis` fallback + log.
- `src/holodeck/optimizer/proposers/numeric.py` — `tell` accepts/ignores `report`.
- `src/holodeck/optimizer/loop.py` — pass `report` into the scored-trial `tell` call.
- `src/holodeck/optimizer/agents/critic.yaml` (+ maybe `applier.yaml`) — prompt accepts
  prior-attempt/loss context for the gradient.

## Testing strategy

- **Unit (`tests/unit/optimizer/test_textual.py`)**, fake `invoker` returning scripted
  JSON (deterministic, no network):
  - First `ask` uses `best_text` + `best_report` failing context.
  - After `tell(loss, accepted=False, report=r2)`, second `ask` chains from the **last
    attempt's text** (momentum, not best) and uses `r2`'s failing context.
  - `ask` keeps returning proposals across iterations (not `None` after one).
  - `>1 axis` → one-pass-per-axis fallback + the log line.
- **Loop integration (`test_loop.py`)**: a textual proposer + a scorer with a known
  gradient; assert the textual phase runs up to `max_trials`, `patience` halts a flat
  chain, and `best_agent` never regresses below the single-step result.
- **e2e (`tests/integration/test_optimize_e2e.py`)**: extend or add a stub-backend case
  where iterative textual steps reduce loss across a phase.
- **Determinism:** all LLM calls stubbed via the fake invoker; assert per-step inputs,
  not LLM phrasing.

## Boundaries

**Always**
- Reuse `TestExecutor` unchanged; never mutate the original `agent.yaml`.
- Keep `NumericProposer` behaviour byte-identical; the `tell` arg is additive/optional.
- Return `best_agent` (loop revert) as the result — a drifted chain is never the output.
- Log every refinement as a `TrialRecord` (already happens via `_run_phase`).

**Ask first**
- Adding any new config knob or schema field (decision 3 says reuse).
- Changing the `min_delta` accept gate or introducing repeats/holdout/variance bars
  (separate deferred work in `plan.md` Phase scope).
- Implementing multi-axis iterative ordering (decision 4 defers it).

**Never**
- Put resolved secrets into Critic/Applier prompts. The gradient uses instruction text +
  failing-case context (`test_input`, truncated `agent_response`, metric names/scores) —
  never credentials; keep it that way when adding the prior-attempt context.
- Let the proposer's internal momentum chain leak into `best_agent` except through the
  loop's accept gate.

## Out of scope

- Multi-axis iterative ordering (deferred; single-axis only here).
- Repeats (`k>1`), train/holdout split, variance-aware acceptance — already deferred by
  the MVP `plan.md`; this spec does not change the acceptance statistics.
- Few-shot demonstration optimization — still the named #1 v2 item in `optimizer.md`.
- `rebase-on-best` as default (kept as a documented fallback only).
