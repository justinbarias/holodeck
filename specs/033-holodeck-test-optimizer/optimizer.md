# Intent: `holodeck test optimize`

**Status:** Confirmed (interview, 2026-05-29)
**Feeds:** `specs/033-holodeck-test-optimizer/`
**Why this file exists:** The original design (2026-05-16) framed the optimizer as
"SGD/AdamW-inspired" but its mechanism was single-step variant search from a fixed
baseline — improvements never compounded. A stress-test interview surfaced that gap
plus several downstream inconsistencies. This record is the corrected baseline the
spec rewrite is verified against.

## Intent

- **Outcome:** A *compounding* coordinate-descent optimizer for HoloDeck agents that
  stacks accepted instruction edits and hyperparameter wins into one improved
  `agent.yaml`, with a full per-trial audit trail.
- **User:** HoloDeck agent authors who currently hand-tune (change knob → run eval →
  eyeball score → change another knob) and want that outer loop automated.
- **Why now:** The test runner already emits normalized per-metric signal
  (`metric_results[].score`, `pass_rate`, per-metric averages, multi-turn evals).
  The missing piece is the loop that reads it and proposes the next config.
- **Success:** A baseline→best loss improvement that is *real* (clears measured
  noise, holds on holdout) and *adoptable* (copy-paste candidate `agent.yaml`),
  produced within a bounded budget.
- **Constraint:** Reuse `TestExecutor` unchanged; never mutate the original
  `agent.yaml`; every trial auditable; ingest once per run.

## Decisions that correct the 2026-05-16 design

1. **Compounding via coordinate descent.** `best_candidate` advances the baseline
   after each accept. The per-trial Strategy Router (numeric-vs-textual every step)
   is replaced by a **phase scheduler**: sweep numeric → sweep textual → repeat
   until a full cycle produces zero accepts.

2. **Fresh TPE study per numeric phase.** A textual edit changes the objective the
   numeric axes are scored against, so observations from a prior prompt are stale.
   Each new numeric phase starts a new Optuna study. The original single immortal
   study (valid only because the old design never compounded) is removed.

3. **Two-level budgets.** Per-phase stop (axis converged / no improvement in K) plus
   global stop (a full numeric→textual cycle yields zero accepts). The flat
   `max_iterations` cap is restructured so TPE warmups don't starve the textual
   passes that carry the largest single wins.

4. **Repeated trials, one global `repeats` knob.** Each trial config runs k times;
   loss is the mean over repeats. Budget and cost math multiply by k.

5. **Variance-aware acceptance.** The spread across the k repeats sets the accept
   bar — an improvement must clear the noise actually measured this run — replacing
   the fixed `holdout_tolerance: 0.02` / `ε=1e-4` thresholds that sat *below* the
   single-case-flip granularity of a 15–35 case eval set. Lenient at low k (the std
   estimate is itself noisy; a strict gate would reject real wins).

6. **Honest rename + reframe.** "prompt optimizer" → **"instruction & hyperparameter
   optimizer."** In current industry usage (DSPy/MIPRO), "prompt optimization"
   implies *few-shot demonstration* selection/bootstrapping, which the MVP does not
   do. Renaming avoids overclaiming.

### Smaller corrections folded into the rewrite

- **Errored metric runs** are *excluded and flagged*, not scored as `0.0` (scoring a
  transient API error as a total failure injects noise into the loss).
- **Run-id reproducibility** is reworded to a *config*-reproducibility claim, not
  bit-for-bit replay — stochastic repeats plus LLM non-determinism break exact
  reproduction.
- **Sampling axes (`temperature`, `top_p`)** stay in the MVP — defensible *because*
  repeats average their noise; they would have been indefensible at k=1.

## Out of scope (MVP)

- **Few-shot demonstration optimization** — the highest-leverage lever in the
  dominant industry framework (DSPy bootstraps demonstrations from training examples
  with a pass/fail oracle, which HoloDeck's `train_cases_file` + graders already
  provide). Deferred deliberately: it needs a first-class, individually-selectable
  `demonstrations` slot on the agent schema, backend prompt injection, and a third
  *selector* proposer — its own design pass. **Named the #1 v2 roadmap item**, not
  "v2 if ever."
- Ingestion-time axes, K-fold cross-validation, parallel trials, population/
  evolutionary methods — unchanged from the original deferral list.

## Industry-fit summary

The instruction-text side sits squarely in the TextGrad / OPRO / ProTeGi lineage
(natural-language "gradient" → applied edit). The numeric side is standard Bayesian
HPO (Optuna TPE). The combination via coordinate descent is the honest way to make
the two coexist without TPE fitting a moving target. The known gap versus industry
is demonstration optimization, addressed by scope-cut + rename + v2 headline above.
