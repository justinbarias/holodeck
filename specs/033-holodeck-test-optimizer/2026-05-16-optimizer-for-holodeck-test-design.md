# `holodeck test optimize` — Instruction & Hyperparameter Optimizer — Design

**Status:** Draft for review (rev. 2026-05-29 — coordinate-descent rewrite)
**Date:** 2026-05-16
**Author:** justinbarias (with Claude)
**Related:** `specs/032-multi-turn-test-cases`, `specs/031-eval-runs-dashboard`, `docs/intent/optimizer.md`

> **Naming note.** This is an *instruction & hyperparameter* optimizer, not a "prompt
> optimizer" in the DSPy/MIPRO sense. It tunes instruction text, tool descriptions, and
> query-time numeric knobs. It does **not** select or bootstrap few-shot demonstrations —
> the highest-leverage lever in current prompt-optimization practice — which is the named
> #1 v2 item (see Out of Scope). The original 2026-05-16 draft called this a "prompt
> optimizer" and framed it as SGD/AdamW; both were corrected in the 2026-05-29 rewrite.

## Motivation

HoloDeck users iterate on agents the same way ML researchers tuned models in 2010: change a knob, run the eval, eyeball the score, change another knob. The test runner already produces rich, normalized signal (`metric_results[].score`, `pass_rate`, `average_scores` per metric, multi-turn evaluations). What's missing is an outer loop that reads that signal and proposes the next configuration.

This spec adds `holodeck test optimize` — a **compounding coordinate-descent** optimizer that tunes the agent's parameters (instruction text, tool descriptions, query-time numeric hyperparameters) against a per-trial weighted-metric loss. It alternates *phases*: a numeric phase sweeps the hyperparameters with Bayesian search (Optuna TPE), then a textual phase improves the prompt with a TextGrad-style critic + applier subagent pair. Each accepted win is **folded back into the baseline** before the next phase, so improvements compound into a single better `agent.yaml`.

> **Why coordinate descent, not "SGD on weights".** There is no real gradient and no
> learning rate here. The earlier "treat parameters as weights, run SGD" framing was a
> metaphor that broke down in two ways: (a) the loop perturbed a *fixed* baseline every
> trial, so wins never compounded — it was single-step variant search, not optimization;
> and (b) interleaving textual edits under one long-lived TPE study makes the study fit a
> *moving* objective (a prompt edit changes what the numeric loss even means). Coordinate
> descent fixes both: compounding happens *between* phases (the baseline advances); valid
> joint Bayesian search happens *within* a numeric phase against a *frozen* prompt.

The key constraints driving the design:

1. **Reuse the existing test runner unchanged.** The optimizer is a wrapper around `TestExecutor`, not a rewrite of it.
2. **Every change is auditable.** No silent edits; every trial writes a structured rationale to disk.
3. **Honest methodology.** Explicit train/holdout split is required; the optimizer cannot accidentally optimize against the same cases it scores. Accept decisions clear *measured* noise (repeated trials), not a magic threshold.
4. **Compounding.** Accepted wins advance the baseline; the final candidate stacks the best prompt edits *and* the best hyperparameters, not whichever single perturbation scored best.
5. **YAML-first, like the rest of HoloDeck.** Optimizer config lives under `evaluations.optimizer:` in `agent.yaml`; CLI flags are short-lived overrides.
6. **Bound blast radius.** Original `agent.yaml` is never touched; candidates are written under `results/optimizer/<run-id>/`.

## Architecture overview

The optimizer is a phased loop wrapped around the existing test runner. Nothing inside `TestExecutor` changes; the optimizer treats it as a black box `(agent_config, test_cases) → TestReport`.

```
┌──────────────────────────────────────────────────────────────────────┐
│  holodeck test optimize  (new subcommand of `test`)                  │
│                                                                      │
│   Trial controller                                                   │
│     ├── reads optimizer config from agent.yaml                       │
│     ├── owns the running baseline (advances on every accept)         │
│     ├── owns the trial history (edit log + per-phase Optuna studies) │
│     ├── enforces budgets (per-phase + global; wall-clock / $)        │
│     └── runs PHASES via the Phase Scheduler                          │
│                                                                      │
│   Phase Scheduler  (replaces the old per-trial Strategy Router)      │
│     repeat until a full numeric→textual cycle yields zero accepts:   │
│       NUMERIC phase:                                                 │
│         • spin up a FRESH Optuna TPE study (prompt is frozen)        │
│         • ask/tell until per-phase stop (converged / no-improve-in-K)│
│         • fold best numeric point into the baseline                  │
│       TEXTUAL phase:                                                 │
│         • critic+applier passes against the (new) baseline           │
│         • each accepted edit compounds onto the prompt               │
│         • stop on no-accept-in-N                                     │
│                                                                      │
│   Per trial (either phase):                                          │
│     1. build candidate = apply proposal to CURRENT baseline          │
│     2. run candidate ×`repeats` against TRAIN (existing TestExecutor)│
│     3. loss = mean over repeats; also keep the spread (std)          │
│     4. if train-loss improves: run HOLDOUT (×`repeats`)              │
│     5. accept iff improvement clears measured noise (variance-aware) │
│     6. on accept: advance baseline + persist candidate               │
│     7. persist trial row + decision rationale (accept OR reject)     │
│                                                                      │
│   Proposers (pluggable)                                              │
│     ├── NumericProposer  → Optuna TPE study (one per numeric phase)  │
│     └── TextualProposer  → Critic subagent + Applier subagent        │
│                                                                      │
│   Loss function                                                      │
│     scalar = 1 − Σ wᵢ · scoreᵢ   (mean over `repeats`; errored       │
│                                    metric runs excluded + flagged)   │
│                                                                      │
│   Output writer                                                      │
│     results/optimizer/<run-id>/                                      │
│       trials.jsonl   (one row per trial, full audit)                 │
│       candidates/iter-NN.yaml                                        │
│       best.yaml -> candidates/iter-NN.yaml                           │
│       trajectory.csv                                                 │
│       report.md                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Two design points to call out:

- **The runner is reused as-is.** `TestExecutor.__init__` already accepts `agent_config: Agent` and `resolved_execution_config: ExecutionConfig` as injected parameters. The optimizer mutates the `Agent` model in memory per trial and reuses everything else.
- **The Phase Scheduler is the novel coordinator** (it replaces the per-trial Strategy Router from the original draft). Everything else is either standard HPO machinery (Optuna), existing HoloDeck (`TestExecutor`, evaluations, backend selector), or two new Claude subagents (Critic, Applier).

**Why phases instead of a per-trial router.** The original draft picked numeric-vs-textual *each step* and kept one long-lived Optuna study. That study's sample-efficiency claim depends on a *stationary* objective — but a textual edit changes the prompt, which changes the loss surface the numeric axes are scored against, so prior numeric observations become stale. Phasing makes each TPE study's lifetime exactly equal to a window where the prompt is frozen (stationary, valid), and lets compounding live in the hand-off *between* phases. A re-opened numeric phase against an improved prompt routinely finds *more* numeric headroom than the first one did, because the optimal hyperparameters move with the prompt.

## How it plugs into the existing test flow

### Today: `holodeck test run agent.yaml`

```
CLI: holodeck test run
  → ConfigLoader.load_agent_yaml(path) → Agent
  → _resolve_execution_config()        → ExecutionConfig
  → TestExecutor(
      agent_config_path=path,
      agent_config=<Agent>,                    ← injected
      resolved_execution_config=<ExecCfg>,
    ).execute_tests() → TestReport
  → _save_report() / _persist_eval_run()
  → sys.exit(0/1)
```

### With optimizer: `holodeck test optimize agent.yaml`

The new subcommand wraps the same flow inside a phased loop. **Nothing inside `TestExecutor` changes.**

```
CLI: holodeck test optimize
  → ConfigLoader.load_agent_yaml(path) → baseline Agent
  → _resolve_execution_config()        → ExecutionConfig
  → OptimizerConfig.from_agent(baseline)
       (loss weights, train/holdout files, budgets, repeats,
        numeric_axes, textual_axes, acceptance rule)
  → OptimizerLoop:
       baseline ← initial Agent
       trial 0: cand = baseline; measure train + holdout loss (×repeats)
                best_train, best_holdout ← trial 0 losses

       repeat (full cycle) until a numeric+textual cycle yields zero accepts
                                 OR a global budget trips:

         ── NUMERIC PHASE ───────────────────────────────────────────
         study = fresh TPESampler study        # prompt frozen → stationary
         while not per-phase stop (converged / no-accept-in-K / phase budget):
           optuna_trial, params = study.ask()
           cand   = apply_axes(baseline, params)           # from CURRENT baseline
           train  = mean_loss(cand, TRAIN, repeats)        # + std
           if improves(train.mean, best_train):
             hold = mean_loss(cand, HOLDOUT, repeats)
             accept = variance_aware_accept(train, hold, best_train, best_holdout)
           else:
             accept = False
           study.tell(optuna_trial, train.mean)
           TrialLogger.write(...)
           if accept:                                       # COMPOUND
             baseline ← cand
             best_train, best_holdout ← train.mean, hold.mean
             save_candidate(cand); update best.yaml

         ── TEXTUAL PHASE ───────────────────────────────────────────
         while not per-phase stop (no-accept-in-N / phase budget):
           failures = failing traces from the LATEST baseline train report
           gradient = await CriticSubagent.run(failures, current_text, history)
           if gradient.confidence < min_confidence: continue   # short-circuit
           edit     = await ApplierSubagent.run(gradient, current_text)
           cand     = apply_textual_edit(baseline, axis, edit)  # from CURRENT baseline
           train    = mean_loss(cand, TRAIN, repeats)
           if improves(train.mean, best_train):
             hold = mean_loss(cand, HOLDOUT, repeats)
             accept = variance_aware_accept(train, hold, best_train, best_holdout)
           else:
             accept = False
           TrialLogger.write(...)
           if accept:                                       # COMPOUND
             baseline ← cand
             best_train, best_holdout ← train.mean, hold.mean
             save_candidate(cand); update best.yaml
  → write report.md
```

The single most important change from the original draft: **`baseline ← cand` on every accept.** Both proposers build candidates from the *current* baseline, not a fixed origin, so a prompt edit accepted in a textual phase is present when the next numeric phase sweeps, and vice-versa. Wins stack.

### `_run_trial` and `mean_loss` — the only adapters needed

`_run_trial` runs the candidate against a case subset *once*; `mean_loss` repeats it `repeats` times and aggregates. The single-run adapter is unchanged from the original design:

```python
async def _run_trial(cand: Agent, cases: list[TestCaseModel]) -> TestReport:
    cand_with_subset = cand.model_copy(update={"test_cases": cases})
    executor = TestExecutor(
        agent_config_path=path,
        agent_config=cand_with_subset,
        resolved_execution_config=resolved,
        force_ingest=False,           # never re-ingest mid-optimization
        progress_callback=trial_progress_cb,
    )
    try:
        return await executor.execute_tests()
    finally:
        await executor.shutdown()


async def mean_loss(cand: Agent, cases, repeats: int) -> TrialLoss:
    """Run `repeats` independent sweeps; return mean loss + spread.

    repeats > 1 is what makes acceptance honest when the agent samples
    stochastically (temperature/top_p > 0): a single run's loss is a random
    variable, so we average it and keep the std to set the accept bar.
    """
    losses = []
    for _ in range(repeats):
        report = await _run_trial(cand, cases)
        losses.append(scalarize(report))
    return TrialLoss(mean=mean(losses), std=pstdev(losses), n=repeats, runs=losses)
```

`repeats` is a single global knob (`optimizer.repeats`, default `3`). It is *not* per-axis: the stochasticity comes from the agent's own sampling, not from which axis is being perturbed, so it is a property of the run. The total test-sweep cost of the whole optimization multiplies by `repeats` — budgets account for this (see Budget and termination).

### What's new vs reused

| Concern | Today | With optimizer |
|---|---|---|
| Agent config load | `ConfigLoader.load_agent_yaml` | **Reused** (loaded once as baseline) |
| Execution config resolution | `_resolve_execution_config` | **Reused** (resolved once, passed to every trial) |
| Test execution | `TestExecutor.execute_tests` | **Reused per trial, unchanged** |
| Backend init / tool ingest | `BackendSelector` + `_initialize_tools` | **Reused; ingest happens once on trial 0** (`force_ingest=False` thereafter) |
| Evaluations | `evaluators` dict | **Reused** — same `metric_results` feed the loss |
| Result persistence | `_save_report` + `_persist_eval_run` | **Skipped per trial; replaced** by `TrialLogger.write` |
| Train/holdout split | n/a | New: `agent.model_copy(update={"test_cases": subset})` per call |
| Loss + acceptance | n/a | New: `OptimizerLoop` |
| Proposers | n/a | New: `NumericProposer` (Optuna), `CriticSubagent` + `ApplierSubagent` |
| Trial output | n/a | New: `results/optimizer/<run-id>/` |

### One ingest, many trials — the key efficiency lever

The largest cost in financial-assistant is the convfinqa ingest (qdrant + contextual embeddings + 3,300-page PDF). The optimizer must not re-ingest per trial.

1. **Trial 0** runs with `force_ingest=False`; ingests only if the vector store is empty.
2. **Trials 1…N** also pass `force_ingest=False`. Same `agent_config_path` → same collection name → all trials hit the same already-populated qdrant collection.
3. **Numeric axes that affect retrieval** (`top_k`, `semantic_weight`, `keyword_weight`, `min_score`, `rrf_k`) are *query-time* — safe to vary across trials.
4. **Numeric axes that would require re-ingestion** (`chunking_strategy`, `max_chunk_tokens`, `contextual_embeddings`, `embedding_model`) are explicitly **excluded from v1** (see Out of Scope).

## Loss function and trial history

### Per-trial loss formula

Given a `TestReport` from `_run_trial(cand, cases)`:

```
L(report) = 1 − ( Σᵢ wᵢ · scorēᵢ(report) ) / Σᵢ wᵢ
```

- **`wᵢ`** — weight for metric `i`, from `evaluations.optimizer.loss` in `agent.yaml`. Metrics configured in `evaluations.metrics` but absent from `loss` get weight 0 (informational only).
- **`scorēᵢ`** — aggregate score for metric `i` across all test cases in this trial:
  - Numeric scores (already 0–1): **mean**.
  - Pass/fail metrics: **pass-rate** (mean of 0/1). ⚠️ The codebase's `ReportSummary.pass_rate` is a **percentage (0–100)**, so the scalarizer must divide it (and any pass-rate-derived score) by 100 before it enters the `[0,1]` loss. Failing to normalize makes the loss go negative and silently corrupts every accept decision — call this out in the loss unit tests.
  - Errored metric runs (score=null): **excluded from the aggregate and flagged**, not scored as `0.0`. A transient API/backend error is not evidence the candidate is bad; scoring it as a total failure would inject noise into the loss and could reject a good candidate. The excluded count is recorded per trial; if more than a configurable fraction (`max_errored_fraction`, default `0.25`) of a metric's runs errored, that trial's loss for that metric is marked **unreliable** and the trial cannot be accepted on that basis (logged with reason).
- **Per-turn metrics** (multi-turn cases): first averaged within the case, then averaged across cases. Avoids long conversations dominating short ones.
- **Across repeats:** each of the `repeats` sweeps produces one trial loss; the reported trial loss is their **mean**, and the **std** is carried into the acceptance rule (see below).

Loss range: `[0, 1]`. **0 = perfect on every weighted metric.** Optuna minimizes.

### Optional regularizers

Off by default. When set, added on top of the weighted-score loss:

```yaml
optimizer:
  penalties:
    cost_per_usd: 0.05      # add 0.05 per $1 of LLM spend per trial
    latency_per_sec: 0.001  # add 0.001 per second of mean test wall-time
```

```
L_total = L(report) + cost_per_usd · trial_usd + latency_per_sec · mean_latency_sec
```

`trial_usd` derived by **summing `token_usage` across `report.results[]`** (the per-case `TestResult.token_usage`) × published per-token rates, then × `repeats`. Note: `ReportSummary` does **not** carry a `token_usage` field — costs must be aggregated from the per-result tokens, not read off the summary. Mean latency from `result.execution_time_ms` per case.

### Trial history (the audit log)

Every trial — accepted *or rejected* — appends one row to `results/optimizer/<run-id>/trials.jsonl`:

```json
{
  "trial_id": 7,
  "timestamp": "2026-05-16T13:42:17Z",
  "cycle": 1,
  "phase": "textual",
  "axis_target": "instructions.inline",
  "baseline_before_path": "candidates/iter-05.yaml",
  "proposal_source": {
    "kind": "textual",
    "critic": {
      "model": "claude-sonnet-4-6",
      "failing_pattern": "Agent answers single-fact lookups from the table but skips intermediate-step calculations on multi-row questions.",
      "root_cause_hypothesis": "Prompt does not instruct the agent to verbalize the arithmetic before calling subtract/divide.",
      "suggested_change_direction": "Add a 'plan-then-call' instruction before tool invocation steps.",
      "confidence": 0.78,
      "citations": ["Single_MRO/2007/page_134.pdf-1", "Single_AAPL/2019/page_88.pdf-3"]
    },
    "applier": {
      "model": "claude-sonnet-4-6",
      "rationale": "Inserted a sentence after the tool list…",
      "edit_type": "insert",
      "diff_lines_added": 3,
      "diff_lines_removed": 0
    }
  },
  "candidate_path": "candidates/iter-07.yaml",
  "repeats": 3,
  "train": {
    "report_summary": { "pass_rate": 82.0, "average_scores": {} },
    "loss_components": { "numeric": 0.78, "turn_program_equivalence": 0.91 },
    "penalty_components": { "cost_per_usd": 0.011, "latency_per_sec": 0.004 },
    "loss": 0.184,
    "loss_std": 0.011,
    "loss_runs": [0.179, 0.184, 0.189],
    "errored_runs_excluded": 0,
    "duration_sec": 141.6,
    "trial_usd": 0.66
  },
  "holdout": {
    "loss": 0.196,
    "loss_std": 0.014,
    "loss_runs": [0.182, 0.196, 0.210],
    "report_summary": { "pass_rate": 80.0, "average_scores": {} }
  },
  "decision": {
    "best_train_mean_before": 0.221,
    "train_improvement": 0.037,
    "noise_bar": 0.013,
    "improvement_clears_noise": true,
    "holdout_regression": 0.012,
    "holdout_within_noise": true,
    "accepted": true,
    "reason": "train improved 0.037 ≥ noise bar 0.013 (1×pooled std); holdout regressed 0.012, within its 0.014 spread — accept + advance baseline"
  },
  "study_state": {
    "cycle": 1,
    "phase": "textual",
    "numeric_study_id": null,
    "tpe_n_observations_this_study": null,
    "best_trial_id": 7
  }
}
```

The row is the audit trail. Anyone reading `trials.jsonl` can answer: *what changed, why it was tried, what it scored on train and holdout, why it was accepted or rejected.* No state lives outside this log.

**Cost/duration field semantics.** `train.trial_usd` and `train.duration_sec` cover the **train sweeps only, already ×`repeats`** (so `0.66 = 3 × 0.22`, `141.6s ≈ 3 × 47s`). Holdout sweeps, when run, add their own cost on top; the *whole-trial* total (train + holdout, ×repeats) is what `trajectory.csv`'s `trial_usd`/`duration_sec` columns report and what the `max_usd` global budget accumulates. The two views differ on purpose: the JSONL separates train from holdout so per-axis cost attribution is exact; the CSV is the flat total for plotting.

### What history buys the loop

| Use | Mechanism |
|---|---|
| TPE sample-efficiency (within a phase) | Each numeric phase's Optuna study replays *its own phase's* observations on resume; studies are never shared across phases |
| Per-phase stop | `mean(Δloss)` over the phase's trials; stop when the axis stops paying out |
| Critic context | Critic subagent is given the last 3 *rejected* textual trials so it doesn't re-propose ideas that already failed |
| Global early stop | Stop when a full numeric→textual cycle yields zero accepts (plus per-phase patience) |
| `report.md` synthesis | Loss trajectory, per-phase + per-axis attribution |
| `--resume` | Replay rows: advance baseline through accepted trials, rebuild each numeric phase's study from *that phase's* rows, carry forward bests |

### What we are deliberately *not* doing

- No surrogate LLM judge of "trial quality" beyond configured metrics. The loss function is exactly what the YAML says.
- No per-test gradient. The loss is a scalar over the trial; the critic gets *raw failures*, not per-axis gradients.
- No reward shaping. A score of 0.7 from a numeric metric is just 0.7. No squaring the gap to penalize harder.
- No Bayesian model over textual proposals. TPE only models numeric axes — and only within a single phase against a frozen prompt.
- No single long-lived study across the whole run. A study's observations are only valid while the prompt is frozen; each numeric phase gets a fresh study.

## Numeric proposer (Optuna TPE)

### Setup

Each numeric axis from `agent.yaml` becomes a dimension in the search space:

```yaml
axes:
  numeric:
    - path: model.temperature                              ; range: [0.0, 1.0]
    - path: model.top_p                                    ; range: [0.5, 1.0]
    - path: tools[name=convfinqa_archive].top_k            ; range: [3, 20]; type: int
    - path: tools[name=convfinqa_archive].semantic_weight  ; range: [0.2, 0.8]
```

Optuna calls each axis a *parameter*. A *trial* is one point in joint space plus its observed loss. The *study* is the running collection of `(point, loss)` pairs.

### Why TPE over grid / random

- **Grid.** 4 axes × 5 levels = 625 trials. Each trial = one full test sweep against a real LLM. Non-starter past 2–3 axes.
- **Random.** Surprisingly competitive vs grid (Bergstra & Bengio 2012) but ignores everything you've already learned.
- **Bayesian (TPE).** Uses every observation. Within ~15–25 trials it concentrates on promising regions. Right point on the cost/value curve when each trial is expensive.

### What TPE actually does

TPE = Tree-structured Parzen Estimator. Sequential model-based optimization. The "model" is two density estimates over parameter space:

1. **`l(x)`** — density of points where loss was *good* (below quantile γ of observed losses).
2. **`g(x)`** — density of points where loss was *bad* (above γ).

Each iteration, TPE samples many candidate points from `l(x)` and picks the one maximizing `l(x)/g(x)` — "looks like the good points and unlike the bad ones."

```
phase trial 1–10: warmup, mostly random samples (study can't model anything yet)
phase trial 11+:  TPE.ask uses l/g to bias sampling toward the good region
                  — but never collapses entirely (always mixes in exploration)
```

Mental shorthand: **TPE is "I've seen N attempts *against this prompt*; here's where the next attempt is most likely to be better than the median."** Not exact gradient descent — there is no derivative — but converges to good regions much faster than random.

**Warmup cost is per phase, by design.** Because each numeric phase starts a fresh study (the prompt underneath changed), it re-pays the ~10-trial warmup each cycle. This is the price of correctness: a study carried over from a previous prompt would be modelling a loss surface that no longer exists. The per-phase numeric budget must be set with this warmup in mind (a phase budget below ~`n_startup_trials` does no Bayesian work at all — it's pure random search). The phase scheduler logs a warning if a numeric phase budget is smaller than the warmup.

### Explainability for numeric trials

TPE doesn't produce a per-axis derivative, but it *does* expose, for any candidate, the per-axis density values. We log them:

```json
"proposal_source": {
  "kind": "numeric",
  "numeric_study_id": "numeric-phase-2",
  "params": { "model.temperature": 0.30, "model.top_p": 0.80, "top_k": 8, "semantic_weight": 0.55 },
  "tpe_quantile": 0.25,
  "candidates_evaluated": 24,
  "selected_candidate_score": 1.83,
  "axis_marginals": {
    "model.temperature":  { "value": 0.30, "good_density": 2.1, "bad_density": 0.4 },
    "top_k":              { "value": 8,    "good_density": 2.4, "bad_density": 0.3 },
    "semantic_weight":    { "value": 0.55, "good_density": 1.6, "bad_density": 0.7 }
  }
}
```

The raw **`params`** dict (exact suggested value per axis) and **`numeric_study_id`** are mandatory on every numeric row — they are what `--resume` needs to reconstruct the phase's study via `study.add_trial(params, value=train_loss)`. The `axis_marginals` are explainability only; Optuna cannot rebuild a study from densities. The trial row also answers "why this point?" — *"TPE identified `top_k=8` and `semantic_weight=0.55` as concentrated in the good region; ratio of good-density to bad-density was 1.83× the median."*

### Plug-in

A `NumericProposer` instance wraps **one** study and is created fresh at the start of each numeric phase (`NumericProposer(axes, seed, phase_index)`). It never outlives the phase.

```python
# src/holodeck/optimizer/proposers/numeric.py
import optuna

class NumericProposer:
    def __init__(self, axes: list[NumericAxis], seed: int, phase_index: int):
        self.axes = axes
        self.phase_index = phase_index
        # Fresh study per numeric phase. Seed is offset by phase so repeated
        # phases don't draw the identical warmup sequence while staying
        # deterministic for --resume.
        self.study = optuna.create_study(
            direction="minimize",
            study_name=f"numeric-phase-{phase_index}",
            sampler=optuna.samplers.TPESampler(
                seed=seed + phase_index,
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
            ),
        )

    def propose(self) -> tuple[optuna.Trial, dict[str, Any]]:
        trial = self.study.ask()
        params: dict[str, Any] = {}
        for axis in self.axes:
            if axis.type == "float":
                params[axis.path] = trial.suggest_float(axis.path, axis.lo, axis.hi)
            elif axis.type == "int":
                params[axis.path] = trial.suggest_int(axis.path, axis.lo, axis.hi)
            elif axis.type == "categorical":
                params[axis.path] = trial.suggest_categorical(axis.path, axis.choices)
        return trial, params

    def tell(self, trial: optuna.Trial, loss: float) -> None:
        self.study.tell(trial, loss)

    def explain(self, trial: optuna.Trial) -> dict:
        """Return marginals + ratios for the audit log."""
        ...
```

The numeric-phase driver is then a tiny adapter over the *current* baseline:

```python
proposer = NumericProposer(self.numeric_axes, seed, phase_index)
while not phase_stop:
    optuna_trial, params = proposer.propose()
    cand = apply_axes(self.baseline, params)     # CURRENT baseline (compounded)
    train = await mean_loss(cand, train_cases, repeats)
    proposer.tell(optuna_trial, train.mean)
    ... accept/reject ...
    if accept:
        self.baseline = cand                     # COMPOUND before next ask
```

Note `self.baseline = cand` inside the phase: because TPE proposes *deltas to the axes* and `apply_axes` always starts from the current baseline, an accept mid-phase compounds within the phase too. The study's observations remain valid because the *prompt* (the thing TPE doesn't model) is frozen for the whole phase.

`--resume`: read `trials.jsonl`; for each numeric phase, filter that phase's rows (`phase=="numeric" && cycle==c`) and `study.add_trial(...)` them into a fresh study for that phase before resuming the phase that was in flight. Studies are reconstructed per phase, never merged.

### Choices made explicitly

- **TPESampler over GPSampler.** TPE handles mixed continuous/integer/categorical natively; GP is for purely continuous low-dimensional setups.
- **`multivariate=True`** — correlations between axes are real (e.g. `temperature` and `top_p`).
- **Fresh study per numeric phase.** A study is valid only while the prompt is frozen; re-opening numeric after a textual phase starts a new study. This is the core correctness decision of the rewrite.
- **Sequential, not parallel.** Optuna supports parallel `ask`/`tell` but our trials hit shared resources. v2.
- **Seed defaults to 42, surfaced in the run-id; offset per phase (`seed + phase_index`)** so repeated numeric phases explore differently yet reproducibly.

### References

- Bergstra et al., 2011. [Algorithms for Hyper-Parameter Optimization](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).
- Bergstra & Bengio, 2012. [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).
- Optuna docs. [TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [Efficient Optimization Algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).
- Akiba et al., 2019. [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902).

## Textual proposer (TextGrad-style critic + applier)

### Intuition: natural-language gradients

In NN backprop, the gradient at a parameter answers: *"if you nudge me in this direction, the loss will go down."* For text, the equivalent is a sentence: *"the prompt is failing on multi-step arithmetic because it doesn't tell the agent to plan the calculation before invoking tools — add a planning instruction."*

That sentence is the gradient. It shares three properties with a real gradient:

1. **Local.** About *this* prompt's *current* failures, not abstract advice.
2. **Directional.** Not "make the prompt better" but specifically "add planning."
3. **An input to a step function.** Something else takes (current_text, gradient) → new_text — the analog of `θ ← θ − η·∇L`.

This is the TextGrad framing (Yuksekgonul et al., 2024). The split into two LLM calls — one that *describes* the gradient, one that *applies* it — makes the system explainable and ablatable.

### The two subagents

Both are Claude subagents (`AgentDefinition`), called via the Claude Agent SDK with structured output. Auto-installed on first run at `.holodeck/optimizer/agents/{critic,applier}.yaml`; user-editable.

**Critic subagent** — produces the gradient.

```yaml
name: optimizer-critic
description: Reads failing test traces and produces a structured natural-language gradient.
model:
  provider: anthropic
  name: claude-sonnet-4-6
  temperature: 0.2
instructions:
  inline: |
    You are a senior prompt engineer auditing a failing AI agent.
    You will receive:
      - The current text of the parameter being optimized
      - 5–10 failing test traces (input, expected, actual, tool calls)
      - History of previously rejected gradients for this same parameter

    Your job: identify the single highest-leverage failure pattern and explain
    the direction the text needs to move. You do NOT rewrite the text. You
    produce a diagnosis, in JSON, with the following fields. Be specific.
    Cite test names.
response_format:
  type: json_schema
  schema:
    type: object
    required: [failing_pattern, root_cause_hypothesis, suggested_change_direction, confidence, citations]
    properties:
      failing_pattern:        { type: string }
      root_cause_hypothesis:  { type: string }
      suggested_change_direction: { type: string }
      confidence:             { type: number, minimum: 0, maximum: 1 }
      citations:              { type: array, items: { type: string } }
      avoid:                  { type: array, items: { type: string } }
```

**Applier subagent** — performs the step.

```yaml
name: optimizer-applier
description: Takes a current text and a critic's gradient, produces the new text.
model:
  provider: anthropic
  name: claude-sonnet-4-6
  temperature: 0.4
instructions:
  inline: |
    You receive:
      - The current text of a parameter (system prompt or tool description)
      - A critic's gradient
      - A maximum length constraint

    Apply the gradient as a focused edit. Prefer minimal, targeted changes
    over rewrites. Justify the edit in one sentence. Output JSON.
response_format:
  type: json_schema
  schema:
    type: object
    required: [edit_type, rationale, new_text, diff_summary]
    properties:
      edit_type:    { enum: [insert, replace, delete, restructure] }
      rationale:    { type: string }
      new_text:     { type: string }
      diff_summary: { type: string }
```

A textual phase runs this proposal repeatedly against the **current** (compounding) baseline until it stops accepting. `current_text` and `latest_train_report` are always read from the baseline *as it stands now* — so after one edit is accepted and folded in, the next critic call diagnoses the *already-improved* prompt's remaining failures, not the phase-entry prompt's. This is how textual edits compound within a phase.

```
TextualProposer.propose(current_text, latest_train_report, history):

  1. failures = [r for r in latest_train_report.results if not r.passed][:10]
     if not failures:
        return None  # nothing left to fix on this axis → end the textual phase

  2. gradient = await critic_subagent.run(
        current_text=current_text,                 # CURRENT baseline's text
        failing_traces=[_compact(r) for r in failures],
        previous_gradients=[h.critic for h in history[-3:]
                            if not h.decision.accepted])

  3. if gradient.confidence < min_confidence:       # default 0.4
        log "low-confidence gradient, skipping textual trial (cheap short-circuit)"
        return None                                  # ends phase if it recurs (no-accept-in-N)

  4. edit = await applier_subagent.run(
        current_text=current_text,
        gradient=gradient,
        max_chars=axis.max_chars)

  5. return TextualProposal(
        new_text=edit.new_text,
        audit={ "critic": gradient, "applier": edit })
```

A textual phase iterates over the configured textual axes (e.g. `instructions.inline`, then each `tools[X].description`), one target per trial. The phase ends when a full pass over the axes produces no accept (`no-accept-in-N`) or the phase budget trips. There is no "fall back to numeric" — phase hand-off is owned by the scheduler, not the proposer.

### Persisting the textual edit

When a textual trial is accepted:

- **`instructions.file` path:** Applier writes the new prompt to `results/optimizer/<run-id>/instructions/system-prompt.iter-NN.md`. The candidate `agent.yaml` (at `results/optimizer/<run-id>/candidates/iter-NN.yaml`) gets `instructions.file: ../instructions/system-prompt.iter-NN.md`. The relative path resolves correctly when the candidate is loaded directly via `holodeck test run candidates/iter-NN.yaml` (HoloDeck resolves `instructions.file` relative to the agent.yaml's directory). On adoption, the user copies the prompt file back into their project layout and updates `instructions.file` accordingly — the "How to adopt" section in `report.md` provides the exact `cp` and `sed` commands.
- **`tools[X].description` (inline):** updated inline in the candidate `agent.yaml`. No separate file.
- **`instructions.inline` (rare; user explicitly chose inline):** updated inline in the candidate `agent.yaml`.

### Why two stages, not one

A single "rewrite this prompt to fix these failures" call collapses diagnosis and remedy. You lose:

- Ability to reject low-confidence diagnoses without spending a full trial. Step 3 above short-circuits before the Applier even runs, before tests are run, saving real money.
- Ability to feed back rejected diagnoses. "Here are 3 things you've already proposed that didn't work" is the closest thing to learning the textual proposer has.
- Ablation. Is the optimizer failing because the critic is wrong, or because the applier rewrites poorly? Two stages let you swap one and hold the other constant.

### What we are deliberately *not* doing

- **No few-shot demonstration optimization.** This is the single highest-leverage lever in DSPy/MIPRO-style prompt optimization — automatically selecting and *bootstrapping* `input→tool-calls→output` demonstrations from training examples that pass the grader, then choosing which subset/order goes in the prompt. The MVP does not do it, which is precisely why this feature is named an *instruction & hyperparameter* optimizer rather than a "prompt optimizer." It is the **#1 v2 item** (see Out of Scope), not "v2 if ever" — HoloDeck's `train_cases_file` + graders already supply the example pool and pass/fail oracle that bootstrapping needs; what's missing is a first-class `demonstrations` slot on the agent schema, backend injection, and a selector proposer.
- No co-optimization across multiple textual axes per trial. One trial perturbs one textual target.
- No evolutionary / population methods.
- No self-judging. Critic gets failure traces; never grades the new prompt.

### Cost reality

```
Critic call:   ~2k input + ~400 output tokens          (once per textual trial)
Applier call:  ~2k input + (axis.max_chars / 4) output  (once per textual trial)
Test sweep on TRAIN cases × repeats  (the expensive part)
```

For financial-assistant: a single test sweep is ~$0.20–$0.40. **With `repeats: 3` (default), a TRAIN-only trial costs ~$0.60–$1.20**, and a trial that also runs HOLDOUT roughly doubles again. The two critic/applier calls add ~$0.02 — negligible against the repeated sweeps. So the cost model changed shape from the original draft: the dominant lever is now `repeats × (trials per phase) × cycles`, not `max_iterations` alone. Budget guidance:

- `repeats` multiplies *everything*. Lowering it from 3→1 cuts cost ~3× but removes the noise averaging that makes acceptance honest on stochastic agents — only do it if the agent is effectively deterministic (temperature 0 and no sampled tools).
- A numeric phase pays its ~10-trial warmup every cycle; keep numeric phase budgets ≥ warmup or the phase is just random search.
- Textual trials are cheap to *reject* (the confidence short-circuit skips the sweep entirely), so a textual phase's cost is dominated by its *accepts*.

### References

- Yuksekgonul et al., 2024. [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496).
- Khattab et al., 2024. [DSPy](https://arxiv.org/abs/2310.03714) and [MIPRO](https://arxiv.org/abs/2406.11695).
- Pryzant et al., 2023. [Automatic Prompt Optimization with "Gradient Descent" and Beam Search (ProTeGi)](https://arxiv.org/abs/2305.03495).
- Yang et al., 2023. [Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2309.03409).
- Anthropic. [Claude Agent SDK — AgentDefinition](https://docs.anthropic.com/en/docs/claude-code/sdk).

## Train/holdout split + acceptance rule

### The split

```yaml
optimizer:
  train_cases_file: data/convfinqa_train.yaml
  holdout_cases_file: data/convfinqa_holdout.yaml
  repeats: 3                  # runs per trial; loss = mean, spread = accept bar
  accept_sigma: 1.0           # improvement must exceed accept_sigma × pooled std
  min_holdout_cases: 5        # hard floor (was a soft warning); below → error
  holdout_eval_policy: on_train_improve
```

Constraints enforced at load time:

- Both files exist and parse as `list[TestCaseModel]` (same schema as `test_cases_file`).
- Disjoint by `test_name`. Any overlap → error with the offending names.
- Both non-empty. **Holdout `< min_holdout_cases` (default 5) → error**, not a warning. Below the floor the variance estimate is too weak for the acceptance rule to mean anything; the original draft let this through as a warning, which let users make sub-noise accept decisions unknowingly.
- Both reference the same metric set (or holdout is a subset). Missing metrics → error at startup.

If only one file is provided → error: *"Optimizer requires both `train_cases_file` and `holdout_cases_file`. To run without a holdout, set `holdout_eval_policy: skip` and acknowledge the overfit risk."*

### The acceptance rule (variance-aware)

The original draft accepted on `train_loss < best_train_loss − ε` (`ε = 1e-4`) and a flat `holdout_tolerance = 0.02`. Both numbers sit *below the noise floor* of a real eval set: on a 15-case holdout, one case flipping pass→fail moves loss by `1/15 ≈ 0.067`, so a 0.02 gate and a 0.0001 ε are finer than the smallest change the data can express. We were making accept/reject decisions at a precision the measurement can't support. **Repeated trials fix this by measuring the noise directly and setting the bar from it.**

Each trial:

1. **Always run TRAIN ×`repeats`.** Compute `train.mean`, `train.std`.
2. **Decide whether to run HOLDOUT** based on `holdout_eval_policy`:

   | Policy | When holdout runs | Cost | Use case |
   |---|---|---|---|
   | `on_train_improve` (default) | Only when `train.mean` beats the noise bar vs `best_train` | Cheapest | Production default |
   | `every_trial` | Every trial | 2× the repeated sweep | Diagnostic / small test sets |
   | `skip` | Never | 1× | Acknowledged-overfit mode; flagged loudly in `report.md` |

3. **Compute the noise bar** from the spread actually measured this trial:

   ```
   pooled_std = sqrt(train.std² + best_train.std²)      # incumbent std is stored on the best
   noise_bar  = accept_sigma · pooled_std                # accept_sigma default 1.0
   ```

4. **Accept iff the improvement clears the noise on train AND holdout did not significantly regress:**

   ```
   (best_train.mean − train.mean) ≥ noise_bar
   AND
   (holdout.mean − best_holdout.mean) ≤ accept_sigma · sqrt(holdout.std² + best_holdout.std²)
   ```

   In words: the train gain must be at least one measured standard deviation (lenient by default), and the holdout must not have regressed by more than its *own* measured noise. There is no magic `0.02` — the bar is recomputed every trial from that trial's spread, so it is automatically correct at any eval-set size. **Why lenient (`accept_sigma = 1.0`, not 2.0):** at `repeats = 3` the std estimate is itself noisy, so a strict gate would have low power and reject real improvements. `1σ` is the honest-but-forgiving setting; raise `accept_sigma` for stricter, more expensive runs.

5. **Update bests on accept** — store the *distribution*, not just the mean, because the next trial's noise bar needs the incumbent's std:

   ```
   best_train   = train      # {mean, std, runs}
   best_holdout = holdout     # {mean, std, runs}
   best_candidate = candidate
   baseline       = candidate # COMPOUND: advance the baseline
   ```

   Bests (and their stds) only update on accept — not every time holdout is evaluated — so a high-train/low-holdout fluke can't tighten or loosen a future bar without first becoming the best.

### Decision rationale (audit row)

Every trial — accepted or rejected — writes the reasoning explicitly:

```json
"decision": {
  "best_train_mean_before": 0.221,
  "train_mean": 0.184,
  "train_std": 0.011,
  "train_improvement": 0.037,
  "pooled_train_std": 0.013,
  "accept_sigma": 1.0,
  "noise_bar": 0.013,
  "improvement_clears_noise": true,
  "holdout_mean": 0.196,
  "best_holdout_mean_before": 0.184,
  "holdout_regression": 0.012,
  "holdout_noise_bar": 0.014,
  "holdout_within_noise": true,
  "accepted": true,
  "reason": "train improved 0.037 ≥ noise bar 0.013 (1σ); holdout regressed 0.012 ≤ its 0.014 noise — accept + advance baseline"
}
```

Rejection example (improvement does not clear the noise it was measured against):

```json
"decision": {
  "best_train_mean_before": 0.184,
  "train_mean": 0.171,
  "train_std": 0.018,
  "train_improvement": 0.013,
  "pooled_train_std": 0.021,
  "accept_sigma": 1.0,
  "noise_bar": 0.021,
  "improvement_clears_noise": false,
  "accepted": false,
  "reason": "train 'improved' 0.013 but that is below the 0.021 noise bar (1σ) — indistinguishable from sampling noise, reject"
}
```

These rationales feed into the Critic for future textual trials, so the optimizer learns which kinds of edits overfit or fail to clear noise.

### What `--resume` does to the bests

1. Replay every accepted trial in order, advancing `baseline`, `best_train` (mean+std), `best_holdout` (mean+std), `best_candidate_path`.
2. For each numeric phase, replay *that phase's* numeric trials into a fresh Optuna study via `study.add_trial`. Studies are reconstructed per phase; never merged across phases.
3. Replay the last 3 *rejected* textual trials per axis into the Critic's `previous_gradients` memory.
4. Resume the in-flight phase, continuing from `len(trials) + 1`.

No special "resume bookkeeping" file. The audit log *is* the state.

### What this rule does *not* do

- No full statistical significance test (paired bootstrap on per-case scores). The variance-aware bar (improvement ≥ `accept_sigma`×pooled-std from `repeats`) is the cheap, honest version — it uses measured spread instead of a magic constant, but with small `repeats` it is not a publishable p-value. A proper paired bootstrap is a v2 upgrade.
- No K-fold rotation. Single train/holdout split is enough variance for v1.
- No automatic split if user only provides `test_cases_file`. Auto-split hides a methodological choice; explicit configuration is right for HoloDeck.
- No early-stop on holdout regression alone. Per-phase patience and the global dry-cycle stop are on accepted trials.

### Honest failure mode this design accepts

If `train_cases` and `holdout_cases` come from the same distribution (e.g. random split of convfinqa), holdout catches single-case overfitting but not *distributional* overfitting. The optimizer can produce a prompt that's great at "convfinqa-style table arithmetic questions" and terrible at production traffic. Mitigation is documentation: README and `report.md` both call out that holdout cases should resemble production traffic.

### Summary in one sentence

Train drives the proposer; holdout gates acceptance; acceptance requires the gain to clear the noise measured across `repeats`; every accept advances the baseline so wins compound; bests (mean+std) only update on accept; resume replays the audit log.

## State, outputs, and the report

### On-disk layout

Everything for one optimization run lives under `results/optimizer/<run-id>/`. The `<run-id>` is `YYYY-MM-DDTHH-MM-SS_<short-hash>`, where the hash is the first 8 chars of `sha256(baseline_agent_yaml + optimizer_config + seed)`. Same baseline + config + seed → same run-id, so `--resume` is unambiguous.

> **Reproducibility is config-level, not bit-for-bit.** The run-id identifies *the run's inputs* (baseline, config, seed) — it does **not** promise that two runs with the same id produce identical trajectories. With `repeats` averaging stochastic sampling and LLM backends that are non-deterministic even at temperature 0, trial losses vary run-to-run; the Optuna seed makes the *proposal sequence* reproducible, but not the *measured losses*. So "same inputs → same run-id" supports unambiguous resume and provenance, not exact replay. The original draft over-claimed this as reproducibility; it is reproducible *configuration*, replayable *decisions* (via the audit log), but not a deterministic objective.

```
results/optimizer/2026-05-16T14-22-08_a3f2c891/
├── run.json              ← run metadata (immutable after trial 0)
├── trials.jsonl          ← append-only audit log
├── trajectory.csv        ← flat per-trial metrics for plotting
├── candidates/
│   ├── iter-00.yaml      ← baseline snapshot
│   ├── iter-03.yaml      ← only accepted trials write a candidate
│   ├── iter-07.yaml
│   └── iter-12.yaml
├── instructions/
│   ├── system-prompt.iter-03.md   ← only accepted textual trials
│   ├── system-prompt.iter-07.md     that touched the prompt file
│   └── system-prompt.iter-12.md
├── best.yaml -> candidates/iter-12.yaml         (symlink)
├── best-instructions.md -> instructions/system-prompt.iter-12.md
└── report.md             ← human-readable summary, written at exit
```

**Why three artifact dirs:**

- `candidates/` — the runnable agent.yaml. User adopts via `cp candidates/iter-12.yaml ../../../agent.yaml`. Each is self-contained and points at its corresponding `instructions/` file via relative path.
- `instructions/` — versioned prompt files. Applier writes a new file per accepted textual trial; candidate's `instructions.file` points at it.
- `best.yaml` + `best-instructions.md` — symlinks to current leader. Updated atomically (write-and-rename) on every accepted trial (every accept advances the baseline, so every accept is the new leader).

**Numbering:** `iter-NN` matches `trial_id` in `trials.jsonl`. Skipped numbers tell the reader at a glance which trials weren't accepted. Because the baseline compounds, `iter-12.yaml` already contains every prior accepted edit (it *is* the running baseline at trial 12), so the latest accepted candidate is always the full stacked result — not a single perturbation.

### `run.json` — immutable header

Written once on trial 0; never modified. Lets `--resume` and post-hoc tools reconstruct context without parsing every JSONL row.

```json
{
  "run_id": "2026-05-16T14-22-08_a3f2c891",
  "started_at": "2026-05-16T14:22:08Z",
  "holodeck_version": "0.18.3",
  "baseline_agent_path": "../../../agent.yaml",
  "baseline_sha256": "a3f2c891b4...",
  "seed": 42,
  "optimizer_config": {
    "loss_weights": { "numeric": 0.7, "turn_program_equivalence": 0.3 },
    "penalties": { "cost_per_usd": 0.0, "latency_per_sec": 0.0 },
    "axes": { "numeric": [], "textual": [] },
    "repeats": 3,
    "accept_sigma": 1.0,
    "budget": {
      "global": { "max_cycles": 4, "max_minutes": 60, "max_usd": 15.0 },
      "numeric_phase": { "max_trials": 15, "patience": 5 },
      "textual_phase": { "max_trials": 10, "patience": 4 }
    },
    "holdout_eval_policy": "on_train_improve",
    "min_holdout_cases": 5
  },
  "train_cases_count": 35,
  "holdout_cases_count": 15,
  "subagents": {
    "critic":  { "model": "claude-sonnet-4-6", "agent_path": ".holodeck/optimizer/agents/critic.yaml" },
    "applier": { "model": "claude-sonnet-4-6", "agent_path": ".holodeck/optimizer/agents/applier.yaml" }
  }
}
```

### `trajectory.csv` — for plotting

Flat table. The existing dashboard and any external tool can consume it without parsing JSON. One row per trial.

```csv
trial_id,timestamp,cycle,phase,axis_target,train_mean,train_std,holdout_mean,best_train,best_holdout,noise_bar,accepted,trial_usd,duration_sec
0,2026-05-16T14:22:08Z,0,baseline,—,0.234,0.010,0.241,0.234,0.241,—,true,1.08,171.0
1,2026-05-16T14:22:51Z,1,numeric,model.temperature,0.221,0.014,,0.234,0.241,0.017,false,0.59,118.0
2,2026-05-16T14:23:30Z,1,numeric,top_k,0.215,0.009,0.243,0.215,0.241,0.013,true,1.05,168.0
3,2026-05-16T14:24:12Z,1,textual,instructions.inline,0.184,0.011,0.196,0.184,0.196,0.014,true,1.10,172.0
```

Notes on the columns:
- Empty `holdout_mean` means holdout wasn't run (so the row's `trial_usd`/`duration_sec` cover train ×`repeats` only — that's why row 1, a no-holdout reject, costs ~half of the holdout-bearing rows).
- `best_train`/`best_holdout` are the incumbent values **after** this trial's accept/reject decision (on a reject they equal the prior incumbent; on an accept they equal this trial's means). The JSONL `decision.best_train_mean_before` is the *pre*-decision counterpart.
- `noise_bar` is the per-trial accept threshold (`accept_sigma × pooled std`); a trial is accepted only when `best_train_before − train_mean ≥ noise_bar` (and holdout stays within its own noise).
- `train_std` and the `trial_usd`/`duration_sec` totals all reflect the `repeats` runs aggregated into each row; `trial_usd` here is the **whole-trial** total (train + holdout, ×repeats), unlike the JSONL which splits them.

### `report.md` — the artifact a user reads

Written on exit (success, budget exhaustion, Ctrl-C, or fatal error — `try/finally` guarantees it). The single thing you need to skim to decide *should I adopt this?*

Contains:

- Result summary: baseline → best loss, % improvement, exit reason, duration, total cost, `repeats` used.
- Trajectory: ASCII sparkline of `train_mean` and `holdout_mean` vs `trial_id`, with phase boundaries marked.
- Per-phase + per-axis attribution table: cycle, phase, trials, accepted count, Δloss contributed. Shows the coordinate-descent story (e.g. "numeric cycle 2 found more headroom after the prompt improved in textual cycle 1").
- Per-metric impact: baseline → best, per metric.
- Diff: baseline → best, both `agent.yaml` and `instructions/system-prompt.*.md` (the diff is the *full stack* of accepted edits, since the baseline compounded).
- Key proposer rationales for accepted textual trials.
- "How to adopt" section with copy-paste commands.
- Methodological notes: train/holdout sizes, `repeats`/`accept_sigma`, holdout policy, distributional-overfit warning, and the explicit note that reproducibility is config-level, not bit-for-bit.

### Streaming output during the run

One line per trial, grouped by phase — same visual idiom as `holodeck test run`. A phase header announces each numeric/textual block (and, for numeric, that a fresh study started):

```
── cycle 1 · numeric phase (fresh TPE study #1) ──────────────────────
[c1/n 2] top_k                 train 0.215 ±0.009 (▼0.019 > bar 0.013)  hold 0.243  accept ✓  $1.05  168s
[c1/n 3] semantic_weight       train 0.213 ±0.012 (▼0.002 < bar 0.017)  ——          noise ✗  $0.59  118s
── cycle 1 · textual phase ───────────────────────────────────────────
[c1/t 1] instructions.inline   train 0.184 ±0.011 (▼0.031 > bar 0.014)  hold 0.196  accept ✓  $1.10  172s
[c1/t 2] tools[subtract].descr train 0.184 (—)                          ——          crit  ✗  $0.014  3s  (low confidence)
── cycle 2 · numeric phase (fresh TPE study #2, vs improved prompt) ───
[c2/n 1] top_k                 train 0.171 ±0.010 (▼0.013 ≥ bar 0.013)  hold 0.190  accept ✓  $1.05  168s
```

Columns: cycle/phase + trial index, axis_target (truncated), `train_mean ± std` with delta-vs-best and the noise bar it was compared against, holdout_mean, outcome (`accept`/`noise`/`crit`-skipped), cost (×`repeats`), duration. `▼` = improvement. The explicit `> bar` / `< bar` makes the variance-aware decision legible at a glance — a reject labelled `noise` means the gain was real-looking but below the measured noise floor. `-q` suppresses these but still writes JSONL.

### What `holodeck test view` learns to do (v2 follow-up, not v1)

Existing dashboard renders eval-run JSON. Extending for optimizer runs is a small addition: detect optimizer run by presence of `run.json`, render `trajectory.csv` as a real chart, show per-axis attribution interactively, render diffs side-by-side, side panel with rationales. **Not in v1.** v1 ships file-based outputs only; dashboard support is a clean v2 follow-up because the file layout is stable.

### What we are deliberately *not* writing to disk

- No per-trial `TestReport` files. Each trial's full report could be tens of MB. We extract summary + loss components into JSONL and discard the rest. To get the full report for a specific trial, re-run via `holodeck test run candidates/iter-NN.yaml`.
- No raw Critic/Applier SDK responses. Their structured outputs are in `trials.jsonl`. Aggregate `trial_usd` is enough for budget tracking.
- No vector store snapshots. Qdrant collection is referenced by name; trials share it.

### Cleanup, retention, .gitignore

- **Default retention:** never auto-delete. User owns disk.
- **Suggested `.gitignore`:** `results/optimizer/` should be gitignored. Sample agent's existing `.gitignore` already covers `results/`.
- **`--keep-best-only`:** at exit, prunes `candidates/iter-*.yaml` and `instructions/system-prompt.iter-*.md` for non-best trials. Audit log and `best.yaml` preserved. Useful for CI.

### Recovery from mid-run crash

`trials.jsonl` and `candidates/` are append-only and atomic-write. If process crashes mid-trial:

- Partial trial: nothing written for it (the trial row is built up in memory and written only after holdout decision). Resume picks up from the last *complete* row.
- Symlink update: written atomically (`os.symlink` to a tempname, then `os.replace`). No torn state.
- `report.md`: only written on exit. After a crash, `--report-only` regenerates it from JSONL without doing more trials.

## CLI surface

`optimize` is a sibling subcommand of the existing `holodeck test` group:

```
holodeck test
  ├── run         (existing)
  ├── view        (existing)
  └── optimize    (NEW)
```

```
holodeck test optimize [AGENT_CONFIG] [OPTIONS]

Arguments:
  AGENT_CONFIG                  Path to agent.yaml. Defaults to ./agent.yaml.

Global budget overrides (override agent.yaml; CLI wins):
  --max-cycles INT              Hard cap on numeric→textual cycles.
  --max-minutes FLOAT           Wall-clock budget (whole run).
  --max-usd FLOAT               LLM-spend budget (whole run).

Per-phase budget overrides:
  --numeric-max-trials INT      Cap trials within each numeric phase.
  --numeric-patience INT        Early-stop a numeric phase after N non-accepts.
  --textual-max-trials INT      Cap trials within each textual phase.
  --textual-patience INT        Early-stop a textual phase after N non-accepts.

Run control:
  --repeats INT                 Runs per trial for noise averaging (default: 3).
  --accept-sigma FLOAT          Accept bar in pooled-std units (default: 1.0).
  --resume RUN_ID               Continue an existing run; replays trials.jsonl.
  --seed INT                    Override Optuna seed (default: 42; offset per phase).
  --report-only                 Skip trials; regenerate report.md from existing JSONL.
  --keep-best-only              Prune non-best candidates/instructions at exit.

Diagnostics:
  --dry-run                     Validate config, print resolved axes/budget/loss, exit 0.
  --print-baseline              Run trial 0 only, print loss + per-metric breakdown, exit 0.

Output:
  -o, --output-dir PATH         Override results/optimizer/<run-id>/ location.
  -q, --quiet                   Suppress per-trial progress lines (still writes JSONL).
  -v, --verbose                 Show Critic/Applier rationales inline as produced.
```

Exit codes:

- `0` — completed (budget exhausted, patience triggered, or `--dry-run`/`--report-only` succeeded).
- `1` — fatal error (config invalid, backend init failed, etc.).
- `2` — interrupted (Ctrl-C). Partial state intact; `--resume` works.

The command does **not** exit non-zero on "no improvement found." That's a legitimate outcome.

### Precedence

CLI flag → `agent.yaml` `optimizer:` block → built-in default. Standard precedence. CLI is for short-lived overrides during exploration; YAML is canonical for reproducible runs.

## Budget and termination

Budgets are **two-level**: per-phase stops decide when to *leave a phase*, global stops decide when to *end the run*. **Stop on first hit** at each level.

**Per-phase termination** (checked each trial; ends the current phase, hands back to the scheduler):

| Condition | Default | Trigger |
|---|---|---|
| `numeric_phase.max_trials` | 15 | trials in this numeric phase ≥ cap |
| `numeric_phase.patience` | 5 | no accept in last N trials of this phase |
| numeric convergence | — | TPE's best predicted expected-improvement (in loss units, same `[0,1]` scale as the loss) falls below `numeric_convergence_eps` (default `1e-3`). This is a *proposer*-side "nothing promising left to try" signal, distinct from acceptance — it is not the removed acceptance ε; it's a named, configurable knob. |
| `textual_phase.max_trials` | 10 | trials in this textual phase ≥ cap |
| `textual_phase.patience` | 4 | no accept in last N trials of this phase (incl. low-confidence skips) |
| textual exhaustion | — | no failing traces left to diagnose, or critic returns low confidence repeatedly |

**Global termination** (checked between phases / trials; ends the whole run):

| Condition | Default | Trigger |
|---|---|---|
| dry cycle | — | a full numeric→textual cycle produced **zero accepts** (the natural convergence of coordinate descent) |
| `max_cycles` | 4 | completed cycles ≥ cap |
| `max_minutes` | unset | `elapsed_wall_time ≥ max_minutes` |
| `max_usd` | unset | `cumulative_usd ≥ max_usd` |
| Ctrl-C / SIGTERM | — | Trap, let in-flight trial finish, write report, exit 2 |

Trigger is recorded in `report.md` and `run.json` as `exit_reason` (and, for a phase end, `phase_exit_reasons`).

**Why two levels.** A flat trial cap interacts badly with re-sweeping: each numeric phase re-pays its ~10-trial TPE warmup, so a single global `max_iterations: 30` would be eaten by warmups and never reach the textual passes that carry the largest single wins. Per-phase caps guarantee each axis gets a fair, bounded shot every cycle; the global dry-cycle stop is what actually ends a converged run.

**On budget exhaustion mid-trial:** the in-flight trial completes (its repeated runs, holdout, and decision are honored). We don't kill a multi-dollar repeated trial to save seconds.

**`max_usd` accounting** sums per-case `TestResult.token_usage` across `report.results[]` (× `repeats`) converted via the model's published per-token rates — *not* `report.summary.token_usage`, which does not exist. Critic + Applier costs added on top. If a model lacks pricing data (custom Ollama endpoint, e.g.), `max_usd` is silently skipped with a warning at startup.

## Validation at startup

Before any trial runs, in this order:

1. Load + validate `agent.yaml` (existing path).
2. `evaluations.optimizer` block exists. If absent: error with starter-config snippet for copy-paste.
3. Train and holdout case files load, parse, are disjoint, share metric set. Holdout `≥ min_holdout_cases` (default 5) → **error** below the floor (variance estimate too weak for the acceptance rule).
4. Loss weights reference metrics actually configured in `evaluations.metrics`. Unknown metric names → error.
5. `repeats ≥ 1` and `accept_sigma ≥ 0`. If `repeats == 1`, warn that the candidate contributes no measured spread (its `std = 0`), so the noise bar reduces to `accept_sigma × incumbent_std` — the bar still exists (the incumbent's stored std carries it), but it stops adapting to the candidate's own variance. Recommend `repeats ≥ 2` unless the agent is deterministic (temperature 0, no sampled tools).
6. Budgets parse: every per-phase `max_trials ≥ 1`; warn if a `numeric_phase.max_trials` is below the TPE warmup (`n_startup_trials`, default 10) — that phase will be pure random search.
7. Numeric axes parse: paths resolve against the Agent model, ranges non-empty, types valid. Each axis is *applied* to the baseline as a sanity check.
8. Textual axes parse: paths resolve, current text loads, `max_chars` ≥ current length.
9. Critic + Applier subagents load and validate. Auto-installed at `.holodeck/optimizer/agents/{critic,applier}.yaml` if missing — copied from package data dir. User can edit; we don't overwrite.
10. Backend credentials validated (existing `validate_credentials()` call).
11. Resume target (if `--resume`) exists, has matching `baseline_sha256`. If baseline drifted: error *"baseline changed since run X. Start a new run instead."*

All validation errors surfaced together when possible (single pass, collect, report).

## Out of scope for v1

| Out-of-scope | Why deferred | Path forward |
|---|---|---|
| **Few-shot demonstration optimization (#1 v2 item)** | The highest-leverage lever in DSPy/MIPRO-style prompt optimization, but it needs a first-class `demonstrations` slot on the agent schema, backend prompt injection, and a new *selector* proposer — its own design pass. The MVP's instruction-text tuning is the lower-leverage half of "prompt optimization"; this is why the feature is named *instruction & hyperparameter* optimizer. | **v2, headlined.** Bootstrap demos from `train_cases_file` (run agent on train inputs, keep traces that pass the grader), then select subset/order as a third proposer. The example pool + pass/fail oracle already exist in the test harness. |
| Ingestion-time axes (`chunking_strategy`, `max_chunk_tokens`, `contextual_embeddings`, `embedding_model`) | Re-ingest per trial = ~10min × $X. Would dominate cost. | Separate `holodeck test ingest-sweep` subcommand. |
| Parallel trials | Optuna supports it but trials hit shared resources (qdrant, OTEL, rate limits, file numbering). Multiplies failure modes. Compounding makes this harder still (a phase's baseline advances mid-flight). | v2: per-trial backend instance, `--concurrency N`, locking on writes. |
| K-fold cross-validation | K× the cost. Single split is sufficient signal for v1. | v2: `holdout_eval_policy: k_fold; k: 5`. |
| Population-based search (evolutionary, beam over candidates) | Multiplies cost by population size; coordinate descent is the single-track v1. | v2 or external: as a separate proposer plugin. |
| Full statistical significance testing (paired bootstrap) | The variance-aware bar (`accept_sigma`×pooled-std from `repeats`) is the cheap honest version; a paired bootstrap is more rigorous but costlier. | v2: replace the σ-bar with a paired test, configurable α. |
| Co-optimization of multiple textual axes per trial | Conflates blame; harder to explain. | v2: `co_optimize: [...]` opt-in. |
| Auto-discovery of axes | Most numeric leaves aren't safely tunable (e.g. `model.name`). Explicit declaration is right for v1. | v2: `--suggest-axes` reads agent.yaml and prints starter axis block. |
| Cross-run learning (TPE warm-start from a different agent's run) | Distribution mismatch. | Probably never. |
| Live dashboard rendering of optimizer runs | `holodeck test view` doesn't know optimizer file shape. | v2: detect `run.json`, render trajectory as chart, side panel with rationales. |
| Meta-optimization of the optimizer | Yak-shaving. Defaults are reasonable. | Never as a feature. |

## v1 contract

- One subcommand: `holodeck test optimize` — an **instruction & hyperparameter** optimizer (not few-shot demos; those are v2).
- **Compounding coordinate descent:** every accept advances the baseline; the final candidate stacks all accepted edits + hyperparameters.
- **Phase scheduler** (not a per-trial router): alternate numeric ↔ textual phases until a full cycle is dry.
- Two proposers: `NumericProposer` (Optuna TPE, **a fresh study per numeric phase**) and `TextualProposer` (Critic + Applier subagents).
- Sequential trials only. **Repeated trials** (`repeats`, default 3) average sampling noise.
- **Variance-aware acceptance:** improvement must clear `accept_sigma`×pooled-std measured from the repeats; no magic threshold.
- **Two-level budgets:** per-phase (`max_trials`, `patience`) + global (`max_cycles`, `max_minutes`, `max_usd`, dry-cycle stop).
- One train/holdout split, user-declared in YAML; holdout `≥ min_holdout_cases` enforced.
- Six output artifacts per run: `run.json`, `trials.jsonl`, `trajectory.csv`, `candidates/`, `instructions/`, `report.md` + symlinks.
- One CLI shape with three diagnostic modes: `--dry-run`, `--print-baseline`, `--report-only`.
- Numeric axes only at query time (excludes ingestion-time params).
- One textual axis perturbed per textual trial.
- Errored metric runs excluded + flagged, never scored as 0.0.
- Reproducibility is config-level (resume + provenance), not bit-for-bit.
- Auto-installed Critic/Applier subagents at `.holodeck/optimizer/agents/`, user-editable.

## Full agent.yaml `optimizer:` block (reference)

```yaml
evaluations:
  metrics:
    - type: standard
      metric: numeric
      absolute_tolerance: 0.5
    - type: code
      grader: graders.turn_program_equivalence:turn_program_equivalence

  optimizer:
    loss:
      numeric: 0.7
      turn_program_equivalence: 0.3
    penalties:                          # optional
      cost_per_usd: 0.0
      latency_per_sec: 0.0

    train_cases_file: data/convfinqa_train.yaml
    holdout_cases_file: data/convfinqa_holdout.yaml
    holdout_eval_policy: on_train_improve
    min_holdout_cases: 5

    repeats: 3                          # runs per trial; loss = mean, std = accept bar
    accept_sigma: 1.0                   # improvement must exceed accept_sigma × pooled std
    max_errored_fraction: 0.25          # above this, a metric's trial loss is "unreliable"

    axes:
      numeric:
        - path: model.temperature       # defensible to tune because repeats average its noise
          range: [0.0, 1.0]
        - path: model.top_p
          range: [0.5, 1.0]
        - path: tools[name=convfinqa_archive].top_k
          range: [3, 20]
          type: int
        - path: tools[name=convfinqa_archive].semantic_weight
          range: [0.2, 0.8]
      textual:
        - path: instructions.inline
          max_chars: 4000
        - path: tools[name=subtract].description
          max_chars: 200

    budget:
      global:
        max_cycles: 4                   # numeric→textual cycles before stop
        max_minutes: 60
        max_usd: 15.0                   # note: ~3× the old number — repeats=3
      numeric_phase:
        max_trials: 15                  # keep ≥ TPE warmup (10) or it's random search
        patience: 5
      textual_phase:
        max_trials: 10
        patience: 4
```

## Open questions resolved during design

1. **Path syntax for axes** — `tools[name=…].top_k`. Bracket-equals form preferred over JSONPath for readability.
2. **Async concurrency** — sequential v1; parallel deferred to v2.
3. **Resume semantics** — `--resume RUN_ID` reads `trials.jsonl`, advances the baseline through accepted trials, rebuilds each numeric phase's study from that phase's rows, continues from `len(trials)`.
4. **Where Applier writes textual edits** — `results/optimizer/<run-id>/instructions/system-prompt.iter-NN.md`. Candidate's `instructions.file: ../instructions/system-prompt.iter-NN.md` (relative to the candidate yaml's directory). Tool descriptions (which live inline) updated inline in the candidate `agent.yaml`.

## Open questions resolved in the 2026-05-29 rewrite

5. **Do improvements compound?** Yes — every accept advances the baseline (the original single-step-from-fixed-baseline behaviour was a bug). This is the core of the rewrite.
6. **How do compounding and TPE coexist?** Coordinate descent: a fresh TPE study per numeric phase, scoped to a window where the prompt is frozen. The original single immortal study would have fit a moving objective.
7. **Per-trial router vs phases?** Phases. The Strategy Router is removed; a phase scheduler alternates numeric/textual blocks and stops on a dry cycle.
8. **How is acceptance made honest at small eval sizes?** Repeated trials (`repeats`, default 3) + a variance-aware bar (`accept_sigma`×pooled-std), replacing the sub-noise-floor `0.02` / `1e-4` thresholds.
9. **Is this a "prompt optimizer"?** No — it's an *instruction & hyperparameter* optimizer. "Prompt optimization" in current usage implies few-shot demonstration selection, which is the #1 v2 item, not v1.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Optimizer overfits to training cases | Required holdout split + variance-aware acceptance (holdout must stay within its own noise). |
| Optimizer overfits to a non-representative holdout | README + `report.md` call out the distributional-overfit limitation. |
| Accepting changes that are just sampling noise | `repeats` measures the noise; accept bar = `accept_sigma`×pooled-std, recomputed per trial — no sub-noise-floor threshold. |
| TPE fits a moving objective after a prompt edit | Fresh TPE study per numeric phase; studies never span a prompt change. |
| Repeats triple the cost vs the original estimate | `max_usd`/per-phase trial caps account for `repeats`; cost-per-trial shown ×repeats in the progress line; `repeats` lowerable for deterministic agents. |
| Numeric phase wasted on pure warmup | Validation warns if `numeric_phase.max_trials` < TPE warmup. |
| User burns budget on a misconfigured run | `--dry-run` and `--print-baseline` validate before spending. Validation pass collects all errors. |
| Ingest re-runs per trial | `force_ingest=False` always; ingestion-time axes excluded from v1. |
| Mid-run crash loses state | Append-only JSONL + atomic candidate writes + `--report-only` regen. Baseline reconstructed from accepted rows on resume. |
| Critic/Applier proposes the same bad edit repeatedly | History of last 3 rejected gradients fed back as `previous_gradients`. |
| User expects bit-for-bit reproduction | `run-id` identifies inputs (baseline + config + seed) and makes the *proposal sequence* reproducible; losses vary (stochastic). Stated explicitly in `report.md`. |
| `agent.yaml` accidentally mutated | Original never touched; candidates under `results/optimizer/<run-id>/`. |

## Implementation surface — modules to create

- `src/holodeck/optimizer/__init__.py`
- `src/holodeck/optimizer/loop.py` — `OptimizerLoop`, `_run_trial`, `mean_loss` (repeats aggregation), variance-aware accept/reject, baseline-advance (compounding).
- `src/holodeck/optimizer/config.py` — Pydantic models for `evaluations.optimizer.*` (incl. `repeats`, `accept_sigma`, two-level `budget`, `min_holdout_cases`).
- `src/holodeck/optimizer/loss.py` — scalarizer (errored runs excluded+flagged), pooled-std / noise-bar computation, penalty calculation, per-trial USD accounting.
- `src/holodeck/optimizer/scheduler.py` — phase scheduler (replaces the per-trial Strategy Router): alternates numeric/textual phases, owns per-phase + global stop conditions, dry-cycle detection.
- `src/holodeck/optimizer/proposers/numeric.py` — Optuna TPE wrapper; **one study per numeric phase**, seed offset per phase.
- `src/holodeck/optimizer/proposers/textual.py` — Critic + Applier orchestration.
- `src/holodeck/optimizer/agent_mutator.py` — applies axis paths to a baseline `Agent` (Pydantic `model_copy(update=...)` plumbing for `tools[name=X].field` paths).
- `src/holodeck/optimizer/output.py` — `TrialLogger`, `run.json` writer, `trajectory.csv`, symlink atomicity.
- `src/holodeck/optimizer/report.py` — `report.md` synthesis, ASCII sparklines, diffs, attribution.
- `src/holodeck/optimizer/agents/critic.yaml` — packaged Critic subagent.
- `src/holodeck/optimizer/agents/applier.yaml` — packaged Applier subagent.
- `src/holodeck/cli/commands/test.py` — register `optimize` subcommand on existing group.

Tests:

- `tests/unit/optimizer/test_loop.py` — variance-aware accept/reject (improvement above/below noise bar; holdout within/over its noise); **baseline advances on accept (compounding)**; resume rebuilds baseline.
- `tests/unit/optimizer/test_loss.py` — scalarizer with weights, missing metrics; **errored runs excluded+flagged, not 0.0**; `max_errored_fraction` marks unreliable; pooled-std / noise-bar math; penalties.
- `tests/unit/optimizer/test_scheduler.py` — phase alternation; per-phase patience/max_trials stops; **dry-cycle global stop**; numeric-phase-below-warmup warning.
- `tests/unit/optimizer/test_proposers_numeric.py` — **fresh study per phase** (seed offset); deterministic proposal sequence with seed; per-phase resume replays only that phase's rows.
- `tests/unit/optimizer/test_proposers_textual.py` — Critic/Applier mocked; low-confidence short-circuits; previous_gradients fed back; edits compound against the current baseline.
- `tests/unit/optimizer/test_agent_mutator.py` — axis paths apply correctly; bracket-name resolution.
- `tests/unit/optimizer/test_output.py` — JSONL append safety; symlink atomicity; trajectory.csv columns (mean/std/noise_bar); `--report-only`.
- `tests/unit/optimizer/test_report.py` — report markdown shape, per-phase attribution, ASCII sparkline with phase boundaries, diff rendering.
- `tests/unit/cli/commands/test_optimize.py` — CLI parsing (two-level budget flags, `--repeats`, `--accept-sigma`), exit codes, validation pass (min_holdout_cases error).
- `tests/integration/optimizer/test_end_to_end.py` — full coordinate-descent loop against a tiny in-process agent + small train/holdout split, `repeats=2`. Mocked Optuna sampler and Critic/Applier; asserts a numeric+textual win compound into one candidate.

## Dependencies to add

- `optuna` (~3.x) — TPE sampler. Add to `pyproject.toml` extras `[optimizer]`. Not pulled into base install.

The Critic + Applier subagents reuse the existing Claude Agent SDK; no new SDK dependency.
