# Implementation Plan: `holodeck test optimize` — MVP

> **Scope cut (confirmed with user, 2026-05-31):** This MVP implements both phases of the
> spec's coordinate-descent optimizer (numeric Optuna + textual Critic/Applier) with
> **compounding**, but **defers statistical rigor**: no train/holdout split, no repeats
> (`k=1`), no variance-aware acceptance bar (accept on raw aggregate-score delta with a
> fixed `min_delta` epsilon), no USD/minute budgets, no resume (resume/checkpointing is
> now planned as **Phase 7**, below). Those deferred pieces are
> what turn this MVP into the full spec v1. Authoritative design remains
> `optimizer.md` + `2026-05-16-optimizer-for-holodeck-test-design.md`.

## Overview

Automate the agent author's hand-tuning loop (change knob/instruction → `holodeck test`
→ eyeball → repeat) as a compounding coordinate-descent optimizer exposed as
`holodeck test optimize <agent.yaml>`. It alternates a numeric phase (Optuna TPE over
declared query-time axes) and a textual phase (Critic produces a natural-language
gradient, Applier rewrites the instructions), advancing a `best_candidate` baseline on
every accepted improvement so wins compound into one candidate `agent.yaml`. The original
`agent.yaml` is never mutated; every trial is logged.

## Architecture decisions

1. **Compounding coordinate descent.** `baseline ← candidate` after each accept; phase
   scheduler sweeps numeric → textual → repeat until a full cycle yields zero accepts or
   `max_cycles` is hit. (Spec decision 1.)
2. **Fresh Optuna TPE study per numeric phase** — a textual edit changes the objective, so
   prior numeric observations are stale. (Spec decision 2.)
3. **MVP acceptance = raw delta.** Accept iff `best_loss - loss(candidate) > min_delta`
   (loss is minimized). No repeats/holdout/variance bar in the MVP. (Deferred: spec
   decisions 4 & 5.)
4. **Reuse `TestExecutor` unchanged**, injecting a mutated `Agent` via `agent_config=`;
   ingest once (`force_ingest=False`).
5. **Errored metric runs excluded** from the scalarized loss, not scored as 0. (Spec
   smaller-correction.)
6. **Objective** = a minimized loss `1 − weighted_mean` of `TestReport`'s per-metric
   averages using `evaluations.optimizer.loss` weights (metric scores must be in
   `[0, 1]`).

## Verified integration points

- Suite run: `TestExecutor(agent_config, agent_config_path, force_ingest=False).execute_tests()`
  → `TestReport` (`src/holodeck/lib/test_runner/executor.py:431-445,804`).
- Score signal: `TestReport.summary.average_scores` / `pass_rate`
  (`src/holodeck/models/test_result.py:263-287`); failing-case context from
  `TestResult.metric_results[].score`, `.passed`, `.errors`.
- Prompt: `Agent.instructions` (`agent.py:29-62,80`); candidates via Pydantic v2
  `model_copy(update=...)`.
- LLM for Critic/Applier: `BackendSelector.select(...)` → `invoke_once()` (has
  `structured_output`); subagent YAMLs via `src/holodeck/lib/agents/loader.py`.
- CLI: register `optimize` under the `test` group (mirror `@test.command("run")` in
  `cli/commands/test.py`; groups registered in `cli/main.py:86-93`).
- `OptimizerError` is **absent** — add to `lib/errors.py`. **Optuna is not installed** —
  `uv add optuna`.

## Task list

### Phase 1: Foundations + scoring spine
- [ ] T1 Package skeleton, `OptimizerError`, `OptimizerConfig`, `models.py`
- [ ] T2 `loss.scalarize` + `mutator.apply_axes` / `apply_textual_edit`
- [ ] T3 `scorer.score` wrapping `TestExecutor`

### Checkpoint A
- [ ] `score()` reproduces eval-runner semantics (equality vs direct executor on stub backend)

### Phase 2: Loop (prove compounding with stub proposers)
- [ ] T4 `OptimizerLoop.run()` + stub proposer — compounding, accept rule, patience/max_cycles/dry-cycle stop, deterministic

### Checkpoint B
- [ ] Loop semantics reviewed

### Phase 3: Numeric proposer
- [ ] T5 `proposers/numeric.py` (Optuna TPE, fresh seeded study per phase) + `uv add optuna`

### Phase 4: Textual proposer
- [ ] T6 `proposers/textual.py` + `agents/{critic,applier}.yaml` + structured output

### Checkpoint C
- [ ] Both proposers verified in isolation

### Phase 5: Outputs + CLI
- [ ] T7 `output.py` → `results/optimizer/<run-id>/{best.yaml,trials.jsonl,report.md}` (original untouched)
- [ ] T8 `cli/commands/optimize.py` + register on `test` group; streams scores; validates ≥1 test case

### Checkpoint D
- [ ] Manual smoke run on fixture agent

### Phase 6: Integration, schema, docs, CI
- [ ] T9 E2E test (`best_loss ≤ baseline_loss`) + `agent.schema.json` `evaluations.optimizer` + docs + tasks.md

### Checkpoint: Complete
- [ ] `make ci` clean; smoke run writes 3 artifacts; source `agent.yaml` byte-identical

### Phase 7: Checkpointing & resume (post-MVP) — interruption resilience
> **Why this matters:** every trial is one full eval (live, billable, minutes long).
> A 10–20+ trial run killed by a crash, Ctrl-C, OOM, or dropped session currently
> loses all of it — `T7` only persists artifacts at the *end*. The design's
> `--resume` contract is already written (design.md:350-355, 427, 491, 776-790,
> 936-994); the MVP just never built it. This phase realizes it: **the audit log
> *is* the checkpoint** — no separate pickle. Resume replays `trials.jsonl`,
> advances the baseline through accepted rows, and rebuilds each numeric phase's
> Optuna study via `study.add_trial(params, value)`.
- [ ] T10 Per-trial atomic persistence: append a row to `trials.jsonl` (atomic
  write + fsync) after **each** completed trial *and* the baseline; write
  `run.json` once on trial 0 (baseline, resolved-config fingerprint, seed).
  Switch the run-id to deterministic `sha256(agent_yaml + optimizer_config +
  seed)[:8]` (design.md:804) so `--resume` is unambiguous. Partial/in-flight
  trials write nothing → resume picks up from the last *complete* row.
- [ ] T11 `--resume` replay in `OptimizerLoop`: load `trials.jsonl` + `run.json`,
  re-derive `best` by replaying accepted rows in order, rebuild each numeric
  phase's study from its own rows (`study.add_trial`), carry bests forward, and
  continue the in-flight phase from `len(trials)+1`. (Numeric rows already carry
  `params`; that is what the study rebuild needs — design.md:427.)
- [ ] T12 CLI `--resume RUN_ID` (auto-detect latest incomplete run when omitted)
  + config-drift guard (refuse on fingerprint mismatch unless `--force`) +
  Ctrl-C → exit code 2 with intact state (design.md:994) + docs/tests.

### Checkpoint E
- [ ] Kill a run mid-trial (SIGINT); `--resume` continues from the last completed
  trial with bests + each phase's TPE study intact; resumed run reaches the same
  accept/reject decisions as an uninterrupted one (decision-level, not bit-exact).

## Acceptance criteria & verification

Per-task AC/verification are detailed in `tasks.md`. End-to-end:

```bash
source .venv/bin/activate && uv add optuna
pytest tests/unit/optimizer/ -n auto -v
pytest tests/integration/test_optimize_e2e.py -n auto
holodeck test optimize tests/fixtures/agents/optimizer_agent.yaml -v
make format && make lint && make type-check && make security
```

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Raw-delta acceptance chases eval noise (no repeats) | Med | Fixed `min_delta` epsilon; document that variance-aware accept is the v1 follow-up |
| Re-ingestion per trial blows up cost | High | `force_ingest=False`; MVP numeric axes are query-time only (temperature, top_k, weights) |
| Optuna new dependency | Low | Pin via `uv add optuna`; isolate behind `proposers/numeric.py` |
| Critic/Applier structured output flakiness | Med | Validate JSON against schema; skip textual trial on parse failure (counts toward patience) |
| Mutating nested Pydantic via `model_copy` | Med | Axis-path parser unit-tested; bad path → `OptimizerError` |

## Out of scope (becomes spec v1)

Train/holdout + `min_holdout_cases`; `repeats` (k>1); variance-aware acceptance
(`accept_sigma`, pooled-std bar); USD/minute budgets; `trajectory.csv` /
instruction-iteration files / `best.yaml` symlink atomics; few-shot
demonstration optimization; ingestion-time axes; parallel trials.
(`--resume` + `run.json` per-trial persistence: now **Phase 7** above.)
