# Implementation Plan: `holodeck test optimize` — MVP

> **Scope cut (confirmed with user, 2026-05-31):** This MVP implements both phases of the
> spec's coordinate-descent optimizer (numeric Optuna + textual Critic/Applier) with
> **compounding**, but **defers statistical rigor**: no train/holdout split, no repeats
> (`k=1`), no variance-aware acceptance bar (accept on raw aggregate-score delta with a
> fixed `min_delta` epsilon), no USD/minute budgets, no resume. Those deferred pieces are
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
3. **MVP acceptance = raw delta.** Accept iff `score(candidate) - best_score > min_delta`.
   No repeats/holdout/variance bar in the MVP. (Deferred: spec decisions 4 & 5.)
4. **Reuse `TestExecutor` unchanged**, injecting a mutated `Agent` via `agent_config=`;
   ingest once (`force_ingest=False`).
5. **Errored metric runs excluded** from the scalarized loss, not scored as 0. (Spec
   smaller-correction.)
6. **Objective** = weighted mean of `TestReport.summary.average_scores` using
   `evaluations.optimizer.loss` weights.

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
- [ ] T9 E2E test (`best_score ≥ baseline_score`) + `agent.schema.json` `evaluations.optimizer` + docs + tasks.md

### Checkpoint: Complete
- [ ] `make ci` clean; smoke run writes 3 artifacts; source `agent.yaml` byte-identical

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
(`accept_sigma`, pooled-std bar); USD/minute budgets; `--resume`; `trajectory.csv` /
`run.json` / instruction-iteration files / `best.yaml` symlink atomics; few-shot
demonstration optimization; ingestion-time axes; parallel trials.
