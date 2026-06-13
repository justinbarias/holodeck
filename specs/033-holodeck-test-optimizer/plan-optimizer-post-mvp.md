# Implementation Plan: `holodeck test optimize` — Post-MVP → Spec v1

**Spec (authoritative):** `2026-05-16-optimizer-for-holodeck-test-design.md` (rev. 2026-05-29)
**Predecessor plans:** `plan.md` (MVP, shipped), `plan-text-proposer.md` (iterative textual, shipped),
`plan-optimize-otel.md` (observability, shipped)
**Status:** Draft for review (2026-06-07) — no code written yet.
**Companion:** this file embeds its own task list (same convention as `plan-optimize-otel.md`).

## Overview

The MVP (`plan.md` T1–T9), iterative textual refinement (`plan-text-proposer.md`), and
observability (`plan-optimize-otel.md`) all shipped in commit `#335`. What remains is the
work that turns the **compounding-coordinate-descent MVP** into the **authoritative spec v1**:
the statistical-rigor core the MVP explicitly deferred (`plan.md:3-10`, confirmed with the
user 2026-05-31), the full audit/artifact set, resume/checkpointing (`plan.md` Phase 7,
already specced as T10–T12), the remaining CLI diagnostics, and proposer-fidelity fixes.

The current acceptance rule is a flat `min_delta` (`loop.py:178`) against a single test set
scored once — the very "magic threshold below the noise floor" the 2026-05-29 rewrite was
written to eliminate. The spine of this plan is replacing that with **train/holdout +
`repeats` + variance-aware acceptance**, then making every trial durable and auditable.

Few-shot demonstration optimization (the design's headlined **#1 v2 item**) is **out of
scope** here — it needs its own schema/backend/selector design pass (see "Out of scope").

## Architecture Decisions

1. **Statistical rigor lands as one coherent slice, not piecemeal.** `repeats`, train/holdout
   split, and the variance-aware bar are mutually dependent (the bar needs the spread from
   `repeats` measured on `train`, gated by `holdout`). They ship together in Phase A behind a
   single checkpoint, so the loop is never in a half-honest state.
2. **The audit log is the checkpoint.** Resume replays `trials.jsonl`; no separate pickle
   (`plan.md:99-102`). This forces richer per-trial records (Phase B) to land *before* resume
   (Phase C) — Phase B is a hard dependency of Phase C.
3. **`TestExecutor` stays unchanged.** Train/holdout is `agent.model_copy(update={"test_cases":
   subset})` per call (spec §"What's new vs reused"); the scorer gains a `mean_loss` wrapper for
   `repeats`. No runner edits.
4. **Backward-compatible config.** Every new `OptimizerConfig` field has a spec default so
   existing `agent.yaml` files keep parsing; the *behavioural* switch from `min_delta` to the
   σ-bar is the only intentional break, gated by validation that warns when `repeats == 1`.
5. **Deterministic run-id enables resume.** Switch from `timestamp + random hex`
   (`optimize.py:251-255`) to `sha256(baseline_yaml + optimizer_config + seed)[:8]`
   (spec §"On-disk layout") so `--resume` is unambiguous.
6. **Secrets never leak into artifacts.** Existing `overlay_axes` on the unsubstituted template
   (`optimize.py:286-297`) is preserved for every candidate written; span/CSV/JSONL rules from
   `plan-optimize-otel.md` decision 5 carry over.

## Verified integration points

- **Config:** `OptimizerConfig` (`src/holodeck/optimizer/config.py`) — add fields + a
  `model_validator` for the new cross-field constraints.
- **Scorer:** `score()` (`src/holodeck/optimizer/scorer.py:16-42`) — wrap in `mean_loss` and
  accept a `cases` subset; today it runs `agent.test_cases` once.
- **Loop:** `OptimizerLoop._run_phase` accept gate (`loop.py:178`), `run()` baseline
  (`loop.py:74-79`), `_record` (`loop.py:225-229`); proposer `tell` already carries `report`.
- **Loss:** `scalarize` / `_metric_averages` (`src/holodeck/optimizer/loss.py:26-102`) — flat
  averaging today; needs per-case-then-across + penalties + `max_errored_fraction`.
- **Models:** `TrialRecord` / `OptimizationResult` (`src/holodeck/optimizer/models.py`).
- **Output:** `write_outputs` (`src/holodeck/optimizer/output.py:98-115`) — 3 artifacts today.
- **Numeric proposer:** fresh study per phase already (`numeric.py:39-43`) but no
  `multivariate`/`n_startup_trials`/`n_ei_candidates`, no per-phase seed offset.
- **Textual proposer:** `_critique`/`_apply_edit` (`textual.py:241-302`) emit a bare `gradient`
  string + `new_text`; no `confidence` short-circuit; subagents loaded from package dir
  (`textual.py:42-58`), not auto-installed/user-editable.
- **CLI:** `optimize` command (`src/holodeck/cli/commands/optimize.py`) — add flags + 3
  diagnostic modes.
- **Resume task detail:** already specced in `plan.md:94-121` and `tasks.md:176-237` (T10–T12).
- **Optuna:** currently a **base** dependency (`pyproject.toml:57`); spec wants it under an
  `[optimizer]` extra.

---

## Task List

### Phase A — Statistical rigor (MVP → honest acceptance)

- [ ] **A1 — Config surface for rigor + budgets.**
  Add to `OptimizerConfig`: `train_cases_file`, `holdout_cases_file`, `holdout_eval_policy`
  (`on_train_improve|every_trial|skip`, default `on_train_improve`), `min_holdout_cases`
  (default 5), `repeats` (default 3), `accept_sigma` (default 1.0), `max_errored_fraction`
  (default 0.25), `penalties` (`cost_per_usd`, `latency_per_sec`, both default 0.0),
  global `budget` (`max_minutes`, `max_usd`, both optional), and
  `numeric_convergence_eps` (default 1e-3). Keep `min_delta` for the `repeats == 1`/skip path.
  **Acceptance:**
  - [ ] All fields parse with spec defaults; an existing MVP `agent.yaml` (no new keys) still loads.
  - [ ] `model_validator` rejects `min_holdout_cases < 1`, `repeats < 1`, `accept_sigma < 0`.
  - [ ] `holdout_eval_policy` outside the enum → `ValidationError`.
  **Verification:** `pytest tests/unit/optimizer/test_config.py -n auto`; `make type-check`.
  **Dependencies:** None.
  **Files:** `src/holodeck/optimizer/config.py`, `tests/unit/optimizer/test_config.py`.
  **Scope:** S.

- [ ] **A2 — Startup validation pass (collect-all-errors).**
  Implement the spec §"Validation at startup" 11-step pass as a pure function returning a list
  of errors (raise a single `OptimizerError` aggregating them). Covers: optimizer block present;
  train/holdout load + parse + disjoint by `test_name` + share metric set + holdout ≥
  `min_holdout_cases`; loss weights reference configured metrics; `repeats`/`accept_sigma`
  bounds with the `repeats == 1` warning; per-phase `max_trials ≥ 1` + numeric-below-warmup
  warning; each numeric axis *applies* to the baseline as a sanity check; each textual axis path
  resolves and `max_chars ≥ current length`.
  **Acceptance:**
  - [ ] Multiple simultaneous misconfigurations are reported together, not one-at-a-time.
  - [ ] Overlapping train/holdout names → error naming the offenders.
  - [ ] Holdout below floor → error (not warning); `repeats == 1` → warning, not error.
  **Verification:** `pytest tests/unit/optimizer/test_validation.py -n auto`.
  **Dependencies:** A1.
  **Files:** `src/holodeck/optimizer/validation.py` (new), `tests/unit/optimizer/test_validation.py`.
  **Scope:** M.

- [ ] **A3 — Train/holdout loading + subset injection.**
  Load both case files into `list[TestCaseModel]`; thread `train`/`holdout` subsets into the
  scorer via `agent.model_copy(update={"test_cases": subset})`. When only `test_cases_file` is
  present and no split is configured, error per spec (point at `holdout_eval_policy: skip`).
  **Acceptance:**
  - [ ] Scorer runs against the train subset by default; holdout subset available on demand.
  - [ ] `holdout_eval_policy: skip` runs train-only and is flagged for the report.
  **Verification:** `pytest tests/unit/optimizer/test_scorer.py -n auto`.
  **Dependencies:** A1.
  **Files:** `src/holodeck/optimizer/scorer.py`, `src/holodeck/cli/commands/optimize.py`,
  `tests/unit/optimizer/test_scorer.py`.
  **Scope:** M.

- [ ] **A4 — `mean_loss`: repeats aggregation (mean + std).**
  Add `mean_loss(cand, cases, repeats) -> TrialLoss{mean, std, n, runs}` running `repeats`
  independent sweeps (`force_ingest=False` throughout). Carry the spread for the accept bar.
  **Acceptance:**
  - [ ] `repeats=3` runs the executor 3× and returns `pstdev` of the per-run losses.
  - [ ] `repeats=1` returns `std=0` (degenerate, documented).
  **Verification:** `pytest tests/unit/optimizer/test_scorer.py -n auto` (stub executor).
  **Dependencies:** A3.
  **Files:** `src/holodeck/optimizer/scorer.py`, `src/holodeck/optimizer/models.py`
  (add `TrialLoss`), `tests/unit/optimizer/test_scorer.py`.
  **Scope:** S–M.

- [ ] **A5 — Loss: per-turn/per-case averaging, `max_errored_fraction`, penalties.**
  In `loss.py`: average per-turn metrics within a case then across cases (not flat across all
  `metric_results`); mark a metric's trial loss *unreliable* when its errored fraction exceeds
  `max_errored_fraction` (block accept-on-that-basis, logged); add the optional cost/latency
  penalties (`trial_usd` summed from per-result `token_usage` × repeats, mean latency from
  `execution_time_ms`).
  **Acceptance:**
  - [ ] A multi-turn case does not dominate a single-turn case in the aggregate.
  - [ ] >25% errored runs for a weighted metric marks the trial unreliable.
  - [ ] Penalties off by default; when set, add to the scalar loss per the spec formula.
  **Verification:** `pytest tests/unit/optimizer/test_loss.py -n auto`.
  **Dependencies:** A1.
  **Files:** `src/holodeck/optimizer/loss.py`, `tests/unit/optimizer/test_loss.py`.
  **Scope:** M.

- [ ] **A6 — Variance-aware acceptance + holdout gating in the loop.**
  Replace `min_delta` accept (`loop.py:178`) with: always run TRAIN ×`repeats`; decide holdout
  per `holdout_eval_policy`; `pooled_std = sqrt(train.std² + best_train.std²)`,
  `noise_bar = accept_sigma·pooled_std`; accept iff `(best_train.mean − train.mean) ≥ noise_bar`
  AND holdout did not regress beyond its own σ. Update bests (mean+std) only on accept; advance
  baseline (compounding preserved). Fall back to `min_delta` when `repeats == 1` *and* policy
  `skip` (no measurable spread).
  **Acceptance:**
  - [ ] A sub-noise "improvement" is rejected with the spec rationale string.
  - [ ] A holdout regression beyond its σ blocks an otherwise-accepted train win.
  - [ ] Baseline still advances on accept; wins still compound across phases.
  **Verification:** `pytest tests/unit/optimizer/test_loop.py -n auto`.
  **Dependencies:** A4, A5.
  **Files:** `src/holodeck/optimizer/loop.py`, `tests/unit/optimizer/test_loop.py`.
  **Scope:** M.

- [ ] **A7 — Global budgets + numeric convergence stop.**
  Track cumulative wall-clock and USD; stop the run when `max_minutes`/`max_usd` trips
  (in-flight trial finishes first). Add `numeric_convergence_eps` as a per-phase numeric stop
  (TPE expected-improvement below ε). Skip `max_usd` with a startup warning when a model lacks
  pricing.
  **Acceptance:**
  - [ ] `max_usd`/`max_minutes` end the run between trials; recorded as `exit_reason`.
  - [ ] Numeric phase stops early when EI < ε; logged distinctly from patience/max_trials.
  **Verification:** `pytest tests/unit/optimizer/test_loop.py tests/unit/optimizer/test_numeric.py -n auto`.
  **Dependencies:** A6.
  **Files:** `src/holodeck/optimizer/loop.py`, `src/holodeck/optimizer/proposers/numeric.py`,
  tests.
  **Scope:** M.

#### 🛑 Checkpoint A — honest acceptance
- [ ] Train drives proposals; holdout gates accepts; accept bar is measured, not magic.
- [ ] A known-noisy stub agent yields zero false accepts at `accept_sigma=1.0`.
- [ ] `make test-unit && make type-check` clean; existing MVP configs still run.

---

### Phase B — Audit & artifacts (spec-complete outputs)

- [ ] **B1 — Richer `TrialRecord` + `OptimizationResult`.**
  Extend `TrialRecord` to the spec shape: `timestamp`, `cycle`, `phase`, `axis_target`,
  `baseline_before_path`, `proposal_source` (numeric: params/study_id/marginals; textual:
  critic+applier blocks), `train`/`holdout` blocks (`loss`, `loss_std`, `loss_runs`,
  `report_summary`, `errored_runs_excluded`, `duration_sec`, `trial_usd`), `decision` rationale
  (noise_bar, improvement_clears_noise, holdout_within_noise, reason), `study_state`.
  **Acceptance:**
  - [ ] A trial round-trips to/from the spec JSON example without loss.
  - [ ] Existing readers of the minimal record are updated; serialization is `extra="forbid"`-clean.
  **Verification:** `pytest tests/unit/optimizer/test_models.py -n auto`.
  **Dependencies:** A6 (decision fields exist to record).
  **Files:** `src/holodeck/optimizer/models.py`, `tests/unit/optimizer/test_models.py`.
  **Scope:** M.

- [ ] **B2 — `run.json` header + deterministic run-id.**
  Switch run-id to `YYYY-MM-DDTHH-MM-SS_<sha256(baseline+config+seed)[:8]>`; write immutable
  `run.json` (spec §"run.json") on trial 0. Note `Date.now()`-equivalents are fine in CLI
  (timestamp from `datetime.now`), but the *hash* drives resume identity.
  **Acceptance:**
  - [ ] Same `(baseline, config, seed)` → identical hash suffix; any change → different.
  - [ ] `run.json` round-trips and carries the config fingerprint + subagent paths.
  **Verification:** `pytest tests/unit/optimizer/test_output.py -n auto`.
  **Dependencies:** A1.
  **Files:** `src/holodeck/optimizer/output.py`, `src/holodeck/cli/commands/optimize.py`, tests.
  **Scope:** S–M.

- [ ] **B3 — Per-trial atomic persistence (`trials.jsonl`).**
  Append one row after **each** completed trial (and the baseline) via atomic write + fsync,
  instead of one bulk write at the end (`output.py:112`). Partial/in-flight trials write nothing.
  **Acceptance:**
  - [ ] After N trials, `trials.jsonl` has N (+1 baseline) parseable rows.
  - [ ] A simulated kill mid-trial leaves the file ending on a complete row.
  **Verification:** `pytest tests/unit/optimizer/test_output.py -n auto`.
  **Dependencies:** B1, B2.
  **Files:** `src/holodeck/optimizer/output.py`, `src/holodeck/optimizer/loop.py`, tests.
  **Scope:** M. *(This is `plan.md`/`tasks.md` **T10** — realize it here.)*

- [ ] **B4 — `trajectory.csv` + versioned candidates/instructions + symlinks.**
  Write `trajectory.csv` (spec columns: trial_id…accepted,trial_usd,duration_sec); write
  `candidates/iter-NN.yaml` and `instructions/system-prompt.iter-NN.md` on each accept; maintain
  `best.yaml`/`best-instructions.md` as atomically-replaced symlinks (write tmp + `os.replace`).
  Add `--keep-best-only` pruning at exit.
  **Acceptance:**
  - [ ] CSV row count = trials; empty `holdout_mean` when holdout not run.
  - [ ] `iter-NN` numbering matches `trial_id`; symlinks point at the current leader.
  - [ ] Candidates carry `${VAR}` placeholders (via existing `overlay_axes`), no resolved secrets.
  **Verification:** `pytest tests/unit/optimizer/test_output.py -n auto`.
  **Dependencies:** B3.
  **Files:** `src/holodeck/optimizer/output.py`, tests.
  **Scope:** M.

- [ ] **B5 — `report.md` enrichment.**
  Extend the report (`output.py:49-95`) with: result summary (incl. `repeats`/exit reason),
  ASCII sparkline of train/holdout with phase boundaries, per-phase + per-axis attribution
  table, per-metric baseline→best, full baseline→best diff (yaml + instructions), key accepted
  rationales, "How to adopt" `cp`/`sed` commands, methodology notes (split sizes,
  `accept_sigma`, holdout policy, distributional-overfit + config-level-reproducibility warnings).
  **Acceptance:**
  - [ ] Report renders for a multi-cycle run with both phases and ≥1 accept.
  - [ ] Sparkline marks phase boundaries; attribution sums match accepted Δloss.
  **Verification:** `pytest tests/unit/optimizer/test_report.py -n auto`.
  **Dependencies:** B1, B4.
  **Files:** `src/holodeck/optimizer/report.py` (new) + `output.py`, `tests/unit/optimizer/test_report.py`.
  **Scope:** M.

#### 🛑 Checkpoint B — durable & auditable
- [ ] A run produces the full 6-artifact set + symlinks; `trials.jsonl` survives a mid-run kill.
- [ ] `make test-unit && make lint && make type-check` clean.

---

### Phase C — Resume & resilience (`plan.md` Phase 7)

- [ ] **C1 — `--resume` replay in `OptimizerLoop`.**
  Load `trials.jsonl` + `run.json`; re-derive bests by replaying accepted rows in order; rebuild
  each numeric phase's study from *that phase's* rows via `study.add_trial(params, value)`; carry
  bests (mean+std) forward; replay last 3 rejected textual gradients into the Critic memory;
  continue from `len(trials)+1`. Studies reconstructed per phase, never merged.
  **Acceptance:**
  - [ ] Replaying a captured `trials.jsonl` reproduces the same `best`/`best_loss` decisions.
  - [ ] A resumed numeric phase's study contains all prior in-phase trials (TPE not reset).
  - [ ] No duplicate trial ids; cycle numbering continues correctly.
  **Verification:** `pytest tests/unit/optimizer/test_resume.py -n auto`.
  **Dependencies:** B3 (per-trial persistence is the checkpoint).
  **Files:** `src/holodeck/optimizer/loop.py`, `src/holodeck/optimizer/resume.py` (new), tests.
  **Scope:** M–L. *(This is **T11**.)*

- [ ] **C2 — CLI `--resume` + drift guard + signal handling.**
  Add `--resume RUN_ID` (auto-detect latest incomplete run when omitted); refuse on
  `baseline_sha256` mismatch unless `--force`; trap Ctrl-C/SIGTERM → let the in-flight trial
  finish, write `report.md`, exit code 2 with intact state.
  **Acceptance:**
  - [ ] `--resume <id>` continues a stopped run to completion.
  - [ ] Mismatched config → non-zero exit with a clear message; `--force` overrides.
  - [ ] Ctrl-C → exit 2; a subsequent `--resume` succeeds.
  **Verification:** `pytest tests/unit/cli/test_optimize*.py -n auto`; manual SIGINT smoke.
  **Dependencies:** C1.
  **Files:** `src/holodeck/cli/commands/optimize.py`, tests.
  **Scope:** M. *(This is **T12**.)*

#### 🛑 Checkpoint C — interruption resilience
- [ ] Kill a real run mid-trial (SIGINT); `--resume` continues from the last completed trial with
  bests + each phase's TPE study intact, reaching the same accept/reject decisions (not bit-exact).

---

### Phase D — CLI diagnostics & proposer fidelity

- [ ] **D1 — New CLI flags + diagnostic modes.**
  Add `--repeats`, `--accept-sigma`, `--max-minutes`, `--max-usd` overrides and the three
  diagnostic modes: `--dry-run` (validate + print resolved axes/budget/loss, exit 0),
  `--print-baseline` (trial 0 only + per-metric breakdown, exit 0), `--report-only` (regenerate
  `report.md` from existing JSONL, no trials). Preserve CLI→YAML→default precedence.
  **Acceptance:**
  - [ ] `--help` lists every spec flag; `--dry-run` spends nothing and exits 0.
  - [ ] `--report-only` rebuilds the report from a captured run dir.
  **Verification:** `pytest tests/unit/cli/test_optimize*.py -n auto`.
  **Dependencies:** A1 (config), B5 (report regen).
  **Files:** `src/holodeck/cli/commands/optimize.py`, tests.
  **Scope:** M.

- [ ] **D2 — Critic/Applier structured schema + confidence short-circuit.**
  Replace the bare `gradient` string with the spec schema (Critic: `failing_pattern`,
  `root_cause_hypothesis`, `suggested_change_direction`, `confidence`, `citations`, `avoid`;
  Applier: `edit_type`, `rationale`, `new_text`, `diff_summary`). Short-circuit before the
  Applier + sweep when `confidence < min_confidence` (default 0.4) — recorded as a cheap skip.
  Persist both blocks into `proposal_source` (B1).
  **Acceptance:**
  - [ ] Low-confidence gradient skips the trial without running the Applier or a sweep.
  - [ ] `citations`/`edit_type`/`diff_summary` land in the trial audit row.
  **Verification:** `pytest tests/unit/optimizer/test_textual.py -n auto`.
  **Dependencies:** B1.
  **Files:** `src/holodeck/optimizer/proposers/textual.py`,
  `src/holodeck/optimizer/agents/{critic,applier}.yaml`, tests.
  **Scope:** M.

- [ ] **D3 — Auto-install user-editable subagents.**
  On first run, copy `critic.yaml`/`applier.yaml` from package data to
  `.holodeck/optimizer/agents/{critic,applier}.yaml` if missing; load from there so users can
  edit without losing changes (never overwrite). Record the resolved paths in `run.json`.
  **Acceptance:**
  - [ ] Fresh project auto-installs both; an edited local copy is preserved across runs.
  **Verification:** `pytest tests/unit/optimizer/test_textual.py -n auto`.
  **Dependencies:** D2.
  **Files:** `src/holodeck/optimizer/proposers/textual.py`, tests.
  **Scope:** S.

- [ ] **D4 — TPE fidelity + `[optimizer]` extra.**
  Configure `TPESampler(seed=seed+phase_index, n_startup_trials=10, n_ei_candidates=24,
  multivariate=True)`; offset the seed per numeric phase; add the warmup-below-`max_trials`
  warning; expose `axis_marginals`/`numeric_study_id`/`params` for the audit row; move `optuna`
  to a `[optimizer]` extra in `pyproject.toml` (out of base install) with an import guard.
  **Acceptance:**
  - [ ] Repeated numeric phases draw different warmup sequences yet stay reproducible.
  - [ ] `optuna` not pulled by base install; clear error if the extra is missing.
  **Verification:** `pytest tests/unit/optimizer/test_numeric.py -n auto`; clean-env install check.
  **Dependencies:** A7, B1.
  **Files:** `src/holodeck/optimizer/proposers/numeric.py`, `pyproject.toml`, tests.
  **Scope:** S–M.

#### 🛑 Checkpoint D — spec-complete CLI & proposers
- [ ] All spec CLI flags + diagnostics present; structured subagents with confidence skip; TPE
  configured per spec; `make ci` clean.

---

### Phase E — Docs, schema, hygiene

- [ ] **E1 — Schema + docs for the new config.**
  Extend `schemas/agent.schema.json` for the new `evaluations.optimizer` fields; update
  `docs/guides/evaluations.md` + `AGENTS.md` with the train/holdout, repeats, acceptance, budget,
  and resume sections; run `python scripts/generate_agent_schema.py --check`.
  **Acceptance:**
  - [ ] Schema validates the spec's full reference `optimizer:` block; drift guard green.
  **Verification:** `make ci`.
  **Dependencies:** A1.
  **Files:** `schemas/agent.schema.json`, `docs/guides/evaluations.md`, `AGENTS.md`,
  `specs/033-holodeck-test-optimizer/optimizer.md`.
  **Scope:** S–M.

- [ ] **E2 — Refresh stale predecessor docs.**
  Mark `plan-text-proposer.md` as shipped (it still says "no code written yet") and tick its
  task checkboxes in `tasks-text-proposer.md`; tick the shipped MVP boxes in `tasks.md` T1–T9;
  move resume out of `plan.md` Phase 7 with a pointer to this file (single source of truth).
  **Acceptance:**
  - [ ] No predecessor plan claims unshipped status for shipped code.
  **Verification:** doc review.
  **Dependencies:** None (can run anytime).
  **Files:** `specs/033-holodeck-test-optimizer/{plan,tasks,plan-text-proposer,tasks-text-proposer}.md`.
  **Scope:** XS.

- [ ] **E3 — End-to-end integration test.**
  Full coordinate-descent loop against a tiny in-process agent + small train/holdout split,
  `repeats=2`, mocked Optuna sampler + Critic/Applier; assert a numeric+textual win **compounds**
  into one candidate and the σ-bar rejects a sub-noise trial.
  **Acceptance:**
  - [ ] E2E green; `best_loss ≤ baseline_loss`; both phases contribute to the final candidate.
  **Verification:** `pytest tests/integration/optimizer/test_end_to_end.py -n auto`.
  **Dependencies:** A6, B4, C1.
  **Files:** `tests/integration/optimizer/test_end_to_end.py`.
  **Scope:** M.

#### 🛑 Checkpoint: Complete
- [ ] All acceptance criteria met; `make ci` clean; full spec v1 contract satisfied.
- [ ] Manual run on `sample/financial-assistant/claude` writes the 6-artifact set, honest
  accept decisions, and a resumable run. Human review before merge.

---

## Dependency Graph

```
A1 (config) ──┬── A2 (validation)
              ├── A3 (train/holdout) ── A4 (mean_loss/repeats) ──┐
              ├── A5 (loss: penalties/errors/per-turn) ──────────┤
              └──────────────────────────────────────────────── A6 (variance accept)
                                                                  └── A7 (budgets/convergence)
A6 ── B1 (rich TrialRecord) ── B3 (atomic jsonl) ── B4 (csv/candidates/symlinks) ── B5 (report)
A1 ── B2 (run.json/run-id) ───┘                                   │
B3 ── C1 (resume replay) ── C2 (cli resume/signals)              │
B1 ── D2 (subagent schema) ── D3 (auto-install)                  │
A7,B1 ── D4 (TPE fidelity/extra)        A1,B5 ── D1 (cli flags/diagnostics)
A1 ── E1 (schema/docs)   (E2 anytime)   A6,B4,C1 ── E3 (e2e)
```

Bottom-up order: **A → B → C / D in parallel → E**. Phase B must precede Phase C (the audit
log is the resume checkpoint). D1–D4 are independent of C and can run in parallel once their
own deps land.

## Parallelization Opportunities

- **Parallel-safe:** A5 alongside A2/A3 (independent of each other, both need only A1); D2/D3
  alongside C1/C2 (textual vs resume are independent); E2 anytime.
- **Sequential (shared state):** A4→A6→A7 (the accept-gate chain); B1→B3→B4→B5 (artifact
  pipeline); B3→C1→C2 (resume chain).
- **Contract-first:** B1's `TrialRecord` shape gates B3/B5/C1/D2 — land it before fanning out.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `repeats×cycles×phases` triples cost vs MVP | High | `max_usd`/`max_minutes` budgets (A7); cost shown ×repeats in progress; `repeats` lowerable for deterministic agents; confidence short-circuit (D2) skips sweeps. |
| Switching `min_delta`→σ-bar silently changes existing runs' outcomes | Med | Validation warns on `repeats==1`; keep `min_delta` fallback for the deterministic/skip path; E3 asserts the new gate; document in E1. |
| Rich `TrialRecord` churns many downstream readers | Med | Land B1 first as a contract; `extra="forbid"` catches drift; update output/report/resume together. |
| Resume study rebuild diverges from a live run | High | Per-phase `add_trial` from that phase's rows only; C-checkpoint asserts decision-level (not bit-exact) equivalence. |
| Optuna moved to extra breaks base imports | Med | Import guard with a clear "install holodeck[optimizer]" error; clean-env install check in D4. |
| Train/holdout from same distribution hides distributional overfit | Med (honest limitation) | Documented in B5 report + E1 docs; not solvable in v1. |

## Out of Scope (explicit — not in this plan)

These remain deferred per the authoritative spec (§"Out of scope for v1"):

- **Few-shot demonstration optimization** — the design's headlined **#1 v2 item**; needs a
  first-class `demonstrations` schema slot, backend prompt injection, and a new *selector*
  proposer (its own design pass). **Not** decomposed here.
- Ingestion-time axes (`chunking_strategy`, `max_chunk_tokens`, `contextual_embeddings`,
  `embedding_model`) — a separate `holodeck test ingest-sweep`.
- Parallel trials; K-fold CV; population/evolutionary search; full paired-bootstrap significance
  testing; co-optimization of multiple textual axes per trial; auto-discovery of axes;
  cross-run TPE warm-start; live dashboard rendering of optimizer runs.

## Open Questions

- **`min_delta` retirement:** keep it permanently as the `repeats==1 && skip` fallback, or
  remove once the σ-bar is proven? (Plan keeps it; flag for reviewer.)
- **Run-id timestamp vs pure hash:** spec uses `timestamp_<hash>`; the timestamp is cosmetic and
  the hash drives resume — confirm the timestamp stays for human readability.
- **`--report-only` vs `--report-only --keep-best-only` interaction:** confirm report-only never
  prunes.
