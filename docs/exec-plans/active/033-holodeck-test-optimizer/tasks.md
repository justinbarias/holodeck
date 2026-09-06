# Tasks: `holodeck test optimize` — MVP

Dependency-ordered. Each task: acceptance criteria (AC), verification (V), dependencies,
files, scope. 🛑 = human checkpoint. Scope cut per `plan.md` (both phases, no rigor).

Run pytest with `-n auto`. After each task: `make format lint type-check`.

---

## Phase 1 — Foundations + scoring spine

### T1: Package skeleton + `OptimizerError` + `OptimizerConfig` + models ✅
**Description:** Create `src/holodeck/optimizer/` with `__init__.py`, `config.py`
(`OptimizerConfig` parsed from `evaluations.optimizer`: `loss` weights,
`axes.numeric[]`{path,range,type}, `axes.textual[]`{path,max_chars}, `max_cycles`,
`numeric_phase`{max_trials,patience}, `textual_phase`{max_trials,patience}, `min_delta`,
`seed`), and `models.py` (`TrialRecord`, `OptimizationResult`). Add
`OptimizerError(HoloDeckError)` to `src/holodeck/lib/errors.py`.
**AC:**
- [ ] Package imports cleanly; `OptimizerError` importable from `holodeck.lib.errors`.
- [ ] Invalid config (e.g. empty loss weights, bad axis type) raises `ValidationError`.
- [ ] `TrialRecord` / `OptimizationResult` round-trip serialize.
**V:** `pytest tests/unit/optimizer/test_config.py tests/unit/optimizer/test_models.py -n auto`
**Dependencies:** None · **Files:** `optimizer/{__init__,config,models}.py`, `lib/errors.py`,
2 test files · **Scope:** M

### T2: `loss.scalarize` + `mutator` (axis application) ✅
**Description:** `loss.scalarize(report, loss_weights) -> float` (weighted mean of
`summary.average_scores`, errored metrics excluded + flagged). `mutator.apply_axes(agent,
params)` and `mutator.apply_textual_edit(agent, axis, new_text)` via nested `model_copy`,
with an axis-path parser for `model.temperature` and `tools[name=X].<field>`.
**AC:**
- [ ] Scalarized score matches a hand-computed weighted mean on a fixture report.
- [ ] Errored metric run is excluded from the aggregate.
- [ ] `model.temperature` and `tools[name=X].top_k` mutate a fixture Agent; original Agent
      object unchanged (new instance returned).
- [ ] Unknown axis path raises `OptimizerError`.
**V:** `pytest tests/unit/optimizer/test_loss.py tests/unit/optimizer/test_mutator.py -n auto`
**Dependencies:** T1 · **Files:** `optimizer/{loss,mutator}.py`, 2 test files · **Scope:** M

### T3: `scorer.score` wrapping `TestExecutor` ✅
**Description:** `async scorer.score(agent, agent_config_path) -> tuple[float, TestReport]`
building `TestExecutor(agent_config=agent, agent_config_path=path, force_ingest=False)`
and calling `execute_tests()`, then `scalarize`.
**AC:**
- [ ] Against a stubbed backend, returned float equals
      `scalarize(execute_tests().summary, weights)` for the same agent.
- [ ] Uses `force_ingest=False`.
**V:** `pytest tests/unit/optimizer/test_scorer.py -n auto`
**Dependencies:** T2 · **Files:** `optimizer/scorer.py`, 1 test file · **Scope:** S

### 🛑 Checkpoint A
- [ ] `score()` reproduces eval-runner semantics — reviewed before building the loop.

---

## Phase 2 — Loop (prove compounding)

### T4: `OptimizerLoop.run()` + stub proposer ✅
**Description:** Coordinate-descent driver: baseline loss → repeat (numeric phase →
textual phase) until a full cycle yields 0 accepts or `max_cycles`. Accept iff
`best_loss - loss > min_delta` (loss is minimized); on accept `baseline ← candidate`. Per-phase `patience`
+ `max_trials`. Proposers injected (use a stub here).
**AC:**
- [ ] Baseline advances on each accepted improvement (compounding verified).
- [ ] Sub-`min_delta` / tie proposals rejected.
- [ ] Stops on phase patience, `max_cycles`, and dry full cycle.
- [ ] Deterministic given fixed seed + stub.
**V:** `pytest tests/unit/optimizer/test_loop.py -n auto`
**Dependencies:** T3 · **Files:** `optimizer/loop.py`, 1 test file · **Scope:** M

### 🛑 Checkpoint B
- [ ] Loop semantics (compounding, accept rule, stopping) reviewed.

---

## Phase 3 — Numeric proposer

### T5: `proposers/numeric.py` (Optuna TPE) ✅
**Description:** `uv add optuna`. Wrapper creating a **fresh seeded study per numeric
phase**; for each declared numeric axis call `suggest_float/int/categorical` within its
range; return a params dict for `mutator.apply_axes`. Wire into the loop's numeric phase.
**AC:**
- [ ] Fresh study per phase; suggestions respect declared ranges/types.
- [ ] Numeric-only run on a synthetic objective (stub scorer with known optimum) improves
      toward the optimum.
**V:** `pytest tests/unit/optimizer/test_numeric.py -n auto`
**Dependencies:** T4 · **Files:** `optimizer/proposers/numeric.py`, `pyproject.toml`,
1 test file · **Scope:** M

---

## Phase 4 — Textual proposer

### T6: `proposers/textual.py` + Critic/Applier subagents ✅
**Description:** Build failing-case context from the latest baseline `TestReport`; call
Critic (`agents/critic.yaml`) → structured gradient JSON; call Applier
(`agents/applier.yaml`) → structured edit JSON with `new_text`; one textual axis per
trial. Use `BackendSelector` + `invoke_once` structured output. Skip trial on JSON parse
failure (counts toward patience).
**AC:**
- [ ] From a stub backend returning canned gradient + edit JSON, produces an edited prompt.
- [ ] Applied edit changes only `instructions`; other fields untouched.
- [ ] Parse failure → trial skipped, no crash.
**V:** `pytest tests/unit/optimizer/test_textual.py -n auto`
**Dependencies:** T4 · **Files:** `optimizer/proposers/textual.py`,
`optimizer/agents/{critic,applier}.yaml`, 1 test file · **Scope:** L

### 🛑 Checkpoint C
- [ ] Both proposers verified in isolation before CLI wiring.

---

## Phase 5 — Outputs + CLI

### T7: `output.py` artifacts ✅
**Description:** Write `results/optimizer/<run-id>/`: `best.yaml` (serialized best
candidate Agent), `trials.jsonl` (one `TrialRecord` per line), `report.md` (baseline vs
best, accepted edits, per-phase summary).
**AC:**
- [ ] All three files written under a run-id dir.
- [ ] Original `agent.yaml` byte-identical after a run.
- [ ] `trials.jsonl` has one valid row per trial.
**V:** `pytest tests/unit/optimizer/test_output.py -n auto`
**Dependencies:** T4 · **Files:** `optimizer/output.py`, 1 test file · **Scope:** S

### T8: `cli/commands/optimize.py` + register on `test` group ✅
**Description:** `holodeck test optimize [AGENT_CONFIG]` with `--max-cycles`,
`--numeric-max-trials`, `--numeric-patience`, `--textual-max-trials`, `--textual-patience`,
`--seed`, `-o/--output-dir`, `-v/--verbose`, `-q/--quiet`. `asyncio.run(OptimizerLoop...)`.
Stream per-trial scores via `click.echo`. Validate ≥1 test case (else `OptimizerError`,
non-zero exit). Register on the `test` group.
**AC:**
- [ ] `holodeck test optimize --help` lists all flags.
- [ ] Fixture run streams per-trial scores and writes outputs.
- [ ] Missing/empty test cases → non-zero exit with clear message.
**V:** `pytest tests/unit/cli/test_optimize_cli.py -n auto`
**Dependencies:** T5, T6, T7 · **Files:** `cli/commands/optimize.py`, `cli/main.py` (or
`commands/test.py` registration), 1 test file · **Scope:** M

### 🛑 Checkpoint D
- [ ] Manual smoke run on a fixture agent.

---

## Phase 6 — Integration, schema, docs, CI

### T9: E2E test + schema + docs + tasks ✅
**Description:** Integration test on a tiny fixture agent (NLP grader, stub/cheap backend):
baseline → optimize → outputs, assert `best_loss ≤ baseline_loss`. Add
`evaluations.optimizer` to `schemas/agent.schema.json`. Document `holodeck test optimize`
in `docs/` + `AGENTS.md`.
**AC:**
- [ ] E2E test green; `best_loss ≤ baseline_loss`.
- [ ] Schema validates a sample `evaluations.optimizer` block.
- [ ] Docs + AGENTS.md updated.
- [ ] `make ci` clean.
**V:** `pytest tests/integration/test_optimize_e2e.py -n auto && make ci`
**Dependencies:** T8 · **Files:** `tests/integration/test_optimize_e2e.py`,
`tests/fixtures/agents/optimizer_agent.yaml`, `schemas/agent.schema.json`, `docs/`,
`AGENTS.md` · **Scope:** M

---

## Checkpoint: Complete
- [x] All unit + integration tests pass under `-n auto` (4954 unit + optimizer + e2e).
- [x] Smoke run writes `best.yaml`, `trials.jsonl`, `report.md`; source `agent.yaml`
      byte-identical (asserted by `tests/integration/test_optimize_e2e.py`).
- [x] `make format` / `lint` / `security` clean; all optimizer code mypy-clean.
      NOTE: `make type-check` (whole-`src`) reports one **pre-existing**, unrelated
      error — `claude_backend.py:23` unused `# type: ignore` (present on the branch
      before this feature). Not fixed here per surgical-change discipline.

---

## Phase 7 — Checkpointing & resume (post-MVP)

Interruption resilience. Each trial is one full eval (live, billable, minutes long); a
multi-trial run killed mid-flight currently loses everything because `T7` only persists at
the end. The design's `--resume` contract is already written
(`2026-05-16-...-design.md:350-355, 427, 491, 776-790, 936-994`) — **the audit log *is* the
checkpoint** (no separate pickle). This phase builds it.

### T10: Per-trial atomic persistence + deterministic run-id ⬜
**Description:** Persist incrementally instead of only at the end. After **each** completed
trial *and* the baseline, append one `TrialRecord` row to `<run-id>/trials.jsonl` via
atomic write (`tmp` + `os.replace`) + `fsync`. Write `<run-id>/run.json` once on trial 0:
baseline score, seed, and a **fingerprint** = `sha256(agent_yaml + resolved optimizer_config
+ seed)`. Switch `run_id` from the timestamp+uuid to deterministic
`<fingerprint>[:8]` (design.md:804) so resume is unambiguous. In-flight/partial trials
write nothing — resume starts from the last *complete* row.
**AC:**
- [ ] After N trials, `trials.jsonl` has exactly N (+1 baseline) valid rows, each parseable.
- [ ] A `SIGKILL` mid-trial leaves `trials.jsonl` uncorrupted, ending on a complete row.
- [ ] Same `(agent_yaml, config, seed)` → identical `run_id`; any change → different `run_id`.
- [ ] `run.json` round-trips and carries the config fingerprint.
**V:** `pytest tests/unit/optimizer/test_persistence.py -n auto`
**Dependencies:** T7 · **Files:** `optimizer/output.py` (or new `persistence.py`),
`optimizer/loop.py` (per-trial hook), `optimizer/models.py` (`RunMetadata`), 1 test file ·
**Scope:** M

### T11: `--resume` replay in `OptimizerLoop` ⬜
**Description:** Reconstruct loop state from the audit log. Load `trials.jsonl` + `run.json`;
re-derive `best` by replaying accepted rows in order; rebuild each numeric phase's Optuna
study from *its own* rows via `study.add_trial(params, value)` (numeric rows already carry
`params` — design.md:427); carry bests forward; continue the in-flight phase from
`len(trials)+1`. Studies are rebuilt per phase, never merged.
**AC:**
- [ ] Replaying a captured `trials.jsonl` reproduces the same `best`/`best_loss` the
      uninterrupted loop held at that point.
- [ ] A resumed numeric phase's study contains all prior in-phase trials (TPE not reset).
- [ ] Resume continues numbering/cycles from where it stopped (no duplicate trial ids).
**V:** `pytest tests/unit/optimizer/test_resume.py -n auto`
**Dependencies:** T10, T5 · **Files:** `optimizer/loop.py`, `optimizer/proposers/numeric.py`
(study rehydrate), 1 test file · **Scope:** L

### T12: CLI `--resume` + drift guard + signal handling ⬜
**Description:** Add `--resume RUN_ID` (auto-detect the latest incomplete run under
`--output-dir` when the id is omitted). Before resuming, compare the current config
fingerprint to `run.json`; refuse with a clear `OptimizerError` unless `--force`. Handle
`SIGINT` so Ctrl-C exits with code **2** leaving state intact (design.md:994). Document the
crash-recovery workflow.
**AC:**
- [ ] `holodeck test optimize --resume <id>` continues a stopped run to completion.
- [ ] Mismatched config → non-zero exit with a clear message; `--force` overrides.
- [ ] Ctrl-C during a run → exit code 2; a subsequent `--resume` succeeds.
- [ ] Docs (`docs/guides/evaluations.md`, `AGENTS.md`) cover resume + recovery.
**V:** `pytest tests/unit/cli/test_optimize_resume_cli.py -n auto && make ci`
**Dependencies:** T11 · **Files:** `cli/commands/optimize.py`, `docs/`, `AGENTS.md`,
1 test file · **Scope:** M

### 🛑 Checkpoint E
- [ ] Kill a real run mid-trial (SIGINT); `--resume` continues from the last completed
      trial with bests + each phase's TPE study intact, reaching the same accept/reject
      decisions as an uninterrupted run (decision-level, not bit-exact).

---

## Out of scope (spec v1 follow-up)
Train/holdout, `repeats`, variance-aware acceptance, USD/minute budgets,
`trajectory.csv`/iteration files/symlinks, few-shot demos, ingestion-time
axes, parallel trials. (`--resume` + `run.json`: now **Phase 7** above.)
