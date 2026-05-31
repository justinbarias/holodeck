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
**Description:** Coordinate-descent driver: baseline score → repeat (numeric phase →
textual phase) until a full cycle yields 0 accepts or `max_cycles`. Accept iff
`score - best_score > min_delta`; on accept `baseline ← candidate`. Per-phase `patience`
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

### T5: `proposers/numeric.py` (Optuna TPE)
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

### T6: `proposers/textual.py` + Critic/Applier subagents
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

### T7: `output.py` artifacts
**Description:** Write `results/optimizer/<run-id>/`: `best.yaml` (serialized best
candidate Agent), `trials.jsonl` (one `TrialRecord` per line), `report.md` (baseline vs
best, accepted edits, per-phase summary).
**AC:**
- [ ] All three files written under a run-id dir.
- [ ] Original `agent.yaml` byte-identical after a run.
- [ ] `trials.jsonl` has one valid row per trial.
**V:** `pytest tests/unit/optimizer/test_output.py -n auto`
**Dependencies:** T4 · **Files:** `optimizer/output.py`, 1 test file · **Scope:** S

### T8: `cli/commands/optimize.py` + register on `test` group
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

### T9: E2E test + schema + docs + tasks
**Description:** Integration test on a tiny fixture agent (NLP grader, stub/cheap backend):
baseline → optimize → outputs, assert `best_score ≥ baseline_score`. Add
`evaluations.optimizer` to `schemas/agent.schema.json`. Document `holodeck test optimize`
in `docs/` + `AGENTS.md`.
**AC:**
- [ ] E2E test green; `best_score ≥ baseline_score`.
- [ ] Schema validates a sample `evaluations.optimizer` block.
- [ ] Docs + AGENTS.md updated.
- [ ] `make ci` clean.
**V:** `pytest tests/integration/test_optimize_e2e.py -n auto && make ci`
**Dependencies:** T8 · **Files:** `tests/integration/test_optimize_e2e.py`,
`tests/fixtures/agents/optimizer_agent.yaml`, `schemas/agent.schema.json`, `docs/`,
`AGENTS.md` · **Scope:** M

---

## Checkpoint: Complete
- [ ] All unit + integration tests pass under `-n auto`.
- [ ] Smoke run writes `best.yaml`, `trials.jsonl`, `report.md`; source `agent.yaml`
      byte-identical.
- [ ] `make format lint type-check security` clean.

## Out of scope (spec v1 follow-up)
Train/holdout, `repeats`, variance-aware acceptance, USD/minute budgets, `--resume`,
`trajectory.csv`/`run.json`/iteration files/symlinks, few-shot demos, ingestion-time
axes, parallel trials.
