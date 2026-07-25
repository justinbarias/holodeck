# Spec: Selectable Optimizer Backend — GEPA Textual Engine

**Status:** Draft (research + scoping interview, 2026-06-10)
**Builds on:** `specs/033-holodeck-test-optimizer/` (coordinate-descent optimizer, shipped in PR #335)
**Research:** [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (MIT, v0.1.x), [`optimize_anything` blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/), [paper](https://arxiv.org/html/2605.19633v1)

## Objective

Let agent authors choose the strategy used for the **textual** optimization phase of
`holodeck test optimize` via a new `optimizer.type` flag:

- `type: default` (the default) — current behavior: Optuna TPE for numeric axes,
  in-house Critic/Applier (TextGrad-style) for textual axes.
- `type: gepa` — Optuna TPE for numeric axes (unchanged), **GEPA reflective text
  evolution** for textual axes, configured by a new `optimizer.gepa` section.

**User:** HoloDeck agent authors whose instruction tuning plateaus with the greedy
single-candidate Critic/Applier — especially those with multiple textual axes or
eval suites where different prompts win on different cases.

**Why GEPA, scoped to text only (decided):** GEPA mutates *text* exclusively; its
benchmark claim of "matching Optuna" comes from evolving solver *source code*, not
from proposing numeric values. For HoloDeck's small, bounded, expensive-to-evaluate
numeric spaces (`model.temperature`, `top_k`, …), Optuna TPE remains strictly
better (direct, sample-efficient, seed-reproducible). GEPA's genuine advantages are
textual:

1. **Pareto-frontier candidate pool** over per-test-case scores instead of greedy
   accept/reject — resists local optima and prompt collapse.
2. **Native multi-component candidates** — a dict of named text parts evolved
   together, where the in-house proposer degrades to one rewrite per axis.
3. **Minibatch reflective evaluation** — each GEPA metric call runs a *subset* of
   test cases, cheaper per iteration than full-suite-per-trial.
4. **Actionable Side Information** — per-case metric failures from `TestReport`
   feed GEPA's reflection LM as diagnostic context.

**Success looks like:** an author flips `type: gepa`, adds an `optimizer.gepa`
block, reruns `holodeck test optimize`, and gets the same artifacts
(`best.yaml`, `trials.jsonl`, `report.md`) with the textual phases driven by GEPA —
no other workflow changes.

## Decisions (from scoping interview, 2026-06-10)

| Decision | Choice |
| --- | --- |
| GEPA scope | **Text only.** Numeric axes always use Optuna, regardless of `type`. The flag selects the textual strategy. |
| Scoring granularity | **Per-test-case.** Test cases map to GEPA "examples"; the scorer exposes per-case scalarized losses, unlocking Pareto selection and minibatching. |
| Reflection model config | **Reuse HoloDeck `LLMProvider` schema** (`optimizer.gepa.reflection_model`), bridged to GEPA via a callable backed by `BackendSelector`. Defaults to the agent's own `model` when omitted. |
| Dependency | **Optional extra** — `pip install holodeck[gepa]`. `type: gepa` without it installed raises a clear, actionable `ConfigError`. |

## Tech Stack

- Python 3.10+, Pydantic v2, Click — existing repo stack.
- `gepa` (PyPI, MIT) as a new **optional** dependency group `gepa` in `pyproject.toml`.
- Existing: `optuna` (numeric phase, unchanged), `TestExecutor` (scoring, reused).

## Design

### Config schema (`evaluations.optimizer`)

```yaml
evaluations:
  optimizer:
    loss: { relevance: 1.0 }
    type: gepa                      # NEW: "default" | "gepa" (default: "default")
    axes:
      numeric:                      # unchanged — always Optuna
        - path: model.temperature
          type: float
          range: [0.0, 1.0]
      textual:                      # required non-empty when type: gepa
        - path: instructions.inline
          max_chars: 8000
    max_cycles: 3                   # unchanged — coordinate descent retained
    numeric_phase: { max_trials: 10, patience: 5 }   # unchanged
    gepa:                           # NEW section — only valid when type: gepa
      reflection_model:             # optional LLMProvider block; default: agent.model
        provider: anthropic
        name: claude-sonnet-4-6
      max_metric_calls: 150         # per textual phase; one call = one candidate × one minibatch case
      minibatch_size: 3             # examples per reflection step
      merge: true                   # enable GEPA's system-aware merge proposals
```

Validation rules (Pydantic, in `src/holodeck/optimizer/config.py`):

- `type: gepa` requires `axes.textual` to be non-empty (GEPA tunes text only).
- `gepa:` section present with `type: default` → validation error (no silently
  ignored config).
- `type: gepa` with `gepa:` omitted → defaults apply (reflection model = agent's
  model, `max_metric_calls: 150`, `minibatch_size: 3`, `merge: true`).
- `textual_phase:` is ignored when `type: gepa` (GEPA manages its own budget via
  `max_metric_calls`); presence alongside `type: gepa` → validation error, same
  no-silent-config principle.
- `GepaConfig.reflection_model` reuses `holodeck.models.llm.LLMProvider`. Note:
  `config.py` currently imports only Pydantic + stdlib to avoid circular imports
  with `holodeck.models.evaluation` — the plan phase must resolve where
  `GepaConfig` lives so that constraint holds (candidate: keep `GepaConfig` in
  `config.py` with `reflection_model: dict | None` validated/coerced at the CLI
  boundary, or restructure imports).

### Integration shape: GEPA as the textual-phase engine

The coordinate-descent loop (`OptimizerLoop`) is retained. The mismatch is that
GEPA is an *engine that drives its own eval loop* (Pareto pool, minibatches,
budget), not an ask/tell `Proposer`. Resolution:

- Introduce a `TextualPhaseEngine` protocol alongside `Proposer`:
  `async run(best_agent, best_report) -> EngineResult` where `EngineResult`
  carries the best evolved candidate text(s) and per-candidate trial records.
- When `type: gepa`, the textual phase of each cycle calls the engine instead of
  the ask/tell loop. Numeric phases are untouched.
- **Acceptance stays with the loop:** GEPA optimizes internally on per-case
  minibatch scores; when its budget is exhausted, `OptimizerLoop` scores the
  GEPA-best candidate with one **full-suite** run and applies the existing
  `min_delta` accept/reject against the current baseline. Compounding,
  audit-trail, and non-destructive semantics are preserved exactly.

### GEPA adapter mapping

| GEPA concept | HoloDeck mapping |
| --- | --- |
| Candidate | Dict of textual-axis path → current text (multi-axis native). `max_chars` enforced post-mutation; over-limit candidates scored as failed. |
| Example | One test case from the agent's eval suite. |
| Evaluator score | Per-case scalarized loss: `1 − weighted_mean(metric scores)` for that case, reusing `loss.py` weights. Errored metric runs excluded per existing policy. |
| Side information (ASI) | Per-case metric names/scores/reasons from `TestReport`, plus response excerpts. |
| `reflection_lm` | Callable bridging to a HoloDeck backend session built from `gepa.reflection_model` (or agent's model) via `BackendSelector`. |
| trainset / valset | Both = full test-case list (MVP). A holdout split is out of scope. |

### Scorer changes

`scorer.py` gains a per-case mode: evaluate a candidate agent against a *subset*
of test cases and return per-case losses (the existing suite-level `score()` is
retained for baselines, numeric phases, and the post-GEPA validation run).
Constraint carried over from 033: **ingest once per run** — per-case scoring must
reuse the already-ingested vector stores (`force_ingest=False`).

### Observability & artifacts

- Each GEPA candidate evaluation is recorded as a `TrialRecord` with
  `phase="textual"` (aggregated minibatch loss), plus the final full-suite
  validation trial; `trials.jsonl` and `report.md` formats are unchanged, with a
  `report.md` note of the textual strategy in use.
- `OptimizerTelemetry` gains spans for GEPA reflection/mutation steps following
  existing span conventions; root span attribute `holodeck.optimize.type` added.
- Reproducibility caveat (consistent with 033): `type: gepa` is
  *config*-reproducible, not replay-reproducible — LLM-driven mutation is
  non-deterministic; `seed` continues to govern only the Optuna study.

### Dependency handling

- `pyproject.toml`: `[project.optional-dependencies] gepa = ["gepa>=0.1,<0.2"]`.
- Import is lazy and guarded: `type: gepa` with the package missing →
  `ConfigError("optimizer.type 'gepa' requires the gepa extra: pip install holodeck[gepa]")`.
- `schemas/agent.schema.json` regenerated to include `type` and `gepa` fields.

## Commands

```bash
source .venv/bin/activate
uv sync --extra gepa                      # install with the optional extra
holodeck test optimize agent.yaml         # entry point (unchanged)
pytest tests/unit/optimizer/ -n auto -v   # unit tests
make format && make lint && make type-check && make security
make ci
```

## Project Structure

```
src/holodeck/optimizer/
  config.py                 # + type field, GepaConfig, cross-field validators
  loop.py                   # + TextualPhaseEngine dispatch in textual phase
  scorer.py                 # + per-case scoring mode (subset of test cases)
  proposers/
    base.py                 # + TextualPhaseEngine protocol, EngineResult
    gepa_engine.py          # NEW: GEPA adapter + engine (lazy gepa import)
src/holodeck/cli/commands/optimize.py     # _build_proposers branches on type
schemas/agent.schema.json   # regenerated
tests/unit/optimizer/
  test_gepa_config.py       # NEW: config validation matrix
  test_gepa_engine.py       # NEW: adapter mapping, engine result, import guard
specs/037-gepa-optimizer/   # this spec + plan/tasks to follow
docs/                       # optimizer docs page: type flag + gepa section
```

## Code Style

Match the existing optimizer module exactly — Google-style docstrings, strict
typing, `ConfigDict(extra="forbid")`, validators raising `ValueError` with
path-bearing messages. Example, in the established idiom:

```python
class GepaConfig(BaseModel):
    """Typed view of the ``evaluations.optimizer.gepa`` block."""

    model_config = ConfigDict(extra="forbid")

    max_metric_calls: int = Field(
        default=150, gt=0, description="Per-phase budget of GEPA metric calls."
    )
    minibatch_size: int = Field(
        default=3, gt=0, description="Examples per reflection step."
    )
    merge: bool = Field(
        default=True, description="Enable GEPA merge proposals."
    )
```

Errors via `holodeck.lib.errors` (`ConfigError`, `OptimizerError`); `logging`
not `print`; async throughout the engine path.

## Testing Strategy

Framework: pytest (`-n auto`), AAA, `@pytest.mark.unit` for all of the below.
GEPA's engine is mocked at the adapter boundary — no live LLM calls in unit tests.

- **Config validation matrix:** `type` default value; `gepa:` with
  `type: default` rejected; `type: gepa` without textual axes rejected;
  `textual_phase` with `type: gepa` rejected; defaults applied when `gepa:` omitted.
- **Import guard:** `type: gepa` without the package installed raises the
  actionable `ConfigError` (simulate via `sys.modules` patching).
- **Adapter mapping:** candidate dict ↔ textual axes round-trip; per-case loss
  computation matches `scalarize` semantics on single-case reports; ASI content
  includes metric names/reasons; `max_chars` enforcement.
- **Engine/loop integration:** GEPA-best candidate goes through full-suite
  validation and existing `min_delta` accept/reject; rejected GEPA result leaves
  baseline untouched; trial records written for engine candidates.
- **Regression:** entire existing optimizer suite passes unchanged with
  `type: default` (and with `type` omitted).
- **Schema:** regenerated `agent.schema.json` accepts the new fields and rejects
  unknown keys under `gepa:`.

## Boundaries

- **Always:** keep `type: default` behavior byte-for-byte identical; lazy-import
  `gepa`; reuse `TestExecutor`/ingested stores (ingest once per run); never
  mutate the original `agent.yaml`; record every candidate evaluation in
  `trials.jsonl`; run `make format && make lint && make type-check && make security`
  after each task.
- **Ask first:** adding any transitive dependency beyond the `gepa` extra itself;
  changing `TrialRecord`/`OptimizationResult` schemas; changing `TestExecutor`'s
  public interface for per-case scoring; pinning a different `gepa` version range.
- **Never:** use GEPA for numeric axes or `optimize_anything`-style code
  evolution; send agent data to any service beyond the configured LLM providers;
  break `schemas/agent.schema.json` backward compatibility for existing configs;
  remove or weaken the in-house Critic/Applier path.

## Success Criteria

1. `optimizer.type` accepts `default`/`gepa`, defaults to `default`, and existing
   agent.yaml files validate and run unchanged.
2. With `type: gepa` and the extra installed, `holodeck test optimize` completes
   a run where textual phases are GEPA-driven and numeric phases are Optuna-driven,
   producing `best.yaml`, `trials.jsonl`, `report.md` in the existing formats.
3. GEPA receives per-test-case scores and diagnostic side information; its metric
   calls run case subsets without re-ingestion.
4. GEPA-best candidates are accepted only after a full-suite validation clears
   `min_delta` — the audit trail shows both the engine trials and the validation
   trial.
5. `type: gepa` without the extra installed fails fast with the actionable error;
   without textual axes fails validation.
6. `make ci` green; new unit tests cover the validation matrix, import guard,
   adapter mapping, and loop integration.

## Out of Scope

- GEPA for numeric hyperparameters or `optimize_anything` code/config evolution.
- A `gepa`-replaces-the-whole-loop mode.
- trainset/valset holdout splits for GEPA (both = full suite in MVP).
- Few-shot demonstration optimization (still the #1 v2 item from 033).
- Parallel candidate evaluation (`EngineConfig.parallel`) — single-flight MVP,
  matching the existing loop's sequential trials.

## Open Questions

1. **Where does `GepaConfig` live** given `config.py`'s no-`holodeck.models`
   import constraint, if `reflection_model` reuses `LLMProvider`? (Plan-phase
   decision; two candidates noted in Design.)
2. **Trial-record fidelity:** is one `TrialRecord` per GEPA candidate
   (aggregated minibatch loss) sufficient for the audit trail, or should
   per-case minibatch scores be embedded (e.g., a `gepa` detail field)?
3. **Budget semantics check:** `max_metric_calls` is per textual phase (so a
   3-cycle run can spend up to 3×); confirm this is the intended cost model vs.
   a whole-run budget.
4. **`gepa` version pin:** `>=0.1,<0.2` assumed; verify API stability of
   `GEPAAdapter` across recent releases during plan phase.
