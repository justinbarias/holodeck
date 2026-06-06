# Implementation Plan: `holodeck test optimize` — OpenTelemetry wiring

> **Goal:** bring `holodeck test optimize` to observability parity with `holodeck test`,
> then layer an **optimizer-specific span tree and metric set** on top so a run is fully
> traceable: one root span per run, nested cycle → phase → trial spans, the existing
> GenAI spans from each trial's eval nesting underneath, plus optimizer KPIs (loss,
> accept rate, improvement, trial duration) as OTel metrics.
>
> **Why now:** `optimize.py` never calls `initialize_observability()` — unlike
> `test.py:219`, `chat.py:190`, `serve.py:126`. The visible symptom is the
> `claude_backend` WARNING *"Observability context not initialized; GenAI instrumentation
> disabled"* on every trial, and **zero** GenAI traces/metrics for optimize runs even when
> the agent's `observability.enabled: true`. Authoritative observability design lives under
> `src/holodeck/lib/observability/`; this plan does not change that subsystem, only consumes it.

## Overview

Three stacked slices, each independently shippable and verifiable:

1. **Context parity** — wire `initialize_observability` / `shutdown_observability` into the
   `optimize` command exactly as `test.py` does, reordering the early `setup_logging` so the
   branch can read `agent.observability`. Wrap `loop.run()` in one root span. *Effect:* warning
   gone; each trial's eval emits GenAI spans/metrics, all nested under one optimize root span.
2. **Optimizer span tree** — instrument `OptimizerLoop` with guarded, no-op-when-disabled spans:
   `baseline`, `cycle`, `phase`, `trial` (and a `propose` span around textual Critic/Applier
   calls). The trial span is opened *around the scorer call* so the eval's GenAI spans nest
   correctly.
3. **Optimizer metrics** — a small instrument set via `get_meter` (trial counter, accept/skip
   counters, loss + best-loss + improvement histograms/gauges, trial-duration histogram),
   recorded from the loop, guarded the same way.

Then tests (no-op-when-disabled + emit-when-enabled via in-memory exporters), docs, and CI.

## Architecture decisions

1. **Mirror `test.py` for context lifecycle, not invent a new one.** Same import trio
   (`ObservabilityContext`, `initialize_observability`, `shutdown_observability`), same
   "OTel replaces `setup_logging` when `agent.observability.enabled`" branch, same `finally:
   shutdown_observability(obs_context)` (`test.py:216-224,393-396`). Keeps one mental model.
2. **Reorder logging in `optimize.py`.** Today `setup_logging(verbose, quiet)` runs at
   `optimize.py:153` *before* `load_agent_with_config` at `:158`. The observability branch needs
   the loaded `agent`, so logging setup moves *after* the load — `setup_logging` only in the
   `else` (observability-disabled) arm, exactly like `test.py:216-224`.
3. **Instrument the loop, not just the CLI.** Cycle/phase/trial spans must be *active while the
   scorer runs* so the eval's backend GenAI spans become their children. A post-hoc
   `progress_callback` can't do that (it fires after the trial). So `OptimizerLoop` gains guarded
   spans/metrics directly. This couples the loop to the observability package the same way
   `TestExecutor` already is — acceptable and idiomatic.
4. **Guarded no-op pattern, never a hard dependency.** Every span uses
   `tracer.start_as_current_span(...) if get_observability_context() is not None else
   nullcontext()` (the `test.py:319-327` pattern). Instruments are created once per run and only
   when context exists; when disabled the loop behaves byte-for-byte as today. No new config keys,
   no behavior change to acceptance/scoring.
5. **No prompt/secret text in span attributes.** Record `edit_summary`, axis *names*, numeric
   *params*, losses, booleans, counts — never full `instructions` text or resolved
   `${VAR}` values. `RedactingSpanProcessor` runs on the provider, but we do not lean on it for
   correctness: the optimizer already guards secrets in `best.yaml` (`optimize.py:229-240`,
   `loss.py`), and span attributes follow the same rule. Keeps spans small and leak-free.
6. **`holodeck.optimize.*` namespace, GenAI-style attribute keys.** Custom optimizer spans/metrics
   are namespaced `holodeck.optimize.*` (the codebase is "OpenTelemetry Native … GenAI semantic
   conventions" per CLAUDE.md). We add optimizer KPIs, not redefine GenAI ones — the per-trial
   GenAI spans/metrics still come from the backend unchanged.

## Verified integration points

- **CLI to patch:** `src/holodeck/cli/commands/optimize.py` — `setup_logging` at `:153`
  (reorder), agent load at `:158`, `asyncio.run(loop.run())` at `:224`, broad `finally`-less
  `try/except` at `:155-257` (add `finally: shutdown_observability`).
- **Pattern source:** `src/holodeck/cli/commands/test.py:188,216-224,317-332,393-396` — context
  var init, OTel-vs-`setup_logging` branch, parent-span async wrapper (`run_tests_with_cleanup`),
  `finally` shutdown.
- **Loop seams:** `src/holodeck/optimizer/loop.py` — `run()` (`:70-101`, baseline + cycle loop),
  `_run_phase()` (`:103-175`, per-trial body wraps `self.scorer(candidate)` at `:141`),
  `_record()` (`:177-181`). These are the exact span/metric emission points.
- **Provider API:** `from holodeck.lib.observability import get_tracer, get_meter,
  get_observability_context, initialize_observability, shutdown_observability,
  ObservabilityContext` (all exported, `observability/__init__.py:34-64`).
  `get_tracer(__name__).start_as_current_span(name)`; `get_meter(__name__).create_counter(...)
  / .create_histogram(...)`.
- **Enabled flag:** `agent.observability and agent.observability.enabled`
  (`models/observability.py`; same predicate as `test.py:216`).
- **Trial data already on hand for attributes:** `TrialRecord` fields — `trial_id, cycle, phase,
  loss, baseline_loss, accepted, params, textual_axis, edit_summary, error`
  (`optimizer/models.py:10-52`). Run-level: `run_id`, `config.max_cycles`, `config.seed`,
  `len(config.axes.numeric)`, `len(config.axes.textual)`, `config.loss`.
- **Per-trial GenAI source (the warning):** `src/holodeck/lib/backends/claude_backend.py:~2350`
  early-returns + WARNs when `get_observability_context()` is None. Fixed transitively by Phase 1.
- **No-op-when-disabled idiom:** `from contextlib import nullcontext` (`test.py:11,326`).
- **Timing:** no per-trial duration exists today; add `time.perf_counter()` around the scorer
  call in `_run_phase` to feed the duration histogram (Phase 3).

## Span tree (target)

```
holodeck.optimize                      (root; run_id, agent.name, max_cycles, seed,
│                                        axes.numeric/textual counts, loss weights)
├── holodeck.optimize.baseline         (baseline_loss)
│   └── <GenAI spans from the baseline eval>      ← free once context is up
├── holodeck.optimize.cycle            (cycle index)
│   ├── holodeck.optimize.phase        (phase=numeric|textual, cycle, accepts)
│   │   ├── holodeck.optimize.propose  (textual only: critic/applier; phase, axis)
│   │   │   └── <GenAI spans from Critic/Applier invoke_once>
│   │   └── holodeck.optimize.trial    (trial_id, phase, loss, baseline_loss,
│   │       │                            accepted, axis|params-keys, edit_summary, error)
│   │       └── <GenAI spans from the trial eval>  ← nests because span is active
│   │                                                 around self.scorer(candidate)
│   └── …
└── …
```

## Metrics (target)

All instruments `holodeck.optimize.*`, created once per run via `get_meter(__name__)`, recorded
from the loop with attributes `{phase, accepted}` where meaningful.

| Instrument | Type | Recorded | Attributes |
| --- | --- | --- | --- |
| `holodeck.optimize.trials` | Counter | +1 per completed trial | `phase`, `accepted` |
| `holodeck.optimize.trials.skipped` | Counter | +1 per errored/skipped trial | `phase` |
| `holodeck.optimize.trial.loss` | Histogram | candidate loss per trial | `phase` |
| `holodeck.optimize.trial.duration` | Histogram (s) | scorer wall-time per trial | `phase` |
| `holodeck.optimize.best_loss` | Histogram | best loss after each accept | `phase` |
| `holodeck.optimize.improvement` | Histogram | `baseline_loss − best_loss` at run end | — |
| `holodeck.optimize.cycles` | Counter | +1 per completed cycle | — |

> Gauges are avoided (OTel sync gauges need observable callbacks); a histogram of `best_loss`
> sampled at each accept gives the same "loss over time" story in Aspire/Grafana without
> callback plumbing.

## Task list

### Phase 1: Context parity (the slice that kills the warning)
- [x] **T1 — Wire observability lifecycle into `optimize.py`.** _(done — commit `9cb04fe`;
  helper extraction deferred to Phase 2, params-on-span = single JSON string per build decisions.)_
  Import the trio; add `obs_context: ObservabilityContext | None = None`; reorder so
  `load_agent_with_config` runs first, then branch: `initialize_observability(agent.observability,
  agent.name, verbose=verbose, quiet=effective_quiet)` when enabled, else the existing
  `setup_logging`. Wrap `loop.run()` in an async helper that opens
  `get_tracer(__name__).start_as_current_span("holodeck.optimize")` (root attrs) when context
  exists, else `nullcontext()`. Add `finally: if obs_context: shutdown_observability(obs_context)`
  to the outer `try`.
  **Acceptance:**
  - With `observability.enabled: true`, the `claude_backend` "context not initialized" WARNING no
    longer appears during an optimize run.
  - GenAI spans from every trial's eval appear and nest under a single `holodeck.optimize` root span.
  - With observability disabled (or absent), behavior + logging are unchanged from today.
  **Verification:** `pytest tests/unit/optimizer/ -n auto`;
  unit test asserting `initialize_observability`/`shutdown_observability` called once when enabled
  and never when disabled (mirror any existing `test.py` observability test);
  manual: `holodeck test optimize <agent.yaml> --max-cycles 1 --numeric-max-trials 1` against an
  agent with observability enabled → no warning in logs.
  **Dependencies:** none.
  **Files:** `src/holodeck/cli/commands/optimize.py`, `tests/unit/cli/test_optimize_command*.py`.
  **Scope:** Small (1 src + 1 test).

### Checkpoint A — parity
- [ ] Warning gone; one root span per run with GenAI children; disabled path untouched; `make lint type-check` clean.

### Phase 2: Optimizer span tree
- [ ] **T2 — Guarded span helper + root/baseline/cycle spans in `OptimizerLoop`.**
  Add a tiny internal helper `self._span(name, **attrs)` returning
  `tracer.start_as_current_span(name, attributes=...)` or `nullcontext()` based on
  `get_observability_context()`. Open `holodeck.optimize.baseline` around the baseline scorer call
  in `run()`; open `holodeck.optimize.cycle` per cycle iteration. (Root span owned by the CLI from T1.)
  **Acceptance:** baseline + one cycle span per cycle, correctly parented; no-op when disabled.
  **Verification:** unit test with an `InMemorySpanExporter` asserting span names/parentage for a
  2-cycle stub-proposer run; a second run with context disabled asserts zero optimizer spans.
  **Dependencies:** T1.
  **Files:** `src/holodeck/optimizer/loop.py`, `tests/unit/optimizer/test_loop_observability.py`.
  **Scope:** Small–Medium.
- [ ] **T3 — Phase, trial, and propose spans in `_run_phase`.**
  Open `holodeck.optimize.phase` around the phase loop (attrs `phase`, `cycle`; set `accepts` at
  end). For textual proposers, open `holodeck.optimize.propose` around `proposer.ask()`. Open
  `holodeck.optimize.trial` around `self.scorer(candidate)` so the eval's GenAI spans nest;
  set attrs from the `TrialRecord` (`trial_id, phase, loss, baseline_loss, accepted`, axis name or
  numeric param *keys*, `edit_summary`, `error`). Record skipped-trial spans for the
  `proposal.error` branch too.
  **Acceptance:** trial spans wrap the scorer; GenAI spans from the eval are children of the trial
  span; textual `propose` spans wrap Critic/Applier; **no instruction text or secrets** in attributes.
  **Verification:** extend the in-memory-exporter test: assert a trial span exists per trial,
  that a stubbed "genai" child span (emitted inside the stub scorer) is parented to the trial span,
  and that no attribute value contains the agent's instruction text. Assert numeric params are
  recorded as keys/values without secret-looking strings.
  **Dependencies:** T2.
  **Files:** `src/holodeck/optimizer/loop.py`, same test module.
  **Scope:** Medium.

### Checkpoint B — span tree
- [ ] Full `baseline/cycle/phase/trial/propose` tree verified in-memory; GenAI nesting proven; disabled = zero optimizer spans.

### Phase 3: Optimizer metrics
- [ ] **T4 — Instrument set + per-trial timing.**
  Create the 7 instruments once per run (guarded). Wrap the scorer call with
  `time.perf_counter()` to feed `trial.duration`. Record `trials`(+`accepted` attr),
  `trials.skipped`, `trial.loss`, `trial.duration` per trial; `best_loss` on each accept;
  `cycles` per cycle; `improvement` once in `run()` before returning the result.
  **Acceptance:** all 7 instruments emit with correct attributes during an enabled run; nothing
  emits when disabled; scoring/acceptance results are identical with metrics on vs off.
  **Verification:** unit test with an `InMemoryMetricReader` asserting each instrument's
  presence, point count, and `{phase, accepted}` attributes for a small stub run; a disabled run
  asserts the reader collected nothing from `holodeck.optimize.*`.
  **Dependencies:** T1 (context), T2 (helper). Independent of T3 but ordered after for a clean tree.
  **Files:** `src/holodeck/optimizer/loop.py`, `tests/unit/optimizer/test_loop_observability.py`.
  **Scope:** Medium.

### Checkpoint C — metrics
- [ ] 7 instruments verified; identical optimization decisions with metrics on/off.

### Phase 4: Docs, schema check, CI
- [ ] **T5 — Docs + CI sweep.**
  Document the optimize span tree + metric table (where the optimizer is documented — `AGENTS.md`
  / `docs/` / this spec folder's `optimizer.md`); note that an agent must set
  `observability.enabled: true` (+ an OTLP exporter) for optimize traces to flow, identical to
  `holodeck test`. Confirm **no `agent.schema.json` change** is needed (no new config keys — the
  feature reuses the existing `observability` block). Run `make format lint type-check security`
  and the full `tests/unit/optimizer` + `tests/unit/cli` suites.
  **Acceptance:** docs updated; `make ci` clean; schema unchanged (drift guard still green).
  **Verification:** `make format && make lint && make type-check && make security`;
  `pytest tests/unit/optimizer tests/unit/cli -n auto`;
  `python scripts/generate_agent_schema.py --check`.
  **Dependencies:** T1–T4.
  **Files:** `AGENTS.md` and/or `docs/…`, `specs/033-holodeck-test-optimizer/optimizer.md`.
  **Scope:** Small.

### Checkpoint: Complete
- [ ] `make ci` clean; manual optimize run on an observability-enabled agent shows the full
  `holodeck.optimize` span tree with nested GenAI spans in the collector (Aspire), the 7
  optimizer metrics present, and **no** "context not initialized" warning; disabled agents run
  exactly as before.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Reordering `setup_logging` after agent load changes early log behavior | Med | Mirror `test.py` exactly; keep `setup_logging` in the disabled branch; load errors already surface via the outer `try/except`. Add a test for the disabled path. |
| Trial span doesn't parent GenAI spans (wrong context activation) | High | Use `start_as_current_span` (sets current context) *around* `self.scorer(candidate)`; prove nesting with an in-memory exporter test (T3). |
| Instruction text or resolved secrets leak into span attributes | High | Decision 5: only summaries/names/params/numbers as attrs; test asserts instruction text absent; `RedactingSpanProcessor` is defense-in-depth, not the primary guard. |
| Coupling generic `OptimizerLoop` to the observability package | Low | Guarded import + `nullcontext`; same precedent as `TestExecutor`; disabled path is a strict no-op (tested). |
| MeterProvider/MetricReader lifecycle warnings (seen in prior run logs) | Low | Instruments created only after `initialize_observability`; `shutdown_observability` force-flushes then shuts down providers (`providers.py:354-386`); never create instruments when context is None. |
| Per-trial duration timing adds overhead | Low | `time.perf_counter()` is ~free; only the histogram record is guarded. |

## Out of scope

- Changes to the observability subsystem itself (exporters, redaction, providers).
- New config keys or schema changes (reuses the existing `observability` block).
- `--resume` checkpoint spans (belongs to `plan.md` Phase 7).
- Cost/latency-aware optimization signals (separate idea doc:
  `docs/ideas/cost-latency-aware-optimization.md`).
- Backfilling observability into other CLI commands (already wired in `test/chat/serve`).
