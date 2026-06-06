# Optimizing Agents (`holodeck test optimize`)

Once you have an evaluation suite, `holodeck test optimize` automates the
hand-tuning loop — change a knob or instruction, re-run `holodeck test`, eyeball
the score, repeat — as a **compounding coordinate-descent optimizer**. It
alternates a *numeric* phase (Optuna TPE over declared query-time axes) and a
*textual* phase (a Critic produces a natural-language "gradient", an Applier
rewrites the instructions), advancing a best-candidate baseline on every
accepted improvement so wins compound into one improved `agent.yaml`.

The original `agent.yaml` is **never modified**. Every trial is logged, and the
best candidate is written to `results/optimizer/<run-id>/best.yaml`.

## How it works

The optimizer nests four concepts. From smallest unit to largest:

| Term | What it is |
| --- | --- |
| **Trial** | The atomic unit. One candidate config is scored = one full eval pass over your test set = one `loss` number. Every row in `trials.jsonl` is a trial. |
| **Phase** | A sweep over *one kind of axis*. The **numeric** phase tunes the numeric knobs (e.g. `top_k`, `min_score`) via Optuna; the **textual** phase rewrites instruction text. Each phase runs many trials. |
| **Cycle** | One numeric phase **followed by** one textual phase. This is coordinate descent: *freeze the text, tune the numbers; then freeze the numbers, tune the text.* Repeating cycles lets each kind of change compound on the other's gains. |
| **Baseline** | The unchanged agent, scored once at the very start. The bar every trial must beat (by `min_delta`) to be accepted. |

So the loop nests **run → cycles → phases → trials → (one eval pass each)**:

```
RUN (max_cycles: 2)
│
├─ baseline ─────────────────► score once  →  the bar to beat
│
├─ CYCLE 1
│   ├─ NUMERIC PHASE  (max_trials 10, patience 4)
│   │     trial → score   tweak top_k/min_score/…, keep if it beats best
│   │     trial → score          ⋮  (stops early after 4 misses in a row)
│   │     … up to 10
│   └─ TEXTUAL PHASE  (max_trials 6, patience 3)
│         trial → score   rewrite instructions, keep if it beats best
│         trial → score          ⋮  (stops early after 3 misses in a row)
│         … up to 6
│
└─ CYCLE 2          ← starts from cycle 1's best, repeats both phases
    ├─ NUMERIC PHASE  …
    └─ TEXTUAL PHASE  …

best.yaml = the single lowest-loss candidate found across every trial
```

`max_cycles` controls the outer loop; each phase's `max_trials` is a **ceiling**,
not a guarantee — `patience` stops a phase early once it stalls (that many
consecutive non-improving trials). So the example above runs *up to*
`2 × (10 + 6) = 32` trials plus 1 baseline = **33 scoring passes** worst-case,
and often far fewer.

> Each trial is a full evaluation run (real LLM + metric work), so trial count is
> the main driver of wall-time and token cost. Size your budgets accordingly.

## Configuration

Add an `evaluations.optimizer` block declaring the scalarized loss (per-metric
weights; loss = `1 − weighted_mean`) and the axes to tune:

```yaml
evaluations:
  metrics:
    - type: standard
      metric: groundedness
    - type: geval
      name: Conciseness
      criteria: "The response is concise and avoids redundancy."
  optimizer:
    loss:                       # metric weights; loss = 1 - weighted_mean
      groundedness: 2.0
      Conciseness: 1.0
    axes:
      numeric:                  # query-time hyperparameters (Optuna TPE)
        - path: model.temperature
          type: float
          range: [0.0, 1.0]
        - path: tools[name=knowledge_base].top_k
          type: int
          range: [3, 12]
      textual:                  # instruction text rewritten by Critic/Applier
        - path: instructions.inline
          max_chars: 6000
    max_cycles: 3               # numeric→textual cycles before stopping
    numeric_phase: { max_trials: 12, patience: 5 }
    textual_phase: { max_trials: 5, patience: 3 }
    min_delta: 0.01             # minimum loss reduction required to accept
    seed: 42
```

**Axis paths** support dotted attributes (`model.temperature`,
`instructions.inline`) and a `tools[name=X].<field>` selector for per-tool
fields. Numeric axes accept `float`/`int` (a `[low, high]` range) or
`categorical` (a list of choices).

**Phase budgets.** `numeric_phase.{max_trials,patience}` cap the Optuna trials
in a numeric phase. For the textual phase the budget drives *iterative
refinement*: with a **single** textual axis, `textual_phase.max_trials` is the
number of successive Critic→Applier refinement steps taken on that axis (each
step builds on the previous attempt and the failing cases it produced), and
`patience` stops the phase after that many consecutive non-improving steps. A
drifting chain never regresses the result — only an accepted, loss-improving
step advances the best agent. With **more than one** textual axis the proposer
falls back to a single rewrite per axis (iterative multi-axis ordering is not
yet supported).

## Running

```bash
holodeck test optimize agent.yaml
holodeck test optimize agent.yaml --max-cycles 2 --numeric-max-trials 20 --seed 7
```

CLI flags (`--max-cycles`, `--numeric-max-trials`, `--numeric-patience`,
`--textual-max-trials`, `--textual-patience`, `--seed`, `-o/--output-dir`)
override the YAML config for a single run. The command streams per-trial losses
and prints the baseline → best summary on completion.

## Outputs

`results/optimizer/<run-id>/` contains:

- `best.yaml` — the best candidate agent, ready to copy over your original.
  Secrets stay templated as `${VAR}` (rebuilt from the unsubstituted source), so
  the file never leaks resolved credentials.
- `trials.jsonl` — one record per trial (the full audit trail).
- `report.md` — baseline vs best, the accepted edits, and a per-phase summary.

## Acceptance and scoring

The optimizer **minimizes a loss** of `1 − weighted_mean`, where the weighted
mean is a renormalized weighted average of the per-metric averages using your
`loss` weights (so a perfect agent has loss `0.0`). Metric scores must be
normalized to `[0, 1]`; an average outside that range aborts the run. Metric runs
that **error** are excluded from the mean (not scored as zero); legitimate `0.0`
scores are kept. A candidate is accepted only when its loss undercuts the current
best by more than `min_delta`.

> **Note (MVP):** acceptance uses the raw loss delta — there is no
> train/holdout split, repeated trials, or variance-aware bar yet, so a small
> `min_delta` can chase evaluation noise. Keep `min_delta` above your suite's
> single-case-flip granularity. Statistical rigor is the planned v1 follow-up.

## Observability

`holodeck test optimize` emits OpenTelemetry traces and metrics on the same
terms as `holodeck test`: set an `observability` block on the agent with
`enabled: true` and an OTLP exporter, and the optimize run exports to your
collector (e.g. Aspire, Grafana). No extra configuration — it reuses the agent's
existing `observability` block. When observability is disabled the optimizer
behaves exactly as before (no spans, no metrics).

**Span tree.** One root span per run, with each trial's evaluation GenAI spans
nesting under its trial span:

```
holodeck.optimize                 run_id, agent_name, max_cycles, seed, axes counts, loss
├── holodeck.optimize.baseline    baseline scoring of the original agent
└── holodeck.optimize.cycle       one coordinate-descent cycle
    └── holodeck.optimize.phase   numeric | textual (records accepts)
        ├── holodeck.optimize.propose   textual Critic/Applier calls (GenAI spans nest)
        └── holodeck.optimize.trial     trial_id, phase, baseline_loss, loss, accepted,
            └── <eval GenAI spans>        edit_summary, axis, params (JSON), error
```

Trial spans carry only primitives — numeric params as a single JSON string,
the textual axis *name* and a human-readable `edit_summary` — never instruction
text or resolved secrets.

**Metrics** (all `holodeck.optimize.*`, attributed by `phase` where meaningful):

| Metric | Type | Meaning |
| --- | --- | --- |
| `trials` | counter | completed trials (`phase`, `accepted`) |
| `trials.skipped` | counter | trials skipped because a proposer errored (`phase`) |
| `trial.loss` | histogram | candidate loss per trial (`phase`) |
| `trial.duration` | histogram (s) | scorer wall-time per trial (`phase`) |
| `best_loss` | histogram | best loss after each accepted improvement (`phase`) |
| `improvement` | histogram | `baseline_loss − best_loss` at run end |
| `cycles` | counter | completed coordinate-descent cycles |

## Next Steps

- See [Evaluations](evaluations.md) for building the metric suite the optimizer scores against.
- See [Observability](observability.md) for wiring up an OTLP collector and dashboard.
- See the [CLI Reference](../api/cli.md#holodeck-test-optimize) for the full flag list.
