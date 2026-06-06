# Cost & Latency-Aware Optimization

## Problem Statement
How might we let the optimizer trade agent quality against real cost and latency
— so "best" means "good enough, cheaply enough" — without breaking the
`loss = 1 − weighted_mean` objective (loss ∈ [0,1]) or the compounding
single-`best_loss` accept rule that drives the coordinate-descent loop?

## Recommended Direction
**Treat cost and latency as a hard budget (feasibility gate), not as terms in
the loss.** Quality remains the sole objective the optimizer *minimizes*; a
candidate is accepted only if it (a) beats the current best loss by `min_delta`
AND (b) stays within the declared cost and latency ceilings.

This keeps `scalarize` and the [0,1] loss invariant completely intact, leaves
Optuna's `direction="minimize"` and the compounding loop unchanged, and answers
the founding concern — runaway model cost — with a ceiling expressed in real
dollars (which the absolute-price-table + priced-token choices make natural).
The accept rule gains one conjunct; nothing else in the objective moves.

Cost basis is **priced tokens**: `Σ over test cases (tokens × per-model rate)`,
sourced from a price table in the optimizer config. Local models (Ollama) are
priced at $0, so for them the cost ceiling is vacuous and the **latency ceiling
carries the economic signal** — which is exactly the regime worth caring about.

## Key Assumptions to Validate
- [ ] **`token_usage` is populated by BOTH backends.** Cost is uncomputable if
      the SK/Ollama path returns `None`. → Run one trial each on Claude and an
      Ollama agent; assert `report.results[*].token_usage` is non-null. This is
      the #1 kill risk — if SK doesn't fill it, the cost gate can't fire.
- [ ] **Latency is stable enough to gate on.** Wall-clock on a shared/dev box
      with cold starts and network jitter is high-variance; a flaky gate would
      reject good candidates nondeterministically. → Measure run-to-run variance
      of `execution_time_ms` for a fixed agent; if noisy, ship latency as
      *recorded-only* first and gate on cost alone.
- [ ] **A maintainable price table is acceptable.** Rates drift. → Ship a small
      default map for common models + YAML override; unknown model → error (not
      silent $0).

## MVP Scope
**In:**
- `optimizer/cost.py`: a `PriceTable` (model_id → $/1M prompt+completion[+cache]
  rates) and `report_cost(report, prices) -> float`, `report_latency(report) ->
  float` (mean of `execution_time_ms` across test cases).
- `OptimizerConfig.budget`: optional `{ cost_max: $, latency_max_ms: ms,
  prices: {model_id: rates} }`. Absent budget ⇒ current behaviour, unchanged.
- Accept rule in `loop.py`: `feasible = within(cost_max) and within(latency_max)`;
  `accepted = feasible and (best_loss − loss > min_delta)`.
- `TrialRecord.cost` / `TrialRecord.latency_ms` + columns in `report.md`.
- Infeasible-baseline handling: if the original agent already exceeds budget,
  accept the first *feasible* candidate regardless of loss, then resume the
  normal rule (so the run can't dead-end against an unreachable baseline).
- Tests: cost computation from a known token_usage + price table; gate rejects
  an over-budget improvement; gate accepts an in-budget improvement; infeasible
  baseline path.

**Out:** soft penalty / λ weights, Pareto, percentile (p95) latency, automatic
price fetching.

## Not Doing (and Why)
- **Penalty term in the loss** — forces λ to fuse unit-conversion with
  preference and breaks the [0,1] invariant. Revisit only if the binary budget
  proves too coarse (the "won't economize within budget" gap).
- **Pareto / multi-objective** — no single `best_loss`; that's the spine of the
  compounding accept rule. It's a rewrite, not an extension.
- **Latency gating by default if it proves noisy** — fall back to recording it
  and gating cost only, rather than shipping a flaky gate.

## Open Questions
- Latency aggregate: mean (simple, less noisy to explain) vs p95 (SLA-meaningful
  but needs more runs)? MVP leans mean.
- Cache-token pricing: fold `cache_read`/`cache_creation` at their own rates, or
  approximate into prompt rate for v1?
- Where does the price table live — inline in `agent.yaml` under
  `evaluations.optimizer.budget.prices`, or a shared `~/.holodeck` price file
  referenced by path?
