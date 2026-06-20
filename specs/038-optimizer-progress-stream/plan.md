# Implementation Plan: Structured progress event stream for `holodeck test optimize`

**Spec:** `specs/038-optimizer-progress-stream/spec.md`
**Status:** Ready for review (scoping decisions resolved 2026-06-18)
**Builds on:** `033-holodeck-test-optimizer` (the `OptimizerLoop`), `037-gepa-optimizer` (spec-only today; reuses the same `OptimizerLoop` seam, so it inherits this stream for free).

## Overview

Add an opt-in, versioned NDJSON progress stream to `holodeck test optimize` so a
subprocess (HoloDeck Studio's live Optimizer tab, CI, notebooks) can reconstruct a
live run without scraping human logs or reading artifacts mid-run. A new
`--progress {plain,json}` flag selects the channel; `plain` (default) is byte-identical
to today. The structured `TrialRecord` already exists and is persisted to
`trials.jsonl` and sent to OTel — this feature adds a **third sink** that streams the
same records (plus lifecycle events) live to stdout, with human/library logs routed to
stderr. Additive only: no algorithm change; `trials.jsonl`, `best.yaml`, `report.md`,
and OTel spans stay unchanged.

## Scoping decisions (resolved with the requester)

| Question | Decision | Consequence |
| --- | --- | --- |
| `best_loss` on an accepted trial | **Post-decision running best** (accepted → = this trial's loss; rejected → unchanged) | The scored-trial path is reordered so the `trial` event carries the *post*-accept best, while `TrialRecord.baseline_loss` stays = the *pre*-trial best (captured in a local). Keeps `trials.jsonl` byte-identical. |
| Event representation | **Pydantic event models** (discriminated union on `event`) | JSON Schema is generated from the models via `model_json_schema()`, published to `schemas/`, and validated in tests. Prevents key typos; matches the repo's Pydantic-only standard. |
| v1 scope of optional extras | **Defer both** `--progress-file` (FR-009) and per-trial `token_usage`/`latency_ms` | v1 = `--progress {plain,json}` stdout/stderr split + flag only. `trials.jsonl` stays byte-identical (acceptance #3). Both become fast-follows. |
| Plan/task location | `specs/038-optimizer-progress-stream/` | This file + `tasks.md`, beside `spec.md`, matching siblings 033/037. |

### Minor decisions taken with sensible defaults (no further input needed)

- **`started_at`** — the CLI computes `run_id` and `started_at` from one
  `datetime.now(timezone.utc)` and passes `started_at` to the loop, so the
  `run_started.started_at` matches the run-id-encoded timestamp. The loop falls back to
  stamping at `run()` entry when no value is passed (keeps direct-construction tests simple).
- **`run_completed.artifacts` paths** — emitted exactly as `write_outputs` resolves them
  (`<output-dir>/<run-id>/{best.yaml,trials.jsonl,report.md}`), i.e. relative when
  `--output-dir` was relative. Matches the worked example and is usable from the
  subprocess cwd.
- **`--progress json` is the machine channel** — under `json`, all human `click.echo`
  ("Optimizing…", final summary) is redirected to **stderr** (`err=True`) and the
  human per-trial streamer (`on_trial`) is disabled for stdout (the `trial` event covers
  it). `-v`/`-q` still control stderr verbosity. (Implements the spec's "implies `-q`
  for stdout" recommendation.)
- **Golden-fixture comparison** — the worked-example test compares **parsed JSON
  objects** (one `json.loads` per line), not raw bytes, so JSON field ordering is never a
  source of fragility.

## Architecture decisions

1. **A `ProgressEmitter` seam modeled on `optimizer/telemetry.py`.** `NullEmitter`
   (default, strict no-op) and `JsonlEmitter(stream)` (writes `model_dump_json()+"\n"`,
   flushes per event, stamps `schema` centrally). Injected into `OptimizerLoop` exactly
   like `OptimizerTelemetry`, so the loop behaves identically whether or not the caller
   opts in. The existing `progress_callback` (human stdout streamer) is left untouched —
   the emitter is an independent additional sink.
2. **The loop owns the run/cycle/phase/trial events; the CLI owns
   `run_completed` and the fatal `error`.** `run_completed` needs the three artifact
   paths, which are only known after `write_outputs` returns (in the CLI). A fatal
   failure is caught in the CLI's `except` blocks, so the terminal `error` event is
   emitted there. Everything else is emitted at the sites that already log/produce records.
3. **The `trial` event is built from the same `TrialRecord` instance** that is appended
   to `trials.jsonl` (`Trial(best_loss=…, **record.model_dump())`), guaranteeing FR-004
   field parity. A source-of-truth test enforces it, so a future `TrialRecord` field
   change fails loudly until the event is updated.
4. **Stdout purity under `--progress json`.** Logs already go to stderr
   (`logging_config.py:131`). The remaining stdout writers are the command's own
   `click.echo` calls → redirected to stderr under `json`. A test asserts no
   `holodeck.*` log handler targets stdout in `json` mode (guards the observability-on edge).

## Dependency graph

```
Task 1: progress.py — event models + emitter + published JSON Schema   (foundation)
   │
   ├── Task 2: wire OptimizerLoop (run_started … trial … cycle_completed)
   │              │
   │              └── Task 3: CLI --progress flag + stdout/stderr split
   │                          + run_completed + fatal error
   │                             │
   │                             └── Task 4: docs (guide + --help)
```

Bottom-up order; each task leaves the system green (plain mode always works; `json`
becomes reachable after Task 3).

## Task list

### Phase 1 — Event contract (foundation)
- [ ] **Task 1:** `progress.py` — Pydantic event models, `ProgressEmitter`/`NullEmitter`/`JsonlEmitter`, published `schemas/optimize-progress.schema.json`.

#### Checkpoint: Contract
- [ ] Emitter unit tests green; schema published and self-validating; `make type-check` clean for the new module.

### Phase 2 — Loop wiring
- [ ] **Task 2:** Emit `run_started`, `baseline`, `cycle_started`, `phase_started`, `trial` (post-decision `best_loss`), `phase_completed`, `cycle_completed` from `OptimizerLoop`.

#### Checkpoint: Loop emits the run
- [ ] Worked-example NDJSON reproduced deterministically; ordering-invariant (FR-003) and source-of-truth (FR-004) tests green.
- [ ] **All pre-existing optimizer tests still pass** (NullEmitter default ⇒ no behavior change).

### Phase 3 — CLI transport
- [ ] **Task 3:** `--progress {plain,json}` flag; stdout/stderr split; `run_completed` after `write_outputs`; terminal `error` on failure; plain mode byte-identical.

#### Checkpoint: End-to-end stream
- [ ] `--progress json` e2e: every stdout line parses as JSON, last line is `run_completed`, logs on stderr.
- [ ] Plain-mode stdout byte-identical to a golden pre-change run (FR-007).
- [ ] `trials.jsonl`, `best.yaml`, `report.md` byte-identical between a plain and a `json` run on the same seed/fixture (acceptance #3).

### Phase 4 — Documentation
- [ ] **Task 4:** Optimizer guide section + `--help` text + schema reference (FR-010).

#### Checkpoint: Complete
- [ ] All FR-001–FR-008 and FR-010 met; FR-009 explicitly deferred (recorded here).
- [ ] `make format && make lint && make type-check && make test` green; `make security` clean.

## Functional-requirement → task map

| FR | Task |
| --- | --- |
| FR-001 (`--progress {plain,json}`, default plain) | 3 |
| FR-002 (NDJSON→stdout, logs→stderr) | 3 |
| FR-003 (schema tag + ordering invariant) | 1 (tag), 2 (ordering) |
| FR-004 (trial event == trials.jsonl row + best_loss) | 2 |
| FR-005 (`cycle_started` cycle/of) | 2 |
| FR-006 (`run_completed` summary + artifacts) | 3 |
| FR-007 (plain byte-identical, zero events) | 3 |
| FR-008 (recoverable trial error → `trial.error`; fatal → terminal `error`) | 2 (recoverable), 3 (fatal) |
| FR-009 (`--progress-file`) | **Deferred** (fast-follow) |
| FR-010 (docs + `--help`) | 4 |

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Reorder for post-decision `best_loss` accidentally changes `TrialRecord.baseline_loss` → `trials.jsonl` drift | High | Capture pre-trial best in a local; `baseline_loss` keeps using it. Add a byte-identical `trials.jsonl` assertion (plain vs json run, same seed). |
| Observability console exporter writes spans/logs to **stdout** under `json`, polluting NDJSON | Med | Logs already on stderr. Add a test asserting no `holodeck.*` handler targets stdout in `json` mode; document that `--progress json` assumes non-stdout OTel export. Verify console-exporter target during Task 3. |
| Golden NDJSON test brittle to JSON field order | Low | Compare parsed objects, not raw bytes. |
| `Trial` event drifts from `TrialRecord` if a field is later added | Med | Build the event from `record.model_dump()`; source-of-truth test fails until updated. |
| Default `NullEmitter` not wired → existing loop callers break | Med | `emitter` defaults to `NullEmitter()`; existing tests construct the loop without it and must stay green (Phase 2 checkpoint). |

## Deferred (fast-follow, out of scope for v1)

- **`--progress-file PATH`** (FR-009) — mirror NDJSON to a file regardless of mode.
- **Per-trial `token_usage`/`latency_ms`** on the `trial` event — requires new
  `TrialRecord` fields (would change `trials.jsonl`); revisit when acceptance #3 can relax.
- **GEPA (037)** — not yet implemented; it reuses `OptimizerLoop`, so it inherits this
  stream once built. No work here.

## Open questions for review

- None blocking. The four scoping decisions above are locked; confirm the deferral of
  FR-009 and cost/latency is acceptable for the Studio integration's v1.
