# Spec: Structured progress event stream for `holodeck test optimize`

**Status:** Draft (motivated by a real Studio integration finding, 2026-06-18)
**Builds on:** `specs/033-holodeck-test-optimizer/` (coordinate-descent optimizer), `specs/037-gepa-optimizer/` (textual engine — emits the same `TrialRecord`s)
**Motivation source:** HoloDeck Studio's live Optimizer tab, which today must regex-scrape `test optimize` stdout to render live progress (real v0.7.0 run captured below).

## Objective

Expose a **stable, versioned, machine-readable progress event stream** for
`holodeck test optimize`, opt-in via a `--progress json` flag, so programmatic consumers can render
a *live* run **without scraping human logs or reading files mid-run**. Today's human output is
preserved unchanged when the flag is off.

**User:** tooling that drives the optimizer as a subprocess — **HoloDeck Studio**'s live Optimizer
tab (primary), plus CI dashboards, notebooks, and any `holodeck test optimize` automation.

**Why now (the concrete failure):** Studio currently parses progress out of stdout with regexes.
A real v0.7.0 run emits:

```
2026-06-18 21:52:19 - holodeck.optimizer.loop - INFO - Baseline loss: 0.5000
  trial 1 [numeric] loss=0.7500 (best 0.5000) — rejected
  trial 2 [textual] loss=0.5000 (best 0.5000) — rejected
2026-06-18 21:56:40 - holodeck.optimizer.loop - INFO - Cycle 0 produced no accepts — stopping.
Baseline loss 0.5000 → best 0.5000 (0 accepted over 1 cycles).
Artifacts written to results/optimizer-smoke/20260618-115036-9b76c2
```

Scraping this is brittle for three reasons, all real:
1. **Format coupling** — wording/spacing/em-dash changes silently break consumers with no error.
2. **Noisy channel** — the same stdout interleaves LiteLLM warnings, gRPC `FD from fork parent …`
   lines, and timestamps.
3. **Missing fields** — the per-trial line has **no cycle index**; `cycle c/N` lives only in separate
   `Cycle N …` lines, so a "cycle 2 of 3" counter can't be built from the trial line.

**The data already exists, structured.** `OptimizerLoop` builds a `TrialRecord`
(`src/holodeck/optimizer/models.py`) per trial and persists it to `trials.jsonl`
(`src/holodeck/optimizer/output.py`); `telemetry.py` already emits per-trial attributes to
OpenTelemetry spans. The gap is only that this structured record is **persisted at the end and sent
to OTel**, never **streamed live to the caller**. This spec adds that third sink.

**Success looks like:** a caller runs `holodeck test optimize --progress json`, reads stdout
line-by-line as NDJSON, and reconstructs the live run (cycle x/N, current trial, running best,
accept/reject, final artifacts) with zero regexes and zero mid-run file reads; human logs are
untouched on stderr; `trials.jsonl`/`best.yaml`/`report.md` are byte-identical to today.

## Decisions (proposed — confirm in scoping)

| Decision | Choice |
| --- | --- |
| Transport | **NDJSON to stdout** under `--progress json`; human logs + library noise routed to **stderr**. A subprocess reads a clean stdout = pure events. |
| Source of truth | The `trial` event payload **is** `TrialRecord.model_dump()` (+ a running `best_loss`). No re-derivation — `trials.jsonl` becomes "the persisted `trial` events." |
| Scope | `test optimize` only. Same `TrialRecord` is produced by the default and GEPA (037) textual engines, so both are covered for free. |
| Backward compatibility | `--progress plain` (**default**) = today's behavior, **byte-identical stdout, zero events**. Purely additive. |
| Versioning | Every line tagged `"schema":"holodeck.optimize.progress/v1"`; breaking changes bump the version, consumers branch on it. |
| Secondary file sink | `--progress-file PATH` (mirror events to a file) — **proposed as optional/fast-follow**, see Open Questions. |

## Tech Stack

- Python 3.10+, Pydantic v2, Click — existing repo stack.
- No new dependencies. The emitter is stdlib `json` + a writable stream.
- Reuses: `OptimizerLoop` (`optimizer/loop.py`), `TrialRecord`/`OptimizerResult` (`optimizer/models.py`),
  the artifact writer (`optimizer/output.py`), and the per-trial hook pattern already in
  `optimizer/telemetry.py`.

## Design

### Config vs. flag

Progress emission is a **CLI/runtime concern, not agent config** — it does not belong in
`evaluations.optimizer` (`optimizer/config.py`). It is controlled entirely by the `--progress` flag
(and optional `--progress-file`), so the same `agent.yaml` behaves identically with or without it.

### Transport: `--progress {plain,json}`

Add to the `test optimize` Click command:

| mode | behavior |
| --- | --- |
| `plain` (default) | unchanged — human logs to stdout, no events |
| `json` | NDJSON events → **stdout** (one object/line, flushed per event); the `holodeck.optimizer.*` log handlers + library noise → **stderr** |

Routing logs to stderr under `json` is the crux: the consumer's stdout is then *only* well-formed
NDJSON, no filtering. Mirrors the `--format json` convention already in the CLI.

### Event schema (NDJSON, versioned)

Each line: one JSON object, `schema` tag + `event` discriminator. In emission order:

| `event` | Fields (besides `schema`,`event`) | Site |
| --- | --- | --- |
| `run_started` | `run_id`, `agent`, `max_cycles`, `axes:{numeric:[{path,type,range}],textual:[{path,max_chars}]}`, `loss_weights:{<metric>:<weight>}`, `started_at` | `OptimizerLoop` entry |
| `baseline` | `loss` | `loop.py:79` (the baseline log site) |
| `cycle_started` | `cycle` (0-based), `of` (= `max_cycles`) | top of each cycle — **supplies cycle x/N** |
| `phase_started` | `cycle`, `phase` (`numeric`\|`textual`) | start of each phase |
| `trial` | **`TrialRecord.model_dump()`** (`trial_id,cycle,phase,loss,baseline_loss,accepted,params,textual_axis,edit_summary,excluded_metrics,error`) **+ `best_loss`** | each trial scored (`loop.py` ~142–195) |
| `phase_completed` | `cycle`, `phase`, `trials`, `accepted` | end of each phase |
| `cycle_completed` | `cycle`, `accepted`, `best_loss`, `stop_reason` (`null`\|`"no_accepts"`) | `loop.py:96` (the cycle-stop site) |
| `run_completed` | `run_id`, `baseline_loss`, `best_loss`, `accepted`, `cycles`, `artifacts:{best_yaml,trials_jsonl,report_md}` | `optimizer/output.py` (run_dir known) |
| `error` | `message`, `fatal` (bool) | recoverable trial error (`TrialRecord.error`) or fatal failure |

Rules:
- One object per line, UTF-8, no pretty-printing, **flushed per event** (live consumers see each
  immediately).
- `cycle` is **0-based**, matching `TrialRecord.cycle` and the existing `Cycle 0 …` log.
- The `trial` event MUST be the same `TrialRecord` instance that is appended to `trials.jsonl` —
  emit then persist, never two code paths.

### Worked example — the captured run as NDJSON (golden fixture)

```jsonl
{"schema":"holodeck.optimize.progress/v1","event":"run_started","run_id":"20260618-115036-9b76c2","agent":"financial-assistant","max_cycles":1,"axes":{"numeric":[{"path":"tools[name=convfinqa_archive].top_k","type":"int","range":[4,12]},{"path":"tools[name=convfinqa_archive].min_score","type":"float","range":[0.0,0.8]}],"textual":[{"path":"instructions.inline","max_chars":6000}]},"loss_weights":{"numeric":1.0},"started_at":"2026-06-18T11:50:36Z"}
{"schema":"holodeck.optimize.progress/v1","event":"baseline","loss":0.5}
{"schema":"holodeck.optimize.progress/v1","event":"cycle_started","cycle":0,"of":1}
{"schema":"holodeck.optimize.progress/v1","event":"phase_started","cycle":0,"phase":"numeric"}
{"schema":"holodeck.optimize.progress/v1","event":"trial","trial_id":1,"cycle":0,"phase":"numeric","loss":0.75,"baseline_loss":0.5,"best_loss":0.5,"accepted":false,"params":{"tools[name=convfinqa_archive].top_k":7,"tools[name=convfinqa_archive].min_score":0.7605},"textual_axis":null,"edit_summary":null,"excluded_metrics":[],"error":null}
{"schema":"holodeck.optimize.progress/v1","event":"phase_completed","cycle":0,"phase":"numeric","trials":1,"accepted":0}
{"schema":"holodeck.optimize.progress/v1","event":"phase_started","cycle":0,"phase":"textual"}
{"schema":"holodeck.optimize.progress/v1","event":"trial","trial_id":2,"cycle":0,"phase":"textual","loss":0.5,"baseline_loss":0.5,"best_loss":0.5,"accepted":false,"params":null,"textual_axis":"instructions.inline","edit_summary":"Added a dedicated \"Year-switching\" subsection …","excluded_metrics":[],"error":null}
{"schema":"holodeck.optimize.progress/v1","event":"phase_completed","cycle":0,"phase":"textual","trials":1,"accepted":0}
{"schema":"holodeck.optimize.progress/v1","event":"cycle_completed","cycle":0,"accepted":0,"best_loss":0.5,"stop_reason":"no_accepts"}
{"schema":"holodeck.optimize.progress/v1","event":"run_completed","run_id":"20260618-115036-9b76c2","baseline_loss":0.5,"best_loss":0.5,"accepted":0,"cycles":1,"artifacts":{"best_yaml":"results/optimizer-smoke/20260618-115036-9b76c2/best.yaml","trials_jsonl":"results/optimizer-smoke/20260618-115036-9b76c2/trials.jsonl","report_md":"results/optimizer-smoke/20260618-115036-9b76c2/report.md"}}
```

### Implementation shape

- **`ProgressEmitter` seam** (new, e.g. `optimizer/progress.py`), modeled on `optimizer/telemetry.py`:
  a `NullEmitter` (default, no-op) and a `JsonlEmitter(stream)` that does
  `stream.write(json.dumps(event) + "\n"); stream.flush()`. The emitter stamps `schema` centrally.
- **Inject into `OptimizerLoop`.** Add `emitter.emit(...)` calls at the sites that already log/produce
  records: baseline (`loop.py:79`), cycle start/stop (`:96` and the cycle top), each `TrialRecord`
  construction (`~:142–195`), and the final `OptimizerResult`. Mirror exactly where `telemetry.py` is
  already invoked per trial — ideally share the call site so telemetry, `trials.jsonl`, and the
  progress stream all consume one record.
- **`trial` payload** = `record.model_dump(mode="json")` + `{"best_loss": self.best_loss}`.
- **`run_completed`** is emitted from `output.py` (or right after it returns) where the run dir and the
  three artifact paths are known.
- **Stderr logging under `--progress json`.** When stdout is the JSON sink, attach the
  `holodeck.optimizer.*` logging handlers to stderr so stdout stays pure.

Additive only — no algorithm changes; `trials.jsonl`, `best.yaml`, `report.md`, OTel spans unchanged.

## Functional requirements

- **FR-001** `holodeck test optimize` gains `--progress {plain,json}`, default `plain`.
- **FR-002** `--progress json` emits NDJSON events to stdout (one object/line, flushed per event) and
  routes human/library logs to stderr.
- **FR-003** Every event carries `schema:"holodeck.optimize.progress/v1"` and an `event` field, and
  follows the ordering invariant: `run_started` → `baseline` → (`cycle_started` → (`phase_started` →
  `trial`+ → `phase_completed`)+ → `cycle_completed`)+ → `run_completed`; exactly one `run_started`
  and one `run_completed`.
- **FR-004** The `trial` event equals the matching `trials.jsonl` row (`TrialRecord`) field-for-field,
  plus a running `best_loss`.
- **FR-005** `cycle_started` carries `{cycle (0-based), of}` so consumers can show cycle x/N.
- **FR-006** `run_completed` carries `baseline_loss`, `best_loss`, `accepted`, `cycles`, and the three
  artifact paths.
- **FR-007** With `--progress plain` (or omitted), stdout is byte-identical to current behavior and no
  events are emitted.
- **FR-008** A recoverable trial error surfaces as a `trial` event with `error != null` (and a
  matching `trials.jsonl` row); a fatal failure surfaces as a terminal `error` event.
- **FR-009** *(optional, see Open Questions)* `--progress-file PATH` mirrors the same NDJSON to a file
  regardless of `--progress` mode, leaving stdout human-readable.
- **FR-010** The optimizer guide and `test optimize --help` document the flag and the event schema.

## Testing

- **Unit (small):** a stubbed `OptimizerLoop` (fake scorer/proposers) drives `JsonlEmitter` and yields
  the §"Worked example" NDJSON deterministically; each event validates against a published JSON Schema.
- **Ordering invariant test:** assert the FR-003 grammar over the emitted sequence.
- **Source-of-truth test:** parse the emitted `trial` events and the written `trials.jsonl`; assert the
  rows are field-equal (FR-004).
- **Backward-compat test:** no flag → zero events, stdout byte-identical to a golden plain run (FR-007).
- **e2e (cheap):** `holodeck test optimize --progress json` on a 1–2 case fixture (a la the captured
  smoke run) → every stdout line parses as JSON, last line is `run_completed`, logs appear on stderr.

## Acceptance criteria

1. FR-001–FR-008 and FR-010 met; FR-009 met or explicitly deferred with a reason.
2. The event stream alone is sufficient to render a live view (cycle x/N, current trial + running
   best + accept/reject, final outcome + artifact paths) with no mid-run file reads.
3. `trials.jsonl`, `best.yaml`, `report.md`, OTel spans, and the optimization result are unchanged.
4. Default-mode stdout is byte-identical to pre-change behavior.

## Open questions

- **`--progress-file` in v1?** Ship `--progress json` (stdout/stderr split) alone, or also the file
  sink? (File decouples events from stdout buffering and fits Studio's "wait on the run dir" model;
  recommend flag in v1, file as fast-follow.)
- **Per-trial cost/latency?** Add `token_usage`/`latency_ms` to the `trial` event for dashboards (the
  runner already tracks per-turn usage)?
- **Does `--progress json` imply `-q`?** Recommend yes for *stdout* (it's the machine channel), while
  `-v` still controls stderr verbosity.

## Consumer adoption note (HoloDeck Studio)

This directly unblocks Studio's live Optimizer tab (which prompted this spec):
- Studio's runner (`src/lib/holodeck/optimize-runner.ts`) spawns with `--progress json`; its existing
  `lineStream` yields one stdout line at a time → each becomes `JSON.parse(line)`.
- `src/components/optimizer/liveRun.ts` drops its two brittle regexes (`TRIAL_RE`, `BASELINE_RE`) and
  switches on `event`. Cycle x/N comes from `cycle_started`; `run_completed.artifacts.trials_jsonl`
  lets it skip the post-run directory poll. The live view becomes a faithful mirror of the real run in
  both seed/fake and live modes, with no format coupling.
