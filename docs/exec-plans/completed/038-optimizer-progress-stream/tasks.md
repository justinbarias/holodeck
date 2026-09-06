# Tasks: Structured progress event stream for `holodeck test optimize`

Companion to `plan.md`. Tasks are dependency-ordered and vertically sliced. Run
`make format && make lint && make type-check` after each task (project workflow rule).
Use `pytest -n auto`.

---

## Task 1: Event models, emitter seam, and published JSON Schema

**Description:** Create the versioned event contract and the emitter seam that all later
tasks depend on. New module `src/holodeck/optimizer/progress.py` defines the nine event
types as a Pydantic discriminated union (on `event`), each carrying
`schema="holodeck.optimize.progress/v1"`; a `ProgressEmitter` protocol with a no-op
`NullEmitter` and a `JsonlEmitter(stream)` that writes one `model_dump_json()` line per
event and flushes. Generate and commit `schemas/optimize-progress.schema.json` from the
union.

**Event models (fields beyond `schema`,`event`):**
- `RunStarted` — `run_id, agent, max_cycles, axes:{numeric:[{path,type,range}], textual:[{path,max_chars}]}, loss_weights:{<metric>:<weight>}, started_at`
- `Baseline` — `loss`
- `CycleStarted` — `cycle` (0-based), `of`
- `PhaseStarted` — `cycle, phase` (`numeric`|`textual`)
- `Trial` — all `TrialRecord` fields (`trial_id,cycle,phase,loss,baseline_loss,accepted,params,textual_axis,edit_summary,excluded_metrics,error`) **+ `best_loss`**; constructed via `Trial(best_loss=…, **record.model_dump(mode="json"))`
- `PhaseCompleted` — `cycle, phase, trials, accepted`
- `CycleCompleted` — `cycle, accepted, best_loss, stop_reason` (`null`|`"no_accepts"`)
- `RunCompleted` — `run_id, baseline_loss, best_loss, accepted, cycles, artifacts:{best_yaml,trials_jsonl,report_md}`
- `ErrorEvent` (`event:"error"`) — `message, fatal`

**Acceptance criteria:**
- [ ] All nine event models construct, serialize to NDJSON (no pretty-printing), and carry the `schema` tag centrally (not repeated by callers).
- [ ] `JsonlEmitter` writes exactly one JSON object per line and flushes per event; `NullEmitter.emit(...)` is a no-op.
- [ ] `schemas/optimize-progress.schema.json` is generated from the union and validates every line of the spec's "Worked example" fixture.

**Verification:**
- [ ] Unit tests pass: `pytest tests/unit/optimizer/test_progress.py -n auto`
- [ ] Each worked-example line validates against the published schema via `jsonschema` (already a dep).
- [ ] Drift test: committed `schemas/optimize-progress.schema.json` equals the freshly generated schema.
- [ ] `make type-check` clean for `progress.py`.

**Dependencies:** None.

**Files likely touched:**
- `src/holodeck/optimizer/progress.py` (new)
- `schemas/optimize-progress.schema.json` (new)
- `tests/unit/optimizer/test_progress.py` (new)
- `tests/fixtures/optimizer/worked_example.jsonl` (new — the spec's golden NDJSON)

**Estimated scope:** Medium (3–5 files)

---

## Task 2: Emit lifecycle events from `OptimizerLoop`

**Description:** Inject `emitter: ProgressEmitter = NullEmitter()` into
`OptimizerLoop.__init__` (default no-op preserves all current callers/tests) and emit the
run/cycle/phase/trial events at the sites that already log/produce records. Reorder the
scored-trial path minimally so the `trial` event carries the **post-decision** running
`best_loss` while `TrialRecord.baseline_loss` keeps using the **pre-trial** best (captured
in a local). The `trial` event is built from the same `TrialRecord` appended to
`trials.jsonl`. Add `started_at` (optional; stamp at `run()` entry if not passed).

**Emission sites:**
- `run_started` — top of `run()`; build `axes` from `config.axes` (numeric path/type/range, textual path/max_chars), `loss_weights` from `config.loss`, `agent`=`original_agent.name`, `max_cycles`, `run_id`, `started_at`.
- `baseline` — after baseline scoring (`loop.py:78–79`).
- `cycle_started` — top of the cycle loop, `{cycle, of: config.max_cycles}`.
- `phase_started` — start of `_run_phase`, `{cycle, phase}`.
- `trial` — for scored trials, **after** the accept/reject decision (post-decision `best_loss`); for skipped/error trials, at the skip-record site (best unchanged). Same `TrialRecord` instance as persisted.
- `phase_completed` — end of `_run_phase`, `{cycle, phase, trials:<count>, accepted}`.
- `cycle_completed` — end of each cycle, `{cycle, accepted, best_loss, stop_reason}` (`"no_accepts"` when `accepts==0`, else `null`).

**Acceptance criteria:**
- [ ] A stubbed loop (fake scorer/proposers) driving a `JsonlEmitter`→`StringIO` reproduces the spec's worked-example sequence (parsed-object equality).
- [ ] Emitted sequence satisfies the FR-003 grammar: one `run_started` → `baseline` → (`cycle_started` → (`phase_started` → `trial`+ → `phase_completed`)+ → `cycle_completed`)+ → exactly the loop's events (no `run_completed` here — CLI owns it).
- [ ] On an **accepting** trial, `trial.best_loss == trial.loss` and `trial.baseline_loss ==` the pre-trial best; on a rejecting trial, `trial.best_loss` is unchanged.
- [ ] Each emitted `trial` event's `TrialRecord` subset is field-equal to the matching `trials.jsonl` row (FR-004).
- [ ] A recoverable proposer error emits a `trial` event with `error != null` and a matching `trials.jsonl` row (FR-008, recoverable half).

**Verification:**
- [ ] `pytest tests/unit/optimizer/test_loop.py tests/unit/optimizer/test_progress_loop.py -n auto`
- [ ] **All pre-existing optimizer tests pass unchanged** (NullEmitter default): `pytest tests/unit/optimizer -n auto`
- [ ] `trials.jsonl` produced by `write_outputs` is byte-identical before/after the reorder (compare against a captured golden).

**Dependencies:** Task 1.

**Files likely touched:**
- `src/holodeck/optimizer/loop.py`
- `tests/unit/optimizer/test_progress_loop.py` (new)
- `tests/unit/optimizer/test_loop.py` (add ordering/source-of-truth/best_loss cases)

**Estimated scope:** Medium (3–5 files)

---

## Task 3: CLI transport — `--progress` flag, stdout/stderr split, `run_completed`, fatal `error`

**Description:** Add `--progress [plain|json]` (default `plain`) to the `optimize` command.
Under `json`: construct `JsonlEmitter(sys.stdout)` and pass it to the loop; redirect the
command's human `click.echo` calls ("Optimizing…", final summary) to **stderr**
(`err=True`); disable the `on_trial` stdout streamer (the `trial` event replaces it);
emit `run_completed` after `write_outputs` returns (artifact paths known); emit a terminal
`error` event (`fatal=true`) on caught failure before `sys.exit(1)`. Under `plain`: use
`NullEmitter` and keep today's behavior byte-identical.

**Acceptance criteria:**
- [ ] `--progress` defaults to `plain`; `plain`/omitted ⇒ zero events and stdout byte-identical to pre-change (FR-001, FR-007).
- [ ] `--progress json` ⇒ every stdout line is a single JSON object; first line `run_started`, last line `run_completed`; all human text and logs appear on **stderr** (FR-002).
- [ ] `run_completed` carries `baseline_loss, best_loss, accepted, cycles` and `artifacts.{best_yaml,trials_jsonl,report_md}` resolved as `<output-dir>/<run-id>/…` (FR-006).
- [ ] A fatal failure under `json` emits a terminal `error` event (`fatal=true`) to stdout, then exits non-zero (FR-008, fatal half).
- [ ] No `holodeck.*` log handler targets stdout in `json` mode (stdout-purity guard).

**Verification:**
- [ ] CLI unit tests (CliRunner with `mix_stderr=False`): `pytest tests/unit/cli/test_optimize_cli.py -n auto`
- [ ] e2e: `pytest tests/integration/test_optimize_e2e.py -n auto` — a `--progress json` run on the stub-backend fixture yields a fully-ordered stream ending in `run_completed`.
- [ ] `trials.jsonl`, `best.yaml`, `report.md` byte-identical between a `plain` and a `json` run on the same seed/fixture (acceptance #3).
- [ ] Manual: `holodeck test optimize --progress json <fixture> 1>events.ndjson 2>logs.txt` → `events.ndjson` is pure NDJSON; `logs.txt` holds the human/log output.

**Dependencies:** Task 2.

**Files likely touched:**
- `src/holodeck/cli/commands/optimize.py`
- `tests/unit/cli/test_optimize_cli.py`
- `tests/integration/test_optimize_e2e.py`

**Estimated scope:** Medium (3–5 files)

---

## Task 4: Documentation — guide section, `--help`, schema reference

**Description:** Document the flag and event schema (FR-010). Add a "Progress event
stream" section to `docs/guides/optimizer.md` (the `--progress` flag, NDJSON to stdout /
logs to stderr, the event table, the `schema` version tag and bump policy, and the Studio
consumer note). Ensure the `--progress` option's Click `help=` text is clear. Reference
the published `schemas/optimize-progress.schema.json`.

**Acceptance criteria:**
- [ ] `docs/guides/optimizer.md` documents `--progress {plain,json}`, the stdout/stderr split, every event type, and the versioning policy.
- [ ] `holodeck test optimize --help` describes `--progress` and its default.
- [ ] The guide links to / references the published JSON Schema file.

**Verification:**
- [ ] `holodeck test optimize --help` shows the `--progress` option (manual).
- [ ] Docs build / markdown lint passes if configured; otherwise visual review.
- [ ] Full local CI: `make format && make lint && make type-check && make test && make security`.

**Dependencies:** Task 3.

**Files likely touched:**
- `docs/guides/optimizer.md`
- `src/holodeck/cli/commands/optimize.py` (help text only)

**Estimated scope:** Small (1–2 files)

---

## Checkpoints (mirrors `plan.md`)

- **After Task 1 — Contract:** ✅ emitter + schema tests green (9 passed); schema published and self-validating.
- **After Task 2 — Loop emits the run:** ✅ worked-example, ordering, and source-of-truth tests green (8 passed); all pre-existing optimizer tests still pass (97).
- **After Task 3 — End-to-end stream:** ✅ plain emits zero events; `json` e2e ends in `run_completed` with artifacts; artifacts byte-identical across modes; CLI (7) + integration (3) green; `make format`/`lint`/`type-check` clean. 557 passed overall.
- **After Task 4 — Complete:** ⏳ docs deferred (not requested in this pass); FR-009 deferred; full `make ci`/`security` pending.

## Pre-implementation verification (skill checklist)

- [x] Every task has acceptance criteria and a verification step.
- [x] Dependencies identified and ordered (1 → 2 → 3 → 4).
- [x] No task touches more than ~5 files.
- [x] Checkpoints exist between phases.
- [ ] Human has reviewed and approved this plan.
