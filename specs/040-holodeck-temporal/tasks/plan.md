# Implementation Plan: HoloDeck Agents on Temporal (spec 040)

## Overview

Build the four deliverables of `specs/040-holodeck-temporal/spec.md`: an
activity library that turns an `EdgeNode` into a Temporal activity (D1), gate
validation inside the activity (D2), deterministic helpers that pass Temporal's
workflow sandbox (D3), and a `holodeck worker` CLI that hosts agents from a
`worker.yaml` (D4). The 036 primitives (`edge.py`, `table_eval.py`, `feel.py`,
`models/workflow.py`, `models/decision_table.py`) are reused, never rewritten.

There is deliberately no no-code way to author a workflow. `worker.yaml`
registers agents as named activities; the workflow that calls them is always
user-authored Temporal Python.

## Architecture Decisions (grilling session, 2026-08-29)

All decisions were put to the user and confirmed. SDK facts were verified
against a live install of `temporalio 1.32.0` (closures as activities,
`pydantic_data_converter`, `SandboxedWorkflowRunner.prepare_workflow`,
`ApplicationError` mapping, activities-only workers, `TracingInterceptor`
span nesting).

| # | Decision |
|---|---|
| 1 | The activity factory takes an `EdgeNode` (id + `edge.agent` + `gate.schema`). Gate is mandatory by construction; `resolve_agent_path` + `load_gate_schema` are the only path/schema seams. |
| 2 | No ungated agent activity exists in v1. An escape hatch is an ask-first decision. |
| 3 | Activity input is a typed Pydantic model: `AgentActivityInput {message: str, context: dict \| None}`. |
| 4 | Activity return is an envelope: `AgentActivityResult {output: <gate-validated dict>, token_usage, num_turns, agent_id}`. `output` is the canonical value (FR-008); raw model text never crosses. |
| 5 | Stateless: one activity call = one `invoke_once`. No `AgentSession` continuity in v1. |
| 6 | Retryability is decided activity-side with typed errors. Gate failure → retryable `GateValidationError`; authoring faults (bad agent.yaml, unloadable gate, path escape) → `ApplicationError(non_retryable=True)`; backend transport / model `is_error` → retryable. SC-003's model-vs-authoring split, translated to Temporal. |
| 7 | Decision tables load at workflow-module import time (policy-as-code, versioned with workflow code). Not loaded in activities, not passed through history. |
| 8 | D4 config is `worker.yaml` (Pydantic model) + env-var overrides + `--config`/`--task-queue` flags. |
| 9 | Edge-node entries are declared inline in `worker.yaml` under `nodes:`; paths resolve relative to the worker.yaml directory through `resolve_agent_path`. `nodes:` is registration only — zero control flow. |
| 10 | *(amended 2026-08-29 after Codex review of PR #366)* Timeout/retry options are caller-side in Temporal: they ride the workflow's `execute_activity` command, and the caller must set `start_to_close_timeout` or `schedule_to_close_timeout` — no server default exists. `ActivityParameters` is therefore a **workflow-side, sandbox-safe helper** (ships with the D3 surface) that expands into `execute_activity` kwargs; `worker.yaml` carries no execution timeouts (registration only). `heartbeat_timeout` is unsupported in v1 — the activity does not heartbeat, and exposing the knob without heartbeats invites concurrent duplicate LLM calls. |
| 11 | Activity name = edge-node `id` (replay-load-bearing; rename-safe against file moves). |
| 12 | `temporalio` ships only in the `holodeck[temporal]` extra. Import guard gives a clear install message. |
| 13 | Exact pin `temporalio==1.32.0` — the sandbox-safety unit test leans on APIs marked "not yet stable". Same policy precedent as `bkflow-feel`. Loosen when the sandbox surface stabilizes. |
| 14 | Developer API is both layers: factory functions are the testable core; `HoloDeckPlugin` (a `SimplePlugin` that sets the pydantic data converter and registers activities) is sugar over them. |
| 15 | `temporalio.contrib.pydantic.pydantic_data_converter` is required and set by the plugin; payload models stay Pydantic. |
| 16 | Sandbox-safety testing is two layers: `SandboxedWorkflowRunner.prepare_workflow` as a fast serverless unit test, plus Worker-init validation in the integration suite as the public-API backstop. |
| 17 | OTel: the activity emits the existing GenAI spans from the agent's observability config; `holodeck worker` also enables `TracingInterceptor` so GenAI spans nest under Temporal's activity spans. The experimental `OpenTelemetryPlugin`/ReplaySafe providers are not adopted. |
| 18 | Sample is the hardship-determination story (evidence-extractor agent → DMN table → letter-writer agent), committed under `tests/integration/temporal/fixtures/` with a thin runnable copy in git-ignored `sample/`. |

Granularity note: `temporalio.contrib.openai_agents` puts the agent loop in
workflow code and makes each model call an activity. Spec 040 does the
opposite — one activity per whole agent run, gate at the activity boundary.
What transfers from that precedent: the plugin pattern, instance-method
activities carrying worker-side state (credentials never enter payloads or
history), and the timeout/retry parameters object.

## Task List

### Phase 1: Foundation (D1 core)

- [x] T1: `holodeck[temporal]` extra, exact pin, package skeleton
- [x] T2: Payload and parameter models
- [ ] T3: Activity factory
- [ ] T4: Error taxonomy and retry classification

### Checkpoint 1: Foundation
- [ ] Factory output is a valid Temporal activity definition (mocked backend)
- [ ] `make test-unit`, `make lint`, `make type-check`, `make security` clean

### Phase 2: Deterministic helpers + plugin (D3, D1 sugar)

- [ ] T5: D3 helper surface + sandbox-safety unit test
- [ ] T6: `HoloDeckPlugin`

### Checkpoint 2: Helpers
- [ ] D3 modules pass `prepare_workflow` sandbox validation
- [ ] Plugin registers activities and sets the pydantic data converter

### Phase 3: Worker CLI (D4)

- [ ] T7: `WorkerConfig` model and loader
- [ ] T8: `holodeck worker` command

### Checkpoint 3: Worker
- [ ] `holodeck worker --help` works without temporalio installed (guarded import)
- [ ] Worker starts (mocked client) from a fixture worker.yaml

### Phase 4: Integration, sample, acceptance criteria

- [ ] T9: Hardship fixtures
- [ ] T10: Integration tests AC-1/AC-2/AC-3
- [ ] T11: Integration tests AC-4/AC-5 + sandbox Worker-init backstop
- [ ] T12: OTel test AC-6
- [ ] T13: Live smoke test, `sample/` copy, docs, index row
- [ ] T14: Gate-schema codegen (`holodeck generate models`)

### Checkpoint: Complete
- [ ] AC-1 through AC-6 demonstrated by tests
- [ ] `make ci` clean; spec status updated in `specs/index.md`

## Tasks

### Task 1: `holodeck[temporal]` extra, exact pin, package skeleton

**Description:** Add the optional dependency and the package that guards it.
`pyproject.toml` gains extra `temporal = ["temporalio==1.32.0"]` with a comment
naming the sandbox-test dependency as the reason for the exact pin.
`src/holodeck/temporal/__init__.py` raises a clear `ConfigError` ("install
holodeck[temporal]") on import when `temporalio` is missing.

**Acceptance criteria:**
- [x] `uv sync --extra temporal` installs `temporalio==1.32.0` on Python 3.10
- [x] Importing `holodeck.temporal` without the extra raises the guarded error, not `ModuleNotFoundError`
- [x] Core install (`uv sync`) does not pull `temporalio`

**Verification:**
- [x] `pytest tests/unit/temporal/test_import_guard.py -n auto`
- [x] `make lint && make type-check && make security`

**Dependencies:** None
**Files likely touched:** `pyproject.toml`, `uv.lock`, `src/holodeck/temporal/__init__.py`, `tests/unit/temporal/test_import_guard.py`
**Estimated scope:** S

### Task 2: Payload and parameter models

**Description:** Pydantic models in `src/holodeck/temporal/models.py`:
`AgentActivityInput` (message, optional context dict), `AgentActivityResult`
(output, token_usage, num_turns, agent_id — all JSON-serializable),
`ActivityParameters` (start_to_close / schedule_to_close / schedule_to_start
timeouts, retry policy fields; no heartbeat — decision 10). Payload models
round-trip through `pydantic_data_converter`. `ActivityParameters` is a
workflow-side helper: `to_activity_kwargs()` expands it into
`execute_activity` keyword arguments, and validation requires at least one of
`start_to_close`/`schedule_to_close` (Temporal has no server default). The
module must stay sandbox-safe (it ships to workflow code with the D3 surface).

**Acceptance criteria:**
- [ ] Payload models serialize/deserialize through `temporalio.contrib.pydantic` converter
- [ ] `AgentActivityResult.output` holds a plain dict (the gate-validated object), never model text
- [ ] `ActivityParameters.to_activity_kwargs()` yields valid `RetryPolicy`/timedelta kwargs; a parameters object with neither closing timeout is refused at validation

**Verification:**
- [ ] `pytest tests/unit/temporal/test_models.py -n auto`
- [ ] Quality gates as in T1

**Dependencies:** T1
**Files likely touched:** `src/holodeck/temporal/models.py`, `tests/unit/temporal/test_models.py`
**Estimated scope:** S

### Task 3: Activity factory

**Description:** `src/holodeck/temporal/activity.py`:
`agent_activity(node: EdgeNode, base_dir: Path) -> Callable`. (No timeout/retry
kwargs — those are caller-side, decision 10.)
Resolves the agent through `edge.resolve_agent_path`, loads the gate through
`load_gate_schema`, invokes through `BackendSelector` (`invoke_once`,
stateless), validates `structured_output` against the gate, returns the
envelope. Activity name = `node.id` via `activity.defn(name=...)`. Backend
and gate are bound at factory time (worker-side state; nothing secret in
payloads). Async activity, async backend call.

**Acceptance criteria:**
- [ ] Factory output passes Temporal activity-definition introspection with name = node id
- [ ] Gate-validated dict lands in `AgentActivityResult.output`; raw response text is absent from the envelope
- [ ] Agent path escaping the base dir refuses via `resolve_agent_path` (existing `ConfigError`)

**Verification:**
- [ ] `pytest tests/unit/temporal/test_activity_factory.py -n auto` (mocked backend)
- [ ] Quality gates

**Dependencies:** T2
**Files likely touched:** `src/holodeck/temporal/activity.py`, `tests/unit/temporal/test_activity_factory.py`
**Estimated scope:** M

### Task 4: Error taxonomy and retry classification

**Description:** Map failures to Temporal semantics inside the activity.
Gate validation failure raises `GateValidationError` (plain exception →
retryable `ApplicationError` typed by class name). Authoring faults
(`ConfigError`, unloadable gate schema, missing agent.yaml) re-raise as
`ApplicationError(non_retryable=True)`. Backend transport errors and
`ExecutionResult.is_error` stay retryable. Document the class-name contract
(`RetryPolicy(non_retryable_error_types=[...])` matches by string).

**Acceptance criteria:**
- [ ] Gate failure → retryable error carrying the validation detail
- [ ] Authoring faults → `non_retryable=True`
- [ ] Model-fault vs authoring-fault channels never mix (SC-003)

**Verification:**
- [ ] `pytest tests/unit/temporal/test_retry_classification.py -n auto`
- [ ] Quality gates

**Dependencies:** T3
**Files likely touched:** `src/holodeck/temporal/activity.py` (or `errors.py`), `tests/unit/temporal/test_retry_classification.py`
**Estimated scope:** S

### Task 5: D3 helper surface + sandbox-safety unit test

**Description:** `src/holodeck/temporal/deterministic.py` re-exports the
workflow-safe surface: `evaluate` (table_eval), a standalone
`check_gate(obj, schema)` function, `ActivityParameters` (the workflow-side
scheduling helper from T2), and `load_decision_table` documented as
import-time-only. Unit test builds a tiny workflow that imports the D3
helpers (not passed through) and calls
`SandboxedWorkflowRunner().prepare_workflow(...)`; a restricted call must
fail the same harness (positive control). Extends
`tests/unit/workflow/test_import_purity.py` coverage to the new module.

**Acceptance criteria:**
- [ ] D3 modules (`table_eval`, `feel`, `decision_table`, gate-check half of `edge`) pass sandbox validation
- [ ] Harness proves it can fail (workflow with import-time nondeterminism is rejected)
- [ ] `holodeck.temporal.deterministic` imports without the backend stack

**Verification:**
- [ ] `pytest tests/unit/temporal/test_sandbox_safety.py tests/unit/workflow/test_import_purity.py -n auto`
- [ ] Quality gates

**Dependencies:** T1 (independent of T3)
**Files likely touched:** `src/holodeck/temporal/deterministic.py`, `tests/unit/temporal/test_sandbox_safety.py`, `tests/unit/workflow/test_import_purity.py`
**Estimated scope:** M

### Task 6: `HoloDeckPlugin`

**Description:** `src/holodeck/temporal/plugin.py`: a `SimplePlugin` that sets
`pydantic_data_converter` and registers the activities built from a list of
`EdgeNode`s (calls the T3 factory; no parallel implementation). Client
plugins auto-propagate to workers.

**Acceptance criteria:**
- [ ] Plugin sets the data converter and registers one activity per node
- [ ] Plugin and manual factory wiring produce identical activity definitions

**Verification:**
- [ ] `pytest tests/unit/temporal/test_plugin.py -n auto`
- [ ] Quality gates

**Dependencies:** T3
**Files likely touched:** `src/holodeck/temporal/plugin.py`, `tests/unit/temporal/test_plugin.py`
**Estimated scope:** S

### Task 7: `WorkerConfig` model and loader

**Description:** `src/holodeck/temporal/worker_config.py`: Pydantic model for
`worker.yaml` — `temporal:` (address, namespace, task_queue, TLS) and
`nodes:` (inline EdgeNode entries). No execution timeouts or retry policy in
this file — those are caller-side (decision 10); worker.yaml is registration
only. Env-var overrides
(`TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, …) follow the existing
ConfigLoader precedence. Node paths resolve relative to the worker.yaml
directory through `resolve_agent_path`.

**Acceptance criteria:**
- [ ] Valid worker.yaml parses; unknown keys refuse (`extra="forbid"`)
- [ ] Env vars override file values
- [ ] A node whose agent path escapes the config directory is refused at load

**Verification:**
- [ ] `pytest tests/unit/temporal/test_worker_config.py -n auto`
- [ ] Quality gates

**Dependencies:** T2
**Files likely touched:** `src/holodeck/temporal/worker_config.py`, `tests/unit/temporal/test_worker_config.py`
**Estimated scope:** M

### Task 8: `holodeck worker` command

**Description:** `src/holodeck/cli/commands/worker.py`: Click command with
`--config` (default `worker.yaml`) and `--task-queue` override. Loads
`WorkerConfig`, builds activities via the factory, connects with
`pydantic_data_converter` + `TracingInterceptor`, starts an activities-only
`Worker`, handles graceful shutdown (`graceful_shutdown_timeout`, SIGINT).
Import of temporalio stays inside the command so `holodeck --help` works
without the extra.

**Acceptance criteria:**
- [ ] `holodeck worker --help` works with and without the extra installed
- [ ] Startup registers one activity per configured node (mocked client test)
- [ ] Missing extra yields the T1 guard message, not a traceback

**Verification:**
- [ ] `pytest tests/unit/temporal/test_worker_command.py -n auto`
- [ ] Quality gates

**Dependencies:** T6, T7
**Files likely touched:** `src/holodeck/cli/commands/worker.py`, `src/holodeck/cli/main.py` (register verb), `tests/unit/temporal/test_worker_command.py`
**Estimated scope:** M

### Task 9: Hardship fixtures

**Description:** Committed fixtures under
`tests/integration/temporal/fixtures/hardship/`: two agent.yaml files
(evidence extractor, letter writer) with response_format, two gate schemas,
one `.dmn.yaml` decision table (reusing the 036 test-suite table so AC-3 can
compare Verdicts), a `worker.yaml`, and the sample workflow
(`workflow.py`: extract → table verdict → letter). The workflow schedules
each activity with `ActivityParameters.to_activity_kwargs()` — the sample is
the proof that timeout/retry configuration is caller-side and functional
(feeds AC-2).

**Acceptance criteria:**
- [ ] Fixture agents load through the existing `Agent` model
- [ ] Table is the same one the 036 test suite evaluates (AC-3 comparability)
- [ ] Workflow module passes the T5 sandbox harness

**Verification:**
- [ ] `pytest tests/unit/temporal/test_fixtures_load.py -n auto`
- [ ] Quality gates

**Dependencies:** T5 (sandbox harness), T7 (worker.yaml shape)
**Files likely touched:** `tests/integration/temporal/fixtures/hardship/*`
**Estimated scope:** S

### Task 10: Integration tests AC-1/AC-2/AC-3

**Description:** Against `temporal server start-dev` with a mocked backend
(deterministic canned `structured_output`). AC-1: workflow receives a
gate-validated envelope. AC-2: backend rigged to emit a gate-failing object
first, passing object second — activity fails once, retry succeeds, history
shows both attempts. AC-3: table helper in workflow code returns the same
`Verdict` as the 036 suite for the same inputs. Skip cleanly when the
`temporal` CLI binary is absent.

**Acceptance criteria:**
- [ ] AC-1, AC-2, AC-3 each demonstrated by a named test
- [ ] History assertion: no unvalidated payload in any completed activity result

**Verification:**
- [ ] `pytest tests/integration/temporal/ -n auto -m integration`
- [ ] Quality gates

**Dependencies:** T4, T9
**Files likely touched:** `tests/integration/temporal/test_activity_acceptance.py`, `tests/integration/temporal/conftest.py`
**Estimated scope:** M

### Task 11: Integration tests AC-4/AC-5 + sandbox backstop

**Description:** AC-4: spawn `holodeck worker` as a subprocess against
start-dev with the fixture worker.yaml; run the sample workflow end to end.
AC-5: replay the completed workflow history with the Temporal SDK replayer;
assert zero backend invocations (mock call counter). Sandbox backstop:
constructing the real Worker with the sample workflow validates it through
public API.

**Acceptance criteria:**
- [ ] AC-4 and AC-5 each demonstrated by a named test
- [ ] Replay executes `table_eval` deterministically without any activity re-execution

**Verification:**
- [ ] `pytest tests/integration/temporal/ -n auto -m integration`
- [ ] Quality gates

**Dependencies:** T8, T10
**Files likely touched:** `tests/integration/temporal/test_worker_e2e.py`, `tests/integration/temporal/test_replay.py`
**Estimated scope:** M

### Task 12: OTel test AC-6

**Description:** With an in-memory span exporter: run the same agent once
through the `holodeck test` execution path and once through the activity;
assert the GenAI span set (names + GenAI semconv attributes) matches. With
`TracingInterceptor` active, assert GenAI spans are children of the Temporal
activity span.

**Acceptance criteria:**
- [ ] Span parity between `holodeck test` and activity execution (AC-6)
- [ ] GenAI spans nest under the interceptor's activity span

**Verification:**
- [ ] `pytest tests/integration/temporal/test_otel.py -n auto -m integration`
- [ ] Quality gates

**Dependencies:** T10
**Files likely touched:** `tests/integration/temporal/test_otel.py`
**Estimated scope:** M

### Task 13: Live smoke, `sample/` copy, docs, index row

**Description:** One `@pytest.mark.slow` test running a real Claude call
through the sample workflow (manual execution). Thin runnable demo under
`sample/temporal-hardship/` (git-ignored). Docs page for the temporal
integration (activity factory, plugin, worker.yaml, D3 pattern with the
import-time table-load rationale). Update `specs/index.md` row 040 and the
spec status line.

**Acceptance criteria:**
- [ ] Smoke test passes manually with credentials
- [ ] Docs cover both personas (Python developer; worker.yaml host)
- [ ] `specs/index.md` row reflects final task count and status

**Verification:**
- [ ] `pytest -m slow tests/integration/temporal/ -n auto` (manual)
- [ ] `make ci`

**Dependencies:** T11, T12
**Files likely touched:** `tests/integration/temporal/test_smoke_live.py`, `docs/…`, `specs/index.md`, `specs/040-holodeck-temporal/spec.md`
**Estimated scope:** S

### Task 14: Gate-schema codegen (`holodeck generate models`)

**Description:** *(added 2026-08-30, user-requested scope addition)* CLI verb
that generates typed Pydantic models from gate JSON Schemas, so workflow code
gets static typing over `AgentActivityResult.output` instead of a bare dict.
Reads edge-node declarations (a `worker.yaml` `nodes:` list or explicit
`agent.yaml` paths), resolves each gate schema through the existing
`load_gate_schema` seam, and emits a generated module (e.g. `models_gen.py`)
via `datamodel-code-generator`. Pairs with `AgentActivityResult.output_as()`
(added in T2 follow-up): `result.output_as(EvidenceOutput)`. Generated code
must be pure Pydantic — sandbox-safe, importable from workflow code. The wire
envelope is untouched: `output` stays a plain dict in history (FR-008);
codegen is developer-ergonomics only.

**Acceptance criteria:**
- [ ] `holodeck generate models --config worker.yaml` emits a module with one model per edge node gate schema, deterministic output (stable ordering, no timestamps)
- [ ] Generated module imports cleanly inside the workflow sandbox (reuse the T5 harness)
- [ ] `result.output_as(GeneratedModel)` round-trips the hardship fixtures
- [ ] Staleness detectable: regenerating over an unchanged schema is a no-op diff (CI-checkable)

**Verification:**
- [ ] `pytest tests/unit/temporal/test_codegen.py -n auto`
- [ ] Quality gates as in T1

**Dependencies:** T5 (sandbox harness), T7 (`WorkerConfig`/worker.yaml loader), T9 (fixtures)
**Files likely touched:** `src/holodeck/cli/commands/generate.py`, `src/holodeck/temporal/codegen.py`, `tests/unit/temporal/test_codegen.py`, `pyproject.toml` (dev/extra dep `datamodel-code-generator`)
**Estimated scope:** M

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sandbox APIs (`prepare_workflow`, `workflow_sandbox`) marked "not yet stable" | Med | Exact pin `temporalio==1.32.0`; Worker-init public-API backstop (T11); keep the unit harness tolerant of minor drift |
| `temporal` CLI binary unavailable in CI | Med | Integration tests skip cleanly when absent; add an install step to CI in T10; unit suite carries the contract regardless |
| Claude backend subprocess (Node.js) inside activity workers | Med | Reuse existing `validators.py` preflight at worker startup; fail fast with the existing error, not mid-activity |
| Envelope bloats event history | Low | Envelope carries only the validated dict + small usage ints; no raw text, no message history |
| Import-time table load conflicts with sandbox re-imports | Low | T5 harness proves the pattern before T9 depends on it; fallback documented pattern is passthrough of the table module |
| `pydantic_data_converter` interaction with sandbox | Low | README-recommended `pydantic` passthrough applied in worker/test setup |

## Open Questions

None. The spec §12 open questions were resolved: extra name = `holodeck[temporal]` (D12), worker config = `worker.yaml` + env + flags (D8–D10), and the `contrib.openai_agents` seam study concluded (granularity note above).
