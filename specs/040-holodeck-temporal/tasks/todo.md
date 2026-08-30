# TODO: HoloDeck Agents on Temporal (spec 040)

Task details, acceptance criteria, and decision record live in `plan.md`.
Run `make format`, `make lint`, `make type-check`, `make security` after each task.

## Phase 1 — Foundation (D1 core)

- [x] **T1** `holodeck[temporal]` extra, exact pin `temporalio==1.32.0`, package skeleton with import guard (S)
- [x] **T2** Payload models: `AgentActivityInput`, `AgentActivityResult`; `ActivityParameters` as workflow-side scheduling helper (`to_activity_kwargs()`, no heartbeat); `output_as()` typed accessor (S)
- [x] **T3** Activity factory: `EdgeNode` → named async activity; gate mandatory; `BackendSelector.invoke_once`; envelope return (M)
- [x] **T4** Error taxonomy: gate failure retryable, authoring faults `non_retryable=True`, transport retryable (S)
- [x] **T15** Live phase-1 e2e: real Temporal workflow + agent activity against live Claude (`start_local` dev server, oauth, `SKIP_LLM_INTEGRATION_TESTS=false`) (S)

### Checkpoint 1
- [x] Factory output is a valid activity definition (mocked backend); unit suite + quality gates clean

## Phase 2 — Deterministic helpers + plugin (D3)

- [x] **T5** `holodeck.temporal.deterministic` surface + sandbox-safety unit test (`prepare_workflow`, with positive control) (M)
- [x] **T6** `HoloDeckPlugin` (`SimplePlugin`): pydantic data converter + activity registration over the T3 factory (S)

### Checkpoint 2
- [x] D3 modules pass sandbox validation; plugin and manual wiring produce identical definitions

## Phase 3 — Worker CLI (D4)

- [x] **T7** `WorkerConfig` model + loader: worker.yaml (temporal connection + inline nodes, registration only — no timeouts), env overrides, path confinement (M)
- [ ] **T8** `holodeck worker` command: activities-only Worker, `TracingInterceptor`, graceful shutdown, guarded import (M)
- [ ] **T16** Live worker-command integration test: real `holodeck worker` subprocess against `WorkflowEnvironment.start_local` (real dev server, no mocks), workflow executes end to end, SIGINT shuts down gracefully (M)

### Checkpoint 3
- [ ] `holodeck worker --help` works without the extra; worker starts from fixture config (mocked client); live worker e2e green

## Phase 4 — Integration, sample, acceptance criteria

- [ ] **T9** Hardship fixtures: 2 agents, 2 gates, 036 table, worker.yaml, sample workflow (S)
- [ ] **T10** Integration AC-1/AC-2/AC-3 against `temporal server start-dev`, mocked backend (M)
- [ ] **T11** Integration AC-4 (worker subprocess e2e) + AC-5 (replay, zero LLM calls) + Worker-init sandbox backstop (M)
- [ ] **T12** OTel AC-6: span parity with `holodeck test`; GenAI spans nest under activity span (M)
- [ ] **T13** Live smoke `@slow`, `sample/temporal-hardship/` demo, docs, `specs/index.md` row (S)
- [ ] **T14** Gate-schema codegen: `holodeck generate models` — typed Pydantic models from gate schemas, pairs with `output_as()` (M)

### Checkpoint: Complete
- [ ] AC-1 … AC-6 demonstrated by named tests; `make ci` clean; spec status updated
