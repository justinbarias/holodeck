# SPEC: HoloDeck Agents on Temporal

**Status:** Draft for review
**Date:** 2026-08-29
**Supersedes:** `specs/036-deterministic-spine/` (archived)
**Method note:** This spec replaces the speckit workflow. Future specs live in this format.

## 1. Objective

HoloDeck agents become first-class citizens on Temporal. A developer authors a
Temporal workflow in Python. HoloDeck supplies the agents that the workflow
calls. HoloDeck does not ship a workflow engine, a workflow YAML format, or a
DAG runner.

The target user is a developer who builds agent systems on Temporal. This user
wants durable, retryable agent calls without custom plumbing. The user defines
each agent in `agent.yaml` and keeps the no-code surface at the agent level.

### Why this pivot

The durable-execution market settled the reliability argument (Temporal, Dapr,
DurableTask). Model-led control flow settled the orchestration argument. The
036 overlay engine competed with that consensus. This spec joins it instead.
The full analysis is in `docs/ideas/agent-workflows-comparative-analysis-2026.md`.

Temporal's own rule enforces the one surviving 036 invariant. Workflow code
must be deterministic. LLM calls live in activities. Therefore the LLM is
never in workflow code. Typed gates hold the boundary.

## 2. Deliverables

### D1 — `holodeck.temporal` activity library

A Python package that turns an agent definition into a Temporal activity.

- Input: a path to `agent.yaml`, or a loaded `Agent` model.
- Output: an activity function that a workflow can call.
- The activity invokes the agent through `BackendSelector`. No new backend code.
- The activity resolves an `EdgeNode` agent reference through
  `edge.resolve_agent_path`, never by a plain path join. That function is the
  path-confinement control, and it has no other caller that keeps it honest.
- Retry and timeout values come from configuration, with Temporal defaults as fallback.
- The activity emits the existing OTel GenAI spans (specs 018 and 022).

### D2 — Gate validation inside the activity

- The activity validates `structured_output` against the node's JSON Schema gate before it returns.
- If validation fails, the activity attempt fails. Temporal's retry policy then runs the agent again.
- As a result, Temporal event history contains validated objects only.
- The gate loader from 036 (`holodeck.lib.workflow.edge`) is reused, not rewritten.

### D3 — Deterministic helpers for workflow code

Pure functions that are safe inside Temporal workflow code:

- **Decision-table evaluation.** The 036 DMN/FEEL evaluator
  (`holodeck.lib.workflow.table_eval`, `holodeck.models.decision_table`)
  becomes a callable step. Same tables, same hit policies, same `Verdict`.
- **Gate check.** Schema validation as a standalone function, for authors who
  validate in workflow code as well.

### D4 — `holodeck worker` CLI

A command that hosts agents as a Temporal worker.

- Input: one or more `agent.yaml` paths and Temporal connection configuration.
- The command registers one activity per agent and starts the worker.
- A developer who writes no Python can host the agents. The workflow side
  stays Python.

### Records (no new deliverable)

Temporal event history records every validated activity result. The existing
OTel spans record the GenAI trail. This spec adds no record artifact. A
portable, engine-independent record is deferred until a concrete buyer asks
for one.

## 3. Acceptance criteria

- **AC-1:** A sample Temporal workflow calls a HoloDeck agent as an activity and receives a gate-validated object.
- **AC-2:** When the agent returns an object that fails the gate, the activity attempt fails. Temporal retries it under the configured policy.
- **AC-3:** A workflow calls the decision-table helper in workflow code and gets the same `Verdict` as the 036 test suite.
- **AC-4:** `holodeck worker` starts against `temporal server start-dev`, registers the sample agents, and serves the sample workflow end to end.
- **AC-5:** Replay of a completed workflow by the Temporal SDK does not invoke an LLM.
- **AC-6:** Activity execution emits the same OTel GenAI spans as `holodeck test` execution of the same agent.

## 4. Commands

Development commands are unchanged. See `CLAUDE.md` for the full list.

```bash
make test                     # All tests (parallel)
make ci                       # Full CI locally
temporal server start-dev     # Local Temporal, single binary
holodeck worker --help        # New verb (this spec)
```

## 5. Project structure

New code lands here:

```
src/holodeck/temporal/        # D1-D3: activity factory, gate seam, helpers
src/holodeck/cli/commands/worker.py   # D4
tests/unit/temporal/          # Unit tests, mocked Temporal + mocked backend
tests/integration/temporal/   # Against temporal server start-dev
```

Reused, not moved: `holodeck.lib.workflow.table_eval`, `holodeck.lib.workflow.edge`,
`holodeck.models.decision_table`, `holodeck.lib.backends.*`.

Removed with the pivot (not shipped): the `holodeck workflow` CLI verb, the
DAG runner (`holodeck.lib.workflow.runner`), the `input_data` validator, the
DAG models (`Workflow`, `PolicyNode`, `HumanNode`), and
`schemas/workflow.schema.json`. Only the `EdgeNode` family remains in
`holodeck.models.workflow`, because the gate executor consumes it.

## 6. Tech stack and constraints

- `temporalio` (Temporal Python SDK) as a new dependency. Pin the version at
  implementation time, after a source check of the current release.
- Study `temporalio.contrib.openai_agents` before design. It is the precedent
  for "agent SDK as first-class Temporal citizen". Copy its seams where they
  fit. Record where they do not.
- Temporal is a hard dependency for this feature. There is no engine
  abstraction layer. A second engine gets a seam only when it is real.
- Python 3.10+ and the existing HoloDeck standards apply (`CLAUDE.md`).

## 7. Code style

Follow `CLAUDE.md`: Google style, Black, Ruff, MyPy strict, Pydantic v2 for
all configuration, errors from `holodeck.lib.errors`, async I/O only.

One addition. Workflow-safe helpers (D3) must not import I/O modules, and a
test must prove that they pass Temporal's workflow sandbox validation.

## 8. Testing strategy

- Unit tests mock the Temporal SDK and the agent backend. They cover the
  activity factory, the gate seam, and retry classification.
- Integration tests run against `temporal server start-dev` and a mocked
  backend. They cover AC-1 through AC-5.
- One live smoke test (marked `@pytest.mark.slow`) runs a real Claude call
  through the sample workflow, for manual execution.
- The 036 table-eval and gate test suites keep running unchanged. They defend
  the reused primitives.

## 9. Boundaries

**Always:**

- Keep the LLM out of workflow code. Agents run in activities only.
- Validate agent output at the gate before the activity returns.
- Route agent calls through `BackendSelector`.
- Run `make format`, `make lint`, `make type-check`, and `make security` after each task.

**Ask first:**

- Any new HoloDeck-owned workflow definition format. This spec forbids one.
  A future exception needs an explicit decision.
- A second orchestration engine, or an abstraction layer for one.
- A portable run-record artifact.

**Never:**

- Do not put a model call, or any nondeterministic call, in workflow code.
- Do not let an unvalidated agent payload into Temporal event history.
- Do not rewrite the 036 table evaluator or gate loader. Reuse them.

## 10. Out of scope

- A HoloDeck workflow YAML, DSL, or visual builder.
- Human-task UI. Human approval uses Temporal signals, authored by the user.
- Engine-independent replay or record artifacts.
- Migration tooling from `holodeck workflow run` to Temporal.

## 11. Decision record (from the 2026-08-29 grilling session)

| # | Decision |
|---|---|
| 1 | Thesis moved from a decisions product to orchestration. |
| 2 | HoloDeck agents are components inside user-authored Temporal workflows. HoloDeck is not the orchestrator. |
| 3 | The primitives are: durable agent step, schema gate, record. Not a workflow format. |
| 4 | "The LLM is never the spine" survives at boundaries only. Temporal's determinism rule enforces it. |
| 5 | The gate lives inside the activity. A gate failure is a retryable activity fault. |
| 6 | Records = Temporal event history + existing OTel spans. Nothing new in v1. |
| 7 | DMN/FEEL table evaluation survives as a deterministic helper in workflow code. |
| 8 | 036 is archived. The kept primitives (gate, table-eval, FEEL) merge and are reused. The overlay engine (runner, DAG models, workflow schema, CLI verb) is deleted before merge. T7+ is not built. |
| 9 | The stated durability pressure (retry and timeout of agent calls) lives in agent execution, not in the 036 spine. |

## 12. Open questions

- Package name for the dependency extra: `holodeck[temporal]` is the working assumption.
- Worker configuration shape: flags, environment variables, or a small `worker.yaml`. Decide during D4 design.
- Which `temporalio.contrib.openai_agents` seams transfer, and which do not. Answer with a source check at design time.

## 13. Glossary of retained 036 identifiers

The kept modules cite requirement tags from `specs/036-deterministic-spine/`,
which is archived. This table gives each tag a live one-line meaning.

| Tag | Meaning |
| --- | --- |
| FR-006 | The edge agent is invoked through `BackendSelector`, never a concrete backend. |
| FR-007 | The agent's `structured_output` is validated against the node's JSON Schema gate. |
| FR-008 | The gate-validated object is the canonical value for every downstream consumer. The raw model text never crosses. |
| FR-010 | FEEL is restricted to a fixed subset. Constructs outside it are rejected statically at load. |
| FR-012 | Failures are loud. No silent fallback, no default that hides an error. |
| FR-030 | A generated table had a human review gate. Removed with the overlay engine; `Provenance.awaiting_review` is its remnant. |
| FR-032 | Provenance metadata is not executable. FEEL cannot reference it. |
| SC-003 | A gate rejection counts as evidence about model output. The error channels keep model faults separate from authoring faults. |
| T1, T3, T5, T10 | 036 task numbers: FEEL conformance suite (T1), determination engine (T3), edge executor (T5), run-record snapshot (T10). |
| refinements §1 | The POC validated the Claude backend only. Dispatch still goes through `BackendSelector`. |
| research.md caveats 1–6 | Verified quirks of the embedded FEEL evaluator (date handling, numeric strictness, silent `None` reads, native exceptions). The archived file holds the details; the docstrings in `feel.py` restate what the code corrects. |
