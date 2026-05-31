# Feature Specification: Deterministic Spine — Composable Workflow Determination Gates

**Feature Branch**: `036-deterministic-spine`
**Created**: 2026-05-30
**Status**: Draft for review
**Author**: justinbarias (with Claude)
**Input**: User description: "A new HoloDeck feature to model deterministic workflows and rules with BPMN & DMN. An engine that models deterministic workflows while using HoloDeck agents on the edges (leaf nodes). Ship 'just the gate' as a composable primitive; show how to compose multiple determination levels. FEEL required. `requires_human` = CLI prompt. Anchor sample: loan hardship underwriting."
**Related**: `docs/ideas/deterministic-spine.md` (idea one-pager), `specs/029-subagent-orchestration`, `specs/018-otel-observability`, `specs/022-otel-genai-semconv`, `specs/006-agent-test-execution`
**Concept ref**: *The Deterministic Spine* (FDE-01/A)

## Motivation

HoloDeck today has no way to **put rules in charge of a decision**. Its multi-agent
story (sequential / handoff / group-chat / magentic, all still aspirational per
VISION) is *emergent* by design — the opposite of what a regulated or enterprise
decision needs. When an AI step sits anywhere on the path to a consequential,
contestable decision, the LLM becomes the de-facto spine: non-reproducible,
unexplainable, and impossible to audit. That is the failure mode administrative
law forbids — and the one Robodebt embodied.

This feature introduces the **deterministic spine**: a workflow modeled as a
directed acyclic graph (DAG) of **determination nodes**. A determination node
takes typed inputs and runs them through **policy-as-code (a DMN decision table
with FEEL expressions)**, emitting a typed, audited verdict. HoloDeck agents are
confined to the **leaf (edge) nodes**; their output crosses a **schema gate** —
the typed boundary where validated objects (and nothing else) enter the spine.
Every node above the edges is deterministic; the top node is the **bright line**
where a named human decides and the AI may draft reasons only.

The target user is the **enterprise + regulated/government FDE** whose killer
requirement is **policy-as-code**: decision logic that lives in versioned,
testable, publishable DMN tables the AI cannot reach around. This turns
HoloDeck's existing evaluation framework into a *policy-testing* tool — a
capability LangSmith, MLflow, and PromptFlow do not offer.

This is a deliberately **thin proof-of-concept**, not a workflow engine. It does
the smallest thing that proves the spine: *agent → typed object → versioned DMN
table → composed, human-accountable verdict → replayable record.*

Scope of the claim: the POC proves **reproducibility, replayability, and
policy-as-code**. A durable, regulated-grade audit record (signed, tamper-evident,
queryable) is **North Star, not POC** — the POC's audit surface is OpenTelemetry
spans plus the local run record, which are sufficient to *replay* a determination
but are not themselves a legal-grade audit store.

## Architectural shape and what it adds

The spine is a new, self-contained subsystem. It **reuses** the backend
abstraction (`BackendSelector` / `AgentBackend` / `ExecutionResult`) to run edge
agents, and OpenTelemetry instrumentation (specs 018, 022) to emit the audit
trace. It introduces **no** orchestration engine, no BPMN/DMN XML parsing, and
no events/timers/message-flows.

| Concern | How the spine handles it |
|---|---|
| Authoring | Native HoloDeck YAML DSL — `workflow.yaml` (the graph) + `tables/*.dmn.yaml` (decision tables) + `schemas/*.json` (gates). No BPMN/DMN XML in v1. |
| Composition | A DAG: a node's verdict is a **named input** to nodes above it (DMN's Decision Requirement Diagram). Strictly one-directional in v1. |
| Edge (AI) node | Dispatches a HoloDeck agent via `BackendSelector`; its `ExecutionResult.structured_output` is validated at a per-node `gate:` block before it may cross. |
| Policy node | Pure DMN table evaluation over named inputs, using FEEL expressions and a hit policy (`UNIQUE` / `FIRST` / `PRIORITY`). No AI. |
| Human node | `requires_human: true` pauses for a **CLI prompt**; `ai_may_draft: reasons` lets AI draft proposed reasons only — no AI output crosses into the verdict. |
| FEEL | An **embedded** evaluator (library TBD — see Open Questions). FEEL is not built from scratch. |
| Audit trace | **OpenTelemetry GenAI spans** per node for now (specs 018/022). A durable DB-backed determination record is North Star / out of scope. |
| Replay | A minimal **local run record** persists each edge node's *gated output* **plus a content snapshot of every decision table and gate schema used**. `holodeck workflow replay` re-evaluates the deterministic policy + human layers from the recorded edge outputs **against the snapshotted tables** — **it does not re-invoke the LLM** (non-deterministic) and **does not depend on a policy registry** (out of scope). |
| CLI | New `holodeck workflow` verb: `holodeck workflow run workflow.yaml`, `holodeck workflow replay <record>`. |

### Terminology (one word, one meaning)

- **Determination node** — a node that emits a *verdict* by evaluating a DMN table: a **policy node** (pure) or a **human node** (the bright line).
- **Edge node** — a leaf node that runs a HoloDeck agent; it emits a gate-validated *input object*, never a verdict.
- **Schema gate** (`gate:`) — the typed boundary on an edge node where the agent's `structured_output` is validated before it may cross into the spine. **"Gate" always means this boundary** — never the DMN node. (The feature title's "Determination Gates" is shorthand for "determination nodes fed across schema gates.")
- **Verdict** — a determination node's typed output; becomes a named input to dependent nodes.

### Why "workflow" not "determination"

The user-facing artifact and command use **`workflow`** (`workflow.yaml`,
`holodeck workflow run`) to match the BPMN mental model the feature derives from.
Internally the nodes remain *gates* and the act remains a *determination* — but
the operator types `holodeck workflow`.

### The one rule (non-negotiable invariant)

**The LLM is never the spine.** It feeds the spine typed, validated objects. The
spine — not the model — holds state, applies the policy-as-code, and records the
trace. Any path that lets the model produce a verdict (rather than an input to
one) is off-pattern and MUST be rejected at validation time.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Run a single determination node (Priority: P1)

An FDE defines one edge node (an agent that extracts fields), a per-node schema
gate, and one DMN decision table. They run `holodeck workflow run workflow.yaml`.
HoloDeck invokes the agent, validates its structured output against the gate
schema, evaluates the DMN table over the validated object, and emits a typed
verdict plus an OTel span.

**Why this priority**: This is the atomic primitive — "AI on a leash, rules in
charge." Without it, nothing else exists. A single gate is already a viable MVP:
it proves a probabilistic agent can feed a deterministic policy through a typed
boundary.

**Independent Test**: A `workflow.yaml` with one edge node + one policy node, a
gate schema, and a `UNIQUE`-hit-policy table. Run it; assert the verdict matches
the table's expected output for the agent's (mocked) extraction, and that
free-text or schema-invalid agent output is rejected at the gate.

**Acceptance Scenarios**:

1. **Given** an edge node whose agent returns a schema-valid object, **When** the workflow runs, **Then** the object crosses the gate and the policy node emits the verdict dictated by the matching DMN rule.
2. **Given** an edge node whose agent returns free text or a schema-invalid object, **When** the workflow runs, **Then** the gate rejects it and the workflow fails with a typed validation error — no verdict is produced.
3. **Given** a workflow run completes, **When** the trace is inspected, **Then** an OTel span exists for the edge node and the policy node, capturing inputs, the matched rule, and the verdict.

---

### User Story 2 — Compose multiple determination levels (Priority: P1)

An FDE composes several edge nodes and policy nodes into a multi-level DAG: Level
1 edge nodes produce typed inputs; Level 2 policy nodes compose those inputs;
a Level 3 node composes the Level 2 verdicts. Each node names its `inputs` by the
`id` of nodes below it. HoloDeck builds the DAG, topologically orders it, and
evaluates bottom-up.

**Why this priority**: This is the headline capability — *multiple determination
levels composed without an orchestration engine*. It is what distinguishes the
spine from "an agent that calls a rules function."

**Independent Test**: A `workflow.yaml` with three Level-1 edges, two Level-2
tables, and one Level-3 table; assert evaluation order is a valid topological
sort, that each higher node receives the lower nodes' verdicts as named inputs,
and that the final verdict reflects the composed sub-verdicts.

**Acceptance Scenarios**:

1. **Given** a node declares `inputs: [a, b]`, **When** the workflow runs, **Then** nodes `a` and `b` are evaluated first and their verdicts are passed as named inputs.
2. **Given** a workflow whose `inputs` form a cycle, **When** it is loaded, **Then** validation fails with a clear cycle error before any agent is invoked.
3. **Given** a node references an `inputs` id that does not exist, **When** it is loaded, **Then** validation fails with an unresolved-reference error.

---

### User Story 3 — Human-accountable determination (Priority: P1)

The top node sets `requires_human: true` and `ai_may_draft: reasons`. When the
workflow reaches it, HoloDeck presents the composed inputs and the AI-drafted
reasons at the **CLI**, then prompts the named human to select the determination
(e.g. approve / refer / decline). The human's choice — not any AI output —
becomes the verdict and is recorded.

**Why this priority**: The bright line — a named human deciding while the AI
drafts only — is the headline promise for regulated users. The MVP is incomplete
without it: a spine with no human determination is a spine with its head cut off.
It depends on US1/US2 but is core, not additive, so it is P1.

**Independent Test**: A workflow whose final node `requires_human`; run it with a
scripted CLI answer; assert the recorded verdict equals the human's choice, that
the AI-drafted reasons are present but advisory, and that no AI-produced field
can populate the verdict.

**Acceptance Scenarios**:

1. **Given** a `requires_human` node, **When** the workflow reaches it, **Then** it pauses and presents composed inputs + AI-drafted reasons and waits for a human selection at the CLI.
2. **Given** `ai_may_draft: reasons`, **When** the node executes, **Then** the AI may produce a `reasons` draft only; attempting to let AI output set the verdict is rejected at validation time.
3. **Given** the human selects a determination, **When** the workflow completes, **Then** the run record attributes the verdict to the named human, with a timestamp.

---

### User Story 4 — Reproducible replay (Priority: P2)

An FDE re-runs a completed determination with `holodeck workflow replay <record>`.
HoloDeck reads the recorded gated edge outputs and the recorded human decision,
re-evaluates the deterministic policy and human layers, and produces a verdict
**identical** to the original — without calling any LLM.

**Why this priority**: Replay is the "prove it" property regulated buyers need.
It depends on US1/US2 existing, so P2.

**Independent Test**: Run a workflow to completion; capture the record; run
`replay` on that record; assert every policy node's matched rule and the final
verdict are byte-identical to the original run, and assert no agent/LLM call is
made during replay.

**Acceptance Scenarios**:

1. **Given** a completed run record, **When** `replay` is invoked, **Then** the deterministic layers re-evaluate from recorded edge outputs and produce an identical verdict.
2. **Given** a replay, **When** it executes, **Then** no edge agent is invoked (zero LLM calls) — only the recorded gated outputs are consumed.
3. **Given** a record whose referenced decision table has changed version, **When** `replay` runs, **Then** it uses the table version recorded at run time (or fails loudly if that version is unavailable) — it never silently replays against a newer policy.

---

### User Story 5 — Test decision tables as policy-as-code (Priority: P3)

An FDE writes test cases that assert a DMN table's verdict for given inputs,
using HoloDeck's existing test/eval framework — testing the *policy*, not an
agent. They run them in CI to guard the rules against regression.

**Why this priority**: High strategic value (turns the evaluator into a
policy-testing tool), but the spine is usable without it, so P3.

**Independent Test**: A test file asserting expected verdicts for several input
rows against a `tables/*.dmn.yaml`; run via the test runner; assert pass/fail
reflects table correctness independent of any agent.

**Acceptance Scenarios**:

1. **Given** a decision-table test case with inputs and an expected verdict, **When** the test runs, **Then** it passes iff the table produces that verdict.
2. **Given** a table edited so a rule changes, **When** the policy tests run, **Then** the affected test fails with a diff of expected vs. actual verdict.

---

### User Story 6 — Loan hardship underwriting sample (Priority: P3)

A complete, runnable sample lives under `sample/` demonstrating three Level-1
edge agents (income/expense extraction, residency verification, document-fraud
flag), two Level-2 tables (affordability `UNIQUE`, risk tier `FIRST`), and one
Level-3 human determination (`PRIORITY`: auto-approve / refer-to-officer /
decline). It runs one applicant end-to-end and then replays the record.

**Why this priority**: Proof and documentation, not core engine; P3.

**Independent Test**: `holodeck workflow run sample/loan-hardship/workflow.yaml`
completes end-to-end with a scripted human answer and prints a composed
determination; `holodeck workflow replay` reproduces it identically.

**Acceptance Scenarios**:

1. **Given** the sample, **When** run end-to-end, **Then** it produces a final determination composed from both Level-2 verdicts and a human decision.
2. **Given** the sample's run record, **When** replayed, **Then** the determination is reproduced identically with no LLM calls.

---

### Edge Cases

- **DMN no match**: a `UNIQUE`/`FIRST`/`PRIORITY` table where no rule matches the inputs → the node MUST fail loudly (or emit a declared default output if the table defines one), never silently emit null.
- **DMN multiple matches under `UNIQUE`**: more than one rule matches → validation/eval error (the `UNIQUE` contract is violated by the table or the data).
- **FEEL evaluation error**: a malformed FEEL expression or a type mismatch (string vs. number) → fail at table load (static where possible) or at eval with a precise locator (table id, rule index, cell).
- **Cyclic or unresolved `inputs`**: detected at load, before any agent invocation.
- **Low-confidence edge output**: if the gate schema marks fields with confidence and a field is below threshold, the gate rejects (or routes to the human node, if the workflow declares that) — it is never silently accepted.
- **Human aborts the CLI prompt**: the run terminates with an explicit "no determination" status; no partial verdict is recorded as final.
- **On-disk table edited after a run**: replay is unaffected — it uses the snapshot embedded in the record, not the current file. A record with missing or corrupt snapshots fails loudly rather than falling back to current tables.
- **Non-deterministic agent on a fresh run** (not replay): expected and allowed — the *edge* is probabilistic; determinism is a property of the *spine* and of *replay*, not of a fresh run's edges.

## Requirements *(mandatory)*

### Functional Requirements

**Authoring & validation**
- **FR-001**: System MUST parse a native YAML workflow artifact (`workflow.yaml`) declaring an ordered list of nodes, each with a unique `id`.
- **FR-002**: System MUST support three node kinds: **edge** (`edge.agent` + `gate`), **policy** (`decision` + `inputs` + `hit_policy`), and **human** (`decision` + `inputs` + `requires_human` + optional `draft` agent + optional `ai_may_draft`).
- **FR-003**: System MUST construct a DAG from each node's `inputs` (referencing other node ids) and MUST reject cycles and unresolved references at load time, before invoking any agent.
- **FR-004**: System MUST validate the workflow against a published JSON schema (consistent with `schemas/agent.schema.json` conventions) with Pydantic v2 models.
- **FR-005**: The workflow schema MUST make "AI output as a verdict" *unrepresentable*: only policy and human nodes emit verdicts (via a DMN table), and edge nodes emit only gate-validated inputs. There is no node kind through which AI output can become a verdict, so the "LLM is never the spine" invariant holds *by construction of the schema* rather than by a runtime check.

**Edge nodes & schema gates**
- **FR-006**: System MUST invoke an edge node's agent via `BackendSelector`, provider-agnostically, reusing `ExecutionResult`.
- **FR-007**: System MUST validate the edge agent's `structured_output` against the node's per-node `gate.schema` (JSON Schema / Pydantic) and MUST reject free text or schema-invalid output at the gate.
- **FR-008**: System MUST treat the gate-validated typed object — not the raw LLM output — as the canonical value that crosses into the spine, and MUST record it for replay.

**Policy nodes (DMN + FEEL)**
- **FR-009**: System MUST evaluate DMN decision tables (`tables/*.dmn.yaml`) over named inputs resolved from the verdicts/objects of input nodes.
- **FR-010**: System MUST evaluate FEEL expressions in rule cells via an **embedded** FEEL evaluator (not built in-house), restricted to a **bounded subset**: numeric literals and comparisons (`<`, `<=`, `>`, `>=`, `=`, `!=`), ranges (`[a..b]`, `(a..b]`, etc.), boolean `and`/`or`/`not`, string equality, list membership (`in`), and date literals with comparison and difference. Expressions outside this subset MUST be rejected at table-load time (statically where possible), never silently mis-evaluated. [NEEDS CLARIFICATION: which FEEL evaluator/library covers exactly this subset — resolve in `/speckit.plan` via the FEEL spike.]
- **FR-011**: System MUST support the hit policies **UNIQUE**, **FIRST**, and **PRIORITY** — the three the anchor sample uses — with standard DMN semantics for each. (`COLLECT` and other DMN hit policies are deferred; see Out of Scope.)
- **FR-012**: System MUST fail loudly on no-match (absent a declared default output) and on a `UNIQUE` table with multiple matches.
- **FR-013**: Each decision table MUST carry a **version** identifier (a label). The run record MUST embed a **content snapshot** of each table used (not merely the version id), so replay does not depend on an external registry or the current on-disk file.

**Human nodes**
- **FR-014**: System MUST pause at a `requires_human` node and present, at the CLI, the composed inputs and — if the node declares a `draft` agent with `ai_may_draft: reasons` — the AI-drafted reasons, then prompt for a human selection among the table's declared outputs.
- **FR-015**: System MUST record the human's selection as the verdict, attributed to the named human with a timestamp; AI output MUST NOT populate the verdict.
- **FR-015a**: When a human node declares a `draft` block, the named drafting agent MUST be invoked **only** to produce the advisory fields listed in `ai_may_draft` (e.g. `reasons`); its output MUST be shown to the human as advisory and MUST NOT populate the verdict or any decision input. A human node without a `draft` block presents inputs only (no AI drafting). The drafting agent's output is recorded as advisory context, not as part of the deterministic spine.

**Composition & execution**
- **FR-016**: System MUST evaluate nodes in a valid topological order, passing each node's verdict to dependent nodes as a named input.
- **FR-017**: The DAG MUST be strictly one-directional in v1 (no callback from a policy node to re-trigger an edge).

**Audit, record & replay**
- **FR-018**: System MUST emit OpenTelemetry GenAI spans (per specs 018/022) for each node, capturing inputs, table version, matched rule(s), and verdict.
- **FR-019**: System MUST persist a **local run record** containing each edge node's gated output, each policy node's matched rule and verdict, a **content snapshot of every decision table and gate schema used** (with version labels), and the human decision — sufficient to replay deterministically with no external dependencies.
- **FR-020**: `holodeck workflow replay <record>` MUST re-evaluate the policy and human layers from the recorded gated outputs and recorded human decision, producing an identical verdict, and MUST NOT invoke any LLM/agent.
- **FR-021**: Replay MUST evaluate against the **table snapshots embedded in the run record** (not the current on-disk tables), and MUST fail loudly if the record is incomplete or its snapshots fail an integrity check — it MUST NOT silently fall back to current tables or a registry.

**CLI**
- **FR-022**: System MUST add a `holodeck workflow` command group with `run <workflow.yaml>` and `replay <record>` subcommands, using Click and `echo()` for output per repo conventions.

**Testing of policy**
- **FR-023**: System MUST allow decision tables to be tested via the existing HoloDeck test/eval framework — asserting a table's verdict for given inputs independent of any agent.

**Sample**
- **FR-024**: System MUST ship a runnable `loan-hardship` sample under `sample/` exercising three edge agents, two policy tables, and one human determination, plus a replay.

### Key Entities

- **Workflow**: the DAG artifact (`workflow.yaml`); an ordered set of Nodes plus metadata (name, version).
- **Node**: one of: **EdgeNode** (agent + schema gate; emits a gate-validated input, never a verdict), **PolicyNode** (decision table + inputs + hit policy), **HumanNode** (decision table + inputs + `requires_human` + optional `draft` agent + optional `ai_may_draft`). Has a unique `id` and declared `inputs`.
- **Gate**: a per-node schema (`gate.schema`) defining the typed object that may cross from an edge agent into the spine.
- **DecisionTable**: a versioned DMN table (`tables/*.dmn.yaml`) — input expressions, rules (FEEL), outputs, and a hit policy.
- **HitPolicy**: `UNIQUE` | `FIRST` | `PRIORITY` (`COLLECT` and others deferred — see Out of Scope).
- **Verdict**: a node's typed output; becomes a named input to dependent nodes.
- **RunRecord**: the persisted, replayable artifact — gated edge outputs, matched rules, **content snapshots of the tables and gate schemas used** (with version labels), human decision, advisory AI drafts, timestamps.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A workflow with ≥3 determination levels (edge → policy → human) runs end-to-end via `holodeck workflow run` and produces a single composed verdict.
- **SC-002**: Replaying any completed run — **using only the run record** (no on-disk tables, no registry) — reproduces the final verdict and every policy node's matched rule **100% identically**, with **0** LLM/agent calls during replay.
- **SC-003**: A schema-invalid or free-text edge output is rejected at the gate in **100%** of cases — no verdict is ever produced from ungated AI output.
- **SC-004**: The three hit policies (`UNIQUE`/`FIRST`/`PRIORITY`) evaluate correctly against a conformance test suite, including no-match and `UNIQUE` multi-match edge cases.
- **SC-005**: The FEEL subset required by the loan-hardship sample (numeric comparison, ranges, and at least one date computation) evaluates correctly via the embedded evaluator.
- **SC-006**: A decision table can be tested in isolation; editing a rule causes the corresponding policy test to fail with an expected-vs-actual diff.
- **SC-007**: The loan-hardship sample runs end-to-end and replays identically, demonstrated in CI or a documented quickstart.
- **SC-008**: The workflow schema makes "AI output as a verdict" unrepresentable — there is no node kind that lets AI output become a verdict or a decision input; verified by schema-validation tests, not runtime fuzzing.

## Out of Scope (POC)

- BPMN/DMN **XML** import, visual modeling, sequence flows, events, timers, message flows, BPMN gateways.
- DMN hit policies beyond `UNIQUE`/`FIRST`/`PRIORITY` (e.g. `COLLECT`, `RULE ORDER`, `OUTPUT ORDER`, aggregations) — only the three the sample uses are in v1.
- A durable, DB-backed / signed determination record (POC uses OTel spans + a local run record). *North Star.*
- A web/task-inbox HITL UI (CLI prompt only).
- Deploy-as-API for workflows (`holodeck serve` / `deploy` integration).
- Versioned policy **registry**, compliance export, replay UI.
- Bidirectional graphs / callback re-extraction (strictly one-directional v1).
- Coupling to the aspirational `experiment.yaml` orchestration.

## Open Questions (resolve in `/speckit.plan`)

1. **Which Python FEEL evaluator to embed**, and the precise FEEL subset the real tables need (numeric, ranges, dates, `contains`, list functions). Candidates to spike: SpiffWorkflow's DMN/FEEL, or a standalone FEEL library. *(Recommended: run the FEEL spike before planning — it is the one assumption that can invalidate the approach.)*
2. **Run record format & location** — JSON shape and where it is written (e.g. `.holodeck/runs/<id>.json`), given the DB-backed record is deferred.
3. **Gate ↔ `ExecutionResult.structured_output` binding** — confirm the per-node `gate:` block validates `structured_output` cleanly for both SK and Claude backends, including the low-confidence-field convention.
4. **`workflow` CLI namespace** — confirm `holodeck workflow` is a free verb (no clash with existing/planned commands). The relationship to `experiment` is already decided: decoupled (see Out of Scope).

## North Star (named, not built)

"**Governable Agents**" — visual modeler, versioned policy registry, durable
signed determination records, replay UI, and a real HITL task inbox. This POC is
the first vertebra of that spine.
