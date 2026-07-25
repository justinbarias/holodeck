# Implementation Plan: Deterministic Spine (036)

> Sources: `specs/036-deterministic-spine/spec.md` + `refinements.md` (binding
> decisions: Claude-only edges; human-node table computes a recommendation;
> new `WorkflowTestExecutor`; confidence gating cut; FEEL syntax is the
> contract; `decided_by` identity). Plan produced 2026-06-10 after read-only
> codebase research and a FEEL-library landscape scan.
> Design input for T2/T3: `../dmn-yaml-mapping.md` (DMN ↔ YAML element
> mapping, DRD semantics, open design points, `source:` annotation proposal).

## Overview

Build a new self-contained subsystem — `src/holodeck/lib/workflow/` (engine),
`src/holodeck/models/workflow.py` + `decision_table.py` + `run_record.py`
(models), `src/holodeck/cli/commands/workflow.py` (CLI) — that runs a DAG of
determination nodes: edge nodes (agents behind schema gates) and workflow-level
`input_data` (facts of record) feed policy nodes (DMN tables + FEEL) feeding a
human node (CLI decision), with an OTel trace, a replayable local run record, and
a policy-test executor.

**Anchor sample: Targeted Compliance Framework compliance determination**
(retargeted 2026-07-25 from invented loan-hardship underwriting to real, public,
citable Australian policy — see `spec.md` US6). There is now an explicit **MVP
ship line**; see `spec.md` § "The MVP ship line" and `todo.md`.

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| FEEL evaluator | Spike **bkflow-feel** (MIT, lark grammar) first; fallback **pySFeel** (GPLv3 — needs license sign-off) | Landscape scan: bkflow-feel covers the FR-010 subset standalone (verify `date - date` in spike); SpiffWorkflow has **no** FEEL parser and lacks FIRST/PRIORITY — ruled out. FEEL syntax stays the contract; subset may narrow (refinements §5). |
| Hit-policy logic | In-house table evaluation (`UNIQUE`/`FIRST`/`PRIORITY`), FEEL cells via the embedded evaluator | We need exact control of no-match / UNIQUE-multi-match errors, version snapshots, and matched-rule reporting. (Spike may revisit if `bkflow-dmn` fits.) |
| Topological sort + cycle detection | stdlib `graphlib.TopologicalSorter` | Python ≥3.10 floor; cycle detection for free; no new dependency. |
| Node model | Discriminated union following the `ToolUnion` pattern (`src/holodeck/models/tool.py:916`) with `extra="forbid"` | Makes "AI output as a verdict" unrepresentable by construction (FR-005/SC-008). |
| Workflow schema publication | `scripts/generate_workflow_schema.py` + `schemas/workflow.schema.json` + sync test, mirroring `generate_agent_schema.py` / `tests/unit/test_agent_schema_sync.py` | FR-004, existing convention. |
| Edge invocation | `BackendSelector.select()` → `invoke_once()`; gate validates `ExecutionResult.structured_output` via `jsonschema` | Claude backend already produces/validates structured output; **POC validates Claude only** (refinements §1). |
| Human prompt | `click.prompt`/`click.confirm` (not InquirerPy) | Scriptable with `CliRunner(input=...)` — required by US3's independent test. Wizard-style InquirerPy is not needed for a select-one prompt. |
| Run record | `.holodeck/runs/<run-id>.json`, canonical JSON (sorted keys), sha256 integrity over embedded table/gate snapshots | Matches `.holodeck/` conventions; canonical form makes "byte-identical replay" testable. |
| OTel | `get_tracer()` + `start_as_current_span` per node (pattern: `agent_factory.py:161`), gated on `ObservabilityConfig` | FR-018, specs 018/022. Minimal spans land with the runner; full GenAI attributes are a dedicated task. |
| Errors | New `WorkflowError` family under `holodeck.lib.errors` (`GateValidationError`, `TableEvalError`, …) | Repo standard; CLI exit-code mapping follows `deploy.py:102` context-manager pattern. |

## Dependency Graph

```
T1 FEEL spike
 │
 ├────────────► T3 table model + FEEL wrapper ──► T3a provenance
 │                        │
 │                        └──► T4 hit policies ──────────────┐
 │                                                           │
T2 workflow models + schema ──► T2a input_data               │
 │                    │                                      │
 │                    └──► T5 edge executor ─────────────────┤
 │                                                           ▼
 └───────────────────────────────────► T6 runner + CLI run + review gate
                                                  │
                                          T7 multi-level composition
                                                  │
                                          T8 human node prompt
                                                  │
                                          T10 run record ──► T11 replay
                                                  │
                                    T13a "active months" spike
                                                  │
                                          T14 TCF sample
        ═══════════════════════════════════════════════════════ MVP SHIP LINE
                          post-MVP: T9 draft · T12 tests · T13 OTel · T15 docs
```

The T6 → T7 edge still holds, but its meaning changed: **T7's implementation
landed inside T6** (the named-input resolver the runner needed gives multi-level
composition for free). T7 now depends on T6 as *tests depend on the code they
cover*, not as unbuilt work waiting on a prerequisite. See T7.

Parallelizable once Phase 1 lands: T12 (needs only T4), T13 (needs T6).
**T9 is no longer a dependency of T14** — the MVP sample's human node ships with
no `draft:` block (permitted by FR-015a). T9 restores it post-MVP.

## Task List

### Phase 0 — De-risk (the one assumption that can invalidate the approach)

#### Task 1: FEEL evaluator spike

**Description:** Add `bkflow-feel` (uv add) and write a conformance test
exercising the full FR-010 subset: numeric comparisons, all four range bracket
forms, `and`/`or`/`not()`, string equality, list membership (`in`), date
literals, date comparison, **date difference**. Decide bkflow-feel vs pySFeel
(GPLv3 needs explicit sign-off) and whether `bkflow-dmn` replaces in-house
table logic. Record the decision + the exact supported subset in
`specs/036-deterministic-spine/research.md`.

**Acceptance criteria:**
- [ ] Every FR-010 subset feature has a pass/fail verdict in `research.md`, including `date(..) - date(..)`.
- [ ] A library is chosen with license noted; unsupported expressions enumerated (these become the static-rejection list in T3).
- [ ] If the subset must narrow, the narrowed FEEL-compatible subset is documented and the sample tables (T14) are noted as bending to it (refinements §5).

**Verification:** `pytest tests/unit/workflow/test_feel_conformance.py -n auto -v`

**Dependencies:** None · **Files:** `pyproject.toml`, `tests/unit/workflow/test_feel_conformance.py`, `specs/036-deterministic-spine/research.md` · **Scope:** S

---

### Phase 1 — US1: a single determination node end-to-end (P1)

#### Task 2: Workflow Pydantic models + published JSON schema

**Description:** `src/holodeck/models/workflow.py`: `Workflow`, node
discriminated union (`EdgeNode` = `edge.agent` + `gate`; `PolicyNode` =
`decision` + `inputs` + `hit_policy`; `HumanNode` = + `requires_human`,
optional `draft`/`ai_may_draft: [reasons]`, optional `decided_by`),
`extra="forbid"`. Model-level validation: unique ids, unresolved `inputs`
refs, cycle rejection via `graphlib`. Publish
`schemas/workflow.schema.json` via `scripts/generate_workflow_schema.py` +
sync test (mirror agent-schema pattern).

**Acceptance criteria:**
- [ ] Valid workflow YAML parses; cycles and unresolved refs raise `ValidationError` naming the offending ids (US2 scenarios 2–3, FR-003).
- [ ] SC-008 tests: an edge node declaring `decision`/`hit_policy`, or any field routing AI output to a verdict, is rejected by the schema (FR-005).
- [ ] Schema sync test passes (FR-004).

**Verification:** `pytest tests/unit/models/test_workflow.py tests/unit/test_workflow_schema_sync.py -n auto` · `make type-check`

**Dependencies:** None · **Files:** `src/holodeck/models/workflow.py`, `scripts/generate_workflow_schema.py`, `schemas/workflow.schema.json`, 2 test files · **Scope:** M

#### Task 3: Decision-table model + loader + FEEL subset wrapper

**Description:** `src/holodeck/models/decision_table.py` (`DecisionTable`,
`Rule`, required `version` label) loading `tables/*.dmn.yaml` relative to
`workflow.yaml` (ConfigLoader conventions, env substitution not needed).
`src/holodeck/lib/workflow/feel.py` wraps the chosen evaluator and statically
rejects out-of-subset expressions at table load with a precise locator
(table id, rule index, cell) — FR-010, edge case "FEEL evaluation error".

**Acceptance criteria:**
- [ ] A valid table loads with version; a missing `version` fails validation (FR-013 first half).
- [ ] Malformed FEEL or out-of-subset expression fails at load with table/rule/cell locator.
- [ ] Evaluator wrapper evaluates every in-subset expression class from T1's conformance list.

**Verification:** `pytest tests/unit/workflow/test_decision_table.py -n auto`

**Dependencies:** T1 · **Files:** `src/holodeck/models/decision_table.py`, `src/holodeck/lib/workflow/feel.py`, tests · **Scope:** M

#### Task 2a: Workflow-level `input_data:` *(amends landed T2)* — **DONE** (`1d9d2eb`)

**Description:** Add an optional workflow-level `input_data:` block to
`src/holodeck/models/workflow.py`: a mapping of name → `{schema: <path>}`
declaring typed facts supplied from outside the spine. `holodeck workflow run`
gains `--input <payload.json>`; every declared fact is JSON-Schema-validated
**before any node executes**, and a missing or invalid fact fails loudly.
`input_data` names resolve in a node's `inputs:` list and in table input
expressions identically to node verdicts. The model MUST make an agent-produced
`input_data` entry unrepresentable (no `edge`/`agent` field). Re-publish
`schemas/workflow.schema.json` via the generator + sync test.

**Why an amendment:** T2 is committed. The schema is soft now and hardens with
every task after T4 — this is the cheap moment (same reasoning that put `source:`
into T2 early).

**Acceptance criteria:**
- [x] Declared facts validate pre-execution; missing/invalid fails with a typed error naming the fact (FR-025).
- [x] `input_data` referenceable from `inputs:` and from table expressions (FR-026). Fact names resolve in `inputs:` but are **excluded from the executed topological order** — they are data, not nodes.
- [x] SC-010: schema-validation test proves no agent can produce an `input_data` value (FR-027) — asserted against the **published** `workflow.schema.json`, not only the Pydantic model.
- [x] Schema sync test passes.

**Verification:** `pytest tests/unit/models/test_workflow.py tests/unit/test_workflow_schema_sync.py -n auto` · `make type-check`

**Dependencies:** T2 · **Files:** `src/holodeck/models/workflow.py`, `scripts/generate_workflow_schema.py`, `schemas/workflow.schema.json`, tests · **Scope:** M

#### Task 3a: `provenance:` on decision tables *(amends landed T3)* — **DONE** (`1d9d2eb`)

**Description:** Add an optional non-executable `provenance:` block to
`DecisionTable` — `generated_by`, `source`, `source_doc`, `source_sha256`,
`reviewed_by`, `reviewed_at`. A hand-authored table omits it entirely.
`provenance` MUST NOT be referenceable from FEEL expressions or affect rule
matching (FR-029, FR-032). Enforcement of the review gate itself lands in T6.

**Why now:** spec 039 will emit generated tables. Without provenance the engine
cannot distinguish an LLM-written table from a hand-written one, and the review
gate (FR-030) has nothing to check.

**Acceptance criteria:**
- [x] Table with full `provenance` loads; table without it loads unchanged.
- [x] A FEEL expression referencing `provenance.*` is rejected at load with a locator. Unreachable by **two independent routes**: a reserved-root check at load, and `provenance` is never placed in the evaluation context. FR-032 survived 16 evasion attempts in audit.

**Verification:** `pytest tests/unit/workflow/test_decision_table.py -n auto`

**Dependencies:** T3 · **Files:** `src/holodeck/models/decision_table.py`, tests · **Scope:** S

#### Task 4: Hit-policy evaluation + conformance suite — **DONE** (`1d9d2eb`)

**Description:** `src/holodeck/lib/workflow/table_eval.py`:
`evaluate(table, named_inputs) -> Verdict` (value, matched rule index/id,
table version). Implements `UNIQUE`/`FIRST`/`PRIORITY` with standard DMN
semantics; loud `TableEvalError` on no-match (absent declared default) and on
UNIQUE multi-match (FR-011, FR-012).

**Acceptance criteria:**
- [x] SC-004 conformance suite passes: each hit policy, no-match (with and without default), UNIQUE multi-match.
- [x] `Verdict` carries matched-rule identity + table version (needed by record/replay and OTel).

**Verification:** `pytest tests/unit/workflow/test_table_eval.py -n auto`

**Dependencies:** T3 · **Files:** `src/holodeck/lib/workflow/table_eval.py`, tests · **Scope:** M

#### Task 5: Edge-node executor + schema gate — **DONE** (`1d9d2eb`)

**Description:** `src/holodeck/lib/workflow/edge.py`: load the node's agent
YAML (`ConfigLoader.load_agent_yaml`), invoke via `BackendSelector.select()` /
`invoke_once()`, validate `ExecutionResult.structured_output` against
`gate.schema` (JSON Schema file resolved relative to workflow.yaml) and emit a
`GatedOutput`. Free text (`structured_output is None`) or schema-invalid
output raises `GateValidationError` (FR-006/007/008). Claude-only per
refinements §1 — no SK structured-output work.

**Acceptance criteria:**
- [x] Valid structured output crosses; the gated object (not raw text) is the canonical value. The gate schema is snapshotted **by content**.
- [x] Free-text and schema-invalid outputs are rejected with a typed error naming node id + validation failure (US1 scenario 2, SC-003) — three distinct failure channels: output rejected / gate unusable / nothing produced to judge.
- [x] Tests use a mocked `AgentBackend` — zero live LLM calls; gate validation never touches the network.

**Verification:** `pytest tests/unit/workflow/test_edge_gate.py -n auto`

**Dependencies:** T2 · **Files:** `src/holodeck/lib/workflow/edge.py`, errors additions in `holodeck/lib/errors`, tests · **Scope:** M

#### Task 6: Workflow runner + `holodeck workflow run` (single level) — **DONE** (`1d9d2eb`), with one recorded gap

**Description:** `src/holodeck/lib/workflow/runner.py`: topo-order via
`graphlib`, execute edge nodes then policy nodes, hold per-node results,
minimal OTel span per node (`get_tracer` pattern, gated on observability
config). `src/holodeck/cli/commands/workflow.py`: `workflow` group +
`run <workflow.yaml>` subcommand registered in `cli/main.py`, error→exit-code
mapping per the `deploy.py` context-manager pattern, `click.echo` output.

**Acceptance criteria:**
- [x] US1 scenarios 1–2 pass as an integration-style test (mocked backend, CliRunner): valid edge output → table verdict echoed; invalid → exit non-zero with gate error. Exit codes separate misauthored / could-not-decide / gate-rejected / invocation-failed.
- [x] Load-time validation failures occur before any agent invocation (FR-003) — `prepare_workflow` performs **all** validation and **constructs no backend**, so FR-003 is structural rather than merely ordered. `execute_workflow` is the separate execution half.
- [x] **Delivered T7's implementation as a side effect** — `_named_inputs` gives multi-level composition at arbitrary depth. T7 is retitled to proof-and-cover work accordingly.
- [ ] ⚠ **US1 scenario 3 / FR-018 NOT met.** `Workflow` has no `observability:` block and is `extra="forbid"`, so `holodeck workflow run` has nowhere to obtain an `ObservabilityConfig` and emits **zero spans**. `execute_workflow` accepts one and is unit-tested with an in-memory exporter, but the CLI path is unreachable end to end. **To be closed by T13.**

**Verification:** `pytest tests/unit/cli/test_workflow_command.py tests/unit/workflow/test_runner.py -n auto` · `holodeck workflow run --help`

**Dependencies:** T4, T5 · **Files:** `src/holodeck/lib/workflow/runner.py`, `src/holodeck/cli/commands/workflow.py`, `src/holodeck/cli/main.py`, tests · **Scope:** M

### Checkpoint 1 — US1 complete — **PASSED 2026-07-25**
- [x] `make format && make lint && make type-check` clean; `pytest tests/unit -n auto` green — **5298 passing, 4 skipped; security and pre-commit also clean**. Full `make ci` re-run green at checkpoint.
- [x] A single edge→policy workflow runs via CLI with a mocked agent — **and live**, via `sample/pbas-points/`: free text classified by Claude, gate-validated, then `tables/points.dmn.yaml` rule 10 awarded 20 points `per_week`. The determination came from the table, not the model (SC-003, FR-005 demonstrated rather than asserted).
- [x] **Human review before Phase 2** — done by the maintainer, who ran the live workflow.

**Phase 2 is unblocked.** Carried forward as known-incomplete, not silently: FR-018
unreachable from the CLI (see T6, closed by T13); `format: date` → `datetime.date`
conversion absent (T13a prerequisite); no adversarial pass has yet audited the
*seams between* tasks — each was audited in isolation, and both of the worst
findings so far (the `!=` silent verdict, remote `$ref` fetching) lived at
boundaries.

---

### Phase 2 — US2: composed determination levels (P1)

#### Task 7: Multi-level composition — **prove and cover**

> **The implementation landed inside T6. Do not re-implement it.** Building the
> runner required `runner._named_inputs`, which resolves each declared input name
> to a fact of record, an upstream edge node's gated object, or an upstream policy
> node's `Verdict.outputs`. Once that exists, arbitrary depth follows from
> `graphlib`. T7's original deliverable — "node `a`'s verdict (or gated object) is
> available as variable `a` in dependent tables' FEEL context (FR-009, FR-016)" —
> is therefore **already satisfied**. Verified empirically against the committed
> code, unmodified: a 3-edge → 2-policy → 1-policy DAG executes in order
> `('a','b','c','mid1','mid2','top')` and composes correctly, with node `b`
> feeding **both** mid-level nodes (a diamond, not a tree) and the top table
> dot-pathing into upstream verdicts (`mid1.band`).

**Description:** What remains is **proof, not build**. Nothing currently defends
multi-level composition: there is no `test_composition.py`, and `test_runner.py`
covers only single-level DAGs. That is a live risk, not a theoretical one — the
recent audit found that flipping `all()` to `any()` in the hit-policy matcher
survived mutation testing entirely, because correct code with no test defending
it is indistinguishable from broken code. Composition is in exactly that state
today. Write the 3→2→1 integration test (mocked edges, diamond shape, dot-paths
into upstream verdicts), close the CLI-level gap on US2 scenario 3, and add the
sample's second level.

**Acceptance criteria:**
- [ ] US2 scenario 1 covered by a 3-edge → 2-policy → 1-policy integration test with mocked edges: evaluation order is a valid topological sort, each higher node receives lower nodes' verdicts as named inputs, and the final verdict reflects the composed sub-verdicts. Use the diamond shape (one edge feeding two mid-level nodes) and at least one dot-path into an upstream verdict — a tree with scalar inputs under-tests the resolver.
- [ ] US2 scenario 3 re-asserted **through the CLI**: an unresolved `inputs` reference → clean error, exit code 2, no agent invoked. *(Scenario 2, the cycle, is already covered end to end by `tests/unit/cli/test_workflow_command.py::test_cycle_exits_two_without_invoking_an_agent`; unresolved references are covered only at model and `prepare_workflow` level.)*
- [ ] **Sample-proof (blocking):** `sample/pbas-points/` demonstrates a genuine two-level DRD it cannot demonstrate today. Footnote (1) of the pinned source — *"These tasks and activities are available to Workforce Australia Services participants only"* — is an **eligibility precondition attached to rows**, not a points rule, and is currently encoded inline as rules 5/6 of `tables/points.dmn.yaml`. Extract it into `tables/eligibility.dmn.yaml`, keyed on `activity.activity_type` + the existing `participant.stream` fact, whose verdict becomes a named input to `points_award`. This is a **refactor plus a level**: the verdict for every existing case — in particular `drivers_licence_attainment` + `workforce_australia_online` → 0 points / `not_available` — must be unchanged. README's "What is modelled" table updated to name the new decision.

> **Source note (verified against the PDF, p.2).** Footnote (1) covers exactly
> **one** of the eleven modelled activities (driver's licence attainment), so the
> eligibility table restricts in one direction only. Footnote **(2)** is the
> mirror case — *"available to Workforce Australia Online participants only"*
> (online learning modules; Career coaching – Youth Advisory Sessions), neither
> currently in the taxonomy. Adding one is optional scope; without it the level
> is real but thin, which is honest rather than wrong.

**Verification:** `pytest tests/unit/workflow/test_composition.py tests/unit/cli/test_workflow_command.py -n auto` · `holodeck workflow run sample/pbas-points/workflow.yaml --input sample/pbas-points/case.json`

**Dependencies:** T6 · **Files:** `tests/unit/workflow/test_composition.py` + fixture YAMLs, `tests/unit/cli/test_workflow_command.py`, `sample/pbas-points/**`. **No `runner.py` change is expected** — if one turns out to be needed, that is a defect the new test found, which is the point. · **Scope:** S

### Checkpoint 2 — composition proven
- [ ] Full DAG executes bottom-up with named inputs; CI targets clean.

---

### Phase 3 — US3: human-accountable determination (P1)

#### Task 8: Human node — recommendation + CLI decision prompt

**Description:** Per refinements §2/§6: at a `requires_human` node the runner
evaluates the node's table to produce a **recommendation** (+ matched rule),
presents composed inputs + recommendation at the CLI, displays declared
`decided_by` and asks the operator to confirm/enter their name, then
`click.prompt`s a selection among the table's declared outputs. Override
(human ≠ recommendation) is flagged. Abort (Ctrl-C / EOF) → run ends with
explicit "no determination" status, nothing recorded as final (edge case).

**Acceptance criteria:**
- [ ] US3 scenarios 1 & 3 pass via `CliRunner(input=...)`: pause + presentation; verdict = human choice, attributed to confirmed `decided_by` name with timestamp.
- [ ] Recommendation and override flag are captured in the node result (feeds T10).
- [ ] Abort path terminates with "no determination" and non-zero exit.
- [ ] **Sample-proof (blocking):** `sample/pbas-points/` gains a `requires_human` node modelling the source's **activity-bonus Note** (PDF p.2): *"Providers and the Digital Services Contact Centre (DSCC) may increase the values of certain tasks or activities through an activity bonus to reflect the individual circumstances of the participant and the task or activity they are doing."* `corpus-manifest.md` names this the corpus's **canonical unmappable clause** — pure discretion, where a table must refuse to decide. The node takes `points_award` as its recommendation and a named `decided_by` delegate confirms it or records that a bonus applies. README item 2 of "What is deliberately NOT modelled" moves from *excluded* to *routed to a human*, and the run is scripted end to end with `CliRunner(input=...)`.

> **This is the sample's most important framing point: the human node exists
> because the source says a rule cannot.** Two constraints come from the document,
> not from design taste:
> 1. The Note authorises an **increase only** — never a reduction. "Confirm or
>    adjust" would overstate it.
> 2. It states **no amount and no criteria**. No table may compute a bonus value,
>    so the delegate's act is a **categorical selection among the table's declared
>    outputs** (per FR-014), not free numeric entry. A sample that let the operator
>    type an arbitrary points figure would be inventing a mechanism the source
>    does not contain.

> ### Transport: build the CLI only, but do not foreclose serve
>
> `holodeck serve` integration and a task-inbox HITL UI are both **Out of Scope**
> (`spec.md`). Build the CLI prompt and nothing else. Two constraints exist only
> to stop that choice becoming a dead end.
>
> **1. Keep `click` out of the runner.** One narrow seam: the runner asks
> *something* for a decision and records what came back. It must return the
> selected output, `decided_by`, a timestamp, and be able to say **aborted / no
> determination**. CLI is the only implementation built now.
>
> *Stated tension:* a Protocol with a single implementation is speculative under
> CLAUDE.md §2 ("no abstractions for single-use code"). It is justified here only
> because the alternative — a runner importing `click` — is a **known** dead end
> rather than a hypothetical one, and the seam is a few lines. If it grows past
> that, it has become the thing the rule warns about. Do not add a registry, a
> plugin point, or a second implementation.
>
> **2. If serve is ever wired: the runner emits the AG-UI events, never a model.**
> AG-UI can carry this as a frontend `prompt_user` tool call — the mechanism at
> `serve/protocols/agui.py:782`, where a run resumes from a `tool` message. But in
> the *normal* AG-UI flow the **model** emits the tool call and the result returns
> **to the model's context**. Routing a human node that way would put Claude
> between the delegate and the verdict, free to paraphrase, drop, or re-interpret
> the determination. That is precisely the invariant this spec exists to hold.
>
> Nothing in AG-UI requires a tool call to originate from an LLM — they are events
> on a stream. So the runner emits `ToolCallStart/Args/End` for `prompt_user`
> itself, awaits the `tool` message, and writes the decision straight into the
> verdict. AG-UI stays pure transport and no model is in the loop. **The obvious
> implementation is the wrong one; record this before anyone reaches for it.**
>
> **3. Durability is T10's problem, and it is nearly free.** A frontend tool
> round-trip only works while someone is watching the stream; a delegate deciding
> tomorrow needs the run to *suspend*, which an SSE connection cannot hold. The
> **run record is already the resume token** — T11 replay reconstructs every
> verdict up to the human node from snapshots with zero LLM calls, which is
> exactly what resume requires. The only addition is a record whose status is
> `awaiting_human` with the decision absent. So suspend/resume is **additive over
> T10+T11, not a redesign** — a reason to keep them ahead of T14. Build none of it
> in T8; just do not design a decision seam that cannot later be satisfied by a
> value arriving from somewhere other than a terminal.

**Verification:** `pytest tests/unit/workflow/test_human_node.py -n auto` · scripted `holodeck workflow run sample/pbas-points/workflow.yaml --input sample/pbas-points/case.json`

**Dependencies:** T7 · **Files:** `src/holodeck/lib/workflow/human.py`, `runner.py`, `cli/commands/workflow.py`, tests, `sample/pbas-points/**` · **Scope:** M

#### Task 9: Draft agent advisory path (`ai_may_draft: reasons`) — **POST-MVP**

> Documented here beside T8 for cohesion, but it sits **after the ship line**.
> The MVP sample's human node has no `draft:` block (FR-015a permits it), so T14
> no longer depends on this. T9 restores the block post-MVP.

**Description:** When a human node declares `draft`, invoke the drafting agent
(via `BackendSelector`, mocked in tests) to produce only the fields in
`ai_may_draft` (POC: `reasons`); show as clearly-advisory text before the
prompt; record as advisory context (FR-015a). Schema already prevents AI
output reaching the verdict (T2); add an explicit test.

**Acceptance criteria:**
- [ ] US3 scenario 2: reasons draft displayed as advisory; no AI field can populate the verdict (schema-validation test, SC-008).
- [ ] Human node without `draft` presents inputs only.
- [ ] **Sample-proof (blocking):** `sample/pbas-points/`'s human node gains a `draft:` block with `ai_may_draft: [reasons]`, producing advisory reasons bearing on whether an activity bonus is warranted — **never the points value and never a bonus amount**. The source states no amount, so a drafted figure would be invented policy; the README says so explicitly.

**Verification:** `pytest tests/unit/workflow/test_draft_agent.py -n auto`

**Dependencies:** T8 · **Files:** `human.py`, tests · **Scope:** S

### Checkpoint 3 — bright line proven
- [ ] Scripted end-to-end: 3 edges → 2 policies → human decision with draft reasons, all mocked, deterministic.
- [ ] **Human review before Phase 4.**

---

### Phase 4 — US4: record & replay (P2)

#### Task 10: RunRecord model + persistence

**Description:** `src/holodeck/models/run_record.py` + writer in
`src/holodeck/lib/workflow/record.py`: persist to `.holodeck/runs/<run-id>.json`
(canonical JSON, sorted keys): gated edge outputs, per-policy matched rule +
verdict, **content snapshots of every table and gate schema used** (version
labels + sha256), human recommendation/decision/override/`decided_by`,
advisory drafts, timestamps (FR-013, FR-019).

**Acceptance criteria:**
- [ ] A completed run writes a record sufficient for replay with the `tables/` and `schemas/` dirs deleted.
- [ ] Record round-trips through the Pydantic model; snapshot hashes verify.
- [ ] **Sample-proof (blocking):** a `sample/pbas-points/` run writes `.holodeck/runs/<id>.json` carrying the validated `input_data` facts (`participant`, `claim`), the gated activity classification, the matched rule of **each** policy node (eligibility and points award), the delegate's decision and override flag, and content snapshots of both tables and the gate schema **with their `provenance` blocks** (`source`, `source_doc`, `source_sha256`) and sha256 integrity.

**Verification:** `pytest tests/unit/workflow/test_run_record.py -n auto` · inspect `.holodeck/runs/` after a PBAS run

**Dependencies:** T8 · **Files:** `src/holodeck/models/run_record.py`, `src/holodeck/lib/workflow/record.py`, `runner.py`, tests · **Scope:** M

#### Task 11: `holodeck workflow replay <record>`

**Description:** Replay re-evaluates policy + human layers from the record's
snapshots and recorded human decision; never constructs a backend (FR-020/021).
Integrity failure (missing/corrupt snapshot, hash mismatch) fails loudly with
no fallback to on-disk tables.

**Acceptance criteria:**
- [ ] US4 scenarios 1–3: identical verdict + matched rules (byte-identical canonical JSON comparison, SC-002); zero LLM calls (assert `BackendSelector` never invoked); edited/missing on-disk tables don't affect replay; corrupt record → loud failure.
- [ ] **Sample-proof (blocking):** the `sample/pbas-points/` run record replays to an **identical** verdict with `sample/pbas-points/tables/` and `sample/pbas-points/schemas/` **deleted from disk** and **zero** LLM calls. Demonstrated as a copy-pasteable sequence in the sample README.

**Verification:** `pytest tests/unit/workflow/test_replay.py -n auto` · `holodeck workflow replay .holodeck/runs/<id>.json` with the PBAS `tables/` + `schemas/` dirs moved aside

**Dependencies:** T10 · **Files:** `src/holodeck/lib/workflow/replay.py`, `cli/commands/workflow.py`, tests · **Scope:** M

### Checkpoint 4 — "prove it" property
- [ ] Run → record → replay loop demonstrated in tests; CI targets clean.

---

### Phase 6 — US5: policy-as-code testing (P3 — **POST-MVP**, parallelizable after T4)

#### Task 12: `WorkflowTestExecutor` + `holodeck workflow test`

**Description:** Per refinements §3: a new executor mirroring the existing
test framework's case/result *shapes* (`TestCaseModel`-like YAML cases,
`TestResult`-like outputs) that evaluates a decision table over given input
rows — no agent, no backend, no LLM. CLI subcommand
`holodeck workflow test <policy-tests.yaml>` with expected-vs-actual diff
output on failure (FR-023, SC-006).

**Acceptance criteria:**
- [ ] US5 scenarios: pass iff table yields expected verdict; rule edit → failing test with expected-vs-actual diff.
- [ ] No `TestExecutor` refactor; existing test framework untouched.
- [ ] **Sample-proof (blocking):** `sample/pbas-points/` carries a **committed** policy-test file over `tables/points.dmn.yaml`, covering at minimum (a) the threshold tier boundary at **exactly 15** contact hours — *"contact hours up to 15 hours per week"* is inclusive, so 15 must yield **15** points, not 20 — and (b) an `activity_type` the table does not cover, asserting the loud no-match (`TableEvalError`, no silent zero). Case (b) is **only** reachable through the policy tester: the gate's closed `enum` and the table's rules cover the same eleven activities, so no *run* can produce it. That is the point of SC-006.

**Verification:** `pytest tests/unit/workflow/test_policy_executor.py -n auto` · `holodeck workflow test sample/pbas-points/<policy-tests>.yaml`

**Dependencies:** T4 · **Files:** `src/holodeck/lib/workflow/policy_test.py`, `cli/commands/workflow.py`, models for policy test cases, tests, `sample/pbas-points/**` · **Scope:** M

---

### Phase 7 — Observability completion (**POST-MVP**)

#### Task 13: Full OTel GenAI span attributes

**Description:** Upgrade the minimal spans from T6 to specs-018/022-conformant
attributes: per-node span with node id/kind, named inputs, table version,
matched rule(s), verdict; gate-rejection events; replay spans marked as replay.
Respect `ObservabilityConfig` enable/disable; reuse `RedactingSpanProcessor`
pipeline.

**Acceptance criteria:**
- [ ] FR-018: in-memory-exporter test asserts span names + attributes for edge, policy, human nodes.
- [ ] Observability disabled → zero spans, zero overhead path.
- [ ] **Closes T6's recorded gap:** `holodeck workflow run` must be able to obtain an `ObservabilityConfig` at all. Today `Workflow` is `extra="forbid"` with no `observability:` block, so the CLI path emits zero spans however well the runner is instrumented. Whatever shape this takes (a `workflow.yaml` block, a CLI flag, or project config) it re-publishes `workflow.schema.json` and its sync test.
- [ ] **Sample-proof (blocking):** a `sample/pbas-points/` run emits spans through the **CLI** carrying node id and kind, table version, matched rule, and each table's provenance — captured against an in-memory exporter, not asserted by hand.

**Verification:** `pytest tests/unit/workflow/test_otel_spans.py -n auto`

**Dependencies:** T11 · **Files:** `runner.py`, `replay.py`, possibly `lib/observability` constants, tests · **Scope:** S

---

### Phase 5 — US6: anchor sample (P1 — the MVP's proof)

#### Task 13a: "Active months" FEEL spike *(before T14)*

**Description:** Resolve spec Open Question 5. TCF expires demerits after "6
**active** months" and resets after "3 **active** months compliant" — a count of
months in which the participant was active, **not** elapsed calendar time, so it
is not a date difference. Determine whether it is expressible via
`inputs[].expression`, or must be pre-computed and supplied as `input_data`.
Recall `research.md` caveat 6: `date(variable)` does not parse; date-typed
fields cross the gate as `datetime.date` and subtract bare, yielding a
`timedelta` that caveat 1 requires converting to days.

> **Prerequisite the spike must record.** `research.md` caveat 6's
> `format: date` → `datetime.date` conversion **does not exist**. `input_data`
> stores the raw JSON value, so a `date`-typed table column receives a **string**
> and bare subtraction raises. T13a and T14 both depend on date arithmetic, so
> the spike's verdict must say who converts and where.

**Acceptance criteria:**
- [ ] A written verdict in `research.md`: expressible, or pre-computed into `input_data` with the shape specified.
- [ ] If the FEEL subset must narrow further, it is documented and SC-005 is amended (refinements §5: the sample bends to the subset, never the reverse).
- [ ] The date-conversion gap above is either closed or explicitly assigned to T14.
- [ ] **Sample-proof: none, deliberately.** This task is a written verdict, not code, and `sample/pbas-points/` has no date arithmetic to extend. Do not invent a sample change for it.

**Dependencies:** T3 · **Files:** `research.md`, a conformance test · **Scope:** S

#### Task 14: Targeted Compliance Framework sample

**Description:** `sample/tcf-compliance/` — one mutual-obligation compliance
event under the Australian TCF, modelled as a **state transition**:

- `input_data`: prior compliance state (zone, active demerits + accrual dates, prior penalty count, months compliant) with a JSON Schema. Never LLM-touched.
- One edge node: an agent (anthropic provider — Claude-only POC) assessing the participant's stated reason against the enumerated factors of the *Reasonable Excuse Determination 2018* s5(2)(a)–(j), across a schema gate derived from that closed list.
- `tables/zone.dmn.yaml` (`UNIQUE`) — next zone from prior state + this event.
- `tables/penalty.dmn.yaml` (`FIRST`) — escalation: 1 week → 2 weeks → cancellation.
- A `requires_human` node with `decided_by` (delegate) and **no `draft:` block** (T9 is post-MVP; FR-015a permits this).
- Tables bend to the T1-verified FEEL subset and T13a's verdict.
- `README.md` carrying the framing statement required by US6 — **not optional**.

Source documents are pinned in `specs/039-policy-generator/corpus/` with hashes
in `corpus-manifest.md`; each table's `provenance.source` cites its authority.

**Acceptance criteria:**
- [ ] US6 scenarios: end-to-end run composes zone + penalty verdicts and the human decision; replay reproduces identically with zero LLM calls.
- [ ] US6 scenario 3: no agent produces any `input_data` fact (SC-010).
- [ ] An integration test (marked `@pytest.mark.integration`) or documented quickstart demonstrates SC-007; unit-level path uses mocked edges.
- [ ] README states the framing up front.
- [ ] **Sample-proof (blocking):** `sample/pbas-points/` still **runs and replays** after T14 lands, with every feature T7–T11 added to it intact. TCF is a second sample, not a replacement — the demonstrator must not rot while attention is on the anchor.

> **T14 must be rescoped before it starts — `sample/` is deliberately untracked.**
> Decided 2026-07-25: `.gitignore:45` keeps `/sample` ignored repo-wide and
> samples stay local-only (Open Question 4). So
> `tests/integration/test_tcf_sample.py` **cannot exist as written** — it would
> reference untracked paths and fail on a fresh clone. Pick one before starting:
> move the assertions onto committed fixtures under `tests/fixtures/workflow/`,
> or make the sample proof a documented quickstart run by hand. SC-007 permits
> the latter ("demonstrated in CI **or a documented quickstart**"), so the
> criterion is still satisfiable — just not automatically verifiable.

**Verification:** `holodeck workflow run sample/tcf-compliance/workflow.yaml --input case.json` (scripted input) · sample proof per the rescoping note above — **not** `tests/integration/test_tcf_sample.py` as originally specified

**Dependencies:** T11, T13a · **Files:** `sample/tcf-compliance/**` (YAML/JSON/MD), one integration test · **Scope:** M (mostly YAML)

---

## ▲ MVP SHIP LINE ▲

**MVP = T4, T5, T6, T7, T8, T10, T11, T14** plus amendments **T2a**, **T3a** and
spike **T13a**. Delivers SC-001…SC-005 and SC-007…SC-010; only **SC-006**
(a table tested in isolation, needs T12) falls past the line.

### Two samples, two jobs

`sample/pbas-points/` is the **running demonstrator**: it exists from T6 onward
and grows one feature per task, so every task above T6 carries a *blocking*
sample-proof criterion naming what PBAS must newly demonstrate. A task is not
done while the demonstrator cannot show its feature. `sample/tcf-compliance/`
(T14) is the **anchor MVP proof** — real statute, prior-state transition, human
delegate, a gate derived from an enumerated legal list — and lands once,
complete. It is not retargeted onto PBAS and its scope is not weakened by
anything PBAS proves. Full statement of the convention: `todo.md`.

---

### Phase 8 — Documentation (**POST-MVP**)

#### Task 15: Documentation

**Description:** `docs/` page: workflow concepts (spine, gates, determination
nodes, the one rule), authoring reference for `workflow.yaml` /
`tables/*.dmn.yaml`, CLI usage (`run`/`replay`/`test`), replay guarantees and
their limits (POC ≠ legal-grade audit store). Update `AGENTS.md` pointers if
conventions require.

**Acceptance criteria:**
- [ ] Quickstart reproduces the sample run + replay verbatim.
- [ ] The "LLM is never the spine" invariant and POC scope-of-claim are stated.
- [ ] **Sample-proof (blocking):** the quickstart reproduces the `sample/pbas-points/` **run and replay verbatim** — the commands copy-paste unmodified and the printed verdict matches what the page claims. Executed once as part of verification, not assumed.

**Verification:** Manual doc review; quickstart commands executed once.

**Dependencies:** T14 · **Files:** `docs/workflows.md` (or docsite equivalent) · **Scope:** S

### Checkpoint 5 — feature complete
- [ ] All SC-001…SC-010 demonstrably met (map each to its test in the PR description).
- [ ] `make ci` clean. **Human review / PR.**

---

## Successor: 039 policy generator

`specs/039-policy-generator/` — AI-drafted decision tables and `workflow.yaml`
DRDs from policy documents, with TODO markers where edge nodes go. Specced in
parallel with this work so its schema needs land here as additions (T2a, T3a)
rather than as migrations later; **built only after this spec's MVP ships**.

036 ships the two things that make a drafted table safe to run — the
`provenance` block (T3a) and the review gate (T6) — and nothing else about
generation. The golden corpus is already pinned:
`specs/039-policy-generator/corpus-manifest.md`.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| bkflow-feel gaps (date difference unverified; no duration literals; dormant since 2023) | High — invalidates FR-010 approach | T1 spike first; pin or vendor (MIT allows); fallback pySFeel **only with GPLv3 sign-off**; last resort: narrow the subset (refinements §5 forbids non-FEEL syntax) |
| Byte-identical replay brittleness (float/dict-order/timestamp drift) | Med | Canonical JSON (sorted keys), timestamps excluded from the compared verdict payload, sha256 snapshot integrity |
| Human prompt untestable | Med | `click.prompt` + `CliRunner(input=...)`; no InquirerPy on this path |
| `structured_output` quirks on the live Claude backend (vs mocks) | Med | Sample integration test (T14) is the live check; unit suite fully mocked |
| Scope creep toward a workflow engine | Med | Out-of-Scope list is binding; any events/timers/callbacks proposal is rejected in review |
| OTel attribute bloat / PII in spans | Low | Reuse `RedactingSpanProcessor`; record full payloads only in the run record, not spans |

## Open Questions (need human input before/at execution)

1. ~~**GPLv3 tolerance** if bkflow-feel fails the spike~~ — moot; T1 chose bkflow-feel (MIT).
2. **Policy-test verb**: plan assumes `holodeck workflow test <file>`. Confirm vs e.g. `holodeck test --policy`.
3. ~~**Sample model choice** for the edge agents~~ — **settled**: `anthropic` / `claude-opus-5`, `temperature: 0.0` (`sample/pbas-points/agents/activity-classifier.yaml`), run end to end successfully.
4. ~~**`sample/` is not tracked by git**~~ — **DECIDED 2026-07-25: leave ignored,
   defer.** `.gitignore:45` keeps `/sample` ignored repo-wide; samples remain
   local-only and `.gitignore` is not to be changed. Revisit only with a concrete
   reason. Binding consequences: **every sample-proof criterion in this plan is a
   manual check, never a CI gate** — report them as such rather than implying a
   pipeline confirmed them; **T14 needs rescoping before it starts** (see the note
   under T14 — `tests/integration/test_tcf_sample.py` cannot reference untracked
   paths); and **SC-007 keeps only its documented-quickstart limb**, remaining
   satisfiable but not automatically verifiable.

## Execution Notes

- Always `source .venv/bin/activate`; run `make format && make lint && make type-check` after each task (CLAUDE.md).
- Tests: `pytest -n auto`, AAA, `@pytest.mark.unit` default; live-LLM only in T14's integration test.
- Conventional commits, no Claude attribution; branch `036-deterministic-spine`.
