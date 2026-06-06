# The Deterministic Spine — Composable Determination Gates for HoloDeck

> Status: idea (pre-spec) · Author: FDE · Date: 2026-05-30
> Companion concept doc: *The Deterministic Spine* (FDE-01/A)
>
> **Superseded by `specs/036-deterministic-spine/spec.md` (2026-05-30).** The spec
> updates naming: the artifact is `workflow.yaml` (not `determination.yaml`), the
> CLI is `holodeck workflow run|replay`, and "gate" means the schema boundary on
> an edge node (the DMN nodes are "determination nodes"). This doc is retained as
> the original ideation record; the spec is authoritative.

## Problem Statement

**How might we let HoloDeck users model a deterministic business process — its
rules and its decision composition — as the authoritative "spine," and confine
probabilistic HoloDeck agents to the leaf nodes, so the determination stays
reproducible, replayable, and auditable while the AI does only the
judgment-light work?**

Today HoloDeck has no way to put rules in charge. Its multi-agent orchestration
story (sequential / handoff / group-chat / magentic) is *emergent* by design —
the opposite of what a regulated or enterprise decision needs. When an AI step
sits anywhere on the path to a consequential, contestable decision, the LLM
becomes the de-facto spine: non-reproducible, unexplainable, and impossible to
audit. That is precisely the failure mode administrative law forbids (and that
Robodebt embodied).

## Recommended Direction

Ship **"just the gate" as a composable primitive**, not a workflow engine.

A *gate* is a single determination node: it takes a typed object, runs it
through **policy-as-code (a DMN decision table with FEEL expressions)**, and
emits a typed, audited verdict. The LLM never crosses the gate — at the leaf
level an agent produces a schema-validated input object; everything above is
deterministic.

Gates **compose by feeding their verdict upward as a named input to the gate
above** — which is exactly DMN's Decision Requirement Diagram (DRD). This is
what gives us *multiple determination levels* for free, with no orchestration
engine:

```
                    ┌──────────────────────────────────┐
  LEVEL 3           │  FINAL DETERMINATION   ◆ gate SHUT │   human-accountable
  human-accountable │  DMN · PRIORITY hit policy         │   AI drafts reasons only
                    └──────────────────┬─────────────────┘
                          ┌────────────┴────────────┐
  LEVEL 2          ┌──────▼───────┐          ┌───────▼──────┐
  policy-as-code   │ AFFORDABILITY│          │  RISK TIER   │   no AI reaches in —
  (pure DMN/FEEL)  │ DMN · UNIQUE │          │ DMN · FIRST  │   composes verdicts below
                   └───┬──────┬───┘          └──────┬───────┘
              ┌────────┘      └─────┐               │
  LEVEL 1     ▼ AI edge        AI edge ▼        AI edge ▼
  probabilistic  income-extract   residency-verify   doc-fraud-flag
  (agent → gate) agent→schema     agent→schema       agent→schema
```

Every upward arrow is a typed object crossing a schema gate. Level 1 is the only
probabilistic part; Levels 2–3 are deterministic and replayable; Level 3 is the
bright line where a named human decides and the AI may draft reasons only.

**Authoring:** native HoloDeck YAML DSL (no BPMN/DMN XML in the POC). The YAML is
the canonical spine and is designed so a `.dmn`/`.bpmn` importer can transpile
*into* it later — but standard XML import is explicitly deferred. This keeps the
feature true to HoloDeck's no-code, git-diffable, hand-editable ethos.

```yaml
# determination.yaml — a determination graph of composable gates
determinations:
  - id: income                      # LEVEL 1 — AI edge
    edge: { agent: agents/income-extractor/agent.yaml }
    gate: { schema: schemas/income.json }   # only typed objects cross

  - id: residency                   # LEVEL 1 — AI edge
    edge: { agent: agents/residency-verifier/agent.yaml }
    gate: { schema: schemas/residency.json }

  - id: doc_fraud_flag              # LEVEL 1 — AI edge
    edge: { agent: agents/doc-fraud-detector/agent.yaml }
    gate: { schema: schemas/fraud-flag.json }

  - id: affordability               # LEVEL 2 — pure policy
    decision: tables/affordability.dmn.yaml
    inputs: [income, residency]
    hit_policy: UNIQUE

  - id: risk_tier                   # LEVEL 2 — pure policy
    decision: tables/risk.dmn.yaml
    inputs: [doc_fraud_flag, income]
    hit_policy: FIRST

  - id: final_determination         # LEVEL 3 — the shut gate
    decision: tables/determination.dmn.yaml
    inputs: [affordability, risk_tier]
    hit_policy: PRIORITY
    requires_human: true            # CLI prompt in the POC; named human decides
    ai_may_draft: reasons           # AI confined to drafting reasons, nothing more
```

Why this direction wins for **enterprise + regulated/gov FDEs** whose killer
property is **policy-as-code**: the decision logic lives in versioned, testable,
publishable DMN tables that the AI literally cannot reach around; the
composition graph is the audit trail; and each node — *including the policy
tables* — is testable with HoloDeck's existing eval framework. It turns
HoloDeck's evaluator into a policy-testing tool, which none of LangSmith /
MLflow / PromptFlow offer.

## Anchor Sample: Loan Hardship Underwriting

Chosen because it ties to the existing `financial-assistant` sample and has
naturally tiered determinations. (Swappable to benefits-eligibility or KYC —
same YAML shape, different tables.)

| Level | Node | Type | Proves |
|---|---|---|---|
| 1 | Extract income & expenses from payslips/statements | AI edge → schema gate | Probabilistic work confined behind a typed gate |
| 1 | Verify residency/identity from documents | AI edge → schema gate | A second independent edge feeding upward |
| 1 | Flag document-fraud signals | AI edge → schema gate | A third edge feeding a different branch |
| 2 | **Affordability** decision (DMN `UNIQUE`, FEEL ranges/date math) | pure policy | Policy composing two AI inputs |
| 2 | **Risk tier** decision (DMN `FIRST`) | pure policy | A parallel sub-determination |
| 3 | **Final determination**: auto-approve / refer-to-officer / decline (DMN `PRIORITY` + HITL CLI prompt) | policy + human | Composition of two Level-2 verdicts + the shut gate, AI drafts reasons only |

The demo runs one applicant end-to-end, prints the composed determination, and
**re-runs the recorded inputs to produce a byte-identical trace** — the "prove
it" moment.

## Key Assumptions to Validate

- [ ] **FEEL is required and best embedded, not built.** — Validate by spiking a
      Python FEEL evaluator (SpiffWorkflow's DMN/FEEL, or a standalone FEEL lib)
      against the three sample tables, including a date-math and a range
      expression. Building FEEL from scratch is out of POC scope; if no embeddable
      evaluator covers the needed subset, narrow the FEEL surface and document it.
- [ ] **The gate is differentiated from "an agent calling a rules function."** —
      The defensible difference is the *reproducible audit record + composition
      graph + policy-level tests*, not the decision itself. Test: the POC must
      emit a replayable determination record and re-run it identically.
- [ ] **A thin gate runtime is enough — no orchestration engine needed.** — Test:
      the determination graph executes purely as a DAG topological evaluation
      (resolve Level 1 edges → evaluate Level 2 tables → evaluate Level 3),
      without sequence flows, events, or BPMN gateways.
- [ ] **HITL as a CLI prompt satisfies the bright-line demo.** — Test: a
      `requires_human` node pauses, presents the AI-drafted reasons + composed
      inputs, records the human's choice into the trace, and refuses to let AI
      output cross.
- [ ] **Schema gates can reuse `ExecutionResult.structured_output` + Pydantic.** —
      Test: bind an agent's structured output to a JSON-Schema gate and reject
      free text / low-confidence fields at the boundary.

## MVP Scope

**In:**
- Native YAML DSL: `determination.yaml` (graph) + `tables/*.dmn.yaml` (decision
  tables) + `schemas/*.json` (gates).
- A gate primitive: schema-validate a typed input → evaluate a DMN table → emit a
  typed verdict + audit record.
- DMN evaluation with **FEEL expressions** (embedded evaluator) and the four hit
  policies used by the sample: `UNIQUE`, `FIRST`, `PRIORITY`, `COLLECT`.
- DAG composition: a node's verdict is a named input to nodes above it.
- `requires_human` nodes via CLI prompt; `ai_may_draft: reasons` confines AI to
  drafting.
- A reproducible determination record + a `--replay` that re-runs recorded inputs
  identically.
- The loan-hardship sample with three Level-1 agents, two Level-2 tables, one
  Level-3 determination.
- Policy-level tests: HoloDeck test cases asserting against decision-table outputs.

**Out (POC):**
- BPMN/DMN XML import, visual modeling, sequence flows, events, BPMN gateways.
- A task-inbox UI / web HITL (CLI only).
- Deploy-as-API for determination graphs (`holodeck serve`/`deploy` integration).
- Versioned policy registry, compliance export, replay UI.

## Not Doing (and Why)

- **No BPMN 2.0 XML in v1** — verbose, tool-generated, breaks the no-code/YAML
  ethos, and is a large spec surface we don't control. The YAML is designed so an
  importer can transpile into it later.
- **No general workflow/orchestration engine** — determination is a DAG of gates,
  not a process with events, timers, and message flows. Coupling to HoloDeck's
  still-aspirational `experiment.yaml` orchestration would inherit vapor.
- **No FEEL built from scratch** — embed an existing evaluator; FEEL is a full
  expression language and re-implementing it would consume the whole POC.
- **No human-task UI** — a CLI prompt proves the bright line; the inbox is a
  flagship-tier concern, not a POC concern.
- **No emergent/agentic determination** — by definition the AI is never the
  spine. Anything that lets the model decide is off-pattern.

## Open Questions

- Which Python FEEL evaluator do we embed, and what FEEL subset do the real
  tables actually need (ranges, dates, `contains`, list functions)?
- How is the audit/determination record represented — extend OpenTelemetry GenAI
  spans, or a separate signed determination-record artifact (closer to the
  regulated "prove it" requirement)?
- Where does the schema gate live relative to `ExecutionResult.structured_output`
  — is it a new `gate` config on a node, or a reusable wrapper any agent can
  declare?
- Do Level-2 pure-policy nodes ever need to *call back* for more AI input
  (re-extraction on low confidence), or is the DAG strictly one-directional in v1?
  (Recommend strictly one-directional for the POC.)

## North Star (named, not built)

"**Governable Agents**" — visual modeler, versioned policy registry, replay UI,
compliance export, and a real HITL task inbox. This is HoloDeck's governance
positioning against competitors that explicitly lack it. The POC is the first
vertebra of that spine.
