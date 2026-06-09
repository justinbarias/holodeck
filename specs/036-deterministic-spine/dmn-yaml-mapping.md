# DMN ↔ HoloDeck YAML Mapping

> Companion artifact to `spec.md` / `refinements.md` / `tasks/plan.md`.
> Status: **design input** for T2 (workflow models) and T3 (table model) —
> illustrative, not normative until those tasks land. Produced 2026-06-10
> from a design discussion on how the native YAML DSL relates to the OMG DMN
> standard.

## The honest claim

The v1 artifacts carry **DMN semantics, not the DMN interchange format**:

- **Standard-faithful:** FEEL expression syntax (refinements §5: FEEL syntax
  is the contract), hit-policy semantics (`UNIQUE`/`FIRST`/`PRIORITY`),
  decision-table structure (input expressions, unary-test cells, output
  values), DRD composition (decisions requiring decisions and input data,
  acyclic by validation).
- **Not the standard artifact:** no DMN 1.x XML import/export, no DMNDI
  (diagram geometry), no boxed expressions. Spec Out of Scope, by design.
- **Pitch wording:** "DMN-semantics policy-as-code" — not "DMN-compatible"
  and not "bring your existing DMN files" (those need the deferred XML
  importer; see Migration path).

## Division of labor

| File | DMN concept it carries |
|---|---|
| `workflow.yaml` | The **DRD** (logical form): decisions, input data, information requirements |
| `tables/*.dmn.yaml` | One **decision table** each (`<decision>` + `<decisionTable>`) |
| `schemas/*.json` (gates) | **`<itemDefinition>`** — the type of each input-data fact |

## Decision table mapping (`tables/*.dmn.yaml`)

Example — the Level-2 affordability table:

```yaml
# DMN: <decision> + <decisionTable>
id: affordability
name: Hardship affordability assessment
version: "2026-06-01.1"          # FR-013: required label, snapshotted into the run record
hit_policy: UNIQUE               # DMN: hitPolicy — exactly one rule may match

# DMN: <input>/<inputExpression> — full FEEL expressions live HERE only.
# `income` / `residency` are workflow node ids; dot-paths reach into the
# gated object (edge node) or verdict (policy node).
inputs:
  - name: surplus_ratio
    expression: (income.net_monthly_income - income.monthly_expenses) / income.net_monthly_income
    type: number
  - name: statement_age
    expression: date(application_date) - date(income.statement_date)   # FEEL date difference
    type: days
  - name: residency_status
    expression: residency.status
    type: string

# DMN: <output>; `values` = outputValues (and the priority order under PRIORITY)
outputs:
  - name: affordability
    type: string
    values: [affordable, marginal, unaffordable]

# DMN: <rule> — each `when` cell is a FEEL unary test against the input in
# the same position; "-" is the irrelevant cell, exactly as in DMN.
rules:
  - when: [">= 0.25",      "<= 90", '"verified"']
    then: { affordability: affordable }
    annotation: Comfortable surplus, fresh statements, verified residency
  - when: ["[0.10..0.25)", "<= 90", '"verified"']
    then: { affordability: marginal }
    annotation: Thin surplus — refer for officer review
  - when: ["< 0.10",       "-",     "-"]
    then: { affordability: unaffordable }
    annotation: No realistic repayment capacity
  - when: ["-",            "> 90",  "-"]
    then: { affordability: unaffordable }
    annotation: Statements too stale to assess

# DMN: defaultOutputEntry — without it, no-match fails loudly (FR-012)
# default: { affordability: unaffordable }
```

Element-by-element:

| `*.dmn.yaml` | DMN element | Notes |
|---|---|---|
| `id`, `name` | `<decision id name>` | |
| `version` | (no direct DMN equivalent) | HoloDeck addition; FR-013 snapshot key |
| `hit_policy` | `hitPolicy` | UNIQUE multi-match errors per FR-012; PRIORITY resolves by `values` order (DMN's output-values-as-priority rule) |
| `inputs[].expression` | `<inputExpression>` (FEEL) | Full FEEL allowed here only |
| `inputs[].type` | `typeRef` | |
| `rules[].when[i]` | `<inputEntry>` (FEEL unary test) | `>= 0.25`, `[0.10..0.25)`, `"verified"`, `-` |
| `rules[].then` | `<outputEntry>` | |
| `rules[].annotation` | rule annotation | |
| `outputs[].values` | `outputValues` | Doubles as PRIORITY ordering, highest first |
| `default` | `defaultOutputEntry` | Optional; absence ⇒ loud no-match failure |

### Open design points for T3

1. **Positional vs keyed `when` cells.** Positional mirrors DMN's column
   layout (and is what a transpiler would emit); keyed
   (`when: {surplus_ratio: ">= 0.25", ...}`) is diff-friendlier and immune to
   column-reorder bugs. Lean: **keyed for hand-authored YAML**, accept
   positional from a future transpiler. T3 decides.
2. **Expression placement rule.** Full FEEL (arithmetic, date math) is valid
   only in `inputs[].expression`; rule cells are restricted to unary tests.
   Faithful to DMN and matches what the FEEL-library research verified best
   (bkflow-feel). Encode as a T3 validation rule.

## DRD mapping (`workflow.yaml`)

The workflow file **is** the DRD in logical form:

```yaml
name: loan-hardship-underwriting
version: "2026-06-01.1"

nodes:
  # DMN <inputData> — produced by an agent; the gate schema is the
  # <itemDefinition> (the type of the fact).
  - id: income
    edge: { agent: agents/income-extractor/agent.yaml }
    gate: { schema: schemas/income.json }
  - id: residency
    edge: { agent: agents/residency-verifier/agent.yaml }
    gate: { schema: schemas/residency.json }
  - id: doc_fraud_flag
    edge: { agent: agents/doc-fraud-detector/agent.yaml }
    gate: { schema: schemas/fraud-flag.json }

  # DMN <decision> — `inputs:` is the <informationRequirement> list.
  - id: affordability
    decision: tables/affordability.dmn.yaml
    inputs: [income, residency]          # requiredInput ×2
    hit_policy: UNIQUE
  - id: risk_tier
    decision: tables/risk.dmn.yaml
    inputs: [doc_fraud_flag, income]
    hit_policy: FIRST

  - id: final_determination
    decision: tables/determination.dmn.yaml
    inputs: [affordability, risk_tier]   # requiredDecision ×2
    hit_policy: PRIORITY
    requires_human: true
    decided_by: "Hardship Officer"
    draft: { agent: agents/reasons-drafter/agent.yaml }
    ai_may_draft: [reasons]
```

| DMN DRD element | HoloDeck equivalent |
|---|---|
| `<decision>` | policy or human node (one table each) |
| `<inputData>` | edge node's gated output |
| `<itemDefinition>` | the gate's JSON Schema |
| `<informationRequirement>` (`requiredInput` / `requiredDecision`) | the `inputs:` list (kind implicit — resolved by what the id names) |
| Acyclic requirements graph (DMN mandate) | FR-003 load-time cycle rejection — same constraint |
| `<knowledgeSource>` | **not modeled** — see proposal below |
| `<businessKnowledgeModel>` (reusable invocable logic) | **not modeled** — every table is bound to exactly one node |
| DMNDI (diagram layout) | **not modeled** — logical DRD only; visual modeler is North Star |

### Behavioral deltas from standard DRD

- DMN allows any decision as an evaluation entry point; v1 always evaluates
  the full DAG to the top. Evaluating one table in isolation is the policy
  test executor's job (T12).
- One `workflow.yaml` holds exactly one DRD; a DMN definitions file may hold
  several sharing elements.

## Proposal: `source:` annotation (knowledgeSource-lite)

`<knowledgeSource>` is DMN's record of the *authority* for a decision —
"this table implements §72 of the Hardship Policy." For the regulated
audience that traceability link is substance, not decoration, and it costs
one optional field:

```yaml
# on a node or table
source: "Hardship Policy v4.2 §72"
```

- Non-executable annotation; flows into the run record and OTel span
  attributes.
- Recommended for T2/T3 model fields (one-line schema addition now vs. a
  migration later). A future "draft table from policy document" feature
  would populate it automatically.

## Authoring routes (where tables come from)

1. **Hand-authored YAML** — the only route in v1; guarded by the published
   JSON schema (T2), load-time FEEL validation with table/rule/cell locators
   (T3), and policy tests (T12).
2. **Imported from DMN XML** — deferred but architecturally preserved: a
   transpiler walks `<decisionTable>` and emits exactly the shape above
   (decision tables only; no DRD-XML, BKM, or boxed expressions). Likely
   small once T3's model exists, if "bring your existing DMN" becomes a
   sales requirement.
3. **AI-drafted from a policy document** — not in any spec yet. On-pattern
   (AI drafts, human reviews/versions/commits — same move as
   `ai_may_draft: reasons`; the committed table stays deterministic), and
   US5 policy tests are exactly how a drafted table would be verified.
   Belongs in North Star / a follow-up spec, possibly dogfooded as a
   HoloDeck agent whose `response_format` is the table schema.
