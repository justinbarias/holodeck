# DMN ↔ HoloDeck YAML Mapping

> Companion artifact to `spec.md` / `refinements.md` / `tasks/plan.md`.
> Status: **design input** for T2 (workflow models) and T3 (table model) —
> illustrative, not normative until those tasks land. Produced 2026-06-10
> from a design discussion on how the native YAML DSL relates to the OMG DMN
> standard.
>
> **Note (2026-07-25):** the worked examples below still use the original
> loan-hardship scenario. They are retained **as DMN-mapping illustrations only** —
> the anchor sample has since moved to the Targeted Compliance Framework
> (`spec.md` US6). The element-by-element mappings, the DRD semantics, and the
> `source:`/`input_data` rows are all current; only the domain in the examples
> is stale. The `date(..)` expression on line ~51 has been corrected in place.

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
  # CORRECTED 2026-07-25 — was `date(application_date) - date(income.statement_date)`,
  # which research.md caveat 6 proved unevaluable: bkflow-feel's `date_func`
  # production accepts only a quoted literal, not a variable. Date-typed gate
  # fields cross the boundary as Python `datetime.date` and subtract bare;
  # caveat 1 requires the wrapper to convert the resulting timedelta to days.
  - name: statement_age
    expression: application_date - income.statement_date
    type: days
  - name: residency_status
    expression: residency.status
    type: string

# DMN: <output>; `values` = outputValues (and the priority order under PRIORITY)
outputs:
  - name: affordability
    type: string
    values: [affordable, marginal, unaffordable]

# DMN: <rule> — each `when` cell is a FEEL unary test, keyed by input name.
# An input omitted from `when` is the irrelevant cell (always matches), which
# is DMN's "-"; an explicit "-" is accepted and means the same thing.
# CORRECTED 2026-07-25 — this example was positional (`when: [">= 0.25", ...]`)
# until T3 landed. Open design point 1 below resolved to keyed cells, so the
# positional form no longer parses. See `models/decision_table.py::Rule`.
rules:
  - when: { surplus_ratio: ">= 0.25", statement_age: "<= 90", residency_status: '"verified"' }
    then: { affordability: affordable }
    annotation: Comfortable surplus, fresh statements, verified residency
  - when: { surplus_ratio: "[0.10..0.25)", statement_age: "<= 90", residency_status: '"verified"' }
    then: { affordability: marginal }
    annotation: Thin surplus — refer for officer review
  - when: { surplus_ratio: "< 0.10" }
    then: { affordability: unaffordable }
    annotation: No realistic repayment capacity
  - when: { statement_age: "> 90" }
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
| `rules[].when[<input name>]` | `<inputEntry>` (FEEL unary test) | `>= 0.25`, `[0.10..0.25)`, `"verified"`; an omitted key (or `-`) is the irrelevant cell |
| `rules[].then` | `<outputEntry>` | |
| `rules[].annotation` | rule annotation | |
| `outputs[].values` | `outputValues` | Doubles as PRIORITY ordering, highest first |
| `default` | `defaultOutputEntry` | Optional; absence ⇒ loud no-match failure |

### Open design points for T3

1. ~~**Positional vs keyed `when` cells.**~~ **RESOLVED in T3 — keyed only.**
   Positional mirrors DMN's column layout, but keyed cells are diff-friendlier
   and immune to column-reorder bugs, and a reordered column silently changing
   every rule's meaning is exactly the class of error this spec exists to
   prevent. `Rule.when` is a `dict[str, str]` keyed by input name; a cell
   naming an undeclared input is rejected at load. Positional entries do not
   parse. A future transpiler (spec 039) emits keyed cells.
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
  # CORRECTED 2026-07-25 — `hit_policy:` was shown on each node until the T3
  # refactor moved it onto the referenced table, where DMN puts it (one table,
  # one hit policy, one source of truth). `PolicyNode` has no such field and
  # forbids extras, so the earlier form no longer parses.
  - id: affordability
    decision: tables/affordability.dmn.yaml   # the table declares UNIQUE
    inputs: [income, residency]          # requiredInput ×2
  - id: risk_tier
    decision: tables/risk.dmn.yaml            # the table declares FIRST
    inputs: [doc_fraud_flag, income]

  - id: final_determination
    decision: tables/determination.dmn.yaml   # the table declares PRIORITY
    inputs: [affordability, risk_tier]   # requiredDecision ×2
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
| `<inputData>` **not agent-produced** | the workflow-level `input_data:` block (facts of record — prior state, case facts). Distinct from an edge node's gated output, which *is* agent-produced. |
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
3. **AI-drafted from a policy document** — **now specced: `specs/039-policy-generator`**
   (was "not in any spec yet"). Emits tables *and* the `workflow.yaml` DRD, with
   TODO markers where edge nodes go; edge agents stay hand-authored. On-pattern
   (AI drafts, human reviews/versions/commits — same move as
   `ai_may_draft: reasons`; the committed table stays deterministic), and
   US5 policy tests are exactly how a drafted table is verified.

   036 ships only what makes a drafted table *safe to run*: the `provenance`
   block (T3a) and the review gate — `holodeck workflow run` refuses a table with
   `generated_by` and no `reviewed_by` (FR-030, SC-009). This extends "the LLM is
   never the spine" to authoring time: without it, a model could be the spine by
   writing the rules rather than by producing a verdict.

   039 is **built only after 036's MVP ships**. Its golden corpus is pinned at
   `specs/039-policy-generator/corpus-manifest.md`.
