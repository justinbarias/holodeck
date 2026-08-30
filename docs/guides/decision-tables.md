# Decision Tables

HoloDeck decision tables are deterministic policy-as-code. A table is a
versioned `.dmn.yaml` file. It converts named facts into one auditable verdict.
The decision does not call an LLM.

Use a table when policy must stay separate from agent interpretation. An agent
can extract a gate-validated object, and a table can decide what that object
means. The table ID, version, matched rule, and outputs remain available in the
result.

## Table format

This hardship table is the running example. It calculates a surplus ratio,
checks residency, and emits an affordability decision.

```yaml
# tables/hardship.dmn.yaml
id: affordability                    # Stable policy identifier
version: 2026-06-01.1                # Required version label
hit_policy: UNIQUE                   # More than one match is an error

# Full S-FEEL expressions calculate the input columns from named facts.
inputs:
  - name: surplus_ratio
    expression: (income.net - income.expenses) / income.net
    type: number
  - name: residency_status
    expression: residency.status
    type: string

# values constrains the output. For PRIORITY, its order is highest first.
outputs:
  - name: affordability
    type: string
    values:
      - affordable
      - marginal
      - unaffordable

# Each when entry is a unary test against its named input column.
rules:
  - when:
      surplus_ratio: '>= 0.25'
      residency_status: '"verified"'
    then:
      affordability: affordable
    annotation: Comfortable surplus
  - when:
      surplus_ratio: '[0.10..0.25)'
    then:
      affordability: marginal
  - when:
      surplus_ratio: < 0.10
    then:
      affordability: unaffordable
```

The second and third rules omit `residency_status`. An omitted input is an
irrelevant cell and always matches. An explicit `-` has the same meaning.
Every cell present in one rule must match for that rule to match.

### Field reference

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Table identifier copied to each verdict. |
| `name` | no | Human-readable table name. |
| `version` | yes | Version label copied to each verdict. |
| `hit_policy` | yes | `UNIQUE`, `FIRST`, or `PRIORITY`. |
| `inputs` | yes | One or more calculated input columns. |
| `inputs[].name` | yes | Column name used by `rules[].when`. |
| `inputs[].expression` | yes | Full S-FEEL expression over the evaluation context. |
| `inputs[].type` | yes | `number`, `string`, `boolean`, `days`, or `date`. |
| `outputs` | yes | One or more fields emitted in `Verdict.outputs`. |
| `outputs[].name` | yes | Output name used by every `then` and `default`. |
| `outputs[].type` | yes | Declared output type. |
| `outputs[].values` | no | Allowed strings; also the ranking order for `PRIORITY`. |
| `rules` | yes | One or more rules, in document order. |
| `rules[].when` | no | Unary tests keyed by input name. An empty map matches all inputs. |
| `rules[].then` | yes | One value for every declared output. |
| `rules[].annotation` | no | Human-readable reason copied to a matched verdict. |
| `default` | no | Complete output entry used when no rule matches. |
| `source` | no | Policy authority annotation. It does not affect evaluation. |
| `provenance` | no | Non-executable authorship and review metadata. |

All model objects reject unknown fields. Input and output names must be unique.
Each `when` key must name an input. Each `then` and `default` must contain
exactly the declared outputs. If an output has `values`, every emitted value
must be in that list.

!!! note "Type enforcement"
    A `days` input accepts a number or a date difference and converts it to
    floating-point days. A `number` input rejects booleans. Other input types
    are declarative until a rule evaluates them, and output types are not
    checked at evaluation time. Use `values` when an output has a fixed set.

The optional `provenance` fields are `generated_by`, `source`, `source_doc`,
`source_sha256`, `reviewed_by`, and `reviewed_at`. S-FEEL cannot read
`provenance`, so authorship metadata cannot change a decision.

## Input expressions and S-FEEL

HoloDeck supports a bounded subset of S-FEEL. Full expressions belong in
`inputs[].expression`. Rule cells contain unary tests against the calculated
value of one input column.

### Supported forms

| Form | Example | Where |
| --- | --- | --- |
| Names and dot paths | `income.net`, `residency.status` | input expression |
| Number, string, boolean, and null literals | `12.5`, `"verified"`, `true`, `null` | input expression or unary test |
| Arithmetic | `(income.net - income.expenses) / income.net` | input expression |
| Comparisons | `ratio >= 0.25`, `status = "verified"` | input expression |
| Boolean logic | `a > 1 and b <= 90`, `not(flag)` | input expression |
| List membership | `tier in ["low", "medium"]` | input expression |
| Ranges | `ratio in [0.10..0.25)` | input expression |
| Date literals | `date("2026-06-01")` | input expression |
| Date comparison and subtraction | `application_date - statement_date` | input expression |
| Irrelevant cell | `-` | unary test |
| Unary comparison | `>= 0.25`, `!= 0`, `= true` | unary test |
| Unary range | `[0.10..0.25)`, `(0.10..0.25]` | unary test |
| Unary equality | `"verified"`, `5` | unary test |
| Unary negation | `not("verified")`, `not([1..10])` | unary test |

Ranges support all four endpoint combinations: `[a..b]`, `[a..b)`, `(a..b]`,
and `(a..b)`. Square brackets include the endpoint. Parentheses exclude it.
Negative numbers are valid in expressions, comparisons, and ranges.

String cells need quotes inside the YAML string. For example,
`residency_status: '"verified"'` compares the input with the string
`verified`. An unquoted cell such as `verified` is treated as a variable and
is rejected.

`date("2026-06-01")` accepts a quoted date literal only. Do not write
`date(application_date)`. Supply date values in the context and subtract the
bare names. A `days` input converts the resulting `datetime.timedelta` to a
number of days.

### Rejected forms

The loader parses every expression and walks an allowlist before the table can
run. It rejects these forms:

- Quantifiers, such as `some x in [1, 2] satisfies x > 1`.
- Conditionals and iteration, such as `if ... then ... else` and
  `for ... return`.
- Functions outside the subset, such as `duration("P30D")`.
- Comma-separated unary tests, including `not("a", "b")`.
- A rule cell that reads another variable, such as `>= threshold`.
- Any expression or cell that reads the reserved `provenance` root.
- Malformed syntax.

This allowlist is intentional. The embedded grammar accepts some unsupported
constructs, and unknown functions can return `None` without an error. Static
rejection prevents an unsupported expression from producing a plausible but
incorrect verdict.

At evaluation time, every referenced name and dot path must resolve. Missing
names, missing attributes, division by zero, and incompatible types fail with
a table, rule, or cell locator. They do not fall through to `default`.

## Hit policies

The hit policy says how HoloDeck resolves overlapping rules. Assume these
three rules all match the same context. The output declares
`values: [unaffordable, marginal, affordable]`, in highest-first order:

| Rule order | Output |
| --- | --- |
| 1 | `affordable` |
| 2 | `marginal` |
| 3 | `unaffordable` |

| Policy | Result |
| --- | --- |
| `UNIQUE` | Raises `TableEvalError` because rules 1, 2, and 3 all matched. |
| `FIRST` | Returns rule 1 because it appears first in the file. Later rules are not evaluated. |
| `PRIORITY` | Returns rule 3 because `unaffordable` ranks first, regardless of rule order. |

For the running example, `values` ranks `affordable` above `marginal`, then
`unaffordable`. Reverse that list if `unaffordable` must have the highest
priority. Every output in a `PRIORITY` table must declare `values`.

With multiple outputs, HoloDeck compares their priority ranks from left to
right. The first declared output dominates. Later outputs break ties. Document
order breaks a complete tie.

!!! warning "PRIORITY outputs are string-valued"
    `outputs[].values` is a list of strings. Therefore, `PRIORITY` currently
    supports ranked string output values. Use `UNIQUE` or `FIRST` for an output
    that cannot use a string ranking.

## Load and evaluate a table

Load the file once, then pass the table and named context to `evaluate()`:

```python
from pathlib import Path
from typing import Any

from holodeck.lib.workflow.table_eval import Verdict, evaluate
from holodeck.models.decision_table import DecisionTable, load_decision_table

TABLE_PATH = Path(__file__).parent / "tables" / "hardship.dmn.yaml"
TABLE: DecisionTable = load_decision_table(TABLE_PATH)


def decide(context: dict[str, Any]) -> Verdict:
    return evaluate(TABLE, context)


verdict = decide(
    {
        "income": {"net": 5000, "expenses": 3000},
        "residency": {"status": "verified"},
    }
)

assert verdict.outputs == {"affordability": "affordable"}
assert verdict.rule_identity == "rule 1"
```

`load_decision_table()` reads YAML with the safe loader and validates the full
model. Read errors, invalid YAML, non-mapping YAML, and structural errors raise
`DecisionTableError`. Unsupported S-FEEL raises `FeelValidationError`. Basic
field-shape errors raise Pydantic `ValidationError`.

`evaluate(table, context)` calculates every input expression once. It then
tests the rules and applies the hit policy. It returns one `Verdict`:

| Field | Meaning |
| --- | --- |
| `table_id` | ID of the table that made the decision. |
| `table_version` | Version of that table. |
| `outputs` | Complete output mapping from the winning rule or default. |
| `matched_rule_index` | One-based rule index, or `None` for a default. |
| `matched_rule_annotation` | Winning rule annotation, if present. |
| `is_default` | `True` only when no rule matched and `default` was used. |
| `rule_identity` | `rule N`, or `default, no rule matched`. |

`rule_identity` is a convenience property. It is not included in
`model_dump()`. The other fields can be stored as the decision record.

### Default outcomes

Add a complete `default` output when no-match is a valid policy outcome:

```yaml
default:
  affordability: unaffordable
```

When no rule matches, the verdict has `is_default: true`, no matched rule
index, and no annotation. If the table has no `default`, every hit policy
raises `TableEvalError`. HoloDeck never invents an outcome.

## Use a table with an agent workflow

Run a table on gate-validated agent output. In the hardship workflow, the gate
shape is also the table context shape:

```python
evidence = await workflow.execute_activity(...)
verdict = evaluate(TABLE, evidence.output)
```

The gate requires `income` and `residency`, which are the roots used by the
table input expressions. No mapping layer is necessary. This
**gate-shape-equals-table-input-shape** pattern makes the policy boundary clear
and keeps invalid model output away from the decision.

For the complete activity, sandbox, and sibling-module loading pattern, see
the [Temporal Integration guide](temporal.md#keep-workflow-code-deterministic).
Temporal requires the table load in a passed-through sibling module so replay
does not perform file I/O. That rule is specific to Temporal; the table format
and evaluator are not.
