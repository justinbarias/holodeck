# Research: FEEL Evaluator Spike (T1, Phase 0)

> **ARCHIVED as a spec, RETAINED as rationale (2026-08-29).** 036 is
> archived (see `spec.md`), but the FEEL evaluator and decision-table
> caveats recorded here still govern the kept primitives in
> `holodeck.lib.workflow` and are cited from their docstrings.


> Resolves the FR-010 NEEDS CLARIFICATION and refinements §5 ("FEEL syntax is
> the contract"). Produced 2026-06-11 from a hands-on spike on branch
> `036-deterministic-spine`. Behavior pinned by
> `tests/unit/workflow/test_feel_conformance.py` (40 tests, all passing).

## Decision

**Embed [bkflow-feel](https://github.com/TencentBlueKing/bkflow-feel) 1.2.0
(pinned).** MIT-licensed, lark-grammar FEEL parser from TencentBlueKing.
It covers the full FR-010 subset — no subset narrowing needed (the
refinements §5 fallback posture goes unused). The core spike risk — FEEL
date difference — is retired: `date("2026-01-31") - date("2026-01-01")`
evaluates to `timedelta(days=30)`.

### Candidates considered

| Library | Verdict | Why |
|---|---|---|
| **bkflow-feel 1.2.0** | **chosen** | MIT; full FR-010 coverage verified; active org (TencentBlueKing); lark grammar |
| pySFeel | fallback, unused | GPLv3 (viral — incompatible with embedding in an MIT-adjacent product without license review); dormant |
| SpiffWorkflow (DMN module) | ruled out | No real FEEL parser — Python-expression translation; hit policies limited to UNIQUE/COLLECT |

License note: bkflow-feel's PyPI metadata has an empty `license` field, but
the repository carries an MIT LICENSE file and the project is published by
TencentBlueKing under their standard MIT terms.

## FR-010 subset verdicts

All verified against bkflow-feel **1.2.0** (the spike initially probed
1.2.1rc2; behavior matched except where noted).

| FR-010 feature | Verdict | Notes |
|---|---|---|
| Numeric comparisons (`<` `<=` `>` `>=` `=` `!=`) | ✅ pass | including arithmetic operands: `(a - b) / a >= 0.25` |
| Ranges, all four brackets `[a..b]` `[a..b)` `(a..b]` `(a..b)` | ✅ pass | endpoint inclusion/exclusion exactly per FEEL |
| `and` / `or` / `not(..)` | ✅ pass | |
| String equality | ✅ pass | `status = "verified"` with context vars |
| List membership `x in [..]` | ✅ pass | strings and numbers |
| Date literals `date("YYYY-MM-DD")` | ✅ pass | |
| Date comparison | ✅ pass | including date-in-date-range |
| Date difference | ✅ pass* | returns `datetime.timedelta` — see caveat 1 |
| Dot-path context access | ✅ pass | `income.net_monthly_income` into nested dicts — how composed node outputs flow into input expressions |

## Caveats (design inputs for T3)

1. **Date difference yields a timedelta, which is not number-comparable.**
   `(date(..) - date(..)) > 30` raises `ValidationError` ("Type of both
   operators must be same"). → The T3 FEEL wrapper must convert
   timedelta-valued input expressions to **days (number)** before rule cells
   compare them. This also matches the `type: days` column in
   `dmn-yaml-mapping.md`.
2. **No duration literals, and unknown functions fail silently.**
   `duration("P30D")` has no builtin; in 1.2.0 an unknown function call
   returns `None` **silently** (the internal `ValueError` is caught and
   logged, not raised). → T3's static validation must reject
   non-allowlisted function names itself; runtime gives nothing to catch
   until the `None` poisons a comparison.
3. **The grammar accepts out-of-subset FEEL — static rejection needs its own
   allowlist.** `some x in [1, 2] satisfies x > 1` parses and evaluates fine;
   `for .. return` happens to be rejected at parse time. The parser is not a
   subset gate. → T3 must walk the lark parse tree (or token stream) and
   enforce an allowlist of node types.
4. **No unknown-variable error.** A missing context variable resolves to
   `None` and fails later as a type mismatch (`ValidationError`, NoneType vs
   int) — there is no dedicated unknown-variable diagnostic. → T3's wrapper
   must verify required inputs are present **before** evaluation so failures
   carry the table/rule/cell locator (FR-012 loud failures).
   **Partly resolved (T3), completed later:** the first fix extracted the
   referenced *root* names from the parse tree and raised
   `FeelEvaluationError(locator, ..)` for any root absent from the context.
   That covered a typo'd root but not a missing **attribute** under a bound
   root, which `ContextItem.evaluate` also resolves to `None` — and because
   `NotEqual` has no operand validator (it is a bare Python `!=`), a `!=` rule
   cell then matched *unconditionally* and the table emitted a real-looking
   verdict. `feel.evaluate_expression` now resolves every full dot-path
   against the bindings before evaluating, and rejects three distinct cases:
   an unbound root, an attribute absent from a bound mapping, and an attribute
   read through a value that is not a `dict` (which can never resolve). A leaf
   explicitly bound to `None` is still a bound fact, as before — presence, not
   truthiness, is the test.

   One route could still defeat the guard: a fact named `true`, `false` or
   `null` parses as a FEEL *literal*, never as a variable, so the binding was
   invisible to both the expression and the check. Those three names are now
   rejected as fact names when a context is evaluated. That check is at
   evaluation time, not load time, because the workflow model — not the FEEL
   wrapper — owns node/fact names.
5. **Error taxonomy for the wrapper:** syntax errors surface as
   `lark.exceptions.UnexpectedInput` (parse time); type/value errors as
   `bkflow_feel.exceptions.ValidationError` (eval time). Both must be caught
   and re-raised through `holodeck.lib.errors` with locators.
   **Resolved, and wider than stated:** those are not the only two channels.
   `bkflow_feel.api.parse_expression` re-raises *any* other exception
   unchanged, and two are reachable from schema-valid data — a
   `ZeroDivisionError` from a ratio expression with a zero denominator, and a
   bare `TypeError` from range cells (`In`/`RangeGroup` bypass the operand
   validator and compare with raw Python). The wrapper therefore catches
   broadly and re-raises everything as `FeelEvaluationError` with the locator,
   preserving the cause.

7. **Numeric comparison is exact-typed and asymmetric.**
   `BinaryOperationValidator` is `isinstance(left, type(right))`, so a `float`
   column value refuses an `int` cell literal and vice versa — a monetary
   column with cents against a whole-number threshold (`<= 2000`) loads fine
   and fails at runtime on the first fractional value. Ranges bypass the
   validator entirely, so `[0..90]` and `<= 90` disagreed on the same input.
   → Unary tests are evaluated with both sides widened to `float`: the column
   value in `feel.evaluate_unary_test`, and every numeric literal via a
   `FEELTransformer` subclass. `bool` is deliberately not widened (it must
   still compare against `true`/`false`) and is rejected outright in `number`
   and `days` columns, where it would otherwise compare silently as 1/0.
   Full expressions keep stock literal typing — widening there would break
   arithmetic over `int`-typed facts for no gain.
6. **`date(..)` accepts only a quoted literal, not a variable.** `date(income.
   statement_date)` does **not** parse — the `date_func` production wants a
   string literal (`date("2026-01-01")`), so the `dmn-yaml-mapping.md` example
   `date(application_date) - date(income.statement_date)` is unevaluable as
   written. → Date-typed inputs are supplied to the FEEL context as Python
   `datetime.date` objects and subtracted directly (`application_date -
   income.statement_date`), which yields a `timedelta` (caveat 1). The
   gate-schema `format: date` fields become `date` objects at the workflow
   boundary; the table's `inputs[].expression` does bare subtraction, never
   `date(variable)`.

## Static-rejection list (input to T3)

Out-of-subset constructs T3 must reject at load time (grammar does NOT
reject them all):

- Quantified expressions: `some/every .. satisfies` (grammar **accepts**)
- Any function invocation outside the allowlist — initially `date(..)` only
  (grammar accepts unknown functions and returns silent `None`)
- `for .. return` (grammar rejects, but reject statically anyway for a
  uniform locator-bearing error)
- `if .. then .. else`, context literals `{..}`, `instance of`,
  `between` — to be probed/enumerated as T3 builds the allowlist walker;
  the posture is **allowlist, not blocklist**, so anything unprobed is
  rejected by default.

## Dependency findings

`bkflow-feel` declares `pytz<2024` (and `lark>=1.1.7,<2`, `python-dateutil<3`,
`pydantic<3`). The pytz pin is overly conservative — pytz's API is stable —
and a naive `uv add` downgraded repo-wide pytz 2025.2→2023.4 and dragged
optuna 4.9.0→4.8.0.

**Resolution (applied):**

- Pinned `bkflow-feel==1.2.0` (stable; uv had selected pre-release 1.2.1rc2
  because the repo allows prereleases for semantic-kernel).
- Added `[tool.uv] override-dependencies = ["pytz>=2024"]` to neutralize the
  pin. Restored: pytz 2026.2, optuna 4.9.0.
- Compatibility of bkflow-feel 1.2.0 with modern pytz is verified by the
  conformance suite (date functions exercise the tz-touching code paths).

**Risk register:** bkflow-feel's release cadence is slow and 1.2.1 has sat
in rc since publication. The pin + conformance suite make upgrades
deliberate: bump, run `tests/unit/workflow/test_feel_conformance.py`, read
the diff. MIT license permits vendoring if the project goes dormant.

## Hit-policy evaluation: in-house (T4)

bkflow-feel is an **expression** evaluator only — it has no decision-table
or hit-policy layer (its sibling project bkflow-dmn does, but it brings a
different table format and more dependency surface). UNIQUE/FIRST/PRIORITY
over already-evaluated unary tests is ~50 lines of well-specified logic with
SC-004 conformance tests planned in T4. **Decision: implement hit policies
in-house in T4; use bkflow-feel strictly for expressions and unary tests.**
