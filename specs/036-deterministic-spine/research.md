# Research: FEEL Evaluator Spike (T1, Phase 0)

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
5. **Error taxonomy for the wrapper:** syntax errors surface as
   `lark.exceptions.UnexpectedInput` (parse time); type/value errors as
   `bkflow_feel.exceptions.ValidationError` (eval time). Both must be caught
   and re-raised through `holodeck.lib.errors` with locators.

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
