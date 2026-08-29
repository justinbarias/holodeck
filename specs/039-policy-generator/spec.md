# Feature Specification: Policy Generator — AI-drafted decision tables and DRDs

> **BLOCKED ON THE PIVOT (2026-08).** This spec depends on 036, which was
> archived when HoloDeck pivoted to Temporal-first orchestration (see
> `specs/040-holodeck-temporal/spec.md`). The `workflow.schema.json`, `holodeck
> workflow run`, and DRD surfaces referenced below were removed with the 036
> overlay engine and no longer exist. The decision-table format and its
> loader survive; if this feature is revived it re-targets the D3 table-step
> design.

**Feature Branch**: `039-policy-generator`
**Created**: 2026-07-25
**Status**: Draft — **spec only, not scheduled for build**
**Author**: justinbarias (with Claude)
**Depends on**: `specs/036-deterministic-spine` — **built only after 036's MVP ships**
**Corpus**: `corpus-manifest.md` (pinned, hashed)

## Why this spec exists now, unbuilt

036's schema is soft *today* and hardens with every task after T4. Writing this
spec now — before building any of it — lets the generator's requirements land in
036 as **one-line additions** rather than migrations later. That is not
hypothetical: 036's `source:` annotation was added to T2 early for exactly this
reason, before any generator existed.

Two requirements have already been pushed back into 036 as a result:

| 036 task | What it adds | Why the generator needs it |
|---|---|---|
| **T3a** | `provenance:` block on `DecisionTable` | Without it the engine cannot tell an LLM-written table from a hand-written one |
| **T6** | Review gate — `run` refuses `generated_by` without `reviewed_by` | Closes the authoring-time hole in "the LLM is never the spine" |

**If nothing else from this spec is ever built, those two still earn their keep** —
they make 036 honest about a capability the ecosystem will add with or without us.

## Objective

Draft a **candidate** `workflow.yaml` DRD and its `tables/*.dmn.yaml` from a
policy document, for a human to review, correct, and commit. The generator never
produces a runnable determination on its own: 036's review gate refuses to
execute what it emits until a person has signed it.

**In scope:** decision tables, the DRD that composes them, and `TODO` markers
where edge nodes must be hand-authored.

**Out of scope:** generating edge agents, gate schemas, or `agent.yaml` files.
The generator marks where they go; a human writes them. Inventing the type system
*and* the prompts is a different, much larger problem.

## The one rule, extended

036's invariant is *the LLM is never the spine* — the model feeds typed inputs,
never verdicts. A generator moves the model **upstream of the rules themselves**.
The invariant survives only if generated policy cannot execute unreviewed:

> **AI may draft policy. AI may never enact policy.**

Enforced in 036 by FR-030/SC-009, not here — the engine is the right place for it,
and it must hold whether or not this generator is the thing that wrote the table.

## User Scenarios

### User Story 1 — Draft tables from a policy section (Priority: P1)

An FDE points the generator at a policy document section. It emits one or more
`tables/*.dmn.yaml` with `provenance.source` citing the clause, `generated_by`
set, and `reviewed_by` **absent**. The FDE reviews, corrects, adds their name as
reviewer, and commits.

**Independent Test**: run against a pinned corpus document; assert every emitted
table parses, validates against `workflow.schema.json`, passes T3's static FEEL
check, and is **refused** by `holodeck workflow run` until a reviewer is recorded.

### User Story 2 — Draft the DRD (Priority: P2)

The generator proposes the decomposition: which decisions exist, how they
compose, and where facts must enter. Edge nodes appear as `TODO` markers naming
the fact required and the clause requiring it.

### User Story 3 — Refuse to map discretion (Priority: P1)

Where a clause is not mechanizable — *"the Secretary is not satisfied that the
matter directly prevented…"* — the generator MUST emit an explicit unmapped
marker citing the clause, **not** an invented rule.

**This is the acceptance-critical behaviour.** A generator that quietly invents a
threshold for a discretionary clause produces a table that looks authoritative
and is wrong in the one way that matters. Silence here is worse than failure.

### User Story 4 — Fidelity evaluation (Priority: P1)

Generation accuracy is scored against a pinned golden corpus of policy documents
with hand-authored expected tables.

## Requirements

- **FR-001**: MUST emit `tables/*.dmn.yaml` conforming to 036's published schema, passing T3's static FEEL validation.
- **FR-002**: MUST populate `provenance`: `generated_by`, `source`, `source_doc`, `source_sha256`. MUST NOT populate `reviewed_by` — only a human does that.
- **FR-003**: MUST emit an explicit unmapped marker, citing the clause, for any provision it cannot express — never an invented rule (US3).
- **FR-004**: MUST emit a candidate `workflow.yaml` DRD with `TODO` markers where edge nodes and `input_data` are required (US2).
- **FR-005**: MUST NOT emit `agent.yaml` files or gate schemas.
- **FR-006**: MUST score against pinned corpus snapshots, never a live URL.
- **FR-007**: MUST report table fidelity as a number and decomposition quality qualitatively (see Open Question 1).

## Success Criteria

- **SC-001**: ≥80% rule-level match against the golden corpus. *(Threshold provisional — see Open Question 2.)*
- **SC-002**: **100%** of known-discretionary clauses produce an unmapped marker rather than a rule. No partial credit; this is a safety property, not a quality metric.
- **SC-003**: **100%** of emitted tables are refused by `holodeck workflow run` until reviewed (inherits 036 SC-009).
- **SC-004**: Every emitted table parses and passes static FEEL validation — a generated artifact is never merely *plausible*, it is always well-formed.

## Corpus

Pinned in `corpus/`, hashed in `corpus-manifest.md`. Domain: Australian
employment services compliance.

The anchor is the **Reasonable Excuse Determination 2018** (7pp), which contains
the full difficulty gradient in one document:

- **s5(2)(a)–(j)** — a closed enumerated list of ten factors → a *gate schema*, not a table
- **s5(3)** — nested sub-tests → mechanizable
- **s6(3)** — *"the Secretary is not satisfied… directly prevented"* → **must refuse to map** (SC-002)
- **s6(4)** — four conjunctive conditions defeated by any of four exceptions → the hardest positive case

A generator that drops s6(4)'s `unless` limb produces a table that wrongly denies
excuses to people who *did* engage with treatment. That is the concrete failure
SC-001 exists to catch, and it is why the corpus is real policy rather than
documents we wrote ourselves.

## Known risks

**The corpus costs what the generator saves.** Golden expectations mean
hand-authoring the tables the generator is meant to produce. Accepted: keep N
small (4–6 sections) and treat the corpus as a durable asset, not throwaway.

**Real policy may resist tabulation.** If the mechanizable fraction is low, the
fidelity number is measured over a narrow base. That is itself a finding worth
publishing — *what fraction of real policy is determinable* is a question the
target buyer will ask. The Determination 2018 is encouraging: three of four
provisions are mechanizable, and the fourth is a clean negative case.

**Policy drifts under you.** The Guidelines were already at v1.24 (1 July 2026)
against search metadata's v1 July 2025 — at least annual revision. Mitigated by
hashing; a mismatch means re-derive expectations, not "the test is broken".

**Optics.** This domain determines welfare penalties. The framing obligation on
036's sample README applies here too: the point is *constraining* automated
decision-making, and that must be stated, not inferred.

## Open Questions

1. **Does the fidelity SC score decomposition, or only table content?** Recommendation: tables numerically, decomposition qualitatively — "is this the right set of decisions" has no clean oracle.
2. **Is 80% the right SC-001 threshold?** Provisional until the corpus is built; set it from a measured baseline, not a guess.
3. **How is an unmapped marker represented** — a `provenance.unmapped: []` list, a sibling report file, or inline comments? Must survive review and reach the reviewer's eye.
4. **Cross-source consistency**: does the generator produce the same table from the statute (Determination 2018) and from its provider-facing restatement (Guidelines pp. 271–273)? A useful robustness check, possibly SC-005.
5. **Delivery shape** — a HoloDeck agent with `response_format` set to the table schema (dogfooding), or a dedicated CLI verb (`holodeck workflow draft`)?

## Not doing

- Generating edge agents, gate schemas, or prompts
- DMN XML import (separate, deferred — see `036/dmn-yaml-mapping.md` route 2)
- Auto-committing, auto-reviewing, or any path that lets generated policy execute unreviewed
- Fine-tuning or training on the corpus — it is an evaluation set, not training data
