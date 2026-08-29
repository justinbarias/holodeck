# TODO: Deterministic Spine (036)

> **ARCHIVED — PIVOTED, 2026-08-29.** Work stopped after Phase 1 plus the
> review-round hardening. T7 and everything after it will not be built. See
> the archive banner in `../spec.md` and the successor `specs/040-holodeck-temporal/spec.md`.

> Condensed task list — full acceptance criteria and verification steps in
> `plan.md`. Restructured 2026-07-25 around an explicit **MVP ship line**;
> anchor sample retargeted from loan-hardship to the Targeted Compliance
> Framework. See `spec.md` § "The MVP ship line".
>
> Every unfinished task carries a **sample-proof** criterion — a blocking
> acceptance criterion naming what `sample/pbas-points/` must newly demonstrate.
> See § "The running demonstrator" below the ship line.

## Phase 0 — De-risk
- [x] **T1** FEEL evaluator spike (bkflow-feel → pySFeel fallback); record decision + supported subset in `research.md` — **chose bkflow-feel 1.2.0 (MIT); full FR-010 subset covered; caveats + static-rejection list in research.md**

## Phase 1 — US1: single determination node (P1)
- [x] **T2** Workflow Pydantic models + DAG validation + `schemas/workflow.schema.json` + sync test — **shape-discriminated node union (edge/policy/human), extra=forbid closes the AI-as-verdict route (SC-008); `source:` annotation added; 24 tests green**
- [x] **T3** Decision-table model + loader + FEEL subset wrapper (static rejection w/ locator) *(needs T1)* — **`DecisionTable` owns `hit_policy`; keyed `when` cells; three failure channels (DecisionTableError / FeelValidationError w/ table·rule·cell locator / pydantic ValidationError); allowlist walker over the lark tree; 104 workflow tests green. research.md caveat 6 records `date(literal)`-only.**
- [x] **T2a** *(amendment to landed code)* Workflow-level `input_data:` block — declaration, JSON-Schema validation, `--input` payload binding, no-agent-representable check *(FR-025…FR-028, SC-010)*; re-publish `workflow.schema.json` — **per-fact JSON Schema validated before any node executes; SC-010 asserted against the *published* `workflow.schema.json`, not just the Pydantic model; fact names resolve in `inputs:` but are excluded from the executed topological order**
- [x] **T3a** *(amendment to landed code)* `provenance:` block on `DecisionTable` — non-executable, not FEEL-referenceable *(FR-029, FR-032)* — **unreachable from FEEL by two independent routes: a reserved-root check rejects it at load, and it is never placed in the evaluation context. FR-032 survived 16 evasion attempts in audit.**
- [x] **T4** Hit-policy evaluation UNIQUE/FIRST/PRIORITY + SC-004 conformance suite *(needs T3)* — **standard DMN semantics for all three; loud no-match (absent a declared default) and UNIQUE multi-match; `Verdict` carries matched-rule identity + table version; SC-004 suite green**
- [x] **T5** Edge-node executor + schema gate (Claude-only, mocked backend tests) *(needs T2)* — **only the validated object crosses; the gate schema is snapshotted by content; three distinct failure channels (output rejected / gate unusable / nothing produced to judge); gate validation never touches the network**
- [x] **T6** Runner + `holodeck workflow run` (single level, minimal OTel spans) + **review-gate refusal** *(FR-030, SC-009)* *(needs T4, T5, T2a, T3a)* — **`prepare_workflow` does all validation and constructs no backend, making FR-003 structural rather than ordered; `execute_workflow` runs the DAG; the review gate refuses unreviewed generated tables at prepare time (FR-030/SC-009); `holodeck workflow run [--input]` maps errors to exit codes separating misauthored / could-not-decide / gate-rejected / invocation-failed**
  - **Delivered T7's implementation as a side effect:** `runner._named_inputs` makes multi-level composition work at arbitrary depth. T7 is now a proof-and-cover task — see below.
  - ⚠ **Known gap — FR-018 is NOT met.** `Workflow` has no `observability:` block and is `extra="forbid"`, so the CLI has nowhere to obtain an `ObservabilityConfig` and `holodeck workflow run` emits **zero spans** today. `execute_workflow` accepts one and is unit-tested with an in-memory exporter, but that path is unreachable end to end. **Closed by T13.**
- [x] **CHECKPOINT 1** — **PASSED 2026-07-25.** `make ci` clean ✔ (5298 unit tests passing, 4 skipped; format, lint, type-check, security, pre-commit all clean) · single edge→policy workflow runs via CLI ✔ — exercised against a **live** Claude call, not a mock: `sample/pbas-points/` classified free text, the gate validated it, and `tables/points.dmn.yaml` rule 10 awarded 20 points `per_week` · human review ✔ (maintainer, who ran the live workflow). Phase 2 is unblocked.

## Phase 2 — US2: composition (P1)
- [ ] **T7** Multi-level composition — **prove and cover** *(needs T6)*
  - ⚠ **The wiring already landed in T6 — do not re-implement it.** `runner._named_inputs` resolves each declared input to a fact, an upstream gated object, or an upstream `Verdict.outputs`; arbitrary depth then follows from `graphlib`. Verified on the committed code, unmodified: a 3→2→1 DAG executes `('a','b','c','mid1','mid2','top')` and composes correctly, with `b` feeding **both** mid nodes (a diamond) and the top table dot-pathing into `mid1.band`. FR-009/FR-016 are satisfied.
  - **What remains is proof, not build.** Nothing defends composition today — there is no `test_composition.py` and `test_runner.py` covers only single-level DAGs. The audit's `all()`→`any()` mutation in the hit-policy matcher survived because correct code with no test defending it looks exactly like broken code. Write the 3→2→1 integration test (diamond shape, dot-path into an upstream verdict), and close US2 scenario 3 **through the CLI** — unresolved reference → exit 2, no agent invoked. *(The cycle, scenario 2, is already covered end to end.)*
  - **Sample-proof:** PBAS gains a genuine two-level DRD. Footnote (1) of the source — *"These tasks and activities are available to Workforce Australia Services participants only"* — is an eligibility precondition attached to rows, not a points rule; it is currently encoded inline as rules 5/6 of `tables/points.dmn.yaml`. Split it into `tables/eligibility.dmn.yaml`, keyed on `activity.activity_type` + `participant.stream`, whose verdict is a named input to `points_award`. **Refactor plus a level:** the verdict for every existing case, including the `workforce_australia_online` + `drivers_licence_attainment` case, must not change.
  - *Note:* footnote (1) covers exactly **one** of the eleven modelled activities (driver's licence attainment), so the eligibility table is thin unless the taxonomy also picks up a footnote **(2)** row (*"available to Workforce Australia Online participants only"* — online learning modules, Youth Advisory Sessions). Adding one is optional but makes the level restrict in both directions instead of one.
  - **Deferred from the 036 review round (2026-07-25), not part of T7's proof:**
    `runner._check_table_inputs` checks only the dot-path *roots* an input
    expression reads, so a typo'd *field* (`evidence.net_incom`, where the node
    does declare `evidence`) still reaches evaluation and costs an edge-agent
    call. It is now cheaply checkable — `PreparedWorkflow.gate_schemas` holds
    each edge node's gate by content, and `input_data` schemas are on disk — but
    it is new capability, not a defect fix: a root check and a field check are
    different validations. Note it only holds for gates that actually declare
    `properties`; an open gate (`{}`) can promise nothing about its fields.
- [ ] **CHECKPOINT 2** — composition proven

## Phase 3 — US3: human determination (P1)
- [ ] **T8** Human node: table recommendation + CLI prompt + `decided_by` + override flag + abort path *(needs T7)*
  - ⚠ **Keep `click` out of the runner** — one narrow decision seam (selected output, `decided_by`, timestamp, abort). CLI is the only implementation built; serve and a task-inbox UI are Out of Scope. See the transport note under Task 8 in `plan.md`.
  - ⚠ **If serve is ever wired, the runner emits the AG-UI `prompt_user` events — never a model.** In the normal AG-UI flow the model emits the tool call and the result lands in the model's context, which would put Claude between the delegate and the verdict. The obvious implementation is the wrong one.
  - ⚠ Durability (a delegate deciding tomorrow) is **additive over T10+T11** — the run record is the resume token. Build none of it here; just don't design a seam that only a terminal can satisfy.
  - **Sample-proof:** PBAS gains a `requires_human` node modelling the source's activity-bonus Note — *"Providers and the Digital Services Contact Centre (DSCC) may increase the values of certain tasks or activities through an activity bonus to reflect the individual circumstances of the participant."* `corpus-manifest.md` names this the corpus's canonical **unmappable** clause: pure discretion, no threshold to extract. The node presents `points_award` as the recommendation and a named `decided_by` delegate selects among the table's declared outputs. **This is the sample's most important framing point — the human node exists because the source says a rule cannot.** The README's "deliberately NOT modelled" item 2 moves from *excluded* to *routed to a human*.
  - *Two source constraints, not design choices:* the Note authorises an **increase only** (never a reduction), and it states **no amount and no criteria** — so no table may compute a bonus value, and the delegate's choice is a categorical selection among declared outputs (e.g. confirm / bonus applied), not free numeric entry. FR-014 already restricts the prompt to the table's declared outputs.
- [ ] **CHECKPOINT 3** — scripted end-to-end with human decision; human review

## Phase 4 — US4: record & replay (P2)
- [ ] **T10** RunRecord model + `.holodeck/runs/<id>.json` writer with `input_data`, table/gate snapshots, provenance + sha256 *(needs T8)*
  - **Sample-proof:** a PBAS run writes a record carrying the validated `input_data` facts (`participant`, `claim`), the gated classification, the matched rule of each policy node, the delegate's decision, and content snapshots of both tables and the gate schema with their `provenance` blocks.
- [ ] **T11** `holodeck workflow replay <record>` — snapshot-only, zero LLM, loud integrity failures *(needs T10)*
  - **Sample-proof:** the PBAS record replays to an identical verdict with `sample/pbas-points/tables/` and `sample/pbas-points/schemas/` **deleted** and **zero** LLM calls.
- [ ] **CHECKPOINT 4** — run → record → replay loop proven

## Phase 5 — US6: TCF sample (P1 — MVP proof)
- [ ] **T13a** *(spike, before T14)* Is "active months" expressible? Resolve Open Question 5; if not, define how it is pre-computed into `input_data`
  - **Sample-proof: none — deliberately.** This is a written verdict in `research.md`, not code, and it changes nothing in `sample/pbas-points/` (PBAS has no date arithmetic). Do not invent a sample change for it.
  - ⚠ **Prerequisite the spike must record:** `research.md` caveat 6's `format: date` → `datetime.date` conversion **does not exist**. `input_data` stores the raw JSON value, so a `date`-typed table column receives a **string** and bare subtraction raises. Both T13a and T14 depend on date arithmetic.
- [ ] **T14** `sample/tcf-compliance/` — `input_data` prior state, 1 edge agent (reasonable-excuse assessment vs s5/s6), 2 tables (zone `UNIQUE`, penalty `FIRST`), human delegate, **no `draft:` block**; README framing statement; scripted run + replay; integration test *(needs T11, T13a)*
  - **Sample-proof:** unchanged in intent — TCF is the anchor, not a PBAS extension. Additionally: `sample/pbas-points/` must still **run and replay** after T14 lands, so the demonstrator does not rot while attention is on TCF.

---

# ▲ MVP SHIP LINE ▲

**MVP = T4, T5, T6, T7, T8, T10, T11, T14** (+ amendments T2a, T3a).
Delivers SC-001…SC-005, SC-007…SC-010. Only **SC-006** falls past the line.

---

## The running demonstrator (convention)

Two samples, two jobs. Neither substitutes for the other.

- **`sample/pbas-points/` is the running demonstrator.** It exists from T6 onward
  and **grows one feature per task**: every task above carries a *blocking*
  sample-proof criterion naming what PBAS must newly demonstrate. A task is not
  done while the demonstrator cannot show its feature. It stays cheap to run —
  one edge agent, one live Claude call, one `--input` payload.
- **`sample/tcf-compliance/` (T14) is the anchor MVP proof.** A different job: a
  real statute, a prior-state → next-state transition, a human delegate, and a
  gate whose fields are drawn from an enumerated legal list (*Reasonable Excuse
  Determination 2018* s5(2)(a)–(j)). It lands once, complete. It is **not**
  retargeted onto PBAS and its scope is not weakened by anything PBAS proves.

Why the split: PBAS proves each primitive the moment it is built, so a regression
surfaces at the next task rather than at T14. TCF proves the whole claim on
policy nobody can accuse us of having invented.

---

## Phase 6 — Post-MVP
- [ ] **Layering: `models` → `lib` inversion across two private names.**
      `models/workflow.py` imports `_LITERAL_NAMES` (and `referenced_roots`) from
      `lib.workflow.feel` and `_RESERVED_FEEL_ROOTS` from
      `models.decision_table`. Two of the three are underscore-prefixed, so a
      model layer now depends on private members of the lib layer. It works and
      is not circular, but the coupling is real: node-id validity is defined by
      the embedded FEEL grammar, which lives in `lib`. Either promote the two
      names to public API (`LITERAL_NAMES`, `RESERVED_FEEL_ROOTS`) or move the
      name-validity predicate into `lib.workflow.feel` and have the model call
      one public function. Raised in the 036 review round (2026-07-25).
- [ ] **T9** Draft agent advisory path (`ai_may_draft: reasons`) *(needs T8)* — restores the `draft:` block to the sample
  - **Sample-proof:** PBAS's human node gains a `draft:` block producing advisory **reasons** for the activity-bonus decision — never the points value, and never a bonus amount. The source states no amount, so a drafted number would be invented policy.
- [ ] **T12** `WorkflowTestExecutor` + `holodeck workflow test` with expected-vs-actual diff *(needs T4)* — delivers SC-006
  - **Sample-proof:** a committed policy-test file over `tables/points.dmn.yaml` in `sample/pbas-points/`, covering at minimum the tier boundary at **exactly 15** contact hours (`"up to 15"` is inclusive → 15 points, not 20) and an `activity_type` the table does not cover → loud no-match. The second case is only reachable through the policy tester: the gate's closed `enum` and the table's rules cover the same eleven activities, so a *run* can never produce it.
- [ ] **T13** Full OTel GenAI span attributes per specs 018/022; replay spans marked; provenance on spans *(needs T11)*
  - **Sample-proof:** a PBAS run emits spans carrying node id and kind, table version, matched rule, and table provenance. This also closes T6's known gap: the CLI must be able to obtain an `ObservabilityConfig` at all (today `Workflow` is `extra="forbid"` with no `observability:` block, so `holodeck workflow run` emits zero spans).
- [ ] **T15** Docs: concepts, authoring reference, CLI usage, replay guarantees *(needs T14)*
  - **Sample-proof:** the quickstart reproduces the PBAS **run and replay verbatim** — commands copy-paste and the printed verdict matches.
- [ ] **CHECKPOINT 5** — SC-001…SC-010 mapped to tests; `make ci` clean; PR

## Successor
- **039 policy generator** — AI-drafted tables + DRD from policy documents.
  Spec written in parallel (`specs/039-policy-generator/`) so its schema needs
  land here as additions while this schema is soft. **Built only after the MVP.**
  Corpus pinned: see `specs/039-policy-generator/corpus-manifest.md`.

## Open questions (human input)
- [x] ~~`sample/` is not tracked by git~~ — **DECIDED 2026-07-25: leave ignored,
      defer.** `.gitignore:45` ignores `/sample` repo-wide and `git ls-files
      sample/` returns **zero** files; samples stay local-only. Revisit only with
      a concrete reason. Three consequences follow and are binding until then:
  - **Every sample-proof criterion above is a manual check, not a CI gate.** A
    task claiming its sample proof is asserting something no pipeline can
    confirm. Say so when reporting a task complete rather than implying CI
    covered it.
  - **T14's `tests/integration/test_tcf_sample.py` cannot exist as specified** —
    it would reference untracked paths and fail on a fresh clone. T14 needs
    rescoping before it starts: either the assertions move to committed fixtures
    under `tests/fixtures/workflow/`, or the sample proof is a documented
    quickstart run by hand. Do not discover this mid-task.
  - **SC-007 keeps only its second limb.** It reads "demonstrated in CI **or a
    documented quickstart**" — the quickstart route survives intact, the CI route
    does not. SC-007 is still satisfiable; it is not automatically verifiable.
- [ ] Confirm policy-test verb: `holodeck workflow test`
- [ ] Which TCF slice the sample models (Open Question 6)
- [x] ~~Sample edge-agent model choice (Claude — which tier)~~ — **settled: `anthropic` / `claude-opus-5`, `temperature: 0.0` (`sample/pbas-points/agents/activity-classifier.yaml`), run end to end successfully** (2026-07-25)
- [x] ~~Repo weight: 4.6 MB Guidelines PDF in `039/corpus/`~~ — **accepted, tracked in git** (2026-07-25)
- [x] ~~GPLv3 acceptable for pySFeel fallback?~~ — moot, T1 chose bkflow-feel (MIT)
