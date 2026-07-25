# TODO: Deterministic Spine (036)

> Condensed task list — full acceptance criteria and verification steps in
> `plan.md`. Restructured 2026-07-25 around an explicit **MVP ship line**;
> anchor sample retargeted from loan-hardship to the Targeted Compliance
> Framework. See `spec.md` § "The MVP ship line".

## Phase 0 — De-risk
- [x] **T1** FEEL evaluator spike (bkflow-feel → pySFeel fallback); record decision + supported subset in `research.md` — **chose bkflow-feel 1.2.0 (MIT); full FR-010 subset covered; caveats + static-rejection list in research.md**

## Phase 1 — US1: single determination node (P1)
- [x] **T2** Workflow Pydantic models + DAG validation + `schemas/workflow.schema.json` + sync test — **shape-discriminated node union (edge/policy/human), extra=forbid closes the AI-as-verdict route (SC-008); `source:` annotation added; 24 tests green**
- [x] **T3** Decision-table model + loader + FEEL subset wrapper (static rejection w/ locator) *(needs T1)* — **`DecisionTable` owns `hit_policy`; keyed `when` cells; three failure channels (DecisionTableError / FeelValidationError w/ table·rule·cell locator / pydantic ValidationError); allowlist walker over the lark tree; 104 workflow tests green. research.md caveat 6 records `date(literal)`-only.**
- [ ] **T2a** *(amendment to landed code)* Workflow-level `input_data:` block — declaration, JSON-Schema validation, `--input` payload binding, no-agent-representable check *(FR-025…FR-028, SC-010)*; re-publish `workflow.schema.json`
- [ ] **T3a** *(amendment to landed code)* `provenance:` block on `DecisionTable` — non-executable, not FEEL-referenceable *(FR-029, FR-032)*
- [ ] **T4** Hit-policy evaluation UNIQUE/FIRST/PRIORITY + SC-004 conformance suite *(needs T3)*
- [ ] **T5** Edge-node executor + schema gate (Claude-only, mocked backend tests) *(needs T2)*
- [ ] **T6** Runner + `holodeck workflow run` (single level, minimal OTel spans) + **review-gate refusal** *(FR-030, SC-009)* *(needs T4, T5, T2a, T3a)*
- [ ] **CHECKPOINT 1** — US1 scenarios green; `make ci` clean; human review

## Phase 2 — US2: composition (P1)
- [ ] **T7** Multi-level composition: named-input FEEL context + 3→2→1 DAG integration test *(needs T6)*
- [ ] **CHECKPOINT 2** — composition proven

## Phase 3 — US3: human determination (P1)
- [ ] **T8** Human node: table recommendation + CLI prompt + `decided_by` + override flag + abort path *(needs T7)*
- [ ] **CHECKPOINT 3** — scripted end-to-end with human decision; human review

## Phase 4 — US4: record & replay (P2)
- [ ] **T10** RunRecord model + `.holodeck/runs/<id>.json` writer with `input_data`, table/gate snapshots, provenance + sha256 *(needs T8)*
- [ ] **T11** `holodeck workflow replay <record>` — snapshot-only, zero LLM, loud integrity failures *(needs T10)*
- [ ] **CHECKPOINT 4** — run → record → replay loop proven

## Phase 5 — US6: TCF sample (P1 — MVP proof)
- [ ] **T13a** *(spike, before T14)* Is "active months" expressible? Resolve Open Question 5; if not, define how it is pre-computed into `input_data`
- [ ] **T14** `sample/tcf-compliance/` — `input_data` prior state, 1 edge agent (reasonable-excuse assessment vs s5/s6), 2 tables (zone `UNIQUE`, penalty `FIRST`), human delegate, **no `draft:` block**; README framing statement; scripted run + replay; integration test *(needs T11, T13a)*

---

# ▲ MVP SHIP LINE ▲

**MVP = T4, T5, T6, T7, T8, T10, T11, T14** (+ amendments T2a, T3a).
Delivers SC-001…SC-005, SC-007…SC-010. Only **SC-006** falls past the line.

---

## Phase 6 — Post-MVP
- [ ] **T9** Draft agent advisory path (`ai_may_draft: reasons`) *(needs T8)* — restores the `draft:` block to the sample
- [ ] **T12** `WorkflowTestExecutor` + `holodeck workflow test` with expected-vs-actual diff *(needs T4)* — delivers SC-006
- [ ] **T13** Full OTel GenAI span attributes per specs 018/022; replay spans marked; provenance on spans *(needs T11)*
- [ ] **T15** Docs: concepts, authoring reference, CLI usage, replay guarantees *(needs T14)*
- [ ] **CHECKPOINT 5** — SC-001…SC-010 mapped to tests; `make ci` clean; PR

## Successor
- **039 policy generator** — AI-drafted tables + DRD from policy documents.
  Spec written in parallel (`specs/039-policy-generator/`) so its schema needs
  land here as additions while this schema is soft. **Built only after the MVP.**
  Corpus pinned: see `specs/039-policy-generator/corpus-manifest.md`.

## Open questions (human input)
- [ ] Confirm policy-test verb: `holodeck workflow test`
- [ ] Sample edge-agent model choice (Claude — which tier)
- [ ] Which TCF slice the sample models (Open Question 6)
- [x] ~~Repo weight: 4.6 MB Guidelines PDF in `039/corpus/`~~ — **accepted, tracked in git** (2026-07-25)
- [x] ~~GPLv3 acceptable for pySFeel fallback?~~ — moot, T1 chose bkflow-feel (MIT)
