# TODO: Deterministic Spine (036)

> Condensed task list — full acceptance criteria and verification steps in
> `plan.md`. Order respects the dependency graph; T12 and T13 can run in
> parallel once their dependencies land.

## Phase 0 — De-risk
- [x] **T1** FEEL evaluator spike (bkflow-feel → pySFeel fallback); record decision + supported subset in `research.md` — **chose bkflow-feel 1.2.0 (MIT); full FR-010 subset covered; caveats + static-rejection list in research.md**

## Phase 1 — US1: single determination node (P1)
- [ ] **T2** Workflow Pydantic models + DAG validation + `schemas/workflow.schema.json` + sync test
- [ ] **T3** Decision-table model + loader + FEEL subset wrapper (static rejection w/ locator) *(needs T1)*
- [ ] **T4** Hit-policy evaluation UNIQUE/FIRST/PRIORITY + SC-004 conformance suite *(needs T3)*
- [ ] **T5** Edge-node executor + schema gate (Claude-only, mocked backend tests) *(needs T2)*
- [ ] **T6** Runner + `holodeck workflow run` (single level, minimal OTel spans) *(needs T4, T5)*
- [ ] **CHECKPOINT 1** — US1 scenarios green; `make ci` clean; human review

## Phase 2 — US2: composition (P1)
- [ ] **T7** Multi-level composition: named-input FEEL context + 3→2→1 DAG integration test *(needs T6)*
- [ ] **CHECKPOINT 2** — composition proven

## Phase 3 — US3: human determination (P1)
- [ ] **T8** Human node: table recommendation + CLI prompt + `decided_by` + override flag + abort path *(needs T7)*
- [ ] **T9** Draft agent advisory path (`ai_may_draft: reasons`) *(needs T8)*
- [ ] **CHECKPOINT 3** — scripted end-to-end with human decision; human review

## Phase 4 — US4: record & replay (P2)
- [ ] **T10** RunRecord model + `.holodeck/runs/<id>.json` writer with table/gate snapshots + sha256 *(needs T8)*
- [ ] **T11** `holodeck workflow replay <record>` — snapshot-only, zero LLM, loud integrity failures *(needs T10)*
- [ ] **CHECKPOINT 4** — run → record → replay loop proven

## Phase 5 — US5: policy testing (P3, parallel after T4)
- [ ] **T12** `WorkflowTestExecutor` + `holodeck workflow test` with expected-vs-actual diff *(needs T4)*

## Phase 6 — Observability
- [ ] **T13** Full OTel GenAI span attributes per specs 018/022; replay spans marked *(needs T11)*

## Phase 7 — US6: sample + docs (P3)
- [ ] **T14** `sample/loan-hardship/` — 3 edge agents, 2+1 tables, scripted run + replay; integration test *(needs T9, T11)*
- [ ] **T15** Docs: concepts, authoring reference, CLI usage, replay guarantees *(needs T14)*
- [ ] **CHECKPOINT 5** — SC-001…SC-008 mapped to tests; `make ci` clean; PR

## Open questions (human input)
- [ ] GPLv3 acceptable for pySFeel fallback? (only if T1 fails)
- [ ] Confirm policy-test verb: `holodeck workflow test`
- [ ] Sample edge-agent model choice (Claude — which tier)
