---
description: "Tasks for User Story 5 — Report per-turn results in the test dashboard (P3)"
---

# Tasks: User Story 5 — Render Multi-Turn Test Cases in the Dashboard (P3)

**Input**: `/specs/032-multi-turn-test-cases/` — spec.md (US5), plan.md §Phase 2 step 5 (dashboard), contracts/turn-result-schema.md §1–§4 (wire contract the dashboard reads), feature 031 dashboard code under `src/holodeck/dashboard/`.
**Approach**: Test-Driven Development — but grounded in the actual Dash SSR architecture (3-column layout: runs → cases → right detail pane with `html.Details` disclosures), NOT a "parent row with child disclosure" model. Verify agents confirmed `explorer_data.py` emits per-case dataclasses (`CaseSummary`, `CaseDetail`, `ConversationView`, etc.) consumed by `views/explorer.py:282–312` (cases column) and `views/explorer.py:809–822` (right detail pane).

**Story Goal**: When a run contains multi-turn test cases, (a) the cases column shows an optional turn-count chip (`4 turns · 3 passed`), (b) the right detail pane's `_conversation_section` (`views/explorer.py:597`) renders each turn as its own `html.Details` block reusing `_tool_call_panel`, `_metric_row_div`, and other existing sub-components. Single-turn rendering byte-identical. Additive only — no new discriminator, no dataclass rename.

**Reporter markdown/JSON hierarchy has moved to US1** (Phase 4, tasks T034–T039). This file no longer duplicates that work. One JSON regression test remains here as a cross-feature guard.

**Independent Test**: After US1 + US2 land, open the dashboard on a run with a multi-turn test case. The case row shows a turn-count chip. Selecting it opens the detail pane whose conversation section contains N expandable `html.Details` blocks, one per turn. Single-turn cases render exactly as before.

**Depends on**: US1 (foundational — `TestResult.turns` wire format + per-turn markdown reporter). Naturally stronger when US2/US3/US4 data is present but does not strictly require them.

---

## Phase 1: Setup

- [ ] T001 [US5] Add `TurnView` dataclass stub to `src/holodeck/dashboard/explorer_data.py` (fields TBD — empty shell). Makes it importable by the Phase 2 tests before `build_case_detail` is wired. Concrete testable deliverable (not a reading task).
- [ ] T002 [US5] Extend `src/holodeck/dashboard/seed_data.py` with a `build_multi_turn_seed_case()` helper (or extend one existing seed run) that produces a mixed run with: one legacy single-turn test case, one 3-turn test case (all passed), and one 4-turn test case with turn-2 failing. Shape per contracts/turn-result-schema.md. **Do not create a JSON fixture file** — dashboard tests consume seed data via `build_seed_runs()` (see `tests/unit/dashboard/test_explorer_data.py:16`); the existing convention is Python-builder, not JSON.

---

## Phase 2: Explorer data layer (TDD — additive, not a redesign)

### Tests first

- [ ] T003 [P] [US5] Write `tests/unit/dashboard/test_explorer_data_multi_turn.py::test_single_turn_case_detail_unchanged` — for a legacy `TestResult` (no `turns`), `build_case_detail(run, case_name, conversations_map)` returns a `CaseDetail` whose `ConversationView` has no `turns` field populated (or `turns is None`). Existing tests in `tests/unit/dashboard/test_explorer_data.py` stay green (SC-002 regression guard at the dashboard layer).
- [ ] T004 [P] [US5] Add `test_explorer_data_multi_turn.py::test_build_case_detail_emits_turn_views_when_turns_present` — for a multi-turn `TestResult`, `build_case_detail(...)` populates `CaseDetail.conversation.turns: list[TurnView]` with one `TurnView` per `TurnResult`.
- [ ] T005 [P] [US5] Add `test_explorer_data_multi_turn.py::test_case_summary_carries_turn_counts` — `CaseSummary` produced for a multi-turn case carries optional `turns_total`, `turns_passed`, `turns_failed` fields (default `None` for single-turn cases). These feed the cases-column chip.
- [ ] T006 [P] [US5] Add `test_explorer_data_multi_turn.py::test_turn_view_fields` — each `TurnView` includes `turn_index`, `input`, `response`, `tool_invocations` (same shape the existing `ToolCallView` consumes), `metric_results`, `tools_matched`, `arg_match_details`, `errors`, `skipped`, `execution_time_ms`, `token_usage`.
- [ ] T007 [P] [US5] Add `test_explorer_data_multi_turn.py::test_skipped_turn_marked_in_view` — a turn with `skipped=True` lands in the payload with a `state: "skipped"` marker so the renderer can style it distinctly.
- [ ] T008 [P] [US5] Add `test_explorer_data_multi_turn.py::test_metric_kind_code_surfaced_on_turn_view` — when `MetricResult.kind == "code"` (US4 output), the `TurnView.metric_results` preserves the kind so the compare view and metric-trend panels can filter on it.

### Implementation

- [ ] T009 [US5] In `src/holodeck/dashboard/explorer_data.py`, flesh out `TurnView` (added as stub in T001) with the fields asserted in T006. Reuse existing serializers (`_build_conversation`, `ToolCallView`, `MetricRow`) — do NOT invent parallel shapes.
- [ ] T010 [US5] Add optional `turns: list[TurnView] | None = None` to `ConversationView` (or to `CaseDetail` directly — pick whichever keeps existing consumers stable). Populate in `build_case_detail` when `result.turns is not None`. Legacy single-turn path unchanged.
- [ ] T011 [US5] Extend `CaseSummary` with optional `turns_total: int | None`, `turns_passed: int | None`, `turns_failed: int | None`; populate from `result.turns` when present. Default `None` preserves the single-turn cases-column rendering.

---

## Phase 3: Rendering (TDD — Dash SSR structural tests)

### Tests first

- [ ] T012 [P] [US5] Write `tests/unit/dashboard/test_explorer_multi_turn_rendering.py::test_conversation_section_renders_per_turn_details_when_turns_present` — calls the `_conversation_section` helper (`views/explorer.py:597`) with a `ConversationView` carrying three `TurnView`s; asserts the returned tree (via `component.to_plotly_json()` — the same idiom as `tests/unit/dashboard/test_assistant_rendering.py`) contains three `html.Details` blocks, each labelled with `turn_index`.
- [ ] T013 [P] [US5] Add `test_explorer_multi_turn_rendering.py::test_single_turn_conversation_section_unchanged` — for a `ConversationView` with `turns is None`, the returned tree matches the pre-feature structure byte-for-byte (compare against a golden serialized from the current single-turn rendering path before the change).
- [ ] T014 [P] [US5] Add `test_explorer_multi_turn_rendering.py::test_cases_column_renders_turn_count_chip_when_present` — for a `CaseSummary` with `turns_total=4, turns_passed=3, turns_failed=1`, the rendered row contains a chip element with text like `"4 turns · 3/4 passed"`. For `turns_total=None`, the row has no such chip (single-turn path unchanged).
- [ ] T015 [P] [US5] Add `test_explorer_multi_turn_rendering.py::test_failing_turn_surfaces_reason_via_existing_helpers` — a failing `TurnView` inside the conversation section reuses `_tool_call_panel`'s existing error affordance (views/explorer.py:484–569) and `_metric_row_div`'s existing reasoning affordance (views/explorer.py:757). Assert the rendered children include calls to those helpers, not a brand-new error component.

### Implementation

- [ ] T016 [US5] Extend `_conversation_section` in `src/holodeck/dashboard/views/explorer.py` to branch on `conversation.turns`: if `None`, render the existing single-turn conversation flow untouched; if populated, render one `html.Details(open=False, children=[_turn_thread_block(turn) for turn in conversation.turns])` group with a summary label per turn. No client-side state — `html.Details` handles expansion natively.
- [ ] T017 [US5] Add private helper `_turn_thread_block(turn: TurnView) -> html.Div` in `views/explorer.py` that composes existing sub-components (`_tool_call_panel`, `_metric_row_div`, the token-usage renderer) into a per-turn block. Style `turn.skipped` turns distinctly (e.g. dimmed heading).
- [ ] T018 [US5] Extend `_cases_column` in `src/holodeck/dashboard/views/explorer.py:235-312` to render a `mini-metric` chip when `case.turns_total is not None`: text `"{turns_total} turns · {turns_passed}/{turns_total} passed"`. Chip absent when `turns_total is None`.

---

## Phase 4: JSON regression guard (US1 owns the markdown/JSON reporter; this is a cross-feature check)

- [ ] T019 [P] [US5] Add `tests/unit/lib/test_runner/test_reporter_multi_turn.py::test_json_report_has_turns_array_cross_check` (or reuse the US1 equivalent from US1 T037). Purpose: catch a regression where the reporter drops `turns` from JSON even though the model carries it. One-line cross-feature guard. No implementation task — reporter hierarchy already lands in US1 Phase 4.

---

## Phase 5: End-to-end dashboard validation

- [ ] T023 [P] [US5] Add `tests/integration/test_dashboard_multi_turn_e2e.py::test_dashboard_renders_multi_turn_run` — bootstraps the dashboard layout function against the seed from T002 (`build_multi_turn_seed_case()`); reuses the existing Dash-test harness from `tests/unit/dashboard/`. Asserts: (a) cases column shows the turn-count chip for the 4-turn case, (b) selecting it populates the detail pane's conversation section with 4 `html.Details` blocks, (c) the failing turn-2 block surfaces error text via the existing helpers.
- [ ] T024 [US5] Run `make format && make lint && make type-check && pytest -n auto`. All green.

---

## Phase 6: Docs touch-up

- [ ] T025 [P] [US5] Add two lines to `docs/guides/dashboard.md` calling out multi-turn disclosure behavior: "Multi-turn test cases render with a turn-count chip in the cases column; the right detail pane's conversation section expands each turn independently."

---

## Dependencies

- Phase 1 → Phase 2 → Phase 3 (UI rendering consumes Phase 2's dataclasses).
- Phase 4 is a guard independent of UI; can run any time after US1 reporter lands.
- Phase 5 depends on Phases 2 and 3.
- Within Phase 2: T003–T008 parallel; T009–T011 serial (same module).
- Within Phase 3: T012–T015 parallel; T016–T018 serial (views/explorer.py).

## Independent test criteria (recap)

- Single-turn rendering byte-identical (T003, T013, plus existing `test_explorer_data.py` staying green).
- Multi-turn detail pane shows per-turn `html.Details` blocks reusing existing helpers (T012, T015).
- Cases column chip summarizes turn counts so a user can scan a run without opening each case (T014).
- `kind: "code"` / `"rag"` / `"geval"` flow through to compare view intact (T008) — preserves feature-031 filter contracts.

## Notes on scope

- US5 is shippable after US1 alone: a 3-turn chit-chat run with no ground truths already exercises `html.Details` per-turn rendering and the cases-column chip. Per-turn metric/tool badges get richer when US2/US3/US4 ship, but the rendering path does not require them.
- No "parent/child row discriminator" is introduced — the dashboard model stays case-detail-oriented, and multi-turn detail lives inside `ConversationView` (additive field). This is the codebase-verified architecture of feature 031.
