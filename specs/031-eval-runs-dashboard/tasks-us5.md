---
description: "Task list — User Story 5: Explorer + Compare views"
---

# Tasks — US5: Test View Dashboard — Explorer (drilldown) + Compare (3 variants) (Priority: P2)

## ⭐ Primary Source of Truth

**The design handoff bundle is AUTHORITATIVE for every visual, interaction, and layout decision in this task list.** When spec.md and the handoff differ, **the handoff wins**. Spec.md defines what the Explorer must contain functionally; the handoff defines how it must look, behave, and sequence. Compare is introduced by the handoff and has no spec counterpart — the handoff is the ONLY source of truth for that view.

Always consult, in this order of precedence:

1. **[design_handoff_holodeck_eval_dashboard/README.md](./design_handoff_holodeck_eval_dashboard/README.md)** — Views §2 (Explorer) + §3 (Compare) + §State (Streamlit mapping) + §Streamlit notes
2. **[design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html](./design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html)** — open in a browser; click the **Explorer** and **Compare** tabs and try every interaction before writing code
3. **[design_handoff_holodeck_eval_dashboard/explorer.js](./design_handoff_holodeck_eval_dashboard/explorer.js)** — Explorer component source (run list collapse, case list, collapsibles, tool-call rendering, JsonPretty highlighting, 500-byte collapse threshold)
4. **[design_handoff_holodeck_eval_dashboard/compare.js](./design_handoff_holodeck_eval_dashboard/compare.js)** — Compare component source; the 3 variants, case matrix, delta pills, compare tray, `+` button lifecycle — ALL come from here
5. **[design_handoff_holodeck_eval_dashboard/styles.js](./design_handoff_holodeck_eval_dashboard/styles.js)** — exact CSS for chat bubbles, tool-call panels, config-diff highlighting, heatmap cells
6. **[design_handoff_holodeck_eval_dashboard/data.js](./design_handoff_holodeck_eval_dashboard/data.js)** — the `sampleConversation` map (`data.js:160–178`) feeds the Explorer detail panel; US4 ports this as `SEED_CONVERSATIONS`
7. Our `spec.md` / `plan.md` — only for FR-032 (the spec says ≥4KB collapses, the handoff says 500B — **the handoff wins** per T410)

**Rule for every implementation task below**: before writing code, open the corresponding handoff JS file and read the referenced line ranges. Every code task cites `explorer.js:nn-nn` or `compare.js:nn-nn` — treat those citations as mandatory reading.

---

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 5 (P2)
**Plan**: [plan.md](./plan.md)
**Contract**: [contracts/cli.md](./contracts/cli.md)

**Goal**: Replace the US4 stubs with two full views:

1. **Explorer** (`explorer.js`, 364 LOC) — 3-column drilldown: runs list (340px, collapsible to 48px) → cases list (340px) → case detail panel (flex-1). Detail panel sections: case header, agent config snapshot (collapsible JSON + tool chips), conversation thread (chat bubbles + inline tool-call panels with collapse-when-large), expected-tools coverage, per-metric evaluations grouped by kind with judge reasoning.
2. **Compare** (`compare.js`, 576 LOC) — select up to 3 runs via the floating Compare tray; render one of three layout variants (side-by-side / baseline+deltas / matrix-first) switched by a segmented control; heatmap case matrix with regression/improvement rings.

**Independent Test**:
- **Explorer**: From Summary, click any run row → Explorer tab opens with that run pre-selected → cases list populated → click a case → detail panel renders agent config, conversation, tool calls (with collapse-when-bytes>500), expected-tools indicators, and evaluations grouped by kind. Total drill-in: ≤ 3 clicks (SC-006).
- **Compare**: From Summary table, click `+` on three runs → Compare tray shows `1/3, 2/3, 3/3` slots → click "Open Compare" → all three variants render via segmented control → each variant shows summary stats + config diff + case matrix with heatmap coloring and regression/improvement outlines.

**TDD discipline**: Data-assembly helpers (`select_run`, `build_case_detail`, `run_stats`, `compute_compare_callouts`, `case_matrix_dataframe`) are unit-tested. Streamlit UI is verified visually against the HTML prototype (`Evaluation Dashboard.html`) and via optional AppTest smoke tests.

**Dependency**: US4 scaffolds `app.py`, theme, state helpers, seed data (for development without real runs), and stub view modules. US5 replaces `views/explorer.py` and `views/compare.py` stubs and upgrades the Compare tray placeholder in `app.py`.

---

## Phase 1: Setup

None — all infrastructure ships with US4. US5 edits existing files.

---

## Phase 2: Foundational

None.

---

## Phase 3: Explorer — data assembly (TDD)

Reference: `design_handoff_holodeck_eval_dashboard/explorer.js:186–337` (CaseDetail component).

### Tests first

- [ ] T401 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `select_run(runs, run_id) -> EvalRun | None` — given a list and an id (filename stem or `run-XXX`), returns the match; `None` when missing; `runs[-1]` fallback semantics left to the caller
- [ ] T402 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `list_case_summaries(run) -> list[CaseSummary]` with fields `[name, passed, geval_score, rag_avg_score]` — geval pulled from the case's first `kind=="geval"` metric, rag avg from `mean(score for m in kind=="rag")`, `None` when absent (mirrors `explorer.js::CaseList`)
- [ ] T403 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `build_case_detail(run, case_name, conversations_map) -> CaseDetail` returning a dataclass with fields: `header` (pass/fail, run_ts, prompt_version, model, temperature, commit), `agent_snapshot` (model.provider/name/temp/max_tokens, embedding.provider/name, claude.extended_thinking, prompt.version/author/file_path/source/tags, tools list with kind+name), `conversation` (`user`, `assistant`, `tool_calls: list[ToolCallView]`), `expected_tools_coverage` (list of `{name, was_called}`), `evaluations` (dict keyed by `standard|rag|geval` → list of metric rows)
- [ ] T404 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `ToolCallView` dataclass — `name`, `args: Any`, `result: Any`, `result_size_bytes` (from `len(json.dumps(result))`), `large: bool` (True when `result_size_bytes > 500` — handoff `explorer.js:156`)
- [ ] T405 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: expected-tools coverage — case-insensitive comparison against `tools_called`; returns both `matched` (configured + called) and `missed` (configured, not called) counts; optionally exposes `unexpected` (called but not configured)
- [ ] T406 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: missing/absent `conversations_map` key — `build_case_detail` falls back to the data.js pattern of using `refund_eligible_standard` as the default conversation (matches `explorer.js:197`); if neither present, returns an empty conversation without raising
- [ ] T407 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `evaluations` dict ordering — keys always iterate in `geval, rag, standard` order (matches handoff's detail-panel rendering order `explorer.js:204–208`), empty groups are omitted from the output, not left as empty lists

### Implementation

- [ ] T408 [US5] Create `src/holodeck/dashboard/explorer_data.py` with: `select_run`, `list_case_summaries`, `build_case_detail`, and dataclasses `CaseSummary`, `CaseDetail`, `ToolCallView`, `AgentSnapshot`, `MetricRow`
- [ ] T409 [US5] Data source precedence inside `build_case_detail`:
    1. **Real-run mode (primary)**: consume `run.report.results[i].conversation: list[ConversationTurn]` and `run.report.results[i].tool_events: list[ToolEvent]` — both fields are added in US1 Phase 2b migrations (US1 T010e–T010l). The ConversationTurn discriminated union (`role: "user"|"assistant"|"tool_call"|"tool_result"`) drives the chat thread; ToolEvent drives the amber tool-call panels with full `args`/`result`/`bytes` data
    2. **Legacy real-run fallback**: when `conversation` is empty (runs persisted before US1 Phase 2b), synthesize a 2-turn conversation from `test_input` (UserTurn) + `agent_response` (AssistantTurn); interleave empty tool_call placeholders from the name-only `tool_calls: list[str]` with a UI hint "args/result not captured — re-run after US1 Phase 2b lands"
    3. **Seed mode**: fall back to `seed_data.SEED_CONVERSATIONS[case_name]` (US4 T313) when both of the above are empty. Dev-only path guarded by `HOLODECK_DASHBOARD_USE_SEED=1`
    Document the precedence in `explorer_data.py`'s module docstring so reviewers see the data-flow without tracing calls
- [ ] T410 [US5] Centralise the tool-call large-result threshold constant `LARGE_TOOL_RESULT_BYTES = 500` (matches handoff, keeps US5 consistent with `explorer.js:156`; note this differs from FR-032's 4KB threshold — the handoff wins here because it's the user-validated experience)

---

## Phase 4: Explorer — Streamlit view

Reference: open `Evaluation Dashboard.html` in a browser, click **Explorer** tab.

- [ ] T411 [US5] Replace `src/holodeck/dashboard/views/explorer.py` stub with `render_explorer(runs)`:
    1. Read `run_id` from `st.session_state.explorer_run_id` (US4 sets this when user clicks a run in Summary); if `None`, default to newest run
    2. Read `case_name` from `st.session_state.explorer_case_name`; if `None`, default to first case of selected run
    3. Read `st.session_state.explorer_runs_collapsed: bool` for the runs-column collapse state (default `True` per `explorer.js:340`)
- [ ] T412 [US5] **3-column layout**: `col_runs, col_cases, col_detail = st.columns([0.07, 0.19, 0.74])` when collapsed, `[0.20, 0.20, 0.60]` when expanded — Streamlit can't natively change column widths per render without a full rerun, so gate on `explorer_runs_collapsed` and `st.rerun()` after toggling
- [ ] T413 [US5] **Runs column** (when expanded) in `col_runs`:
    - Header row `▸ Runs <count>` with click toggling `explorer_runs_collapsed`
    - Inside `st.container(height=700)` for vertical scroll: for each run, a custom-styled clickable row using `st.button(..., use_container_width=True, type="tertiary")` — button label combines timestamp, prompt version in accent color, pass-rate pill, passed/total, model-suffix (last two hyphen segments of model name, matches `explorer.js:88`)
    - Each row also shows the `+` compare-queue button inline (reuse US4 `state.push_to_compare_queue`)
    - Active run is marked via CSS class `.hd-run-row--active` (bold accent border)
- [ ] T414 [US5] **Runs column** (when collapsed, 48px wide): vertical rotated text `RUNS <count>` with click handler that expands. Use `st.markdown` + inline CSS `writing-mode: vertical-rl; transform: rotate(180deg);` per `explorer.js:50–62`
- [ ] T415 [US5] **Cases column** in `col_cases`:
    - Header `Cases <count>` with "pass" count chip
    - Inside `st.container(height=700)`: one clickable row per case with pass/fail icon (`✓` accent / `✕` coral), case name (mono), G-Eval score on the right, RAG avg small/muted
    - Active case highlighted
- [ ] T416 [US5] **Detail panel** in `col_detail` — scrollable `st.container(height=700)` with five sections, each wrapped in `st.expander(..., expanded=default_open)` where `default_open=True` for case header, `False` for the four content sections (matches handoff Collapsible component `explorer.js:3`)
- [ ] T417 [US5] **Section 1: Case header** (always open, not in expander) — row with pass/fail pill + case name (large); second row with inline metadata badges `run <ts>` · `prompt <version-in-accent>` · `model <name>` · `temp <value>` · `commit <sha>` (matches `explorer.js:212–224`)
- [ ] T418 [US5] **Section 2: Agent config snapshot** — `st.expander("AGENT CONFIG SNAPSHOT · Configuration at run time", expanded=False)`:
    - Grid of key:value pairs via `st.columns(3)` rows showing `model.provider`, `model.name`, `model.temperature`, `model.max_tokens`, `embedding.provider`, `embedding.name`, `claude.extended_thinking`, `prompt.version` (accent), `prompt.author`, `prompt.file_path`, `prompt.source`, `prompt.tags`
    - Tools subsection: eyebrow `TOOLS (n)` + chip row — each chip shows `<kind small-muted> <name>` (matches `explorer.js:249–257`)
    - Sub-right: a "View raw JSON" button opening a nested `st.expander` with `st.json(agent_config_dict)` where `api_key` and any `SecretStr` field is `"***"` (secrets redacted in the data layer, README "Streamlit notes — Redact api_key")
- [ ] T419 [US5] **Section 3: Conversation thread** — `st.expander("CONVERSATION · Thread with tool calls", expanded=False)`:
    - User turn: `st.chat_message("user")` with the input text
    - Every tool call rendered inline BETWEEN user and assistant as a custom-bordered `st.container(border=True)` with class `.hd-tool-call` (amber tint) — header `TOOL · <name>() · <size>B`, body has collapsible `args` (always full) and `result` (collapsed when `ToolCallView.large`); use `st.json(args, expanded=False)` and either `st.json(result, expanded=False)` when small OR `st.expander(f"Expand ({size}B)").json(result)` when large (matches `explorer.js::ToolCall`)
    - Assistant turn: `st.chat_message("assistant")` with caption `AGENT · <model.name>` + response text
- [ ] T420 [US5] **Section 4: Expected tools** — `st.expander("EXPECTED TOOLS · Tool-call coverage", expanded=False)`:
    - Right side of expander title shows match pill: `3/4 matched` (pass style if all matched, fail otherwise)
    - List of rows, each `✓ <name> called` (accent tint background) or `✕ <name> not invoked` (coral tint)
    - "No expected tools configured" fallback when list empty (matches `explorer.js:292`)
- [ ] T421 [US5] **Section 5: Evaluations** — `st.expander("EVALUATIONS · Per-metric results", expanded=False)`:
    - Iterate `geval, rag, standard` (order from T407); for each non-empty group emit an eyebrow label + a list of metric rows
    - Each metric row: 4-column grid `[name, score (big mono, accent if passed else coral), threshold (mono), pass/fail pill]`; if `m.reasoning` present, below the row render a `st.caption` or small indented block with the judge's explanation (matches `explorer.js:314–332`)
- [ ] T422 [US5] **Empty state** — when `runs == []`: center a panel `∅ No runs found · Run holodeck test agent.yaml to generate one` (matches `explorer.js:346–352`)
- [ ] T423 [US5] Bind click handlers so navigation works: clicking a run in column 1 sets `explorer_run_id`, clicking a case in column 2 sets `explorer_case_name`, both trigger `st.rerun()`. When a new run is selected, auto-select first case (matches `explorer.js:357`)

---

## Phase 5: Compare — data assembly (TDD)

Reference: `design_handoff_holodeck_eval_dashboard/compare.js`.

### Tests first

- [ ] T424 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `run_stats(run) -> RunStats` dataclass with fields `pass_rate`, `passed`, `total`, `duration_ms`, `geval_avg`, `rag_avg`, `est_cost` — cost is `duration_ms/1000 * rate` where rate is 0.018 for sonnet models, 0.012 otherwise (matches `compare.js:23–42`). Aggregates match handoff arithmetic to 4 decimals
- [ ] T425 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_case_matrix(runs) -> DataFrame` — rows = union of case names across all runs sorted alphabetically, columns = one per run (`run_id, score, passed, regression: bool, improvement: bool`); score derived from first geval metric, else rag avg, else `1 if passed else 0` (matches `compare.js::caseScore`)
- [ ] T426 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: regression/improvement flags — for runs[1:], `regression = baseline.passed and not this.passed`, `improvement = (not baseline.passed) and this.passed` (matches `compare.js:225–226`)
- [ ] T427 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_config_diff(runs) -> list[ConfigRow]` — rows `[label, values_per_run, all_same: bool]` for `prompt_version, model_name, temperature, tags_joined, git_commit, extended_thinking` (matches `compare.js:94–101`)
- [ ] T428 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_compare_callouts(runs) -> list[Callout]` — for each non-baseline run, list up to 3 regressions and 3 improvements by case name with `+N` overflow count (matches `compare.js::CompareV3:353–364`)
- [ ] T429 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_summary_rows(runs) -> list[StatRow]` — returns rows for `[pass_rate, passed_ratio, geval_avg, rag_avg, duration_ms, est_cost]`, each with a `delta_polarity: "normal"|"invert"` field (invert for duration + cost; matches `compare.js:85–92`)

### Implementation

- [ ] T430 [US5] Create `src/holodeck/dashboard/compare_data.py` with `run_stats`, `compute_case_matrix`, `compute_config_diff`, `compute_compare_callouts`, `compute_summary_rows`, and `COMPARE_PALETTE = ["#7bff5a", "#5ae0a6", "#ffcf5a"]` constant (baseline, run-1, run-2)
- [ ] T431 [US5] Expose a single helper `delta_pill_class(value, *, invert=False) -> str` returning `"hd-delta-pos"|"hd-delta-neg"|"hd-delta-neutral"` so the UI can spray CSS classes consistently (matches `compare.js::deltaClass`)

---

## Phase 6: Compare — Streamlit view

Reference: open `Evaluation Dashboard.html`, click **Compare** tab, try all three layout variants.

- [ ] T432 [US5] Replace `src/holodeck/dashboard/views/compare.py` stub with `render_compare(runs)`:
    1. Resolve `queue = st.session_state.compare_queue` to concrete `EvalRun` instances (filter out any IDs no longer in `runs` — e.g., after filter change)
    2. If `len(queue_runs) < 2`, render the empty-state CTA (T433); otherwise dispatch to the variant renderer (T436/T437/T438) based on `st.session_state.compare_variant`
- [ ] T433 [US5] **Empty state** (`compare.js::CompareEmpty`):
    - Centered panel with an inline SVG of three offset rectangles in the palette colors (via `st.markdown(unsafe_allow_html=True)`)
    - `Pick runs to compare` heading
    - Explanation paragraph ("first-selected = baseline; others show deltas")
    - Two shortcut buttons: `Compare latest 2 runs` (primary) + `Compare latest 3 runs` (ghost) — each populates the queue from `sorted(runs, key=created_at, reverse=True)[:n]`
- [ ] T434 [US5] **Toolbar** (when ≥ 2 runs selected): left side eyebrow `COMPARE` + heading `<N> runs · baseline <version-in-palette[0]-color>`; right side `layout` label + `st.segmented_control("layout", options=["side-by-side","baseline + deltas","matrix-first"], key="compare_variant_label")` + `Clear` button that empties the queue
- [ ] T435 [US5] **Compare tray** (floating, rendered near the TOP of `app.py` when `len(compare_queue) > 0`, visible across all tabs — README "Compare tray"):
    - Move this out of views/compare.py into `src/holodeck/dashboard/components/compare_tray.py` so Summary, Explorer, and Compare all call it
    - Layout: `st.container(border=True)` with CSS making it sticky-ish via `position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);` (README acknowledges Streamlit has no native floater — this is the closest native approximation)
    - Shows up to 3 slot pills, first tagged `base`, each with a × remove button; empty slots say `slot N`
    - `Clear` button + `Open Compare →` primary button (disabled when fewer than 2 slots filled). `Open Compare` sets `st.session_state.tab = "Compare"` and calls `st.rerun()`
- [ ] T436 [US5] **Variant 1 — Side-by-side** (`compare.js::CompareV1`, ~180 LOC reference):
    - Column headers: one per run — dot in palette color + slot label (`BASELINE` / `RUN 1` / `RUN 2`) + timestamp + version-in-color + model name + commit (matches `RunSlotHeader`)
    - Summary block: eyebrow `SUMMARY` + heading `Headline stats` + description; then one row per entry from `compute_summary_rows`. Each non-baseline cell shows the value + a delta pill (pp for rates, raw for scores, inverted for duration/cost). Use a manual `st.columns(1 + len(runs))` grid with custom CSS classes; alternatively a `st.dataframe` with `column_config` — manual grid wins on fidelity
    - Config diff block: eyebrow `CONFIG DIFF` + heading `What's different?`; one row per `compute_config_diff` entry. Cells that differ from baseline render with amber left-border (class `.hd-cfg-cell--different`) + a `changed` badge
    - Case matrix block (T439) rendered at the bottom
- [ ] T437 [US5] **Variant 2 — Baseline + deltas** (`compare.js::CompareV2`):
    - Two-column grid `st.columns([1.4, 1, 1])` where baseline takes 1.4fr
    - Baseline card: slot label + big version + timestamp + model/temp/commit + big pass-rate number (mono, accent color) + subtitle `pass rate · <passed>/<total>` + small 2x2 grid of `geval / rag / dur / cost` + tag chip row
    - Each delta card: slot label + version + timestamp + model (with `changed` badge if different from baseline) + 5 delta rows (`pass rate`, `geval`, `rag`, `duration`, `cost`) — each row `label + value + DeltaPill(value - baseline.value, invert?)`
    - Case matrix below
- [ ] T438 [US5] **Variant 3 — Matrix-first** (`compare.js::CompareV3`):
    - Compact strip: `st.columns(len(runs))`, each cell a mini card with dot + slot-label + version + model + big pass-rate + mini `geval / rag / duration` triple
    - Callouts block: per non-baseline run, card listing first 3 regressions + first 3 improvements with `+N` overflow and fallback "No case-level changes from baseline"
    - Case matrix takes visual precedence below
- [ ] T439 [US5] **Case matrix** (shared across all 3 variants; reference `compare.js::CaseMatrix`):
    - Render via **Plotly heatmap** (`plotly.graph_objects.Heatmap`): rows = case names, columns = runs, `z = score_matrix`, colorscale `[(0, "#ff9d7e"), (0.5, "#1c2b25"), (1, "#7bff5a")]` (coral → dark → green gradient matching handoff)
    - Cell text: `"✓ 0.89"` or `"✕ 0.42"` via `fig.update_traces(texttemplate="%{text}", text=text_matrix, textfont_color="#050b09")`
    - Regression/improvement outlines: iterate the regression/improvement flag DataFrame from T426 and add `fig.add_shape(type="rect", ...)` per flagged cell — coral dashed outline for regressions, green dashed for improvements (matches `compare.js:229` CSS rings)
    - Legend row below the chart: 4 swatches — pass, fail, regression (coral dashed outline), improvement (green dashed outline)
    - Height: `max(400, 36 * len(case_names))`
- [ ] T440 [US5] Add a "Remove from queue" action on each column header's `×` button (available across all 3 variants) — reuses US4 `state.remove_from_compare_queue`
- [ ] T441 [US5] Add `+` / slot-indicator column to the US4 Summary runs table that also appears in the Explorer runs list (T413) — both need to let users build the compare queue without leaving their current view. Centralise the `+` button widget in `src/holodeck/dashboard/components/compare_add_button.py` (matches `compare.js::CompareAddButton`)

---

## Phase 7: Wiring + smoke tests

- [ ] T442 [US5] In `app.py`, replace the US4 compare-tray placeholder with `compare_tray.render(runs)` (T435) — shown near the top of the page on every tab when queue is non-empty
- [ ] T443 [US5] In Summary's runs table (US4 T361), replace the placeholder `+` CheckboxColumn with the real `CompareAddButton` (T441), so the user journey `Summary click + → Compare tray appears → Open Compare → variants render` works end-to-end
- [ ] T444 [US5] In Explorer's runs column (T413), wire the same `CompareAddButton` on each run row
- [ ] T445 [US5] (optional, slow) Extend `tests/integration/dashboard/test_app_smoke.py`:
    - Navigate to Explorer with `HOLODECK_DASHBOARD_USE_SEED=1` + `st.session_state.explorer_run_id="run-020"`; assert the detail panel's case list appears, first case detail is visible, no exceptions
    - Seed `st.session_state.compare_queue = ["run-018","run-020","run-023"]`, switch to Compare tab, assert all three variants render without exception (toggle `compare_variant` through 1→2→3 via `AppTest.segmented_control`)
    - Assert the case matrix Plotly figure is present (`stPlotlyChart` node count ≥ 1 on Compare tab)

---

## Phase 8: Visual fidelity — Chrome MCP side-by-side inspection (Explorer + Compare)

**Why this phase exists**: Explorer and Compare have the most complex layouts in the app. `AppTest` confirms components render; only a real browser side-by-side with the prototype confirms visual match. The HTML prototype is the ground truth (§Primary Source of Truth, item 2).

**Setup** (same as US4 Phase 8):
- Terminal A: `holodeck test view --seed` → `http://localhost:8501`
- Terminal B: `python -m http.server 8000 -d specs/031-eval-runs-dashboard/design_handoff_holodeck_eval_dashboard` → `http://localhost:8000/Evaluation%20Dashboard.html`

### Explorer parity (T446–T453)

- [ ] T446 [US5] **Open both tabs and baseline screenshots**: `mcp__claude-in-chrome__tabs_create_mcp` with the prototype URL, click the **Explorer** tab button in the prototype header via `mcp__claude-in-chrome__click`, `take_screenshot` → `visual-baselines/prototype-explorer.png`. Then `tabs_create_mcp` with `http://localhost:8501/?tab=explorer`, `take_screenshot` → `visual-baselines/streamlit-explorer-v1.png`
- [ ] T447 [US5] **3-column layout parity** — confirm Explorer shows 3 columns with widths matching the handoff: Runs (340px when expanded, 48px when collapsed) · Cases (340px) · Detail (flex-1). Use `mcp__claude-in-chrome__javascript_tool` to run `document.querySelectorAll('.explorer > *').forEach(e => console.log(e.getBoundingClientRect().width))` on both tabs; values on the Streamlit side should be within ±15px of the prototype. If off, iterate on T412
- [ ] T448 [US5] **Runs column collapse** — on the Streamlit tab, `click` the collapse arrow; screenshot at 48px; expand again; screenshot. Confirm vertical rotated `RUNS <count>` text appears in collapsed state (matches `explorer.js:48–62`). Fix T414 if missing
- [ ] T449 [US5] **Case detail: section sequence parity** — `get_page_text` on the Streamlit detail panel; extract the five eyebrow labels in order. MUST be: `[case header]` (no eyebrow — pass/fail + name) → `AGENT CONFIG SNAPSHOT` → `CONVERSATION` → `EXPECTED TOOLS` → `EVALUATIONS`. Wrong order = T417–T421 sequencing bug. Reference: `explorer.js:210–335`
- [ ] T450 [US5] **Conversation thread: tool-call panels** — expand the conversation section; confirm every tool-call panel has: `▸` caret + `TOOL` badge + `name()` + `<size>B` byte count in the header row; expands to show `args` (full JSON) and `result` (collapsed with `Expand (<size>B)` button when size > 500B); amber-tinted panel background. Reference: `explorer.js::ToolCall:152–184`. Screenshot, diff against prototype, fix T419 if the amber tint or collapse button is missing
- [ ] T451 [US5] **Expected-tools indicators** — confirm check/cross icons render in accent green (`#7bff5a`) / coral (`#ff9d7e`) respectively, with `called` / `not invoked` note text, and the top-right of the expander shows `N/M matched` pill (pass style when all matched). Reference: `explorer.js:285–302`
- [ ] T452 [US5] **Evaluations: group order + reasoning expander** — confirm eyebrow labels `GEVAL`, `RAG`, `STANDARD` appear in that order (T407 guarantees this) and that G-Eval metric rows have an expandable reasoning block below the score. Screenshot with reasoning expanded
- [ ] T453 [US5] **Drill-in click count (SC-006)** — from a fresh Summary load, measure clicks to reach a test-case detail:
    1. `click` a row in Summary table
    2. `click` a case in Explorer cases column
    3. (case detail already visible)
    Total ≤ 3 clicks. If the cases column requires an extra click (e.g. a run-selector step in-between), the default-case fallback in T411 is broken

### Compare parity (T454–T462)

- [ ] T454 [US5] **Populate compare queue from Summary** — on the Streamlit tab, go to Summary; click the `+` button on 3 rows via `mcp__claude-in-chrome__click`. Confirm the floating compare tray appears with 3 slot pills, first tagged `base`, and that the `Open Compare →` button enables. Reference: `compare.js::CompareTray:512–549`. Screenshot the tray → `visual-baselines/streamlit-compare-tray.png`; diff against the prototype tray (`compare.js` docs)
- [ ] T455 [US5] **Tray is sticky across tabs** — switch to Explorer (`click` the Explorer tab). Confirm the tray remains visible and functional. Switch to Compare. Confirm the tray remains visible. If the tray disappears on tab switch, T442 (`app.py` wiring) is wrong
- [ ] T456 [US5] **Empty state CTA** — clear the queue via the tray's `Clear` button; open the Compare tab. Confirm the empty-state panel shows the SVG three-rectangles icon + headline "Pick runs to compare" + two CTAs (`Compare latest 2 runs` primary, `Compare latest 3 runs` ghost). Reference: `compare.js::CompareEmpty:435–457`. Click `Compare latest 2 runs` — queue should populate with the 2 newest runs and variant-1 rendering should appear
- [ ] T457 [US5] **Variant toolbar parity** — `get_page_text` on the Compare toolbar. Confirm: eyebrow `COMPARE` + `<N> runs · baseline <version>` + `layout` label + segmented control with three options `side-by-side`, `baseline + deltas`, `matrix-first` + `Clear` button. Reference: `compare.js:484–501`
- [ ] T458 [US5] **Variant 1 — Side-by-side** — set `compare_variant=1` by clicking the first segmented option; `take_screenshot` of the full page. Confirm:
    - Column headers per run with palette dot, slot label, timestamp, version-in-color, model, commit
    - Summary block with 6 rows (pass rate, passed, geval, rag, duration, cost) and delta pills on non-baseline cells
    - Config diff block with differing cells having amber left-border + `changed` badge
    - Case matrix heatmap at bottom
    Save → `visual-baselines/streamlit-compare-v1.png`; compare with the prototype's V1 screenshot
- [ ] T459 [US5] **Variant 2 — Baseline + deltas** — click segmented option 2; screenshot. Confirm the baseline card is visually emphasized (1.4fr grid; large pass-rate number in accent), delta cards are compact with 5 delta-rows each. Reference: `compare.js::CompareV2:253–343`
- [ ] T460 [US5] **Variant 3 — Matrix-first** — click segmented option 3; screenshot. Confirm compact run-card strip at top + callouts block (regressions/improvements listed by case name with `+N` overflow) + matrix dominant below. Reference: `compare.js::CompareV3:347–431`
- [ ] T461 [US5] **Case matrix heatmap parity** — on any variant, hover over a cell via `mcp__claude-in-chrome__hover` — confirm Plotly tooltip shows `case_name / run_label / score / pass|fail`. Confirm regression cells have a coral dashed outer outline and improvement cells a green dashed outline. Reference: `compare.js::CaseMatrix:182–250`. If outlines are missing, fix T439
- [ ] T462 [US5] **Delta polarity sanity** — pick two runs where run[1] has LONGER duration than baseline; confirm the duration delta pill shows coral (delta-neg) because duration polarity is inverted (lower = better). Same for est. cost. Reference: `compare.js::deltaClass:17–21`. If polarity is wrong, fix T431

### Global sweep (T463–T466)

- [ ] T463 [US5] **Console cleanliness** — `mcp__claude-in-chrome__read_console_messages` with `pattern: "(error|warning)"` on the Streamlit tab while navigating Summary → Explorer → Compare and toggling all 3 variants. Zero errors from our modules. Any React-style Streamlit warnings from the framework itself are acceptable
- [ ] T464 [US5] **Navigation record** using `mcp__claude-in-chrome__gif_creator`: name it `dashboard_tour.gif` — record the journey from Summary → click row → Explorer with case selected → add `+` to queue x3 → Compare tab → toggle all three variants. Commit alongside `visual-baselines/`
- [ ] T465 [US5] **URL-state shareability test** — with a filter applied on Summary AND a case open in Explorer AND a compare queue populated, copy the full URL via `mcp__claude-in-chrome__javascript_tool` running `window.location.href`. Open that URL in a new tab (`tabs_create_mcp`). Verify the app restores: filter state, Explorer's selected run+case, and the compare queue. If the compare queue does not survive, session-state persistence (README "State") needs work
- [ ] T466 [US5] **Accessibility check** — `mcp__chrome-devtools__lighthouse_audit` on the Streamlit app; confirm Accessibility score ≥ 85 (contrast on the terminal-green theme is the likely risk area). File follow-up tasks for any blocker-level issues; do not block US5 merge on this, but capture the score

**Outputs of Phase 8**: `visual-baselines/` directory contains prototype + Streamlit screenshots for each view/variant; `dashboard_tour.gif` documents the UX; any deltas logged and resolved against T411–T441.

---

## Dependencies

- US4 fully complete: `app.py`, theme, state helpers, seed data, Summary view, CLI `view` command.
- T401–T407 (Explorer TDD) blocks T408–T410.
- T408–T410 blocks T411–T423.
- T424–T429 (Compare TDD) blocks T430–T431.
- T430–T431 blocks T432–T440.
- T441 (`CompareAddButton` component) blocks T443, T444 (Summary + Explorer use it).
- T435 (compare tray component) blocks T442.

### Parallel Opportunities

```bash
# Data-assembly TDD — Explorer and Compare are independent:
Task: "tests/unit/dashboard/test_explorer_data.py (T401–T407)"
Task: "tests/unit/dashboard/test_compare_data.py (T424–T429)"

# View modules run in parallel (separate files):
Task: "src/holodeck/dashboard/views/explorer.py (T411–T423)"
Task: "src/holodeck/dashboard/views/compare.py (T432–T440)"
Task: "src/holodeck/dashboard/components/compare_tray.py (T435)"
Task: "src/holodeck/dashboard/components/compare_add_button.py (T441)"
```

---

## Acceptance Scenario Traceability

### US5 (Explorer — Spec AC1–AC8)

| AC | Covered by |
|---|---|
| AC1 (click run → list cases) | T402, T413, T415 |
| AC2 (agent config snapshot in detail) | T403, T418 |
| AC3 (chat-style conversation) | T419 |
| AC4 (tool calls with formatted JSON args + result) | T404, T419 |
| AC5 (expected tools with match indicators) | T405, T420 |
| AC6 (metric results with reasoning) | T421 |
| AC7 (errors displayed prominently) | covered in T417 case header for per-case errors |
| AC8 (large results collapsed) | T404 (`large` flag), T410, T419 |

### Design-handoff Compare (additive to spec, introduced by handoff README §3)

| Handoff requirement | Covered by |
|---|---|
| 3 layout variants switchable via segmented control | T434, T436, T437, T438 |
| Compare queue max 3, baseline = first added | T435, US4 T345–T346 |
| Delta pills with inverted polarity for duration + cost | T431, T436, T437 |
| Config-diff highlighting (amber left-border, `changed` badge) | T436 (T427 supplies data) |
| Heatmap case matrix with regression/improvement outlines | T425, T426, T439 |
| Floating Compare tray visible across tabs | T435, T442 |
| `+` button for run rows that cycles "add / in-slot-N / remove" | T441, T443, T444 |
| "Compare latest 2 / latest 3" empty-state shortcuts | T433 |

---

## Implementation Strategy

### Recommended order (matches user directive: scaffold → Summary → **Explorer** → **Compare**)

1. **Explorer first** (T401–T423). Smaller surface (1 file, ~300 LOC equivalent), uses only the seed data and US4 state.
   - Port `explorer.js` section by section: case detail panel → runs column → cases column → wiring. Verify each section against the HTML prototype before moving on.
2. **Compare second** (T424–T444). Larger surface but orthogonal to Explorer.
   - Build data layer (T424–T431) — fully TDD'd.
   - Ship the floating Compare tray (T435, T442) first so users can build queues from Summary even before the Compare view itself lands.
   - Add empty state + toolbar (T433, T434).
   - Build variants in order: **V1 side-by-side → V2 baseline+deltas → V3 matrix-first**. Each variant reuses the shared case-matrix heatmap (T439).
3. **Wire compare buttons into Summary + Explorer** (T441, T443, T444) — the `+` must be everywhere the user sees a run.
4. **Smoke tests** (T445).
5. **Visual fidelity sweep via Chrome MCP** (T446–T466) — every view and variant diffed against the HTML prototype in a real browser. No visual delta is acceptable at US5 merge.

### Parallel team strategy

Two developers can split cleanly:
- **Dev A**: Explorer (T401–T423) + Compare tray (T435)
- **Dev B**: Compare data layer + variants (T424–T440)

They converge on the wiring step (T441–T444).

### Visual fidelity checklist

This checklist is enforced via the Phase 8 Chrome MCP tasks (T446–T466). For each item below, capture before/after screenshots via `mcp__claude-in-chrome__take_screenshot` with the HTML prototype and the Streamlit app loaded side-by-side. Commit both under `specs/031-eval-runs-dashboard/visual-baselines/`. No item may remain un-ticked at US5 merge:
- [ ] Colors match (accent `#7bff5a`, fail `#ff9d7e`, warn `#ffcf5a`)
- [ ] Mono font used for all numeric values
- [ ] Eyebrow labels uppercase, 10px, `.15em` letter-spacing, accent-soft color
- [ ] Card borders + gradient backgrounds
- [ ] Pills use correct tier colors (green ≥ 85%, yellow 65–85%, coral < 65%)
- [ ] Compare palette dots are visually distinct (`#7bff5a`, `#5ae0a6`, `#ffcf5a`)
- [ ] Delta pills use correct polarity (inverted for duration/cost)
- [ ] Regression dots on Summary pass-rate chart are coral with dark outline
- [ ] Heatmap case matrix gradient matches coral→dark→green scheme
- [ ] Regression/improvement ring outlines are dashed, not solid
