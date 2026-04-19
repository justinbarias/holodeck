---
description: "Task list — User Story 5: Explorer + Compare views (Dash framework)"
---

# Tasks — US5: Test View Dashboard — Explorer (drilldown) + Compare (3 variants) (Priority: P2)

> **Framework: Dash (Plotly).** An earlier draft of this list targeted Streamlit; visual deltas against the handoff were unacceptable and US4 has already been rebuilt on Dash (see `tasks-us4.md`). Explorer and Compare mount into the same `app.py` scaffold as additional view modules. All state references below assume the US4 `dcc.Store(id="app-state")` payload + pattern-match callbacks; there is no `st.session_state`.

## ⭐ Primary Source of Truth

**The design handoff bundle is AUTHORITATIVE for every visual, interaction, and layout decision in this task list.** When spec.md and the handoff differ, **the handoff wins**. Spec.md defines what the Explorer must contain functionally; the handoff defines how it must look, behave, and sequence. Compare is introduced by the handoff and has no spec counterpart — the handoff is the ONLY source of truth for that view.

Always consult, in this order of precedence:

1. **[design_handoff_holodeck_eval_dashboard/README.md](./design_handoff_holodeck_eval_dashboard/README.md)** — Views §2 (Explorer) + §3 (Compare) + §State mapping. (Ignore any "Streamlit notes" — we're on Dash now; map those to `dcc.Store` + `@callback`.)
2. **[design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html](./design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html)** — open in a browser; click the **Explorer** and **Compare** tabs and try every interaction before writing code
3. **[design_handoff_holodeck_eval_dashboard/explorer.js](./design_handoff_holodeck_eval_dashboard/explorer.js)** — Explorer component source (run list collapse, case list, collapsibles, tool-call rendering, JsonPretty highlighting, 500-byte collapse threshold). Since Dash is React-backed, components map almost 1:1 to `html.*`/`dcc.*` trees.
4. **[design_handoff_holodeck_eval_dashboard/compare.js](./design_handoff_holodeck_eval_dashboard/compare.js)** — Compare component source; the 3 variants, case matrix, delta pills, compare tray, `+` button lifecycle — ALL come from here
5. **[design_handoff_holodeck_eval_dashboard/styles.js](./design_handoff_holodeck_eval_dashboard/styles.js)** — exact CSS for chat bubbles, tool-call panels, config-diff highlighting, heatmap cells. Copied verbatim into `assets/02-holodeck.css` by US4 T344; extend (don't fork) that file here.
6. **[design_handoff_holodeck_eval_dashboard/data.js](./design_handoff_holodeck_eval_dashboard/data.js)** — the `sampleConversation` map (`data.js:160–178`) feeds the Explorer detail panel; US4 T313 ports this as `SEED_CONVERSATIONS`
7. Our `spec.md` / `plan.md` — only for FR-032 (the spec says ≥4KB collapses, the handoff says 500B — **the handoff wins** per T410)

**Rule for every implementation task below**: before writing code, open the corresponding handoff JS file and read the referenced line ranges. Every code task cites `explorer.js:nn-nn` or `compare.js:nn-nn` — treat those citations as mandatory reading.

---

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 5 (P2)
**Plan**: [plan.md](./plan.md)
**Contract**: [contracts/cli.md](./contracts/cli.md)

**Goal**: Replace the US4 stubs with two full views:

1. **Explorer** (`views/explorer.py`) — 3-column drilldown: runs list (340px, collapsible to 48px) → cases list (340px) → case detail panel (flex-1). Detail panel sections: case header, agent config snapshot (collapsible JSON + tool chips), conversation thread (chat bubbles + inline tool-call panels with collapse-when-large), expected-tools coverage, per-metric evaluations grouped by kind with judge reasoning.
2. **Compare** (`views/compare.py`) — select up to 3 runs via the floating Compare tray; render one of three layout variants (side-by-side / baseline+deltas / matrix-first) switched by a segmented control; Plotly heatmap case matrix with regression/improvement rings.

**Independent Test**:
- **Explorer**: From Summary, click any run row → Explorer tab opens with that run pre-selected (via `dcc.Store` state + `dcc.Location.search` sync) → cases list populated → click a case → detail panel renders agent config, conversation, tool calls (with collapse-when-bytes>500), expected-tools indicators, and evaluations grouped by kind. Total drill-in: ≤ 3 clicks (SC-006).
- **Compare**: From Summary table, click `+` on three runs → Compare tray shows `1/3, 2/3, 3/3` slots → click "Open Compare" → all three variants render via segmented control → each variant shows summary stats + config diff + case matrix with heatmap coloring and regression/improvement outlines.

**TDD discipline**: Data-assembly helpers (`select_run`, `build_case_detail`, `run_stats`, `compute_compare_callouts`, `case_matrix_dataframe`) are unit-tested. Dash layout builders return plain Python trees and ARE testable by shape/className/id. Visual fidelity is verified against the HTML prototype (`Evaluation Dashboard.html`) via Chrome MCP (Phase 8).

**Dependency**: US4 scaffolds `app.py`, assets/CSS, `state.py` helpers, seed data, and stub `views/explorer.py` + `views/compare.py`. US5 replaces the stubs and upgrades the US4 compare-tray placeholder in `app.py`.

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
- [ ] T404 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `ToolCallView` dataclass — `name: str`, `args: dict[str, Any]`, `result: Any`, `result_size_bytes: int` (copied from the persisted `ToolInvocation.bytes` when available, else computed via `len(json.dumps(result, default=str))` for legacy-mode fallback), `large: bool` (True when `result_size_bytes > 500` — handoff `explorer.js:156`), `duration_ms: int | None`, `error: str | None`. This view object projects one `ToolInvocation` (US1 Migration B) per row with the minimum data the Dash layout needs; it is NOT the persisted on-disk model.
- [ ] T405 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: expected-tools coverage — case-insensitive comparison against `tools_called`; returns both `matched` (configured + called) and `missed` (configured, not called) counts; optionally exposes `unexpected` (called but not configured)
- [ ] T406 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: missing/absent `conversations_map` key — `build_case_detail` falls back to the data.js pattern of using `refund_eligible_standard` as the default conversation (matches `explorer.js:197`); if neither present, returns an empty conversation without raising
- [ ] T407 [P] [US5] (TDD) `tests/unit/dashboard/test_explorer_data.py`: `evaluations` dict ordering — keys always iterate in `geval, rag, standard` order (matches handoff's detail-panel rendering order `explorer.js:204–208`), empty groups are omitted from the output, not left as empty lists

### Implementation

- [ ] T408 [US5] Create `src/holodeck/dashboard/explorer_data.py` with: `select_run`, `list_case_summaries`, `build_case_detail`, and dataclasses `CaseSummary`, `CaseDetail`, `ToolCallView`, `AgentSnapshot`, `MetricRow`
- [ ] T409 [US5] Data source precedence inside `build_case_detail`. The interleaved `ConversationTurn` union was originally drafted for US1 Migration C but has been **deferred** — the handoff's Explorer renders a simple user→tool_calls→assistant shape that does not need role-tagged turns (see US1 tasks-us1.md "Deferred — conversation discriminated union"). The three data-source paths are therefore:
    1. **Real-run mode (primary)**: consume `test_input` (user turn), `agent_response` (assistant turn), and `run.report.results[i].tool_invocations: list[ToolInvocation]` (US1 Migration B, US1 T010e–T010j). Each `ToolInvocation` maps 1:1 to a `ToolCallView` — the amber tool-call panel renders its `name`, `args`, `result`, and uses `bytes` for the large-result collapse at the `LARGE_TOOL_RESULT_BYTES` threshold (T410). `error` surfaces as a coral "Tool failed: <msg>" strip inside the panel.
    2. **Legacy real-run fallback**: when `tool_invocations` is empty but the name-only `tool_calls: list[str]` is not (runs persisted before US1 Migration B shipped), render chat bubbles from `test_input` + `agent_response` AND emit one amber panel per name with a muted "args/result not captured — re-run after upgrading to this HoloDeck version" subtitle.
    3. **Seed mode**: fall back to `seed_data.SEED_CONVERSATIONS[case_name]` (US4 T313) when both `tool_invocations` and `tool_calls` are empty. Dev-only path guarded by `HOLODECK_DASHBOARD_USE_SEED=1`.
    Document the precedence in `explorer_data.py`'s module docstring so reviewers see the data flow without tracing calls.
- [ ] T410 [US5] Centralise the tool-call large-result threshold constant `LARGE_TOOL_RESULT_BYTES = 500` (matches handoff, keeps US5 consistent with `explorer.js:156`; note this differs from FR-032's 4KB threshold — the handoff wins here because it's the user-validated experience)

---

## Phase 4: Explorer — Dash view

Reference: open `Evaluation Dashboard.html` in a browser, click **Explorer** tab. All layouts below return `html.Div` trees; interactions are wired via `@callback` with `Input/Output/State` + `dash.ALL` pattern-match IDs.

- [ ] T411 [US5] Replace `src/holodeck/dashboard/views/explorer.py` stub with `render_explorer(state, runs) -> html.Div`:
    1. Read `run_id = state.get("explorer_run_id")` (US4 `state.open_in_explorer` sets this when the user clicks a row in Summary); if `None`, default to newest run (`runs[-1]`)
    2. Read `case_name = state.get("explorer_case_name")`; if `None`, default to first case of the selected run
    3. Read `runs_collapsed = state.get("explorer_runs_collapsed", True)` for the runs-column collapse state (default `True` per `explorer.js:340`)
    4. All three values round-trip through `dcc.Location.search` via the US4 URL-sync callback so refresh/share works
- [ ] T412 [US5] **3-column flex layout**: `html.Div([col_runs, col_cases, col_detail], className="hd-explorer-grid")` with CSS `display: grid; grid-template-columns: 340px 340px 1fr; gap: 12px;` when expanded, `48px 340px 1fr` when collapsed. Swap class (`.hd-explorer-grid--collapsed`) based on `runs_collapsed` — Dash re-emits the tree on state change, no `st.rerun()` equivalent required
- [ ] T413 [US5] **Runs column** (expanded) in `col_runs`:
    - Header row `html.Div([html.Span("▸"), html.H4(f"Runs {len(runs)}")], id="explorer-runs-toggle", role="button")` — click toggles `state["explorer_runs_collapsed"]` via `@callback(Input("explorer-runs-toggle","n_clicks"), State("app-state","data"), Output("app-state","data"))`
    - Scroll wrapper: `html.Div(children=[...run rows...], className="hd-explorer-runs-scroll", style={"max-height":"700px","overflow-y":"auto"})`
    - Each run row: `html.Div([timestamp span, prompt-version span (accent color), pass-rate pill (`.hd-pill-pass|warn|fail`), passed/total span, model-suffix span, CompareAddButton (T441)], id={"type":"explorer-run-row","run_id":run.id}, className="hd-run-row" + " hd-run-row--active" if active)`. Model suffix = last two hyphen segments of `model.name` (matches `explorer.js:88`)
    - Row click (pattern-match): `@callback(Input({"type":"explorer-run-row","run_id":ALL},"n_clicks"), State("app-state","data"), Output("app-state","data"))` — sets `explorer_run_id`, clears `explorer_case_name` to auto-select first case (matches `explorer.js:357`)
- [ ] T414 [US5] **Runs column** (collapsed, 48px wide): `html.Div(html.Span(f"RUNS {len(runs)}", style={"writing-mode":"vertical-rl","transform":"rotate(180deg)"}), id="explorer-runs-toggle", className="hd-explorer-runs-collapsed")`. Same callback as T413 toggles back open. Matches `explorer.js:50–62`
- [ ] T415 [US5] **Cases column** in `col_cases`:
    - Header `html.Div([html.H4(f"Cases {len(cases)}"), html.Span(f"{pass_count} pass", className="hd-pill-pass hd-pill-sm")], className="hd-explorer-cases-head")`
    - Scroll wrapper `className="hd-explorer-cases-scroll"` with one row per case: `html.Div([icon, name span (mono), geval score span, rag-avg span muted], id={"type":"explorer-case-row","case_name":c.name}, className="hd-case-row" + active class)`. Icon `✓` in accent green for pass, `✕` in coral for fail
    - Row click (pattern-match) sets `state["explorer_case_name"]`
- [ ] T416 [US5] **Detail panel** in `col_detail` — `html.Div(children=[section1..5], className="hd-explorer-detail", style={"max-height":"700px","overflow-y":"auto"})`. Sections 2–5 are `html.Details([html.Summary(...), body])` (HTML-native collapsible, maps directly to handoff's Collapsible component `explorer.js:3`). Section 1 (case header) is always open
- [ ] T417 [US5] **Section 1: Case header** (always visible, not wrapped in `<details>`) — `html.Div([html.Div([pass/fail pill, html.H3(case.name)], className="hd-case-header-title"), html.Div([badge("run", ts), badge("prompt", version, accent=True), badge("model", name), badge("temp", val), badge("commit", sha)], className="hd-case-header-meta")])`. Each badge is `html.Span([html.Span(label, className="hd-eyebrow-inline"), html.Span(value)], className="hd-badge")`. Matches `explorer.js:212–224`
- [ ] T418 [US5] **Section 2: Agent config snapshot** — `html.Details([html.Summary("AGENT CONFIG SNAPSHOT · Configuration at run time"), body])` open=False by default:
    - Grid of key:value pairs via `html.Div(className="hd-cfg-grid")` with CSS `grid-template-columns: repeat(3, 1fr)`; rows show `model.provider`, `model.name`, `model.temperature`, `model.max_tokens`, `embedding.provider`, `embedding.name`, `claude.extended_thinking`, `prompt.version` (accent), `prompt.author`, `prompt.file_path`, `prompt.source`, `prompt.tags`
    - Tools subsection: eyebrow `TOOLS (n)` + chip row — each chip `html.Span([html.Span(kind, className="hd-muted"), html.Span(name)], className="hd-chip hd-chip--static")` (matches `explorer.js:249–257`)
    - Raw JSON drawer: nested `html.Details([html.Summary("View raw JSON"), html.Pre(json.dumps(agent_config_dict, indent=2), className="hd-code")])` where `api_key` and any `SecretStr` field is `"***"` (secrets redacted in the data layer — see T409's docstring requirement and README "Redact api_key")
- [ ] T419 [US5] **Section 3: Conversation thread** — `html.Details([html.Summary("CONVERSATION · Thread with tool calls"), body])`:
    - User turn: `html.Div([html.Span("USER", className="hd-eyebrow"), html.P(user_text)], className="hd-chat-user")`
    - Every tool call rendered inline BETWEEN user and assistant as `html.Div(children=[header, args_details, result_details], className="hd-tool-call")` — amber tint via CSS `background: rgba(255,207,90,.08); border-left: 3px solid var(--hd-warn);`. Header: `html.Div([html.Span("TOOL", className="hd-eyebrow-inline"), html.Code(f"{name}()"), html.Span(f"{size}B", className="hd-muted")], className="hd-tool-call-head")`. `args` rendered via `html.Details([html.Summary("args"), html.Pre(json.dumps(args, indent=2))])` always open; `result` via `html.Details([html.Summary(f"result{' — Expand ({size}B)' if large else ''}"), html.Pre(json.dumps(result, indent=2))])` with `open=not large` (matches `explorer.js::ToolCall`)
    - Assistant turn: `html.Div([html.Span(f"AGENT · {model.name}", className="hd-eyebrow"), html.P(assistant_text)], className="hd-chat-assistant")`
- [ ] T420 [US5] **Section 4: Expected tools** — `html.Details([html.Summary([...]), body])`:
    - Summary line has match pill on the right: `html.Summary(["EXPECTED TOOLS · Tool-call coverage", html.Span(f"{matched}/{total} matched", className="hd-pill-pass" if matched==total else "hd-pill-fail")])`
    - Body: list of rows, each `html.Div([icon, html.Code(name), html.Span("called" if was_called else "not invoked")], className="hd-tool-match hd-tool-match--ok|--miss")` with accent-tint / coral-tint backgrounds via CSS
    - "No expected tools configured" fallback when list empty (matches `explorer.js:292`)
- [ ] T421 [US5] **Section 5: Evaluations** — `html.Details([html.Summary("EVALUATIONS · Per-metric results"), body])`:
    - Iterate `geval, rag, standard` (order from T407); for each non-empty group emit `html.Div([html.Div(kind.upper(), className="hd-eyebrow"), *metric_rows])`
    - Each metric row: `html.Div([html.Div(m.name, className="hd-metric-name"), html.Div(f"{m.score:.2f}", className=f"hd-metric-score hd-mono{' hd-metric-score--fail' if not m.passed else ''}"), html.Div(f"≥ {m.threshold:.2f}", className="hd-mono hd-muted"), html.Span("pass" if m.passed else "fail", className=f"hd-pill-{'pass' if m.passed else 'fail'}")], className="hd-metric-row hd-metric-row--5")` using CSS grid `grid-template-columns: 1fr 80px 80px 60px`
    - If `m.reasoning` present, below the row render `html.Div(m.reasoning, className="hd-judge-reasoning")` — indented block with muted border (matches `explorer.js:314–332`)
- [ ] T422 [US5] **Empty state** — when `runs == []`: `html.Div([html.Span("∅", className="hd-empty-glyph"), html.H3("No runs found"), html.P(html.Code("Run holodeck test agent.yaml to generate one"))], className="hd-empty-state")` (matches `explorer.js:346–352`)
- [ ] T423 [US5] Wire navigation callbacks. All three live in `app.py` so they can share the `Output("app-state","data")` sink:
    - Run-row pattern-match click → set `explorer_run_id`, auto-select first case
    - Case-row pattern-match click → set `explorer_case_name`
    - Collapse toggle → flip `explorer_runs_collapsed`
    URL-sync callback (US4 T346) already encodes these into `dcc.Location.search`

---

## Phase 5: Compare — data assembly (TDD)

Reference: `design_handoff_holodeck_eval_dashboard/compare.js`.

### Tests first

- [ ] T424 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `run_stats(run) -> RunStats` dataclass with fields `pass_rate`, `passed`, `total`, `duration_ms`, `geval_avg`, `rag_avg`, `total_tokens`, `est_cost`. **Cost computation precedence**: (a) when every `TestResult.token_usage` (US1 Migration C) is populated, sum `input_tokens` and `output_tokens` across test results and multiply by the rate from a `PRICING_TABLE: dict[str, tuple[float, float]]` constant (per-model input/output USD-per-1M-tokens); (b) when any `token_usage` is `None` — legacy runs or backends that did not report tokens — fall back to the handoff's synthetic formula `duration_ms/1000 × rate` (rate 0.018 for sonnet, 0.012 otherwise; matches `compare.js:23–42`). `total_tokens` is `None` in the fallback path so the UI can surface "estimated from duration" instead of a token count. Aggregates match handoff arithmetic to 4 decimals when on the fallback path.
- [ ] T425 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_case_matrix(runs) -> DataFrame` — rows = union of case names across all runs sorted alphabetically, columns = one per run (`run_id, score, passed, regression: bool, improvement: bool`); score derived from first geval metric, else rag avg, else `1 if passed else 0` (matches `compare.js::caseScore`)
- [ ] T426 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: regression/improvement flags — for runs[1:], `regression = baseline.passed and not this.passed`, `improvement = (not baseline.passed) and this.passed` (matches `compare.js:225–226`)
- [ ] T427 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_config_diff(runs) -> list[ConfigRow]` — rows `[label, values_per_run, all_same: bool]` for `prompt_version, model_name, temperature, tags_joined, git_commit, extended_thinking` (matches `compare.js:94–101`)
- [ ] T428 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_compare_callouts(runs) -> list[Callout]` — for each non-baseline run, list up to 3 regressions and 3 improvements by case name with `+N` overflow count (matches `compare.js::CompareV3:353–364`)
- [ ] T429 [P] [US5] (TDD) `tests/unit/dashboard/test_compare_data.py`: `compute_summary_rows(runs) -> list[StatRow]` — returns rows for `[pass_rate, passed_ratio, geval_avg, rag_avg, duration_ms, total_tokens, est_cost]`, each with a `delta_polarity: "normal"|"invert"` field (invert for duration, total_tokens, and est_cost; matches `compare.js:85–92`). `total_tokens` is omitted from the row list entirely when all runs' `RunStats.total_tokens is None` (synthetic-only mode) — we don't surface a column with nothing but dashes.

### Implementation

- [ ] T430 [US5] Create `src/holodeck/dashboard/compare_data.py` with `run_stats`, `compute_case_matrix`, `compute_config_diff`, `compute_compare_callouts`, `compute_summary_rows`, the `COMPARE_PALETTE = ["#7bff5a", "#5ae0a6", "#ffcf5a"]` constant (baseline, run-1, run-2), AND a module-level `PRICING_TABLE: dict[str, tuple[float, float]]` mapping model name (e.g. `"claude-sonnet-4.5"`, `"gpt-4o"`) to `(input_usd_per_1M_tokens, output_usd_per_1M_tokens)`. Pricing values are best-effort documented placeholders — comment the source URL inline; update them when vendor pricing changes. `PRICING_TABLE` misses (unknown model) fall to the synthetic duration-based formula.
- [ ] T431 [US5] Expose a single helper `delta_pill_class(value, *, invert=False) -> str` returning `"hd-delta-pos"|"hd-delta-neg"|"hd-delta-neutral"` so layout builders can spray CSS classes consistently (matches `compare.js::deltaClass`). Used by T436/T437/T438 to render delta pill spans.

---

## Phase 6: Compare — Dash view

Reference: open `Evaluation Dashboard.html`, click **Compare** tab, try all three layout variants.

- [ ] T432 [US5] Replace `src/holodeck/dashboard/views/compare.py` stub with `render_compare(state, runs) -> html.Div`:
    1. Resolve `queue = state.get("compare_queue", [])` to concrete `EvalRun` instances (filter out any IDs no longer in `runs` — e.g., after filter change)
    2. If `len(queue_runs) < 2`, return the empty-state CTA (T433); otherwise dispatch to the variant renderer (T436/T437/T438) based on `state.get("compare_variant", 1)`
- [ ] T433 [US5] **Empty state** (`compare.js::CompareEmpty`):
    - `html.Div(children=[svg_icon, html.H3("Pick runs to compare"), html.P("first-selected = baseline; others show deltas"), html.Div([primary_btn, ghost_btn])], className="hd-compare-empty")`
    - `svg_icon` — inline SVG of three offset rectangles in palette colors via `html.Img` with a data URL OR `dash_svg`/`html.Div(dangerously_set_inner_html=...)` alternative. Cleanest path: embed a literal `dash.html.Div([html.Div(className="hd-compare-empty-rect", style={"background": color, ...})])` stack of three absolutely-positioned divs — avoids any HTML-injection machinery
    - Two shortcut buttons: `html.Button("Compare latest 2 runs", id="compare-quick-2", className="hd-btn hd-btn--primary")` + `html.Button("Compare latest 3 runs", id="compare-quick-3", className="hd-btn hd-btn--ghost")` — each wired to a callback that populates `state["compare_queue"]` from `sorted(runs, key=created_at, reverse=True)[:n]`
- [ ] T434 [US5] **Toolbar** (rendered at the top of Compare view when ≥ 2 runs selected): `html.Div([left_block, right_block], className="hd-compare-toolbar")`:
    - Left: `html.Span("COMPARE", className="hd-eyebrow")` + `html.H2([f"{N} runs · baseline ", html.Span(baseline.prompt_version, style={"color": COMPARE_PALETTE[0]})])`
    - Right: `html.Span("layout")` + `dcc.RadioItems(id="compare-variant", options=[{"label":"side-by-side","value":1},{"label":"baseline + deltas","value":2},{"label":"matrix-first","value":3}], value=state.get("compare_variant",1), className="hd-segmented", inline=True)` + `html.Button("Clear", id="compare-clear", className="hd-btn hd-btn--ghost")`
    - Variant-change callback: `Input("compare-variant","value") → State("app-state","data") → Output("app-state","data")` sets `state["compare_variant"]`
- [ ] T435 [US5] **Compare tray** (floating, always rendered near the TOP of `app.py` when `len(compare_queue) > 0`, visible across all tabs — README "Compare tray"):
    - Move this out of `views/compare.py` into `src/holodeck/dashboard/components/compare_tray.py` so Summary, Explorer, and Compare all mount the same layout
    - Layout: `html.Div([...slot pills..., html.Div([clear_btn, open_btn])], id="compare-tray", className="hd-compare-tray")` with CSS `position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 50;` — Dash + HTML gives us native `position: fixed` (Streamlit couldn't do this without an iframe hack)
    - Shows up to 3 slot pills, first tagged `base`, each with a `×` remove button (pattern-match id `{"type":"compare-slot-remove","run_id":rid}`); empty slots say `slot N` (ghost style)
    - `html.Button("Clear", id="compare-tray-clear")` + `html.Button("Open Compare →", id="compare-tray-open", className="hd-btn hd-btn--primary", disabled=len(queue)<2)`. `Open Compare` callback sets `state["tab"] = "compare"` via `state.open_in_compare(state)`; the US4 view-dispatch callback handles the re-render
- [ ] T436 [US5] **Variant 1 — Side-by-side** (`compare.js::CompareV1`, ~180 LOC reference):
    - `html.Div([header_row, summary_grid, config_grid, case_matrix], className="hd-compare-v1")`
    - Column headers: `html.Div([RunSlotHeader(run, i) for i, run in enumerate(runs)], className="hd-compare-headers")` — each header has a palette-colored dot, slot label (`BASELINE`/`RUN 1`/`RUN 2`), timestamp, version in palette color, model name, commit
    - Summary block: `html.Div([html.Span("SUMMARY", className="hd-eyebrow"), html.H3("Headline stats"), html.P("..."), summary_grid])` where `summary_grid` is a CSS grid with `grid-template-columns: 160px repeat(N, 1fr)`; one row per `compute_summary_rows` entry. Each non-baseline cell shows value + `html.Span(delta_text, className=delta_pill_class(delta, invert=row.invert))`
    - Config diff block: same grid layout; cells that differ from baseline render with `className="hd-cfg-cell hd-cfg-cell--different"` (amber left-border) + `html.Span("changed", className="hd-badge-changed")`
    - Case matrix block (T439) rendered at the bottom
- [ ] T437 [US5] **Variant 2 — Baseline + deltas** (`compare.js::CompareV2`):
    - `html.Div([baseline_card, *delta_cards, case_matrix], className="hd-compare-v2")` with CSS `grid-template-columns: 1.4fr 1fr 1fr`
    - Baseline card: slot label + big version (accent color) + timestamp + model/temp/commit + big pass-rate number (`hd-kpi-value hd-accent`) + subtitle `pass rate · <passed>/<total>` + small 2x2 grid of `geval / rag / dur / cost` + tag chip row
    - Each delta card: slot label + version + timestamp + model (with `changed` badge if different from baseline) + 5 delta rows (`pass rate`, `geval`, `rag`, `duration`, `cost`) — each row `html.Div([html.Span(label), html.Span(value), html.Span(delta_text, className=delta_pill_class(delta, invert=...))], className="hd-compare-delta-row")`
    - Case matrix below
- [ ] T438 [US5] **Variant 3 — Matrix-first** (`compare.js::CompareV3`):
    - `html.Div([compact_strip, callouts_block, case_matrix], className="hd-compare-v3")`
    - Compact strip: `html.Div([MiniRunCard(r, i) for i, r in enumerate(runs)], className="hd-compare-mini-strip")` with CSS `grid-template-columns: repeat(N, 1fr)` — each cell a mini card with palette dot + slot-label + version + model + big pass-rate + mini `geval / rag / duration` triple
    - Callouts block: `html.Div([CalloutCard(run, callouts) for run in non_baseline], className="hd-compare-callouts")` — each card lists first 3 regressions + first 3 improvements with `+N` overflow (`+{extra_count} more regressions`) and fallback "No case-level changes from baseline"
    - Case matrix takes visual precedence below
- [ ] T439 [US5] **Case matrix** (shared across all 3 variants; reference `compare.js::CaseMatrix`):
    - Render via **Plotly heatmap** in `dcc.Graph`: `plotly.graph_objects.Heatmap(x=run_labels, y=case_names, z=score_matrix, colorscale=[(0,"#ff9d7e"),(0.5,"#1c2b25"),(1,"#7bff5a")])` (coral → dark → green gradient matching handoff)
    - Cell text: `go.Heatmap(..., text=text_matrix, texttemplate="%{text}", textfont_color="#050b09")` where text is `"✓ 0.89"` / `"✕ 0.42"`
    - Regression/improvement outlines: iterate the regression/improvement flag DataFrame from T426 and `fig.add_shape(type="rect", x0=..., x1=..., y0=..., y1=..., line=dict(color="#ff9d7e", dash="dash", width=2))` per flagged cell — coral dashed for regressions, `"#7bff5a"` for improvements (matches `compare.js:229` CSS rings)
    - Legend row below: `html.Div([swatch("pass", accent), swatch("fail", coral), swatch("regression", coral_dashed), swatch("improvement", green_dashed)], className="hd-compare-matrix-legend")` — rendered as separate HTML so Plotly's native legend (which can't show dashed outline swatches) stays off
    - Height: `fig.update_layout(height=max(400, 36 * len(case_names)))`; `dcc.Graph(id="chart-compare-matrix", figure=fig, config={"displayModeBar": False})`
- [ ] T440 [US5] **Column-header `×` button** — available across all 3 variants. Each header row cell includes `html.Button("×", id={"type":"compare-header-remove","run_id":rid}, className="hd-btn-icon")`. Reuses the tray's remove callback (T435) — a single pattern-match callback on `Input({"type":"compare-slot-remove","run_id":ALL}, "n_clicks")` covers both surfaces
- [ ] T441 [US5] **`CompareAddButton` component** — shared widget placed in `src/holodeck/dashboard/components/compare_add_button.py`:
    - `render(run_id: str, queue: list[str]) -> html.Button` returns `html.Button(label, id={"type":"compare-add","run_id":run_id}, className="hd-cmp-add hd-cmp-add--in|out|full", title=tooltip)` where label is `"+"` when not in queue, `"1"`/`"2"`/`"3"` when in a slot (with slot index), disabled + `className` full when queue is full and run not already queued
    - Matches `compare.js::CompareAddButton` cycle: add → show slot index → click again removes → `+` returns
    - Single pattern-match callback: `@callback(Input({"type":"compare-add","run_id":ALL},"n_clicks"), State("app-state","data"), Output("app-state","data"))` calls `state.push_to_compare_queue` or `state.remove_from_compare_queue` based on current queue membership

---

## Phase 7: Wiring + smoke tests

- [ ] T442 [US5] In `app.py`, replace the US4 compare-tray placeholder with `compare_tray.render(state, runs)` (T435) — rendered above `html.Div(id="view-container")` on every view when queue is non-empty. Put it inside `app.layout` directly (not in a callback output) and use a separate callback `Input("app-state","data") → Output("compare-tray","style")` to toggle visibility (`display: none` when queue empty) — avoids re-rendering the whole tree on every state change
- [ ] T443 [US5] In Summary's runs table (US4 T361), replace the placeholder `+` column with the real `CompareAddButton` (T441). Since `dash_table.DataTable` can't host arbitrary Dash components in cells, render the runs table via `html.Table` (the US4 T361 implementation already landed on this path to avoid the DataTable markdown workaround) and inject `CompareAddButton.render(run_id, queue)` as the first cell of each row. The user journey `Summary click + → Compare tray appears → Open Compare → variants render` then works end-to-end via the shared pattern-match callback
- [ ] T444 [US5] In Explorer's runs column (T413), mount the same `CompareAddButton` on each run row — same callback, zero additional wiring
- [ ] T445 [US5] (optional, slow) Extend `tests/integration/dashboard/test_app_smoke.py`:
    - Import `holodeck.dashboard.app` under `monkeypatch.setenv("HOLODECK_DASHBOARD_USE_SEED","1")`
    - Call `render_explorer({"explorer_run_id": "run-020", ...}, build_seed_runs())`; assert the returned tree contains a `.hd-explorer-grid` class, 3 columns, and the detail panel has 5 sections (`html.Details` count + case header div == 5)
    - Call `render_compare({"compare_queue": ["run-018","run-020","run-023"], "compare_variant": 1}, runs)`; assert it returns a `.hd-compare-v1` tree and contains a `dcc.Graph` with `id="chart-compare-matrix"`. Repeat for variants 2 and 3 (`.hd-compare-v2`, `.hd-compare-v3`)
    - `@pytest.mark.slow`

---

## Phase 8: Visual fidelity — Chrome MCP side-by-side inspection (Explorer + Compare)

**Why this phase exists**: Explorer and Compare have the most complex layouts in the app. Component-tree tests confirm structure; only a real browser side-by-side with the prototype confirms visual match. The HTML prototype is the ground truth (§Primary Source of Truth, item 2).

**Setup** (same as US4 Phase 8):
- Terminal A: `holodeck test view --seed` → `http://127.0.0.1:8501`
- Terminal B: `python -m http.server 8000 -d specs/031-eval-runs-dashboard/design_handoff_holodeck_eval_dashboard` → `http://localhost:8000/Evaluation%20Dashboard.html`

### Explorer parity (T446–T453)

- [ ] T446 [US5] **Open both tabs and baseline screenshots**: `mcp__claude-in-chrome__tabs_create_mcp` with the prototype URL, click the **Explorer** tab button in the prototype header via `mcp__claude-in-chrome__click`, `take_screenshot` → `visual-baselines/prototype-explorer.png`. Then `tabs_create_mcp` with `http://127.0.0.1:8501/?tab=explorer`, `take_screenshot` → `visual-baselines/dash-explorer-v1.png`
- [ ] T447 [US5] **3-column layout parity** — confirm Explorer shows 3 columns with widths matching the handoff: Runs (340px when expanded, 48px when collapsed) · Cases (340px) · Detail (flex-1). Use `mcp__claude-in-chrome__javascript_tool` to run `document.querySelectorAll('.hd-explorer-grid > *').forEach(e => console.log(e.getBoundingClientRect().width))` on both tabs; values on the Dash side should match the prototype to the pixel (CSS grid columns are deterministic). If off, iterate on T412
- [ ] T448 [US5] **Runs column collapse** — on the Dash tab, `click` `#explorer-runs-toggle`; screenshot at 48px; click again to expand; screenshot. Confirm vertical rotated `RUNS <count>` text appears in collapsed state (matches `explorer.js:48–62`). Fix T414 if missing
- [ ] T449 [US5] **Case detail: section sequence parity** — `get_page_text` on the Dash detail panel (`.hd-explorer-detail`); extract the five eyebrow labels in order. MUST be: `[case header]` (no eyebrow — pass/fail + name) → `AGENT CONFIG SNAPSHOT` → `CONVERSATION` → `EXPECTED TOOLS` → `EVALUATIONS`. Wrong order = T417–T421 sequencing bug. Reference: `explorer.js:210–335`
- [ ] T450 [US5] **Conversation thread: tool-call panels** — click open the conversation `<details>`; confirm every `.hd-tool-call` panel has: caret (native `<summary>` marker) + `TOOL` eyebrow + `name()` monospace + `<size>B` byte count in the header row; `args` `<details>` open, `result` `<details>` collapsed with `Expand (<size>B)` summary text when size > 500B; amber-tinted panel background (`rgba(255,207,90,.08)` per T419 CSS). Reference: `explorer.js::ToolCall:152–184`. Screenshot, diff against prototype, fix T419 if the amber tint or collapse behavior is wrong
- [ ] T451 [US5] **Expected-tools indicators** — confirm check/cross glyphs render in accent green (`#7bff5a`) / coral (`#ff9d7e`) respectively, with `called` / `not invoked` note text, and the `<summary>` row shows the `N/M matched` pill on the right (via flex `justify-content: space-between`). Reference: `explorer.js:285–302`
- [ ] T452 [US5] **Evaluations: group order + reasoning block** — confirm eyebrow labels `GEVAL`, `RAG`, `STANDARD` appear in that order (T407 guarantees this) and that G-Eval metric rows have a `.hd-judge-reasoning` block below the score. Screenshot with reasoning visible
- [ ] T453 [US5] **Drill-in click count (SC-006)** — from a fresh Summary load, measure clicks to reach a test-case detail:
    1. `click` a run row in the Summary `<html.Table>` runs list
    2. `click` a case in Explorer cases column
    3. (case detail already visible — first case auto-selected by T413's "clears `explorer_case_name`" logic)
    Total ≤ 3 clicks. If the cases column requires an extra interaction step (e.g. the auto-first-case fallback from T411 is missing), the contract is broken

### Compare parity (T454–T462)

- [ ] T454 [US5] **Populate compare queue from Summary** — on the Dash tab, go to Summary; click `+` on 3 runs via `mcp__claude-in-chrome__click` on `[id^="{\"type\":\"compare-add\""]` selectors. Confirm the floating compare tray appears with 3 slot pills, first tagged `base`, and that the `Open Compare →` button enables. Reference: `compare.js::CompareTray:512–549`. Screenshot the tray → `visual-baselines/dash-compare-tray.png`; diff against the prototype tray
- [ ] T455 [US5] **Tray is sticky across tabs** — switch to Explorer (`click` the Explorer tab). Confirm the tray remains visible and functional (position: fixed + z-index: 50 survives view-container re-render because the tray is mounted OUTSIDE `#view-container`). Switch to Compare. Confirm the tray remains visible. If the tray disappears on tab switch, T442 (`app.py` layout position) is wrong — it must be a sibling of `#view-container`, not a child
- [ ] T456 [US5] **Empty state CTA** — clear the queue via the tray's `Clear` button; open the Compare tab. Confirm the empty-state panel shows the three-rectangles SVG (or the absolute-positioned div stack per T433) + headline "Pick runs to compare" + two CTAs (`Compare latest 2 runs` primary, `Compare latest 3 runs` ghost). Reference: `compare.js::CompareEmpty:435–457`. Click `Compare latest 2 runs` — queue should populate with the 2 newest runs and variant-1 rendering should appear
- [ ] T457 [US5] **Variant toolbar parity** — `get_page_text` on `.hd-compare-toolbar`. Confirm: eyebrow `COMPARE` + `<N> runs · baseline <version>` + `layout` label + radio group with three options `side-by-side`, `baseline + deltas`, `matrix-first` + `Clear` button. Reference: `compare.js:484–501`
- [ ] T458 [US5] **Variant 1 — Side-by-side** — set `compare-variant` radio to the first option via `click`; `take_screenshot` of the full page. Confirm:
    - Column headers per run with palette dot, slot label, timestamp, version-in-color, model, commit
    - Summary block with up to 7 rows (pass rate, passed, geval, rag, duration, total_tokens if available, cost) and delta pills on non-baseline cells
    - Config diff block with differing cells having amber left-border + `changed` badge
    - Case matrix Plotly heatmap at bottom (`dcc.Graph` `#chart-compare-matrix` present)
    Save → `visual-baselines/dash-compare-v1.png`; compare with the prototype's V1 screenshot
- [ ] T459 [US5] **Variant 2 — Baseline + deltas** — click radio option 2; screenshot. Confirm the baseline card is visually emphasized (1.4fr grid column; large pass-rate number in accent), delta cards are compact with 5 delta-rows each. Reference: `compare.js::CompareV2:253–343`
- [ ] T460 [US5] **Variant 3 — Matrix-first** — click radio option 3; screenshot. Confirm compact run-card strip at top + callouts block (regressions/improvements listed by case name with `+N` overflow) + matrix dominant below. Reference: `compare.js::CompareV3:347–431`
- [ ] T461 [US5] **Case matrix heatmap parity** — on any variant, hover over a cell via `mcp__claude-in-chrome__hover` — confirm Plotly tooltip shows `case_name / run_label / score / pass|fail`. Confirm regression cells have a coral dashed outer outline and improvement cells a green dashed outline. Reference: `compare.js::CaseMatrix:182–250`. If the outlines are missing, `fig.add_shape(..., line=dict(dash="dash"))` calls in T439 need review
- [ ] T462 [US5] **Delta polarity sanity** — pick two runs where run[1] has LONGER duration than baseline; confirm the duration delta pill shows coral (`hd-delta-neg`) because duration polarity is inverted (lower = better). Same for `est_cost` and (when present) `total_tokens`. Reference: `compare.js::deltaClass:17–21`. If polarity is wrong, fix T431. Additionally, confirm the cost cell shows a tooltip or caption indicating whether it was computed from `token_usage × PRICING_TABLE` (real) or from the synthetic `duration × rate` fallback — so users know when the number is provisional

### Global sweep (T463–T466)

- [ ] T463 [US5] **Console cleanliness** — `mcp__claude-in-chrome__read_console_messages` with `pattern: "(error|warning)"` on the Dash tab while navigating Summary → Explorer → Compare and toggling all 3 variants. Zero errors from our modules. Dash/werkzeug framework warnings are acceptable but should be reviewed
- [ ] T464 [US5] **Navigation record** using `mcp__claude-in-chrome__gif_creator`: name it `dashboard_tour.gif` — record the journey from Summary → click row → Explorer with case selected → add `+` to queue ×3 → Compare tab → toggle all three variants. Commit alongside `visual-baselines/`
- [ ] T465 [US5] **URL-state shareability test** — with a filter applied on Summary AND a case open in Explorer AND a compare queue populated, copy the full URL via `mcp__claude-in-chrome__javascript_tool` running `window.location.href`. Open that URL in a new tab (`tabs_create_mcp`). Verify the app restores: filter state, Explorer's selected run+case, and the compare queue. Requires that US4's URL-sync callback (T346) serializes the Explorer + Compare keys in addition to filter params — extend `url_search_from_state` / `state_from_url_search` if `explorer_run_id` / `explorer_case_name` / `compare_queue` don't round-trip. If the compare queue does not survive, the serde on the URL-sync side needs work
- [ ] T466 [US5] **Accessibility check** — `mcp__chrome-devtools__lighthouse_audit` on the Dash app; confirm Accessibility score ≥ 85 (contrast on the terminal-green theme is the likely risk area). File follow-up tasks for any blocker-level issues; do not block US5 merge on this, but capture the score

**Outputs of Phase 8**: `visual-baselines/` directory contains prototype + Dash screenshots for each view/variant; `dashboard_tour.gif` documents the UX; any deltas logged and resolved against T411–T441.

---

## Dependencies

- US4 fully complete: `app.py`, assets/CSS, `state.py` helpers, seed data, Summary view, CLI `view` command, `views/explorer.py` + `views/compare.py` stubs.
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
| Compare queue max 3, baseline = first added | T435, US4 T345 |
| Delta pills with inverted polarity for duration + cost | T431, T436, T437 |
| Config-diff highlighting (amber left-border, `changed` badge) | T436 (T427 supplies data) |
| Heatmap case matrix with regression/improvement outlines | T425, T426, T439 |
| Floating Compare tray visible across tabs | T435, T442 |
| `+` button for run rows that cycles "add / in-slot-N / remove" | T441, T443, T444 |
| "Compare latest 2 / latest 3" empty-state shortcuts | T433 |

---

## Implementation Strategy

### Recommended order (matches user directive: scaffold → Summary → **Explorer** → **Compare**)

1. **Explorer first** (T401–T423). Smaller surface (1 view module, ~300 LOC equivalent), uses only the seed data and US4 state.
   - Port `explorer.js` section by section: case detail panel → runs column → cases column → wiring. Verify each section against the HTML prototype before moving on. Because Dash layouts are pure functions over state, you can preview any section in isolation via `python -c "import ...; print(render_section(...))"`.
2. **Compare second** (T424–T444). Larger surface but orthogonal to Explorer.
   - Build data layer (T424–T431) — fully TDD'd.
   - Ship the floating Compare tray (T435, T442) first so users can build queues from Summary even before the Compare view itself lands.
   - Add empty state + toolbar (T433, T434).
   - Build variants in order: **V1 side-by-side → V2 baseline+deltas → V3 matrix-first**. Each variant reuses the shared case-matrix heatmap (T439).
3. **Wire compare buttons into Summary + Explorer** (T441, T443, T444) — the `+` must be everywhere the user sees a run. A single pattern-match callback handles all three surfaces.
4. **Smoke tests** (T445).
5. **Visual fidelity sweep via Chrome MCP** (T446–T466) — every view and variant diffed against the HTML prototype in a real browser. No visual delta is acceptable at US5 merge.

### Parallel team strategy

Two developers can split cleanly:
- **Dev A**: Explorer (T401–T423) + Compare tray (T435)
- **Dev B**: Compare data layer + variants (T424–T440)

They converge on the wiring step (T441–T444).

### Visual fidelity checklist

This checklist is enforced via the Phase 8 Chrome MCP tasks (T446–T466). For each item below, capture before/after screenshots via `mcp__claude-in-chrome__take_screenshot` with the HTML prototype and the Dash app loaded side-by-side. Commit both under `specs/031-eval-runs-dashboard/visual-baselines/`. No item may remain un-ticked at US5 merge:
- [ ] Colors match (accent `#7bff5a`, fail `#ff9d7e`, warn `#ffcf5a`)
- [ ] JetBrains Mono font used for all numeric values
- [ ] Eyebrow labels uppercase, 10px, `.15em` letter-spacing, accent-soft color
- [ ] Card borders + gradient backgrounds
- [ ] Pills use correct tier colors (green ≥ 85%, yellow 65–85%, coral < 65%)
- [ ] Compare palette dots are visually distinct (`#7bff5a`, `#5ae0a6`, `#ffcf5a`)
- [ ] Delta pills use correct polarity (inverted for duration/cost)
- [ ] Regression dots on Summary pass-rate chart are coral with dark outline
- [ ] Heatmap case matrix gradient matches coral→dark→green scheme
- [ ] Regression/improvement ring outlines are dashed, not solid
