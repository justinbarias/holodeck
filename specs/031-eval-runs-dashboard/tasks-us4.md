---
description: "Task list — User Story 4: Test View Dashboard - Scaffold + Summary View (Dash framework)"
---

# Tasks — US4: Test View Dashboard — Scaffold + Summary View (Priority: P2)

> **Framework: Dash (Plotly).** An earlier draft of this list targeted Streamlit; the visual deltas against the handoff were unacceptable. Dash is component-driven (React under the hood), lets us paste the handoff's CSS tokens verbatim into `assets/`, exposes `html.Div`/`dcc` primitives that map 1:1 to the prototype's DOM, and ships `dash_table.DataTable` with per-cell conditional styling so the runs-table pill+bar cell no longer needs a workaround. Plotly is already a direct dependency so no chart-library churn.

## ⭐ Primary Source of Truth

**The design handoff bundle is AUTHORITATIVE for every visual, interaction, data-shape, and copy decision in this task list.** When spec.md and the handoff differ, **the handoff wins**. The handoff is the user-validated experience; spec.md captures functional requirements but does not prescribe visual fidelity.

Always consult, in this order of precedence:

1. **[design_handoff_holodeck_eval_dashboard/README.md](./design_handoff_holodeck_eval_dashboard/README.md)** — design tokens, data model, view layouts. (Ignore any "Streamlit notes" — we're on Dash now.)
2. **[design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html](./design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html)** — open in a browser; the interactive prototype is the ground-truth reference for every pixel
3. **[design_handoff_holodeck_eval_dashboard/styles.js](./design_handoff_holodeck_eval_dashboard/styles.js)** — exact CSS values (colors, spacings, gradients, shadows). Copied verbatim into `assets/holodeck.css`.
4. **[design_handoff_holodeck_eval_dashboard/summary.js](./design_handoff_holodeck_eval_dashboard/summary.js)** — Summary view React component source (chart maths, regression detection, filter semantics, KPI layout). Since Dash is React-backed, components map almost 1:1.
5. **[design_handoff_holodeck_eval_dashboard/data.js](./design_handoff_holodeck_eval_dashboard/data.js)** — the canonical seed dataset and data-shape reference
6. Our `spec.md` / `plan.md` / `data-model.md` — functional requirements; apply when silent on visuals

**Rule for every implementation task below**: before writing code, open the corresponding handoff file in a viewer. Every code task references the specific handoff line ranges that define its output. Treat those references as **mandatory reading**, not footnotes.

---

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 4 (P2)
**Plan**: [plan.md](./plan.md)
**Research**: [research.md](./research.md) — R2, R6, R7
**Contract**: [contracts/cli.md](./contracts/cli.md) — `holodeck test view` (to be updated from Streamlit → Dash in T342a)

**Goal**: Scaffold the entire Dash dashboard app (shell, design tokens in `assets/`, client-side stores for state, seed data) AND deliver the first of three views — **Summary**. Summary reproduces the handoff's `summary.js`: KPI strip, pass-rate-over-time area chart (Plotly, with coral regression dots + dashed prompt-version boundaries), per-metric trend chart (Plotly, `standard|rag|geval` radio control, threshold line at 0.7), three breakdown panels (Standard / RAG / G-Eval horizontal bars), and a filterable runs table. Explorer and Compare (US5) mount into the same scaffold as additional tab layouts.

**Independent Test**: Run `holodeck test view` → Dash app serves on `http://127.0.0.1:8501/` → three tabs visible (`Summary`, `Explorer`, `Compare`) → Summary renders with the `data.py` seed dataset (24 runs × 12 cases) before any real `results/` exist → KPI strip, pass-rate chart, metric trend chart, three breakdown panels, and runs table all render with the terminal-green theme. Switching the seed data source to `results/<slug>/` via env var loads real `EvalRun` JSON files instead.

**TDD discipline**: UI assembly (Dash layout builders + callback wiring) IS unit-testable because `layout()` returns a tree of plain Python `html.*`/`dcc.*` objects and callbacks are plain functions. Tests assert the component tree shape (children count, className, id) and callback return values. Visual fidelity is still verified against the HTML prototype via Chrome MCP (Phase 8). The DATA layer (`data_loader.py`, `filters.py`, `seed_data.py`, the port of `data.js`) IS unit-tested — tasks marked "(TDD)" write failing tests first.

**Dependency**: US1 must be merged so the `EvalRun` Pydantic model exists and can validate both real runs and seed-data instances.

---

## Phase 1: Setup & Optional Extra

- [ ] T301 [US4] Add optional extra `dashboard` to `pyproject.toml`: `dash>=2.17,<3.0`, `plotly>=5.20,<6.0`, `pandas>=2.0`. Dash is chosen over Streamlit for pixel-fidelity to the handoff — raw `html.Div`+`className` accepts the handoff's CSS tokens verbatim; `dash_table.DataTable` supports per-cell conditional styling; `dcc.Location`+`dcc.Store` give native URL-query and client-side state without `st.session_state` ceremony. Keep `dash[testing]` out of the core extra; anyone who wants browser-driven tests adds it separately. Run `uv lock`
- [ ] T302 [US4] Create package tree under `src/holodeck/dashboard/`: `__init__.py`, `__main__.py` (argparse entry; calls `app.run`), `app.py` (Dash instance + layout + callback registration), `data_loader.py`, `seed_data.py`, `filters.py`, `charts.py`, `state.py` (callback helpers for `dcc.Store` payloads + URL-param serde), `views/__init__.py`, `views/summary.py`, `views/explorer.py` (US5 placeholder), `views/compare.py` (US5 placeholder), `assets/tokens.css` (CSS variables from handoff), `assets/holodeck.css` (component styles). Module docstring on `dashboard/__init__.py`: **"Dash dashboard for the eval-runs viewer. Imported ONLY when the `dashboard` extra is installed. Nothing outside this package may import from here. NOTE: This package is distinct from `holodeck.lib.ui/` (which contains terminal/CLI rendering utilities — `colors.py`, `spinner.py`, `terminal.py`). The two packages render different surfaces (Dash HTML/React vs. ANSI TTY) and must not cross-import."** Reciprocal note already exists on `src/holodeck/lib/ui/__init__.py`.
- [ ] T303 [US4] Add a ruff per-file ignore (or top-level `# noqa`) letting `src/holodeck/dashboard/**` import `dash`, `plotly`, `pandas` unconditionally — the package itself is guarded upstream by the CLI's `find_spec` check

---

## Phase 2: Foundational — Click group conversion + seed data port

> **Framework-agnostic — unchanged from prior draft.** The Click group and seed-data port have nothing to do with the UI framework. Listed here for completeness.

### Click group

- [ ] T304 [US4] Convert `src/holodeck/cli/commands/test.py` from a single `@click.command()` to a `@click.group(invoke_without_command=True)`, moving today's test logic into the default callback so `holodeck test agent.yaml` still works identically (plan.md §"Key architecture decisions"). Wire `test.add_command(view)` to register the subcommand
- [ ] T305 [US4] (TDD) `tests/unit/cli/test_test_group_preserves_default.py`: assert `holodeck test agent.yaml` exit code and side effects are unchanged; assert `holodeck test --help` documents the `view` subcommand

### Seed data: port `data.js` → `data.py` (TDD)

The design-handoff dataset (24 runs, 7 prompt versions, pass-rate trajectory 0.58→0.93) is the golden fixture for UI development before any real `EvalRun` files exist. It also doubles as the backing data for pytest-level component-tree smoke tests.

- [ ] T306 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: `build_seed_runs() -> list[EvalRun]` returns exactly 24 instances, each passing `EvalRun.model_validate`
- [ ] T307 [P] [US4] (TDD) pass-rate trajectory matches the handoff's `trajectory` array exactly (`[0.58, 0.60, 0.62, 0.55, 0.52, 0.50, 0.63, 0.68, 0.72, 0.74, 0.76, 0.78, 0.80, 0.81, 0.79, 0.83, 0.85, 0.84, 0.88, 0.90, 0.89, 0.92, 0.91, 0.93]`)
- [ ] T308 [P] [US4] (TDD) prompt-version distribution — 7 distinct versions (`v1.0, v1.1, v1.2, v1.2.1, v1.3, v1.4, v2.0`), each covering ~4 consecutive runs (`data.js:54`)
- [ ] T309 [P] [US4] (TDD) model distribution — runs 0..13 use `claude-sonnet-4.5`; run 14+ alternates `gpt-4o` on `i % 3 == 0`, `claude-sonnet-4.5` otherwise (`data.js:55`)
- [ ] T310 [P] [US4] (TDD) every run has 12 test cases; `refund_eligible_standard` and `refund_outside_window` carry synthetic conversation payloads matching `data.js:161–178`
- [ ] T311 [P] [US4] (TDD) metric distribution — every case has one `geval` metric (`tone_of_voice|policy_compliance|escalation_appropriateness` cycled by case index); every even-indexed case has the five RAG sub-metrics; cases 0–2 also carry three standard metrics (bleu/rouge/meteor). Mirrors `data.js:63–98`
- [ ] T312 [P] [US4] (TDD) determinism — two calls to `build_seed_runs()` return equal `list[EvalRun]` (same hashes, timestamps, scores)
- [ ] T313 [US4] Implement `src/holodeck/dashboard/seed_data.py` porting `data.js`:
    - `_CASE_SEED(i, j)` → `math.sin((i*7 + j*3))` then `(sin + 1) / 2`
    - `_HASH(n)` → `abs(int(math.sin(n) * 1e8)) % int(1e6)` matches `caseHash` in `data.js:155`
    - Hard-code `trajectory`, `promptVersions`, `models`, `baseCases`, `RAG_METRICS`, `GEVAL_METRICS`, `STD_METRICS` verbatim from the handoff
    - Construct `EvalRun` / `EvalRunMetadata` / `PromptVersion` instances using US1/US2 models; attach handoff-only fields (e.g., `summary.duration_ms`, synthetic `git_commit`, nested `conversation` on `TestResult`) via `extra` dicts or the parallel `SEED_CONVERSATIONS: dict[str, dict]` constant
    - Expose `SEED_CONVERSATIONS` mapping case-name → `{user, assistant, tool_calls}` ported from `data.js:160–178` (Explorer/US5 consumes this)
    - Expose `build_seed_runs() -> list[EvalRun]` and `SEED_AGENT_DISPLAY_NAME = "customer-support"`
- [ ] T314 [US4] Add a fixture file `tests/fixtures/dashboard/seed_runs.json` generated once from `build_seed_runs()` (committed) so component-tree tests in US4/US5 load it from disk without recomputing

---

## Phase 3: Data layer (real + seed) — TDD

> **Framework-agnostic — unchanged from prior draft.** All aggregations return `pandas.DataFrame` objects consumed identically by Dash callbacks.

### Tests first

- [ ] T315 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `load_all(results_dir: Path) -> list[EvalRun]` — given a dir with three valid run files, returns 3 instances sorted by `report.timestamp` ascending
- [ ] T316 [P] [US4] (TDD) skip-on-corrupt — given 2 valid + 1 truncated JSON, returns 2 instances and logs ONE `WARNING` naming the skipped file (FR-024, R6)
- [ ] T317 [P] [US4] (TDD) schema-violation skip — given a file that parses as JSON but fails `EvalRun.model_validate_json`, skip with WARNING
- [ ] T318 [P] [US4] (TDD) empty dir / missing dir → returns `[]` without error (FR-023)
- [ ] T319 [P] [US4] (TDD) `load_runs_for_app() -> list[EvalRun]` — reads `HOLODECK_DASHBOARD_USE_SEED=1` → returns `build_seed_runs()`; otherwise reads `HOLODECK_DASHBOARD_RESULTS_DIR` → `load_all(results_dir)`
- [ ] T320 [P] [US4] (TDD) `to_summary_dataframe(runs) -> pandas.DataFrame` schema `[id, timestamp, pass_rate, passed, total, duration_ms, prompt_version, model_name, git_commit, tags, pass_rate_tier]`, one row per run, sorted newest-first. `pass_rate_tier` is `"pass"|"warn"|"fail"` per handoff thresholds (≥0.85/0.65–0.85/<0.65, `summary.js:360`) — used by `DataTable.style_data_conditional` to color the pass-rate cell natively (no emoji workaround needed)
- [ ] T321 [P] [US4] (TDD) `to_metric_trend_dataframe(runs, kind) -> DataFrame` — wide format (one column per metric name) for Plotly; per-run averages; threshold drawn separately (`summary.js::MetricTrendChart`)
- [ ] T322 [P] [US4] (TDD) `to_breakdown_dataframe(runs, kind, recent_n=6) -> DataFrame` — last N runs; columns `[metric_name, avg_score, pass_count, total]`; mirrors `summary.js::BreakdownPanel` rows
- [ ] T323 [P] [US4] (TDD) `detect_regressions(runs, drop_threshold=0.04) -> list[int]` — indices where `pass_rate[i] - pass_rate[i-1] < -0.04` (coral dots in `summary.js:173-175`)
- [ ] T324 [P] [US4] (TDD) `detect_version_boundaries(runs) -> list[tuple[int, str]]` — index + version at prompt-version change boundaries (dashed lines in `summary.js:177-183`)
- [ ] T325 [P] [US4] (TDD) `distinct_values(runs, field) -> list[str]` populates filter-option lists for `prompt_version`, `model_name`, `tags` (FR-028a)
- [ ] T326 [P] [US4] (TDD) `tests/unit/dashboard/test_filters.py`: `Filters` dataclass with fields `date_from, date_to, prompt_versions: list[str], model_names: list[str], min_pass_rate: float, tags: list[str], metric_kind: Literal["standard","rag","geval"]`; `apply(filters, runs) -> list[EvalRun]` AND-combines all non-empty fields
- [ ] T327 [P] [US4] (TDD) `filters_to_query_params(filters) -> dict[str, str]` and `filters_from_query_params(dict) -> Filters` round-trip with empty defaults (FR-028b, handoff shows `?versions=v1.3,v1.4&tags=rag-tuning` in `summary.js:108`)

### Implementation

- [ ] T328 [US4] Implement `src/holodeck/dashboard/data_loader.py`: all of the above. **Normalize codebase↔handoff shape drift in one place**:
    - `ReportSummary.pass_rate` is 0–100 in our model (`src/holodeck/models/test_result.py:127`); handoff expects 0..1 (`data.js:45-49`, `summary.js:171`). Divide by 100 when building the summary DataFrame. Document in the module docstring
    - `ReportSummary.total_duration_ms` (`test_result.py:128`) → handoff's `summary.duration_ms` (`data.js:117`). Emit both keys; prefer `duration_ms` downstream
    - Add tests: given `ReportSummary(pass_rate=87.5, total_duration_ms=19000)`, the DataFrame row has `pass_rate=0.875` and `duration_ms=19000`
- [ ] T329 [US4] Implement `src/holodeck/dashboard/filters.py`: `Filters` dataclass + `apply` + query-param serde. Multi-select fields comma-joined; booleans `"1"/"0"`. `min_pass_rate` stored 0..1

**Checkpoint**: `python -m pytest tests/unit/dashboard/` green; all aggregations produce DataFrames ready for `dcc.Graph` / `dash_table.DataTable`, pass_rate on 0..1 scale and duration under `duration_ms`.

---

## Phase 4: CLI subprocess boundary — TDD

### Tests first

- [ ] T330 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: with `importlib.util.find_spec` patched to return `None` for `"dash"`, `holodeck test view agent.yaml` prints the install hint (updated to say `dashboard` extra installs Dash, not Streamlit) to stderr, exits 2, no traceback (FR-022, SC-007)
- [ ] T331 [P] [US4] (TDD) with `find_spec` non-None and `subprocess.Popen` patched, argv is `[sys.executable, "-m", "holodeck.dashboard", "--port", <port>, "--host", "127.0.0.1"]`; env contains `HOLODECK_DASHBOARD_RESULTS_DIR`, `HOLODECK_DASHBOARD_AGENT_NAME`, `HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME`, and `HOLODECK_DASHBOARD_USE_SEED` (absent unless `--seed`)
- [ ] T332 [P] [US4] (TDD) `--seed` flag (dev-only, hidden) sets `HOLODECK_DASHBOARD_USE_SEED=1`
- [ ] T333 [P] [US4] (TDD) network-safety warning printed to stderr before Popen (FR-020). Copy text now says "Dash binds to 127.0.0.1 by default; if you override --host, firewall the port on shared infra."
- [ ] T334 [P] [US4] (TDD) results-dir resolves against `agent_base_dir`, not CWD
- [ ] T335 [P] [US4] (TDD) `--port 9000` forwarded; default 8501 (keep the 8501 default so the URL contract is stable across the Streamlit→Dash migration)
- [ ] T336 [P] [US4] (TDD) invalid `AGENT_CONFIG` exits 2 with single-line message, no traceback
- [ ] T337 [P] [US4] (TDD) Ctrl+C forwards SIGINT → `wait(timeout=5)` → `kill()` on timeout (research R2)

### Implementation

- [ ] T338 [US4] Create `src/holodeck/cli/commands/test_view.py`: `view` Click subcommand, positional `agent_config` default `"agent.yaml"`, options `--port` (default 8501), `--host` (default `127.0.0.1`), `--no-browser`, `--seed` (hidden)
- [ ] T339 [US4] Pre-flight: load agent config, resolve `agent_base_dir`, `slug = slugify(agent.name)`, `results_dir = agent_base_dir / "results" / slug`
- [ ] T340 [US4] Pre-flight: `if find_spec("dash") is None` → emit install-hint block → `raise click.exceptions.Exit(code=2)` — never import dash in the CLI module
- [ ] T341 [US4] Build argv + env + Popen + SIGINT forwarding per research R2; emit network warning first
- [ ] T342 [US4] Register `view` in `test.py` via `test.add_command(view)` (depends on T304)
- [ ] T342a [US4] Update `specs/031-eval-runs-dashboard/contracts/cli.md` from Streamlit to Dash: replace the `python -m streamlit run ...` argv block with `python -m holodeck.dashboard --port=<port> --host=127.0.0.1`; replace the "binds to 0.0.0.0" warning copy; update the error-taxonomy row (`streamlit not installed` → `dash not installed`); replace "Streamlit's own error" with "Dash / werkzeug's own error" for the port-in-use row. Leave invariants and exit codes unchanged
- [ ] T342b [US4] Implement `src/holodeck/dashboard/__main__.py`: argparse with `--port` (int), `--host` (str), `--debug` (flag, default False); imports `app` from `.app` and calls `app.run(host=args.host, port=args.port, debug=args.debug)`. Must not import `dash` at module top — all imports inside `main()` so that if someone runs `python -m holodeck.dashboard` without the extra installed they get a clean `ModuleNotFoundError` pointing at the `dashboard` extra

---

## Phase 5: App scaffold — shell, theme, state, navigation

> All of Phase 5 returns plain Python `dash.html.*`/`dash.dcc.*` objects. Layout builder functions are pure and testable; callbacks are plain functions registered via `@app.callback`.

- [ ] T343 [US4] Populate `src/holodeck/dashboard/assets/tokens.css` with the handoff's full token set — colors (`--hd-bg-body=#050b09`, `--hd-accent=#7bff5a`, `--hd-fg=#e8f5ec`, fail `#ff9d7e`, warn `#ffcf5a`, compare palette `[#7bff5a, #5ae0a6, #ffcf5a]`), typography (`Inter` 400/500/600/700, `JetBrains Mono` 400/500, sizes 10/11/12/13/15/16/18), radii (`--radius-sm=6px`, `-md=10px`, `-lg=16px`, `-pill=999px`), motion (`--dur-quick=140ms`, `--dur-base=220ms`, `--ease-standard=cubic-bezier(.2,.7,.3,1)`), and the page background radial gradient. Copy **verbatim** from `styles.js` — no translation needed. Dash auto-loads any CSS file in `assets/` on startup
- [ ] T344 [US4] Populate `src/holodeck/dashboard/assets/holodeck.css` with component styles lifted from `styles.js`:
    - `body { background: ...radial... ; color: var(--hd-fg); font-family: 'Inter', ...; }`
    - `.hd-eyebrow { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--hd-accent-soft); }`
    - `.hd-pill-pass { background: rgba(123,255,90,.12); color: var(--hd-accent); padding: 2px 8px; border-radius: var(--radius-pill); }` / `.hd-pill-warn` / `.hd-pill-fail`
    - `.hd-card { background: linear-gradient(rgba(10,17,15,.95), #070c0a 60%); border: 1px solid var(--hd-border); border-radius: var(--radius-lg); padding: 14px 18px; }`
    - `.hd-kpi-value { font-family: 'JetBrains Mono', ui-monospace; }`
    - `.hd-chip { ... }` / `.hd-chip--selected { ... }` for filter-rail chips
    - `.hd-tab-bar { display: flex; gap: 8px; }` / `.hd-tab[aria-selected="true"] { border-bottom: 2px solid var(--hd-accent); }`
    - `.hd-chat-user` / `.hd-chat-assistant` (Explorer, US5)
    - `.hd-delta-pos` / `.hd-delta-neg` (Compare, US5)
    - Override Dash's default white-on-white by setting `.dash-table-container .dash-spreadsheet-container { background: var(--hd-card); color: var(--hd-fg); }`
- [ ] T345 [US4] Create `src/holodeck/dashboard/state.py`: helpers that operate on `dcc.Store` payloads (plain JSON-serializable dicts). Functions: `default_state() -> dict` returns `{tab: "summary", filters: {...}, explorer_run_id: None, explorer_case_name: None, compare_queue: [], compare_variant: 1}`; `push_to_compare_queue(state, run_id) -> dict` (3-slot cap, returns new state); `remove_from_compare_queue(state, run_id) -> dict`; `open_in_explorer(state, run_id) -> dict` (sets `tab="explorer"`, `explorer_run_id=run_id`). All pure — no mutation, easy to unit test. Mirrors README "State" mapping.
- [ ] T346 [US4] `state.py`: URL sync — `url_search_from_state(state) -> str` (returns `?versions=...&tags=...`) and `state_from_url_search(search) -> dict` which delegate to `filters_from_query_params` + the `tab` query param. Used by the callback that binds `dcc.Location.search` ↔ `dcc.Store` state
- [ ] T347 [US4] Create `src/holodeck/dashboard/app.py`:
    1. `app = dash.Dash(__name__, title="HoloDeck · Evaluation Dashboard", update_title=None, suppress_callback_exceptions=True, assets_folder="assets")`
    2. `@functools.lru_cache(maxsize=1)` wrapper around `data_loader.load_runs_for_app()` keyed by env-var tuple so we don't re-parse on every callback. A separate `dcc.Interval(id="refresh", interval=60_000)` ticks the cache every 60s in real-mode to pick up new runs (FR equivalent of Streamlit's `cache_data(ttl=60)`)
    3. `app.layout = html.Div([dcc.Location(id="url"), dcc.Store(id="app-state", storage_type="memory"), header, tab_bar, html.Div(id="view-container")])`
    4. Header: `html.Header([html.Span("HoloDeck", className="hd-brand"), html.Span("·"), html.Span(AGENT_DISPLAY_NAME, className="hd-accent"), html.Span(sub_line, className="hd-muted")], className="hd-header")`
    5. Tab bar: `html.Nav([html.Button("Summary", id="tab-summary", className="hd-tab"), html.Button("Explorer", ...), html.Button("Compare", ...)], className="hd-tab-bar")` — plain buttons over `dcc.Tabs` so we can (a) badge the Compare tab with the queue count and (b) style the active underline via `aria-selected` without fighting Dash's default tab chrome
    6. Floating Compare tray (US5) mounts into a `html.Div(id="compare-tray")` visible when `state.compare_queue` is non-empty — placeholder in US4
    7. Main callback: `Input("app-state", "data") -> Output("view-container", "children")` dispatches to `render_summary(state, runs)` / `render_explorer(state, runs)` / `render_compare(state, runs)` based on `state["tab"]`
    8. URL-sync callback: bidirectional binding between `dcc.Location.search` and `dcc.Store` via `callback_context.triggered` to avoid loops
- [ ] T348 [US4] Create `src/holodeck/dashboard/views/__init__.py` exporting `render_summary`, `render_explorer` (stub in US4), `render_compare` (stub in US4)
- [ ] T349 [US4] Stub `src/holodeck/dashboard/views/explorer.py`: `render_explorer(state, runs) -> html.Div` returning `html.Div("Explorer — see US5", className="hd-empty-state")`
- [ ] T350 [US4] Stub `src/holodeck/dashboard/views/compare.py`: `render_compare(state, runs) -> html.Div` returning `html.Div("Compare — see US5", className="hd-empty-state")`

---

## Phase 6: Summary view — Plotly charts + breakdowns + runs table

Visual reference: `design_handoff_holodeck_eval_dashboard/summary.js` + open `Evaluation Dashboard.html` and click the **Summary** tab. Target "pixel-adjacent, not pixel-perfect" (README "Fidelity").

- [ ] T351 [US4] Create `src/holodeck/dashboard/charts.py` — pure functions returning `plotly.graph_objects.Figure`. **Unchanged from the Streamlit plan** — Plotly figures are the same regardless of host framework. Every chart calls `_apply_theme(fig)` setting `plot_bgcolor="#070c0a"`, `paper_bgcolor="rgba(0,0,0,0)"`, `font=dict(color="#e8f5ec", family="Inter, system-ui, sans-serif")`, `xaxis_gridcolor="rgba(28,43,37,.5)"`, `yaxis_gridcolor="rgba(28,43,37,.5)"`, `margin=dict(l=40, r=16, t=16, b=28)`
- [ ] T352 [US4] `charts.pass_rate_chart(runs) -> Figure`:
    - Filled area: `go.Scatter(fill="tozeroy", fillcolor="rgba(123,255,90,.15)")`
    - Main line `#7bff5a` width 2
    - Dots: normal `#7bff5a` r=3; regression `#ff9d7e` r=4.5 with `#050b09` outline (`detect_regressions`)
    - Dashed vertical lines at version boundaries via `fig.add_vline(x=boundary_ts, line_dash="dot", line_color="rgba(123,255,90,.18)", annotation_text=version, annotation_position="top")`
    - Y-axis 0–1 with 0.25 tick spacing, `%` format (`summary.js:194–200`)
    - X-axis datetime with `Apr 18` style labels
- [ ] T353 [US4] `charts.metric_trend_chart(runs, kind) -> Figure`:
    - One line per metric, palette `['#7bff5a','#5ae0a6','#53ff9c','#9bff5f','#a7f0ba','#ffcf5a']`
    - `fig.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,120,80,.7)", annotation_text="thresh 0.7")` (matches `summary.js:258`)
    - Legend right, single-column; Y 0.0–1.0
- [ ] T354 [US4] `charts.breakdown_bar(df, palette) -> Figure`:
    - Horizontal bar, one per metric; solid per-bar color from palette (Plotly doesn't render CSS gradients; single color is acceptable per README "Fidelity")
    - Threshold marker at x=0.7 via `add_vline(line_dash="dash")`
    - `height=max(180, 48 * len(df))`
- [ ] T355 [US4] Create `src/holodeck/dashboard/views/summary.py`. Top-level: `render_summary(state, runs) -> html.Div` returning a two-column flex layout — LEFT `html.Aside(filter_rail, className="hd-sidebar")` at ~18% width, RIGHT `html.Main(content_stack, className="hd-main")` at ~82% width. Use raw flexbox in CSS (no `dbc.Row/Col`) so the handoff's token spacings apply cleanly
- [ ] T356 [US4] **Filter rail** — reproduce `summary.js::FilterRail`:
    - Header row `html.H3("Filters")` + a `html.Button("Reset", id="filter-reset", className="hd-link")` that clears state via callback
    - `dcc.DatePickerRange(id="filter-date")` for date range
    - `dcc.Dropdown(id="filter-versions", multi=True, options=distinct_values(runs, "prompt_version"))` — restyle selected items as handoff-green chips via `.hd-chip` CSS
    - `dcc.Dropdown(id="filter-models", multi=True, ...)`
    - `dcc.Dropdown(id="filter-tags", multi=True, ...)`
    - `dcc.Slider(id="filter-min-pass", min=0, max=100, step=1, value=0, marks={0:"0%", 100:"100%"})`
    - `html.Pre(id="filter-url-preview", className="hd-code")` showing current query string + a `html.Button("Copy URL", id="filter-copy-url")` with a `dcc.Clipboard(target_id="filter-url-preview")`
    - Callback: any filter input → updates `dcc.Store` state → URL-sync callback emits `dcc.Location.search` (FR-028b)
- [ ] T357 [US4] **KPI strip** — reproduce `summary.js::KpiStrip`. Four cards in a flex row, each a `html.Div(className="hd-card hd-kpi")`:
    1. `Latest pass rate` — big `hd-kpi-value` number + `hd-kpi-delta` (▲/▼ vs prior) + inline sparkline via `dcc.Graph(figure=_sparkline(last_n), config={"displayModeBar": False, "staticPlot": True}, style={"height": "32px"})`
    2. `Runs (filtered)` — count + `6 wks` caption
    3. `Avg G-Eval score` — value `/ 1.00` + sparkline of last 8 G-Eval averages
    4. `Median duration` — e.g., `19.4s` + `per run` caption. **Compute `statistics.median(durations_ms)`**, not the mean. The handoff's reference JS labels the card `Median duration` but computes an average (`summary.js:123, 138`: `avgDur = ... / runs.length`) — that is a bug in the prototype. We honor the LABEL. Document this intentional deviation in `visual-baselines/README.md` and in the view docstring so Chrome MCP parity checks expect a value difference here
- [ ] T358 [US4] **Pass-rate panel** — `html.Div(className="hd-card")` containing:
    - `html.Div("TRENDS", className="hd-eyebrow")`
    - `html.H3("Pass rate over time")`
    - `html.P(f"{len(runs)} runs of {agent} · regressions flagged in coral · dashed lines mark prompt-version boundaries", className="hd-muted")`
    - Inline legend row (green swatch `pass rate` · coral swatch `regression`)
    - `dcc.Graph(id="chart-pass-rate", figure=pass_rate_chart(filtered_runs), config={"displayModeBar": False})`
- [ ] T359 [US4] **Metric trend panel** — `html.Div(className="hd-card")`:
    - Eyebrow `METRIC TRENDS` · title `Per-metric average scores` · subtitle with threshold note
    - `dcc.RadioItems(id="metric-kind", options=[{"label":"RAG","value":"rag"}, {"label":"G-Eval","value":"geval"}, {"label":"Standard","value":"standard"}], value="rag", className="hd-segmented", inline=True)` — style as segmented control via `.hd-segmented label { ... }` CSS
    - `dcc.Graph(id="chart-metric-trend")` driven by callback `Input("metric-kind","value") -> Output("chart-metric-trend","figure")` returning `metric_trend_chart(filtered_runs, kind)`
- [ ] T360 [US4] **Breakdown panels** — `html.Div([...three cards...], className="hd-grid-3")` with CSS grid `grid-template-columns: repeat(3, 1fr); gap: 16px;`. Each card: `html.Div([eyebrow, title, description, dcc.Graph(figure=breakdown_bar(df, palette))], className="hd-card")`:
    - `BREAKDOWN · STANDARD` · "NLP metrics" · palette `['#7bff5a','#5ae0a6','#9bff5f']` · kind `standard`
    - `BREAKDOWN · RAG` · "Retrieval & grounding" · palette full · kind `rag`
    - `BREAKDOWN · G-EVAL` · "Custom LLM judges" · palette `['#7bff5a','#ffcf5a','#5ae0a6']` · kind `geval`
- [ ] T361 [US4] **Runs table** — `dash_table.DataTable(id="runs-table", ...)`. **Dash DataTable lets us drop the Streamlit pill-string workaround entirely** — use `style_data_conditional` to render colored pills natively:
    - `columns=[{"name":"+","id":"queue","presentation":"markdown"}, {"name":"Timestamp","id":"timestamp"}, {"name":"Pass rate","id":"pass_rate_display"}, {"name":"Tests","id":"tests"}, {"name":"Prompt","id":"prompt_version"}, {"name":"Model","id":"model_name"}, {"name":"Duration","id":"duration"}, {"name":"Commit","id":"git_commit"}]`
    - `style_data_conditional=[{"if":{"filter_query":"{pass_rate_tier} = 'pass'","column_id":"pass_rate_display"},"backgroundColor":"rgba(123,255,90,.12)","color":"var(--hd-accent)"}, ...warn, ...fail]` — per-cell tiered pill colors, matching the handoff's in-cell pill. The `pass_rate_tier` hidden column (T320) drives the conditional
    - `style_header` + `style_cell` lifted from `holodeck.css` tokens
    - `row_selectable="single"`, `sort_action="native"`, `page_size=24`
    - `+` column: markdown cell rendering `"☑"` if run in compare queue else `"☐"`; clicking toggles queue membership via callback on `active_cell`
    - Row click: `Input("runs-table","active_cell") -> Output("app-state","data")` calls `state.open_in_explorer(state, run_id)` (README "Interactions: Run row click → Explorer")
    - Above the table: `html.Button("Export CSV")` + `dcc.Download(id="runs-csv")` with callback serving `summary_df.to_csv()`
- [ ] T362 [US4] **Empty state** — when `filtered_runs == []`, `render_summary` returns `html.Div([html.Span("∅"), html.H3("No runs match your filters"), html.P("Clear filters or run holodeck test.")], className="hd-empty-state")` centered. Mirrors `summary.js:394–403`
- [ ] T363 [US4] Wire filter changes: each control's value flows into `dcc.Store("app-state")` via a callback; a second callback binds state ↔ `dcc.Location.search` so refresh/share works (FR-028b). On app boot, `state_from_url_search(location.search)` seeds initial state

---

## Phase 7: Smoke / component-tree tests

> Dash lets us unit-test layout builders directly because they return plain Python objects. No Selenium required for the default suite. The `dash[testing]` extra (Selenium-driven) is optional and NOT wired into CI.

- [ ] T364 [US4] `tests/unit/dashboard/test_summary_layout.py`: import `render_summary`, call with `(default_state(), build_seed_runs())`, assert the returned tree contains exactly one `dash_table.DataTable`, exactly three `dcc.Graph` elements inside the breakdown grid, exactly one `dcc.Graph` with `id="chart-pass-rate"`, and exactly four elements with `className` containing `"hd-kpi"`. Assert no exceptions raised on empty runs list (the empty-state branch)
- [ ] T365 [US4] `tests/unit/dashboard/test_callbacks.py`: callbacks are plain functions decorated with `@app.callback`. Import them directly (or extract to standalone functions and register in `app.py`), call them with sample inputs, assert return structures. Cover: filter-change updates state; tab-click updates `state["tab"]`; runs-table row click produces the expected `state["explorer_run_id"]`; URL-sync round-trips
- [ ] T366 [US4] `tests/integration/dashboard/test_app_importable.py`: `pytest.importorskip("dash")`; import `holodeck.dashboard.app` and assert `app.layout` is a `dash.html.Div`. Marks the full Dash app boots without exceptions under seed mode (`monkeypatch.setenv("HOLODECK_DASHBOARD_USE_SEED","1")` before import). Mark `@pytest.mark.slow`; register the marker in `pyproject.toml` if not already present

---

## Phase 8: Visual fidelity — Chrome MCP side-by-side inspection

**Why this phase exists**: Component-tree tests verify structure but not pixels. Every task below uses Chrome MCP to drive a real browser, open the HTML prototype and the live Dash app side-by-side, screenshot, and diff. The prototype is ground truth.

**Setup**:
- Terminal A: `holodeck test view --seed` (launches Dash on `http://127.0.0.1:8501/` with seed data)
- Terminal B: `python -m http.server 8000 -d specs/031-eval-runs-dashboard/design_handoff_holodeck_eval_dashboard` → prototype at `http://localhost:8000/Evaluation%20Dashboard.html`

### Chrome MCP verification tasks

- [ ] T367 [US4] Open the prototype: `mcp__claude-in-chrome__tabs_context_mcp` → `tabs_create_mcp` with `url="http://localhost:8000/Evaluation%20Dashboard.html"` → wait for load → `take_screenshot`. Save as `specs/031-eval-runs-dashboard/visual-baselines/prototype-summary.png`
- [ ] T368 [US4] Open the Dash app: `tabs_create_mcp` with `url="http://127.0.0.1:8501/?tab=summary"` → wait for the main chart (`find` on `#chart-pass-rate`) → `take_screenshot`. Save as `specs/031-eval-runs-dashboard/visual-baselines/dash-summary-v1.png`
- [ ] T369 [US4] **KPI strip parity** — `get_page_text` both tabs, extract the four KPI labels (`Latest pass rate`, `Runs (filtered)`, `Avg G-Eval score`, `Median duration`). Labels must match exactly; numeric values must match within rounding EXCEPT for `Median duration` (intentional — median vs prototype's mean; see T357 + `visual-baselines/README.md`). Any other delta >0.1pp or any missing sparkline is a failing visual bug
- [ ] T370 [US4] **Pass-rate chart parity** — screenshot the chart region on both tabs (find DOM element, screenshot by element). Verify:
    - Regression points in coral at the same run indices (`detect_regressions`, `summary.js:173–175`)
    - Dashed vertical lines at the same prompt-version boundaries (`summary.js:177–183`)
    - Area gradient fades `rgba(123,255,90,.35) → transparent`
    - Y-axis 0/25/50/75/100%
    Iterate on T352 if deltas exist
- [ ] T371 [US4] **Metric-trend parity** — click each `rag`/`geval`/`standard` radio on both tabs via `mcp__claude-in-chrome__click`; confirm line count + colors match per kind, 0.7 dashed threshold present. If not, fix T353/T359
- [ ] T372 [US4] **Breakdown panels parity** — 3 panels in a row, each with eyebrow + title + description + bars. Hover (`mcp__claude-in-chrome__hover`) a bar; confirm Plotly tooltip content. If panels stack vertically, the `hd-grid-3` CSS in T344/T360 needs fixing
- [ ] T373 [US4] **Runs table parity** — column order exactly `[+] | Timestamp | Pass rate | Tests | Prompt | Model | Duration | Commit` (`summary.js:345–355`). Confirm the pass-rate cell renders a colored pill natively (green/yellow/coral) via `style_data_conditional` — this is the full fidelity win over the Streamlit draft, which had to fall back to emoji prefixes. Click a row, confirm `dcc.Location.search` updates and the Explorer tab receives focus with the correct `run_id` in the URL
- [ ] T374 [US4] **Filter rail parity** — left column shows: Date range · Prompt version chips · Model chips · Tag chips · Min pass rate slider · Share URL block. Apply a filter (click a chip via `click`); watch URL change via `javascript_tool` running `window.location.search`. URL MUST encode the filter (FR-028b)
- [ ] T375 [US4] **Theme parity** — `javascript_tool` runs `getComputedStyle(document.body).backgroundColor` on both tabs; values numerically close. Confirm `--hd-accent` resolves to `rgb(123, 255, 90)`. If not, iterate on T343–T344 (likely a `assets/*.css` load-order issue — Dash loads alphabetically, so `holodeck.css` should depend on `tokens.css` being first; rename to `01-tokens.css` / `02-holodeck.css` if necessary)
- [ ] T376 [US4] **Console cleanliness** — `read_console_messages` with `pattern: "(error|warning)"`. Zero errors; warnings only acceptable if they come from Dash/Plotly internals. Anything from our `app.py` / view modules must be cleared
- [ ] T377 [US4] **Responsive check** — `resize_page` to 1440×900, 1280×800, 1024×768; screenshot each. Confirm the three breakdown panels collapse gracefully (2+1 or stack on narrow viewports is acceptable as long as nothing clips). With Dash + CSS grid, `grid-template-columns` media queries in `holodeck.css` give us control Streamlit didn't
- [ ] T378 [US4] **Record a walkthrough** with `gif_creator` named `summary_walkthrough.gif`: load Summary → apply a version filter → toggle metric-trend radio → click a table row to jump to Explorer. Commit alongside `visual-baselines/`

**Outputs of Phase 8**: a `visual-baselines/` directory with prototype and Dash screenshots, a walkthrough GIF, and zero delta between the two on every parity check above. Any delta is a P0 bug against T351–T363.

---

## Dependencies

- US1 Phase 2b runtime-shape migrations BLOCK every real-run rendering path in US4. Specifically:
  - **Migration A — `MetricResult.kind`** (US1 T010a–T010d): required by breakdown panels (T360), metric-trend kind filter (T359), and `kind`-aware aggregations (T321, T322).
  - **Migration B — `TestResult.tool_invocations`** (US1 T010e–T010j): not directly consumed by US4 but required by US5.
  - **Migration C — `TestResult.token_usage`** (US1 T010k–T010m): Compare-view concern in US5.
  - The interleaved `conversation: list[ConversationTurn]` is deferred (US1 Phase 2b note). US4 is unaffected.
  - US4 work against the SEED dataset can proceed without any of these; real-run parity cannot.
- T301–T303 must complete first (extra installed, package tree exists).
- T304 blocks T305, T342.
- T306–T313 (seed data) blocks T319, T366.
- T315–T327 (data/filters TDD) blocks T328–T329.
- T330–T337 (CLI TDD) blocks T338–T342b.
- T328 blocks T352–T354, T357–T361 (charts + views consume `to_*_dataframe`).
- T343–T344 (CSS) blocks T347 (header/layout uses the classes).
- T345–T346 (state) blocks T347, T361 (table row-click navigation).
- T349–T350 stubs unblock US5 (which replaces them).
- T351 blocks T352–T354.

### Parallel Opportunities

```bash
# Phase 2 TDD — all independent test files:
Task: "tests/unit/dashboard/test_seed_data.py (T306–T312)"
Task: "tests/unit/cli/test_test_group_preserves_default.py (T305)"

# Phase 3 TDD:
Task: "tests/unit/dashboard/test_data_loader.py (T315–T325)"
Task: "tests/unit/dashboard/test_filters.py (T326, T327)"

# Phase 4 TDD:
Task: "tests/unit/cli/test_view_command.py (T330–T337)"

# Phase 5 CSS + state in parallel:
Task: "assets/tokens.css + holodeck.css (T343, T344)"
Task: "state.py (T345, T346)"

# Phase 6 — charts and view modules:
Task: "src/holodeck/dashboard/charts.py (T351–T354)"
Task: "src/holodeck/dashboard/views/summary.py (T355–T363)"
```

---

## Acceptance Scenario Traceability

| AC | Covered by |
|---|---|
| AC1 (auto-discovery + Summary tab) | T315, T319, T347, T355 |
| AC2 (pass-rate + per-metric trends) | T320, T321, T352, T353, T358, T359 |
| AC3 (standard/rag/geval breakdowns) | T322, T354, T360 |
| AC4 (per-experiment scoping) | T339, T334 |
| AC5 (empty state) | T318, T362 |
| AC6 (install hint, no traceback) | T330, T340 |
| AC7 (Ctrl+C clean shutdown) | T337, T341 |
| FR-028a (faceted filters compose) | T326, T356 |
| FR-028b (URL query string) | T327, T346, T356, T363 |
| **Design handoff Summary KPIs** | T357 |
| **Design handoff pass-rate chart (regression dots, version boundaries)** | T323, T324, T352, T358 |
| **Design handoff per-metric trend with kind toggle + threshold line** | T353, T359 |
| **Design handoff three breakdown panels** | T354, T360 |
| **Design handoff runs table w/ colored pill cells + row-click drilldown** | T361 |
| **Design handoff filter rail** | T356 |

---

## Implementation Strategy

### Recommended order (scaffold → Summary → Explorer → Compare)

1. **Setup + Foundational** (T301–T314): install deps, convert CLI to group, port `data.js` → `data.py`. Seed dataset lets every UI task iterate without real runs.
2. **Data layer** (T315–T329): TDD aggregations feeding every chart. Pure Pandas; fast to iterate.
3. **CLI subprocess boundary** (T330–T342b): TDD with `Popen` mocked; update `contracts/cli.md` to reflect Dash; add `--seed` so `holodeck test view --seed` works immediately.
4. **App scaffold** (T343–T350): CSS tokens + state + `app.py` three-tab shell. Launch with `holodeck test view --seed`; confirm all three tabs load with terminal-green aesthetic. Explorer and Compare are stubs.
5. **Summary view** (T351–T363): build each panel with the matching handoff file open. After each panel lands, run the matching Chrome MCP parity task (T369–T377) against the prototype. Fix deltas before the next panel — visual drift compounds.
6. **Component-tree tests** (T364–T366): structure sanity.
7. **Visual fidelity sweep** (T367–T378): Chrome MCP side-by-side. Gate-keeping before handoff to US5 — no visual deltas carry over.
8. Hand off to US5 (Explorer + Compare) — scaffold stubs already mounted, CSS + state + seed data fully validated.

### Parallel team strategy

Three developers can work in parallel after T314:
- **Dev A**: data layer (T315–T329) + CLI (T330–T342b)
- **Dev B**: CSS + state + app scaffold (T343–T350)
- **Dev C**: Plotly charts (T351–T354) — independent pure functions testable with seed data

Then merge and assemble Summary together (T355–T363).
