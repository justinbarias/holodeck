---
description: "Task list — User Story 4: Test View Dashboard - Scaffold + Summary View"
---

# Tasks — US4: Test View Dashboard — Scaffold + Summary View (Priority: P2)

## ⭐ Primary Source of Truth

**The design handoff bundle is AUTHORITATIVE for every visual, interaction, data-shape, and copy decision in this task list.** When spec.md and the handoff differ, **the handoff wins**. The handoff is the user-validated experience; spec.md captures functional requirements but does not prescribe visual fidelity.

Always consult, in this order of precedence:

1. **[design_handoff_holodeck_eval_dashboard/README.md](./design_handoff_holodeck_eval_dashboard/README.md)** — design tokens, data model, view layouts, Streamlit-specific hints
2. **[design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html](./design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html)** — open in a browser; the interactive prototype is the ground-truth reference for every pixel
3. **[design_handoff_holodeck_eval_dashboard/styles.js](./design_handoff_holodeck_eval_dashboard/styles.js)** — exact CSS values (colors, spacings, gradients, shadows); when the README is ambiguous, read styles.js
4. **[design_handoff_holodeck_eval_dashboard/summary.js](./design_handoff_holodeck_eval_dashboard/summary.js)** — Summary view component source (chart maths, regression detection, filter semantics, KPI layout)
5. **[design_handoff_holodeck_eval_dashboard/data.js](./design_handoff_holodeck_eval_dashboard/data.js)** — the canonical seed dataset and data-shape reference
6. Our `spec.md` / `plan.md` / `data-model.md` — functional requirements; apply when silent on visuals

**Rule for every implementation task below**: before writing code, open the corresponding handoff file in a viewer. Every code task references the specific handoff line ranges that define its output. Treat those references as **mandatory reading**, not footnotes.

---

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 4 (P2)
**Plan**: [plan.md](./plan.md)
**Research**: [research.md](./research.md) — R2, R6, R7
**Contract**: [contracts/cli.md](./contracts/cli.md) — `holodeck test view`

**Goal**: Scaffold the entire Streamlit dashboard app (shell, design tokens, session state, seed data) AND deliver the first of three views — **Summary**. Summary reproduces the design handoff's `summary.js`: KPI strip, pass-rate-over-time area chart (Plotly, with coral regression dots + dashed prompt-version boundaries), per-metric trend chart (Plotly, `standard|rag|geval` segmented control, threshold line at 0.7), three breakdown panels (Standard / RAG / G-Eval horizontal bars), and a filterable runs table. Explorer and Compare (US5) mount into the same scaffold.

**Independent Test**: Run `holodeck test view` → the Streamlit app loads → three tabs are visible (`Summary`, `Explorer`, `Compare`) → Summary renders with the `data.py` seed dataset (24 runs × 12 cases) before any real `results/` exist → KPI strip, pass-rate chart, metric trend chart (Plotly), three breakdown panels, and runs table all render with the terminal-green theme. Switching the seed data source to `results/<slug>/` via env var loads real `EvalRun` JSON files instead.

**TDD discipline**: UI layer (Streamlit + Plotly chart assembly) is NOT unit-tested — verified visually against the HTML prototype (`design_handoff_holodeck_eval_dashboard/Evaluation Dashboard.html`) and via optional `streamlit.testing.v1.AppTest` smoke tests (research R7). The DATA layer (`data_loader.py`, `filters.py`, `seed_data.py`, the port of `data.js`) IS unit-tested — tasks marked "(TDD)" write failing tests first.

**Dependency**: US1 must be merged so the `EvalRun` Pydantic model exists and can validate both real runs and seed-data instances.

---

## Phase 1: Setup & Optional Extra

- [ ] T301 [US4] Add optional extra `dashboard` to `pyproject.toml`: `streamlit>=1.36,<2.0`, `plotly>=5.20,<6.0`, `pandas>=2.0`. Plotly is required (not Altair) because the handoff relies on dashed prompt-version boundary lines, coral regression dots, threshold horizontal rules, and heatmap rectangle annotations (README "Streamlit notes"). Run `uv lock`
- [ ] T302 [US4] Create package tree under `src/holodeck/dashboard/`: `__init__.py`, `app.py`, `data_loader.py`, `seed_data.py`, `filters.py`, `theme.py`, `charts.py`, `state.py`, `views/__init__.py`, `views/summary.py`, `views/explorer.py` (US5 placeholder), `views/compare.py` (US5 placeholder). Module docstring on `dashboard/__init__.py`: **"Streamlit dashboard for the eval-runs viewer. Imported ONLY when the `dashboard` extra is installed. Nothing outside this package may import from here. NOTE: This package is distinct from `holodeck.lib.ui/` (which contains terminal/CLI rendering utilities — `colors.py`, `spinner.py`, `terminal.py`). The two packages render different surfaces (Streamlit HTML vs. ANSI TTY) and must not cross-import."** Also add a reciprocal note to `src/holodeck/lib/ui/__init__.py` pointing at `holodeck.dashboard` for the browser-based UI.
- [ ] T303 [US4] Add a ruff per-file ignore (or top-level `# noqa`) letting `src/holodeck/dashboard/**` import `streamlit`, `plotly`, `pandas` unconditionally — the package itself is guarded upstream by the CLI's `find_spec` check

---

## Phase 2: Foundational — Click group conversion + seed data port

### Click group (same as previous US4 plan — unchanged scope, re-stated for completeness)

- [ ] T304 [US4] Convert `src/holodeck/cli/commands/test.py` from a single `@click.command()` to a `@click.group(invoke_without_command=True)`, moving today's test logic into the default callback so `holodeck test agent.yaml` still works identically (plan.md §"Key architecture decisions"). Wire `test.add_command(view)` to register the subcommand
- [ ] T305 [US4] (TDD) `tests/unit/cli/test_test_group_preserves_default.py`: assert `holodeck test agent.yaml` exit code and side effects are unchanged; assert `holodeck test --help` documents the `view` subcommand

### Seed data: port `data.js` → `data.py` (TDD)

The design-handoff dataset (24 runs, 7 prompt versions, pass-rate trajectory 0.58→0.93) is the golden fixture for UI development before any real `EvalRun` files exist. It also doubles as the backing data for `streamlit.testing.v1.AppTest` smoke tests.

- [ ] T306 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: `build_seed_runs() -> list[EvalRun]` returns exactly 24 instances, each passing `EvalRun.model_validate` — confirms the port produces valid Pydantic objects, not dicts
- [ ] T307 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: pass-rate trajectory matches the handoff's `trajectory` array exactly (`[0.58, 0.60, 0.62, 0.55, 0.52, 0.50, 0.63, 0.68, 0.72, 0.74, 0.76, 0.78, 0.80, 0.81, 0.79, 0.83, 0.85, 0.84, 0.88, 0.90, 0.89, 0.92, 0.91, 0.93]`)
- [ ] T308 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: prompt-version distribution — 7 distinct versions (`v1.0, v1.1, v1.2, v1.2.1, v1.3, v1.4, v2.0`), each covering ~4 consecutive runs (`data.js:54`, `Math.floor(i/4)`)
- [ ] T309 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: model distribution — runs 0..13 use `claude-sonnet-4.5`; run 14+ alternates `gpt-4o` on `i % 3 == 0`, `claude-sonnet-4.5` otherwise (`data.js:55`)
- [ ] T310 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: every run has 12 test cases; the `refund_eligible_standard` and `refund_outside_window` cases carry synthetic conversation payloads matching `data.js:161–178`
- [ ] T311 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: metric distribution — every case has one `geval` metric (`tone_of_voice|policy_compliance|escalation_appropriateness` cycled by case index); every even-indexed case has the five RAG sub-metrics; cases 0–2 also carry three standard metrics (bleu/rouge/meteor). Mirrors `data.js:63–98`
- [ ] T312 [P] [US4] (TDD) `tests/unit/dashboard/test_seed_data.py`: determinism — two calls to `build_seed_runs()` return equal `list[EvalRun]` (same hashes, timestamps, scores); needed so AppTest snapshot tests are stable
- [ ] T313 [US4] Implement `src/holodeck/dashboard/seed_data.py` porting `data.js`:
    - `_CASE_SEED(i, j)` → `math.sin((i*7 + j*3))` then `(sin + 1) / 2` matches the JS closure exactly
    - `_HASH(n)` → `abs(int(math.sin(n) * 1e8)) % int(1e6)` matches `caseHash` in data.js:155
    - Hard-code `trajectory`, `promptVersions`, `models`, `baseCases`, `RAG_METRICS`, `GEVAL_METRICS`, `STD_METRICS` verbatim from the handoff
    - Construct `EvalRun` / `EvalRunMetadata` / `PromptVersion` instances using the US1/US2 models; when a handoff field doesn't exist in our models (e.g., `summary.duration_ms`, synthetic `git_commit`, nested `conversation` on `TestResult`), attach it via `extra` dicts or a parallel `SEED_CONVERSATIONS: dict[str, dict]` constant exposed alongside runs
    - Expose `SEED_CONVERSATIONS` mapping case-name → `{user, assistant, tool_calls}` ported from `data.js:160–178` (Explorer/US5 consumes this)
    - Expose `build_seed_runs() -> list[EvalRun]` and a `SEED_AGENT_DISPLAY_NAME = "customer-support"` constant
- [ ] T314 [US4] Add a fixture file `tests/fixtures/dashboard/seed_runs.json` generated once from `build_seed_runs()` (committed) so AppTest-based smoke tests in US4 and US5 load it from disk without recomputing — reduces flakiness

---

## Phase 3: Data layer (real + seed) — TDD

### Tests first

- [ ] T315 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `load_all(results_dir: Path) -> list[EvalRun]` — given a directory with three valid run files, returns 3 instances sorted by `report.timestamp` ascending
- [ ] T316 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: skip-on-corrupt — given a dir with 2 valid files and 1 truncated JSON, returns 2 instances and logs ONE `WARNING` naming the skipped file (FR-024, R6)
- [ ] T317 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: schema-violation skip — given a file that parses as JSON but fails `EvalRun.model_validate_json`, skip with WARNING
- [ ] T318 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: empty dir / missing dir → returns `[]` without error (FR-023)
- [ ] T319 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `load_runs_for_app() -> list[EvalRun]` — reads `HOLODECK_DASHBOARD_USE_SEED=1` env var → returns `build_seed_runs()`; otherwise reads `HOLODECK_DASHBOARD_RESULTS_DIR` → `load_all(results_dir)`; lets the app boot in seed mode during development
- [ ] T320 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `to_summary_dataframe(runs) -> pandas.DataFrame` schema `[id, timestamp, pass_rate, passed, total, duration_ms, prompt_version, model_name, git_commit, tags]`, one row per run, sorted newest-first (FR-025, matches handoff's RunTable sort `data.js:327`)
- [ ] T321 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `to_metric_trend_dataframe(runs, kind) -> DataFrame` — wide format (one column per metric name) for Plotly line plotting; values are per-run averages; threshold line drawn at 0.7 separately (matches `summary.js::MetricTrendChart`)
- [ ] T322 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `to_breakdown_dataframe(runs, kind, recent_n=6) -> DataFrame` — last N runs; columns `[metric_name, avg_score, pass_count, total]`; mirrors `summary.js::BreakdownPanel` rows
- [ ] T323 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `detect_regressions(runs, drop_threshold=0.04) -> list[int]` — returns indices where `pass_rate[i] - pass_rate[i-1] < -0.04` (coral dots in `summary.js:173-175`)
- [ ] T324 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `detect_version_boundaries(runs) -> list[tuple[int, str]]` — index + version at prompt-version change boundaries (dashed lines in `summary.js:177-183`)
- [ ] T325 [P] [US4] (TDD) `tests/unit/dashboard/test_data_loader.py`: `distinct_values(runs, field) -> list[str]` populates filter-option lists for `prompt_version`, `model_name`, `tags` (FR-028a)
- [ ] T326 [P] [US4] (TDD) `tests/unit/dashboard/test_filters.py`: `Filters` dataclass with fields `date_from, date_to, prompt_versions: list[str], model_names: list[str], min_pass_rate: float, tags: list[str], metric_kind: Literal["standard","rag","geval"]`; `apply(filters, runs) -> list[EvalRun]` AND-combines all non-empty fields (matches `summary.js::FilterRail` semantics)
- [ ] T327 [P] [US4] (TDD) `tests/unit/dashboard/test_filters.py`: `filters_to_query_params(filters) -> dict[str, str]` and `filters_from_query_params(dict) -> Filters` round-trip with empty defaults (FR-028b, handoff shows `?versions=v1.3,v1.4&tags=rag-tuning` in `summary.js:108`)

### Implementation

- [ ] T328 [US4] Implement `src/holodeck/dashboard/data_loader.py`: `load_all`, `load_runs_for_app`, `to_summary_dataframe`, `to_metric_trend_dataframe`, `to_breakdown_dataframe`, `detect_regressions`, `detect_version_boundaries`, `distinct_values`. **Normalize codebase↔handoff shape drift in one place**:
    - `ReportSummary.pass_rate` is stored on a 0–100 scale (src/holodeck/models/test_result.py:127); handoff expects 0..1 (data.js:45-49, summary.js:171). Divide by 100 when building the summary DataFrame and any EvalRun-summary projection. Document the normalization in `data_loader.py`'s module docstring
    - `ReportSummary.total_duration_ms` (test_result.py:128) maps to the handoff's `summary.duration_ms` (data.js:117). Emit both keys when writing DataFrames — prefer `duration_ms` for downstream code to match the handoff + seed data; keep `total_duration_ms` as a hidden column only if a pipeline needs it
    - Add explicit tests covering both normalizations (extend T320/T323 coverage): given a `ReportSummary(pass_rate=87.5, total_duration_ms=19000)`, the DataFrame row has `pass_rate=0.875` and `duration_ms=19000`
- [ ] T329 [US4] Implement `src/holodeck/dashboard/filters.py`: `Filters` dataclass + `apply` + query-param serde. Use `dataclasses.asdict` + comma-joined lists for multi-select fields; booleans as `"1"/"0"`. The `min_pass_rate` field is 0..1 (handoff scale) — normalize consumers accordingly

**Checkpoint**: `python -m pytest tests/unit/dashboard/` green; all handoff-specific aggregations produce DataFrames ready for Plotly, with pass_rate on 0..1 scale and duration under the `duration_ms` key.

---

## Phase 4: CLI subprocess boundary — TDD

### Tests first

- [ ] T330 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: with `importlib.util.find_spec` patched to return `None` for `"streamlit"`, `holodeck test view agent.yaml` prints the install hint from contracts/cli.md to stderr, exits 2, no traceback (FR-022, SC-007)
- [ ] T331 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: with `find_spec` non-None and `subprocess.Popen` patched, argv is `[sys.executable, "-m", "streamlit", "run", <app_path>, "--server.port=<port>", "--server.headless=true", "--browser.gatherUsageStats=false"]`; env contains `HOLODECK_DASHBOARD_RESULTS_DIR`, `HOLODECK_DASHBOARD_AGENT_NAME`, `HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME`, and `HOLODECK_DASHBOARD_USE_SEED` (absent unless `--seed` flag passed)
- [ ] T332 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: `--seed` flag (dev-only, hidden) sets `HOLODECK_DASHBOARD_USE_SEED=1` — lets developers launch the app without any real runs against the handoff dataset
- [ ] T333 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: network-safety warning printed to stderr before Popen (FR-020)
- [ ] T334 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: results-dir resolves against `agent_base_dir`, not CWD
- [ ] T335 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: `--port 9000` forwarded as `--server.port=9000`; default 8501
- [ ] T336 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: invalid `AGENT_CONFIG` exits 2 with single-line message, no traceback
- [ ] T337 [P] [US4] (TDD) `tests/unit/cli/test_view_command.py`: Ctrl+C forwards SIGINT → `wait(timeout=5)` → `kill()` on timeout (research R2)

### Implementation

- [ ] T338 [US4] Create `src/holodeck/cli/commands/test_view.py`: `view` Click subcommand, positional `agent_config` default `"agent.yaml"`, options `--port` (default 8501), `--no-browser`, `--seed` (hidden)
- [ ] T339 [US4] Pre-flight: load agent config, resolve `agent_base_dir`, `slug = slugify(agent.name)`, `results_dir = agent_base_dir / "results" / slug`
- [ ] T340 [US4] Pre-flight: `if find_spec("streamlit") is None` → emit install-hint block → `raise click.exceptions.Exit(code=2)` — never import streamlit in the CLI module
- [ ] T341 [US4] Build argv + env + Popen + SIGINT forwarding per research R2; emit network warning first
- [ ] T342 [US4] Register `view` in `test.py` via `test.add_command(view)` (depends on T304)

---

## Phase 5: App scaffold — shell, theme, state, navigation

> UI-layer code below is verified visually against the handoff HTML prototype. No unit tests on these modules.

- [ ] T343 [US4] Create `src/holodeck/dashboard/theme.py`: `inject_theme()` function that emits a single `st.markdown(..., unsafe_allow_html=True)` with the handoff's full token set — colors (`--hd-bg-body=#050b09`, `--hd-accent=#7bff5a`, `--hd-fg=#e8f5ec`, fail `#ff9d7e`, warn `#ffcf5a`, compare palette `[#7bff5a, #5ae0a6, #ffcf5a]`), typography (`Inter` 400/500/600/700, `JetBrains Mono` 400/500, sizes 10/11/12/13/15/16/18), radii (`--radius-sm=6px`, `-md=10px`, `-lg=16px`, `-pill=999px`), motion (`--dur-quick=140ms`, `--dur-base=220ms`, `--ease-standard=cubic-bezier(.2,.7,.3,1)`), and the page background radial gradient from README "Page background radial" — copy verbatim. Source values from `design_handoff_holodeck_eval_dashboard/styles.js` where the README is ambiguous.
- [ ] T344 [US4] `theme.py`: CSS overrides to achieve "pixel-adjacent" fidelity — hide default Streamlit chrome (`#MainMenu`, footer), restyle `.stApp` with the gradient, monospace numeric text via `[data-testid="stMetricValue"] { font-family: "JetBrains Mono", ui-monospace; }`, eyebrow class `.hd-eyebrow { font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: var(--hd-accent-soft); }`, pill classes `.hd-pill-pass { background: rgba(123,255,90,.12); color: var(--hd-accent); }` / `.hd-pill-warn` / `.hd-pill-fail`, card styling, chat-bubble classes for Explorer, delta-pill classes for Compare
- [ ] T345 [US4] Create `src/holodeck/dashboard/state.py`: helpers reading/writing `st.session_state` for: `tab` (default from `st.query_params["tab"]`), `filters` (Filters dataclass), `explorer_run_id`, `explorer_case_name`, `compare_queue: list[str]` (max 3), `compare_variant: Literal[1,2,3]` — mirrors README "State (Streamlit mapping)"
- [ ] T346 [US4] `state.py`: `ensure_defaults()` function called at top of `app.py` once per run, `push_to_compare_queue(run_id)` / `remove_from_compare_queue(run_id)` with 3-slot cap, `open_in_explorer(run_id)` (sets tab="explorer" + explorer_run_id, calls `st.rerun()`)
- [ ] T347 [US4] Create `src/holodeck/dashboard/app.py`:
    1. Call `st.set_page_config(page_title="HoloDeck · Evaluation Dashboard", page_icon="▸", layout="wide", initial_sidebar_state="expanded")`
    2. Call `theme.inject_theme()`, `state.ensure_defaults()`
    3. Load runs via `data_loader.load_runs_for_app()`; cache via `@st.cache_data(ttl=60)` keyed by the env-var values so real-mode picks up new runs within a minute without full reload
    4. Render header: `HoloDeck · <AGENT_DISPLAY_NAME>` in `--hd-accent`, sub-line `<N> runs · <date range>`
    5. Navigation: `tab = st.radio("view", ["Summary", "Explorer", "Compare"], horizontal=True, label_visibility="collapsed", key="tab")` with CSS overrides giving the accent underline — `st.tabs` has no badge support (README "Streamlit notes"); radio is chosen so Compare can show a `(n/3)` count from the queue
    6. Render the floating Compare tray (US5) via a top-of-page container when `len(compare_queue) > 0` — placeholder in US4, implemented in US5
    7. Dispatch to view module based on tab value
- [ ] T348 [US4] Create `src/holodeck/dashboard/views/__init__.py` exporting `render_summary`, `render_explorer` (stub in US4), `render_compare` (stub in US4)
- [ ] T349 [US4] Stub `src/holodeck/dashboard/views/explorer.py` with `render_explorer(runs)` → `st.info("Explorer — see US5")` (US5 T411 replaces this)
- [ ] T350 [US4] Stub `src/holodeck/dashboard/views/compare.py` with `render_compare(runs)` → `st.info("Compare — see US5")` (US5 replaces this)

---

## Phase 6: Summary view — Plotly charts + breakdowns + runs table

Visual reference: `design_handoff_holodeck_eval_dashboard/summary.js` + open `Evaluation Dashboard.html` and click the **Summary** tab. Target "pixel-adjacent, not pixel-perfect" (README "Fidelity").

- [ ] T351 [US4] Create `src/holodeck/dashboard/charts.py` — pure functions returning `plotly.graph_objects.Figure`, so they're unit-testable shape-wise and reusable across views. Every chart must call `_apply_theme(fig)` setting `plot_bgcolor="#070c0a"`, `paper_bgcolor="rgba(0,0,0,0)"`, `font=dict(color="#e8f5ec", family="Inter, system-ui, sans-serif")`, `xaxis_gridcolor="rgba(28,43,37,.5)"`, `yaxis_gridcolor="rgba(28,43,37,.5)"`, `margin=dict(l=40, r=16, t=16, b=28)` (README "Streamlit notes")
- [ ] T352 [US4] `charts.pass_rate_chart(runs) -> Figure`:
    - Filled area under the line with gradient `rgba(123,255,90,.35) → rgba(123,255,90,0)` (use `go.Scatter(fill="tozeroy", fillcolor="rgba(123,255,90,.15)")`)
    - Main line in `#7bff5a` width 2
    - Dots: normal runs `#7bff5a` r=3; regression runs `#ff9d7e` r=4.5 with `#050b09` outline (use `data_loader.detect_regressions`)
    - Dashed vertical lines at prompt-version boundaries — `fig.add_vline(x=boundary_ts, line_dash="dot", line_color="rgba(123,255,90,.18)", annotation_text=version, annotation_position="top")` for each boundary from `detect_version_boundaries`
    - Y-axis: 0–1 with 0.25 tick spacing, formatted as `%` (handoff `summary.js:194–200`)
    - X-axis: datetime with abbreviated labels (`Apr 18`)
- [ ] T353 [US4] `charts.metric_trend_chart(runs, kind) -> Figure`:
    - One line per metric name in the selected `kind`, colors cycled through the palette `['#7bff5a','#5ae0a6','#53ff9c','#9bff5f','#a7f0ba','#ffcf5a']`
    - Horizontal threshold line at 0.7 — `fig.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,120,80,.7)", annotation_text="thresh 0.7")` (matches `summary.js:258`)
    - Legend on the right, single-column
    - Y-axis 0.0–1.0
- [ ] T354 [US4] `charts.breakdown_bar(df, palette) -> Figure`:
    - Horizontal bar, one bar per metric name, `barmode="stack"` if needed (single-segment here)
    - Bar fill: linear gradient approximation via two-segment bar — since Plotly doesn't render CSS gradients, use a single solid color per bar from the palette
    - Threshold marker at x=0.7 via `add_vline(line_dash="dash", line_color="var-equivalent")`
    - Height responsive to row count: `height=max(180, 48 * len(df))`
- [ ] T355 [US4] Create `src/holodeck/dashboard/views/summary.py`. Top-level layout: `sidebar_col, main_col = st.columns([0.18, 0.82])` — LEFT `sidebar_col` = filter rail, RIGHT `main_col` = content stack. (Streamlit's native `st.sidebar` can ALSO be used for Tweaks/global controls per README "Streamlit notes".)
- [ ] T356 [US4] **Filter rail** in left column — reproduce `summary.js::FilterRail`:
    - Header row "Filters" + a `Reset` link-style button that clears session_state filters
    - `st.date_input` for date range (two dates)
    - Multi-select chip row for prompt versions — use `st.multiselect` but restyle with CSS so selected items render as handoff-style green chips; option list from `distinct_values(runs, "prompt_version")`
    - Multi-select for model names
    - Multi-select for tags
    - `st.slider("Min pass rate", 0, 100, 0)` (percentage)
    - A monospace read-only `st.code` showing the current query string (e.g. `?versions=v1.3,v1.4&tags=rag-tuning`) + a `Copy URL` button (use `st.code(language="text")` — Streamlit renders a copy icon natively)
    - On any change, write to `st.query_params` via `filters_to_query_params` so refresh/share works (FR-028b)
- [ ] T357 [US4] **KPI strip** — reproduce `summary.js::KpiStrip` with 4 `st.metric` cards inside 4 `st.columns`:
    1. `Latest pass rate` — big value + delta vs. prior run (`st.metric` native `delta` arg shows `▲/▼` and color) + a mini sparkline (embed a tiny Plotly figure underneath the metric)
    2. `Runs (filtered)` — count with `sub="6 wks"` caption
    3. `Avg G-Eval score` — value `/ 1.00` + sparkline of last 8 runs' G-Eval averages
    4. `Median duration` — e.g., `19.4s` with `per run` caption. **Compute `statistics.median(durations_ms)`**, not the mean. The handoff's reference implementation labels the card `Median duration` but internally computes an average (`summary.js:123, 138`: `avgDur = ... / runs.length`) — that is a bug in the prototype. We honor the LABEL (the user-facing promise) by computing the actual median. Document this intentional deviation in `visual-baselines/README.md` and in the `charts.py` or view docstring so Chrome MCP parity checks expect a value difference here
    - Wrap each `st.metric` in a `st.container(border=True)` with custom CSS class for the card styling
- [ ] T358 [US4] **Pass-rate panel** — `st.container(border=True)` with:
    - Eyebrow: `TRENDS`
    - Title: `Pass rate over time`
    - Subtitle: `<N> runs of <agent> · regressions flagged in coral · dashed lines mark prompt-version boundaries`
    - Legend (small horizontal row): `pass rate` (green swatch) · `regression` (coral swatch)
    - `st.plotly_chart(pass_rate_chart(filtered_runs), use_container_width=True, config={"displayModeBar": False})`
- [ ] T359 [US4] **Metric trend panel** — `st.container(border=True)` with:
    - Eyebrow: `METRIC TRENDS` · title: `Per-metric average scores` · subtitle with threshold note
    - Segmented control `st.segmented_control("kind", options=["rag","geval","standard"], default="rag")` (Streamlit ≥1.36) bound to `filters.metric_kind`
    - `st.plotly_chart(metric_trend_chart(filtered_runs, kind))`
- [ ] T360 [US4] **Breakdown panels** — three `st.columns(3)` cells, each a `st.container(border=True)` with eyebrow + title + description + `st.plotly_chart(breakdown_bar(...))`:
    - `BREAKDOWN · STANDARD` · "NLP metrics" · palette `['#7bff5a','#5ae0a6','#9bff5f']` · kind `standard`
    - `BREAKDOWN · RAG` · "Retrieval & grounding" · palette full · kind `rag`
    - `BREAKDOWN · G-EVAL` · "Custom LLM judges" · palette `['#7bff5a','#ffcf5a','#5ae0a6']` · kind `geval`
- [ ] T361 [US4] **Runs table** — `st.dataframe(summary_df, selection_mode="single-row", on_select="rerun", hide_index=True, use_container_width=True, height=360)`. Columns configured via `st.column_config`:
    - `+` column: `st.column_config.CheckboxColumn` bound to compare-queue membership (check = in queue). When user toggles, call `state.push_to_compare_queue` / `state.remove_from_compare_queue` and `st.rerun()`
    - `timestamp`: datetime formatted `Apr 18 14:22`
    - `pass_rate`: `st.column_config.TextColumn("Pass rate")` rendering a pre-formatted pill string — `data_loader.to_summary_dataframe` emits the column as e.g. `"🟢 93.3%"` / `"🟡 72.0%"` / `"🔴 51.0%"` with the emoji chosen by the handoff's tier cutoffs (≥ 0.85 green, 0.65–0.85 yellow, < 0.65 coral; `summary.js:360`). This is the pragmatic replacement for the handoff's pill-next-to-inline-bar: `st.column_config.ProgressColumn` is bar-only and cannot color-tier per row, and Streamlit's DataFrame does not support per-cell CSS classes at the column_config level. The inline bar is acknowledged visual redundancy in the prototype; we keep the pill (which carries the tier color information). Document this fidelity trade-off in the visual-baselines README and in T373's Chrome MCP check (the parity assertion changes from "pill+bar" to "tiered pill string")
    - `passed/total`: text `12/12`
    - `prompt_version`: plain text in accent color (set via DataFrame cell styling)
    - `model`, `duration`, `commit`: text
    - On row select: call `state.open_in_explorer(run_id)` — clicking a row navigates to Explorer with that run pre-selected (README "Interactions: Run row click (Summary): navigate to Explorer")
    - Above the table: `Export CSV` button → `st.download_button` serving `summary_df.to_csv()`
- [ ] T362 [US4] **Empty state** — when `filtered_runs == []`, render a centered panel: `∅` icon + "No runs match your filters" + hint to clear filters or run `holodeck test`. Mirrors `summary.js:394–403`
- [ ] T363 [US4] Wire filter changes: every widget update calls `filters_to_query_params(filters)` → `st.query_params.update(...)` so the URL reflects state (FR-028b). On app boot, `filters_from_query_params(st.query_params)` seeds the Filters dataclass (READ at top of `app.py`, WRITE inside Summary view)

---

## Phase 7: Smoke tests (optional, `importorskip`-guarded)

- [ ] T364 [US4] `tests/integration/dashboard/test_app_smoke.py`: `pytest.importorskip("streamlit")`; use `monkeypatch.setenv("HOLODECK_DASHBOARD_USE_SEED", "1")` **before** `AppTest.from_file("src/holodeck/dashboard/app.py")` — AppTest inherits `os.environ` at exec time but provides no in-call env injection. Additionally, `app.py` must read the env var inside `data_loader.load_runs_for_app()` at call time (not cached at module import) so that repeat `AppTest.run()` calls honor env changes between tests. Run → assert no exceptions, at least one `stPlotlyChart`, at least one `stDataframe`. Package this pattern as a reusable `seed_mode_app` fixture in `tests/integration/dashboard/conftest.py` so US5 smoke tests inherit the setup without reimplementation
- [ ] T365 [US4] Extend smoke test: flip `tab` to each value, assert `Summary` variant renders with KPI strip (4 metrics), `Explorer` variant renders the stub, `Compare` variant renders the stub
- [ ] T366 [US4] `@pytest.mark.slow` on smoke tests; register the marker in `pyproject.toml` if not already present

---

## Phase 8: Visual fidelity — Chrome MCP side-by-side inspection

**Why this phase exists**: Streamlit is opinionated about layout and `AppTest` smoke tests only verify that components render without exceptions — they say nothing about visual match to the handoff. Every task below uses the Chrome MCP tools (`mcp__claude-in-chrome__*`) to drive a real browser, open the HTML prototype and the live Streamlit app side-by-side, capture screenshots, and diff them. The prototype is the ground truth (§Primary Source of Truth, item 2).

**Setup for this phase**:
- Terminal A: `holodeck test view --seed` (launches Streamlit on `http://localhost:8501` with seed data)
- Terminal B: a static server for the HTML prototype, e.g. `python -m http.server 8000 -d specs/031-eval-runs-dashboard/design_handoff_holodeck_eval_dashboard` → prototype at `http://localhost:8000/Evaluation%20Dashboard.html`

### Chrome MCP verification tasks

- [ ] T367 [US4] Open the prototype: call `mcp__claude-in-chrome__tabs_context_mcp` to enumerate tabs, then `mcp__claude-in-chrome__tabs_create_mcp` with `url="http://localhost:8000/Evaluation%20Dashboard.html"`. Wait for load, then `mcp__claude-in-chrome__take_screenshot` of the full page. Save the PNG as `specs/031-eval-runs-dashboard/visual-baselines/prototype-summary.png`
- [ ] T368 [US4] Open the Streamlit app: `mcp__claude-in-chrome__tabs_create_mcp` with `url="http://localhost:8501/?tab=summary"`. Wait for the main chart to render (use `mcp__claude-in-chrome__find` to locate a stable DOM element, e.g. the `stPlotlyChart` container), then `take_screenshot`. Save as `specs/031-eval-runs-dashboard/visual-baselines/streamlit-summary-v1.png`
- [ ] T369 [US4] **KPI strip parity check** — use `mcp__claude-in-chrome__get_page_text` on both tabs and extract the four KPI labels (`Latest pass rate`, `Runs (filtered)`, `Avg G-Eval score`, `Median duration`). Labels MUST match exactly; numeric values must match within rounding EXCEPT for the `Median duration` card, which intentionally differs: the Streamlit implementation computes `statistics.median(durations_ms)` per T357 decision (honoring the card's label) while the prototype's JS computes `avgDur` (a prototype bug). Document this single allowed delta in `visual-baselines/README.md` with before/after values. Any delta >0.1pp on the other three KPIs or any missing sparkline is a failing visual bug; re-work T357 until parity is achieved
- [ ] T370 [US4] **Pass-rate chart parity** — on both tabs, scroll to the pass-rate panel and screenshot JUST the chart region (use `mcp__claude-in-chrome__take_screenshot` with the DOM element ID returned by `find`). Verify:
    - Regression points appear in coral at the same run indices (detected by `detect_regressions`, handoff `summary.js:173–175`)
    - Dashed vertical lines appear at the same prompt-version boundaries (`summary.js:177–183`)
    - Area gradient is present and fades `rgba(123,255,90,.35) → transparent`
    - Y-axis ticks at 0/25/50/75/100%
    If any visual property differs, iterate on T352 and re-screenshot
- [ ] T371 [US4] **Metric-trend parity** — click the `rag`/`geval`/`standard` segmented control on both tabs; confirm each toggle produces the same set of lines (line count, colors) and that the 0.7 threshold horizontal line is present and dashed. Use `mcp__claude-in-chrome__click` to drive the toggle, then `take_screenshot` per position. If missing, fix T353 and T359
- [ ] T372 [US4] **Breakdown panels parity** — confirm 3 panels in a row, each with eyebrow + title + description + bars. On hover (use `mcp__claude-in-chrome__hover` on a bar), confirm Plotly tooltip content is sane. If the panels stack vertically instead of 3-across, the `st.columns(3)` wiring in T360 needs fixing
- [ ] T373 [US4] **Runs table parity** — confirm columns match the handoff order exactly: `[+] | Timestamp | Pass rate | Tests | Prompt | Model | Duration | Commit` (`summary.js:345–355`). Confirm the **pass-rate pill string** format (per the T361 decision replacing `ProgressColumn` with a tiered TextColumn): runs with pass_rate ≥ 0.85 render as `🟢 <pct>%`; 0.65–0.85 as `🟡 <pct>%`; < 0.65 as `🔴 <pct>%`. Screenshot the column; document in `visual-baselines/README.md` that our cell shows a tiered emoji+pct string in place of the prototype's pill-plus-inline-bar, by design (Streamlit DataFrame cannot render both). Click a row (`mcp__claude-in-chrome__click`) and confirm `st.query_params` updates and the Explorer tab receives focus with the correct `run_id`
- [ ] T374 [US4] **Filter rail parity** — confirm left column shows: Date range · Prompt version chips · Model chips · Tag chips · Min pass rate slider · Share URL block. Apply a filter (click a prompt-version chip via `mcp__claude-in-chrome__click`), watch the URL change via `mcp__claude-in-chrome__get_page_text` on the URL bar or `javascript_tool` running `window.location.search`. The URL MUST encode the filter (FR-028b)
- [ ] T375 [US4] **Theme parity** — use `mcp__claude-in-chrome__javascript_tool` to run `getComputedStyle(document.body).backgroundColor` on both tabs. Values SHOULD be numerically close (allowing for Streamlit's inherited wrappers). Confirm `--hd-accent` is computable as `rgb(123, 255, 90)`. If not, iterate on T343–T344
- [ ] T376 [US4] **Console cleanliness** — run `mcp__claude-in-chrome__read_console_messages` with `pattern: "(error|warning)"` on the Streamlit tab. Zero errors is the bar; warnings that originate from Streamlit's own components are acceptable, but anything from our `app.py` / view modules must be cleared
- [ ] T377 [US4] **Responsive check** — use `mcp__claude-in-chrome__resize_page` to set viewport to 1440×900, then 1280×800, then 1024×768. Screenshot each size. Confirm the three breakdown panels collapse gracefully (they may go 2+1 or stack on narrow viewports — acceptable as long as nothing clips or overlaps). Report the breakpoint where the layout breaks
- [ ] T378 [US4] **Record a short walkthrough** using `mcp__claude-in-chrome__gif_creator` named `summary_walkthrough.gif`: load Summary → apply a version filter → toggle metric-trend segmented control → click a table row to jump to Explorer. Commit the GIF alongside `visual-baselines/` as documentation

**Outputs of Phase 8**: a `visual-baselines/` directory committed with prototype and Streamlit screenshots, a short walkthrough GIF, and zero delta between the two on every parity check above. Any delta is a P0 bug against the T351–T363 implementation.

---

## Dependencies

- US1 Phase 2b runtime-shape migrations (US1 T010a–T010l: `MetricResult.kind`, `TestResult.tool_events`, `TestResult.conversation`) BLOCK every real-run rendering path in US4 — specifically the breakdown panels (T360), metric-trend kind filter (T359), and Summary's `kind`-aware aggregations (T321, T322). US4 work against the SEED dataset can proceed without these; real-run parity cannot.
- T301–T303 must complete first (extra installed, package tree exists).
- T304 blocks T305, T342 (group conversion is foundational for the subcommand).
- T306–T313 (seed data TDD + impl) blocks T319 (data_loader's seed branch), T364 (smoke test).
- T315–T327 (data/filters TDD) blocks T328–T329.
- T330–T337 (CLI TDD) blocks T338–T342.
- T328 blocks T352–T354, T357–T361 (charts + views consume `to_*_dataframe` functions).
- T343–T344 (theme) blocks T347 (app.py injects theme).
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
| AC4 (per-experiment scoping) | T339, T334 (results_dir scoped by slug) |
| AC5 (empty state) | T318, T362 |
| AC6 (install hint, no traceback) | T330, T340 |
| AC7 (Ctrl+C clean shutdown) | T337, T341 |
| FR-028a (faceted filters compose) | T326, T356 |
| FR-028b (URL query string) | T327, T356, T363 |
| **Design handoff Summary KPIs** | T357 |
| **Design handoff pass-rate chart (regression dots, version boundaries)** | T323, T324, T352, T358 |
| **Design handoff per-metric trend with kind toggle + threshold line** | T353, T359 |
| **Design handoff three breakdown panels** | T354, T360 |
| **Design handoff runs table w/ compare `+` and row-click drilldown** | T361 |
| **Design handoff filter rail** | T356 |

---

## Implementation Strategy

### Recommended order (matches user directive: scaffold → Summary → Explorer → Compare)

1. **Setup + Foundational** (T301–T314): install deps, convert CLI to group, port `data.js` → `data.py`. The seed dataset lets every subsequent UI task iterate without real runs.
2. **Data layer** (T315–T329): TDD the aggregations feeding every chart. All are pure Pandas; fast to iterate.
3. **CLI subprocess boundary** (T330–T342): TDD with `Popen` mocked; add the `--seed` flag so launching `holodeck test view --seed` works immediately against the in-memory dataset.
4. **App scaffold** (T343–T350): theme + state + `app.py` with three-tab shell. Launch with `holodeck test view --seed`; confirm all three tabs load with terminal-green aesthetic and navigation works. Explorer and Compare are visible stubs at this point.
5. **Summary view** (T351–T363): build each panel with the corresponding handoff file open in a viewer. After each panel is rendered in Streamlit, immediately run the matching Chrome MCP parity task (T369–T377) against the live prototype. Fix deltas before moving to the next panel — visual drift compounds.
6. **Smoke tests** (T364–T366): AppTest-level sanity.
7. **Visual fidelity sweep** (T367–T378): Chrome MCP side-by-side inspection against the HTML prototype. This is gate-keeping before handoff to US5 — no visual deltas may carry over.
8. Hand off to US5 (Explorer + Compare) — scaffold stubs already mounted, theme + state + seed data fully validated.

### Parallel team strategy

Three developers can work in parallel after T314:
- **Dev A**: data layer (T315–T329) + CLI (T330–T342)
- **Dev B**: theme + state + app scaffold (T343–T350)
- **Dev C**: Plotly charts (T351–T354) — independent pure functions testable with the seed data

Then merge and assemble Summary together (T355–T363).
