# Handoff: HoloDeck Evaluation Dashboard

## Overview

A dashboard for viewing agent-evaluation test runs — the sort of UI you'd scaffold after running a `holodeck test` command against an agent config. It shows pass/fail trends, per-metric breakdowns, individual test-case detail with full tool-call traces, and a 3-run comparison view with delta analysis.

There are three primary views:

1. **Summary** — headline KPIs, trend charts, metric breakdowns, filterable run table
2. **Explorer** — three-column drilldown (runs → cases → detail) with conversation thread + tool calls
3. **Compare** — pick up to 3 runs; see them side-by-side, baseline+deltas, or matrix-first

The user wants this reproduced in **Streamlit** (see "Streamlit notes" section below).

## About the Design Files

The files in this bundle are **design references created in HTML/React** — prototypes showing intended look and behavior, not production code to copy directly. The task is to **recreate these designs in Streamlit**, using Streamlit's native components (st.metric, st.dataframe, st.plotly_chart, st.tabs, st.columns, etc.) and CSS overrides where needed to approximate the visual style.

The HTML prototypes are fully interactive. Open `Evaluation Dashboard.html` in a browser to explore every interaction before you start.

## Fidelity

**High-fidelity.** The HTML prototype has final colors, typography, spacing, and a coherent dark-mode "terminal-green" aesthetic. Recreate in Streamlit as closely as Streamlit allows — Streamlit is opinionated about layout, so some compromises are expected. Aim for pixel-adjacent, not pixel-perfect.

---

## Design Tokens

### Colors

| Token | Hex | Usage |
|---|---|---|
| `--hd-bg-body` | `#050b09` | Page background |
| `--hd-bg-alt` | `#0c1412` | Surface |
| `--hd-card` | `#0a110f` | Raised card |
| `--hd-card-2` | `#070c0a` | Card gradient end |
| `--hd-border` | `#1c2b25` | Borders, dividers |
| `--hd-fg` | `#e8f5ec` | Primary foreground |
| `--hd-muted` | `#9bb3a5` | Secondary text |
| `--hd-accent` | `#7bff5a` | Primary accent (terminal green) |
| `--hd-accent-2` | `#53ff9c` | Gradient stop |
| `--hd-accent-soft` | `#5ae0a6` | Softer accent (labels) |
| `--hd-accent-glow` | `rgba(123,255,90,.18)` | Hover/glow |
| *(fail)* | `#ff9d7e` | Regressions, fails (coral) |
| *(warn)* | `#ffcf5a` | Warn / changed-config |

### Compare-palette (per slot)

```js
['#7bff5a', '#5ae0a6', '#ffcf5a']  // baseline, run-1, run-2
```

### Page background radial

```css
background:
  radial-gradient(circle at 22% 18%, rgba(123,255,90,.09), transparent 26%),
  radial-gradient(circle at 78% 12%, rgba(123,255,90,.06), transparent 22%),
  #050b09;
```

### Typography

- **Sans**: `"Inter", -apple-system, system-ui, sans-serif` — weights 400/500/600/700
- **Mono**: `"JetBrains Mono", ui-monospace, Menlo, monospace` — weights 400/500
- **Scale**: h3 18px · body 16px · small 15px · code 13px · tag 12px · label 11px
- **Eyebrow labels**: 10px, `letter-spacing: .15em`, uppercase, color `--hd-accent-soft`
- **Numeric values**: always in mono

### Radius / Spacing

- `--radius-sm` 6px · `--radius-md` 10px · `--radius-lg` 16px · `--radius-pill` 999px
- Cards: 14–18px padding, 1px solid `--hd-border`, linear-gradient bg `rgba(10,17,15,.95)` → `#070c0a` 60%
- Gap scale: 2/4/6/8/10/12/14/16/20/24px

### Motion

- `--dur-quick` 140ms · `--dur-base` 220ms · `--ease-standard` `cubic-bezier(.2,.7,.3,1)`

---

## Data Model

See `data.js` for the full synthetic dataset. The shape each real run should match:

```ts
type Run = {
  id: string;                           // "run_20251103_104300_a1b2"
  created_at: string;                   // ISO 8601
  git_commit: string;                   // short SHA
  summary: {
    total: number;
    passed: number;
    failed: number;
    pass_rate: number;                  // 0..1
    duration_ms: number;
  };
  metadata: {
    agent_config: {
      model: { name: string; temperature: number; max_tokens: number; };
      claude: { extended_thinking: boolean; /*...*/ };
      api_key: string;                  // always redacted to "***" in UI
      /* other agent config */
    };
    prompt_version: {
      version: string;                  // "v0.4.2"
      tags: string[];                   // ["triage", "cautious"]
    };
  };
  test_results: TestCase[];
};

type TestCase = {
  name: string;
  passed: boolean;
  input: string;                        // user turn that started the case
  conversation: Message[];              // chat-style thread, includes tool calls
  expected_tools: string[];             // names of tools the agent SHOULD call
  actual_tools: string[];               // names actually called
  metric_results: MetricResult[];
};

type Message =
  | { role: 'user' | 'assistant'; content: string; }
  | { role: 'tool_call'; name: string; arguments: any; }
  | { role: 'tool_result'; name: string; result: any; bytes?: number; };

type MetricResult = {
  kind: 'standard' | 'rag' | 'geval';
  name: string;                         // e.g. "faithfulness", "answer_relevancy"
  score: number;                        // 0..1
  threshold: number;                    // 0.7 typical
  passed: boolean;
  reasoning?: string;                   // for geval, judge's explanation
};
```

---

## Views

### 1. Summary

**Purpose**: Dashboard landing. Answer "is the agent regressing?" at a glance.

**Layout** — 2 columns:
- **Left rail** (240px fixed): filter controls (collapsible via Tweak)
- **Main** (flex-1): stacked content

**Main content order**:
1. **KPI strip** — 4 cards in a row: Pass rate (big number + delta vs. prev run), Run count, Avg G-Eval, Avg duration
2. **Pass-rate-over-time chart** (area chart) — x: time, y: pass rate. Coral dots on regressions (drop >10pp from previous run). Dashed vertical lines at prompt-version boundaries, labeled with version.
3. **Per-metric trend** (line chart) — one line per metric, with segmented control (`standard | rag | geval`) to filter. Horizontal dashed line at threshold 0.7.
4. **Three breakdown panels** side-by-side (grid of 3 cols): Standard / RAG / G-Eval. Each shows per-metric averages as horizontal bars with threshold markers.
5. **Runs table** — sortable columns: `[+]` | Timestamp | Pass rate (pill + inline bar) | Tests (passed/total) | Prompt version | Model | Duration | Commit. Click row → opens in Explorer. `[+]` adds to compare queue.

**Filter rail contents**:
- Date range
- Prompt version chips (multi-select)
- Model chips (multi-select)
- Min pass-rate slider (0–100%)
- Tag chips (multi-select)
- Shareable URL field (encodes filters in querystring)

**Pill colors**: pass ≥ 85% → green, 65–85% → yellow, <65% → coral.

### 2. Explorer

**Purpose**: Deep-dive on a specific test case.

**Layout** — 3 columns:
- **Runs list** (340px, collapsible to 48px via arrow): vertical list, newest first. Each item shows: timestamp, version, pass-rate pill, passed/total, model suffix, `[+]` button.
- **Cases list** (340px): vertical list of cases for selected run. Each shows pass/fail icon, case name, G-Eval score, RAG avg.
- **Detail panel** (flex-1): everything about the selected case.

**Detail panel sections** (scrollable):
1. **Case header**: name, pass/fail pill, expected-tools coverage (check/cross per tool name)
2. **Agent config snapshot**: collapsible card showing `agent_config` as pretty-printed JSON with `api_key` redacted to `***`
3. **Conversation thread**: chat-style bubbles
   - `user` → right-aligned light bubble
   - `assistant` → left-aligned fg bubble
   - `tool_call` → distinct amber-tinted panel showing tool name, args as JSON
   - `tool_result` → distinct green-tinted panel. If `bytes > 500`, collapse by default with "show result (1.2 KB)" toggle.
4. **Metric evaluations**: grouped by kind (standard/rag/geval). Each metric: name · score (mono, big) · threshold · pass/fail pill · (for geval) judge's reasoning as expandable text.

### 3. Compare

**Purpose**: Pick 2–3 runs, see what changed. Baseline = first run added.

**Empty state**: friendly CTA with "Compare latest 2 runs" / "Compare latest 3 runs" shortcuts.

**3 layout variants** (segmented control at top):

**Variant 1 — Side-by-side** (default):
- Column headers row (one per run, color-coded dot)
- **Summary block**: rows `pass rate / passed / avg geval / avg rag / duration / est. cost`, with a delta pill next to non-baseline values (green for improvement, coral for regression; duration + cost invert polarity)
- **Config diff block**: rows for `prompt version / model / temp / tags / commit / extended thinking`. Cells where the value differs from baseline get an amber left-border and a "changed" badge.
- **Case matrix**: one row per case, one column per run. Each cell shows ✓/✕ + per-case score. Regression cells (baseline passed, this run failed) get a coral outer ring; improvement cells the inverse with green.

**Variant 2 — Baseline + deltas**:
- Big baseline card on the left (~1.4fr), takes full stats including big pass-rate number
- Right: compact delta cards (1fr each) showing `label · value · delta-pill` rows
- Case matrix below

**Variant 3 — Matrix-first**:
- Compact strip of run cards at top (equal-width)
- **Callouts block**: per non-baseline run, a card listing first 3 regressions and first 3 improvements with `+N` overflow
- Full case matrix below

**Delta polarity**: positive values green, negative coral. For `duration` and `cost`, invert (lower = better).

**Compare tray** (floating, fixed bottom-center, shown across all tabs when queue > 0): slot pills for 3 items, "base" badge on first, remove ×, "Open Compare →" CTA. Enabled when ≥ 2 slots filled.

---

## Interactions & Behavior

- **Tab persistence**: selected tab stored in localStorage
- **Compare queue**: stored in localStorage (array of run IDs, max 3)
- **Run row click** (Summary): navigate to Explorer with that run selected
- **`[+]` button** (Summary table + Explorer runs list): add/remove from queue. Shows slot number (1/2/3) when in queue.
- **Variant switch**: segmented control at top of Compare view
- **Hover states**: rows brighten; cards get subtle glow
- **Filter changes**: live-update chart + table

---

## State (Streamlit mapping)

Use `st.session_state` for:
- `tab`: "summary" | "explorer" | "compare"
- `filters`: `{ versions: [], models: [], tags: [], min_pass: 0, metric_kind: "rag" }`
- `explorer_run_id`, `explorer_case_name`
- `compare_queue`: `list[str]` (max 3)
- `compare_variant`: `1 | 2 | 3`

Persist across reruns with `streamlit_local_storage` or a sidecar JSON file if cross-session persistence is needed.

---

## Streamlit notes

Streamlit-specific implementation hints:

- **Tabs**: `st.tabs(["Summary", "Explorer", "Compare"])` — simple but doesn't support badges; consider `st.radio(horizontal=True)` with custom CSS if you want the count badges from the HTML design.
- **Charts**: use **Plotly** (`st.plotly_chart`), not Altair — you need fine control over the dashed prompt-version lines, coral regression dots, and threshold markers. Set `plot_bgcolor="#070c0a"`, `paper_bgcolor="rgba(0,0,0,0)"`, font color `#e8f5ec`, grid `rgba(28,43,37,.5)`.
- **Tables**: `st.dataframe` supports clickable rows via `on_select="rerun"` and `selection_mode="single-row"`. Use it for the runs table. For the per-metric breakdown, a small `st.dataframe` per group works.
- **3-column Explorer layout**: `st.columns([1, 1, 2])`. Put the lists in scrollable containers: `with st.container(height=600):`.
- **Case matrix in Compare**: easiest approach is a Plotly heatmap — rows=cases, cols=runs, z=score, colorscale matching the HTML's green/coral gradient. Overlay regression/improvement markers with `add_annotation`.
- **Compare tray**: Streamlit has no fixed-position floaters natively — use `st.sidebar` OR inject a small CSS-positioned `st.empty()` container. Or just render it at the top of the page when `len(compare_queue) > 0`.
- **Global CSS**: inject once at the top:
  ```python
  st.markdown("""<style>
    :root { --hd-accent: #7bff5a; ... }
    .stApp { background: radial-gradient(...) #050b09; color: #e8f5ec; }
    /* mono font for metrics, etc. */
  </style>""", unsafe_allow_html=True)
  ```
- **Tool-call bubbles in the conversation thread**: render as `st.json(expanded=False)` inside an `st.container(border=True)`. For `bytes > 500`, wrap in `st.expander("show result (1.2 KB)")`.
- **Redact `api_key`**: do it in the data layer before rendering, never in UI.
- **Tweaks**: Streamlit doesn't need a tweaks panel — put equivalent controls in `st.sidebar`: density, default tab, compare layout.

---

## Files in this handoff

- `Evaluation Dashboard.html` — entry HTML; load this in a browser to explore the prototype
- `data.js` — synthetic dataset (24 runs, 7 prompt versions). **Use this as your seed data for Streamlit development.** Parse it with a small JS-to-JSON shim or manually port the structure.
- `styles.js` — all CSS (injected as a `<style>` tag at runtime). Good source for exact color/spacing/shadow values.
- `summary.js` — Summary view React components (KPIs, charts, breakdowns, runs table, filter rail)
- `explorer.js` — Explorer view (run list, case list, case detail with tool-call rendering)
- `compare.js` — Compare view (3 variants, tray, `[+]` button, empty state)
- `app.jsx` — root component wiring, state management, Tweaks panel

---

## Assets

No bespoke image assets — everything is SVG inline or CSS. The dataset's run IDs and commit SHAs are synthetic. Swap `data.js` for your real eval pipeline's output and the UI should just work.
