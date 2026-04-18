# Quickstart: Eval Runs, Prompt Versioning, and Test View Dashboard

**Feature**: 031-eval-runs-dashboard
**Audience**: A developer evaluating whether this feature works end-to-end after Phase 2 implementation completes.

This quickstart is a hand-runnable recipe. It doubles as the "golden path" integration test script.

---

## 0. Prerequisites

```bash
# From a clean clone
make init
source .venv/bin/activate
holodeck --version
```

To exercise the dashboard, install the optional extra:

```bash
uv add 'holodeck-ai[dashboard]'
# or: pip install 'holodeck-ai[dashboard]'
```

---

## 1. Create a minimal agent with a frontmatter-annotated prompt

```bash
mkdir -p ./demo && cd ./demo
```

`demo/instructions.md`:

```markdown
---
version: "1.0"
author: "jane@example.com"
description: "Initial support prompt"
tags: [support, v1]
---
You are a helpful customer support agent. Always answer in one paragraph.
```

`demo/agent.yaml`:

```yaml
name: support-demo
model:
  provider: ollama
  name: gpt-oss:20b
  temperature: 0.2
instructions:
  file: instructions.md
test_cases:
  - name: "Refund question"
    input: "Do you have a money-back guarantee?"
    ground_truth: "We offer a 30-day money-back guarantee."
evaluations:
  metrics:
    - type: standard
      metric: bleu
      threshold: 0.1
```

---

## 2. Run the tests — persistence kicks in automatically

```bash
holodeck test agent.yaml
```

Expected console output (abbreviated):

```
✓ Refund question ... passed
1/1 tests passed
EvalRun persisted: results/support-demo/2026-04-18T14-22-09.812Z.json
```

> **Note**: run files are intended to be committed (per spec Assumption 5) so experiment history stays diffable and reviewable alongside code changes. If your team prefers to keep runs local-only, add `results/` to `.gitignore` — HoloDeck itself does not modify your `.gitignore`.

Verify the artifact:

```bash
ls results/support-demo/
# → 2026-04-18T14-22-09.812Z.json

python - <<'PY'
from holodeck.models.eval_run import EvalRun
import pathlib, json
p = next(pathlib.Path("results/support-demo").glob("*.json"))
run = EvalRun.model_validate_json(p.read_text())
print("version:", run.metadata.prompt_version.version)
print("author: ", run.metadata.prompt_version.author)
print("tags:   ", run.metadata.prompt_version.tags)
print("source: ", run.metadata.prompt_version.source)
print("model:  ", run.metadata.agent_config.model.name)
print("tests:  ", run.report.summary.total_tests)
PY
```

Expected:
```
version: 1.0
author:  jane@example.com
tags:    ['support', 'v1']
source:  file
model:   gpt-oss:20b
tests:   1
```

---

## 3. Verify auto-version when no manual `version:` is present

Edit `instructions.md`, remove only the `version: "1.0"` line (keep the rest of the frontmatter). Re-run:

```bash
holodeck test agent.yaml
```

Inspect the new file:

```bash
python -c "
from holodeck.models.eval_run import EvalRun
import pathlib
files = sorted(pathlib.Path('results/support-demo').glob('*.json'))
latest = files[-1]
run = EvalRun.model_validate_json(latest.read_text())
print(run.metadata.prompt_version.version)  # e.g., auto-3f9a1c2b
"
```

Re-run `holodeck test agent.yaml` **without** editing the prompt body. The auto-version in the new file MUST match the previous auto-version — this is SC-003.

Now change one character of the prompt body (e.g., `one paragraph` → `two paragraphs`). Re-run. The auto-version MUST differ — this is SC-004.

---

## 4. Verify the snapshot is frozen (Story 3)

1. Copy the latest run file aside:
   ```bash
   cp results/support-demo/*.json /tmp/pinned_run.json
   ```
2. Edit `agent.yaml`: change `temperature: 0.2` → `temperature: 0.9`.
3. Re-load `/tmp/pinned_run.json`:
   ```bash
   python -c "
   from holodeck.models.eval_run import EvalRun
   r = EvalRun.model_validate_json(open('/tmp/pinned_run.json').read())
   print(r.metadata.agent_config.model.temperature)  # MUST print 0.2
   "
   ```

---

## 5. Verify secret redaction

Set a fake API key and run:

```bash
OPENAI_API_KEY=sk-not-a-real-key holodeck test agent.yaml
```

Inspect the newest run file:

```bash
python -c "
import json, pathlib
p = sorted(pathlib.Path('results/support-demo').glob('*.json'))[-1]
doc = json.loads(p.read_text())
ak = doc['metadata']['agent_config']['model'].get('api_key')
print(repr(ak))  # '***' or absent, NEVER 'sk-not-a-real-key'
"
```

---

## 6. Launch the dashboard (requires `[dashboard]` extra)

```bash
holodeck test view agent.yaml
```

Expected console:

```
Warning: Streamlit binds to 0.0.0.0 by default. results/ may contain
sensitive conversation/tool data — firewall this port on shared infrastructure.

  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

A browser tab opens showing:
- **Summary** page:
  - Sortable table: `timestamp`, `pass_rate`, `total_tests`, `prompt_version`, `model_name`.
  - Line chart: pass rate over time.
  - Line chart: per-metric average score over time.
  - Three breakdown panels: `standard` (bleu/rouge/meteor), `rag` (5 sub-metrics), `geval` (per custom name).
  - Sidebar filters: date range, prompt-version multi-select, model-name multi-select, pass-rate slider, tag multi-select.
  - Changing filters updates the URL query string; copy-pasting the URL into a new tab reproduces the filtered view.
- **Explorer** page (linked from the run table):
  - Click a run → list of test cases with pass/fail badges.
  - Click a test case → detail panel with agent config snapshot, chat-style conversation, tool-call panels (collapsed when large), expected-tools check/cross, metric results with score/threshold/reasoning.

Ctrl+C in the terminal terminates the server cleanly.

---

## 7. Verify the missing-extra install hint

Without the `dashboard` extra installed:

```bash
uv remove streamlit  # simulate
holodeck test view agent.yaml
```

Expected (exit code 2):

```
Dashboard not installed. Install the optional extra:
  uv add 'holodeck-ai[dashboard]'   # or: pip install 'holodeck-ai[dashboard]'
```

No Python traceback (SC-007).

---

## 8. Verify empty state

In a fresh directory with no prior runs:

```bash
mkdir /tmp/empty-demo && cd /tmp/empty-demo
# copy agent.yaml + instructions.md, do NOT run `holodeck test`
holodeck test view agent.yaml
```

The dashboard loads with an empty-state panel:

> No runs found yet — execute `holodeck test` to generate one.

No traceback.

---

## 10. Verify tool-call capture (dashboard Explorer prerequisite)

The dashboard's Explorer view needs `{name, args, result, bytes}` per tool call to render the `ToolCall` panel. Confirm `tool_invocations` is populated end-to-end.

Add a trivial function tool to `demo/agent.yaml` (or use any fixture with an MCP/function tool) and a test case that triggers it. Then:

```bash
holodeck test agent.yaml
python - <<'PY'
import pathlib
from holodeck.models.eval_run import EvalRun
p = sorted(pathlib.Path("results/support-demo").glob("*.json"))[-1]
run = EvalRun.model_validate_json(p.read_text())
tc = run.report.results[0]
print("tool_invocations:", len(tc.tool_invocations))
for inv in tc.tool_invocations:
    print(f"  - {inv.name}({inv.args}) -> {inv.bytes} bytes")
print("token_usage:", tc.token_usage)
print("metric kinds:", {m.kind for m in tc.metric_results})
PY
```

Expected:
- `tool_invocations` is non-empty when the test triggered at least one tool.
- Each invocation has a non-empty `name`, a dict `args` (possibly empty), a `result`, and `bytes == len(json.dumps(result, default=str))`.
- `token_usage` is a `TokenUsage` object (not `None`) when the backend reported it.
- Every `MetricResult` has `kind ∈ {"standard","rag","geval"}`.

If `tool_invocations` is empty but `tool_calls` (names-only) is not, the pairing layer (see research.md R8) didn't run — report as a bug. Back-compat readers will fall through to `tool_calls`.

---

## 11. Success checklist

| Criterion | Verified by step |
|---|---|
| SC-001: Locate persisted file in <10 s | §2 |
| SC-002: Reconstruct agent config from JSON alone | §4 |
| SC-003: Same body → same auto-version | §3 |
| SC-004: One-char edit → different auto-version | §3 |
| SC-005: Summary renders for ≥50 runs | Populate `results/` with 50 files and re-run §6 |
| SC-006: Drill to test case in ≤3 clicks | §6 (Explorer) |
| SC-007: Clean install hint on missing extra | §7 |
| SC-008: <200 ms persistence overhead | Micro-benchmark: `time holodeck test` with 1 test case, compare before/after |
| SC-009: `--output` output byte-equivalent | `diff` against pre-feature output on the same config |
| SC-010: <5 s P95 load for 1000 runs | Populate 1000 synthetic runs and time Summary first-render |
| SC-011: Tool-call args/result captured | §10 |
| SC-012: Token usage persisted for cost computation | §10 |
| SC-013: MetricResult.kind populated for every metric | §10 |
