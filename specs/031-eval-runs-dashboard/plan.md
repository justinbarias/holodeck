# Implementation Plan: Eval Runs, Prompt Versioning, and Test View Dashboard

**Branch**: `031-eval-runs-dashboard` | **Date**: 2026-04-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification at `/specs/031-eval-runs-dashboard/spec.md`

## Summary

Extend HoloDeck's `test` command with three cohesive capabilities:

1. **Persistence (`EvalRun`)** — every `holodeck test` invocation writes a strongly-typed JSON artifact at `results/<slugified-agent-name>/<ISO-timestamp>.json` that wraps the existing `TestReport` and adds a `metadata` block capturing a full snapshot of the validated `Agent` config, the prompt version, and run provenance. Secrets are redacted via a minimal name-allowlist plus `SecretStr`-type-driven policy. Writes are atomic (`.tmp` + `fsync` + `os.replace`).
2. **Prompt versioning (`python-frontmatter`)** — optional YAML frontmatter on instruction files is parsed via `python-frontmatter`; the prompt body is stripped before it reaches the LLM; a `PromptVersion` is attached to `EvalRun.metadata`. Manual `version:` overrides an auto `auto-<sha256[:8]>` derived from the prompt body.
3. **Dashboard (`holodeck test view`)** — a new Click subcommand of `test` that launches a Streamlit app (shipped as optional extra `holodeck[dashboard]`) in a subprocess. The app auto-discovers runs under `results/<slugified-agent-name>/`, renders a Summary view with faceted filters + trend/breakdown charts, and an Explorer view for per-test-case drill-down.

Scope is purely additive to the file-on-disk contract: `TestReport` keeps its shape; `TestResult` gains **additive** fields only (`tool_invocations`, `token_usage`, and `MetricResult.kind`) — legacy EvalRun files remain loadable via Pydantic defaults. Existing `--output` behaviour is preserved byte-equivalent.

**Why `TestResult` must grow** (not in the original plan, added after the design handoff landed): the dashboard's Explorer view requires `{name, args, result, bytes}` per tool invocation; its Compare view references per-run cost (derivable from token usage); every Summary breakdown pivots on `MetricResult.kind`. Without these fields the dashboard can only render the synthetic seed dataset, not real runs. See [data-model.md](./data-model.md) "Dashboard Field Projections" for the full on-disk ↔ dashboard name-mapping table and [research.md](./research.md) R8 / R9 for pairing and token-usage decisions.

## Technical Context

**Language/Version**: Python 3.10+ (matches existing HoloDeck target).
**Primary Dependencies**:
- Core (new): `python-frontmatter>=1.1,<2.0` (MIT; transitive dep is `PyYAML` which is already present).
- Optional extra `dashboard` (new): `streamlit>=1.36,<2.0` (for `st.Page`/`st.navigation` and `st.query_params`), `altair>=5.0` (bundled with streamlit), `pandas>=2.0` (bundled with streamlit).
- Existing: Pydantic v2, Click, PyYAML, asyncio, OpenTelemetry SDK, claude-agent-sdk, semantic-kernel.
**Storage**: Filesystem. Runs persisted as UTF-8 JSON at `results/<slugified-agent-name>/<ISO-timestamp>.json`. No database.
**Testing**: pytest with markers (`unit`, `integration`, `slow`). `-n auto` for parallel runs. New suites for frontmatter parsing, redaction, atomic write, dashboard data loader. Streamlit UI is not unit-tested; data-loader is unit-tested and the app receives integration smoke-tests via `streamlit.testing.v1.AppTest` where cheap.
**Target Platform**: Developer workstations (macOS/Linux) and CI. Dashboard runs locally (Streamlit subprocess bound to `0.0.0.0` per Streamlit default).
**Project Type**: Single project (existing `src/holodeck/` layout).
**Performance Goals**:
- SC-008: EvalRun persistence adds <200 ms per invocation.
- SC-010: Dashboard Summary view renders in <5 s P95 for 1000 runs.
**Constraints**:
- Round-trip equivalence for `EvalRun` via `model_dump_json()` / `model_validate_json()` (modulo redacted secrets).
- Atomic writes (FR-009b): no reader ever observes a partial file.
- Optional-extra boundary: no `streamlit` import at Click-command import time; detection via `importlib.util.find_spec`.
- Dashboard must not prescribe a design system — the user is building one separately. The plan specifies Streamlit primitives and layout semantics only (no custom CSS/themes beyond Streamlit defaults in v1).
- **`results/` path resolution**: `results/` is rooted at the agent.yaml's directory (`agent_base_dir` context variable, the same convention already used by `instructions.file`, `response_format` files, and vector-store source paths). A run launched as `holodeck test subdir/agent.yaml` writes to `subdir/results/<slug>/`. The dashboard resolves the same path.
- **`.gitignore` policy**: This feature intentionally does NOT modify `.gitignore`. Per spec Assumption 5, EvalRun files are designed to be committed (they are under 1 MB, redacted, and are the basis for reviewable experiment history). Teams that prefer to keep `results/` local can add `results/` to their own `.gitignore` — the code does not opine.
**Scale/Scope**:
- Up to 1000 runs per agent in `results/` (spec edge case).
- Typical run file <1 MB (assumption in spec).
- Experiments identified by `agent.name` (no cross-agent aggregation in v1).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment | Notes |
|---|---|---|
| I. No-Code-First Agent Definition | **PASS** | All new behaviour is opt-out-free: persistence happens automatically; frontmatter is optional YAML inside a markdown file (no Python). Dashboard is a viewer — users never write Python. No new YAML keys in `agent.yaml`. |
| II. MCP for API Integrations | **N/A** | No external API integration added. Dashboard reads local filesystem only. |
| III. Test-First with Multimodal Support | **PASS** | New models have unit tests before wiring. Multimodal test-case `files:` snapshotting is path-only by spec decision — documented in FR-009a; no regression to existing multimodal support. |
| IV. OpenTelemetry-Native Observability | **PASS** | No new telemetry-bearing code paths in the hot agent loop. Persistence happens post-run and is logged at INFO/WARNING as appropriate. No OTel regression. Git-commit capture is best-effort and synchronous but bounded (one `git rev-parse HEAD`). |
| V. Evaluation Flexibility with Model Overrides | **PASS** | No changes to evaluation model selection. Dashboard reads existing `metric_results` including `model_used`. |
| Architecture Constraints (3 engines) | **PASS** | EvalRun lives in the evaluation engine boundary (persistence of test output). No coupling introduced across engines. Dashboard is a separate process — strictly a consumer of on-disk artifacts. |
| Code Quality (MyPy strict, Black, Ruff, Bandit) | **PASS** | New modules are type-annotated; new models are Pydantic v2. `subprocess.Popen` call for Streamlit uses fully qualified `sys.executable` and a list argv (no shell). |
| Minimum 80% coverage | **PASS (plan)** | Coverage targets: persistence writer, redactor, PromptVersion resolver, slugifier, dashboard data loader, CLI `test view` entry (mocked subprocess). |

**Result**: No violations. No entries needed in the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/031-eval-runs-dashboard/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── eval_run.schema.json        # JSON Schema for EvalRun
│   ├── prompt_version.schema.json  # JSON Schema for PromptVersion
│   └── cli.md                      # CLI contract for `holodeck test view`
└── checklists/          # (exists from /speckit.specify stage)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   ├── eval_run.py                # NEW: EvalRun, EvalRunMetadata, PromptVersion
│   ├── test_result.py             # EDIT: MetricResult.kind (new Literal field),
│   │                              #       TestResult.tool_invocations: list[ToolInvocation],
│   │                              #       TestResult.token_usage: TokenUsage | None,
│   │                              #       ToolInvocation (new co-located model).
│   │                              #       Name deliberately chosen to NOT collide with
│   │                              #       models/tool_event.py::ToolEvent (streaming) or
│   │                              #       lib/backends/base.py::ToolEvent (hook-stream).
│   ├── llm.py                     # EDIT: LLMProvider.api_key: str → SecretStr
│   ├── tool.py                    # EDIT: VectorStoreConfig.connection_string,
│   │                              #       VectorStoreConfig.api_key,
│   │                              #       FunctionConfig auth-bearing fields → SecretStr
│   ├── observability.py           # EDIT: AzureMonitorExporterConfig
│   │                              #       .connection_string → SecretStr
│   └── config.py                  # EDIT: ExecutionConfig-side connection_string
│                                  #       (if secret-bearing) → SecretStr
│   # NOTE: Each SecretStr migration requires updating every reader callsite
│   # (SK backend, Claude backend, Azure exporter init, vector-store clients)
│   # to call `.get_secret_value()`. Migration targets are enumerated in
│   # data-model.md §"Redaction Policy Surface / Migration required".
│
├── lib/
│   ├── eval_run/                  # NEW package
│   │   ├── __init__.py
│   │   ├── writer.py              # atomic write: .tmp + fsync + os.replace;
│   │   │                          # resolves results/ against agent_base_dir
│   │   ├── redactor.py            # name-allowlist + SecretStr-driven redaction
│   │   ├── slugify.py             # agent-name slugification
│   │   └── metadata.py            # build EvalRunMetadata (git rev-parse, cli args, ts)
│   ├── test_runner/
│   │   ├── executor.py            # EDIT: wire ToolInvocation pairing (research.md R8)
│   │   │                          #       and TestResult.token_usage (research.md R9)
│   │   └── tool_invocation_builder.py  # NEW: pair_tool_calls(exec_result, backend_kind)
│   │                                   # → list[ToolInvocation]; per-backend correlation
│   │                                   # (SK: index, Claude: tool_use_id)
│   ├── prompt_version.py          # NEW: frontmatter parse + auto-hash resolver,
│   │                              #       exposes `resolve_prompt_version(instructions,
│   │                              #       base_dir) -> PromptVersion`. Does NOT modify
│   │                              #       instruction_resolver.py (additive module).
│   └── instruction_resolver.py    # UNCHANGED: existing callers
│                                  #       (agent_factory.py:1151, claude_backend.py:275)
│                                  #       keep their `-> str` contract. The eval-run
│                                  #       writer invokes resolve_prompt_version()
│                                  #       separately on the same Instructions object.
│
├── cli/
│   └── commands/
│       ├── test.py                # EDIT: convert `@click.command()` → `@click.group(
│       │                          #       invoke_without_command=True)` with the current
│       │                          #       test logic moved into a default callback that
│       │                          #       runs when no subcommand is given (preserving
│       │                          #       `holodeck test agent.yaml` backward compat).
│       │                          #       Register `view` via `test.add_command(view)`.
│       │                          #       Also wire EvalRun persistence after test
│       │                          #       execution.
│       └── test_view.py           # NEW: the `view` subcommand — detects `streamlit`
│                                  #       via find_spec, resolves results/ from
│                                  #       agent_base_dir, spawns streamlit subprocess,
│                                  #       forwards signals.
│
└── dashboard/                     # NEW package — only imported when extra installed
    ├── __init__.py
    ├── app.py                     # Streamlit entry (target of `streamlit run`)
    ├── data_loader.py             # Scan results/<agent>/*.json, skip-on-corrupt,
    │                              # return list[EvalRun] + typed DataFrames
    ├── pages/
    │   ├── summary.py             # Summary view (filters + trends + breakdowns)
    │   └── explorer.py            # Explorer view (run → test-case drill-down)
    └── filters.py                 # Filter state ↔ st.query_params plumbing

tests/
├── unit/
│   ├── models/
│   │   ├── test_eval_run.py
│   │   ├── test_metric_result_kind.py           # NEW: MetricResult.kind
│   │   ├── test_tool_invocation.py              # NEW: ToolInvocation shape + bytes computation
│   │   └── test_test_result_token_usage.py      # NEW: TestResult.token_usage round-trip
│   ├── lib/
│   │   ├── eval_run/
│   │   │   ├── test_writer_atomic.py
│   │   │   ├── test_redactor.py
│   │   │   └── test_slugify.py
│   │   ├── test_runner/
│   │   │   └── test_tool_invocation_builder.py  # NEW: SK index-pairing, Claude id-pairing
│   │   └── test_prompt_version.py
│   ├── cli/
│   │   └── test_view_command.py   # subprocess spawn mocked
│   └── dashboard/
│       └── test_data_loader.py    # no streamlit import; exercises pure logic
└── integration/
    ├── cli/
    │   └── test_test_persists_eval_run.py
    └── dashboard/
        └── test_app_smoke.py      # optional, skipif no streamlit extra
```

**Structure Decision**: Option 1 (single project). New code lives in dedicated sub-packages (`lib/eval_run/`, `dashboard/`) to keep imports boundaries clear — in particular, nothing under `dashboard/` may be imported from the rest of the codebase, so Streamlit remains genuinely optional. The CLI detects the extra with `importlib.util.find_spec("streamlit")` before any dashboard import.

**Key architecture decisions recorded during `/speckit.verify`**:
- **Click shape**: `test` becomes a group with an invoke-without-command default callback that preserves today's `holodeck test agent.yaml` behavior; `view` is registered as a subcommand of that group. No top-level `test-view` command.
- **SecretStr migration is in-scope for this feature.** Every secret-bearing field in the agent model tree must migrate from `str` to `SecretStr`, and every reader callsite must be updated to use `.get_secret_value()`. The migration is itemized in `data-model.md` and is a blocker for FR-005 Acceptance Scenario 3.
- **`resolve_prompt_version` is additive**, not a signature change to `resolve_instructions`. Existing callers are untouched.
- **`results/` is rooted at `agent_base_dir`** — consistent with how `instructions.file`, `response_format` files, and vector-store source paths are already resolved.

**Key architecture decisions recorded after the design handoff landed (2026-04-18 revisit)**:
- **`TestResult` grows additively** to carry dashboard-required runtime data: `tool_invocations: list[ToolInvocation]`, `token_usage: TokenUsage | None`, and `MetricResult.kind: Literal["standard","rag","geval"]`. These are additive — legacy EvalRun files load via Pydantic defaults.
- **`ToolInvocation` is the persisted tool-call shape**, deliberately named to avoid the two existing `ToolEvent` classes (streaming UI event, backend hook event). Fields: `name, args, result, bytes, duration_ms?, error?`. See [research.md](./research.md) R8 for per-backend pairing semantics.
- **Conversation discriminated union deferred**: the dashboard's Explorer view renders `test_input` + `agent_response` + parallel `tool_invocations` without needing interleaved turns. `ConversationTurn` union is not in US1 scope.
- **Cost is dashboard-side**: HoloDeck persists `token_usage`; the Streamlit layer multiplies by a pricing table. No dollar values on disk.
- **`pass_rate` scale**: stays `0..100` on disk (existing `ReportSummary` semantics); loader normalises to `0..1` for the dashboard. See [data-model.md](./data-model.md) "Dashboard Field Projections".
- **`run.id` is dashboard-assigned**: loader sets `id = filepath.stem`; no persisted field. Stable across refresh because the filename is stable.

## Complexity Tracking

> No Constitution violations; table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
