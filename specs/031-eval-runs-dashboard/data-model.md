# Phase 1: Data Model

**Feature**: 031-eval-runs-dashboard

This document enumerates the new Pydantic models introduced by the feature, their fields, validation rules, and the relationships to existing models. All models live under `src/holodeck/models/eval_run.py` unless noted otherwise.

## Entity Map

```
EvalRun
├── report: TestReport              # existing shape, additively extended
│   ├── summary: ReportSummary      # existing, unchanged on-disk
│   ├── results: list[TestResult]
│   │   ├── test_input: str         # existing — the user turn
│   │   ├── agent_response: str     # existing — the assistant turn
│   │   ├── tool_calls: list[str]   # existing — names only, kept for back-compat
│   │   ├── tool_invocations: list[ToolInvocation]   # NEW (additive)
│   │   ├── token_usage: TokenUsage | None           # NEW (additive)
│   │   └── metric_results: list[MetricResult]
│   │       └── kind: "standard" | "rag" | "geval"   # NEW (additive)
│   └── timestamp: str              # existing — dashboard projects to `created_at`
└── metadata: EvalRunMetadata
    ├── agent_config: Agent         # existing, snapshot (redacted)
    ├── prompt_version: PromptVersion
    ├── holodeck_version: str       # dashboard projects to root-level `holodeck_version`
    ├── cli_args: list[str]         # sanitized argv
    └── git_commit: str | None      # dashboard projects to root-level `git_commit`
```

> **No persisted `id` field.** `run.id` in the handoff dataset is a **dashboard-assigned** identifier computed by `data_loader` from the filename stem (e.g. `2026-04-18T14-22-09.812Z` → id `2026-04-18T14-22-09.812Z`). No on-disk schema change; documented here so future dashboard contributors know where the id originates.

---

## `PromptVersion`

Location: `src/holodeck/models/eval_run.py`

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | `str` | ✅ | Manual value from frontmatter `version:` key, or auto-derived `auto-<sha256[:8]>`. |
| `author` | `str \| None` | — | From frontmatter `author:` key. |
| `description` | `str \| None` | — | From frontmatter `description:` key. |
| `tags` | `list[str]` | — | Default `[]`. From frontmatter `tags:` key. Accepts YAML list. |
| `source` | `Literal["file", "inline"]` | ✅ | `"file"` when derived from `instructions.file`; `"inline"` otherwise. |
| `file_path` | `str \| None` | — | Absolute or agent-relative path; `None` when `source="inline"`. |
| `body_hash` | `str` | ✅ | Full 64-char SHA-256 hex of the prompt body (frontmatter stripped). |
| `extra` | `dict[str, Any]` | — | Default `{}`. Any frontmatter keys outside the documented schema are preserved here (FR-016). |

**Validation**:
- `model_config = ConfigDict(extra="forbid")` on the model itself so the documented keys are authoritative; unknown *frontmatter* keys land in `extra` rather than failing validation (the allowlist is applied by the parser, not by Pydantic).
- `body_hash` must match `^[a-f0-9]{64}$`.
- When `source == "inline"`, `file_path` MUST be `None`.
- When `source == "file"`, `file_path` MUST be non-empty.

**Derivation rules** (FR-011 to FR-016):
1. If `instructions.inline` is used → `source="inline"`, `file_path=None`, `version="auto-" + sha256(inline)[:8]`.
2. Else `frontmatter.load(file_path)`:
   - `post.content` → hashed → `body_hash`.
   - `post.metadata.get("version")` → `version` if present, else `"auto-" + body_hash[:8]`.
   - Recognised keys: `version`, `author`, `description`, `tags`.
   - All other keys → `extra`.
3. Malformed YAML → `ConfigError` re-raise (FR-017).

---

## `EvalRunMetadata`

Location: `src/holodeck/models/eval_run.py`

| Field | Type | Required | Description |
|---|---|---|---|
| `agent_config` | `Agent` | ✅ | Deep snapshot of the validated `Agent` model, post-redaction. |
| `prompt_version` | `PromptVersion` | ✅ | Prompt identity at run time. |
| `holodeck_version` | `str` | ✅ | `importlib.metadata.version("holodeck-ai")` or `"0.0.0.dev0"`. |
| `cli_args` | `list[str]` | ✅ | `sys.argv[1:]` with any secret-like token values (`--api-key=xxx`) sanitized; stored as `list[str]` for faithful preservation of order/quoting. In v1, sanitization is conservative — we emit the argv verbatim unless a token matches a secret-like option prefix (empty set in v1; reserved for future). |
| `git_commit` | `str \| None` | — | Best-effort `git rev-parse HEAD` in CWD; `None` when not a git repo or `git` binary missing. |

> **Timestamp source of truth**: `EvalRunMetadata` intentionally does not carry its own `created_at` field. `EvalRun.report.timestamp` (the existing `TestReport.timestamp`, stamped at end of test execution in `src/holodeck/models/test_result.py:171`) is the single source of truth for both on-disk ordering and filename derivation. This avoids a second timestamp drifting from the first by the persistence-assembly delta.

**Validation**:
- `model_config = ConfigDict(extra="forbid")`.

**Provenance**:
- `git_commit` uses a bounded `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, timeout=2)` call. Any non-zero exit or timeout yields `None`.

---

## `ToolInvocation` (new — persisted tool-call record)

Location: `src/holodeck/models/test_result.py` (co-located with `TestResult`; avoids cross-module import cycles).

> **Naming**: deliberately **not** `ToolEvent`. Two `ToolEvent` classes already exist (`models/tool_event.py` for streaming UI events, `lib/backends/base.py` for backend hook events). `ToolInvocation` unambiguously denotes a completed call+result pair.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | ✅ | Tool name invoked by the agent (e.g. `"lookup_order"`). |
| `args` | `dict[str, Any]` | ✅ | Tool input parameters. Empty dict when the tool takes no args. |
| `result` | `Any` | ✅ | Tool output — may be a JSON-serialisable scalar, dict, list, or string. |
| `bytes` | `int` | ✅ | `len(json.dumps(result, default=str))`. Used by the dashboard to collapse-by-default large results. |
| `duration_ms` | `int \| None` | — | Tool execution time in milliseconds. `None` when the backend does not report it (SK Plugin path). |
| `error` | `str \| None` | — | Error message when the tool invocation failed. `result` should be `None` in that case. |

**Validation**:
- `model_config = ConfigDict(extra="forbid")`.
- `bytes >= 0`.
- When `error` is set, `result` SHOULD be `None` — not enforced, but a writer invariant.

**Pairing semantics** (see research.md R8):
- **SK backend**: `tool_calls` and `tool_results` are parallel lists produced in call-order. Pair by index. If lengths diverge (rare — tool raised before returning), pad the shorter list with synthesised records carrying `error`.
- **Claude backend**: every `FunctionCallContent` carries a `tool_use_id` matched by the downstream `FunctionResultContent`. Pair by id. Unmatched ids → record with `error="no result received"`.

---

## `TestResult` additive fields

Location: `src/holodeck/models/test_result.py` (existing model — new fields appended).

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_invocations` | `list[ToolInvocation]` | — | Default `[]`. Parallel to `tool_calls` (names-only) for back-compat. Dashboard prefers `tool_invocations` when non-empty. |
| `token_usage` | `TokenUsage \| None` | — | Default `None`. Populated from `ExecutionResult.token_usage`. Required for Compare view cost computation (dashboard multiplies tokens by a pricing table; HoloDeck itself does not persist a dollar value). |

> **Deferred**: the `conversation: list[ConversationTurn]` discriminated union originally proposed for US1 is **deferred to a later user story**. The dashboard's Explorer view renders the existing `test_input` + `agent_response` + parallel `tool_invocations` shape without needing interleaved turns. Revisit if multi-turn agent sessions become a first-class test-case scenario.

---

## `EvalRun`

Location: `src/holodeck/models/eval_run.py`

| Field | Type | Required | Description |
|---|---|---|---|
| `report` | `TestReport` | ✅ | Existing model, re-exported from `holodeck.models.test_result`. Additive fields on `TestResult` (see above) apply transitively. |
| `metadata` | `EvalRunMetadata` | ✅ | Snapshot and provenance. |

**Validation**:
- `model_config = ConfigDict(extra="forbid")`.
- `report.agent_name` must equal `metadata.agent_config.name` (consistency check).

**Round-trip** (FR-007, SC-002):
```python
data = run.model_dump_json(indent=2)
rehydrated = EvalRun.model_validate_json(data)
assert rehydrated == run  # modulo already-redacted secrets
```

**Snapshot semantics** (Story 3):
- The `metadata.agent_config` field is a **deep copy at serialization time**. The writer builds `EvalRun` immediately after test execution; `Agent` models are immutable post-validation, so a direct reference is safe within the invocation. On disk, subsequent edits to `agent.yaml` have no effect on prior `EvalRun` files (FR-032 / Acceptance Scenario 5).

---

## Redaction Policy Surface

Centralised constant (FR-005):

```python
# src/holodeck/lib/eval_run/redactor.py
REDACTED_FIELD_NAMES: frozenset[str] = frozenset({"api_key", "password", "secret"})
REDACTED_PLACEHOLDER: str = "***"
```

Two-rule policy:

1. Walk every `BaseModel` in the nested `Agent` tree:
   - For each field whose **leaf name** is in `REDACTED_FIELD_NAMES` → replace with `"***"`.
   - For each field whose **annotated type** is `SecretStr` (or `SecretStr | None`) → replace with `"***"`.
2. Non-matching fields are persisted verbatim.

**Migration required before this feature merges** (per spec Assumption 4): audit the following models and confirm each secret-bearing field is typed as `SecretStr`. Any plain-`str` offender must be migrated in the same PR family:

| Model | Field(s) expected to be `SecretStr` |
|---|---|
| `holodeck.models.llm.LLMProvider` | `api_key` (existing allowlist name — also covered by rule 1) |
| Anthropic-specific config (`LLMProvider` / `ClaudeConfig`) | `auth_token`, any `oauth_token` |
| AWS creds path (bedrock/vertex routes) | `aws_access_key_id`, `aws_secret_access_key` |
| Azure config | `connection_string`, `api_key` |
| MCP tool config (stdio env, SSE headers) | any `Authorization` header values; any field named `*_token` |

The migration is **required** so that non-allowlisted names (e.g. `auth_token`, `connection_string`) get covered by rule 2. Story 1 Acceptance Scenario 3 is blocked until this migration lands.

---

## Filename Convention

Path: `<agent_base_dir>/results/<slugify(agent.name)>/<iso_ts>.json`

Where:
- `agent_base_dir` is the directory containing `agent.yaml`, resolved via the existing `holodeck.config.context.agent_base_dir` context variable. This matches the convention already used for `instructions.file`, `response_format` files, and vector-store source paths.
- `slugify` is as defined in research.md R4.
- `iso_ts` is `report.timestamp` reformatted for filesystem safety:
  - `2026-04-18T14:22:09.812Z` → `2026-04-18T14-22-09.812Z` (colons replaced with hyphens).
- If the existing `TestReport.timestamp` does not already have millisecond precision, the writer normalises it to ms precision before deriving the filename (source field unchanged in the persisted JSON).
- Collision suffix: on pre-existing path, append `-<4-hex>` before `.json` (FR-008).

---

## State & Invariants

- `EvalRun` is **immutable** once written.
- The dashboard is read-only; it never mutates nor deletes files.
- Readers tolerate `EvalRun` files where newly-added optional fields are absent (Pydantic defaults). The existing-only `TestReport` is unchanged, so legacy reports remain loadable.

---

## Dashboard Field Projections

The Streamlit dashboard (US4/US5) consumes `EvalRun` JSON via `dashboard/data_loader.py`. Design-handoff field names (from `design_handoff_holodeck_eval_dashboard/data.js`) **do not** match Python model field names 1:1. The loader is the single place where projection happens; no on-disk rename.

| On-disk (Pydantic) | Dashboard (handoff) | Used by |
|---|---|---|
| `report.timestamp` | `run.created_at` | every view (axis labels, sort) |
| `report.summary.total_tests` | `summary.total` | Summary table "Tests" col, Compare matrix |
| `report.summary.total_duration_ms` | `summary.duration_ms` | KPI strip, Compare "Duration" row |
| `report.summary.pass_rate` **(0..100)** | `summary.pass_rate` **(0..1)** | **Loader divides by 100** — see below |
| `report.results[].test_input` | `testCase.input` | Explorer conversation user bubble |
| `report.results[].tool_calls` (names) | `testCase.tools_called` | Compare matrix tool-call summary |
| `report.results[].tool_invocations` | `testCase.tool_calls` (in Explorer detail) | Explorer `ToolCall` panel |
| `agent_config.tools[].type` (ToolUnion discriminator) | `agent_config.tools[].kind` | Explorer config-snapshot tool chips |
| `agent_config.embedding_provider` | `agent_config.embedding` | Explorer config-snapshot rows |
| `metadata.holodeck_version` | `run.holodeck_version` (root) | (informational) |
| `metadata.git_commit` | `run.git_commit` (root) | Summary table "Commit" col, Compare header |
| `metadata.prompt_version.*` | `run.metadata.prompt_version.*` | **No projection** — already nested correctly |

**Scale conversion**: `pass_rate` is persisted as an integer-ish percentage `0..100` (existing `ReportSummary` semantics, documented in the Pydantic field description). The design-handoff assumes `0..1`. The loader divides on read — decided in preference to a breaking migration of `ReportSummary` that would churn the markdown reporter, existing TestReport JSON files on disk, and every integration test that asserts against the markdown output.

**`run.id` synthesis**: not a persisted field. The loader computes `id = filepath.stem` (e.g. `"2026-04-18T14-22-09.812Z"` or `"2026-04-18T14-22-09.812Z-a3f9"` when the writer appended a collision suffix). This id is stable across loader invocations — it is **safe** to store in `st.session_state.compare_queue` / `st.query_params` because it survives server restarts as long as the file is on disk.
