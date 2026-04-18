# Phase 1: Data Model

**Feature**: 031-eval-runs-dashboard

This document enumerates the new Pydantic models introduced by the feature, their fields, validation rules, and the relationships to existing models. All models live under `src/holodeck/models/eval_run.py` unless noted otherwise.

## Entity Map

```
EvalRun
├── report: TestReport              # existing, unchanged
└── metadata: EvalRunMetadata
    ├── agent_config: Agent         # existing, snapshot (redacted)
    ├── prompt_version: PromptVersion
    ├── created_at: str             # ISO-8601 with ms
    ├── holodeck_version: str
    ├── cli_args: list[str]         # sanitized argv
    └── git_commit: str | None
```

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

## `EvalRun`

Location: `src/holodeck/models/eval_run.py`

| Field | Type | Required | Description |
|---|---|---|---|
| `report` | `TestReport` | ✅ | Existing model, re-exported from `holodeck.models.test_result`. Unchanged. |
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
