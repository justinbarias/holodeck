# Phase 0 Research: Eval Runs, Prompt Versioning, and Test View Dashboard

**Feature**: 031-eval-runs-dashboard
**Status**: Complete — no `NEEDS CLARIFICATION` remains.

All clarifications in the spec (Clarifications / Session 2026-04-18) have already been resolved by the product owner. This document records the technical research supporting those decisions and the library-level choices required by the plan.

---

## R1. `python-frontmatter` — prompt versioning parser

**Decision**: Adopt `python-frontmatter>=1.1,<2.0` as a core dependency.

**Rationale**:
- MIT licensed, pure Python, minimal surface area, widely adopted by static-site tooling.
- Its only runtime dep is `PyYAML`, already present transitively in HoloDeck.
- It handles the three spec-required cases cleanly:
  - Frontmatter present → `post.metadata` is a dict, `post.content` is the body.
  - No frontmatter → `post.metadata == {}`, `post.content == full file`. **No exception is raised**, satisfying FR-011 and Acceptance Scenario 3 of Story 2.
  - Malformed YAML inside a `---` block → raises `yaml.YAMLError` (subclasses `ScannerError`/`ParserError`), which we catch and re-raise as `ConfigError` per FR-017.
- Accepts `Path`, `str` path, or file-like object via `frontmatter.load(...)`; accepts a string via `frontmatter.loads(...)`. The latter is what we use for the `instructions.inline` case — actually we *skip* parsing for inline per FR-014 (no `---` fences expected).

**API usage pattern**:

```python
import frontmatter
import yaml
from holodeck.lib.errors import ConfigError

try:
    post = frontmatter.load(instructions_path)
except yaml.YAMLError as e:
    raise ConfigError(
        "instructions.file",
        f"Malformed YAML frontmatter in {instructions_path}: {e}",
    ) from e

body: str = post.content                 # stripped of frontmatter block
metadata: dict[str, Any] = post.metadata # {} when no frontmatter
```

**Integration point (decided during /speckit.verify)**: this logic lives in a **new** module `src/holodeck/lib/prompt_version.py` that exposes `resolve_prompt_version(instructions: Instructions, base_dir: Path | None) -> PromptVersion`. The existing `resolve_instructions()` keeps its `-> str` contract and is **not** modified — its two callers (`test_runner/agent_factory.py:1151` and `lib/backends/claude_backend.py:275`) stay untouched. The eval-run writer invokes `resolve_prompt_version()` separately on the same `Instructions` object at persist time. This avoids a cross-cutting signature change.

**Alternatives considered**:
- **Hand-roll `---` fence parser** — rejected: non-trivial edge cases (escaped fences, BOM, trailing whitespace); the library is tiny and mature.
- **`markdown-it-py` front-matter plugin** — rejected: pulls in a full markdown tokenizer we don't need.
- **`ruamel.yaml` with manual fence splitting** — rejected: more code for no gain; `python-frontmatter` already uses `PyYAML` under the hood.

**Version auto-derivation (FR-013)**:

```python
import hashlib
body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
auto_version = f"auto-{body_hash[:8]}"
```

The full 64-hex-char `body_hash` is stored on `PromptVersion.body_hash` so the dashboard can recover deterministic equality across runs even when a manual version is present.

---

## R2. Streamlit — dashboard runtime

**Decision**: Adopt `streamlit>=1.36,<2.0` as an **optional extra** (`holodeck[dashboard]`). Launch via `subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path, ...])`.

**Rationale**:
- Streamlit ≥1.36 provides the modern multi-page API (`st.Page`, `st.navigation`) which is declarative and avoids the legacy `pages/` directory convention. It also provides `st.query_params` (stable since 1.30) which FR-028b requires for shareable filter state.
- Spawning `streamlit` as a subprocess via the `-m streamlit` entry point is the officially recommended approach and is what the `streamlit` console script does internally. It gives:
  - Correct venv/interpreter resolution (`sys.executable`).
  - Clean signal handling — forwarding `SIGINT` to the child on Ctrl+C terminates the server predictably (Acceptance Scenario 7 of Story 4).
  - No dependency on Streamlit's private `streamlit.web.bootstrap` API, which has changed between minor versions.
- Binding to `0.0.0.0` is Streamlit's default — no flag needed. Per clarification, network exposure is the user's responsibility; we emit a one-line launch-time warning per FR-020.

**Subprocess pattern**:

```python
import os, signal, subprocess, sys
from pathlib import Path

app_path = Path(holodeck.dashboard.__file__).parent / "app.py"
env = {**os.environ,
       "HOLODECK_DASHBOARD_RESULTS_DIR": str(results_dir),
       "HOLODECK_DASHBOARD_AGENT_NAME": agent_name_slug}
argv = [sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false"]
proc = subprocess.Popen(argv, env=env)
try:
    proc.wait()
except KeyboardInterrupt:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
```

**Configuration passing**: Environment variables (`HOLODECK_DASHBOARD_*`) are preferred over `sys.argv` forwarding — they are inherited cleanly by the child and trivially read inside the Streamlit app via `os.environ`. `st.query_params` is reserved for user-driven filter state (FR-028b), not bootstrapping.

**Install detection** (FR-022):

```python
from importlib.util import find_spec
if find_spec("streamlit") is None:
    click.echo(
        "Dashboard not installed. Install the optional extra:\n"
        "  uv add 'holodeck-ai[dashboard]'   # or: pip install 'holodeck-ai[dashboard]'",
        err=True,
    )
    raise click.exceptions.Exit(code=2)
```

`find_spec` does **not** import Streamlit (which is heavy and has side effects), so the hint fires instantly and cleanly.

**Layout primitives used** (no design-system opinions — user is building one separately):

| Spec Requirement | Streamlit Primitive |
|---|---|
| Multi-page navigation (Summary / Explorer) | `st.Page(...)` + `st.navigation([...])` |
| Sidebar with faceted filters (FR-028a) | `st.sidebar` + `st.date_input`, `st.multiselect`, `st.slider` |
| Run table (FR-025) | `st.dataframe(df, on_select="rerun")` for click-to-drill |
| Pass-rate-over-time chart (FR-026) | `st.line_chart(df)` or `st.altair_chart(...)` for tooltips |
| Per-metric avg trends (FR-027) | `st.line_chart(df)` per metric type |
| Breakdown panels for `standard`/`rag`/`geval` (FR-028) | `st.tabs(...)` or three `st.container()` panels |
| Chat-style conversation (Explorer) | `st.chat_message("user")` / `st.chat_message("assistant")` |
| Tool calls distinct from chat | `st.container(border=True)` + `st.json(..., expanded=False)` |
| Collapsible large tool results (FR-032) | `st.expander("Tool result (N KB)", expanded=False)` + `st.json` |
| Filter state in URL (FR-028b) | `st.query_params.to_dict()` / `st.query_params.from_dict(...)` |

**Alternatives considered**:
- **Gradio** — rejected: more opinionated about ML-model demos, weaker multi-page story, no native query-param plumbing.
- **Plain FastAPI + a single HTML page** — rejected: requires a JS frontend to match filter interactivity; vastly more code for a read-only viewer.
- **Jupyter Lab / Voilà** — rejected: not a zero-config install for non-notebook users; awkward for a CLI-launched experience.
- **Invoking `streamlit.web.bootstrap.run()` in-process** — rejected: private API, unstable across versions, and complicates signal handling (Streamlit uses Tornado's ioloop which competes with Click's signal setup).

---

## R3. Atomic JSON writes

**Decision**: `tmp + fsync + os.replace` pattern on POSIX; `os.replace` is atomic on both POSIX and Windows.

**Rationale**: Spec FR-009b mandates readers see either the old file or the new file, never a partial one. Python's `os.replace` provides an atomic rename on the same filesystem. The sequence is:

```python
import os, tempfile, json
from pathlib import Path

def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)          # atomic
    except BaseException:
        try: os.unlink(tmp_name)
        except FileNotFoundError: pass
        raise
```

- `mkstemp` in the same directory ensures `os.replace` is on the same filesystem (rename within a directory is always same-fs on POSIX).
- `fsync` flushes the page cache to disk before replace — required if the machine crashes mid-write.
- We do **not** fsync the parent directory after replace. This is a deliberate simplification: a post-crash directory listing may briefly omit the new filename, but no reader will ever observe a partial file — which is the spec's stated invariant. Adding parent-directory fsync is a future hardening option, not a v1 requirement.

**Collision handling (FR-008)**: Timestamp uses millisecond precision (`2026-04-18T14-22-09.812Z.json`); if collision occurs (two runs in the same ms, rare), append a 4-hex random suffix before the extension. Implemented in `writer.py`.

---

## R4. Slugification for `results/<slugified-agent-name>/`

**Decision**: Lowercase + alphanumerics + `-`; non-conforming characters replaced with `-`; consecutive `-` collapsed; leading/trailing `-` stripped. Empty result fails loudly (but spec notes this can't happen since `agent.name` validation already rejects empty).

**Rationale**: The existing HoloDeck codebase has no single canonical slugifier; this trivial rule avoids adding a dependency (e.g. `python-slugify`) and matches the spec's "documented rule" (Assumption 9). Collisions across distinct agent names are a known v1 limitation.

```python
import re

def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"agent.name slugified to empty string: {name!r}")
    return slug
```

---

## R5. Secret redaction policy

**Decision**: Two-rule redaction enforced by a single module, `lib/eval_run/redactor.py`:

1. **Name allowlist** (centralised as `REDACTED_FIELD_NAMES = frozenset({"api_key", "password", "secret"})`). Exact-match on the leaf field name.
2. **Type-driven** (`isinstance(value, pydantic.SecretStr)` after Pydantic dump, OR discovered by walking model fields via `model_fields` and checking annotation).

The redactor operates on the **model instance**, not the serialized JSON, so it can see Pydantic field types. Redacted values become the literal string `"***"`.

**Migration requirement** (per spec Assumption 4): Before this feature ships, any provider-secret field typed as plain `str` (e.g. Anthropic `auth_token`, OAuth tokens, AWS access keys, connection strings) MUST be migrated to `SecretStr`. The Phase-1 data-model deliverable enumerates the fields that need migration.

**Rationale**: Whitelisting the exact allowlist in one module satisfies FR-005 ("centralised in a single module-level constant for discoverability"). Type-driven redaction is the durable mechanism — future secret fields added with `SecretStr` are covered automatically without updating the allowlist.

---

## R6. Dashboard data loading — skip-on-corrupt

**Decision**: The dashboard's `data_loader.load_all(results_dir)` iterates `*.json`, attempting `EvalRun.model_validate_json(path.read_text())`. On `ValidationError`, `json.JSONDecodeError`, or `OSError`, the file is skipped with a `logging.warning(...)` and loading continues (FR-024).

**Rationale**: Defense-in-depth. The atomic-write path (R3) should prevent partial files produced by `holodeck test`, but files produced by other tooling, manual edits, or interrupted legacy writes must not crash the viewer.

---

## R7. Streamlit testing posture

**Decision**: No unit tests against the Streamlit UI itself. Unit tests cover:
- `dashboard/data_loader.py` — pure Python, exercises loading, filtering, aggregation, skip-on-corrupt.
- `dashboard/filters.py` — round-trip of filter state ↔ `dict` (the `st.query_params` boundary is thin).

An optional smoke integration test under `tests/integration/dashboard/test_app_smoke.py` uses `streamlit.testing.v1.AppTest` if available; it is `pytest.importorskip("streamlit")`-guarded so the core test suite still runs without the extra installed.

**Rationale**: Streamlit UI tests are fragile and expensive. The data layer carries the correctness-critical logic; the UI is thin glue.

---

## R8. Tool call/result pairing for `ToolInvocation` persistence

**Decision**: Pair backend-emitted tool-call records with their result records at executor time, producing a list of `ToolInvocation(name, args, result, bytes, duration_ms?, error?)` on each `TestResult`. Pairing strategy differs per backend; both land in a single uniform `ToolInvocation` shape.

**Background**: `ExecutionResult.tool_calls` and `ExecutionResult.tool_results` are two parallel `list[dict[str, Any]]` fields ([backends/base.py:38–39](../../src/holodeck/lib/backends/base.py)). Today the executor only extracts tool **names** via `extract_tool_names(exec_result.tool_calls)` and persists them to `TestResult.tool_calls: list[str]` — the `args` and the `result` payload are discarded. The design dashboard's Explorer view (explorer.js:152–184) cannot render `{name, args, result, bytes}` panels without this data.

**Pairing semantics**:

| Backend | Call source | Result source | Correlation key | Failure mode |
|---|---|---|---|---|
| Semantic Kernel (`sk_backend.py`) | `FunctionCallContent` items on chat history | `FunctionResultContent` items on chat history | Positional index in call-order | If the call raised before returning, `tool_results` is shorter than `tool_calls` — pad with `ToolInvocation(error="no result received", result=None)` preserving call-order. |
| Claude Agent SDK (`claude_backend.py`) | `ToolUseBlock` in assistant messages | `ToolResultBlock` in user-role messages | `tool_use_id` field (guaranteed unique within a turn) | Unmatched `tool_use_id` → record with `error="no result received"`. A result without a prior call is a protocol violation — log WARNING and skip. |

**Shape contract** (enforced by `ToolInvocation` validator):

```python
class ToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    bytes: int = Field(ge=0)
    duration_ms: int | None = None
    error: str | None = None
```

`bytes = len(json.dumps(result, default=str))` — `default=str` ensures non-JSON-serialisable values (datetimes, Pydantic models) don't raise during size measurement. The persisted `result` is coerced to a JSON-safe shape by the writer (`json.loads(json.dumps(result, default=str))`) before it reaches `ToolInvocation` — this is lossy for non-serialisable values but matches on-disk semantics.

**Implementation locus**: a new helper `pair_tool_calls(exec_result: ExecutionResult, backend_kind: str) -> list[ToolInvocation]` in `src/holodeck/lib/test_runner/tool_invocation_builder.py`. The executor calls it at [executor.py:602–604](../../src/holodeck/lib/test_runner/executor.py) immediately after `exec_result` arrives, replacing the current `tool_calls = extract_tool_names(...)` line with both a names-list (for back-compat) and a `tool_invocations` list.

**Alternatives considered**:
- **Dict-merge heuristic** (match by tool name only) — rejected: ambiguous when the same tool is called multiple times in one turn.
- **Single unified stream of `ToolEvent(kind="start"|"end")` events from the backend** — already exists as `lib/backends/base.py::ToolEvent` for streaming, but consumers must materialise a paired list anyway; we'd be pushing complexity into every consumer.

---

## R9. `token_usage` persistence on `TestResult`

**Decision**: Add `TestResult.token_usage: TokenUsage | None = None`. Executor copies from `exec_result.token_usage`. `TokenUsage` model already exists at `src/holodeck/models/token_usage.py` and is already carried on `ExecutionResult` by both backends — the only gap is that the executor drops it on the floor today.

**Rationale**: Required by the Compare view's cost computation. The design handoff (compare.js:31–32) synthesises cost from `duration_ms * rate` with a hardcoded per-model rate — acceptable for a prototype, not for a real dashboard. Persisting token counts lets the dashboard multiply by a pricing table (also out-of-scope for US1; V1 dashboard may still fall back to the synthetic formula until the pricing table ships). HoloDeck itself never persists a dollar value — the aggregation lives in the viewer.

**Why not in US1 scope before now**: `token_usage` was already on `ExecutionResult` but the executor discarded it because no downstream consumer needed it pre-dashboard. Adding it here costs one field + one line in the executor.

**Executor wire-up** (single line at [executor.py:670–684](../../src/holodeck/lib/test_runner/executor.py) `TestResult(...)` construction):

```python
return TestResult(
    ...,
    tool_invocations=tool_invocations,   # from R8
    token_usage=exec_result.token_usage if exec_result else None,   # from R9
    ...
)
```

---

## Summary of key decisions

| Topic | Choice |
|---|---|
| Frontmatter parser | `python-frontmatter>=1.1` (core dep) |
| Prompt-version resolver | New `lib/prompt_version.py` (additive); `resolve_instructions` unchanged |
| Dashboard framework | Streamlit ≥1.36 (optional `dashboard` extra) |
| Dashboard launch | `subprocess.Popen([sys.executable, "-m", "streamlit", "run", ...])` |
| Dashboard config passing | Env vars `HOLODECK_DASHBOARD_*` |
| URL filter state | `st.query_params` |
| Install detection | `importlib.util.find_spec("streamlit")` |
| Atomic write | `mkstemp` + `fsync` + `os.replace` (no parent-dir fsync in v1) |
| Slugifier | Hand-rolled `[a-z0-9-]+`; no new dep |
| Redaction | Name allowlist + `SecretStr` type-driven; centralised constant |
| `results/` root | `agent_base_dir/results/<slug>/` (matches `instructions.file` resolution) |
| Click shape | `holodeck test` converts to a group with invoke-without-command default callback; `view` registered as subcommand |
| SecretStr migration | In-scope for this feature; every callsite updated to `.get_secret_value()` |
| Corrupt-file handling | Log + skip (dashboard) |
| UI testing | Data layer unit-tested; Streamlit UI smoke-tested optionally |
| Tool-call persistence shape | `ToolInvocation{name, args, result, bytes, duration_ms?, error?}` — pairing per R8 |
| Token usage persistence | `TestResult.token_usage: TokenUsage \| None` — copied from `ExecutionResult` |
| Cost computation | Dashboard-side from `token_usage × pricing_table` (HoloDeck persists no dollar value) |
| Conversation discriminated union | **Deferred** to a follow-up user story — `test_input` + `agent_response` + parallel `tool_invocations` suffices for US4/US5 |
