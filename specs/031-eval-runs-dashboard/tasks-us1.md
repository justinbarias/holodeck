---
description: "Task list — User Story 1: Persist Strongly-Typed EvalRun Per Test Invocation"
---

# Tasks — US1: Persist Strongly-Typed EvalRun Per Test Invocation (Priority: P1) 🎯 MVP

**Feature**: 031-eval-runs-dashboard
**Spec**: [spec.md](./spec.md) — User Story 1 (P1)
**Plan**: [plan.md](./plan.md)
**Data model**: [data-model.md](./data-model.md)
**Research**: [research.md](./research.md)
**Contract**: [contracts/cli.md](./contracts/cli.md)

**Goal**: Every `holodeck test` invocation writes a strongly-typed `EvalRun` JSON artifact to `<agent_base_dir>/results/<slugified-agent-name>/<ISO-timestamp>.json`. The artifact wraps the existing `TestReport` unchanged and adds an `EvalRunMetadata` block with a redacted `Agent` snapshot and run provenance. Writes are atomic.

**Independent Test**: Run `holodeck test agent.yaml` in a project with no prior runs → a file appears at `results/<slug>/<ts>.json` → `EvalRun.model_validate_json(path.read_text())` round-trips without loss (modulo redacted secrets).

**TDD discipline**: For every task marked "(TDD)", write the failing test first, confirm it fails for the expected reason, then implement and make it pass. Do not implement ahead of the test.

---

## Phase 1: Setup (US1-specific)

- [ ] T001 [US1] Create package directory `src/holodeck/lib/eval_run/` with empty `__init__.py`
- [ ] T002 [US1] Create empty module `src/holodeck/models/eval_run.py` with module docstring and future imports only

---

## Phase 2: Foundational — SecretStr migration (Blocks US1 Acceptance Scenario 3)

**⚠️ CRITICAL**: The `SecretStr` migration is a prerequisite for the type-driven redaction rule (FR-005 rule 2). US1 AC3 cannot pass until this lands.

### Tests first (TDD)

- [ ] T003 [P] [US1] (TDD) Add failing unit test `tests/unit/models/test_secret_fields.py` asserting that `LLMProvider.model_fields["api_key"].annotation` is `Optional[SecretStr]` (or `SecretStr | None`) — expect failure until migration lands
- [ ] T004 [P] [US1] (TDD) Add failing unit tests in `tests/unit/models/test_secret_fields.py` for every field enumerated below (T006): scope is limited to fields that EXIST in the current codebase. Do NOT write tests for `auth_token`, `aws_access_key_id`, `aws_secret_access_key` — those fields do not exist yet and migrating them is out-of-scope for this feature

### Implementation

- [ ] T005 [US1] Migrate `LLMProvider.api_key` from `str | None` to `SecretStr | None` in `src/holodeck/models/llm.py:44`
- [ ] T006 [US1] Migrate the following *existing* plain-`str` secret fields to `SecretStr | None`, verified present in the codebase as of this task list:
    - `VectorStoreConfig.connection_string` (src/holodeck/models/tool.py:212)
    - `KeywordIndexConfig.password` (src/holodeck/models/tool.py:250) and `KeywordIndexConfig.api_key` (src/holodeck/models/tool.py:251)
    - `AzureMonitorExporterConfig.connection_string` (src/holodeck/models/observability.py:229) — handled separately in T007 for clarity
    - **NOTE**: Anthropic `auth_token`, OAuth tokens, and AWS creds (`aws_access_key_id`, `aws_secret_access_key`) are NOT fields that currently exist in `LLMProvider` or `ClaudeConfig`. `ClaudeConfig.AuthProvider` has the ENUM value `oauth_token` but no actual token field. This feature will NOT add new secret fields; any future provider secret must be typed `SecretStr` from day one. Document this invariant in `src/holodeck/models/llm.py` via a module-level comment
- [ ] T007 [US1] Migrate `AzureMonitorExporterConfig.connection_string` to `SecretStr | None` in `src/holodeck/models/observability.py:229`
- [ ] T008 [US1] Audit MCP tool configuration in `src/holodeck/models/tool.py` for any header/token fields currently typed `str`; migrate to `SecretStr | None` only when the field's purpose is to carry a secret. If the audit yields no additional fields, mark T008 as N/A in the commit message (do not invent fields)
- [ ] T009 [US1] Update every reader callsite for the migrated fields above to `.get_secret_value()`: SK backend in `src/holodeck/lib/backends/sk_backend.py`, Claude backend in `src/holodeck/lib/backends/claude_backend.py`, Azure exporter init under `src/holodeck/lib/observability/`, vector-store clients under `src/holodeck/lib/vector_store.py`, MCP bridge in `src/holodeck/lib/backends/mcp_bridge.py`, and any `KeywordIndexConfig` consumer under `src/holodeck/lib/keyword_search.py`
- [ ] T010 [US1] Project-wide grep for attribute accesses on the migrated fields (`.api_key`, `.connection_string`, `.password`) — for each hit, confirm it's either (a) reading via `.get_secret_value()`, (b) on a non-migrated model, or (c) in a test fixture that needs updating to pass `SecretStr(...)` instead of a plain str

**Checkpoint**: All existing tests still pass (no regression); new `test_secret_fields.py` passes.

---

## Phase 2b: Foundational — Runtime data shape migrations (Blocks US4/US5 dashboard rendering)

**⚠️ CRITICAL**: The design handoff's dashboard architecture requires runtime data the codebase does not currently capture. Three `TestResult`-adjacent migrations must land in US1 so that US4/US5 can render real runs (not just seed data). Seed data already has the right shape; the codebase does not.

### Migration A — `MetricResult.kind` discriminator

Without this field, Summary's three breakdown panels (`summary.js:442-458`), the metric-trend kind toggle (`summary.js:433-437`), Explorer's evaluations-by-kind grouping (`explorer.js:204-208`), and Compare's case-matrix scoring (`compare.js:189-196`) cannot render from real `EvalRun` JSON files.

- [ ] T010a [P] [US1] (TDD) `tests/unit/models/test_metric_result_kind.py`: assert `MetricResult.model_fields["kind"].annotation` is `Literal["standard","rag","geval"]`; assert `kind` is required (no default); round-trip via `model_dump_json()` / `model_validate_json()` preserves it
- [ ] T010b [US1] Extend `MetricResult` in `src/holodeck/models/test_result.py:35` with `kind: Literal["standard","rag","geval"] = Field(..., description="Metric family — distinguishes standard NLP / RAG / GEval")`. The discriminator already exists on the evaluation CONFIG side (`models/evaluation.py:43,149,250`) as `type: Literal["standard"|"geval"|"rag"]`; T010b propagates it to runtime results
- [ ] T010c [US1] Update every evaluator writer under `src/holodeck/lib/evaluators/` to populate `kind` when constructing `MetricResult`: NLP metrics (bleu/rouge/meteor) → `"standard"`; GEval judges → `"geval"`; RAG metrics → `"rag"`. Grep for `MetricResult(` and patch each construction site
- [ ] T010d [US1] Back-compat: when loading legacy `EvalRun` JSON files produced before T010b, attempt to infer `kind` from `metric_name` via a small classifier in `src/holodeck/lib/eval_run/legacy.py` (names in {bleu, rouge, meteor, exact_match} → `standard`; names in RAG metric enum → `rag`; unknown → `geval`). Log a WARNING when inference is used

### Migration B — Structured tool-call capture

Without this, Explorer's Conversation panel renders empty tool-call panels for real runs (the handoff's `explorer.js::ToolCall:152-184` requires `{name, args, result, bytes}` per invocation).

- [ ] T010e [P] [US1] (TDD) `tests/unit/models/test_tool_events.py`: assert `TestResult` has a new field `tool_events: list[ToolEvent]` with `default_factory=list`; `ToolEvent` has fields `name: str`, `args: dict[str, Any]`, `result: Any`, `bytes: int` (computed from `len(json.dumps(result))`); existing `tool_calls: list[str]` remains for back-compat
- [ ] T010f [US1] Add `class ToolEvent(BaseModel)` to `src/holodeck/models/tool_event.py` (the file already exists — inspect first for existing shape; extend or replace accordingly). Add `tool_events: list[ToolEvent]` field to `TestResult` in `src/holodeck/models/test_result.py`
- [ ] T010g [US1] Update `src/holodeck/lib/test_runner/executor.py` to capture `ToolEvent` from both SK and Claude backends. Reference: SK exposes tool-call events via `FunctionCallContent` / `FunctionResultContent`; Claude Agent SDK exposes them via message stream. Populate `tool_events` parallel to the existing `tool_calls: list[str]` (which continues to record just names for back-compat)
- [ ] T010h [US1] Back-compat: `tool_calls` list[str] remains; readers (including US5 Explorer) prefer `tool_events` when non-empty, else fall back to rendering `tool_calls` as name-only panels with "args/result not captured" placeholder

### Migration C — Multi-turn conversation capture

Without this, Explorer's Conversation section degrades to "one user bubble + one assistant bubble" for real runs — no interleaved tool-call events, no multi-turn history.

- [ ] T010i [P] [US1] (TDD) `tests/unit/models/test_conversation_turns.py`: assert `TestResult.conversation: list[ConversationTurn]` exists with `default_factory=list`; `ConversationTurn` is a discriminated union `Annotated[UserTurn | AssistantTurn | ToolCallTurn | ToolResultTurn, Discriminator("role")]`. Each turn has `role`; bodies carry `content: str` (user/assistant) or `name + args` (tool_call) or `name + result` (tool_result)
- [ ] T010j [US1] Implement the `ConversationTurn` union in `src/holodeck/models/test_result.py` (or a sibling `conversation.py` if the file grows large). Use the same discriminator pattern already in use for `ToolUnion` (`models/tool.py:913`)
- [ ] T010k [US1] Update `src/holodeck/lib/test_runner/executor.py` to record turns as they occur — backends emit events in order, so this is a stream-consumer pattern: accumulate user → (zero or more tool_call/tool_result pairs) → assistant. Single-turn test cases produce exactly one `UserTurn` + one `AssistantTurn` with any tool events between them
- [ ] T010l [US1] Back-compat: `test_input` and `agent_response` fields remain; when `conversation` is empty (legacy runs), readers synthesize a 2-turn conversation from them

**Checkpoint for Phase 2b**: Unit tests for T010a/T010e/T010i pass. A manual `holodeck test` on a fixture with at least one tool-using case produces a persisted `EvalRun` whose `metadata.agent_config` snapshot AND `report.results[0].tool_events` / `report.results[0].conversation` are non-empty.

---

## Phase 3: US1 Core — EvalRun Persistence

### Tests first (TDD) — write all before implementation

- [ ] T011 [P] [US1] (TDD) `tests/unit/models/test_eval_run.py`: assert `EvalRun` has required `report: TestReport` and `metadata: EvalRunMetadata` fields, enforces `extra="forbid"`, and raises on consistency mismatch when `report.agent_name != metadata.agent_config.name`
- [ ] T012 [P] [US1] (TDD) `tests/unit/models/test_eval_run.py`: round-trip test — build `EvalRun` fixture, `dump = run.model_dump_json()`, `rehydrated = EvalRun.model_validate_json(dump)`, assert `rehydrated == run` (FR-007, AC2)
- [ ] T013 [P] [US1] (TDD) `tests/unit/models/test_eval_run.py`: assert `EvalRunMetadata` has all required fields (`agent_config`, `prompt_version`, `holodeck_version`, `cli_args`, `git_commit`) and enforces `extra="forbid"`; assert it does NOT carry its own `created_at` field (timestamp source of truth is `report.timestamp`)
- [ ] T014 [P] [US1] (TDD) `tests/unit/lib/eval_run/test_slugify.py`: cover lowercase, alphanumerics+`-`, consecutive `-` collapse, leading/trailing `-` strip, non-ASCII → `-`, spaces → `-`, empty-result raises `ValueError` (R4)
- [ ] T015 [P] [US1] (TDD) `tests/unit/lib/eval_run/test_redactor.py`: for each rule — (a) name-allowlist (`api_key`, `password`, `secret`) in any nested model leaf redacted to `"***"`; (b) `SecretStr`-typed field redacted to `"***"` even when its name is not in the allowlist; (c) non-matching fields persisted verbatim; (d) `REDACTED_FIELD_NAMES` is a module-level `frozenset` (FR-005 centralisation)
- [ ] T016 [P] [US1] (TDD) `tests/unit/lib/eval_run/test_writer_atomic.py`: (a) happy path writes final file; (b) patched `os.replace` raising mid-write leaves no `.tmp` file dangling and no partial target; (c) assert `fsync` is called on the temp fd before replace; (d) collision on pre-existing target path appends a 4-hex suffix and writes anyway (FR-008, FR-009b)
- [ ] T017 [P] [US1] (TDD) `tests/unit/lib/eval_run/test_metadata.py`: `build_eval_run_metadata(...)` — git_commit captured when in a git repo (patched subprocess), `None` when `git rev-parse` fails or times out (2s bound), `holodeck_version` read via `importlib.metadata.version`, `cli_args` echoes `sys.argv[1:]`
- [ ] T018 [US1] (TDD) `tests/unit/lib/eval_run/test_writer_path_resolution.py`: writer resolves `results/` against `agent_base_dir` (not CWD); given `agent_base_dir=/tmp/subdir`, target path is `/tmp/subdir/results/<slug>/<ts>.json`
- [ ] T019 [US1] (TDD) `tests/integration/cli/test_test_persists_eval_run.py`: run `holodeck test <fixture>.yaml` end-to-end via `CliRunner`, assert exactly one JSON file appears under `results/<slug>/`, assert `EvalRun.model_validate_json(path.read_text())` succeeds, assert `--output report.md` AND persistence both happen (FR-006, AC5)
- [ ] T020 [US1] (TDD) `tests/integration/cli/test_persistence_failure_does_not_mask_exit.py`: chmod the target directory to `0o400` (or patch writer to raise), assert test exit code is unchanged and a single WARNING line is logged (FR-009)
- [ ] T021 [US1] (TDD) `tests/integration/cli/test_test_persists_eval_run.py`: assert no run file is written when `test_cases` is empty (contracts/cli.md "Emitted when at least one test result is produced")
- [ ] T022 [US1] (TDD) `tests/integration/cli/test_test_persists_eval_run.py`: with a fixture containing `api_key: ${OPENAI_API_KEY}` and `OPENAI_API_KEY=sk-fake-not-real` in env, assert the persisted JSON contains `"api_key": "***"` and never contains the literal `"sk-fake-not-real"` (AC3)

### Implementation (make the tests pass, in order)

- [ ] T023 [P] [US1] Implement `src/holodeck/lib/eval_run/slugify.py` per research.md R4
- [ ] T024 [P] [US1] Implement `PromptVersion` as a stub in `src/holodeck/models/eval_run.py` (fields only, no derivation logic yet — US2 owns derivation); enough to satisfy `EvalRun` type references
- [ ] T025 [US1] Implement `EvalRunMetadata` and `EvalRun` Pydantic models in `src/holodeck/models/eval_run.py` per data-model.md (with `extra="forbid"` and the `report.agent_name == metadata.agent_config.name` consistency validator)
- [ ] T026 [US1] Implement `src/holodeck/lib/eval_run/redactor.py` with module-level `REDACTED_FIELD_NAMES = frozenset({"api_key", "password", "secret"})`, `REDACTED_PLACEHOLDER = "***"`, and `redact(agent: Agent) -> Agent` that walks the nested Pydantic tree (two-rule policy per research.md R5)
- [ ] T027 [US1] Implement `src/holodeck/lib/eval_run/metadata.py` with `build_eval_run_metadata(agent, prompt_version, argv=None) -> EvalRunMetadata` (bounded `git rev-parse HEAD` via `subprocess.run` with `timeout=2`)
- [ ] T028 [US1] Implement `src/holodeck/lib/eval_run/writer.py` with atomic `write_eval_run(run: EvalRun, agent_base_dir: Path) -> Path` using `mkstemp` + `fsync` + `os.replace` per research.md R3, filename derived from `run.report.timestamp` normalised to ms precision, colons replaced with hyphens, `.json` extension, 4-hex collision suffix
- [ ] T029 [US1] Export public API from `src/holodeck/lib/eval_run/__init__.py`: `write_eval_run`, `redact`, `slugify`, `build_eval_run_metadata`, `REDACTED_FIELD_NAMES`
- [ ] T030 [US1] Wire persistence into `src/holodeck/cli/commands/test.py`: after `executor.run()` produces a `TestReport` and after any `--output` emission, build the `EvalRun`, call `write_eval_run`, emit a single informational line `EvalRun persisted: results/<slug>/<ts>.json` (suppressed under `--quiet`), catch `OSError`/`PermissionError` → log WARNING and surface a single-line CLI notice without changing exit code. **Fail-loud stub guard**: before calling `write_eval_run`, if `prompt_version.version == "auto-00000000"` OR `prompt_version.body_hash == "0" * 64`, raise `ConfigError("prompt_version", "PromptVersion resolver stub returned sentinel values — US2 (frontmatter parsing) has not shipped. Refusing to persist an EvalRun with a fake version; this would silently break SC-003/SC-004.")`. Removes the risk of US1 reaching production without US2
- [ ] T031 [US1] Pass `agent_base_dir` through from the CLI loader to the writer (use the existing `holodeck.config.context.agent_base_dir` context variable)
- [ ] T032 [US1] Invoke the US2 `resolve_prompt_version(...)` once its implementation lands; until then, pass a stub `PromptVersion` with `source="file"|"inline"`, `body_hash="0"*64`, `version="auto-00000000"` so US1 can ship independently (US2 replaces the stub call)

**Checkpoint**: All US1 unit+integration tests green. Quickstart steps §2 and §5 reproducible.

---

## Dependencies

- T001–T002 have no dependencies.
- T003–T010 (SecretStr migration) block T015 (redactor test rule 2) and T022 (integration redaction test).
- T011–T022 (all TDD tests) MUST be written and failing before T023–T032 implementation.
- T023–T024 block T025.
- T025–T027 block T028 (writer uses `EvalRun` type and metadata builder).
- T028–T029 block T030.
- T030 blocks T031.
- T032 can be deferred until US2 is merged; US1 ships with the stub.

### Parallel Opportunities

```bash
# TDD phase — all independent test files:
Task: "tests/unit/models/test_eval_run.py (T011–T013)"
Task: "tests/unit/lib/eval_run/test_slugify.py (T014)"
Task: "tests/unit/lib/eval_run/test_redactor.py (T015)"
Task: "tests/unit/lib/eval_run/test_writer_atomic.py (T016)"
Task: "tests/unit/lib/eval_run/test_metadata.py (T017)"

# Implementation phase — independent leaf modules:
Task: "slugify.py (T023)"
Task: "PromptVersion stub in eval_run.py (T024)"
```

---

## Acceptance Scenario Traceability

| AC | Covered by |
|---|---|
| AC1 (file appears at `results/<slug>/<ts>.json`) | T019 |
| AC2 (round-trip) | T012 |
| AC3 (secret redaction) | T015, T022 (requires T003–T010) |
| AC4 (dir auto-created) | T016 (`parent.mkdir(parents=True, exist_ok=True)` inside writer), covered by T019 |
| AC5 (`--output` + persistence coexist) | T019 |
| AC6 (slug sanitised) | T014 |

---

## Implementation Strategy

1. Land Phase 2 (SecretStr migration) first — this is non-negotiable for AC3.
2. Write all TDD tests (T011–T022), confirm they fail for the expected reasons.
3. Implement leaf modules (slugify, PromptVersion stub, redactor, metadata, writer) in dependency order.
4. Wire into the CLI last (T030–T031).
5. Run `make test-unit` and `make test-integration` filtered to the new files.
6. Run quickstart §2 and §5 by hand to validate end-to-end.
