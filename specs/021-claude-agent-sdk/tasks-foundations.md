# Tasks: Native Claude Agent SDK Integration — Foundations (Phase 0–3)

**Feature**: `021-claude-agent-sdk`
**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Scope**: Phases 0–3 only (SDK installation → verification → model extensions → validation layer).
Phases 4–12 will be in `tasks.md` once generated via `/speckit.tasks`.
**Approach**: TDD — test tasks are written and **must fail** before implementation tasks begin
**References**: Each task includes a document reference `(doc:L<line>)` pointing to the relevant design decision

---

## Format: `[ID] [P?] Description (doc:Lline)`

- **[P]**: Task is parallelizable (different files, no unresolved dependencies)
- No `[US?]` labels in Phases 0–3: all tasks are setup/foundational prerequisites
- **TDD rule**: Test tasks MUST precede their implementation tasks in every phase

---

## Phase 0: SDK Installation + Verification Smoke Test

**Purpose**: Mandatory gate — install the SDK, then confirm the actual Claude Agent SDK API against all `[ASSUMED]` names in `research.md §2` before any production code is written. **No implementation begins until this phase passes.**

**Gate criteria**: All `[ASSUMED]` markers in `research.md §2` are replaced with confirmed names. If any name differs from the spec, `plan.md`, `data-model.md`, and `quickstart.md` are updated before Phase 1 begins.

> **TDD note**: Phase 0 is exploratory verification, not production code. There are no failing unit tests to write first — the smoke test script *is* the test.

- [x] T001 Add `claude-agent-sdk>=0.1.39,<0.2.0` to `pyproject.toml` dependencies and run `uv sync` to verify resolution — **must run before T002; the smoke test cannot import the SDK until this completes** (plan.md:L180-183, research.md:L11-13)
- [x] T002 Create `scripts/smoke_test_sdk.py` importing and exercising `ClaudeAgentOptions`, `PermissionMode`, `@tool`, `create_sdk_mcp_server`, `ResultMessage`, `ClaudeSDKClient`, and `query` to confirm actual class/function names (plan.md:L144-154, quickstart.md:L57-94, research.md:L48-55)
- [x] T003 Extend `scripts/smoke_test_sdk.py` with a 2-turn conversation using `ClaudeSDKClient` to confirm whether multi-turn state is automatic or requires `continue_conversation=True` in `ClaudeAgentOptions` (plan.md:L155-168, research.md:L56-58)
- [x] T004 Update `research.md §2` replacing every `[ASSUMED]` marker with the confirmed API name from T002–T003 (plan.md:L170-172, research.md:L27-58)
- [x] T005 Update `plan.md`, `data-model.md`, and `quickstart.md` where any assumed API name proved incorrect in T002–T003 (plan.md:L172-173)

**Checkpoint**: All `[ASSUMED]` markers in `research.md §2` resolved. After T004/T005, scan all line number references in T006–T023b and update any that shifted due to doc edits before starting Phase 1.

---

## Phase 1: Backends Package + Core Interfaces

**Purpose**: Establish the `lib/backends/` package skeleton and implement the provider-agnostic `ExecutionResult` / `AgentSession` / `AgentBackend` interfaces that all subsequent phases depend on.

**⚠️ CRITICAL**: `base.py` (T008) must be complete before ANY user story phase can proceed. All downstream consumers — test runner, chat layer — depend on `ExecutionResult`.

### Tests for Phase 1 (write these first — they MUST fail before T008)

- [x] T006 [P] Write unit tests for `ExecutionResult` construction, field defaults, and all error condition scenarios (`max_turns`, subprocess crash, tool failure, timeout) in `tests/unit/lib/backends/test_base.py`. Use `TokenUsage.zero()` (not `TokenUsage()`) for the default token_usage field. Import `TokenUsage` from `holodeck.models.token_usage`. (plan.md:L193, contracts/execution-result.md:L17-72)

### Implementation for Phase 1

- [x] T007 [P] Create `src/holodeck/lib/backends/__init__.py` (empty package marker) (plan.md:L185)
- [x] T007b Create `tests/unit/lib/backends/__init__.py` (empty init required for pytest package discovery — all test files T006, T017–T022 live in this package and will not be discovered without it)
- [x] T008 Create `src/holodeck/lib/backends/base.py` with:
  - `ExecutionResult` dataclass (all fields from contract; `token_usage: TokenUsage = field(default_factory=TokenUsage.zero)` — import `TokenUsage` from `holodeck.models.token_usage`, NOT from `holodeck.models.test_result` which does not exist)
  - `AgentSession` Protocol with `send`, `send_streaming`, and `close` — **type `send_streaming` as `AsyncGenerator[str, None]` (async generator with `yield` inside), not `-> AsyncIterator[str]` (which implies returning an external iterator object; these are not the same in Python's type system)**
  - `AgentBackend` Protocol with `initialize`, `invoke_once`, `create_session`, `teardown`
  - `BackendError`, `BackendInitError`, `BackendSessionError`, `BackendTimeoutError` exception hierarchy
  (plan.md:L187-193, data-model.md:L201-338, contracts/execution-result.md:L21-146)

**Checkpoint**: `test_base.py` tests now pass. `ExecutionResult`, `AgentSession`, and `AgentBackend` are importable from `holodeck.lib.backends.base`.

---

## Phase 2: Model Extensions

**Purpose**: Extend HoloDeck's Pydantic models with Claude-native fields. Backward compatibility — existing YAML configurations must continue to parse without error.

**⚠️ CRITICAL**: These models are prerequisites for Phase 3 validators, Phase 4 SK backend refactor, and Phase 8 Claude backend. No backend work can begin until `claude_config.py`, `llm.py`, and `agent.py` are updated and tested.

### Tests for Phase 2 (write these first — they MUST fail before T014–T016)

- [x] T009 [P] Write unit tests for `AuthProvider` enum values (`api_key`, `oauth_token`, `bedrock`, `vertex`, `foundry`) and `PermissionMode` enum values (`manual`, `acceptEdits`, `acceptAll`) in `tests/unit/models/test_claude_config.py`. **Note**: T009 and T010 target the same file — `[P]` applies only when two developers own each task; a single developer should execute them sequentially. (plan.md:L219-223, data-model.md:L14-35)
- [x] T010 [P] Write unit tests for `ClaudeConfig` valid construction covering: minimal (all defaults), `ExtendedThinkingConfig` enabled, `BashConfig` with excluded commands, `FileSystemConfig` all flags, `SubagentConfig` with `max_parallel`, and `extra="forbid"` rejects unknown fields in `tests/unit/models/test_claude_config.py`. **Note**: Same file as T009 — parallelizable only with a second developer. (plan.md:L219-223, data-model.md:L74-121)
- [x] T011 [P] Write unit tests for `LLMProvider.auth_provider` field: valid values accepted, `None` default, and `logging.warning` emitted (verified via `caplog` fixture — not `pytest.warns`) when `auth_provider` is set for a non-Anthropic provider in `tests/unit/models/test_llm_extensions.py` (plan.md:L219-223, data-model.md:L125-142)
- [x] T012 [P] Write unit tests for `Agent.embedding_provider` (accepts `LLMProvider | None`) and `Agent.claude` (accepts `ClaudeConfig | None`) field acceptance in `tests/unit/models/test_agent_extensions.py` (plan.md:L219-223, data-model.md:L145-175)
- [x] T013 [P] Write backward compatibility tests: load each YAML file in `tests/fixtures/agents/` through `Agent(**config)` and assert no `ValidationError` is raised. For each fixture also assert `agent.claude is None` and `agent.embedding_provider is None`, confirming new optional fields default correctly when absent in `tests/unit/models/test_agent_extensions.py` (plan.md:L222-223, contracts/agent-yaml-schema.md:L200-205)

### Implementation for Phase 2

- [x] T014 Create `src/holodeck/models/claude_config.py` with `AuthProvider(str, Enum)`, `PermissionMode(str, Enum)`, `ExtendedThinkingConfig(BaseModel)`, `BashConfig(BaseModel)`, `FileSystemConfig(BaseModel)`, `SubagentConfig(BaseModel)`, and `ClaudeConfig(BaseModel)` — all with `extra="forbid"` and all capabilities defaulting to disabled (plan.md:L201-208, data-model.md:L14-121, contracts/agent-yaml-schema.md:L28-56)
- [x] T015 Update `src/holodeck/models/llm.py` to add `auth_provider: AuthProvider | None = Field(default=None, ...)` and a new `model_validator(mode="after")` method (give it a unique name distinct from the existing `check_endpoint_required` validator) that emits `logging.warning(...)` when `auth_provider` is set for a non-Anthropic provider (plan.md:L210-212, data-model.md:L125-142, contracts/agent-yaml-schema.md:L66-84)
- [x] T016 Update `src/holodeck/models/agent.py` to add `embedding_provider: LLMProvider | None = Field(default=None, ...)` and `claude: ClaudeConfig | None = Field(default=None, ...)`, preserving `model_config = ConfigDict(extra="forbid")` (plan.md:L214-217, data-model.md:L145-175, contracts/agent-yaml-schema.md:L10-25)

**Checkpoint**: All Phase 2 tests pass. `claude_config.py` importable. `Agent`, `LLMProvider` accept new fields. All `tests/fixtures/agents/` YAML files parse without error.

---

## Phase 3: Validation Layer

**Purpose**: Implement all startup validators that gate Claude-native agent execution. These validators are called by `ClaudeBackend.initialize()` (Phase 8) before any subprocess is spawned — early, clear errors instead of cryptic runtime failures.

**⚠️ CRITICAL**: `validators.py` is a direct dependency of `ClaudeBackend`. Phase 8 cannot begin until all validators are complete and tested.

> **`ConfigError` convention**: All validators raise `ConfigError(field, message)` with **two positional arguments**: `field` is the configuration field name (e.g. `"nodejs"`, `"ANTHROPIC_API_KEY"`, `"embedding_provider"`); `message` is the human-readable description. This matches the actual constructor signature at `lib/errors.py:L26`. Do **not** call `ConfigError` with a single string argument — that raises `TypeError`.

### Tests for Phase 3 (write these first — they MUST fail before T023)

> **NOTE**: Write ALL validator tests before implementing `validators.py`. Run `pytest tests/unit/lib/backends/test_validators.py -n auto` to confirm all fail with `ImportError` or `ModuleNotFoundError` before T023.

- [ ] T017 [P] Write unit tests for `validate_nodejs()`: passes when `shutil.which("node")` returns a path; raises `ConfigError("nodejs", "Node.js is required…")` (2-arg form) with install instructions when it returns `None` in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-233, research.md:L204-219)
- [ ] T018 [P] Write unit tests for `validate_credentials()` covering all 5 `auth_provider` values: `api_key` raises `ConfigError("ANTHROPIC_API_KEY", "…")` when absent; `oauth_token` raises `ConfigError("CLAUDE_CODE_OAUTH_TOKEN", "… run 'claude setup-token'")` when absent; `bedrock` returns env dict with `{"CLAUDE_CODE_USE_BEDROCK": "1"}`; `vertex` returns `{"CLAUDE_CODE_USE_VERTEX": "1"}`; `foundry` returns `{"CLAUDE_CODE_USE_FOUNDRY": "1"}` in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-234, research.md:L83-94, quickstart.md:L279-317)
- [ ] T019 [P] Write unit tests for `validate_embedding_provider()` covering 4 cases: (1) passes when no vectorstore tools present; (2) passes when vectorstore tool present and `embedding_provider` set to a valid provider (`openai`); (3) raises `ConfigError("embedding_provider", "…")` when `provider=anthropic` + vectorstore tool + no `embedding_provider` configured; (4) raises `ConfigError("embedding_provider", "… anthropic cannot generate embeddings")` when `embedding_provider.provider == anthropic` — only `openai` and `azure_openai` are valid embedding providers in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-235, research.md:L244-250, data-model.md:L170-175)
- [ ] T020 [P] Write unit tests for `validate_tool_filtering()`: emits `logging.warning(...)` (verified via `caplog` fixture) when `provider=anthropic` + `tool_filtering` is not `None` — **validator warns only and does NOT mutate `agent.tool_filtering`; `ClaudeBackend` ignores the field internally**; passes silently when `tool_filtering` is `None` in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-236, research.md:L279-284, data-model.md:L172-174)
- [ ] T021 [P] Write unit tests for `validate_working_directory()`: passes when `working_directory=None`; passes when path has no `CLAUDE.md`; emits `logging.warning(...)` (verified via `caplog` fixture — this is a warning, not a `ConfigError`) when `<working_directory>/CLAUDE.md` contains `# CLAUDE.md` header in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-237, research.md:L270-274, data-model.md:L175)
- [ ] T022 [P] Write unit tests for `validate_response_format()` covering 4 cases: (1) passes when `None`; (2) passes when a valid serialisable JSON Schema `dict`; (3) raises `ConfigError("response_format", "…")` when dict contains non-serialisable types; (4) when `response_format` is a `str` (file path), loads and parses the file as JSON Schema — raises `ConfigError("response_format", "…")` if the file is missing or its parsed content is non-serialisable, passes if the file contains a valid JSON Schema dict in `tests/unit/lib/backends/test_validators.py` (plan.md:L231-238, research.md:L226-239)

### Implementation for Phase 3

- [ ] T023 Create `src/holodeck/lib/backends/validators.py` implementing:
  - `validate_nodejs() -> None` — uses `shutil.which("node")`; raises `ConfigError("nodejs", "…")` if absent
  - `validate_credentials(model: LLMProvider) -> dict[str, str]` — maps each `auth_provider` value to its required env vars; raises `ConfigError(var_name, "…")` if a required var is missing; uses `holodeck.config.env_loader.get_env_var` for resolution
  - `validate_embedding_provider(agent: Agent) -> None` — raises `ConfigError("embedding_provider", "…")` when vectorstore/hierarchical-doc tools are present and `embedding_provider` is absent, or when `embedding_provider.provider == anthropic`
  - `validate_tool_filtering(agent: Agent) -> None` — emits `logging.warning(...)` when `tool_filtering` is set for Anthropic provider; **does not mutate `agent.tool_filtering`**
  - `validate_working_directory(path: str | None) -> None` — emits `logging.warning(...)` on CLAUDE.md collision; does not raise
  - `validate_response_format(response_format: dict | str | None) -> None` — if `str`, load file and parse as JSON Schema dict; validate JSON serializability for dict cases; raises `ConfigError("response_format", "…")` on failure
  - All `ConfigError` calls use the **2-arg form** `ConfigError(field_name, message)` from `holodeck.lib.errors`
  (plan.md:L231-253, research.md:L204-219, quickstart.md:L279-317)
- [ ] T023b Remove the local `class AgentFactoryError(Exception)` redefinition from `src/holodeck/lib/test_runner/agent_factory.py` (approximately line 100) and replace all references within that file with `from holodeck.lib.errors import AgentFactoryError`. Run `make test-unit` to confirm zero regressions. This resolves the duplicate-class conflict before Phase 4 begins its SK backend refactor. (lib/errors.py:L153)

**Checkpoint**: All Phase 3 tests pass. `validators.py` importable from `holodeck.lib.backends.validators`. Gate confirmed: startup errors surface before any subprocess spawns. `AgentFactoryError` consolidated to `lib/errors.py` only — Phase 4 can now safely `from holodeck.lib.errors import AgentFactoryError` without ambiguity.

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 0 (SDK install + smoke test — mandatory gate)
    │
    ▼
Phase 1 (backends/ package + core interfaces)
    │
    ▼
Phase 2 (Model Extensions)
    │
    ▼
Phase 3 (Validation Layer)
    │
    ▼ (unblocks Phase 4–12 in main tasks.md)
```

### Within Each Phase: TDD Order

```
1. Write test tasks (marked with "MUST fail before")
2. Verify tests fail: pytest <test_file> -n auto
3. Implement: make test tasks pass
4. Run: make format && make lint-fix && make type-check
5. Checkpoint: confirm tests pass, proceed to next phase
```

### Parallel Opportunities Per Phase

**Phase 0**: Sequential — T001 must complete before T002; T002 before T003; T004/T005 can overlap.

**Phase 1**:
- T006 (tests), T007, and T007b (init files) can all run in parallel
- T008 (implementation) requires T006 + T007 + T007b complete

**Phase 2**:
- T009–T013 (all five test tasks) can run in parallel — different test functions (note: T009+T010 share a file; parallel only with two developers)
- T014 (claude_config.py) requires T009, T010
- T015 (llm.py) requires T011, T014 (needs `AuthProvider` from T014)
- T016 (agent.py) requires T012, T013, T014 (needs `ClaudeConfig` from T014)

**Phase 3**:
- T017–T022 (all six test tasks) can run in parallel — different test functions in same file
- T023 (validators.py) requires T017–T022 all written and confirmed failing
- T023b (AgentFactoryError consolidation) can run in parallel with T023 — different file, no shared dependency

### Parallel Example: Phase 2

```bash
# All Phase 2 test tasks can start together (different files):
# Developer A: tests/unit/models/test_claude_config.py (T009, T010 — same file, do sequentially)
# Developer B: tests/unit/models/test_llm_extensions.py (T011)
# Developer C: tests/unit/models/test_agent_extensions.py (T012, T013)

# After all test tasks confirmed failing:
# T014 → T015 → T016 (sequential; each depends on the prior model)
```

### Parallel Example: Phase 3

```bash
# All six validator tests can be written simultaneously (same file, different functions):
# T017: test_validate_nodejs
# T018: test_validate_credentials (5 parametrized cases)
# T019: test_validate_embedding_provider (4 cases including anthropic-as-embedding rejection)
# T020: test_validate_tool_filtering (caplog-based, warn-only)
# T021: test_validate_working_directory (caplog-based)
# T022: test_validate_response_format (4 cases including str file-path resolution)

# After all confirmed failing → T023 and T023b can run in parallel (different files)
```

---

## Task Summary

| Phase | Tasks | Tests | Implementations | Parallel |
|-------|-------|-------|-----------------|---------|
| Phase 0 (SDK install + Smoke Test) | T001–T005 | 0 | 0 | T004∥T005 |
| Phase 1 (Backends Package) | T006–T008 + T007b | 1 | 3 | T006, T007, T007b |
| Phase 2 (Models) | T009–T016 | 5 | 3 | T009–T013 |
| Phase 3 (Validators) | T017–T023b | 6 | 2 | T017–T022, T023b∥T023 |
| **Total** | **25 tasks** | **12** | **8** | |

---

## Code Quality Gates

Run after each phase:

```bash
make format         # Black + Ruff formatting
make lint-fix       # Auto-fix linting issues
make type-check     # MyPy strict mode
make test-unit      # pytest tests/unit/ -n auto
make security       # Bandit + Safety + detect-secrets
```

---

## Notes

- Phase 0 is a **hard gate** — do not modify any production source file until T004 is complete
- T001 (SDK install) is the first task in Phase 0; the smoke test (T002) cannot run without it
- All `[ASSUMED]` names in `research.md §2` carry risk; Phase 0 eliminates that risk before code is written
- `ExecutionResult` (T008) is the most critical deliverable in Phases 0–3; every downstream phase depends on it
- `TokenUsage` default: use `TokenUsage.zero()` — `TokenUsage()` with no args raises `ValidationError` (all three fields are required with sum constraint enforcement)
- `TokenUsage` import path: `from holodeck.models.token_usage import TokenUsage` — there is no `holodeck.models.test_result` module
- `ConfigError` signature: always 2-arg `ConfigError(field, message)` — single-arg form raises `TypeError`
- `send_streaming` typing: use `AsyncGenerator[str, None]` (async generator with `yield`) — not `-> AsyncIterator[str]`
- `validate_tool_filtering` warns only — it does **not** mutate `Agent.tool_filtering`; `ClaudeBackend` ignores the field internally
- Backward compatibility (T013) scope: `tests/fixtures/agents/*.yaml` only — not deploy, ollama, or wizard fixtures
- T023b resolves the `AgentFactoryError` duplication before Phase 4's SK refactor needs a clean import
