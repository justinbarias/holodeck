# Tasks: Phase 8 — Claude Backend (Core) — Claude Agent SDK

**Input**: Design documents from `/specs/021-claude-agent-sdk/`
**Prerequisites**: Phases 1–7 complete (base.py, validators.py, sk_backend.py, selector.py, tool_adapters.py, mcp_bridge.py, otel_bridge.py)

**Tests**: TDD approach — write failing tests first, then implement.

**Organization**: Phase 8 is the core Claude backend implementation. Tasks are organized into two workstreams: (A) `ClaudeBackend` + `ClaudeSession` core implementation, and (B) `BackendSelector` update. Workstream B depends on A.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 8A: ClaudeSession & ClaudeBackend — Core Implementation

**Goal**: Create `claude_backend.py` implementing `ClaudeSession(AgentSession)` and `ClaudeBackend(AgentBackend)`. This is the core execution path for `provider: anthropic` agents — single-turn invocations via `query()`, stateful multi-turn sessions via `ClaudeSDKClient`, streaming support, subprocess lifecycle management, permission mode mapping, and structured output handling.

**Plan Reference**: `plan.md` lines 410–474 (Phase 8: Claude Backend Core)
**Spec Reference**: `spec.md` lines 27–41 (US1 — Run a Claude-Native Agent), lines 93–108 (US5 — Streaming Chat), lines 179–193 (US10 — Structured Output), lines 111–123 (US6 — Parallel Subagents), lines 125–141 (US7 — File System Access), lines 143–157 (US8 — Permission Governance), lines 159–177 (US9 — Flexible Auth)
**Research Reference**: `research.md` lines 30–101 (§2 — API Entry Points, §2a — ClaudeAgentOptions), lines 106–118 (§3 — Permission Mode Mapping), lines 122–137 (§4 — Authentication), lines 199–230 (§7 — Streaming Message Types), lines 234–243 (§8 — Retry Behaviour), lines 266–281 (§10 — Structured Output)
**Data Model Reference**: `data-model.md` lines 202–257 (§4 — ExecutionResult), lines 260–295 (§5 — AgentSession), lines 298–338 (§6 — AgentBackend), lines 342–402 (§7–8 — Entity Relationships, Lifecycle)
**Quickstart Reference**: `quickstart.md` lines 139–188 (§2 — Building ClaudeAgentOptions), lines 190–248 (§3 — Stateless Test Invocation), lines 250–278 (§4 — Streaming Chat Session), lines 280–324 (§5 — Authentication Env Vars)
**Contract Reference**: `contracts/execution-result.md` lines 17–72 (ExecutionResult, Error Conditions), lines 74–97 (AgentSession), lines 100–128 (AgentBackend Lifecycle)

**Independent Test**: Mock `claude_agent_sdk.query()` to return a canned `AssistantMessage` + `ResultMessage` stream. Call `ClaudeBackend.invoke_once("Hello")`. Assert `ExecutionResult.response` contains the expected text, `is_error is False`, and `token_usage` is populated.

### Tests for Phase 8A (TDD — write first, verify they FAIL)

- [ ] T001 [P] [US1] Write unit test for `build_options()` in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:433–434` ("Build `ClaudeAgentOptions` via `build_options()`"), `quickstart.md:139–188`
  - Create a minimal `Agent` fixture with `provider: anthropic`, `model.name: claude-sonnet-4-6`, and optional `claude` block.
  - Assert returned `ClaudeAgentOptions` has correct `model`, `system_prompt`, `permission_mode`, `max_turns`, `env`, `mcp_servers`, `allowed_tools`.
  - Test with no `claude` block → defaults: `permission_mode=None`, `max_turns=None`, no extra MCP servers.
  - Test with full `claude` block → all fields mapped: working dir → `cwd`, `extended_thinking.budget_tokens` → `max_thinking_tokens`, `web_search`, `bash`, `file_system`, `subagents`, `allowed_tools`.
  - Test `response_format` dict → `output_format` translated correctly (NOT using SK's `_wrap_response_format`).
  - **Ref**: `research.md:266–281` (§10 — Structured Output)

- [ ] T002 [P] [US1] Write unit test for permission mode mapping in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `research.md:106–118` (§3 — Permission Mode Mapping), `plan.md:446`
  - Assert: `manual` → `"default"`, `acceptEdits` → `"acceptEdits"`, `acceptAll` → `"bypassPermissions"`.
  - Test override for test mode: when `mode="test"` and `permission_mode` is not `manual`, override to `"bypassPermissions"`.
  - Test `mode="test"` with `permission_mode: manual` → keeps `"default"` (no override).
  - Test `mode="test"` with `bash.enabled: true` and `allow_side_effects=False` → bash force-disabled.
  - Test `mode="test"` with `file_system.write: true` and `allow_side_effects=False` → file_system.write force-disabled.
  - Test `mode="test"` with `bash.enabled: true` and `allow_side_effects=True` → bash NOT force-disabled (preserved).

- [ ] T003 [P] [US1] Write unit test for `ClaudeBackend.__init__()` stores config only (no I/O) in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:424` ("stores config only — no I/O, no subprocess"), `plan.md:295–298` (lazy-init pattern)
  - Construct `ClaudeBackend(agent, tool_instances, mode="test")` — assert no SDK imports triggered, no subprocess spawned.
  - Assert `._initialized is False` after construction.
  - Assert `._options is None` after construction (options built lazily in `initialize()`).

- [ ] T004 [P] [US1] Write unit test for `ClaudeBackend.initialize()` calls validators in correct order in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:425–434` (initialize() steps 1–9), FR-029 (CLAUDE.md collision warning)
  - Mock all validators (`validate_nodejs`, `validate_credentials`, `validate_embedding_provider`, `validate_tool_filtering`, `validate_working_directory`, `validate_response_format`).
  - Assert they are called in the documented order.
  - Assert `validate_working_directory()` is called with `agent.claude.working_directory` (covers FR-029 CLAUDE.md collision detection).
  - Assert `._initialized is True` after `initialize()`.
  - Assert `._options` is a `ClaudeAgentOptions` instance after `initialize()`.

- [ ] T005 [P] [US1] Write unit test for `ClaudeBackend.invoke_once()` happy path in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:435–441` (invoke_once steps), `quickstart.md:190–248`, `contracts/execution-result.md:17–32`
  - Mock `claude_agent_sdk.query()` to yield: `AssistantMessage(content=[TextBlock(text="Hello world")])`, then `ResultMessage(is_error=False, num_turns=1, usage={"input_tokens": 10, "output_tokens": 5}, structured_output=None, session_id="abc")`.
  - Assert `ExecutionResult.response == "Hello world"`.
  - Assert `ExecutionResult.is_error is False`.
  - Assert `ExecutionResult.token_usage.prompt_tokens == 10`.
  - Assert `ExecutionResult.token_usage.completion_tokens == 5`.
  - Assert `ExecutionResult.num_turns == 1`.

- [ ] T006 [P] [US2] Write unit test for `invoke_once()` tool call extraction in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:437` ("Collects AssistantMessage blocks — text, tool_use, tool_result"), `research.md:229`, `contracts/execution-result.md:36–52`
  - Mock stream with `ToolUseBlock(id="toolu_01", name="kb_search", input={"query": "refund"})` and `ToolResultBlock(tool_use_id="toolu_01", content=[TextBlock(text="30-day guarantee")], is_error=False)`.
  - Assert `ExecutionResult.tool_calls[0] == {"name": "kb_search", "arguments": {"query": "refund"}, "call_id": "toolu_01"}`.
  - Assert `ExecutionResult.tool_results[0] == {"call_id": "toolu_01", "result": "30-day guarantee", "is_error": False}`.

- [ ] T007 [P] [US10] Write unit test for `invoke_once()` structured output handling in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:439` ("Validates structured output against schema"), `research.md:266–281` (§10), `spec.md:179–193` (US10)
  - Mock `ResultMessage.structured_output = {"name": "Widget", "price": 9.99}`.
  - Assert `ExecutionResult.structured_output == {"name": "Widget", "price": 9.99}`.
  - Assert `ExecutionResult.response` contains the JSON-serialized string for NLP/G-Eval consumption.
  - Test schema validation failure: structured output does not match configured schema → `ExecutionResult.is_error is True`, `error_reason` mentions "schema validation".

- [ ] T008 [P] [US1] Write unit test for `invoke_once()` max_turns exceeded detection in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:441` ("Detects max_turns exceeded via ResultMessage.num_turns >= max_turns"), `contracts/execution-result.md:68` (error condition)
  - Mock `ResultMessage.num_turns = 10` with `max_turns = 10` in agent config.
  - Assert `ExecutionResult.is_error is True`.
  - Assert `ExecutionResult.error_reason == "max_turns limit reached"`.
  - Assert partial response text is preserved.

- [ ] T009 [P] [US1] Write unit test for `invoke_once()` subprocess crash handling in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:442` ("Session-level retry on ProcessError"), `research.md:234–243` (§8), `contracts/execution-result.md:69`, `spec.md:210` (edge case — subprocess terminates)
  - Mock `claude_agent_sdk.query()` to raise `ProcessError` on first call, succeed on second.
  - Assert retry happens (up to 3 attempts, exponential backoff).
  - Mock all 3 retries failing → assert `BackendSessionError` is raised with "subprocess terminated unexpectedly".

- [ ] T010 [P] [US9] Write unit test for auth env var handling in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:434` ("validate_credentials → stores env dict"), `research.md:122–137` (§4), `quickstart.md:280–324`, `contracts/agent-yaml-schema.md:67–84`
  - **Design note**: The SDK merges `os.environ` with `ClaudeAgentOptions.env` automatically. `validate_credentials()` returns only *extra* env vars needed (e.g., `CLAUDE_CODE_USE_BEDROCK=1`). For `api_key`/`oauth_token`, it validates presence and returns `{}` — the keys are inherited via `os.environ`.
  - Test `auth_provider: api_key` → `validate_credentials()` returns `{}` (validates ANTHROPIC_API_KEY exists, inherited by subprocess automatically).
  - Test `auth_provider: oauth_token` → `validate_credentials()` returns `{}` (validates CLAUDE_CODE_OAUTH_TOKEN exists, inherited automatically).
  - Test `auth_provider: bedrock` → `validate_credentials()` returns `{"CLAUDE_CODE_USE_BEDROCK": "1"}` in `ClaudeAgentOptions.env`.
  - Test `auth_provider: vertex` → `validate_credentials()` returns `{"CLAUDE_CODE_USE_VERTEX": "1"}` in `ClaudeAgentOptions.env`.
  - Test `auth_provider: foundry` → `validate_credentials()` returns `{"CLAUDE_CODE_USE_FOUNDRY": "1"}` in `ClaudeAgentOptions.env`.
  - Assert extra env vars are merged into `ClaudeAgentOptions.env`.

- [ ] T011 [P] [US5] Write unit test for `ClaudeSession.send_streaming()` in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:419` ("yields text chunks from StreamEvent / AssistantMessage blocks"), `quickstart.md:250–278`, `research.md:199–230` (§7)
  - Mock `ClaudeSDKClient.query()` and `receive_response()` to yield multiple `AssistantMessage` with `TextBlock` chunks.
  - Assert chunks arrive progressively (not all at once).
  - Assert final `ResultMessage` terminates the stream.

- [ ] T012 [P] [US5] Write unit test for `ClaudeSession.send()` (non-streaming) in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:418` ("collects full response from receive_response() iterator"), `data-model.md:275–280`
  - Mock `ClaudeSDKClient.query()` + `receive_response()` yielding `AssistantMessage` + `ResultMessage`.
  - Assert `ExecutionResult.response` is the concatenated text from all `TextBlock`s.
  - Assert `token_usage` is extracted from `ResultMessage.usage`.

- [ ] T013 [P] [US1] Write unit test for `ClaudeSession.close()` calls `client.disconnect()` in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:421` ("close() → client.disconnect()"), `data-model.md:392–396`
  - Mock `ClaudeSDKClient.disconnect()`.
  - Assert `close()` calls `disconnect()` exactly once.

- [ ] T014 [P] [US5] Write unit test for `ClaudeSession` multi-turn state tracking in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `research.md:59–80` (multi-turn state is OPT-IN), `research.md:81` ("session wrapper must pass continue_conversation=True and track session_id")
  - Mock two successive `send()` calls.
  - Assert first call uses the base options (no `continue_conversation`, no `resume`).
  - Assert second call creates a NEW `ClaudeAgentOptions` instance (not mutating the original) with `continue_conversation=True` and `resume=<session_id from first ResultMessage>`.
  - Assert the original base options object is NOT mutated after two turns.

- [ ] T015 [P] [US1] Write unit test for `ClaudeBackend.initialize()` lazy-init guard in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:295–298` ("initialize() is called lazily on the first invoke_once() or create_session()")
  - Call `invoke_once()` without calling `initialize()` first → assert `initialize()` is called automatically.
  - Call `create_session()` without calling `initialize()` first → assert `initialize()` is called automatically.
  - Call `initialize()` explicitly, then `invoke_once()` → assert `initialize()` is NOT called a second time.

- [ ] T016 [P] [US1] Write unit test for `ClaudeBackend.teardown()` in `tests/unit/lib/backends/test_claude_backend.py`
  - **Ref**: `plan.md:444` ("teardown() → cleanup"), `data-model.md:336–338`
  - Assert `teardown()` resets `_initialized = False` and `_options = None`.
  - Assert calling `invoke_once()` after `teardown()` triggers re-initialization.

### Implementation for Phase 8A

- [ ] T017 [US1] Create `src/holodeck/lib/backends/claude_backend.py` with module docstring, imports, and helper functions
  - **Ref**: `plan.md:414`, `quickstart.md:139–188`
  - Import from `claude_agent_sdk`: `ClaudeAgentOptions`, `ClaudeSDKClient`, `query`, `AssistantMessage`, `ResultMessage`, `StreamEvent`, `ProcessError`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`.
  - Import from `holodeck`: `ExecutionResult`, `AgentSession`, `BackendSessionError`, `BackendInitError`, `TokenUsage`, `Agent`.
  - Implement helper `_extract_result_text(content: list) -> str` — extracts text from content blocks.
  - Implement helper `_build_permission_mode(claude_config, mode) -> str | None` — maps HoloDeck permission modes to SDK literals per `research.md:106–118`.
  - Implement helper `_build_output_format(response_format) -> dict | None` — translates to plain `dict[str, Any]` for `ClaudeAgentOptions.output_format`, NOT using SK's `_wrap_response_format`. Note: SDK has no `OutputFormat` class — `output_format` is `dict[str, Any] | None`.
  - **Extract shared instruction resolver**: Move `AgentFactory._load_instructions()` logic from `src/holodeck/lib/test_runner/agent_factory.py` into a shared utility (e.g., `src/holodeck/lib/instruction_resolver.py` or `src/holodeck/config/loader.py`). Implement `resolve_instructions(instructions: Instructions, base_dir: Path | None = None) -> str`. Both SK backend and Claude backend import from this shared location. Update `AgentFactory` to use the shared function.

- [ ] T018 [US1] Implement `build_options()` function in `src/holodeck/lib/backends/claude_backend.py`
  - **Ref**: `plan.md:433–434`, `quickstart.md:139–188`, `research.md:83–101` (§2a — ClaudeAgentOptions fields)
  - Accept `agent: Agent`, `tool_server: McpSdkServerConfig | None`, `mcp_configs: dict`, `auth_env: dict`, `otel_env: dict`, `mode: str`, `allow_side_effects: bool`.
  - Build `system_prompt` from `agent.instructions` (resolve inline or file).
  - Build `mcp_servers` dict: merge `holodeck_tools` server (if tool_server) + external MCP configs.
  - Build `env` dict: merge `auth_env` + `otel_env`.
  - Map `extended_thinking.budget_tokens` → `max_thinking_tokens` (only if `extended_thinking.enabled`).
  - Map `allowed_tools` from `agent.claude.allowed_tools` merged with SDK tool names.
  - Handle test mode safety: when `mode="test"`, force `bash.enabled=False` and `file_system.write=False` unless `allow_side_effects=True`.
  - Return `ClaudeAgentOptions(...)` with all fields.

- [ ] T019 [US1] Implement `ClaudeSession` class in `src/holodeck/lib/backends/claude_backend.py`
  - **Ref**: `plan.md:416–421`, `quickstart.md:250–278`, `research.md:59–80` (multi-turn OPT-IN)
  - `__init__(self, options: ClaudeAgentOptions)` — store options as `_base_options` (immutable reference), set `_session_id = None`, `_client = None`.
  - `async _ensure_client(self)` — create `ClaudeSDKClient` and enter async context if not already.
  - `_build_turn_options(self) -> ClaudeAgentOptions` — for turn 1, return a copy of `_base_options`. For turn 2+, create a NEW `ClaudeAgentOptions` with `continue_conversation=True` and `resume=_session_id`. NEVER mutate `_base_options`.
  - `async send(self, message: str) -> ExecutionResult` — call `_client.query(message)`, collect from `receive_response()`, extract text/tool_calls/tool_results/token_usage, update `_session_id` from `ResultMessage`, build `ExecutionResult`.
  - `async send_streaming(self, message: str) -> AsyncGenerator[str, None]` — yield `TextBlock.text` chunks progressively from `receive_response()`. Return type matches `AgentSession` Protocol (`AsyncGenerator[str, None]`).
  - `async close(self)` — call `_client.disconnect()` if client exists, set `_client = None`.
  - Handle `ProcessError` → raise `BackendSessionError("subprocess terminated unexpectedly: {detail}")`.

- [ ] T020 [US1] Implement `ClaudeBackend` class in `src/holodeck/lib/backends/claude_backend.py`
  - **Ref**: `plan.md:423–444`, `data-model.md:298–338` (§6 — AgentBackend), `quickstart.md:190–248`
  - `__init__(self, agent: Agent, tool_instances: dict[str, VectorStoreTool | HierarchicalDocumentTool] | None = None, mode: str = "test", allow_side_effects: bool = False)` — store config only, no I/O. Set `_initialized = False`, `_options = None`.
  - `async initialize(self)` — idempotent (check `_initialized`). Steps:
    1. `validate_nodejs()`
    2. `auth_env = validate_credentials(agent.model)`
    3. `validate_embedding_provider(agent)` (if vectorstore tools)
    4. `validate_tool_filtering(agent)` (warn + skip)
    5. Build tool adapters from `tool_instances` → `create_tool_adapters()` + `build_holodeck_sdk_server()`
    6. Build MCP bridge configs → `build_claude_mcp_configs()`
    7. Build OTel env vars → `translate_observability()` (if observability configured)
    8. Build `ClaudeAgentOptions` via `build_options()`
    9. `validate_working_directory(claude.working_directory)`
    10. `validate_response_format(agent.response_format)`
    11. Set `_initialized = True`
  - `async invoke_once(self, message: str, context=None) -> ExecutionResult` — lazy-init guard → call `initialize()` if not yet. Use `query(prompt=message, options=self._options)`. Collect `AssistantMessage` blocks (text, tool_use, tool_result) + `ResultMessage` for token usage/structured output. Detect `max_turns` exceeded. Retry on `ProcessError` (3 attempts, exponential backoff). Return `ExecutionResult`.
  - `async create_session(self) -> ClaudeSession` — lazy-init guard → return `ClaudeSession(self._options)`.
  - `async teardown(self)` — reset `_initialized = False`, `_options = None`.

- [ ] T021 [US1] Implement subprocess retry logic with exponential backoff in `src/holodeck/lib/backends/claude_backend.py`
  - **Ref**: `plan.md:442` ("3 attempts, exponential backoff"), `research.md:234–243` (§8 — Retry Behaviour)
  - Retry only on `ProcessError` (subprocess spawn/crash). NOT on API-level errors (SDK handles those internally per FR-013).
  - Backoff: 1s → 2s → 4s (exponential).
  - After 3 failures: raise `BackendSessionError("Claude subprocess failed after 3 retries: {last_error}")`.
  - Log each retry attempt at WARNING level.

- [ ] T022 [US10] Implement structured output schema validation in `src/holodeck/lib/backends/claude_backend.py`
  - **Ref**: `plan.md:439` ("Validates structured output against schema if response_format configured"), `spec.md:189–193` (US10 acceptance), `research.md:266–281` (§10)
  - When `response_format` is configured and `ResultMessage.structured_output` is returned:
    1. Validate against configured JSON Schema using `jsonschema.validate()`.
    2. Set `ExecutionResult.structured_output` to the validated object.
    3. Set `ExecutionResult.response` to `json.dumps(structured_output)` for NLP/G-Eval metrics.
  - On validation failure: set `ExecutionResult.is_error = True`, `error_reason = "Structured output does not match configured schema: {detail}"`.

**Checkpoint**: At this point, `claude_backend.py` should contain a complete `ClaudeSession` and `ClaudeBackend` with all unit tests passing against mocked SDK. Run: `pytest tests/unit/lib/backends/test_claude_backend.py -n auto -v`

---

## Phase 8B: BackendSelector Update

**Goal**: Update `selector.py` to route `provider: anthropic` agents to `ClaudeBackend` instead of raising `BackendInitError`. This completes the routing so that `BackendSelector.select()` can produce both SK and Claude backends.

**Plan Reference**: `plan.md` lines 448–464 (Phase 8.2 — selector.py)
**Spec Reference**: `spec.md` lines 218–233 (FR-001, FR-010, FR-011)
**Data Model Reference**: `data-model.md` lines 362–373 (BackendSelector routing)

**Independent Test**: Call `BackendSelector.select(agent)` with a `provider: anthropic` agent. Assert a `ClaudeBackend` instance is returned (not `BackendInitError`). Call with a `provider: openai` agent → assert `SKBackend` is returned (unchanged).

### Tests for Phase 8B (TDD — write first, verify they FAIL)

- [ ] T023 [P] [US1] Write unit test for `BackendSelector` routing `anthropic` to `ClaudeBackend` in `tests/unit/lib/backends/test_selector.py`
  - **Ref**: `plan.md:448–464`, `data-model.md:362–373`
  - **Replaces existing `test_anthropic_raises_backend_init_error()`** — after Phase 8B, Anthropic no longer raises an error.
  - Mock `ClaudeBackend.__init__` and `initialize()`.
  - Create agent with `provider: anthropic` → assert `BackendSelector.select()` returns `ClaudeBackend` instance.
  - Assert `initialize()` was called on the returned backend.

- [ ] T024 [P] [US1] Write unit test for `BackendSelector` passing `tool_instances` and `mode` to `ClaudeBackend` in `tests/unit/lib/backends/test_selector.py`
  - **Ref**: `plan.md:453–458` (selector signature with tool_instances, mode)
  - Mock `ClaudeBackend.__init__`.
  - Call `BackendSelector.select(agent, tool_instances={"kb": mock_tool}, mode="chat")`.
  - Assert `ClaudeBackend` was constructed with `tool_instances={"kb": mock_tool}` and `mode="chat"`.

- [ ] T025 [P] [US1] Write unit test for unsupported provider raises `BackendInitError` in `tests/unit/lib/backends/test_selector.py`
  - **Ref**: `selector.py:46` (existing behaviour)
  - Create agent with a hypothetical unsupported provider → assert `BackendInitError` raised.
  - **Note**: Existing SK-path tests (`test_openai_returns_sk_backend`, `test_azure_openai_returns_sk_backend`, `test_ollama_returns_sk_backend`) already cover FR-010 and must continue to pass after the `select()` signature change. No new SK-path tests needed.

### Implementation for Phase 8B

- [ ] T026 [US1] Update `src/holodeck/lib/backends/selector.py` to route `ProviderEnum.ANTHROPIC` to `ClaudeBackend`
  - **Ref**: `plan.md:448–464`
  - Import `ClaudeBackend` from `claude_backend`.
  - Update `select()` signature to accept optional `tool_instances: dict[str, VectorStoreTool | HierarchicalDocumentTool] | None = None`, `mode: str = "test"`, `allow_side_effects: bool = False`. New params have defaults so existing callers are unaffected.
  - For `ProviderEnum.ANTHROPIC`: instantiate `ClaudeBackend(agent, tool_instances, mode, allow_side_effects)`, call `await backend.initialize()`, return.
  - Remove the current `BackendInitError("Anthropic provider is not yet supported...")` block.
  - Preserve `SKBackend` path for `OPENAI`, `AZURE_OPENAI`, `OLLAMA` — no changes.
  - **Existing tests must still pass**: `test_openai_returns_sk_backend`, `test_azure_openai_returns_sk_backend`, `test_ollama_returns_sk_backend`, `test_initialize_awaited_on_returned_backend`.

- [ ] T027 [US1] Update `src/holodeck/lib/backends/__init__.py` to export `ClaudeBackend` and `ClaudeSession`
  - **Ref**: `plan.md:95` (claude_backend.py in project structure)
  - Add imports: `from holodeck.lib.backends.claude_backend import ClaudeBackend, ClaudeSession`.
  - Add to `__all__`: `"ClaudeBackend"`, `"ClaudeSession"`.

**Checkpoint**: At this point, `BackendSelector.select()` should route Anthropic agents to `ClaudeBackend`. Existing SK-path tests must still pass. Run: `pytest tests/unit/lib/backends/test_selector.py tests/unit/lib/backends/test_claude_backend.py -n auto -v`

---

## Phase 8C: Code Quality & Verification

**Goal**: Run all code quality checks and verify zero regressions across the full test suite.

**Plan Reference**: `plan.md` lines 551–561 (Phase 12: Code Quality — run after each phase)

- [ ] T028 Run `make format` to apply Black + Ruff formatting to all new/modified files
  - **Ref**: `plan.md:556`
  - Files: `src/holodeck/lib/backends/claude_backend.py`, `src/holodeck/lib/backends/selector.py`, `src/holodeck/lib/backends/__init__.py`, `tests/unit/lib/backends/test_claude_backend.py`, `tests/unit/lib/backends/test_selector.py`.

- [ ] T029 Run `make lint-fix` to auto-fix linting issues in new/modified files
  - **Ref**: `plan.md:557`

- [ ] T030 Run `make type-check` to verify MyPy strict mode passes for all new code
  - **Ref**: `plan.md:558`
  - All new functions must have type hints on parameters and return values.
  - Fix any MyPy errors before proceeding.

- [ ] T031 Run `make test-unit` to verify zero regressions across the full unit test suite
  - **Ref**: `plan.md:559`, `plan.md:299` ("existing unit tests MUST pass without modification")
  - Run: `pytest tests/unit/ -n auto -v`
  - All existing tests from Phases 1–7 must still pass.
  - All new Phase 8 tests must pass.

- [ ] T032 Run `make security` to verify Bandit + Safety + detect-secrets pass
  - **Ref**: `plan.md:560`
  - No new security warnings allowed in `claude_backend.py` or `selector.py`.

**Checkpoint**: Full CI pipeline green. Ready for Phase 9 (Test Runner Updates).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 8A Tests (T001–T016)**: No dependencies on each other — all [P] parallelizable. All depend on Phases 1–7 being complete.
- **Phase 8A Implementation (T017–T022)**: T017 must come first (module skeleton + shared instruction resolver). T018–T022 depend on T017. T019 and T020 can be partially parallelized but T020 (`ClaudeBackend`) references T019 (`ClaudeSession`).
- **Phase 8B Tests (T023–T025)**: Depend on Phase 8A implementation being complete. All [P] parallelizable among themselves.
- **Phase 8B Implementation (T026–T027)**: Depend on Phase 8A implementation + Phase 8B tests. T027 depends on T026.
- **Phase 8C Quality (T028–T032)**: Depends on all Phase 8A + 8B being complete. T028–T030 are parallelizable. T031 depends on T028–T030. T032 depends on T031.

### Execution Order (Critical Path)

```
T001–T016 (parallel)  ─── all Phase 8A tests
      │
T017                  ─── module skeleton + instruction resolver extraction
      │
T018, T019, T020, T021, T022  ─── Phase 8A implementation (mostly sequential)
      │
T023–T025 (parallel)  ─── Phase 8B tests
      │
T026, T027            ─── Phase 8B implementation (sequential)
      │
T028–T030 (parallel)  ─── formatting, linting, type-check
      │
T031                  ─── full test suite
      │
T032                  ─── security scan
```

### Within Phase 8A Implementation

- T017 (skeleton + instruction resolver) → T018 (build_options) → T019 (ClaudeSession) → T020 (ClaudeBackend) → T021 (retry logic) → T022 (structured output validation)
- T021 and T022 can be done in either order after T020.

### Parallel Opportunities

```bash
# Launch all Phase 8A tests in parallel:
Task: T001 — build_options unit test
Task: T002 — permission mode mapping test
Task: T003 — __init__ no-I/O test
Task: T004 — initialize() validator order test
Task: T005 — invoke_once() happy path test
Task: T006 — tool call extraction test
Task: T007 — structured output test
Task: T008 — max_turns exceeded test
Task: T009 — subprocess crash retry test
Task: T010 — auth env var handling test
Task: T011 — send_streaming() test
Task: T012 — send() non-streaming test
Task: T013 — close() test
Task: T014 — multi-turn state tracking test
Task: T015 — lazy-init guard test
Task: T016 — teardown() test

# Launch all Phase 8B tests in parallel:
Task: T023 — selector routes anthropic to ClaudeBackend (replaces existing error test)
Task: T024 — selector passes tool_instances/mode
Task: T025 — unsupported provider error

# Launch quality checks in parallel:
Task: T028 — format
Task: T029 — lint-fix
Task: T030 — type-check
```

---

## Implementation Strategy

### MVP First (Phase 8A Only)

1. Write all T001–T016 tests (TDD — verify they FAIL)
2. Implement T017–T022 (claude_backend.py)
3. Verify all Phase 8A tests pass
4. **STOP and VALIDATE**: `ClaudeBackend.invoke_once()` works with mocked SDK
5. This alone is a demonstrable milestone — the core execution engine exists

### Full Phase 8

1. Complete Phase 8A (MVP above)
2. Write Phase 8B tests (T023–T025)
3. Implement Phase 8B (T026–T027 — selector update + exports)
4. Run quality checks (T028–T032)
5. **STOP and VALIDATE**: `BackendSelector.select()` routes to both backends

### User Story Coverage

| User Story | Tasks | Key Deliverable |
|---|---|---|
| US1 (Run Claude-Native Agent) | T001–T005, T008–T009, T013, T015–T020, T023–T027 | Core execution path |
| US2 (Vectorstore/Doc Tools) | T006 (tool extraction) | Tool calls via adapters (Phase 5) |
| US3 (MCP Tools) | — | MCP bridge (Phase 6, complete) |
| US5 (Streaming Chat) | T011–T012, T014 | `ClaudeSession.send_streaming()` |
| US6 (Parallel Subagents) | — | Subagent config passed via `build_options()` |
| US7 (File System Access) | — | File system config passed via `build_options()` |
| US8 (Permission Governance) | T002 | Permission mode mapping |
| US9 (Flexible Auth) | T010 | Auth env var handling |
| US10 (Structured Output) | T007, T022 | Schema validation + output handling |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- TDD: write tests first, verify they fail, then implement
- All `claude_agent_sdk` calls are mocked in unit tests — no API key needed
- Integration tests (requiring real API keys) are deferred to Phase 11 per plan.md:541–548
- Commit after each logical group (Phase 8A tests → Phase 8A impl → Phase 8B tests → Phase 8B impl → quality)
- Stop at any checkpoint to validate independently
