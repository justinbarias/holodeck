# Tasks: Configure a Microsoft Agent Framework Agent via YAML (US2)

**Input**: Design documents from `/specs/023-choose-your-backend/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Tests**: TDD approach — write tests FIRST, verify they FAIL, then implement.
**Dependency**: US2 depends on US1's setup phase (BackendEnum, ProviderEnum, BackendSelector routing). Tasks below assume those foundational changes are already merged.

**Organization**: Tasks are grouped by phase to enable incremental delivery. All tasks belong to User Story 2.

## Format: `[ID] [P?] [US2] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US2]**: All tasks belong to User Story 2
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/holodeck/`, `tests/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add AF optional dependency groups to pyproject.toml. BackendEnum, ProviderEnum, and BackendSelector changes are assumed complete from US1.

**CRITICAL**: US1 setup phase MUST be complete before starting. Specifically: BackendEnum (in `src/holodeck/models/llm.py`), ProviderEnum updates, and BackendSelector routing (in `src/holodeck/lib/backends/selector.py`) must already exist.

- [ ] T001 [US2] Add `agent-framework` optional dependency group to `pyproject.toml` under `[project.optional-dependencies]` with `agent-framework-core==1.0.0rc4`. Add `agent-framework-anthropic` sub-extra group with `agent-framework-core==1.0.0rc4` and `agent-framework-anthropic==1.0.0rc4`. Add `agent-framework-ollama` sub-extra group with `agent-framework-core==1.0.0rc4` and `agent-framework-ollama==1.0.0rc4`. Run `uv lock` to regenerate the lock file
- [ ] T002 [US2] Verify AF package installs correctly by running `uv pip install -e ".[agent-framework]"` and confirming that `from agent_framework import Agent` (or verified import path from P0-1 prerequisite) is importable. Fix package names in `pyproject.toml` if verification reveals different names

---

## Phase 2: Foundational (AF-Specific Prerequisites)

**Purpose**: Create the AF-specific config model and add the `agent_framework` field to the Agent model. If US1 already added the `agent_framework` field stub to `src/holodeck/models/agent.py`, skip T004.

**CRITICAL**: Phase 1 must be complete before starting.

- [ ] T003 [US2] Create `src/holodeck/models/af_config.py` with the `AgentFrameworkConfig` Pydantic model. Include: `CompactionStrategy` enum (`none`, `sliding_window`, `summarization` — use `str, Enum` base classes), `compaction_strategy: CompactionStrategy` field (default `CompactionStrategy.NONE`), `max_tool_rounds: int | None` field (default `None`, `ge=1` constraint). Set `model_config = ConfigDict(extra="forbid")`. Add docstrings per PEP 257 and type hints per project standards
- [ ] T004 [US2] Add `agent_framework: AgentFrameworkConfig | None = None` field to the `Agent` model in `src/holodeck/models/agent.py`. Import `AgentFrameworkConfig` from `holodeck.models.af_config`. Add Field with description "Agent Framework backend-specific configuration". This field is only validated when resolved backend is `agent_framework`; silently ignored otherwise
- [ ] T005 [P] [US2] Create unit test file `tests/unit/models/test_af_config.py` with tests for `AgentFrameworkConfig`: (1) test default values (`compaction_strategy=none`, `max_tool_rounds=None`), (2) test valid `compaction_strategy` values (`none`, `sliding_window`, `summarization`), (3) test invalid `compaction_strategy` value raises `ValidationError`, (4) test `max_tool_rounds` with valid value (e.g., `5`), (5) test `max_tool_rounds=0` raises `ValidationError` (ge=1), (6) test `extra="forbid"` rejects unknown fields. Import from `holodeck.models.af_config`
- [ ] T006 [US2] Run tests in `tests/unit/models/test_af_config.py` with `pytest tests/unit/models/test_af_config.py -n auto -v` and verify all pass

**Checkpoint**: AF config model ready. Agent model accepts `agent_framework` config section.

---

## Phase 3: User Story 2 Implementation

**Purpose**: Implement the AF backend, session, tool adapters, OTel wrapping, and provider client selection. This is the core of US2.

### 3a: AF Tool Adapters

- [ ] T007 [P] [US2] Create unit test file `tests/unit/lib/backends/test_af_tool_adapters.py` with tests for all tool type conversions. Tests MUST be written FIRST and verified to FAIL before implementation. Include: (1) `test_vectorstore_tool_creates_function_tool` — mock a VectorstoreTool instance, call adapter, assert returns AF `FunctionTool` wrapping the search function, (2) `test_hierarchical_document_tool_creates_function_tool` — same pattern for HierarchicalDocumentTool, (3) `test_function_tool_creates_af_function_tool` — load a Python callable, assert wrapped as AF `FunctionTool`, (4) `test_mcp_stdio_creates_mcp_stdio_tool` — MCPConfig with `transport: stdio`, `command`, `args`, assert creates `MCPStdioTool(name, command, args)`, (5) `test_mcp_sse_creates_streamable_http_tool` — MCPConfig with `transport: sse` and `url`, assert creates `MCPStreamableHTTPTool(name, url)`, (6) `test_mcp_websocket_creates_websocket_tool` — MCPConfig with `transport: websocket` and `url`, assert creates `MCPWebsocketTool(name, url)`. Use lazy import mocking for all AF SDK types
- [ ] T008 [US2] Create `src/holodeck/lib/backends/af_tool_adapters.py` with `adapt_tools_for_af(tool_configs: list[ToolUnion], tool_instances: dict[str, Any]) -> list[Any]` function. Use LAZY IMPORTS for all AF SDK types (guard with try/except ImportError raising `BackendInitError` with install instructions). Implement adapter logic per research.md R4: VectorstoreTool/HierarchicalDocumentTool → `FunctionTool(func=instance.search)`, FunctionTool → `FunctionTool(func=loaded_fn)`, MCPConfig stdio → `MCPStdioTool(name, command, args)`, MCPConfig sse/http → `MCPStreamableHTTPTool(name, url)`, MCPConfig websocket → `MCPWebsocketTool(name, url)`. Add docstrings and type hints
- [ ] T009 [US2] Run tests in `tests/unit/lib/backends/test_af_tool_adapters.py` with `pytest tests/unit/lib/backends/test_af_tool_adapters.py -n auto -v` and verify all pass

### 3b: AF OTel Span Wrapping

- [ ] T010 [P] [US2] Create `src/holodeck/lib/backends/af_otel.py` with basic OTel span wrapping for AF backend operations. Implement: (1) `trace_af_invoke(func)` async decorator that wraps `invoke_once()` calls in an OTel span named `holodeck.af.invoke` with attributes `af.model_name`, `af.provider`, (2) `trace_af_session_send(func)` async decorator that wraps `session.send()` calls in an OTel span named `holodeck.af.session.send`, (3) `trace_af_tool_call(func)` async decorator for tool invocations with span name `holodeck.af.tool`. Use lazy import for `opentelemetry.trace` — if not available, decorators should be no-ops (return the unwrapped function). Add docstrings

### 3c: AF Backend and Session

- [ ] T011 [P] [US2] Create unit test file `tests/unit/lib/backends/test_af_backend.py` with tests for AFBackend and AFSession. Tests MUST be written FIRST and verified to FAIL. Mock ALL AF SDK imports lazily. Include: (1) `test_initialize_creates_openai_client` — agent with `model.provider=openai`, `model.name=gpt-4o`, call `initialize()`, assert `OpenAIChatClient` instantiated with correct model and API key from env, (2) `test_initialize_creates_azure_client` — agent with `model.provider=azure_openai`, assert `AzureOpenAIChatClient` instantiated with endpoint, api_version, api_key, (3) `test_initialize_creates_anthropic_client` — agent with `model.provider=anthropic`, assert `AnthropicClient` instantiated, (4) `test_initialize_creates_ollama_client` — agent with `model.provider=ollama`, assert `OllamaChatClient` instantiated with endpoint, (5) `test_initialize_google_provider_raises` — agent with `model.provider=google` and `backend=agent_framework`, assert raises `BackendInitError` with message about Google not being supported by AF, (6) `test_invoke_once_returns_execution_result` — mock `agent.run()` to return a mock `AgentResponse` with `.text`, `.usage_details`, `.messages`, call `invoke_once("hello")`, assert `ExecutionResult` has correct `response`, `token_usage`, `tool_calls`, `is_error=False`, (7) `test_invoke_once_populates_tool_calls` — mock `agent.run()` response with tool call messages, assert `ExecutionResult.tool_calls` and `ExecutionResult.tool_results` populated correctly, (8) `test_invoke_once_handles_error` — mock `agent.run()` to raise exception, assert `ExecutionResult` returned with `is_error=True` and `error_reason` set, (9) `test_create_session_returns_af_session` — call `create_session()`, assert returned object has `send`, `send_streaming`, `close` methods, (10) `test_session_send_maintains_history` — create session, send two messages, verify AF native session passed to `agent.run()` on both calls, (11) `test_session_close_cleans_up` — create and close session, verify resources released, (12) `test_teardown_cleans_backend` — initialize and teardown, verify `_agent` and `_client` set to None, (13) `test_af_import_error_gives_install_instructions` — mock AF import to raise ImportError, call `initialize()`, assert `BackendInitError` raised with message containing `pip install "holodeck[agent-framework]"`
- [ ] T012 [US2] Create `src/holodeck/lib/backends/af_backend.py` implementing `AFBackend` class that satisfies the `AgentBackend` protocol from `src/holodeck/lib/backends/base.py`. Use LAZY IMPORTS for all AF SDK types. Use import alias `from agent_framework import AgentSession as AFNativeSession` to avoid collision with HoloDeck's `AgentSession` protocol. Constructor takes `agent: Agent`, `tool_instances: dict[str, Any]`. Implement `initialize()`: (a) try-import AF packages, raise `BackendInitError` with install instructions on ImportError, (b) select client class based on `agent.model.provider` per research.md R5 mapping (openai → `OpenAIChatClient`, azure_openai → `AzureOpenAIChatClient`, anthropic → `AnthropicClient`, ollama → `OllamaChatClient`), (c) raise `BackendInitError` for `google` provider with clear error message, (d) instantiate client with model name, API key, endpoint etc from agent config and environment, (e) call `adapt_tools_for_af()` to convert tools, (f) create agent via `client.as_agent(name=agent.name, instructions=resolved_instructions, tools=adapted_tools)`, (g) apply `AgentFrameworkConfig` settings (compaction_strategy, max_tool_rounds) if present. Implement `invoke_once(message: str) -> ExecutionResult`: (a) call `await self._agent.run(message)`, (b) extract response text from `result.text`, (c) extract token usage from `result.usage_details` into `TokenUsage`, (d) extract tool calls/results from `result.messages`, (e) return populated `ExecutionResult`. Implement `create_session() -> AFSession`. Implement `teardown()`: cleanup agent and client references
- [ ] T013 [US2] In the same file `src/holodeck/lib/backends/af_backend.py`, implement `AFSession` class satisfying the `AgentSession` protocol. Constructor takes `agent` and creates AF native session via `agent.create_session()`. Implement `send(message: str) -> ExecutionResult`: call `await self._agent.run(message, session=self._session)`, extract results same as `invoke_once()`. Implement `send_streaming(message: str) -> AsyncGenerator[str, None]`: call `self._agent.run(message, session=self._session, stream=True)`, iterate `ResponseStream` yielding text chunks. Implement `close()`: release session reference. Note: MCP tools require `async with self._agent:` context manager — handle this in `initialize()` by entering the context and storing the exit for `teardown()`
- [ ] T014 [US2] Run tests in `tests/unit/lib/backends/test_af_backend.py` with `pytest tests/unit/lib/backends/test_af_backend.py -n auto -v` and verify all pass

### 3d: Backend Exports and Selector Integration

- [ ] T015 [US2] Update `src/holodeck/lib/backends/__init__.py` to export `AFBackend` and `AFSession` types. Use conditional/lazy import pattern: only import from `af_backend` when needed (e.g., inside a function or behind TYPE_CHECKING guard) to avoid import-time AF dependency
- [ ] T016 [US2] Verify BackendSelector routing for AF in `src/holodeck/lib/backends/selector.py` — confirm that `BackendEnum.AGENT_FRAMEWORK` routes to `AFBackend` and that auto-detection maps `openai` and `azure_openai` providers to `agent_framework` backend (clean break, no deprecation warning). `ollama` auto-detects to `claude` backend (clean break, no deprecation warning). If US1 left placeholder routing, replace with actual `AFBackend` import (lazy). Add test in `tests/unit/lib/backends/test_selector.py`: (1) `test_select_af_backend_for_openai_provider` — agent with `provider=openai`, no explicit backend, assert AF backend selected directly with no deprecation warning emitted, (2) `test_select_af_backend_for_azure_provider` — agent with `provider=azure_openai`, assert AF backend selected directly with no deprecation warning emitted, (3) `test_select_af_backend_explicit` — agent with `backend=agent_framework`, assert AF backend selected, (4) `test_select_af_backend_google_provider_rejected` — agent with `provider=google`, `backend=agent_framework`, assert raises validation error, (5) `test_select_claude_backend_for_ollama_provider` — agent with `provider=ollama`, no explicit backend, assert Claude backend selected directly with no deprecation warning emitted

### 3e: Acceptance Scenario Verification Tests

- [ ] T017 [P] [US2] Write acceptance test `test_openai_provider_autodetects_af_backend` in `tests/unit/lib/backends/test_af_backend.py` — create a full Agent config with `model.provider=openai`, `model.name=gpt-4o`, instructions (inline), no explicit backend field. Mock AF SDK. Call `BackendSelector.select()`, assert returns AFBackend directly (no deprecation warning emitted — use `warnings.catch_warnings()` to assert zero warnings). Call `initialize()`, mock `OpenAIChatClient`, assert client created. Call `invoke_once("What is AI?")`, mock `agent.run()` to return response text "AI is...". Assert `ExecutionResult.response == "AI is..."` and `is_error is False`
- [ ] T018 [P] [US2] Write acceptance test `test_azure_openai_uses_af_backend` in `tests/unit/lib/backends/test_af_backend.py` — create Agent config with `model.provider=azure_openai`, `model.name=gpt-4o`, `model.endpoint=https://myendpoint.openai.azure.com`, `model.api_version=2024-02-01`. Mock AF SDK. Assert `AzureOpenAIChatClient` created with correct endpoint, api_version, and API key
- [ ] T019 [P] [US2] Write acceptance test `test_mcp_stdio_tools_via_af` in `tests/unit/lib/backends/test_af_backend.py` — create Agent config with `provider=openai` and an MCP tool config (transport=stdio, command="node", args=["server.js"]). Mock AF SDK. Initialize backend, assert `MCPStdioTool` created in adapted tools list. Mock `agent.run()` returning a response with tool call records. Assert `ExecutionResult.tool_calls` contains the MCP tool invocation
- [ ] T020 [P] [US2] Write acceptance test `test_multiturn_chat_via_af_session` in `tests/unit/lib/backends/test_af_backend.py` — create Agent config with `provider=openai`. Mock AF SDK. Create session via `backend.create_session()`. Send message 1, assert response returned. Send message 2, assert AF native session object was passed to `agent.run()` on the second call (verifying conversation state maintained). Close session, assert cleanup completed

---

## Phase 4: Polish

**Purpose**: Code quality, documentation, and final validation.

- [ ] T021 [P] [US2] Run `make format` to format all new and modified files with Black + Ruff
- [ ] T022 [P] [US2] Run `make lint` and fix any Ruff + Bandit violations in `src/holodeck/models/af_config.py`, `src/holodeck/lib/backends/af_backend.py`, `src/holodeck/lib/backends/af_tool_adapters.py`, `src/holodeck/lib/backends/af_otel.py`
- [ ] T023 [US2] Run `make type-check` and fix any MyPy errors — ensure all AF lazy imports have proper type annotations (use `Any` for unresolvable AF SDK types behind `TYPE_CHECKING`)
- [ ] T024 [US2] Run full test suite `make test` to verify no regressions across entire codebase
- [ ] T025 [US2] Verify quickstart.md examples for AF backend match implementation — confirm YAML examples with `provider: openai` and `provider: azure_openai` are accurate and include `agent_framework` config section examples
- [ ] T026 [US2] Add unit test `test_af_execution_result_has_evaluator_required_fields` in `tests/unit/lib/backends/test_af_backend.py` — verify ExecutionResult returned by AFBackend contains all fields required for evaluator compatibility (FR-016). Mock AF SDK and `agent.run()` to return a realistic response. Call `invoke_once()`, then assert field presence and types on the returned ExecutionResult: `response` (str), `tool_calls` (list), `tool_results` (list), `token_usage` (TokenUsage), `num_turns` (int), `is_error` (bool), `error_reason` (str | None). This is a lightweight field-presence/type check, not a full evaluator integration test

---

## Dependencies & Execution Order

### Cross-Story Dependencies

- **US1 Setup (REQUIRED BEFORE US2)**: BackendEnum, ProviderEnum updates, BackendSelector routing infrastructure in `src/holodeck/models/llm.py` and `src/holodeck/lib/backends/selector.py` must be completed in US1 before US2 can begin.
- **US2 is independent of US1's Google ADK implementation** — only US1's shared setup phase is required.

### Phase Dependencies

- **Phase 1 (Setup)**: Depends on US1 setup completion — add AF deps to pyproject.toml
- **Phase 2 (Foundational)**: Depends on Phase 1 — AF config model + Agent model field
- **Phase 3 (Implementation)**: Depends on Phase 2 — all AF backend components
  - Phase 3a (Tool Adapters) and 3b (OTel) can run in parallel — different files, no dependencies
  - Phase 3c (Backend/Session) depends on 3a (uses tool adapters) and 3b (uses OTel decorators)
  - Phase 3d (Exports/Selector) depends on 3c (exports the backend classes)
  - Phase 3e (Acceptance Tests) depends on 3d (needs selector routing working)
- **Phase 4 (Polish)**: Depends on all Phase 3 sub-phases being complete

### Within Each Sub-Phase

- Tests MUST be written and FAIL before implementation (TDD)
- Implementation after tests
- Verify tests PASS after implementation

---

## Parallel Example: Phase 3a + 3b

```bash
# These tasks can run in parallel (different files, no dependencies):
Task T007: "Write tool adapter tests in tests/unit/lib/backends/test_af_tool_adapters.py"
Task T010: "Create OTel span wrappers in src/holodeck/lib/backends/af_otel.py"
```

## Parallel Example: Acceptance Scenarios

```bash
# These tasks can run in parallel (independent test functions, same file):
Task T017: "Write test_openai_provider_autodetects_af_backend"
Task T018: "Write test_azure_openai_uses_af_backend"
Task T019: "Write test_mcp_stdio_tools_via_af"
Task T020: "Write test_multiturn_chat_via_af_session"
```

---

## Implementation Strategy

### MVP First (Phases 1-3c Only)

1. Complete Phase 1: Setup (T001-T002) — AF deps installed
2. Complete Phase 2: Foundational (T003-T006) — AF config model ready
3. Complete Phase 3a: Tool Adapters (T007-T009) — tools convert correctly
4. Complete Phase 3c: Backend/Session (T011-T014) — core AF backend works
5. **STOP and VALIDATE**: Run `make test` — AF backend initializes, invokes, sessions work
6. Deploy/demo if ready

### Incremental Delivery

1. Phase 1 → AF dependency available
2. Phase 2 → AF config model validated, Agent model accepts it
3. Phase 3a → Tool adapters tested independently
4. Phase 3b → OTel wrapping available (no-op if OTel not installed)
5. Phase 3c → Core backend functional (MVP!)
6. Phase 3d → Integrated into BackendSelector routing
7. Phase 3e → All acceptance scenarios verified
8. Phase 4 → Code quality validated, no regressions

### Key Files Created

| File | Phase | Purpose |
|------|-------|---------|
| `src/holodeck/models/af_config.py` | Phase 2 | AgentFrameworkConfig Pydantic model with CompactionStrategy enum |
| `src/holodeck/lib/backends/af_tool_adapters.py` | Phase 3a | HoloDeck tool → AF tool conversion for all 5 tool types + MCP transports |
| `src/holodeck/lib/backends/af_otel.py` | Phase 3b | Basic OTel span decorators for invoke, session.send, tool calls |
| `src/holodeck/lib/backends/af_backend.py` | Phase 3c | AFBackend + AFSession implementing AgentBackend/AgentSession protocols |
| `tests/unit/models/test_af_config.py` | Phase 2 | Unit tests for AF config model |
| `tests/unit/lib/backends/test_af_tool_adapters.py` | Phase 3a | Unit tests for tool adapters |
| `tests/unit/lib/backends/test_af_backend.py` | Phase 3c | Unit tests for backend + session + acceptance scenarios |

### Key Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `pyproject.toml` | Phase 1 | `[agent-framework]`, `[agent-framework-anthropic]`, `[agent-framework-ollama]` optional groups |
| `src/holodeck/models/agent.py` | Phase 2 | `agent_framework: AgentFrameworkConfig \| None` field |
| `src/holodeck/lib/backends/__init__.py` | Phase 3d | Export AFBackend, AFSession |
| `src/holodeck/lib/backends/selector.py` | Phase 3d | Verify/update AF routing (may already be done by US1) |
| `tests/unit/lib/backends/test_selector.py` | Phase 3d | AF-specific routing tests |

---

## Notes

- [P] tasks = different files, no dependencies between them
- [US2] label on every task for traceability
- All AF SDK imports MUST be lazy (inside functions/methods, not at module level) to prevent import-time failures when AF is not installed
- Import alias required: `from agent_framework import AgentSession as AFNativeSession` to avoid collision with HoloDeck's `AgentSession` protocol in `base.py`
- `provider: google` is NOT supported by AF — must raise clear error with message directing user to use `backend: google_adk` instead
- AF's `MCPStdioTool` requires `async with agent:` context manager — AFBackend.initialize() must enter this context and store the exit for teardown()
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
