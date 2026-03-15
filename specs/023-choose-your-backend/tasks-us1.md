# Tasks: User Story 1 — Configure a Google ADK Agent via YAML

**Feature**: 023-choose-your-backend | **Story**: US1 | **Date**: 2026-03-15
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md)

**User Story**: A platform user wants to use Google's Gemini models through the Google ADK backend. They create an agent.yaml specifying `provider: google` and a Gemini model name. HoloDeck auto-detects the ADK backend, initializes the agent, and executes the request.

**Acceptance Scenarios**:
1. Valid agent.yaml with `model.provider: google` and `model.name: gemini-2.5-flash` → auto-detect ADK, invoke, return response with text and token usage.
2. agent.yaml with `provider: google` and vectorstore tools → tools initialized and available to ADK agent.
3. agent.yaml with `provider: google` and MCP tools → MCP tools invoked, results returned with tool call/result records.
4. agent.yaml with `provider: google` → `holodeck chat` establishes multi-turn session with context maintained.

---

## Phase 1: Setup — Shared Infrastructure

These tasks create the foundational enums, dependencies, and routing changes that US1 (and future stories) depend on.

- [ ] T001 [US1] Add `google-adk>=1.2.0,<2.0.0` as an optional dependency group in `pyproject.toml` at `/Users/justinbarias/Documents/Git/python/agentlab/pyproject.toml`. Add a new `[project.optional-dependencies]` entry: `google-adk = ["google-adk>=1.2.0,<2.0.0"]`. Do NOT add it to the core `dependencies` list (lazy import only).

- [ ] T002 [US1] Add `GOOGLE = "google"` to `ProviderEnum` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/llm.py`. Place it after the existing `OLLAMA` entry. This enables `model.provider: google` in agent.yaml.

- [ ] T003 [US1] Create `BackendEnum` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/llm.py`. Add a new `BackendEnum(str, Enum)` class with four values: `SEMANTIC_KERNEL = "semantic_kernel"`, `CLAUDE = "claude"`, `GOOGLE_ADK = "google_adk"`, `AGENT_FRAMEWORK = "agent_framework"`. Place it after `ProviderEnum`. This enum identifies the agent runtime, decoupled from the LLM provider.

- [ ] T004 [US1] Add optional `backend` field to the `Agent` model in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/agent.py`. Import `BackendEnum` from `holodeck.models.llm`. Add field: `backend: BackendEnum | None = Field(default=None, description="Agent runtime override. When None, auto-detected from model.provider.")`. Place it after the `model` field.

- [ ] T005 [US1] Add optional `google_adk` config field to the `Agent` model in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/agent.py`. Import `GoogleADKConfig` from `holodeck.models.google_adk_config` (will be created in T008). Add field: `google_adk: GoogleADKConfig | None = Field(default=None, description="Google ADK-specific settings. Applicable when resolved backend is google_adk.")`. Place it after the `claude` field.

- [ ] T006 [US1] Write unit tests for `ProviderEnum.GOOGLE` and `BackendEnum` in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/models/test_llm.py`. Add tests verifying: (a) `ProviderEnum("google") == ProviderEnum.GOOGLE`, (b) all four `BackendEnum` values are accessible and have correct string values, (c) invalid enum values raise `ValueError`. Run with `pytest tests/unit/models/test_llm.py -n auto`.

---

## Phase 2: Foundational — Blocking Prerequisites

These tasks implement the BackendSelector routing changes and Agent model validators that must exist before the ADK backend can be wired up.

- [ ] T007 [US1] Update `BackendSelector.select()` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/selector.py` to route by the `backend` field first, falling back to auto-detect from `model.provider`. Import `BackendEnum` from `holodeck.models.llm`. When `agent.backend` is explicitly set, use it directly. When `agent.backend is None`, resolve from `model.provider` using the default routing table: `openai` → `agent_framework`, `azure_openai` → `agent_framework`, `anthropic` → `claude`, `ollama` → `claude`, `google` → `google_adk`. This is a clean break from the legacy SK routing — users who need Semantic Kernel must set `backend: semantic_kernel` explicitly. For `BackendEnum.GOOGLE_ADK`, lazily import `ADKBackend` from `holodeck.lib.backends.adk_backend` and instantiate it. For `BackendEnum.AGENT_FRAMEWORK`, lazily import `AFBackend` from `holodeck.lib.backends.af_backend` and instantiate it (stub — implemented in US2). For `BackendEnum.CLAUDE`, use existing `ClaudeBackend`. Wrap the import in a `try/except ImportError` that raises `BackendInitError` with message: `"Google ADK backend requires the 'google-adk' package. Install it with: pip install holodeck[google-adk]"`.

- [ ] T008 [US1] Create `GoogleADKConfig` Pydantic model in new file `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/google_adk_config.py`. Define `StreamingMode(str, Enum)` with values `NONE = "none"` and `SSE = "sse"`. Define `GoogleADKConfig(BaseModel)` with `ConfigDict(extra="forbid")` and fields: `streaming_mode: StreamingMode = Field(default=StreamingMode.NONE)`, `code_execution: bool = Field(default=False)`, `max_iterations: int | None = Field(default=None, ge=1)`, `output_key: str = Field(default="output")`. Follow the pattern of `ClaudeConfig` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/claude_config.py`.

- [ ] T009 [US1] Create `@model_validator(mode='after')` in the `Agent` model at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/agent.py` that enforces `embedding_provider` is set when the resolved backend is `google_adk` (or `claude`) AND vectorstore or hierarchical_document tools are configured. Import `BackendEnum` and `ProviderEnum`. Resolve the effective backend: if `self.backend` is set, use it; otherwise auto-detect from `self.model.provider` using the default routing table (same as T007: `openai` → `agent_framework`, `azure_openai` → `agent_framework`, `anthropic` → `claude`, `ollama` → `claude`, `google` → `google_adk`). Check if any tool in `self.tools` has `type == "vectorstore"` or `type == "hierarchical_document"`. If so, and `self.embedding_provider is None`, raise `ValueError` with message: `"embedding_provider is required when using vectorstore or hierarchical_document tools with the {backend} backend"`. This fixes the pre-existing gap for `anthropic` provider.

- [ ] T010 [US1] Create backend/provider compatibility validator as a `@model_validator(mode='after')` in the `Agent` model at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/models/agent.py`. When `self.backend` is explicitly set, validate that `self.model.provider` is compatible using the compatibility matrix: `google_adk` supports `google`, `openai`, `azure_openai`, `anthropic`, `ollama`; `claude` supports `anthropic` only; `semantic_kernel` supports `openai`, `azure_openai`, `ollama`. Raise `ValueError` with a clear message listing compatible providers for the chosen backend if validation fails.

- [ ] T011 [US1] Write unit tests for `GoogleADKConfig` in new file `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/models/test_google_adk_config.py`. Test: (a) default values are correct, (b) `streaming_mode: "sse"` parses to `StreamingMode.SSE`, (c) `max_iterations: 0` raises `ValidationError` (ge=1), (d) `max_iterations: null` is valid, (e) extra fields are rejected (`extra="forbid"`), (f) `code_execution: true` is accepted. Run with `pytest tests/unit/models/test_google_adk_config.py -n auto`.

- [ ] T012 [US1] Write unit tests for updated `BackendSelector` routing in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/lib/backends/test_selector.py`. Add tests: (a) `provider: google` auto-detects to ADK backend (mock `ADKBackend`), (b) explicit `backend: google_adk` routes to ADK, (c) explicit `backend: google_adk` with `provider: openai` routes to ADK (cross-provider), (d) missing `google-adk` package raises `BackendInitError` with install instructions, (e) `provider: openai` auto-routes to `agent_framework` (NOT `semantic_kernel` — clean break), (f) `provider: azure_openai` auto-routes to `agent_framework`, (g) `provider: ollama` auto-routes to `claude`, (h) existing `provider: anthropic` routing still works (Claude), (i) explicit `backend: semantic_kernel` with `provider: openai` routes to SK (opt-in only). Run with `pytest tests/unit/lib/backends/test_selector.py -n auto`.

- [ ] T013 [US1] Write unit tests for Agent model validators (embedding_provider and backend/provider compatibility) in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/models/test_agent.py`. Add tests: (a) `provider: google` + vectorstore tool + no `embedding_provider` → `ValueError`, (b) `provider: google` + vectorstore tool + `embedding_provider` set → valid, (c) `provider: anthropic` + vectorstore tool + no `embedding_provider` → `ValueError`, (d) `backend: claude` + `provider: openai` → `ValueError` with compatible provider list, (e) `backend: google_adk` + `provider: openai` → valid, (f) `backend: google_adk` + `provider: google` → valid. Run with `pytest tests/unit/models/test_agent.py -n auto`.

---

## Phase 3: User Story 1 Implementation — ADK Backend

These tasks implement the ADK backend, session, tool adapters, and OTel instrumentation.

- [ ] T014 [US1] Create `ADKBackend` class in new file `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/adk_backend.py`. The class must implement the `AgentBackend` protocol from `holodeck.lib.backends.base`. Use lazy imports: `from google.adk.agents import LlmAgent`, `from google.adk.runners import Runner`, `from google.adk.sessions import InMemorySessionService`. Constructor accepts `agent_config: Agent` and `tool_instances: dict[str, Any] | None`. State attributes: `_agent: LlmAgent | None`, `_runner: Runner | None`, `_session_service: InMemorySessionService | None`, `_tools: list[Any]`, `_config: GoogleADKConfig | None` (from `agent_config.google_adk`). Implement `initialize()`: create `InMemorySessionService`, build the `LlmAgent` with name from `agent_config.name`, model from `agent_config.model.name` (for `provider: google`) or LiteLLM prefix `{provider}/{name}` (for cross-provider), instruction from resolved instructions, and adapted tools. Create `Runner(agent=agent, session_service=session_service)`. Implement `teardown()`: set references to None.

- [ ] T015 [US1] Implement `invoke_once()` on `ADKBackend` in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/adk_backend.py`. Create a temporary session via `session_service.create_session(app_name=agent_name, user_id=unique_id)`. Build `new_message` using `google.genai.types.Content(role="user", parts=[Part.from_text(message)])`. Call `runner.run_async(user_id, session_id, new_message)` and iterate the `AsyncGenerator[Event]`. Collect events: extract final response text from `event.content.parts` when `event.is_final_response()`, collect tool calls from `event.get_function_calls()`, extract token usage from `event.usage_metadata` (map to `TokenUsage(prompt_tokens, completion_tokens, total_tokens)`). Count turns by tracking function call/response cycles. Return `ExecutionResult` with all fields populated. Wrap errors in `BackendSessionError`.

- [ ] T016 [US1] Create `ADKSession` class in `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/adk_backend.py`. Implements `AgentSession` protocol. Constructor accepts `runner: Runner`, `session_service: InMemorySessionService`, `app_name: str`, `user_id: str`, `session_id: str`. Implement `send()`: build Content message, call `runner.run_async()`, collect events same as `invoke_once()`, return `ExecutionResult`. Conversation history is automatically maintained by the `InMemorySessionService` across calls with the same `session_id`. Implement `send_streaming()`: use `RunConfig(streaming_mode=StreamingMode.SSE)` passed to `runner.run_async()`, yield partial text from non-final events. Implement `close()`: no-op (in-memory session cleanup is handled by garbage collection). Implement `create_session()` on `ADKBackend`: generate unique `user_id` and `session_id` using `python-ulid`, create session via `session_service.create_session()`, return `ADKSession` instance.

- [ ] T017 [US1] Create ADK tool adapter module at `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/adk_tool_adapters.py`. Implement `adapt_tools_for_adk(tools: list[ToolUnion], tool_instances: dict[str, Any] | None) -> list[Any]` that converts HoloDeck tool definitions to ADK-compatible tools. Handle each tool type: (a) **VectorstoreTool**: Create an async callable wrapping `tool_instances[name].search(query)` that takes a `query: str` parameter and returns search results as a string. ADK auto-wraps plain callables as `FunctionTool`. (b) **HierarchicalDocumentTool**: Same pattern as VectorstoreTool. (c) **FunctionTool**: Load the Python function from `tool.module` and `tool.function` using `importlib`, pass the callable directly (ADK auto-wraps). (d) **MCPTool (stdio)**: Create `McpToolset(connection_params=StdioConnectionParams(command=tool.command, args=tool.args))`. Import from `google.adk.tools.mcp_tool.mcp_toolset`. (e) **MCPTool (sse)**: Create `McpToolset(connection_params=SseConnectionParams(url=tool.url))`. (f) **MCPTool (http)**: Create `McpToolset(connection_params=StreamableHTTPConnectionParams(url=tool.url))`. All ADK imports must be lazy (inside the function body). **Note**: SkillTool (`type: skill`) adapter is OUT OF SCOPE for US1 — it is deferred to US7 (tasks T730–T738). US1 builds adapters for 4 tool types only: vectorstore, hierarchical_document, function, and MCP. If a SkillTool is encountered, raise `ToolError` with message: `"SkillTool adapter is not yet supported for the ADK backend (planned for US7)"`.

- [ ] T018 [US1] Create basic OTel span wrapping for ADK backend in new file `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/adk_otel.py`. Implement a decorator or context manager `adk_span(operation_name: str)` that creates an OpenTelemetry span wrapping ADK operations. Use `opentelemetry.trace.get_tracer("holodeck.backends.adk")`. Wrap the following operations: `invoke_once`, `session.send`, and tool calls. Add span attributes: `gen_ai.system = "google_adk"`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.prompt_tokens`, `gen_ai.usage.completion_tokens`. Apply the decorator/context manager in `adk_backend.py` `invoke_once()` and `ADKSession.send()` methods.

- [ ] T019 [US1] Update `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/__init__.py` to conditionally export ADK types. Add a try/except block that imports `ADKBackend` and `ADKSession` from `holodeck.lib.backends.adk_backend` and adds them to `__all__`. If `ImportError` occurs (google-adk not installed), skip the exports silently. This ensures existing imports work when google-adk is not installed.

- [ ] T020 [US1] Write unit tests for `ADKBackend` in new file `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/lib/backends/test_adk_backend.py`. Mock all `google.adk` imports using `pytest-mock`. Test: (a) `initialize()` creates `InMemorySessionService`, `LlmAgent`, and `Runner` with correct parameters, (b) `invoke_once()` sends message and returns `ExecutionResult` with response text, token usage, and turn count, (c) `invoke_once()` with tool calls populates `tool_calls` and `tool_results` in `ExecutionResult`, (d) `create_session()` returns an `ADKSession` with unique user_id and session_id, (e) `teardown()` cleans up references, (f) `ADKSession.send()` returns `ExecutionResult` with conversation context maintained, (g) `ADKSession.close()` completes without error, (h) cross-provider model name construction (e.g., `provider: openai` + `name: gpt-4o` → model string `"openai/gpt-4o"`), (i) `provider: google` + `name: gemini-2.5-flash` → model string `"gemini-2.5-flash"` (no prefix). Use `@pytest.mark.asyncio` for all async tests. Run with `pytest tests/unit/lib/backends/test_adk_backend.py -n auto`.

- [ ] T021 [US1] Write unit tests for ADK tool adapters in new file `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/lib/backends/test_adk_tool_adapters.py`. Mock `google.adk` imports. Test: (a) VectorstoreTool adaptation creates a callable that wraps `instance.search()`, (b) HierarchicalDocumentTool adaptation creates a callable that wraps `instance.search()`, (c) FunctionTool adaptation loads the Python function and returns it directly, (d) MCPTool stdio creates `McpToolset` with `StdioConnectionParams`, (e) MCPTool sse creates `McpToolset` with `SseConnectionParams`, (f) empty tools list returns empty list, (g) unknown tool type raises `ToolError`, (h) SkillTool (`type: skill`) raises `ToolError` indicating deferral to US7. **Note**: US1 only covers 4 tool types (vectorstore, hierarchical_document, function, MCP). SkillTool adapter is deferred to US7 (tasks T730–T738). Run with `pytest tests/unit/lib/backends/test_adk_tool_adapters.py -n auto`.

---

## Phase 4: Polish — Docs, Cleanup, Quickstart Validation

- [ ] T022 [US1] Validate quickstart examples work end-to-end. Verify the "Google ADK with Gemini" example from `/Users/justinbarias/Documents/Git/python/agentlab/specs/023-choose-your-backend/quickstart.md` can be loaded as a valid `Agent` config. Create a simple integration-style unit test in `/Users/justinbarias/Documents/Git/python/agentlab/tests/unit/models/test_adk_quickstart_configs.py` that parses the three ADK quickstart YAML examples (basic Gemini, cross-provider OpenAI, streaming with `google_adk` section) through the `Agent` Pydantic model and asserts they validate without errors. This does NOT require google-adk installed — it only tests config parsing.

- [ ] T023 [US1] Run full test suite and code quality checks. Execute `make test-unit`, `make lint`, `make format-check`, and `make type-check` from the project root at `/Users/justinbarias/Documents/Git/python/agentlab`. Fix any failures introduced by US1 changes. Ensure all existing tests still pass (zero regressions). All new files must pass Black formatting, Ruff linting, and MyPy type checking.

---

## Dependencies & Execution Order

```
T001 (pyproject.toml deps)
  └─ no dependencies — start immediately

T002 (ProviderEnum.GOOGLE) ──┐
T003 (BackendEnum) ──────────┤
                              ├──→ T004 (Agent.backend field)
                              ├──→ T007 (BackendSelector routing)
                              └──→ T009 (embedding_provider validator)
                                   T010 (backend/provider compatibility validator)

T008 (GoogleADKConfig model) ──→ T005 (Agent.google_adk field)
                               └──→ T014 (ADKBackend constructor uses config)

T004 + T005 ──→ T009, T010, T013 (Agent validators + tests)

T006 (enum tests) ── can run after T002 + T003
T011 (GoogleADKConfig tests) ── can run after T008
T012 (BackendSelector tests) ── can run after T007

T007 (BackendSelector) ──→ T014 (ADKBackend)
T017 (tool adapters) ──→ T014 (ADKBackend uses adapted tools)
T014 ──→ T015 (invoke_once)
T014 ──→ T016 (ADKSession + create_session)
T014 ──→ T018 (OTel spans applied to backend)
T014 ──→ T019 (__init__.py exports)
T014 + T015 + T016 ──→ T020 (ADKBackend tests)
T017 ──→ T021 (tool adapter tests)

T020 + T021 + T013 ──→ T022 (quickstart validation)
T022 ──→ T023 (full suite + quality checks)
```

## Parallel Execution Example

The following groups can be executed in parallel within each phase:

**Phase 1 parallel group:**
- T001 (pyproject.toml) | T002 (ProviderEnum) | T003 (BackendEnum) | T008 (GoogleADKConfig) — all independent

**Phase 2 parallel group (after Phase 1):**
- T004 + T005 (Agent model fields) — depend on T002, T003, T008
- T006 (enum tests) — depends on T002, T003
- T011 (config tests) — depends on T008
- T007 (BackendSelector) — depends on T002, T003

**Phase 2 second wave:**
- T009 + T010 (Agent validators) — depend on T004
- T012 (BackendSelector tests) — depends on T007
- T013 (Agent validator tests) — depends on T009, T010

**Phase 3 parallel group (after Phase 2):**
- T017 (tool adapters) | T018 (OTel) — independent of each other, both depend on Phase 2
- T014 + T015 + T016 (backend + session) — depend on T007, T008, T017

**Phase 3 second wave:**
- T019 (__init__ exports) | T020 (backend tests) | T021 (adapter tests) — after T014-T017

**Phase 4 sequential:**
- T022 → T023

## Implementation Strategy

### Guiding Principles

1. **Lazy imports only**: All `google.adk.*` imports MUST be inside function/method bodies or guarded by `try/except ImportError`. The ADK package must never cause import-time failures when not installed.

2. **Protocol compliance**: `ADKBackend` must satisfy `AgentBackend` protocol and `ADKSession` must satisfy `AgentSession` protocol from `/Users/justinbarias/Documents/Git/python/agentlab/src/holodeck/lib/backends/base.py`. Use `isinstance()` checks in tests to verify protocol compliance.

3. **ExecutionResult completeness**: Every `invoke_once()` and `session.send()` call must return a fully populated `ExecutionResult` with: `response` (text), `tool_calls` (list of dicts with tool name/args), `tool_results` (list of dicts with tool output), `token_usage` (TokenUsage dataclass), `num_turns` (int), `is_error` (bool), `error_reason` (str | None).

4. **Cross-provider model naming**: For `provider: google`, pass model name directly to ADK (e.g., `"gemini-2.5-flash"`). For other providers with `backend: google_adk`, construct LiteLLM prefix: `"{provider}/{model_name}"` (e.g., `"openai/gpt-4o"`). See research.md R1.

5. **Error handling**: Wrap all ADK exceptions in HoloDeck backend errors: `BackendInitError` for initialization failures, `BackendSessionError` for runtime invocation failures, `BackendTimeoutError` for timeouts.

6. **Test isolation**: All unit tests must mock `google.adk` imports entirely. No test should require the google-adk package to be installed. Use `unittest.mock.patch` or `pytest-mock` to mock the `google.adk` module hierarchy.

### Key Files Reference

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | MODIFY | Add google-adk optional dep |
| `src/holodeck/models/llm.py` | MODIFY | Add GOOGLE to ProviderEnum, create BackendEnum |
| `src/holodeck/models/agent.py` | MODIFY | Add backend, google_adk fields + validators |
| `src/holodeck/models/google_adk_config.py` | NEW | GoogleADKConfig Pydantic model |
| `src/holodeck/lib/backends/selector.py` | MODIFY | Route by backend field, lazy ADK import |
| `src/holodeck/lib/backends/adk_backend.py` | NEW | ADKBackend + ADKSession |
| `src/holodeck/lib/backends/adk_tool_adapters.py` | NEW | HoloDeck → ADK tool conversion |
| `src/holodeck/lib/backends/adk_otel.py` | NEW | Basic OTel span wrapping |
| `src/holodeck/lib/backends/__init__.py` | MODIFY | Conditional ADK exports |
| `tests/unit/models/test_llm.py` | MODIFY | Enum tests |
| `tests/unit/models/test_google_adk_config.py` | NEW | Config model tests |
| `tests/unit/models/test_agent.py` | MODIFY | Validator tests |
| `tests/unit/lib/backends/test_selector.py` | MODIFY | Routing tests |
| `tests/unit/lib/backends/test_adk_backend.py` | NEW | Backend + session tests |
| `tests/unit/lib/backends/test_adk_tool_adapters.py` | NEW | Tool adapter tests |
| `tests/unit/models/test_adk_quickstart_configs.py` | NEW | Quickstart YAML validation |
