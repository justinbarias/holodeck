# Implementation Plan: Native Claude Agent SDK Integration

**Branch**: `021-claude-agent-sdk` | **Date**: 2026-02-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/021-claude-agent-sdk/spec.md`

---

## Summary

Replace the Semantic Kernel agent conversation loop for Anthropic-provider agents with a native Claude Agent SDK integration. The Claude Agent SDK spawns a managed subprocess that owns the agent loop, tool execution, permission enforcement, and streaming output. HoloDeck configures the subprocess, wraps its existing tools (vectorstore, hierarchical document) as SDK-compatible in-process tools, bridges external MCP server configs, and maps subprocess event output back to its evaluation and chat interfaces.

The implementation introduces a clean **backend abstraction layer** (`lib/backends/`) that decouples HoloDeck's test runner and chat layer from Semantic Kernel types — making the Claude-native backend a first-class peer and positioning HoloDeck to remove SK from the agent loop for additional providers in the future.

---

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
  - `claude-agent-sdk>=0.1.39,<0.2.0` (new — Claude subprocess management)
  - `semantic-kernel>=1.37.1,<2.0.0` (retained — embeddings + non-Anthropic providers)
  - `anthropic>=0.72.0` (retained — already present)

**Storage**: N/A — stateless per invocation; vector indexes managed by existing tools (unchanged)
**Testing**: pytest with `-n auto` parallel execution
**Target Platform**: macOS / Linux (cross-platform); Node.js required on PATH
**Project Type**: Single project (Python CLI + library)
**Performance Goals**: Streaming chat begins within 3 seconds of user submission (SC-004)
**Constraints**:
  - Node.js must be present on PATH (validated at startup)
  - Backward compatible: existing YAML configs require zero changes
  - SK stays installed; only agent loop replaced for Anthropic providers
  - All new `agent.yaml` fields optional with safe defaults (least-privilege)
**Scale/Scope**: Single-provider path change; all non-Anthropic paths unchanged

---

## Constitution Check

*GATE: Must pass before implementation. Re-checked after design.*

### I. No-Code-First Agent Definition ✅
All Claude-native capabilities (working directory, permission mode, max turns, extended thinking, file system, bash, subagents, web search) are configured via YAML extensions to `agent.yaml`. Zero Python code required from users. Tool registration is automatic and transparent (FR-023).

### II. MCP for API Integrations ✅
External API integrations remain MCP-only. The Claude-native backend uses the SDK's native MCP client for stdio servers via `McpStdioServerConfig`. The `mcp_bridge.py` translates HoloDeck `MCPTool` config without requiring users to change their YAML (FR-005, FR-035).

### III. Test-First with Multimodal Support ✅
FR-008 requires all evaluation metrics (NLP, G-Eval, RAG) to work with Claude-native agents. SC-001 through SC-016 provide measurable acceptance criteria. Multimodal file inputs continue through `FileProcessor` unchanged.

### IV. OpenTelemetry-Native Observability ✅
FR-036–FR-038 explicitly address OTel for Claude-native agents. `otel_bridge.py` translates `observability:` YAML into Claude Code environment variables before subprocess spawn. No SK OTel decorators are used for Anthropic agents.

### V. Evaluation Flexibility with Model Overrides ✅
FR-008 requires all three metric types to work against Claude-native agent responses. The provider-agnostic `ExecutionResult` interface (FR-012b) enables the existing evaluation pipeline to operate without modification.

**Architecture Constraints**: Agent Engine (new Claude backend) remains decoupled from Evaluation Framework and Deployment Engine. Cross-engine communication uses `ExecutionResult` (not SK types). ✅

**Complexity Tracking**: No violations. The `lib/backends/` abstraction layer is warranted by the dual-backend requirement and the user directive to decouple from Semantic Kernel.

---

## Project Structure

### Documentation (this feature)

```text
specs/021-claude-agent-sdk/
├── plan.md              # This file
├── research.md          # Phase 0 output (complete)
├── data-model.md        # Phase 1 output (complete)
├── quickstart.md        # Phase 1 output (complete)
├── contracts/
│   ├── execution-result.md      # ExecutionResult + AgentBackend interface
│   └── agent-yaml-schema.md     # YAML schema extensions
└── tasks.md             # Phase 2 output (run /speckit.tasks)
```

### Source Code

```text
src/holodeck/
├── models/
│   ├── agent.py                  # UPDATED: embedding_provider, claude fields
│   ├── chat.py                   # UPDATED: history type changed to list[dict]
│   ├── llm.py                    # UPDATED: auth_provider field
│   └── claude_config.py          # NEW: ClaudeConfig, AuthProvider, sub-models
│
├── lib/
│   ├── backends/                 # NEW PACKAGE: provider backend abstraction
│   │   ├── __init__.py
│   │   ├── base.py               # ExecutionResult, AgentSession, AgentBackend
│   │   ├── selector.py           # BackendSelector: routes by provider
│   │   ├── sk_backend.py         # SK backend (moved from agent_factory.py)
│   │   ├── claude_backend.py     # Claude Agent SDK backend
│   │   ├── tool_adapters.py      # VectorStore/HierarchicalDoc → Claude @tool
│   │   ├── mcp_bridge.py         # MCPTool config → McpStdioServerConfig
│   │   ├── otel_bridge.py        # ObservabilityConfig → Claude Code env vars
│   │   └── validators.py         # Node.js, credentials, embedding_provider checks
│   │
│   └── test_runner/
│       ├── executor.py           # UPDATED: use ExecutionResult interface
│       └── agent_factory.py      # UPDATED: thin facade over BackendSelector
│
└── chat/
    ├── executor.py               # UPDATED: streaming support via AgentSession
    ├── session.py                # UPDATED: consume streaming responses
    └── streaming.py              # UPDATED: SDK event stream handlers

tests/
├── unit/
│   └── lib/
│       └── backends/             # NEW: unit tests for each backend module
│           ├── test_base.py
│           ├── test_sk_backend.py
│           ├── test_claude_backend.py
│           ├── test_tool_adapters.py
│           ├── test_mcp_bridge.py
│           ├── test_otel_bridge.py
│           └── test_validators.py
│
└── integration/
    └── backends/                 # NEW: integration tests (require API keys)
        ├── test_claude_e2e.py
        └── test_sk_unchanged.py

tests/fixtures/
└── claude_minimal_agent.yaml    # NEW: minimal Claude-native test fixture
```

**Structure Decision**: Single project, Option 1. The new `lib/backends/` package is the primary addition. Refactoring moves SK-specific code from `agent_factory.py` into `sk_backend.py` — no new top-level packages.

---

## Implementation Phases

### Phase 0: SDK Verification Smoke Test (Mandatory Gate)

**This phase is a mandatory gate. No implementation begins until it passes.**

**FR**: FR-011 (verified SDK integration)

The class names, method signatures, and module paths in the spec were researched from SDK v0.1.39 documentation. Phase 0 confirms the actual API before any code is written.

**0.1** — Install `claude-agent-sdk` and write a smoke test that imports and exercises:

- `ClaudeAgentOptions` (or actual class name) — confirm constructor fields and types
- `PermissionMode` — confirm it's an enum or string literal; confirm `"bypassPermissions"` casing exactly
- `@tool` decorator — signature confirmed: `(name, description, input_schema)` (third param is `input_schema`, not `schema_dict`); returns `SdkMcpTool`
- `create_sdk_mcp_server()` — confirm function exists and its signature `(name, tools)` → server config object
- `ResultMessage.structured_output` — confirm field name exists and type
- `ClaudeSDKClient` — confirm class name (vs alternative stateful API shape)
- `query()` — confirm it is an async generator entry point at module level

**0.2** — Multi-turn state verification: Write a 2-turn conversation test using `ClaudeSDKClient`:

```python
async with ClaudeSDKClient(options=options) as client:
    await client.query("What is 2+2?")
    async for msg in client.receive_response():
        ...
    # Second turn — does the client track the session automatically?
    await client.query("What did I just ask you?")
    async for msg in client.receive_response():
        ...
```

Confirm whether `continue_conversation=True` in `ClaudeAgentOptions` is required, or whether `ClaudeSDKClient` tracks the `session_id` internally across calls. Record the verified mechanism.

**0.3** — Output: Document all confirmed class/method names in `research.md` §2 ("API Entry Points"). **Phase 0 complete** — zero `[ASSUMED]` markers remain in `research.md §2`. See `scripts/smoke_test_sdk.py` for full verification output.

**Gate**: If any API detail differs from spec assumptions, update the relevant sections of `plan.md`, `data-model.md`, `quickstart.md`, and `research.md` before proceeding to Phase 1.

---

### Phase 1: Dependency + SDK Setup

**FR**: FR-009a (Node.js check)

**1.1** — Add `claude-agent-sdk` to `pyproject.toml` dependencies:
```
claude-agent-sdk>=0.1.39,<0.2.0
```

**1.2** — Create `src/holodeck/lib/backends/__init__.py` (empty).

**1.3** — Create `src/holodeck/lib/backends/base.py`:
- `ExecutionResult` dataclass (response, tool_calls, tool_results, token_usage, structured_output, num_turns, is_error, error_reason)
- `AgentSession` Protocol (send, send_streaming, close)
- `AgentBackend` Protocol (initialize, invoke_once, create_session, teardown)
- `BackendError`, `BackendInitError`, `BackendSessionError`, `BackendTimeoutError` exceptions

**Test**: Unit tests for `ExecutionResult` construction and field defaults.

---

### Phase 2: Model Extensions

**FR**: FR-002a, FR-012a, FR-032, FR-039

**2.1** — Create `src/holodeck/models/claude_config.py`:
- `AuthProvider(str, Enum)` — api_key, oauth_token, bedrock, vertex, foundry
- `PermissionMode(str, Enum)` — manual, acceptEdits, acceptAll
- `ExtendedThinkingConfig(BaseModel)` — enabled: bool, budget_tokens: int
- `BashConfig(BaseModel)` — enabled: bool, excluded_commands: list[str], allow_unsafe: bool
- `FileSystemConfig(BaseModel)` — read: bool, write: bool, edit: bool
- `SubagentConfig(BaseModel)` — enabled: bool, max_parallel: int
- `ClaudeConfig(BaseModel)` — all fields from data-model.md §1

**2.2** — Update `src/holodeck/models/llm.py`:
- Add `auth_provider: AuthProvider | None = None` field
- Add `model_validator` to warn if `auth_provider` set for non-Anthropic provider

**2.3** — Update `src/holodeck/models/agent.py`:
- Add `embedding_provider: LLMProvider | None = None`
- Add `claude: ClaudeConfig | None = None`
- Preserve `model_config = ConfigDict(extra="forbid")`

**Test**: Unit tests covering:
- Valid `ClaudeConfig` construction with all combinations
- `model_post_init` validation errors for missing `auth_provider` credentials
- Backward compatibility: existing YAML fixtures parse without error

---

### Phase 3: Validation Layer

**FR**: FR-009a, FR-009b, FR-012a, FR-032a, FR-032b

**3.1** — Create `src/holodeck/lib/backends/validators.py`:

```python
def validate_nodejs() -> None: ...
    # shutil.which("node") → ConfigError if absent

def validate_credentials(model: LLMProvider) -> dict[str, str]: ...
    # Check env vars per auth_provider → ConfigError if missing
    # Returns env dict to pass to subprocess

def validate_embedding_provider(agent: Agent) -> None: ...
    # provider==anthropic + vectorstore/hierarchical tools + no embedding_provider → ConfigError

def validate_tool_filtering(agent: Agent) -> None: ...
    # provider==anthropic + tool_filtering set → warning + clear

def validate_working_directory(path: str | None) -> None: ...
    # Check CLAUDE.md collision → warning

def validate_response_format(response_format: dict | None) -> None: ...
    # Check JSON Schema is serialisable → ConfigError if not
```

**Test**: Unit tests for each validator covering pass/fail cases and error messages.

---

### Phase 4: SK Backend (Refactor)

**FR**: FR-010, FR-011, FR-012b

Extract SK-specific code from `agent_factory.py` into the backends package. The existing code is not deleted — it is moved and wrapped to implement the `AgentBackend` Protocol.

**Files modified in this phase**:
- `lib/backends/sk_backend.py` (new)
- `lib/test_runner/agent_factory.py` (updated)
- `lib/chat_history_utils.py` (refactored)
- `models/chat.py` (updated — `history` type change)

**4.1** — Create `src/holodeck/lib/backends/sk_backend.py`:
- Move `AgentFactory` logic into `SKBackend(AgentBackend)`
- Move `AgentThreadRun` logic into `SKSession(AgentSession)`
- Wrap `AgentExecutionResult` → produce `ExecutionResult` (replace `chat_history: ChatHistory` with nothing — history is internal to `SKSession`)
- `SKSession.send()`: calls existing `execute_turn()` logic, converts result
- `SKSession.send_streaming()`: returns full response as single yield (no-op; SK is non-streaming for now)
- Keep all existing SK imports internal to this module — nothing SK-specific leaks to callers
- Move `extract_last_assistant_content()` from `chat_history_utils.py` into `sk_backend.py` as a private function `_extract_response(chat_history)`. `SKBackend.invoke_once()` MUST call `_extract_response(result.chat_history)` internally before constructing `ExecutionResult`. Callers receive a populated `response` field — never an empty string.
- After Phase 4, no module outside `sk_backend.py` imports `ChatHistory` from `semantic_kernel`.

**4.2** — Refactor `lib/chat_history_utils.py`:
- Remove `extract_last_assistant_content()` — moved to `sk_backend.py` as `_extract_response()`.
- Keep `extract_tool_names()` (SK-free) in `chat_history_utils.py` or inline at the call site.
- Any remaining imports of `ChatHistory` in this module must be removed.

**4.3** — Update `models/chat.py`:
- Change `history: ChatHistory` → `history: list[dict[str, Any]] = []`
- SK backend serialises `ChatHistory` messages to `{"role": str, "content": str}` dicts internally before storing in session state. Claude backend leaves `history = []` (conversation state is managed by the SDK subprocess).
- Update anything reading `session.history` to expect `list[dict]` instead of `ChatHistory`.

**4.4** — Update `agent_factory.py`:

> **Preserved constructor contract**: For SK agents, `AgentFactory.__init__()` still creates the SK kernel synchronously (no behaviour change). For Claude agents, `__init__()` stores config only (async init is lazy). The observable difference: SK callers get a ready kernel immediately; Claude callers get a configured-but-uninitialised backend. This is not a "thin facade" — it is a conditional init strategy based on provider.
>
> `AgentFactory` keeps its class name for backward compatibility but internally delegates to `BackendSelector.create()`.

**4.5** — Lazy-init pattern for `ClaudeBackend`:

> `ClaudeBackend.__init__()` stores config only — no I/O, no subprocess spawn. `initialize()` is called lazily on the first `invoke_once()` or `create_session()` call — identical to the existing `AgentFactory._ensure_tools_initialized()` pattern. The lifecycle diagram `[Created] → initialize() → [Ready]` in `data-model.md` applies only to explicit call sites; internal callers trigger init automatically.

**Test**: Run existing unit tests — they MUST pass without modification. The refactor is behaviour-preserving.

---

> **Integration Gate (Phase 4 → 5)**: Before starting Phase 5, run a full `holodeck test` workflow against a real non-Anthropic agent with at least one MCP tool and one vectorstore tool. All existing tests must pass. This prevents asyncio lifecycle regressions from the SK refactor from being buried under Claude backend code.

---

### Phase 5: Tool Adapters

**FR**: FR-003, FR-004, FR-023

**5.1** — Create `src/holodeck/lib/backends/tool_adapters.py`:

```python
class VectorStoreToolAdapter:
    """Wraps VectorStoreTool.search() as a Claude SDK @tool."""

    def __init__(self, config: VectorstoreTool, instance: VectorStoreTool): ...
    def to_sdk_tool(self) -> SdkMcpTool: ...
    # Uses @tool decorator with tool_name from config.name

class HierarchicalDocToolAdapter:
    """Wraps HierarchicalDocumentTool.search() as a Claude SDK @tool."""

    def __init__(self, config: HierarchicalDocumentToolConfig, instance): ...
    def to_sdk_tool(self) -> SdkMcpTool: ...

def build_holodeck_sdk_server(
    adapters: list[VectorStoreToolAdapter | HierarchicalDocToolAdapter]
) -> tuple[McpSdkServerConfig, list[str]]:
    """Create in-process MCP server; return (server_config, allowed_tool_names)."""
```

**5.2** — Tool adapter loop: Use a factory function — not a bare `for` loop — when creating tool functions. Python closures capture by reference; a factory function is mandatory to avoid all adapters sharing the last loop variable:

```python
def make_search_fn(t: VectorStoreTool, name: str) -> SdkMcpTool:
    @tool(name, f"Search {t.config.name}", {"query": str})
    async def search_fn(args: dict) -> dict:
        result = await t.search(args["query"])
        return {"content": [{"type": "text", "text": result}]}
    return search_fn

sdk_tools = [make_search_fn(t, f"{t.config.name}_search") for t in vectorstore_tools]
```

**5.3** — Tool instance initialisation (embeddings):
- Vectorstore and hierarchical document tools still use SK embedding services.
- `ClaudeBackend.initialize()` calls existing `AgentFactory._register_embedding_service()` using `agent.embedding_provider` credentials — specifically `agent.embedding_provider.provider.value` (NOT `agent.model.provider`). For Claude agents, `model.provider` is `"anthropic"` — passing this to the embedding tool would select wrong vector dimensions.
- Tools are initialised in the parent process before the Claude subprocess starts.

**Test**: Unit tests mock `tool.search()` — verify adapter produces correct SDK tool format and `allowed_tools` entry names.

---

### Phase 6: MCP Bridge

**FR**: FR-005, FR-035

**6.1** — Create `src/holodeck/lib/backends/mcp_bridge.py`:

```python
def build_claude_mcp_configs(mcp_tools: list[MCPTool]) -> dict[str, McpStdioServerConfig]:
    """Translate HoloDeck MCPTool configs → Claude SDK MCP server specs.

    Stdio transport only; non-stdio tools emit a warning and are skipped.
    """
```

- Resolve env vars using existing `_resolve_mcp_env()` logic from `mcp/factory.py`
- Output format: `{"server_name": {"type": "stdio", "command": "...", "args": [...], "env": {...}}}`

**Test**: Unit tests for config translation, env var resolution, and non-stdio warning.

---

### Phase 7: OTel Bridge

**FR**: FR-036, FR-037, FR-038

**7.1** — Create `src/holodeck/lib/backends/otel_bridge.py`:

```python
def translate_observability(config: ObservabilityConfig) -> dict[str, str]:
    """Map agent.yaml observability: section to Claude Code env vars.

    Returns env dict to merge into ClaudeAgentOptions.env.
    """
    # CLAUDE_CODE_ENABLE_TELEMETRY, OTEL_METRICS_EXPORTER,
    # OTEL_LOGS_EXPORTER, OTEL_EXPORTER_OTLP_PROTOCOL,
    # OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_METRIC_EXPORT_INTERVAL,
    # OTEL_LOGS_EXPORT_INTERVAL, OTEL_LOG_USER_PROMPTS, OTEL_LOG_TOOL_DETAILS
```

**7.2** — Unmapped field warnings: `otel_bridge.py` must emit named startup warnings for each `ObservabilityConfig` field that has no Claude Code env var equivalent:

- `exporters.azure_monitor` → no mapping
- `exporters.prometheus` → no mapping
- `traces.redaction_patterns` → no mapping
- `traces.sample_rate` → no mapping
- `logs.filter_namespaces` → no mapping

Warning format: `"The following observability settings are not supported by the Claude-native backend and will be ignored: {field_list}. Use a non-Anthropic provider or remove these fields."`

This is a named warning, not an error — the session proceeds.

**Test**: Unit tests for each mapping, default-off privacy controls, and unmapped-field warning emission.

---

### Phase 8: Claude Backend (Core)

**FR**: FR-001, FR-006, FR-007, FR-009, FR-009b, FR-011, FR-013, FR-015, FR-016, FR-039

**8.1** — Create `src/holodeck/lib/backends/claude_backend.py`:

#### `ClaudeSession(AgentSession)`
- Wraps `ClaudeSDKClient` for stateful chat
- `send(message)` → collects full response from `receive_response()` iterator
- `send_streaming(message)` → yields text chunks from `StreamEvent` / `AssistantMessage` blocks
- `close()` → `client.disconnect()`
- Handles `ProcessError` (subprocess crash) → raises `BackendSessionError`

#### `ClaudeBackend(AgentBackend)`
- `__init__(agent, tools_instances)` — stores config only (no I/O, no subprocess)
- `initialize()` (called lazily on first `invoke_once()` or `create_session()`):
  1. `validate_nodejs()`
  2. `validate_credentials(agent.model)` → stores env dict
  3. `validate_embedding_provider(agent)` (if vectorstore tools)
  4. `validate_tool_filtering(agent)` (warn + clear if needed)
  5. Initialise tool instances (embedding services via SK, using `embedding_provider`)
  6. Build tool adapters + SDK server
  7. Build MCP bridge configs
  8. Build `ClaudeAgentOptions` via `build_options()`
  9. `validate_working_directory(claude.working_directory)`
- `invoke_once(message, context=None)`:
  1. Uses `claude_agent_sdk.query(prompt=message, options=self._options)`
  2. Collects `AssistantMessage` blocks (text, tool_use, tool_result)
  3. Extracts `ResultMessage` for token usage + structured output
  4. Validates structured output against schema if `response_format` configured
  5. Returns `ExecutionResult`
  6. Detects `max_turns` exceeded via `ResultMessage.num_turns >= max_turns`
  7. Session-level retry on `ProcessError` (spawn failure): 3 attempts, exponential backoff
- `create_session()` → returns `ClaudeSession(self._options)`
- `teardown()` → cleanup (no persistent resources for Claude backend)

**Permission mode for test runs**: Override to `"bypassPermissions"` during `holodeck test` unless `permission_mode: manual` is explicitly set. However, when `bash.enabled: true` or `file_system.write: true` is configured AND `--allow-side-effects` is NOT passed to `holodeck test`, emit a startup warning and force-disable bash and file system access for the test run. Users who want side effects in test runs must pass `--allow-side-effects` explicitly.

**8.2** — Create `src/holodeck/lib/backends/selector.py`:

```python
class BackendSelector:
    @staticmethod
    async def create(
        agent: Agent,
        tool_instances: list,
        mode: Literal["test", "chat"] = "test",
    ) -> AgentBackend:
        if agent.model.provider == ProviderEnum.ANTHROPIC:
            backend = ClaudeBackend(agent, tool_instances, mode=mode)
        else:
            backend = SKBackend(agent, tool_instances, mode=mode)
        await backend.initialize()
        return backend
```

**Test**: Unit tests with mocked SDK calls covering:
- Successful single-turn invocation
- Tool call extraction from AssistantMessage
- ResultMessage processing (token usage, structured output)
- max_turns exceeded detection
- Subprocess crash → BackendSessionError
- All auth provider env var mappings
- Structured output schema validation pass/fail

---

### Phase 9: Test Runner Updates

**FR**: FR-008, FR-012b, FR-013

**9.1** — Update `lib/test_runner/executor.py`:
- Replace `AgentFactory(config).create_thread_run()` with `BackendSelector.select(agent, tools)`
- Replace `AgentExecutionResult` with `ExecutionResult` (from `backends.base`)
- Remove all `ChatHistory`, `FunctionCallContent`, `FunctionResultContent` imports
- Skip `_create_agent_factory()` when `_backend` is set (BackendSelector handles tool init). Note: remove `_create_agent_factory()` entirely after chat migration in Phase 10.
- `max_turns` exceeded: check `result.is_error and result.error_reason == "max_turns limit reached"` → mark test as failed (not evaluation error)
- Subprocess crash: `result.is_error and "subprocess terminated"` → mark as execution error, continue test suite

**9.2** — Update `lib/test_runner/agent_factory.py`:
- Thin facade: `AgentFactory` delegates to `BackendSelector`
- Preserves public API for any code that imports `AgentFactory` directly
- `AgentExecutionResult` kept as-is for backward compat (separate dataclass, not re-exported from `ExecutionResult`); add deprecation comment directing new code to use `ExecutionResult`

**9.3** — Add `--allow-side-effects` flag to the `holodeck test` CLI command:
- When absent (default): `ClaudeBackend` disables bash and file_system access for Anthropic-provider test runs (even if `bash.enabled: true` or `file_system.write: true` is in the YAML).
- When present: configured bash and file_system settings are respected, and a startup warning is emitted noting that the test run may have system side effects.
- This flag has no effect for non-Anthropic backends.

**Test**: Run full existing test suite; zero regressions on non-Anthropic agents.

---

### Phase 10: Streaming Chat Refactor

**FR**: FR-007, FR-012b

**10.1** — Update `src/holodeck/chat/executor.py`:
- Add `execute_turn_streaming(message: str) -> AsyncIterator[str]` method
- Internally calls `session.send_streaming(message)` → yields text chunks
- Falls back to `session.send(message)` → yield complete response for SK backend
- Replace `get_history() -> ChatHistory` with two methods:
  - `get_message_count() -> int` — satisfiable by both backends
  - `get_history() -> list[dict[str, Any]]` — returns empty list for Claude sessions (state lives in the subprocess); returns serialised SK history for SK sessions

**10.2** — Update `src/holodeck/chat/session.py`:
- `process_message()` uses `execute_turn_streaming()` when available
- Updates token usage from `ExecutionResult` after stream completes
- Any caller reading `session.history` must expect `list[dict[str, Any]]` (not `ChatHistory`)

**10.3** — Update `src/holodeck/chat/streaming.py`:
- Add handlers for SDK message types where needed

**Test**: Unit tests with mocked `ClaudeSession.send_streaming()` verifying chunks arrive progressively.

---

### Phase 11: Tests

**Coverage target**: 80%+ for all new modules.

**Unit tests** (fast, no API calls):
- `tests/unit/lib/backends/test_base.py` — ExecutionResult construction
- `tests/unit/lib/backends/test_validators.py` — all validators pass/fail
- `tests/unit/lib/backends/test_tool_adapters.py` — adapter output format
- `tests/unit/lib/backends/test_mcp_bridge.py` — config translation
- `tests/unit/lib/backends/test_otel_bridge.py` — env var mapping
- `tests/unit/lib/backends/test_claude_backend.py` — mocked SDK interactions
- `tests/unit/lib/backends/test_sk_backend.py` — existing SK behaviour preserved
- `tests/unit/models/test_claude_config.py` — model validation
- `tests/unit/models/test_agent_extensions.py` — new agent fields

**Integration tests** (require API key, marked `@pytest.mark.integration`):
- `tests/integration/backends/test_claude_e2e.py` — end-to-end chat + test run
- `tests/integration/backends/test_sk_unchanged.py` — non-Anthropic agents unchanged

**Test fixtures**:
- `tests/fixtures/claude_minimal_agent.yaml` — minimal Anthropic agent (no tools)
- `tests/fixtures/claude_vectorstore_agent.yaml` — Anthropic agent with vectorstore tool

---

### Phase 12: Code Quality

Run after each phase:

```bash
make format        # Black + Ruff formatting
make lint-fix      # Auto-fix linting issues
make type-check    # MyPy strict mode (all new code must pass)
make test-unit     # Run unit tests in parallel (-n auto)
make security      # Bandit + Safety + detect-secrets
```

---

## Implementation Order (Dependency Graph)

```
Phase 0 (SDK verification — mandatory gate)
    │
Phase 1 (SDK setup)
    │
Phase 2 (models)
    │
Phase 3 (validators)
    │
Phase 4 (SK backend refactor) ← must be stable + tested first
    │
    ├── Phase 5 (tool adapters)     ─┐
    ├── Phase 6 (MCP bridge)         ├── parallel after Phase 4 confirmed green
    └── Phase 7 (OTel bridge)       ─┘
                                       │
                                   Phase 8 (Claude backend)
                                       │
                                   Phase 9 (test runner + --allow-side-effects)
                                       │
                                   Phase 10 (streaming chat + get_history fix)
                                       │
                                   Phase 11 (tests)
                                       │
                                   Phase 12 (quality)
```

**Critical path**: **0 → 1 → 2 → 3 → 4 → [5+6+7] → 8 → 9 → 10 → 11 → 12**

Phases 5, 6, 7 can progress in parallel after Phase 4 is confirmed green (integration gate passed).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Claude Agent SDK API changes (alpha) | Medium | High | Pin `<0.2.0`; wrap SDK calls in adapters to isolate changes |
| SDK API differs from assumed names (Phase 0) | Medium | High | Phase 0 smoke test gates all implementation; fixes applied before Phase 1 |
| SK refactor breaks existing tests | Low | High | Keep SK backend behaviour identical; run full test suite after Phase 4 |
| Node.js absent on CI | Medium | Medium | Add `node --version` to CI prerequisites; validate in Phase 3 |
| Tool adapter IPC overhead | Low | Medium | Measure latency in integration tests; SC-004 (3s streaming start) |
| Structured output schema incompatibility | Low | Low | Validate schema at startup (Phase 3); clear error before subprocess spawn |
| CLAUDE.md collision in working directory | Medium | Low | Startup warning (not error); user can ignore or adjust path |
| Unintended side effects in test runs | Low | Medium | `--allow-side-effects` gate disables bash/file_system by default |

---

## Deferred (Out of Scope)

Per spec Out of Scope section:
- `can_use_tool` callback (Python callables across subprocess boundary — follow-up spec)
- Pre/post execution hooks (same subprocess boundary constraint)
- Streaming for non-Anthropic backends (SK-based streaming is a separate spec)
- Tool filtering for Claude-native backend (SK-dependent; follow-up spec)
- SSE/WebSocket/HTTP MCP transports
- `holodeck deploy` integration
