# Research: Native Claude Agent SDK Integration

**Feature**: 021-claude-agent-sdk
**Date**: 2026-02-20
**Status**: Complete — all NEEDS CLARIFICATION resolved

---

## 1. SDK Package Identity

**Decision**: Use `claude-agent-sdk` (v0.1.39, released 2026-02-19)

**Rationale**: The package was renamed from the deprecated `claude-code-sdk`. PyPI canonical name is `claude-agent-sdk`. Python module import is `claude_agent_sdk`.

**Installation**:
```
uv add claude-agent-sdk
```

**Requirements**: Python 3.10+, Node.js on PATH (subprocess host runtime).

**Alternatives considered**: `claude-code-sdk` (deprecated), direct `anthropic` SDK (no subprocess/tool model).

---

## 2. API Entry Points

> **Note**: All names in this section were verified against SDK v0.1.39 via the Phase 0 smoke test (`scripts/smoke_test_sdk.py`). Zero `[ASSUMED]` markers remain.

**Decision**: Use `query()` for test runs (stateless), `ClaudeSDKClient` for chat sessions (stateful).

**Rationale**:
- `query()` creates a fresh session per call — ideal for isolated test case execution where each invocation must not bleed conversation state from the previous test.
- `ClaudeSDKClient` maintains session state across calls via `session_id` — required for multi-turn interactive chat where the agent must remember prior messages.

```python
# Test runs — stateless
async for message in query(prompt=user_input, options=options):
    ...

# Chat sessions — stateful
async with ClaudeSDKClient(options=options) as client:
    await client.query(user_input)
    async for message in client.receive_response():
        ...
```

**Confirmed names (Phase 0 verified — `scripts/smoke_test_sdk.py`):**

| Assumed Name | Confirmed Name | Notes |
|---|---|---|
| `ClaudeSDKClient` | `ClaudeSDKClient` | ✓ Confirmed |
| `ClaudeAgentOptions` | `ClaudeAgentOptions` | ✓ Confirmed. Full field list in §2a below. |
| `PermissionMode` | `PermissionMode` | ✓ Confirmed. **`Literal` type alias, NOT an Enum class.** Use string literals: `"default"`, `"acceptEdits"`, `"plan"`, `"bypassPermissions"` |
| `create_sdk_mcp_server()` | `create_sdk_mcp_server()` | ✓ Confirmed. Signature: `(name: str, version: str = '1.0.0', tools: list[SdkMcpTool] \| None = None) -> McpSdkServerConfig` |
| `ResultMessage.structured_output` | `ResultMessage.structured_output` | ✓ Confirmed. Type: `Any` |
| `@tool(name, description, schema_dict)` | `@tool(name, description, input_schema)` | **CORRECTED**: third param is `input_schema`, not `schema_dict`. Returns `SdkMcpTool`, not the function. |

**Multi-turn state (Phase 0 verified):**

Multi-turn state is **OPT-IN** — it is NOT automatic. Each `query()` call starts a fresh session unless `continue_conversation=True` is explicitly passed to `ClaudeAgentOptions`. To resume a specific prior session, additionally pass `resume=session_id`.

```python
# Turn 1 — fresh session
opts_1 = ClaudeAgentOptions(permission_mode="bypassPermissions", max_turns=1)
async for event in query(prompt="What is 2+2?", options=opts_1):
    if isinstance(event, ResultMessage):
        session_id = event.session_id

# Turn 2 — continues previous session (OPT-IN via continue_conversation + resume)
opts_2 = ClaudeAgentOptions(
    permission_mode="bypassPermissions",
    max_turns=1,
    continue_conversation=True,  # Required — state is NOT automatic
    resume=session_id,           # Resume specific session by ID
)
async for event in query(prompt="And what is that times 3?", options=opts_2):
    ...
```

**Phase 8 `ClaudeSession` implementation implication**: The session wrapper must pass `continue_conversation=True` and track `session_id` across turns. Do not assume automatic context carry-over.

### §2a — `ClaudeAgentOptions` Key Fields

Full signature verified via smoke test. Key fields for HoloDeck use:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `permission_mode` | `Literal['default','acceptEdits','plan','bypassPermissions'] \| None` | `None` | Autonomy level |
| `continue_conversation` | `bool` | `False` | **Set True for multi-turn** |
| `resume` | `str \| None` | `None` | Resume by session_id |
| `mcp_servers` | `dict[str, McpStdioServerConfig \| McpSSEServerConfig \| McpHttpServerConfig \| McpSdkServerConfig] \| str \| Path` | `{}` | MCP server registrations |
| `allowed_tools` | `list[str]` | `[]` | Tool allowlist |
| `system_prompt` | `str \| SystemPromptPreset \| None` | `None` | Override system prompt |
| `model` | `str \| None` | `None` | Model override |
| `env` | `dict[str, str]` | `{}` | Subprocess env vars (auth, etc.) |
| `cwd` | `str \| Path \| None` | `None` | Working directory |
| `max_turns` | `int \| None` | `None` | Turn limit |
| `include_partial_messages` | `bool` | `False` | Enable `StreamEvent` token deltas |
| `output_format` | `dict[str, Any] \| None` | `None` | Structured output schema |

**Alternatives considered**: Using `ClaudeSDKClient` for all modes (adds unnecessary session overhead for tests), using raw `anthropic` SDK (loses tool-calling, MCP, permission model).

---

## 3. Permission Mode Mapping

**Decision**: Map HoloDeck YAML permission levels to SDK `PermissionMode` literals as follows:

| HoloDeck YAML (`permission_mode`) | Claude SDK `PermissionMode` | Behaviour |
|---|---|---|
| `manual` | `"default"` | Standard approval prompts for all actions |
| `acceptEdits` | `"acceptEdits"` | Auto-approve file edits; prompt for others |
| `acceptAll` | `"bypassPermissions"` | No prompts; fully autonomous |

**Rationale**: The SDK's `"plan"` mode is a preview-only mode not relevant to HoloDeck's autonomy model. `"bypassPermissions"` is the correct mapping for `acceptAll` — it skips all permission checks for permitted tools.

**For test runs**: Override to `"bypassPermissions"` automatically during `holodeck test` unless `permission_mode: manual` is explicitly configured. However, when `bash.enabled: true` or `file_system.write: true` is configured, bash and file system are force-disabled unless `--allow-side-effects` is explicitly passed to `holodeck test`. This prevents unattended side effects in automated test environments. The `--allow-side-effects` flag is added to the `holodeck test` CLI command in Phase 9.

---

## 4. Authentication Architecture

**Decision**: Map `auth_provider` to environment variables injected into the subprocess environment dict (`ClaudeAgentOptions.env`).

| `auth_provider` value | Required env vars | SDK behaviour |
|---|---|---|
| `api_key` (default) | `ANTHROPIC_API_KEY` | Direct Anthropic API |
| `oauth_token` | `CLAUDE_CODE_OAUTH_TOKEN` | Anthropic OAuth |
| `bedrock` | AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` etc.) | `CLAUDE_CODE_USE_BEDROCK=1` |
| `vertex` | GCP credentials (`GOOGLE_APPLICATION_CREDENTIALS` etc.) | `CLAUDE_CODE_USE_VERTEX=1` |
| `foundry` | Azure credentials (`AZURE_TENANT_ID` etc.) | `CLAUDE_CODE_USE_FOUNDRY=1` |

HoloDeck validates credential presence before spawning the subprocess. Missing credentials raise a `ConfigError` with a specific message naming the missing variable.

**Rationale**: All auth is env-var driven — no SDK-level auth objects exist. HoloDeck reads from `os.environ` + `.env` files (existing `EnvLoader` behaviour) and passes the resolved vars into `ClaudeAgentOptions.env`.

---

## 5. Tool Registration for Vectorstore/Hierarchical Document Tools

**Decision**: Wrap HoloDeck's tool `search()` methods using the Claude Agent SDK `@tool` decorator, bundle them into an in-process MCP server via `create_sdk_mcp_server()`, and register that server in `ClaudeAgentOptions.mcp_servers`.

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

# Factory function required — Python closures capture by reference, not value.
# A bare for-loop creates all tool functions sharing the last loop variable.
def make_search_fn(t: VectorStoreTool, name: str) -> SdkMcpTool:
    @tool(name, f"Search {t.config.name}", {"query": str})
    async def search_fn(args: dict) -> dict:
        result = await t.search(args["query"])
        return {"content": [{"type": "text", "text": result}]}
    return search_fn

sdk_tools = [make_search_fn(t, f"{t.config.name}_search") for t in vectorstore_tools]

sdk_server = create_sdk_mcp_server(
    name="holodeck_tools",
    tools=sdk_tools,
)

options = ClaudeAgentOptions(
    mcp_servers={"holodeck_tools": sdk_server},
    allowed_tools=[f"mcp__holodeck_tools__{t.config.name}_search" for t in vectorstore_tools],
)
```

**Rationale**: The SDK's subprocess communicates tool invocations back to the parent process transparently through the SDK's IPC. HoloDeck does not need to implement a custom IPC server. The `@tool` + `create_sdk_mcp_server()` pattern is the SDK's canonical in-process tool mechanism.

**Tool name convention**: `mcp__<server_name>__<tool_name>` — must match entries in `allowed_tools`.

**Alternatives considered**: Using the SK `@kernel_function` decorator (incompatible with Claude subprocess); building a custom MCP stdio server process (unnecessary complexity when SDK provides in-process path).

---

## 6. MCP Tool Integration (External MCP Servers)

**Decision**: Translate HoloDeck's `MCPTool` config (stdio transport) to `McpStdioServerConfig` dicts for `ClaudeAgentOptions.mcp_servers`. The SDK subprocess manages MCP server processes directly.

```python
# HoloDeck MCPTool → SDK MCP server config
mcp_servers["my_mcp"] = McpStdioServerConfig(
    type="stdio",
    command=mcp_tool.command,
    args=mcp_tool.args or [],
    env=resolve_env(mcp_tool),
)
```

HoloDeck's existing `MCPStdioPlugin` (SK) is NOT used for Anthropic-provider agents. The SK MCP plugin factory produces SK plugins for the Kernel — the Claude subprocess has its own native MCP client that accepts the `McpStdioServerConfig` dict format.

**Routing**: `lib/backends/selector.py` routes MCP config translation based on backend:
- SK backend → existing `mcp/factory.py` → `MCPStdioPlugin`
- Claude backend → new `mcp_bridge.py` → `McpStdioServerConfig`

---

## 7. Streaming Message Types

**Decision**: Consume `AssistantMessage.content` blocks for streaming text output; consume `ResultMessage` for final token usage and structured output.

Key message types from the SDK's async iterator:

```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]   # TextBlock, ToolUseBlock, ThinkingBlock
    model: str

@dataclass
class ResultMessage:
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None
    usage: dict[str, Any] | None
    result: str | None
    structured_output: Any       # Set when output_format configured

@dataclass
class StreamEvent:               # Enabled via include_partial_messages=True
    uuid: str
    event: dict[str, Any]        # Raw SDK stream event for token-level streaming
```

For streaming chat, use `include_partial_messages=True` to get `StreamEvent` messages with partial token deltas.

For tool call extraction: scan `AssistantMessage.content` for `ToolUseBlock` (call) and `ToolResultBlock` (result).

---

## 8. Retry Behaviour

**Decision**: The Claude Agent SDK subprocess handles API-level retries (rate limit, 5xx, timeout) internally. HoloDeck adds retry logic only at the **session-spawn level** — when the subprocess itself fails to start or crashes unexpectedly.

**Rationale**: FR-013 explicitly warns against double-retry stacking. If the SDK already retries API calls 3 times internally, adding HoloDeck's own 3 retries would produce up to 9 effective attempts.

**Implementation**: Verify SDK retry behaviour in integration tests. If confirmed, HoloDeck's `invoke_with_retry` for Claude backend covers only:
- `CLINotFoundError` — not retryable (raises immediately with guidance)
- `ProcessError` on spawn — retryable (up to 3 session-level retries, exponential backoff)
- `RateLimitError` from `anthropic` SDK (if SDK does NOT retry) — retryable

---

## 9. Node.js Prerequisite

**Decision**: Check `shutil.which("node")` at agent startup before spawning subprocess. Raise `ConfigError` with install instructions if absent.

```python
import shutil
from holodeck.lib.errors import ConfigError

def validate_nodejs() -> None:
    if shutil.which("node") is None:
        raise ConfigError(
            "Node.js is required by the Claude Agent SDK but was not found on PATH.\n"
            "Install Node.js: https://nodejs.org/en/download/"
        )
```

**Rationale**: The SDK bundles the `claude` CLI — no separate install needed. But Node.js itself must be present as the runtime host. A missing Node.js produces a cryptic subprocess spawn error without this check.

---

## 10. Structured Output

**Decision**: Translate HoloDeck's `response_format` dict to the SDK's `OutputFormat` dataclass:

```python
from claude_agent_sdk import OutputFormat

output_format = OutputFormat(
    type="json_schema",
    schema=agent.response_format,  # Already a JSON Schema dict
)
```

The existing SK wrapper (`_wrap_response_format`) produces `{"type": "json_schema", "json_schema": {...}, "strict": true}` — a different envelope format. This MUST NOT be used for the Claude-native path.

Structured output is returned in `ResultMessage.structured_output`. Validation against the configured schema is performed before returning to downstream consumers. The serialised text form (JSON string) is extracted for NLP/G-Eval evaluation metrics.

---

## 11. Embedding Provider Architecture

**Decision**: Keep SK embedding services for vectorstore and hierarchical document tool index generation, even when the agent conversation loop uses the Claude backend. Add a top-level `embedding_provider: LLMProvider` field to `Agent`.

**Rationale**: The Claude SDK subprocess does not generate embeddings. Vectorstore index generation must happen in-process (parent) before the subprocess starts. The existing `VectorStoreTool` and `HierarchicalDocumentTool` already use SK embedding services. Adding `embedding_provider` lets Anthropic-provider agents use, e.g., `provider: openai, name: text-embedding-3-small` for embeddings independently of the chat provider.

**Validation**: At startup, if `provider: anthropic` AND any vectorstore/hierarchical-document tool is configured AND `embedding_provider` is absent → raise `ConfigError` before any subprocess is spawned.

---

## 12. Backend Abstraction Architecture

**Decision**: Create `src/holodeck/lib/backends/` package with a clean `AgentBackend` Protocol and `ExecutionResult` dataclass. Both SK and Claude-native backends implement this Protocol. All downstream consumers (test runner, chat executor) interact only with the interface.

**Rationale** (from user directive): Decouple HoloDeck's test runner and chat layer from SK-specific types (`ChatHistory`, `ChatMessageContent`, `FunctionCallContent`). This makes the Claude-native backend a first-class peer rather than a bolt-on, and positions HoloDeck to remove SK from the agent loop entirely in the future without touching the test or chat layers.

**Key interface types**:
- `ExecutionResult` — provider-agnostic single-turn result
- `AgentSession` — stateful multi-turn session (Protocol)
- `AgentBackend` — backend factory Protocol

See `contracts/execution-result.md` and `contracts/agent-backend.md` for full specifications.

---

## 13. CLAUDE.md Collision Risk

**Decision**: At startup, warn if `working_directory` is set AND `<working_directory>/CLAUDE.md` exists AND its content appears to be a developer/project-level CLAUDE.md (heuristic: contains `# CLAUDE.md` header or `CLAUDE.md` in first 5 lines).

**Warning message**: Direct users to either (a) use a more specific working directory, or (b) provide agent instructions via `agent.yaml`'s `instructions:` field instead.

This is a startup warning, not an error — the session proceeds unless the user overrides.

---

## 14. Tool Filtering (Disabled for Claude Backend)

**Decision**: `tool_filtering` configuration is silently skipped with a user-visible warning when `provider: anthropic` is set. No error is raised.

**Rationale**: `tool_filtering` uses SK's in-process kernel for semantic search over tool descriptions — incompatible with the subprocess model. A follow-up spec will address this for the Claude-native backend.

**Warning**: "tool_filtering is not supported for provider: anthropic and will be skipped. Configure allowed_tools instead."

---

## 15. OTel Bridge — Unsupported Fields

**Decision**: `otel_bridge.py` maps supported `ObservabilityConfig` fields to Claude Code environment variables. Fields that have no equivalent Claude Code env var are silently skipped with named startup warnings.

**Unsupported fields** (emit named warnings, do not raise errors):

| `ObservabilityConfig` field | Reason |
|---|---|
| `exporters.azure_monitor` | No Claude Code env var equivalent |
| `exporters.prometheus` | No Claude Code env var equivalent |
| `traces.redaction_patterns` | No Claude Code env var equivalent |
| `traces.sample_rate` | No Claude Code env var equivalent |
| `logs.filter_namespaces` | No Claude Code env var equivalent |

**Warning format**:
```
"The following observability settings are not supported by the Claude-native backend
and will be ignored: {field_list}. Use a non-Anthropic provider or remove these fields."
```

This is a named warning, not an error — the session proceeds normally. The listed fields are collected at startup before subprocess spawn and emitted as a single consolidated warning (not one per field).

**Rationale**: Silently ignoring these fields without any warning would create a confusing debugging experience for users who configure observability expecting it to take effect. A named warning makes the limitation explicit without blocking the agent from running.

---

## 16. Chat Layer Decoupling Architecture

**Decision**: Decouple `chat/executor.py` from `AgentFactory`/`AgentThreadRun` using the same
`BackendSelector` → `AgentBackend` → `AgentSession` pattern applied to the test runner in Phase 9.
Full cutover (no dual-path with legacy factory).

**Rationale**: The chat executor currently imports `AgentFactory` and `AgentThreadRun` directly,
making `holodeck chat` impossible for Claude-native agents. The test runner was decoupled in Phase 9
using a dual-path approach (backend + legacy factory). The chat executor uses a full cutover instead
because no external consumer injects `AgentFactory` into it — only the CLI and `ChatSessionManager`
call it.

**Key architectural differences from test runner decoupling**:

| Aspect | Test Runner (Phase 9) | Chat Executor (Phase 10) |
|---|---|---|
| Entry point | `AgentBackend.invoke_once()` | `AgentBackend.create_session()` → `AgentSession.send()` |
| State model | Stateless (fresh per test) | Stateful (multi-turn, history preserved) |
| Dual-path | Yes (backend + legacy factory) | No (full cutover) |
| Mode | `mode="test"` | `mode="chat"` |
| Streaming | N/A | `AgentSession.send_streaming()` (net-new) |
| Permission mode | Forced `bypassPermissions` | Respects YAML `permission_mode` |

**History tracking**: The executor maintains a local `list[dict]` of `{role, content}` dicts.
This avoids extending the `AgentSession` protocol. Both backends handle conversation state
internally (SK via `ChatHistory`, Claude via subprocess `session_id`) — the executor's local
history is for display/export only.

**Streaming architecture**: `AgentSession.send_streaming()` is already defined in the protocol
(`backends/base.py`). `ClaudeSession` yields real text chunks from the subprocess event stream.
`SKSession` falls back to yielding the complete response as a single chunk. The chat CLI writes
chunks directly to stdout for progressive display.

**Alternatives considered**: Dual-path approach matching test runner (rejected — unnecessary
complexity since no external consumer injects `AgentFactory` into the chat executor).
