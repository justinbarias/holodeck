# Quickstart: Native Claude Agent SDK Integration

**Feature**: 021-claude-agent-sdk
**Audience**: HoloDeck developers implementing the feature

---

## Prerequisites

1. Node.js installed on PATH: `node --version`
2. `ANTHROPIC_API_KEY` set in environment or `.env` file
3. SDK installed: `uv add claude-agent-sdk`

---

## Architecture Overview

```
holodeck chat/test
       │
       ▼
BackendSelector (lib/backends/selector.py)
       │
       ├── provider: anthropic → ClaudeBackend
       │        │
       │        ├── invoke_once() → uses claude_agent_sdk.query()
       │        └── create_session() → uses claude_agent_sdk.ClaudeSDKClient
       │
       └── provider: openai/azure/ollama → SKBackend (unchanged)

Both backends produce: ExecutionResult
Consumed by: TestExecutor, ChatSessionManager
```

---

## New Package: `lib/backends/`

```
src/holodeck/lib/backends/
├── __init__.py
├── base.py              # ExecutionResult, AgentSession, AgentBackend protocols
├── selector.py          # BackendSelector — routes by provider
├── sk_backend.py        # SK backend (refactored from agent_factory.py)
├── claude_backend.py    # Claude Agent SDK backend
├── tool_adapters.py     # VectorStore/HierarchicalDoc → Claude @tool
├── mcp_bridge.py        # MCPTool config → McpStdioServerConfig
├── otel_bridge.py       # ObservabilityConfig → Claude Code env vars
└── validators.py        # Node.js check, credential check, embedding check
```

---

## Key Implementation Patterns

### 0. Phase 0: SDK Verification Smoke Test (Run Before Any Implementation)

```python
# smoke_test_sdk.py — run this before writing any backend code
# Goal: confirm actual API vs. assumed API; update research.md §2 with results

from claude_agent_sdk import (
    ClaudeAgentOptions,   # ✓ Confirmed class name
    PermissionMode,        # ✓ Confirmed — Literal type alias, NOT an Enum. Use string literals.
    query,                 # ✓ Confirmed — keyword-only arg: query(prompt=..., options=...)
    ResultMessage,         # ✓ Confirmed — fields: structured_output, num_turns, usage, session_id
    AssistantMessage,      # ✓ Confirmed — field: content (list[TextBlock|ThinkingBlock|ToolUseBlock|ToolResultBlock])
)

# Verify tool decorator and server factory
from claude_agent_sdk import tool, create_sdk_mcp_server  # ✓ Confirmed

@tool("test_tool", "A test tool", {"query": str})  # Third param: input_schema (NOT schema_dict)
async def my_tool(args: dict) -> dict:
    return {"content": [{"type": "text", "text": "ok"}]}

server = create_sdk_mcp_server(name="test_server", tools=[my_tool])

# Multi-turn state is OPT-IN — NOT automatic.
# Each query() call starts a fresh session unless continue_conversation=True is passed.
from claude_agent_sdk import ClaudeSDKClient  # ✓ Confirmed class name

async def verify_multiturn():
    # Turn 1 — fresh session
    opts_1 = ClaudeAgentOptions(permission_mode="bypassPermissions", max_turns=1)
    async for event in query(prompt="What is 2+2?", options=opts_1):
        if isinstance(event, ResultMessage):
            session_id = event.session_id

    # Turn 2 — OPT-IN continuation (continue_conversation=True + resume required)
    opts_2 = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        max_turns=1,
        continue_conversation=True,  # Required — state is NOT automatic
        resume=session_id,
    )
    async for event in query(prompt="What did I just ask you?", options=opts_2):
        ...

# Phase 0 complete: all [ASSUMED] markers replaced. See research.md §2 for full verification table.
```

### 1. Registering HoloDeck Tools with Claude SDK

```python
# lib/backends/tool_adapters.py

from claude_agent_sdk import tool, create_sdk_mcp_server

def build_sdk_tools_server(
    vectorstore_tools: list[VectorStoreTool],
    hierdoc_tools: list[HierarchicalDocumentTool],
) -> tuple[McpSdkServerConfig, list[str]]:
    """Build in-process MCP server from HoloDeck tool instances.

    Returns (server_config, allowed_tool_names).
    """
    sdk_tools = []
    allowed = []

    # Factory function required — Python closures capture by reference, not value.
    # A bare for-loop would cause all closures to share the last loop variable.
    def make_search_fn(t: VectorStoreTool, name: str) -> SdkMcpTool:
        @tool(name, f"Search {t.config.name}", {"query": str})  # input_schema arg
        async def search_fn(args: dict) -> dict:
            result = await t.search(args["query"])
            return {"content": [{"type": "text", "text": result}]}
        return search_fn

    for vs_tool in vectorstore_tools:
        tool_name = f"{vs_tool.config.name}_search"
        sdk_tools.append(make_search_fn(vs_tool, tool_name))
        allowed.append(f"mcp__holodeck_tools__{tool_name}")

    server = create_sdk_mcp_server(name="holodeck_tools", tools=sdk_tools)
    return server, allowed
```

### 2. Building `ClaudeAgentOptions` from `Agent` Config

```python
# lib/backends/claude_backend.py

from claude_agent_sdk import ClaudeAgentOptions, OutputFormat

def build_options(agent: Agent, tool_server, mcp_configs: dict) -> ClaudeAgentOptions:
    """Translate Agent config to ClaudeAgentOptions."""

    # Permission mode mapping
    perm_map = {
        "manual": "default",
        "acceptEdits": "acceptEdits",
        "acceptAll": "bypassPermissions",
    }
    perm_mode = None
    if agent.claude and agent.claude.permission_mode:
        perm_mode = perm_map[agent.claude.permission_mode.value]

    # MCP servers: holodeck in-process + external MCP tools
    mcp_servers = {}
    if tool_server:
        mcp_servers["holodeck_tools"] = tool_server
    mcp_servers.update(mcp_configs)

    # Auth env vars
    env = build_auth_env(agent.model)

    # OTel env vars (if observability configured)
    if agent.observability:
        env.update(translate_observability(agent.observability))

    return ClaudeAgentOptions(
        system_prompt=resolve_instructions(agent.instructions),
        model=agent.model.name,
        mcp_servers=mcp_servers,
        permission_mode=perm_mode,
        max_turns=agent.claude.max_turns if agent.claude else None,
        max_thinking_tokens=(
            agent.claude.extended_thinking.budget_tokens
            if agent.claude and agent.claude.extended_thinking and agent.claude.extended_thinking.enabled
            else None
        ),
        allowed_tools=build_allowed_tools(agent, tool_server),
        cwd=agent.claude.working_directory if agent.claude else None,
        output_format=build_output_format(agent.response_format),
        env=env,
    )
```

### 3. Stateless Test Invocation

```python
# lib/backends/claude_backend.py

from claude_agent_sdk import query, AssistantMessage, ResultMessage

async def invoke_once(self, message: str, context=None) -> ExecutionResult:
    from holodeck.models.test_result import TokenUsage  # confirm import path
    tool_calls = []
    tool_results = []
    response_text = ""
    structured_output = None
    token_usage = TokenUsage()

    async for msg in query(prompt=message, options=self._options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append({
                        "name": block.name,
                        "arguments": block.input,
                        "call_id": block.id,
                    })
                elif isinstance(block, ToolResultBlock):
                    tool_results.append({
                        "call_id": block.tool_use_id,
                        "result": _extract_result_text(block.content),
                        "is_error": block.is_error or False,
                    })
        elif isinstance(msg, ResultMessage):
            if msg.usage:
                token_usage = TokenUsage(
                    prompt_tokens=msg.usage.get("input_tokens", 0),
                    completion_tokens=msg.usage.get("output_tokens", 0),
                    total_tokens=(
                        msg.usage.get("input_tokens", 0)
                        + msg.usage.get("output_tokens", 0)
                    ),
                )
            structured_output = msg.structured_output
            if msg.is_error:
                return ExecutionResult(
                    response=response_text,
                    is_error=True,
                    error_reason=msg.result or "Agent returned error result",
                    token_usage=token_usage,
                )

    return ExecutionResult(
        response=response_text,
        tool_calls=tool_calls,
        tool_results=tool_results,
        token_usage=token_usage,
        structured_output=structured_output,
    )
```

### 4. Streaming Chat Session

```python
# lib/backends/claude_backend.py

from claude_agent_sdk import ClaudeSDKClient, StreamEvent

class ClaudeSession:
    def __init__(self, options: ClaudeAgentOptions):
        self._client = ClaudeSDKClient(options=options)

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def send_streaming(self, message: str) -> AsyncIterator[str]:
        await self._client.query(message)
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield block.text
            elif isinstance(msg, ResultMessage):
                # Session complete for this turn
                break

    async def close(self):
        await self._client.disconnect()
```

### 5. Authentication Environment Variables

```python
# lib/backends/validators.py

def build_auth_env(model: LLMProvider) -> dict[str, str]:
    """Build env vars for Claude Code subprocess based on auth_provider."""
    from holodeck.config.env_loader import get_env_var
    from holodeck.lib.errors import ConfigError

    auth = model.auth_provider or AuthProvider.api_key
    env: dict[str, str] = {}

    if auth == AuthProvider.api_key:
        key = get_env_var("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is required for auth_provider: api_key. "
                "Set it in your environment or .env file."
            )
        env["ANTHROPIC_API_KEY"] = key

    elif auth == AuthProvider.oauth_token:
        token = get_env_var("CLAUDE_CODE_OAUTH_TOKEN")
        if not token:
            raise ConfigError(
                "CLAUDE_CODE_OAUTH_TOKEN is required for auth_provider: oauth_token. "
                "Run 'claude setup-token' to obtain a token."
            )
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token

    elif auth == AuthProvider.bedrock:
        env["CLAUDE_CODE_USE_BEDROCK"] = "1"
        # AWS credentials come from the environment (AWS CLI, IAM role, etc.)

    elif auth == AuthProvider.vertex:
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        # GCP credentials come from the environment

    elif auth == AuthProvider.foundry:
        env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
        # Azure credentials come from the environment

    return env
```

### 6. MCP Tool Bridge

```python
# lib/backends/mcp_bridge.py

from typing import Any
from holodeck.models.tool import MCPTool

def build_claude_mcp_configs(mcp_tools: list[MCPTool]) -> dict[str, Any]:
    """Translate HoloDeck MCPTool configs to Claude SDK MCP server specs.

    Supports stdio transport only (SSE/HTTP deferred per spec).
    Returns dict suitable for ClaudeAgentOptions.mcp_servers.
    """
    configs: dict[str, Any] = {}
    for tool in mcp_tools:
        if tool.transport == "stdio":
            configs[tool.name] = {
                "type": "stdio",
                "command": tool.command,
                "args": tool.args or [],
                "env": _resolve_mcp_env(tool),
            }
        else:
            # SSE/HTTP/WebSocket deferred; skip with warning
            import logging
            logging.getLogger(__name__).warning(
                "MCP transport '%s' for tool '%s' is not supported by the "
                "Claude-native backend (stdio only). Skipping.",
                tool.transport, tool.name
            )
    return configs
```

---

## Refactoring Plan for Existing Code

### `agent_factory.py` → split into:
- `lib/backends/sk_backend.py` — all SK-specific code (Kernel, ChatCompletionAgent, ChatHistory)
- `lib/backends/selector.py` — provider routing
- `lib/backends/claude_backend.py` — Claude-native implementation

### `chat/executor.py` — full decoupling + streaming:

**Decoupling** (same pattern as `test_runner/executor.py` in Phase 9):

```python
# BEFORE (coupled to AgentFactory)
from holodeck.lib.test_runner.agent_factory import AgentFactory, AgentThreadRun

class AgentExecutor:
    def __init__(self, agent_config):
        self._factory = AgentFactory(agent_config)
        self._thread_run = None

    async def execute_turn(self, message):
        if self._thread_run is None:
            self._thread_run = await self._factory.create_thread_run()
        result = await self._thread_run.invoke(message)  # SK-only
        return AgentResponse(content=result.response, ...)

# AFTER (provider-agnostic via BackendSelector)
from holodeck.lib.backends.base import AgentBackend, AgentSession, BackendSelector

class AgentExecutor:
    def __init__(self, agent_config, backend=None):
        self._backend = backend
        self._session = None
        self._history = []

    async def _ensure_backend_and_session(self):
        if self._session is not None:
            return
        if self._backend is None:
            self._backend = await BackendSelector.select(
                self.agent_config, mode="chat"
            )
        self._session = await self._backend.create_session()

    async def execute_turn(self, message):
        await self._ensure_backend_and_session()
        exec_result = await self._session.send(message)  # Works with both backends
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": exec_result.response})
        return AgentResponse(content=exec_result.response, ...)
```

**Streaming** (net-new capability):

```python
    async def execute_turn_streaming(self, message):
        await self._ensure_backend_and_session()
        collected = []
        async for chunk in self._session.send_streaming(message):
            collected.append(chunk)
            yield chunk
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": "".join(collected)})
```

### `chat/session.py` — add streaming method:
- New: `ChatSessionManager.process_message_streaming()` → yields chunks from executor
- Existing `process_message()` unchanged (non-streaming path preserved)
- `start()` simplified — constructor no longer does I/O

### `cli/commands/chat.py` — streaming REPL:
- Replace spinner thread with progressive text output: `sys.stdout.write(chunk)` per chunk
- Both backends work: Claude gets real streaming, SK gets single-chunk (identical to current)

### `lib/test_runner/executor.py` — remove SK type dependencies:
- Replace `AgentExecutionResult` references with `ExecutionResult`
- Remove any `ChatHistory`, `FunctionCallContent`, `FunctionResultContent` imports
- Use `BackendSelector.create(agent, tools)` instead of `AgentFactory(config)`

---

## Testing Patterns

### Unit Tests

```python
# tests/unit/lib/backends/test_claude_backend.py

@pytest.mark.unit
async def test_invoke_once_returns_execution_result(mock_query):
    """ClaudeBackend.invoke_once() returns valid ExecutionResult."""
    mock_query.return_value = make_mock_message_stream(response="Hello world")

    backend = ClaudeBackend(agent_config=minimal_claude_agent())
    await backend.initialize()
    result = await backend.invoke_once("Say hello")

    assert isinstance(result, ExecutionResult)
    assert result.response == "Hello world"
    assert result.is_error is False
```

### Integration Tests

```python
# tests/integration/test_claude_backend_integration.py

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires API key")
async def test_claude_native_chat_response():
    """End-to-end: Claude-native agent returns coherent response."""
    agent = Agent(**load_yaml("tests/fixtures/claude_minimal_agent.yaml"))
    backend = BackendSelector.create(agent, tools=[])
    await backend.initialize()

    result = await backend.invoke_once("What is 2 + 2?")
    assert "4" in result.response
    assert result.is_error is False
```

---

## SDK Compatibility Notes

- **Package**: `claude-agent-sdk` (not `claude-code-sdk` — that package is deprecated)
- **Status**: Alpha — API may change between minor versions
- **Pin version**: `claude-agent-sdk>=0.1.39,<0.2.0` in `pyproject.toml`
- **Node.js**: Required on PATH; verified via `shutil.which("node")` at startup
- **Retry behaviour**: SDK handles API-level retries internally; HoloDeck adds session-spawn retries only
