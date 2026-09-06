# Contract: Provider-Agnostic Execution Interface

**Module**: `src/holodeck/lib/backends/base.py`
**Feature**: 021-claude-agent-sdk (FR-012b)

---

## Overview

This contract defines the provider-agnostic interface that all agent execution backends must implement. It replaces the Semantic Kernel type dependencies (`ChatHistory`, `ChatMessageContent`, `FunctionCallContent`) throughout HoloDeck's test runner and chat layer.

**Consumers**: `TestExecutor`, `ChatSessionManager`, `AgentExecutor` (chat)
**Producers**: `SKBackend`, `ClaudeBackend`

---

## `ExecutionResult`

Single-turn invocation result. Produced by both SK and Claude-native backends.

```python
@dataclass
class ExecutionResult:
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    structured_output: Any | None = None
    num_turns: int = 1
    is_error: bool = False
    error_reason: str | None = None
```

### Field Schemas

**`tool_calls`** — each element:
```json
{
  "name": "vectorstore_search",
  "arguments": {"query": "refund policy"},
  "call_id": "toolu_01Abc..."
}
```

**`tool_results`** — each element:
```json
{
  "call_id": "toolu_01Abc...",
  "result": "Our refund policy is 30 days...",
  "is_error": false
}
```

**`token_usage`** — keys:
```json
{
  "input_tokens": 450,
  "output_tokens": 180,
  "total_tokens": 630
}
```

### Error Conditions

| `is_error` | `error_reason` | Meaning |
|---|---|---|
| `False` | `None` | Successful turn |
| `True` | `"max_turns limit reached"` | Agent hit configured `max_turns` |
| `True` | `"subprocess terminated unexpectedly"` | Claude subprocess crashed |
| `True` | `"tool execution failed: <name>"` | Critical tool failure |
| `True` | `"timeout exceeded"` | Agent exceeded timeout |

---

## `AgentSession` Protocol

Stateful multi-turn session. Used by `ChatSessionManager`.

```python
class AgentSession(Protocol):
    async def send(self, message: str) -> ExecutionResult: ...
    async def send_streaming(self, message: str) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...
```

### Implementation Notes

**SK implementation** (`SKSession`):
- `send()`: invokes `AgentThreadRun.execute_turn()`, wraps result in `ExecutionResult`
- `send_streaming()`: currently non-streaming; yields full response as single chunk (no-op streaming)
- `close()`: releases SK kernel resources

**Claude implementation** (`ClaudeSession`):
- `send()`: calls `client.query()`, collects full response from `receive_response()` iterator
- `send_streaming()`: calls `client.query()`, yields text chunks from `StreamEvent` partial messages
- `close()`: calls `client.disconnect()`

---

## `AgentBackend` Protocol

Backend factory. `BackendSelector` creates the appropriate implementation.

```python
class AgentBackend(Protocol):
    async def initialize(self) -> None: ...
    async def invoke_once(self, message: str, context: list[dict] | None = None) -> ExecutionResult: ...
    async def create_session(self) -> AgentSession: ...
    async def teardown(self) -> None: ...
```

### Lifecycle

```
BackendSelector.create(agent_config, tools)
    │
    ▼
backend.initialize()          # Tool index setup, credential validation
    │
    ├─► backend.invoke_once() × N    # For test cases (TestExecutor)
    │
    └─► session = backend.create_session()
             session.send() × N      # For chat turns (ChatSessionManager)
             session.close()
    │
    ▼
backend.teardown()            # Resource cleanup
```

---

## Error Types

```python
class BackendError(HoloDeckError):
    """Base for all backend errors."""

class BackendInitError(BackendError):
    """Raised during initialize() — startup validation failures."""

class BackendSessionError(BackendError):
    """Raised during send() — session-level failures (subprocess crash)."""

class BackendTimeoutError(BackendError):
    """Raised when a single invocation exceeds configured timeout."""
```

---

## Compatibility Guarantee

1. **SK backend** `ExecutionResult.response` is identical to the current `AgentThreadRun` response string.
2. **SK backend** `ExecutionResult.tool_calls` format is identical to the current `AgentExecutionResult.tool_calls` dict structure.
3. `TestExecutor` and `ChatSessionManager` are modified to use `ExecutionResult` and `AgentSession` **only**. No `ChatHistory`, `FunctionCallContent`, or any other SK type is referenced in these consumers after this feature is implemented.
4. Non-Anthropic agents continue to work without modification — they use the SK backend which implements the same interface.
