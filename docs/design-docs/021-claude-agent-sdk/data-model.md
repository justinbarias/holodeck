# Data Model: Native Claude Agent SDK Integration

**Feature**: 021-claude-agent-sdk
**Date**: 2026-02-20

---

## 1. New Model: `ClaudeConfig` (claude_config.py)

Grouped under `src/holodeck/models/claude_config.py`. All Claude Agent SDK-specific configuration fields live here to keep `agent.py` clean and the Claude-native surface explicit.

### `AuthProvider` Enum

```python
class AuthProvider(str, Enum):
    api_key    = "api_key"      # ANTHROPIC_API_KEY (default)
    oauth_token = "oauth_token" # CLAUDE_CODE_OAUTH_TOKEN
    bedrock    = "bedrock"      # AWS credentials + CLAUDE_CODE_USE_BEDROCK=1
    vertex     = "vertex"       # GCP credentials + CLAUDE_CODE_USE_VERTEX=1
    foundry    = "foundry"      # Azure credentials + CLAUDE_CODE_USE_FOUNDRY=1
```

### `PermissionMode` Enum

```python
class PermissionMode(str, Enum):
    manual      = "manual"       # Every action requires approval
    acceptEdits = "acceptEdits"  # File edits auto-approved; others prompt
    acceptAll   = "acceptAll"    # Fully autonomous; no prompts
```

### `ExtendedThinkingConfig`

```python
class ExtendedThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    budget_tokens: int = Field(default=10_000, ge=1_000, le=100_000)
```

### `BashConfig`

```python
class BashConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    excluded_commands: list[str] = Field(default_factory=list)
    allow_unsafe: bool = False
```

### `FileSystemConfig`

```python
class FileSystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read: bool = False
    write: bool = False
    edit: bool = False
```

### `SubagentConfig`

```python
class SubagentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_parallel: int = Field(default=4, ge=1, le=16)
```

### `ClaudeConfig` (top-level)

```python
class ClaudeConfig(BaseModel):
    """Claude Agent SDK-specific settings.

    All fields optional. All capabilities default to disabled (least-privilege).
    """
    model_config = ConfigDict(extra="forbid")

    working_directory: str | None = Field(
        default=None,
        description="Scope agent file access to this path. Subprocess cwd."
    )
    permission_mode: PermissionMode = Field(
        default=PermissionMode.manual,
        description="Level of autonomous action. Defaults to manual (safest)."
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Maximum agent loop iterations. None = SDK default."
    )
    extended_thinking: ExtendedThinkingConfig | None = Field(
        default=None,
        description="Extended reasoning (deep thinking) configuration."
    )
    web_search: bool = Field(
        default=False,
        description="Enable built-in web search capability."
    )
    bash: BashConfig | None = Field(
        default=None,
        description="Shell command execution settings."
    )
    file_system: FileSystemConfig | None = Field(
        default=None,
        description="File read/write/edit access settings."
    )
    subagents: SubagentConfig | None = Field(
        default=None,
        description="Parallel sub-agent execution settings."
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Explicit tool allowlist. None = all configured tools."
    )
```

---

## 2. Updated Model: `LLMProvider` (llm.py)

**Change**: Add `auth_provider: AuthProvider | None`.

```python
class LLMProvider(BaseModel):
    # ... existing fields ...
    auth_provider: AuthProvider | None = Field(
        default=None,
        description=(
            "Authentication method for Anthropic provider. "
            "Defaults to api_key. Ignored for non-Anthropic providers."
        )
    )
```

**Validation**: `auth_provider` is only meaningful when `provider == ProviderEnum.ANTHROPIC`. A `model_validator` should emit a warning (not an error) if `auth_provider` is set for a non-Anthropic provider.

---

## 3. Updated Model: `Agent` (agent.py)

**Changes**: Add `embedding_provider` and `claude` fields.

```python
class Agent(BaseModel):
    # ... existing fields ...

    embedding_provider: LLMProvider | None = Field(
        default=None,
        description=(
            "Provider used exclusively for embedding generation. "
            "Required when provider: anthropic + vectorstore/hierarchical tools. "
            "Supports openai and azure_openai only."
        )
    )
    claude: ClaudeConfig | None = Field(
        default=None,
        description=(
            "Claude Agent SDK-specific settings. "
            "Applicable only when model.provider: anthropic."
        )
    )
```

**Startup Validation** (in `AgentFactory` / backend selector):

1. If `model.provider == anthropic` AND any tool is `VectorstoreTool | HierarchicalDocumentToolConfig` AND `embedding_provider is None` → raise `ConfigError`.
2. If `model.provider == anthropic` AND `tool_filtering is not None` → emit warning, clear `tool_filtering`.
3. If `model.provider != anthropic` AND `claude is not None` → emit warning that `claude:` block is ignored.
4. If `claude.working_directory` is set AND `{working_directory}/CLAUDE.md` exists → emit CLAUDE.md collision warning.

---

## 3b. Updated Model: `ChatSession` (models/chat.py)

**Change**: `history: ChatHistory` → `history: list[dict[str, Any]] = []`

```python
from typing import Any

class ChatSession(BaseModel):
    # ... existing fields ...
    history: list[dict[str, Any]] = []
    """Conversation history as plain dicts.

    SK backend serialises ChatHistory messages to list[dict] on session creation.
    Claude backend leaves history=[] — conversation state is managed by the SDK subprocess.
    Dict format: {"role": "user" | "assistant", "content": str}
    """
```

**Rationale**: `ChatHistory` is a Semantic Kernel type. Exposing it in `models/chat.py` creates a hard dependency on SK from a data model that should be provider-agnostic. Changing to `list[dict[str, Any]]` makes `ChatSession` compatible with both SK and Claude backends. Any callers of `session.history` that expect `ChatHistory` must be updated to expect `list[dict]`.

---

## 4. New Interface: `ExecutionResult` (lib/backends/base.py)

Provider-agnostic result from a single agent invocation turn. Both SK and Claude-native backends produce this type. All downstream consumers depend only on this type — no SK types leak through.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExecutionResult:
    """Provider-agnostic result of a single agent turn.

    Both SK and Claude-native backends produce this type.
    All downstream consumers (TestExecutor, ChatSessionManager)
    depend only on this interface — no SK types leak through.
    """

    response: str
    """The agent's final text response."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Tool calls made during the turn.
    Each dict: {"name": str, "arguments": dict, "call_id": str}
    """

    tool_results: list[dict[str, Any]] = field(default_factory=list)
    """Tool results returned during the turn.
    Each dict: {"call_id": str, "result": str, "is_error": bool}
    """

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    """Token counts using the existing TokenUsage Pydantic model.

    Fields: prompt_tokens, completion_tokens, total_tokens.
    Claude backend translates SDK response: input_tokens→prompt_tokens,
    output_tokens→completion_tokens.
    Import: from holodeck.models.test_result import TokenUsage (confirm path).
    """

    structured_output: Any | None = None
    """Validated structured object when response_format is configured. None otherwise."""

    num_turns: int = 1
    """Number of agent loop iterations used in this result."""

    is_error: bool = False
    """True if the agent produced an error result (not an evaluation failure)."""

    error_reason: str | None = None
    """Human-readable error description when is_error=True."""
```

**Field notes**:
- `tool_calls` / `tool_results` use consistent dict schema across both backends. SK's `FunctionCallContent` / `FunctionResultContent` are translated into this format in `sk_backend.py`.
- `structured_output` is populated only when `response_format` is configured. NLP/G-Eval metrics receive the serialised JSON string via `response`.
- `num_turns` enables `max_turns` exceeded detection: if `num_turns >= max_turns`, the test case is marked failed with reason `"max_turns limit reached"`.

---

## 5. New Interface: `AgentSession` Protocol (lib/backends/base.py)

Stateful multi-turn session abstraction used by the chat executor.

```python
from typing import AsyncIterator, Protocol, runtime_checkable

@runtime_checkable
class AgentSession(Protocol):
    """Stateful multi-turn conversation session.

    Used by ChatSessionManager for interactive chat.
    SK backend wraps AgentThreadRun; Claude backend wraps ClaudeSDKClient.
    """

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and return the complete response.

        Used by non-streaming chat consumers.
        """
        ...

    async def send_streaming(self, message: str) -> AsyncIterator[str]:
        """Send a message and stream text tokens as they arrive.

        Used by the holodeck chat terminal interface.
        Yields text chunks; call sites collect for complete response.
        """
        ...

    async def close(self) -> None:
        """Release all session resources cleanly."""
        ...
```

---

## 6. New Interface: `AgentBackend` Protocol (lib/backends/base.py)

Factory Protocol for creating test runs and chat sessions.

```python
@runtime_checkable
class AgentBackend(Protocol):
    """Provider backend factory.

    Selects the correct backend implementation based on agent.model.provider.
    """

    async def invoke_once(
        self,
        message: str,
        context: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute a single-turn invocation (stateless).

        Used by the test runner for isolated test case execution.
        A fresh conversation is created for every call.
        """
        ...

    async def create_session(self) -> AgentSession:
        """Create a stateful multi-turn chat session.

        Used by ChatSessionManager for interactive chat.
        """
        ...

    async def initialize(self) -> None:
        """Initialize backend resources (tool indexes, embedding services, etc.).

        Must be called before invoke_once() or create_session().
        """
        ...

    async def teardown(self) -> None:
        """Release all backend resources cleanly."""
        ...
```

---

## 7. Entity Relationships

```
Agent (YAML)
├── model: LLMProvider            ← adds auth_provider
├── embedding_provider: LLMProvider   ← NEW (for Anthropic + vectorstore)
├── claude: ClaudeConfig          ← NEW (all Claude-native settings)
│   ├── working_directory
│   ├── permission_mode
│   ├── max_turns
│   ├── extended_thinking: ExtendedThinkingConfig
│   ├── web_search
│   ├── bash: BashConfig
│   ├── file_system: FileSystemConfig
│   ├── subagents: SubagentConfig
│   └── allowed_tools
├── tools: list[ToolUnion]        ← unchanged
├── evaluations: EvaluationConfig ← unchanged
└── test_cases: list[TestCase]    ← unchanged

BackendSelector
├── provider == anthropic → ClaudeBackend
└── provider != anthropic → SKBackend

ClaudeBackend: AgentBackend
├── invoke_once() → ExecutionResult   (uses query())
└── create_session() → ClaudeSession  (uses ClaudeSDKClient)

SKBackend: AgentBackend
├── invoke_once() → ExecutionResult   (uses AgentThreadRun)
└── create_session() → SKSession      (uses AgentThreadRun multi-turn)

ExecutionResult (both backends produce this)
└── consumed by: TestExecutor, ChatSessionManager
```

---

## 8. State Transitions: AgentSession Lifecycle

```
[Created] ──initialize()──► [Ready]
                                │
                    ┌───────────┼───────────────────┐
                    │           │                   │
              send()        send_streaming()   close()
                    │           │                   │
                    ▼           ▼                   ▼
             [Processing]  [Streaming]         [Closed]
                    │           │
             [Ready]◄───────────┘
                    │
               close()
                    │
               [Closed]
```

**Error paths**:
- Subprocess crash during [Processing]: → `[Closed]`, raises `AgentSessionError`
- Tool failure during [Processing]: → continues; `ExecutionResult.tool_results[n].is_error = True`
- `max_turns` reached: → returns `ExecutionResult(is_error=True, error_reason="max_turns limit reached")`

---

## 9. Chat Layer Integration

The chat layer (`chat/executor.py`, `chat/session.py`) uses the same backend abstraction
as the test runner but through a different entry point:

- **Test runner**: `AgentBackend.invoke_once()` — stateless, fresh session per test case
- **Chat executor**: `AgentBackend.create_session()` → `AgentSession.send()` / `send_streaming()` — stateful multi-turn

### Chat Executor → AgentSession Mapping

| Current (Pre-Phase 10) | After Phase 10 |
|---|---|
| `AgentFactory(agent_config)` | `BackendSelector.select(agent, mode="chat")` |
| `_factory.create_thread_run()` | `_backend.create_session()` |
| `_thread_run.invoke(message)` → `AgentExecutionResult` | `_session.send(message)` → `ExecutionResult` |
| `_thread_run.chat_history.messages` | `self._history` (executor-local `list[dict]`) |
| `_factory.shutdown()` | `_session.close()` + `_backend.teardown()` |

### Conversation History Strategy

History is tracked locally in `AgentExecutor._history: list[dict[str, Any]]` as
`{"role": "user" | "assistant", "content": str}` dicts. This is backend-agnostic:

- **SK sessions**: `SKSession` internally maintains `ChatHistory` for the SK agent loop.
  The executor does NOT read from it — it records messages from its own call/response pairs.
- **Claude sessions**: `ClaudeSession` tracks `session_id` for multi-turn state.
  Conversation context lives in the SDK subprocess. The executor's local history is for
  display/export only.

This approach avoids extending the `AgentSession` protocol with a `get_history()` method,
keeping the protocol minimal (send, send_streaming, close).

### Streaming Architecture

```
User Input (CLI)
    │
    ▼
ChatSessionManager.process_message_streaming()
    │
    ▼
AgentExecutor.execute_turn_streaming()
    │
    ▼
AgentSession.send_streaming()
    │
    ├── ClaudeSession: real token-by-token via ClaudeSDKClient
    └── SKSession: complete response as single yield (fallback)
    │
    ▼
CLI: sys.stdout.write(chunk) per chunk → progressive terminal display
```

Token usage and tool call details are NOT available during the streaming path.
The CLI tracks only the full response text for display. Token accumulation
continues to work through the non-streaming `process_message()` path if needed.
