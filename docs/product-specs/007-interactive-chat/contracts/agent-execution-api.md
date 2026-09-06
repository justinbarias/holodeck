# Agent Execution API Contract

**Feature**: `007-interactive-chat`
**Contract Type**: Internal Python API
**Date**: 2025-11-22

## Overview

This contract defines the internal API for the Agent Execution Runtime, which orchestrates LLM interactions, tool execution, and conversation management. This API is consumed by the `holodeck chat` CLI command and will be reused by the `holodeck test` command.

---

## Module: `holodeck.agent.executor`

### Class: `AgentExecutor`

Orchestrates agent execution using Semantic Kernel.

**Purpose**: Provides a high-level interface for executing agents with conversation context.

#### Constructor

```python
def __init__(
    self,
    agent_config: AgentConfig,
    enable_observability: bool = False
) -> None:
    """
    Initialize agent executor.

    Args:
        agent_config: Loaded agent configuration
        enable_observability: Enable OpenTelemetry instrumentation

    Raises:
        AgentInitializationError: If agent cannot be initialized
        InvalidToolError: If tool configuration is invalid
        LLMConnectionError: If LLM provider is unreachable
    """
```

#### Methods

**`async def execute_turn(message: str) -> AgentResponse`**

Execute a single conversation turn.

```python
async def execute_turn(self, message: str) -> AgentResponse:
    """
    Process user message and generate agent response.

    Args:
        message: User message (already validated)

    Returns:
        AgentResponse containing response text, tool executions, and token usage

    Raises:
        MessageProcessingError: If message processing fails
        ToolExecutionError: If tool execution fails critically
        LLMError: If LLM call fails
    """
```

**`def get_history() -> ChatHistory`**

Retrieve current conversation history.

```python
def get_history(self) -> ChatHistory:
    """
    Get conversation history.

    Returns:
        Semantic Kernel ChatHistory instance
    """
```

**`def clear_history() -> None`**

Clear conversation history (keep system message).

```python
def clear_history(self) -> None:
    """Clear conversation history but preserve system message."""
```

**`async def shutdown() -> None`**

Clean up resources.

```python
async def shutdown(self) -> None:
    """Gracefully shutdown agent executor and clean up resources."""
```

---

## Module: `holodeck.agent.session`

### Class: `ChatSessionManager`

Manages chat session lifecycle and state.

**Purpose**: Handles session initialization, message tracking, and context management.

#### Constructor

```python
def __init__(
    self,
    agent_config: AgentConfig,
    config: ChatConfig
) -> None:
    """
    Initialize chat session manager.

    Args:
        agent_config: Loaded agent configuration
        config: Chat runtime configuration

    Raises:
        SessionInitializationError: If session cannot be initialized
    """
```

#### Methods

**`async def start() -> ChatSession`**

Start new chat session.

```python
async def start(self) -> ChatSession:
    """
    Initialize and start chat session.

    Returns:
        ChatSession instance with initialized state

    Raises:
        SessionInitializationError: If initialization fails
    """
```

**`async def process_message(message: str) -> AgentResponse`**

Process user message through agent.

```python
async def process_message(self, message: str) -> AgentResponse:
    """
    Validate, process, and respond to user message.

    Args:
        message: Raw user input

    Returns:
        AgentResponse with response and metadata

    Raises:
        ValidationError: If message validation fails
        MessageProcessingError: If processing fails
    """
```

**`def get_session() -> ChatSession`**

Retrieve current session state.

```python
def get_session(self) -> ChatSession:
    """
    Get current session.

    Returns:
        ChatSession instance
    """
```

**`def should_warn_context_limit() -> bool`**

Check if approaching context limit.

```python
def should_warn_context_limit(self) -> bool:
    """
    Check if conversation approaching message limit.

    Returns:
        True if at 80% of max_messages threshold
    """
```

**`async def terminate() -> None`**

Terminate session and cleanup.

```python
async def terminate(self) -> None:
    """Gracefully terminate session and release resources."""
```

---

## Module: `holodeck.agent.message`

### Class: `MessageValidator`

Validates user messages.

**Purpose**: Implements validation pipeline for input sanitization.

#### Methods

**`def validate(message: str) -> tuple[bool, str | None]`**

```python
def validate(self, message: str) -> tuple[bool, str | None]:
    """
    Validate user message.

    Args:
        message: Raw user input

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, "error reason") if invalid
    """
```

---

## Module: `holodeck.agent.streaming`

### Class: `ToolExecutionStream`

Streams tool execution events.

**Purpose**: Provides real-time tool execution updates for display.

#### Methods

**`async def stream_execution(tool_call: ToolExecution) -> AsyncIterator[ToolEvent]`**

```python
async def stream_execution(
    self,
    tool_call: ToolExecution
) -> AsyncIterator[ToolEvent]:
    """
    Stream tool execution events.

    Args:
        tool_call: Tool execution metadata

    Yields:
        ToolEvent instances (started, progress, completed, failed)
    """
```

---

## Data Transfer Objects

### `AgentResponse`

Response from agent execution.

```python
@dataclass
class AgentResponse:
    """Agent response container."""

    content: str
    """Agent response text"""

    tool_executions: list[ToolExecution]
    """Tools executed during this turn"""

    tokens_used: TokenUsage | None
    """Token usage for this turn"""

    execution_time: float
    """Total execution time in seconds"""
```

### `ToolEvent`

Tool execution event for streaming.

```python
@dataclass
class ToolEvent:
    """Tool execution event."""

    event_type: ToolEventType
    """Event type (started, progress, completed, failed)"""

    tool_name: str
    """Tool name"""

    timestamp: datetime
    """Event timestamp"""

    data: dict[str, Any]
    """Event-specific data"""
```

**ToolEventType Enum**:
```python
class ToolEventType(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
```

---

## Error Hierarchy

All agent execution errors inherit from `AgentError` (defined in `holodeck.lib.errors`).

```python
class AgentError(HoloDeckError):
    """Base class for agent execution errors."""
    pass

class AgentInitializationError(AgentError):
    """Agent initialization failed."""
    pass

class MessageProcessingError(AgentError):
    """Message processing failed."""
    pass

class ToolExecutionError(AgentError):
    """Tool execution failed."""
    pass

class LLMError(AgentError):
    """LLM provider error."""
    pass

class LLMConnectionError(LLMError):
    """Cannot connect to LLM provider."""
    pass

class SessionInitializationError(AgentError):
    """Session initialization failed."""
    pass

class ValidationError(AgentError):
    """Input validation failed."""
    pass
```

---

## Usage Example

```python
from holodeck.config.loader import ConfigLoader
from holodeck.agent.executor import AgentExecutor
from holodeck.agent.session import ChatSessionManager
from holodeck.models.chat import ChatConfig

# Load agent configuration
config_loader = ConfigLoader()
agent_config = config_loader.load_agent("examples/weather-agent.yaml")

# Create chat configuration
chat_config = ChatConfig(
    agent_config_path=Path("examples/weather-agent.yaml"),
    verbose=True,
    enable_observability=False,
    max_messages=50
)

# Initialize session
session_manager = ChatSessionManager(agent_config, chat_config)
session = await session_manager.start()

# Interactive loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['exit', 'quit']:
        break

    try:
        response = await session_manager.process_message(user_input)
        print(f"Agent: {response.content}")

        # Display tool executions (if verbose)
        for tool in response.tool_executions:
            print(f"[Tool] {tool.tool_name} → {tool.status} ({tool.execution_time}s)")

    except ValidationError as e:
        print(f"Error: {e}")

# Cleanup
await session_manager.terminate()
```

---

## Thread Safety

- `AgentExecutor`: **NOT thread-safe** - Single session per instance
- `ChatSessionManager`: **NOT thread-safe** - Single interactive session
- Concurrent sessions require separate instances

---

## Testing Requirements

All public methods must have:
- Unit tests with mocked Semantic Kernel components
- Integration tests with real LLM providers (test accounts)
- Error case coverage (invalid configs, LLM failures, tool errors)

**Test Markers**:
- `@pytest.mark.unit` for isolated tests
- `@pytest.mark.integration` for end-to-end tests
- `@pytest.mark.slow` for LLM-dependent tests
