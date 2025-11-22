# Data Model: Interactive Agent Testing

**Feature**: `007-interactive-chat`
**Date**: 2025-11-22
**Status**: Complete

## Overview

This document defines the data models for the interactive chat feature. All models use Pydantic for validation and serialization.

---

## Core Entities

### 1. ChatSession

Represents an active conversation between developer and agent.

**Purpose**: Manages conversation state, history, and lifecycle for a single interactive session.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | `str` | Yes | Unique session identifier (UUID) |
| `agent_config` | `AgentConfig` | Yes | Loaded agent configuration |
| `history` | `ChatHistory` | Yes | Semantic Kernel ChatHistory instance |
| `started_at` | `datetime` | Yes | Session start timestamp |
| `message_count` | `int` | Yes | Total messages exchanged |
| `metadata` | `dict[str, Any]` | No | Optional session metadata |

**State Transitions**:
- `INITIALIZING` → `ACTIVE` (agent loaded successfully)
- `ACTIVE` → `TERMINATED` (user exits or error occurs)

**Relationships**:
- Contains multiple `Message` instances (via `ChatHistory`)
- Executes tools defined in `AgentConfig`

**Validation Rules**:
- `session_id` must be valid UUID format
- `message_count` must be non-negative
- `started_at` cannot be in the future

**Pydantic Model**:
```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID, uuid4
from semantic_kernel.contents import ChatHistory
from holodeck.models.agent import AgentConfig

class SessionState(str, Enum):
    """Chat session state."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    TERMINATED = "terminated"

class ChatSession(BaseModel):
    """Interactive chat session model."""

    session_id: UUID = Field(default_factory=uuid4)
    agent_config: AgentConfig
    history: ChatHistory
    started_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, ge=0)
    state: SessionState = Field(default=SessionState.INITIALIZING)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # Allow ChatHistory

    @field_validator('started_at')
    @classmethod
    def validate_started_at(cls, v: datetime) -> datetime:
        if v > datetime.utcnow():
            raise ValueError("started_at cannot be in the future")
        return v
```

---

### 2. Message

Represents a single exchange in the conversation.

**Purpose**: Encapsulates message content, role, and metadata for display and tracking.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | `MessageRole` | Yes | Message role (user, assistant, system) |
| `content` | `str` | Yes | Message content |
| `timestamp` | `datetime` | Yes | Message creation time |
| `tool_calls` | `list[ToolExecution]` | No | Tool calls made during this exchange |
| `tokens_used` | `TokenUsage \| None` | No | Token usage for this message |

**Enums**:
```python
class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
```

**Validation Rules**:
- `content` must not be empty (after stripping whitespace)
- `content` length must be ≤ 10,000 characters for user messages
- `role` must be valid MessageRole value
- `tool_calls` can only exist for assistant messages

**Pydantic Model**:
```python
from pydantic import BaseModel, Field, field_validator

class Message(BaseModel):
    """Chat message model."""

    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: list[ToolExecution] = Field(default_factory=list)
    tokens_used: TokenUsage | None = None

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str, info) -> str:
        # Strip whitespace
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty")

        # Enforce size limit for user messages
        if info.data.get('role') == MessageRole.USER and len(v) > 10000:
            raise ValueError("User message exceeds 10,000 character limit")

        return v

    @field_validator('tool_calls')
    @classmethod
    def validate_tool_calls(cls, v: list[ToolExecution], info) -> list[ToolExecution]:
        # Tool calls only allowed for assistant messages
        if v and info.data.get('role') != MessageRole.ASSISTANT:
            raise ValueError("Tool calls only allowed for assistant messages")
        return v
```

---

### 3. ToolExecution

Metadata about tool calls made during agent processing.

**Purpose**: Tracks tool invocations for logging, display, and observability.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tool_name` | `str` | Yes | Name of the tool/function called |
| `parameters` | `dict[str, Any]` | Yes | Tool parameters (sanitized) |
| `result` | `str \| None` | No | Tool execution result (sanitized) |
| `status` | `ToolStatus` | Yes | Execution status |
| `execution_time` | `float` | No | Execution time in seconds |
| `error_message` | `str \| None` | No | Error details if status=FAILED |

**Enums**:
```python
class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
```

**Validation Rules**:
- `tool_name` must match configured tool in agent
- `execution_time` must be non-negative if provided
- `error_message` required if `status=FAILED`
- `result` must be sanitized (no terminal escape sequences)

**Pydantic Model**:
```python
from pydantic import BaseModel, Field, field_validator

class ToolExecution(BaseModel):
    """Tool execution metadata."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    status: ToolStatus = Field(default=ToolStatus.PENDING)
    execution_time: float | None = Field(default=None, ge=0)
    error_message: str | None = None

    @field_validator('error_message')
    @classmethod
    def validate_error_message(cls, v: str | None, info) -> str | None:
        if info.data.get('status') == ToolStatus.FAILED and not v:
            raise ValueError("error_message required when status=FAILED")
        return v

    @field_validator('result')
    @classmethod
    def sanitize_result(cls, v: str | None) -> str | None:
        if v:
            # Sanitize output (implemented in validation.py)
            from holodeck.lib.validation import sanitize_tool_output
            return sanitize_tool_output(v)
        return v
```

---

### 4. TokenUsage

Token consumption tracking for LLM calls.

**Purpose**: Enables cost tracking and observability for interactive sessions.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt_tokens` | `int` | Yes | Tokens in the prompt |
| `completion_tokens` | `int` | Yes | Tokens in the completion |
| `total_tokens` | `int` | Yes | Total tokens (prompt + completion) |

**Validation Rules**:
- All token counts must be non-negative
- `total_tokens` must equal `prompt_tokens + completion_tokens`

**Pydantic Model**:
```python
from pydantic import BaseModel, Field, field_validator

class TokenUsage(BaseModel):
    """Token usage tracking."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @field_validator('total_tokens')
    @classmethod
    def validate_total(cls, v: int, info) -> int:
        prompt = info.data.get('prompt_tokens', 0)
        completion = info.data.get('completion_tokens', 0)
        if v != prompt + completion:
            raise ValueError(
                f"total_tokens ({v}) must equal "
                f"prompt_tokens ({prompt}) + completion_tokens ({completion})"
            )
        return v
```

---

### 5. ChatConfig

Runtime configuration for chat sessions.

**Purpose**: Captures CLI options and runtime settings for a chat session.

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_config_path` | `Path` | Yes | Path to agent YAML configuration |
| `verbose` | `bool` | No | Show detailed tool execution (default: False) |
| `enable_observability` | `bool` | No | Enable OpenTelemetry tracing (default: False) |
| `max_messages` | `int` | No | Max conversation messages (default: 50) |

**Validation Rules**:
- `agent_config_path` must exist and be readable
- `max_messages` must be positive

**Pydantic Model**:
```python
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

class ChatConfig(BaseModel):
    """Chat session runtime configuration."""

    agent_config_path: Path
    verbose: bool = Field(default=False)
    enable_observability: bool = Field(default=False)
    max_messages: int = Field(default=50, gt=0)

    @field_validator('agent_config_path')
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Agent config not found: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v
```

---

## Entity Relationships

```
ChatSession (1) ───────> (1) AgentConfig [loaded from YAML]
     │
     │
     └─> (1) ChatHistory ───────> (N) Message
                                        │
                                        └─> (N) ToolExecution
                                        └─> (0..1) TokenUsage
```

**Lifecycle Flow**:
1. User runs `holodeck chat agent.yaml`
2. `ChatConfig` created from CLI arguments
3. `AgentConfig` loaded from YAML file
4. `ChatSession` initialized with empty `ChatHistory`
5. User sends message → `Message(role=USER)` added to history
6. Agent processes → `Message(role=ASSISTANT)` with `ToolExecution` instances
7. Repeat steps 5-6 until user exits
8. Session terminated, history discarded (MVP - no persistence)

---

## File Locations

All models will be implemented in:

```
src/holodeck/models/
├── chat.py              # NEW: ChatSession, Message, ChatConfig
├── tool_execution.py    # NEW: ToolExecution, ToolStatus
├── token_usage.py       # NEW: TokenUsage
├── agent.py             # EXISTING: AgentConfig
└── llm.py               # EXISTING: LLMConfig
```

---

## Validation Architecture

Input validation happens in layers:

1. **Pydantic model validation** (structure, types, basic constraints)
2. **Custom validators** (business logic via `@field_validator`)
3. **ValidationPipeline** (runtime validation for user input)
4. **Sanitization** (output cleaning via `sanitize_tool_output()`)

See `research.md` Section 4 for detailed validation pipeline design.
