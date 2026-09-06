# Data Model: Agent Local Server

**Feature**: 017-agent-local-server
**Date**: 2025-12-29

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentServer                               │
│  - agent_config: Agent                                           │
│  - protocol: ProtocolType                                        │
│  - port: int                                                     │
│  - sessions: SessionStore                                        │
│  - is_ready: bool                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ manages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SessionStore                              │
│  - sessions: dict[str, ServerSession]                            │
│  - ttl_seconds: int = 1800 (30 min)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ contains
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ServerSession                             │
│  - session_id: str (ULID)                                        │
│  - agent_executor: AgentExecutor                                 │
│  - created_at: datetime                                          │
│  - last_activity: datetime                                       │
│  - message_count: int                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AgentExecutor (existing)                      │
│  - agent_config: Agent                                           │
│  - _thread_run: AgentThreadRun                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Entities

### 1. AgentServer

The main server instance that hosts a single agent.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `agent_config` | `Agent` | Loaded agent configuration | Required, validated |
| `protocol` | `ProtocolType` | Active protocol (ag-ui or rest) | Enum, default: ag-ui |
| `port` | `int` | Server listening port | 1-65535, default: 8000 |
| `host` | `str` | Server bind address | default: 0.0.0.0 |
| `sessions` | `SessionStore` | Session storage | Lazy initialized |
| `is_ready` | `bool` | Server ready state | Set after agent init |
| `cors_origins` | `list[str]` | CORS allowed origins | default: ["*"] |
| `debug` | `bool` | Debug logging enabled | default: False |

**State Transitions**:
```
INITIALIZING → READY → RUNNING → SHUTTING_DOWN → STOPPED
```

---

### 2. SessionStore

In-memory session storage with automatic TTL cleanup.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `sessions` | `dict[str, ServerSession]` | Active sessions | Keyed by session_id |
| `ttl_seconds` | `int` | Session timeout | default: 1800 (30 min) |
| `_cleanup_task` | `asyncio.Task` | Background cleanup | Started on init |

**Operations**:
- `get(session_id) -> ServerSession | None`
- `create(agent_executor) -> ServerSession`
- `delete(session_id) -> bool`
- `touch(session_id) -> None` (updates last_activity)
- `cleanup_expired() -> int` (returns count removed)

---

### 3. ServerSession

Individual conversation session with an agent.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `session_id` | `str` | Unique identifier | ULID format |
| `agent_executor` | `AgentExecutor` | Agent execution context | Required |
| `created_at` | `datetime` | Session creation time | UTC, immutable |
| `last_activity` | `datetime` | Last request time | UTC, updated on activity |
| `message_count` | `int` | Messages in session | Incremented per message |

**Lifecycle**:
```
Created (new request without session_id)
    ↓
Active (processing requests)
    ↓
Expired (30 min inactivity) → Cleaned up
    or
Deleted (explicit DELETE request) → Cleaned up
```

---

## Request/Response Models

### 4. ChatRequest

Request payload for chat endpoints.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `message` | `str` | User message content | Required, max 10,000 chars |
| `session_id` | `str | None` | Session identifier | Optional, ULID if provided |
| `files` | `list[FileContent] | None` | Attached files | Optional, max 10 files |

**Validation Rules**:
- `message` must not be empty or whitespace-only
- `session_id` if provided must be valid ULID format
- Total file size must not exceed 100 MB
- Individual file size must not exceed 50 MB

---

### 4a. FileContent

Binary file content for multimodal inputs.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `content` | `str` | Base64-encoded file data | Required for JSON requests |
| `mime_type` | `str` | MIME type of the file | Required, validated |
| `filename` | `str | None` | Original filename | Optional |

**Supported MIME Types**:
- Images: `image/png`, `image/jpeg`, `image/gif`, `image/webp`
- Documents: `application/pdf`
- Office: `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (docx), `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (xlsx), `application/vnd.openxmlformats-officedocument.presentationml.presentation` (pptx)
- Text: `text/plain`, `text/csv`, `text/markdown`

**Processing**: Files are processed by HoloDeck's `FileProcessor` before being sent to the agent:
- Images → OCR text extraction
- PDFs → Text extraction
- Office docs → Markdown conversion

---

### 5. ChatResponse

Response payload for synchronous chat endpoint (REST only).

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `message_id` | `str` | Unique message identifier | ULID |
| `content` | `str` | Agent response text | May be empty |
| `session_id` | `str` | Session used/created | ULID |
| `tool_calls` | `list[ToolCallInfo]` | Tools invoked | May be empty |
| `tokens_used` | `TokenUsage | None` | Token consumption | Optional |
| `execution_time_ms` | `int` | Request latency | Milliseconds |

---

### 6. ToolCallInfo

Tool execution information in response.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `name` | `str` | Tool name | Required |
| `arguments` | `dict[str, Any]` | Tool parameters | JSON-serializable |
| `status` | `str` | Execution status | "success" or "error" |

---

### 7. HealthResponse

Response for health check endpoints.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `status` | `str` | Overall status | "healthy" or "unhealthy" |
| `agent_name` | `str | None` | Loaded agent name | Present if agent loaded |
| `agent_ready` | `bool` | Agent ready state | True when initialized |
| `active_sessions` | `int` | Session count | >= 0 |
| `uptime_seconds` | `float` | Server uptime | >= 0 |

---

### 8. ProblemDetail

RFC 7807 error response format.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `type` | `str` | Error type URI | URL format |
| `title` | `str` | Short description | Human-readable |
| `status` | `int` | HTTP status code | 4xx or 5xx |
| `detail` | `str | None` | Detailed message | Optional |
| `instance` | `str | None` | Request identifier | Optional, for tracing |

---

## Enumerations

### ProtocolType

```python
class ProtocolType(str, Enum):
    AG_UI = "ag-ui"
    REST = "rest"
```

### ServerState

```python
class ServerState(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
```

---

## Pydantic Model Definitions

```python
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from python_ulid import ULID


class ProtocolType(str, Enum):
    AG_UI = "ag-ui"
    REST = "rest"


SUPPORTED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/gif", "image/webp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain", "text/csv", "text/markdown",
}


class FileContent(BaseModel):
    """Binary file content for multimodal inputs."""

    content: str = Field(..., description="Base64-encoded file data")
    mime_type: str = Field(..., description="MIME type of the file")
    filename: str | None = Field(default=None, description="Original filename")

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        if v not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        import base64
        try:
            # Validate it's valid base64
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("content must be valid base64")
        return v


class ChatRequest(BaseModel):
    """Request payload for chat endpoints."""

    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None)
    files: list[FileContent] | None = Field(default=None, max_length=10)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v

    @field_validator("session_id")
    @classmethod
    def valid_ulid(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                ULID.from_str(v)
            except ValueError:
                raise ValueError("session_id must be valid ULID")
        return v


class ToolCallInfo(BaseModel):
    """Tool execution information."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"


class ChatResponse(BaseModel):
    """Response payload for synchronous chat."""

    message_id: str
    content: str
    session_id: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    tokens_used: dict[str, int] | None = None
    execution_time_ms: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_name: str | None = None
    agent_ready: bool = False
    active_sessions: int = 0
    uptime_seconds: float = 0.0


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details error response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
```

---

## Relationship to Existing Models

| New Model | Existing Model | Relationship |
|-----------|----------------|--------------|
| `ServerSession` | `ChatSession` | Extends with HTTP context |
| `ChatRequest` | N/A | New for HTTP layer |
| `FileContent` | `TestCaseFile` | Similar structure for multimodal |
| `ChatResponse` | `AgentResponse` | Wraps for HTTP response |
| `ToolCallInfo` | `ToolExecution` | Simplified for API |
| `HealthResponse` | N/A | New for health checks |

---

## Storage Considerations

| Entity | Storage | Persistence | Cleanup |
|--------|---------|-------------|---------|
| `AgentServer` | Memory | Process lifetime | On shutdown |
| `SessionStore` | Memory | Process lifetime | On shutdown |
| `ServerSession` | Memory | 30-min TTL | Background task |

**Note**: Persistent storage for sessions is explicitly deferred per spec assumptions. Future work may add Redis or database backing.
