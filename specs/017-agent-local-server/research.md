# Research: Agent Local Server

**Feature**: 017-agent-local-server
**Date**: 2025-12-29

## Executive Summary

This research addresses the technical decisions required to implement a local agent server with two protocol options: AG-UI (default) and FastAPI REST. The findings are based on analysis of the AG-UI protocol specification, sample implementations, and the existing HoloDeck codebase.

---

## 1. AG-UI Protocol Integration

### 1.1 Protocol Overview

The **Agent User Interaction Protocol (AG-UI)** is a lightweight, event-driven framework for seamless communication between frontend applications and AI agents. Key design principles:

- **Transport Agnostic**: Supports SSE, WebSockets, webhooks, and other mechanisms
- **Bidirectional**: Enables collaborative human-AI workflows
- **Minimal Opinions**: Requires only 16 standardized event types
- **Middleware Layer**: Events need only be AG-UI-compatible, enabling flexibility

**Core Abstraction**:
```
run(input: RunAgentInput) -> Observable<BaseEvent>
```

### 1.2 Decision: Use `ag-ui-protocol` Python SDK

**Rationale**: The official `ag-ui-protocol` package (v0.1.10) provides:
- Strongly-typed Pydantic models for all 16+ event types
- Built-in `EventEncoder` for SSE streaming with content negotiation
- Format negotiation based on HTTP Accept headers
- Two transport options: HTTP SSE (text) and HTTP Binary Protocol (high-performance)
- Full Python 3.9+ compatibility (HoloDeck requires 3.10+)

**Alternatives Considered**:
1. **Implement protocol from scratch** - Rejected: Unnecessary duplication, higher maintenance burden
2. **Use raw JSON SSE** - Rejected: Loses type safety and format negotiation benefits

### 1.3 Complete Event Type Reference

AG-UI organizes events into distinct categories:

#### Lifecycle Events

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `RunStartedEvent` | `thread_id`, `run_id`, `parent_run_id?`, `input?` | Initiates execution |
| `RunFinishedEvent` | `thread_id`, `run_id`, `result?` | Signals successful completion |
| `RunErrorEvent` | `message`, `code?` | Indicates failure |
| `StepStartedEvent` | `step_name` | Brackets discrete work unit start |
| `StepFinishedEvent` | `step_name` | Brackets discrete work unit end |

#### Text Message Events (Start-Content-End Pattern)

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `TextMessageStartEvent` | `message_id`, `role` | Opens message stream |
| `TextMessageContentEvent` | `message_id`, `delta` | Streams incremental text chunks |
| `TextMessageEndEvent` | `message_id` | Closes message stream |
| `TextMessageChunkEvent` | (convenience) | Auto-expands to Start→Content→End |

#### Tool Call Events

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `ToolCallStartEvent` | `tool_call_id`, `tool_call_name`, `parent_message_id?` | Initiates tool execution |
| `ToolCallArgsEvent` | `tool_call_id`, `delta` | Streams argument JSON fragments |
| `ToolCallEndEvent` | `tool_call_id` | Completes argument transmission |
| `ToolCallResultEvent` | `tool_call_id`, `content` | Delivers tool output |

#### State Management Events (Snapshot-Delta Pattern)

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `StateSnapshotEvent` | `snapshot` | Complete state baseline |
| `StateDeltaEvent` | `delta` | RFC 6902 JSON Patch operations |
| `MessagesSnapshotEvent` | `messages` | Full conversation history |

#### Special Events

| Event Type | Fields | Purpose |
|------------|--------|---------|
| `CustomEvent` | `name`, `value` | Application-defined events |
| `RawEvent` | `event`, `source?` | Wraps external system events |
| `ActivitySnapshotEvent` | `message_id`, `activity_type`, `content` | Frontend-only UI updates |
| `ActivityDeltaEvent` | `delta` | Incremental activity patches |

### 1.4 RunAgentInput Structure

The input to every AG-UI agent run:

```python
class RunAgentInput(BaseModel):
    """Input parameters for agent execution."""
    thread_id: str           # Conversation thread identifier
    run_id: str              # Unique run execution identifier
    messages: list[Message]  # Conversation history
    tools: list[Tool] | None # Available tools (frontend-defined)
    context: dict | None     # Optional contextual data
```

**Key Insight**: Tools are frontend-defined and passed to the agent during execution. This enables:
- Frontend control over available capabilities
- Dynamic tool addition/removal based on context
- Security via application-controlled sensitive operations

### 1.5 Message Types

AG-UI supports multiple participant roles:

```python
# User message (supports multimodal)
UserMessage(
    id="msg-123",
    role="user",
    content="What's in this image?",  # Can be str or list[InputContent]
)

# Assistant message (may include tool calls)
AssistantMessage(
    id="msg-124",
    role="assistant",
    content="I'll analyze that for you.",
    tool_calls=[ToolCall(...)]  # Optional
)

# Tool result message
ToolMessage(
    id="msg-125",
    role="tool",
    tool_call_id="tc-123",
    content="Tool execution result"
)

# System instruction
SystemMessage(
    id="msg-001",
    role="system",
    content="You are a helpful assistant."
)
```

### 1.6 State Management

AG-UI provides efficient bidirectional state synchronization:

**State Snapshots** - Complete state representations:
```python
yield StateSnapshotEvent(
    type=EventType.STATE_SNAPSHOT,
    snapshot={"steps": [...], "status": "in_progress"}
)
```

**State Deltas** - Incremental updates via JSON Patch (RFC 6902):
```python
import jsonpatch

previous_state = {"count": 1}
current_state = {"count": 2}
patch = jsonpatch.make_patch(previous_state, current_state)

yield StateDeltaEvent(
    type=EventType.STATE_DELTA,
    delta=patch.patch  # [{"op": "replace", "path": "/count", "value": 2}]
)
```

**Best Practices**:
- Deploy snapshots at run start and after error recovery
- Use deltas for incremental changes (bandwidth efficient)
- Apply patches atomically with error fallback to snapshot

### 1.7 Serialization and EventEncoder

The `EventEncoder` handles format negotiation based on HTTP Accept headers:

```python
from ag_ui.encoder import EventEncoder

# Create encoder from Accept header
accept_header = request.headers.get("accept")
encoder = EventEncoder(accept=accept_header)

# Encode events for streaming
async def event_generator():
    yield encoder.encode(RunStartedEvent(...))
    yield encoder.encode(TextMessageContentEvent(...))
    yield encoder.encode(RunFinishedEvent(...))

# Return streaming response
return StreamingResponse(
    event_generator(),
    media_type=encoder.get_content_type()
)
```

**Transport Options**:
- `text/event-stream`: SSE text format (default, debugging friendly)
- `application/octet-stream`: Binary protocol (high performance)

### 1.8 Implementation Patterns from Examples

#### Pattern 1: Agentic Chat (Text Streaming)

```python
async def agentic_chat_endpoint(input_data: RunAgentInput, request: Request):
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def event_generator():
        # 1. Emit run started
        yield encoder.encode(RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
        ))

        # 2. Stream text response
        message_id = str(uuid.uuid4())
        yield encoder.encode(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role="assistant"
        ))

        # Stream chunks from agent
        async for chunk in agent.execute(input_data.messages):
            yield encoder.encode(TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=chunk
            ))

        yield encoder.encode(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id
        ))

        # 3. Emit run finished
        yield encoder.encode(RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
        ))

    return StreamingResponse(event_generator(), media_type=encoder.get_content_type())
```

#### Pattern 2: Tool Calls with Incremental Arguments

```python
async def send_tool_call_events():
    tool_call_id = str(uuid.uuid4())
    tool_call_name = "search_knowledge_base"
    args = {"query": "return policy", "limit": 10}

    # 1. Start tool call
    yield ToolCallStartEvent(
        type=EventType.TOOL_CALL_START,
        tool_call_id=tool_call_id,
        tool_call_name=tool_call_name
    )

    # 2. Stream arguments (can be chunked for large args)
    yield ToolCallArgsEvent(
        type=EventType.TOOL_CALL_ARGS,
        tool_call_id=tool_call_id,
        delta=json.dumps(args)
    )

    # 3. End tool call
    yield ToolCallEndEvent(
        type=EventType.TOOL_CALL_END,
        tool_call_id=tool_call_id
    )
```

#### Pattern 3: Generative UI with State Updates

```python
async def generative_ui_endpoint(input_data: RunAgentInput, request: Request):
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def event_generator():
        yield encoder.encode(RunStartedEvent(...))

        # Initial state snapshot
        state = {"steps": [{"name": f"Step {i}", "status": "pending"} for i in range(5)]}
        yield encoder.encode(StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=state
        ))

        # Update steps progressively
        previous_state = copy.deepcopy(state)
        for i, step in enumerate(state["steps"]):
            step["status"] = "completed"
            patch = jsonpatch.make_patch(previous_state, state)
            yield encoder.encode(StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=patch.patch
            ))
            previous_state = copy.deepcopy(state)
            await asyncio.sleep(0.5)

        yield encoder.encode(RunFinishedEvent(...))

    return StreamingResponse(event_generator(), media_type=encoder.get_content_type())
```

#### Pattern 4: Backend Tool Rendering (Messages Snapshot)

```python
async def send_backend_tool_call_events(messages: list):
    tool_call_id = str(uuid.uuid4())

    # Create assistant message with tool call
    new_message = AssistantMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        tool_calls=[
            ToolCall(
                id=tool_call_id,
                type="function",
                function={
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"})
                }
            )
        ]
    )

    # Create tool result message
    result_message = ToolMessage(
        id=str(uuid.uuid4()),
        role="tool",
        content=json.dumps({"temperature": "72F", "conditions": "sunny"}),
        tool_call_id=tool_call_id
    )

    # Emit complete messages snapshot
    all_messages = list(messages) + [new_message, result_message]
    yield MessagesSnapshotEvent(
        type=EventType.MESSAGES_SNAPSHOT,
        messages=all_messages
    )
```

#### Pattern 5: Custom Events (Predictive State)

```python
async def predictive_state_events():
    # Emit custom event for frontend state prediction
    yield CustomEvent(
        type=EventType.CUSTOM,
        name="PredictState",
        value=[
            {
                "state_key": "document",
                "tool": "write_document",
                "tool_argument": "content"
            }
        ]
    )

    # Then emit tool call that will update predicted state
    yield ToolCallStartEvent(...)
```

### 1.9 HoloDeck Integration Mapping

| AG-UI Concept | HoloDeck Component | Implementation |
|--------------|-------------------|----------------|
| `RunAgentInput` | `ChatRequest` + session | Map session_id to thread_id, generate run_id |
| `messages` | `ChatSession.history` | Convert to AG-UI message format |
| `tools` | Agent tools from YAML | Expose via AG-UI tool definitions |
| Text streaming | `AgentExecutor.execute_turn()` | Wrap response stream in AG-UI events |
| Tool calls | Semantic Kernel tools | Map SK tool invocations to AG-UI events |
| State | Session metadata | Expose session state via StateSnapshotEvent |

### 1.10 Middleware Considerations

AG-UI middleware can transform, filter, and augment event streams:

```python
# Example: Logging middleware
class LoggingMiddleware:
    def process(self, event: BaseEvent) -> BaseEvent:
        logger.info(f"Event: {event.type}")
        return event

# Example: Tool filtering
class ToolFilterMiddleware:
    def __init__(self, allowed_tools: list[str]):
        self.allowed_tools = allowed_tools

    def process(self, event: BaseEvent) -> BaseEvent | None:
        if event.type == EventType.TOOL_CALL_START:
            if event.tool_call_name not in self.allowed_tools:
                return None  # Block this tool
        return event
```

**HoloDeck Application**: Use middleware for:
- Debug logging of all events
- Rate limiting event emission
- Filtering sensitive tool calls
- Adding observability metadata

---

## 2. FastAPI REST Protocol Design

### Decision: Follow HoloDeck API patterns from VISION.md

**Rationale**: VISION.md defines the expected REST API contract:
- Endpoint: `/agent/<agent-name>/chat` (sync) and `/agent/<agent-name>/chat/stream` (SSE)
- Request body: `{"message": "...", "session_id": "..."}`
- Response: JSON or SSE stream

**Alternatives Considered**:
1. **OpenAI-compatible API** - Rejected: Not aligned with HoloDeck's simpler model
2. **Custom protocol** - Rejected: Increases integration complexity

### REST Endpoint Structure

```
POST /agent/{agent_name}/chat
POST /agent/{agent_name}/chat/stream
GET  /health
GET  /ready
GET  /health/agent
DELETE /sessions/{session_id}
```

### Error Response Format

Per clarification: RFC 7807 Problem Details
```json
{
  "type": "https://holodeck.dev/errors/agent-not-found",
  "title": "Agent Not Found",
  "status": 404,
  "detail": "Agent 'xyz' is not loaded on this server"
}
```

---

## 3. Multimodal Input Support

### Decision: Support binary content in both AG-UI and REST protocols

**Rationale**:
- HoloDeck's Test-First principle (Constitution III) requires multimodal support
- AG-UI protocol natively supports `BinaryInputContent` for images, PDFs, documents
- REST protocol should provide equivalent functionality via multipart/form-data or base64

### AG-UI Multimodal Support

AG-UI supports multimodal inputs through `BinaryInputContent`:

```python
# AG-UI BinaryInputContent structure
{
    "type": "binary",
    "mimeType": "image/png",  # Required
    # One of the following (required):
    "data": "base64-encoded-content",  # Inline base64
    "url": "https://example.com/image.png",  # Remote URL
    "id": "file-123",  # Reference to uploaded file
    # Optional:
    "filename": "screenshot.png"
}
```

**Supported MIME Types** (aligned with HoloDeck file processor):
- Images: `image/png`, `image/jpeg`, `image/gif`, `image/webp`
- Documents: `application/pdf`, `application/vnd.openxmlformats-officedocument.*`
- Text: `text/plain`, `text/csv`, `text/markdown`

### REST Multimodal Support

For REST protocol, two approaches are supported:

**Option 1: Base64 in JSON** (simple, small files)
```json
{
    "message": "What's in this image?",
    "files": [
        {
            "content": "base64-encoded-data",
            "mime_type": "image/png",
            "filename": "screenshot.png"
        }
    ]
}
```

**Option 2: Multipart Form Data** (recommended for large files)
```
POST /agent/{agent_name}/chat
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="message"

What's in this image?
--boundary
Content-Disposition: form-data; name="files"; filename="image.png"
Content-Type: image/png

<binary data>
--boundary--
```

### Integration with Existing File Processor

HoloDeck's `lib/file_processor.py` already handles multimodal processing:
- Images → OCR via markitdown
- PDFs → Text extraction via pypdf
- Office documents → markitdown conversion

The server will leverage this existing infrastructure:

```python
from holodeck.lib.file_processor import FileProcessor

async def process_multimodal_request(message: str, files: list[FileContent]):
    processor = FileProcessor()
    processed_content = []

    for file in files:
        result = await processor.process(file.content, file.mime_type)
        processed_content.append(result)

    # Combine message with processed file content
    full_message = f"{message}\n\n{chr(10).join(processed_content)}"
    return full_message
```

### File Size Limits

| Context | Limit | Rationale |
|---------|-------|-----------|
| Base64 in JSON | 10 MB | Prevent JSON parsing issues |
| Multipart upload | 50 MB | Reasonable for local dev |
| Per-request total | 100 MB | Memory constraints |

**Implementation**: Use FastAPI's `UploadFile` with size validation middleware.

---

## 4. Session Management Architecture

### Decision: In-memory session store with 30-minute TTL

**Rationale**:
- Per clarification: Sessions expire after 30 minutes of inactivity
- Existing `ChatSessionManager` provides session lifecycle patterns
- In-memory is sufficient for local development (per spec assumptions)

**Implementation Approach**:
1. Extend or reuse `ChatSessionManager` for HTTP context
2. Use Python dict with `session_id` → `SessionData` mapping
3. Background task for TTL cleanup (asyncio periodic task)

**Alternatives Considered**:
1. **Redis sessions** - Deferred: Out of scope for local server spec
2. **File-based persistence** - Rejected: Unnecessary complexity for dev use case

### Session Data Structure

```python
@dataclass
class ServerSession:
    session_id: str
    agent_executor: AgentExecutor
    created_at: datetime
    last_activity: datetime
    message_count: int
```

---

## 5. Server Framework Selection

### Decision: FastAPI with Uvicorn

**Rationale**:
- FastAPI is already a HoloDeck architecture component (see VISION.md Deployment Engine)
- Native async support aligns with existing `AgentExecutor`
- Built-in OpenAPI documentation (FR-015)
- SSE streaming support via `StreamingResponse`

**Alternatives Considered**:
1. **Starlette only** - Rejected: Loses OpenAPI auto-generation
2. **Flask** - Rejected: No native async, poor streaming support
3. **aiohttp** - Rejected: Less ecosystem integration

### Dependency Addition

```toml
# pyproject.toml additions
"fastapi>=0.115.0,<1.0.0",
"uvicorn[standard]>=0.34.0,<1.0.0",
"ag-ui-protocol>=0.1.10,<1.0.0",
```

---

## 6. Protocol Adapter Pattern

### Decision: Use adapter pattern for protocol abstraction

**Rationale**: Clean separation between agent execution and protocol encoding enables:
- Single agent execution path regardless of protocol
- Easy addition of future protocols
- Testable in isolation

### Architecture

```
CLI (holodeck serve)
    ↓
ServerFactory (creates appropriate server)
    ↓
┌─────────────────────────────────────────┐
│         AgentServer (FastAPI app)        │
│  ┌─────────────────┬─────────────────┐  │
│  │  AGUIProtocol   │  RESTProtocol   │  │
│  │  (EventEncoder) │  (JSON/SSE)     │  │
│  └────────┬────────┴────────┬────────┘  │
│           ↓                  ↓           │
│     AgentEndpointHandler                 │
│           ↓                              │
│     SessionManager                       │
│           ↓                              │
│     AgentExecutor (existing)             │
└─────────────────────────────────────────┘
```

---

## 7. Logging and Observability

### Decision: Structured logging with request metadata

**Rationale**: Per clarification:
- Default: Log timestamp, session_id, endpoint, latency
- Debug mode: Full request/response content

**Implementation**:
- Use existing `holodeck.lib.logging_config` infrastructure
- Add request/response middleware for metadata capture
- Use structured JSON logging format

---

## 8. CORS Configuration

### Decision: FastAPI CORSMiddleware with configurable origins

**Rationale**: Per clarification:
- Default: Allow all origins (`*`) for local development
- Configurable via `--cors-origins` flag

**Implementation**:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 9. Graceful Shutdown

### Decision: Signal handlers + asyncio cleanup

**Rationale**: FR-013 requires graceful shutdown on SIGINT/SIGTERM

**Implementation**:
1. Register signal handlers in Uvicorn server
2. On signal: stop accepting new requests
3. Wait for in-flight requests (with timeout)
4. Cleanup all sessions (call `AgentExecutor.shutdown()`)
5. Exit cleanly

---

## 10. CLI Command Design

### Decision: `holodeck serve` with protocol option

**Rationale**: Follows existing CLI patterns (Click-based)

**Command Signature**:
```bash
holodeck serve <agent.yaml> [OPTIONS]

Options:
  --port INTEGER          Server port (default: 8000)
  --protocol [ag-ui|rest] Protocol to use (default: ag-ui)
  --cors-origins TEXT     CORS allowed origins (default: *)
  --debug                 Enable debug logging
  --open                  Open browser on startup
```

---

## 11. File Structure

### New Files Required

```
src/holodeck/
├── serve/                      # New module
│   ├── __init__.py
│   ├── server.py               # AgentServer class
│   ├── protocols/
│   │   ├── __init__.py
│   │   ├── base.py             # Protocol ABC
│   │   ├── agui.py             # AG-UI protocol adapter
│   │   └── rest.py             # REST protocol adapter
│   ├── session_store.py        # In-memory session management
│   ├── middleware.py           # Logging, CORS, error handling
│   └── models.py               # Request/Response Pydantic models
├── cli/
│   └── commands/
│       └── serve.py            # New CLI command
```

---

## Summary of Decisions

| Topic | Decision | Key Dependency |
|-------|----------|----------------|
| AG-UI SDK | `ag-ui-protocol>=0.1.10` | PyPI package |
| AG-UI Events | 16+ event types in 5 categories | See Section 1.3 |
| AG-UI Patterns | Start-Content-End, Snapshot-Delta | See Section 1.8 |
| Web Framework | FastAPI + Uvicorn | `fastapi`, `uvicorn` |
| Multimodal Input | Base64 JSON + multipart form-data | Existing FileProcessor |
| Session Storage | In-memory with 30-min TTL | None (stdlib) |
| Error Format | RFC 7807 Problem Details | Built-in |
| Logging | Structured metadata + debug mode | Existing logging |
| Protocol Pattern | Adapter for AG-UI and REST | Custom code |
| CORS | Configurable, default `*` | FastAPI middleware |

## Key AG-UI Implementation References

| Pattern | Example File | HoloDeck Use Case |
|---------|-------------|-------------------|
| Agentic Chat | `agentic_chat.py` | Basic agent responses |
| Tool Calls | `human_in_the_loop.py` | HoloDeck tool execution |
| State Updates | `agentic_generative_ui.py` | Session metadata sync |
| Backend Tools | `backend_tool_rendering.py` | Server-side tool results |
| Custom Events | `predictive_state_updates.py` | Extended functionality |
| Shared State | `shared_state.py` | Complex state snapshots |
