# Research: Agent Local Server

**Feature**: 017-agent-local-server
**Date**: 2025-12-29

## Executive Summary

This research addresses the technical decisions required to implement a local agent server with two protocol options: AG-UI (default) and FastAPI REST. The findings are based on analysis of the AG-UI protocol specification, sample implementations, and the existing HoloDeck codebase.

---

## 1. AG-UI Protocol Integration

### Decision: Use `ag-ui-protocol` Python SDK

**Rationale**: The official `ag-ui-protocol` package (v0.1.10) provides:
- Strongly-typed Pydantic models for all 16 event types
- Built-in `EventEncoder` for Server-Sent Events (SSE) streaming
- Format negotiation based on HTTP Accept headers
- Full Python 3.9+ compatibility (HoloDeck requires 3.10+)

**Alternatives Considered**:
1. **Implement protocol from scratch** - Rejected: Unnecessary duplication, higher maintenance burden
2. **Use raw JSON SSE** - Rejected: Loses type safety and format negotiation benefits

### Key AG-UI Events to Implement

Based on sample analysis, the minimum viable event set for HoloDeck:

| Event Type | Purpose | HoloDeck Mapping |
|------------|---------|------------------|
| `RunStartedEvent` | Signals workflow start | Session initialization |
| `RunFinishedEvent` | Signals workflow completion | Session end |
| `TextMessageStartEvent` | Opens message stream | Response start |
| `TextMessageContentEvent` | Streams text chunks | Agent response chunks |
| `TextMessageEndEvent` | Closes message stream | Response complete |
| `ToolCallStartEvent` | Tool invocation start | Tool execution start |
| `ToolCallArgsEvent` | Tool arguments | Tool parameters |
| `ToolCallEndEvent` | Tool invocation complete | Tool execution end |
| `StateSnapshotEvent` | Session state sync | Session metadata |

### Sample Implementation Pattern

From `agentic_chat.py`:
```python
from ag_ui.core import (
    RunAgentInput, EventType, RunStartedEvent, RunFinishedEvent,
    TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent,
)
from ag_ui.encoder import EventEncoder

async def event_generator():
    yield RunStartedEvent(thread_id=thread_id, run_id=run_id)
    yield TextMessageStartEvent(message_id=msg_id, role="assistant")
    for chunk in agent_response:
        yield TextMessageContentEvent(message_id=msg_id, delta=chunk)
    yield TextMessageEndEvent(message_id=msg_id)
    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)

return StreamingResponse(
    EventEncoder(accept_header).encode_events(event_generator()),
    media_type=EventEncoder(accept_header).content_type
)
```

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
| Web Framework | FastAPI + Uvicorn | `fastapi`, `uvicorn` |
| Multimodal Input | Base64 JSON + multipart form-data | Existing FileProcessor |
| Session Storage | In-memory with 30-min TTL | None (stdlib) |
| Error Format | RFC 7807 Problem Details | Built-in |
| Logging | Structured metadata + debug mode | Existing logging |
| Protocol Pattern | Adapter for AG-UI and REST | Custom code |
| CORS | Configurable, default `*` | FastAPI middleware |
