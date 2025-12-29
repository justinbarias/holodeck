# Quickstart: Agent Local Server

**Feature**: 017-agent-local-server

This guide shows how to serve a HoloDeck agent locally using the new `holodeck serve` command.

---

## Prerequisites

- HoloDeck installed (`pip install holodeck-ai`)
- A valid agent configuration file (e.g., `agent.yaml`)
- Agent tested with `holodeck chat` or `holodeck test`

---

## Basic Usage

### Start with AG-UI Protocol (Default)

```bash
holodeck serve agent.yaml
```

Output:
```
🚀 Starting HoloDeck Agent Server...

Agent: customer-support
Protocol: ag-ui
Server: http://localhost:8000

AG-UI endpoint ready for CopilotKit or compatible clients.
Press Ctrl+C to stop.
```

### Start with REST Protocol

```bash
holodeck serve agent.yaml --protocol rest
```

Output:
```
🚀 Starting HoloDeck Agent Server...

Agent: customer-support
Protocol: rest
Server: http://localhost:8000

Endpoints:
  POST /agent/customer-support/chat        - Synchronous chat
  POST /agent/customer-support/chat/stream - Streaming chat (SSE)
  GET  /health                             - Health check
  GET  /ready                              - Readiness check

OpenAPI docs: http://localhost:8000/docs
Press Ctrl+C to stop.
```

---

## CLI Options

```bash
holodeck serve <agent.yaml> [OPTIONS]

Options:
  --port INTEGER          Server port (default: 8000)
  --protocol [ag-ui|rest] Protocol to use (default: ag-ui)
  --cors-origins TEXT     CORS allowed origins (default: *)
  --debug                 Enable debug logging
  --open                  Open browser on startup
  --help                  Show this message and exit
```

### Examples

```bash
# Custom port
holodeck serve agent.yaml --port 3000

# REST with specific CORS origins
holodeck serve agent.yaml --protocol rest --cors-origins "http://localhost:3000,https://myapp.com"

# Debug mode with browser
holodeck serve agent.yaml --protocol rest --debug --open
```

---

## REST API Usage

### Synchronous Chat

```bash
# Start new conversation
curl -X POST http://localhost:8000/agent/customer-support/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'

# Response
{
  "message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "content": "Our return policy allows returns within 30 days...",
  "session_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
  "tool_calls": [],
  "execution_time_ms": 1250
}

# Continue conversation
curl -X POST http://localhost:8000/agent/customer-support/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What about electronics?", "session_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}'
```

### Streaming Chat

```bash
curl -X POST http://localhost:8000/agent/customer-support/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about your products"}'

# SSE events
event: stream_start
data: {"session_id": "01ARZ3...", "message_id": "01ARZ3..."}

event: message_delta
data: {"delta": "We offer a wide", "message_id": "01ARZ3..."}

event: message_delta
data: {"delta": " range of products", "message_id": "01ARZ3..."}

event: stream_end
data: {"message_id": "01ARZ3...", "execution_time_ms": 1500}
```

### Multimodal Chat (with Files)

The server supports multimodal inputs including images, PDFs, and Office documents.

**Option 1: Base64 in JSON**

```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -i screenshot.png)

# Send with message
curl -X POST http://localhost:8000/agent/customer-support/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"What's in this image?\",
    \"files\": [{
      \"content\": \"$IMAGE_BASE64\",
      \"mime_type\": \"image/png\",
      \"filename\": \"screenshot.png\"
    }]
  }"
```

**Option 2: Multipart Form Data (recommended for large files)**

```bash
# Upload files directly
curl -X POST http://localhost:8000/agent/customer-support/chat \
  -F "message=Summarize this document" \
  -F "files=@report.pdf"

# Multiple files
curl -X POST http://localhost:8000/agent/customer-support/chat \
  -F "message=Compare these documents" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf"
```

**Supported File Types:**
- Images: PNG, JPEG, GIF, WebP (OCR extraction)
- Documents: PDF (text extraction)
- Office: DOCX, XLSX, PPTX (markdown conversion)
- Text: TXT, CSV, Markdown

**File Limits:**
- Max 10 files per request
- Max 50MB per file
- Max 100MB total per request

---

### Health Checks

```bash
# Server health
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "agent_name": "customer-support",
  "agent_ready": true,
  "active_sessions": 2,
  "uptime_seconds": 3600.5
}

# Readiness (for load balancers)
curl http://localhost:8000/ready

# Response
{"ready": true}
```

### Delete Session

```bash
curl -X DELETE http://localhost:8000/sessions/01ARZ3NDEKTSV4RRFFQ69G5FAW
# Returns 204 No Content
```

---

## AG-UI Protocol Usage

When using AG-UI protocol, connect with a compatible client like CopilotKit:

```typescript
import { CopilotKit } from "@copilotkit/react-core";

function App() {
  return (
    <CopilotKit runtimeUrl="http://localhost:8000">
      <YourApp />
    </CopilotKit>
  );
}
```

---

## Session Management

- Sessions are created automatically on first request
- Sessions expire after 30 minutes of inactivity
- Provide `session_id` to continue a conversation
- Delete sessions explicitly when done (optional)

---

## Error Handling

Errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://holodeck.dev/errors/invalid-request",
  "title": "Invalid Request",
  "status": 400,
  "detail": "The 'message' field is required and cannot be empty"
}
```

Common errors:
- `400` - Invalid request (missing/invalid fields)
- `404` - Agent not found
- `503` - Agent not ready (still initializing)

---

## Debugging

Enable debug mode for detailed logging:

```bash
holodeck serve agent.yaml --debug
```

Debug mode logs:
- Full request/response content
- Tool invocations with arguments
- Token usage per request
- Timing information

---

## Next Steps

- [OpenAPI Documentation](/docs) - Interactive API explorer (REST only)
- [AG-UI Protocol Docs](https://docs.ag-ui.com) - Full protocol specification
- [CopilotKit Integration](https://docs.copilotkit.ai) - React UI components
