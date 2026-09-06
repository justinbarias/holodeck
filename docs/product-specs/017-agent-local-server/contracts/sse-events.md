# Server-Sent Events (SSE) Contract

**Feature**: 017-agent-local-server
**Protocol**: REST streaming (`/agent/{agent_name}/chat/stream`)

## Overview

The streaming endpoint uses Server-Sent Events (SSE) to deliver real-time agent responses. This document defines the event types, data formats, and expected behavior.

**Multimodal Support**: The streaming endpoint accepts files via JSON (base64) or multipart form-data. Files are processed server-side (OCR, text extraction) before the agent is invoked. The SSE event format remains the same regardless of whether files were included in the request.

## Event Format

All events follow the SSE specification:

```
event: <event_type>
data: <json_payload>

```

Note the empty line after `data:` which terminates each event.

---

## Event Types

### 1. `stream_start`

Sent at the beginning of a response stream.

```
event: stream_start
data: {"session_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV", "message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID (ULID) |
| `message_id` | string | Message ID for this response (ULID) |

---

### 2. `message_delta`

Sent for each chunk of the agent's response text.

```
event: message_delta
data: {"delta": "Our return policy", "message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}
```

| Field | Type | Description |
|-------|------|-------------|
| `delta` | string | Text chunk (may be partial word) |
| `message_id` | string | Message ID reference |

---

### 3. `tool_call_start`

Sent when the agent begins invoking a tool.

```
event: tool_call_start
data: {"tool_call_id": "tc_01", "name": "search_knowledge_base", "message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW"}
```

| Field | Type | Description |
|-------|------|-------------|
| `tool_call_id` | string | Unique tool call identifier |
| `name` | string | Tool name |
| `message_id` | string | Parent message ID |

---

### 4. `tool_call_args`

Sent to stream tool call arguments (may be chunked for large arguments).

```
event: tool_call_args
data: {"tool_call_id": "tc_01", "args_delta": "{\"query\": \"return"}
```

| Field | Type | Description |
|-------|------|-------------|
| `tool_call_id` | string | Tool call identifier |
| `args_delta` | string | JSON fragment of arguments |

---

### 5. `tool_call_end`

Sent when tool invocation completes.

```
event: tool_call_end
data: {"tool_call_id": "tc_01", "status": "success"}
```

| Field | Type | Description |
|-------|------|-------------|
| `tool_call_id` | string | Tool call identifier |
| `status` | string | "success" or "error" |

---

### 6. `stream_end`

Sent at the end of a successful response stream.

```
event: stream_end
data: {"message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW", "tokens_used": {"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225}, "execution_time_ms": 1250}
```

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | string | Message ID |
| `tokens_used` | object | Token usage statistics |
| `execution_time_ms` | integer | Total processing time |

---

### 7. `error`

Sent when an error occurs during streaming.

```
event: error
data: {"type": "https://holodeck.dev/errors/agent-error", "title": "Agent Error", "status": 500, "detail": "LLM provider timeout"}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Error type URI |
| `title` | string | Short description |
| `status` | integer | HTTP-equivalent status |
| `detail` | string | Detailed message |

---

## Event Sequence

### Successful Response (no tools)

```
1. stream_start
2. message_delta (repeated)
3. stream_end
```

### Successful Response (with tools)

```
1. stream_start
2. tool_call_start
3. tool_call_args (repeated)
4. tool_call_end
5. message_delta (repeated)
6. stream_end
```

### Error During Stream

```
1. stream_start
2. message_delta (0 or more)
3. error
```

---

## Client Implementation Notes

### JavaScript/TypeScript

```javascript
const eventSource = new EventSource('/agent/support/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello' })
});

eventSource.addEventListener('message_delta', (e) => {
  const data = JSON.parse(e.data);
  appendToUI(data.delta);
});

eventSource.addEventListener('stream_end', (e) => {
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  showError(data.detail);
  eventSource.close();
});
```

### Python

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream(
        'POST',
        'http://localhost:8000/agent/support/chat/stream',
        json={'message': 'Hello'}
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith('event:'):
                event_type = line[7:].strip()
            elif line.startswith('data:'):
                data = json.loads(line[5:])
                handle_event(event_type, data)
```

---

## Connection Handling

| Scenario | Behavior |
|----------|----------|
| Client disconnects | Server stops generation, cleans up |
| Server restart | Client receives error, should reconnect |
| Timeout (30s no events) | Server sends keepalive comment |
| Network error | Client should implement retry with backoff |

### Keepalive

To prevent proxy/load balancer timeouts, the server sends a comment every 15 seconds during long-running operations:

```
: keepalive
```

(Lines starting with `:` are SSE comments, ignored by clients)
