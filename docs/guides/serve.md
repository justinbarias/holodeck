# Agent Server Guide

Deploy a HoloDeck agent as an HTTP server with `holodeck serve`.

## Quick start

Serve a Claude-backend agent and hit its endpoints.

```yaml
# agent.yaml
name: research
model:
  provider: anthropic
  name: claude-sonnet-4-5
instructions:
  inline: "You are a helpful research assistant."
```

```bash
holodeck serve agent.yaml
# Server listening at http://127.0.0.1:8000 (AG-UI protocol)
```

```bash
# Health check
curl http://127.0.0.1:8000/health
# {"status":"healthy","agent_name":"research","agent_ready":true,...}  → HTTP 200

# Chat via the AG-UI endpoint
curl -X POST http://127.0.0.1:8000/awp -H 'content-type: application/json' \
  -d '{"threadId":"t1","runId":"r1","state":{},"messages":[{"id":"m1","role":"user","content":"Hello"}],"tools":[],"context":[],"forwardedProps":{}}'
# → HTTP 200, streamed AG-UI events
```

!!! info "Backend support"
    `holodeck serve` is fully supported on the Claude backend (`provider: anthropic`) and Ollama. On the OpenAI Agents backend (`provider: openai` / `azure_openai`) serve is **on the roadmap** — see the [OpenAI Backend guide](openai-backend.md).

## How it works

`holodeck serve` wraps your configured agent in an HTTP server exposing two protocols: **AG-UI** (default, for frontends like CopilotKit) and **REST** (`--protocol rest`, JSON/SSE endpoints). Both stream responses, manage sessions, accept multimodal uploads, and expose `/health` and `/ready` for orchestrators. The AG-UI `thread_id` maps directly to a HoloDeck `session_id` so conversations persist across requests. Pick AG-UI when wiring a web frontend; pick REST for traditional service-to-service calls.

## CLI reference

```bash
holodeck serve <agent_config> [OPTIONS]
```

| Option           | Description                      | Default                 |
| ---------------- | -------------------------------- | ----------------------- |
| `agent_config`   | Path to agent.yaml               | `agent.yaml`            |
| `--port, -p`     | Port to listen on                | `8000`                  |
| `--host, -h`     | Host to bind to                  | `127.0.0.1`             |
| `--protocol`     | `ag-ui` or `rest`                | `ag-ui`                 |
| `--cors-origins` | Comma-separated CORS origins     | `http://localhost:3000` |
| `--verbose, -v`  | Verbose debug logging            | `false`                 |
| `--quiet, -q`    | Suppress INFO logging            | `false`                 |

```bash
# Production: all interfaces, custom CORS, quiet
holodeck serve agent.yaml --host 0.0.0.0 --port 8080 \
  --cors-origins "https://myapp.com" --quiet
```

---

## AG-UI protocol (default)

AG-UI integrates with AI agent frontends like CopilotKit and the Vercel AI SDK.

```
POST /awp
```

Accepts `RunAgentInput` from the AG-UI specification and streams protocol events back. The AG-UI `thread_id` maps directly to a HoloDeck `session_id` for conversation continuity.

### CopilotKit integration

Point a CopilotKit `HttpAgent` at the `/awp` endpoint:

```typescript
// src/app/api/copilotkit/route.ts
import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const runtime = new CopilotRuntime({
  agents: {
    // Agent name must match your HoloDeck agent name
    research: new HttpAgent({ url: "http://127.0.0.1:8000/awp" }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
```

```typescript
// src/app/layout.tsx — wrap the app
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit" agent="research">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
```

Run both servers (`holodeck serve agent.yaml` and `npm run dev`), then open http://localhost:3000. Install the client deps with `npm install @copilotkit/react-core @copilotkit/react-ui @copilotkit/runtime @ag-ui/client`.

---

## REST protocol

Run with `--protocol rest` to expose traditional REST endpoints.

| Endpoint                                       | Description                          |
| ---------------------------------------------- | ------------------------------------ |
| `POST /agent/{name}/chat`                      | Synchronous chat (JSON)              |
| `POST /agent/{name}/chat/stream`               | Streaming chat (SSE)                 |
| `POST /agent/{name}/chat/multipart`            | Chat with file uploads               |
| `POST /agent/{name}/chat/stream/multipart`     | Streaming chat with file uploads     |
| `DELETE /sessions/{session_id}`                | Delete session (`204 No Content`)    |
| `GET /docs`                                    | Interactive Swagger UI               |

### Synchronous chat

```
POST /agent/{agent_name}/chat
{ "message": "Hello", "session_id": "01HQXYZ..." }   // session_id optional
```

```json
{
  "message_id": "01HQABC...",
  "content": "I'm a helpful assistant...",
  "session_id": "01HQXYZ...",
  "tool_calls": [],
  "tokens_used": { "prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225 },
  "execution_time_ms": 1250
}
```

### Streaming chat (SSE)

`POST /agent/{agent_name}/chat/stream` returns a Server-Sent Events stream:

```
event: stream_start
data: {"session_id": "01HQXYZ...", "message_id": "01HQABC..."}

event: message_delta
data: {"delta": "Quantum computing is", "message_id": "01HQABC..."}

event: stream_end
data: {"message_id": "01HQABC...", "tokens_used": {...}, "execution_time_ms": 2500}
```

Keepalive comments (`:`) are sent every 15 seconds to prevent connection timeout.

### File uploads

Multipart endpoints accept up to 10 files per request (max 50MB/file, 100MB/request). Supported types: images (PNG, JPEG, GIF, WebP), PDF, Office (DOCX, XLSX, PPTX), text (TXT, CSV, Markdown).

---

## Health endpoints

```
GET /health   → {"status":"healthy","agent_name":"research","agent_ready":true,"active_sessions":5,"uptime_seconds":3600.5}
GET /ready     → {"ready": true}
```

`/ready` is intended for load balancers and container orchestrators.

---

## Sessions

Sessions are created automatically on first request and identified by a ULID. Defaults: 30-minute inactivity TTL, 1000 concurrent sessions, cleanup every 5 minutes. Each session holds conversation history, the agent executor, timestamps, and a message count. Include `session_id` in a request to continue a conversation; an expired session is replaced by a new one.

---

## CORS

```bash
holodeck serve agent.yaml --cors-origins "http://localhost:3000,https://myapp.com"
holodeck serve agent.yaml --host 0.0.0.0 --cors-origins "*"   # dev only
```

---

## SSE event types

| Event             | Description              | Data                               |
| ----------------- | ------------------------ | ---------------------------------- |
| `stream_start`    | Stream begins            | `session_id`, `message_id`         |
| `message_delta`   | Text chunk               | `delta`, `message_id`              |
| `tool_call_start` | Tool invocation begins   | `tool_call_id`, `name`             |
| `tool_call_args`  | Tool arguments (chunked) | `tool_call_id`, `args_delta`       |
| `tool_call_end`   | Tool completes           | `tool_call_id`, `status`           |
| `stream_end`      | Stream completes         | `tokens_used`, `execution_time_ms` |
| `error`           | Error occurred           | RFC 7807 problem details           |

---

## Troubleshooting

Errors are returned in RFC 7807 Problem Details format:

```json
{
  "type": "https://holodeck.dev/errors/invalid-request",
  "title": "Invalid Request",
  "status": 400,
  "detail": "Message cannot be empty",
  "instance": "/agent/research/chat"
}
```

| Status | Type                  | Description                |
| ------ | --------------------- | -------------------------- |
| 400    | `invalid-request`     | Malformed request body     |
| 404    | `not-found`           | Agent or session not found |
| 503    | `service-unavailable` | Agent not ready            |
| 500    | `internal-error`      | Unexpected server error    |

- **`503 service-unavailable`** — the agent is still initializing. Poll `/ready` until it returns `{"ready": true}`.
- **Serving an OpenAI-provider agent fails** — serve is not yet supported on the OpenAI Agents backend. Use a Claude-backend agent for now; see the [OpenAI Backend guide](openai-backend.md).

---

## Next steps

- [Agent Configuration Guide](agent-configuration.md) — agent.yaml reference
- [Tools Guide](tools.md) — adding tools to your agent
- [Deployment Guide](deployment.md) — ship the server as a container
- [Observability Guide](observability.md) — tracing and monitoring
