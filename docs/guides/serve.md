# Agent Server Guide

This guide explains how to deploy HoloDeck agents as HTTP servers using the `holodeck serve` command.

## Overview

`holodeck serve` exposes your configured agent as an HTTP server with two protocol options:

- **AG-UI Protocol** (default) - Standard protocol for AI agent frontends like CopilotKit
- **REST API Protocol** - Traditional REST endpoints with JSON/SSE responses

Both protocols support streaming responses, session management, multimodal file uploads, and health checks.

## Quick Start

```bash
# Start server with AG-UI protocol (default)
holodeck serve agent.yaml

# Start server with REST API
holodeck serve agent.yaml --protocol rest

# Custom port and host
holodeck serve agent.yaml --port 3000 --host 0.0.0.0
```

The server displays startup information:

```
============================================================
  HoloDeck Agent Server
============================================================

  Agent:    research
  Protocol: ag-ui
  URL:      http://127.0.0.1:8000

  Endpoints:
    POST /awp                    AG-UI Protocol
    GET  /health                 Health Check
    GET  /ready                  Readiness Check

  Press Ctrl+C to stop
============================================================
```

## CLI Reference

```bash
holodeck serve <agent_config> [OPTIONS]
```

### Arguments

| Argument       | Description                           | Default      |
| -------------- | ------------------------------------- | ------------ |
| `agent_config` | Path to agent.yaml configuration file | `agent.yaml` |

### Options

| Option           | Description                      | Default                 |
| ---------------- | -------------------------------- | ----------------------- |
| `--port, -p`     | Port to listen on                | `8000`                  |
| `--host, -h`     | Host to bind to                  | `127.0.0.1`             |
| `--protocol`     | Protocol type: `ag-ui` or `rest` | `ag-ui`                 |
| `--cors-origins` | Comma-separated CORS origins     | `http://localhost:3000` |
| `--verbose, -v`  | Enable verbose debug logging     | `false`                 |
| `--quiet, -q`    | Suppress INFO logging output     | `false`                 |

### Examples

```bash
# Development with verbose logging
holodeck serve agent.yaml --verbose

# Production with all interfaces
holodeck serve agent.yaml --host 0.0.0.0 --port 8080

# REST API with custom CORS
holodeck serve agent.yaml --protocol rest --cors-origins "http://localhost:3000,https://myapp.com"
```

---

## AG-UI Protocol

AG-UI is the default protocol, designed for integration with AI agent frontends like CopilotKit, Vercel AI SDK, and similar frameworks.

### Endpoint

```
POST /awp
```

Accepts `RunAgentInput` from the AG-UI specification and streams protocol events back to the client.

### Architecture

```
┌─────────────────────────┐
│  Web Frontend           │
│  (CopilotKit, etc.)     │
└────────────┬────────────┘
             │ AG-UI Protocol
             ▼
┌─────────────────────────┐
│  HoloDeck Server        │
│  POST /awp              │
└────────────┬────────────┘
             │ LLM + Tools
             ▼
┌─────────────────────────┐
│  Agent Execution        │
│  (Semantic Kernel)      │
└─────────────────────────┘
```

### Thread/Session Mapping

AG-UI `thread_id` maps directly to HoloDeck `session_id` for conversation continuity.

---

## REST API Protocol

When running with `--protocol rest`, traditional REST endpoints are exposed.

### Chat Endpoints

#### Synchronous Chat

```
POST /agent/{agent_name}/chat
Content-Type: application/json

{
  "message": "Hello, how can you help me?",
  "session_id": "01HQXYZ..."  // Optional, auto-generated if omitted
}
```

Response:

```json
{
  "message_id": "01HQABC...",
  "content": "I'm a helpful assistant...",
  "session_id": "01HQXYZ...",
  "tool_calls": [],
  "tokens_used": {
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
  },
  "execution_time_ms": 1250
}
```

#### Streaming Chat (SSE)

```
POST /agent/{agent_name}/chat/stream
Content-Type: application/json

{
  "message": "Explain quantum computing",
  "session_id": "01HQXYZ..."
}
```

Response: Server-Sent Events stream

```
event: stream_start
data: {"session_id": "01HQXYZ...", "message_id": "01HQABC..."}

event: message_delta
data: {"delta": "Quantum computing is", "message_id": "01HQABC..."}

event: message_delta
data: {"delta": " a type of computation...", "message_id": "01HQABC..."}

event: stream_end
data: {"message_id": "01HQABC...", "tokens_used": {...}, "execution_time_ms": 2500}
```

#### Multipart File Upload

```
POST /agent/{agent_name}/chat/multipart
Content-Type: multipart/form-data

message: "What's in this image?"
files: <binary file data>
session_id: "01HQXYZ..."  // Optional
```

Supports up to 10 files per request with the following limits:

- Max 50MB per file
- Max 100MB total per request

**Supported file types:**

- Images: PNG, JPEG, GIF, WebP
- Documents: PDF
- Office: DOCX, XLSX, PPTX
- Text: TXT, CSV, Markdown

#### Streaming with Multipart

```
POST /agent/{agent_name}/chat/stream/multipart
```

Combines streaming responses with multipart file upload.

### Session Management

#### Delete Session

```
DELETE /sessions/{session_id}
```

Removes session and conversation history. Returns `204 No Content`.

### Health Endpoints

#### Health Check

```
GET /health
```

```json
{
  "status": "healthy",
  "agent_name": "research",
  "agent_ready": true,
  "active_sessions": 5,
  "uptime_seconds": 3600.5
}
```

#### Readiness Check

```
GET /ready
```

```json
{
  "ready": true
}
```

Used by load balancers and container orchestrators.

### API Documentation

```
GET /docs
```

Interactive Swagger UI for testing endpoints (REST protocol only).

---

## CopilotKit Integration

HoloDeck integrates seamlessly with [CopilotKit](https://copilotkit.ai) using the AG-UI protocol.

### Architecture

```
┌─────────────────────────┐
│  CopilotKit Frontend    │
│  (Next.js React App)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Next.js API Route                  │
│  /api/copilotkit                    │
│  HttpAgent → http://127.0.0.1:8000  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  HoloDeck AG-UI Server              │
│  POST /awp                          │
└─────────────────────────────────────┘
```

### Setup

#### 1. Start HoloDeck Server

```bash
holodeck serve agent.yaml
# Server starts at http://127.0.0.1:8000
```

#### 2. Create Next.js Project

```bash
npx create-next-app@latest my-copilot --typescript --tailwind
cd my-copilot
```

#### 3. Install Dependencies

```bash
npm install @copilotkit/react-core @copilotkit/react-ui @copilotkit/runtime @ag-ui/client
```

#### 4. Create API Route

Create `src/app/api/copilotkit/route.ts`:

```typescript
import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
  agents: {
    // Agent name must match your HoloDeck agent name
    research: new HttpAgent({ url: "http://127.0.0.1:8000/awp" }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
```

#### 5. Create Layout with Provider

Update `src/app/layout.tsx`:

```typescript
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
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

#### 6. Create Chat Page

Create `src/app/page.tsx`:

```typescript
"use client";

import { CopilotChat } from "@copilotkit/react-ui";

export default function Home() {
  return (
    <main className="h-screen">
      <CopilotChat
        labels={{
          title: "Research Assistant",
          initial: "Hi! I'm your research assistant. How can I help?",
        }}
      />
    </main>
  );
}
```

#### 7. Run Both Servers

```bash
# Terminal 1: HoloDeck
holodeck serve agent.yaml

# Terminal 2: Next.js
npm run dev
```

Open http://localhost:3000 to chat with your agent.

---

## Session Management

HoloDeck maintains conversation sessions with automatic cleanup.

### Session Behavior

- **Auto-creation**: Sessions are created automatically on first request
- **Session ID format**: ULID (Universally Unique Lexicographically Sortable Identifier)
- **TTL**: 30 minutes of inactivity
- **Max sessions**: 1000 concurrent sessions
- **Cleanup interval**: Every 5 minutes

### Session Data

Each session maintains:

- Conversation history
- Agent executor instance
- Created/last activity timestamps
- Message count

### Continuing Conversations

Include `session_id` in requests to continue a conversation:

```json
{
  "message": "Tell me more about that",
  "session_id": "01HQXYZ..."
}
```

If the session has expired, a new session is automatically created.

---

## Error Handling

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

### Error Types

| Status | Type                  | Description                |
| ------ | --------------------- | -------------------------- |
| 400    | `invalid-request`     | Malformed request body     |
| 404    | `not-found`           | Agent or session not found |
| 503    | `service-unavailable` | Agent not ready            |
| 500    | `internal-error`      | Unexpected server error    |

---

## SSE Event Types

For streaming endpoints, the following event types are emitted:

| Event             | Description              | Data                               |
| ----------------- | ------------------------ | ---------------------------------- |
| `stream_start`    | Stream begins            | `session_id`, `message_id`         |
| `message_delta`   | Text chunk               | `delta`, `message_id`              |
| `tool_call_start` | Tool invocation begins   | `tool_call_id`, `name`             |
| `tool_call_args`  | Tool arguments (chunked) | `tool_call_id`, `args_delta`       |
| `tool_call_end`   | Tool completes           | `tool_call_id`, `status`           |
| `stream_end`      | Stream completes         | `tokens_used`, `execution_time_ms` |
| `error`           | Error occurred           | RFC 7807 problem details           |

Keepalive comments (`:`) are sent every 15 seconds to prevent connection timeout.

---

## CORS Configuration

Configure allowed origins for cross-origin requests:

```bash
# Single origin
holodeck serve agent.yaml --cors-origins "http://localhost:3000"

# Multiple origins
holodeck serve agent.yaml --cors-origins "http://localhost:3000,https://myapp.com"

# All interfaces for development
holodeck serve agent.yaml --host 0.0.0.0 --cors-origins "*"
```

---

## Complete Examples

### Basic AG-UI Server

```yaml
# agent.yaml
name: assistant
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "You are a helpful assistant."
```

```bash
holodeck serve agent.yaml
```

### REST API with Tools

```yaml
# agent.yaml
name: research
model:
  provider: ollama
  name: llama3.2:latest
instructions:
  file: instructions/research.md
tools:
  - type: vectorstore
    name: search_docs
    store: chroma
    collection: research_papers
```

```bash
holodeck serve agent.yaml --protocol rest --port 8080
```

### Production Deployment

```bash
# With all interfaces exposed
holodeck serve agent.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --cors-origins "https://myapp.com" \
  --quiet

# With Docker
docker run -p 8000:8000 \
  -v $(pwd)/agent.yaml:/app/agent.yaml \
  holodeck-ai serve /app/agent.yaml --host 0.0.0.0
```

---

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for agent.yaml reference
- See [Tools Guide](tools.md) for adding tools to your agent
- See [Observability Guide](observability.md) for tracing and monitoring
- See [Global Configuration](global-config.md) for shared settings
