# Feature Specification: Agent Local Server

**Feature Branch**: `017-agent-local-server`
**Created**: 2025-12-29
**Status**: Draft
**Input**: User description: "Options to publish agent capabilities via ag-ui (default) or FastAPI REST using /agent/<agent-name>/chat endpoint"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Start Agent Server with Default Protocol (Priority: P1)

A developer wants to quickly expose their configured agent as an interactive server for client integration. They run a single CLI command to start a local server that exposes their agent using the AG-UI protocol (Agent User Interface protocol), which is the default option. The server starts on a configurable port and the agent is immediately accessible to AG-UI compatible clients.

**Why this priority**: This is the primary way users will deploy agents locally. AG-UI is the default protocol because it provides a standardized interface for agent interactions that many UI frameworks support out of the box.

**Independent Test**: Can be fully tested by starting the server and connecting with an AG-UI compatible client to verify the agent responds correctly.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml configuration file, **When** the user runs `holodeck serve agent.yaml`, **Then** a server starts on the default port (8000) using AG-UI protocol and displays the server URL.

2. **Given** a running AG-UI server, **When** an AG-UI compatible client connects and sends a message, **Then** the agent processes the message and streams the response back following AG-UI protocol specifications.

3. **Given** a valid agent.yaml, **When** the user runs `holodeck serve agent.yaml --port 3000`, **Then** the server starts on port 3000 instead of the default.

---

### User Story 2 - Start Agent Server with FastAPI REST Protocol (Priority: P1)

A developer prefers to expose their agent through a simple REST API for integration with custom applications or services that don't support AG-UI. They specify the REST protocol option when starting the server, which exposes the agent at a `/agent/<agent-name>/chat` endpoint.

**Why this priority**: REST API is a universal integration method that allows any HTTP client to interact with the agent. This is critical for teams building custom frontends or integrating with existing systems.

**Independent Test**: Can be fully tested by starting the server with REST protocol and making HTTP POST requests to the chat endpoint.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with name "customer-support", **When** the user runs `holodeck serve agent.yaml --protocol rest`, **Then** a FastAPI server starts and exposes `/agent/customer-support/chat` endpoint.

2. **Given** a running REST server, **When** a client sends a POST request to `/agent/<agent-name>/chat` with a JSON body containing a message, **Then** the agent responds with the appropriate JSON response.

3. **Given** a running REST server, **When** a client sends a POST request to `/agent/<agent-name>/chat/stream`, **Then** the agent streams the response using Server-Sent Events (SSE).

---

### User Story 3 - Health Check and Monitoring Endpoints (Priority: P2)

A developer or operations team needs to monitor the health of the agent server and verify the agent is operational. The server provides standard health check endpoints for integration with monitoring tools and load balancers.

**Why this priority**: Health monitoring is essential for production deployments and integration with container orchestration systems.

**Independent Test**: Can be fully tested by starting the server and making requests to health endpoints.

**Acceptance Scenarios**:

1. **Given** a running agent server, **When** a client requests `/health`, **Then** the server returns status information indicating overall health.

2. **Given** a running agent server, **When** a client requests `/health/agent`, **Then** the server returns the health status of the loaded agent including its name and ready state.

3. **Given** a running agent server, **When** the server is ready to accept requests, **Then** the `/ready` endpoint returns a successful status.

---

### User Story 4 - Session Management (Priority: P2)

A developer building a multi-turn conversational application needs the server to maintain conversation context across multiple requests. The server supports session-based interactions where conversation history is preserved.

**Why this priority**: Session management enables stateful conversations which are critical for complex agent interactions, but basic single-turn functionality should work first.

**Independent Test**: Can be fully tested by sending multiple messages with the same session ID and verifying context is maintained.

**Acceptance Scenarios**:

1. **Given** a running agent server, **When** a client includes a `session_id` in the request, **Then** the server maintains conversation history for that session.

2. **Given** an active session, **When** the client sends a new message with the same `session_id`, **Then** the agent has access to previous conversation context.

3. **Given** an active session, **When** the client sends a DELETE request to the session endpoint, **Then** the session and its history are cleared.

---

### User Story 5 - Interactive Server Startup Feedback (Priority: P3)

A developer starting the server wants clear feedback about the server status, available endpoints, and how to interact with the agent. Upon startup, the server displays helpful information and optionally opens a browser to a test interface.

**Why this priority**: Good developer experience accelerates adoption and reduces friction when getting started.

**Independent Test**: Can be fully tested by starting the server and verifying the console output contains all expected information.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml, **When** the server starts successfully, **Then** the console displays the server URL, agent name, protocol, and available endpoints.

2. **Given** a running REST server, **When** the user accesses the root URL in a browser, **Then** the server displays OpenAPI documentation with interactive testing capabilities.

3. **Given** the `--open` flag is provided, **When** the server starts successfully, **Then** the default browser opens to the server URL.

---

### Edge Cases

- What happens when the agent configuration file is invalid or missing? System displays a clear error message and does not start the server.
- How does the system handle when the specified port is already in use? System displays an error indicating port conflict and suggests an alternative.
- What happens when an agent encounters an error during request processing? System returns an appropriate error response without crashing the server.
- How does the system handle when a session expires or is not found? System creates a new session automatically; expired sessions (inactive >30 minutes) are cleaned up.
- What happens when a request is received while the agent is still initializing? System returns a "service unavailable" response until the agent is ready.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `holodeck serve` CLI command to start an agent server
- **FR-002**: System MUST support AG-UI protocol as the default server protocol
- **FR-003**: System MUST support FastAPI REST protocol as an alternative server protocol
- **FR-004**: System MUST expose REST endpoints at `/agent/<agent-name>/chat` for synchronous requests
- **FR-005**: System MUST expose REST endpoints at `/agent/<agent-name>/chat/stream` for streaming responses using SSE
- **FR-006**: System MUST allow configuration of the server port via `--port` flag (default: 8000)
- **FR-007**: System MUST allow protocol selection via `--protocol` flag with options: `ag-ui` (default), `rest`
- **FR-008**: System MUST validate agent configuration before starting the server
- **FR-009**: System MUST provide a `/health` endpoint for server health checks
- **FR-010**: System MUST provide a `/ready` endpoint for readiness checks
- **FR-011**: System MUST support session-based conversations via `session_id` parameter
- **FR-012**: System MUST display the server URL and available endpoints upon startup
- **FR-013**: System MUST handle graceful shutdown on SIGINT/SIGTERM signals
- **FR-014**: System MUST log request metadata (timestamp, session_id, endpoint, latency) by default; full request/response content logged only in debug mode
- **FR-015**: System MUST provide OpenAPI documentation at the root URL for REST protocol
- **FR-016**: System MUST support the `--open` flag to open a browser upon server startup
- **FR-017**: System MUST expire inactive sessions after 30 minutes and clean up associated resources
- **FR-018**: System MUST enable CORS for REST protocol with configurable origins via `--cors-origins` flag, defaulting to allow all (`*`)
- **FR-019**: System MUST return error responses in RFC 7807 Problem Details format (type, title, status, detail)
- **FR-020**: System MUST support multimodal inputs (images, PDFs, Office documents) via base64-encoded JSON or multipart form-data uploads
- **FR-021**: System MUST process uploaded files using existing FileProcessor (OCR for images, text extraction for documents) before sending to agent

### Key Entities

- **Agent Server**: The HTTP server instance that hosts a single agent
- **Agent Endpoint**: The URL path where the agent is accessible
- **Session**: A conversation context maintained across multiple requests, identified by session_id; expires after 30 minutes of inactivity
- **Chat Request**: An incoming message from a client to the agent, containing message content, optional session_id, and optional file attachments
- **Chat Response**: The agent's reply, which may be synchronous JSON or a streamed SSE response

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can start an agent server with a single command in under 5 seconds
- **SC-002**: Agent responses begin streaming to the client within 2 seconds of request receipt
- **SC-003**: Server supports at least 100 concurrent sessions without degradation
- **SC-004**: Health check endpoints respond in under 100 milliseconds
- **SC-005**: 95% of users successfully connect to their agent on the first attempt using provided documentation
- **SC-006**: Server gracefully handles connection errors and returns appropriate error responses within 1 second

## Clarifications

### Session 2025-12-29

- Q: What is the session expiration behavior? → A: Sessions expire after 30 minutes of inactivity
- Q: What should be logged by default (non-debug mode)? → A: Log request metadata only (timestamp, session_id, endpoint, latency)
- Q: What CORS behavior for REST protocol? → A: CORS enabled with configurable origins, default allow all (`*`)
- Q: What error response format for API errors? → A: RFC 7807 Problem Details (type, title, status, detail)

## Assumptions

- The AG-UI protocol follows the Agent User Interface protocol standard for agent communication
- Users have a valid agent.yaml configuration that has been tested with `holodeck test` or `holodeck chat`
- The server runs in a development/local environment; production deployment features (authentication, rate limiting, TLS) will be addressed in a separate specification
- Session storage will use in-memory storage by default; persistent session storage will be addressed in a future specification
- The REST API follows OpenAPI 3.0 specifications for documentation
- Multi-agent publishing will be supported through agent orchestration in a future specification
