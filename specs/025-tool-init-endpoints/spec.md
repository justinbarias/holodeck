# Feature Specification: Async Tool Initialization Endpoints

**Feature Branch**: `025-tool-init-endpoints`
**Created**: 2026-03-23
**Status**: Draft
**Input**: User description: "Write a spec that extends the endpoints holodeck serve exposes - which will allow holodeck users to trigger initialization of vector-store based (vectorstore, hierarchical_document) tools. The endpoints must be asynchronous (returns 201 created), and returns an href to get /tools/<tool-id>/init"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trigger Tool Initialization (Priority: P1)

As a HoloDeck user running `holodeck serve`, I want to trigger the initialization (ingestion/indexing) of a vector-store-based tool via an API call so that I can prepare tools for use without blocking the calling client. Currently, tool initialization happens lazily on the first chat request, which causes the first user to experience a long delay while documents are ingested and indexed. By exposing an explicit initialization endpoint, operators can pre-warm tools at deployment time or on-demand.

The user sends a POST request to initialize a specific tool. The server immediately returns a 201 Created response with a `Location` header and response body containing an href to a status resource. The initialization proceeds in the background. The user can poll the status endpoint to check progress and completion.

**Why this priority**: This is the core capability. Without the ability to trigger initialization and receive an async tracking reference, no other stories are possible.

**Independent Test**: Can be fully tested by starting `holodeck serve` with an agent config containing a vectorstore tool, sending a POST to the init endpoint, receiving a 201 with a status href, and then polling the status href until initialization completes.

**Acceptance Scenarios**:

1. **Given** a running holodeck server with a configured vectorstore tool, **When** the user sends `POST /tools/{tool-name}/init`, **Then** the server returns 201 Created with a `Location` header pointing to `/tools/{tool-name}/init` and a response body containing the status href and initial state.
2. **Given** a running holodeck server with a configured hierarchical_document tool, **When** the user sends `POST /tools/{tool-name}/init`, **Then** the server returns 201 Created with the same async tracking pattern.
3. **Given** a running holodeck server, **When** the user sends `POST /tools/{tool-name}/init` for a tool that is not vectorstore or hierarchical_document type, **Then** the server returns 400 Bad Request indicating the tool type does not support initialization.
4. **Given** a running holodeck server, **When** the user sends `POST /tools/{tool-name}/init` for a tool name that does not exist in the agent config, **Then** the server returns 404 Not Found.

---

### User Story 2 - Poll Initialization Status (Priority: P1)

As a HoloDeck user who has triggered tool initialization, I want to poll the status endpoint to know when initialization has completed, failed, or is still in progress so that I can take appropriate action (e.g., begin sending chat requests, retry on failure, or alert an operator).

**Why this priority**: This is co-equal with US1 because the async pattern is only useful if the client can check status. Without polling, the 201 response is a dead end.

**Independent Test**: Can be tested by triggering an init via US1, then repeatedly calling `GET /tools/{tool-name}/init` and verifying state transitions from `pending` through `in_progress` to `completed` or `failed`.

**Acceptance Scenarios**:

1. **Given** a tool initialization has been triggered, **When** the user sends `GET /tools/{tool-name}/init`, **Then** the server returns 200 OK with the current status including state (`pending`, `in_progress`, `completed`, `failed`), timestamps, and progress details.
2. **Given** a tool initialization has completed successfully, **When** the user polls the status endpoint, **Then** the response shows state `completed` with a completion timestamp and summary of what was initialized (e.g., number of documents ingested).
3. **Given** a tool initialization has failed, **When** the user polls the status endpoint, **Then** the response shows state `failed` with an error message describing what went wrong.
4. **Given** no initialization has been triggered for a tool, **When** the user sends `GET /tools/{tool-name}/init`, **Then** the server returns 404 Not Found indicating no initialization job exists for this tool.

---

### User Story 3 - List All Tools and Their Init Status (Priority: P2)

As a HoloDeck operator, I want to list all configured tools and see which ones support initialization and their current init status so that I can get an overview of system readiness without querying each tool individually.

**Why this priority**: This is a convenience feature that improves the operator experience but is not required for the core async init flow.

**Independent Test**: Can be tested by calling `GET /tools` on a running server and verifying the response includes all configured tools with their types and init status.

**Acceptance Scenarios**:

1. **Given** a running holodeck server with multiple tools configured (mix of vectorstore, hierarchical_document, function, mcp types), **When** the user sends `GET /tools`, **Then** the server returns a list of all tools with their name, type, and whether they support initialization.
2. **Given** some tools have been initialized and others have not, **When** the user sends `GET /tools`, **Then** each tool entry includes its current init status (`not_started`, `pending`, `in_progress`, `completed`, `failed`) if it supports initialization.

---

### User Story 4 - Re-initialize a Tool (Priority: P3)

As a HoloDeck user, I want to re-trigger initialization for a tool that has already been initialized (e.g., after updating source documents) so that the tool picks up new or changed content without restarting the server.

**Why this priority**: Re-initialization is an important operational capability but is secondary to the initial init flow.

**Independent Test**: Can be tested by initializing a tool, then sending another POST to the same init endpoint and verifying a new initialization job is created.

**Acceptance Scenarios**:

1. **Given** a tool that has already been initialized (status `completed`), **When** the user sends `POST /tools/{tool-name}/init`, **Then** a new initialization is triggered, replacing the previous status, and the server returns 201 Created with a fresh status href.
2. **Given** a tool that is currently initializing (status `in_progress`), **When** the user sends `POST /tools/{tool-name}/init`, **Then** the server returns 409 Conflict indicating initialization is already in progress.

---

### Edge Cases

- What happens when the server shuts down while a tool initialization is in progress? The initialization job should be cancelled gracefully and the status should reflect the incomplete state.
- What happens when two concurrent POST requests arrive for the same tool? The second request should receive 409 Conflict if the first is still in progress.
- What happens when embedding provider credentials are invalid or missing? Initialization should fail with a descriptive error in the status resource.
- What happens when source documents referenced by the tool config are missing or unreadable? Initialization should fail with details about which files could not be processed.
- How does a `force-ingest` style re-initialization differ from a normal re-init? A `force` query parameter on the POST request should force full re-ingestion even if the tool was previously initialized with the same content.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose a `POST /tools/{tool-name}/init` endpoint that triggers asynchronous initialization of vectorstore and hierarchical_document tools.
- **FR-002**: The POST endpoint MUST return 201 Created with a `Location` header set to `/tools/{tool-name}/init` and a response body containing the status resource representation.
- **FR-003**: System MUST expose a `GET /tools/{tool-name}/init` endpoint that returns the current initialization status for the specified tool.
- **FR-004**: The initialization status MUST include: state (`pending`, `in_progress`, `completed`, `failed`), timestamps (created_at, started_at, completed_at), and a human-readable message or error detail.
- **FR-005**: System MUST return 400 Bad Request when initialization is requested for a tool type that does not support it (e.g., function, mcp, prompt).
- **FR-006**: System MUST return 404 Not Found when initialization is requested for a tool name that does not exist in the agent configuration.
- **FR-007**: System MUST return 409 Conflict when initialization is requested for a tool that is currently being initialized.
- **FR-008**: System MUST expose a `GET /tools` endpoint that lists all configured tools with their type and current initialization status.
- **FR-009**: System MUST gracefully cancel in-progress initialization jobs during server shutdown.
- **FR-010**: The POST endpoint MUST accept an optional `force` query parameter that forces full re-ingestion regardless of prior initialization state.
- **FR-011**: System MUST track initialization progress (e.g., documents processed count) and include it in the status response when available.
- **FR-012**: These endpoints MUST be registered on the main application alongside health/ready endpoints, making them available regardless of protocol mode (REST or AG-UI).
- **FR-013**: System MUST enforce a configurable maximum number of concurrent initialization jobs (default: 3). When the limit is reached, additional init requests MUST return 429 Too Many Requests with a message indicating the concurrent limit has been reached.
- **FR-014**: Init job lifecycle events (start, progress, completion, failure) and any embedding model calls during initialization MUST be instrumented with traces and metrics following OpenTelemetry GenAI semantic conventions, consistent with the existing observability infrastructure.
- **FR-015**: System MUST validate that tool names are unique within an agent configuration at config load time. If duplicate tool names are detected, the system MUST return a clear validation error before the server starts.
- **FR-016**: The tool config `source` field MUST accept remote URIs in addition to local paths. Supported schemes: `s3://` (AWS S3), `az://` (Azure Blob Storage), `https://` / `http://` (HTTP). Local paths (relative and absolute) MUST continue to work unchanged.
- **FR-017**: Remote sources MUST be downloaded to a temporary directory before processing. The temporary directory MUST be cleaned up when the init job reaches a terminal state (completed, failed) or on server shutdown.
- **FR-018**: Cloud provider SDKs (boto3 for S3, azure-storage-blob for Azure) MUST be optional dependencies installed via extras (`pip install holodeck[s3]`, `pip install holodeck[azure-blob]`). If a remote source scheme is used but the required SDK is not installed, the system MUST return a clear error message indicating which package to install.
- **FR-019**: Remote source credentials MUST be resolved via environment variables following cloud-native conventions (AWS credential chain for S3, `AZURE_STORAGE_CONNECTION_STRING` or managed identity for Azure, optional `HOLODECK_HTTP_AUTH_HEADER` for HTTP). No credentials are stored in YAML configuration.
- **FR-020**: HTTP sources MUST support single file downloads only (no archive extraction — dropped for security reasons: zip bombs, path traversal, symlink attacks). S3 and Azure Blob sources MUST support prefix-based listing and download of multiple files preserving directory structure. All downloaded object keys/blob names MUST be validated against path traversal before writing to the temp directory.
- **FR-021**: Remote source downloads MUST retry on transient failures (HTTP 5xx, network errors) with exponential backoff (3 attempts). Authentication failures (HTTP 401/403, invalid credentials) MUST fail immediately with a sanitized error message.

### Key Entities

- **Tool**: A configured tool within the agent definition. Key attributes: name (unique identifier from config), type (vectorstore, hierarchical_document, function, mcp, prompt), supports_init (derived from type).
- **InitJob**: Represents an asynchronous initialization task for a tool. Key attributes: tool_name, state (pending/in_progress/completed/failed), created_at, started_at, completed_at, message, error_detail, progress (documents_processed, total_documents).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can trigger tool initialization and receive an async tracking reference within 500ms of the POST request.
- **SC-002**: Users can determine tool readiness by polling the status endpoint, with status updates reflecting real state within 2 seconds of a state change.
- **SC-003**: 100% of vectorstore and hierarchical_document tools can be initialized via the endpoint without requiring server restart.
- **SC-004**: First chat request latency is reduced by the duration of document ingestion/indexing when tools are pre-warmed via the init endpoint, since the chat flow's lazy init can skip re-ingestion of already-indexed data.
- **SC-005**: Operators can assess full system readiness with a single `GET /tools` call, seeing all tools and their init status in one response.

## Clarifications

### Session 2026-03-23

- Q: Should there be a limit on concurrent tool initialization jobs? → A: Yes, server-managed limit with a default of 3 concurrent init jobs.
- Q: Should a pre-warmed tool skip lazy initialization on the first chat request? → A: No. Pre-warming validates config and ingests/indexes documents, but the chat flow still initializes its own tool instances. The latency benefit comes from data already being ingested, not from reusing the init endpoint's tool instance.

## Assumptions

- Tool names are unique within an agent configuration and serve as stable identifiers for the endpoint path parameter.
- The existing `initialize_tools()` pipeline in `tool_initializer.py` can be reused for the actual ingestion/indexing work. However, the serve layer currently has no background job infrastructure (no job queue, status tracking, or async task management). The orchestration layer — including job state machine, progress reporting, cancellation, and concurrent job limiting — is net-new infrastructure that must be built.
- Initialization state is held in-memory on the server instance (not persisted across restarts). This is consistent with the existing session management pattern.
- The `force` query parameter on POST triggers the same behavior as the existing `--force-ingest` CLI flag.
- These endpoints are protocol-agnostic and are registered on the main FastAPI app alongside existing health/ready endpoints, not within a specific protocol handler.
- Pre-warming via init endpoints handles document ingestion and indexing only. The chat flow still performs its own tool instance initialization (lazy init), but benefits from skipping re-ingestion since data is already indexed. The existing ingestion pipeline already has index verification logic that detects previously-indexed data and skips re-ingestion; this mechanism is what delivers the latency benefit described in SC-004.
