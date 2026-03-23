# Serve (Agent Local Server)

The `holodeck.serve` package provides HTTP server functionality for exposing
HoloDeck agents via AG-UI (default) or REST protocols. It includes session
management, middleware, multimodal file handling, and protocol adapters.

---

## Server

The main entry point. `AgentServer` wraps a FastAPI application and manages
the full server lifecycle -- initialization, request routing, session cleanup,
and graceful shutdown.

::: holodeck.serve.server.AgentServer
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - is_ready
        - uptime_seconds
        - create_app
        - start
        - stop

---

## Models

Pydantic request/response models shared across both AG-UI and REST protocols,
plus health-check and error-response schemas.

### ProtocolType

::: holodeck.serve.models.ProtocolType
    options:
      docstring_style: google
      show_source: true

### ServerState

::: holodeck.serve.models.ServerState
    options:
      docstring_style: google
      show_source: true

### FileContent

::: holodeck.serve.models.FileContent
    options:
      docstring_style: google
      show_source: true
      members:
        - validate_mime_type
        - validate_base64

### ChatRequest

::: holodeck.serve.models.ChatRequest
    options:
      docstring_style: google
      show_source: true
      members:
        - message_not_blank
        - valid_ulid

### ChatResponse

::: holodeck.serve.models.ChatResponse
    options:
      docstring_style: google
      show_source: true
      members:
        - valid_message_id_ulid
        - valid_session_id_ulid

### ToolCallInfo

::: holodeck.serve.models.ToolCallInfo
    options:
      docstring_style: google
      show_source: true

### HealthResponse

::: holodeck.serve.models.HealthResponse
    options:
      docstring_style: google
      show_source: true

### ProblemDetail

::: holodeck.serve.models.ProblemDetail
    options:
      docstring_style: google
      show_source: true

### SUPPORTED_MIME_TYPES

::: holodeck.serve.models.SUPPORTED_MIME_TYPES
    options:
      docstring_style: google
      show_source: true

---

## Middleware

Cross-cutting concerns for the FastAPI application: structured logging with
optional OpenTelemetry tracing, and RFC 7807 error handling.

### LoggingMiddleware

::: holodeck.serve.middleware.LoggingMiddleware
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - dispatch

### ErrorHandlingMiddleware

::: holodeck.serve.middleware.ErrorHandlingMiddleware
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - dispatch

---

## Session Store

In-memory session storage with TTL-based expiration. Sessions maintain
conversation context (via `AgentExecutor`) across multiple HTTP requests.

### ServerSession

::: holodeck.serve.session_store.ServerSession
    options:
      docstring_style: google
      show_source: true

### SessionStore

::: holodeck.serve.session_store.SessionStore
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - active_count
        - get
        - get_all
        - create
        - delete
        - touch
        - cleanup_expired
        - start_cleanup_task
        - stop_cleanup_task

---

## File Utilities

Shared utilities for multimodal content processing across REST and AG-UI
protocols, including MIME-type mappings, temporary file management, and
binary content extraction.

### Constants

::: holodeck.serve.file_utils.MAX_FILE_SIZE_MB
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.MAX_TOTAL_SIZE_MB
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.MIME_TO_FILE_TYPE
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.MIME_TO_EXTENSION
    options:
      docstring_style: google
      show_source: true

### Functions

::: holodeck.serve.file_utils.create_temp_file_from_bytes
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.convert_file_content_to_file_input
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.convert_binary_dict_to_file_input
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.cleanup_temp_file
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.cleanup_temp_files
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.process_multimodal_files
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.file_utils.extract_binary_parts_from_content
    options:
      docstring_style: google
      show_source: true

---

## Protocols

### Base Protocol

Abstract base class that both AG-UI and REST protocol adapters implement.

::: holodeck.serve.protocols.base.Protocol
    options:
      docstring_style: google
      show_source: true
      members:
        - handle_request
        - name
        - content_type

### REST Protocol

REST protocol adapter providing synchronous and streaming (SSE) chat
endpoints, plus multipart file upload support.

::: holodeck.serve.protocols.rest.RESTProtocol
    options:
      docstring_style: google
      show_source: true
      members:
        - name
        - content_type
        - handle_request
        - handle_sync_request
        - process_files

::: holodeck.serve.protocols.rest.SSEEvent
    options:
      docstring_style: google
      show_source: true
      members:
        - format
        - stream_start
        - message_delta
        - tool_call_start
        - tool_call_args
        - tool_call_end
        - stream_end
        - error
        - keepalive

::: holodeck.serve.protocols.rest.convert_upload_file_to_file_content
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.rest.process_multipart_files
    options:
      docstring_style: google
      show_source: true

### AG-UI Protocol

AG-UI protocol adapter implementing the
[ag-ui-protocol](https://github.com/ag-ui-protocol/ag-ui) event-driven
streaming pattern for agent interaction.

::: holodeck.serve.protocols.agui.AGUIProtocol
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - name
        - content_type
        - handle_request

::: holodeck.serve.protocols.agui.AGUIEventStream
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - content_type
        - encode

#### Event Factory Functions

::: holodeck.serve.protocols.agui.extract_message_and_files_from_input
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.extract_message_from_input
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.map_session_id
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.generate_run_id
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_run_started_event
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_run_finished_event
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_run_error_event
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_text_message_start
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_text_message_content
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_text_message_end
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_tool_call_start
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_tool_call_args
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_tool_call_end
    options:
      docstring_style: google
      show_source: true

::: holodeck.serve.protocols.agui.create_tool_call_events
    options:
      docstring_style: google
      show_source: true
