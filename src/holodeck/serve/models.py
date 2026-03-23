"""Pydantic models for the Agent Local Server.

This module defines request/response models for both AG-UI and REST protocols,
as well as health check and error response models.
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from ulid import ULID


class ProtocolType(str, Enum):
    """Protocol types supported by the Agent Local Server."""

    AG_UI = "ag-ui"
    REST = "rest"


class ServerState(str, Enum):
    """Server lifecycle states."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class InitJobState(str, Enum):
    """States for tool initialization jobs."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class InitJobProgress(BaseModel):
    """Progress tracking for tool initialization."""

    documents_processed: int = Field(
        ..., description="Number of documents processed so far"
    )
    total_documents: int | None = Field(
        default=None, description="Total documents to process (None if unknown)"
    )


class InitJobResponse(BaseModel):
    """Response body for tool initialization endpoints."""

    tool_name: str = Field(..., description="Name of the tool being initialized")
    state: InitJobState = Field(..., description="Current job state")
    href: str = Field(..., description="URL to the job status resource")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: datetime | None = Field(
        default=None, description="Processing start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Completion timestamp"
    )
    message: str | None = Field(
        default=None, description="Human-readable status message"
    )
    error_detail: str | None = Field(
        default=None, description="Sanitized error detail on failure"
    )
    progress: InitJobProgress | None = Field(
        default=None, description="Progress info during processing"
    )
    force: bool = Field(
        ..., description="Whether force re-initialization was requested"
    )


class ToolInfoResponse(BaseModel):
    """Information about a single tool."""

    name: str = Field(..., description="Tool name")
    type: str = Field(..., description="Tool type (e.g. vectorstore, mcp)")
    supports_init: bool = Field(
        ..., description="Whether this tool supports initialization"
    )
    init_status: InitJobState | None = Field(
        default=None, description="Current init job state, if any"
    )


class ToolListResponse(BaseModel):
    """Response for listing available tools."""

    tools: list[ToolInfoResponse] = Field(..., description="List of tool info")
    total: int = Field(..., description="Total number of tools")


# Supported MIME types for file uploads
SUPPORTED_MIME_TYPES = {
    # Images
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    # Documents
    "application/pdf",
    # Office documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Text
    "text/plain",
    "text/csv",
    "text/markdown",
}


class FileContent(BaseModel):
    """Binary file content for multimodal inputs.

    Files are base64-encoded for JSON transport. Supported file types
    include images, PDFs, Office documents, and text files.
    """

    content: str = Field(..., description="Base64-encoded file data")
    mime_type: str = Field(..., description="MIME type of the file")
    filename: str | None = Field(default=None, description="Original filename")

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate MIME type is supported."""
        if v not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate content is valid base64."""
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("content must be valid base64") from None
        return v


class ChatRequest(BaseModel):
    """Request payload for chat endpoints.

    Used by both synchronous and streaming chat endpoints in the REST protocol.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="User message content",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier (ULID format)",
    )
    files: list[FileContent] | None = Field(
        default=None,
        max_length=10,
        description="Attached files for multimodal input",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        """Validate message is not just whitespace."""
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v

    @field_validator("session_id")
    @classmethod
    def valid_ulid(cls, v: str | None) -> str | None:
        """Validate session_id is valid ULID format if provided."""
        if v is not None:
            try:
                ULID.from_str(v)
            except (ValueError, TypeError):
                raise ValueError("session_id must be valid ULID") from None
        return v


class ToolCallInfo(BaseModel):
    """Tool execution information in response.

    Contains details about a tool call made by the agent during
    message processing.
    """

    name: str = Field(..., description="Tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters (JSON-serializable)",
    )
    status: str = Field(
        default="success",
        description="Execution status: 'success' or 'error'",
    )


class ChatResponse(BaseModel):
    """Response payload for synchronous chat endpoint.

    Contains the agent's response, tool calls, and execution metadata.
    """

    message_id: str = Field(..., description="Unique message identifier (ULID)")
    content: str = Field(..., description="Agent response text")
    session_id: str = Field(..., description="Session used/created (ULID)")
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list,
        description="Tools invoked during processing",
    )
    tokens_used: dict[str, int] | None = Field(
        default=None,
        description="Token consumption statistics",
    )
    execution_time_ms: int = Field(..., description="Request latency in milliseconds")

    @field_validator("message_id")
    @classmethod
    def valid_message_id_ulid(cls, v: str) -> str:
        """Validate message_id is valid ULID format."""
        try:
            ULID.from_str(v)
        except (ValueError, TypeError):
            raise ValueError("message_id must be valid ULID") from None
        return v

    @field_validator("session_id")
    @classmethod
    def valid_session_id_ulid(cls, v: str) -> str:
        """Validate session_id is valid ULID format."""
        try:
            ULID.from_str(v)
        except (ValueError, TypeError):
            raise ValueError("session_id must be valid ULID") from None
        return v


class HealthResponse(BaseModel):
    """Response for health check endpoints.

    Provides server and agent status information.
    """

    status: str = Field(
        ...,
        description=("Overall health status: 'healthy', 'degraded', " "or 'unhealthy'"),
    )
    agent_name: str | None = Field(
        default=None,
        description="Loaded agent name",
    )
    agent_ready: bool = Field(
        default=False,
        description="Agent ready state",
    )
    active_sessions: int = Field(
        default=0,
        description="Number of active sessions",
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="Server uptime in seconds",
    )
    backend_ready: bool = Field(
        default=True,
        description="Whether backend prerequisites are satisfied",
    )
    backend_diagnostics: list[str] = Field(
        default_factory=list,
        description="Diagnostic messages when backend is degraded/unhealthy",
    )


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details error response.

    Standard format for HTTP API error responses.
    """

    type: str = Field(
        default="about:blank",
        description="Error type URI",
    )
    title: str = Field(..., description="Short human-readable description")
    status: int = Field(..., description="HTTP status code")
    detail: str | None = Field(
        default=None,
        description="Detailed error message",
    )
    instance: str | None = Field(
        default=None,
        description="Request identifier for tracing",
    )
