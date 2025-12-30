"""Server runtime package for exposing agents via HTTP.

This module provides HTTP server functionality for HoloDeck agents,
supporting AG-UI (default) and REST protocols.
"""

from holodeck.serve.middleware import ErrorHandlingMiddleware, LoggingMiddleware
from holodeck.serve.models import (
    SUPPORTED_MIME_TYPES,
    ChatRequest,
    ChatResponse,
    FileContent,
    HealthResponse,
    ProblemDetail,
    ProtocolType,
    ServerState,
    ToolCallInfo,
)
from holodeck.serve.server import AgentServer
from holodeck.serve.session_store import ServerSession, SessionStore

__all__ = [
    # Server
    "AgentServer",
    # Models
    "ChatRequest",
    "ChatResponse",
    "FileContent",
    "HealthResponse",
    "ProblemDetail",
    "ProtocolType",
    "ServerState",
    "SUPPORTED_MIME_TYPES",
    "ToolCallInfo",
    # Session management
    "ServerSession",
    "SessionStore",
    # Middleware
    "ErrorHandlingMiddleware",
    "LoggingMiddleware",
]
