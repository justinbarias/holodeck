"""Unit tests for serve module Pydantic models.

Tests cover validation rules for ChatRequest, ChatResponse, FileContent,
and other models used in the Agent Local Server.
"""

import base64

import pytest
from pydantic import ValidationError

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


class TestProtocolType:
    """Tests for ProtocolType enum."""

    def test_protocol_type_ag_ui_value(self) -> None:
        """Test AG-UI protocol enum value."""
        assert ProtocolType.AG_UI.value == "ag-ui"

    def test_protocol_type_rest_value(self) -> None:
        """Test REST protocol enum value."""
        assert ProtocolType.REST.value == "rest"

    def test_protocol_type_is_string_enum(self) -> None:
        """Test that ProtocolType values are strings."""
        assert isinstance(ProtocolType.AG_UI, str)
        assert isinstance(ProtocolType.REST, str)


class TestServerState:
    """Tests for ServerState enum."""

    def test_server_state_values(self) -> None:
        """Test all server state enum values."""
        assert ServerState.INITIALIZING.value == "initializing"
        assert ServerState.READY.value == "ready"
        assert ServerState.RUNNING.value == "running"
        assert ServerState.SHUTTING_DOWN.value == "shutting_down"
        assert ServerState.STOPPED.value == "stopped"


class TestFileContent:
    """Tests for FileContent model validation."""

    def test_file_content_valid_base64(self) -> None:
        """Test FileContent with valid base64 content."""
        valid_base64 = base64.b64encode(b"test content").decode()
        file_content = FileContent(
            content=valid_base64,
            mime_type="text/plain",
            filename="test.txt",
        )
        assert file_content.content == valid_base64
        assert file_content.mime_type == "text/plain"
        assert file_content.filename == "test.txt"

    def test_file_content_invalid_base64_raises(self) -> None:
        """Test FileContent rejects invalid base64."""
        with pytest.raises(ValidationError, match="content must be valid base64"):
            FileContent(
                content="not-valid-base64!!!",
                mime_type="text/plain",
            )

    def test_file_content_valid_mime_types(self) -> None:
        """Test FileContent accepts all supported MIME types."""
        valid_base64 = base64.b64encode(b"test").decode()
        for mime_type in SUPPORTED_MIME_TYPES:
            file_content = FileContent(content=valid_base64, mime_type=mime_type)
            assert file_content.mime_type == mime_type

    def test_file_content_unsupported_mime_type_raises(self) -> None:
        """Test FileContent rejects unsupported MIME types."""
        valid_base64 = base64.b64encode(b"test").decode()
        with pytest.raises(ValidationError, match="Unsupported MIME type"):
            FileContent(
                content=valid_base64,
                mime_type="application/octet-stream",
            )

    def test_file_content_filename_optional(self) -> None:
        """Test FileContent filename is optional."""
        valid_base64 = base64.b64encode(b"test").decode()
        file_content = FileContent(content=valid_base64, mime_type="text/plain")
        assert file_content.filename is None


class TestChatRequest:
    """Tests for ChatRequest model validation."""

    def test_chat_request_valid_message(self) -> None:
        """Test ChatRequest with valid message."""
        request = ChatRequest(message="Hello, agent!")
        assert request.message == "Hello, agent!"
        assert request.session_id is None
        assert request.files is None

    def test_chat_request_empty_message_raises(self) -> None:
        """Test ChatRequest rejects empty message."""
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_blank_message_raises(self) -> None:
        """Test ChatRequest rejects whitespace-only message."""
        with pytest.raises(ValidationError, match="message cannot be blank"):
            ChatRequest(message="   ")

    def test_chat_request_max_length_message(self) -> None:
        """Test ChatRequest accepts message up to max length."""
        max_message = "a" * 10_000
        request = ChatRequest(message=max_message)
        assert len(request.message) == 10_000

    def test_chat_request_exceeds_max_length_raises(self) -> None:
        """Test ChatRequest rejects message exceeding max length."""
        with pytest.raises(ValidationError):
            ChatRequest(message="a" * 10_001)

    def test_chat_request_valid_session_id(self) -> None:
        """Test ChatRequest with valid ULID session_id."""
        # Valid ULID format: 01ARZ3NDEKTSV4RRFFQ69G5FAV
        valid_ulid = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        request = ChatRequest(message="test", session_id=valid_ulid)
        assert request.session_id == valid_ulid

    def test_chat_request_invalid_session_id_raises(self) -> None:
        """Test ChatRequest rejects invalid session_id."""
        with pytest.raises(ValidationError, match="session_id must be valid ULID"):
            ChatRequest(message="test", session_id="invalid-session-id")

    def test_chat_request_with_files(self) -> None:
        """Test ChatRequest with attached files."""
        valid_base64 = base64.b64encode(b"test").decode()
        files = [FileContent(content=valid_base64, mime_type="text/plain")]
        request = ChatRequest(message="Check this file", files=files)
        assert len(request.files) == 1

    def test_chat_request_max_files(self) -> None:
        """Test ChatRequest accepts up to 10 files."""
        valid_base64 = base64.b64encode(b"test").decode()
        files = [
            FileContent(content=valid_base64, mime_type="text/plain") for _ in range(10)
        ]
        request = ChatRequest(message="test", files=files)
        assert len(request.files) == 10

    def test_chat_request_exceeds_max_files_raises(self) -> None:
        """Test ChatRequest rejects more than 10 files."""
        valid_base64 = base64.b64encode(b"test").decode()
        files = [
            FileContent(content=valid_base64, mime_type="text/plain") for _ in range(11)
        ]
        with pytest.raises(ValidationError):
            ChatRequest(message="test", files=files)


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_chat_response_all_fields(self) -> None:
        """Test ChatResponse with all fields populated."""
        response = ChatResponse(
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            content="Hello! How can I help?",
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            tool_calls=[
                ToolCallInfo(
                    name="search", arguments={"query": "test"}, status="success"
                )
            ],
            tokens_used={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            execution_time_ms=150,
        )
        assert response.message_id == "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        assert response.content == "Hello! How can I help?"
        assert response.session_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert len(response.tool_calls) == 1
        assert response.tokens_used["total_tokens"] == 30
        assert response.execution_time_ms == 150

    def test_chat_response_defaults(self) -> None:
        """Test ChatResponse with default values."""
        response = ChatResponse(
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            content="Hello!",
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            execution_time_ms=100,
        )
        assert response.tool_calls == []
        assert response.tokens_used is None

    def test_chat_response_invalid_message_id_raises(self) -> None:
        """Test ChatResponse rejects invalid message_id ULID."""
        with pytest.raises(ValidationError, match="message_id must be valid ULID"):
            ChatResponse(
                message_id="not-a-valid-ulid",
                content="Hello!",
                session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
                execution_time_ms=100,
            )

    def test_chat_response_invalid_session_id_raises(self) -> None:
        """Test ChatResponse rejects invalid session_id ULID."""
        with pytest.raises(ValidationError, match="session_id must be valid ULID"):
            ChatResponse(
                message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
                content="Hello!",
                session_id="invalid-session-id",
                execution_time_ms=100,
            )


class TestToolCallInfo:
    """Tests for ToolCallInfo model."""

    def test_tool_call_info_defaults(self) -> None:
        """Test ToolCallInfo with default values."""
        tool_call = ToolCallInfo(name="search")
        assert tool_call.name == "search"
        assert tool_call.arguments == {}
        assert tool_call.status == "success"

    def test_tool_call_info_with_arguments(self) -> None:
        """Test ToolCallInfo with arguments."""
        tool_call = ToolCallInfo(
            name="search",
            arguments={"query": "test", "limit": 10},
            status="success",
        )
        assert tool_call.arguments == {"query": "test", "limit": 10}

    def test_tool_call_info_error_status(self) -> None:
        """Test ToolCallInfo with error status."""
        tool_call = ToolCallInfo(name="search", status="error")
        assert tool_call.status == "error"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response_defaults(self) -> None:
        """Test HealthResponse with default values."""
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"
        assert response.agent_name is None
        assert response.agent_ready is False
        assert response.active_sessions == 0
        assert response.uptime_seconds == 0.0

    def test_health_response_all_fields(self) -> None:
        """Test HealthResponse with all fields populated."""
        response = HealthResponse(
            status="healthy",
            agent_name="support-agent",
            agent_ready=True,
            active_sessions=5,
            uptime_seconds=3600.5,
        )
        assert response.agent_name == "support-agent"
        assert response.agent_ready is True
        assert response.active_sessions == 5
        assert response.uptime_seconds == 3600.5


class TestProblemDetail:
    """Tests for ProblemDetail model (RFC 7807)."""

    def test_problem_detail_rfc7807_format(self) -> None:
        """Test ProblemDetail follows RFC 7807 format."""
        problem = ProblemDetail(
            type="https://holodeck.dev/errors/validation-error",
            title="Validation Error",
            status=400,
            detail="The message field is required.",
            instance="/agent/support/chat",
        )
        assert problem.type == "https://holodeck.dev/errors/validation-error"
        assert problem.title == "Validation Error"
        assert problem.status == 400
        assert problem.detail == "The message field is required."
        assert problem.instance == "/agent/support/chat"

    def test_problem_detail_defaults(self) -> None:
        """Test ProblemDetail with default values."""
        problem = ProblemDetail(title="Not Found", status=404)
        assert problem.type == "about:blank"
        assert problem.detail is None
        assert problem.instance is None

    def test_problem_detail_serialization(self) -> None:
        """Test ProblemDetail JSON serialization."""
        problem = ProblemDetail(
            title="Internal Server Error",
            status=500,
            detail="Unexpected error occurred.",
        )
        json_data = problem.model_dump()
        assert json_data["type"] == "about:blank"
        assert json_data["title"] == "Internal Server Error"
        assert json_data["status"] == 500
        assert json_data["detail"] == "Unexpected error occurred."
