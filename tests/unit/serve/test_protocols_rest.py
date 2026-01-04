"""Unit tests for REST protocol adapter.

Tests for:
- T031: REST protocol handlers (sync and stream)
- T032: SSE event serialization (stream_start, message_delta, tool_call_*, etc.)
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# T031: Unit tests for REST protocol handlers
# =============================================================================


class TestRESTProtocolProperties:
    """Tests for RESTProtocol class properties."""

    def test_protocol_name_is_rest(self) -> None:
        """Test protocol name is 'rest'."""
        from holodeck.serve.protocols.rest import RESTProtocol

        protocol = RESTProtocol()
        assert protocol.name == "rest"

    def test_protocol_content_type_is_event_stream(self) -> None:
        """Test protocol content type for streaming is 'text/event-stream'."""
        from holodeck.serve.protocols.rest import RESTProtocol

        protocol = RESTProtocol()
        assert protocol.content_type == "text/event-stream"


class TestRESTProtocolSyncHandler:
    """Tests for RESTProtocol.handle_sync_request method."""

    @pytest.mark.asyncio
    async def test_handle_sync_request_basic(self) -> None:
        """Test handle_sync_request returns ChatResponse."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        # Create request
        request = ChatRequest(message="Hello, world!")

        # Mock session
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help?"
        mock_response.tool_executions = []
        mock_response.tokens_used = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        # Create protocol and handle request
        protocol = RESTProtocol()
        response = await protocol.handle_sync_request(request, mock_session)

        # Verify response
        assert response.content == "Hello! How can I help?"
        assert response.session_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert response.execution_time_ms >= 0
        assert len(response.message_id) == 26  # ULID length

    @pytest.mark.asyncio
    async def test_handle_sync_request_with_tool_calls(self) -> None:
        """Test handle_sync_request includes tool_calls in response."""
        from holodeck.models.tool_execution import ToolExecution, ToolStatus
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        # Create request
        request = ChatRequest(message="Search for info")

        # Mock response with tool execution
        tool_exec = ToolExecution(
            tool_name="search",
            parameters={"query": "test"},
            status=ToolStatus.SUCCESS,
        )

        mock_response = MagicMock()
        mock_response.content = "Here are the results"
        mock_response.tool_executions = [tool_exec]
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        # Create protocol and handle request
        protocol = RESTProtocol()
        response = await protocol.handle_sync_request(request, mock_session)

        # Verify tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].status == "success"

    @pytest.mark.asyncio
    async def test_handle_sync_request_with_tokens(self) -> None:
        """Test handle_sync_request includes tokens_used."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        # Create request
        request = ChatRequest(message="Hello")

        # Mock response with tokens
        mock_response = MagicMock()
        mock_response.content = "Hi!"
        mock_response.tool_executions = []
        mock_response.tokens_used = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()
        response = await protocol.handle_sync_request(request, mock_session)

        assert response.tokens_used is not None
        assert response.tokens_used["prompt_tokens"] == 10
        assert response.tokens_used["completion_tokens"] == 5
        assert response.tokens_used["total_tokens"] == 15


class TestRESTProtocolStreamHandler:
    """Tests for RESTProtocol.handle_request (streaming) method."""

    @pytest.mark.asyncio
    async def test_handle_request_yields_bytes(self) -> None:
        """Test handle_request yields bytes for streaming."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        # Create request
        request = ChatRequest(message="Hello")

        # Mock session
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_executions = []
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()
        events = []
        async for event_bytes in protocol.handle_request(request, mock_session):
            events.append(event_bytes)
            assert isinstance(event_bytes, bytes)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_handle_request_starts_with_stream_start(self) -> None:
        """Test handle_request first event is stream_start."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        # Create request
        request = ChatRequest(message="Hello")

        # Mock session
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_executions = []
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()
        events = []
        async for event_bytes in protocol.handle_request(request, mock_session):
            events.append(event_bytes.decode("utf-8"))

        # First event should be stream_start
        assert "event: stream_start" in events[0]

    @pytest.mark.asyncio
    async def test_handle_request_ends_with_stream_end(self) -> None:
        """Test handle_request last event is stream_end."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        request = ChatRequest(message="Hello")

        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_executions = []
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()
        events = []
        async for event_bytes in protocol.handle_request(request, mock_session):
            events.append(event_bytes.decode("utf-8"))

        # Last event should be stream_end
        assert "event: stream_end" in events[-1]


class TestFileContentToFileInput:
    """Tests for converting FileContent to FileInput for FileProcessor."""

    def test_convert_image_file_content(self) -> None:
        """Test converting image FileContent to FileInput."""
        from holodeck.serve.file_utils import convert_file_content_to_file_input
        from holodeck.serve.models import FileContent

        # Create a valid base64-encoded 1x1 PNG
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        file_content = FileContent(
            content=base64.b64encode(png_bytes).decode(),
            mime_type="image/png",
            filename="test.png",
        )

        file_input = convert_file_content_to_file_input(file_content)

        assert file_input.type == "image"
        assert file_input.path is not None

    def test_convert_pdf_file_content(self) -> None:
        """Test converting PDF FileContent to FileInput."""
        from holodeck.serve.file_utils import convert_file_content_to_file_input
        from holodeck.serve.models import FileContent

        # Create a minimal PDF content (just base64 placeholder)
        pdf_content = base64.b64encode(b"%PDF-1.0\n%%EOF").decode()
        file_content = FileContent(
            content=pdf_content,
            mime_type="application/pdf",
            filename="test.pdf",
        )

        file_input = convert_file_content_to_file_input(file_content)

        assert file_input.type == "pdf"

    def test_convert_text_file_content(self) -> None:
        """Test converting text FileContent to FileInput."""
        from holodeck.serve.file_utils import convert_file_content_to_file_input
        from holodeck.serve.models import FileContent

        text_content = base64.b64encode(b"Hello, world!").decode()
        file_content = FileContent(
            content=text_content,
            mime_type="text/plain",
            filename="test.txt",
        )

        file_input = convert_file_content_to_file_input(file_content)

        assert file_input.type == "text"


# =============================================================================
# T032: Unit tests for SSE event serialization
# =============================================================================


class TestSSEEventFormat:
    """Tests for SSE event format compliance with sse-events.md."""

    def test_sse_event_format_structure(self) -> None:
        """Test SSE event has correct structure: event: type\\ndata: json\\n\\n."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.format("test_event", {"key": "value"})

        assert event.startswith("event: test_event\n")
        assert "data: " in event
        assert event.endswith("\n\n")

    def test_sse_event_data_is_json(self) -> None:
        """Test SSE event data is valid JSON."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.format("test_event", {"key": "value"})

        # Extract data line
        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        json_str = data_line[6:]  # Remove "data: " prefix

        data = json.loads(json_str)
        assert data["key"] == "value"


class TestStreamStartEvent:
    """Tests for stream_start event format."""

    def test_stream_start_format(self) -> None:
        """Test stream_start event has session_id and message_id."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.stream_start(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
        )

        assert "event: stream_start" in event

        # Parse data
        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["session_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert data["message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAW"


class TestMessageDeltaEvent:
    """Tests for message_delta event format."""

    def test_message_delta_format(self) -> None:
        """Test message_delta event has delta and message_id."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.message_delta(
            delta="Hello, world!",
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
        )

        assert "event: message_delta" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["delta"] == "Hello, world!"
        assert data["message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAW"

    def test_message_delta_with_special_chars(self) -> None:
        """Test message_delta properly encodes special characters."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.message_delta(
            delta='Special chars: "quotes" and \\backslash',
            message_id="msg-1",
        )

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert '"quotes"' in data["delta"]


class TestToolCallStartEvent:
    """Tests for tool_call_start event format."""

    def test_tool_call_start_format(self) -> None:
        """Test tool_call_start event has tool_call_id, name, message_id."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.tool_call_start(
            tool_call_id="tc_01",
            name="search_knowledge_base",
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
        )

        assert "event: tool_call_start" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["tool_call_id"] == "tc_01"
        assert data["name"] == "search_knowledge_base"
        assert data["message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAW"


class TestToolCallArgsEvent:
    """Tests for tool_call_args event format."""

    def test_tool_call_args_format(self) -> None:
        """Test tool_call_args event has tool_call_id and args_delta."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.tool_call_args(
            tool_call_id="tc_01",
            args_delta='{"query": "return',
        )

        assert "event: tool_call_args" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["tool_call_id"] == "tc_01"
        assert data["args_delta"] == '{"query": "return'


class TestToolCallEndEvent:
    """Tests for tool_call_end event format."""

    def test_tool_call_end_format(self) -> None:
        """Test tool_call_end event has tool_call_id and status."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.tool_call_end(
            tool_call_id="tc_01",
            status="success",
        )

        assert "event: tool_call_end" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["tool_call_id"] == "tc_01"
        assert data["status"] == "success"

    def test_tool_call_end_error_status(self) -> None:
        """Test tool_call_end with error status."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.tool_call_end(
            tool_call_id="tc_01",
            status="error",
        )

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["status"] == "error"


class TestStreamEndEvent:
    """Tests for stream_end event format."""

    def test_stream_end_format(self) -> None:
        """Test stream_end event has message_id, tokens_used, execution_time_ms."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.stream_end(
            message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            tokens_used={
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
            },
            execution_time_ms=1250,
        )

        assert "event: stream_end" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAW"
        assert data["tokens_used"]["prompt_tokens"] == 150
        assert data["execution_time_ms"] == 1250

    def test_stream_end_without_tokens(self) -> None:
        """Test stream_end with None tokens_used."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.stream_end(
            message_id="msg-1",
            tokens_used=None,
            execution_time_ms=500,
        )

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["tokens_used"] is None
        assert data["execution_time_ms"] == 500


class TestErrorEvent:
    """Tests for error event format (RFC 7807)."""

    def test_error_event_format(self) -> None:
        """Test error event has RFC 7807 fields."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.error(
            type="https://holodeck.dev/errors/agent-error",
            title="Agent Error",
            status=500,
            detail="LLM provider timeout",
        )

        assert "event: error" in event

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["type"] == "https://holodeck.dev/errors/agent-error"
        assert data["title"] == "Agent Error"
        assert data["status"] == 500
        assert data["detail"] == "LLM provider timeout"

    def test_error_event_minimal(self) -> None:
        """Test error event with minimal fields."""
        from holodeck.serve.protocols.rest import SSEEvent

        event = SSEEvent.error(
            type="about:blank",
            title="Internal Error",
            status=500,
            detail=None,
        )

        lines = event.strip().split("\n")
        data_line = [line for line in lines if line.startswith("data: ")][0]
        data = json.loads(data_line[6:])

        assert data["type"] == "about:blank"
        assert data["title"] == "Internal Error"
        assert data["status"] == 500


class TestKeepaliveComment:
    """Tests for keepalive comment format."""

    def test_keepalive_format(self) -> None:
        """Test keepalive is SSE comment format."""
        from holodeck.serve.protocols.rest import SSEEvent

        keepalive = SSEEvent.keepalive()

        # SSE comment starts with ":"
        assert keepalive == ": keepalive\n"

    def test_keepalive_is_single_line(self) -> None:
        """Test keepalive is a single line comment."""
        from holodeck.serve.protocols.rest import SSEEvent

        keepalive = SSEEvent.keepalive()

        # Should be exactly one line ending with newline
        assert keepalive.count("\n") == 1
        assert keepalive.endswith("\n")


# =============================================================================
# Additional tests for edge cases
# =============================================================================


class TestRESTProtocolErrorHandling:
    """Tests for error handling in RESTProtocol."""

    @pytest.mark.asyncio
    async def test_handle_request_emits_error_on_failure(self) -> None:
        """Test handle_request emits error event when agent fails."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        request = ChatRequest(message="Hello")

        # Mock executor that raises an error
        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(
            side_effect=RuntimeError("Agent execution failed")
        )

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()
        events = []
        async for event_bytes in protocol.handle_request(request, mock_session):
            events.append(event_bytes.decode("utf-8"))

        # Should contain error event
        all_content = "".join(events)
        assert "event: error" in all_content
        assert "Agent execution failed" in all_content

    @pytest.mark.asyncio
    async def test_handle_sync_request_raises_on_failure(self) -> None:
        """Test handle_sync_request raises exception on agent failure."""
        from holodeck.serve.models import ChatRequest
        from holodeck.serve.protocols.rest import RESTProtocol

        request = ChatRequest(message="Hello")

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(
            side_effect=RuntimeError("Agent execution failed")
        )

        mock_session = MagicMock()
        mock_session.session_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        mock_session.agent_executor = mock_executor

        protocol = RESTProtocol()

        with pytest.raises(RuntimeError, match="Agent execution failed"):
            await protocol.handle_sync_request(request, mock_session)
