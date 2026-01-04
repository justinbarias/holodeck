"""Unit tests for AG-UI protocol adapter.

Tests for:
- T015: AG-UI event mapping (lifecycle, text message, tool call events)
- T016: RunAgentInput to HoloDeck request mapping
- T029a: Unit tests for AG-UI BinaryInputContent parsing
- T029b: Unit tests for AG-UI multimodal content to FileInput conversion
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from ag_ui.core.events import (
    EventType,
)
from ag_ui.core.types import UserMessage

if TYPE_CHECKING:
    pass


# Minimal 1x1 transparent PNG for testing
MINIMAL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Minimal PDF header for testing (not a valid PDF, just for header validation)
MINIMAL_PDF_BASE64 = base64.b64encode(b"%PDF-1.4\n%test minimal pdf content").decode()

# Long MIME type constants for Office documents
WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


# =============================================================================
# T015: Unit tests for AG-UI event mapping
# =============================================================================


class TestAGUIEventMapping:
    """Tests for AG-UI lifecycle event creation."""

    def test_create_run_started_event(self) -> None:
        """Test RunStartedEvent creation with thread_id and run_id."""
        from holodeck.serve.protocols.agui import create_run_started_event

        event = create_run_started_event(
            thread_id="thread-123",
            run_id="run-456",
        )

        assert event.type == EventType.RUN_STARTED
        assert event.thread_id == "thread-123"
        assert event.run_id == "run-456"

    def test_create_run_finished_event(self) -> None:
        """Test RunFinishedEvent creation with thread_id and run_id."""
        from holodeck.serve.protocols.agui import create_run_finished_event

        event = create_run_finished_event(
            thread_id="thread-123",
            run_id="run-456",
        )

        assert event.type == EventType.RUN_FINISHED
        assert event.thread_id == "thread-123"
        assert event.run_id == "run-456"

    def test_create_run_error_event_with_message(self) -> None:
        """Test RunErrorEvent creation with message only."""
        from holodeck.serve.protocols.agui import create_run_error_event

        event = create_run_error_event(message="Something went wrong")

        assert event.type == EventType.RUN_ERROR
        assert event.message == "Something went wrong"

    def test_create_run_error_event_with_code(self) -> None:
        """Test RunErrorEvent creation with message and code."""
        from holodeck.serve.protocols.agui import create_run_error_event

        event = create_run_error_event(
            message="Rate limit exceeded",
            code="RATE_LIMIT",
        )

        assert event.type == EventType.RUN_ERROR
        assert event.message == "Rate limit exceeded"
        assert event.code == "RATE_LIMIT"


class TestTextMessageEvents:
    """Tests for text message event sequence."""

    def test_create_text_message_start_event(self) -> None:
        """Test TextMessageStartEvent with message_id and role."""
        from holodeck.serve.protocols.agui import create_text_message_start

        event = create_text_message_start(message_id="msg-123")

        assert event.type == EventType.TEXT_MESSAGE_START
        assert event.message_id == "msg-123"
        assert event.role == "assistant"

    def test_create_text_message_content_event(self) -> None:
        """Test TextMessageContentEvent with delta text."""
        from holodeck.serve.protocols.agui import create_text_message_content

        event = create_text_message_content(
            message_id="msg-123",
            delta="Hello, world!",
        )

        assert event.type == EventType.TEXT_MESSAGE_CONTENT
        assert event.message_id == "msg-123"
        assert event.delta == "Hello, world!"

    def test_create_text_message_content_with_whitespace_delta(self) -> None:
        """Test TextMessageContentEvent with whitespace delta."""
        from holodeck.serve.protocols.agui import create_text_message_content

        # AG-UI SDK requires non-empty delta, so test with whitespace
        event = create_text_message_content(
            message_id="msg-123",
            delta=" ",
        )

        assert event.type == EventType.TEXT_MESSAGE_CONTENT
        assert event.delta == " "

    def test_create_text_message_end_event(self) -> None:
        """Test TextMessageEndEvent with message_id."""
        from holodeck.serve.protocols.agui import create_text_message_end

        event = create_text_message_end(message_id="msg-123")

        assert event.type == EventType.TEXT_MESSAGE_END
        assert event.message_id == "msg-123"


class TestToolCallEvents:
    """Tests for tool call event sequence."""

    def test_create_tool_call_start_event(self) -> None:
        """Test ToolCallStartEvent with tool_call_id, name, and parent."""
        from holodeck.serve.protocols.agui import create_tool_call_start

        event = create_tool_call_start(
            tool_call_id="tc-123",
            tool_call_name="search_knowledge_base",
            parent_message_id="msg-456",
        )

        assert event.type == EventType.TOOL_CALL_START
        assert event.tool_call_id == "tc-123"
        assert event.tool_call_name == "search_knowledge_base"
        assert event.parent_message_id == "msg-456"

    def test_create_tool_call_args_event(self) -> None:
        """Test ToolCallArgsEvent with args delta."""
        from holodeck.serve.protocols.agui import create_tool_call_args

        event = create_tool_call_args(
            tool_call_id="tc-123",
            delta='{"query": "return policy"}',
        )

        assert event.type == EventType.TOOL_CALL_ARGS
        assert event.tool_call_id == "tc-123"
        assert event.delta == '{"query": "return policy"}'

    def test_create_tool_call_end_event(self) -> None:
        """Test ToolCallEndEvent with tool_call_id."""
        from holodeck.serve.protocols.agui import create_tool_call_end

        event = create_tool_call_end(tool_call_id="tc-123")

        assert event.type == EventType.TOOL_CALL_END
        assert event.tool_call_id == "tc-123"

    def test_create_tool_call_events_from_execution(self) -> None:
        """Test creating complete tool call event sequence from ToolExecution."""
        from holodeck.serve.protocols.agui import create_tool_call_events

        # Mock ToolExecution
        tool_execution = MagicMock()
        tool_execution.tool_name = "search_knowledge_base"
        tool_execution.parameters = {"query": "return policy", "limit": 10}

        events = create_tool_call_events(
            tool_execution=tool_execution,
            message_id="msg-123",
        )

        assert len(events) == 3
        # Check ToolCallStartEvent
        assert events[0].type == EventType.TOOL_CALL_START
        assert events[0].tool_call_name == "search_knowledge_base"
        assert events[0].parent_message_id == "msg-123"
        # Check ToolCallArgsEvent
        assert events[1].type == EventType.TOOL_CALL_ARGS
        args = json.loads(events[1].delta)
        assert args["query"] == "return policy"
        assert args["limit"] == 10
        # Check ToolCallEndEvent
        assert events[2].type == EventType.TOOL_CALL_END
        # All should have same tool_call_id
        assert (
            events[0].tool_call_id == events[1].tool_call_id == events[2].tool_call_id
        )


# =============================================================================
# T016: Unit tests for RunAgentInput to HoloDeck request mapping
# =============================================================================


class TestRunAgentInputMapping:
    """Tests for mapping RunAgentInput to HoloDeck request."""

    def test_extract_last_user_message(self) -> None:
        """Test extracting the last user message from RunAgentInput.messages."""
        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import AssistantMessage

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello"),
                AssistantMessage(id="msg-2", role="assistant", content="Hi there!"),
                UserMessage(id="msg-3", role="user", content="What's the weather?"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        message = extract_message_from_input(input_data)
        assert message == "What's the weather?"

    def test_extract_message_single_user_message(self) -> None:
        """Test extracting when there's only one user message."""
        from ag_ui.core.events import RunAgentInput

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        message = extract_message_from_input(input_data)
        assert message == "Hello"

    def test_extract_message_empty_messages_raises(self) -> None:
        """Test error handling when messages list is empty."""
        from ag_ui.core.events import RunAgentInput

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        with pytest.raises(ValueError, match="No user messages"):
            extract_message_from_input(input_data)

    def test_extract_message_no_user_messages_raises(self) -> None:
        """Test error handling when no user messages in list."""
        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import AssistantMessage, SystemMessage

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                AssistantMessage(id="msg-1", role="assistant", content="Hi there!"),
                SystemMessage(id="msg-2", role="system", content="You are helpful"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        with pytest.raises(ValueError, match="No user messages"):
            extract_message_from_input(input_data)

    def test_map_thread_id_to_session_id(self) -> None:
        """Test that AG-UI thread_id maps to HoloDeck session_id."""
        from holodeck.serve.protocols.agui import map_session_id

        session_id = map_session_id("thread-123")
        assert session_id == "thread-123"

    def test_map_thread_id_valid_ulid(self) -> None:
        """Test that valid ULID thread_id is preserved."""
        from holodeck.serve.protocols.agui import map_session_id

        ulid_str = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        session_id = map_session_id(ulid_str)
        assert session_id == ulid_str

    def test_generate_unique_run_id(self) -> None:
        """Test that each call generates a unique run_id (ULID)."""
        from holodeck.serve.protocols.agui import generate_run_id

        run_id_1 = generate_run_id()
        run_id_2 = generate_run_id()

        assert run_id_1 != run_id_2
        assert len(run_id_1) == 26  # ULID length
        assert len(run_id_2) == 26


class TestAGUIEventStream:
    """Tests for AG-UI event encoding wrapper."""

    def test_event_stream_default_content_type(self) -> None:
        """Test default content type is text/event-stream."""
        from holodeck.serve.protocols.agui import AGUIEventStream

        stream = AGUIEventStream()
        assert stream.content_type == "text/event-stream"

    def test_event_stream_encode_run_started(self) -> None:
        """Test encoding RunStartedEvent."""
        from holodeck.serve.protocols.agui import (
            AGUIEventStream,
            create_run_started_event,
        )

        stream = AGUIEventStream()
        event = create_run_started_event("thread-1", "run-1")
        encoded = stream.encode(event)

        # Encoder returns string (SSE format) which we convert to bytes
        assert isinstance(encoded, str | bytes)
        # SSE format should contain event type and data
        decoded = encoded if isinstance(encoded, str) else encoded.decode("utf-8")
        assert "RUN_STARTED" in decoded or "run_started" in decoded.lower()

    def test_event_stream_encode_text_message(self) -> None:
        """Test encoding TextMessageContentEvent."""
        from holodeck.serve.protocols.agui import (
            AGUIEventStream,
            create_text_message_content,
        )

        stream = AGUIEventStream()
        event = create_text_message_content("msg-1", "Hello!")
        encoded = stream.encode(event)

        # Encoder returns string (SSE format) which we convert to bytes
        assert isinstance(encoded, str | bytes)
        decoded = encoded if isinstance(encoded, str) else encoded.decode("utf-8")
        assert "Hello!" in decoded


class TestAGUIProtocolProperties:
    """Tests for AGUIProtocol class properties."""

    def test_protocol_name(self) -> None:
        """Test protocol name is 'ag-ui'."""
        from holodeck.serve.protocols.agui import AGUIProtocol

        protocol = AGUIProtocol()
        assert protocol.name == "ag-ui"

    def test_protocol_content_type(self) -> None:
        """Test protocol content type is 'text/event-stream'."""
        from holodeck.serve.protocols.agui import AGUIProtocol

        protocol = AGUIProtocol()
        assert protocol.content_type == "text/event-stream"

    def test_protocol_with_accept_header(self) -> None:
        """Test protocol initialization with accept header."""
        from holodeck.serve.protocols.agui import AGUIProtocol

        protocol = AGUIProtocol(accept_header="text/event-stream")
        assert protocol._accept_header == "text/event-stream"


# =============================================================================
# Additional coverage tests for message extraction edge cases
# =============================================================================


class TestExtractMessageDictFormat:
    """Tests for extract_message_from_input with dict message format.

    At runtime, JSON deserialization may produce dicts instead of Message objects.
    These tests cover lines 65-67 in agui.py.
    """

    def test_extract_message_from_dict_messages(self) -> None:
        """Test extracting message when messages are dicts (runtime JSON)."""
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        # Simulate runtime JSON deserialization where messages are dicts
        input_data = MagicMock()
        input_data.messages = [
            {"id": "msg-1", "role": "system", "content": "You are helpful"},
            {"id": "msg-2", "role": "user", "content": "Hello from dict"},
        ]

        message = extract_message_from_input(input_data)
        assert message == "Hello from dict"

    def test_extract_message_from_dict_with_list_content(self) -> None:
        """Test extracting message from dict with list content parts.

        Covers lines 76-84 where content is a list of content parts.
        """
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = MagicMock()
        input_data.messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " World"},
                ],
            },
        ]

        message = extract_message_from_input(input_data)
        assert message == "Hello  World"

    def test_extract_message_from_dict_with_string_content_parts(self) -> None:
        """Test extracting message from dict with string content parts.

        Covers lines 83-84 where content parts are strings.
        """
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = MagicMock()
        input_data.messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": ["Part 1", "Part 2", "Part 3"],
            },
        ]

        message = extract_message_from_input(input_data)
        assert message == "Part 1 Part 2 Part 3"

    def test_extract_message_from_dict_with_mixed_content_parts(self, caplog) -> None:
        """Test extracting message from dict with mixed content parts."""
        import logging
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = MagicMock()
        input_data.messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": [
                    {"type": "text", "text": "From dict"},
                    "From string",
                    {"type": "image", "url": "http://example.com/image.png"},
                ],
            },
        ]

        with caplog.at_level(logging.WARNING, logger="holodeck.serve.protocols.agui"):
            message = extract_message_from_input(input_data)

        # Only text types and strings should be included
        assert "From dict" in message
        assert "From string" in message
        # Warning should be logged for non-text content
        assert "Skipping non-text content part" in caplog.text
        assert "image" in caplog.text

    def test_extract_message_raises_on_only_non_text_content(self, caplog) -> None:
        """Test that ValueError is raised when content has no text parts."""
        import logging
        from unittest.mock import MagicMock

        import pytest

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = MagicMock()
        input_data.messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": [
                    {"type": "image", "url": "http://example.com/image.png"},
                    {"type": "audio", "url": "http://example.com/audio.mp3"},
                ],
            },
        ]

        with (
            caplog.at_level(logging.WARNING, logger="holodeck.serve.protocols.agui"),
            pytest.raises(ValueError, match="No text content found"),
        ):
            extract_message_from_input(input_data)

        # Warnings should still be logged for skipped content
        assert "Skipping non-text content part" in caplog.text

    def test_extract_message_skips_non_user_dict_messages(self) -> None:
        """Test that non-user dict messages are skipped."""
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        input_data = MagicMock()
        input_data.messages = [
            {"id": "msg-1", "role": "assistant", "content": "I am assistant"},
            {"id": "msg-2", "role": "system", "content": "System message"},
            {"id": "msg-3", "role": "user", "content": "User message"},
        ]

        message = extract_message_from_input(input_data)
        assert message == "User message"

    def test_extract_message_from_message_object_with_list_content(self) -> None:
        """Test extracting message from Message object with list content.

        Tests the getattr path (lines 69-70) with list content.
        """
        from unittest.mock import MagicMock

        from holodeck.serve.protocols.agui import extract_message_from_input

        # Simulate a Message object with list content
        mock_message = MagicMock()
        mock_message.role = "user"
        mock_message.content = [
            {"type": "text", "text": "Hello from object"},
            "Plain string part",
        ]

        input_data = MagicMock()
        input_data.messages = [mock_message]

        message = extract_message_from_input(input_data)
        assert "Hello from object" in message
        assert "Plain string part" in message


# =============================================================================
# Tests for AGUIEventStream binary encoding (lines 158-159)
# =============================================================================


class TestAGUIEventStreamBinaryFormat:
    """Tests for AGUIEventStream with binary format negotiation."""

    def test_event_stream_with_binary_accept_header(self) -> None:
        """Test event stream initialization with binary accept header."""
        from holodeck.serve.protocols.agui import AGUIEventStream

        # Initialize with binary format accept header
        stream = AGUIEventStream(accept_header="application/octet-stream")
        # Content type should reflect the format
        assert stream.encoder is not None

    def test_event_stream_returns_bytes(self) -> None:
        """Test that encode always returns bytes."""
        from holodeck.serve.protocols.agui import (
            AGUIEventStream,
            create_run_started_event,
        )

        stream = AGUIEventStream()
        event = create_run_started_event("thread-1", "run-1")
        encoded = stream.encode(event)

        # Result should be bytes
        assert isinstance(encoded, bytes)


# =============================================================================
# Tests for AGUIProtocol.handle_request
# =============================================================================


class TestAGUIProtocolHandleRequest:
    """Tests for AGUIProtocol.handle_request method."""

    @pytest.mark.asyncio
    async def test_handle_request_success(self) -> None:
        """Test handle_request returns proper event sequence on success."""
        from unittest.mock import AsyncMock, MagicMock

        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import UserMessage

        from holodeck.serve.protocols.agui import AGUIProtocol

        # Create input
        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        # Mock session
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help?"
        mock_response.tool_executions = []

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "session-123"
        mock_session.agent_executor = mock_executor

        # Create protocol and handle request
        protocol = AGUIProtocol()
        events = []
        async for event_bytes in protocol.handle_request(input_data, mock_session):
            events.append(event_bytes)

        # Should have multiple events: RunStarted, TextMessageStart,
        # TextMessageContent, TextMessageEnd, RunFinished
        assert len(events) >= 5

        # Verify events contain expected content
        all_content = b"".join(events).decode("utf-8")
        assert "RUN_STARTED" in all_content or "run_started" in all_content.lower()
        assert "RUN_FINISHED" in all_content or "run_finished" in all_content.lower()

    @pytest.mark.asyncio
    async def test_handle_request_with_tool_executions(self) -> None:
        """Test handle_request includes tool call events."""
        from unittest.mock import AsyncMock, MagicMock

        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import UserMessage

        from holodeck.models.tool_execution import ToolExecution, ToolStatus
        from holodeck.serve.protocols.agui import AGUIProtocol

        # Create input
        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Search for info"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        # Mock response with tool execution
        tool_exec = ToolExecution(
            tool_name="search",
            parameters={"query": "test"},
            status=ToolStatus.SUCCESS,
        )

        mock_response = MagicMock()
        mock_response.content = "Here are the results"
        mock_response.tool_executions = [tool_exec]

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "session-123"
        mock_session.agent_executor = mock_executor

        # Create protocol and handle request
        protocol = AGUIProtocol()
        events = []
        async for event_bytes in protocol.handle_request(input_data, mock_session):
            events.append(event_bytes)

        # Verify tool call events are present
        all_content = b"".join(events).decode("utf-8")
        assert "TOOL_CALL" in all_content or "tool_call" in all_content.lower()

    @pytest.mark.asyncio
    async def test_handle_request_error(self) -> None:
        """Test handle_request emits error event on failure."""
        from unittest.mock import AsyncMock, MagicMock

        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import UserMessage

        from holodeck.serve.protocols.agui import AGUIProtocol

        # Create input
        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        # Mock executor that raises an error
        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(
            side_effect=RuntimeError("Agent execution failed")
        )

        mock_session = MagicMock()
        mock_session.session_id = "session-123"
        mock_session.agent_executor = mock_executor

        # Create protocol and handle request
        protocol = AGUIProtocol()
        events = []
        async for event_bytes in protocol.handle_request(input_data, mock_session):
            events.append(event_bytes)

        # Should contain error event
        all_content = b"".join(events).decode("utf-8")
        assert "RUN_ERROR" in all_content or "run_error" in all_content.lower()
        assert "Agent execution failed" in all_content

    @pytest.mark.asyncio
    async def test_handle_request_with_accept_header(self) -> None:
        """Test handle_request with custom accept header."""
        from unittest.mock import AsyncMock, MagicMock

        from ag_ui.core.events import RunAgentInput
        from ag_ui.core.types import UserMessage

        from holodeck.serve.protocols.agui import AGUIProtocol

        # Create input
        input_data = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello"),
            ],
            state=None,
            tools=[],
            context=[],
            forwarded_props=None,
        )

        # Mock session
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_executions = []

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock()
        mock_session.session_id = "session-123"
        mock_session.agent_executor = mock_executor

        # Create protocol with accept header
        protocol = AGUIProtocol(accept_header="text/event-stream")
        events = []
        async for event_bytes in protocol.handle_request(input_data, mock_session):
            events.append(event_bytes)

        assert len(events) > 0


# =============================================================================
# T029a: Unit tests for AG-UI BinaryInputContent parsing
# =============================================================================


class TestBinaryInputContentParsing:
    """Tests for extracting binary content parts from AG-UI message content."""

    def test_extract_binary_parts_empty_content(self) -> None:
        """Test extracting binary parts from empty content list."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        result = extract_binary_parts_from_content([])
        assert result == []

    def test_extract_binary_parts_text_only(self) -> None:
        """Test extracting binary parts when content only has text."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Hello world"},
            {"type": "text", "text": "More text"},
        ]
        result = extract_binary_parts_from_content(content)
        assert result == []

    def test_extract_binary_parts_with_base64_image(self) -> None:
        """Test extracting binary parts with inline base64 image data."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "binary",
                "mimeType": "image/png",
                "data": MINIMAL_PNG_BASE64,
                "filename": "test.png",
            },
        ]
        result = extract_binary_parts_from_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "binary"
        assert result[0]["mimeType"] == "image/png"
        assert result[0]["data"] == MINIMAL_PNG_BASE64
        assert result[0]["filename"] == "test.png"

    def test_extract_binary_parts_with_url_reference(self) -> None:
        """Test extracting binary parts with URL reference."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "binary",
                "mimeType": "image/jpeg",
                "url": "https://example.com/image.jpg",
            },
        ]
        result = extract_binary_parts_from_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "binary"
        assert result[0]["mimeType"] == "image/jpeg"
        assert result[0]["url"] == "https://example.com/image.jpg"

    def test_extract_binary_parts_with_file_id(self) -> None:
        """Test extracting binary parts with file ID reference."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Process this file"},
            {
                "type": "binary",
                "mimeType": "application/pdf",
                "id": "file-12345",
            },
        ]
        result = extract_binary_parts_from_content(content)

        assert len(result) == 1
        assert result[0]["type"] == "binary"
        assert result[0]["mimeType"] == "application/pdf"
        assert result[0]["id"] == "file-12345"

    def test_extract_binary_parts_mixed_content(self) -> None:
        """Test extracting multiple binary parts from mixed content."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Look at these files:"},
            {
                "type": "binary",
                "mimeType": "image/png",
                "data": MINIMAL_PNG_BASE64,
            },
            {"type": "text", "text": "And also:"},
            {
                "type": "binary",
                "mimeType": "application/pdf",
                "data": MINIMAL_PDF_BASE64,
            },
        ]
        result = extract_binary_parts_from_content(content)

        assert len(result) == 2
        assert result[0]["mimeType"] == "image/png"
        assert result[1]["mimeType"] == "application/pdf"

    def test_extract_binary_parts_skips_unsupported_mime(self, caplog, capsys) -> None:
        """Test that unsupported MIME types are skipped with warning."""
        import logging

        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[dict[str, Any]] = [
            {
                "type": "binary",
                "mimeType": "video/mp4",  # Not supported
                "data": "somebase64data",
            },
            {
                "type": "binary",
                "mimeType": "image/png",  # Supported
                "data": MINIMAL_PNG_BASE64,
            },
        ]

        with caplog.at_level(logging.WARNING):
            result = extract_binary_parts_from_content(content)

        # Only the supported MIME type should be returned
        assert len(result) == 1
        assert result[0]["mimeType"] == "image/png"

        # Warning should be logged for unsupported MIME type
        # Check both caplog.text and stdout (logging handler timing)
        captured_stdout = capsys.readouterr().out
        log_output = caplog.text + captured_stdout
        assert "video/mp4" in log_output or "unsupported" in log_output.lower()

    def test_extract_binary_parts_with_string_parts(self) -> None:
        """Test extracting binary parts ignores plain string parts."""
        from holodeck.serve.protocols.agui import extract_binary_parts_from_content

        content: list[Any] = [
            "Plain string content",
            {"type": "binary", "mimeType": "image/png", "data": MINIMAL_PNG_BASE64},
        ]
        result = extract_binary_parts_from_content(content)

        assert len(result) == 1
        assert result[0]["mimeType"] == "image/png"


# =============================================================================
# T029b: Unit tests for AG-UI binary content to FileInput conversion
# =============================================================================


class TestBinaryContentToFileInput:
    """Tests for converting AG-UI binary content to FileInput."""

    def test_convert_base64_png_creates_temp_file(self) -> None:
        """Test converting base64 PNG to FileInput creates temp file."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "image/png",
            "data": MINIMAL_PNG_BASE64,
            "filename": "test.png",
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "image"
        assert file_input.path is not None
        assert Path(file_input.path).exists()
        assert file_input.description == "test.png"

        # Verify file content
        with open(file_input.path, "rb") as f:
            content = f.read()
        assert content == base64.b64decode(MINIMAL_PNG_BASE64)

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_base64_pdf_creates_temp_file(self) -> None:
        """Test converting base64 PDF to FileInput creates temp file."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "application/pdf",
            "data": MINIMAL_PDF_BASE64,
            "filename": "document.pdf",
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "pdf"
        assert file_input.path is not None
        assert Path(file_input.path).exists()
        assert file_input.path.endswith(".pdf")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_base64_jpeg_creates_temp_file(self) -> None:
        """Test converting base64 JPEG to FileInput."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        # Minimal JPEG-like data
        jpeg_data = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 10).decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "image/jpeg",
            "data": jpeg_data,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "image"
        assert file_input.path.endswith(".jpg")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_url_reference_returns_none_for_security(self, caplog, capsys) -> None:
        """Test that URL references return None (disabled for SSRF security)."""
        import logging

        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "image/png",
            "url": "https://example.com/test.png",
        }

        with caplog.at_level(logging.WARNING, logger="holodeck.serve.file_utils"):
            file_input = convert_binary_dict_to_file_input(binary_content)

        # URL references should return None (disabled for security)
        assert file_input is None
        # Should log a warning about SSRF prevention
        # Check both caplog.text and stdout (logging handler timing)
        captured_stdout = capsys.readouterr().out
        log_output = caplog.text + captured_stdout
        assert "SSRF" in log_output

    def test_file_id_logs_warning_returns_none(self, caplog, capsys) -> None:
        """Test that file ID references log warning and return None."""
        import logging

        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "application/pdf",
            "id": "file-12345",
        }

        with caplog.at_level(logging.WARNING, logger="holodeck.serve.file_utils"):
            file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is None
        # Check both caplog.text and stdout (logging handler timing)
        captured_stdout = capsys.readouterr().out
        log_output = caplog.text + captured_stdout
        assert "file-12345" in log_output or "not supported" in log_output.lower()

    def test_cleanup_removes_temp_files(self) -> None:
        """Test cleanup_temp_file removes temporary files."""
        from holodeck.serve.file_utils import (
            cleanup_temp_file,
            convert_binary_dict_to_file_input,
        )

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "image/png",
            "data": MINIMAL_PNG_BASE64,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)
        assert file_input is not None
        assert Path(file_input.path).exists()

        cleanup_temp_file(file_input)
        assert not Path(file_input.path).exists()

    def test_invalid_base64_raises_error(self) -> None:
        """Test that invalid base64 data raises an error."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "image/png",
            "data": "not-valid-base64!!!",
        }

        with pytest.raises(ValueError, match="Invalid base64"):
            convert_binary_dict_to_file_input(binary_content)

    def test_convert_word_document(self) -> None:
        """Test converting Word document MIME type."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        docx_data = base64.b64encode(b"PK\x03\x04" + b"\x00" * 20).decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": WORD_MIME,
            "data": docx_data,
            "filename": "doc.docx",
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "word"
        assert file_input.path.endswith(".docx")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_excel_spreadsheet(self) -> None:
        """Test converting Excel spreadsheet MIME type."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        xlsx_data = base64.b64encode(b"PK\x03\x04" + b"\x00" * 20).decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": EXCEL_MIME,
            "data": xlsx_data,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "excel"
        assert file_input.path.endswith(".xlsx")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_powerpoint_presentation(self) -> None:
        """Test converting PowerPoint presentation MIME type."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        pptx_data = base64.b64encode(b"PK\x03\x04" + b"\x00" * 20).decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": PPTX_MIME,
            "data": pptx_data,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "powerpoint"
        assert file_input.path.endswith(".pptx")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_text_file(self) -> None:
        """Test converting text file MIME type."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        text_data = base64.b64encode(b"Hello, this is a text file.").decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "text/plain",
            "data": text_data,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "text"
        assert file_input.path.endswith(".txt")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)

    def test_convert_csv_file(self) -> None:
        """Test converting CSV file MIME type."""
        from holodeck.serve.file_utils import convert_binary_dict_to_file_input

        csv_data = base64.b64encode(b"name,value\nfoo,1\nbar,2").decode()
        binary_content: dict[str, Any] = {
            "type": "binary",
            "mimeType": "text/csv",
            "data": csv_data,
        }

        file_input = convert_binary_dict_to_file_input(binary_content)

        assert file_input is not None
        assert file_input.type == "csv"
        assert file_input.path.endswith(".csv")

        # Cleanup
        Path(file_input.path).unlink(missing_ok=True)
