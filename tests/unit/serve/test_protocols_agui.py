"""Unit tests for AG-UI protocol adapter.

Tests for:
- T015: AG-UI event mapping (lifecycle, text message, tool call events)
- T016: RunAgentInput to HoloDeck request mapping
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from ag_ui.core.events import (
    EventType,
)
from ag_ui.core.types import UserMessage

if TYPE_CHECKING:
    pass


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
