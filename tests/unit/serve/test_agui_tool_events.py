"""Unit tests for AG-UI real-time tool event streaming.

Tests for:
- _tool_event_to_agui() — ToolEvent to AG-UI event conversion
- handle_request() — real-time tool events via queue drain
- handle_request() — SK fallback (post-hoc tool events)
"""

from __future__ import annotations

import json

from ag_ui.core.events import EventType

from holodeck.lib.backends.base import ToolEvent
from holodeck.serve.protocols.agui import _tool_event_to_agui

# ---------------------------------------------------------------------------
# _tool_event_to_agui converter
# ---------------------------------------------------------------------------


class TestToolEventToAgui:
    """Tests for _tool_event_to_agui converter."""

    def test_start_event_produces_start_and_args(self) -> None:
        """Start event -> [TextMessageStart, ToolCallStart, ToolCallArgs]."""
        event = ToolEvent(
            kind="start",
            tool_name="search",
            tool_use_id="tu_123",
            tool_input={"query": "hello"},
        )
        tool_msg_ids: dict[str, str] = {}
        events = _tool_event_to_agui(event, tool_msg_ids)

        assert len(events) == 3
        assert events[0].type == EventType.TEXT_MESSAGE_START
        assert events[1].type == EventType.TOOL_CALL_START
        assert events[1].tool_call_id == "tu_123"
        assert events[1].tool_call_name == "search"
        assert events[2].type == EventType.TOOL_CALL_ARGS
        assert events[2].tool_call_id == "tu_123"
        assert json.loads(events[2].delta) == {"query": "hello"}

        # tool_msg_ids should be populated
        assert "tu_123" in tool_msg_ids

    def test_start_event_does_not_emit_end(self) -> None:
        """Start event must NOT include ToolCallEnd (tool is still running)."""
        event = ToolEvent(
            kind="start",
            tool_name="search",
            tool_use_id="tu_123",
        )
        tool_msg_ids: dict[str, str] = {}
        events = _tool_event_to_agui(event, tool_msg_ids)

        event_types = [e.type for e in events]
        assert EventType.TOOL_CALL_END not in event_types

    def test_start_event_with_none_input(self) -> None:
        """Start event with None tool_input serializes as empty dict."""
        event = ToolEvent(
            kind="start",
            tool_name="noop",
            tool_use_id="tu_0",
            tool_input=None,
        )
        tool_msg_ids: dict[str, str] = {}
        events = _tool_event_to_agui(event, tool_msg_ids)

        assert len(events) == 3
        assert json.loads(events[2].delta) == {}

    def test_end_event_produces_end_result_and_message_end(self) -> None:
        """End event -> [ToolCallEnd, ToolCallResult, TextMessageEnd]."""
        event = ToolEvent(
            kind="end",
            tool_name="search",
            tool_use_id="tu_123",
            tool_response="found 3 results",
        )
        # Simulate a prior start having populated the mapping
        tool_msg_ids: dict[str, str] = {"tu_123": "parent_msg_1"}
        events = _tool_event_to_agui(event, tool_msg_ids)

        assert len(events) == 3
        assert events[0].type == EventType.TOOL_CALL_END
        assert events[0].tool_call_id == "tu_123"
        assert events[1].type == EventType.TOOL_CALL_RESULT
        assert events[1].tool_call_id == "tu_123"
        assert events[1].content == "found 3 results"
        assert events[1].role == "tool"
        assert events[2].type == EventType.TEXT_MESSAGE_END

        # tool_msg_ids should be consumed
        assert "tu_123" not in tool_msg_ids

    def test_end_event_with_none_response(self) -> None:
        """End event with None tool_response uses empty string."""
        event = ToolEvent(
            kind="end",
            tool_name="search",
            tool_use_id="tu_123",
            tool_response=None,
        )
        tool_msg_ids: dict[str, str] = {"tu_123": "parent_msg_1"}
        events = _tool_event_to_agui(event, tool_msg_ids)

        assert events[1].content == ""

    def test_error_event_produces_end_result_and_message_end(self) -> None:
        """Error event -> [ToolCallEnd, ToolCallResult, TextMessageEnd]."""
        event = ToolEvent(
            kind="error",
            tool_name="search",
            tool_use_id="tu_456",
            error="connection timeout",
        )
        tool_msg_ids: dict[str, str] = {"tu_456": "parent_msg_2"}
        events = _tool_event_to_agui(event, tool_msg_ids)

        assert len(events) == 3
        assert events[0].type == EventType.TOOL_CALL_END
        assert events[0].tool_call_id == "tu_456"
        assert events[1].type == EventType.TOOL_CALL_RESULT
        assert events[1].tool_call_id == "tu_456"
        assert events[1].content == "Error: connection timeout"
        assert events[1].role == "tool"
        assert events[2].type == EventType.TEXT_MESSAGE_END

        # tool_msg_ids should be consumed
        assert "tu_456" not in tool_msg_ids

    def test_uses_tool_use_id_as_tool_call_id(self) -> None:
        """tool_use_id from SDK is used as tool_call_id in AG-UI events."""
        event = ToolEvent(
            kind="start",
            tool_name="foo",
            tool_use_id="sdk_unique_id",
        )
        tool_msg_ids: dict[str, str] = {}
        events = _tool_event_to_agui(event, tool_msg_ids)

        # ToolCallStart and ToolCallArgs should use the tool_use_id
        for evt in events:
            if hasattr(evt, "tool_call_id"):
                assert evt.tool_call_id == "sdk_unique_id"
