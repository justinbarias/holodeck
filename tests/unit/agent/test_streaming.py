"""Unit tests for tool execution streaming."""

from __future__ import annotations

import pytest

from holodeck.chat.streaming import ToolEvent, ToolEventType, ToolExecutionStream
from holodeck.models.tool_execution import ToolExecution, ToolStatus


class TestToolEvent:
    """Tests for ToolEvent model."""

    def test_tool_event_creation(self) -> None:
        """ToolEvent can be instantiated."""
        event = ToolEvent(
            event_type=ToolEventType.STARTED,
            tool_name="test_tool",
            data={},
        )
        assert event is not None

    def test_tool_event_with_data(self) -> None:
        """ToolEvent can store execution data."""
        event = ToolEvent(
            event_type=ToolEventType.COMPLETED,
            tool_name="test_tool",
            data={"result": "success", "execution_time": 0.5},
        )
        assert event.data["result"] == "success"
        assert event.data["execution_time"] == 0.5

    def test_tool_event_with_error_data(self) -> None:
        """ToolEvent can store error data."""
        event = ToolEvent(
            event_type=ToolEventType.FAILED,
            tool_name="test_tool",
            data={"error": "Tool not found"},
        )
        assert event.data["error"] == "Tool not found"


class TestToolExecutionStream:
    """Tests for ToolExecutionStream."""

    @pytest.mark.parametrize(
        "verbose, expected_verbose",
        [
            pytest.param(False, False, id="non_verbose"),
            pytest.param(True, True, id="verbose"),
        ],
    )
    def test_stream_initialization(self, verbose: bool, expected_verbose: bool) -> None:
        """ToolExecutionStream initializes with correct verbosity."""
        stream = ToolExecutionStream(verbose=verbose)
        assert stream is not None
        assert stream.verbose is expected_verbose

    def test_stream_default_not_verbose(self) -> None:
        """ToolExecutionStream defaults to non-verbose."""
        stream = ToolExecutionStream()
        assert stream.verbose is False

    @pytest.mark.asyncio
    async def test_stream_execution_success(self) -> None:
        """Stream emits events for successful tool execution."""
        stream = ToolExecutionStream(verbose=False)
        tool_call = ToolExecution(
            tool_name="test_tool",
            parameters={"arg": "value"},
            result="success result",
            status=ToolStatus.SUCCESS,
            execution_time=0.5,
        )

        events = []
        async for event in stream.stream_execution(tool_call):
            events.append(event)

        # Should emit at least START and COMPLETED
        assert len(events) >= 2
        assert events[0].event_type == ToolEventType.STARTED
        assert events[-1].event_type == ToolEventType.COMPLETED

    @pytest.mark.asyncio
    async def test_stream_execution_failure(self) -> None:
        """Stream emits events for failed tool execution."""
        stream = ToolExecutionStream(verbose=False)
        tool_call = ToolExecution(
            tool_name="test_tool",
            parameters={"arg": "value"},
            status=ToolStatus.FAILED,
            error_message="Tool execution failed",
        )

        events = []
        async for event in stream.stream_execution(tool_call):
            events.append(event)

        # Should emit START and FAILED
        assert len(events) >= 2
        assert events[0].event_type == ToolEventType.STARTED
        assert events[-1].event_type == ToolEventType.FAILED

    @pytest.mark.asyncio
    async def test_stream_execution_includes_tool_name(self) -> None:
        """Stream events include tool name."""
        stream = ToolExecutionStream()
        tool_call = ToolExecution(
            tool_name="my_special_tool",
            parameters={},
            result="ok",
            status=ToolStatus.SUCCESS,
        )

        events = []
        async for event in stream.stream_execution(tool_call):
            events.append(event)

        assert all(e.tool_name == "my_special_tool" for e in events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "verbose",
        [
            pytest.param(True, id="verbose_mode"),
            pytest.param(False, id="standard_mode"),
        ],
    )
    async def test_stream_execution_emits_events(self, verbose: bool) -> None:
        """Stream emits events in both verbose and standard modes."""
        stream = ToolExecutionStream(verbose=verbose)
        tool_call = ToolExecution(
            tool_name="test_tool",
            parameters={"key": "value"},
            result="test result",
            status=ToolStatus.SUCCESS,
            execution_time=1.5,
        )

        events = []
        async for event in stream.stream_execution(tool_call):
            events.append(event)

        assert len(events) > 0
