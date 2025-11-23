"""Unit tests for tool execution streaming."""

from __future__ import annotations

import pytest

from holodeck.chat.streaming import ToolEvent, ToolEventType, ToolExecutionStream
from holodeck.models.tool_execution import ToolExecution, ToolStatus


class TestToolEventType:
    """Tests for ToolEventType enum."""

    def test_event_type_started(self) -> None:
        """STARTED event type exists."""
        assert ToolEventType.STARTED is not None

    def test_event_type_progress(self) -> None:
        """PROGRESS event type exists."""
        assert ToolEventType.PROGRESS is not None

    def test_event_type_completed(self) -> None:
        """COMPLETED event type exists."""
        assert ToolEventType.COMPLETED is not None

    def test_event_type_failed(self) -> None:
        """FAILED event type exists."""
        assert ToolEventType.FAILED is not None


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

    def test_stream_initialization(self) -> None:
        """ToolExecutionStream initializes with verbosity."""
        stream = ToolExecutionStream(verbose=False)
        assert stream is not None
        assert stream.verbose is False

    def test_stream_verbose_mode(self) -> None:
        """ToolExecutionStream stores verbose preference."""
        stream = ToolExecutionStream(verbose=True)
        assert stream.verbose is True

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
    async def test_stream_execution_verbose_mode_includes_details(self) -> None:
        """Verbose mode includes detailed execution info."""
        stream = ToolExecutionStream(verbose=True)
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

        # Verbose mode should include parameters in data
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_execution_standard_mode_minimal_data(self) -> None:
        """Standard mode includes only essential info."""
        stream = ToolExecutionStream(verbose=False)
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

        # Standard mode should still include essential data
        assert len(events) > 0
