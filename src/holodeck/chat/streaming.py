"""Tool execution streaming utilities."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from holodeck.models.tool_event import ToolEvent, ToolEventType
from holodeck.models.tool_execution import ToolExecution, ToolStatus

# Re-export for convenient access
__all__ = ["ToolExecutionStream", "ToolEvent", "ToolEventType"]


class ToolExecutionStream:
    """Streams tool execution events to the caller.

    Emits ToolEvent instances as a tool executes, allowing callers
    to display real-time progress. Supports both standard and verbose modes.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the stream with verbosity preference.

        Args:
            verbose: If True, include detailed execution data (parameters, results).
                    If False, emit minimal data (tool name, status, timing).
        """
        self.verbose = verbose

    async def stream_execution(
        self, tool_call: ToolExecution
    ) -> AsyncIterator[ToolEvent]:
        """Stream execution events for a tool call.

        Simulates the execution lifecycle by emitting events:
        1. STARTED - Tool execution begins
        2. PROGRESS - (optional, for long operations)
        3. COMPLETED or FAILED - Execution finished

        Args:
            tool_call: Tool execution with status and result data.

        Yields:
            ToolEvent instances representing execution progression.
        """
        # Emit STARTED event
        yield ToolEvent(
            event_type=ToolEventType.STARTED,
            tool_name=tool_call.tool_name,
            timestamp=datetime.utcnow(),
            data={"parameters": tool_call.parameters} if self.verbose else {},
        )

        # Emit COMPLETED or FAILED based on status
        if tool_call.status == ToolStatus.SUCCESS:
            yield ToolEvent(
                event_type=ToolEventType.COMPLETED,
                tool_name=tool_call.tool_name,
                timestamp=datetime.utcnow(),
                data={
                    "result": tool_call.result if self.verbose else None,
                    "execution_time": tool_call.execution_time or 0,
                },
            )
        elif tool_call.status == ToolStatus.FAILED:
            yield ToolEvent(
                event_type=ToolEventType.FAILED,
                tool_name=tool_call.tool_name,
                timestamp=datetime.utcnow(),
                data={
                    "error": tool_call.error_message or "Unknown error",
                },
            )
        else:
            # For PENDING or RUNNING, emit as PROGRESS
            yield ToolEvent(
                event_type=ToolEventType.PROGRESS,
                tool_name=tool_call.tool_name,
                timestamp=datetime.utcnow(),
                data={"status": tool_call.status.value},
            )
