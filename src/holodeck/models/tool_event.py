"""Tool execution event models for streaming."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolEventType(str, Enum):
    """Type of tool execution event."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolEvent(BaseModel):
    """Event emitted during tool execution.

    Represents a single event in the lifecycle of a tool execution,
    used for streaming updates about tool progress and results.
    """

    event_type: ToolEventType
    tool_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return human-readable event description."""
        if self.event_type == ToolEventType.STARTED:
            return f"🚀 {self.tool_name} started"
        elif self.event_type == ToolEventType.PROGRESS:
            progress = self.data.get("progress", "")
            return f"⏳ {self.tool_name}: {progress}"
        elif self.event_type == ToolEventType.COMPLETED:
            execution_time = self.data.get("execution_time", 0)
            return f"✅ {self.tool_name} completed in {execution_time:.2f}s"
        # event_type == ToolEventType.FAILED
        error = self.data.get("error", "Unknown error")
        return f"❌ {self.tool_name} failed: {error}"
