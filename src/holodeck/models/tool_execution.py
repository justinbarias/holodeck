"""Tool execution metadata models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from holodeck.lib.validation import sanitize_tool_output


class ToolStatus(str, Enum):
    """Tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ToolExecution(BaseModel):
    """Tool execution metadata."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    status: ToolStatus = Field(default=ToolStatus.PENDING)
    execution_time: float | None = Field(default=None, ge=0)
    error_message: str | None = None

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, value: str | None, info: Any) -> str | None:
        """Require error_message when status is FAILED."""
        if info.data.get("status") == ToolStatus.FAILED and not value:
            raise ValueError("error_message required when status=FAILED")
        return value

    @field_validator("result")
    @classmethod
    def sanitize_result(cls, value: str | None) -> str | None:
        """Sanitize tool output before display."""
        if value is None:
            return None
        return sanitize_tool_output(value)

    @model_validator(mode="after")
    def validate_failure(self) -> ToolExecution:
        """Ensure failed executions include an error message."""
        if self.status == ToolStatus.FAILED and not self.error_message:
            raise ValueError("error_message required when status=FAILED")
        return self
