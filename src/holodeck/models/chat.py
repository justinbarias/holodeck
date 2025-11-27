"""Chat-related models for interactive sessions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
from semantic_kernel.contents import ChatHistory

from holodeck.models.agent import Agent
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution


class SessionState(str, Enum):
    """Chat session lifecycle state."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    TERMINATED = "terminated"


class MessageRole(str, Enum):
    """Role of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Chat message model."""

    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: list[ToolExecution] = Field(default_factory=list)
    tokens_used: TokenUsage | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str, info: Any) -> str:
        """Strip content and enforce size limits."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty")
        role = info.data.get("role")
        if role == MessageRole.USER and len(cleaned) > 10_000:
            raise ValueError("User message exceeds 10,000 character limit")
        return cleaned

    @field_validator("tool_calls")
    @classmethod
    def validate_tool_calls(
        cls, calls: list[ToolExecution], info: Any
    ) -> list[ToolExecution]:
        """Ensure tool calls are only attached to assistant messages."""
        if calls and info.data.get("role") != MessageRole.ASSISTANT:
            raise ValueError("Tool calls only allowed for assistant messages")
        return calls


class ChatSession(BaseModel):
    """Interactive chat session model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: UUID = Field(default_factory=uuid4)
    agent_config: Agent
    history: ChatHistory
    started_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, ge=0)
    state: SessionState = Field(default=SessionState.INITIALIZING)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: datetime) -> datetime:
        """Ensure started_at is not in the future."""
        if value > datetime.utcnow():
            raise ValueError("started_at cannot be in the future")
        return value


class ChatConfig(BaseModel):
    """Runtime configuration for chat sessions."""

    agent_config_path: Path
    verbose: bool = Field(default=False)
    enable_observability: bool = Field(default=False)
    max_messages: int = Field(default=50, gt=0)
    force_ingest: bool = Field(default=False)
    llm_timeout: int | None = Field(
        default=None, description="LLM API call timeout in seconds"
    )

    @field_validator("agent_config_path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        """Ensure the agent config path exists and is a file."""
        if not value.exists():
            raise ValueError(f"Agent config not found: {value}")
        if not value.is_file():
            raise ValueError(f"Path is not a file: {value}")
        return value
