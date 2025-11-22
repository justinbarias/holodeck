"""Tests for chat-related models."""

from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from semantic_kernel.contents import ChatHistory

from holodeck.models.agent import Agent, Instructions
from holodeck.models.chat import ChatConfig, ChatSession, Message, MessageRole
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus


def _make_agent() -> Agent:
    """Create a minimal Agent instance for tests."""
    return Agent(
        name="test-agent",
        description="desc",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
    )


class TestMessage:
    """Message validation behavior."""

    def test_user_message_trims_and_accepts_content(self) -> None:
        """User content is stripped and stored."""
        message = Message(role=MessageRole.USER, content="  hello  ")
        assert message.content == "hello"
        assert message.timestamp <= datetime.utcnow()

    def test_user_message_over_limit_rejected(self) -> None:
        """User content over 10k chars is rejected."""
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="a" * 10001)

    def test_tool_calls_only_allowed_for_assistant(self) -> None:
        """Non-assistant role cannot hold tool calls."""
        tool_call = ToolExecution(tool_name="echo", status=ToolStatus.SUCCESS)
        with pytest.raises(ValidationError):
            Message(
                role=MessageRole.USER,
                content="hello",
                tool_calls=[tool_call],
            )


class TestChatSession:
    """ChatSession validation behavior."""

    def test_future_started_at_rejected(self) -> None:
        """started_at cannot be set in the future."""
        future = datetime.utcnow() + timedelta(days=1)
        with pytest.raises(ValidationError):
            ChatSession(
                agent_config=_make_agent(),
                history=ChatHistory(),
                started_at=future,
            )

    def test_session_defaults(self) -> None:
        """Defaults populate ids and counters."""
        session = ChatSession(agent_config=_make_agent(), history=ChatHistory())
        assert isinstance(session.session_id, UUID)
        assert session.message_count == 0
        assert session.state.value == "initializing"


class TestChatConfig:
    """ChatConfig validation behavior."""

    def test_path_must_exist_and_be_file(self, tmp_path: Path) -> None:
        """Config path must exist and be a file."""
        missing = tmp_path / "missing.yaml"
        with pytest.raises(ValidationError):
            ChatConfig(agent_config_path=missing)

        directory = tmp_path / "dir"
        directory.mkdir()
        with pytest.raises(ValidationError):
            ChatConfig(agent_config_path=directory)

    def test_positive_max_messages(self, tmp_path: Path) -> None:
        """max_messages must be positive."""
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text("name: test")
        with pytest.raises(ValidationError):
            ChatConfig(agent_config_path=agent_path, max_messages=0)


class TestTokenAndTools:
    """Token and tool validations used by messages."""

    def test_token_usage_must_match_totals(self) -> None:
        """total_tokens must equal prompt + completion."""
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=2)

    def test_tool_execution_failed_requires_error(self) -> None:
        """Failed tool calls must include an error message."""
        with pytest.raises(ValidationError):
            ToolExecution(tool_name="echo", status=ToolStatus.FAILED)
