"""Unit tests for Claude SDK tool event hooks.

Tests for:
- _build_tool_hooks() — PreToolUse, PostToolUse, PostToolUseFailure hooks
- _maybe_emit_subagent_message() — subagent_message events from msg stream
- ClaudeSession.tool_events — queue property
- _TaskBoundSession.tool_events — passthrough property
- AgentExecutor.tool_event_queue — passthrough property
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from holodeck.lib.backends.base import ToolEvent
from holodeck.lib.backends.claude_backend import (
    _build_tool_hooks,
    _maybe_emit_subagent_message,
)

# ---------------------------------------------------------------------------
# Lightweight stand-ins for SDK message types — only the class name and the
# attributes used by the production code are reproduced here.
# ---------------------------------------------------------------------------


class TextBlock:
    """Stand-in for ``claude_agent_sdk.TextBlock``."""

    def __init__(self, text: str) -> None:
        self.text = text


class AssistantMessage:
    """Stand-in for ``claude_agent_sdk.AssistantMessage``."""

    def __init__(
        self,
        content: list[object],
        parent_tool_use_id: str | None = None,
    ) -> None:
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id


class ResultMessage:
    """Stand-in for ``claude_agent_sdk.ResultMessage``."""

    def __init__(self, parent_tool_use_id: str | None = None) -> None:
        self.parent_tool_use_id = parent_tool_use_id


# ---------------------------------------------------------------------------
# _build_tool_hooks
# ---------------------------------------------------------------------------


class TestBuildToolHooks:
    """Tests for _build_tool_hooks hook factory."""

    def test_returns_three_hook_events(self) -> None:
        """Hook dict contains PreToolUse, PostToolUse, PostToolUseFailure."""
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        hooks = _build_tool_hooks(queue)

        assert "PreToolUse" in hooks
        assert "PostToolUse" in hooks
        assert "PostToolUseFailure" in hooks

    def test_each_event_has_one_matcher(self) -> None:
        """Each hook event has exactly one HookMatcher."""
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        hooks = _build_tool_hooks(queue)

        for matchers in hooks.values():
            assert len(matchers) == 1

    @pytest.mark.asyncio
    async def test_pre_tool_use_pushes_start_event(self) -> None:
        """PreToolUse hook pushes a 'start' ToolEvent."""
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        hooks = _build_tool_hooks(queue)

        callback = hooks["PreToolUse"][0].hooks[0]
        input_data = {
            "tool_name": "search",
            "tool_use_id": "tu_123",
            "tool_input": {"query": "hello"},
            "session_id": "s1",
            "transcript_path": "/tmp/t",  # noqa: S108
            "cwd": "/tmp",  # noqa: S108
        }
        result = await callback(input_data, None, {})

        assert result == {}
        assert not queue.empty()
        event = queue.get_nowait()
        assert event.kind == "start"
        assert event.tool_name == "search"
        assert event.tool_use_id == "tu_123"
        assert event.tool_input == {"query": "hello"}

    @pytest.mark.asyncio
    async def test_post_tool_use_pushes_end_event(self) -> None:
        """PostToolUse hook pushes an 'end' ToolEvent."""
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        hooks = _build_tool_hooks(queue)

        callback = hooks["PostToolUse"][0].hooks[0]
        input_data = {
            "tool_name": "search",
            "tool_use_id": "tu_123",
            "tool_input": {"query": "hello"},
            "tool_response": "found 3 results",
            "session_id": "s1",
            "transcript_path": "/tmp/t",  # noqa: S108
            "cwd": "/tmp",  # noqa: S108
        }
        result = await callback(input_data, None, {})

        assert result == {}
        event = queue.get_nowait()
        assert event.kind == "end"
        assert event.tool_name == "search"
        assert event.tool_use_id == "tu_123"
        assert event.tool_response == "found 3 results"

    @pytest.mark.asyncio
    async def test_post_tool_failure_pushes_error_event(self) -> None:
        """PostToolUseFailure hook pushes an 'error' ToolEvent."""
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        hooks = _build_tool_hooks(queue)

        callback = hooks["PostToolUseFailure"][0].hooks[0]
        input_data = {
            "tool_name": "search",
            "tool_use_id": "tu_456",
            "tool_input": {"query": "hello"},
            "error": "connection timeout",
            "session_id": "s1",
            "transcript_path": "/tmp/t",  # noqa: S108
            "cwd": "/tmp",  # noqa: S108
        }
        result = await callback(input_data, None, {})

        assert result == {}
        event = queue.get_nowait()
        assert event.kind == "error"
        assert event.tool_name == "search"
        assert event.tool_use_id == "tu_456"
        assert event.error == "connection timeout"


# ---------------------------------------------------------------------------
# _maybe_emit_subagent_message
# ---------------------------------------------------------------------------


class TestMaybeEmitSubagentMessage:
    """Tests for the message-stream tap that surfaces subagent text."""

    def test_pushes_subagent_message_event_when_parent_id_set(self) -> None:
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        msg = AssistantMessage(
            content=[TextBlock("Reading src/foo.py")],
            parent_tool_use_id="task_42",
        )

        _maybe_emit_subagent_message(msg, queue)

        evt = queue.get_nowait()
        assert evt.kind == "subagent_message"
        assert evt.tool_name == "Task"
        assert evt.tool_use_id == "task_42"
        assert evt.parent_tool_use_id == "task_42"
        assert evt.text == "Reading src/foo.py"

    def test_no_event_when_parent_id_missing(self) -> None:
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        msg = AssistantMessage(content=[TextBlock("hi")], parent_tool_use_id=None)

        _maybe_emit_subagent_message(msg, queue)

        assert queue.empty()

    def test_no_event_for_non_assistant_messages(self) -> None:
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        msg = ResultMessage(parent_tool_use_id="task_1")

        _maybe_emit_subagent_message(msg, queue)

        assert queue.empty()

    def test_no_event_for_empty_text(self) -> None:
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        msg = AssistantMessage(content=[TextBlock("   \n  ")], parent_tool_use_id="t")

        _maybe_emit_subagent_message(msg, queue)

        assert queue.empty()


# ---------------------------------------------------------------------------
# ClaudeSession.tool_events
# ---------------------------------------------------------------------------


class TestClaudeSessionToolEvents:
    """Tests for ClaudeSession.tool_events property."""

    def test_returns_queue(self) -> None:
        """tool_events returns an asyncio.Queue."""
        from holodeck.lib.backends.claude_backend import ClaudeSession

        mock_options = MagicMock()
        mock_options.hooks = None
        session = ClaudeSession(mock_options)

        assert isinstance(session.tool_events, asyncio.Queue)

    def test_returns_same_queue_instance(self) -> None:
        """tool_events returns the same queue on repeated access."""
        from holodeck.lib.backends.claude_backend import ClaudeSession

        mock_options = MagicMock()
        mock_options.hooks = None
        session = ClaudeSession(mock_options)

        assert session.tool_events is session.tool_events


# ---------------------------------------------------------------------------
# _TaskBoundSession.tool_events passthrough
# ---------------------------------------------------------------------------


class TestTaskBoundSessionToolEvents:
    """Tests for _TaskBoundSession.tool_events passthrough."""

    def test_passes_through_inner_session_queue(self) -> None:
        """tool_events returns the inner session's queue."""
        from holodeck.chat.executor import _TaskBoundSession

        inner = MagicMock()
        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        inner.tool_events = queue

        wrapper = _TaskBoundSession(inner)
        assert wrapper.tool_events is queue

    def test_returns_none_when_inner_has_no_queue(self) -> None:
        """tool_events returns None when inner session lacks the property."""
        from holodeck.chat.executor import _TaskBoundSession

        inner = MagicMock(spec=["send", "send_streaming", "close"])
        wrapper = _TaskBoundSession(inner)
        assert wrapper.tool_events is None


# ---------------------------------------------------------------------------
# AgentExecutor.tool_event_queue passthrough
# ---------------------------------------------------------------------------


class TestAgentExecutorToolEventQueue:
    """Tests for AgentExecutor.tool_event_queue property."""

    def test_returns_none_before_session_init(self) -> None:
        """tool_event_queue is None when session not yet initialized."""
        from holodeck.chat.executor import AgentExecutor

        agent = MagicMock()
        agent.name = "test"
        executor = AgentExecutor(agent)
        assert executor.tool_event_queue is None

    def test_returns_queue_from_session(self) -> None:
        """tool_event_queue returns the session's tool_events queue."""
        from holodeck.chat.executor import AgentExecutor

        agent = MagicMock()
        agent.name = "test"
        executor = AgentExecutor(agent)

        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        mock_session = MagicMock()
        mock_session.tool_events = queue
        executor._session = mock_session

        assert executor.tool_event_queue is queue
