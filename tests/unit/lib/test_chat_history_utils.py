"""Unit tests for holodeck.lib.chat_history_utils module."""

from unittest.mock import MagicMock

import pytest

from holodeck.lib.chat_history_utils import (
    extract_last_assistant_content,
    extract_tool_names,
)


@pytest.mark.unit
class TestExtractLastAssistantContent:
    """Tests for extract_last_assistant_content function."""

    def test_extracts_content_from_last_assistant_message(self) -> None:
        """Test extraction of content from the last assistant message."""
        # Create mock messages
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "Hi there!"

        # Create mock history
        history = MagicMock()
        history.messages = [user_msg, assistant_msg]

        result = extract_last_assistant_content(history)
        assert result == "Hi there!"

    def test_returns_last_assistant_when_multiple_exist(self) -> None:
        """Test returns content from most recent assistant message."""
        assistant1 = MagicMock()
        assistant1.role = "assistant"
        assistant1.content = "First response"

        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Follow up"

        assistant2 = MagicMock()
        assistant2.role = "assistant"
        assistant2.content = "Second response"

        history = MagicMock()
        history.messages = [assistant1, user_msg, assistant2]

        result = extract_last_assistant_content(history)
        assert result == "Second response"

    def test_returns_empty_string_for_empty_history(self) -> None:
        """Test returns empty string when history has no messages."""
        history = MagicMock()
        history.messages = []

        result = extract_last_assistant_content(history)
        assert result == ""

    def test_returns_empty_string_for_none_history(self) -> None:
        """Test returns empty string when history is None."""
        result = extract_last_assistant_content(None)  # type: ignore
        assert result == ""

    def test_returns_empty_string_when_no_assistant_messages(self) -> None:
        """Test returns empty string when no assistant messages exist."""
        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        history = MagicMock()
        history.messages = [user_msg]

        result = extract_last_assistant_content(history)
        assert result == ""

    def test_handles_none_content_gracefully(self) -> None:
        """Test handles assistant message with None content."""
        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.content = None

        history = MagicMock()
        history.messages = [assistant_msg]

        result = extract_last_assistant_content(history)
        assert result == ""

    def test_handles_history_without_messages_attribute(self) -> None:
        """Test handles history object without messages attribute."""
        history = MagicMock(spec=[])  # No attributes

        result = extract_last_assistant_content(history)
        assert result == ""


@pytest.mark.unit
class TestExtractToolNames:
    """Tests for extract_tool_names function."""

    def test_extracts_names_from_tool_calls(self) -> None:
        """Test extraction of tool names from tool call list."""
        tool_calls = [
            {"name": "search", "arguments": {"query": "test"}},
            {"name": "fetch", "arguments": {"url": "http://example.com"}},
        ]

        result = extract_tool_names(tool_calls)
        assert result == ["search", "fetch"]

    def test_returns_empty_list_for_empty_tool_calls(self) -> None:
        """Test returns empty list when no tool calls provided."""
        result = extract_tool_names([])
        assert result == []

    def test_skips_entries_without_name_key(self) -> None:
        """Test skips tool call entries that don't have 'name' key."""
        tool_calls = [
            {"name": "valid_tool", "arguments": {}},
            {"arguments": {"param": "value"}},  # Missing 'name'
            {"name": "another_tool", "arguments": {}},
        ]

        result = extract_tool_names(tool_calls)
        assert result == ["valid_tool", "another_tool"]

    def test_handles_empty_name_value(self) -> None:
        """Test handles tool calls with empty string names."""
        tool_calls = [
            {"name": "", "arguments": {}},
            {"name": "valid", "arguments": {}},
        ]

        result = extract_tool_names(tool_calls)
        # Empty string is still a valid value from get()
        assert result == ["", "valid"]

    def test_preserves_tool_call_order(self) -> None:
        """Test preserves the order of tool calls in output."""
        tool_calls = [
            {"name": "first", "arguments": {}},
            {"name": "second", "arguments": {}},
            {"name": "third", "arguments": {}},
        ]

        result = extract_tool_names(tool_calls)
        assert result == ["first", "second", "third"]
