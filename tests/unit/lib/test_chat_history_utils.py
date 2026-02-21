"""Unit tests for holodeck.lib.chat_history_utils module."""

import pytest

from holodeck.lib.chat_history_utils import extract_tool_names


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
