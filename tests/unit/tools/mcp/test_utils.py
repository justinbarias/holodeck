"""Tests for MCP utility functions."""

from holodeck.tools.mcp.utils import normalize_tool_name


class TestNormalizeToolName:
    """Test normalize_tool_name utility function."""

    def test_normalize_tool_name_replaces_dots(self) -> None:
        """Dots should be replaced with hyphens."""
        assert normalize_tool_name("read.file") == "read-file"

    def test_normalize_tool_name_replaces_slashes(self) -> None:
        """Slashes should be replaced with hyphens."""
        assert normalize_tool_name("read/file") == "read-file"

    def test_normalize_tool_name_replaces_spaces(self) -> None:
        """Spaces should be replaced with hyphens."""
        assert normalize_tool_name("read file") == "read-file"

    def test_normalize_tool_name_preserves_hyphens(self) -> None:
        """Existing hyphens should be preserved."""
        assert normalize_tool_name("read-file") == "read-file"

    def test_normalize_tool_name_preserves_underscores(self) -> None:
        """Underscores should be preserved."""
        assert normalize_tool_name("read_file") == "read_file"

    def test_normalize_tool_name_preserves_alphanumeric(self) -> None:
        """Alphanumeric characters should be preserved."""
        assert normalize_tool_name("readFile2") == "readFile2"
        assert normalize_tool_name("read_file_v2") == "read_file_v2"

    def test_normalize_tool_name_handles_mixed_characters(self) -> None:
        """Mixed special characters should all be replaced."""
        assert normalize_tool_name("read.file/v2 final") == ("read-file-v2-final")

    def test_normalize_tool_name_preserves_case(self) -> None:
        """Case should be preserved."""
        assert normalize_tool_name("ReadFile") == "ReadFile"
        assert normalize_tool_name("readFILE") == "readFILE"

    def test_normalize_tool_name_handles_special_chars(self) -> None:
        """Various special characters should be replaced."""
        assert normalize_tool_name("tool@v1") == "tool-v1"
        assert normalize_tool_name("tool#1") == "tool-1"
        assert normalize_tool_name("tool$test") == "tool-test"
        assert normalize_tool_name("tool%test") == "tool-test"

    def test_normalize_tool_name_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_tool_name("") == ""
