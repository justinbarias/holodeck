"""Tests for MCPPluginWrapper base class."""

from holodeck.tools.mcp.plugin import MCPPluginWrapper


class TestMCPPluginWrapperNormalization:
    """Test MCPPluginWrapper tool name normalization."""

    def test_normalize_tool_name_replaces_dots(self) -> None:
        """Dots should be replaced with hyphens."""
        assert MCPPluginWrapper.normalize_tool_name("read.file") == "read-file"

    def test_normalize_tool_name_replaces_slashes(self) -> None:
        """Slashes should be replaced with hyphens."""
        assert MCPPluginWrapper.normalize_tool_name("read/file") == "read-file"

    def test_normalize_tool_name_replaces_spaces(self) -> None:
        """Spaces should be replaced with hyphens."""
        assert MCPPluginWrapper.normalize_tool_name("read file") == "read-file"

    def test_normalize_tool_name_preserves_hyphens(self) -> None:
        """Existing hyphens should be preserved."""
        assert MCPPluginWrapper.normalize_tool_name("read-file") == "read-file"

    def test_normalize_tool_name_preserves_underscores(self) -> None:
        """Underscores should be preserved."""
        assert MCPPluginWrapper.normalize_tool_name("read_file") == "read_file"

    def test_normalize_tool_name_preserves_alphanumeric(self) -> None:
        """Alphanumeric characters should be preserved."""
        assert MCPPluginWrapper.normalize_tool_name("readFile2") == "readFile2"
        assert MCPPluginWrapper.normalize_tool_name("read_file_v2") == "read_file_v2"

    def test_normalize_tool_name_handles_mixed_characters(self) -> None:
        """Mixed special characters should all be replaced."""
        assert MCPPluginWrapper.normalize_tool_name("read.file/v2 final") == (
            "read-file-v2-final"
        )

    def test_normalize_tool_name_preserves_case(self) -> None:
        """Case should be preserved."""
        assert MCPPluginWrapper.normalize_tool_name("ReadFile") == "ReadFile"
        assert MCPPluginWrapper.normalize_tool_name("readFILE") == "readFILE"

    def test_normalize_tool_name_handles_special_chars(self) -> None:
        """Various special characters should be replaced."""
        assert MCPPluginWrapper.normalize_tool_name("tool@v1") == "tool-v1"
        assert MCPPluginWrapper.normalize_tool_name("tool#1") == "tool-1"
        assert MCPPluginWrapper.normalize_tool_name("tool$test") == "tool-test"
        assert MCPPluginWrapper.normalize_tool_name("tool%test") == "tool-test"

    def test_normalize_tool_name_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert MCPPluginWrapper.normalize_tool_name("") == ""
