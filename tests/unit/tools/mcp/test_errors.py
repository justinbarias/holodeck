"""Tests for MCP error classes."""

from holodeck.lib.errors import ConfigError, HoloDeckError
from holodeck.tools.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolNotFoundError,
)


class TestMCPErrorHierarchy:
    """Test MCP error class inheritance and attributes."""

    def test_mcp_config_error_extends_config_error(self) -> None:
        """MCPConfigError should extend ConfigError."""
        error = MCPConfigError(field="transport", message="Invalid transport")
        assert isinstance(error, ConfigError)
        assert error.field == "transport"
        assert "Invalid transport" in str(error)

    def test_mcp_error_extends_holodeck_error(self) -> None:
        """MCPError should extend HoloDeckError."""
        error = MCPError(message="Test error", server="test-server")
        assert isinstance(error, HoloDeckError)
        assert error.server == "test-server"
        assert "Test error" in str(error)

    def test_mcp_error_server_optional(self) -> None:
        """MCPError should work without server."""
        error = MCPError(message="Test error")
        assert error.server is None

    def test_mcp_connection_error_extends_mcp_error(self) -> None:
        """MCPConnectionError should extend MCPError."""
        error = MCPConnectionError(
            message="Failed to connect",
            server="my-server",
            command="npx",
        )
        assert isinstance(error, MCPError)
        assert error.server == "my-server"
        assert error.command == "npx"

    def test_mcp_connection_error_command_optional(self) -> None:
        """MCPConnectionError should work without command."""
        error = MCPConnectionError(message="Failed to connect")
        assert error.command is None
        assert error.server is None

    def test_mcp_timeout_error_extends_connection_error(self) -> None:
        """MCPTimeoutError should extend MCPConnectionError."""
        error = MCPTimeoutError(
            message="Request timed out",
            server="my-server",
            timeout=60.0,
        )
        assert isinstance(error, MCPConnectionError)
        assert isinstance(error, MCPError)
        assert error.timeout == 60.0

    def test_mcp_timeout_error_timeout_optional(self) -> None:
        """MCPTimeoutError should work without timeout value."""
        error = MCPTimeoutError(message="Request timed out")
        assert error.timeout is None

    def test_mcp_protocol_error_extends_mcp_error(self) -> None:
        """MCPProtocolError should extend MCPError."""
        error = MCPProtocolError(
            message="Protocol error",
            server="my-server",
            error_code=-32600,
        )
        assert isinstance(error, MCPError)
        assert error.error_code == -32600

    def test_mcp_protocol_error_code_optional(self) -> None:
        """MCPProtocolError should work without error_code."""
        error = MCPProtocolError(message="Protocol error")
        assert error.error_code is None

    def test_mcp_tool_not_found_error_extends_mcp_error(self) -> None:
        """MCPToolNotFoundError should extend MCPError."""
        error = MCPToolNotFoundError(tool_name="read_file", server="fs-server")
        assert isinstance(error, MCPError)
        assert error.tool_name == "read_file"
        assert error.server == "fs-server"

    def test_mcp_tool_not_found_error_message_includes_tool_name(self) -> None:
        """MCPToolNotFoundError message should include tool name."""
        error = MCPToolNotFoundError(tool_name="read_file")
        assert "read_file" in str(error)
        assert "not found" in str(error).lower()


class TestMCPConfigError:
    """Test MCPConfigError specific behavior."""

    def test_mcp_config_error_field_and_message(self) -> None:
        """MCPConfigError should store field and message."""
        error = MCPConfigError(
            field="command",
            message="Invalid command 'bash'. Supported commands: npx, uvx, docker",
        )
        assert error.field == "command"
        assert "Invalid command" in str(error)
        assert "npx" in str(error)

    def test_mcp_config_error_for_missing_url(self) -> None:
        """MCPConfigError for missing URL field."""
        error = MCPConfigError(
            field="url",
            message="'url' is required for sse transport",
        )
        assert error.field == "url"
        assert "required" in str(error).lower()
