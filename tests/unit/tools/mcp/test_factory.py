"""Tests for MCP plugin factory."""

import pytest

from holodeck.models.tool import CommandType, MCPTool, TransportType
from holodeck.tools.mcp.errors import MCPConfigError
from holodeck.tools.mcp.factory import create_mcp_plugin


class TestCreateMCPPlugin:
    """Test factory function for creating MCP plugins."""

    def test_stdio_transport_not_yet_implemented(self) -> None:
        """Stdio transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            server="test-server",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"

    def test_sse_transport_not_yet_implemented(self) -> None:
        """SSE transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            server="test-server",
            transport=TransportType.SSE,
            url="https://example.com/sse",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"

    def test_websocket_transport_not_yet_implemented(self) -> None:
        """WebSocket transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            server="test-server",
            transport=TransportType.WEBSOCKET,
            url="wss://example.com/ws",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"

    def test_http_transport_not_yet_implemented(self) -> None:
        """HTTP transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            server="test-server",
            transport=TransportType.HTTP,
            url="https://example.com/stream",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"
