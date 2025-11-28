"""MCP (Model Context Protocol) tool module for HoloDeck.

This module provides MCP server integration capabilities:
- create_mcp_plugin: Factory function for creating SK MCP plugin instances
- normalize_tool_name: Utility for normalizing MCP tool names
- MCP error types: MCPError, MCPConfigError, MCPConnectionError, etc.

Note: Semantic Kernel's MCP plugins handle lifecycle, tool discovery,
and response processing automatically. HoloDeck uses SK plugins directly
with thin configuration translation.
"""

from holodeck.tools.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolNotFoundError,
)
from holodeck.tools.mcp.factory import create_mcp_plugin
from holodeck.tools.mcp.utils import normalize_tool_name

__all__ = [
    # Factory
    "create_mcp_plugin",
    # Utility
    "normalize_tool_name",
    # Errors
    "MCPError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPToolNotFoundError",
]
