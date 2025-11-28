"""MCP (Model Context Protocol) tool module for HoloDeck.

This module provides MCP server integration capabilities:
- MCPPluginWrapper: Base wrapper for Semantic Kernel MCP plugins
- MCPToolResult: Standardized result from MCP tool invocations
- ContentBlock types: TextContent, ImageContent, AudioContent, BinaryContent
- create_mcp_plugin: Factory function for creating appropriate plugin wrappers
- MCP error types: MCPError, MCPConfigError, MCPConnectionError, etc.
"""

from holodeck.tools.mcp.content import (
    AudioContent,
    BinaryContent,
    ContentBlock,
    ImageContent,
    MCPToolResult,
    TextContent,
)
from holodeck.tools.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPProtocolError,
    MCPTimeoutError,
    MCPToolNotFoundError,
)
from holodeck.tools.mcp.factory import create_mcp_plugin
from holodeck.tools.mcp.plugin import MCPPluginWrapper

__all__ = [
    # Content types
    "TextContent",
    "ImageContent",
    "AudioContent",
    "BinaryContent",
    "ContentBlock",
    "MCPToolResult",
    # Plugin wrapper
    "MCPPluginWrapper",
    # Factory
    "create_mcp_plugin",
    # Errors
    "MCPError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPToolNotFoundError",
]
