"""MCP plugin factory.

Creates the appropriate MCPPluginWrapper based on transport type.
This factory pattern allows the caller to create plugin wrappers
without knowing the concrete implementation details.

Usage:
    from holodeck.tools.mcp.factory import create_mcp_plugin
    from holodeck.models.tool import MCPTool, CommandType

    config = MCPTool(
        name="filesystem",
        description="File operations",
        server="@modelcontextprotocol/server-filesystem",
        command=CommandType.NPX,
    )
    plugin = create_mcp_plugin(config)
    async with plugin:
        result = await plugin.call_tool("read_file", {"path": "/tmp/test.txt"})
"""

from holodeck.models.tool import MCPTool, TransportType
from holodeck.tools.mcp.errors import MCPConfigError
from holodeck.tools.mcp.plugin import MCPPluginWrapper


def create_mcp_plugin(config: MCPTool) -> MCPPluginWrapper:
    """Create an MCP plugin wrapper based on transport type.

    This factory function creates the appropriate MCPPluginWrapper subclass
    based on the transport type specified in the configuration. Each transport
    type maps to a specific Semantic Kernel MCP plugin implementation.

    Transport mapping:
    - stdio -> MCPStdioPluginWrapper (wraps MCPStdioPlugin)
    - sse -> MCPSsePluginWrapper (wraps MCPSsePlugin)
    - websocket -> MCPWebsocketPluginWrapper (wraps MCPWebsocketPlugin)
    - http -> MCPStreamableHttpPluginWrapper (wraps MCPStreamableHttpPlugin)

    Args:
        config: MCP tool configuration from agent.yaml

    Returns:
        Appropriate MCPPluginWrapper subclass instance

    Raises:
        MCPConfigError: If transport type is not supported or not yet implemented

    Example:
        >>> config = MCPTool(
        ...     name="filesystem",
        ...     description="File operations",
        ...     server="@modelcontextprotocol/server-filesystem",
        ...     command=CommandType.NPX,
        ... )
        >>> plugin = create_mcp_plugin(config)
        >>> type(plugin).__name__
        'MCPStdioPluginWrapper'
    """
    # Import concrete implementations here to avoid circular imports
    # These will be implemented in Phase 3+ (User Story implementations)

    if config.transport == TransportType.STDIO:
        # TODO: Implement in T011 (Phase 3 - User Story 1)
        # from holodeck.tools.mcp.stdio import MCPStdioPluginWrapper
        # return MCPStdioPluginWrapper(config)
        raise MCPConfigError(
            field="transport",
            message="Stdio transport not yet implemented. Coming in Phase 3.",
        )

    elif config.transport == TransportType.SSE:
        # TODO: Implement in T044 (Phase 8 - User Story 6)
        # from holodeck.tools.mcp.sse import MCPSsePluginWrapper
        # return MCPSsePluginWrapper(config)
        raise MCPConfigError(
            field="transport",
            message="SSE transport not yet implemented. Coming in Phase 8.",
        )

    elif config.transport == TransportType.WEBSOCKET:
        # TODO: Implement in T048 (Phase 9 - User Story 7)
        # from holodeck.tools.mcp.websocket import MCPWebsocketPluginWrapper
        # return MCPWebsocketPluginWrapper(config)
        raise MCPConfigError(
            field="transport",
            message="WebSocket transport not yet implemented. Coming in Phase 9.",
        )

    elif config.transport == TransportType.HTTP:
        # TODO: Implement in T051 (Phase 10 - User Story 8)
        # from holodeck.tools.mcp.http import MCPStreamableHttpPluginWrapper
        # return MCPStreamableHttpPluginWrapper(config)
        raise MCPConfigError(
            field="transport",
            message="HTTP transport not yet implemented. Coming in Phase 10.",
        )

    else:
        raise MCPConfigError(
            field="transport",
            message=f"Unknown transport type: {config.transport}",
        )
