"""MCP plugin factory.

Creates the appropriate Semantic Kernel MCP plugin based on transport type.
This factory translates HoloDeck MCPToolConfig to SK plugin constructor arguments.

Note: SK plugins handle full lifecycle management via async context managers.
HoloDeck does NOT need custom lifecycle wrappers - we return SK plugins directly.

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
        # SK plugin handles tool discovery and invocation automatically
        tools = await plugin.list_tools()
"""

from typing import Any

from holodeck.models.tool import MCPTool, TransportType
from holodeck.tools.mcp.errors import MCPConfigError


def create_mcp_plugin(config: MCPTool) -> Any:
    """Create an SK MCP plugin based on transport type.

    This factory function creates the appropriate Semantic Kernel MCP plugin
    based on the transport type specified in the configuration. Each transport
    type maps to a specific SK plugin:

    Transport mapping:
    - stdio -> MCPStdioPlugin
    - sse -> MCPSsePlugin
    - websocket -> MCPWebsocketPlugin
    - http -> MCPStreamableHttpPlugin

    Args:
        config: MCP tool configuration from agent.yaml

    Returns:
        Appropriate SK MCP plugin instance (MCPStdioPlugin, MCPSsePlugin, etc.)

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
        >>> # plugin is an MCPStdioPlugin instance
    """
    # TODO: Add environment variable resolution using substitute_env_vars()
    # before passing to SK plugin constructors (T007)

    if config.transport == TransportType.STDIO:
        # TODO: Implement in T008 (Phase 3 - User Story 1)
        # from semantic_kernel.connectors.mcp import MCPStdioPlugin
        # return MCPStdioPlugin(
        #     name=config.name,
        #     command=config.command.value if config.command else "npx",
        #     args=[config.server] + (config.args or []),
        #     env=config.env,
        #     encoding=config.encoding or "utf-8",
        # )
        raise MCPConfigError(
            field="transport",
            message="Stdio transport not yet implemented. Coming in Phase 3.",
        )

    elif config.transport == TransportType.SSE:
        # TODO: Implement in T022 (Phase 7 - User Story 6)
        # from semantic_kernel.connectors.mcp import MCPSsePlugin
        # return MCPSsePlugin(
        #     name=config.name,
        #     url=config.url,
        #     headers=config.headers,
        #     timeout=config.timeout,
        #     sse_read_timeout=config.sse_read_timeout,
        # )
        raise MCPConfigError(
            field="transport",
            message="SSE transport not yet implemented. Coming in Phase 7.",
        )

    elif config.transport == TransportType.WEBSOCKET:
        # TODO: Implement in T025 (Phase 8 - User Story 7)
        # from semantic_kernel.connectors.mcp import MCPWebsocketPlugin
        # return MCPWebsocketPlugin(
        #     name=config.name,
        #     url=config.url,
        # )
        raise MCPConfigError(
            field="transport",
            message="WebSocket transport not yet implemented. Coming in Phase 8.",
        )

    elif config.transport == TransportType.HTTP:
        # TODO: Implement in T027 (Phase 9 - User Story 8)
        # from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
        # return MCPStreamableHttpPlugin(
        #     name=config.name,
        #     url=config.url,
        #     headers=config.headers,
        #     timeout=config.timeout,
        #     sse_read_timeout=config.sse_read_timeout,
        #     terminate_on_close=config.terminate_on_close,
        # )
        raise MCPConfigError(
            field="transport",
            message="HTTP transport not yet implemented. Coming in Phase 9.",
        )

    else:
        raise MCPConfigError(
            field="transport",
            message=f"Unknown transport type: {config.transport}",
        )
