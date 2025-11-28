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
        command=CommandType.NPX,
        args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
    )
    plugin = create_mcp_plugin(config)
    async with plugin:
        # SK plugin handles tool discovery and invocation automatically
        tools = await plugin.list_tools()
"""

import json
from typing import TYPE_CHECKING

from holodeck.config.env_loader import load_env_file, substitute_env_vars
from holodeck.models.tool import MCPTool, TransportType
from holodeck.tools.mcp.errors import MCPConfigError

# Default encoding for MCP stdio transport
DEFAULT_STDIO_ENCODING: str = "utf-8"

if TYPE_CHECKING:
    from semantic_kernel.connectors.mcp import MCPStdioPlugin


def _resolve_env_vars(config: MCPTool) -> dict[str, str]:
    """Resolve environment variables for MCP plugin.

    Loads env_file if specified, then resolves ${VAR} patterns
    in env dict values. Fail-fast on missing variables.

    Precedence (highest to lowest):
    1. Explicit env vars from config.env
    2. Variables loaded from config.env_file
    3. Process environment (for ${VAR} substitution)

    Args:
        config: MCP tool configuration

    Returns:
        Dictionary of resolved environment variables

    Raises:
        ConfigError: If a referenced environment variable is not found
    """
    resolved_env: dict[str, str] = {}

    # Load env_file first (lower precedence)
    if config.env_file:
        resolved_env.update(load_env_file(config.env_file))

    # Apply explicit env vars (higher precedence)
    if config.env:
        for key, value in config.env.items():
            resolved_env[key] = substitute_env_vars(value)

    # Pass config as JSON via MCP_CONFIG env var if provided
    if config.config:
        resolved_env["MCP_CONFIG"] = json.dumps(config.config)

    return resolved_env


def create_mcp_plugin(config: MCPTool) -> "MCPStdioPlugin":
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
        MCPStdioPlugin instance. Other transport types (SSE, WebSocket, HTTP)
        will return their respective plugin types when implemented.

    Raises:
        MCPConfigError: If transport type is not supported or not yet implemented

    Example:
        >>> config = MCPTool(
        ...     name="filesystem",
        ...     description="File operations",
        ...     command=CommandType.NPX,
        ...     args=["-y", "@modelcontextprotocol/server-filesystem"],
        ... )
        >>> plugin = create_mcp_plugin(config)
        >>> # plugin is an MCPStdioPlugin instance
    """
    # Resolve environment variables (env_file + explicit env + config passthrough)
    resolved_env = _resolve_env_vars(config)

    if config.transport == TransportType.STDIO:
        # Import SK plugin lazily to avoid hard dependency
        try:
            from semantic_kernel.connectors.mcp import MCPStdioPlugin
        except ImportError as e:
            raise MCPConfigError(
                field="transport",
                message=(
                    "Semantic Kernel MCP support not installed. "
                    "Install with: pip install semantic-kernel[mcp]"
                ),
            ) from e

        return MCPStdioPlugin(
            name=config.name,
            command=config.command.value if config.command else "npx",
            args=(config.args or []),
            env=resolved_env if resolved_env else None,
            encoding=config.encoding or DEFAULT_STDIO_ENCODING,
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
