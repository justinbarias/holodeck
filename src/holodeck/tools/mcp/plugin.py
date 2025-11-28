"""MCP plugin wrapper for HoloDeck.

Provides a unified interface for interacting with MCP servers via
Semantic Kernel's MCP plugin implementations. The MCPPluginWrapper
abstract base class defines the contract that all transport-specific
implementations must follow.

Concrete implementations (to be added in later phases):
- MCPStdioPluginWrapper: Wraps MCPStdioPlugin for local subprocess
- MCPSsePluginWrapper: Wraps MCPSsePlugin for SSE transport
- MCPWebsocketPluginWrapper: Wraps MCPWebsocketPlugin for WebSocket
- MCPStreamableHttpPluginWrapper: Wraps MCPStreamableHttpPlugin for HTTP
"""

import re
from abc import ABC, abstractmethod
from typing import Any

from holodeck.models.tool import MCPTool
from holodeck.tools.mcp.content import MCPToolResult


class MCPPluginWrapper(ABC):
    """Abstract base wrapper for Semantic Kernel MCP plugins.

    This class provides a unified interface for all MCP transport types.
    Concrete implementations wrap specific Semantic Kernel plugin classes
    and handle the details of each transport protocol.

    Usage:
        async with plugin_wrapper:
            result = await plugin_wrapper.call_tool("tool_name", {"arg": "value"})
            print(result.content)

    Attributes:
        config: The MCPTool configuration from agent.yaml
        name: The tool name (from config)
        is_connected: Whether the plugin is currently connected

    Example:
        >>> config = MCPTool(
        ...     name="filesystem",
        ...     description="File operations",
        ...     server="@modelcontextprotocol/server-filesystem",
        ...     command=CommandType.NPX,
        ... )
        >>> plugin = create_mcp_plugin(config)
        >>> async with plugin:
        ...     result = await plugin.call_tool("read_file", {"path": "/tmp/test.txt"})
    """

    def __init__(self, config: MCPTool) -> None:
        """Initialize the plugin wrapper.

        Args:
            config: MCP tool configuration from agent.yaml
        """
        self.config = config
        self.name = config.name
        self._connected = False
        self._discovered_tools: list[dict[str, Any]] = []
        self._discovered_prompts: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Check if the plugin is connected to the MCP server.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server.

        This method should:
        1. Create the underlying Semantic Kernel plugin
        2. Enter the plugin's async context
        3. Discover available tools/prompts if load_tools/load_prompts is True
        4. Set _connected = True on success

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If connection times out
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server.

        This method should:
        1. Exit the plugin's async context
        2. Clean up any resources
        3. Set _connected = False
        """
        pass

    @abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> MCPToolResult:
        """Invoke a tool on the MCP server.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments (optional)

        Returns:
            MCPToolResult with success/error status and content blocks

        Raises:
            MCPToolNotFoundError: If tool doesn't exist on server
            MCPProtocolError: If server returns a protocol error
            MCPTimeoutError: If request times out
            MCPConnectionError: If not connected
        """
        pass

    @abstractmethod
    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Get a prompt from the MCP server.

        Args:
            prompt_name: Name of the prompt to retrieve
            arguments: Prompt arguments for template substitution (optional)

        Returns:
            Rendered prompt string

        Raises:
            MCPProtocolError: If prompt doesn't exist or server error
            MCPConnectionError: If not connected
        """
        pass

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server.

        Returns:
            List of tool definitions, each containing:
            - name: Tool name
            - description: Tool description
            - inputSchema: JSON Schema for tool parameters

        Raises:
            MCPConnectionError: If not connected
        """
        pass

    @abstractmethod
    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts from the MCP server.

        Returns:
            List of prompt definitions, each containing:
            - name: Prompt name
            - description: Prompt description
            - arguments: List of argument definitions

        Raises:
            MCPConnectionError: If not connected
        """
        pass

    async def __aenter__(self) -> "MCPPluginWrapper":
        """Async context manager entry - connect to server.

        Returns:
            Self after connection is established
        """
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - disconnect from server.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception instance if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.disconnect()

    @staticmethod
    def normalize_tool_name(name: str) -> str:
        """Normalize tool name by replacing invalid characters with '-'.

        Per Semantic Kernel pattern, tool names must be valid identifiers.
        This method replaces any character that is not alphanumeric or
        underscore with a hyphen.

        Args:
            name: Original tool name from MCP server

        Returns:
            Normalized name safe for use as identifier

        Example:
            >>> MCPPluginWrapper.normalize_tool_name("read.file")
            'read-file'
            >>> MCPPluginWrapper.normalize_tool_name("read/write")
            'read-write'
            >>> MCPPluginWrapper.normalize_tool_name("read_file_v2")
            'read_file_v2'
        """
        return re.sub(r"[^a-zA-Z0-9_]", "-", name)
