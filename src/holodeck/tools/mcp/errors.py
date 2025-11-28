"""MCP-specific error types.

Error Hierarchy:
    HoloDeckError (existing)
    ├── ConfigError (existing)
    │   └── MCPConfigError (new) - Invalid MCP configuration
    └── MCPError (new) - Base MCP runtime error
        ├── MCPConnectionError - Failed to connect to server
        │   └── MCPTimeoutError - Connection/request timeout
        ├── MCPProtocolError - Protocol-level error from server
        └── MCPToolNotFoundError - Tool not found on server
"""

from holodeck.lib.errors import ConfigError, HoloDeckError


class MCPConfigError(ConfigError):
    """MCP configuration error (invalid transport, missing fields, etc.)."""

    def __init__(self, field: str, message: str) -> None:
        """Initialize MCP configuration error.

        Args:
            field: The configuration field that caused the error
            message: Descriptive error message
        """
        super().__init__(field, message)


class MCPError(HoloDeckError):
    """Base exception for MCP runtime errors."""

    def __init__(self, message: str, server: str | None = None) -> None:
        """Initialize MCP error.

        Args:
            message: Descriptive error message
            server: MCP server identifier (optional)
        """
        self.server = server
        super().__init__(message)


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""

    def __init__(
        self, message: str, server: str | None = None, command: str | None = None
    ) -> None:
        """Initialize MCP connection error.

        Args:
            message: Descriptive error message
            server: MCP server identifier (optional)
            command: Command that was attempted (optional)
        """
        self.command = command
        super().__init__(message, server)


class MCPTimeoutError(MCPConnectionError):
    """MCP server connection or request timeout."""

    def __init__(
        self,
        message: str,
        server: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize MCP timeout error.

        Args:
            message: Descriptive error message
            server: MCP server identifier (optional)
            timeout: Timeout value that was exceeded (optional)
        """
        self.timeout = timeout
        super().__init__(message, server)


class MCPProtocolError(MCPError):
    """MCP protocol-level error returned by server."""

    def __init__(
        self,
        message: str,
        server: str | None = None,
        error_code: int | None = None,
    ) -> None:
        """Initialize MCP protocol error.

        Args:
            message: Descriptive error message
            server: MCP server identifier (optional)
            error_code: MCP protocol error code (optional)
        """
        self.error_code = error_code
        super().__init__(message, server)


class MCPToolNotFoundError(MCPError):
    """Tool not found on MCP server."""

    def __init__(self, tool_name: str, server: str | None = None) -> None:
        """Initialize MCP tool not found error.

        Args:
            tool_name: Name of the tool that was not found
            server: MCP server identifier (optional)
        """
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found on server", server)
