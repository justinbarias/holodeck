# Contract: MCP Registry Client

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## Overview

Internal Python client for the MCP Registry API. Not a public API - this defines the interface between the CLI commands and the registry service.

## Service Interface

```python
class MCPRegistryClient:
    """Client for MCP Registry API."""

    def __init__(
        self,
        base_url: str = "https://registry.modelcontextprotocol.io",
        timeout: float = 5.0,
    ) -> None:
        """Initialize client with base URL and timeout."""

    def search(
        self,
        query: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> SearchResult:
        """Search for MCP servers.

        Args:
            query: Optional search term (substring match on name)
            limit: Maximum results per page (default 25)
            cursor: Pagination cursor for next page

        Returns:
            SearchResult with servers and pagination info

        Raises:
            RegistryConnectionError: Network/timeout issues
            RegistryAPIError: API returned error status
        """

    def get_server(
        self,
        name: str,
        version: str = "latest",
    ) -> RegistryServer:
        """Get specific server by name and version.

        Args:
            name: Server name (reverse-DNS format)
            version: Version string or "latest"

        Returns:
            RegistryServer with full details

        Raises:
            ServerNotFoundError: Server doesn't exist
            RegistryConnectionError: Network/timeout issues
        """

    def list_versions(
        self,
        name: str,
    ) -> list[str]:
        """List available versions for a server.

        Args:
            name: Server name (reverse-DNS format)

        Returns:
            List of version strings (newest first)

        Raises:
            ServerNotFoundError: Server doesn't exist
            RegistryConnectionError: Network/timeout issues
        """
```

## Data Types

```python
@dataclass
class SearchResult:
    """Search result with pagination."""
    servers: list[RegistryServer]
    next_cursor: str | None
    total_count: int

class RegistryConnectionError(HoloDeckError):
    """Network or timeout error connecting to registry."""

class RegistryAPIError(HoloDeckError):
    """Registry API returned an error response."""

class ServerNotFoundError(HoloDeckError):
    """Requested server not found in registry."""
```

## Error Handling Contract

| HTTP Status | Exception | User Message |
|-------------|-----------|--------------|
| Timeout | RegistryConnectionError | "Registry unavailable (timeout). Check connection." |
| Connection error | RegistryConnectionError | "Cannot connect to registry. Check connection." |
| 404 | ServerNotFoundError | "Server '{name}' not found in registry." |
| 429 | RegistryAPIError | "Rate limited. Please wait and try again." |
| 5xx | RegistryAPIError | "Registry service error. Try again later." |

## Usage Example

```python
from holodeck.services.mcp_registry import MCPRegistryClient

client = MCPRegistryClient()

# Search for servers
result = client.search(query="filesystem")
for server in result.servers:
    print(f"{server.name}: {server.description}")

# Get specific server
server = client.get_server("io.github.modelcontextprotocol/server-filesystem")
print(server.packages[0].identifier)
```
