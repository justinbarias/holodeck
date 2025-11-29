# API Contract: MCP Registry Client

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Overview

Internal client for fetching MCP server information from the official Model Context Protocol registry.

## Module: `holodeck.lib.mcp_registry`

### Class: `MCPRegistryClient`

```python
class MCPRegistryClient:
    """Client for official MCP server registry API.

    Fetches available MCP servers for wizard selection.
    Handles pagination, caching, and error scenarios.
    """
```

### Constants

```python
REGISTRY_BASE_URL = "https://registry.modelcontextprotocol.io"
REGISTRY_API_VERSION = "v0"
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_PAGE_LIMIT = 100
```

### Methods

#### `__init__`

```python
def __init__(
    self,
    base_url: str = REGISTRY_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """Initialize registry client.

    Args:
        base_url: Registry base URL (for testing)
        timeout: Request timeout in seconds
    """
```

#### `list_servers`

```python
def list_servers(
    self,
    search: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> list[MCPServerInfo]:
    """Fetch list of MCP servers from registry.

    Args:
        search: Optional search term to filter servers
        limit: Maximum servers to return (max 100)

    Returns:
        List of MCPServerInfo objects

    Raises:
        MCPRegistryError: If registry is unreachable or returns error
        MCPRegistryTimeoutError: If request times out
    """
```

#### `get_server_choices`

```python
def get_server_choices(self) -> list[MCPServerChoice]:
    """Get servers formatted for wizard selection.

    Fetches servers, marks defaults, and returns as choice objects.

    Returns:
        List of MCPServerChoice with display info

    Raises:
        MCPRegistryError: If registry is unreachable
    """
```

### Data Classes

#### `MCPServerInfo`

```python
class MCPServerInfo(BaseModel):
    """MCP server information from registry."""
    name: str  # Fully qualified name
    description: str
    version: str
    packages: list[MCPPackage]
    is_official: bool
    is_default: bool  # One of the 3 default servers

    @property
    def short_name(self) -> str: ...
    @property
    def primary_package(self) -> MCPPackage | None: ...
```

#### `MCPPackage`

```python
class MCPPackage(BaseModel):
    """Package installation info."""
    registry_type: str  # "npm" | "pypi"
    identifier: str  # Package name
    transport_type: str  # "stdio" | "sse" | "http"
```

#### `MCPServerChoice`

```python
class MCPServerChoice(BaseModel):
    """Formatted choice for wizard display."""
    value: str  # Package identifier for config
    display_name: str  # Human-readable name
    description: str  # Brief description
    enabled: bool  # Pre-selected in wizard
```

### Exceptions

```python
class MCPRegistryError(Exception):
    """Base exception for registry errors."""
    pass

class MCPRegistryNetworkError(MCPRegistryError):
    """Network connectivity error."""
    pass

class MCPRegistryTimeoutError(MCPRegistryError):
    """Request timeout error."""
    pass

class MCPRegistryResponseError(MCPRegistryError):
    """Invalid response from registry."""
    pass
```

## External API Contract

### Endpoint: `GET /v0/servers`

**Base URL**: `https://registry.modelcontextprotocol.io`

**Request**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Max results (default: 100, max: 100) |
| `cursor` | string | No | Pagination cursor |
| `search` | string | No | Search filter |

**Response (200 OK)**:

```json
{
  "servers": [
    {
      "server": {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-09-29/server.schema.json",
        "name": "io.github.modelcontextprotocol/server-filesystem",
        "description": "MCP server providing filesystem access",
        "version": "1.0.0",
        "packages": [
          {
            "registryType": "npm",
            "identifier": "@modelcontextprotocol/server-filesystem",
            "transport": {
              "type": "stdio"
            }
          }
        ]
      },
      "_meta": {
        "io.modelcontextprotocol.registry/official": {
          "status": "active",
          "publishedAt": "2025-09-16T16:43:44.243Z",
          "updatedAt": "2025-09-16T16:43:44.243Z",
          "isLatest": true
        }
      }
    }
  ],
  "metadata": {
    "count": 1,
    "nextCursor": null
  }
}
```

**Error Responses**:

| Status | Meaning |
|--------|---------|
| 400 | Invalid request parameters |
| 500 | Server error |
| Timeout | Network unreachable |

## Default MCP Servers

These servers are pre-selected in the wizard:

| Package Identifier | Short Name | Description |
|-------------------|------------|-------------|
| `@modelcontextprotocol/server-filesystem` | filesystem | File system access |
| `@modelcontextprotocol/server-memory` | memory | Key-value memory storage |
| `@modelcontextprotocol/server-sequential-thinking` | sequential-thinking | Structured reasoning |

## Error Handling

1. **Network Error**: Catch `requests.RequestException`, wrap in `MCPRegistryNetworkError`
2. **Timeout**: Catch `requests.Timeout`, wrap in `MCPRegistryTimeoutError`
3. **Invalid JSON**: Catch `json.JSONDecodeError`, wrap in `MCPRegistryResponseError`
4. **Unexpected Status**: Check status code, wrap in `MCPRegistryResponseError`

Per spec requirement FR-014:
> System MUST display a clear error message and exit if the MCP registry API is unreachable

Error message format:
```
Error: Cannot fetch MCP servers from registry.
Network error: {details}
Check your internet connection and try again.
```

## Caching Strategy

For wizard responsiveness:
- Cache registry response for session duration
- No persistent cache (always fresh on new init)
- Configurable timeout for slow networks

## Testing Considerations

1. **Mock responses**: Provide fixture files for unit tests
2. **Network simulation**: Test timeout and error scenarios
3. **Pagination**: Test with multi-page responses (if applicable)
4. **Default marking**: Verify default servers are correctly flagged
