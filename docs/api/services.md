# Services

The `holodeck.services` package contains client classes for interacting with
external services such as the MCP Registry API.

## MCP Registry Client

`MCPRegistryClient` provides methods to search, retrieve, and list MCP servers
from the official registry at <https://registry.modelcontextprotocol.io>.

::: holodeck.services.mcp_registry.MCPRegistryClient
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - search
        - get_server
        - list_versions
        - close

## Module-Level Functions

### find_stdio_package

::: holodeck.services.mcp_registry.find_stdio_package
    options:
      docstring_style: google
      show_source: true

### registry_to_mcp_tool

::: holodeck.services.mcp_registry.registry_to_mcp_tool
    options:
      docstring_style: google
      show_source: true

## Constants

### SUPPORTED_REGISTRY_TYPES

::: holodeck.services.mcp_registry.SUPPORTED_REGISTRY_TYPES
    options:
      docstring_style: google
      show_source: true
