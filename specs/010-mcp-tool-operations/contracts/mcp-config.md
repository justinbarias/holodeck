# MCP Configuration Contract

**Feature**: 010-mcp-tool-operations
**Date**: 2025-11-28
**Version**: 1.0.0

## Overview

This document defines the YAML configuration contract for MCP (Model Context Protocol) tools in HoloDeck agent.yaml files.

## Configuration Schema

### Full MCP Tool Schema

```yaml
tools:
  - name: string                    # Required: Unique tool identifier
    description: string             # Required: Human-readable description
    type: mcp                       # Required: Must be "mcp"
    server: string                  # Required: MCP server identifier

    # Transport configuration (mutually exclusive groups)
    transport: stdio | sse | websocket | http  # Optional: Default "stdio"

    # Stdio transport fields
    command: npx | uvx | docker     # Required for stdio: Supported commands only
    args: [string]                  # Optional: Command arguments
    env:                            # Optional: Environment variables
      KEY: "${ENV_VAR}"            # Supports ${VAR} substitution
    envFile: string                 # Optional: Path to .env file
    encoding: string                # Optional: Stream encoding (default: utf-8)

    # HTTP/SSE/WebSocket transport fields
    url: string                     # Required for sse/websocket/http
    headers:                        # Optional for sse/http
      Authorization: "Bearer ${TOKEN}"
    timeout: float                  # Optional: Connection timeout (seconds)
    sse_read_timeout: float         # Optional: SSE read timeout (seconds)
    terminate_on_close: boolean     # Optional for http: Close on finish

    # Common optional fields
    config:                         # Optional: Server-specific configuration
      key: value
    load_tools: boolean             # Optional: Auto-discover tools (default: true)
    load_prompts: boolean           # Optional: Auto-discover prompts (default: true)
    request_timeout: integer        # Optional: Operation timeout (default: 60)
```

## Transport-Specific Contracts

### Stdio Transport Contract

**Use Case**: Local MCP servers executed as child processes

**Supported Commands**: Only the following commands are allowed for security and consistency:

| Command | Use Case | Example |
|---------|----------|---------|
| `npx` | Node.js/npm MCP servers | `@modelcontextprotocol/server-filesystem` |
| `uvx` | Python/uv MCP servers | `mcp-server-python` |
| `docker` | Containerized MCP servers | `mcp-server:latest` |

```yaml
tools:
  - name: filesystem
    description: "File system operations via MCP"
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    transport: stdio                # Optional, stdio is default
    command: npx                    # Required: npx | uvx | docker
    args:                           # Optional
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
    env:                            # Optional
      ALLOWED_PATHS: "/workspace"
    envFile: ".env"                 # Optional
    encoding: "utf-8"               # Optional
    config:                         # Optional, passed to server init
      allowed_directories:
        - "/workspace/data"
    request_timeout: 60             # Optional, default 60
```

**Required Fields**: `name`, `description`, `type`, `server`, `command`

**Validation Rules**:
- `command` MUST be one of: `npx`, `uvx`, `docker` (case-sensitive)
- Invalid commands will fail at config load time with: `MCPConfigError: Invalid command 'X'. Supported commands: npx, uvx, docker`
- `envFile` path is relative to agent.yaml location
- All `${VAR}` references must resolve at config load time

### SSE Transport Contract

**Use Case**: Remote MCP servers with Server-Sent Events streaming

```yaml
tools:
  - name: cloud_mcp
    description: "Cloud-hosted MCP server"
    type: mcp
    server: "cloud-mcp-service"
    transport: sse                  # Required for SSE
    url: "https://mcp.example.com/sse"  # Required
    headers:                        # Optional
      Authorization: "Bearer ${API_TOKEN}"
      X-Custom-Header: "value"
    timeout: 30                     # Optional, connection timeout
    sse_read_timeout: 120           # Optional, read timeout
    request_timeout: 60             # Optional
```

**Required Fields**: `name`, `description`, `type`, `server`, `transport`, `url`

**Validation Rules**:
- `url` must be a valid HTTPS URL (HTTP allowed for localhost only)
- Header values support `${VAR}` substitution

### WebSocket Transport Contract

**Use Case**: Bidirectional real-time MCP communication

```yaml
tools:
  - name: realtime_mcp
    description: "Real-time MCP server"
    type: mcp
    server: "realtime-service"
    transport: websocket            # Required for WebSocket
    url: "wss://mcp.example.com/ws" # Required
    request_timeout: 60             # Optional
```

**Required Fields**: `name`, `description`, `type`, `server`, `transport`, `url`

**Validation Rules**:
- `url` must be a valid `wss://` or `ws://` URL

### HTTP Transport Contract (Streamable)

**Use Case**: HTTP with streaming response support

```yaml
tools:
  - name: http_mcp
    description: "HTTP MCP server with streaming"
    type: mcp
    server: "http-mcp-service"
    transport: http                 # Required for HTTP
    url: "https://mcp.example.com/stream"  # Required
    headers:                        # Optional
      Authorization: "Bearer ${TOKEN}"
    timeout: 30                     # Optional
    sse_read_timeout: 120           # Optional
    terminate_on_close: true        # Optional
    request_timeout: 60             # Optional
```

**Required Fields**: `name`, `description`, `type`, `server`, `transport`, `url`

## Environment Variable Patterns

**Implementation Note (DRY)**: All environment variable resolution reuses existing functions from `holodeck.config.env_loader`:
- `substitute_env_vars()` - Standard `${VAR}` pattern (already fail-fast)
- `load_env_file()` - Load variables from `.env` files

### Standard Pattern: `${VAR_NAME}`

Substitutes value from process environment using `substitute_env_vars()`.

```yaml
env:
  API_KEY: "${MY_API_KEY}"          # Reads MY_API_KEY from environment
  DATABASE_URL: "${DB_CONNECTION}"  # Reads DB_CONNECTION from environment
```

**Behavior** (provided by existing `substitute_env_vars()`):
- If `MY_API_KEY` is not set, configuration load fails with error:
  `ConfigError: Environment variable 'MY_API_KEY' not found`

## Validation Error Messages

| Error Condition | Error Message |
|-----------------|---------------|
| Invalid `command` | `MCPConfigError: Invalid command 'X'. Supported commands: npx, uvx, docker` |
| Missing `command` for stdio | `MCPConfigError: 'command' is required for stdio transport` |
| Missing `url` for sse | `MCPConfigError: 'url' is required for sse transport` |
| Unresolved env var | `ConfigError: Environment variable 'VAR_NAME' not found` |
| Invalid URL scheme | `MCPConfigError: 'url' must use https:// (or http:// for localhost)` |
| Empty server | `ValidationError: 'server' must be a non-empty identifier` |

## Backward Compatibility

### Legacy Format (v0.x)

```yaml
tools:
  - name: old_mcp_tool
    type: mcp
    server: "@modelcontextprotocol/server-memory"
    config:
      key: value
```

**Migration**: Legacy configs without `transport` or `command` will:
1. Default to `stdio` transport
2. Auto-detect command from `server` field if it's an npm package
3. Emit deprecation warning recommending explicit `command` field

## Examples

### Example 1: Filesystem Server

```yaml
tools:
  - name: files
    description: "Read and write files in workspace"
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    config:
      allowed_directories:
        - "${workspaceFolder}/data"
        - "${workspaceFolder}/output"
```

### Example 2: GitHub Integration

```yaml
tools:
  - name: github
    description: "GitHub repository operations"
    type: mcp
    server: "@modelcontextprotocol/server-github"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
    config:
      repositories:
        - "owner/repo"
```

### Example 3: Database Server

```yaml
tools:
  - name: postgres
    description: "PostgreSQL database queries"
    type: mcp
    server: "@modelcontextprotocol/server-postgres"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
```

### Example 4: Python MCP Server (uvx)

```yaml
tools:
  - name: python_mcp
    description: "Python MCP server via uvx"
    type: mcp
    server: "mcp-server-fetch"
    command: uvx
    args: ["mcp-server-fetch"]
    env:
      CUSTOM_CONFIG: "${CUSTOM_CONFIG}"
```

### Example 5: Docker MCP Server

```yaml
tools:
  - name: docker_mcp
    description: "Containerized MCP server"
    type: mcp
    server: "mcp-server-custom"
    command: docker
    args: ["run", "--rm", "-i", "mcp-server:latest"]
    env:
      API_KEY: "${API_KEY}"
```

### Example 6: Cloud-Hosted SSE Server

```yaml
tools:
  - name: cloud_api
    description: "Cloud AI service via MCP"
    type: mcp
    server: "cloud-ai-mcp"
    transport: sse
    url: "https://api.example.com/mcp/sse"
    headers:
      Authorization: "Bearer ${CLOUD_API_KEY}"
      X-Request-ID: "holodeck-agent"
    timeout: 10
    sse_read_timeout: 300
    request_timeout: 120
```
