# Data Model: MCP Tool Operations

**Feature**: 010-mcp-tool-operations
**Date**: 2025-11-28

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCPToolConfig                            │
│  (Enhanced MCPTool model in models/tool.py)                     │
├─────────────────────────────────────────────────────────────────┤
│  name: str                    # Tool identifier                  │
│  description: str             # Human-readable description       │
│  type: Literal["mcp"]         # Tool type discriminator          │
│  server: str                  # MCP server package/path          │
│  transport: TransportType     # stdio|sse|websocket|http         │
│  command: str | None          # Stdio: command to run            │
│  args: list[str] | None       # Stdio: command arguments         │
│  env: dict[str,str] | None    # Environment variables            │
│  envFile: str | None          # Path to .env file                │
│  encoding: str | None         # Stdio: stream encoding           │
│  url: str | None              # HTTP/SSE/WS: server URL          │
│  headers: dict[str,str] | None# HTTP/SSE: request headers        │
│  timeout: float | None        # Connection timeout               │
│  sse_read_timeout: float|None # SSE read timeout                 │
│  terminate_on_close: bool|None# HTTP: terminate on close         │
│  config: dict[str,Any] | None # Server-specific config           │
│  load_tools: bool = True      # Auto-discover tools              │
│  load_prompts: bool = True    # Auto-discover prompts            │
│  request_timeout: int = 60    # Operation timeout (seconds)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ creates
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MCPPluginWrapper                         │
│  (HoloDeck wrapper for SK MCP plugins)                          │
├─────────────────────────────────────────────────────────────────┤
│  name: str                    # Plugin name                      │
│  config: MCPToolConfig        # Original configuration           │
│  _plugin: MCPPluginBase       # Underlying SK plugin             │
│  _connected: bool             # Connection state                 │
│  _discovered_tools: list      # Discovered tool definitions      │
│  _discovered_prompts: list    # Discovered prompt definitions    │
├─────────────────────────────────────────────────────────────────┤
│  + connect() -> None          # Establish connection             │
│  + disconnect() -> None       # Close connection                 │
│  + call_tool(name, **kw)      # Invoke MCP tool                  │
│  + get_prompt(name, **kw)     # Get MCP prompt                   │
│  + list_tools() -> list       # Get available tools              │
│  + list_prompts() -> list     # Get available prompts            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ wraps one of
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Semantic Kernel MCP Plugin Classes                 │
├─────────────────────────────────────────────────────────────────┤
│  MCPStdioPlugin               # stdio transport                  │
│  MCPSsePlugin                 # SSE transport                    │
│  MCPWebsocketPlugin           # WebSocket transport              │
│  MCPStreamableHttpPlugin      # Streamable HTTP transport        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ returns
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCPToolResult                               │
│  (Converted MCP response for HoloDeck)                          │
├─────────────────────────────────────────────────────────────────┤
│  success: bool                # Operation succeeded              │
│  content: list[ContentBlock]  # Response content blocks          │
│  error: MCPError | None       # Error details if failed          │
│  metadata: dict[str, Any]     # Additional response metadata     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ contains
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ContentBlock                               │
│  (Union type for MCP content)                                   │
├─────────────────────────────────────────────────────────────────┤
│  TextContent:                                                    │
│    type: "text"                                                  │
│    text: str                                                     │
│                                                                  │
│  ImageContent:                                                   │
│    type: "image"                                                 │
│    data: str (base64)                                            │
│    mime_type: str                                                │
│                                                                  │
│  AudioContent:                                                   │
│    type: "audio"                                                 │
│    data: str (base64)                                            │
│    mime_type: str                                                │
│                                                                  │
│  BinaryContent:                                                  │
│    type: "binary"                                                │
│    data: bytes                                                   │
│    mime_type: str | None                                         │
│    uri: str | None                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Entity Details

### MCPToolConfig (Enhanced MCPTool)

**Purpose**: YAML configuration model for MCP tools in agent.yaml

**Validation Rules**:
- `name`: Required, non-empty string, unique within agent
- `server`: Required, non-empty string (npm package or local path)
- `transport`: Default "stdio", must be one of: stdio, sse, websocket, http
- `command`: Required for stdio transport, MUST be one of: `npx`, `uvx`, `docker`
- `url`: Required for sse, websocket, http transports
- `env` values: Must resolve via `${VAR}` pattern at config load time
- `request_timeout`: Default 60, must be positive integer

**Supported Commands** (security constraint):
| Command | Runtime | Use Case |
|---------|---------|----------|
| `npx` | Node.js | npm-based MCP servers (e.g., `@modelcontextprotocol/*`) |
| `uvx` | Python/uv | Python MCP servers via uv package manager |
| `docker` | Docker | Containerized MCP servers |

Arbitrary commands are NOT allowed to prevent command injection attacks.

**Transport-Specific Validation**:
| Transport | Required Fields | Optional Fields |
|-----------|-----------------|-----------------|
| stdio | command | args, env, envFile, encoding |
| sse | url | headers, timeout, sse_read_timeout |
| websocket | url | - |
| http | url | headers, timeout, sse_read_timeout, terminate_on_close |

**Environment Variable Resolution** (DRY - reuse existing):
- `${VAR}` pattern: Use `holodeck.config.env_loader.substitute_env_vars()`
- `env_file` loading: Use `holodeck.config.env_loader.load_env_file()`

### TransportType

**Purpose**: Enum for supported MCP transport types

```python
class TransportType(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"
    HTTP = "http"
```

### CommandType

**Purpose**: Enum for allowed stdio commands (security constraint)

```python
class CommandType(str, Enum):
    NPX = "npx"      # Node.js/npm package runner
    UVX = "uvx"      # Python/uv package runner
    DOCKER = "docker"  # Docker container runner
```

**Validation**: The `command` field in MCPToolConfig MUST be validated against this enum.
Invalid commands raise `MCPConfigError` at config load time.

### MCPPluginWrapper

**Purpose**: HoloDeck adapter wrapping Semantic Kernel MCP plugins

**State Transitions**:
```
                    ┌──────────────┐
                    │  CREATED     │
                    │  (initial)   │
                    └──────┬───────┘
                           │ connect()
                           ▼
┌──────────────┐    ┌──────────────┐
│  ERROR       │◄───│  CONNECTING  │
│  (terminal)  │    │              │
└──────────────┘    └──────┬───────┘
       ▲                   │ success
       │                   ▼
       │            ┌──────────────┐
       │            │  CONNECTED   │◄──┐
       │            │  (active)    │   │ tool notification
       │            └──────┬───────┘───┘ (refresh tools)
       │                   │ disconnect()
       │                   ▼
       │            ┌──────────────┐
       └────────────│  DISCONNECTED│
         error      │  (terminal)  │
                    └──────────────┘
```

### MCPToolResult

**Purpose**: Standardized result from MCP tool invocation

**Fields**:
- `success`: Boolean indicating operation success
- `content`: List of ContentBlock items (may be empty)
- `error`: MCPError instance if failed, None if success
- `metadata`: Dict for tool-specific metadata (timing, request_id, etc.)

### ContentBlock

**Purpose**: Union type representing MCP content types

**Type Discrimination**: Via `type` field (Literal discriminator)

## Error Types

**Note**: Extends existing error hierarchy from `holodeck.lib.errors` (DRY compliance)

```
HoloDeckError (existing)
├── ConfigError (existing) ─────────────────────────────────────────┐
│   └── MCPConfigError          # Invalid MCP configuration (NEW)   │
│       ├── MissingTransportFieldError   # Missing required field   │
│       └── UnresolvedEnvVarError        # Env var not found        │
│           (Note: uses existing ConfigError from substitute_env_vars)
│
└── MCPError (NEW)              # Base MCP runtime error
    ├── MCPConnectionError      # Failed to connect to server
    │   └── MCPTimeoutError     # Connection/request timeout
    ├── MCPProtocolError        # Protocol-level error from server
    └── MCPToolNotFoundError    # Tool not found on server
```

**Reuse Note**: `UnresolvedEnvVarError` is NOT needed - `substitute_env_vars()`
already raises `ConfigError` with the variable name. MCP code should let this
propagate without wrapping.

## Configuration Examples

### Stdio Transport (Default)

```yaml
tools:
  - name: filesystem
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    config:
      allowed_directories: ["/workspace/data"]
```

### SSE Transport

```yaml
tools:
  - name: remote_api
    type: mcp
    server: "my-mcp-server"
    transport: sse
    url: "https://mcp.example.com/sse"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    timeout: 30
    sse_read_timeout: 120
```

### WebSocket Transport

```yaml
tools:
  - name: realtime
    type: mcp
    server: "realtime-server"
    transport: websocket
    url: "wss://mcp.example.com/ws"
```

### HTTP Transport (Streamable)

```yaml
tools:
  - name: streaming_api
    type: mcp
    server: "streaming-server"
    transport: http
    url: "https://mcp.example.com/stream"
    terminate_on_close: true
```

## Database Schema

N/A - MCP tools do not persist data to HoloDeck storage. MCP servers manage their own state.

## Indexes and Constraints

N/A - No database tables.

## Migration Notes

**Existing MCPTool Model**: The current `MCPTool` in `models/tool.py` has minimal fields:
- `name`, `description`, `type`, `server`, `config`

**Migration Strategy**:
1. Add new fields with defaults (backward compatible)
2. Existing configs with only `server` will use stdio transport with auto-detected command
3. No data migration needed (config-only change)
