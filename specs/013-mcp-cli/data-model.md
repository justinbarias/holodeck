# Data Model: MCP CLI Command Group

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## Entities

### RegistryServer

Represents an MCP server from the official registry API response.

```python
class RegistryServerPackage(BaseModel):
    """Package distribution info from registry."""
    registry_type: Literal["npm", "pypi", "docker", "oci", "nuget", "mcpb"]
    identifier: str  # Package name (e.g., "@modelcontextprotocol/server-filesystem")
    version: str | None = None
    transport: TransportConfig
    environment_variables: list[EnvVarConfig] = []

class TransportConfig(BaseModel):
    """Transport configuration from registry."""
    type: Literal["stdio", "sse", "streamable-http"]
    # Additional fields for HTTP-based transports
    url: str | None = None

class EnvVarConfig(BaseModel):
    """Environment variable requirement."""
    name: str
    description: str | None = None
    required: bool = True

class RegistryServerMeta(BaseModel):
    """Registry metadata."""
    status: Literal["active", "deprecated", "deleted"] = "active"
    published_at: datetime | None = None
    updated_at: datetime | None = None
    is_latest: bool = False

class RegistryServer(BaseModel):
    """MCP server from registry."""
    name: str  # Reverse-DNS format: "io.github.user/server-name"
    description: str
    title: str | None = None
    version: str
    repository: RepositoryInfo | None = None
    website_url: str | None = None
    packages: list[RegistryServerPackage] = []
    meta: RegistryServerMeta | None = None

class RepositoryInfo(BaseModel):
    """Repository information."""
    url: str
    source: str | None = None  # "github", "gitlab", etc.
```

### GlobalConfig (Existing - MODIFY)

**Location**: `src/holodeck/models/config.py`

The existing `GlobalConfig` model needs one new field added:

```python
# EXISTING fields in GlobalConfig:
# - providers: dict[str, LLMProvider] | None
# - vectorstores: dict[str, VectorstoreConfig] | None
# - execution: ExecutionConfig | None
# - deployment: DeploymentConfig | None

# NEW field to add:
mcp_servers: list[MCPTool] | None = Field(
    None, description="Global MCP server configurations"
)
```

This extends the existing model without breaking compatibility - the field is optional with `None` default.

### MCPTool (Existing - MODIFY)

Already defined in `src/holodeck/models/tool.py`. Used for storing server configurations.

**Modification Required**: Add `registry_name` field to store the full reverse-DNS identifier from the registry. This enables proper duplicate detection and conflict warnings.

```python
# Key fields for MCP CLI context:
class MCPTool(BaseModel):
    name: str  # Short display name (e.g., "filesystem")
    description: str
    type: Literal["mcp"] = "mcp"
    transport: TransportType = TransportType.STDIO
    command: CommandType | None = None  # For stdio
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None  # For HTTP/SSE
    # ... other fields

    # NEW field for registry tracking:
    registry_name: str | None = Field(
        None,
        description=(
            "Full reverse-DNS name from registry (e.g., 'io.github.user/server'). "
            "Used for duplicate detection and origin tracking. "
            "None for manually configured servers."
        ),
    )
```

**Why this approach:**
- `name` remains the user-facing identifier for agent config references
- `registry_name` stores the canonical identifier for duplicate detection
- Servers added manually (not from registry) have `registry_name=None`
- Enables detection and warning when two servers share the same short name

## Relationships

```
RegistryServer (API Response)
    │
    ├── packages: list[RegistryServerPackage]
    │       └── transport: TransportConfig
    │       └── environment_variables: list[EnvVarConfig]
    │
    └── meta: RegistryServerMeta

GlobalConfig (~/.holodeck/config.yaml)
    │
    └── mcp_servers: list[MCPTool]

AgentConfig (agent.yaml) - Existing
    │
    └── tools: list[ToolUnion]  # includes MCPTool
```

## Transformations

### Registry Server → MCPTool

```python
def registry_to_mcp_tool(server: RegistryServer, package: RegistryServerPackage) -> MCPTool:
    """Convert registry server to MCPTool configuration."""

    # Map transport type
    transport_map = {
        "stdio": TransportType.STDIO,
        "sse": TransportType.SSE,
        "streamable-http": TransportType.HTTP,
    }

    # Map registry type to command
    command_map = {
        "npm": CommandType.NPX,
        "pypi": CommandType.UVX,
        "docker": CommandType.DOCKER,
    }

    # Build args based on registry type
    if package.registry_type == "npm":
        args = ["-y", f"{package.identifier}@{package.version or 'latest'}"]
    elif package.registry_type == "pypi":
        args = [package.identifier]
    elif package.registry_type == "docker":
        args = ["run", "-i", f"{package.identifier}:{package.version or 'latest'}"]

    # Extract env vars
    env = {ev.name: f"${{{ev.name}}}" for ev in package.environment_variables}

    # Extract short name for display, store full name for duplicate detection
    short_name = server.name.split("/")[-1] if "/" in server.name else server.name

    return MCPTool(
        name=short_name,  # User-facing display name
        description=server.description,
        type="mcp",
        transport=transport_map.get(package.transport.type, TransportType.STDIO),
        command=command_map.get(package.registry_type),
        args=args,
        env=env if env else None,
        url=package.transport.url,  # For HTTP-based transports
        registry_name=server.name,  # Full reverse-DNS name for duplicate detection
    )
```

## Validation Rules

### Server Name
- Must be non-empty
- Registry names use reverse-DNS format: `io.github.user/server-name`
- Local names (in config) use short form: `server-name`

### Global Config
- `version` must be "1.0" (for now)
- `mcp_servers` must contain valid MCPTool objects
- No duplicate servers within the list (see duplicate detection rules below)

### Agent Config Integration
- MCP servers in global config merge with agent tools
- Agent-level MCP tools with same name override global
- Override matching uses `name` field (short name)

### Duplicate Detection Rules

Duplicate detection uses a tiered approach:

1. **Registry-to-Registry**: Compare `registry_name` fields
   - `io.github.alice/filesystem` != `io.github.bob/filesystem` (different servers, OK)
   - `io.github.alice/filesystem` == `io.github.alice/filesystem` (same server, DUPLICATE)

2. **Manual-to-Manual**: Compare `name` fields (no registry_name)
   - Two manually-added servers with same `name` are duplicates

3. **Registry-to-Manual**: Compare `name` fields with warning
   - Registry server with `name=filesystem` and manual server with `name=filesystem`
   - Treated as override (manual takes precedence) but warn user

**Short Name Conflict Warning**: When adding a registry server, if another registry server with the same short `name` (but different `registry_name`) already exists, display a warning:
```
⚠️  Warning: Server 'filesystem' (io.github.bob/filesystem) has the same
    short name as existing server (io.github.alice/filesystem).
    Consider using --name to specify a unique name.
```

**Implementation**: The `holodeck mcp add` command should accept an optional `--name` flag to override the short name when conflicts occur.

## State Transitions

### Server Installation State

```
NOT_INSTALLED → INSTALLED_LOCAL (add to agent.yaml)
NOT_INSTALLED → INSTALLED_GLOBAL (add with -g flag)
INSTALLED_LOCAL → NOT_INSTALLED (remove from agent.yaml)
INSTALLED_GLOBAL → NOT_INSTALLED (remove with -g flag)
INSTALLED_BOTH → INSTALLED_GLOBAL (remove without -g)
INSTALLED_BOTH → INSTALLED_LOCAL (remove with -g)
```

### File Modification State

```
READ → PARSE → VALIDATE → MODIFY → VALIDATE → WRITE
         │         │                    │
         └─ ERROR ─┴─────── ERROR ──────┘
              │
         (abort, no changes)
```
