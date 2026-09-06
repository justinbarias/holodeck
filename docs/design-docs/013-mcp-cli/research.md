# Research: MCP CLI Command Group

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## MCP Registry API

### Decision: Use Official MCP Registry API v0.1

**Rationale**: The official MCP registry at `https://registry.modelcontextprotocol.io` provides a well-documented REST API following OpenAPI specifications. No authentication required for read operations.

**Alternatives Considered**:
- Custom registry: Rejected - unnecessary complexity, MCP ecosystem uses official registry
- Local server list only: Rejected - would defeat the purpose of discovery

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v0.1/servers` | GET | List/search servers (paginated) |
| `/v0.1/servers/{name}/versions` | GET | List versions of a server |
| `/v0.1/servers/{name}/versions/{version}` | GET | Get specific server version |

### Query Parameters (for `/v0.1/servers`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `search` | string | Filter by server name (substring match) |
| `cursor` | string | Pagination cursor (format: `{name}:{version}`) |
| `limit` | integer | Max items per page |

### Response Structure

```json
{
  "servers": [
    {
      "server": {
        "name": "io.github.user/server-name",
        "description": "Server description",
        "title": "Human-readable title",
        "version": "1.0.0",
        "repository": {
          "url": "https://github.com/user/repo",
          "source": "github"
        },
        "packages": [
          {
            "registryType": "npm|pypi|docker",
            "identifier": "package-name",
            "version": "1.0.0",
            "transport": {
              "type": "stdio|sse|http"
            },
            "environmentVariables": [
              {
                "name": "API_KEY",
                "description": "Required API key"
              }
            ]
          }
        ]
      },
      "_meta": {
        "io.modelcontextprotocol.registry/official": {
          "status": "active|deprecated",
          "publishedAt": "2025-01-01T00:00:00Z",
          "isLatest": true
        }
      }
    }
  ],
  "metadata": {
    "nextCursor": "io.github.user/next:1.0.0",
    "count": 25
  }
}
```

## YAML Modification Strategy

### Decision: Use ruamel.yaml for Comment Preservation

**Rationale**: PyYAML does not preserve comments. The `ruamel.yaml` library maintains comments and formatting when modifying YAML files. However, since HoloDeck already uses PyYAML extensively, we will use PyYAML but document that comments may be lost during modification.

**Alternatives Considered**:
- ruamel.yaml: Would preserve comments but adds dependency
- Custom YAML parser: Too complex, not worth the effort
- PyYAML with manual comment extraction: Fragile and error-prone

**Decision Update**: Use PyYAML (existing dependency) with clear documentation that YAML modifications may not preserve comments. This is acceptable for CLI tool usage.

## Global Config Structure

### Decision: Add mcp_servers Field to Existing GlobalConfig

**Rationale**: The `GlobalConfig` model already exists in `src/holodeck/models/config.py` with fields for `providers`, `vectorstores`, `execution`, and `deployment`. The `ConfigLoader` in `src/holodeck/config/loader.py` already handles loading and merging global config. We extend this infrastructure rather than creating new components.

**Changes Required**:
1. Add `mcp_servers: list[MCPTool] | None` field to `GlobalConfig` model
2. Extend `ConfigLoader.merge_configs()` to merge global MCP servers with agent tools

```yaml
# ~/.holodeck/config.yaml (extended structure)
providers:
  openai:
    provider: openai
    api_key: ${OPENAI_API_KEY}

vectorstores:
  default:
    provider: postgres
    connection_string: ${DATABASE_URL}

# NEW: MCP servers section
mcp_servers:
  - name: filesystem
    description: "File system access"
    type: mcp
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"

  - name: github
    description: "GitHub API access"
    type: mcp
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-github"
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
```

### Merge Behavior

When loading agent configuration:
1. Load global `mcp_servers` from `~/.holodeck/config.yaml`
2. Load agent `tools` from `agent.yaml`
3. For MCP tools with matching names: agent config takes precedence
4. For unique MCP tools: combine both lists

## Transport Type Mapping

### Decision: Map Registry Transport to MCPTool Model

| Registry Transport | MCPTool Transport | Command |
|-------------------|-------------------|---------|
| stdio | TransportType.STDIO | From package registryType |
| sse | TransportType.SSE | N/A (uses URL) |
| streamable-http | TransportType.HTTP | N/A (uses URL) |

### Command Mapping by Registry Type

| registryType | CommandType | Args Pattern |
|--------------|-------------|--------------|
| npm | CommandType.NPX | `["-y", "{identifier}@{version}"]` |
| pypi | CommandType.UVX | `["{identifier}"]` |
| docker | CommandType.DOCKER | `["run", "-i", "{identifier}:{version}"]` |

## Error Handling Strategy

### Decision: Fail Fast with Actionable Messages

**Rationale**: CLI tools should provide immediate, clear feedback. 5-second timeout prevents hanging on slow networks.

| Error Type | User Message |
|------------|--------------|
| Network timeout | "Registry unavailable. Check your connection and try again." |
| 404 Not Found | "Server '{name}' not found in registry." |
| Invalid YAML | "Invalid YAML in {file}: {details}. File not modified." |
| Server already exists | "Server '{name}' already configured in {location}." |
| No agent.yaml found | "No agent.yaml found. Use --agent or -g for global install." |

## Dependencies

### New Dependencies: None

All required functionality is covered by existing dependencies:
- `requests`: HTTP client for registry API
- `pyyaml`: YAML parsing/writing
- `click`: CLI framework
- `pydantic`: Model validation

### Existing Code Reuse

| Component | Location | Modification |
|-----------|----------|--------------|
| GlobalConfig model | `src/holodeck/models/config.py` | ADD `mcp_servers` field |
| MCPTool model | `src/holodeck/models/tool.py` | USE as-is (no changes) |
| ConfigLoader | `src/holodeck/config/loader.py` | EXTEND `merge_configs()` for MCP servers |
| CLI main.py | `src/holodeck/cli/main.py` | ADD `mcp` command group registration |
| CLI patterns | `src/holodeck/cli/commands/` | FOLLOW existing patterns for new `mcp.py` |

### Key Insight

The existing `ConfigLoader.merge_configs()` method already:
- Merges global LLM provider configs into agent model
- Resolves vectorstore references from global config
- Follows agent-takes-precedence pattern

We follow this same pattern for MCP servers, adding a `_merge_mcp_servers()` helper method.
