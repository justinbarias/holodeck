# Contract: ConfigLoader Extensions

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## Overview

Extensions to the existing `ConfigLoader` class in `src/holodeck/config/loader.py` to support MCP server management and merging.

## Existing Infrastructure

The `ConfigLoader` already provides:
- `load_global_config()` - loads `~/.holodeck/config.yaml` into `GlobalConfig`
- `merge_configs()` - merges agent config with global config
- YAML parsing and validation

## Required Modifications

### 1. GlobalConfig Model (`src/holodeck/models/config.py`)

Add new field to existing `GlobalConfig` class:

```python
from holodeck.models.tool import MCPTool

class GlobalConfig(BaseModel):
    # ... existing fields ...

    mcp_servers: list[MCPTool] | None = Field(
        None, description="Global MCP server configurations"
    )
```

### 2. ConfigLoader.merge_configs() Extension

Extend the existing `merge_configs()` method to merge global MCP servers with agent tools:

```python
def merge_configs(
    self, agent_config: dict[str, Any], global_config: GlobalConfig | None
) -> dict[str, Any]:
    """Merge agent config with global config using proper precedence.

    Extended to support MCP server merging:
    - Global mcp_servers are added to agent tools list
    - Agent-level MCP tools with same name override global
    """
    # ... existing merge logic ...

    # NEW: Merge global MCP servers into agent tools
    if global_config and global_config.mcp_servers:
        self._merge_mcp_servers(agent_config, global_config.mcp_servers)

    return agent_config

def _merge_mcp_servers(
    self,
    agent_config: dict[str, Any],
    global_mcp_servers: list[MCPTool],
) -> None:
    """Merge global MCP servers into agent tools.

    - Agent-level MCP tools take precedence over global
    - Global servers with unique names are added to tools list
    """
    if "tools" not in agent_config:
        agent_config["tools"] = []

    # Get names of existing MCP tools in agent config
    existing_names = {
        tool.get("name")
        for tool in agent_config["tools"]
        if isinstance(tool, dict) and tool.get("type") == "mcp"
    }

    # Add global MCP servers that don't conflict
    for server in global_mcp_servers:
        if server.name not in existing_names:
            agent_config["tools"].append(server.model_dump())
```

### 3. New Helper Methods for CLI

Add helper methods to support CLI operations (can be added to ConfigLoader or a new utility module):

```python
def save_global_config(config: GlobalConfig, config_path: Path | None = None) -> None:
    """Save global configuration to ~/.holodeck/config.yaml.

    Creates directory if it doesn't exist.
    """
    if config_path is None:
        config_path = Path.home() / ".holodeck" / "config.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(exclude_none=True), f, default_flow_style=False)


def add_mcp_server_to_agent(
    agent_path: Path,
    server: MCPTool,
) -> bool:
    """Add MCP server to agent.yaml tools section.

    Returns True if added, False if already exists.
    """
    # Load, modify, save agent.yaml
    ...


def remove_mcp_server_from_agent(
    agent_path: Path,
    server_name: str,
) -> bool:
    """Remove MCP server from agent.yaml tools section.

    Returns True if removed, False if not found.
    """
    # Load, modify, save agent.yaml
    ...
```

## Merge Behavior

| Scenario | Behavior |
|----------|----------|
| Server in global only | Added to agent tools at runtime |
| Server in agent only | Used as-is |
| Server in both (same name) | Agent config takes precedence |
| No tools section in agent | Created with global servers |

## File Format Examples

### Global Config (`~/.holodeck/config.yaml`)

```yaml
providers:
  openai:
    provider: openai
    api_key: ${OPENAI_API_KEY}

mcp_servers:
  - name: filesystem
    description: "File system access"
    type: mcp
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
```

### Agent Config (`agent.yaml`)

```yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o

tools:
  - name: custom-tool
    type: mcp
    # ... agent-specific MCP tool
```

### Merged Result (at runtime)

```yaml
tools:
  - name: custom-tool      # From agent.yaml
    type: mcp
  - name: filesystem       # From global config (merged)
    type: mcp
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
```
