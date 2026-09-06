# Contract: CLI Commands

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## Overview

Click command group `holodeck mcp` with subcommands for MCP server management.

## Command Group

```
holodeck mcp
├── search [QUERY]     Search MCP registry
├── list               List installed servers
├── add <SERVER>       Add server to config
└── remove <SERVER>    Remove server from config
```

## Command Specifications

### `holodeck mcp search [QUERY]`

Search the MCP registry for available servers.

**Arguments**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| QUERY | string | No | Search term (substring match on name) |

**Options**:
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit` | int | 25 | Results per page |
| `--json` | flag | false | Output as JSON |

**Output (table format)**:
```
NAME                                    DESCRIPTION                    TRANSPORT
io.github.modelcontextprotocol/server-  Filesystem access             stdio
io.github.modelcontextprotocol/server-  GitHub API integration        stdio
```

**Output (JSON format)**:
```json
{
  "servers": [
    {
      "name": "io.github.modelcontextprotocol/server-filesystem",
      "description": "Filesystem access",
      "transports": ["stdio"]
    }
  ]
}
```

**Exit Codes**:
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Network/API error |
| 2 | Invalid arguments |

---

### `holodeck mcp list`

List installed MCP servers from agent and/or global configuration.

**Options**:
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--agent` | path | agent.yaml | Agent config file |
| `-g, --global` | flag | false | Show global config only |
| `--all` | flag | false | Show both agent and global |
| `--json` | flag | false | Output as JSON |

**Output (table format)**:
```
SOURCE    NAME        DESCRIPTION              TRANSPORT
agent     filesystem  Filesystem access        stdio
global    github      GitHub API integration   stdio
```

**Output (JSON format)**:
```json
{
  "servers": [
    {
      "source": "agent",
      "name": "filesystem",
      "description": "Filesystem access",
      "transport": "stdio"
    }
  ]
}
```

**Exit Codes**:
| Code | Meaning |
|------|---------|
| 0 | Success (including empty list) |
| 1 | Config file error |
| 2 | Invalid arguments |

---

### `holodeck mcp add <SERVER>`

Add an MCP server to agent or global configuration.

**Arguments**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| SERVER | string | Yes | Server name (from registry) |

**Options**:
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--agent` | path | agent.yaml | Agent config file |
| `-g, --global` | flag | false | Add to global config |
| `--version` | string | latest | Server version |
| `--transport` | string | stdio | Transport type |

**Behavior**:
1. Fetch server details from registry
2. Convert to MCPTool configuration
3. Check for duplicates
4. Add to specified config file
5. Display required environment variables (if any)

**Output (success)**:
```
Added 'filesystem' to agent.yaml

Required environment variables:
  ALLOWED_PATHS - Paths the server can access
```

**Output (already exists)**:
```
Server 'filesystem' is already configured in agent.yaml
```

**Exit Codes**:
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Server not found / network error |
| 2 | Config file error |
| 3 | Already exists |

---

### `holodeck mcp remove <SERVER>`

Remove an MCP server from agent or global configuration.

**Arguments**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| SERVER | string | Yes | Server name to remove |

**Options**:
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--agent` | path | agent.yaml | Agent config file |
| `-g, --global` | flag | false | Remove from global config |

**Behavior**:
1. Load specified config file
2. Find server by name
3. Remove from tools/mcp_servers list
4. Save config file

**Output (success)**:
```
Removed 'filesystem' from agent.yaml
```

**Output (not found)**:
```
Server 'filesystem' is not configured in agent.yaml
```

**Exit Codes**:
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Server not found |
| 2 | Config file error |

## Common Error Messages

| Condition | Message |
|-----------|---------|
| No agent.yaml, no -g | "No agent.yaml found in current directory. Use --agent to specify a file or -g for global install." |
| Registry timeout | "Registry unavailable (timeout). Check your connection and try again." |
| Invalid YAML | "Invalid YAML in {file}: {error}. File not modified." |
| Permission denied | "Cannot write to {file}: permission denied." |

## Help Text

```
$ holodeck mcp --help
Usage: holodeck mcp [OPTIONS] COMMAND [ARGS]...

  Manage MCP (Model Context Protocol) servers.

  Search the official MCP registry, add servers to your agent configuration,
  and manage installed servers.

Commands:
  search  Search the MCP registry for available servers
  list    List installed MCP servers
  add     Add an MCP server to your configuration
  remove  Remove an MCP server from your configuration
```
