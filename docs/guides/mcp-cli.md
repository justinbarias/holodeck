# MCP CLI Guide

The `holodeck mcp` command group provides tools to discover, install, and manage MCP (Model Context Protocol) servers for your agents.

## Overview

MCP servers extend your agent's capabilities by providing access to external tools and services. The MCP CLI integrates with the [official MCP Registry](https://registry.modelcontextprotocol.io) to help you:

- **Search** for available MCP servers
- **Add** servers to your agent or global configuration
- **List** installed servers
- **Remove** servers you no longer need

## Quick Reference

| Command | Description |
|---------|-------------|
| `holodeck mcp search [query]` | Search the MCP registry for servers |
| `holodeck mcp add <server>` | Add a server to your configuration |
| `holodeck mcp list` | List installed MCP servers |
| `holodeck mcp remove <server>` | Remove a server from your configuration |

---

## Commands

### `holodeck mcp search`

Search the official MCP registry for available servers.

**Usage:**

```bash
holodeck mcp search [QUERY] [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY` | No | Search term to filter servers by name. If omitted, lists all servers. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | 25 | Maximum results to return (1-100) |
| `--json` | - | Output results as JSON |

**Examples:**

```bash
# Search for filesystem-related servers
holodeck mcp search filesystem

# List all available servers
holodeck mcp search

# Search with more results
holodeck mcp search github --limit 50

# Get JSON output for scripting
holodeck mcp search memory --json
```

**Sample Output:**

```
Found 3 servers matching 'filesystem':

NAME                                              VERSION   DESCRIPTION                           TRANSPORT
io.github.modelcontextprotocol/server-filesystem  1.0.0     Secure file operations with co...     stdio
io.github.example/fs-tools                        2.1.0     Advanced filesystem utilities         stdio
io.github.another/file-manager                    0.5.0     Simple file management                stdio

Use 'holodeck mcp add <name>' to install a server.
```

---

### `holodeck mcp add`

Add an MCP server to your agent or global configuration.

**Usage:**

```bash
holodeck mcp add <SERVER> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `SERVER` | Yes | Server name from MCP registry (e.g., `io.github.modelcontextprotocol/server-filesystem`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--agent` | `agent.yaml` | Path to agent configuration file |
| `-g, --global` | - | Add to global config (`~/.holodeck/config.yaml`) instead |
| `--version` | latest | Server version to install |
| `--transport` | stdio | Transport type (stdio, sse, http) |
| `--name` | auto | Custom name for the server |

**Examples:**

```bash
# Add to agent.yaml in current directory
holodeck mcp add io.github.modelcontextprotocol/server-filesystem

# Add to a specific agent file
holodeck mcp add io.github.modelcontextprotocol/server-github --agent my-agent.yaml

# Add to global configuration (available to all agents)
holodeck mcp add io.github.modelcontextprotocol/server-memory -g

# Install a specific version
holodeck mcp add io.github.example/server --version 1.2.0

# Use a custom name
holodeck mcp add io.github.modelcontextprotocol/server-filesystem --name file_tools
```

**Sample Output:**

```
Fetching server 'io.github.modelcontextprotocol/server-filesystem' from registry...

Server added successfully to agent.yaml

  Name: filesystem
  Version: 1.0.0
  Transport: stdio
  Command: npx @modelcontextprotocol/server-filesystem

Required environment variables:
  - FILESYSTEM_ROOT: Root directory for file operations

Configure these in your .env file or shell environment.
```

**Environment Variables:**

When a server requires environment variables, they are displayed after installation. Add them to your `.env` file:

```bash
# .env
FILESYSTEM_ROOT=/path/to/allowed/directory
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

---

### `holodeck mcp list`

List MCP servers installed in your agent or global configuration.

**Usage:**

```bash
holodeck mcp list [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--agent` | `agent.yaml` | Path to agent configuration file |
| `-g, --global` | - | Show only global servers |
| `--all` | - | Show both agent and global servers |
| `--json` | - | Output as JSON |

**Examples:**

```bash
# List servers from agent.yaml
holodeck mcp list

# List global servers only
holodeck mcp list -g

# List all servers from both sources
holodeck mcp list --all

# JSON output for scripting
holodeck mcp list --all --json
```

**Sample Output (default):**

```
MCP servers in agent.yaml:

NAME        VERSION   TRANSPORT   DESCRIPTION
filesystem  1.0.0     stdio       Secure file operations
github      2.0.0     stdio       GitHub integration
memory      1.0.0     stdio       Persistent memory store
```

**Sample Output (--all):**

```
MCP servers:

NAME        VERSION   TRANSPORT   DESCRIPTION                  SOURCE
filesystem  1.0.0     stdio       Secure file operations       agent
github      2.0.0     stdio       GitHub integration           agent
memory      1.0.0     stdio       Persistent memory store      global
fetch       1.0.0     stdio       HTTP request capabilities    global
```

---

### `holodeck mcp remove`

Remove an MCP server from your agent or global configuration.

**Usage:**

```bash
holodeck mcp remove <SERVER> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `SERVER` | Yes | Name of the server to remove (e.g., `filesystem`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--agent` | `agent.yaml` | Path to agent configuration file |
| `-g, --global` | - | Remove from global config instead |

**Examples:**

```bash
# Remove from agent.yaml
holodeck mcp remove filesystem

# Remove from a specific agent file
holodeck mcp remove github --agent my-agent.yaml

# Remove from global configuration
holodeck mcp remove memory -g
```

**Sample Output:**

```
Server 'filesystem' removed from agent.yaml
```

---

## Configuration Locations

MCP servers can be installed in two locations:

### Agent Configuration (`agent.yaml`)

Servers added to an agent file appear in the `tools` section:

```yaml
# agent.yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o

tools:
  - name: filesystem
    type: mcp
    description: Secure file operations
    transport:
      type: stdio
      command: npx
      args:
        - "@modelcontextprotocol/server-filesystem"
```

### Global Configuration (`~/.holodeck/config.yaml`)

Servers added globally appear in the `mcp_servers` section:

```yaml
# ~/.holodeck/config.yaml
mcp_servers:
  - name: memory
    type: mcp
    description: Persistent memory store
    transport:
      type: stdio
      command: npx
      args:
        - "@modelcontextprotocol/server-memory"
```

### Precedence Rules

When an agent runs:

1. **Agent-level servers** take precedence over global servers
2. If the same server exists in both locations, the agent configuration is used
3. Global servers provide defaults available to all agents

---

## Common Workflows

### Discovering and Adding a Server

```bash
# 1. Search for what you need
holodeck mcp search "github"

# 2. Add the server you want
holodeck mcp add io.github.modelcontextprotocol/server-github

# 3. Configure required environment variables
echo "GITHUB_TOKEN=ghp_xxxx" >> .env

# 4. Verify installation
holodeck mcp list
```

### Setting Up Global Defaults

```bash
# Add commonly used servers globally
holodeck mcp add io.github.modelcontextprotocol/server-memory -g
holodeck mcp add io.github.modelcontextprotocol/server-fetch -g

# Verify global installation
holodeck mcp list -g
```

### Viewing Combined Configuration

```bash
# See all servers available to an agent
holodeck mcp list --all

# Check what's coming from where
holodeck mcp list --all --json | jq '.servers[] | {name, source}'
```

---

## Troubleshooting

### Registry Connection Error

```
Error: Failed to connect to MCP registry
```

**Cause:** Network connectivity issue or registry unavailable.

**Solution:** Check your internet connection and try again. The CLI uses a 5-second timeout.

### Server Not Found

```
Error: Server 'xyz' not found in registry
```

**Cause:** The server name doesn't exist in the registry.

**Solution:** Use `holodeck mcp search` to find the correct server name. Server names use reverse-DNS format (e.g., `io.github.user/server-name`).

### Server Already Configured

```
Error: Server 'filesystem' is already configured
```

**Cause:** The server already exists in the target configuration.

**Solution:** Use `holodeck mcp remove` first if you want to reconfigure, or check if it's in a different location (agent vs global).

### Transport Not Supported

```
Error: Server does not support stdio transport
```

**Cause:** The server only supports transports that HoloDeck doesn't currently implement (SSE, HTTP).

**Solution:** Choose a different server that supports stdio transport, or wait for future HoloDeck releases with additional transport support.

### No Agent Configuration Found

```
Error: No agent.yaml found in current directory
```

**Cause:** Running `holodeck mcp add` or `holodeck mcp list` without an agent file.

**Solution:** Either:
- Navigate to a directory with `agent.yaml`
- Use `--agent path/to/agent.yaml` to specify the file
- Use `-g` flag to work with global configuration

---

## Next Steps

- [Tools Guide](tools.md) - Learn about MCP tool configuration in detail
- [Global Configuration Guide](global-config.md) - Understand global vs agent configuration
- [Agent Configuration Guide](agent-configuration.md) - Complete agent setup reference
