# Quickstart: MCP CLI Commands

**Feature**: 013-mcp-cli
**Date**: 2025-12-13

## Prerequisites

- HoloDeck CLI installed (`pip install holodeck-ai` or `uv tool install holodeck-ai`)
- Network access to `registry.modelcontextprotocol.io`
- An existing agent project (for local install) OR write access to `~/.holodeck/` (for global install)

## Basic Usage

### 1. Search for MCP Servers

Find available MCP servers in the registry:

```bash
# Search all servers
holodeck mcp search

# Search by keyword
holodeck mcp search filesystem

# Search with JSON output
holodeck mcp search github --json
```

### 2. Add a Server to Your Agent

Add an MCP server to your agent configuration:

```bash
# Add to agent.yaml in current directory
holodeck mcp add io.github.modelcontextprotocol/server-filesystem

# Add to a specific agent file
holodeck mcp add io.github.modelcontextprotocol/server-github --agent my-agent.yaml

# Add to global config (available to all agents)
holodeck mcp add io.github.modelcontextprotocol/server-filesystem -g
```

### 3. List Installed Servers

View MCP servers configured for your agent:

```bash
# List from agent.yaml
holodeck mcp list

# List from global config
holodeck mcp list -g

# List from both sources
holodeck mcp list --all
```

### 4. Remove a Server

Remove an MCP server from your configuration:

```bash
# Remove from agent.yaml
holodeck mcp remove filesystem

# Remove from global config
holodeck mcp remove filesystem -g
```

## Common Workflows

### Setting Up a New Agent with MCP Tools

```bash
# Initialize a new agent project
holodeck init my-agent

cd my-agent

# Search for useful servers
holodeck mcp search

# Add the servers you need
holodeck mcp add io.github.modelcontextprotocol/server-filesystem
holodeck mcp add io.github.modelcontextprotocol/server-github

# Verify installation
holodeck mcp list
```

### Global vs Local Installation

**Local (agent-specific)**:
- Stored in `agent.yaml` under `tools:`
- Only available to this specific agent
- Use when the server is specific to one project

**Global (user-wide)**:
- Stored in `~/.holodeck/config.yaml` under `mcp_servers:`
- Available to all agents on this machine
- Use for commonly-used servers like filesystem or git

```bash
# Common servers go global
holodeck mcp add io.github.modelcontextprotocol/server-filesystem -g

# Project-specific servers go local
holodeck mcp add io.github.mycompany/server-internal
```

### Handling Environment Variables

Some MCP servers require environment variables. After adding a server, you'll see the required variables:

```
$ holodeck mcp add io.github.modelcontextprotocol/server-github

Added 'github' to agent.yaml

Required environment variables:
  GITHUB_TOKEN - Your GitHub personal access token

Set these in your .env file or shell environment.
```

Add to your `.env` file:
```bash
GITHUB_TOKEN=ghp_your_token_here
```

## Troubleshooting

### "No agent.yaml found"

You're not in an agent project directory. Either:
- `cd` to your agent project
- Use `--agent path/to/agent.yaml`
- Use `-g` for global installation

### "Registry unavailable (timeout)"

Network issue connecting to the MCP registry. Check:
- Your internet connection
- Proxy settings (if applicable)
- Try again in a few moments

### "Server already configured"

The server is already in your config. To update it:
1. Remove the existing server: `holodeck mcp remove <name>`
2. Add it again: `holodeck mcp add <name>`

### Checking Server Details

To see full details about a server before adding:
```bash
holodeck mcp search <server-name> --json | jq '.servers[0]'
```

## Next Steps

After adding MCP servers:

1. **Configure environment variables** - Set any required API keys or tokens
2. **Test your agent** - Run `holodeck test` to verify the agent works
3. **Start chatting** - Run `holodeck chat` to interact with your agent

For more information:
- [MCP Server Documentation](https://modelcontextprotocol.io)
- [HoloDeck Agent Configuration](https://useholodeck.ai/docs/agents)
