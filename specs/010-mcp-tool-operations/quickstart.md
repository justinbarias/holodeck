# Quickstart: MCP Tool Operations

**Feature**: 010-mcp-tool-operations
**Date**: 2025-11-28

## Overview

This guide shows how to add MCP (Model Context Protocol) tools to your HoloDeck agent. MCP tools enable your agent to interact with external systems through standardized servers.

## Prerequisites

- HoloDeck installed (`pip install holodeck-ai`)
- Node.js installed (for npm-based MCP servers)
- An existing HoloDeck agent project

## Quick Start: Add Your First MCP Tool

### Step 1: Choose an MCP Server

Browse available MCP servers:
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)
- Common options:
  - `@modelcontextprotocol/server-filesystem` - File operations
  - `@modelcontextprotocol/server-github` - GitHub integration
  - `@modelcontextprotocol/server-postgres` - PostgreSQL queries
  - `@modelcontextprotocol/server-memory` - In-memory key-value store

**Supported Commands**: HoloDeck only allows these commands for security:
| Command | Use Case |
|---------|----------|
| `npx` | Node.js/npm MCP servers |
| `uvx` | Python MCP servers (via uv) |
| `docker` | Containerized MCP servers |

### Step 2: Add MCP Tool to agent.yaml

Edit your `agent.yaml` to add an MCP tool:

```yaml
name: "my-agent"
description: "Agent with MCP tools"

model:
  provider: openai
  name: gpt-4o-mini

instructions:
  file: instructions/system-prompt.md

tools:
  # Add MCP tool here
  - name: files
    description: "Read and write files in the workspace"
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    config:
      allowed_directories:
        - "./data"
        - "./output"
```

### Step 3: Test Your Agent

```bash
# Interactive chat to test the tool
holodeck chat agent.yaml

# Or run test cases
holodeck test agent.yaml
```

## Common MCP Tool Configurations

### File System Access

```yaml
tools:
  - name: filesystem
    description: "File system operations"
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
    config:
      allowed_directories:
        - "./data"
```

### GitHub Integration

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
        - "your-org/your-repo"
```

**Setup**: Set your GitHub token in the environment:
```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

### PostgreSQL Database

```yaml
tools:
  - name: database
    description: "Query PostgreSQL database"
    type: mcp
    server: "@modelcontextprotocol/server-postgres"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
```

**Setup**: Set your database connection string:
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/mydb"
```

### Memory Store

```yaml
tools:
  - name: memory
    description: "Store and retrieve key-value data"
    type: mcp
    server: "@modelcontextprotocol/server-memory"
    command: npx
    args: ["-y", "@modelcontextprotocol/server-memory"]
```

## Using Environment Variables

### From Environment

Reference environment variables with `${VAR_NAME}`:

```yaml
tools:
  - name: api_tool
    type: mcp
    server: "my-server"
    command: npx
    args: ["-y", "my-mcp-server"]
    env:
      API_KEY: "${MY_API_KEY}"      # Read from environment
      SECRET: "${MY_SECRET}"        # Read from environment
```

### From .env File

Load variables from a file:

```yaml
tools:
  - name: api_tool
    type: mcp
    server: "my-server"
    command: npx
    args: ["-y", "my-mcp-server"]
    envFile: ".env"                 # Load from .env file
```

Your `.env` file:
```
MY_API_KEY=secret123
MY_SECRET=supersecret
```

## Advanced: Remote MCP Servers

### SSE Transport (Server-Sent Events)

For cloud-hosted MCP servers:

```yaml
tools:
  - name: cloud_service
    description: "Cloud AI service"
    type: mcp
    server: "cloud-mcp"
    transport: sse
    url: "https://api.example.com/mcp"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    timeout: 30
    sse_read_timeout: 300
```

### WebSocket Transport

For real-time bidirectional communication:

```yaml
tools:
  - name: realtime
    description: "Real-time updates"
    type: mcp
    server: "ws-server"
    transport: websocket
    url: "wss://ws.example.com/mcp"
```

## Troubleshooting

### Error: "Environment variable 'X' not found"

The referenced environment variable is not set. Fix:
```bash
export X="your_value"
```

Or add to your `.env` file.

### Error: "command not found: npx"

Node.js is not installed. Install it:
```bash
# macOS
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm
```

### Error: "MCP server connection failed"

1. Check if the server package exists: `npx -y @modelcontextprotocol/server-NAME --help`
2. Verify network connectivity for remote servers
3. Check firewall settings

### Error: "Tool 'X' not found"

The MCP server may not expose that tool. List available tools:
```bash
holodeck tools list agent.yaml
```

## Testing MCP Tools

### Add Test Cases

```yaml
test_cases:
  - name: "Test file reading"
    input: "Read the contents of data/sample.txt"
    expected_tools: ["filesystem"]

  - name: "Test database query"
    input: "How many users are in the database?"
    expected_tools: ["database"]
```

### Run Tests

```bash
holodeck test agent.yaml
```

## Next Steps

- [MCP Configuration Contract](./contracts/mcp-config.md) - Full configuration reference
- [Data Model](./data-model.md) - Technical data model details
- [VISION.md](../../VISION.md) - HoloDeck MCP section for more examples
