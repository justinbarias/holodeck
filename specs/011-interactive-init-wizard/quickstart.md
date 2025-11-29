# Quickstart: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Overview

The interactive init wizard guides users through configuring a new HoloDeck agent project with smart defaults and intuitive prompts.

## Quick Start

### Interactive Mode (Default)

```bash
holodeck init my-agent
```

This launches the wizard with three prompts:
1. **LLM Provider** - Select your preferred language model provider
2. **Vector Store** - Choose where to store embeddings
3. **MCP Servers** - Select Model Context Protocol integrations

Press `Enter` at each prompt to accept the default (highlighted) option.

### Non-Interactive Mode

```bash
holodeck init my-agent --non-interactive
```

Creates a project with all defaults:
- **LLM**: Ollama (local)
- **Vector Store**: ChromaDB
- **MCP Servers**: Filesystem, Memory, Sequential Thinking

## Customizing via CLI Flags

Skip specific prompts by providing values via flags:

```bash
# Use OpenAI instead of Ollama
holodeck init my-agent --llm openai

# Use Redis for vector storage
holodeck init my-agent --vectorstore redis

# Select specific MCP servers
holodeck init my-agent --mcp filesystem,github,brave-search

# Combine flags
holodeck init my-agent --llm anthropic --vectorstore chromadb --mcp filesystem,memory
```

## LLM Provider Options

| Provider | Flag Value | Description | API Key Required |
|----------|------------|-------------|------------------|
| Ollama | `ollama` | Local inference, no cloud dependency | No |
| OpenAI | `openai` | GPT-4, GPT-3.5-turbo | Yes (`OPENAI_API_KEY`) |
| Azure OpenAI | `azure_openai` | Azure-hosted OpenAI models | Yes (`AZURE_OPENAI_API_KEY`) |
| Anthropic | `anthropic` | Claude 3.5, Claude 3 | Yes (`ANTHROPIC_API_KEY`) |

## Vector Store Options

| Store | Flag Value | Description | Best For |
|-------|------------|-------------|----------|
| ChromaDB | `chromadb` | Embedded database with local persistence | Development, single-user |
| Redis | `redis` | Production-grade with Redis Stack | Production, multi-instance |
| In-Memory | `in-memory` | Ephemeral, no persistence | Testing, prototyping |

## MCP Server Selection

Default pre-selected servers:
- `filesystem` - Access local files
- `memory` - Key-value storage
- `sequential-thinking` - Structured reasoning chains

Additional servers available from the MCP registry:
- `github` - Repository access
- `brave-search` - Web search
- `postgres` - PostgreSQL access
- And more...

### Selecting No MCP Servers

```bash
holodeck init my-agent --mcp none
```

## Generated Project Structure

```
my-agent/
├── agent.yaml          # Main configuration (includes wizard selections)
├── instructions/
│   └── system-prompt.md
├── tools/
│   └── custom_tool.py
├── tests/
│   └── test_cases.yaml
├── data/
│   └── sample.json
└── .env.example        # API key template
```

## Configuration in agent.yaml

The wizard selections are reflected in the generated `agent.yaml`:

```yaml
name: my-agent
description: "Your agent description here"

model:
  provider: openai  # From --llm or wizard selection
  name: gpt-4o
  temperature: 0.7

tools:
  # Vector store from --vectorstore or wizard selection
  - name: knowledge-base
    type: vectorstore
    database:
      provider: chromadb
    source: data/

  # MCP servers from --mcp or wizard selection
  - name: filesystem
    type: mcp
    command: npx
    args: ["@modelcontextprotocol/server-filesystem"]

  - name: memory
    type: mcp
    command: npx
    args: ["@modelcontextprotocol/server-memory"]
```

## Environment Setup

After running the wizard, set up required environment variables:

```bash
cd my-agent

# Copy the example .env file
cp .env.example .env

# Edit with your API keys
# For OpenAI:
echo "OPENAI_API_KEY=sk-..." >> .env

# For Anthropic:
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

## Common Workflows

### 1. Quick Local Development

```bash
holodeck init dev-agent
cd dev-agent
# Uses Ollama (local), ChromaDB, default MCP servers
holodeck chat agent.yaml
```

### 2. Production Setup with OpenAI + Redis

```bash
holodeck init prod-agent --llm openai --vectorstore redis
cd prod-agent
export OPENAI_API_KEY="sk-..."
export REDIS_URL="redis://localhost:6379"
holodeck test agent.yaml
```

### 3. CI/CD Pipeline

```bash
# In CI script
holodeck init test-agent \
  --llm ollama \
  --vectorstore in-memory \
  --mcp none \
  --non-interactive \
  --force
```

## Troubleshooting

### "Cannot fetch MCP servers - network error"

The wizard requires internet access to fetch the MCP server list.

**Solutions**:
1. Check your internet connection
2. Use `--mcp` flag to specify servers directly
3. Use `--non-interactive` to skip the MCP prompt

### "Project directory already exists"

Use `--force` to overwrite:

```bash
holodeck init my-agent --force
```

### Terminal doesn't support interactive prompts

The wizard automatically falls back to non-interactive mode when:
- Running in a non-TTY environment (pipes, CI)
- `--non-interactive` flag is set

To verify your terminal supports interactive mode:

```bash
python -c "import sys; print('Interactive' if sys.stdin.isatty() else 'Non-interactive')"
```

## Next Steps

After creating your project:

1. **Configure your agent**: Edit `agent.yaml`
2. **Write system prompts**: Edit `instructions/system-prompt.md`
3. **Add custom tools**: Create files in `tools/`
4. **Run tests**: `holodeck test agent.yaml`
5. **Start chatting**: `holodeck chat agent.yaml`
