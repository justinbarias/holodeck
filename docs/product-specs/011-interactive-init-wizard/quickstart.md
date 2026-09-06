# Quickstart: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Overview

The interactive init wizard guides users through configuring a new HoloDeck agent project with smart defaults and intuitive prompts.

## Quick Start

### Interactive Mode (Default)

```bash
holodeck init
```

This launches the wizard with five prompts:

1. **Agent Name** - Enter a name for your agent
2. **LLM Provider** - Select your preferred language model provider
3. **Vector Store** - Choose where to store embeddings
4. **Evals** - Select evaluation metrics for testing
5. **MCP Servers** - Select Model Context Protocol integrations

Press `Enter` at each prompt to accept the default (highlighted) option.

### Non-Interactive Mode

```bash
holodeck init --name my-agent --non-interactive
```

Creates a project with all defaults:

- **Agent Name**: my-agent
- **LLM**: Ollama (gpt-oss:20b)
- **Vector Store**: ChromaDB (http://localhost:8000)
- **Evals**: rag-faithfulness, rag-answer_relevancy
- **MCP Servers**: brave-search[web-search], memory, sequentialthinking

## Customizing via CLI Flags

Skip specific prompts by providing values via flags:

```bash
# Specify agent name
holodeck init --name my-agent

# Use OpenAI instead of Ollama
holodeck init --name my-agent --llm openai

# Use Redis for vector storage
holodeck init --name my-agent --vectorstore redis

# Select specific evals
holodeck init --name my-agent --evals rag-faithfulness,rag-context_precision

# Select specific MCP servers
holodeck init --name my-agent --mcp filesystem,brave-search,memory

# Combine flags
holodeck init --name my-agent --llm anthropic --vectorstore chromadb --mcp filesystem,memory
```

## LLM Provider Options

| Provider     | Flag Value     | Default Model     | Description                          | API Key Required             |
| ------------ | -------------- | ----------------- | ------------------------------------ | ---------------------------- |
| Ollama       | `ollama`       | gpt-oss:20b       | Local inference, no cloud dependency | No                           |
| OpenAI       | `openai`       | gpt-4o            | GPT-4, GPT-3.5-turbo                 | Yes (`OPENAI_API_KEY`)       |
| Azure OpenAI | `azure_openai` | gpt-4o            | Azure-hosted OpenAI models           | Yes (`AZURE_OPENAI_API_KEY`) |
| Anthropic    | `anthropic`    | claude-3-5-sonnet | Claude 3.5, Claude 3                 | Yes (`ANTHROPIC_API_KEY`)    |

## Vector Store Options

| Store     | Flag Value  | Default Endpoint       | Description                              | Best For                   |
| --------- | ----------- | ---------------------- | ---------------------------------------- | -------------------------- |
| ChromaDB  | `chromadb`  | http://localhost:8000  | Embedded database with local persistence | Development, single-user   |
| Redis     | `redis`     | redis://localhost:6379 | Production-grade with Redis Stack        | Production, multi-instance |
| In-Memory | `in-memory` | N/A                    | Ephemeral, no persistence                | Testing, prototyping       |

## Evaluation Metrics Options

| Metric                | Flag Value              | Description                                 | Default |
| --------------------- | ----------------------- | ------------------------------------------- | ------- |
| RAG Faithfulness      | `rag-faithfulness`      | Measures if response is grounded in context | Yes     |
| RAG Answer Relevancy  | `rag-answer_relevancy`  | Measures if response answers the question   | Yes     |
| RAG Context Precision | `rag-context_precision` | Measures precision of retrieved context     | No      |
| RAG Context Recall    | `rag-context_recall`    | Measures recall of retrieved context        | No      |

## MCP Server Options

Default pre-selected servers:

- `brave-search` - Web search capabilities (brave-search[web-search])
- `memory` - Key-value memory storage (@modelcontextprotocol/server-memory)
- `sequentialthinking` - Structured reasoning (@modelcontextprotocol/server-sequentialthinking)

Additional available servers:

- `filesystem` - Access local files (@modelcontextprotocol/server-filesystem)
- `github` - Repository access
- `postgres` - PostgreSQL database access

### Selecting No MCP Servers

```bash
holodeck init --name my-agent --mcp none
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
  provider: ollama # From --llm or wizard selection
  name: gpt-oss:20b
  temperature: 0.7

tools:
  # Vector store from --vectorstore or wizard selection
  - name: knowledge-base
    type: vectorstore
    database:
      provider: chromadb
      endpoint: http://localhost:8000
    source: data/

  # MCP servers from --mcp or wizard selection
  - name: brave-search
    type: mcp
    command: npx
    args: ["@anthropic/mcp-server-brave-search"]

  - name: memory
    type: mcp
    command: npx
    args: ["@modelcontextprotocol/server-memory"]

  - name: sequentialthinking
    type: mcp
    command: npx
    args: ["@modelcontextprotocol/server-sequentialthinking"]

evaluations:
  metrics:
    - name: rag-faithfulness
      type: ai
    - name: rag-answer_relevancy
      type: ai
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

# For Brave Search MCP:
echo "BRAVE_API_KEY=..." >> .env
```

## Common Workflows

### 1. Quick Local Development

```bash
holodeck init --name dev-agent
cd dev-agent
# Uses Ollama (local), ChromaDB, default evals, default MCP servers
holodeck chat agent.yaml
```

### 2. Production Setup with OpenAI + Redis

```bash
holodeck init --name prod-agent --llm openai --vectorstore redis
cd prod-agent
export OPENAI_API_KEY="sk-..."
export REDIS_URL="redis://localhost:6379"
holodeck test agent.yaml
```

### 3. CI/CD Pipeline

```bash
# In CI script
holodeck init \
  --name test-agent \
  --llm ollama \
  --vectorstore in-memory \
  --evals rag-faithfulness \
  --mcp none \
  --non-interactive \
  --force
```

### 4. Research Agent with Extended Evals

```bash
holodeck init \
  --name research-agent \
  --llm anthropic \
  --evals rag-faithfulness,rag-answer_relevancy,rag-context_precision,rag-context_recall \
  --mcp brave-search,memory,sequentialthinking
```

## Troubleshooting

### "Project directory already exists"

Use `--force` to overwrite:

```bash
holodeck init --name my-agent --force
```

### Terminal doesn't support interactive prompts

The wizard automatically falls back to non-interactive mode when:

- Running in a non-TTY environment (pipes, CI)
- `--non-interactive` flag is set

To verify your terminal supports interactive mode:

```bash
python -c "import sys; print('Interactive' if sys.stdin.isatty() else 'Non-interactive')"
```

### "Agent name is required"

In non-interactive mode, you must provide the agent name:

```bash
holodeck init --name my-agent --non-interactive
```

## Next Steps

After creating your project:

1. **Configure your agent**: Edit `agent.yaml`
2. **Write system prompts**: Edit `instructions/system-prompt.md`
3. **Add custom tools**: Create files in `tools/`
4. **Run tests**: `holodeck test agent.yaml`
5. **Start chatting**: `holodeck chat agent.yaml`
