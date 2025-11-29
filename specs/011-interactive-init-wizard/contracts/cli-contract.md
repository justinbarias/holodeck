# CLI Contract: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Command: `holodeck init`

### Synopsis

```
holodeck init [OPTIONS] PROJECT_NAME
```

### Description

Initialize a new HoloDeck agent project with interactive configuration wizard.

When run in an interactive terminal, prompts user for:
1. LLM provider selection
2. Vector store selection
3. MCP server selection (multi-select)

When run non-interactively (piped input, CI/CD), uses defaults or CLI flag values.

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROJECT_NAME` | Yes | Name of the project directory to create |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--template` | Choice | `conversational` | Project template to use |
| `--description` | String | None | Brief description of agent |
| `--author` | String | None | Project author name |
| `--force` | Flag | `False` | Overwrite existing project |
| `--llm` | Choice | `ollama` | LLM provider (skips prompt) |
| `--vectorstore` | Choice | `chromadb` | Vector store (skips prompt) |
| `--mcp` | String | *(see below)* | Comma-separated MCP servers |
| `--non-interactive` | Flag | `False` | Skip all prompts, use defaults/flags |

### Option Values

**`--llm`** (LLM Provider):
- `ollama` - Local LLM inference (default)
- `openai` - OpenAI API
- `azure_openai` - Azure OpenAI
- `anthropic` - Anthropic Claude

**`--vectorstore`** (Vector Store):
- `chromadb` - ChromaDB local storage (default)
- `redis` - Redis Stack
- `in-memory` - Ephemeral storage

**`--mcp`** (MCP Servers):
- Comma-separated list of package identifiers
- Default: `filesystem,memory,sequential-thinking`
- Use `--mcp none` to select no MCP servers
- Invalid server names are skipped with warning

### Interactive Mode Behavior

When stdin is a TTY and `--non-interactive` is not set:

1. **LLM Provider Prompt** (if `--llm` not provided):
   ```
   ? Select LLM provider: (Use arrow keys)
   > Ollama (local) - Local LLM inference, no API key required
     OpenAI - GPT-4, GPT-3.5-turbo via OpenAI API
     Azure OpenAI - OpenAI models via Azure deployment
     Anthropic Claude - Claude 3.5, Claude 3 via Anthropic API
   ```

2. **Vector Store Prompt** (if `--vectorstore` not provided):
   ```
   ? Select vector store: (Use arrow keys)
   > ChromaDB (default) - Embedded vector database with local persistence
     Redis - Production-grade vector store with Redis Stack
     In-Memory - Ephemeral storage for development/testing
   ```

3. **MCP Server Prompt** (if `--mcp` not provided):
   ```
   ? Select MCP servers (space to toggle): (Use arrow keys, space to select)
   > [X] Filesystem - File system access
     [X] Memory - Key-value memory storage
     [X] Sequential Thinking - Structured reasoning
     [ ] GitHub - GitHub repository access
     [ ] Brave Search - Web search capabilities
     ... (more from registry)
   ```

### Non-Interactive Mode Behavior

When stdin is not a TTY, or `--non-interactive` is set:

- All defaults are used unless overridden by flags
- No prompts displayed
- Info message logged: "Running in non-interactive mode with defaults"

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - project created |
| 1 | Error - validation failure, file I/O error |
| 2 | Abort - user cancelled (Ctrl+C) |

### Examples

**Interactive mode (default)**:
```bash
holodeck init my-agent
```

**Non-interactive with all defaults**:
```bash
holodeck init my-agent --non-interactive
```

**Specify LLM and vector store**:
```bash
holodeck init my-agent --llm openai --vectorstore redis
```

**CI/CD pipeline usage**:
```bash
holodeck init my-agent \
  --llm anthropic \
  --vectorstore chromadb \
  --mcp filesystem,github \
  --non-interactive
```

**Override only MCP servers**:
```bash
holodeck init my-agent --mcp filesystem,memory,github,brave-search
```

**No MCP servers**:
```bash
holodeck init my-agent --mcp none
```

### Error Messages

| Scenario | Message |
|----------|---------|
| Invalid `--llm` value | `Error: Invalid LLM provider 'X'. Valid options: ollama, openai, azure_openai, anthropic` |
| Invalid `--vectorstore` value | `Error: Invalid vector store 'X'. Valid options: chromadb, redis, in-memory` |
| Invalid MCP server in `--mcp` | `Warning: Skipping unknown MCP server 'X'` (continues with valid servers) |
| Project directory exists | `Project directory 'X' already exists. Use --force to overwrite.` |
| MCP registry unreachable | `Error: Cannot fetch MCP servers - network error. Check your internet connection.` |
| User cancels (Ctrl+C) | `Wizard cancelled.` |

### Output Format

**Success**:
```
✓ Project initialized successfully!

Project: my-agent
Location: /path/to/my-agent
Template: conversational
LLM Provider: openai
Vector Store: chromadb
MCP Servers: filesystem, memory, sequential-thinking
Time: 0.42s

Files created:
  • my-agent/agent.yaml
  • my-agent/instructions/system-prompt.md
  • my-agent/tools/
  ... and 5 more file(s)

Next steps:
  1. cd my-agent
  2. Set OPENAI_API_KEY environment variable
  3. Edit agent.yaml to customize your agent
  4. Run tests with: holodeck test agent.yaml
```

**Failure**:
```
✗ Project initialization failed

Error: Cannot fetch MCP servers - network error. Check your internet connection.
```
