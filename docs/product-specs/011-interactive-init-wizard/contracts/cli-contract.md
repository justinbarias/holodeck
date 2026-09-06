# CLI Contract: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Command: `holodeck init`

### Synopsis

```
holodeck init [OPTIONS]
```

### Description

Initialize a new HoloDeck agent project with interactive configuration wizard.

When run in an interactive terminal, prompts user for:

1. Agent name
2. LLM provider selection
3. Vector store selection
4. Evaluation metrics selection (multi-select)
5. MCP server selection (multi-select)

When run non-interactively (piped input, CI/CD), uses defaults or CLI flag values.

### Options

| Option              | Type               | Default          | Description                                   |
| ------------------- | ------------------ | ---------------- | --------------------------------------------- |
| `--name`            | String             | None             | Agent name (required in non-interactive mode) |
| `--template`        | Choice             | `conversational` | Project template to use                       |
| `--description`     | String             | None             | Brief description of agent                    |
| `--author`          | String             | None             | Project author name                           |
| `--force`           | Flag               | `False`          | Overwrite existing project                    |
| `--llm`             | Choice             | `ollama`         | LLM provider (skips prompt)                   |
| `--vectorstore`     | Choice             | `chromadb`       | Vector store (skips prompt)                   |
| `--evals`           | String (comma-sep) | _(see below)_    | Evaluation metrics                            |
| `--mcp`             | String (comma-sep) | _(see below)_    | MCP servers                                   |
| `--non-interactive` | Flag               | `False`          | Skip all prompts, use defaults/flags          |

### Option Values

**`--llm`** (LLM Provider):

- `ollama` - Local LLM inference with gpt-oss:20b (default)
- `openai` - OpenAI API with gpt-4o
- `azure_openai` - Azure OpenAI
- `anthropic` - Anthropic Claude

**`--vectorstore`** (Vector Store):

- `chromadb` - ChromaDB at http://localhost:8000 (default)
- `redis` - Redis Stack
- `in-memory` - Ephemeral storage

**`--evals`** (Evaluation Metrics):

- Comma-separated list of metric identifiers
- Default: `rag-faithfulness,rag-answer_relevancy`
- Available: `rag-faithfulness`, `rag-answer_relevancy`, `rag-context_precision`, `rag-context_recall`
- Use `--evals none` to disable all evals

**`--mcp`** (MCP Servers):

- Comma-separated list of server identifiers
- Default: `brave-search,memory,sequentialthinking`
- Available: `brave-search`, `memory`, `sequentialthinking`, `filesystem`, `github`, `postgres`
- Use `--mcp none` to select no MCP servers
- Invalid server names are skipped with warning

### Interactive Mode Behavior

When stdin is a TTY and `--non-interactive` is not set:

1. **Agent Name Prompt** (if `--name` not provided):

   ```
   ? Enter agent name: my-agent
   ```

2. **LLM Provider Prompt** (if `--llm` not provided):

   ```
   ? Select LLM provider: (Use arrow keys)
   > Ollama (local) - Local LLM inference, gpt-oss:20b
     OpenAI - GPT-4, GPT-3.5-turbo via OpenAI API
     Azure OpenAI - OpenAI models via Azure deployment
     Anthropic Claude - Claude 3.5, Claude 3 via Anthropic API
   ```

3. **Vector Store Prompt** (if `--vectorstore` not provided):

   ```
   ? Select vector store: (Use arrow keys)
   > ChromaDB (default) - Embedded vector database, http://localhost:8000
     Redis - Production-grade vector store with Redis Stack
     In-Memory - Ephemeral storage for development/testing
   ```

4. **Evals Prompt** (if `--evals` not provided):

   ```
   ? Select evaluation metrics (space to toggle): (Use arrow keys, space to select)
   > [X] RAG Faithfulness - Measures if response is grounded in context
     [X] RAG Answer Relevancy - Measures if response answers the question
     [ ] RAG Context Precision - Measures precision of retrieved context
     [ ] RAG Context Recall - Measures recall of retrieved context
   ```

5. **MCP Server Prompt** (if `--mcp` not provided):
   ```
   ? Select MCP servers (space to toggle): (Use arrow keys, space to select)
   > [X] Brave Search - Web search capabilities
     [X] Memory - Key-value memory storage
     [X] Sequential Thinking - Structured reasoning
     [ ] Filesystem - File system access
     [ ] GitHub - GitHub repository access
     [ ] PostgreSQL - PostgreSQL database access
   ```

### Non-Interactive Mode Behavior

When stdin is not a TTY, or `--non-interactive` is set:

- All defaults are used unless overridden by flags
- No prompts displayed
- Info message logged: "Running in non-interactive mode with defaults"
- `--name` is required in non-interactive mode

### Exit Codes

| Code | Meaning                                    |
| ---- | ------------------------------------------ |
| 0    | Success - project created                  |
| 1    | Error - validation failure, file I/O error |
| 2    | Abort - user cancelled (Ctrl+C)            |

### Examples

**Interactive mode (default)**:

```bash
holodeck init
```

**Non-interactive with all defaults**:

```bash
holodeck init --name my-agent --non-interactive
```

**Specify name and LLM provider**:

```bash
holodeck init --name my-agent --llm openai
```

**Specify vector store**:

```bash
holodeck init --name my-agent --vectorstore redis
```

**CI/CD pipeline usage**:

```bash
holodeck init \
  --name my-agent \
  --llm anthropic \
  --vectorstore chromadb \
  --evals rag-faithfulness,rag-context_precision \
  --mcp brave-search,memory \
  --non-interactive
```

**Override only MCP servers**:

```bash
holodeck init --name my-agent --mcp filesystem,memory,brave-search
```

**No MCP servers**:

```bash
holodeck init --name my-agent --mcp none
```

**No evals**:

```bash
holodeck init --name my-agent --evals none
```

### Error Messages

| Scenario                        | Message                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| Missing name in non-interactive | `Error: --name is required in non-interactive mode`                                       |
| Invalid agent name format       | `Error: Agent name must contain only alphanumeric characters, hyphens, and underscores`   |
| Invalid `--llm` value           | `Error: Invalid LLM provider 'X'. Valid options: ollama, openai, azure_openai, anthropic` |
| Invalid `--vectorstore` value   | `Error: Invalid vector store 'X'. Valid options: chromadb, redis, in-memory`              |
| Invalid eval in `--evals`       | `Warning: Skipping unknown eval metric 'X'` (continues with valid evals)                  |
| Invalid MCP server in `--mcp`   | `Warning: Skipping unknown MCP server 'X'` (continues with valid servers)                 |
| Project directory exists        | `Project directory 'X' already exists. Use --force to overwrite.`                         |
| User cancels (Ctrl+C)           | `Wizard cancelled.`                                                                       |

### Output Format

**Success**:

```
✓ Project initialized successfully!

Project: my-agent
Location: /path/to/my-agent
Template: conversational
LLM Provider: ollama (gpt-oss:20b)
Vector Store: chromadb (http://localhost:8000)
Evals: rag-faithfulness, rag-answer_relevancy
MCP Servers: brave-search, memory, sequentialthinking
Time: 0.42s

Files created:
  • my-agent/agent.yaml
  • my-agent/instructions/system-prompt.md
  • my-agent/tools/
  ... and 5 more file(s)

Next steps:
  1. cd my-agent
  2. Edit agent.yaml to customize your agent
  3. Run tests with: holodeck test agent.yaml
```

**Failure**:

```
✗ Project initialization failed

Error: Agent name must contain only alphanumeric characters, hyphens, and underscores
```
