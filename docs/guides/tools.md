# Tools Reference Guide

This guide explains HoloDeck's four tool types that extend agent capabilities.

## Overview

Tools are agent capabilities defined in `agent.yaml`. HoloDeck supports four tool types:

| Tool Type             | Description                    | Status         |
| --------------------- | ------------------------------ | -------------- |
| **Vectorstore Tools** | Semantic search over data      | ✅ Implemented |
| **MCP Tools**         | Model Context Protocol servers | ✅ Implemented |
| **Function Tools**    | Custom Python functions        | 🚧 Planned     |
| **Prompt Tools**      | LLM-powered semantic functions | 🚧 Planned     |

> **Note**: **Vectorstore Tools** and **MCP Tools** are fully implemented. Function and Prompt tools are defined in the configuration schema but not yet functional.

## Common Tool Fields

All tools share these fields:

```yaml
tools:
  - name: tool-id # Required: Tool identifier (unique)
    description: What it does # Required: Human-readable description
    type: vectorstore|function|mcp|prompt # Required: Tool type
```

### Name

- **Required**: Yes
- **Type**: String
- **Format**: 1-100 characters, alphanumeric + underscores
- **Uniqueness**: Must be unique within agent
- **Purpose**: Used to reference tool in test cases, execution logs

```yaml
- name: search_kb
```

### Description

- **Required**: Yes
- **Type**: String
- **Max Length**: 500 characters
- **Purpose**: Helps agent understand when to use this tool

```yaml
- description: Search company knowledge base for answers
```

### Type

- **Required**: Yes
- **Type**: String (Enum)
- **Options**: `vectorstore`, `function`, `mcp`, `prompt`
- **Purpose**: Determines which additional fields are required

```yaml
- type: vectorstore
```

---

## Vectorstore Tools ✅

> **Status**: Fully implemented

Semantic search over unstructured or structured data.

### When to Use

- Searching documents, knowledge bases, FAQs
- Semantic similarity matching
- Context retrieval for RAG (Retrieval-Augmented Generation)

### Basic Example

```yaml
- name: search-kb
  description: Search knowledge base for answers
  type: vectorstore
  source: knowledge_base/
```

### Supported Vector Database Providers

HoloDeck supports multiple vector database backends through Semantic Kernel's VectorStoreCollection abstractions. You can switch providers via configuration without changing your agent code.

| Provider    | Description                         | Connection                       | Install Command                |
| ----------- | ----------------------------------- | -------------------------------- | ------------------------------ |
| `postgres`  | PostgreSQL with pgvector extension  | `postgresql://user:pass@host/db` | `uv add holodeck-ai[postgres]` |
| `qdrant`    | Qdrant vector database              | `http://localhost:6333`          | `uv add holodeck-ai[qdrant]`   |
| `chromadb`  | ChromaDB (local or server)          | Local path or host URL           | `uv add holodeck-ai[chromadb]` |
| `pinecone`  | Pinecone serverless vector database | API key + index name             | `uv add holodeck-ai[pinecone]` |
| `in-memory` | Simple in-memory storage            | None required                    | Built-in                       |

> **Tip**: Install all vector store providers at once with `uv add holodeck-ai[vectorstores]`. Use `in-memory` for development and testing without installing any dependencies. Switch to a persistent provider like `postgres`, `qdrant`, or `chromadb` for production.

#### Database Configuration Examples

**PostgreSQL with pgvector**

```yaml
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database:
    provider: postgres
    connection_string: postgresql://user:password@localhost:5432/mydb
```

**Azure AI Search**

```yaml
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database:
    provider: azure-ai-search
    connection_string: ${AZURE_SEARCH_ENDPOINT}
    api_key: ${AZURE_SEARCH_API_KEY}
```

**Qdrant**

```yaml
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database:
    provider: qdrant
    url: http://localhost:6333
    # api_key: optional-api-key
```

**In-Memory (development only)**

```yaml
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database:
    provider: in-memory
```

**Reference to Global Config**

You can also reference a named vectorstore from your global `config.yaml`:

```yaml
# In agent.yaml
- name: search-kb
  type: vectorstore
  source: knowledge_base/
  database: my-postgres-store # Reference to config.yaml vectorstores section
```

```yaml
# In config.yaml
vectorstores:
  my-postgres-store:
    provider: postgres
    connection_string: ${DATABASE_URL}
```

### Required Fields

#### Source

- **Type**: String (path)
- **Purpose**: Data file or directory to index
- **Formats Supported**:
  - Single files: `.txt`, `.md`, `.pdf`, `.json`, `.csv`
  - Directories: Recursively indexes supported formats
  - Remote URLs: File auto-cached locally

```yaml
source: knowledge_base/
# OR
source: docs.json
# OR
source: https://example.com/data.pdf
```

### Optional Fields

#### Embedding Model

- **Type**: String
- **Purpose**: Which embedding model to use
- **Default**: Provider-specific default
- **Examples**: `text-embedding-3-small`, `text-embedding-ada-002`

```yaml
embedding_model: text-embedding-3-small
```

#### Vector Field

- **Type**: String or List of strings
- **Purpose**: Which field(s) to vectorize (for JSON/CSV)
- **Default**: Auto-detect text fields
- **Note**: XOR with `vector_fields` (use one or the other)

```yaml
vector_field: content
# OR
vector_field: [title, description]
```

#### Meta Fields

- **Type**: List of strings
- **Purpose**: Metadata fields to include in results
- **Default**: All fields included

```yaml
meta_fields: [title, source, date]
```

#### Chunk Size

- **Type**: Integer
- **Purpose**: Characters per chunk for text splitting
- **Default**: 512
- **Constraint**: Must be > 0

```yaml
chunk_size: 1024
```

#### Chunk Overlap

- **Type**: Integer
- **Purpose**: Characters to overlap between chunks
- **Default**: 0
- **Constraint**: Must be >= 0

```yaml
chunk_overlap: 100
```

#### Record Path

- **Type**: String
- **Purpose**: Path to array in nested JSON (dot notation)
- **Example**: For `{data: {items: [{...}]}}`, use `data.items`

```yaml
record_path: data.records
```

#### Record Prefix

- **Type**: String
- **Purpose**: Prefix added to record fields
- **Default**: None

```yaml
record_prefix: record_
```

#### Meta Prefix

- **Type**: String
- **Purpose**: Prefix added to metadata fields
- **Default**: None

```yaml
meta_prefix: meta_
```

### Complete Example

```yaml
- name: search-docs
  description: Search technical documentation
  type: vectorstore
  source: docs/
  embedding_model: text-embedding-3-small
  vector_field: [title, content]
  meta_fields: [source, date, url]
  chunk_size: 1024
  chunk_overlap: 128
```

### Data Format Examples

**Text Files** (`.txt`, `.md`)

```
# Document Title

This is the document content that will be
vectorized for semantic search.
```

**JSON** (Array of objects)

```json
[
  {
    "title": "Getting Started",
    "content": "How to get started with the platform...",
    "source": "docs/intro.md"
  }
]
```

**JSON** (Nested structure)

```json
{
  "data": {
    "records": [
      {
        "id": 1,
        "title": "Article 1",
        "content": "..."
      }
    ]
  }
}
```

Use `record_path: data.records` to access records.

**CSV**

```csv
title,content,source
"Getting Started","How to get started...","docs/intro"
"API Reference","API documentation...","docs/api"
```

---

## MCP Tools ✅

> **Status**: Fully implemented (stdio transport)

Model Context Protocol (MCP) server integrations enable agents to interact with external systems through a standardized protocol. HoloDeck uses Semantic Kernel's MCP plugins for seamless integration.

> **Finding MCP Servers**: Browse the official MCP server registry at [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) for a curated list of available servers including filesystem, GitHub, Slack, Google Drive, PostgreSQL, and many more community-contributed integrations.

### When to Use

- File system operations (read, write, list files)
- GitHub/GitLab operations (issues, PRs, code)
- Database access (SQLite, PostgreSQL)
- Web browsing and search
- Any standardized MCP server

### Basic Example

```yaml
- name: filesystem
  description: Read and write files in the workspace
  type: mcp
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
```

### Complete Example

```yaml
tools:
  # MCP filesystem tool for reading/writing files
  - type: mcp
    name: filesystem
    description: Read and write files in the workspace data directory
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "./sample/data"]
    config:
      allowed_directories: ["./sample/data"]
    request_timeout: 30
```

### Required Fields

#### Command

- **Type**: String (enum: `npx`, `node`, `uvx`, `docker`)
- **Purpose**: How to launch the MCP server
- **Required**: Yes (for stdio transport)

```yaml
command: npx     # For npm packages (auto-installs if needed)
# OR
command: node    # For local .js files or installed packages
# OR
command: uvx     # For Python packages via uv
# OR
command: docker  # For containerized servers
```

**When to use each:**

- `npx` - Run npm packages directly (e.g., `@modelcontextprotocol/server-filesystem`)
- `node` - Run local JavaScript files (e.g., `./tools/my-server.js`)
- `uvx` - Run Python packages via uv (e.g., `mcp-server-fetch`)
- `docker` - Run containerized MCP servers

#### Args

- **Type**: List of strings
- **Purpose**: Command-line arguments for the server
- **Note**: Often includes the server package name and configuration

```yaml
args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
```

### Optional Fields

#### Transport

- **Type**: String (enum: `stdio`, `sse`, `websocket`, `http`)
- **Default**: `stdio`
- **Purpose**: Communication protocol with the server
- **Note**: Currently only `stdio` is implemented

```yaml
transport: stdio # Default, works with most servers
```

#### Config

- **Type**: Object (free-form)
- **Purpose**: Server-specific configuration passed via MCP_CONFIG env var
- **Validation**: Server validates at runtime

```yaml
config:
  allowed_directories: ["./data", "/tmp"]
  max_file_size: 1048576
```

#### Env

- **Type**: Object (string key-value pairs)
- **Purpose**: Environment variables for the server process
- **Supports**: Variable substitution with `${VAR_NAME}`

```yaml
env:
  GITHUB_TOKEN: "${GITHUB_TOKEN}"
  API_KEY: "static-value"
```

#### Env File

- **Type**: String (path)
- **Purpose**: Load environment variables from a file
- **Format**: Standard `.env` file format

```yaml
env_file: .env.mcp
```

#### Request Timeout

- **Type**: Integer (seconds)
- **Default**: 30
- **Purpose**: Timeout for individual MCP requests

```yaml
request_timeout: 60
```

#### Encoding

- **Type**: String
- **Default**: `utf-8`
- **Purpose**: Character encoding for stdio communication

```yaml
encoding: utf-8
```

### Sample MCP Servers

#### Filesystem (stdio)

Read, write, and manage files:

```yaml
- name: filesystem
  type: mcp
  description: File system operations
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
  config:
    allowed_directories: ["./data"]
```

**Tools provided**: `read_file`, `write_file`, `list_directory`, `create_directory`, `move_file`, `search_files`, `get_file_info`

#### GitHub

Interact with GitHub repositories:

```yaml
- name: github
  type: mcp
  description: GitHub repository operations
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
```

**Tools provided**: `search_repositories`, `create_issue`, `list_issues`, `get_file_contents`, `create_pull_request`, `fork_repository`

#### SQLite

Query SQLite databases:

```yaml
- name: sqlite
  type: mcp
  description: SQLite database queries
  command: npx
  args:
    [
      "-y",
      "@modelcontextprotocol/server-sqlite",
      "--db-path",
      "./data/database.db",
    ]
```

**Tools provided**: `read_query`, `write_query`, `create_table`, `list_tables`, `describe_table`

#### Brave Search

Web search capabilities:

```yaml
- name: brave-search
  type: mcp
  description: Web search via Brave
  command: npx
  args: ["-y", "@modelcontextprotocol/server-brave-search"]
  env:
    BRAVE_API_KEY: "${BRAVE_API_KEY}"
```

**Tools provided**: `brave_web_search`, `brave_local_search`

#### Puppeteer (Browser Automation)

Browser automation and web scraping:

```yaml
- name: puppeteer
  type: mcp
  description: Browser automation
  command: npx
  args: ["-y", "@modelcontextprotocol/server-puppeteer"]
```

**Tools provided**: `puppeteer_navigate`, `puppeteer_screenshot`, `puppeteer_click`, `puppeteer_fill`, `puppeteer_evaluate`

### Local Node.js Servers (node)

For local JavaScript MCP server files, use `node`:

```yaml
- name: my-custom-server
  type: mcp
  description: Custom local MCP server
  command: node
  args: ["./tools/my-mcp-server.js", "--config", "./config.json"]
```

> **Note**: Use `node` for local `.js` files. Use `npx` for npm packages.

### Python MCP Servers (uvx)

For Python-based MCP servers, use `uvx`:

```yaml
- name: mcp-server-fetch
  type: mcp
  description: Fetch web content
  command: uvx
  args: ["mcp-server-fetch"]
```

#### Memory (Short-Term Storage)

Scratchpad for agent short-term memory storage:

```yaml
- name: memory
  type: mcp
  description: Scratchpad for short term memory storage
  command: uvx
  args: ["basic-memory", "mcp"]
  request_timeout: 30
```

**Tools provided**: `write_note`, `read_note`, `search_notes`, `delete_note`

> **Use case**: Enable agents to persist information across conversation turns, store intermediate results, or maintain context during multi-step tasks, and especially between chat sessions.

### Docker MCP Servers

For containerized servers:

```yaml
- name: custom-server
  type: mcp
  description: Custom containerized server
  command: docker
  args: ["run", "-i", "--rm", "my-mcp-server:latest"]
```

### Environment Variable Patterns

**Static values:**

```yaml
env:
  API_KEY: "sk-1234567890"
```

**Environment substitution:**

```yaml
env:
  GITHUB_TOKEN: "${GITHUB_TOKEN}" # From process environment
```

**From env file:**

```yaml
env_file: .env.mcp
env:
  OVERRIDE_VAR: "override-value" # Overrides env_file
```

### Error Handling

- **Server unavailable**: Error during agent startup
- **Connection timeout**: Configurable via `request_timeout`
- **Invalid config**: Error during agent startup (validation)
- **Runtime errors**: Logged and returned as tool error responses

### Prerequisites

MCP tools require the appropriate runtime to be installed on your machine based on the `command` you use:

| Command  | Required Software | Installation                                                            |
| -------- | ----------------- | ----------------------------------------------------------------------- |
| `npx`    | Node.js + npm     | [nodejs.org](https://nodejs.org/) or `brew install node`                |
| `node`   | Node.js           | [nodejs.org](https://nodejs.org/) or `brew install node`                |
| `uvx`    | uv (Python)       | `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |
| `docker` | Docker            | [docker.com](https://docker.com/) or `brew install --cask docker`       |

**Verify installation:**

```bash
# For npm-based MCP servers
node --version    # Should show v18+ recommended
npx --version

# For Python-based MCP servers
uv --version
uvx --version

# For containerized servers
docker --version
```

> **Tip**: Most MCP servers use `npx` with npm packages. Ensure Node.js 18+ is installed for best compatibility.

### Lifecycle Management

MCP plugins are automatically managed:

1. **Startup**: Plugin initialized and connected when agent starts
2. **Execution**: Tools discovered and registered on the kernel
3. **Shutdown**: Plugin properly closed when session ends

> **Important**: Always terminate chat sessions properly (`exit` or `quit`) to ensure MCP servers are cleanly shut down.

---

## Function Tools 🚧

> **Status**: Planned - Configuration schema defined, execution not yet implemented

Execute custom Python functions.

### When to Use

- Custom business logic
- Database queries
- System operations
- Complex calculations

### Basic Example

```yaml
- name: get-user
  description: Look up user information
  type: function
  file: tools/users.py
  function: get_user
```

### Required Fields

#### File

- **Type**: String (path)
- **Purpose**: Python file containing the function
- **Path**: Relative to `agent.yaml` directory
- **Format**: Standard Python module

```yaml
file: tools/users.py
```

#### Function

- **Type**: String
- **Purpose**: Function name to call
- **Format**: Valid Python identifier

```yaml
function: get_user
```

### Optional Fields

#### Parameters

- **Type**: Object mapping parameter names to schemas
- **Purpose**: Define function parameters the agent can pass
- **Default**: No parameters (function takes no args)

```yaml
parameters:
  user_id:
    type: string
    description: User identifier
  include_details:
    type: boolean
    description: Include detailed information
```

Parameter schema fields:

- `type`: `string`, `integer`, `float`, `boolean`, `array`, `object`
- `description`: What the parameter is for
- `enum`: Optional list of allowed values
- `default`: Optional default value

### Complete Example

```yaml
- name: create-ticket
  description: Create a support ticket
  type: function
  file: tools/support.py
  function: create_ticket
  parameters:
    title:
      type: string
      description: Ticket title (required)
    priority:
      type: string
      description: Ticket priority
      enum: [low, medium, high]
    description:
      type: string
      description: Detailed description
```

### Python Function Format

```python
# tools/support.py

def create_ticket(title: str, priority: str = "medium", description: str = "") -> dict:
    """
    Create a support ticket.

    Args:
        title: Ticket title
        priority: low|medium|high
        description: Detailed description

    Returns:
        Created ticket data
    """
    return {
        "id": "TICKET-123",
        "status": "open",
        "title": title,
        "priority": priority,
    }
```

### Best Practices

- Keep functions focused on single tasks
- Use clear parameter names
- Add type hints and docstrings
- Handle errors gracefully (return error messages)
- Return JSON-serializable data
- Avoid long-running operations (prefer async tools in future versions)

---

## Prompt Tools 🚧

> **Status**: Planned - Configuration schema defined, execution not yet implemented

LLM-powered semantic functions with template substitution.

### When to Use

- Text generation with templates
- Specialized prompts for specific tasks
- Reusable prompt chains
- A/B testing different prompts

### Basic Example

```yaml
- name: summarize
  description: Summarize text into key points
  type: prompt
  template: "Summarize this in 3 bullet points: {{text}}"
  parameters:
    text:
      type: string
      description: Text to summarize
```

### Required Fields

#### Template or File

Either `template` (inline) or `file` (external), not both:

**Inline Template**

- **Type**: String
- **Max Length**: 5000 characters
- **Syntax**: Mustache-style `{{variable}}`

```yaml
template: "Summarize: {{content}}"
```

**Template File**

- **Type**: String (path)
- **Path**: Relative to `agent.yaml`

```yaml
file: prompts/summarize.txt
```

File contents:

```
Summarize this text in 3 bullet points:

{{text}}

Focus on key takeaways.
```

#### Parameters

- **Type**: Object mapping parameter names to schemas
- **Purpose**: Template variables the agent can fill
- **Required**: Yes (at least one)

```yaml
parameters:
  text:
    type: string
    description: Text to process
```

### Optional Fields

#### Model Override

- **Type**: Model configuration object
- **Purpose**: Use different model for this tool
- **Default**: Uses agent's model

```yaml
model:
  provider: openai
  name: gpt-4 # Different from agent's model
  temperature: 0.2
```

### Complete Example

```yaml
- name: code-reviewer
  description: Review code for best practices
  type: prompt
  file: prompts/code_review.txt
  model:
    provider: openai
    name: gpt-4
    temperature: 0.3
  parameters:
    code:
      type: string
      description: Code to review
    language:
      type: string
      description: Programming language
      enum: [python, javascript, go, java]
```

Template file (`prompts/code_review.txt`):

```
Review this {{language}} code for best practices.

Code:
{{code}}

Provide:
1. Issues found
2. Suggestions for improvement
3. Security considerations
```

### Template Syntax

Variables use Mustache-style syntax:

```
Simple variable: {{name}}

Conditionals (if parameter provided):
{{#if description}}
Description: {{description}}
{{/if}}

Loops (if parameter is array):
{{#each items}}
- {{this}}
{{/each}}
```

---

## Tool Comparison

| Feature        | Vectorstore             | MCP                     | Function        | Prompt          |
| -------------- | ----------------------- | ----------------------- | --------------- | --------------- |
| **Status**     | ✅ Implemented          | ✅ Implemented          | 🚧 Planned      | 🚧 Planned      |
| **Use Case**   | Search data             | External integrations   | Custom logic    | Template-based  |
| **Execution**  | Vector similarity       | MCP protocol (stdio)    | Python function | LLM generation  |
| **Setup**      | Data files              | Server config + runtime | Python files    | Template text   |
| **Parameters** | Implicit (search query) | Server-specific tools   | Defined in code | Defined in YAML |
| **Latency**    | Medium (~100ms)         | Medium (~50-500ms)      | Low (<10ms)     | High (LLM call) |
| **Cost**       | Embedding API           | Server resource         | Internal        | LLM tokens      |

---

## Common Patterns

### Knowledge Base Search

```yaml
- name: search-kb
  type: vectorstore
  source: kb/
  chunk_size: 512
  embedding_model: text-embedding-3-small
```

### Database Query

```yaml
- name: query-db
  type: function
  file: tools/db.py
  function: query
  parameters:
    sql:
      type: string
```

### File Operations (MCP)

```yaml
- name: filesystem
  type: mcp
  description: Read and write files
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
  config:
    allowed_directories: ["./data"]
```

### GitHub Integration (MCP)

```yaml
- name: github
  type: mcp
  description: GitHub repository operations
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
```

### Text Transformation

```yaml
- name: translate
  type: prompt
  template: "Translate to {{language}}: {{text}}"
  parameters:
    text:
      type: string
    language:
      type: string
```

---

## Error Handling

### Vectorstore Tool Errors

- **No data found**: Returns empty results
- **Invalid path**: Error during agent startup (config validation)
- **Unsupported format**: Error during agent startup

### Function Tool Errors

- **Function not found**: Error during agent startup
- **Runtime error**: Caught and returned as error message
- **Type mismatch**: Type checking during agent startup

### MCP Tool Errors

- **Server unavailable**: Error during agent startup (fails fast)
- **Command not found**: Error if runtime (npx, uvx, docker) not installed
- **Connection timeout**: Configurable via `request_timeout`, returns error
- **Invalid config**: Error during agent startup (validation)
- **Runtime errors**: Returned as tool error responses to the LLM

### Prompt Tool Errors

- **Invalid template**: Error during agent startup
- **LLM failure**: Soft failure (logged, error message returned)
- **Template rendering**: Error during execution

---

## Performance Tips

### Vectorstore Tools

- Use appropriate chunk size (larger = fewer embeddings)
- Enable caching for remote files
- Reduce `vector_field` count if possible
- Index only necessary fields

### Function Tools

- Keep functions fast (<1 second)
- Use connection pooling for databases
- Cache results when possible

### MCP Tools

- Use server-side filtering when available
- Limit result sets
- Cache responses locally

### Prompt Tools

- Use simpler models for repeated operations
- Batch processing when possible
- Limit template complexity

---

## Best Practices

1. **Clear Names**: Use descriptive tool names
2. **Clear Descriptions**: Agent uses description to decide when to call tool
3. **Parameters**: Define expected parameters clearly
4. **Error Handling**: Handle errors gracefully
5. **Performance**: Test with realistic data
6. **Versioning**: Manage tool file versions in source control
7. **Testing**: Include test cases that exercise each tool

---

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for tool usage
- See [File References Guide](file-references.md) for path resolution
- See [Examples](../examples/) for complete tool usage
