# Tools Reference Guide

This guide explains HoloDeck's four tool types that extend agent capabilities.

## Overview

Tools are agent capabilities defined in `agent.yaml`. HoloDeck supports four tool types:

1. **Vectorstore Tools** - Semantic search over data
2. **Function Tools** - Custom Python functions
3. **MCP Tools** - Model Context Protocol servers
4. **Prompt Tools** - LLM-powered semantic functions

## Common Tool Fields

All tools share these fields:

```yaml
tools:
  - name: tool-id              # Required: Tool identifier (unique)
    description: What it does  # Required: Human-readable description
    type: vectorstore|function|mcp|prompt  # Required: Tool type
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

## Vectorstore Tools

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

## Function Tools

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

## MCP Tools

Model Context Protocol server integrations.

### When to Use

- GitHub, GitLab operations
- Database access
- File system operations
- Any standardized MCP server

### Basic Example

```yaml
- name: file-system
  description: Access file system
  type: mcp
  server: "@modelcontextprotocol/server-filesystem"
```

### Required Fields

#### Server

- **Type**: String
- **Purpose**: MCP server identifier
- **Formats**:
  - Package name: `@modelcontextprotocol/server-filesystem`
  - Local path: `/path/to/server`
  - Custom identifier: `my-custom-server`

```yaml
server: "@modelcontextprotocol/server-filesystem"
```

### Optional Fields

#### Config

- **Type**: Object (free-form)
- **Purpose**: MCP server-specific configuration
- **Validation**: MCP server validates at runtime

```yaml
config:
  allowed_directories: ["/data", "/tmp"]
  max_file_size: 1048576  # 1MB
```

### Available MCP Servers

#### Filesystem

```yaml
- name: filesystem
  type: mcp
  server: "@modelcontextprotocol/server-filesystem"
  config:
    allowed_directories: ["/home/user/data"]
```

Resources: List files, read files, write files, create directories

#### GitHub

```yaml
- name: github
  type: mcp
  server: "@modelcontextprotocol/server-github"
  config:
    access_token: "${GITHUB_TOKEN}"
    repository: "user/repo"
```

Resources: List issues, create issues, read files, etc.

#### SQLite

```yaml
- name: sqlite
  type: mcp
  server: "@modelcontextprotocol/server-sqlite"
  config:
    database: "/path/to/database.db"
```

Resources: Query database, list tables, etc.

### Custom MCP Servers

To create a custom MCP server:

1. Implement MCP protocol
2. Deploy as standalone service or Python package
3. Reference by path or package identifier

```yaml
- name: custom-service
  type: mcp
  server: "/path/to/custom_server.py"
  config:
    api_url: "http://localhost:8000"
```

---

## Prompt Tools

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
  name: gpt-4  # Different from agent's model
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

| Feature | Vectorstore | Function | MCP | Prompt |
|---------|-------------|----------|-----|--------|
| **Use Case** | Search data | Custom logic | Integrations | Template-based |
| **Execution** | Vector similarity | Python function | MCP protocol | LLM generation |
| **Setup** | Data files | Python files | Server config | Template text |
| **Parameters** | Implicit (search query) | Defined in code | Server-specific | Defined in YAML |
| **Latency** | Medium (~100ms) | Low (<10ms) | Medium | High (LLM call) |
| **Cost** | Embedding API | Internal | Service cost | LLM tokens |

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

### File Operations

```yaml
- name: filesystem
  type: mcp
  server: "@modelcontextprotocol/server-filesystem"
  config:
    allowed_directories: ["/data"]
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

- **Server unavailable**: Soft failure (logged, empty results)
- **Permission denied**: Soft failure (logged)
- **Invalid config**: Error during agent startup

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
