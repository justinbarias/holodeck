# Agent Configuration Guide

This guide explains how to define AI agents using HoloDeck's `agent.yaml` configuration file.

## Overview

An agent configuration file defines a single AI agent with everything it needs:

- **Model settings**: Which LLM provider to use (OpenAI, Azure, Anthropic)
- **Instructions**: System prompt (from a file or inline)
- **Tools**: Capabilities the agent can use (search, functions, APIs, templates)
- **Evaluations**: Metrics to measure quality
- **Test cases**: Scenarios to validate the agent

All configuration is declarative—no code required.

## Basic Structure

```yaml
name: my-agent                    # Required: Agent name
description: Agent description    # Optional: What this agent does
author: "Your Name"               # Optional: Who created this agent

model:                            # Required: LLM configuration
  provider: anthropic             # Required: openai|azure_openai|anthropic|ollama
  name: claude-sonnet-4-20250514  # Required: Model identifier
  auth_provider: oauth_token      # Optional: Uses CLAUDE_CODE_OAUTH_TOKEN env var
  temperature: 0.7                # Optional: 0.0-2.0
  max_tokens: 2000                # Optional: Maximum generation tokens

instructions:                     # Required: System prompt
  inline: "You are a helpful..."  # Option 1: Inline text
  # OR
  # file: instructions.txt        # Option 2: External file

tools: []                         # Optional: Agent capabilities
evaluations:                      # Optional: Quality metrics
  metrics: []
test_cases: []                    # Optional: Test scenarios
```

> **Backend auto-selection**: HoloDeck automatically selects the execution backend based on the `provider` field. `anthropic` routes to the Claude Agent SDK backend; all other providers route to the Semantic Kernel backend.

## Agent Name

- **Required**: Yes
- **Type**: String
- **Format**: 1-100 characters, alphanumeric + hyphens, must start with letter
- **Examples**: `customer-support`, `code-reviewer`, `data-analyzer`

```yaml
name: customer-support
```

## Agent Description

- **Required**: No
- **Type**: String
- **Constraints**: Max 500 characters
- **Purpose**: Describe what this agent does for documentation

```yaml
description: Handles customer support queries with ticket creation
```

## Agent Author

- **Required**: No
- **Type**: String
- **Constraints**: Max 256 characters
- **Purpose**: Document who created or maintains this agent

```yaml
author: "Alice Johnson"
```

This field is useful for:
- Attribution and credit in multi-team environments
- Understanding who to contact for questions about the agent
- Tracking agent ownership and maintenance responsibility

## Model Configuration

Defines which LLM provider and model to use.

### Provider Field

- **Required**: Yes
- **Type**: String (Enum)
- **Options**:
  - `openai` - OpenAI API (GPT-4o, GPT-4o-mini, etc.)
  - `azure_openai` - Azure OpenAI Service
  - `anthropic` - Anthropic Claude (native Claude Agent SDK backend)
  - `ollama` - Local models via Ollama

```yaml
model:
  provider: anthropic
```

### Model Name

- **Required**: Yes
- **Type**: String
- **Purpose**: Identifies specific model within provider
- **Examples by Provider**:
  - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
  - Azure: `gpt-4`, `gpt-4-32k`
  - Anthropic: `claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `claude-3-5-haiku-20241022`
  - Ollama: `llama3.2`, `mistral`, `codellama`

```yaml
model:
  name: gpt-4o
```

### Temperature

- **Optional**: Yes
- **Type**: Float
- **Range**: 0.0 to 2.0
- **Default**: 0.7 (if not specified)
- **Meaning**:
  - `0.0` - Deterministic, focused responses
  - `0.7` - Balanced randomness
  - `1.5+` - Very creative, random responses

```yaml
model:
  temperature: 0.8  # More creative
```

### Max Tokens

- **Optional**: Yes
- **Type**: Integer
- **Constraint**: Must be positive
- **Purpose**: Limit maximum length of generated responses

```yaml
model:
  max_tokens: 4000
```

### Top P

- **Optional**: Yes
- **Type**: Float
- **Range**: 0.0 to 1.0
- **Purpose**: Nucleus sampling (alternative to temperature)

```yaml
model:
  top_p: 0.9
```

### Authentication Provider

- **Optional**: Yes (defaults to `api_key`)
- **Type**: String (Enum)
- **Applies to**: `anthropic` provider only (ignored for other providers)
- **Purpose**: Select the authentication method for Claude Agent SDK

| Method | Env Variable | Use Case |
|--------|-------------|----------|
| `api_key` | `ANTHROPIC_API_KEY` | Direct API access (default) |
| `oauth_token` | `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code OAuth (recommended) |
| `bedrock` | AWS credentials (`AWS_ACCESS_KEY_ID`, etc.) | AWS Bedrock |
| `vertex` | GCP credentials (`GOOGLE_APPLICATION_CREDENTIALS`) | Google Vertex AI |
| `foundry` | Foundry credentials | Foundry platform |

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: oauth_token  # Uses CLAUDE_CODE_OAUTH_TOKEN env var
```

## Embedding Provider

When using `model.provider: anthropic` with **vectorstore** or **hierarchical_document** tools, you **must** define an `embedding_provider` at the agent level. This is because Anthropic does not provide embedding models — a separate provider is needed to generate embeddings for semantic search.

- **Required**: When `provider: anthropic` AND using vectorstore/hierarchical_document tools
- **Type**: LLM provider configuration object

```yaml
# Using Ollama for embeddings (local, free)
embedding_provider:
  provider: ollama
  name: nomic-embed-text:latest
  endpoint: http://localhost:11434

# Or using OpenAI for embeddings
embedding_provider:
  provider: openai
  name: text-embedding-3-small
```

> **Note**: `embedding_provider` is ignored when using `openai`, `azure_openai`, or `ollama` as the main provider, since those providers can generate embeddings natively.

## Claude Agent SDK Settings

When using `model.provider: anthropic`, you can configure Claude Agent SDK-specific capabilities via the `claude` section. All capabilities default to disabled (least-privilege).

### Permission Mode

| Value | Description |
|-------|-------------|
| `manual` | Requires manual approval for all actions (default, safest) |
| `acceptEdits` | Auto-approve file edits, manual approval for other actions |
| `acceptAll` | Auto-approve all tool calls and actions |

### Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permission_mode` | Enum | `manual` | Level of autonomous action |
| `working_directory` | String | None | Scope file access to this path; subprocess cwd |
| `max_turns` | Integer (>=1) | None (SDK default) | Maximum agent loop iterations |
| `extended_thinking` | Object | None | Extended reasoning configuration |
| `web_search` | Boolean | `false` | Enable built-in web search |
| `bash` | Object | None | Shell command execution settings |
| `file_system` | Object | None | File read/write/edit access |
| `subagents` | Object | None | Parallel sub-agent execution |
| `allowed_tools` | List[String] | None (all tools) | Explicit tool allowlist |

### Extended Thinking

Enable deep reasoning for complex tasks:

```yaml
claude:
  extended_thinking:
    enabled: true
    budget_tokens: 10000  # 1,000 - 100,000
```

### Bash Access

```yaml
claude:
  bash:
    enabled: true
    excluded_commands: ["rm", "shutdown"]  # Commands to block
    allow_unsafe: false                    # Dangerous commands require explicit opt-in
```

### File System Access

```yaml
claude:
  file_system:
    read: true
    write: true
    edit: true
```

### Sub-agents

```yaml
claude:
  subagents:
    enabled: true
    max_parallel: 4  # 1-16 parallel sub-agents
```

### Full Annotated Example

```yaml
claude:
  permission_mode: acceptAll       # Auto-approve actions
  working_directory: ./workspace   # Restrict file access
  max_turns: 15                    # Limit agent loop iterations

  extended_thinking:
    enabled: true
    budget_tokens: 20000           # Deep reasoning token budget

  web_search: true                 # Enable web search

  bash:
    enabled: true
    excluded_commands: ["rm -rf", "shutdown"]
    allow_unsafe: false

  file_system:
    read: true
    write: true
    edit: true

  subagents:
    enabled: true
    max_parallel: 8

  allowed_tools:                   # Only expose these tools to the agent
    - knowledge-base
    - web-search
```

> **Note**: The `claude` section is ignored when using non-Anthropic providers.

## Instructions

Defines the system prompt that guides agent behavior.

### Inline Instructions

Embed the prompt directly in `agent.yaml`:

```yaml
instructions:
  inline: |
    You are a customer support specialist.

    Guidelines:
    - Be polite and professional
    - Provide accurate information
    - Escalate complex issues to supervisors
```

### File-Based Instructions

Reference an external file (path relative to `agent.yaml`):

```yaml
instructions:
  file: system_prompt.txt
```

File at `system_prompt.txt`:

```
You are a customer support specialist.

Guidelines:
- Be polite and professional
- Provide accurate information
- Escalate complex issues to supervisors
```

### Rules

- **Exactly one required**: Either `inline` OR `file`, not both
- **Max length** (inline): 5000 characters
- **File path**: Relative to `agent.yaml` directory (see File References guide)

## Response Format

Defines the expected structure of the agent's responses (agent-level only).

The `response_format` field constrains the LLM to generate structured output following a JSON Schema. This is useful for:

- Ensuring consistent response structure
- Integrating with downstream systems that expect specific formats
- Validating response quality programmatically
- Guiding the LLM toward well-formatted outputs

### Inline Response Format

Define the schema directly in `agent.yaml`:

```yaml
response_format:
  type: object
  properties:
    answer:
      type: string
      description: The answer to the user's question
    confidence:
      type: number
      description: Confidence score 0.0-1.0
    sources:
      type: array
      items:
        type: string
      description: References used in the answer
  required:
    - answer
    - confidence
  additionalProperties: false
```

### File-Based Response Format

Reference an external JSON Schema file (path relative to `agent.yaml`):

```yaml
response_format: schemas/response.json
```

File at `schemas/response.json`:

```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "The answer to the user's question"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score 0.0-1.0"
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "References used in the answer"
    }
  },
  "required": ["answer", "confidence"],
  "additionalProperties": false
}
```

### Supported JSON Schema Keywords

Response format uses a **Basic JSON Schema** subset supporting:

- `type` - Data type (string, number, integer, boolean, object, array, null)
- `properties` - Object properties and their schemas
- `required` - Array of required property names
- `items` - Schema for array items
- `enum` - List of allowed values
- `description` - Documentation string
- `minimum` - Minimum numeric value
- `maximum` - Maximum numeric value
- `additionalProperties` - Whether to allow extra properties (true|false)

### Unsupported Keywords

The following keywords are **not supported** and will cause validation errors:

- `$ref` - Schema references
- `anyOf` - Multiple schema options
- `oneOf` - Exactly one schema match
- `allOf` - All schemas must match
- `patternProperties` - Regex-based properties
- `minLength`, `maxLength` - String length constraints
- `minItems`, `maxItems` - Array length constraints
- And other JSON Schema draft keywords

### Rules

- **Optional**: Response format is not required
- **Agent-level only**: Not inherited from global configuration
- **Validated at load time**: Invalid schemas cause errors with clear messages
- **File path**: Relative to `agent.yaml` directory

### Validation

Invalid response formats will produce clear error messages:

```
Error: Invalid JSON in response_format field
File: agent.yaml
Line: 42
Details: Missing required property: 'answer'
```

```
Error: Unknown JSON Schema keyword: '$ref'
File: schemas/response.json
Details: Keyword '$ref' is not supported. Use basic JSON Schema keywords only.
```

## Tools

Define capabilities the agent can use. See the [Tools Reference Guide](tools.md) for detailed documentation.

```yaml
tools:
  - name: search-docs
    description: Search company documentation
    type: vectorstore
    source: docs/

  - name: get-user
    description: Retrieve user information
    type: function
    file: tools/user_tools.py
    function: get_user

  - name: file-system
    description: File system access
    type: mcp
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    config:
      allowed_directories: ["/data", "/tmp"]

  - name: summarize
    description: Summarize text
    type: prompt
    template: "Summarize this in 2-3 sentences: {{text}}"
    parameters:
      text:
        type: string
        description: Text to summarize
```

### Tool Constraints

- **Max tools**: 50 per agent
- **Tool names**: Must be unique, alphanumeric + underscores
- **Required fields**: `name`, `description`, `type`

## Evaluations

Defines metrics to measure agent quality. See the [Evaluations Guide](evaluations.md) for details.

```yaml
evaluations:
  model:  # Optional: default model for all metrics
    provider: openai
    name: gpt-4o

  metrics:
    - metric: groundedness
      threshold: 0.8
      enabled: true

    - metric: relevance
      threshold: 0.75
```

## Test Cases

Defines scenarios to validate agent behavior.

```yaml
test_cases:
  - name: "Support request"
    input: "How do I reset my password?"
    expected_tools: [search-docs]
    ground_truth: "Instructions for password reset"

  - input: "What are your hours?"
    expected_tools: []
```

### Test Case Fields

- **name**: Test identifier (optional)
- **input**: User query (required, max 5000 chars)
- **expected_tools**: Tools that should be called (optional)
- **ground_truth**: Expected response for comparison (optional)
- **files**: Multimodal inputs like images, PDFs (optional, max 10 per test)

### Constraints

- **Max test cases**: 100 per agent
- **Test names**: Must be unique if provided

## Complete Example

```yaml
name: support-agent
description: Handles customer support queries with knowledge base search

model:
  provider: azure_openai
  name: gpt-4o
  temperature: 0.7
  max_tokens: 2000

instructions:
  file: system_prompt.txt

response_format:
  type: object
  properties:
    answer:
      type: string
      description: The support response
    confidence:
      type: number
      description: How confident we are in this answer
    escalation_needed:
      type: boolean
      description: Whether the issue should be escalated
  required:
    - answer
    - confidence

tools:
  - name: search-kb
    description: Search knowledge base
    type: vectorstore
    source: knowledge_base.json
    chunk_size: 500

  - name: create-ticket
    description: Create support ticket
    type: function
    file: tools/support.py
    function: create_ticket
    parameters:
      title:
        type: string
        description: Ticket title
      priority:
        type: string
        description: low|medium|high

evaluations:
  model:
    provider: azure_openai
    name: gpt-4o
    temperature: 0.0

  metrics:
    - metric: f1_score
      threshold: 0.8
    - metric: bleu
      threshold: 0.75

test_cases:
  - name: "Password reset"
    input: "How do I reset my password?"
    expected_tools: [search-kb]
    ground_truth: "Step-by-step password reset instructions"
    evaluations:
      - f1_score
      - bleu

  - name: "Open ticket"
    input: "I need help with my account"
    expected_tools: [search-kb, create-ticket]
    ground_truth: "Ticket created and knowledge base searched"
    evaluations:
      - f1_score
```

## Validation Rules

### Required Fields

- `name`: Must be provided
- `model.provider`: Must be provided and valid
- `model.name`: Must be provided
- `instructions`: Must have either `inline` or `file`

### Mutual Exclusivity

- Instructions: Either `inline` OR `file`, not both
- Prompt tools: Either `template` OR `file`, not both

### Ranges

- Temperature: 0.0 to 2.0
- Max tokens: Must be > 0
- Tool limit: Max 50 per agent
- Test cases: Max 100 per agent

### Response Format Validation

- `response_format`: Optional
- Must be valid JSON Schema (if inline) or valid JSON file (if external)
- Only Basic JSON Schema keywords supported
- Invalid schemas produce clear error messages with file location and details
- Not inherited from global configuration (agent-level only)

### File References

- Paths are relative to `agent.yaml` directory
- Files must exist (checked during loading)
- Absolute paths are supported

## Environment Variables

Replace sensitive values with environment variables:

```yaml
model:
  provider: openai
  name: gpt-4o
  # API key from environment, see global config guide
```

See the [Global Configuration Guide](global-config.md) for environment variable interpolation details.

## Common Patterns

### Minimal Agent (Inline Instructions)

```yaml
name: simple-agent

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: "You are a helpful assistant."
```

### Agent with Structured Output (Response Format)

```yaml
name: structured-agent
description: Returns structured responses

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: "Answer questions in the specified format."

response_format:
  type: object
  properties:
    answer:
      type: string
    confidence:
      type: number
      minimum: 0
      maximum: 1
  required:
    - answer
    - confidence
```

### Agent with File References

```yaml
name: documented-agent

model:
  provider: azure_openai
  name: gpt-4

instructions:
  file: prompts/system.txt

response_format: schemas/output.json

tools:
  - name: search
    type: vectorstore
    source: data/docs.json
```

### Full-Featured Agent

```yaml
name: enterprise-agent
description: Production-ready support agent

model:
  provider: azure_openai
  name: gpt-4o
  temperature: 0.6
  max_tokens: 4096

instructions:
  file: system_prompt.txt

response_format:
  type: object
  properties:
    response:
      type: string
    sources:
      type: array
      items:
        type: string
    confidence:
      type: number
      minimum: 0
      maximum: 1
  required:
    - response

tools:
  - name: knowledge-base
    type: vectorstore
    source: kb/
  - name: system-check
    type: function
    file: tools/system.py
    function: check_status
  - name: external-api
    type: mcp
    command: npx
    args: ["-y", "custom-mcp-server"]

evaluations:
  model:
    provider: azure_openai
    name: gpt-4o
    temperature: 0.0

  metrics:
    - metric: f1_score
      threshold: 0.85
    - metric: bleu
      threshold: 0.8

test_cases:
  - name: "Basic query"
    input: "Hello, can you help?"
    expected_tools: [knowledge-base]
    ground_truth: "Yes, I'm here to help with your support request"
    evaluations:
      - f1_score
      - bleu
```

## Troubleshooting

### Error: "name is required"

- Add `name` field at top level

### Error: "instructions: Either file or inline must be provided"

- Ensure instructions section has either `inline` or `file` field

### Error: "instruction file not found"

- Check file path is correct and relative to `agent.yaml` directory
- Use absolute paths if needed

### Error: "Invalid model provider"

- Use valid provider: `openai`, `azure_openai`, `anthropic`, or `ollama`

### Error: "Tool name must be unique"

- Each tool must have a unique `name` field

### Error: "Invalid JSON in response_format field"

- Ensure response_format is valid JSON (if inline) or references a valid JSON file
- Check for missing quotes, commas, or brackets in JSON

### Error: "response_format file not found"

- Check the file path is correct and relative to `agent.yaml` directory
- File must exist and be readable

### Error: "Unknown JSON Schema keyword: '$ref'"

- Remove unsupported JSON Schema keywords ($ref, anyOf, oneOf, allOf, patternProperties, etc.)
- Use only Basic JSON Schema keywords: type, properties, required, items, enum, description, minimum, maximum, additionalProperties

## Next Steps

- See [Tools Reference](tools.md) for tool configuration details
- See [Evaluations Guide](evaluations.md) for quality metrics
- See [Global Configuration](global-config.md) for shared settings
- See [File References](file-references.md) for path resolution
