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

model:                            # Required: LLM configuration
  provider: openai                # Required: openai|azure_openai|anthropic
  name: gpt-4o                    # Required: Model identifier
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

## Model Configuration

Defines which LLM provider and model to use.

### Provider Field

- **Required**: Yes
- **Type**: String (Enum)
- **Options**:
  - `openai` - OpenAI API (GPT-4o, GPT-4o-mini, etc.)
  - `azure_openai` - Azure OpenAI Service
  - `anthropic` - Anthropic Claude

```yaml
model:
  provider: openai
```

### Model Name

- **Required**: Yes
- **Type**: String
- **Purpose**: Identifies specific model within provider
- **Examples by Provider**:
  - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
  - Azure: `gpt-4`, `gpt-4-32k`
  - Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

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
    server: "@modelcontextprotocol/server-filesystem"
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
  provider: openai
  name: gpt-4o
  temperature: 0.7
  max_tokens: 2000

instructions:
  file: system_prompt.txt

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
  metrics:
    - metric: groundedness
      threshold: 0.8

test_cases:
  - name: "Password reset"
    input: "How do I reset my password?"
    expected_tools: [search-kb]
    ground_truth: "Step-by-step password reset instructions"

  - name: "Open ticket"
    input: "I need help with my account"
    expected_tools: [search-kb, create-ticket]
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

### Agent with File References

```yaml
name: documented-agent

model:
  provider: azure_openai
  name: gpt-4

instructions:
  file: prompts/system.txt

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
  provider: openai
  name: gpt-4o
  temperature: 0.6
  max_tokens: 4096

instructions:
  file: system_prompt.txt

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
    server: "custom-server"

evaluations:
  model:
    provider: openai
    name: gpt-4o
  metrics:
    - metric: groundedness
      threshold: 0.85
    - metric: safety
      threshold: 0.9

test_cases:
  - name: "Basic query"
    input: "Hello, can you help?"
    expected_tools: [knowledge-base]
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

- Use valid provider: `openai`, `azure_openai`, or `anthropic`

### Error: "Tool name must be unique"

- Each tool must have a unique `name` field

## Next Steps

- See [Tools Reference](tools.md) for tool configuration details
- See [Evaluations Guide](evaluations.md) for quality metrics
- See [Global Configuration](global-config.md) for shared settings
- See [File References](file-references.md) for path resolution
