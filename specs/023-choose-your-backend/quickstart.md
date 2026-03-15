# Quickstart: Choose Your Backend

**Feature**: 023-choose-your-backend | **Date**: 2026-03-15

## Prerequisites

Install the backend you want to use:

```bash
# Google ADK backend
pip install holodeck[google-adk]

# Microsoft Agent Framework backend (OpenAI provider)
pip install holodeck[agent-framework]

# Agent Framework with Anthropic provider support
pip install holodeck[agent-framework-anthropic]

# Agent Framework with Ollama provider support
pip install holodeck[agent-framework-ollama]
```

## Usage Examples

### Google ADK with Gemini

```yaml
# agent.yaml
name: gemini-agent
model:
  provider: google
  name: gemini-2.5-flash
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

### Google ADK with OpenAI (Cross-Provider)

```yaml
name: adk-openai-agent
backend: google_adk
model:
  provider: openai
  name: gpt-4o
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

> **Cross-provider selection**: Setting `backend: google_adk` with `provider: openai` runs the ADK runtime with an OpenAI model. ADK uses LiteLLM internally for non-Google models.

### Google ADK with Streaming

```yaml
name: streaming-adk-agent
model:
  provider: google
  name: gemini-2.5-flash
instructions:
  inline: "You are a helpful assistant."
google_adk:
  streaming_mode: sse
  max_iterations: 10
```

### Google ADK with Vectorstore Tools

```yaml
name: adk-rag-agent
model:
  provider: google
  name: gemini-2.5-flash
embedding_provider:          # Required for google_adk with vectorstore tools
  provider: openai
  name: text-embedding-3-small
instructions:
  inline: "Search the knowledge base to answer questions."
tools:
  - name: knowledge_base
    type: vectorstore
    source: ./data/docs/
    description: "Company knowledge base"
```

### Agent Framework with OpenAI

```yaml
name: af-openai-agent
model:
  provider: openai
  name: gpt-4o
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

### Agent Framework with Azure OpenAI

```yaml
name: af-azure-agent
model:
  provider: azure_openai
  name: gpt-4o
  endpoint: https://myorg.openai.azure.com/
  api_version: "2024-12-01-preview"
instructions:
  inline: "You are a helpful assistant."
```

### Agent Framework with MCP Tools

```yaml
name: af-mcp-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Use available tools to help the user."
tools:
  - name: filesystem
    type: mcp
    transport: stdio
    command_type: npx
    command: "@modelcontextprotocol/server-filesystem"
    args: ["./workspace"]
```

### Agent Framework with Vectorstore Tools

```yaml
name: af-rag-agent
model:
  provider: openai
  name: gpt-4o
embedding_provider:              # Required for agent_framework with vectorstore tools
  provider: openai
  name: text-embedding-3-small
instructions:
  inline: "Search the knowledge base to answer questions."
tools:
  - name: knowledge_base
    type: vectorstore
    source: ./data/docs/
    description: "Company knowledge base"
```

### Agent Framework with Anthropic (Cross-Provider)

```yaml
name: af-anthropic-agent
backend: agent_framework
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "You are a helpful assistant."
```

### Using Skills (Inline)

Skills replace the old prompt tool type. They run as sub-agent invocations on the same backend:

```yaml
name: support-agent
model:
  provider: google
  name: gemini-2.5-flash
embedding_provider:              # Required for google_adk with vectorstore tools
  provider: openai
  name: text-embedding-3-small
instructions:
  inline: "You are a customer support agent. Use skills for specialized tasks."
tools:
  - name: knowledge_base
    type: vectorstore
    source: ./data/docs/
    description: "Company knowledge base"
  - name: sentiment-analyzer
    type: skill
    description: "Analyze customer sentiment from their message"
    instructions: "Analyze the given text for sentiment. Return overall sentiment (positive/negative/neutral), confidence score, and key emotions detected."
  - name: escalation-checker
    type: skill
    description: "Determine if a support case needs escalation"
    instructions: "Review the conversation and determine if escalation is needed based on sentiment, topic complexity, and customer frustration level."
    allowed_tools: [knowledge_base]  # can search the KB to check policies
```

### Using Skills (File-Based)

For complex skills with scripts, references, or assets, point to a skill directory:

```yaml
tools:
  - name: research-assistant
    type: skill
    path: ./skills/research-assistant/
    allowed_tools: [knowledge_base, web_search]
```

The skill directory follows the [Agent Skills spec](https://agentskills.io/specification):
```
skills/research-assistant/
├── SKILL.md              # Required: frontmatter + instructions
├── references/           # Optional: detailed reference docs
│   └── search-strategies.md
└── assets/               # Optional: templates, schemas
    └── report-template.md
```

## Switching Backends

Use the `backend` field to run the same model on different runtimes:

```yaml
# Default: auto-detected backend (openai → agent_framework)
model:
  provider: openai
  name: gpt-4o

# Explicit: same model, Semantic Kernel backend
backend: semantic_kernel
model:
  provider: openai
  name: gpt-4o

# Explicit: same model, Google ADK backend (cross-provider)
backend: google_adk
model:
  provider: openai
  name: gpt-4o
```

## Running

```bash
# Test execution
holodeck test agent.yaml

# Interactive chat
holodeck chat agent.yaml

# Verbose mode for debugging
holodeck test agent.yaml -v
```

## Error Handling

If a backend package is not installed:
```
BackendInitError: Google ADK backend requires the 'google-adk' package.
Install it with: pip install holodeck[google-adk]
```

If backend and provider are incompatible:
```
ValidationError: Backend 'claude' is not compatible with provider 'openai'.
The 'claude' backend only supports the 'anthropic' provider.
```
