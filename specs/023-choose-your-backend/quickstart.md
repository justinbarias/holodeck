# Quickstart: Choose Your Backend

**Feature**: 023-choose-your-backend | **Date**: 2026-03-15

## Prerequisites

Install the backend you want to use:

```bash
# Google ADK backend
pip install holodeck[google-adk]

# Microsoft Agent Framework backend (OpenAI provider)
pip install holodeck[agent-framework]

# Agent Framework with Anthropic sub-provider
pip install holodeck[agent-framework-anthropic]

# Agent Framework with Ollama sub-provider
pip install holodeck[agent-framework-ollama]
```

## Usage Examples

### Google ADK with Gemini

```yaml
# agent.yaml
name: gemini-agent
model:
  provider: google_adk
  name: gemini-2.5-flash
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

### Google ADK with OpenAI via LiteLLM

```yaml
name: adk-openai-agent
model:
  provider: google_adk
  name: openai/gpt-4o
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

### Google ADK with Streaming

```yaml
name: streaming-adk-agent
model:
  provider: google_adk
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
  provider: google_adk
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
  provider: agent_framework
  name: gpt-4o
  temperature: 0.7
instructions:
  inline: "You are a helpful assistant."
```

### Agent Framework with Azure OpenAI

```yaml
name: af-azure-agent
model:
  provider: agent_framework
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
  provider: agent_framework
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

### Agent Framework with Native Embeddings

```yaml
name: af-rag-agent
model:
  provider: agent_framework
  name: gpt-4o
instructions:
  inline: "Search the knowledge base to answer questions."
tools:
  - name: knowledge_base
    type: vectorstore
    source: ./data/docs/
    description: "Company knowledge base"
agent_framework:
  use_native_embeddings: true   # Uses AF's OpenAIEmbeddingClient
```

### Agent Framework with Explicit Sub-Provider

```yaml
name: af-ollama-agent
model:
  provider: agent_framework
  name: llama3.1:70b
instructions:
  inline: "You are a helpful assistant."
agent_framework:
  sub_provider: ollama
```

## Switching Backends

Change only `model.provider` and `model.name` — all tools continue to work:

```yaml
# Before: Semantic Kernel with OpenAI
model:
  provider: openai
  name: gpt-4o

# After: Google ADK with Gemini (tools unchanged)
model:
  provider: google_adk
  name: gemini-2.5-flash
embedding_provider:          # Add if vectorstore tools are configured
  provider: openai
  name: text-embedding-3-small

# After: Agent Framework with GPT-4o (tools unchanged)
model:
  provider: agent_framework
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

If model name doesn't match any known AF sub-provider:
```
ValidationError: Cannot auto-detect sub-provider for model 'custom-model'.
Specify 'sub_provider' in the agent_framework config section.
Supported prefixes: gpt-*, o1*, o3*, o4* (OpenAI), claude-* (Anthropic),
llama-*, mistral-*, phi-* (Ollama)
```
