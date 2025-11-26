# LLM Providers Guide

This guide explains how to configure LLM providers in HoloDeck for your AI agents.

## Overview

HoloDeck supports multiple LLM providers, allowing you to choose the best model for your use case. Provider configuration can be defined at two levels:

- **Global Configuration** (`config.yaml`): Shared settings and API credentials
- **Agent Configuration** (`agent.yaml`): Per-agent model selection and overrides

### Supported Providers

| Provider | Description | API Key Required |
|----------|-------------|------------------|
| `openai` | OpenAI API (GPT-4o, GPT-4o-mini, etc.) | Yes |
| `azure_openai` | Azure OpenAI Service | Yes + Endpoint |
| `anthropic` | Anthropic Claude models | Yes |
| `ollama` | Local models via Ollama | No (Endpoint required) |

---

## Quick Start

### Minimal Agent Configuration

```yaml
# agent.yaml
name: my-agent

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: "You are a helpful assistant."
```

### With Global Configuration

```yaml
# config.yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}
```

```yaml
# agent.yaml
name: my-agent

model:
  provider: openai
  # Inherits name, api_key from config.yaml

instructions:
  inline: "You are a helpful assistant."
```

---

## Configuration Fields

All providers share these common fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | string | Yes | - | Provider identifier |
| `name` | string | Yes | - | Model name/identifier |
| `temperature` | float | No | 0.3 | Randomness (0.0-2.0) |
| `max_tokens` | integer | No | 1000 | Maximum response tokens |
| `top_p` | float | No | - | Nucleus sampling (0.0-1.0) |
| `api_key` | string | No | - | API authentication key |
| `endpoint` | string | Varies | - | API endpoint URL |

### Temperature

Controls response randomness:

- **0.0**: Deterministic, focused responses
- **0.3**: Default, balanced
- **0.7**: More creative
- **1.0+**: Highly creative/random

```yaml
model:
  temperature: 0.5  # Moderately creative
```

### Max Tokens

Limits response length. Set based on your use case:

```yaml
model:
  max_tokens: 2000  # Allow longer responses
```

### Top P (Nucleus Sampling)

Alternative to temperature for controlling randomness. While both can be used simultaneously, it's recommended to adjust one or the other for more predictable results:

```yaml
model:
  top_p: 0.9  # Consider top 90% probability tokens
```

---

## OpenAI

OpenAI provides GPT-4o, GPT-4o-mini, and other models through their API.

### Prerequisites

1. Create an account at [platform.openai.com](https://platform.openai.com)
2. Generate an API key in the [API Keys section](https://platform.openai.com/api-keys)
3. Set up billing in your account

### Configuration

**Global Configuration (Recommended):**

```yaml
# config.yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    temperature: 0.3
    max_tokens: 2000
    api_key: ${OPENAI_API_KEY}
```

**Agent Configuration:**

```yaml
# agent.yaml
name: my-agent

model:
  provider: openai
  name: gpt-4o
  temperature: 0.7
  max_tokens: 4000

instructions:
  inline: "You are a helpful assistant."
```

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
```

### Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `gpt-4o` | Most capable, multimodal | 128K tokens |
| `gpt-4o-mini` | Fast and cost-effective | 128K tokens |
| `gpt-4-turbo` | Previous generation flagship | 128K tokens |
| `gpt-3.5-turbo` | Fast, lower cost | 16K tokens |

### Complete Example

```yaml
# config.yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}

  openai-fast:
    provider: openai
    name: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
```

```yaml
# agent.yaml
name: support-agent
description: Customer support with GPT-4o

model:
  provider: openai
  name: gpt-4o
  temperature: 0.5
  max_tokens: 2000

instructions:
  inline: |
    You are a customer support specialist.
    Be helpful, accurate, and professional.
```

---

## Azure OpenAI

Azure OpenAI Service provides OpenAI models through Microsoft Azure with enterprise features.

### Prerequisites

1. Azure subscription with Azure OpenAI access
2. Create an Azure OpenAI resource in the [Azure Portal](https://portal.azure.com)
3. Deploy a model in Azure OpenAI Studio
4. Note your endpoint URL and API key

### Configuration

Azure OpenAI requires both an `endpoint` and `api_key`:

**Global Configuration (Recommended):**

```yaml
# config.yaml
providers:
  azure_openai:
    provider: azure_openai
    name: gpt-4o
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}
    temperature: 0.3
    max_tokens: 2000
```

**Agent Configuration:**

```yaml
# agent.yaml
name: enterprise-agent

model:
  provider: azure_openai
  name: gpt-4o  # Must match your Azure deployment name
  endpoint: https://my-resource.openai.azure.com/
  temperature: 0.5

instructions:
  inline: "You are an enterprise assistant."
```

### Environment Variables

```bash
# .env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

### Endpoint Format

The endpoint URL follows this pattern:

```
https://{resource-name}.openai.azure.com/
```

Find your endpoint in:

1. Azure Portal > Your OpenAI Resource > Keys and Endpoint
2. Azure OpenAI Studio > Deployments > Your Deployment

### Understanding Azure Deployment Names

> **Important**: In Azure OpenAI, the `name` field refers to your **deployment name**, not the base model name. This is different from OpenAI's API.

When you deploy a model in Azure OpenAI Studio, you create a deployment with a custom name:

1. **Base Model**: The underlying model (e.g., `gpt-4o`, `gpt-4o-mini`)
2. **Deployment Name**: Your custom identifier (e.g., `my-gpt4o`, `prod-gpt4`)

The `name` field in HoloDeck must match your **deployment name**:

```yaml
# If your Azure deployment is named "my-gpt4o-production"
# backed by the gpt-4o base model:

model:
  provider: azure_openai
  name: my-gpt4o-production  # Must match deployment name exactly
  endpoint: https://my-resource.openai.azure.com/
```

**Common Mistake:**

```yaml
# WRONG - Using base model name
model:
  provider: azure_openai
  name: gpt-4o  # This won't work unless your deployment is literally named "gpt-4o"

# CORRECT - Using your deployment name
model:
  provider: azure_openai
  name: my-gpt4o-deployment  # Your actual deployment name
```

> **Tip for OpenAI Users**: If you're transitioning from OpenAI to Azure, remember that Azure adds this extra layer of indirection. Your deployment name can be anything, but it must be specified exactly in the configuration.

### Available Models

Azure OpenAI offers the same models as OpenAI, deployed to your resource:

| Model | Azure Deployment | Description |
|-------|------------------|-------------|
| GPT-4o | Deploy in Azure | Most capable |
| GPT-4o-mini | Deploy in Azure | Cost-effective |
| GPT-4 | Deploy in Azure | Previous flagship |
| GPT-3.5-Turbo | Deploy in Azure | Fast, lower cost |

### Complete Example

```yaml
# config.yaml
providers:
  azure_openai:
    provider: azure_openai
    name: gpt-4o-deployment
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}
    temperature: 0.3
    max_tokens: 2000
```

```yaml
# agent.yaml
name: enterprise-support
description: Enterprise support agent on Azure

model:
  provider: azure_openai
  name: gpt-4o-deployment
  temperature: 0.5
  max_tokens: 4000

instructions:
  file: prompts/enterprise-support.txt

evaluations:
  model:
    provider: azure_openai
    name: gpt-4o-deployment
  metrics:
    - metric: f1_score
      threshold: 0.8
```

---

## Anthropic

Anthropic provides the Claude family of models known for safety and helpfulness.

### Prerequisites

1. Create an account at [console.anthropic.com](https://console.anthropic.com)
2. Generate an API key in the Console
3. Set up billing

### Configuration

**Global Configuration (Recommended):**

```yaml
# config.yaml
providers:
  anthropic:
    provider: anthropic
    name: claude-sonnet-4-20250514
    temperature: 0.3
    max_tokens: 4000
    api_key: ${ANTHROPIC_API_KEY}
```

**Agent Configuration:**

```yaml
# agent.yaml
name: claude-agent

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  temperature: 0.5
  max_tokens: 4000

instructions:
  inline: "You are Claude, a helpful AI assistant."
```

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `claude-sonnet-4-20250514` | Best balance of speed and capability | 200K tokens |
| `claude-opus-4-20250514` | Most capable, best for complex tasks | 200K tokens |
| `claude-3-5-sonnet-20241022` | Previous generation Sonnet | 200K tokens |
| `claude-3-5-haiku-20241022` | Fast and cost-effective | 200K tokens |

> **Note**: Model identifiers include version dates (e.g., `20250514`). Check [Anthropic's documentation](https://docs.anthropic.com/en/docs/about-claude/models) for the latest available models and their capabilities.

### Complete Example

```yaml
# config.yaml
providers:
  anthropic:
    provider: anthropic
    name: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  anthropic-fast:
    provider: anthropic
    name: claude-3-5-haiku-20241022
    api_key: ${ANTHROPIC_API_KEY}
```

```yaml
# agent.yaml
name: research-assistant
description: Research assistant powered by Claude

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  temperature: 0.3
  max_tokens: 8000

instructions:
  inline: |
    You are a research assistant.
    Provide thorough, well-sourced answers.
    Be accurate and cite relevant information.
```

---

## Ollama

Ollama enables running open-source LLMs locally on your machine. This is ideal for privacy-sensitive applications, offline deployments, and avoiding API costs.

### Benefits of Ollama

- **Privacy**: Data never leaves your machine
- **No API Costs**: Run unlimited queries without usage fees
- **Offline Support**: Works without internet connection
- **Open-Source Models**: Access to Llama, Mistral, CodeLlama, and more

### Prerequisites

1. Install Ollama from [ollama.com](https://ollama.com)

   **macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   **Windows:**
   Download from [ollama.com/download](https://ollama.com/download)

2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```

3. Verify Ollama is running:
   ```bash
   ollama list
   ```

### Configuration

Ollama requires an `endpoint` pointing to your local Ollama server:

**Global Configuration (Recommended):**

```yaml
# config.yaml
providers:
  ollama:
    provider: ollama
    name: llama3.2
    endpoint: http://localhost:11434
    temperature: 0.7
    max_tokens: 2000
```

**Agent Configuration:**

```yaml
# agent.yaml
name: local-agent

model:
  provider: ollama
  name: llama3.2
  endpoint: http://localhost:11434
  temperature: 0.5

instructions:
  inline: "You are a helpful local assistant."
```

### Environment Variables

```bash
# .env (optional - for custom endpoint)
OLLAMA_ENDPOINT=http://localhost:11434
```

### Available Models

Pull models with `ollama pull <model-name>`:

| Model | Command | Description | Size |
|-------|---------|-------------|------|
| GPT-OSS (20B) | `ollama pull gpt-oss:20b` | Recommended completion model | 40GB |
| Nomic Embed Text | `ollama pull nomic-embed-text:latest` | Recommended embedding model | 274MB |
| Llama 3.2 | `ollama pull llama3.2` | Meta's latest, general purpose | 2GB |
| Llama 3.2 (3B) | `ollama pull llama3.2:3b` | Larger Llama variant | 5GB |
| Mistral | `ollama pull mistral` | Fast and capable | 4GB |
| CodeLlama | `ollama pull codellama` | Optimized for code | 4GB |
| Phi-3 | `ollama pull phi3` | Microsoft's compact model | 2GB |
| Gemma 2 | `ollama pull gemma2` | Google's open model | 5GB |

> **Tip**: Run `ollama list` to see your installed models.

### Running Ollama as a Service

For production use, run Ollama as a background service:

**Start Ollama server:**
```bash
ollama serve
```

**Or with Docker:**
```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama
```

### Context Size Configuration

For agent workloads, we recommend configuring a context size of at least **16k tokens**. By default, Ollama models may use smaller context windows which can limit agent capabilities.

**Create a custom model with extended context:**

```bash
# Create a Modelfile with extended context
cat <<EOF > Modelfile
FROM gpt-oss:20b
PARAMETER num_ctx 16384
EOF

# Create the custom model
ollama create gpt-oss:20b-16k -f Modelfile
```

**For larger context needs (32k):**

```bash
cat <<EOF > Modelfile
FROM gpt-oss:20b
PARAMETER num_ctx 32768
EOF

ollama create gpt-oss:20b-32k -f Modelfile
```

**Use the custom model in your configuration:**

```yaml
model:
  provider: ollama
  name: gpt-oss:20b-16k  # or gpt-oss:20b-32k for larger context
  endpoint: http://localhost:11434
```

> **Note**: Larger context sizes require more memory. A 32k context with a 20B parameter model may require 48GB+ RAM or a GPU with 16GB+ VRAM.

### Complete Example

```yaml
# config.yaml
providers:
  ollama:
    provider: ollama
    name: llama3.2
    endpoint: ${OLLAMA_ENDPOINT}
    temperature: 0.7

  ollama-code:
    provider: ollama
    name: codellama
    endpoint: ${OLLAMA_ENDPOINT}
    temperature: 0.2
```

```yaml
# agent.yaml
name: local-assistant
description: Privacy-focused local assistant

model:
  provider: ollama
  name: llama3.2
  temperature: 0.5
  max_tokens: 4000

instructions:
  inline: |
    You are a helpful assistant running locally.
    All data stays on this machine for privacy.
```

### Troubleshooting Ollama

**Error:** `endpoint is required for ollama provider`

**Solution:** Always include the endpoint:
```yaml
model:
  provider: ollama
  name: llama3.2
  endpoint: http://localhost:11434
```

**Error:** `Connection refused`

**Solutions:**
1. Verify Ollama is running: `ollama list`
2. Start the server: `ollama serve`
3. Check the endpoint URL matches your setup

**Error:** `Model not found`

**Solution:** Pull the model first:
```bash
ollama pull llama3.2
```

---

## Multi-Provider Setup

Configure multiple providers to use different models for different purposes:

```yaml
# config.yaml
providers:
  # Primary provider for agents
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0.3

  # Fast provider for evaluations
  openai-fast:
    provider: openai
    name: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
    temperature: 0.0

  # Enterprise provider
  azure:
    provider: azure_openai
    name: gpt-4o-deployment
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}

  # Alternative provider
  anthropic:
    provider: anthropic
    name: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  # Local provider (no API costs, privacy-focused)
  ollama:
    provider: ollama
    name: llama3.2
    endpoint: ${OLLAMA_ENDPOINT}
```

Use different providers in your agent:

```yaml
# agent.yaml
name: multi-model-agent

model:
  provider: openai
  name: gpt-4o

evaluations:
  model:
    provider: openai
    name: gpt-4o-mini  # Use faster model for evaluations
  metrics:
    - metric: f1_score
      threshold: 0.8
```

---

## Security Best Practices

### Never Commit API Keys

```yaml
# WRONG - Never do this
providers:
  openai:
    api_key: sk-abc123...  # Exposed secret!

# CORRECT - Use environment variables
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

### Use .env Files

Create a `.env` file (add to `.gitignore`):

```bash
# .env - DO NOT COMMIT
OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_ENDPOINT=http://localhost:11434
```

### Create Example Files

Commit a template for other developers:

```bash
# .env.example - Safe to commit
OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OLLAMA_ENDPOINT=http://localhost:11434
```

---

## Troubleshooting

### Invalid API Key

**Error:** `AuthenticationError` or `Invalid API key`

**Solutions:**

1. Verify your API key is correct
2. Check environment variable is set: `echo $OPENAI_API_KEY`
3. Ensure no extra whitespace in the key
4. Regenerate the API key if needed

### Azure Endpoint Issues

**Error:** `endpoint is required for azure_openai provider`

**Solution:** Include the endpoint in your configuration:

```yaml
model:
  provider: azure_openai
  name: my-deployment
  endpoint: https://my-resource.openai.azure.com/
```

### Model Not Found

**Error:** `Model not found` or `Deployment not found`

**Solutions:**

- **OpenAI**: Check the model name is valid (e.g., `gpt-4o`, not `gpt4o`)
- **Azure**: Ensure `name` matches your deployment name exactly
- **Anthropic**: Use full model identifier (e.g., `claude-sonnet-4-20250514`)

### Rate Limits

**Error:** `Rate limit exceeded`

**Solutions:**

1. Implement retry logic with exponential backoff
2. Reduce `max_tokens` to use fewer tokens
3. Use a faster/cheaper model for testing
4. Upgrade your API plan

### Temperature Out of Range

**Error:** `temperature must be between 0.0 and 2.0`

**Solution:** Use a value between 0.0 and 2.0:

```yaml
model:
  temperature: 0.7  # Valid
```

---

## Environment Variable Reference

| Variable | Provider | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | API authentication key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Resource endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | API authentication key |
| `ANTHROPIC_API_KEY` | Anthropic | API authentication key |
| `OLLAMA_ENDPOINT` | Ollama | Server endpoint (default: `http://localhost:11434`) |

---

## Next Steps

- See [Agent Configuration](agent-configuration.md) for complete agent setup
- See [Global Configuration](global-config.md) for shared provider settings and credentials
- See [Evaluations Guide](evaluations.md) for configuring evaluation models (consider using faster models like `gpt-4o-mini` for cost-effective evaluations)
- See [Tools Guide](tools.md) for extending agent capabilities
- See [Vector Stores Guide](vector-stores.md) for semantic search configuration
