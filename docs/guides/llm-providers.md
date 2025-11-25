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

Alternative to temperature for controlling randomness. Use one or the other, not both:

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

### Model Names in Azure

In Azure OpenAI, the `name` field refers to your **deployment name**, not the base model:

```yaml
model:
  provider: azure_openai
  name: my-gpt4o-deployment  # Your deployment name in Azure
```

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
```

### Create Example Files

Commit a template for other developers:

```bash
# .env.example - Safe to commit
OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
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

---

## Next Steps

- See [Agent Configuration](agent-configuration.md) for complete agent setup
- See [Global Configuration](global-config.md) for shared settings
- See [Evaluations Guide](evaluations.md) for testing agent quality
- See [Tools Guide](tools.md) for extending agent capabilities
