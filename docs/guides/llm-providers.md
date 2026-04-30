# LLM Providers

HoloDeck supports multiple LLM providers across two execution backends. Provider configuration can be defined at two levels:

- **Global Configuration** (`config.yaml`): shared settings and credentials.
- **Agent Configuration** (`agent.yaml`): per-agent model selection and overrides.

This page is an index. For setup, advanced features, and full configuration reference, follow the link to the matching backend.

## Supported providers

| Provider       | Backend           | Description                                            |
|----------------|-------------------|--------------------------------------------------------|
| `anthropic`    | Claude Agent SDK  | Anthropic's Claude family (first-class)                |
| `openai`       | Semantic Kernel   | OpenAI API (GPT-4o, GPT-4o-mini, etc.)                 |
| `azure_openai` | Semantic Kernel   | Azure OpenAI Service (enterprise-deployed models)      |
| `ollama`       | Semantic Kernel   | Local models via Ollama (Llama, Mistral, etc.)         |

> **Backend auto-selection**: The `model.provider` field selects the backend automatically. `anthropic` routes to the Claude Agent SDK backend; everything else routes to the Semantic Kernel backend.

## Choose a backend

- **[Claude Backend](claude-backend.md)** — `provider: anthropic`. Quick start, authentication providers (api_key / oauth_token / bedrock / vertex / foundry), and Claude-specific capabilities like permission modes, extended thinking, web search, subagents, bash and file_system scoping.
- **[Semantic Kernel Backend](semantic-kernel-backend.md)** — `provider: openai`, `azure_openai`, or `ollama`. Per-provider quick starts, the Azure deployment-name nuance, and Ollama context-size tuning.

## Shared configuration fields

All providers share these common fields:

| Field         | Type    | Required | Default | Description              |
|---------------|---------|----------|---------|--------------------------|
| `provider`    | string  | yes      | -       | Provider identifier      |
| `name`        | string  | yes      | -       | Model name / identifier  |
| `temperature` | float   | no       | `0.3`   | Randomness (`0.0`–`2.0`) |
| `max_tokens`  | integer | no       | `1000`  | Maximum response tokens  |
| `top_p`       | float   | no       | -       | Nucleus sampling         |
| `api_key`     | string  | varies   | -       | API authentication key   |
| `endpoint`    | string  | varies   | -       | API endpoint URL         |

### Temperature

Controls response randomness:

- `0.0` — deterministic, focused.
- `0.3` — default, balanced.
- `0.7` — more creative.
- `1.0+` — highly creative / random.

```yaml
model:
  temperature: 0.5
```

### Max tokens

```yaml
model:
  max_tokens: 2000
```

### Top P (nucleus sampling)

```yaml
model:
  top_p: 0.9
```

Use either `temperature` or `top_p`, not both, for predictable behaviour.

## Multi-provider setup

Configure multiple providers at the global level so different agents (or different evaluation models) can pick from the same set:

```yaml
# config.yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0.3

  openai-fast:
    provider: openai
    name: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
    temperature: 0.0

  azure:
    provider: azure_openai
    name: gpt-4o-deployment
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}

  anthropic:
    provider: anthropic
    name: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  ollama:
    provider: ollama
    name: llama3.2
    endpoint: ${OLLAMA_ENDPOINT}
```

Use them in an agent:

```yaml
# agent.yaml
name: multi-model-agent

model:
  provider: openai
  name: gpt-4o

evaluations:
  model:
    provider: openai
    name: gpt-4o-mini   # cheaper / faster eval model
  metrics:
    - metric: f1_score
      threshold: 0.8
```

## Security best practices

### Never commit API keys

```yaml
# WRONG
providers:
  openai:
    api_key: sk-abc123...

# CORRECT
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

### Use `.env` files

Add `.env` to `.gitignore`:

```bash
# .env — DO NOT COMMIT
OPENAI_API_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_CODE_OAUTH_TOKEN=...
OLLAMA_ENDPOINT=http://localhost:11434
```

Commit a template (`.env.example`) for collaborators:

```bash
# .env.example — Safe to commit
OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OLLAMA_ENDPOINT=http://localhost:11434
```

## Next steps

- [Claude Backend](claude-backend.md) — Claude-specific setup and capabilities.
- [Semantic Kernel Backend](semantic-kernel-backend.md) — OpenAI / Azure / Ollama setup.
- [Agent Configuration](agent-configuration.md) — full `agent.yaml` structure.
- [Global Configuration](global-config.md) — shared provider settings and credentials.
- [Evaluations](evaluations.md) — configuring evaluation models.
- [Tools](tools.md) — extending agent capabilities.
- [Vector Stores](vector-stores.md) — semantic search configuration.
