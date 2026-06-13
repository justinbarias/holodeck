# LLM Providers

HoloDeck supports multiple LLM providers across two execution backends. The
`model.provider` field selects the backend automatically — you never configure
it directly. Provider configuration can be defined at two levels:

- **Global Configuration** (`config.yaml`): shared settings and credentials.
- **Agent Configuration** (`agent.yaml`): per-agent model selection and overrides.

## Quick start

Pick a provider, set its credentials, and run `holodeck chat`. OpenAI is the
most common entry point:

```yaml
# agent.yaml
name: my-agent

model:
  provider: openai
  name: gpt-4o
  temperature: 0.5
  max_tokens: 2000

instructions:
  inline: "You are a helpful assistant."
```

```bash
# .env
OPENAI_API_KEY=${OPENAI_API_KEY}
```

```bash
holodeck chat agent.yaml
```

Swap `provider`/`name` (and the relevant credentials) to target Anthropic,
Azure OpenAI, or a local Ollama model — see [Supported providers](#supported-providers)
below.

## Supported providers

| Provider       | Backend               | Description                                       |
|----------------|-----------------------|---------------------------------------------------|
| `anthropic`    | Claude backend        | Anthropic's Claude family (first-class)           |
| `openai`       | OpenAI Agents backend | OpenAI API (GPT-4o, GPT-4o-mini, etc.)            |
| `azure_openai` | OpenAI Agents backend | Azure OpenAI Service (enterprise-deployed models) |
| `ollama`       | Claude backend        | Local models via Ollama (Llama, Mistral, etc.)    |

> **Backend auto-selection**: `anthropic` and `ollama` route to the Claude
> backend; `openai` and `azure_openai` route to the OpenAI Agents backend.
> Both backends support `holodeck chat` and `holodeck test`. `holodeck serve`
> and `holodeck deploy` are supported on the **Claude backend** today; on the
> OpenAI Agents backend they are on the roadmap.

## Choose a backend

- **[Claude Backend](claude-backend.md)** — `provider: anthropic` (and local
  `ollama` models). Authentication providers (api_key / oauth_token / bedrock /
  vertex / foundry) and Claude-specific capabilities like permission modes,
  extended thinking, web search, and subagents.
- **[OpenAI Backend](openai-backend.md)** — `provider: openai` or
  `azure_openai`. Per-provider quick starts and the Azure deployment-name
  nuance.

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

### Azure deployment names

!!! warning "`name` refers to the deployment, not the base model"
    For `provider: azure_openai`, the `name` field must match your **deployment
    name**, not the base model. This differs from OpenAI's API, where `name` is
    the model identifier (e.g. `gpt-4o`).

```yaml
# If your deployment is "my-gpt4o-production" backed by gpt-4o:
model:
  provider: azure_openai
  name: my-gpt4o-production
  endpoint: https://my-resource.openai.azure.com/
```

See the [OpenAI Backend](openai-backend.md) guide for the full Azure setup.

## Multi-provider setup

Configure multiple providers at the global level so different agents (or
different evaluation models) can pick from the same set:

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

## Local models with Ollama

Ollama runs open-source LLMs locally — ideal for privacy-sensitive workloads,
offline use, and avoiding API costs. The `ollama` provider routes through the
**Claude backend**, so Claude-backend capabilities and the
[Claude Backend](claude-backend.md) guide apply.

### Prerequisites

1. Install Ollama from [ollama.com](https://ollama.com):

   ```bash
   # macOS / Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull a model:

   ```bash
   ollama pull llama3.2
   ```

3. Verify it's running:

   ```bash
   ollama list
   ```

### Configuration

Ollama requires an `endpoint` pointing to your local server:

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

```yaml
# agent.yaml
name: local-agent

model:
  provider: ollama
  name: llama3.2
  endpoint: http://localhost:11434

instructions:
  inline: "You are a helpful local assistant."
```

```bash
# .env
OLLAMA_ENDPOINT=http://localhost:11434
```

### Available models

Pull with `ollama pull <model>`:

| Model            | Command                                | Description             | Size   |
|------------------|----------------------------------------|-------------------------|--------|
| GPT-OSS (20B)    | `ollama pull gpt-oss:20b`              | Recommended completion  | 40 GB  |
| Nomic Embed Text | `ollama pull nomic-embed-text:latest`  | Recommended embeddings  | 274 MB |
| Llama 3.2        | `ollama pull llama3.2`                 | Meta's latest, general  | 2 GB   |
| Llama 3.2 (3B)   | `ollama pull llama3.2:3b`              | Larger Llama variant    | 5 GB   |
| Mistral          | `ollama pull mistral`                  | Fast and capable        | 4 GB   |
| CodeLlama        | `ollama pull codellama`                | Optimised for code      | 4 GB   |
| Phi-3            | `ollama pull phi3`                     | Microsoft compact model | 2 GB   |
| Gemma 2          | `ollama pull gemma2`                   | Google's open model     | 5 GB   |

### Running Ollama as a service

```bash
ollama serve
```

Or with Docker:

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama
```

### Context size

For agent workloads, configure a context size of at least **16k tokens**.
Default Ollama context windows can be small enough to clip tool outputs.

Create a custom model with extended context:

```bash
cat <<EOF > Modelfile
FROM gpt-oss:20b
PARAMETER num_ctx 16384
EOF

ollama create gpt-oss:20b-16k -f Modelfile
```

For 32k:

```bash
cat <<EOF > Modelfile
FROM gpt-oss:20b
PARAMETER num_ctx 32768
EOF

ollama create gpt-oss:20b-32k -f Modelfile
```

Use it:

```yaml
model:
  provider: ollama
  name: gpt-oss:20b-16k
  endpoint: http://localhost:11434
```

!!! note "Memory pressure scales with context"
    A 32k context with a 20B-parameter model typically requires 48 GB+ RAM or a
    GPU with 16 GB+ VRAM.

### Ollama troubleshooting

**Connection refused**

1. Verify the daemon: `ollama list`.
2. Start the server: `ollama serve`.
3. Confirm the `endpoint` URL.

**Model not found**

```bash
ollama pull llama3.2
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

- [Claude Backend](claude-backend.md) — Claude / Ollama setup and capabilities.
- [OpenAI Backend](openai-backend.md) — OpenAI / Azure OpenAI setup.
- [Agent Configuration](agent-configuration.md) — full `agent.yaml` structure.
- [Global Configuration](global-config.md) — shared provider settings and credentials.
- [Evaluations](evaluations.md) — configuring evaluation models.
- [Tools](tools.md) — extending agent capabilities.
- [Vector Stores](vector-stores.md) — semantic search configuration.
