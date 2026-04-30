# Semantic Kernel Backend

The Semantic Kernel (SK) backend powers HoloDeck agents that run against OpenAI, Azure OpenAI, or local Ollama models. It is automatically selected for any `model.provider` other than `anthropic`.

This guide is for backend-specific behaviour: per-provider setup, deployment-name mechanics, local-model nuances, and the full configuration reference. For shared concepts like tools, observability, and vector stores, see the dedicated guides for each.

## Quick start

OpenAI is the most common entry point. Set an API key and point an agent at it:

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
OPENAI_API_KEY=sk-...
```

Verify:

```bash
holodeck chat agent.yaml
```

For Azure OpenAI or Ollama, see the per-provider sections below.

## Advanced configuration

The SK backend doesn't expose a separate top-level config block — all backend behaviour is driven by the `model` settings of the active provider plus the provider's own infrastructure.

### Provider: OpenAI

OpenAI provides GPT-4o, GPT-4o-mini, and other models through their hosted API.

**Prerequisites:**

1. Create an account at [platform.openai.com](https://platform.openai.com).
2. Generate an API key in the [API Keys section](https://platform.openai.com/api-keys).
3. Set up billing.

**Configuration:**

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

**Environment variables:**

```bash
OPENAI_API_KEY=sk-...
```

**Available models:**

| Model           | Description                       | Context window |
|-----------------|-----------------------------------|----------------|
| `gpt-4o`        | Most capable, multimodal          | 128K tokens    |
| `gpt-4o-mini`   | Fast and cost-effective           | 128K tokens    |
| `gpt-4-turbo`   | Previous-generation flagship      | 128K tokens    |
| `gpt-3.5-turbo` | Fast, lower cost                  | 16K tokens     |

### Provider: Azure OpenAI

Azure OpenAI Service provides OpenAI models via Microsoft Azure with enterprise features.

**Prerequisites:**

1. Azure subscription with Azure OpenAI access.
2. Create an Azure OpenAI resource in the [Azure Portal](https://portal.azure.com).
3. Deploy a model in Azure OpenAI Studio.
4. Note the endpoint URL and API key.

**Configuration:**

Both `endpoint` and `api_key` are required:

```yaml
# config.yaml
providers:
  azure_openai:
    provider: azure_openai
    name: my-gpt4o-deployment    # see "Deployment names" below
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_API_KEY}
    temperature: 0.3
    max_tokens: 2000
```

```yaml
# agent.yaml
name: enterprise-agent

model:
  provider: azure_openai
  name: my-gpt4o-deployment
  endpoint: https://my-resource.openai.azure.com/

instructions:
  inline: "You are an enterprise assistant."
```

**Environment variables:**

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

**Endpoint format:**

```
https://{resource-name}.openai.azure.com/
```

Find your endpoint in: Azure Portal → Your OpenAI Resource → Keys and Endpoint, or Azure OpenAI Studio → Deployments → Your Deployment.

#### Deployment names

!!! warning "`name` refers to the deployment, not the base model"
    In Azure OpenAI, the `name` field must match your **deployment name**, not the base model. This is different from OpenAI's API.

When you deploy a model in Azure OpenAI Studio, you create a deployment with a custom name:

- **Base model**: `gpt-4o`, `gpt-4o-mini`, etc.
- **Deployment name**: your custom identifier (e.g. `my-gpt4o`, `prod-gpt4`).

```yaml
# If your deployment is "my-gpt4o-production" backed by gpt-4o:
model:
  provider: azure_openai
  name: my-gpt4o-production
  endpoint: https://my-resource.openai.azure.com/
```

Common mistake:

```yaml
# WRONG — using the base model name
model:
  provider: azure_openai
  name: gpt-4o    # only works if the deployment is literally named "gpt-4o"

# CORRECT — using your deployment name
model:
  provider: azure_openai
  name: my-gpt4o-deployment
```

### Provider: Ollama

Ollama runs open-source LLMs locally — ideal for privacy-sensitive workloads, offline use, and avoiding API costs.

**Prerequisites:**

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

**Configuration:**

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

**Environment variables:**

```bash
OLLAMA_ENDPOINT=http://localhost:11434
```

**Available models** (pull with `ollama pull <model>`):

| Model            | Command                                     | Description              | Size   |
|------------------|---------------------------------------------|--------------------------|--------|
| GPT-OSS (20B)    | `ollama pull gpt-oss:20b`                   | Recommended completion   | 40 GB  |
| Nomic Embed Text | `ollama pull nomic-embed-text:latest`       | Recommended embeddings   | 274 MB |
| Llama 3.2        | `ollama pull llama3.2`                      | Meta's latest, general   | 2 GB   |
| Llama 3.2 (3B)   | `ollama pull llama3.2:3b`                   | Larger Llama variant     | 5 GB   |
| Mistral          | `ollama pull mistral`                       | Fast and capable         | 4 GB   |
| CodeLlama        | `ollama pull codellama`                     | Optimised for code       | 4 GB   |
| Phi-3            | `ollama pull phi3`                          | Microsoft compact model  | 2 GB   |
| Gemma 2          | `ollama pull gemma2`                        | Google's open model      | 5 GB   |

#### Running Ollama as a service

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

#### Context size

For agent workloads, configure a context size of at least **16k tokens**. Default Ollama context windows can be small enough to clip tool outputs.

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
    A 32k context with a 20B-parameter model typically requires 48 GB+ RAM or a GPU with 16 GB+ VRAM.

### Backend behaviour vs Claude

The SK backend is automatically selected when `model.provider` is anything other than `anthropic`. There is no top-level `claude:` block — capabilities like permission modes, extended thinking, web search, and subagents are Claude-only and are documented in the [Claude Backend](claude-backend.md) guide.

## Configuration reference

### `model.*` fields per provider

All providers share these fields:

| Field         | Type    | Required | Default | Description                                |
|---------------|---------|----------|---------|--------------------------------------------|
| `provider`    | string  | yes      | -       | `openai`, `azure_openai`, or `ollama`      |
| `name`        | string  | yes      | -       | Model name (deployment name for Azure)     |
| `temperature` | float   | no       | `0.3`   | Randomness, `0.0`–`2.0`                    |
| `max_tokens`  | integer | no       | `1000`  | Maximum response tokens                    |
| `top_p`       | float   | no       | -       | Nucleus sampling, `0.0`–`1.0`              |
| `api_key`     | string  | varies   | -       | API key (required for OpenAI / Azure)      |
| `endpoint`    | string  | varies   | -       | Endpoint URL (required for Azure / Ollama) |

Per-provider requirements:

| Provider       | `api_key` | `endpoint` | Notes                                                      |
|----------------|-----------|------------|------------------------------------------------------------|
| `openai`       | required  | -          | -                                                          |
| `azure_openai` | required  | required   | `name` must match the deployment name, not the base model  |
| `ollama`       | -         | required   | Default endpoint `http://localhost:11434`                  |

### Provider-specific environment variables

| Variable                  | Provider     | Description                                          |
|---------------------------|--------------|------------------------------------------------------|
| `OPENAI_API_KEY`          | OpenAI       | API authentication key                               |
| `AZURE_OPENAI_ENDPOINT`   | Azure OpenAI | Resource endpoint URL                                |
| `AZURE_OPENAI_API_KEY`    | Azure OpenAI | API authentication key                               |
| `OLLAMA_ENDPOINT`         | Ollama       | Server endpoint (default `http://localhost:11434`)   |

## Limitations & roadmap

The SK backend currently powers `holodeck serve` and `holodeck deploy build`; the Claude backend does not yet support those commands.

- [Agent Server](serve.md): SK-only.
- [Deployment](deployment.md): SK-only (containerisation).

Both will gain Claude support in a future release.

## Troubleshooting

### Invalid API key

**Error**: `AuthenticationError` or `Invalid API key`.

1. Verify the key: `echo $OPENAI_API_KEY` (or the Azure / Ollama equivalent).
2. Ensure no extra whitespace.
3. Regenerate the API key if needed.

### Azure endpoint missing

**Error**: `endpoint is required for azure_openai provider`.

```yaml
model:
  provider: azure_openai
  name: my-deployment
  endpoint: https://my-resource.openai.azure.com/
```

### Model / deployment not found

- **OpenAI**: check the model identifier (e.g. `gpt-4o`, not `gpt4o`).
- **Azure**: ensure `name` matches the deployment name exactly.
- **Ollama**: pull first with `ollama pull <model>`.

### Ollama: connection refused

1. Verify the daemon: `ollama list`.
2. Start the server: `ollama serve`.
3. Confirm the `endpoint` URL.

### Ollama: model not found

```bash
ollama pull llama3.2
```

### Rate limits

**Error**: `Rate limit exceeded` (OpenAI / Azure).

1. Implement retry with exponential backoff.
2. Reduce `max_tokens`.
3. Use a faster / cheaper model.
4. Upgrade the API plan.

### Temperature out of range

```yaml
model:
  temperature: 0.7   # must be 0.0 – 2.0
```

## Next steps

- [Agent Configuration](agent-configuration.md) — full agent.yaml structure
- [Tools](tools.md) — extending agent capabilities
- [Observability](observability.md) — tracing and metrics
- [Vector Stores](vector-stores.md) — semantic search configuration
- [Agent Server](serve.md) and [Deployment](deployment.md) — running SK agents as services
