# Claude Backend

The Claude backend runs your agent on top of the [Claude Agent SDK](https://docs.anthropic.com/en/api/agent-sdk) — Anthropic's first-class agent runtime. It is automatically selected when `model.provider: anthropic` is set in your agent configuration.

This guide is for backend-specific behaviour: authentication, Claude Agent SDK capabilities (permission modes, extended thinking, web search, subagents, etc.), and the full configuration reference. For shared concepts like tools, observability, and vector stores, see the dedicated guides for each.

## Quick start

### Prerequisites

- **Node.js 18+** — required by the Claude Agent SDK subprocess. Verify with `node --version`.
- An Anthropic credential. The simplest is a Claude Code OAuth token (recommended for Claude Code users).

### Minimal agent

```yaml
# agent.yaml
name: my-claude-agent

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: oauth_token

instructions:
  inline: "You are a helpful assistant."
```

```bash
# .env
CLAUDE_CODE_OAUTH_TOKEN=your-oauth-token
```

Verify:

```bash
holodeck chat agent.yaml
```

## Advanced configuration

All Claude-specific settings live under the top-level `claude:` block in `agent.yaml`. Every capability defaults to disabled (least-privilege).

### Authentication providers

The `auth_provider` field on `model` selects how HoloDeck authenticates with Anthropic. Defaults to `api_key` when omitted.

| Method         | Env variables                                                              | Use case                                |
|----------------|----------------------------------------------------------------------------|-----------------------------------------|
| `api_key`      | `ANTHROPIC_API_KEY`                                                        | Direct API access (default)             |
| `oauth_token`  | `CLAUDE_CODE_OAUTH_TOKEN`                                                  | Claude Code OAuth (recommended)         |
| `bedrock`      | `AWS_REGION` (or `AWS_DEFAULT_REGION`) plus AWS credentials                | AWS Bedrock                             |
| `vertex`       | `CLOUD_ML_REGION` plus `ANTHROPIC_VERTEX_PROJECT_ID` and Google credentials | Google Vertex AI                       |
| `foundry`      | `ANTHROPIC_FOUNDRY_RESOURCE` or `ANTHROPIC_FOUNDRY_BASE_URL`               | Azure AI Foundry                        |

**API key (default):**

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: api_key
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**OAuth token (recommended for Claude Code users):**

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: oauth_token
```

```bash
export CLAUDE_CODE_OAUTH_TOKEN="your-oauth-token"
```

**AWS Bedrock:**

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: bedrock
```

```bash
export AWS_REGION=us-east-1
# plus AWS credentials/profile (env vars, ~/.aws/credentials, or IAM role)
```

**Google Vertex AI:**

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: vertex
```

```bash
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=your-gcp-project-id
# plus Google credentials (GOOGLE_APPLICATION_CREDENTIALS or ADC)
```

**Azure AI Foundry:**

```yaml
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: foundry
```

```bash
export ANTHROPIC_FOUNDRY_RESOURCE=your-foundry-resource
# or:
export ANTHROPIC_FOUNDRY_BASE_URL=https://your-resource.services.ai.azure.com
```

Authentication is provided either via `ANTHROPIC_API_KEY` or the Azure credential chain.

!!! note "Cloud routing is env-driven"
    For `bedrock`, `vertex`, and `foundry`, HoloDeck sets the Claude Code subprocess flag (`CLAUDE_CODE_USE_BEDROCK=1`, `CLAUDE_CODE_USE_VERTEX=1`, `CLAUDE_CODE_USE_FOUNDRY=1`) based on `auth_provider`. `model.endpoint` is **not** consulted for these modes. Missing routing variables fail fast at startup with a configuration error.

### Permission modes

Controls how autonomously the agent can act:

| Value          | Behaviour                                                          |
|----------------|--------------------------------------------------------------------|
| `manual`       | Manual approval for every action (default, safest)                 |
| `acceptEdits`  | Auto-approve file edits; manual approval for other tool calls      |
| `acceptAll`    | Auto-approve all tool calls and actions                            |

```yaml
claude:
  permission_mode: acceptEdits
```

### Extended thinking

Enable deep, internal reasoning before the agent responds:

```yaml
claude:
  extended_thinking:
    enabled: true
    budget_tokens: 10000   # 1,000 - 100,000
```

`budget_tokens` caps how many tokens the agent can spend on hidden reasoning per turn.

### Web search

Built-in web search capability backed by the SDK:

```yaml
claude:
  web_search: true
```

### Bash & file system scoping

Both are off by default. Enable explicitly to grant access:

```yaml
claude:
  bash:
    enabled: true
    excluded_commands: ["rm -rf", "shutdown"]
    allow_unsafe: false              # dangerous commands require explicit opt-in

  file_system:
    read: true
    write: true
    edit: true
```

`working_directory` further scopes file access to a specific path (passed as the subprocess `cwd`):

```yaml
claude:
  working_directory: ./workspace
```

### Subagents

Define named subagents (a multi-agent team) under `claude.agents`. Each entry becomes an `AgentDefinition` registered with the SDK and is invocable by the parent via the `Task` tool.

```yaml
claude:
  agents:
    researcher:
      description: Gathers raw information from the web and primary sources.
      prompt: |
        You are a meticulous researcher. Search the web for primary sources.
        Cite every claim with a URL. Do not analyze — just gather facts.
      tools: [WebSearch, WebFetch]
      model: haiku
    analyst:
      description: Analyses data, identifies patterns, surfaces key insights.
      prompt: |
        You are a data analyst. Study the researcher's findings, identify
        patterns and outliers, and summarise the key insights clearly.
      tools: [Read]
      model: sonnet
    writer:
      description: Produces a polished final report from analyst insights.
      prompt: |
        You are an expert technical writer. Use headers, bullets, and inline
        citations. Do not conduct research yourself.
      # No tools list → inherits all parent tools.
      # No model → inherits parent model.
```

Per-subagent fields:

| Field         | Type                                            | Required | Description                                                       |
|---------------|-------------------------------------------------|----------|-------------------------------------------------------------------|
| `description` | string                                          | yes      | Human-readable description used by the parent for routing         |
| `prompt`      | string                                          | one of   | Inline system prompt for the subagent                             |
| `prompt_file` | string                                          | one of   | Path to a file containing the prompt (mutually exclusive with `prompt`) |
| `tools`       | list of strings                                 | no       | Allowlist of tool names; omit to inherit all parent tools         |
| `model`       | `sonnet` \| `opus` \| `haiku` \| `inherit`      | no       | Model alias for this subagent; omit to inherit the parent's model |

!!! warning "Subagent `model` must be a SDK alias"
    Only `sonnet`, `opus`, `haiku`, or `inherit` are valid. Full model IDs like `claude-sonnet-4-20250514` are silently rejected by the Claude CLI — invalid values fail at config-load time with a clear error. Use `inherit` to track the parent's pinned full ID for version stability.

### Working directory & max turns

```yaml
claude:
  working_directory: ./workspace   # subprocess cwd; restricts file access
  max_turns: 10                    # cap on agent loop iterations
```

### Allowed tools

When set, only the listed tools are visible to the agent (in addition to any defined under the agent's top-level `tools:`):

```yaml
claude:
  allowed_tools:
    - knowledge-base
    - web-search
```

### Embedding provider requirement

Anthropic does not provide embedding models. When you combine `model.provider: anthropic` with **vectorstore** or **hierarchical_document** tools, you **must** define an `embedding_provider` at the agent level:

```yaml
embedding_provider:
  provider: ollama
  name: nomic-embed-text:latest
  endpoint: http://localhost:11434

# or:
embedding_provider:
  provider: openai
  name: text-embedding-3-small
```

### Full annotated example

```yaml
# agent.yaml — full Claude-native agent
name: research-assistant
description: Research assistant powered by Claude

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  auth_provider: oauth_token
  temperature: 0.3
  max_tokens: 8000

embedding_provider:
  provider: ollama
  name: nomic-embed-text:latest
  endpoint: http://localhost:11434

instructions:
  inline: |
    You are a research assistant.
    Provide thorough, well-sourced answers.

claude:
  permission_mode: acceptEdits
  working_directory: ./workspace
  max_turns: 15

  extended_thinking:
    enabled: true
    budget_tokens: 20000

  web_search: true

  bash:
    enabled: true
    excluded_commands: ["rm -rf", "shutdown"]
    allow_unsafe: false

  file_system:
    read: true
    write: true
    edit: true

  agents:
    researcher:
      description: Gathers raw information from the web.
      prompt: "Search the web for primary sources. Cite every claim."
      tools: [WebSearch, WebFetch]
      model: haiku

  allowed_tools:
    - knowledge-base
    - web-search

tools:
  - name: knowledge-base
    type: vectorstore
    description: Search the research knowledge base
    source: ./data/research/
```

## Production considerations: memory and concurrency

The Claude backend runs each turn through a fresh Claude CLI **Node.js subprocess** spawned by `claude-agent-sdk` (see [Anthropic's hosting guide](https://docs.anthropic.com/en/api/agent-sdk/hosting)). This is the dominant memory cost in production and the binding constraint on concurrency. If you're deploying with `holodeck serve` to a container with a fixed memory limit (Azure Container Apps, Cloud Run, ECS Fargate, Kubernetes), the defaults below have been calibrated against real OOM data — but the right number for *your* agent depends on the tools it carries.

### The model in one diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Parent Python process (HoloDeck serve)                         │
│  ┌─────────────────────┐    in-process MCP server               │
│  │  AG-UI / REST API   │◄──── tool calls (vectorstore,          │
│  │  + session store    │      hierarchical_document, function)  │
│  └─────────────────────┘                                         │
│        │ spawn (per turn)                                        │
│        ▼                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │ Node CLI subprocess │    │ Node CLI subprocess │  …N turns    │
│  │   (~300 MiB)        │    │   (~300 MiB)        │              │
│  └─────────────────────┘    └─────────────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

Two things to notice:

1. **Tools run in the parent.** `create_sdk_mcp_server` hosts HoloDeck tools in the Python process and routes calls back to the subprocess via IPC. A heavy `hierarchical_document` tool with a qdrant client and a chunked corpus inflates the *parent* baseline, not the subprocess.
2. **Subprocesses are per-turn, not per-session.** Spec 034 P4 ("Hybrid Sessions") tears the subprocess down at turn end and persists transcripts as JSONL on disk. Idle sessions cost ~30 MiB each (Python object + JSONL handle); concurrent **active turns** cost the full subprocess.

This is why the concurrency cap is keyed on active turns, not open sessions.

### Memory budget defaults

| Component                          | Default | Configurable via                                  |
|-----------------------------------|---------|---------------------------------------------------|
| Parent baseline (server + tools + OTEL) | 400 MiB | (library constant)                          |
| Per concurrent active turn        | 500 MiB | `claude.session_memory_estimate_mib`             |
| Open session ceiling (idle slots) | 1000    | (library constant)                                |

The active-turn cap is derived from the replica's cgroup memory limit:

```
max_concurrent_turns = (memory_limit - baseline) / per_turn_estimate
```

The 500 MiB default covers the ~300 MiB Node CLI steady state plus the simultaneous-startup spike (when N turns hit `serve` at once, V8 initial heap allocation peaks together) plus parent-side transient work during the turn (hybrid search, rerank, context generation). It was calibrated against the spec 034 P4 cloud validation where a 2 GiB Azure Container Apps replica OOM-killed at 4 concurrent turns and ran cleanly at 3 — `(2048-400)/500 = 3` matches that.

### Sizing your replica

| Replica memory | Derived turn cap | Notes                                           |
|---------------|------------------|--------------------------------------------------|
| 1 GiB         | 1                | Bare minimum; one turn at a time                 |
| 2 GiB         | 3                | Default sample (financial-assistant) target     |
| 4 GiB         | 7                | Comfortable production tier                      |
| 8 GiB         | 15               | High-concurrency tier                            |

If you need higher concurrency, scale horizontally (more replicas) rather than vertically — the SDK subprocess model means each turn is independent, so adding replicas linearly increases capacity.

### Tuning `session_memory_estimate_mib`

The default targets the median tool-heavy agent. Adjust it when:

| Situation                                                       | Direction | Suggested override |
|----------------------------------------------------------------|-----------|--------------------|
| Thin agent — no vectorstore, no hierarchical_document, no MCP  | Lower     | `300` (yields cap=5 on 2 GiB) |
| Heavy retrieval — large hybrid index, in-process rerank        | Default   | `500`              |
| Embedded models — local rerank model, ONNX, sentence-transformers | Higher | `700–900`         |
| Multiple MCP servers with persistent state                     | Higher    | `600–700`          |

```yaml
claude:
  session_memory_estimate_mib: 700   # tune for your agent's tool footprint
```

### Excess-capacity behaviour

When concurrent active turns exceed the cap, the serve layer immediately returns HTTP 429 with a `Retry-After` body. It does **not** queue requests — Anthropic's hosting guidance recommends shedding load over buffering, because subprocess startup is expensive and a queued request that times out at the upstream is worse than a fast 429 that the client can retry with backoff.

### Observability

When OpenTelemetry is enabled (`observability.enabled: true`), the serve layer emits:

- `holodeck.serve.active_turns` (gauge) — current in-flight count
- `holodeck.serve.turn_rejections_total` (counter) — 429s due to the cap
- `holodeck.serve.session_count` (gauge) — open sessions (idle + active)

Alert on `turn_rejections_total` rate > 0 sustained for more than a minute — it indicates either undersized replicas or a need to scale horizontally.

!!! warning "Azure WorkingSetBytes is 1-minute averaged"
    Cloud-provider memory metrics typically report 1-minute averages, which mask the instantaneous peak during simultaneous turn startup. If you're sizing from Azure metrics, expect the real peak to be 1.5–2× the average. The 500 MiB default already accounts for this; if you're tuning from your own observed averages, multiply by ~1.7 before dividing into headroom.

### Anthropic's own guidance

Anthropic's [SDK hosting documentation](https://docs.anthropic.com/en/api/agent-sdk/hosting) recommends **1 GiB RAM per agent instance** as a conservative starting point — that includes the parent process. HoloDeck's 400 MiB parent baseline + 500 MiB per turn lands right on that number for cap=1 and scales linearly from there. There is no "shared subprocess pool" pattern documented in the SDK; each turn gets its own subprocess by design.

## Configuration reference

### `model.*` fields

| Field           | Type    | Required | Default     | Description                                                            |
|-----------------|---------|----------|-------------|------------------------------------------------------------------------|
| `provider`      | string  | yes      | -           | Must be `anthropic` to select the Claude backend                       |
| `name`          | string  | yes      | -           | Anthropic model identifier (e.g. `claude-sonnet-4-20250514`)           |
| `auth_provider` | enum    | no       | `api_key`   | One of `api_key`, `oauth_token`, `bedrock`, `vertex`, `foundry`        |
| `temperature`   | float   | no       | `0.3`       | Randomness, `0.0`–`2.0`                                                |
| `max_tokens`    | integer | no       | `1000`      | Maximum response tokens                                                |
| `top_p`         | float   | no       | -           | Nucleus sampling, `0.0`–`1.0`                                          |
| `api_key`       | string  | no       | -           | API key (when `auth_provider: api_key`); prefer `${ANTHROPIC_API_KEY}` |

### `claude.*` fields

| Field                | Type           | Default       | Description                                                              |
|----------------------|----------------|---------------|--------------------------------------------------------------------------|
| `permission_mode`    | enum           | `manual`      | `manual` \| `acceptEdits` \| `acceptAll`                                 |
| `working_directory`  | string         | -             | Restrict file access; subprocess `cwd`                                   |
| `max_turns`          | integer (≥1)   | SDK default   | Maximum agent loop iterations                                            |
| `extended_thinking`  | object         | -             | `{ enabled: bool, budget_tokens: int }`                                  |
| `web_search`         | boolean        | `false`       | Enable built-in web search                                               |
| `bash`               | object         | -             | `{ enabled: bool, excluded_commands: [str], allow_unsafe: bool }`        |
| `file_system`        | object         | -             | `{ read: bool, write: bool, edit: bool }`                                |
| `agents`             | map<str, spec> | -             | Named subagents (see Subagents above)                                    |
| `allowed_tools`      | list of strings | all tools    | Explicit tool allowlist                                                  |
| `session_memory_estimate_mib` | int (50–2000) | `500`  | Per-active-turn memory budget; drives the concurrency cap (see [Production considerations](#production-considerations-memory-and-concurrency)) |
| `max_concurrent_sessions` | int (1–500) | derived    | Hard cap on concurrent active turns; auto-derived from cgroup memory when omitted |

### Available models

| Model                          | Description                                | Context window |
|--------------------------------|--------------------------------------------|----------------|
| `claude-sonnet-4-20250514`     | Best balance of speed and capability       | 200K tokens    |
| `claude-opus-4-20250514`       | Most capable, best for complex tasks       | 200K tokens    |
| `claude-3-5-sonnet-20241022`   | Previous-generation Sonnet                 | 200K tokens    |
| `claude-3-5-haiku-20241022`    | Fast and cost-effective                    | 200K tokens    |

Check [Anthropic's model documentation](https://docs.anthropic.com/en/docs/about-claude/models) for the latest list.

### Anthropic environment variables

| Variable                          | Auth provider           | Description                                       |
|-----------------------------------|-------------------------|---------------------------------------------------|
| `ANTHROPIC_API_KEY`               | `api_key`               | API authentication key                            |
| `CLAUDE_CODE_OAUTH_TOKEN`         | `oauth_token`           | Claude Code OAuth token (recommended)             |
| `AWS_REGION`                      | `bedrock`               | AWS Bedrock region                                |
| `AWS_DEFAULT_REGION`              | `bedrock`               | Alternate AWS region variable                     |
| `CLOUD_ML_REGION`                 | `vertex`                | Vertex region                                     |
| `ANTHROPIC_VERTEX_PROJECT_ID`     | `vertex`                | Vertex project ID                                 |
| `GCLOUD_PROJECT`                  | `vertex`                | Alternate Vertex project context                  |
| `GOOGLE_CLOUD_PROJECT`            | `vertex`                | Alternate Vertex project context                  |
| `GOOGLE_APPLICATION_CREDENTIALS`  | `vertex`                | Service-account credential file path              |
| `ANTHROPIC_FOUNDRY_RESOURCE`      | `foundry`               | Foundry resource name                             |
| `ANTHROPIC_FOUNDRY_BASE_URL`      | `foundry`               | Alternate Foundry base URL                        |

## Troubleshooting

### Cloud auth context missing

**Error**: `AWS_REGION environment variable is not set for auth_provider: bedrock`
**or**: `CLOUD_ML_REGION environment variable is not set for auth_provider: vertex`
**or**: `Missing Foundry target for auth_provider: foundry`

1. Confirm `model.provider: anthropic` and your selected `auth_provider`.
2. Set the required routing variables (see the Authentication providers table above).
3. `model.endpoint` is ignored for cloud auth modes — drop it.

### Subagent not registering

**Symptom**: The parent falls back to the `general-purpose` agent and your custom subagents (`researcher`, `analyst`, …) never appear.

**Cause**: The Claude CLI silently drops `AgentDefinition`s whose `model` is outside `sonnet | opus | haiku | inherit`. Full model IDs are not accepted in the subagent slot.

**Fix**: Use the SDK aliases in `claude.agents.<name>.model`, or omit `model` to inherit the parent's pinned model.

### Embedding provider missing for Anthropic + vectorstore

**Error**: HoloDeck refuses to start when `provider: anthropic` is combined with vectorstore/hierarchical_document tools but no `embedding_provider` is configured.

**Fix**: Add an `embedding_provider` block (Ollama or OpenAI; see above).

### Invalid API key

**Error**: `AuthenticationError` or `Invalid API key`.

1. Verify the key: `echo $ANTHROPIC_API_KEY`.
2. Ensure no extra whitespace.
3. For OAuth, regenerate via the Claude Code CLI.

## Limitations

- `holodeck serve` and container deployment do not currently support `provider: anthropic`. Both are planned in a future release. See [Agent Server](serve.md) and [Deployment](deployment.md) for the current SK-only support matrix.

## Next steps

- [Agent Configuration](agent-configuration.md) — full agent.yaml structure
- [Tools](tools.md) — extending agent capabilities
- [Observability](observability.md) — tracing and metrics
- [Vector Stores](vector-stores.md) — semantic search configuration
