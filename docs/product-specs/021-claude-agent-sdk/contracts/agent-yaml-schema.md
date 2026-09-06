# Contract: Agent YAML Schema Extensions

**Feature**: 021-claude-agent-sdk
**Date**: 2026-02-20

---

## New Top-Level Fields

### `embedding_provider` (FR-012a)

```yaml
embedding_provider:
  provider: openai              # openai | azure_openai only (anthropic excluded)
  name: text-embedding-3-small
  endpoint: null                # Required for azure_openai
  api_key: null                 # Uses env var if null
```

**Rules**:
- Required when `model.provider: anthropic` AND any `tools[].type: vectorstore | hierarchical_document`
- Ignored for non-Anthropic providers
- If absent when required: `ConfigError` at startup (not runtime)

---

### `claude` (FR-002a, FR-001a)

```yaml
claude:
  working_directory: ./workspace  # null = current directory
  permission_mode: manual         # manual | acceptEdits | acceptAll
  max_turns: null                 # int >= 1; null = SDK default (unlimited)

  extended_thinking:
    enabled: false
    budget_tokens: 10000          # 1000–100000

  web_search: false               # Enable built-in web search

  bash:
    enabled: false
    excluded_commands: []         # ["rm -rf", "sudo rm"]
    allow_unsafe: false

  file_system:
    read: false
    write: false
    edit: false

  subagents:
    enabled: false
    max_parallel: 4               # 1–16

  allowed_tools: null             # null = all configured tools; list = explicit allowlist
```

**Rules**:
- All fields optional; all capabilities default to disabled (least-privilege, FR-001a)
- `claude:` block is ignored with a warning if `model.provider != anthropic`
- `permission_mode` defaults to `manual` (safest) when not specified
- `max_turns` is separate from execution `timeout_seconds` — it caps agent loop iterations, not wall-clock time

---

### `model.auth_provider` (FR-032)

```yaml
model:
  provider: anthropic
  name: claude-opus-4-6
  auth_provider: api_key         # api_key | oauth_token | bedrock | vertex | foundry
```

**Required environment variables by `auth_provider`**:

| Value | Required Env Vars |
|---|---|
| `api_key` (default) | `ANTHROPIC_API_KEY` |
| `oauth_token` | `CLAUDE_CODE_OAUTH_TOKEN` |
| `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (or configured AWS profile) |
| `vertex` | `GOOGLE_APPLICATION_CREDENTIALS` or application default credentials |
| `foundry` | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` or managed identity |

---

## Full Example: Claude-Native Agent

```yaml
name: research-assistant
description: Autonomous research agent with file access

model:
  provider: anthropic
  name: claude-opus-4-6
  temperature: 0.5
  max_tokens: 8192
  auth_provider: api_key

embedding_provider:
  provider: openai
  name: text-embedding-3-small

instructions:
  inline: |
    You are a research assistant with access to a knowledge base.
    Search for information before answering questions.

claude:
  working_directory: ./workspace
  permission_mode: acceptEdits
  max_turns: 20
  web_search: true
  file_system:
    read: true
    write: true
    edit: true

tools:
  - name: knowledge_base
    type: vectorstore
    source: ./docs/
    top_k: 5

evaluations:
  model:
    provider: openai
    name: gpt-4o
    temperature: 0.0
  metrics:
    - type: standard
      metric: rouge
      threshold: 0.5
    - type: geval
      name: Accuracy
      criteria: "Does the response accurately answer the question based on retrieved content?"
      evaluation_params: [actual_output, input, retrieval_context]
      threshold: 0.7

test_cases:
  - name: "Knowledge retrieval test"
    input: "What is the refund policy?"
    ground_truth: "30-day money-back guarantee"
    expected_tools: [knowledge_base]
    retrieval_context:
      - "Refund policy: All products include a 30-day money-back guarantee"
```

---

## Full Example: Minimal Claude-Native Agent

```yaml
name: simple-claude-agent
description: Plain conversational agent via Claude native backend

model:
  provider: anthropic
  name: claude-sonnet-4-6

instructions:
  inline: "You are a helpful assistant."

test_cases:
  - input: "What is 2 + 2?"
    ground_truth: "4"
```

No `claude:` block = all defaults (manual mode, no file access, no bash, no extended thinking).

---

## Full Example: Bedrock-Authenticated Agent

```yaml
name: enterprise-agent

model:
  provider: anthropic
  name: claude-opus-4-6
  auth_provider: bedrock

embedding_provider:
  provider: azure_openai
  name: text-embedding-ada-002
  endpoint: https://myorg.openai.azure.com/

instructions:
  file: ./instructions/enterprise.md

claude:
  permission_mode: acceptAll
  max_turns: 50
```

---

## Backward Compatibility

All new fields are optional. Existing `agent.yaml` files without any of the new fields continue to work without modification when `model.provider != anthropic`. For `provider: anthropic` agents:
- No new fields required for basic conversational use (FR-002)
- `embedding_provider` only required if vectorstore/hierarchical document tools are configured
- `claude:` block only required to enable any Claude-native capability

Unknown fields continue to raise a `ValidationError` (Pydantic `extra="forbid"` preserved, FR-002a).
