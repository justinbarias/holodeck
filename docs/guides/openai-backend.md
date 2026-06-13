# OpenAI Backend

The OpenAI backend runs your agent on top of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — OpenAI's native agent runtime. It is automatically selected when `model.provider` is `openai` or `azure_openai`.

This guide is for backend-specific behaviour: the `openai:` configuration block, reasoning effort, budget caps, model fallback, structured output, and tracing. For shared concepts like tools, observability, and vector stores, see the dedicated guides for each. It is the sibling of the [Claude Backend](claude-backend.md) guide.

## Quick start

A minimal `azure_openai` agent with three function tools — deterministic enough to verify with `holodeck test`.

```yaml
# agent.yaml
name: warehouse-agent

model:
  provider: azure_openai
  name: gpt-5.4 # MUST match your Azure deployment name
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  temperature: 0.0

instructions:
  inline: "You are a warehouse assistant. Use the tools to answer stock questions."

tools:
  - name: get_inventory
    type: function
    description: Look up catalog stock and unit price for a SKU.
    file: tools/warehouse.py
    function: get_inventory

test_cases:
  - name: "Single-tool lookup"
    input: "Is SKU WIDGET-1 in stock, and what does one cost?"
    expected_tools: [get_inventory]
    ground_truth: "WIDGET-1 is in stock (120 units) at $12.50 each."
```

```bash
# .env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
```

```bash
holodeck test run agent.yaml -n 1
# => PASS  Single-tool lookup  (tool calls: get_inventory)
```

For plain OpenAI instead, set `model.provider: openai`, `name: gpt-4o-mini` (drop `endpoint`), and supply `OPENAI_API_KEY`.

## How it works

`BackendSelector` routes `provider: openai` and `provider: azure_openai` to the OpenAI Agents backend; Anthropic and Ollama route to the [Claude backend](claude-backend.md). The backend runs the SDK `Runner` loop **in-process** (no per-turn subprocess), driving up to `openai.max_turns` agent iterations per call. The `openai` / `agents` SDK is imported lazily — only when an OpenAI-provider agent is actually selected — so non-OpenAI agents never pay its import cost. HoloDeck tools (function, vectorstore, hierarchical_document, MCP) are adapted onto the SDK's tool surface; structured output, budget, fallback, and tracing are layered on via the SDK's output-schema, `RunHooks`, model-wrapping, and `TracingProcessor` extension points.

## The `openai:` block

All OpenAI-specific settings live under the top-level `openai:` block in `agent.yaml`. Every field is optional.

```yaml
openai:
  max_turns: 20
  effort: high
```

| Field | Type | Default | Constraint | Meaning |
|-------|------|---------|------------|---------|
| `max_turns` | int | `20` | ≥ 1 | Maximum agent loop iterations passed to `Runner.run`. **Shipped.** |
| `effort` | enum | – | `low`\|`medium`\|`high`\|`max` | Reasoning effort for reasoning models (see [Reasoning effort](#reasoning-effort)). **Shipped.** |
| `max_budget_usd` | float | – | > 0 | Hard cap on session spend in USD (see [Budget](#budget)). **Shipped.** |
| `fallback_model` | str | – | – | Model used on a retryable primary failure (see [Fallback](#fallback)). **Shipped.** |
| `disallowed_tools` | list[str] | – | – | Tools removed from the resolved agent at build time. **Shipped.** |
| `session_memory_estimate_mib` | int | `100` | 50–2000 | Estimated peak resident memory per active turn; used by `serve` to derive the concurrency cap. The backend runs in-process (no per-turn subprocess), so this is lower than the Claude default. *Config accepted; serve auto-sizing lands in a later release.* |
| `max_concurrent_sessions` | int | – (derived) | 1–500 | Hard cap on concurrent active turns per serve instance. When unset, derived from the replica's memory limit ÷ `session_memory_estimate_mib`. *Config accepted; serve enforcement lands in a later release.* |
| `permissions.allowed_tools` | list[str] | `null` (all) | – | Explicit tool allowlist. *Config accepted; full enforcement nuance lands in a later release — prefer `disallowed_tools` today.* |
| `permissions.disallowed_tools` | list[str] | – | – | Deny list; takes precedence over `allowed_tools`. *Config accepted; prefer the top-level `disallowed_tools` today.* |
| `i_understand_this_is_unsafe` | bool | `false` | – | Acknowledges that hosted tools (e.g. code interpreter) permit server-side code execution. *Config accepted; hosted tools are not yet shipped — see [Coming soon](#coming-soon).* |
| `disable_default_hooks` | bool | `false` | – | Disables HoloDeck's default credential-redaction output guardrail. *Config accepted; default redaction guardrails are not yet shipped, so this is currently a no-op. OTel attribute redaction runs independently and is unaffected.* |
| `disable_subprocess_env_scrub` | bool | `false` | – | Disables env scrubbing for stdio MCP servers / shelling-out function tools. *Config accepted; full enforcement lands in a later release.* |

!!! note "Honest status labels"
    Fields marked *config accepted* validate and load today but their runtime feature ships later. They are documented so you can author forward-compatible configs, not because the behaviour is live. The four **Shipped** runtime features (`effort`, `max_budget_usd`, `fallback_model`, `disallowed_tools`) plus `max_turns` are fully enforced now.

## Reasoning effort

`effort` requests deeper internal reasoning from reasoning models (o-series and `gpt-5`+). It only affects reasoning models; on non-reasoning models it has no effect.

```yaml
openai:
  effort: high
```

The level maps onto the SDK's `ReasoningEffort` literal:

| `effort` | SDK `ReasoningEffort` |
|----------|------------------------|
| `low` | `low` |
| `medium` | `medium` |
| `high` | `high` |
| `max` | `xhigh` |

!!! note "`max` → `xhigh` is an intentional deviation"
    HoloDeck exposes `max` as the strongest level; it maps to the SDK's `xhigh` rather than a literal `max`. This is deliberate so the HoloDeck vocabulary stays stable as the SDK's effort ceiling evolves.

Setting `effort` also requests reasoning **summaries** (`summary="auto"`), which populate the [`thinking`](#thinking) field on the result. With `effort` unset, no summary is requested.

## Budget

`max_budget_usd` caps total spend across a session. A `RunHooks` cost accountant prices every LLM response from the SDK's per-response token usage against a bundled, versioned per-model price table and accumulates the running total.

```yaml
openai:
  max_budget_usd: 0.50
```

- When the accumulated cost reaches the cap, the hook aborts the turn (`BackendBudgetExceededError`), and the backend surfaces it on the standard error path (`is_error` / `error_reason`) with the **partial response preserved** — whatever assistant text the model produced before the cap tripped.
- The budget is **per session**: one accountant is shared across every turn of a session, so the cap covers the whole conversation.
- An **unknown model** (e.g. an opaque Azure deployment name that doesn't resolve to a base model) logs a single warning and contributes no cost — enforcement is silently disabled for that model rather than crashing the run.

## Fallback

`fallback_model` wraps the primary model so that a **retryable** upstream failure is re-issued **once** against the fallback.

```yaml
openai:
  fallback_model: gpt-4o-mini
```

- **Retryable set (fixed):** HTTP 429 (rate limit) and 5xx (server-side). Everything else — 400/401/403/404/422, connection/timeout errors, non-OpenAI exceptions — propagates unchanged; the fallback is never consulted.
- **Ordering:** if you enable the SDK runner's own retries, the primary retries exhaust **first**, then exactly one fallback attempt. No double-fallback, no fallback mid-retry — the wrapper itself never retries the fallback.
- **Streaming:** the wrapper falls back only if the primary stream fails **before its first event**. Once any event has been emitted, a later failure propagates unchanged (restarting on the fallback would replay already-delivered deltas).
- **Tracing:** both the primary and the fallback attempt open their own generation span, so both are visible in the trace.

For Azure, build the fallback as another deployment on the same endpoint/credentials.

## Structured output

`response_format` constrains every final response to a JSON schema. It accepts a JSON-schema **dict** inline, a **string** path to a JSON file, or `None`. The resolved schema drives the SDK's structured-output path, and the parsed object is populated on the `ExecutionResult.structured_output` field — downstream graders `json.loads` the response.

```yaml
response_format:
  type: object
  additionalProperties: false
  required: [answer]
  properties:
    answer:
      anyOf:
        - type: number
        - type: string
```

!!! warning "Portability: use `anyOf`, never `oneOf`"
    OpenAI structured outputs **reject `oneOf`**. Use `anyOf` instead (we hit this live against Azure OpenAI). The same `anyOf` schema also works on the [Claude backend](claude-backend.md), so authoring with `anyOf` keeps your schema portable across both backends.

**Strict mode** is auto-enabled only when the schema already qualifies: a top-level `type: object` with `additionalProperties: false` and **every** declared property listed in `required`. HoloDeck never rewrites your schema to force strictness — if it doesn't qualify, the schema is still enforced via `jsonschema` validation, just without provider strict-mode guarantees.

## Thinking

The `thinking` field is populated from a reasoning model's **summaries**, which are only requested when `openai.effort` is set (it implies `summary="auto"`). Without `effort`, or on a non-reasoning model, `thinking` is empty.

## Tools

| Tool type | Behaviour on this backend |
|-----------|----------------------------|
| `function` | Python callables (sync or async) wrapped as SDK function tools. |
| `vectorstore` | Wraps the tool's `.search()`; surfaced to the model as `{name}_search`. Requires an `embedding_provider`. |
| `hierarchical_document` | Same wrapping pattern; surfaced as `{name}_search`. Requires an `embedding_provider`. |

The `{name}_search` naming keeps `disallowed_tools` portable across backends. For RAG configuration depth (chunking, hybrid search, databases), see [Tools](tools.md) and [Vector Stores](vector-stores.md) rather than duplicating it here.

## MCP

MCP tools map onto the SDK's MCP server classes by transport:

| `transport` | SDK server |
|-------------|------------|
| `stdio` | `MCPServerStdio` |
| `sse` | `MCPServerSse` |
| `http` | `MCPServerStreamableHttp` |
| `websocket` | **Skipped with a warning** — the SDK has no WebSocket transport. The load does **not** fail. |

A per-server `allowed_tools` list becomes a static SDK tool filter (only the listed MCP tools are exposed). See [MCP CLI](mcp-cli.md) for transport configuration.

## Tracing

The SDK runs its own tracing pipeline. HoloDeck installs an OTel-mirroring `TracingProcessor` that reconstructs each finished SDK span as an OTel span on HoloDeck's global tracer (carrying the redacting span processor and your configured exporters). The mirror is installed only when `observability.enabled` **and** `observability.traces.enabled` are both true.

| Configuration | platform.openai.com upload | OTel mirror |
|---------------|----------------------------|-------------|
| `provider: openai` | retained (default exporter kept) | ✓ |
| `provider: azure_openai` | none (mirror only) | ✓ |
| `observability.disable_provider_tracing: true` (either provider) | none (mirror only) | ✓ |

!!! note "Sensitive data is not uploaded by default"
    The SDK's `trace_include_sensitive_data` is bound to `observability.traces.capture_content` (default **false**). With the default, tool input/output is **not** included in uploaded spans. Set `capture_content: true` only when the data is safe to capture. See [Observability](observability.md).

## Per-backend semantics

Shipped surface only, OpenAI vs Claude:

| Capability | OpenAI | Claude |
|------------|--------|--------|
| Function tools | ✓ | ✓ |
| RAG (vectorstore / hierarchical_document) | ✓ | ✓ |
| MCP stdio / sse / http | ✓ | ✓ |
| Structured output | ✓ (use `anyOf`, not `oneOf`) | ✓ |
| Reasoning / `thinking` | ✓ | ✓ |
| `effort` / `max_budget_usd` / `fallback_model` | ✓ (via `openai:`) | — |
| `holodeck chat` / `holodeck test` | ✓ | ✓ |
| `holodeck serve` / `holodeck deploy` | ✗ (roadmap) | ✓ |

`holodeck serve` and `holodeck deploy` are fully supported on the **Claude
backend** today. On the OpenAI backend they are on the [roadmap](#coming-soon) —
use `holodeck chat` and `holodeck test` for now.

## Coming soon

The following are **not yet available** on this backend — they are roadmap, not shipped:

- **Subagents / handoffs** — a multi-agent `openai.agents` block.
- **YAML hooks** — user-defined `openai.hooks`.
- **Hosted tools** — web search, code interpreter, file search, image generation, hosted MCP (this is what `i_understand_this_is_unsafe` gates).
- **`holodeck serve` & `holodeck deploy`** — running this backend as a REST/AG-UI server or deploying it to a container platform is not yet wired. (Both are fully supported on the [Claude backend](claude-backend.md).) The `max_concurrent_sessions` / `session_memory_estimate_mib` knobs are accepted in config ahead of that work but are not yet enforced.
- **Default credential-redaction guardrails** — the output guardrail that `disable_default_hooks` would turn off.

## Troubleshooting

### Missing Azure credentials

**Error**: `AZURE_OPENAI_API_KEY is required for provider 'azure_openai'` or `AZURE_OPENAI_ENDPOINT ...`

1. Set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` (env or `model.api_key` / `model.endpoint`).
2. Confirm `model.name` matches your Azure **deployment** name, not a base model id.
3. For plain OpenAI, set `OPENAI_API_KEY` and use `provider: openai` (no endpoint).

### `oneOf` schema rejected

**Symptom**: the provider rejects your `response_format` schema.

**Fix**: replace `oneOf` with `anyOf` — OpenAI structured outputs do not accept `oneOf`. The `anyOf` form also works on the Claude backend.

### Reasoning-model sampling params ignored or erroring

Reasoning models (o-series, `gpt-5`+) reject `temperature` / `top_p` and use `max_output_tokens` rather than `max_tokens`. The backend detects reasoning models by name prefix; if your Azure deployment name is opaque (doesn't embed the base model), set sane sampling params explicitly and don't rely on auto-detection.

### MCP WebSocket tool skipped

**Symptom**: a `transport: websocket` MCP tool logs `Skipping MCP tool '...': websocket transport is not supported`.

This is expected — the SDK has no WebSocket transport, so the tool is skipped (the load does not fail). Use `stdio`, `sse`, or `http`.

## Next steps

- [Agent Configuration](agent-configuration.md) — full agent.yaml structure
- [Tools](tools.md) — extending agent capabilities
- [Vector Stores](vector-stores.md) — semantic search configuration
- [Observability](observability.md) — tracing and metrics
- [Claude Backend](claude-backend.md) — the sibling backend
