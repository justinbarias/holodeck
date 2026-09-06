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
| `i_understand_this_is_unsafe` | bool | `false` | – | Acknowledges that `CodeInterpreterTool` runs model-written code in an OpenAI-hosted container. Required to load that hosted tool (see [Hosted tools](#hosted-tools)). **Shipped.** |
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
- **Ordering (bounded):** the primary call carries the OpenAI client's own retries (`max_retries`, default 2, honouring `retry-after`); when those exhaust with a retryable error, exactly **one** fallback attempt follows, and its result or error is returned unchanged. HoloDeck never enables the SDK runner's `ModelSettings.retry`, so the Runner schedules no policy retries of the primary-then-fallback pair: at most `1 + max_retries` primary requests, then at most `1 + max_retries` fallback requests per pair. One SDK compatibility path remains: if the fallback answers HTTP 400 `conversation_locked`, the SDK re-runs the pair up to three more times (1 s / 2 s / 4 s backoff) before raising.
- **Streaming:** the wrapper falls back only if the primary stream fails **before its first event**, including non-text events such as `response.created`. Once any event has been emitted, a later failure propagates unchanged (restarting on the fallback would replay already-delivered deltas).
- **Tracing:** both the primary attempt (recorded with its error) and the fallback attempt open their own `response` span under the same trace, so both are visible on the provider dashboard when upload is permitted and in the OTel mirror.

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
| `skill` | Becomes a handoff-target sub-agent scoped to its `allowed_tools`; see [Skills](#skills). |
| `hosted` | OpenAI-platform tools (web search, file search, code interpreter, image generation, hosted MCP); see [Hosted tools](#hosted-tools). |

The `{name}_search` naming keeps `disallowed_tools` portable across backends. For RAG configuration depth (chunking, hybrid search, databases), see [Tools](tools.md) and [Vector Stores](vector-stores.md) rather than duplicating it here.

## Hosted tools

`type: hosted` entries select one of the SDK's server-side tools. They run on the OpenAI platform through the Responses API, so HoloDeck never executes them locally: no subprocess, no MCP connection, no local guardrail. `name` is the config name you use in `disallowed_tools` and subagent `tools` lists; `tool` picks the SDK class; `params` are that class's constructor parameters.

```yaml
tools:
  - name: web
    type: hosted
    tool: WebSearchTool
    params:
      search_context_size: low
      allowed_domains: [example.com]
      user_location: {city: Austin, country: US}
  - name: policies
    type: hosted
    tool: FileSearchTool
    params:
      vector_store_ids: [vs_123]
      max_num_results: 5
  - name: sandbox
    type: hosted
    tool: CodeInterpreterTool          # needs openai.i_understand_this_is_unsafe: true
    params:
      container: {type: auto, memory_limit: 4g}
  - name: art
    type: hosted
    tool: ImageGenerationTool
    params: {quality: low, size: 1024x1024}
  - name: docs
    type: hosted
    tool: HostedMCPTool
    params:
      server_label: docs
      server_url: https://mcp.example.com
      authorization: ${DOCS_TOKEN}     # ${VAR} substitution, like local MCP headers
      allowed_tools: [search, read]
```

| `tool` | SDK tool name | Required params | Notable params |
|--------|---------------|-----------------|----------------|
| `WebSearchTool` | `web_search` | – | `search_context_size`, `allowed_domains`, `user_location.{city,country,region,timezone}`, `external_web_access` |
| `FileSearchTool` | `file_search` | `vector_store_ids` | `max_num_results` (1–50), `include_search_results`, `ranking_options.{ranker,score_threshold}`, `filters` (OpenAI attribute filter, passed through) |
| `CodeInterpreterTool` | `code_interpreter` | `container` (id or `{type: auto, file_ids, memory_limit, network_policy}`) | Gated by `openai.i_understand_this_is_unsafe` |
| `ImageGenerationTool` | `image_generation` | – | `model`, `quality`, `size`, `output_format`, `output_compression`, `background`, `action`, `moderation`, `partial_images`, `input_fidelity` |
| `HostedMCPTool` | `hosted_mcp` | `server_label` and exactly one of `server_url` / `connector_id` | `authorization`, `headers`, `allowed_tools`, `server_description`; `require_approval` must stay `never` |
| `ComputerTool` | – | – | **Always rejected** at config load; it needs a computer harness HoloDeck does not provide yet (H-012) |

Rules that fail config load: an unknown `tool` value, a missing required param, an unknown param (`extra: forbid`), two hosted entries of the same class (they would share one SDK tool name), a `CodeInterpreterTool` without the opt-in, and `HostedMCPTool` with `require_approval` other than `never` (interactive approval is deferred; an unsupported gate fails closed rather than running the tool). Every problem is reported in the same validation pass as credential and permission errors. A `disallowed_tools` entry naming a hosted tool drops it before construction, so a disallowed code interpreter needs no opt-in. Hosted tools load on the Claude backend only as an error: they are OpenAI Responses features.

Hosted calls appear on the tool-event stream and in `ExecutionResult.tool_calls` / `tool_results` under the SDK tool name (`web_search`, `file_search`, `code_interpreter`, `image_generation`; hosted MCP calls use the remote tool's name with the `server_label` in the arguments), so `expected_tools: [web_search]` works in test cases. Because they run server-side, a hosted call emits `start` and `end` together, its result is a status or summary rather than raw output, and image bytes are omitted.

### Hosted tools on Azure

Hosted entries load on `provider: azure_openai` too; there is no blanket configuration ban. Whether a call succeeds depends on the Azure resource, region, and API surface, and that is only known at run time. When a run fails on Azure and the agent declares hosted tools, the SDK error is preserved verbatim and HoloDeck appends a hint naming the declared hosted classes. If you see it, check the resource's Responses API tool support or remove the entry.

### Limits

- Hosted tools are outside HoloDeck's model-visible guardrails: no output redaction, hooks, or rejection rules apply to what the OpenAI platform executes (OTel attribute redaction still applies to exported spans).
- `openai.fallback_model` falls back to a model on the same provider; a fallback that lacks Responses hosted-tool support fails with the provider's error rather than silently dropping the tools.
- Hosted MCP approval loops and `ComputerTool` are deferred ([H-012, H-013](../exec-plans/tech-debt-tracker.md)).

## MCP

MCP tools map onto the SDK's MCP server classes by transport:

| `transport` | SDK server |
|-------------|------------|
| `stdio` | `MCPServerStdio` |
| `sse` | `MCPServerSse` |
| `http` | `MCPServerStreamableHttp` |
| `websocket` | **Skipped with a warning** — the SDK has no WebSocket transport. The load does **not** fail. |

A per-server `allowed_tools` list becomes a static SDK tool filter (only the listed MCP tools are exposed). See [MCP CLI](mcp-cli.md) for transport configuration.

## Subagents (handoffs)

Declare handoff targets under `openai.agents`. Each entry becomes an SDK `Agent` on the parent's `handoffs`, exposed to the parent model as a `transfer_to_<name>` tool:

```yaml
openai:
  agents:
    researcher:
      description: Finds and cites sources        # shown to the parent for routing
      prompt: You research questions thoroughly.  # or prompt_file: prompts/researcher.md
      tools: [search_kb]                          # parent tool names; omit to inherit all
      model: inherit                              # default; or any OpenAI model id / Azure deployment
    writer:
      description: Drafts the final answer
      prompt_file: prompts/writer.md
      skip_recommended_prefix: true
```

| Field | Behaviour |
|-------|-----------|
| `description` | Required. The SDK `handoff_description` the parent model routes on. |
| `prompt` / `prompt_file` | Exactly one. `prompt_file` is resolved relative to `agent.yaml` and inlined at load. |
| `tools` | Omitted or `null`: the subagent inherits **every** parent tool and MCP server. A list restricts it to those parent tools, named as written under the parent's `tools:` (so `search_kb`, not `search_kb_search`). A name that matches no parent tool fails load. An empty list grants nothing. |
| `model` | `inherit` (default) reuses the parent's model object, including any `fallback_model` wrapper. Any other string is passed to the SDK as a model identifier (an Azure deployment on the same endpoint for `azure_openai`) with no fallback wrapper. The Claude aliases `sonnet`, `opus`, and `haiku` fail load. |
| `skip_recommended_prefix` | By default the SDK's `RECOMMENDED_PROMPT_PREFIX` (from `agents.extensions.handoff_prompt`) is prepended once to the subagent's instructions so it knows it is part of a handoff system. Set `true` to use the prompt verbatim. |

Handoff-history shaping (`handoff_input_filter`, `nest_handoff_history`) stays at SDK defaults. `claude.agents` is Claude-only and is not read by this backend.

**Events.** A handoff surfaces on the same `ToolEvent` stream the Claude backend uses, so `holodeck chat` and the AG-UI tools panel render it without protocol changes: `start` for the `transfer_to_<name>` call, a `subagent_message` announcing the new active agent, `parent_link` for every tool the subagent calls (so the panel nests them), `subagent_message` snapshots of the subagent's text, and finally `end` for the handoff when the run finishes (the target agent owns the conversation until then, like a Claude `Task`). Ordinary tool `start`/`end` and reasoning `thinking` events are emitted on this backend too. Streaming turns emit events live; non-streaming turns emit the same ordered list after the run completes. If the run fails mid-stream, every still-open tool call and handoff is closed with an `error` event carrying the failure. A local tool that raises inside the SDK loop is reported as an `end` event carrying the SDK's error text (the SDK converts the exception into a model-visible string). The per-session event queue is bounded (1000 entries); when nothing drains it, newer events are dropped rather than growing memory.

Subagents also inherit the parent's `response_format` output type and, for `model: inherit`, the parent's model settings; an explicit `model` gets settings rebuilt for that model (reasoning-model rules apply per model). Two parent tools may not share an SDK tool name (a vectorstore `kb` and a function `kb_search` collide) and two handoff targets may not normalise to the same `transfer_to_` tool (`research_assistant` and `research-assistant` collide); both fail load.

A complete, runnable configuration that exercises every surface on this page (function, RAG, MCP, subagents, skills, effort, budget, fallback, tracing, tests) lives in [`sample/openai-agents-full`](https://github.com/justinbarias/holodeck/tree/main/sample/openai-agents-full).

## Skills

`type: skill` tools are scoped sub-agents following the [Agent Skills specification](https://agentskills.io/specification). On this backend a skill becomes a handoff target exactly like a subagent, with two differences: its instructions are used verbatim (no recommended prefix) and it always inherits the parent's model.

```yaml
tools:
  - name: summarise            # lowercase, hyphen-separated (Agent Skills naming)
    type: skill
    description: Summarise a document in three bullets
    instructions: |
      Read the provided text and return three bullet points.
    allowed_tools: [search_kb]  # parent tool names; omit for no tools
  - name: research-assistant
    type: skill
    path: skills/research-assistant   # directory containing SKILL.md
    allowed_tools: [search_kb]
```

A file-based skill's `SKILL.md` must start with a `---` YAML frontmatter block containing `name` and `description`; the Markdown body becomes the instructions. `description` may be omitted in YAML and falls back to the frontmatter. The file is validated when the config loads, so a missing directory, missing `SKILL.md`, or missing frontmatter field fails `holodeck` before any provider call. The SKILL.md `allowed-tools` key is **not** merged: tool scope comes only from the YAML `allowed_tools`, which must name non-skill tools declared on the parent. A skill name that matches an `openai.agents` key fails load (both would produce the same `transfer_to_` tool). See [Tools](tools.md#skill-tools) for the field reference.

`claude.setting_sources` is accepted for portability but has no effect here; loading such an agent on this backend logs `setting_sources is a Claude-only concept; ignored on openai.` No ambient skill discovery happens (tracked as H-020).

## Tracing

The SDK runs its own tracing pipeline, with a single process-global processor list. HoloDeck owns that list: the first OpenAI-backend `initialize()` in a process installs one HoloDeck trace router (replacing the SDK's default exporter), and every backend instance registers its own **tracing policy** with the router — an upload decision plus, when `observability.enabled` **and** `observability.traces.enabled` are both true, an OTel-mirroring `TracingProcessor` that reconstructs each finished SDK span as an OTel span on HoloDeck's global tracer (carrying the redacting span processor and your configured exporters).

Each run tags its trace with the backend's policy id, and the router applies that policy per trace. The upload decision therefore never depends on which agent initialized first, on repeated initialization, or on whether observability is enabled: an Azure agent served next to an OpenAI agent never uploads, and the OpenAI agent still does.

| Configuration | platform.openai.com upload | OTel mirror (when tracing enabled) |
|---------------|----------------------------|------------------------------------|
| `provider: openai` | ✓ | ✓ |
| `provider: azure_openai` | never (also with observability disabled) | ✓ |
| `observability.disable_provider_tracing: true` (either provider) | never | ✓ |

With observability disabled, an OpenAI agent keeps the SDK default upload and an Azure agent emits nothing. Spans produced by a HoloDeck run follow that run's backend even inside a trace you opened yourself with the SDK's `trace()` (the outer trace's own record keeps SDK default behaviour); each span's policy is fixed when it starts, so a span that finishes later, elsewhere, or after its trace ended keeps it. Traces from non-HoloDeck SDK usage in the same process keep the SDK default behaviour. A backend's policy is withdrawn at `teardown()`; from then on its events are dropped rather than uploaded.

!!! note "Sensitive data is not uploaded by default"
    The SDK's `trace_include_sensitive_data` is bound to `observability.traces.capture_content` (default **false**). With the default, tool input/output is **not** included in uploaded spans. Set `capture_content: true` only when the data is safe to capture. See [Observability](observability.md).

## Per-backend semantics

Shipped surface only, OpenAI vs Claude:

| Capability | OpenAI | Claude |
|------------|--------|--------|
| Function tools | ✓ | ✓ |
| RAG (vectorstore / hierarchical_document) | ✓ | ✓ |
| MCP stdio / sse / http | ✓ | ✓ |
| Subagents / handoffs | ✓ (`openai.agents`) | ✓ (`claude.agents`) |
| Skills (`type: skill`) | ✓ (handoff target) | ✗ (not yet adapted) |
| Hosted tools (`type: hosted`) | ✓ (five classes; `ComputerTool` deferred) | ✗ (fails load) |
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

- **YAML hooks** — user-defined `openai.hooks`.
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

### Hosted tool rejected at load

`ComputerTool` is always rejected (H-012). `CodeInterpreterTool` needs `openai.i_understand_this_is_unsafe: true`. `HostedMCPTool` needs `server_label` plus exactly one of `server_url` / `connector_id`, and `require_approval` must be `never`. Two entries of one hosted class collide on the SDK tool name; keep one per class.

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
