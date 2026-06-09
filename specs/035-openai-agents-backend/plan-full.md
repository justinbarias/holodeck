# Implementation Plan: OpenAI Agents SDK Backend — Full Parity (post-MVP)

**Spec:** `specs/035-openai-agents-backend/spec.md`
**Builds on:** `plan-mvp.md` (shipped — function tools, real streaming, routing flip, SK agent-path carve)
**Scope of this plan:** Everything in spec 035 that the MVP deliberately left out, **except** two surfaces deferred to their own specs by decision below: the P3 hardened/Envoy profile and US8 sandbox mode.
**Status:** Draft for review

---

## Overview

The MVP proved one vertical slice: a `provider: openai` / `azure_openai` agent runs the OpenAI
Agents SDK `Runner` loop, calls Python **function** tools, and streams under `holodeck chat`, with
default routing flipped and the SK agent path removed. This plan delivers the **rest of the
backend's parity surface**:

- the remaining four HoloDeck tool types (vectorstore, hierarchical_document, skill, MCP);
- subagents (handoffs) and YAML hooks;
- the spec-026 config mappings (`effort`, `max_budget_usd`, `fallback_model`, `disallowed_tools`);
- hosted tools (`type: hosted`);
- tracing dual-write;
- serve/deploy parity and the spec-034 hardening phases **P1a/P1b/P2a/P2b**.

It is organised as **vertical, feature-complete slices** (model + adapter + backend wiring +
tests per slice) so each phase leaves `holodeck chat`/`test`/`serve` working.

## Decisions (confirmed with user, 2026-06-09)

1. **Namespacing → a new top-level `openai:` block.** Hooks, subagents, permissions, and all
   backend knobs for this backend live under `openai:` (sibling to `claude:`). `claude.hooks` /
   `claude.agents` stay **Claude-only** — they are NOT read by this backend. Cross-backend
   portability is achieved by declaring the relevant block per backend, not by sharing
   `claude.*`. *(This supersedes spec Open Question 1 — neither option (a) "reuse claude.\*" nor
   the `openai_agents:` name; the block is literally `openai:`.)*
2. **P3 hardened profile (Envoy sidecar, `deployment.security_profile`, egress allowlist,
   base-URL rewriting) → deferred to a separate cross-backend spec.** It exists for **no** backend
   today (Claude included); building it here would be net-new shared infra with backend-specific
   divergence. FR-090…FR-093 are out of this plan. P1a/P1b/P2a/P2b stay in.
3. **US8 sandbox mode (`agent_mode: sandbox`) → deferred to a follow-up spec.** SDK 0.17.4 exposes
   `agents.sandbox.SandboxAgent` + `Manifest` but **not** the `UnixLocalSandboxClient` the spec
   names; the real surface (`agents.sandbox.session` / `runtime` / `SandboxRunConfig`) differs from
   the spec, and the spec's own "Out of scope" section already lists `SandboxAgent`. FR-094…FR-099
   are out of this plan. The `openai.i_understand_this_is_unsafe` gate still ships (it gates
   `CodeInterpreterTool`/`ComputerTool`).
4. **Hosted tools allowed on Azure, runtime-gated.** The MVP put Azure on the Responses API
   (`/openai/v1`), so the spec's blanket config-load block (US5 scenario 3 / FR-091) is relaxed:
   hosted tools are accepted on `azure_openai`; if a specific tool isn't available on the resource,
   the SDK surfaces a runtime error that HoloDeck propagates verbatim with a clear hint.

## Spec ↔ reality reconciliations (carried into this plan)

These differ from the spec text; the plan follows reality, not the literal FR wording:

- **No `Backend` enum / no `backend:` override / no `agent_framework` backend exist.** Routing is
  purely by `model.provider` in `BackendSelector` (`src/holodeck/lib/backends/selector.py`), and
  the `openai`/`azure_openai → openai_agents` flip already shipped in the MVP. → **FR-001 and FR-007
  are already satisfied**; US1 scenario 4 (explicit AF opt-out) is **dropped** (no AF backend in
  this repo).
- **`MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` live under `agents.mcp.*`,** not at
  the `agents` top level. Confirmed present in 0.17.4, plus `create_static_tool_filter`.
- **`RedactingSpanProcessor` is already backend-agnostic** (`otel_redaction.py`) — P2b OTel
  redaction (FR-088) needs verification only, no new code.
- **`security_profile` / Envoy infra does not exist** → see Decision 2.

## Architecture decisions

- **One `openai:` config model** (`src/holodeck/models/openai_config.py`, `OpenAIConfig`) mirrors
  the role `ClaudeConfig` plays for Claude. It carries sizing (`max_concurrent_sessions`,
  `session_memory_estimate_mib`), `max_turns`, safety (`i_understand_this_is_unsafe`),
  hook/redaction opt-outs, `permissions`, the spec-026 fields (`effort`, `max_budget_usd`,
  `fallback_model`, `disallowed_tools`), plus `hooks` and `agents` sub-blocks.
- **Lazy import gate preserved.** Every `import agents` / `import openai` stays inside
  functions/methods of the `openai_agents_*` modules (SC-005). Pydantic models in
  `models/openai_config.py` never import the SDK.
- **Tool adapters extend the existing `build_sdk_tools`** in
  `openai_agents_tool_adapters.py`. Today it raises `ConfigError` for every non-function type; each
  phase below removes one type from that error path and returns the right SDK object. Returns become
  a tuple `(tools, mcp_servers, handoffs)` so MCP servers and skill/subagent handoffs flow into
  `Agent(...)`.
- **Reuse the live RAG factories.** Vectorstore / hierarchical_document adapters wrap the existing
  tool `.search()` callables (same objects the Claude adapter wraps in
  `lib/backends/tool_adapters.py`), reusing `tool_initializer.create_embedding_service`. No new
  embedding code.
- **Hooks are observation-first.** `AgentHooks`/`RunHooks` cover `log`/`notify`/`script`;
  `reject` synthesises `needs_approval` / `input_guardrail`; `modify` is inert + load warning (spec
  Open Question 2 → v1 inert).
- **Cost accountant price table = bundled versioned constant + unknown-model warning** (spec Open
  Question 3 → v1). Lives in `openai_agents_cost.py`.
- **Unit tests mock the SDK** (no network, no key); creds-gated integration smokes exercise live
  Azure.

---

## Task List

### Phase A — `openai:` config block foundation

#### Task A1: `OpenAIConfig` model + `openai:` block on `Agent`
**Description:** New `src/holodeck/models/openai_config.py` with `OpenAIConfig` and
`OpenAIPermissionsConfig`. Add `openai: OpenAIConfig | None` to the `Agent` model (sibling to
`claude`). Fields: `max_concurrent_sessions`, `session_memory_estimate_mib` (default `100`),
`max_turns` (default `20`), `i_understand_this_is_unsafe` (default `False`),
`disable_default_hooks`, `disable_subprocess_env_scrub`, `permissions` (`allowed_tools` /
`disallowed_tools`), `effort` (`low|medium|high|max`), `max_budget_usd`, `fallback_model`,
`disallowed_tools`. (`hooks` and `agents` sub-blocks are added in Phases E and D.)
**Acceptance criteria:**
- [ ] `agent.yaml` with an `openai:` block validates; unknown keys rejected (`extra="forbid"`).
- [ ] Defaults match the spec (`session_memory_estimate_mib=100`, `max_turns=20`).
- [ ] `schemas/agent.schema.json` regenerated to include the `openai` block; schema still validates.
**Verification:** `tests/unit/models/test_openai_config.py` (field defaults, validation);
`make type-check`; sample loads.
**Dependencies:** None
**Files:** `src/holodeck/models/openai_config.py`, `src/holodeck/models/agent.py`,
`schemas/agent.schema.json`
**Scope:** M

#### Task A2: Backend consumes `OpenAIConfig`; collect-all-errors validation entrypoint
**Description:** Thread `agent.openai` into `OpenAIAgentsBackend`. Pass `max_turns` to
`Runner.run(..., max_turns=...)`. Add `validate_openai_agents(agent)` to
`lib/backends/validators.py` that runs credential preflight (reusing the MVP `_build_model`
checks) + config consistency (`allowed ∩ disallowed` empty, safety-gate references) in a single
pass, surfacing all errors together (FR-110).
**Acceptance criteria:**
- [ ] `max_turns` from `openai.max_turns` reaches `Runner.run`; default `20` when unset.
- [ ] Missing-credential and conflicting-tool errors are collected and raised together.
- [ ] No SDK import occurs at module import time (SC-005 preserved).
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (max_turns wiring);
`test_validators.py` (collect-all-errors). `make test-unit`.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`,
`src/holodeck/lib/backends/validators.py`
**Scope:** S

### Checkpoint A — Config foundation
- [ ] `openai:` block validates; backend reads sizing/turns; full unit suite green; schema valid.

---

### Phase B — Native tool adapters (vectorstore + hierarchical_document)

#### Task B1: Vectorstore + hierarchical_document adapters
**Description:** In `openai_agents_tool_adapters.py`, translate `VectorstoreTool` and
`HierarchicalDocumentToolConfig` into SDK `FunctionTool`s that wrap the same `.search()` callables
the Claude adapter uses (`lib/backends/tool_adapters.py`), reusing
`tool_initializer.create_embedding_service`. Remove these two from the "unsupported type"
`ConfigError` path. Tool name + `{query: str}` schema match the Claude adapter
(`{name}_search`).
**Acceptance criteria:**
- [ ] A `type: vectorstore` tool loads, initialises its store, and is invocable by the agent loop;
      results reach the model (mocked run + a creds-gated live check).
- [ ] A `type: hierarchical_document` tool behaves identically (results joined `\n---\n`).
- [ ] Embedding provider validation (`validate_embedding_provider`) still fires when these tools
      are present.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (both types,
mocked search). RAG init path unaffected.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`
**Scope:** M

### Checkpoint B — RAG tools work
- [ ] Vectorstore + hier-doc tools answer a grounded query on an `openai_agents` agent (creds-gated);
      tool-init endpoints (already backend-agnostic) smoke green.

---

### Phase C — MCP transports (spec 027)

#### Task C1: MCP adapter (stdio / sse / http) → `agents.mcp.*`
**Description:** Translate `MCPTool` configs into `agents.mcp.MCPServerStdio` /
`MCPServerSse(params={url, headers})` / `MCPServerStreamableHttp(params={url, headers})`. Reuse the
env/header `${VAR}` substitution + relative-arg resolution from `mcp_bridge.py`. `transport:
websocket` → skip with the spec-027 warning. `allowed_tools` subset →
`agents.mcp.create_static_tool_filter(allowed_tool_names=[...])`. Return MCP servers as a separate
list so `build_sdk_tools` hands them to `Agent(mcp_servers=[...])`. Node.js detection already
exists (`validators.agent_needs_nodejs`); no change.
**Acceptance criteria:**
- [ ] stdio/sse/http each produce the right SDK class with substituted url/headers/env (mocked).
- [ ] `transport: websocket` is skipped with a warning; load does not fail.
- [ ] An `allowed_tools` subset yields a static tool filter.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (per-transport).
`make type-check`.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py` (may extract
`openai_agents_mcp.py` if it grows past ~150 lines)
**Scope:** M

---

### Phase D — Subagents (handoffs) + skill tools

#### Task D1: `openai.agents` → SDK sub-Agents + parent `handoffs`
**Description:** Add `agents: dict[str, OpenAISubagentSpec] | None` to `OpenAIConfig`
(`OpenAISubagentSpec`: `description`, `prompt`/`prompt_file`, `tools`, `model`). Build each as an
SDK `Agent(name, instructions, handoff_description, tools, model)` and set the parent's
`handoffs=[...]`. Auto-prepend `agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX` unless
`skip_recommended_prefix: true`. Validate `model`: `inherit` → parent model; reject Claude literals
(`sonnet|opus|haiku`); allow any other string. `tools: null` → inherit parent tools.
**Acceptance criteria:**
- [ ] Three subagents become three `Agent`s on `handoffs`; prefix prepended.
- [ ] `model: sonnet|opus|haiku` fails load; `inherit` uses the parent model; arbitrary id allowed.
- [ ] A subagent with no `tools` inherits the parent's resolved tools.
**Verification:** `tests/unit/lib/backends/test_openai_agents_subagents.py`.
**Dependencies:** A1, B1, C1 (tool resolution must exist to inherit)
**Files:** `src/holodeck/models/openai_config.py`,
`src/holodeck/lib/backends/openai_agents_subagents.py`,
`src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

#### Task D2: Skill tool → handoff target
**Description:** Translate `SkillTool` (inline `instructions`/`description`/`allowed_tools` and
file-based SKILL.md) into a handoff-target `Agent` (same machinery as D1). `allowed_tools`
restricts the skill agent's tool scope. Remove `skill` from the unsupported-type `ConfigError`.
**Acceptance criteria:**
- [ ] Inline and file-based skills each become a handoff `Agent` with matching instructions.
- [ ] `allowed_tools` scopes the skill agent's tools.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (skill happy paths).
**Dependencies:** D1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/openai_agents_subagents.py`
**Scope:** S

#### Task D3: Handoff `ToolEvent`s for AG-UI (FR-006)
**Description:** During streaming, map `AgentUpdatedStreamEvent` to `subagent_message` /
`parent_link` `ToolEvent`s so the AG-UI panel renders handoffs identically to Claude. Push events
onto the serve `tool_event_queue` when present (real-time path); fall back to post-hoc otherwise.
**Acceptance criteria:**
- [ ] A handoff run emits `subagent_message` + `parent_link` events (mocked stream).
- [ ] `agui.py` consumes them with no protocol change.
**Verification:** `tests/unit/serve/` or `test_openai_agents_subagents.py` event-shape test.
**Dependencies:** D1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

### Checkpoint D — Handoffs + skills
- [ ] A multi-agent handoff scenario runs; skill tool routes; AG-UI shows subagent events. Suite green.

---

### Phase E — YAML hooks (spec 028)

#### Task E1: `openai.hooks` model + observation actions
**Description:** Add `hooks` to `OpenAIConfig` (event + matcher + action shape mirroring spec 028).
New `openai_agents_hooks.py` builds `AgentHooks`/`RunHooks` from the config: `PreToolUse` /
`PostToolUse` / `PostToolUseFailure` with `log|notify|script` → `on_tool_start`/`on_tool_end`
(observation-only); `Stop` → `on_end`; `Notification` → `on_llm_start`/`on_llm_end`; `SessionStart`
→ invoked at `create_session()`.
**Acceptance criteria:**
- [ ] Each event fires its `log`/`notify`/`script` action at the mapped lifecycle point (mocked).
- [ ] Hook chain ordering: internal hooks (tool tracking, redaction) before user hooks; declaration
      order preserved (FR-051).
**Verification:** `tests/unit/lib/backends/test_openai_agents_hooks.py`.
**Dependencies:** A1
**Files:** `src/holodeck/models/openai_config.py`,
`src/holodeck/lib/backends/openai_agents_hooks.py`,
`src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

#### Task E2: `reject` (→ approval/guardrail) + `modify` (inert)
**Description:** `PreToolUse` `action: reject` → generate `needs_approval=...` on the matched
function tool returning the rejection synchronously; if the matched tool is a hosted tool without
`needs_approval` support, fail load with a clear error. Entry-agent `input_guardrail` synthesis for
input-matched rejects. `action: modify` → load succeeds with a `not yet supported on openai_agents`
warning (inert).
**Acceptance criteria:**
- [ ] A `reject` on a function tool blocks the call with the configured message.
- [ ] A `reject` on an unsupported hosted tool fails load with a clear error.
- [ ] A `modify` action loads with a warning and has no runtime effect.
**Verification:** `tests/unit/lib/backends/test_openai_agents_hooks.py` (reject + modify paths).
**Dependencies:** E1, G1 (hosted-tool reject error needs the hosted-tool path)
**Files:** `src/holodeck/lib/backends/openai_agents_hooks.py`
**Scope:** M

### Checkpoint E — Hooks
- [ ] log/notify/script observation works; reject gates a tool; modify warns; ordering correct.

---

### Phase F — Spec-026 config mappings (US4)

#### Task F1: `effort` → `ModelSettings(reasoning=...)`
**Description:** Map `openai.effort` to `ModelSettings(reasoning=Reasoning(effort=<value>))` for
reasoning models; `max` → clamp to `high` with a load warning (FR-030/031). Fold into
`_build_model_settings`.
**Acceptance criteria:**
- [ ] `effort: high` + `gpt-5` → `reasoning.effort="high"`.
- [ ] `effort: max` → warning + `effort="high"`.
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (settings build).
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

#### Task F2: `disallowed_tools` config-time filter
**Description:** Remove named tools from the resolved `Agent.tools` + `mcp_servers` at build time;
`allowed ∩ disallowed` non-empty → load fail (FR-034). For hosted tools, refuse to construct a
disallowed one.
**Acceptance criteria:**
- [ ] A disallowed tool is absent from the built agent.
- [ ] A name in both allow + disallow fails load with the spec message.
**Verification:** `tests/unit/lib/backends/test_openai_agents_permissions.py`.
**Dependencies:** A1, B1, C1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

#### Task F3: `max_budget_usd` → cost-accountant `RunHooks`
**Description:** New `openai_agents_cost.py`: a `RunHooks` that accumulates spend from run usage ×
bundled per-model price table on each LLM end; on exhaustion raise `BackendBudgetExceededError`
(new error) carrying partial response + accumulated cost (FR-032). Unknown model → warning +
no-op (degrade, don't crash).
**Acceptance criteria:**
- [ ] A query exceeding the budget aborts with `BackendBudgetExceededError` + partial response.
- [ ] An unknown model logs a warning and does not enforce.
**Verification:** `tests/unit/lib/backends/test_openai_agents_cost.py`.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_cost.py`,
`src/holodeck/lib/errors.py`, `openai_agents_backend.py`
**Scope:** M

#### Task F4: `fallback_model` → wrapping `Model` provider
**Description:** New wrapping `Model` (or `ModelProvider`) that catches the retryable set (429,
5xx) and re-issues against `fallback_model`; both attempts visible in the trace (FR-033). Pinned to
function/MCP tools; documented caveat for hosted tools that need Responses on the fallback.
**Acceptance criteria:**
- [ ] A primary 429 routes the retry to the fallback model; a non-retryable error propagates.
- [ ] Both attempts appear in the trace.
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (fallback path, mocked).
**Dependencies:** A1, F1 (shares model-build)
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py` (or `openai_agents_fallback.py`)
**Scope:** M

### Checkpoint F — Config additions
- [ ] effort/disallowed/budget/fallback each behave per their FR; suite green.

---

### Phase G — Hosted tools (US5)

#### Task G1: `HostedTool` model + 6 tool factories
**Description:** Add `HostedTool` (`type: hosted`, `name` selecting the SDK class, `params` kwargs)
to `ToolUnion` in `models/tool.py`. Factory builds `WebSearchTool` / `FileSearchTool` /
`CodeInterpreterTool` / `ImageGenerationTool` / `ComputerTool` / `HostedMCPTool`. Allowed on both
providers (Decision 4); a tool unavailable on the live Azure resource surfaces a runtime error with
a clear hint (not a config-load block).
**Acceptance criteria:**
- [ ] `type: hosted, name: WebSearchTool` → `WebSearchTool()`; `FileSearchTool` passes
      `vector_store_ids`/`max_num_results`.
- [ ] Unknown hosted name → clear config error.
- [ ] On `azure_openai`, hosted tools load; a runtime unsupported error is propagated verbatim.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (per hosted tool).
**Dependencies:** A1
**Files:** `src/holodeck/models/tool.py`, `schemas/agent.schema.json`,
`src/holodeck/lib/backends/openai_agents_tool_adapters.py`
**Scope:** M

#### Task G2: Safety gate for `CodeInterpreterTool` / `ComputerTool` (P1b)
**Description:** Auto-disallow `CodeInterpreterTool` and `ComputerTool` unless
`openai.i_understand_this_is_unsafe: true`; loading without the opt-in emits the canonical
migration error (FR-083). Function tools' `needs_approval` gates are never weakened (FR-084).
**Acceptance criteria:**
- [ ] Declaring `CodeInterpreterTool` without the opt-in fails load with the canonical error.
- [ ] With the opt-in, it constructs.
**Verification:** `tests/unit/lib/backends/test_openai_agents_permissions.py`.
**Dependencies:** G1, A1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/validators.py`
**Scope:** S

### Checkpoint G — Hosted tools
- [ ] All 6 hosted tools declarable; safety gate enforced; Azure load works (runtime-gated).

---

### Phase H — Tracing (US7)

#### Task H1: OTel-mirroring `TracingProcessor` + provider switch
**Description:** New `openai_agents_tracing.py`: a `TracingProcessor` that mirrors SDK trace events
into the OTel pipeline, registered via `agents.add_trace_processor(...)`. `provider: openai` keeps
the platform.openai.com upload; `provider: azure_openai` calls `set_tracing_disabled(True)` **after**
registration (the MVP already disables Azure upload — fold it here). Add
`observability.disable_provider_tracing: bool` (default `False`) to suppress the upload regardless
of provider. Verify `RedactingSpanProcessor` scrubs the new spans (FR-088 — expected no code).
**Acceptance criteria:**
- [ ] `provider: openai` → OTel mirror active AND upload retained (unless override).
- [ ] `provider: azure_openai` → OTel only; upload disabled after registration.
- [ ] `disable_provider_tracing: true` suppresses upload for either provider.
- [ ] A credential-shaped `tool.output.*` span attribute is redacted before export.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tracing.py`.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_tracing.py`,
`src/holodeck/models/observability.py`, `openai_agents_backend.py`
**Scope:** M

---

### Phase I — Serve / deploy parity + P1a / P2a hardening (US2 / US6)

#### Task I1: Serve active-turn cap, 429, readiness, config echo
**Description:** In `serve/server.py`, derive an active-turn semaphore for `openai_agents` from
`openai.max_concurrent_sessions` (explicit) or `floor(memory_mib / session_memory_estimate_mib)`
(default 100). Overflow → 429 + `Retry-After` + problem+json `type:
…/session-cap-exceeded` (FR-081). `/ready` gates on tool-init completion (already wired); add the
backend credential preflight to serve startup; add an "OpenAI Agents" resolved-config echo section.
**Acceptance criteria:**
- [ ] Concurrent in-flight turns are capped at the resolved value; overflow → 429 + Retry-After.
- [ ] Idle sessions do not consume slots (cap is on active turns, FR-014).
- [ ] Startup echoes the resolved cap + derivation; missing creds exit non-zero at startup (FR-012).
**Verification:** `tests/unit/serve/test_session_semaphore_openai_agents.py`; serve smoke.
**Dependencies:** A2
**Files:** `src/holodeck/serve/server.py`, `src/holodeck/cli/commands/serve.py`
**Scope:** M

#### Task I2: Dockerfile pure-Python branch (P2a)
**Description:** `deploy/dockerfile.py` + the deploy CLI already gate Node.js on
`agent_needs_nodejs` (stdio MCP only). Confirm the `openai_agents` path emits a **pure-Python**
image by default and only installs Node when an MCP stdio server needs it (FR-011/085). Corpus
`chmod a-w` + `/tmp` / `/var/holodeck/work` EmptyDir already exist — verify they apply.
**Acceptance criteria:**
- [ ] An `openai_agents` agent with no Node-MCP → Dockerfile has no `apt-get install nodejs`.
- [ ] An agent with a Node stdio MCP → Node is installed.
- [ ] Corpus dirs read-only; scratch dirs writable.
**Verification:** `tests/unit/deploy/test_dockerfile_generation_openai_agents.py`.
**Dependencies:** A1
**Files:** `src/holodeck/deploy/dockerfile.py` (likely verify-only),
`src/holodeck/cli/commands/deploy.py`
**Scope:** S

#### Task I3: ACA default sizing 1 CPU / 1 GiB + provider-aware session-cap echo (P1a)
**Description:** In `deploy/deployers/azure_containerapps.py`, default `cpu=1.0` / `memory=1Gi`
when `model.provider ∈ {openai, azure_openai}` (vs Claude's 2 GiB), and make the resolved
session-cap echo provider-aware (`floor(memory_mib / 100)` for openai_agents) instead of the
Claude `cpu*2` formula (FR-080).
**Acceptance criteria:**
- [ ] An openai_agents deploy defaults to 1 CPU / 1 GiB; echo reports "10 (derived from 1024 MiB @
      100 MiB/session)".
- [ ] Claude sizing is unchanged.
**Verification:** `tests/unit/deploy/test_aca_template_openai_agents.py`.
**Dependencies:** A1
**Files:** `src/holodeck/deploy/deployers/azure_containerapps.py`,
`src/holodeck/models/deployment.py`
**Scope:** S

### Checkpoint I — Serve/deploy parity
- [ ] `holodeck serve` + `deploy build` work on an openai_agents agent; pure-Python image; 429 cap;
      /health + /ready behave; sizing echo correct.

---

### Phase J — P2b credential redaction + subprocess scrub (US6)

#### Task J1: Default function-tool credential-redaction decorator
**Description:** New default-on decorator in `openai_agents_hooks.py` wrapping each `@function_tool`
at registration to scrub the 5 credential patterns (anthropic key, AWS access key, GitHub token,
JWT, Bearer) from **return values** before the SDK reads them (the only honest path — `on_tool_end`
is observation-only). Opt-out: `openai.disable_default_hooks: true` with the loud warning. Reuse
`redact_credentials()` from `claude_hooks.py`.
**Acceptance criteria:**
- [ ] A tool returning `sk-ant-api03-…` is seen by the model as `[REDACTED:anthropic-key]`.
- [ ] Opt-out disables it with a warning; OTel redaction (J2) still runs.
**Verification:** `tests/unit/lib/backends/test_openai_agents_hooks.py` (redaction + opt-out).
**Dependencies:** B1, C1 (wraps the resolved function tools)
**Files:** `src/holodeck/lib/backends/openai_agents_hooks.py`,
`src/holodeck/lib/backends/openai_agents_tool_adapters.py`
**Scope:** M

#### Task J2: Subprocess env scrub + OTel-redaction verification
**Description:** When the agent has MCP stdio servers or function tools that shell out, default-on a
subprocess env scrub stripping OpenAI/Azure/Anthropic credential vars from children; opt-out
`openai.disable_subprocess_env_scrub: true` (FR-089). Verify `RedactingSpanProcessor` already
covers openai_agents spans (FR-088 — expected no code beyond a test).
**Acceptance criteria:**
- [ ] A spawned MCP stdio child does not inherit the scrubbed credential vars.
- [ ] An openai_agents span's `tool.output.*` credential value is redacted (regression test).
**Verification:** `tests/unit/lib/backends/test_openai_agents_hooks.py`;
`tests/unit/lib/backends/test_otel_redaction.py` (add openai case).
**Dependencies:** C1
**Files:** `src/holodeck/lib/backends/openai_agents_hooks.py` (or a small subprocess wrapper)
**Scope:** S

### Checkpoint J — Hardening P2b
- [ ] Synthetic prompt-injection test (SC-009) passes: model sees redacted value AND span is scrubbed.

---

### Phase K — Sample, integration smokes, docs

#### Task K1: `sample/financial-assistant/openai`
**Description:** Copy the Claude/SK financial-assistant sample, change only `model.provider` →
`azure_openai` (or `openai`) + `model.name`, keep vectorstore + function tools. Proves SC-001.
**Acceptance criteria:**
- [ ] `holodeck test` on the sample runs the full tool loop (creds-gated).
**Verification:** Manual creds-gated run; sample loads in CI without creds.
**Dependencies:** B1, C1
**Files:** `sample/financial-assistant/openai/**`
**Scope:** S

#### Task K2: Creds-gated integration smokes
**Description:** Add integration tests (skip without creds): tool-init endpoints, a hosted-tool
turn, a handoff scenario, and the redaction e2e. Mirrors the spec's test list (minus P3/sandbox).
**Acceptance criteria:**
- [ ] Each skips cleanly without creds; passes with Azure creds.
**Verification:** `make test-integration` with creds; skip-clean without.
**Dependencies:** D1, E2, G1, J1
**Files:** `tests/integration/test_openai_agents_*.py`
**Scope:** M

#### Task K3: Docs + per-backend semantics matrix
**Description:** Backend docs for `openai_agents` (`openai:` block reference); hooks per-backend
semantics matrix (`modify` inert; `reject` via approval/guardrail); note in
`docs/security/aca-limitations.md` that **P3 hardened and US8 sandbox are deferred to follow-up
specs**; migration note that hooks/subagents for this backend live under `openai:`, not `claude:`.
**Acceptance criteria:**
- [ ] Docs describe the `openai:` block, hosted tools, tracing behaviour, and the deferred surfaces.
**Verification:** Docs build / link check; grep for stale `openai_agents:` block references.
**Dependencies:** all prior
**Files:** `docs/api/backends.md`, `docs/guides/*`, `docs/security/aca-limitations.md`, `AGENTS.md`
**Scope:** S

### Checkpoint K — Complete
- [ ] `make format lint type-check security` clean; `make test` (parallel) green.
- [ ] SC-001/003/004/006/007/009 met by tests; sample runs end-to-end (creds-gated).
- [ ] Review with user.

---

## Dependency graph (summary)

```
A1 (openai: config) ──┬── A2 (backend consumes + validate)
                      ├── B1 (vectorstore/hier-doc) ──┐
                      ├── C1 (MCP) ──────────────────┤
                      │                               ├── D1 (subagents) ── D2 (skill) ── D3 (AG-UI events)
                      ├── E1 (hooks) ── E2 (reject/modify) ←─ needs G1
                      ├── F1 (effort) F2 (disallowed) F3 (budget) F4 (fallback)
                      ├── G1 (hosted) ── G2 (safety gate)
                      ├── H1 (tracing)
                      ├── A2 → I1 (serve) ; A1 → I2 (dockerfile) I3 (sizing)
                      └── B1/C1 → J1 (redaction) ; C1 → J2 (env scrub)
K1/K2/K3 depend on the feature phases they exercise.
```

**Parallelisable after A1:** B, C, F, G, H, I are largely independent. E2 needs G1. D needs B/C
(tool inheritance). J needs B/C. Land in separate PRs per phase.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `build_sdk_tools` signature change (now returns tools + mcp_servers + handoffs) ripples into the backend | Med | Single small refactor in A2/B1; covered by existing adapter tests before adding types |
| Hosted tools on Azure fail at runtime in confusing ways (Decision 4) | Med | Propagate the SDK error verbatim + a "this hosted tool may be unavailable on your Azure resource" hint; document in K3 |
| `max_budget_usd` price table staleness | Low | Bundled constant + unknown-model warning (no crash); documented update path |
| `reject` hook → `needs_approval` semantics differ subtly from Claude's PreToolUse deny | Med | Document in the per-backend matrix (K3); cover entry-agent-only guardrail caveat (spec edge case) |
| Cost/fallback wrapping `Model` interferes with Responses/reasoning models | Med | Pin fallback to function/MCP; unit-test against a mocked reasoning model; caveat in docs |
| Serve session-cap divergence from Claude (active turns vs open sessions) confuses operators | Low | Echo the derivation at startup (I1); document the in-process rationale |

## Out of scope (deferred to follow-up specs)

- **P3 hardened profile** — Envoy credential sidecar, `deployment.security_profile`, egress
  allowlist, `OPENAI_BASE_URL`/`AZURE_OPENAI_ENDPOINT` rewriting (FR-090…093). Net-new cross-backend
  infra; own spec covering Claude + openai_agents.
- **US8 sandbox mode** — `agent_mode: sandbox`, `SandboxAgent`/`Manifest`, remote sandbox clients
  (FR-094…099). Re-spec against the real `agents.sandbox.*` surface.
- **`modify` hook action implementation** — v1 is inert + warning (spec Open Question 2).
- **Live `provider: openai` validation** — coded to spec, validated via mocks until a key exists
  (carried over from the MVP).
- **Cross-backend hooks/subagents namespace unification** — this plan keeps `openai:` and `claude:`
  separate by Decision 1.

## Open questions (resolved in this plan)

- Namespacing → new top-level `openai:` block (Decision 1).
- P3 / sandbox → deferred (Decisions 2, 3).
- Azure hosted tools → allowed, runtime-gated (Decision 4).
- Price-table source → bundled constant + unknown-model warning (spec Open Q3, v1).
- `modify` action → inert + warning (spec Open Q2, v1).
