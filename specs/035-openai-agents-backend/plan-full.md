# Implementation Plan: OpenAI Agents SDK Backend — Full Parity (post-MVP)

**Spec:** `specs/035-openai-agents-backend/spec.md`
**Builds on:** `plan-mvp.md` (shipped — function tools, real streaming, routing flip, SK agent-path carve)
**Scope of this plan:** Everything in spec 035 that the MVP deliberately left out, **except** surfaces
deferred to their own specs by decision below: the P3 hardened/Envoy profile, US8 sandbox mode, and
computer-use (`ComputerTool`).
**Status:** Revised through two adversarial review cycles against the installed SDK source
(`openai-agents==0.17.4`, 2026-06-10). Cycle 1 re-based Claude-by-analogy mappings onto the SDK's
native idioms (tool guardrails, trace processors, `RunConfig`). Cycle 2 fixed coverage and
feasibility gaps the rewrite introduced or inherited (MCP tools are SDK-built `FunctionTool`s with
no guardrail attachment point; the MVP already calls `set_tracing_disabled`; `SkillTool` does not
exist in the codebase despite the spec presuming it; run-level input guardrails are tripwire-only).
All SDK claims below were verified against `.venv/.../site-packages/agents` source.

---

## Overview

The MVP proved one vertical slice: a `provider: openai` / `azure_openai` agent runs the OpenAI
Agents SDK `Runner` loop, calls Python **function** tools, and streams under `holodeck chat`, with
default routing flipped and the SK agent path removed. This plan delivers the **rest of the
backend's parity surface**:

- the remaining HoloDeck tool types: vectorstore, hierarchical_document, MCP — plus skill tools,
  whose `SkillTool` model is **net-new** (the spec presumes it exists; it does not — see
  reconciliations);
- subagents (handoffs) and YAML hooks;
- the spec-026 config mappings (`effort`, `max_budget_usd`, `fallback_model`, `disallowed_tools`);
- hosted tools (`type: hosted`);
- structured output + `thinking` parity (FR-004);
- `RunConfig` plumbing (trace sensitivity, workflow naming);
- tracing dual-write;
- serve/deploy parity and the spec-034 hardening phases **P1a/P1b/P2a/P2b**.

It is organised as **vertical, feature-complete slices** (model + adapter + backend wiring +
tests per slice) so each phase leaves `holodeck chat`/`test`/`serve` working.

## Decisions

Confirmed with user 2026-06-09 (1–4) and 2026-06-10 (3 revised; 5–7 added after the SDK review):

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
3. **US8 sandbox mode (`agent_mode: sandbox`) → deferred to a follow-up spec, as a scoping
   decision.** *(Rationale corrected 2026-06-10: the first draft claimed `UnixLocalSandboxClient`
   does not exist in 0.17.4 — that is false; it exists at `agents.sandbox.sandboxes.unix_local`
   and imports on non-Windows platforms.)* The honest reason for deferral: sandbox mode is a
   large, security-sensitive surface (manifests, sandbox runtimes, lifecycle management) that
   deserves its own spec and threat model rather than riding along in a parity plan. FR-094…FR-099
   are out of this plan. The `openai.i_understand_this_is_unsafe` gate still ships (it gates
   `CodeInterpreterTool`).
4. **Hosted tools allowed on Azure, runtime-gated.** The MVP put Azure on the Responses API
   (`/openai/v1`), so the spec's blanket config-load block (US5 scenario 3) is relaxed: hosted
   tools are accepted on `azure_openai`; if a specific tool isn't available on the resource, the
   SDK surfaces a runtime error that HoloDeck propagates verbatim with a clear hint. *(The spec's
   risk table cites FR-091 for this block — a mis-citation; FR-091 is the P3 egress-allowlist FR,
   deferred under Decision 2.)*
5. **`reject` + credential redaction → native per-tool guardrails, scoped to HoloDeck-built
   tools.** The SDK's `FunctionTool.tool_input_guardrails` / `tool_output_guardrails`
   (`ToolGuardrailFunctionOutput.reject_content(message)`) are the implementation for hook
   `reject` actions and for default credential redaction — synchronous, message-bearing, run
   continues. Coverage is **function/vectorstore/hier-doc/skill tools only**: hosted tools execute
   server-side, and MCP-server tools become SDK-built `FunctionTool`s HoloDeck never touches (see
   reconciliations). The first draft's `needs_approval` synthesis is dropped: `needs_approval`
   **interrupts** the run (`RunState.approve()/.reject()` + re-run), which is a human-in-the-loop
   feature, not an auto-reject — deferred to a follow-up spec if wanted.
6. **`modify` hook action stays inert in v1** (load warning, no runtime effect) to keep hook
   semantics symmetric with the Claude backend. Note for later: SDK output guardrails *can*
   replace the model-visible tool output, so modify-on-output is implementable when both backends
   are ready to support it.
7. **`ComputerTool` rejected at config load, computer-use deferred.** `ComputerTool` requires a
   live Python `Computer`/`AsyncComputer` implementation (`tool.py` — required `computer:` field);
   it cannot be constructed from YAML params. `type: hosted, name: ComputerTool` fails load with a
   clear "requires a computer harness — not yet supported" error. Computer-use gets its own
   follow-up spec (likely alongside sandbox).

## Spec ↔ reality reconciliations (verified against installed 0.17.4 source + this repo)

These differ from the spec text and/or earlier drafts; the plan follows reality:

- **No `Backend` enum / no `backend:` override / no `agent_framework` backend exist.** Routing is
  purely by `model.provider` in `BackendSelector` (`src/holodeck/lib/backends/selector.py`), and
  the `openai`/`azure_openai → openai_agents` flip already shipped in the MVP. → **FR-001 and FR-007
  are already satisfied**; US1 scenario 4 (explicit AF opt-out) is **dropped** (no AF backend in
  this repo).
- **`SkillTool` does not exist in the codebase.** The spec (line 37, FR-070, SC-001) presumes an
  "existing HoloDeck `SkillTool` (spec 023)", but `ToolUnion`
  (`models/tool.py`) is `VectorstoreTool | FunctionTool | MCPTool | PromptTool |
  HierarchicalDocumentToolConfig`. D2 therefore **creates** the `SkillTool` model (+ schema regen)
  before translating it.
- **`PromptTool` (`type: prompt`) has no runtime adapter in any backend** — it is defined in
  `models/tool.py` and referenced nowhere else in `src/`. The spec's tool list excludes it. This
  plan keeps it out of scope but downgrades the openai adapter's unsupported-type `ConfigError`
  for `prompt` to a skip-with-warning (B1), so switching providers on an agent that declares one
  matches the de-facto status quo instead of hard-failing.
- **MCP-server tools are SDK-built `FunctionTool`s HoloDeck never sees.** `agents/mcp/util.py`
  (`to_function_tool`) converts each MCP tool to a `FunctionTool` **inside the SDK, per run**,
  with `tool_input_guardrails=None` / `tool_output_guardrails=None`. HoloDeck only passes
  `Agent(mcp_servers=[...])`. Consequence: guardrail-based `reject` (E2), credential redaction
  (J1), and adapter-wrapper failure detection (E1) **cannot cover MCP tools** in v1 — documented
  per task and in the K3 matrix.
- **Directly constructed `FunctionTool`s have no default failure handling.** The SDK's
  error-string-to-model behavior lives in the `function_tool` factory's failure-handling invoker
  (`tool.py` `_build_wrapped_function_tool`); HoloDeck constructs `FunctionTool` directly, so an
  uncaught tool exception propagates and **fails the whole run** (`run_internal/tool_execution.py`
  → `UserError`). E1's adapter wrapper must catch exceptions itself, fire `PostToolUseFailure`,
  and return an error string to the model (mirroring the factory's default).
- **Run-level `InputGuardrail` is tripwire-only.** `GuardrailFunctionOutput(tripwire_triggered)`
  halts execution via `InputGuardrailTripwireTriggered` (`guardrail.py`,
  `run_internal/guardrails.py`) — there is no message-and-continue option at the run-input level.
  Input-matched `reject` hooks therefore **abort the turn** (HoloDeck catches the exception and
  surfaces the configured message); only tool-matched rejects continue the run.
- **The shipped MVP already calls `set_tracing_disabled(True)`** on the Azure path
  (`openai_agents_backend.py` `_build_model`) — which makes the trace provider return
  `NoOpTrace`/`NoOpSpan` (`tracing/provider.py`) and would starve the H1 OTel mirror. H1 removes
  that call; A2's preflight refactor keeps `_build_model`'s credential checks side-effect-free
  (no `set_tracing_disabled` / `set_default_openai_key` during validation).
- **`MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` live under `agents.mcp.*`,** not at
  the `agents` top level. Confirmed present in 0.17.4, plus `create_static_tool_filter`.
- **`needs_approval` is an interrupt, not a synchronous reject** (`tool.py`: "the run will be
  interrupted and the tool call will need to be approved using RunState.approve() or rejected
  using RunState.reject()"). Auto-reject maps to `tool_input_guardrails` instead (Decision 5).
  This also covers the MCP server classes' `require_approval` (it resolves into the generated
  tool's `needs_approval`) — so MCP `reject` cannot be implemented that way without HITL.
- **`on_tool_start`/`on_tool_end` fire for *local* tools only** (lifecycle docstrings; invocation
  sites are in `run_internal/tool_execution.py` / `tool_actions.py`). Hosted tools execute
  server-side and never trigger AgentHooks/RunHooks tool events or guardrails.
- **`ReasoningEffort` accepts `"none" | "minimal" | "low" | "medium" | "high" | "xhigh"`** in the
  installed client. Spec FR-030/031's "clamp `max` to `high`" was written against an older value
  set; this plan maps `max → "xhigh"` (documented deviation).
- **Reasoning summaries must be requested.** `ReasoningItem`s carry summaries only when
  `Reasoning(summary=...)` is set on the request; effort alone yields empty summaries. F5 sets
  `summary="auto"` for reasoning models so `thinking` has content.
- **`AgentOutputSchemaBase` is abstract** and the SDK's only concrete implementation
  (`AgentOutputSchema`) requires a Python type. JSON-schema-dict-driven output (HoloDeck's
  `response_format`) needs a small HoloDeck subclass (F5 writes it).
- **`RunConfig.trace_include_sensitive_data` defaults to True** (env-derived), uploading raw tool
  inputs/outputs with provider traces. A security-focused plan must set it explicitly (Task A3
  ties it to `observability.traces.capture_content`).
- **`CodeInterpreterTool` / `ImageGenerationTool` / `HostedMCPTool` require structured
  `tool_config` objects** (openai-types `CodeInterpreter` / `ImageGeneration` / `Mcp`), not flat
  kwargs — G1 factories build the nested objects from YAML `params`.
- **`HostedMCPTool` approval requires `require_approval` in its `Mcp` tool_config** — the
  `on_approval_request` callback only fires for tools the provider was told need approval. E2's
  hosted-MCP reject mapping must set both halves.
- **The SDK has runner-managed retry** (`ModelSettings.retry: ModelRetrySettings(max_retries,
  backoff, policy)`) that fires on the same 429/5xx class as F4's fallback — F4 must define the
  ordering (retries exhaust first, then fallback).
- **FR-084 reinterpreted.** The FR says function tools' `needs_approval` gates are never weakened;
  v1 ships no `needs_approval` at all (Decision 5), so the FR is satisfied vacuously, and the
  enforced invariant becomes: default guardrail gates (J1 redaction, E2 rejects) are never
  weakened by hosted-tool or permission config.
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
- **One guardrail module** (`openai_agents_guardrails.py`) owns both hook-`reject` input guardrails
  (E2) and the default credential-redaction output guardrail (J1) — same mechanism, one
  attachment point at FunctionTool build time, for **HoloDeck-built tools only** (see
  reconciliations for the MCP gap). Observation hooks (`log`/`notify`/`script`) use
  `AgentHooks`/`RunHooks`; failure detection lives in the adapter wrapper.
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

#### Task A2: Backend consumes `OpenAIConfig`; side-effect-free validation entrypoint
**Description:** Thread `agent.openai` into `OpenAIAgentsBackend`. Pass `max_turns` to
`Runner.run(..., max_turns=...)`. Add `validate_openai_agents(agent)` to
`lib/backends/validators.py` that runs credential preflight + config consistency
(`allowed ∩ disallowed` empty, safety-gate references) in a single pass, surfacing all errors
together (FR-110). The preflight **extracts** `_build_model`'s credential checks into a
side-effect-free helper — validation must not trigger `_build_model`'s global mutations
(`set_tracing_disabled`, `set_default_openai_key`).
**Acceptance criteria:**
- [ ] `max_turns` from `openai.max_turns` reaches `Runner.run`; default `20` when unset.
- [ ] Missing-credential and conflicting-tool errors are collected and raised together.
- [ ] Running validation leaves SDK global state untouched (no tracing/key side effects).
- [ ] No SDK import occurs at module import time (SC-005 preserved).
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (max_turns wiring);
`test_validators.py` (collect-all-errors, no-side-effect). `make test-unit`.
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`,
`src/holodeck/lib/backends/validators.py`
**Scope:** S

#### Task A3: `RunConfig` plumbing (trace sensitivity, workflow identity)
**Description:** Build a `RunConfig` for every `Runner.run(...)` call:
`workflow_name=agent.name`, `group_id=<session id>` for session-based runs (`invoke_once` has no
session and carries `workflow_name` only), `trace_metadata` carrying the HoloDeck run context,
and — security-critical — `trace_include_sensitive_data = observability.traces.capture_content`
(the SDK default is **True** via env, which would upload raw tool inputs/outputs to
platform.openai.com even while J1 redacts what the model sees). `handoff_input_filter` /
`nest_handoff_history` are left at SDK defaults and explicitly documented as unmapped in v1 (see
Out of scope).
**Acceptance criteria:**
- [ ] Every run carries `workflow_name`; session runs additionally carry `group_id`.
- [ ] `capture_content: false` (default) → `trace_include_sensitive_data=False` regardless of env.
- [ ] `capture_content: true` → sensitive payloads included.
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (RunConfig build).
**Dependencies:** A1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

### Checkpoint A — Config foundation
- [ ] `openai:` block validates; backend reads sizing/turns; RunConfig wired; validation
      side-effect-free; full unit suite green; schema valid.

---

### Phase B — Native tool adapters (vectorstore + hierarchical_document)

#### Task B1: Vectorstore + hierarchical_document adapters; `prompt` downgraded to warning
**Description:** In `openai_agents_tool_adapters.py`, translate `VectorstoreTool` and
`HierarchicalDocumentToolConfig` into SDK `FunctionTool`s that wrap the same `.search()` callables
the Claude adapter uses (`lib/backends/tool_adapters.py`), reusing
`tool_initializer.create_embedding_service`. Remove these two from the "unsupported type"
`ConfigError` path. Tool name + `{query: str}` schema match the Claude adapter
(`{name}_search`). Additionally, downgrade `type: prompt` from `ConfigError` to skip-with-warning
(no backend has a runtime adapter for it — see reconciliations).
**Acceptance criteria:**
- [ ] A `type: vectorstore` tool loads, initialises its store, and is invocable by the agent loop;
      results reach the model (mocked run + a creds-gated live check).
- [ ] A `type: hierarchical_document` tool behaves identically (results joined `\n---\n`).
- [ ] A `type: prompt` tool is skipped with a warning; load does not fail.
- [ ] Embedding provider validation (`validate_embedding_provider`) still fires when these tools
      are present.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (both types,
mocked search; prompt warning). RAG init path unaffected.
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
exists (`validators.agent_needs_nodejs`); no change. Note: the SDK converts MCP tools to
`FunctionTool`s internally per run — HoloDeck cannot attach guardrails or failure wrappers to them
(reconciliations; affects E1/E2/J1 coverage).
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
Handoff-history shaping (`handoff_input_filter`, `nest_handoff_history`) stays at SDK defaults in
v1 (documented).
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

#### Task D2: `SkillTool` model (net-new) + skill → handoff target
**Description:** The spec (FR-070, SC-001) presumes an existing `SkillTool`; **none exists** (see
reconciliations). First add `SkillTool` to `models/tool.py` + `ToolUnion` (+ schema regen):
`type: skill`, inline form (`instructions`, `description`, `allowed_tools`) or file-based form
(`path` to a directory whose SKILL.md body + frontmatter supply them), per spec-023 FR-022…026 as
restated in spec-035 FR-070. Then translate it into a handoff-target `Agent` (same machinery as
D1). `allowed_tools` restricts the skill agent's tool scope.
**Acceptance criteria:**
- [ ] `SkillTool` validates in YAML (inline + file-based); schema regenerated.
- [ ] Inline and file-based skills each become a handoff `Agent` with matching instructions.
- [ ] `allowed_tools` scopes the skill agent's tools.
**Verification:** `tests/unit/models/test_tool.py` (SkillTool validation);
`tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (skill happy paths).
**Dependencies:** D1
**Files:** `src/holodeck/models/tool.py`, `schemas/agent.schema.json`,
`src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/openai_agents_subagents.py`
**Scope:** M

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

#### Task E1: `openai.hooks` model + observation actions + failure path
**Description:** Add `hooks` to `OpenAIConfig` (event + matcher + action shape mirroring spec 028).
New `openai_agents_hooks.py` builds `AgentHooks`/`RunHooks` from the config: `PreToolUse` /
`PostToolUse` with `log|notify|script` → `on_tool_start`/`on_tool_end` (observation-only — **local
tools only**, and MCP-server tools are observable here but not wrappable); `PostToolUseFailure` →
HoloDeck's adapter wrapper around the on_invoke callables it builds **catches tool exceptions
itself**, fires the configured action, and returns an error string to the model (directly
constructed `FunctionTool`s have no SDK default failure handling — an uncaught exception fails the
whole run; see reconciliations). `PostToolUseFailure` does not fire for MCP-server tools (SDK
builds their invokers) — load warning when a failure matcher targets MCP tool names. `Stop` →
`on_end`; `Notification` → `on_llm_start`/`on_llm_end`; `SessionStart` → invoked at
`create_session()`.
**Acceptance criteria:**
- [ ] Each event fires its `log`/`notify`/`script` action at the mapped lifecycle point (mocked).
- [ ] A raising function tool fires `PostToolUseFailure` and the model receives an error string;
      the run does not abort.
- [ ] Hook chain ordering: internal hooks (tool tracking, redaction) before user hooks; declaration
      order preserved (FR-051).
- [ ] A hook matcher that can only match hosted tools (or a failure matcher on MCP tool names)
      loads with a "will never fire" warning.
**Verification:** `tests/unit/lib/backends/test_openai_agents_hooks.py`.
**Dependencies:** A1
**Files:** `src/holodeck/models/openai_config.py`,
`src/holodeck/lib/backends/openai_agents_hooks.py`,
`src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

#### Task E2: `reject` (→ tool guardrails) + `modify` (inert)
**Description:** New `openai_agents_guardrails.py`. Reject mapping by target:
- **HoloDeck-built function/local tool** (function, vectorstore, hier-doc, skill) → attach a
  `tool_input_guardrail` returning `ToolGuardrailFunctionOutput.reject_content(<configured
  message>)` — synchronous, the run continues, the model sees the rejection message. (The
  guardrail context exposes `tool_name`/`tool_arguments` — sufficient for spec-028 matchers.)
- **`HostedMCPTool`** → set `require_approval` for the matched tool names inside the `Mcp`
  `tool_config` **and** wire `on_approval_request` to return a rejection with the configured
  message (both halves required — the callback only fires for tools flagged as needing approval).
- **MCP-server tools (`agents.mcp.*`)** → **fail load with a clear error**: HoloDeck cannot attach
  guardrails (SDK builds those `FunctionTool`s internally), and the server classes'
  `require_approval` resolves into `needs_approval` interrupts — i.e. HITL, excluded by
  Decision 5.
- **Other hosted tools** (WebSearch/FileSearch/CodeInterpreter/ImageGeneration — server-side, no
  guardrail path) → fail load with a clear error.
- **Input-matched rejects** → entry-agent `InputGuardrail`. This is **tripwire-only**: the turn
  aborts via `InputGuardrailTripwireTriggered`; the backend catches it and surfaces the configured
  message as the turn's error result. Distinct semantics from tool rejects (run does NOT
  continue) — documented in the K3 matrix.
`action: modify` → load succeeds with a `not yet supported on openai_agents` warning (inert,
Decision 6). **Phase ordering note:** the function-tool and input paths land with E1/B1; the
hosted-tool paths require G1 and are verified at Checkpoint G.
**Acceptance criteria:**
- [ ] A `reject` on a function tool blocks the call via input guardrail; the model receives the
      configured message; the run continues (no interrupt).
- [ ] An input-matched `reject` aborts the turn with the configured message as the error result.
- [ ] A `reject` on an MCP-server tool fails load with a clear error.
- [ ] (After G1) A `reject` on a `HostedMCPTool` sets `require_approval` + rejects via
      `on_approval_request`; a `reject` on `WebSearchTool` (et al.) fails load.
- [ ] A `modify` action loads with a warning and has no runtime effect.
**Verification:** `tests/unit/lib/backends/test_openai_agents_guardrails.py` (reject paths),
`test_openai_agents_hooks.py` (modify warning).
**Dependencies:** E1 (function/input/MCP-error paths); G1 (hosted-tool paths only)
**Files:** `src/holodeck/lib/backends/openai_agents_guardrails.py`,
`src/holodeck/lib/backends/openai_agents_hooks.py`
**Scope:** M

### Checkpoint E — Hooks (function-tool + input scope; hosted paths at Checkpoint G)
- [ ] log/notify/script observation works; failure path fires + returns error string; reject gates
      a function tool without interrupting the run; input reject aborts with message; MCP reject
      fails load clearly; modify warns; ordering correct.

---

### Phase F — Spec-026 config mappings (US4) + parity gaps

#### Task F1: `effort` → `ModelSettings(reasoning=...)`
**Description:** Map `openai.effort` to `ModelSettings(reasoning=Reasoning(effort=<value>))` for
reasoning models. `max` → `"xhigh"` (supported by the installed client's `ReasoningEffort`
literal; documented deviation from FR-030/031's stale clamp-to-`high` wording). Fold into
`_build_model_settings`.
**Acceptance criteria:**
- [ ] `effort: high` + `gpt-5` → `reasoning.effort="high"`.
- [ ] `effort: max` → `reasoning.effort="xhigh"` (no warning needed; documented mapping).
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
bundled per-model price table on each LLM end (`Usage.request_usage_entries` supports per-request
math); on exhaustion raise `BackendBudgetExceededError` (new error) carrying partial response +
accumulated cost (FR-032). Unknown model → warning + no-op (degrade, don't crash).
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
5xx) and re-issues against `fallback_model`; both attempts visible in the trace (FR-033). Define
the ordering against the SDK's runner-managed retry: `ModelSettings.retry`
(`ModelRetrySettings(max_retries, backoff, policy)`) exhausts on the **primary** model first; the
fallback wrapper engages only on the final failure (no double-fallback, no fallback mid-retry).
Pinned to function/MCP tools; documented caveat for hosted tools that need Responses on the
fallback.
**Acceptance criteria:**
- [ ] A primary 429 routes the retry to the fallback model after primary retries exhaust; a
      non-retryable error propagates.
- [ ] Both attempts appear in the trace.
- [ ] Retry/fallback ordering is unit-tested (retry exhaust → one fallback attempt).
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (fallback path, mocked).
**Dependencies:** A1, F1 (shares model-build)
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py` (or `openai_agents_fallback.py`)
**Scope:** M

#### Task F5: Structured output + `thinking` parity (FR-004)
**Description:** Wire the existing `Agent.response_format` (dict JSON schema | str path | None,
`models/agent.py`) to SDK `Agent(output_type=...)`. `AgentOutputSchemaBase` is abstract and the
SDK's concrete `AgentOutputSchema` requires a Python type, so write a small HoloDeck subclass
(`JSONSchemaOutputSchema(AgentOutputSchemaBase)`: name, json_schema, strict flag, `validate_json`
via `jsonschema`). Populate `ExecutionResult.structured_output` from `result.final_output` when an
output type is set. Populate `ExecutionResult.thinking` from `ReasoningItem`s in the run output
(the backend currently hardcodes `thinking=""` — `openai_agents_backend.py:257`) — and set
`Reasoning(summary="auto")` for reasoning models (coupled with F1's effort mapping), since
summaries are only emitted when requested. Cover both `invoke_once` and streaming.
**Acceptance criteria:**
- [ ] A `response_format` JSON schema yields `structured_output` as a parsed dict; absent →
      `None` (current behavior).
- [ ] A reasoning-model run (with summary requested) populates `thinking`; non-reasoning models
      leave it empty.
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` (output schema +
thinking extraction, mocked run); creds-gated live check in K2.
**Dependencies:** A1, F1 (shares `_build_model_settings`)
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

### Checkpoint F — Config additions + parity gaps
- [ ] effort/disallowed/budget/fallback each behave per their FR; structured output + thinking
      populated; suite green.

---

### Phase G — Hosted tools (US5)

#### Task G1: `HostedTool` model + 5 tool factories
**Description:** Add `HostedTool` (`type: hosted`, `name` selecting the SDK class, `params`) to
`ToolUnion` in `models/tool.py`. Factory builds `WebSearchTool` / `FileSearchTool` /
`CodeInterpreterTool` / `ImageGenerationTool` / `HostedMCPTool`. `CodeInterpreterTool`,
`ImageGenerationTool`, and `HostedMCPTool` require **structured `tool_config` objects**
(openai-types `CodeInterpreter` / `ImageGeneration` / `Mcp`) — the factory constructs them from
the YAML `params` mapping (e.g. container spec for code interpreter), with clear errors for
missing required sub-fields. `name: ComputerTool` → config-load error: "requires a computer
harness — not yet supported" (Decision 7). Allowed on both providers (Decision 4); a tool
unavailable on the live Azure resource surfaces a runtime error with a clear hint (not a
config-load block). Unblocks E2's hosted-tool reject paths.
**Acceptance criteria:**
- [ ] `type: hosted, name: WebSearchTool` → `WebSearchTool()`; `FileSearchTool` passes
      `vector_store_ids`/`max_num_results`.
- [ ] `CodeInterpreterTool` params build the required nested `tool_config` (container spec);
      missing container → clear config error (not a bare `TypeError`).
- [ ] Unknown hosted name → clear config error; `ComputerTool` → the Decision-7 error.
- [ ] On `azure_openai`, hosted tools load; a runtime unsupported error is propagated verbatim.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` (per hosted tool).
**Dependencies:** A1
**Files:** `src/holodeck/models/tool.py`, `schemas/agent.schema.json`,
`src/holodeck/lib/backends/openai_agents_tool_adapters.py`
**Scope:** M

#### Task G2: Safety gate for `CodeInterpreterTool` (P1b)
**Description:** Auto-disallow `CodeInterpreterTool` unless
`openai.i_understand_this_is_unsafe: true`; loading without the opt-in emits the canonical
migration error (FR-083). (`ComputerTool` is unconditionally rejected by G1, so the gate no longer
covers it.) Default guardrail gates (J1 redaction, E2 rejects) are never weakened by hosted-tool
config (FR-084 as reinterpreted — see reconciliations).
**Acceptance criteria:**
- [ ] Declaring `CodeInterpreterTool` without the opt-in fails load with the canonical error.
- [ ] With the opt-in (and a valid container `tool_config`), it constructs.
**Verification:** `tests/unit/lib/backends/test_openai_agents_permissions.py`.
**Dependencies:** G1, A1
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/validators.py`
**Scope:** S

### Checkpoint G — Hosted tools (+ E2 hosted paths)
- [ ] 5 hosted tools declarable (nested configs built from YAML); `ComputerTool` cleanly rejected;
      safety gate enforced; Azure load works (runtime-gated); E2's `HostedMCPTool` reject +
      hosted load-fail paths verified.

---

### Phase H — Tracing (US7)

#### Task H1: OTel-mirroring `TracingProcessor` + processor-list management
**Description:** New `openai_agents_tracing.py`: a `TracingProcessor` that mirrors SDK trace events
into the OTel pipeline. **Remove the MVP's `set_tracing_disabled(True)` call** in
`_build_model` (Azure path) — it makes the provider return `NoOpTrace`/`NoOpSpan` and starves
every processor, including the mirror. Processor configuration happens **once at backend
`initialize()`, before any run emits spans** (the default platform exporter is lazily registered
and `set_trace_processors` does not flush what the replaced processor already queued):
`provider: openai` → `agents.add_trace_processor(mirror)` (default platform.openai.com exporter
retained alongside the mirror); `provider: azure_openai` → `agents.set_trace_processors([mirror])`
(replaces the default exporter — spans keep flowing, nothing uploads). Add
`observability.disable_provider_tracing: bool` (default `False`): when true, use
`set_trace_processors([mirror])` regardless of provider. Sensitive-payload inclusion in uploaded
traces is governed by A3 (`trace_include_sensitive_data`). Verify `RedactingSpanProcessor` scrubs
the mirrored spans (FR-088 — expected no code).
**Acceptance criteria:**
- [ ] The `set_tracing_disabled(True)` call is gone; Azure runs deliver spans to the mirror
      (regression test against the NoOp failure mode).
- [ ] `provider: openai` → OTel mirror receives spans AND upload retained (unless override).
- [ ] `provider: azure_openai` → OTel mirror receives spans; no default exporter registered.
- [ ] `disable_provider_tracing: true` suppresses upload for either provider — mirror still
      receives spans.
- [ ] A credential-shaped `tool.output.*` span attribute is redacted before export.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tracing.py`.
**Dependencies:** A1, A3
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

#### Task J1: Default credential-redaction output guardrail
**Description:** In `openai_agents_guardrails.py` (shared with E2), attach a default-on
`tool_output_guardrail` to every **HoloDeck-built** `FunctionTool` (function, vectorstore,
hier-doc, skill) that scrubs the 5 credential patterns (anthropic key, AWS access key, GitHub
token, JWT, Bearer) from tool output **before the model sees it**: clean output → `allow()`;
pattern found → `reject_content(<redacted text>)` (the SDK replaces the model-visible tool result
with the message, and the redacted value is also what lands in the span output). Reuse
`redact_credentials()` from `claude_hooks.py`. Opt-out: `openai.disable_default_hooks: true` with
the loud warning. **Coverage note (documented + tested):** hosted tools execute server-side, and
MCP-server tool outputs flow through SDK-built `FunctionTool`s HoloDeck cannot wrap — both bypass
this redaction (K3 matrix; SC-009 scope is HoloDeck-built tools; J2's OTel-layer redaction still
covers MCP/hosted span attributes).
**Acceptance criteria:**
- [ ] A tool returning `sk-ant-api03-…` is seen by the model as `[REDACTED:anthropic-key]` (via
      the output guardrail, not return-value mutation).
- [ ] Clean outputs pass through unmodified (`allow()` path).
- [ ] Opt-out disables it with a warning; OTel redaction (J2) still runs.
**Verification:** `tests/unit/lib/backends/test_openai_agents_guardrails.py` (redaction + opt-out).
**Dependencies:** B1, E2 (shared guardrail module)
**Files:** `src/holodeck/lib/backends/openai_agents_guardrails.py`,
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
- [ ] Synthetic prompt-injection test (SC-009, HoloDeck-built-tool scope) passes: model sees
      redacted value AND span is scrubbed AND (A3) the uploaded trace excludes sensitive payloads
      by default.

---

### Phase K — Sample, integration smokes, docs

#### Task K1: `sample/financial-assistant/openai`
**Description:** Create the sample fresh — there is no in-tree financial-assistant sample to copy
(`sample/` currently holds research-agent / test-openai-sdk-agent / test-vec, and `/sample` is
**gitignored**, so nothing under it is exercised by CI). Model it on the historical
financial-assistant layout (vectorstore + function tools) with `model.provider: azure_openai` (or
`openai`) + `model.name`. Proves SC-001. Verification is manual and creds-gated only.
**Acceptance criteria:**
- [ ] `holodeck test` on the sample runs the full tool loop (creds-gated, manual).
**Verification:** Manual creds-gated run.
**Dependencies:** B1, C1
**Files:** `sample/financial-assistant/openai/**` (untracked)
**Scope:** S

#### Task K2: Creds-gated integration smokes
**Description:** Add integration tests (skip without creds): tool-init endpoints, a hosted-tool
turn, a handoff scenario, a structured-output turn (F5), and the redaction e2e. Mirrors the spec's
test list (minus P3/sandbox/computer-use).
**Acceptance criteria:**
- [ ] Each skips cleanly without creds; passes with Azure creds.
**Verification:** `make test-integration` with creds; skip-clean without.
**Dependencies:** D1, E2, F5, G1, J1
**Files:** `tests/integration/test_openai_agents_*.py`
**Scope:** M

#### Task K3: Docs + per-backend semantics matrix
**Description:** Backend docs for `openai_agents` (`openai:` block reference); hooks per-backend
semantics matrix: `reject` via tool input guardrails (synchronous, run continues) vs Claude's
PreToolUse deny; input-matched `reject` **aborts the turn** (tripwire) vs tool-matched (run
continues); `modify` inert on both; **coverage table — hosted tools AND MCP-server tools bypass
hooks, guardrails, and model-visible redaction** (OTel-layer redaction still applies);
`effort: max → xhigh` mapping; FR-084 reinterpretation; tracing behaviour incl.
`disable_provider_tracing` + `trace_include_sensitive_data`/`capture_content` coupling. Note in
`docs/security/aca-limitations.md` that **P3 hardened, US8 sandbox, and computer-use are deferred
to follow-up specs**; migration note that hooks/subagents for this backend live under `openai:`,
not `claude:`.
**Acceptance criteria:**
- [ ] Docs describe the `openai:` block, hosted tools, the hook/guardrail coverage table (incl.
      MCP gap), tracing behaviour, and the deferred surfaces.
**Verification:** Docs build / link check; grep for stale `openai_agents:` block references.
**Dependencies:** all prior
**Files:** `docs/api/backends.md`, `docs/guides/*`, `docs/security/aca-limitations.md`, `AGENTS.md`
**Scope:** S

### Checkpoint K — Complete
- [ ] `make format lint type-check security` clean; `make test` (parallel) green.
- [ ] SC-001/003/004/006/007/009 met by tests (SC-009 scoped to HoloDeck-built tools); sample runs
      end-to-end (creds-gated).
- [ ] Review with user.

---

## Dependency graph (summary)

```
A1 (openai: config) ──┬── A2 (backend consumes + side-effect-free validate) ── I1 (serve)
                      ├── A3 (RunConfig) ── H1 (tracing; removes MVP set_tracing_disabled)
                      ├── B1 (vectorstore/hier-doc + prompt warning) ──┐
                      ├── C1 (MCP) ──────────────────────────────────┤
                      │                                               ├── D1 (subagents) ── D2 (SkillTool model + skill) ── D3 (AG-UI events)
                      ├── E1 (hooks + failure wrapper) ── E2 (reject via guardrails; hosted paths ←─ G1)
                      │                                       └── J1 (redaction guardrail) ←─ needs B1
                      ├── F1 (effort) F2 (disallowed) F3 (budget) F4 (fallback) F5 (structured/thinking ←─ F1)
                      ├── G1 (hosted ×5) ── G2 (safety gate)
                      ├── I2 (dockerfile) I3 (sizing)
                      └── C1 → J2 (env scrub)
K1/K2/K3 depend on the feature phases they exercise.
```

**Parallelisable after A1:** A3, B, C, F, G, I are largely independent. E2's hosted paths need G1
(verified at Checkpoint G); J1 needs E2 + B1. D needs B/C (tool inheritance). Land in separate PRs
per phase.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `build_sdk_tools` signature change (now returns tools + mcp_servers + handoffs) ripples into the backend | Med | Single small refactor in A2/B1; covered by existing adapter tests before adding types |
| Hosted tools on Azure fail at runtime in confusing ways (Decision 4) | Med | Propagate the SDK error verbatim + a "this hosted tool may be unavailable on your Azure resource" hint; document in K3 |
| Hosted **and MCP-server** tools silently bypass hooks/guardrails/model-visible redaction | High | Load warnings on unreachable matchers (E1); reject on MCP tools fails load (E2); explicit coverage table in K3; SC-009 scoped; OTel-layer redaction (J2) still covers their spans |
| `max_budget_usd` price table staleness | Low | Bundled constant + unknown-model warning (no crash); documented update path |
| Guardrail `reject_content` semantics differ subtly from Claude's PreToolUse deny; input rejects abort the turn | Low | Document both in the per-backend matrix (K3) |
| Fallback wrapper double-retries with `ModelSettings.retry` | Med | F4 defines ordering (primary retries exhaust → one fallback attempt); unit-tested |
| Cost/fallback wrapping `Model` interferes with Responses/reasoning models | Med | Pin fallback to function/MCP; unit-test against a mocked reasoning model; caveat in docs |
| Nested hosted-tool `tool_config` objects drift with openai-types upgrades | Low | Factories isolate construction in one module; per-tool unit tests catch signature changes |
| Trace-processor swap timing leaks early spans to the platform exporter | Low | H1 configures processors at backend `initialize()` before any run emits spans |
| Serve session-cap divergence from Claude (active turns vs open sessions) confuses operators | Low | Echo the derivation at startup (I1); document the in-process rationale |

## Out of scope (deferred to follow-up specs)

- **P3 hardened profile** — Envoy credential sidecar, `deployment.security_profile`, egress
  allowlist, `OPENAI_BASE_URL`/`AZURE_OPENAI_ENDPOINT` rewriting (FR-090…093). Net-new cross-backend
  infra; own spec covering Claude + openai_agents.
- **US8 sandbox mode** — `agent_mode: sandbox`, `SandboxAgent`/`Manifest`, sandbox clients
  (FR-094…099). Deferred as a scoping decision (the surface exists in 0.17.4, including
  `UnixLocalSandboxClient`); deserves its own spec + threat model.
- **Computer-use** — `ComputerTool` needs a live `Computer`/`AsyncComputer` harness; own follow-up
  spec (Decision 7).
- **Human-in-the-loop tool approval** — `needs_approval` + `RunState.approve()/.reject()` resume
  loop (including MCP server `require_approval`, which resolves to the same interrupt). A distinct
  interactive feature, not an implementation detail of `reject` (Decision 5).
- **Guardrail coverage for MCP-server tools** — requires an SDK attachment point on its
  internally built `FunctionTool`s (or upstream feature request); v1 documents the gap and fails
  load on `reject`-matching MCP tools.
- **`modify` hook action implementation** — v1 inert + warning (Decision 6; SDK output guardrails
  make modify-on-output feasible later).
- **`type: prompt` runtime support** — no backend has an adapter today; openai_agents skips with a
  warning (B1) to match the status quo.
- **Unmapped SDK surfaces, explicitly disposed (not accidental omissions):**
  `tool_use_behavior` / `reset_tool_choice` / `ModelSettings.tool_choice` (tool-loop policy),
  agent/run `output_guardrails` beyond F5, `handoff_input_filter` / `nest_handoff_history`
  (handoff history shaping), `tool_error_formatter` / `tool_not_found_behavior`. All stay at SDK
  defaults in v1; revisit on demand.
- **Live `provider: openai` validation** — coded to spec, validated via mocks until a key exists
  (carried over from the MVP).
- **Cross-backend hooks/subagents namespace unification** — this plan keeps `openai:` and `claude:`
  separate by Decision 1.

## Open questions (resolved in this plan)

- Namespacing → new top-level `openai:` block (Decision 1).
- P3 / sandbox / computer-use → deferred (Decisions 2, 3, 7).
- Azure hosted tools → allowed, runtime-gated (Decision 4).
- `reject` mechanism → native tool guardrails for HoloDeck-built tools; `on_approval_request` for
  HostedMCPTool; load-fail for MCP-server + other hosted tools; tripwire abort for input matchers
  (Decision 5).
- Price-table source → bundled constant + unknown-model warning (spec Open Q3, v1).
- `modify` action → inert + warning (spec Open Q2 + Decision 6, v1).
- `effort: max` → `"xhigh"` (reconciliation vs FR-030/031).
