# Feature Specification: OpenAI Agents SDK Backend

**Feature Branch**: `035-openai-agents-backend`
**Created**: 2026-05-24
**Status**: Draft for review
**Author**: justinbarias (with Claude)
**Input**: User description: "Build a new backend — the OpenAI Agents SDK (latest version). Complete parity with the Claude backend (specs 24, 25, 26, 27, 28, 29, 30); meets the contract in spec 23; meets the same security hardening requirements in spec 034 but for OpenAI-based agents."
**Related**: `specs/023-choose-your-backend`, `specs/024-claude-serve-deploy`, `specs/025-tool-init-endpoints`, `specs/026-sdk-config-additions`, `specs/027-mcp-http-sse-transport`, `specs/028-yaml-hooks-system`, `specs/029-subagent-orchestration`, `specs/030-skills-support`, `specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md`
**External refs:** [OpenAI — The next evolution of the Agents SDK](https://openai.com/index/the-next-evolution-of-the-agents-sdk/), [OpenAI Agents SDK (Python) docs](https://openai.github.io/openai-agents-python/), [OpenAI — Agents & Sandboxes guide](https://developers.openai.com/api/docs/guides/agents/sandboxes)

## Motivation

HoloDeck today routes `provider: openai` and `provider: azure_openai` through Microsoft's Agent Framework (per spec 023's clean-break default). AF is functional but lives at one remove from OpenAI itself — it is a multi-provider runtime that *includes* OpenAI as one of several clients. OpenAI's [Agents SDK](https://openai.github.io/openai-agents-python/) is the first-class Python framework from OpenAI itself, the supported successor to the Assistants API, and the platform OpenAI is investing in (web search, file search, code interpreter, image gen, computer use, hosted MCP — all expose through this SDK first). [The next evolution of the Agents SDK](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) makes the strategic direction explicit.

There are three reasons to add it as a first-class HoloDeck backend:

1. **Provider-native parity.** Anthropic users get the Claude Agent SDK; Google users get ADK; OpenAI users today get a generic multi-provider wrapper. That is asymmetric and gives OpenAI customers the worst version of HoloDeck's "use your provider's native runtime" promise.
2. **Hosted tools become reachable.** `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `ComputerTool`, `HostedMCPTool` — none of these are usable through AF or SK. Adding the Agents SDK is the unlock.
3. **Sessions, handoffs, guardrails are SDK primitives.** Multi-agent (handoffs), session memory (`SQLiteSession`, `OpenAIConversations`), and input/output guardrails are first-class types — not patterns that have to be invented in adapter code. That maps cleanly onto specs 029 (subagents) and 028 (hook semantics, with the impedance mismatch flagged below).

The contract: **meet spec 023's backend protocols, reach feature parity with the Claude backend per specs 024–030, and inherit spec 034's hardening posture — reframed for the SDK's in-process execution model.**

## Architectural shape and what that changes vs. the Claude backend

The Claude Agent SDK spawns a Node.js subprocess per session (`ClaudeSDKClient`) and the spec-034 hardening story was built around that fact: per-subprocess memory accounting, hybrid sessions (P4), Node.js install, subprocess env scrubbing. **The OpenAI Agents SDK runs in-process Python.** That changes three things:

| Concern | Claude backend | OpenAI Agents backend |
|---|---|---|
| Process model | One Node.js subprocess per session | One Python coroutine per turn; no per-session subprocess |
| Memory unit | ~330 MiB per `ClaudeSDKClient` | Conversation state + tool state in the serve process |
| Hybrid sessions (spec 034 P4) | New code path required | Native — `session=SQLiteSession(id)` is the documented pattern |
| Node.js dependency | Required (SDK ships bundled CLI) | Not required (pure Python) |
| Permission posture | SDK ships Bash/Write/Edit/WebFetch as built-ins | No built-in dangerous tools; everything is explicit |
| Tracing | OTel via subprocess env vars | Native SDK tracing to `platform.openai.com/traces` + custom processor hook for OTel |
| MCP transports | `McpStdioServerConfig` / `McpSseServerConfig` / `McpHttpServerConfig` | `MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` |
| "Hooks" semantics | SDK hooks can reject/modify tool calls (`PreToolUse` returns a decision) | `AgentHooks`/`RunHooks` are observation-only; rejection routes through `input_guardrail` / `needs_approval` |
| "Skills" | `setting_sources: [project]` loads `.claude/skills/SKILL.md` | No subsystem; existing HoloDeck `SkillTool` (spec 023) maps onto sub-agents via handoff |

These are the load-bearing differences. The user-facing YAML stays as close to the existing schema as possible, and HoloDeck absorbs the impedance in the adapter layer.

### The sandbox question

The OpenAI Agents SDK exposes `SandboxAgent` ([guide](https://developers.openai.com/api/docs/guides/agents/sandboxes)) as the supported way to give an agent a confined shell + filesystem that persist across turns. This is the OpenAI-side analog to **Claude Code's native Bash/Write/Edit tools** — both deliver "the agent has a shell," and both raise the same question: how do you keep prompt injection from turning that shell into an exfil channel?

`SandboxAgent` and spec 034's container hardening solve **complementary, not substitutable** problems:

| Boundary | Spec 034 hardens | `SandboxAgent` enforces |
|---|---|---|
| Container → host | ✓ (cgroup limits, capability drops, Envoy egress allowlist, credential sidecar) | ✗ |
| Agent → container | ✗ (the agent can still write anywhere the container can write) | ✓ (`Manifest` + `SandboxClient` confines shell + FS to a workspace) |
| Credential exfil via egress | ✓ (P3 sidecar) | ✗ (credentials still in container env unless P3 is active) |
| Concurrent-session OOM | ✓ (P1a sizing + 429 backpressure) | ✗ |
| Prompt-injection → uncontrolled FS writes | partial (P2a `chmod a-w` on corpus) | ✓ (shell can only touch the sandbox workspace) |

The right v1 model is to support **both** — keep spec 034's container hardening as the default posture, and add `agent_mode: sandbox` as the opt-in path for agents that legitimately need shell/code execution. This is the analog of spec 034 P1b's `claude.i_understand_this_is_unsafe` gate: a named opt-in for a capability that hands the agent dangerous primitives.

In `security_profile: hardened`, the local `UnixLocalSandboxClient` swaps for a remote sandbox client (Docker or Modal) so the boundary becomes a kernel boundary, not a process boundary. This is also the realistic path to "per-session ephemeral containers" (Pattern 1 from the hosting doc) that both spec 034 and the base scope of this spec defer — it lands as a follow-up feature gated on operator demand.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Configure an OpenAI Agents SDK Agent via YAML (Priority: P1)

A platform user wants to use OpenAI models through the OpenAI-native Agents SDK. They create an `agent.yaml` with `model.provider: openai` and a model name (e.g. `gpt-5`, `gpt-4o`). When they run `holodeck test`, `holodeck chat`, or `holodeck serve`, HoloDeck auto-detects the `openai_agents` backend, builds an `Agent` + `Runner`, and executes the request through the SDK's agent loop.

**Why this priority**: This is the new default for `openai`/`azure_openai`. Without it, every other story is unreachable.

**Independent Test**: Create `agent.yaml` with `provider: openai`, `name: gpt-4o`, no `backend` field set. Run `holodeck test`. Verify response, tool calls, token usage are populated.

**Acceptance Scenarios**:

1. **Given** `model.provider: openai` and `model.name: gpt-4o`, **When** the user runs `holodeck test`, **Then** the agent invokes via `Runner.run`, returns text + token usage, and `last_agent.name` matches the configured agent name.
2. **Given** `model.provider: azure_openai` with `azure_openai_endpoint` and `azure_openai_api_key` configured, **When** the agent runs, **Then** the SDK uses `AsyncAzureOpenAI` and the response is identical in shape to the OpenAI case.
3. **Given** `model.provider: openai` and an unset `OPENAI_API_KEY`, **When** the agent loads, **Then** `holodeck` raises a clear `BackendInitError` naming the missing env var.
4. **Given** the same YAML as scenario 1 but with `backend: agent_framework` explicitly set, **When** the user runs `holodeck test`, **Then** the AF backend is used and the new backend is skipped — explicit opt-out works.

---

### User Story 2 — Serve and Deploy Parity (Priority: P1)

A platform operator runs `holodeck serve agent.yaml` against an OpenAI-backed agent and gets the same REST + AG-UI surface as Claude- or AF-backed agents. They then run `holodeck deploy build` and `holodeck deploy run` and the agent ships as an Azure Container App with the same probes, readiness semantics, secrets, and ingress defaults as a Claude agent. **No Node.js layer is required.**

**Why this priority**: Parity with specs 024 (serve/deploy) is the production gate.

**Independent Test**: Build + deploy the `sample/financial-assistant/openai` example. Hit `/health`, `/ready`, and `/awp`. Confirm a single AG-UI turn returns a grounded answer.

**Acceptance Scenarios**:

1. **Given** an OpenAI agent and `holodeck serve agent.yaml`, **When** the operator hits the `/awp` endpoint, **Then** the response shape matches the existing AG-UI contract used by the Claude and SK backends.
2. **Given** `holodeck deploy build` on an OpenAI agent, **When** the operator inspects the generated Dockerfile, **Then** Node.js is **not** installed (image is pure Python) unless the agent declares an MCP stdio tool whose `command` requires Node.js.
3. **Given** a deployed OpenAI agent, **When** the agent serves traffic, **Then** `/ready` returns 200 only after vectorstore/hierarchical_document tool init completes (parity with spec 025).
4. **Given** the entrypoint, **When** `OPENAI_API_KEY` (or the Azure credential set) is missing, **Then** the container exits non-zero at startup with a structured error — not after the first request.
5. **Given** the OpenAI backend, **When** OpenTelemetry endpoint env vars are set, **Then** spans are emitted in the same GenAI semconv shape as the Claude backend.

---

### User Story 3 — Tool Init Endpoints, MCP Transports, Hooks, Subagents, Skills (Priority: P1)

A platform user takes an existing agent.yaml that uses the Claude backend, switches `model.provider: anthropic` → `model.provider: openai`, and expects:

- `POST /tools/{name}/init` and `GET /tools` (spec 025) work identically.
- MCP tools declared as `transport: stdio | sse | http` (spec 027) connect through the SDK's `MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` classes.
- `claude.hooks: [...]` entries (spec 028) translate to OpenAI-side observation/approval hooks with the impedance documented (see below).
- `claude.agents: {...}` subagent definitions (spec 029) translate to handoffs.
- `SkillTool` entries (spec 023) translate to handoff targets built from inline instructions or a SKILL.md.

**Why this priority**: The whole point of the multi-backend architecture is portability. Without this, the new backend is a second-class citizen.

**Independent Test**: Take `sample/financial-assistant/claude`, copy to `sample/financial-assistant/openai`, change only `model.provider` and `model.name`, run `holodeck test` + tool-init endpoint smoke tests + a multi-agent handoff scenario.

**Acceptance Scenarios**:

1. **Given** an OpenAI-backed agent with a vectorstore tool, **When** the operator hits `POST /tools/{name}/init`, **Then** initialization runs asynchronously and `GET /tools/{name}/init` reflects state transitions (parity with spec 025 FR-001..FR-013).
2. **Given** an OpenAI-backed agent with `mcp` tools declared as `transport: sse` and `transport: http`, **When** the agent initializes, **Then** the SDK receives `MCPServerSse(params={url: ...})` and `MCPServerStreamableHttp(params={url: ...})` instances respectively, with header substitution from env vars (parity with spec 027).
3. **Given** an OpenAI-backed agent with hooks declared in `claude.hooks` (or its renamed equivalent — see Open Questions), **When** the SDK runs, **Then** `log`/`notify`/`script` actions fire from `AgentHooks` / `RunHooks`; `reject` actions on `PreToolUse` route through a generated `input_guardrail` for the matched input and through `needs_approval=...` on the matched function tool; `modify` actions are accepted at config time but emit a `not yet supported on openai_agents` warning unless implemented (see Open Questions).
4. **Given** an OpenAI-backed agent with `claude.agents` declaring three subagents (researcher / analyst / writer), **When** the SDK runs, **Then** each subagent becomes an `Agent` instance, the parent's `handoffs=[...]` lists them, and `RECOMMENDED_PROMPT_PREFIX` is auto-prepended to each subagent's instructions (parity with spec 029).
5. **Given** a `SkillTool` declared on an OpenAI-backed agent (inline or file-based), **When** the SDK runs, **Then** the skill is exposed as a handoff target whose instructions match the skill's `instructions` / SKILL.md body and whose tool scope is restricted by `allowed_tools`.
6. **Given** an OpenAI-backed agent with `setting_sources: [project]` (spec 030), **When** the agent loads, **Then** the system emits a clear warning that `setting_sources` is a Claude-only concept and is silently ignored on `openai_agents`; the load does **not** fail.

---

### User Story 4 — SDK Config Additions Per Spec 026 (Priority: P2)

The four config fields shipped in spec 026 (`effort`, `max_budget_usd`, `fallback_model`, `disallowed_tools`) need backend-specific mappings because the OpenAI Agents SDK does not expose all four 1:1.

| Spec 026 field | OpenAI Agents SDK mapping |
|---|---|
| `effort: low | medium | high | max` | Translated to `ModelSettings(reasoning=Reasoning(effort=...))` for gpt-5/o-series. `max` clamps to `high` with a load-time warning since the SDK type only accepts `low | medium | high`. |
| `max_budget_usd: float` | Enforced by a HoloDeck-managed `RunHooks` cost accountant: on each `on_llm_end`, accumulate the run's spend from `RunResult.token_usage` × per-model price table; when budget is exhausted, raise a `BackendBudgetExceededError` that the Runner propagates. Not native to the SDK; documented as best-effort. |
| `fallback_model: str` | Implemented via a wrapping `Model` provider that catches a configurable set of upstream errors (rate-limit, 5xx) and re-issues against the fallback. Pinned to function tools / MCP only; hosted tools that depend on Responses API may not survive the fallback if the fallback is a chat-completions-only model. |
| `disallowed_tools: list[str]` | Implemented as a config-time filter: the named tools are removed from the resolved `Agent.tools` and `mcp_servers` lists. For hosted tools we additionally refuse to construct (e.g. `CodeInterpreterTool`) when the name appears in the disallow list. |

**Why this priority**: Parity, but with honest delta. Operators should not be surprised that `max_budget_usd` is best-effort on this backend.

**Acceptance Scenarios**:

1. **Given** `effort: high` and `model.name: gpt-5`, **When** the agent runs, **Then** the SDK receives `ModelSettings(reasoning=Reasoning(effort="high"))`.
2. **Given** `effort: max`, **When** the config loads, **Then** a warning is emitted ("max not supported on openai_agents; clamping to high") and the SDK receives `effort="high"`.
3. **Given** `max_budget_usd: 0.05` and a query that exceeds 5¢, **When** the agent runs, **Then** the run aborts with `BackendBudgetExceededError` and the partial response is surfaced.
4. **Given** `fallback_model: gpt-4o-mini` and the primary model returns a 429, **When** the agent retries, **Then** the next call goes to `gpt-4o-mini` and the trace records both attempts.
5. **Given** `disallowed_tools: [WebSearchTool]` and a YAML that lists `WebSearchTool` in agent tools, **When** the config loads, **Then** load fails with a clear message: "`WebSearchTool` is both declared and disallowed."

---

### User Story 5 — Hosted Tools (Priority: P2)

A platform user wants to use OpenAI's hosted tools directly without standing up infrastructure: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `ComputerTool`, `HostedMCPTool`. They declare them in YAML alongside their function/MCP/vectorstore tools.

**Why this priority**: Hosted tools are an OpenAI differentiator and unblock production use cases that HoloDeck cannot otherwise serve. P2 because the backend ships without them and they can land in a follow-up if scope tightens.

**Independent Test**: Configure an agent with `WebSearchTool` and `FileSearchTool(vector_store_ids=[...])` in YAML, run a query that triggers each, verify citations and grounded output.

**Acceptance Scenarios**:

1. **Given** a tool entry `type: hosted, name: WebSearchTool`, **When** the agent loads, **Then** the SDK receives `WebSearchTool()` in `tools=[...]`.
2. **Given** `type: hosted, name: FileSearchTool, params: {vector_store_ids: [vs_abc], max_num_results: 5}`, **When** the agent loads, **Then** the SDK receives `FileSearchTool(vector_store_ids=["vs_abc"], max_num_results=5)`.
3. **Given** any hosted tool with `provider: azure_openai`, **When** the agent loads, **Then** load fails with a clear error: "Hosted tools require Responses API which is not available on Azure OpenAI yet."
4. **Given** a `HostedMCPTool` with `require_approval: never`, **When** the agent runs, **Then** the SDK invokes the MCP server-side without local approval flow.
5. **Given** `CodeInterpreterTool` or `ComputerTool` declared without `claude.i_understand_this_is_unsafe: true` (or the renamed equivalent — see Open Questions), **When** the config loads, **Then** load fails with a security warning explaining the capability.

---

### User Story 6 — Production Hardening Per Spec 034 (Priority: P1)

The same hardening posture spec 034 brings to the Claude backend must apply here, reframed for the in-process execution model. Phase numbering mirrors 034 so operators can cross-reference. **Phase 4 (hybrid sessions) is dropped** because the OpenAI Agents SDK's native `session=SQLiteSession(id)` already delivers what P4 reconstructs for Claude — idle sessions cost ~zero process resident memory because session state lives in SQLite, not a held subprocess.

| Phase | Adapted scope |
|---|---|
| **P1a — stop the OOM** | Default ACA sizing: 1 CPU / 1 GiB (smaller floor than Claude's 2 GiB; no Node.js subprocess overhead). Memory-derived `max_concurrent_sessions` defaults to `floor(memory_mib / session_memory_estimate_mib)` with `session_memory_estimate_mib` defaulting to `100` (vs Claude's `400`). 429 with Retry-After on overflow. `/ready` distinct from `/health`. `max_turns` default `20` (SDK default is `10`; bumped to match Claude). |
| **P1b — permission posture** | OpenAI Agents SDK has **no built-in dangerous tools** equivalent to Claude's Bash/Write/Edit/WebFetch — every tool is explicit. The 1b posture instead targets *hosted tools* with destructive surface: `CodeInterpreterTool`, `ComputerTool`, `ImageGenerationTool` (cost surface). Auto-disallowed at backend layer unless the agent declares them. New schema field: `openai_agents.i_understand_this_is_unsafe: true` is required to allow `CodeInterpreterTool` or `ComputerTool`. Function tools that declare `needs_approval` keep their gate. |
| **P2a — container hardening** | Generated Dockerfile is pure Python (no Node.js gate). Corpus dirs root-owned + `chmod a-w`. EmptyDir volumes for `/tmp` and `/var/holodeck/work`. Ingress defaults to internal with the same loud warning on `ingress_external: true`. ACA `securityContext` gaps documented identically — `docs/security/aca-limitations.md` covers both backends. |
| **P2b — prompt-injection defenses** | Credential redaction **on function tool return values** via a `holodeck.lib.backends.openai_agents_hooks` module: each `@function_tool` is wrapped at registration time with a post-return scrubber (5 patterns: anthropic key, AWS access key, GitHub token, JWT, Bearer header — same patterns as Claude). `RedactingSpanProcessor` (existing module) scrubs `tool.input.*`, `tool.output.*`, `gen_ai.*` span attributes — already backend-agnostic. **Subprocess env scrub** applies to MCP stdio servers and function tools that shell out: `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`-style behavior generalised. **Note**: openai-agents `AgentHooks.on_tool_end` is observation-only — it cannot mutate the tool result. The wrapping-decorator approach is the only honest implementation. |
| **P3 — credential boundary (opt-in)** | `deployment.security_profile: hardened` works identically: Envoy sidecar holds credentials; agent container has none; domain allowlist derived from YAML. For `provider: openai`, `OPENAI_BASE_URL=http://localhost:<envoy-port>` is injected. For `provider: azure_openai`, the allowlist additionally includes the Azure resource hostname (`<resource>.openai.azure.com`) and `AZURE_OPENAI_ENDPOINT` routes through Envoy — at the application layer this requires the agent to set `azure_endpoint=` to the localhost sidecar URL. Same caveats as Claude P3 (no TLS-terminating proxy in v1). |
| **P4 — hybrid sessions** | **N/A on this backend.** Use `SQLiteSession(session_id)` (the SDK's native session) and idle sessions are SQLite rows, not held processes. The HoloDeck `SessionStore` caps concurrent *active turns*, not open sessions. This is documented as "P4 satisfied by SDK design" in `docs/security/aca-limitations.md`. |

**Acceptance Scenarios**:

1. **Given** the default 1 CPU / 1 GiB ACA sizing, **When** the operator runs `holodeck deploy run`, **Then** the resolved-config echo reports "Concurrent sessions per replica: 10 (derived from 1024 MiB memory limit @ 100 MiB/session)".
2. **Given** an agent that declares `CodeInterpreterTool` without `openai_agents.i_understand_this_is_unsafe: true`, **When** the config loads, **Then** load fails with the canonical migration error pointing at the safety opt-in.
3. **Given** a function tool that returns text containing `sk-ant-api03-XXXX`, **When** the agent runs, **Then** the SDK sees `[REDACTED:anthropic-key]` instead of the raw value and the OTel span's `tool.output.*` attribute is also redacted.
4. **Given** `security_profile: hardened`, **When** the operator deploys, **Then** the agent container env has zero credential-bearing vars; the sidecar holds `OPENAI_API_KEY` (or the Azure equivalents); calls to non-allowlisted domains are rejected by Envoy with a structured error.
5. **Given** the in-process model, **When** 50 sessions are created with sparse activity, **Then** memory does **not** scale with open-session count because session state lives in SQLite — the 429 cap only kicks in based on concurrent in-flight turns.

---

### User Story 7 — Tracing (Priority: P2)

The OpenAI Agents SDK ships with built-in tracing that uploads to `platform.openai.com/traces`. HoloDeck's existing OTel pipeline (per spec 022) is the production observability story. For `provider: openai`, both should be active. For `provider: azure_openai`, the OpenAI dashboard upload must be **disabled** (Azure customers' data should not leak to OpenAI's hosted dashboard).

**Independent Test**: Run an OpenAI-provider agent with `OTEL_EXPORTER_OTLP_ENDPOINT` set; verify spans appear at both the OTel collector and `platform.openai.com/traces`. Re-run with `provider: azure_openai`; verify spans appear only at the OTel collector and the SDK's upload is suppressed.

**Acceptance Scenarios**:

1. **Given** `provider: openai`, **When** an agent runs, **Then** a custom `TracingProcessor` registered by HoloDeck mirrors SDK trace events into the OTel pipeline AND the SDK's default upload to `platform.openai.com/traces` continues.
2. **Given** `provider: azure_openai`, **When** the backend initializes, **Then** `set_tracing_disabled(True)` is called *after* the OTel-mirroring processor is registered, suppressing the OpenAI dashboard upload while preserving OTel emission.
3. **Given** either provider, **When** the user opts out via `observability.disable_provider_tracing: true`, **Then** the OpenAI dashboard upload is suppressed regardless of provider.
4. **Given** either provider, **When** an OTel span carries a `tool.output.*` attribute containing a credential-shaped string, **Then** the `RedactingSpanProcessor` from P2b scrubs it before export.

---

### User Story 8 — Sandboxed Agent Mode for Shell / Code Execution (Priority: P2)

A platform user wants to build an agent that writes and executes Python or shell commands as part of its reasoning loop (codex-style, build-and-test, scripted data analysis). They declare `openai_agents.agent_mode: sandbox` in YAML and opt into the safety gate. The backend constructs a `SandboxAgent` with the SDK's `UnixLocalSandboxClient` instead of a plain `Agent`; the agent gets a confined workspace + shell + persistent filesystem across turns. The same YAML deploys safely under `security_profile: hardened`, where the local sandbox client swaps for a remote sandbox (Docker or Modal) so the boundary is a kernel boundary, not a process boundary.

**Why this priority**: This is the OpenAI-side analog to Claude's Bash/Write/Edit tool surface. It unblocks a real class of agents (codex-style code generation, automated data analysis) that today can only be served by `CodeInterpreterTool` — which is Responses-API-only and therefore unavailable on `provider: azure_openai`. Sandbox mode works for both providers. It's P2 because the base backend (`agent_mode: standard`) ships first; sandbox mode is additive.

**Independent Test**: Set `openai_agents.agent_mode: sandbox` and `openai_agents.i_understand_this_is_unsafe: true` in agent.yaml, give the agent a task that requires writing and running a Python script, and verify the shell commands execute inside the sandbox workspace (not the container's working directory).

**Acceptance Scenarios**:

1. **Given** `openai_agents.agent_mode: sandbox` and `openai_agents.i_understand_this_is_unsafe: true`, **When** the agent is initialized, **Then** the backend constructs `SandboxAgent(manifest=..., client=UnixLocalSandboxClient())` with a generated `Manifest(name=<agent.name>, description=<agent.description>)`.
2. **Given** `agent_mode: sandbox` but `i_understand_this_is_unsafe: false` (or unset), **When** the config loads, **Then** load fails with a structured error pointing at the safety opt-in — same shape as the existing `CodeInterpreterTool` / `ComputerTool` gate (FR-083).
3. **Given** `agent_mode: sandbox`, **When** the agent writes a file via its sandboxed shell, **Then** the file lands inside the sandbox workspace and the container's `/app` directory is unaffected.
4. **Given** `agent_mode: sandbox` and `deployment.security_profile: hardened`, **When** the operator deploys, **Then** the backend swaps `UnixLocalSandboxClient` for the configured remote sandbox client (Docker by default; Modal if `openai_agents.sandbox.remote_client: modal` is set). The hardened-profile Envoy sidecar continues to enforce egress allowlist independently.
5. **Given** `agent_mode: sandbox` and a YAML that also declares `CodeInterpreterTool` in `tools`, **When** the config loads, **Then** load fails with a redundancy error: "Sandbox mode already provides code execution; remove `CodeInterpreterTool` from `tools`."
6. **Given** `agent_mode: sandbox` and `provider: azure_openai`, **When** the agent runs, **Then** the sandbox functions normally — this is the supported code-execution path on Azure (where `CodeInterpreterTool` is unavailable per US5 acceptance scenario 3).
7. **Given** `agent_mode: sandbox`, **When** the agent session ends, **Then** the sandbox workspace is destroyed (for `UnixLocalSandboxClient`, the temp directory is removed; for remote sandbox clients, the remote container is terminated).

---

### Edge Cases

- What happens when `provider: openai` is set but `OPENAI_API_KEY` is unset and no `RunConfig.model_provider` override is supplied? → `BackendInitError` at startup, parity with Claude backend's credential check.
- What happens when an agent declares `model: gpt-5` but the runtime account doesn't have access? → The SDK surfaces the auth error at the first `Runner.run`; HoloDeck propagates the underlying message verbatim.
- What happens when a user mixes `session=` and `previous_response_id=` (or `conversation_id=`) in YAML? → Pydantic validation rejects the config at load time.
- What happens when an MCP stdio server's `command` requires Node.js? → Dockerfile generator detects this from the MCP tool config (same logic as Claude P2a Node.js gate) and installs Node only in that case.
- What happens when a hosted tool is declared on a provider that can't run Responses API (Azure today)? → US5 acceptance scenario 3 — load fails with a clear error.
- What happens when an `input_guardrail` synthesized from a `reject`-type hook trips during a multi-agent handoff chain? → SDK semantics apply: only the *entry/triage* agent's input guardrails run. The spec documents this; operators are pointed to set the rejecting hook on the entry agent.
- What happens when `fallback_model` is set and the primary model raises a non-retryable error (e.g. invalid argument)? → No fallback; the error propagates. Fallback is gated to the configurable retryable-error set.
- What happens when a `claude.hooks` `modify` action is declared on `openai_agents`? → Load succeeds with a warning ("modify action not yet supported on openai_agents; hook will be inert"). Not a load failure — the same YAML must remain portable across backends.
- What happens when `disallowed_tools` includes an MCP tool name that doesn't exist? → Validation warning at config load time (same shape as spec 029 FR-009 for subagent tool references).
- What happens when `setting_sources` is declared on an `openai_agents` agent? → Warning emitted, value ignored (US3 acceptance scenario 6). Cross-backend YAML portability is preserved.
- What happens when `agent_mode: sandbox` is set but the host does not provide the sandbox runtime (e.g. `UnixLocalSandboxClient` cannot allocate a workspace)? → Pre-flight validation at `holodeck serve` time catches this and emits a structured error before accepting requests.
- What happens when `agent_mode: sandbox` is set and the sandbox process inside the workspace runs out of disk? → The SDK surfaces the underlying error; HoloDeck propagates it as a `BackendSessionError` to the client. Quota is bounded by the sandbox client's configuration (`workspace_size_limit_mb`, default 512 MiB; see the SDK manifest fields).
- What happens when a `agent_mode: sandbox` agent is deployed under `security_profile: hardened` but no remote sandbox client credentials are configured? → Load fails at deploy time with a clear error naming the required env vars (`DOCKER_HOST` for Docker; `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` for Modal).

## Scope Boundary

**In scope:**

- New backend module `src/holodeck/lib/backends/openai_agents_backend.py` implementing `AgentBackend` and `AgentSession` protocols (spec 023 FR-003/004/005).
- Tool adapters for the 5 HoloDeck tool types: `vectorstore`, `function`, `mcp` (stdio/sse/http per spec 027), `hierarchical_document`, `skill` (per spec 023 FR-022..FR-026).
- New tool category: `hosted` (US5) covering `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `ComputerTool`, `HostedMCPTool`.
- New agent mode: `agent_mode: sandbox` (US8) constructing `SandboxAgent` with `UnixLocalSandboxClient` by default and a remote sandbox client (Docker / Modal) under `security_profile: hardened`. Gated by the existing `openai_agents.i_understand_this_is_unsafe` opt-in.
- Serve + deploy parity (spec 024): pre-flight validation, Dockerfile generation (pure Python by default), health/ready probes, OTel preservation, subprocess lifecycle (in-process: tracks concurrent turns).
- Tool init endpoints (spec 025) — already backend-agnostic; smoke-test against the new backend.
- LiteLLM embedding adapter (spec 023 FR-019/020) reused as-is for vectorstore tools on this backend.
- SDK config additions (spec 026) with the per-field mappings spelled out in US4.
- MCP HTTP/SSE transports (spec 027) via `MCPServerStreamableHttp` and `MCPServerSse`.
- YAML-defined hooks (spec 028) mapped to `AgentHooks` / `RunHooks` for observation actions; `reject` actions routed via `input_guardrail` + `needs_approval`; `modify` deferred with a clear warning.
- Subagent orchestration (spec 029) via SDK `handoffs=[...]` with `RECOMMENDED_PROMPT_PREFIX` auto-applied.
- Skill tools (spec 023 FR-022..FR-026) implemented as handoff targets; `setting_sources` (spec 030) is a documented no-op with a load warning.
- Hardening posture per spec 034 phases P1a/P1b/P2a/P2b/P3 — adapted to the in-process model. P4 is documented as native.
- Default routing change: `provider: openai` and `provider: azure_openai` route to `openai_agents` by default. Clean-break, same pattern spec 023 used for SK → AF.
- Tracing: dual-write to OTel + `platform.openai.com/traces` for `provider: openai`; OTel-only for `provider: azure_openai` (per session 2026-05-24 clarifications).

**Out of scope:**

- `VoicePipeline` / `RealtimeAgent` / `SandboxAgent` from the openai-agents SDK. These have meaningful schema requirements (audio I/O, sandbox primitives) that don't fit the v1 surface.
- A new deployment target. We reuse the existing `holodeck deploy` Azure Container Apps path (parity with spec 024).
- Per-session ephemeral containers (Pattern 1 from spec 034). Same out-of-scope decision applies here.
- A native equivalent to Claude Code's `.claude/skills/SKILL.md` discovery. HoloDeck's `SkillTool` covers the declarative case; ambient skill discovery is not in v1.
- Implementing the `modify` hook action on this backend (deferred — flagged in Open Questions).
- Replacing the Microsoft Agent Framework backend. AF remains supported and selectable via `backend: agent_framework`.
- A `gpt-image` or `Responses API`-only feature gate beyond the existing hosted-tool guard. If a downstream feature requires Responses API, it MUST live in this backend, not in AF/SK/Claude.

## Requirements *(mandatory)*

### Functional Requirements

**Backend contract (spec 023):**

- **FR-001**: System MUST register a new backend identifier `openai_agents` in the `Backend` enumeration and route to it from the existing `BackendSelector`.
- **FR-002**: The `openai_agents` backend MUST implement the `AgentBackend` protocol exactly: `initialize()`, `invoke_once()`, `create_session()`, `teardown()`.
- **FR-003**: The `openai_agents` session MUST implement the `AgentSession` protocol exactly: `prepare()`, `send()`, `send_streaming()`, `close()`.
- **FR-004**: `invoke_once()` and `send()` MUST return `ExecutionResult` objects with all fields populated: `response`, `tool_calls`, `tool_results`, `token_usage`, `structured_output`, `num_turns`, `is_error`, `error_reason`, `thinking` (empty when the model is non-reasoning).
- **FR-005**: `send_streaming()` MUST yield successive text chunks consistent with the SDK's `RawResponsesStreamEvent`/`ResponseTextDeltaEvent` shape, surfaced through HoloDeck's existing AG-UI bridge with no protocol-level change.
- **FR-006**: System MUST emit `ToolEvent` records (kind: `start` / `end` / `error` / `thinking`) consistent with the Claude backend's contract, so the AG-UI panel renders identically. `subagent_message` and `parent_link` events MUST be emitted during handoff transitions (mapped from `AgentUpdatedStreamEvent`).
- **FR-007**: Default routing table updates: `openai` and `azure_openai` MUST default to `openai_agents`. `anthropic` continues to default to `claude`. `google` continues to default to `google_adk`. `ollama` continues to default to `claude` (per spec 023 FR-031, unchanged). `semantic_kernel` and `agent_framework` remain explicit-only.

**Serve & deploy parity (spec 024):**

- **FR-010**: `holodeck serve agent.yaml` MUST perform pre-flight credential validation at startup for `provider: openai` (`OPENAI_API_KEY`) and `provider: azure_openai` (`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`).
- **FR-011**: `holodeck deploy build` MUST generate a Dockerfile that does **not** install Node.js when the agent uses `openai_agents` and has no Node-requiring MCP stdio servers. When such an MCP server exists, Node.js MUST be installed (parity with the spec 034 P2a gating logic).
- **FR-012**: Container entrypoint MUST validate the prerequisite credential set before starting `holodeck serve` and exit non-zero with a structured error if missing.
- **FR-013**: `/health` MUST report process liveness; `/ready` MUST return 200 only after tool init manager reports all tools initialized (parity with spec 034 P1a).
- **FR-014**: Subprocess lifecycle in serve mode MUST cap concurrent **active turns** (not open sessions) at the resolved `max_concurrent_sessions` value; overflow returns 429 with `Retry-After`. This deliberately diverges from the Claude backend's pre-P4 cap on open sessions, because the in-process model makes idle-session memory ~zero.

**Tool init endpoints (spec 025):**

- **FR-020**: All endpoints in spec 025 (`POST /tools/{name}/init`, `GET /tools/{name}/init`, `GET /tools`) MUST work identically against an `openai_agents` agent. Vectorstore and hierarchical_document tools MUST be initializable; non-init tool types MUST return 400.

**SDK config additions (spec 026, with backend-specific mappings):**

- **FR-030**: `effort: low | medium | high` MUST translate to `ModelSettings(reasoning=Reasoning(effort=<value>))`.
- **FR-031**: `effort: max` MUST emit a load-time warning and clamp to `high`.
- **FR-032**: `max_budget_usd: <float>` MUST be enforced by a HoloDeck-managed `RunHooks` cost accountant; budget exhaustion raises `BackendBudgetExceededError` and aborts the run. Backend MUST surface the partial response and accumulated cost in the error payload.
- **FR-033**: `fallback_model: <str>` MUST be implemented as a wrapping model provider that retries the request against the named model on a configurable set of retryable upstream errors (default: 429 rate-limit, 5xx). The fallback set is not user-tunable in v1.
- **FR-034**: `disallowed_tools: [<str>, ...]` MUST be applied at config-time tool-resolution: named tools are removed from `Agent.tools` and `mcp_servers`. If a tool name appears in both `allowed_tools` and `disallowed_tools`, config load MUST fail.

**MCP transports (spec 027):**

- **FR-040**: MCP tools with `transport: stdio` MUST translate to `MCPServerStdio(params={command, args, env})`.
- **FR-041**: MCP tools with `transport: sse` MUST translate to `MCPServerSse(params={url, headers})`. URL and header env-var substitution MUST work (parity with spec 027 FR-005).
- **FR-042**: MCP tools with `transport: http` MUST translate to `MCPServerStreamableHttp(params={url, headers})` with the same env-var substitution.
- **FR-043**: MCP tools with `transport: websocket` MUST be skipped with the same warning shape spec 027 FR-006 prescribes for the Claude backend.
- **FR-044**: MCP tool filtering MUST be supported via `create_static_tool_filter(allowed_tool_names=[...])` when an MCP tool config declares an `allowed_tools` subset.

**YAML hooks (spec 028, with semantic mapping):**

- **FR-050**: Hook events `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `Notification`, `SessionStart` MUST be supported on `openai_agents` with the following SDK mapping:
  - `PreToolUse` with `action: log | notify | script` → `AgentHooks.on_tool_start` (observation-only).
  - `PreToolUse` with `action: reject` → SDK `function_tool(..., needs_approval=...)` is generated for the matched tool, returning the rejection message synchronously when the matcher fires. If the matched tool is a hosted tool that does not support `needs_approval`, load fails with a clear error.
  - `PreToolUse` with `action: modify` → load succeeds with a warning that the action is not implemented on this backend in v1.
  - `PostToolUse` and `PostToolUseFailure` → `AgentHooks.on_tool_end` (observation-only). Cannot reject post-fact; documented.
  - `Stop` → `RunHooks.on_end` (or `AgentHooks.on_end` if scoped to a specific agent).
  - `Notification` → `RunHooks.on_llm_start` / `on_llm_end` pair, surfaced through the notification action.
  - `SessionStart` → invoked once at `create_session()` time.
- **FR-051**: Hook chain ordering MUST match spec 028 FR-010: HoloDeck-internal hooks (tool tracking, credential redaction) run before user-defined hooks; user-defined hooks run in declaration order; the first terminal action (reject) stops evaluation.

**Subagent orchestration (spec 029):**

- **FR-060**: Each entry in `claude.agents` (or its renamed equivalent — see Open Questions) MUST be translated to an `Agent(name=<key>, instructions=<prompt>, handoff_description=<description>, tools=<resolved>, model=<resolved>)` instance and added to the parent agent's `handoffs=[...]`.
- **FR-061**: When a subagent declares no `tools`, the parent's full tool list MUST be inherited (parity with spec 029 FR-007).
- **FR-062**: When a subagent declares `model: inherit`, the parent's model MUST be used. When it declares `sonnet | opus | haiku`, load MUST fail (these are Claude model literals; not portable). For openai-agents, the allowed values are `inherit` plus any string the SDK accepts as a model identifier.
- **FR-063**: `RECOMMENDED_PROMPT_PREFIX` from `agents.extensions.handoff_prompt` MUST be auto-prepended to each subagent's `instructions` unless the YAML declares `claude.agents.<name>.skip_recommended_prefix: true`.

**Skills (spec 023 SkillTool + spec 030):**

- **FR-070**: `SkillTool` (type: `skill`) entries MUST be translated to handoff targets. Inline form (`instructions`, `description`, `allowed_tools`) becomes an `Agent` with those fields. File-based form (`path` pointing at a directory with SKILL.md) loads the body + frontmatter and constructs the same Agent.
- **FR-071**: `setting_sources` (spec 030) MUST be accepted in YAML for cross-backend portability but emit a load-time warning when the resolved backend is `openai_agents`: "setting_sources is a Claude-only concept; ignored on openai_agents."

**Hardening (spec 034, in-process reframe):**

- **FR-080 (P1a)**: Default ACA sizing for `openai_agents` agents MUST be 1 CPU / 1 GiB. Default `max_concurrent_sessions` MUST derive from cgroup memory: `floor(memory_mib / openai_agents.session_memory_estimate_mib)` with `openai_agents.session_memory_estimate_mib` defaulting to `100`. Operator override remains via `openai_agents.max_concurrent_sessions`. The resolved value MUST be echoed at `holodeck deploy run` and `holodeck serve` time.
- **FR-081 (P1a)**: Overflow MUST return 429 with `Retry-After` and a problem+json body whose `type` is `https://holodeck.dev/errors/session-cap-exceeded`. Parity with spec 034 P1a.
- **FR-082 (P1a)**: `max_turns` default MUST be `20` (SDK default is `10`; bumped for parity with the Claude backend's spec-034 default).
- **FR-083 (P1b)**: Hosted tools `CodeInterpreterTool`, `ComputerTool` MUST be auto-disallowed unless the agent declares `openai_agents.i_understand_this_is_unsafe: true`. Loading without the opt-in emits the canonical migration error.
- **FR-084 (P1b)**: Function tools declared with `needs_approval=...` MUST retain their gate. The default hooks chain MUST NOT silently weaken this.
- **FR-085 (P2a)**: Generated Dockerfile MUST be pure Python by default (no `apt-get install nodejs npm`). Node.js install is gated only on MCP stdio servers that declare a `command` requiring it. Corpus dirs root-owned + `chmod a-w`. EmptyDir volumes for `/tmp` and `/var/holodeck/work`.
- **FR-086 (P2a)**: Ingress defaults to `false` (internal). Setting `ingress_external: true` emits the same loud deploy-time warning as spec 034.
- **FR-087 (P2b)**: A new module `src/holodeck/lib/backends/openai_agents_hooks.py` MUST provide a default function-tool decorator that scrubs credential-shaped strings from tool return values (5 patterns identical to spec 034 P2b: anthropic key, AWS access key, GitHub token, JWT, Bearer header). Default-on; opt-out via `openai_agents.disable_default_hooks: true` with the same loud warning shape as Claude P2b.
- **FR-088 (P2b)**: `RedactingSpanProcessor` from the existing `otel_redaction.py` module MUST scrub `tool.input.*`, `tool.output.*`, `gen_ai.*` span attributes for `openai_agents` spans the same way it does for Claude spans. This is backend-agnostic and requires no new code beyond ensuring the OTel processor sees the new backend's spans.
- **FR-089 (P2b)**: When the agent has MCP stdio servers OR function tools that spawn subprocesses, the backend MUST default-on a subprocess env scrub that strips OpenAI/Azure/Anthropic credential env vars from spawned children. Opt-out via `openai_agents.disable_subprocess_env_scrub: true`. Implementation reuses the same subprocess wrapper pattern as Claude P2b Task 4.5.
- **FR-090 (P3)**: `deployment.security_profile: hardened` MUST work identically: Envoy sidecar holds credentials, agent container has none, domain allowlist derived from YAML.
- **FR-091 (P3)**: Allowlist derivation for `provider: openai` MUST include `api.openai.com`, the embedding-provider endpoint, and each MCP HTTP/SSE endpoint. For `provider: azure_openai`, the allowlist MUST additionally include the resolved Azure resource hostname (`<resource>.openai.azure.com`) extracted from `AZURE_OPENAI_ENDPOINT`.
- **FR-092 (P3)**: For `provider: openai`, the agent container MUST be configured with `OPENAI_BASE_URL=http://localhost:<envoy-port>` and `HTTPS_PROXY=http://localhost:<envoy-port>`. For `provider: azure_openai`, `AZURE_OPENAI_ENDPOINT` MUST be rewritten to the localhost sidecar URL; the original Azure hostname is reachable only from the sidecar.
- **FR-093 (P3)**: The `openai_agents` backend MUST refuse to start in hardened profile if any credential-bearing env var is set on the agent container (the operator has half-migrated).

**Sandbox agent mode (US8):**

- **FR-094**: System MUST accept `openai_agents.agent_mode: standard | sandbox` (default `standard`).
- **FR-095**: When `agent_mode: sandbox`, the backend MUST construct `SandboxAgent(manifest=Manifest(name=<agent.name>, description=<agent.description>, ...), client=<sandbox_client>)` instead of `Agent(...)`. All other agent fields (instructions, tools, model, handoffs, hooks) MUST be forwarded to the `SandboxAgent` with the same shape they have on the standard `Agent`.
- **FR-096**: When `agent_mode: sandbox`, the safety gate MUST require `openai_agents.i_understand_this_is_unsafe: true`. Load without the opt-in fails with the canonical migration error (same shape as FR-083 for `CodeInterpreterTool` / `ComputerTool`).
- **FR-097**: When `agent_mode: sandbox` AND `deployment.security_profile: default`, the sandbox client MUST be `UnixLocalSandboxClient` with the workspace under `/var/holodeck/work/sandbox/<session_id>` (using the same EmptyDir tmpfs mount as FR-085). Workspace MUST be destroyed on session close.
- **FR-098**: When `agent_mode: sandbox` AND `deployment.security_profile: hardened`, the sandbox client MUST switch to a remote sandbox client selected by `openai_agents.sandbox.remote_client: docker | modal` (default `docker`). The remote client's credentials MUST live alongside the other secrets in the Envoy sidecar; the agent container MUST NOT carry them.
- **FR-099**: System MUST reject configurations that declare both `agent_mode: sandbox` and `CodeInterpreterTool` (or `ComputerTool`) in `tools` — the sandbox already provides shell + code execution and the overlap is a configuration error, not a feature.

**Tracing:**

- **FR-100**: When `provider: openai`, HoloDeck MUST register a custom `TracingProcessor` that mirrors SDK trace events into the OTel pipeline AND MUST NOT call `set_tracing_disabled(True)` — the SDK's default upload to `platform.openai.com/traces` continues alongside OTel.
- **FR-101**: When `provider: azure_openai`, HoloDeck MUST register the same custom `TracingProcessor` for OTel emission and MUST call `set_tracing_disabled(True)` after registration to suppress the platform.openai.com upload.
- **FR-102**: A new schema field `observability.disable_provider_tracing: bool` (default `false`) MUST allow operators to suppress the OpenAI-dashboard upload regardless of provider. OTel emission is unaffected.

**Validation at startup:**

- **FR-110**: All validation errors collected in a single pass, surfaced together (parity with spec 034 final validation section).
- **FR-111**: Validation MUST run at `holodeck serve` and `holodeck deploy run` time. The resolved config (sizing, concurrent cap, default hooks, hosted tools, security profile, tracing) MUST be echoed before listening/deploying — same shape as the spec-034 deploy-time echo.

### Key Entities

- **Backend (extended)**: `openai_agents` added to the `Backend` enumeration alongside `semantic_kernel`, `claude`, `google_adk`, `agent_framework`.
- **OpenAIAgentsBackend / OpenAIAgentsSession**: New backend and session implementations wrapping the SDK's `Agent` + `Runner` + `SQLiteSession`.
- **OpenAIAgentsConfig**: New top-level YAML block `openai_agents:` (sibling to `claude:`) with fields `max_concurrent_sessions`, `session_memory_estimate_mib`, `i_understand_this_is_unsafe`, `disable_default_hooks`, `disable_subprocess_env_scrub`, `permissions` (parallels `claude.permissions`).
- **OpenAI Agents Tool Adapters**: Translators from HoloDeck tool definitions to SDK tool objects: `@function_tool`-wrapped callables, `MCPServerStdio`/`MCPServerSse`/`MCPServerStreamableHttp`, hosted-tool factories.
- **Hosted Tool Entry**: New tool type `type: hosted` in the schema, with `name` selecting the SDK class and `params` carrying its constructor kwargs.
- **Cost Accountant**: HoloDeck-managed `RunHooks` implementation that enforces `max_budget_usd`. Per-model price table is bundled with the backend module and versioned.
- **Fallback Model Wrapper**: A `Model` provider implementation that catches a configurable retryable-error set and re-issues against the fallback model.
- **Custom Tracing Processor**: A `TracingProcessor` that mirrors SDK trace events to the OTel pipeline. Used by both `provider: openai` and `provider: azure_openai`.
- **Default Hook Decorator**: A function-tool wrapper that scrubs credentials from return values before the SDK reads them. Lives in `openai_agents_hooks.py`.
- **OpenAIAgentsHardenedProfile**: A composition layer that wires the Envoy sidecar and rewrites `OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` to the localhost sidecar URL for the `security_profile: hardened` deployment.
- **OpenAIAgentsSandboxMode**: A construction switch on the backend. When active, builds `SandboxAgent` with `UnixLocalSandboxClient` (default profile) or a remote sandbox client (`docker` / `modal`, hardened profile). Generates the SDK `Manifest` from the agent's `name` and `description`. Gated by `openai_agents.i_understand_this_is_unsafe`.
- **OpenAIAgentsSandboxConfig**: New sub-block under `openai_agents.sandbox` carrying `remote_client: docker | modal` and per-client tuning (workspace size, idle TTL).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Operators can switch any existing OpenAI- or Azure-OpenAI-backed agent to the new backend by deleting the optional `backend:` field (auto-detect routes to `openai_agents`). All five HoloDeck tool types (`vectorstore`, `function`, `mcp`, `skill`, `hierarchical_document`) continue to function. Hosted tools become additionally available as `type: hosted`.
- **SC-002**: All existing tests pass without modification. The Claude, SK, AF, and ADK backends are unchanged.
- **SC-003**: `openai_agents` passes the spec-023 functional test suite (single-turn, multi-turn session, streaming, tool calling, error handling) at parity with Claude/AF.
- **SC-004**: Agent configs validate within 1 second; invalid configs (missing credentials, conflicting tool lists, unsupported hosted tool on Azure, missing `i_understand_this_is_unsafe` for dangerous hosted tools) produce actionable errors.
- **SC-005**: Operators who do not install the openai-agents extras experience no import-time failures when using other backends (lazy import gate).
- **SC-006**: `openai_agents` results pass all three evaluation metric types (standard NLP, G-Eval, RAG) — no result-shape regressions.
- **SC-007**: All 5 HoloDeck tool types and all 6 hosted tools work on `openai_agents`, verified by per-tool unit tests in `tests/unit/lib/backends/test_openai_agents_tool_adapters.py`.
- **SC-008**: Container deploys on default sizing (1 CPU / 1 GiB) handle 10 concurrent in-flight turns without OOM; overflow returns 429 with `Retry-After`.
- **SC-009**: Synthetic prompt-injection test (function tool returning `sk-ant-api03-XXXX`) verifies both context redaction (model sees `[REDACTED:anthropic-key]`) and OTel attribute redaction.
- **SC-010**: `security_profile: hardened` deploys two containers; the agent container env has zero credential-bearing vars; calls to non-allowlisted domains are rejected by Envoy.
- **SC-011**: With `provider: openai`, traces appear in both the OTel collector AND `platform.openai.com/traces`. With `provider: azure_openai`, traces appear in OTel only — verified end-to-end against a real deploy.
- **SC-012**: An agent declared with `agent_mode: sandbox` and the safety opt-in successfully executes a shell command (e.g. `python -c 'print(1+1)'`) inside the sandbox workspace; the same command issued through a `standard`-mode agent has no shell tool to invoke. Verified by integration test on both `provider: openai` and `provider: azure_openai`.
- **SC-013**: Under `security_profile: hardened` with `agent_mode: sandbox`, the sandbox client is the configured remote variant (Docker by default); workspace state never lands on the agent container filesystem. Verified by inspecting the deployed Container App spec and the sandbox client's remote handle.

## Clarifications

### Session 2026-05-24

- Q: Should `openai_agents` take over the default for `provider: openai` / `azure_openai`? → A: Yes. Clean break, same pattern spec 023 used for SK → AF. `agent_framework` becomes the explicit-opt-in path. No deprecation window.
- Q: How should tracing be wired? → A: For `provider: openai`, keep both OpenAI dashboard upload AND OTel mirror. For `provider: azure_openai`, disable the OpenAI dashboard upload (Azure customers should not leak telemetry to OpenAI's hosted infra); keep OTel. New schema field `observability.disable_provider_tracing` is a per-agent override.
- Q: Does the in-process execution model change the spec 034 hardening shape? → A: Yes. P1a sizing recalibrated (1 CPU / 1 GiB default; `session_memory_estimate_mib: 100`). P1b targets hosted tools (CodeInterpreter / Computer) rather than Bash/Write/Edit/WebFetch. P2a Dockerfile drops Node.js by default. P2b credential redaction implemented as a function-tool wrapper (since `AgentHooks.on_tool_end` is observation-only). P3 unchanged in shape, with `OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` rewritten through the sidecar. P4 is dropped entirely — the SDK's native `session=SQLiteSession(id)` already delivers the hybrid model.
- Q: Should the SDK config additions from spec 026 carry over 1:1? → A: Not exactly. `effort` maps to `ModelSettings.reasoning` (with `max` clamped to `high`). `disallowed_tools` is a config-time filter. `max_budget_usd` is a best-effort `RunHooks` accountant. `fallback_model` is a wrapping `Model` provider for retryable errors. All four are supported in YAML; their backend-specific limitations are documented in the field descriptions.
- Q: Should YAML hook actions `reject` and `modify` work the same as on Claude? → A: `reject` works via generated `input_guardrail` + per-tool `needs_approval`. `modify` is deferred — the SDK has no observation hook that can mutate tool input/output, so v1 emits a warning and treats the hook as inert. Cross-backend YAML stays portable; the warning makes the delta visible.
- Q: Should `setting_sources` (spec 030) raise a load error on `openai_agents`? → A: No. The SDK has no analogous subsystem; emit a warning and silently ignore for cross-backend portability.
- Q: How should subagent `model` field literals be handled? → A: `inherit` is universal. The Claude literals (`sonnet | opus | haiku`) are rejected on `openai_agents`; any string the SDK accepts as a model id is allowed.
- Q: Should we wrap every openai_agents agent in `SandboxAgent` to mirror spec 034's isolation story? → A: No — `SandboxAgent` and spec 034 hardening solve complementary problems (agent→container vs. container→host). Add `agent_mode: sandbox` as an opt-in (US8) for agents needing shell/code execution; keep 034's container hardening as the default posture for all agents regardless of mode. The safety opt-in (`openai_agents.i_understand_this_is_unsafe`) gates both `CodeInterpreterTool`/`ComputerTool` AND `agent_mode: sandbox` — one named flag for "this agent has dangerous primitives."
- Q: Should the `claude.*` schema namespace be reused for hooks/agents or should there be a new top-level `openai_agents.*` block? → **Open Question 1, see below.**

## Assumptions

- The openai-agents SDK Python package (`openai-agents`) is available on PyPI at a tested version. Added as an optional extra (`uv add 'openai-agents[litellm,voice]'` is **not** required; we only need the base extras).
- The SDK's `Agent`, `Runner`, `Session`, `MCPServerStdio`, `MCPServerSse`, `MCPServerStreamableHttp`, `function_tool`, `input_guardrail`, `output_guardrail`, `AgentHooks`, `RunHooks`, `TracingProcessor`, `set_tracing_disabled` symbols are stable in the current release. Pinned to a known version in `pyproject.toml`.
- The custom `TracingProcessor` API for mirroring SDK trace events into OTel is available via `agents.tracing.add_trace_processor(...)`.
- `OPENAI_API_KEY` is the canonical credential env var for `provider: openai`. For `provider: azure_openai`, the credential set is `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (existing HoloDeck convention).
- The SDK's hosted tools (`WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, `ComputerTool`, `HostedMCPTool`) require the OpenAI Responses API and are therefore unavailable on `provider: azure_openai` in v1. This is enforced at config validation, not at runtime.
- The SDK's session backends include `SQLiteSession` (used by HoloDeck), `OpenAIConversations`, Redis, and SQLAlchemy options. HoloDeck defaults to `SQLiteSession`; other backends are documented but not directly exposed in YAML in v1.
- The Microsoft Agent Framework backend (`agent_framework`) remains supported for users who explicitly select it. Spec 023's tool adapter coverage there is unchanged.
- The Envoy sidecar image used for `security_profile: hardened` (Claude P3) is provider-neutral; the only change for `openai_agents` is the allowlist contents and the env-var names rewritten in the agent container (`OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` instead of `ANTHROPIC_BASE_URL`).
- Per-model pricing for the cost accountant is bundled with the backend module (versioned constant). Operators who run new models before the table is updated see the budget enforcement degrade to a no-op with a warning — not a crash.

## Open Questions

1. **Namespacing for hooks / subagents / permissions blocks.** Specs 028 and 029 land hooks and subagents under `claude.hooks` and `claude.agents` respectively. That namespace is misleading once the same concepts run on `openai_agents` (and ADK / AF in the future). Two paths:
   - **(a)** Keep `claude.hooks` / `claude.agents` and document that the names are historical. The YAML stays portable across backends.
   - **(b)** Introduce a backend-agnostic `agents:` / `hooks:` top-level block, with `claude.*` as a deprecated alias. Larger schema change but honest.
   - Recommendation: **(a)** for v1, schedule **(b)** as a separate schema-cleanup spec to avoid breaking shipped configs.

2. **`modify` hook action implementation strategy.** Two options for landing it after v1:
   - Wrap each function tool with a pre-call transformer that mutates `tool_input` before invocation. This works for function tools but cannot reach MCP tools or hosted tools (their input is shaped by the SDK and reaches them out-of-band).
   - Route through a custom `Model` provider that intercepts every tool-call instruction the LLM produces and rewrites it before the SDK dispatches. Works for all tool types but is invasive.
   - v1: warn and treat as inert. v2: pick one of the above based on usage signal.

3. **Cost-accountant price table source of truth.** Bundling a constant in the backend module is the simplest v1 approach but rots fast. Options:
   - Manual update on each model release (status quo).
   - Pull from a versioned manifest in this repo (Renovate-managed PR cadence).
   - Pull at startup from a vendor endpoint — adds a network dependency at agent init time.
   - v1: bundled constant + warning when an unknown model is queried.

4. **`session_memory_estimate_mib` calibration.** Claude P1a calibrated `400 MiB/session` against real observation. The 100-MiB default here is an educated guess — the in-process model means a session is mostly conversation state in SQLite + tool state + a handful of asyncio tasks. Calibrate this in implementation against a load test on the financial-assistant sample before locking the default.

5. **Streaming AG-UI shape parity.** The SDK's `RawResponsesStreamEvent` chunk shape is close to but not identical to Claude's `client.send_streaming()` shape. Confirm during implementation that the existing AG-UI bridge consumes both without protocol changes; if not, the bridge needs a per-backend serializer (separable PR).

## How HoloDeck's structure changes

### Schema additions

```yaml
# New top-level block (sibling to `claude:`)
openai_agents:
  # P1a sizing
  max_concurrent_sessions: null          # default: derived from cgroup memory
  session_memory_estimate_mib: 100       # per-session memory budget for derivation
  # P1b safety
  i_understand_this_is_unsafe: false     # required to enable CodeInterpreter / Computer tools
  # P2b
  disable_default_hooks: false
  disable_subprocess_env_scrub: false
  permissions:
    allowed_tools: [...]
    disallowed_tools: [...]
  # Spec-026-style fields, parity with claude.*
  effort: low | medium | high            # max clamps to high with warning
  max_budget_usd: null
  fallback_model: null
  disallowed_tools: [...]
  # Spec-027 reuses tool.transport — no new field

  # US8 — Sandbox mode
  agent_mode: standard | sandbox         # default: standard
  sandbox:
    remote_client: docker | modal        # default: docker; only used under security_profile: hardened
    workspace_size_limit_mb: 512
    idle_ttl_seconds: 3600

# Tracing (new)
observability:
  disable_provider_tracing: false        # NEW — suppresses platform.openai.com upload
  # existing OTel fields unchanged

# Deployment (parity with spec 034 P3)
deployment:
  security_profile: default | hardened
```

`claude.hooks` and `claude.agents` are read by both backends — see Open Question 1.

### Backend changes

| File | Change |
|---|---|
| `src/holodeck/lib/backends/openai_agents_backend.py` | NEW — `OpenAIAgentsBackend` + `OpenAIAgentsSession` implementing spec-023 protocols |
| `src/holodeck/lib/backends/openai_agents_tool_adapters.py` | NEW — translators for function / vectorstore / hierarchical_document / mcp / skill / hosted tools |
| `src/holodeck/lib/backends/openai_agents_hooks.py` | NEW — default function-tool decorator (credential redaction); `RunHooks`/`AgentHooks` integration |
| `src/holodeck/lib/backends/openai_agents_tracing.py` | NEW — custom `TracingProcessor` mirroring SDK events into OTel; provider-aware `set_tracing_disabled` switch |
| `src/holodeck/lib/backends/openai_agents_cost.py` | NEW — `RunHooks`-based cost accountant for `max_budget_usd` enforcement |
| `src/holodeck/lib/backends/openai_agents_sandbox.py` | NEW — sandbox client factory (Unix local / Docker / Modal); `Manifest` builder; workspace lifecycle |
| `src/holodeck/lib/backends/selector.py` | Updated routing table (FR-007) |
| `src/holodeck/lib/backends/otel_redaction.py` | NO change — already backend-agnostic |
| `src/holodeck/lib/backends/validators.py` | Add `_build_openai_agents_options()`; hosted-tool safety gate; provider-allowed-on-Azure gate |

### Serve / deploy changes

| File | Change |
|---|---|
| `src/holodeck/serve/server.py` | Resolved-config echo grows an "OpenAI Agents" section parallel to "Claude"; reuse existing 429 / readiness wiring |
| `src/holodeck/deploy/dockerfile.py` | New branch for `openai_agents`: pure Python by default; Node.js gated only on stdio MCP server requirement |
| `src/holodeck/deploy/deployers/azure_containerapps.py` | Default sizing 1 CPU / 1 GiB for `openai_agents`; hardened-profile sidecar uses OpenAI/Azure allowlist |
| `src/holodeck/deploy/envoy.py` | Allowlist derivation switch on `model.provider` (Anthropic / OpenAI / Azure) |
| `src/holodeck/cli/commands/deploy.py` | Echo backend-specific resolved config |
| `src/holodeck/cli/commands/serve.py` | Same |

### Model changes

| File | Change |
|---|---|
| `src/holodeck/models/agent.py` (or wherever `Backend` lives) | Add `openai_agents` to the `Backend` enum; update default-routing table |
| `src/holodeck/models/openai_agents_config.py` | NEW — `OpenAIAgentsConfig`, `OpenAIAgentsPermissionsConfig`, hosted-tool models |
| `src/holodeck/models/observability.py` | Add `disable_provider_tracing` field |
| `src/holodeck/models/deployment.py` | Default `cpu`/`memory` to `1.0`/`1Gi` when `model.provider in {openai, azure_openai}` |
| `src/holodeck/models/tool.py` | Add `HostedTool` to `ToolUnion` (type: `hosted`) |

### Tests

- `tests/unit/lib/backends/test_openai_agents_backend.py` — protocol conformance, ExecutionResult population, streaming shape
- `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` — per-tool-type adapter coverage incl. hosted tools
- `tests/unit/lib/backends/test_openai_agents_hooks.py` — default function-tool credential redaction; opt-out path; hook chain ordering
- `tests/unit/lib/backends/test_openai_agents_tracing.py` — OTel mirror; provider-aware `set_tracing_disabled` switch; opt-out via `observability.disable_provider_tracing`
- `tests/unit/lib/backends/test_openai_agents_cost.py` — `max_budget_usd` enforcement; unknown-model degradation
- `tests/unit/lib/backends/test_openai_agents_permissions.py` — hosted-tool safety gate; `i_understand_this_is_unsafe` opt-in
- `tests/unit/lib/backends/test_openai_agents_subagents.py` — handoffs translation; `RECOMMENDED_PROMPT_PREFIX` injection; model-literal validation
- `tests/unit/lib/backends/test_openai_agents_sandbox.py` — safety gate; `Manifest` derivation; client switch under hardened profile; redundancy error for `agent_mode: sandbox` + `CodeInterpreterTool`
- `tests/integration/sandbox/test_openai_agents_sandbox_e2e.py` — sandboxed shell execution against a live agent on both `provider: openai` and `provider: azure_openai`
- `tests/unit/serve/test_session_semaphore_openai_agents.py` — 429 overflow against the new backend
- `tests/unit/deploy/test_dockerfile_generation_openai_agents.py` — pure-Python default; Node.js gating on stdio MCP
- `tests/unit/deploy/test_aca_template_openai_agents.py` — sizing default; securityContext gaps documented
- `tests/unit/deploy/test_envoy_generator_openai_agents.py` — OpenAI / Azure allowlist derivation
- `tests/integration/security/test_openai_agents_redaction_e2e.py` — synthetic prompt-injection against a live agent
- `tests/integration/security/test_openai_agents_hardened_e2e.py` — hardened profile end-to-end
- `tests/integration/openai_agents_serve_deploy.py` — serve + deploy parity smoke

## Phase-by-phase ship criteria

The work decomposes the same way spec 034 did, so the two specs ship in parallel without stepping on each other.

| Phase | Ship criteria |
|---|---|
| **P0 — Backend scaffolding** | New backend module, `BackendSelector` routes; single-turn `Runner.run` works against `gpt-4o`; ExecutionResult populated. Unit tests pass. |
| **P1 — Tool adapters** | All 5 HoloDeck tool types translate; vectorstore + hierarchical_document use LiteLLM embedder; MCP stdio/sse/http via SDK classes; skill → handoff target. Per-tool-type unit tests pass. |
| **P2 — Serve & deploy parity** | `holodeck serve` + `holodeck deploy build/run` work against an OpenAI agent; pure-Python Dockerfile; /health, /ready, /awp behave; integration test exercises a live ACA deploy. |
| **P3 — Spec 026/028/029 mappings** | `effort` / `disallowed_tools` / `fallback_model` / `max_budget_usd` work with the documented per-field semantics. YAML hooks (`log`, `notify`, `script`, `reject`) work; `modify` is inert + warning. Subagents translate to handoffs with `RECOMMENDED_PROMPT_PREFIX`. |
| **P4 — Hosted tools** | `WebSearchTool` / `FileSearchTool` / `CodeInterpreterTool` / `ImageGenerationTool` / `ComputerTool` / `HostedMCPTool` declarable in YAML; per-tool unit tests; safety gate for CodeInterpreter / Computer. |
| **P5 — Tracing** | OTel mirror processor registered; `provider: openai` → both; `provider: azure_openai` → OTel-only; `observability.disable_provider_tracing` honoured. Integration test confirms both destinations on a real deploy. |
| **P6 — Hardening (P1a/P1b/P2a/P2b/P3 of spec 034, reframed)** | Default sizing + 429 + readiness; hosted-tool safety gate; pure-Python Dockerfile with corpus chmod + tmpfs; credential redaction in function-tool returns and OTel; `security_profile: hardened` sidecar with OpenAI/Azure allowlist. |
| **P7 — Default routing flip** | `BackendSelector` updates: `openai` / `azure_openai` default to `openai_agents`. Migration note in release notes; AF remains opt-in via explicit `backend: agent_framework`. |
| **P8 — Sandbox agent mode (US8)** | `agent_mode: sandbox` accepted in YAML; safety gate enforced; `UnixLocalSandboxClient` works under default profile; remote sandbox client (Docker) works under hardened profile. End-to-end test on `provider: openai` + `provider: azure_openai`. Modal client is follow-up. |

P0–P5 are independent and can land in parallel PRs. P6 depends on P0/P2/P3. P7 lands last (the default flip is a single line + sample updates + docs).

## Cost reality

The runtime delta vs. AF is minimal: the SDK's `Runner` overhead is a thin wrapper over the Responses API call. The bigger deltas are operational:

- **Image size**: removing the Node.js layer shrinks the agent image by ~50 MiB → faster pulls, faster cold starts. Net win.
- **Memory floor**: 1 GiB default vs. Claude's 2 GiB default. Halved per-replica memory bill on ACA consumption profile.
- **Concurrent sessions**: 10 default vs. Claude's 2 default for the same memory. 5× concurrency at parity-ish cost.
- **Tracing**: OpenAI dashboard is included in the API spend (no additional line item). OTel costs are operator-side and unchanged.
- **P3 sidecar**: adds the same ~50–80 MiB Envoy overhead and ~1ms latency per upstream request — identical to the Claude P3 cost.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Default routing flip breaks operators' existing OpenAI agents | Spec 023's clean-break precedent applies; release notes call out the change. Operators wanting AF set `backend: agent_framework`. |
| Hosted tools cost more than function tools and operators don't realise | Resolved-config echo lists every declared hosted tool at deploy time; documentation flags the cost surface; `max_budget_usd` is the safety net. |
| `modify` hook semantics drift between Claude and openai_agents because v1 is inert here | Warning at load time names the inert hook; docs/security/hooks.md lists per-backend semantics matrix. |
| `max_budget_usd` price table goes stale | Unknown-model degradation is a warning + no-op; documentation explains how to add a model entry; periodic update via Renovate PR. |
| `fallback_model` masks systematic upstream issues by silently routing traffic | Both attempts recorded in the trace; the fallback is gated to a small retryable-error set, not catch-all. |
| `session_memory_estimate_mib: 100` default is wrong for some workloads | Documented as a heuristic + tunable; calibration is an Open Question for implementation; the resolved cap is echoed so operators can tune. |
| Azure customers accidentally enable a hosted tool that doesn't work there | Config-time validation catches this (FR-091 / US5 acceptance scenario 3); no runtime surprise. |
| Tracing dual-write produces twice the span volume for `provider: openai` operators | Documented; opt-out is `observability.disable_provider_tracing: true`. OTel sampling can be tuned independently. |
| Operators reach for `agent_mode: sandbox` without understanding it grants shell access | Safety gate (`i_understand_this_is_unsafe`) is the same flag that already gates `CodeInterpreterTool` / `ComputerTool`; one named opt-in covers all dangerous primitives. Deploy-time echo lists the active mode. |
| `UnixLocalSandboxClient` is a process boundary, not a kernel boundary — operators may assume more isolation than they get | Documentation is explicit that local sandbox is process-level; kernel-level isolation requires `security_profile: hardened` with a remote sandbox client. |
| Remote sandbox client (Docker / Modal) introduces a new credential surface | Hardened profile already moves credentials to the Envoy sidecar; remote sandbox client credentials land there too, not in the agent container. |

## Out of scope for v1

| Out-of-scope | Why deferred | Path forward |
|---|---|---|
| `VoicePipeline` (STT + agent + TTS) | Audio I/O introduces a substantially different YAML surface | Separate spec — likely 03X-voice-pipeline-backend |
| `RealtimeAgent` (gpt-realtime WebSocket sessions) | Same as above — realtime transport doesn't fit the current AG-UI bridge | Separate spec |
| Modal as a remote sandbox client | Modal credentials and deployment story is its own scope | Follow-up to P8; Docker remote client ships first in P8 |
| `modify` YAML hook action implementation on `openai_agents` | No SDK observation hook can mutate; would require wrapping every tool or a custom Model provider | Follow-up spec; v1 warns and treats inert |
| AF backend removal | AF stays as the explicit-opt-in alternative for operators who depend on its compaction / workflow features | Deprecation TBD |
| Cross-backend hooks/subagents namespace cleanup (`claude.hooks` → top-level `hooks:`) | Schema-breaking; cleaner as its own spec | Schedule a follow-up spec post-flip |
| `setting_sources` parity (loading `.claude/skills/SKILL.md` ambient skills on `openai_agents`) | Requires inventing a skill-discovery layer the SDK doesn't have | Defer until `SkillTool` declarative coverage proves insufficient |
| Per-session ephemeral containers (Pattern 1 from spec 034) | Same out-of-scope as Claude P1 | Separate spec covering hostile multi-tenant workloads |

## v1 contract

- New backend identifier `openai_agents` and module implementing spec-023 protocols.
- Default routing flip: `openai` / `azure_openai` → `openai_agents`. Clean break.
- All 5 HoloDeck tool types work; hosted tools added as `type: hosted`.
- Full parity with specs 024 (serve/deploy), 025 (tool init), 027 (MCP transports), 029 (subagents). Specs 026/028/030 carry over with the documented per-field mappings and one deferred action (`modify`).
- Hardening posture from spec 034 phases P1a/P1b/P2a/P2b/P3 applied with in-process reframe. P4 is N/A (SDK native).
- Tracing dual-write for `provider: openai`; OTel-only for `provider: azure_openai`; per-agent override via `observability.disable_provider_tracing`.
- One safety opt-in (`openai_agents.i_understand_this_is_unsafe: true`) gating `CodeInterpreterTool`, `ComputerTool`, and `agent_mode: sandbox`. Same name signal as Claude P1b.
- Sandbox mode (`agent_mode: sandbox`) opt-in for agents needing shell/code execution; complementary to (not a replacement for) spec-034 container hardening.
- Resolved limits, hosted tools, tracing destination, and security profile echoed at `serve` and `deploy run` time.
- No required user YAML changes for operators on OpenAI defaults; auto-detection + auto-derived caps handle the migration.

## Implementation surface — modules to create / modify

**New modules:**

- `src/holodeck/lib/backends/openai_agents_backend.py`
- `src/holodeck/lib/backends/openai_agents_tool_adapters.py`
- `src/holodeck/lib/backends/openai_agents_hooks.py`
- `src/holodeck/lib/backends/openai_agents_tracing.py`
- `src/holodeck/lib/backends/openai_agents_cost.py`
- `src/holodeck/lib/backends/openai_agents_sandbox.py`
- `src/holodeck/models/openai_agents_config.py`

**Modified modules:**

- `src/holodeck/lib/backends/selector.py` — default routing table; lazy import gate
- `src/holodeck/lib/backends/validators.py` — `_build_openai_agents_options()`
- `src/holodeck/lib/backends/otel_redaction.py` — verify backend-agnostic coverage (no code change expected)
- `src/holodeck/serve/server.py` — resolved-config echo
- `src/holodeck/serve/protocols/agui.py` — verify stream-shape compat (per Open Question 5)
- `src/holodeck/deploy/dockerfile.py` — pure-Python branch for `openai_agents`
- `src/holodeck/deploy/deployers/azure_containerapps.py` — sizing default; hardened-profile env rewriting
- `src/holodeck/deploy/envoy.py` — provider-aware allowlist
- `src/holodeck/models/agent.py` — extend `Backend` enum; default-routing table
- `src/holodeck/models/observability.py` — `disable_provider_tracing`
- `src/holodeck/models/deployment.py` — provider-aware defaults
- `src/holodeck/models/tool.py` — `HostedTool` in `ToolUnion`
- `src/holodeck/cli/commands/deploy.py` — echo resolved config
- `src/holodeck/cli/commands/serve.py` — same
- `pyproject.toml` — add `openai-agents` as an optional extra

**Tests:** see `## How HoloDeck's structure changes → Tests` above for the full list.

## Dependencies to add

- `openai-agents` (Python; pinned to a tested version in `pyproject.toml`) as an optional extra. Lazy-imported by the new backend so users on other backends incur no import cost.
- No new runtime image dep beyond the existing Envoy sidecar (P3 only).
- No new Node.js dependency by default; the existing conditional Node.js install (per spec 034 P2a) handles stdio-MCP-server cases.
