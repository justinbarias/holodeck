# Feature Specification: OpenAI Agents SDK Backend

**Feature Branch**: `035-openai-agents-backend`
**Created**: 2026-05-24
**Status**: Partially implemented — final requirement contract reconciled 2026-09-06; implementation and acceptance remain open.
**Author**: justinbarias (with Claude)
**Input**: OpenAI-native backend with the retained parity of specs 023–030 and in-process adaptations of spec 034.

## Implementation status — 2026-09-06

The [completion execution plan](../../exec-plans/active/035-openai-agents-backend/2026-09-06-complete-035.md) owns execution order.
The [acceptance matrix](../../exec-plans/active/035-openai-agents-backend/acceptance-matrix.md) is the canonical requirement-by-requirement evidence record.
The [reconciliation report](../../exec-plans/active/035-openai-agents-backend/reconciliation.md) records the audited baseline.
Requirements below describe the retained contract, not proof that implementation is complete.
All original FR and SC identifiers remain; deferred criteria retain their proposed behavior and an explicit destination.
The [decision register](#decision-register-2026-09-06) replaces obsolete assumptions throughout the original draft.

## Motivation and architecture

OpenAI and Azure agents use the provider-native OpenAI Agents SDK through `BackendSelector` and the shared backend/session protocols.
Anthropic and Ollama continue through Claude. The native backend supports Responses tools, SDK sessions, handoffs, guardrails, and tracing.
No AF/ADK/SK agent backend or explicit backend-selection field is available in the current repository.

The OpenAI loop runs in-process Python; the Claude SDK uses a subprocess. OpenAI turn capacity therefore counts active turns, while idle session history uses the SDK session abstraction (`SQLiteSession`).
Idle sessions do not hold dedicated processes; this is not a claim that Python metadata uses zero memory.
Images are pure Python unless a configured MCP stdio command requires Node.js.
Container protections and model-visible credential redaction remain distinct boundaries.
The deferred sandbox and Envoy proposals require their own threat models; neither is delivered by this feature.

## User scenarios and acceptance

### User Story 1 — Configure a native agent (P1)

An operator sets `model.provider: openai` or `azure_openai`, configures credentials and a model, and runs `holodeck test`, `chat`, or `serve`.
`BackendSelector` chooses OpenAI Agents without a backend override. `Runner` returns text, tool results, and usage with the configured agent identity.
Azure uses `AsyncOpenAI` with its endpoint normalized to `/openai/v1` and an `OpenAIResponsesModel`; its result contract matches OpenAI.
Missing credentials produce an actionable startup error. Provider account/model-access errors preserve the upstream message.
Explicit AF opt-out from the original US1 scenario 4 is superseded by D01; no replacement backend is promised.

### User Story 2 — Serve and deploy parity (P1)

The same agent serves REST and AG-UI through `/awp` and builds/runs on Azure Container Apps with startup validation, liveness, readiness, internal ingress, and GenAI telemetry.
Use tracked representative fixtures, not ignored `/sample` content, for repeatable tests.
Verify conditional Node installation, non-root execution, read-only corpus, writable scratch space, startup failure before traffic, and actual local image behavior.
Authorized cloud acceptance must verify the candidate image and runtime probes; source inspection alone is insufficient.

### User Story 3 — Tools, hooks, subagents, and skills (P1)

Provider portability uses an `openai:` block for this backend's settings and a separate `claude:` block for Claude.
Verify asynchronous vectorstore/hierarchical-document initialization and polling, plus MCP stdio/SSE/HTTP transport and environment substitution.
Three subagents (researcher, analyst, writer) must become handoff targets with inherited or restricted tools and models; AG-UI displays ordered handoff events.
Inline and SKILL.md skills must produce equivalent restricted targets. `SkillTool` is new implementation work, not an existing model.
Observation, rejection, failure, and inert `modify` behavior follow FR-050; unsupported guardrail targets fail clearly.
`setting_sources` remains accepted with a Claude-only warning and no OpenAI ambient discovery.

### User Story 4 — SDK configuration (P2)

`openai.effort` maps low/medium/high directly and max to xhigh. Reasoning summaries must be requested when reasoning effort is configured; absent summaries leave thinking empty.
Budget accounting is best-effort using per-request usage and a bundled versioned price table. Unknown model prices warn and disable enforcement for that model.
Budget exhaustion returns an error with partial response and accumulated cost. Streaming reports errors after any already-emitted text.
Fallback must exhaust primary retries before one fallback attempt, preserve both attempts in allowed traces, and never duplicate streamed output.
Tool permission conflicts fail load; merely declaring a disallowed tool causes it to be filtered before construction, not a declaration-conflict error.

### User Story 5 — Hosted tools (P2)

Five hosted tools are retained: `WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`, and `HostedMCPTool`.
A validated `type: hosted` entry selects the class and constructor parameters. Unknown names or missing required parameters fail with actionable configuration errors.
File search forwards vector-store IDs and result limits. Code interpreter, image generation, and hosted MCP factories construct the SDK's nested `tool_config` objects.
Code interpreter requires the unsafe opt-in. Computer use fails load with a computer-harness/deferred-feature explanation, even with that opt-in.
Azure accepts hosted configuration; unsupported resource capabilities surface the SDK error with a useful Azure-capability hint. No blanket Azure load-time ban applies.
Hosted MCP with `require_approval: never` runs without a local approval loop; automatic rejection follows FR-050.
Per-factory unit tests and real supported-provider calls must verify tools, citations/grounded results where applicable, and capability-error handling.

### User Story 6 — Production hardening (P1)

Retain spec-034 P1a capacity, P1b code-interpreter safety, P2a container protections, and P2b redaction/managed child-environment controls.
Verify exact resolved active-turn capacity, overflow/recovery, cancellation cleanup, and sparse activity across many idle sessions without per-session processes.
A synthetic credential returned by a HoloDeck-built local tool must be replaced before the model sees it and independently redacted in exported OTel attributes.
Explicit guardrail and subprocess opt-outs warn. They must not alter independent OTel redaction.
P3 sidecar/egress isolation is deferred (D08). SDK sessions address the subprocess-retention motivation of Claude P4; active-turn and memory behavior still require measurement.

### User Story 7 — Tracing (P2)

With OTel enabled, OpenAI agents emit to OTel and the provider dashboard unless provider upload is disabled; Azure emits only to OTel.
Processor selection must preserve spans while suppressing disallowed uploads. Test disabled observability, repeated initialization, and mixed-provider initialization in both orders.
`capture_content: false` explicitly excludes sensitive payloads from SDK traces. OTel redaction applies independently.
Real deployment evidence must observe the permitted destinations and verify absence of prohibited upload; mocked processor construction alone cannot close SC-011.

### User Story 8 — Sandbox mode (deferred)

Shell/filesystem mode, manifests, local workspace cleanup, Docker/Modal remote clients, resource quotas, missing-runtime/credential preflight, and standard/sandbox redundancy checks remain future design work.
FR-094–FR-099 and SC-012/SC-013 retain the proposal, with [H-011](../../exec-plans/tech-debt-tracker.md#h-011) as its destination.
A future sandbox design must test provider support, disk exhaustion, shell execution restricted to its workspace, and workspace deletion at session close.
No `agent_mode` or `sandbox` YAML support is claimed by 035 completion.

## Edge cases

- Missing provider credentials must fail preflight with named configuration/environment remedies; account/model-access failures preserve useful upstream causes.
- Invalid or unknown tool references must produce the existing actionable validation warning/error; do not silently grant unrestricted tools. A nonexistent name in `disallowed_tools` warns at configuration load.
- A primary non-retryable failure must propagate without fallback; cancellation or stream closure must clean up sessions, MCP resources, and capacity slots.
- Input rejection during handoffs applies to the entry/triage agent, following SDK tripwire semantics. Tool rejection and failure wrappers follow their separately documented local-tool coverage.
- Unsupported WebSocket MCP and prompt tools warn and skip. Unsupported hosted names and unreachable rejection targets fail load; unreachable observation/failure matchers warn.
- Missing sandbox runtimes, disk exhaustion, remote credentials, quotas, and local/remote workspace cleanup belong to the explicitly deferred sandbox design, not hidden 035 implementation requirements.

## Scope boundary

In scope: shared backend/session contracts; multimodal and structured results; five HoloDeck tool categories (function, vectorstore, hierarchical-document, MCP, skill); five hosted tools; hook/handoff semantics; serving and deployment; retained hardening and telemetry; LiteLLM inference acceptance; reproducible examples and documentation.

Deferred scope has owners and exit criteria in the [debt tracker](../../exec-plans/tech-debt-tracker.md):

| Excluded capability | Durable destination |
| --- | --- |
| Final SK vector connectors, chunker, package removal | [H-008](../../exec-plans/tech-debt-tracker.md#h-008) |
| Cross-backend P3 Envoy profile | [H-010](../../exec-plans/tech-debt-tracker.md#h-010) |
| Sandbox mode, local/Docker/Modal runtime | [H-011](../../exec-plans/tech-debt-tracker.md#h-011) |
| Computer harness | [H-012](../../exec-plans/tech-debt-tracker.md#h-012) |
| Interactive human approval/resume | [H-013](../../exec-plans/tech-debt-tracker.md#h-013) |
| SDK-built MCP-server guardrails/model-visible redaction | [H-014](../../exec-plans/tech-debt-tracker.md#h-014) |
| Hook `modify` execution | [H-015](../../exec-plans/tech-debt-tracker.md#h-015) |
| Prompt-tool execution (load warns and skips) | [H-016](../../exec-plans/tech-debt-tracker.md#h-016) |
| Unified cross-backend configuration namespace | [H-017](../../exec-plans/tech-debt-tracker.md#h-017) |
| VoicePipeline and RealtimeAgent | [H-018](../../exec-plans/tech-debt-tracker.md#h-018), [H-019](../../exec-plans/tech-debt-tracker.md#h-019) |
| Ambient skill discovery | [H-020](../../exec-plans/tech-debt-tracker.md#h-020) |
| Per-session containers and arbitrary Python containment | [H-021](../../exec-plans/tech-debt-tracker.md#h-021) |

Additional SDK knobs remain at SDK defaults: tool-loop policy (`tool_use_behavior`, `reset_tool_choice`, `tool_choice`), extra run/agent output guardrails, handoff-history shaping, and SDK error formatting/not-found policy. These were never required configurable surfaces.
There is no new deployment target or session/conversation-control YAML surface. Unknown configuration keys must remain invalid; do not add `previous_response_id` or `conversation_id` solely to preserve an obsolete draft edge case.

## Requirements *(mandatory)*

### Functional Requirements

**Backend contract (spec 023):**

- **FR-001**: System MUST expose the `openai_agents` backend through `BackendSelector`. Routing uses `model.provider`; there is no `Backend` enum or `backend:` override. [D01](#d01).
- **FR-002**: The `openai_agents` backend MUST implement the `AgentBackend` protocol exactly: `initialize()`, `invoke_once()`, `create_session()`, `teardown()`.
- **FR-003**: The `openai_agents` session MUST implement the `AgentSession` protocol exactly: `prepare()`, `send()`, `send_streaming()`, `close()`.
- **FR-004**: `invoke_once()` and `send()` MUST return `ExecutionResult` objects with all fields populated: `response`, `tool_calls`, `tool_results`, `token_usage`, `structured_output`, `num_turns`, `is_error`, `error_reason`, `thinking` (empty when the model is non-reasoning).
- **FR-005**: `send_streaming()` MUST yield successive text chunks consistent with the SDK's `RawResponsesStreamEvent`/`ResponseTextDeltaEvent` shape, surfaced through HoloDeck's existing AG-UI bridge with no protocol-level change.
- **FR-006**: System MUST emit `ToolEvent` records (kind: `start` / `end` / `error` / `thinking`) consistent with the Claude backend's contract, so the AG-UI panel renders identically. `subagent_message` and `parent_link` events MUST be emitted during handoff transitions (mapped from `AgentUpdatedStreamEvent`).
- **FR-007**: `openai` and `azure_openai` MUST route to `openai_agents`; `anthropic` and `ollama` MUST route to `claude`. Unsupported providers MUST fail clearly. AF, ADK, and SK agent backends are not selectable. [D01](#d01).

**Serve & deploy parity (spec 024):**

- **FR-010**: `holodeck serve agent.yaml` MUST perform pre-flight credential validation at startup for `provider: openai` (`OPENAI_API_KEY`) and `provider: azure_openai` (`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`).
- **FR-011**: `holodeck deploy build` MUST generate a Dockerfile that does **not** install Node.js when the agent uses `openai_agents` and has no Node-requiring MCP stdio servers. When such an MCP server exists, Node.js MUST be installed (parity with the spec 034 P2a gating logic).
- **FR-012**: Container entrypoint MUST validate the prerequisite credential set before starting `holodeck serve` and exit non-zero with a structured error if missing.
- **FR-013**: `/health` MUST report process liveness. `/ready` MUST return HTTP 200 only after backend/startup prerequisites and required initializable tools are ready: vectorstore/hierarchical-document initialization is COMPLETED or existing initialized data is verified. Uninitialized, pending, in-progress, failed, or cancelled required initialization MUST return HTTP 503, as must shutdown/draining. With zero initializable tools, readiness follows backend prerequisites; function/MCP tools do not require nonexistent ingestion jobs. Keep the existing endpoint name. [D10](#d10).
- **FR-014**: Serve MUST cap concurrent **active turns**, including streaming turns, at the resolved `openai.max_concurrent_sessions` value. Idle sessions MUST NOT consume turn slots. Overflow returns 429 with `Retry-After`; success, failure, timeout, cancellation, disconnect, and generator closure MUST release acquired slots. Session history uses the SDK session abstraction without a dedicated process per idle session.

**Tool init endpoints (spec 025):**

- **FR-020**: All endpoints in spec 025 (`POST /tools/{name}/init`, `GET /tools/{name}/init`, `GET /tools`) MUST work identically against an `openai_agents` agent. Vectorstore and hierarchical_document tools MUST be initializable; non-init tool types MUST return 400.

**SDK config additions (spec 026, with backend-specific mappings):**

- **FR-030**: `effort: low | medium | high` MUST translate to `ModelSettings(reasoning=Reasoning(effort=<value>))`.
- **FR-031**: `openai.effort: max` MUST map to reasoning effort `xhigh`, without a clamping warning. [D04](#d04).
- **FR-032**: `max_budget_usd: <float>` MUST be enforced by a HoloDeck-managed `RunHooks` cost accountant; budget exhaustion raises `BackendBudgetExceededError` and aborts the run. Backend MUST surface the partial response and accumulated cost in the error payload.
- **FR-033**: `openai.fallback_model` MUST wrap the primary model with a bounded fallback policy for 429/5xx errors. Exhaust configured primary retries before one fallback attempt; do not fallback on non-retryable errors or restart a stream after its first emitted event. Both attempts MUST be visible in permitted traces. The retryable set is fixed in v1. Function/MCP support is required; document hosted-tool limitations when the fallback lacks Responses capabilities. The bounded order is [D14](#d14); Runner-level acceptance is recorded in the matrix.
- **FR-034**: `disallowed_tools: [<str>, ...]` MUST be applied at config-time tool-resolution: named tools are removed from `Agent.tools` and `mcp_servers`. If a tool name appears in both `allowed_tools` and `disallowed_tools`, config load MUST fail.

**MCP transports (spec 027):**

- **FR-040**: MCP tools with `transport: stdio` MUST translate to `MCPServerStdio(params={command, args, env})`.
- **FR-041**: MCP tools with `transport: sse` MUST translate to `MCPServerSse(params={url, headers})`. URL and header env-var substitution MUST work (parity with spec 027 FR-005).
- **FR-042**: MCP tools with `transport: http` MUST translate to `MCPServerStreamableHttp(params={url, headers})` with the same env-var substitution.
- **FR-043**: MCP tools with `transport: websocket` MUST be skipped with the same warning shape spec 027 FR-006 prescribes for the Claude backend.
- **FR-044**: MCP tool filtering MUST be supported via `create_static_tool_filter(allowed_tool_names=[...])` when an MCP tool config declares an `allowed_tools` subset.

**YAML hooks (spec 028, with semantic mapping):**

- **FR-050**: `openai.hooks` MUST support the following event/action contract. [D05](#d05).
  - `PreToolUse` and `PostToolUse` observation (`log`, `notify`, `script`) use local SDK tool-start/tool-end lifecycle events. MCP-server calls are observable, but hosted tools execute server-side and do not emit those local lifecycle events.
  - `PostToolUseFailure` uses a HoloDeck adapter exception wrapper: fire the action and return a model-visible error string so the run can continue. SDK-built MCP invokers are outside this wrapper. Warn on unreachable hosted-only matchers and MCP failure matchers.
  - `Stop` uses agent/run end; `Notification` uses LLM start/end; `SessionStart` fires once at session creation. Hook scripts use the managed child-environment boundary in FR-089.
  - Tool-matched `reject` on HoloDeck-built local tools uses a native tool-input guardrail returning the configured rejection message; skip the call and continue the run without an approval interrupt.
  - Input-matched `reject` uses the entry agent's input guardrail: tripwire aborts the turn and the backend surfaces the configured message. It does not continue the run; document the entry-agent limitation during handoffs.
  - `HostedMCPTool` rejection sets matched `require_approval` entries in the nested MCP tool configuration and an `on_approval_request` callback that rejects automatically with the configured message. This is not an interactive resume loop.
  - A reject targeting SDK-built MCP-server tools or other hosted tools MUST fail load clearly because the required attachment point is unavailable. Interactive approval and MCP guardrail expansion are deferred to [H-013](../../exec-plans/tech-debt-tracker.md#h-013) and [H-014](../../exec-plans/tech-debt-tracker.md#h-014).
  - `modify` MUST load with an explicit inert-action warning and have no runtime effect; implementation is deferred to [H-015](../../exec-plans/tech-debt-tracker.md#h-015). Observation hooks cannot reject a tool after execution.
- **FR-051**: Hook chain ordering MUST match spec 028 FR-010: HoloDeck-internal hooks (tool tracking, credential redaction) run before user-defined hooks; user-defined hooks run in declaration order; the first terminal action (reject) stops evaluation.

**Subagent orchestration (spec 029):**

- **FR-060**: Each entry in `openai.agents` MUST become an SDK `Agent(name=<key>, instructions=<prompt>, handoff_description=<description>, tools=<resolved>, model=<resolved>)` in the parent's `handoffs`. Support prompt and prompt-file forms. `claude.agents` remains Claude-only. [D02](#d02).
- **FR-061**: When a subagent declares no `tools`, the parent's full tool list MUST be inherited (parity with spec 029 FR-007).
- **FR-062**: When a subagent declares `model: inherit`, the parent's model MUST be used. When it declares `sonnet | opus | haiku`, load MUST fail (these are Claude model literals; not portable). For openai-agents, the allowed values are `inherit` plus any string the SDK accepts as a model identifier.
- **FR-063**: `RECOMMENDED_PROMPT_PREFIX` from `agents.extensions.handoff_prompt` MUST be auto-prepended to each subagent's `instructions` unless the YAML declares `openai.agents.<name>.skip_recommended_prefix: true`.

**Skills (spec 023 SkillTool + spec 030):**

- **FR-070**: Add a validated `SkillTool` (`type: skill`) to `ToolUnion`; it does not exist at the audited baseline. Inline form (`instructions`, `description`, `allowed_tools`) and a directory containing SKILL.md (body and frontmatter) MUST become equivalent handoff-target Agents. `allowed_tools` MUST restrict the skill agent to its declared tool scope. [D02](#d02).
- **FR-071**: `setting_sources` (spec 030) MUST be accepted in YAML for cross-backend portability but emit a load-time warning when the resolved backend is `openai_agents`: "setting_sources is a Claude-only concept; ignored on openai."

**Hardening (spec 034, in-process reframe):**

- **FR-080 (P1a)**: Retain the supported ACA Consumption default of 1 CPU / 2 GiB. Resolve active-turn capacity from explicit `openai.max_concurrent_sessions` first, otherwise `floor(memory_mib / openai.session_memory_estimate_mib)` with a default estimate of 100 MiB. Use a finite cgroup memory limit when available; otherwise use an explicitly labelled assumed 1024-MiB local budget. Do not subtract the Claude process baseline. Derived values below one MUST fail startup with an actionable override/configuration error. Serve and deploy MUST echo budget/source, estimate, override or derivation, and resolved capacity consistently. Default ACA 2048 MiB resolves to 20 turns; local 1024 MiB resolves to 10. [D11](#d11).
- **FR-081 (P1a)**: Overflow MUST return 429 with `Retry-After` and a problem+json body whose `type` is `https://holodeck.dev/errors/session-cap-exceeded`. Parity with spec 034 P1a.
- **FR-082 (P1a)**: `max_turns` default MUST be `20` (SDK default is `10`; bumped for parity with the Claude backend's spec-034 default).
- **FR-083**: `CodeInterpreterTool` MUST fail configuration load unless `openai.i_understand_this_is_unsafe: true`, with an actionable safety opt-in error. `ComputerTool` MUST fail load regardless of opt-in until its harness is delivered under [H-012](../../exec-plans/tech-debt-tracker.md#h-012). [D06](#d06).
- **FR-084**: Permission or hosted-tool configuration MUST NOT weaken default credential guardrails, configured rejection rules, or existing approval gates. An unsupported approval gate MUST fail closed with an actionable error, never run the tool without approval. Interactive `needs_approval`/resume support is deferred to [H-013](../../exec-plans/tech-debt-tracker.md#h-013); it is not the synchronous reject mechanism. [D05](#d05).
- **FR-085 (P2a)**: Generated Dockerfile MUST be pure Python by default (no `apt-get install nodejs npm`). Node.js install is gated only on MCP stdio servers that declare a `command` requiring it. Corpus dirs root-owned + `chmod a-w`. EmptyDir volumes for `/tmp` and `/var/holodeck/work`.
- **FR-086 (P2a)**: Ingress defaults to `false` (internal). Setting `ingress_external: true` emits the same loud deploy-time warning as spec 034.
- **FR-087**: Default-on tool-output guardrails MUST scrub credential-shaped strings before model consumption for HoloDeck-built function, vectorstore, hierarchical-document, and skill-local tools. Reuse the five spec-034 patterns (Anthropic key, AWS access key, GitHub token, JWT, Bearer header). Clean output passes unchanged; redacted output replaces the model-visible result. Opt-out `openai.disable_default_hooks: true` MUST warn loudly and MUST NOT disable OTel redaction. SDK-built MCP-server and hosted tools are outside this model-visible boundary; publish that coverage explicitly. Use the shared `openai_agents_guardrails.py` attachment point rather than an observation hook. [D05](#d05).
- **FR-088 (P2b)**: `RedactingSpanProcessor` from the existing `otel_redaction.py` module MUST scrub `tool.input.*`, `tool.output.*`, `gen_ai.*` span attributes for `openai_agents` spans the same way it does for Claude spans. This is backend-agnostic and requires no new code beyond ensuring the OTel processor sees the new backend's spans.
- **FR-089**: HoloDeck-managed MCP stdio and function-subprocess paths MUST strip OpenAI/Azure/Anthropic credential variables from child environments by default, including substituted configuration env values. Explicit `openai.disable_subprocess_env_scrub: true` MUST produce an opt-out warning. Scrubbing MUST use per-child environments without process-global mutation. Arbitrary Python tools that spawn their own children outside managed paths are not contained; expanded isolation is deferred to [H-021](../../exec-plans/tech-debt-tracker.md#h-021). [D09](#d09).
- **FR-090 (P3)**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-010](../../exec-plans/tech-debt-tracker.md#h-010). Future requirement: `deployment.security_profile: hardened` MUST work identically: Envoy sidecar holds credentials, agent container has none, domain allowlist derived from YAML.
- **FR-091 (P3)**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-010](../../exec-plans/tech-debt-tracker.md#h-010). Future requirement: Allowlist derivation for `provider: openai` MUST include `api.openai.com`, the embedding-provider endpoint, and each MCP HTTP/SSE endpoint. For `provider: azure_openai`, the allowlist MUST additionally include the resolved Azure resource hostname (`<resource>.openai.azure.com`) extracted from `AZURE_OPENAI_ENDPOINT`.
- **FR-092 (P3)**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-010](../../exec-plans/tech-debt-tracker.md#h-010). Future requirement: For `provider: openai`, the agent container MUST be configured with `OPENAI_BASE_URL=http://localhost:<envoy-port>` and `HTTPS_PROXY=http://localhost:<envoy-port>`. For `provider: azure_openai`, `AZURE_OPENAI_ENDPOINT` MUST be rewritten to the localhost sidecar URL; the original Azure hostname is reachable only from the sidecar.
- **FR-093 (P3)**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-010](../../exec-plans/tech-debt-tracker.md#h-010). Future requirement: The `openai_agents` backend MUST refuse to start in hardened profile if any credential-bearing env var is set on the agent container (the operator has half-migrated).

**Sandbox agent mode (US8):**

- **FR-094**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: System MUST accept `openai.agent_mode: standard | sandbox` (default `standard`).
- **FR-095**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: When `agent_mode: sandbox`, the backend MUST construct `SandboxAgent(manifest=Manifest(name=<agent.name>, description=<agent.description>, ...), client=<sandbox_client>)` instead of `Agent(...)`. All other agent fields (instructions, tools, model, handoffs, hooks) MUST be forwarded to the `SandboxAgent` with the same shape they have on the standard `Agent`.
- **FR-096**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: When `agent_mode: sandbox`, the safety gate MUST require `openai.i_understand_this_is_unsafe: true`. Load without the opt-in fails with the canonical migration error (same shape as FR-083 for `CodeInterpreterTool` / `ComputerTool`).
- **FR-097**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: When `agent_mode: sandbox` AND `deployment.security_profile: default`, the sandbox client MUST be `UnixLocalSandboxClient` with the workspace under `/var/holodeck/work/sandbox/<session_id>` (using the same EmptyDir tmpfs mount as FR-085). Workspace MUST be destroyed on session close.
- **FR-098**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: When `agent_mode: sandbox` AND `deployment.security_profile: hardened`, the sandbox client MUST switch to a remote sandbox client selected by `openai.sandbox.remote_client: docker | modal` (default `docker`). The remote client's credentials MUST live alongside the other secrets in the Envoy sidecar; the agent container MUST NOT carry them.
- **FR-099**: **Deferred proposal**, outside 035 completion; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future requirement: System MUST reject configurations that declare both `agent_mode: sandbox` and `CodeInterpreterTool` (or `ComputerTool`) in `tools` — the sandbox already provides shell + code execution and the overlap is a configuration error, not a feature.

**Tracing:**

- **FR-100**: For `provider: openai`, permitted SDK spans MUST reach the OTel mirror when enabled and the provider exporter unless provider upload is disabled. Select processors before runs emit spans. Do not call `set_tracing_disabled(True)` to suppress only provider upload: it also starves the mirror. [D07](#d07).
- **FR-101**: For `provider: azure_openai`, SDK processor selection MUST exclude the OpenAI dashboard exporter while preserving enabled OTel emission. Suppression MUST hold with observability disabled, repeated initialization, and mixed-provider agents in either initialization order. Never use `set_tracing_disabled(True)` as the Azure upload-suppression mechanism. Process-global processor ownership is [D13](#d13). [D07](#d07).
- **FR-102**: `observability.disable_provider_tracing: bool` (default `false`) MUST suppress provider upload for either provider without disabling otherwise-enabled OTel emission. Every run MUST explicitly derive `trace_include_sensitive_data` from `observability.traces.capture_content` (default false), carry the agent workflow name and run context, and include a group/session ID for session runs. SDK environment defaults MUST NOT override capture-disabled behavior. [D07](#d07).

**Validation at startup:**

- **FR-110**: Validation MUST collect all applicable configuration and credential errors in one pass and surface them together. Preflight MUST be side-effect-free: no SDK trace/key global mutation or agent run.
- **FR-111**: Validation MUST run at `holodeck serve` and `holodeck deploy run` before accepting traffic or deploying. Echo sizing, capacity and derivation, default guardrails, hosted tools, and tracing destinations. Do not present the deferred hardened profile or sandbox configuration as supported. [D08](#d08).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Existing OpenAI/Azure agents route through the native backend by provider selection. Function, vectorstore, MCP, hierarchical-document, and the newly added skill category work; the five retained hosted tools are additionally available. No nonexistent `backend:` override is required. [D01](#d01), [D06](#d06).
- **SC-002**: The required regression suite passes and Claude/Ollama behavior and shared result contracts remain intact. Tests may change to reflect recorded specification decisions; do not preserve or invent tests for removed AF/ADK/SK agent backends. [D01](#d01).
- **SC-003**: `openai_agents` meets the shared spec-023 functional contract: single-turn, multi-turn session, streaming, tool calling, multimodal input, and error handling, with Claude regressions covered.
- **SC-004**: Configuration validation completes within one second on a stated reproducible fixture/environment. Missing credentials, conflicting permission lists, invalid hosted parameters, computer use, and missing code-interpreter opt-in produce actionable errors. Azure resource capability rejection occurs at runtime, not through a blanket configuration ban. [D03](#d03).
- **SC-005**: Operators who do not install the openai-agents extras experience no import-time failures when using other backends (lazy import gate).
- **SC-006**: `openai_agents` results pass all three evaluation metric types (standard NLP, G-Eval, RAG) — no result-shape regressions.
- **SC-007**: All five retained HoloDeck tool categories and all five retained hosted tools are covered by per-tool tests and real supported-provider acceptance. Computer use is explicitly rejected at configuration load and deferred under [H-012](../../exec-plans/tech-debt-tracker.md#h-012). [D06](#d06).
- **SC-008**: A locally constrained 1-CPU / 1-GiB container MUST sustain ten concurrent in-flight turns without OOM and reject the eleventh with 429 and `Retry-After`. An authorized ACA deployment using the supported 1-CPU / 2-GiB default MUST sustain its derived twenty turns without OOM and reject the twenty-first identically. Verify recovery and recorded memory under a representative fixture; idle sessions do not consume slots. Calibration failures keep acceptance open and require an explicit contract revision, not an unrecorded capacity reduction. [D11](#d11).
- **SC-009**: A synthetic credential-returning HoloDeck-built local tool proves model-visible replacement (`[REDACTED:anthropic-key]` for the Anthropic pattern), OTel attribute redaction, and capture-disabled trace payload exclusion. SDK-built MCP and server-side hosted outputs are not claimed as model-visible redaction coverage. [D05](#d05).
- **SC-010**: **Deferred proposal**, not a 035 completion gate; [D08](#d08), [H-010](../../exec-plans/tech-debt-tracker.md#h-010). Future acceptance: `security_profile: hardened` deploys two containers; the agent container env has zero credential-bearing vars; calls to non-allowlisted domains are rejected by Envoy.
- **SC-011**: An authorized real deployment proves OpenAI traces reach enabled OTel and the provider dashboard, while Azure reaches enabled OTel only. Provider-upload override suppresses upload independently of OTel. Mixed-provider/repeated-initialization and capture-disabled acceptance must also pass. [D07](#d07).
- **SC-012**: **Deferred proposal**, not a 035 completion gate; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future acceptance: An agent declared with `agent_mode: sandbox` and the safety opt-in successfully executes a shell command (e.g. `python -c 'print(1+1)'`) inside the sandbox workspace; the same command issued through a `standard`-mode agent has no shell tool to invoke. Verified by integration test on both `provider: openai` and `provider: azure_openai`.
- **SC-013**: **Deferred proposal**, not a 035 completion gate; [D08](#d08), [H-011](../../exec-plans/tech-debt-tracker.md#h-011). Future acceptance: Under `security_profile: hardened` with `agent_mode: sandbox`, the sandbox client is the configured remote variant (Docker by default); workspace state never lands on the agent container filesystem. Verified by inspecting the deployed Container App spec and the sandbox client's remote handle.

<a id="decision-register"></a>

## Decision register — 2026-09-06

These decisions reconcile the original 2026-05-24 proposal against the [full plan's approved decisions and SDK review](../../exec-plans/active/035-openai-agents-backend/plan-full.md#decisions), repository source, and completion scope.
They supersede old narrative, examples, duplicate ship phases, implementation-file forecasts, and quantitative cost claims that conflicted with the retained contract. They are not implementation evidence.

<a id="d01"></a>

### D01

**Provider routing replaces the nonexistent backend enumeration and opt-out.** `src/holodeck/lib/backends/selector.py` routes OpenAI/Azure to OpenAI Agents and Anthropic/Ollama to Claude. No `Backend` enum, AF backend, ADK backend, or explicit SK agent route exists. FR-001/007 and SC-001/002/003 now describe that contract; original US1 scenario 4 is superseded.

<a id="d02"></a>

### D02

**The namespace is `openai:`.** The approved full-plan Decision 1 and `models/openai_config.py` establish `OpenAIConfig`, sibling to `claude:`. Hooks/subagents are not read from `claude.*`. FR-060/063 and all backend-setting examples follow this name. `SkillTool` is required new work, correcting the original assumption that it already existed. Shared namespace cleanup remains H-017.

<a id="d03"></a>

### D03

**Azure uses the v1 Responses client.** `openai_agents_backend.py` constructs `AsyncOpenAI` on normalized `/openai/v1`; the older `AsyncAzureOpenAI` assumption and blanket Azure hosted-tool block are superseded. US1/US5 and SC-004 instead require runtime capability errors to remain actionable. Hosted SDK parameter shapes are verified against the locked SDK during T4, not inferred from old flat-kwargs examples.

<a id="d04"></a>

### D04

**Reasoning max maps to xhigh.** The configuration model and backend mapping support this value. FR-031 and US4's warning/clamp behavior are superseded; no clamping warning is required. Reasoning summaries are requested when effort is configured; absent summaries do not prove a failure.

<a id="d05"></a>

### D05

**Rejection is distinct from interactive approval and observation.** Full-plan Decisions 5/6 replace synchronous `needs_approval` assumptions with local tool guardrails, input tripwires, and hosted-MCP automatic rejection. FR-050/084/087 and SC-009 retain enforceable boundaries and fail closed on unsupported gates. Model-visible redaction does not cover SDK-built MCP or hosted execution. Ordinary MCP tool lifecycle observation remains available; MCP failure wrapping is unavailable. `modify` warns and remains inert. H-013/H-014/H-015 own the deferred expansions.

<a id="d06"></a>

### D06

**Five hosted tools, with computer use deferred.** Full-plan Decision 7 establishes that a live computer harness is required. US5, FR-083, and SC-007 retain web search, file search, code interpreter, image generation, and hosted MCP. H-012 owns computer use. Code interpreter still requires its safety gate and nested SDK configuration.

<a id="d07"></a>

### D07

**Select exporters without suppressing SDK span generation.** `set_tracing_disabled(True)` creates no-op traces and would starve OTel; FR-101 and US7's old sequence are superseded. FR-100–102 require provider-aware routing independently of OTel enablement and explicit capture-content policy. The unsafe process-global first-initialization behavior was replaced by the per-backend policy router in [D13](#d13).

<a id="d08"></a>

### D08

**P3 and sandbox remain explicit exclusions.** Full-plan Decisions 2/3 defer net-new shared Envoy infrastructure and security-sensitive sandbox lifecycle work. FR-090–093/SC-010 go to H-010; FR-094–099/SC-012/013 go to H-011. Their retained MUST wording is future proposal text, not a claim of supported YAML or a current completion gate. P1a/P1b/P2a/P2b remain required. Spec-034 references do not imply shared P3 infrastructure already exists.

<a id="d09"></a>

### D09

**Child scrubbing has a managed boundary.** FR-089 cannot contain arbitrary Python by promising a subprocess wrapper. It now requires per-child sanitization for all HoloDeck-managed paths, including MCP configured environments, without concurrent process-global mutation. H-021 owns stronger isolation for arbitrary child launches; operators must be told the boundary.

<a id="d10"></a>

### D10

**Keep `/ready` and connect it to required tool state.** The route exists in `serve/server.py` and ACA startup/readiness probes use it. The current lifecycle-only 200 does not establish FR-013. T7 must require successful backend prerequisites and completed or verified existing initialization for required vectorstore/hierarchical-document tools. Pending, failed, cancelled, and uninitialized jobs return 503; agents with no initializable tools use backend readiness. Shutdown also returns 503. `/health` remains liveness; no endpoint rename is required.

<a id="d11"></a>

### D11

**Preserve the memory formula and correct the unsupported ACA resource pair.** ACA Consumption requires 2 GiB with 1 CPU; the proposed 1-CPU / 1-GiB ACA default is invalid. Retain the current supported 1-CPU / 2-GiB deployment default, yielding `floor(2048 / 100) = 20`; retain the original ten-turn no-OOM target as a local 1-CPU / 1-GiB constrained-container regression. SC-008 requires both results. See [Azure Container Apps resource requirements](https://learn.microsoft.com/en-us/azure/container-apps/containers#vcpu-and-memory-allocation-requirements), reviewed 2026-09-06.

Explicit capacity overrides take precedence. Without a finite cgroup limit, use a labelled assumed 1024-MiB budget and derive ten by default. Do not subtract Claude's 400-MiB subprocess baseline or silently clamp a derived zero to one: a result below one fails startup with an actionable adjustment/override error. Serve/deploy must share the math and echo its inputs. Calibration remains an acceptance gate; changing the estimate or target requires an explicit decision and evidence.

<a id="d12"></a>

### D12

**Close migration acceptance without claiming final SK removal.** H-008 owns connectors, text splitting, and the SK dependency. LiteLLM dimension/error/span acceptance remains in T2/T10 and H-009. Tracked fixtures, real provider/collector/container evidence, multimodal behavior, evaluation overrides, and clean optional-extra installation remain mandatory completion gates. Prior live-provider deferral in the historical full plan is superseded by T10: missing access is an unresolved blocker, not passing acceptance.

<a id="d13"></a>

### D13

**HoloDeck owns the SDK's process-global trace-processor list through one router.** The SDK exposes a single processor list per process, so a first-initialization flag cannot express an OpenAI agent that uploads next to an Azure agent that must not. At `initialize()` each backend registers a per-instance `TracingPolicy` (`upload`: OpenAI without `disable_provider_tracing`; `mirror`: an OTel mirror when observability tracing is enabled) with `register_tracing_policy`; the first registration installs one HoloDeck router via `set_trace_processors`, replacing the SDK default exporter. Every run tags its trace with the backend's policy id in `RunConfig.trace_metadata` (`holodeck.tracing_policy`) and executes inside an `active_tracing_policy` scope. Trace events resolve by the tag, then the scope. Span identity is resolved once at span start (the scope of the run starting it, then its trace) and pinned to the span id until the span ends, so a run nested in a caller-owned or differently tagged outer `trace()` follows its own backend, and a span finished outside its scope, after its trace ended, or after the bounded trace map evicted its trace never falls back to upload. Policies are resolved from the registry per event, so `teardown()`'s `unregister_tracing_policy` drops further events of that backend immediately (fail closed). Registration is always performed, so Azure suppression holds with observability disabled, across repeated initialization, and for mixed providers in either order. Only traces and spans with no HoloDeck identity at all (non-HoloDeck SDK usage in the same process) keep SDK default behavior; the trace-level record of a caller-owned untagged outer trace also keeps that default. Implemented in `lib/backends/openai_agents_tracing.py` and covered by `test_openai_agents_tracing.py` and `test_openai_agents_backend.py`.

<a id="d14"></a>

### D14

**Fallback order is primary client retries, then one fallback attempt, with no Runner-level policy repeat.** The wrapper in `lib/backends/openai_agents_fallback.py` is the Runner's model, so the SDK's runner-managed retries (`ModelSettings.retry` with a policy) would re-run the whole primary-then-fallback pair per attempt. HoloDeck never sets `ModelSettings.retry`; the primary's retry budget is the OpenAI client's provider-managed retries (`max_retries`, default 2, on 429/5xx/connection errors with `retry-after`), which exhaust inside the primary call before the wrapper makes exactly one fallback attempt and then surfaces the fallback's error unchanged. One SDK compatibility path remains and is documented, not suppressed: on a fallback HTTP 400 `conversation_locked`, openai-agents 0.17.4 rewinds and re-runs the pair up to three more times (bound: four pairs) before raising; the only opt-out (`max_retries=0`) would also disable the client retries, so it is not taken. Streams fall back only before the first event, including non-text events such as `response.created`. Both attempts open their own `response` span under one trace. Runner-level tests over real `OpenAIResponsesModel` instances and an in-process HTTP transport are in `tests/unit/lib/backends/test_openai_agents_fallback_runner.py`; live provider evidence remains T10.

## Implementation and validation ownership

Use the [completion plan](../../exec-plans/active/035-openai-agents-backend/2026-09-06-complete-035.md) for task dependencies and checks; the [matrix](../../exec-plans/active/035-openai-agents-backend/acceptance-matrix.md) owns named evidence and deferral disposition.
The [full plan](../../exec-plans/active/035-openai-agents-backend/plan-full.md) provides detailed adapter designs; its historical claims yield to this reconciled contract.
Existing implementation lives in `src/holodeck/lib/backends/openai_agents_*.py`, `models/openai_config.py`, the shared selector/validators, serve, and deploy modules.
Subagents, skills, hosted-tool schema/factories, hooks, and guardrails remain implementation work; do not treat forecast filenames as delivered modules.
Use the pinned `openai-agents` optional extra and retain lazy imports for other backends. Models must not import SDK modules at import time.
Schema changes require schema regeneration/checks; Python changes require focused style/type/tests; documentation changes require harness/link validation.
Final acceptance includes the completion plan's required repository checks and authorized live validation. No cloud deployment is authorized merely by approving this contract.
