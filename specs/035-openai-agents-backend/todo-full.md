# TODO — OpenAI Agents SDK Backend, Full Parity (post-MVP)

Tracks `plan-full.md` (revised 2026-06-10 after two adversarial SDK review cycles). Deferred to
follow-up specs: **P3 hardened/Envoy**, **US8 sandbox**, **computer-use**, **HITL tool approval**,
**MCP-tool guardrail coverage**. Scope per phase is in the plan. Check off as each task's
acceptance criteria + verification pass.

## Phase A — `openai:` config foundation
- [x] A1 — `OpenAIConfig` model + `openai:` block on `Agent` (+ schema regen)  · M
- [x] A2 — Backend consumes `OpenAIConfig`; **side-effect-free** collect-all-errors `validate_openai_agents`  · S
- [x] A3 — `RunConfig` plumbing: `workflow_name` (all runs) / `group_id` (session runs) / `trace_metadata`; `trace_include_sensitive_data` ← `capture_content`  · S
- [x] **Checkpoint A** — block validates, backend reads sizing/turns, RunConfig wired, validation side-effect-free, suite green, schema valid

## Phase B — Native RAG tool adapters
- [x] B1 — vectorstore + hierarchical_document → SDK `FunctionTool` (reuse `.search()`/embedder); `type: prompt` → skip-with-warning  · M
- [x] **Checkpoint B** — grounded query answers; tool-init endpoints smoke green *(unit/mocked; creds-gated live check deferred to K2)*

## Phase C — MCP transports (spec 027)
- [x] C1 — stdio/sse/http → `agents.mcp.*`; websocket skip; `create_static_tool_filter` (note: SDK-built FunctionTools — no guardrail/wrapper attachment)  · M

## Phase D — Subagents + skills (handoffs)
- [ ] D1 — `openai.agents` → SDK sub-Agents + `handoffs` + `RECOMMENDED_PROMPT_PREFIX` + model-literal validation  · M
- [ ] D2 — **net-new `SkillTool` model** (+ ToolUnion + schema regen) → handoff target (inline + SKILL.md)  · M
- [ ] D3 — handoff `ToolEvent`s (`subagent_message`/`parent_link`) for AG-UI  · S
- [ ] **Checkpoint D** — handoff scenario + skill route; AG-UI shows subagent events

## Phase E — YAML hooks (spec 028)
- [ ] E1 — `openai.hooks` model + log/notify/script observation; failure path = adapter wrapper catches + returns error string (no SDK default for directly built tools); unreachable-matcher warnings; chain ordering  · M
- [ ] E2 — `reject`: function tools → `tool_input_guardrails`/`reject_content`; input matchers → tripwire abort w/ message; MCP-server tools → load fail; HostedMCPTool → `require_approval` + `on_approval_request` (after G1); `modify` inert + warning  · M
- [ ] **Checkpoint E** — observation works; failure fires + error string; function reject continues run; input reject aborts w/ message; MCP reject fails load; modify warns; ordering correct (hosted paths verified at Checkpoint G)

## Phase F — Spec-026 config mappings (US4) + parity gaps
- [ ] F1 — `effort` → `ModelSettings(reasoning=…)`; `max` → `"xhigh"` (documented deviation)  · S
- [ ] F2 — `disallowed_tools` config-time filter; allow∩disallow → load fail  · S
- [ ] F3 — `max_budget_usd` → cost-accountant `RunHooks` + price table + `BackendBudgetExceededError`  · M
- [ ] F4 — `fallback_model` → wrapping `Model`; ordering vs `ModelSettings.retry` defined + tested  · M
- [ ] F5 — structured output: `response_format` → custom `JSONSchemaOutputSchema(AgentOutputSchemaBase)`; `thinking` from `ReasoningItem`s + `Reasoning(summary="auto")` (FR-004)  · M
- [ ] **Checkpoint F** — each field behaves per FR; structured/thinking parity; suite green

## Phase G — Hosted tools (US5)
- [ ] G1 — `HostedTool` model + 5 factories (nested `tool_config` built from YAML); `ComputerTool` → clear load error; allowed on Azure (runtime-gated)  · M
- [ ] G2 — safety gate: CodeInterpreter requires `i_understand_this_is_unsafe` (P1b; FR-084 as reinterpreted)  · S
- [ ] **Checkpoint G** — 5 hosted tools declarable; ComputerTool rejected cleanly; gate enforced; Azure load works; E2 hosted-reject paths verified

## Phase H — Tracing (US7)
- [ ] H1 — OTel-mirror `TracingProcessor`; **remove MVP `set_tracing_disabled(True)`**; `add_trace_processor` (openai) / `set_trace_processors` (azure/override) at backend `initialize()` before any span; `observability.disable_provider_tracing`  · M

## Phase I — Serve/deploy parity + P1a/P2a
- [ ] I1 — serve active-turn cap + 429/Retry-After + readiness + credential preflight + config echo  · M
- [ ] I2 — Dockerfile pure-Python branch (Node only on stdio MCP); corpus chmod/tmpfs verify  · S
- [ ] I3 — ACA default 1 CPU/1 GiB for openai/azure; provider-aware session-cap echo  · S
- [ ] **Checkpoint I** — serve + deploy build work; pure-Python image; 429 cap; sizing echo correct

## Phase J — P2b credential redaction + subprocess scrub
- [ ] J1 — default credential-redaction `tool_output_guardrail` on **HoloDeck-built** tools (allow clean / `reject_content` redacted; + `disable_default_hooks` opt-out; hosted + MCP gap documented)  · M
- [ ] J2 — subprocess env scrub (+ opt-out); verify OTel span redaction covers openai spans (incl. MCP/hosted span attrs)  · S
- [ ] **Checkpoint J** — SC-009 (HoloDeck-built-tool scope) passes: model + span redacted; uploaded trace excludes sensitive payloads by default (A3)

## Phase K — Sample, integration smokes, docs
- [ ] K1 — `sample/financial-assistant/openai` **created fresh** (no in-tree sample to copy; `/sample` is gitignored — manual creds-gated verification only)  · S
- [ ] K2 — creds-gated integration smokes (tool-init, hosted, handoff, structured output, redaction)  · M
- [ ] K3 — docs + per-backend hooks/guardrails-semantics matrix (coverage table incl. MCP gap; input-reject abort vs tool-reject continue; `max→xhigh`; FR-084 note; trace-sensitivity coupling) + deferred-surface notes  · S
- [ ] **Checkpoint K** — `make format lint type-check security` + `make test` green; SCs met; user review

---

### Deferred (separate specs — do NOT start here)
- [ ] ~~P3 hardened/Envoy profile (FR-090…093)~~ → own cross-backend spec
- [ ] ~~US8 sandbox mode (FR-094…099)~~ → own spec + threat model (surface EXISTS in 0.17.4, incl. `UnixLocalSandboxClient` — deferral is scoping, not absence)
- [ ] ~~Computer-use (`ComputerTool` + harness)~~ → own spec (needs a live `Computer`/`AsyncComputer` impl)
- [ ] ~~HITL tool approval (`needs_approval` + `RunState` resume loop; MCP `require_approval`)~~ → own spec (distinct from auto-`reject`)
- [ ] ~~Guardrail coverage for MCP-server tools~~ → needs an SDK attachment point (upstream)
- [ ] ~~`modify` hook implementation~~ → v1 inert (output guardrails make modify-on-output feasible later)
- [ ] ~~`type: prompt` runtime support~~ → no backend has it; skip-with-warning preserves status quo
- [ ] ~~Cross-backend `claude.*`/`openai:` namespace unification~~ → schema-cleanup spec
