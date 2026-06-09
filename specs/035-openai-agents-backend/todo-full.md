# TODO — OpenAI Agents SDK Backend, Full Parity (post-MVP)

Tracks `plan-full.md`. Deferred to follow-up specs: **P3 hardened/Envoy**, **US8 sandbox**.
Scope per phase is in the plan. Check off as each task's acceptance criteria + verification pass.

## Phase A — `openai:` config foundation
- [ ] A1 — `OpenAIConfig` model + `openai:` block on `Agent` (+ schema regen)  · M
- [ ] A2 — Backend consumes `OpenAIConfig`; collect-all-errors `validate_openai_agents`  · S
- [ ] **Checkpoint A** — block validates, backend reads sizing/turns, suite green, schema valid

## Phase B — Native RAG tool adapters
- [ ] B1 — vectorstore + hierarchical_document → SDK `FunctionTool` (reuse `.search()`/embedder)  · M
- [ ] **Checkpoint B** — grounded query answers; tool-init endpoints smoke green

## Phase C — MCP transports (spec 027)
- [ ] C1 — stdio/sse/http → `agents.mcp.*`; websocket skip; `create_static_tool_filter`  · M

## Phase D — Subagents + skills (handoffs)
- [ ] D1 — `openai.agents` → SDK sub-Agents + `handoffs` + `RECOMMENDED_PROMPT_PREFIX` + model-literal validation  · M
- [ ] D2 — skill tool → handoff target (inline + SKILL.md)  · S
- [ ] D3 — handoff `ToolEvent`s (`subagent_message`/`parent_link`) for AG-UI  · S
- [ ] **Checkpoint D** — handoff scenario + skill route; AG-UI shows subagent events

## Phase E — YAML hooks (spec 028)
- [ ] E1 — `openai.hooks` model + log/notify/script observation mapping + chain ordering  · M
- [ ] E2 — `reject` → `needs_approval`/`input_guardrail`; `modify` inert + warning  · M
- [ ] **Checkpoint E** — observation works; reject gates; modify warns; ordering correct

## Phase F — Spec-026 config mappings (US4)
- [ ] F1 — `effort` → `ModelSettings(reasoning=…)`; `max` clamps to `high` + warning  · S
- [ ] F2 — `disallowed_tools` config-time filter; allow∩disallow → load fail  · S
- [ ] F3 — `max_budget_usd` → cost-accountant `RunHooks` + price table + `BackendBudgetExceededError`  · M
- [ ] F4 — `fallback_model` → wrapping `Model` provider on retryable errors  · M
- [ ] **Checkpoint F** — each field behaves per FR; suite green

## Phase G — Hosted tools (US5)
- [ ] G1 — `HostedTool` model + 6 factories; allowed on Azure (runtime-gated)  · M
- [ ] G2 — safety gate: CodeInterpreter/Computer require `i_understand_this_is_unsafe` (P1b)  · S
- [ ] **Checkpoint G** — 6 hosted tools declarable; gate enforced; Azure load works

## Phase H — Tracing (US7)
- [ ] H1 — OTel-mirror `TracingProcessor`; provider-aware `set_tracing_disabled`; `observability.disable_provider_tracing`  · M

## Phase I — Serve/deploy parity + P1a/P2a
- [ ] I1 — serve active-turn cap + 429/Retry-After + readiness + credential preflight + config echo  · M
- [ ] I2 — Dockerfile pure-Python branch (Node only on stdio MCP); corpus chmod/tmpfs verify  · S
- [ ] I3 — ACA default 1 CPU/1 GiB for openai/azure; provider-aware session-cap echo  · S
- [ ] **Checkpoint I** — serve + deploy build work; pure-Python image; 429 cap; sizing echo correct

## Phase J — P2b credential redaction + subprocess scrub
- [ ] J1 — default function-tool credential-redaction decorator (+ `disable_default_hooks` opt-out)  · M
- [ ] J2 — subprocess env scrub (+ opt-out); verify OTel span redaction covers openai spans  · S
- [ ] **Checkpoint J** — SC-009 prompt-injection test passes (model + span redacted)

## Phase K — Sample, integration smokes, docs
- [ ] K1 — `sample/financial-assistant/openai`  · S
- [ ] K2 — creds-gated integration smokes (tool-init, hosted, handoff, redaction)  · M
- [ ] K3 — docs + per-backend hooks-semantics matrix + deferred-surface notes  · S
- [ ] **Checkpoint K** — `make format lint type-check security` + `make test` green; SCs met; user review

---

### Deferred (separate specs — do NOT start here)
- [ ] ~~P3 hardened/Envoy profile (FR-090…093)~~ → own cross-backend spec
- [ ] ~~US8 sandbox mode (FR-094…099)~~ → re-spec against real `agents.sandbox.*`
- [ ] ~~`modify` hook implementation~~ → v1 inert
- [ ] ~~Cross-backend `claude.*`/`openai:` namespace unification~~ → schema-cleanup spec
