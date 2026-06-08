# Implementation Plan: OpenAI Agents SDK Backend — MVP Slice

**Spec:** `specs/035-openai-agents-backend/spec.md`
**Scope of this plan:** A single vertical slice — *chat with one OpenAI/Azure-OpenAI agent that runs the SDK agent loop and calls custom Python (function) tools* — plus the routing flip and the surgical removal of the SK **agent-execution** path.
**Status:** Draft for review

## Overview

Deliver the smallest end-to-end path that proves the new backend: a `provider: openai` or
`provider: azure_openai` agent executes through the OpenAI Agents SDK `Runner` loop, calls
user-defined Python function tools, and works under `holodeck chat` with real token-delta
streaming. Once that path is proven, flip default routing (`openai`/`azure_openai →
openai_agents`, `ollama → claude`) and carve out the SK agent-execution surface while keeping
the SK completion/embedding services the RAG core depends on.

Everything else in spec 035 (serve/deploy, hardening, MCP/vectorstore/hierarchical/skill/hosted
tools, subagents, hooks, cost/fallback/effort, tracing-mirror, sandbox) is **explicitly out of
this MVP** and tracked as follow-ups.

## Decisions (confirmed with user)

1. **Routing flip now.** `anthropic → claude`, `openai`/`azure_openai → openai_agents`,
   `ollama → claude`. The SK provider branch is removed from the selector.
2. **Both providers.** `openai` and `azure_openai`. Because **no OpenAI API key is available**,
   **Azure OpenAI (endpoint + api-key) is the dev/test target**; the `provider: openai` path is
   coded but validated later when a real key exists.
3. **Real token-delta streaming** for `holodeck chat` (not the SK-style fake single-chunk).
4. **Delete the SK agent path wholesale; keep the shared service factories.** Remove
   `SKBackend`/`SKSession` *and* `AgentFactory` entirely. The reusable embedding + completion
   service factories already live in `tool_initializer.py`
   (`create_embedding_service`, `_create_chat_service_from_config`) — **not** in `AgentFactory` —
   so they stay untouched and the RAG/vectorstore/contextualization path is unaffected.
   `AgentFactory`'s `_register_embedding_service`/`_llm_service`/`_create_kernel` are merely
   SK-kernel glue and die with the agent path. `semantic-kernel` stays in `pyproject.toml`.
5. **Auth reality.** A Codex/ChatGPT OAuth token is *not* a usable credential for the Agents
   SDK (different audience/endpoint than `api.openai.com`; rejected by `/v1/responses` and
   `/v1/chat/completions`). Azure is the supported dev path.

## Architecture decisions

- **Azure uses Chat Completions, not Responses.** For `provider: azure_openai` the backend
  builds `AsyncAzureOpenAI(api_key, api_version, azure_endpoint)` and wraps it as
  `OpenAIChatCompletionsModel(model=<deployment>, openai_client=...)`, passed as the agent's
  `model=`. It also calls `set_tracing_disabled(True)` (no dashboard upload; and the SDK won't
  try to upload traces with a key it doesn't have). For `provider: openai` the default Responses
  client + a model-name string is used.
- **Lazy import gate.** `import agents` (the `openai-agents` package) happens *inside* the
  backend module's functions/methods, never at a module top-level that other backends import —
  preserving SC-005 (other backends incur no import cost / failure).
- **Service factories already live outside `AgentFactory`.** `tool_initializer.create_embedding_service`
  and `tool_initializer._create_chat_service_from_config` are the provider-agnostic embedding/completion
  factories the RAG path uses (`vectorstore_tool`, `hierarchical_document_tool`, `LLMContextGenerator`).
  They are untouched by the carve; `AgentFactory` is deleted wholesale.
- **Reuse the existing function-tool loader.** `load_function_tool(cfg, base_dir)` already
  returns a plain callable used by both SK and Claude. The OpenAI adapter wraps that callable as
  an SDK tool via the low-level `agents.FunctionTool(name, description, params_json_schema,
  on_invoke_tool=...)`, feeding it the JSON schema we derive (mirrors the Claude adapter's
  `_derive_input_schema`). This avoids depending on the `@function_tool` signature-introspection
  path for dynamically loaded callables.
- **Multi-turn via `SQLiteSession`.** `OpenAIAgentsSession` holds a `SQLiteSession(session_id)`
  and calls `Runner.run(agent, message, session=...)`; the SDK persists turn history. Idle
  sessions are SQLite rows, not held processes (spec's P4-is-native point).
- **MVP tool surface = function tools only.** Any non-function tool type on an `openai_agents`
  agent raises a clear `ConfigError`: "`<type>` tools are not yet supported on the openai_agents
  backend" — fail fast rather than silently drop tools.
- **Unit tests mock the SDK** (no network, no key). A single Azure-gated integration smoke
  exercises the live path.

## Task List

### Phase 0 — Foundation & wiring

#### Task 1: Add `openai-agents` optional extra + lazy-import gate
**Description:** Add `openai-agents` (pinned) as an optional dependency and confirm the import is
lazy so non-OpenAI backends are unaffected.
**Acceptance criteria:**
- [ ] `openai-agents==<pinned>` added to `pyproject.toml` (optional extra, e.g. `openai_agents`).
- [ ] `import agents` only occurs inside functions/methods of the new backend module.
- [ ] Importing `holodeck.lib.backends.selector` with the extra *uninstalled* does not raise.
**Verification:** `uv sync` resolves; `python -c "import holodeck.lib.backends.selector"` in an
env without the extra succeeds; `make lint` clean.
**Dependencies:** None
**Files:** `pyproject.toml`
**Scope:** XS

#### Task 2: Provider→client/model builder + credential pre-flight
**Description:** A helper that, given an `Agent`, returns the SDK `model=` argument and validates
credentials: `openai` → model-name string (default client) requiring `OPENAI_API_KEY`;
`azure_openai` → `OpenAIChatCompletionsModel` over `AsyncAzureOpenAI` requiring
`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`, and `set_tracing_disabled(True)`.
**Acceptance criteria:**
- [ ] Missing required env var → `BackendInitError` naming the exact var (US1 scenario 3).
- [ ] Azure path builds `AsyncAzureOpenAI` + `OpenAIChatCompletionsModel(model=<deployment>)` and
      disables SDK tracing.
- [ ] OpenAI path returns the model name for the default Responses client.
**Verification:** `tests/unit/lib/backends/test_openai_agents_backend.py` cases for both
providers + missing-cred (mocked SDK). `make type-check`.
**Dependencies:** Task 1
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

### Phase 1 — Backend + agent loop (non-streaming)

#### Task 3: `OpenAIAgentsBackend` + `OpenAIAgentsSession` (non-streaming)
**Description:** Implement the `AgentBackend`/`AgentSession` protocols wrapping SDK `Agent` +
`Runner` + `SQLiteSession`. `invoke_once`/`send` call `Runner.run(...)`; `create_session` returns
a session holding a `SQLiteSession`. Model the shape on `sk_backend.py`.
**Acceptance criteria:**
- [ ] All four `AgentBackend` and four `AgentSession` methods implemented (`prepare`/`close`
      no-ops where appropriate).
- [ ] `invoke_once` and `send` return an `ExecutionResult` with `response` populated from
      `result.final_output`.
- [ ] Runtime failures are wrapped as `BackendSessionError` with `error_reason` set.
**Verification:** Protocol-conformance + single-turn + multi-turn unit tests (mocked `Runner`).
**Dependencies:** Task 2
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

#### Task 4: Full `ExecutionResult` mapping
**Description:** Populate every `ExecutionResult` field from the SDK run result.
**Acceptance criteria:**
- [ ] `tool_calls`/`tool_results` extracted from `result.new_items`
      (tool-call vs tool-output items); `token_usage` from the run usage
      (`input_tokens`/`output_tokens`/`total_tokens`); `num_turns` from raw-response count;
      `thinking=""`.
- [ ] `is_error`/`error_reason` set on failure.
- [ ] Shapes match what `chat/executor.py` and the test runner already consume.
**Verification:** Unit test asserts a tool-calling run yields non-empty `tool_calls`/`tool_results`
and non-zero `token_usage` (mocked run items).
**Dependencies:** Task 3
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** S

### Phase 2 — Function tool calling

#### Task 5: Function-tool adapter for the SDK
**Description:** New adapter module translating a HoloDeck `FunctionTool` into an SDK
`agents.FunctionTool`, reusing `load_function_tool` + a JSON-schema deriver. Non-function tool
types raise a clear `ConfigError`.
**Acceptance criteria:**
- [ ] A YAML `type: function` tool loads its callable and is invoked by the agent loop with args
      from the model; the return value reaches the model (verified via a mocked run that drives
      the tool).
- [ ] `parameters` schema from YAML (when present) is used; otherwise derived from the callable
      signature.
- [ ] A `vectorstore`/`mcp`/`hierarchical_document`/`skill`/`prompt` tool on an `openai_agents`
      agent raises `ConfigError` naming the unsupported type.
**Verification:** `tests/unit/lib/backends/test_openai_agents_tool_adapters.py` — function happy
path + each unsupported-type error.
**Dependencies:** Task 3
**Files:** `src/holodeck/lib/backends/openai_agents_tool_adapters.py`,
`src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

### Checkpoint A — Non-streaming chat works end-to-end
- [ ] Against Azure (real creds), `holodeck chat` on an `provider: azure_openai` agent with one
      Python function tool completes a turn that calls the tool.
- [ ] All unit tests pass; `make format lint type-check` clean.
- [ ] **Review with user before proceeding to the SK carve.**

### Phase 3 — Real streaming

#### Task 6: `send_streaming` via `Runner.run_streamed`
**Description:** Implement true token-delta streaming: iterate `result.stream_events()`, yield
text from `raw_response_event` → `ResponseTextDeltaEvent.delta`; capture final usage/result after
the stream completes so a subsequent `ExecutionResult` (or the chat executor's accounting) is
correct.
**Acceptance criteria:**
- [ ] `send_streaming` yields multiple text chunks for a multi-token response (mocked event
      stream).
- [ ] Final token usage is captured after stream end.
- [ ] `chat/executor.py` consumes the stream with no protocol change (Open Question 5 in spec).
**Verification:** Unit test feeds a fake event stream and asserts chunk sequence + final usage;
manual Azure `holodeck chat` shows incremental output.
**Dependencies:** Task 4
**Files:** `src/holodeck/lib/backends/openai_agents_backend.py`
**Scope:** M

### Phase 4 — Routing flip + SK agent-factory carve

#### Task 7: Selector routing flip
**Description:** Update `BackendSelector`: `anthropic → claude`, `openai`/`azure_openai →
openai_agents`, `ollama → claude`. Remove the SK provider branch.
**Acceptance criteria:**
- [ ] `provider: openai`/`azure_openai` selects `OpenAIAgentsBackend`.
- [ ] `provider: ollama` selects `ClaudeBackend`.
- [ ] No selector path returns `SKBackend`.
**Verification:** `tests/unit/lib/backends/` selector tests updated; `make test-unit` green.
**Dependencies:** Tasks 3–5 (new backend must work before it becomes default)
**Files:** `src/holodeck/lib/backends/selector.py`
**Scope:** S

#### Task 8: Delete the SK agent path (SKBackend + AgentFactory)
**Description:** Delete `sk_backend.py` (`SKBackend`/`SKSession`) and `agent_factory.py`
(`AgentFactory`/`AgentThreadRun`) wholesale. The shared embedding/completion factories in
`tool_initializer.py` (`create_embedding_service`, `_create_chat_service_from_config`) stay
untouched — they are the source of truth the RAG path already uses, and nothing outside the SK
agent path depends on `AgentFactory` for services. Rewire `test_runner/executor.py` pairing
(`_detect_backend_kind`, which assumes "sk" vs "claude") so non-Claude runs go through the
selected backend (`OpenAIAgentsBackend`) rather than `AgentFactory`. Remove obsolete SK
agent-execution tests; keep RAG/service tests.
**Acceptance criteria:**
- [ ] `sk_backend.py` and `agent_factory.py` removed; no import references remain.
- [ ] RAG path (vectorstore/hierarchical-doc init + embeddings + contextualization) still works —
      its factories in `tool_initializer.py` are unchanged.
- [ ] `holodeck test` runs an `openai`/`azure_openai` agent via `OpenAIAgentsBackend`.
- [ ] Obsolete SK agent tests removed; full suite green.
**Verification:** `make test` (full, parallel) green; a vectorstore agent still initializes its
store + answers a grounded query; `make type-check` clean.
**Dependencies:** Task 7
**Files (delete):** `src/holodeck/lib/backends/sk_backend.py`,
`src/holodeck/lib/test_runner/agent_factory.py`
**Files (edit):** `src/holodeck/lib/test_runner/executor.py`,
`src/holodeck/lib/test_runner/__init__.py`, `src/holodeck/lib/errors.py` (drop
`AgentFactoryError` if now unused), obsolete tests under `tests/` (`test_sk_backend.py`,
`test_function_tool_sk.py`, `test_agent_factory*.py`, `test_agent_factory_integration.py`)
**Scope:** L  *(largest task — `test_runner` rewiring is the real work; verify with the full suite)*

### Checkpoint B — Routing flipped, SK carved
- [ ] Full test suite green.
- [ ] `chat` + `test` work for `openai`/`azure_openai` via the new backend; `ollama` via Claude;
      `anthropic` unchanged.
- [ ] RAG/embedding path unaffected.

### Phase 5 — Tests & docs

#### Task 9: Unit-test hardening
**Description:** Round out mocked unit coverage: protocol conformance, ExecutionResult
population, Azure-client construction, missing-cred errors, function-tool invocation, streaming
deltas, unsupported-tool errors.
**Acceptance criteria:**
- [ ] New tests cover both providers and the streaming path (all mocked — no network/key).
- [ ] Coverage for the new modules is comparable to the Claude/SK backend tests.
**Verification:** `make test-unit -n auto`; `make test-coverage` shows the new modules covered.
**Dependencies:** Tasks 6, 8
**Files:** `tests/unit/lib/backends/test_openai_agents_backend.py`,
`tests/unit/lib/backends/test_openai_agents_tool_adapters.py`
**Scope:** M

#### Task 10: Azure end-to-end smoke (creds-gated)
**Description:** One integration test (skipped without Azure creds) that runs a single `holodeck
chat` turn calling a Python tool against a live Azure deployment.
**Acceptance criteria:**
- [ ] Test skips cleanly when `AZURE_OPENAI_*` env is absent.
- [ ] With creds, a tool-calling turn returns a grounded response and streams.
**Verification:** Run locally with Azure creds; confirm pass + skip-without-creds.
**Dependencies:** Task 6
**Files:** `tests/integration/test_openai_agents_chat_e2e.py`
**Scope:** S

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Azure (Chat Completions) streaming events differ from Responses-API shape | Med | Task 6 normalizes on `raw_response_event`/`ResponseTextDeltaEvent`; verify both providers against the SDK adapter during impl |
| Deleting `AgentFactory` breaks the RAG/embedding path | Med | Factories already live in `tool_initializer.py` (verified) and stay untouched; full-suite gate + vectorstore-init verification (Task 8 / Checkpoint B) |
| `test_runner/executor.py` rewiring off `AgentFactory` regresses `holodeck test` | High | `_detect_backend_kind` + executor pairing is the real work in Task 8; covered by `test_executor*`; full suite is the gate |
| No OpenAI key → `provider: openai` path unverified | Med | Code it to spec, validate via mocks; mark live `openai` validation as a follow-up when a key exists |
| Test runner pairing logic assumes "sk" vs "claude" | Med | Task 8 rewires `_detect_backend_kind`; covered by `test_executor*` |
| Codex OAuth token assumed usable | Low | Decided unusable; Azure is the path |

## Out of scope (tracked follow-ups)

Serve/deploy + Dockerfile (US2) · hardening P1a–P3 + sandbox (US6/US8) ·
MCP/vectorstore/hierarchical/skill/hosted tools on this backend (US3/US5) · subagents/handoffs
(US3) · YAML hooks (US3) · cost/fallback/effort/disallowed (US4) · OTel tracing-mirror (US7) ·
live `provider: openai` validation · full `semantic-kernel` dependency removal.

## Open questions

- **Pin version** for `openai-agents` — confirm the latest stable at implementation time and
  that it exposes `Agent`, `Runner`, `SQLiteSession`, `FunctionTool`, `OpenAIChatCompletionsModel`,
  `set_tracing_disabled`.
- **Azure `api_version`** default — pick a current GA version; expose override via existing model
  config if one already exists.
- **`thinking` field** — left `""` for MVP; revisit if a reasoning model surfaces reasoning
  items through the SDK.
