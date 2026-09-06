# Tasks: US5 - Cross-Provider Backend Selection

**User Story**: A platform user wants to run an agent on a specific backend runtime while using a model from a different provider (e.g., Google ADK with OpenAI model, Agent Framework with Anthropic model).

**Dependencies**: US1 (ADK backend), US2 (AF backend) must be functional before US5 implementation.

**Spec References**: FR-030 (compatibility validation), Research R1, R5, R8

---

## Phase 2: Foundational — Compatibility Matrix Definition

- [ ] US5-01 P1 [US5] EXTRACT inline backend/provider compatibility logic from US1 T007's `BackendSelector` code into a named constant `BACKEND_PROVIDER_COMPATIBILITY` in `src/holodeck/lib/backends/selector.py` — a `dict[BackendEnum, set[ProviderEnum]]` encoding the compatibility matrix: `semantic_kernel: {openai, azure_openai, ollama}`, `claude: {anthropic}`, `google_adk: {google, openai, azure_openai, anthropic, ollama}`, `agent_framework: {openai, azure_openai, anthropic, ollama}`
- [ ] US5-02 P1 [US5] EXTRACT inline default-routing logic from US1 T007's `BackendSelector` code into a named constant `PROVIDER_DEFAULT_BACKEND` in `src/holodeck/lib/backends/selector.py` — a `dict[ProviderEnum, BackendEnum]` encoding the default routing table: `openai->agent_framework`, `azure_openai->agent_framework`, `anthropic->claude`, `ollama->claude`, `google->google_adk`

## Phase 3: US5 — Compatibility Validation and Cross-Provider Routing

### Compatibility Validation (FR-030)

- [ ] US5-03 P1 [US5] EXTEND the existing `@model_validator(mode='after')` created by US1 T010 in `Agent` (`src/holodeck/models/agent.py`) to also validate `backend`/`provider` compatibility when `backend` is explicitly set — raise `ValueError` with message listing the compatible providers for the requested backend (e.g., "Backend 'claude' only supports providers: anthropic. Got: openai")
- [ ] US5-04 P1 [US5] REFACTOR the existing `BackendSelector.select()` (already modified by US1 T007/US2) in `src/holodeck/lib/backends/selector.py` to resolve the effective backend using the extracted constants: if `agent.backend` is set, use it directly (compatibility already validated by model); if `None`, look up `PROVIDER_DEFAULT_BACKEND[agent.model.provider]`
- [ ] US5-05 P1 [US5] REFACTOR the existing `BackendSelector.select()` to route by resolved `BackendEnum` value instead of `ProviderEnum` — use a dispatch dict or match statement mapping each `BackendEnum` to its backend class instantiation logic, replacing the inline routing already present from US1/US2

### Cross-Provider Model Name Construction (ADK)

- [ ] US5-06 P1 [US5] In `src/holodeck/lib/backends/adk_backend.py`, implement LiteLLM model name construction in the initialization path: if `agent.model.provider` is `google`, pass `agent.model.name` directly (e.g., `"gemini-2.5-flash"`); otherwise construct `"{provider}/{name}"` (e.g., `"openai/gpt-4o"`) per ADK's LiteLLM integration (research R1)
- [ ] US5-07 P2 [US5] Handle `azure_openai` provider prefix mapping in `adk_backend.py` — Azure models use `"azure/{deployment_name}"` format for LiteLLM, and may require `api_base` and `api_version` environment variables

### Cross-Provider Client Selection (AF)

- [ ] US5-08 P1 [US5] In `src/holodeck/lib/backends/af_backend.py`, implement provider-to-client dispatch: `openai->OpenAIChatClient`, `azure_openai->AzureOpenAIChatClient`, `anthropic->AnthropicClient`, `ollama->OllamaChatClient` (research R5). Ensure the client is selected based on `agent.model.provider` and receives appropriate auth/endpoint config

**Note**: FR-031 (SK deprecation warning) was removed per design decision. Default routing breaks cleanly to AF/Claude. Users add `backend: semantic_kernel` explicitly if needed.

### Error Messages

- [ ] US5-11 P1 [US5] Ensure the `BackendInitError` raised for unsupported backend/provider combos in `selector.py` includes: the requested backend name, the requested provider name, and the full list of compatible providers for that backend (e.g., "Backend 'agent_framework' does not support provider 'google'. Compatible providers: openai, azure_openai, anthropic, ollama")
- [ ] US5-12 P2 [US5] Add a fallback error in `BackendSelector.select()` for unknown `BackendEnum` values (defensive programming — should not occur if enum is validated, but protects against future enum additions without dispatch updates)

## Phase 3: US5 — Tests

- [ ] US5-13 P1 [US5] Add unit tests in `tests/unit/lib/backends/test_selector.py` for compatibility validation: test that `backend: google_adk` + `provider: openai` resolves to ADK backend (acceptance scenario 1)
- [ ] US5-14 P1 [US5] Add unit test: `backend: agent_framework` + `provider: anthropic` resolves to AF backend with AnthropicClient (acceptance scenario 2)
- [ ] US5-15 P1 [US5] Add unit test: `backend: claude` + `provider: openai` raises clear error listing compatible providers (acceptance scenario 3)
- [ ] US5-16 P1 [US5] Add unit tests for all valid backend/provider combinations from the compatibility matrix — parameterize across all 13 valid combos (SK:3 + Claude:1 + ADK:5 + AF:4) to verify each resolves without error
- [ ] US5-17 P1 [US5] Add unit tests for all invalid backend/provider combinations — parameterize across invalid combos (e.g., `semantic_kernel+anthropic`, `claude+openai`, `agent_framework+google`) to verify each raises `BackendInitError` with descriptive message
- [ ] US5-18 P1 [US5] Add unit tests in `tests/unit/models/` for the `Agent` model validator: verify that constructing an `Agent` with incompatible `backend`/`provider` raises `ValidationError`
- [ ] US5-19 P1 [US5] Add unit test for default backend auto-detection: verify each provider resolves to its expected default backend when `backend` is omitted
- [ ] US5-20 P2 [US5] Add unit test for ADK LiteLLM model name construction: `provider: google` + `name: gemini-2.5-flash` -> `"gemini-2.5-flash"`, `provider: openai` + `name: gpt-4o` -> `"openai/gpt-4o"`, `provider: azure_openai` + `name: my-deployment` -> `"azure/my-deployment"`
- [ ] US5-21 P2 [US5] Add unit test for AF client selection: verify each supported provider maps to the correct client class

## Phase 4: Polish

- [ ] US5-23 P2 [US5] Run `make format && make lint && make type-check` — fix any issues introduced by US5 changes
- [ ] US5-24 P2 [US5] Run full test suite `make test` to verify no regressions from selector refactoring
- [ ] US5-25 P3 [US5] Review error messages for consistency: all backend/provider validation errors should follow the same format pattern across model validator and selector

---

## Summary

| Phase | Task Count | P1 | P2 | P3 |
|-------|-----------|----|----|-----|
| Phase 2: Foundational | 2 | 2 | 0 | 0 |
| Phase 3: Implementation | 7 | 5 | 2 | 0 |
| Phase 3: Tests | 9 | 7 | 2 | 0 |
| Phase 4: Polish | 3 | 0 | 2 | 1 |
| **Total** | **21** | **14** | **6** | **1** |
