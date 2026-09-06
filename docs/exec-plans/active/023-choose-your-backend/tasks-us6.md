# Tasks: US6 — Unified Embedding Provider via LiteLLM

**User Story**: A platform user has vectorstore/hierarchical_document tools and wants embeddings to work consistently regardless of backend. LiteLLM replaces SK embedding classes as the unified embedding provider, supporting 20+ providers.

**Branch**: `023-choose-your-backend`
**Depends on**: BackendEnum from US1 setup (for embedding_provider validator); otherwise independent of US1/US2.

---

## Phase 1: Setup — Add LiteLLM Core Dependency

- [ ] T6-01 P1 [US6] Add `litellm` as a core dependency in `pyproject.toml` (`dependencies` list, not optional)
  - Pin to a stable minor range (e.g., `litellm>=1.40.0,<2.0.0`)
  - Run `uv lock` to verify resolution with existing deps
  - Confirm no conflicts with existing `openai`, `anthropic`, `ollama` packages

## Phase 2: Foundational — EmbeddingService Protocol

- [ ] T6-02 P0 [US6] Create `src/holodeck/lib/embedding_protocol.py` with `EmbeddingService` protocol
  - Define `@runtime_checkable` Protocol class with:
    - `async def embed_batch(self, texts: list[str]) -> list[list[float]]`
    - `@property dimensions -> int`
    - `@property model_id -> str`
  - Include docstrings per project style (Google Python Style Guide)

- [ ] T6-03 P0 [US6] Implement `LiteLLMEmbeddingAdapter` in `src/holodeck/lib/embedding_protocol.py`
  - Fields: `_model_id: str`, `_api_key: str | None`, `_api_base: str | None`, `_dimensions: int`
  - `embed_batch()` calls `litellm.aembedding()` with the mapped model string
  - Implement provider mapping logic:
    - `openai` + `text-embedding-3-small` -> `"text-embedding-3-small"`
    - `azure_openai` + `my-deployment` -> `"azure/my-deployment"`
    - `ollama` + `nomic-embed-text` -> `"ollama/nomic-embed-text"`
    - `google` + `text-embedding-004` -> `"gemini/text-embedding-004"`
  - Add `@classmethod from_embedding_provider(cls, provider_config: LLMProvider) -> LiteLLMEmbeddingAdapter` factory method
  - Handle `api_key`, `api_base` (from `endpoint` field), and `api_version` mapping to LiteLLM kwargs

- [ ] T6-04 P1 [US6] Add provider-to-LiteLLM model string mapping helper in `embedding_protocol.py`
  - Function: `def _map_provider_model(provider: ProviderEnum, name: str) -> str`
  - Mapping rules per data-model.md section 5
  - Raise clear error for unsupported providers (e.g., `anthropic` has no embedding models)

## Phase 3: Integration — Refactor tool_initializer.py

- [ ] T6-05 P0 [US6] Replace `create_embedding_service()` in `src/holodeck/lib/tool_initializer.py` to use `LiteLLMEmbeddingAdapter`
  - Remove SK embedding imports: `OpenAITextEmbedding`, `AzureTextEmbedding`, `OllamaTextEmbedding`
  - Remove the `from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding, OpenAITextEmbedding` block (line ~100-103)
  - Remove the `from semantic_kernel.connectors.ai.ollama import OllamaTextEmbedding` lazy import (line ~130)
  - Replace entire `create_embedding_service()` body with:
    - Call `_resolve_embedding_provider()` and `_resolve_embedding_model_config()` (keep these helpers)
    - Instantiate `LiteLLMEmbeddingAdapter.from_embedding_provider(model_config)` or equivalent
  - Return type annotation: change from `Any` to `EmbeddingService`
  - Update module docstring to remove "uses SK TextEmbedding classes" reference

- [ ] T6-06 P1 [US6] Update `_resolve_embedding_provider()` in `tool_initializer.py` to handle `google` provider
  - Currently only checks for `ProviderEnum.ANTHROPIC`; must also check for `GOOGLE` (new provider from US1)
  - All non-SK providers (anthropic, google) require `embedding_provider` to be set
  - Error message should list all providers that need `embedding_provider`

- [ ] T6-07 P1 [US6] Update type annotations in `tool_initializer.py`
  - Change `embedding_service: Any` parameters in `_initialize_vectorstore_tools()` and `initialize_hierarchical_doc_tools()` to `embedding_service: EmbeddingService`
  - Add import of `EmbeddingService` from `embedding_protocol`
  - Update docstrings referencing "SK TextEmbedding service" to "EmbeddingService"

## Phase 4: Validation — embedding_provider Model Validator

- [ ] T6-08 P0 [US6] Add `@model_validator(mode='after')` to `Agent` model in `src/holodeck/models/agent.py`
  - Validator enforces: when resolved backend is `claude`, `google_adk`, or `agent_framework` AND agent has vectorstore or hierarchical_document tools -> `embedding_provider` MUST be set
  - Backend resolution: use `backend` field if set, otherwise use default routing from `model.provider`
  - Import `BackendEnum` from `models.llm` (depends on T1-XX from US1 that adds BackendEnum)
  - Import tool types for isinstance checks: `VectorstoreTool`, `HierarchicalDocumentToolConfig`
  - Error message: "Backend '{backend}' does not provide native embeddings. Configure 'embedding_provider' in agent.yaml for vectorstore/hierarchical_document tools."
  - This is a NEW validator, not a modification of an existing one

- [ ] T6-09 P1 [US6] Handle the `semantic_kernel` backend case in embedding_provider validator
  - SK backend has its own embedding classes (functional when `backend: semantic_kernel` is explicitly set)
  - When resolved backend is `semantic_kernel`, embedding_provider is NOT required (SK handles it internally)
  - Note: `provider: openai` now defaults to `agent_framework` backend; only `backend: semantic_kernel` triggers the SK path
  - Document this distinction in the validator docstring

## Phase 5: Tests

- [ ] T6-10 P0 [US6] Create `tests/unit/lib/test_embedding_protocol.py`
  - Test `EmbeddingService` protocol compliance: verify `LiteLLMEmbeddingAdapter` satisfies `isinstance()` check
  - Test provider mapping function:
    - `openai` + `text-embedding-3-small` -> `"text-embedding-3-small"`
    - `azure_openai` + `my-deployment` -> `"azure/my-deployment"`
    - `ollama` + `nomic-embed-text` -> `"ollama/nomic-embed-text"`
    - `google` + `text-embedding-004` -> `"gemini/text-embedding-004"`
  - Test `from_embedding_provider()` factory with mock LLMProvider configs for each supported provider
  - Test `embed_batch()` with mocked `litellm.aembedding` response
  - Test error on unsupported provider (e.g., anthropic as embedding provider)
  - Test `dimensions` and `model_id` properties
  - Use `@pytest.mark.unit` and `@pytest.mark.asyncio`

- [ ] T6-11 P0 [US6] Add/update tests in `tests/unit/lib/test_tool_initializer.py` for LiteLLM integration
  - Test `create_embedding_service()` returns `LiteLLMEmbeddingAdapter` (not SK types)
  - Test with `provider: openai` + vectorstore tools -> LiteLLM adapter created
  - Test with `provider: anthropic` + `embedding_provider` -> LiteLLM adapter created
  - Test with `provider: google` + `embedding_provider` -> LiteLLM adapter created (acceptance scenario 2)
  - Test with `provider: ollama` + `embedding_provider: { provider: ollama, endpoint: http://localhost:11434 }` -> endpoint mapped (acceptance scenario 4)
  - Verify NO SK embedding imports are used (mock check or import inspection)

- [ ] T6-12 P0 [US6] Add tests for embedding_provider validator in `tests/unit/models/test_agent.py`
  - Test: `provider: anthropic` + vectorstore tools + NO embedding_provider -> `ValidationError` (acceptance scenario 5)
  - Test: `provider: google` + vectorstore tools + NO embedding_provider -> `ValidationError` (acceptance scenario 5)
  - Test: `provider: openai` + backend: `agent_framework` + vectorstore tools + NO embedding_provider -> `ValidationError`
  - Test: `provider: openai` + backend: `semantic_kernel` + vectorstore tools + NO embedding_provider -> PASSES (SK handles it)
  - Test: `provider: openai` + vectorstore tools + embedding_provider set -> PASSES (acceptance scenario 3)
  - Test: agent with NO vectorstore/hierarchical_document tools -> validator is no-op regardless of backend

## Phase 6: Polish

- [ ] T6-13 P2 [US6] Update `src/holodeck/lib/embedding_protocol.py` exports in `__init__.py` files
  - Add `EmbeddingService` and `LiteLLMEmbeddingAdapter` to `src/holodeck/lib/__init__.py` if one exists
  - Ensure public API is importable as `from holodeck.lib.embedding_protocol import EmbeddingService`

- [ ] T6-14 P2 [US6] Run full code quality pipeline and fix issues
  - `make format` (Black + Ruff)
  - `make lint` (Ruff + Bandit)
  - `make type-check` (MyPy — verify EmbeddingService protocol type-checks correctly)
  - `make test-unit -n auto` (all unit tests pass, including new ones)

- [ ] T6-15 P2 [US6] Verify embedding behavior under new default routing: `provider: openai` now defaults to `agent_framework` backend
  - The YAML schema for `embedding_provider` is NOT modified
  - `provider: openai` with vectorstore tools now requires `embedding_provider` (since default backend is `agent_framework`, not SK)
  - `provider: openai` + `backend: semantic_kernel` + vectorstore tools continues to work without `embedding_provider` (SK handles it internally)
  - Existing tests that exercise vectorstore tool initialization must be updated to either set `embedding_provider` or explicitly set `backend: semantic_kernel`

---

## Dependency Graph

```
T6-01 (litellm dep)
  └──> T6-02 (protocol) + T6-03 (adapter) + T6-04 (mapping)
         └──> T6-05 (tool_initializer refactor)
         │      └──> T6-06 (google provider handling)
         │      └──> T6-07 (type annotations)
         └──> T6-08 (validator) ──> requires BackendEnum from US1
         │      └──> T6-09 (SK backend case)
         └──> T6-10 (protocol tests)
         └──> T6-11 (tool_init tests) ──> after T6-05
         └──> T6-12 (validator tests) ──> after T6-08
                └──> T6-13 + T6-14 + T6-15 (polish)
```

## Acceptance Criteria Traceability

| Scenario | Covered By |
|----------|-----------|
| 1. `provider: openai` + vectorstore + `embedding_provider: openai` -> LiteLLM | T6-05, T6-11 |
| 2. `provider: google` + vectorstore + embedding_provider -> LiteLLM (no SK dep) | T6-06, T6-11 |
| 3. `provider: openai` + vectorstore + `embedding_provider` -> embeddings via LiteLLM, backend auto-detected to `agent_framework` | T6-05, T6-11, T6-15 |
| 4. `embedding_provider: { provider: ollama, endpoint: ... }` -> mapped to LiteLLM | T6-03, T6-04, T6-10, T6-11 |
| 5. `provider: anthropic/google` + vectorstore + NO embedding_provider -> error | T6-08, T6-12 |
