# Implementation Plan: Replace SK Embedding and Chat Services with LiteLLM

**Execution handoff:** Use the [completion execution plan](2026-09-06-complete-035.md) for current task order, acceptance gates, and progress.
This record retains the reconciled design and historical task evidence.

**Spec:** `docs/product-specs/035-openai-agents-backend/spec.md`
**Predecessor:** [SK decoupling](plan-sk-decouple.md).
**Status:** Core migration implemented; focused contracts for dimensions, errors, telemetry, and
documentation closed by 035 T2 on 2026-09-06 ([D15](../../../product-specs/035-openai-agents-backend/spec.md#d15)).
Live provider validation remains with 035 T10.
**Reconciled:** 2026-09-06 against `b585e80`; implementation landed in `3805e80` (#338); T2 follow-up on branch `feat/035-t2-litellm-acceptance`.
**Scope:** Replace SK embedding/contextual-chat inference, retaining vector-store abstractions and chunking.

## Evidence and status convention

Checked items indicate implementation or test coverage found in the tree. See the
[reconciliation report](reconciliation.md) for fresh test results and their limits; broad gates
below are not inferred from focused test passes. Unchecked items distinguish unmet draft
acceptance criteria from validation not rerun. An unmet draft criterion is not by itself a
confirmed product regression.
The previous status recorded live Azure validation of both inference seams in-session. That is
historical evidence, not a new test result or proof of OpenAI/Ollama end-to-end acceptance.
Task IDs below retain the original work breakdown while replacing superseded draft instructions.

## Resolved architecture decisions

1. `lib/litellm_support.py` provides `LiteLLMModelSpec` and `resolve_litellm_model`.
   The spec carries `model`, `api_key`, and `api_base`; it does **not** carry `api_version` or
   `dimensions`. Both inference paths share this mapping:

   | Provider | LiteLLM model | Connection |
   | --- | --- | --- |
   | OpenAI | Bare model name | API key; default provider endpoint |
   | Azure OpenAI | `openai/<deployment>` | Endpoint normalized to `/openai/v1`; no API-version argument |
   | Ollama | `ollama/<model>` | Configured endpoint when present |
   | Anthropic (chat only) | `anthropic/<model>` | API key |

   The draft Azure `azure/<deployment>` mapping is superseded.
2. `LiteLLMEmbeddingService.generate_embeddings` calls `litellm.aembedding` and returns ordered
   `list[list[float]]`, preserving the embedding-mixin interface.
3. `LLMContextGenerator` calls `litellm.acompletion`; retry, throttling, truncation, and batch logic remain.
4. **Hard cutover:** no `HOLODECK_RAG_BACKEND` flag or SK inference fallback was shipped. The draft
   one-release fallback and later flag-deletion phase are superseded. Rollback requires a code change.
5. `litellm>=1.80.0,<1.89.0` is a core dependency. SK remains for connectors/chunking.
6. LiteLLM telemetry uses the configured tracer-provider chain with `SPAN_ONLY` content capture
   or `NO_CONTENT`. Redaction covers new and legacy message attributes, LiteLLM's
   `llm.openai.stringified_raw_response`, `error.*` failure attributes, and span-event attributes. **Semconv (T2):** HoloDeck does not pin
   `OTEL_SEMCONV_STABILITY_OPT_IN`; the default LiteLLM 1.88 shape (`litellm_request`, operation on
   `llm.request.type`, provider on `gen_ai.system`) is the documented contract and the
   `gen_ai_latest_experimental` opt-in is documented for operators. Both shapes are tested at the
   exporter in `tests/unit/lib/observability/test_litellm_spans.py`.
7. Ollama live smoke coverage was left manual in the earlier decision; this audit does not claim it ran.
8. **Error boundary (T2):** `LiteLLMEmbeddingService.generate_embeddings` re-raises every provider or
   decoding failure as `EmbeddingServiceError` with the cause chained. Tools propagate it (no placeholder
   fallback when a service is injected); the initializer wraps ingest failures as `ToolInitializerError`.
9. **Dimensions (T2):** `create_embedding_service(agent, dimensions=...)` forwards an override; the
   initializer keeps one shared service per agent and builds a dedicated service per tool that sets
   `embedding_dimensions`, so different tools may request different sizes of the same model. Azure
   deployment aliases (`openai/<deployment>`) allow-list the parameter via `allowed_openai_params`
   because LiteLLM's client-side guard keys on the `text-embedding-3` model marker.

## Task reconciliation

### Task 1 — Dependency and shared resolver

- [x] Core LiteLLM requirement exists in `pyproject.toml`.
- [x] Resolver implements the provider mapping above and rejects unsupported embedding providers
  with `ToolInitializerError`; covered by `tests/unit/lib/test_litellm_support.py`.
- [x] Azure normalization and already-normalized endpoint cases have unit coverage.
- [ ] Re-establish dependency installation and type-check validation; `uv sync` was not rerun here.

**Checkpoint 0:** Resolver implemented. Focused tests passed in the parent audit.
Isolated installation and type-check acceptance remain open. The original “no behavior change yet” condition describes a past intermediate step.

### Task 2 — Embedding shim

- [x] Shim returns floats in response-index order; empty-input and mocked output tests exist.
- [x] Direct shim use forwards `dimensions` when provided and omits it when `None`; unit tests exist.
- [x] **Boundary implemented (T2, 2026-09-06):** the shim raises `EmbeddingServiceError` with the
  provider exception chained (Decision 8); `tests/unit/lib/test_litellm_support.py::TestEmbeddingServiceErrorBoundary`
  and the initializer/tool propagation tests cover it. The earlier silent placeholder fallback in both
  tools was removed.
- [x] Focused embedding tests passed in the parent audit; see the [verification record](reconciliation.md#verification-in-this-audit).

### Task 3 — Route embedding factory through LiteLLM

- [x] `create_embedding_service` unconditionally returns `LiteLLMEmbeddingService`.
- [x] The draft flag and SK fallback criteria are superseded by hard cutover (Decision 4).
- [x] Existing tools retain embedding-dimension mismatch checks; downstream collection APIs still
  consume precomputed vectors.
- [x] **Dimension wiring (T2, 2026-09-06):** per-tool `embedding_dimensions` reaches `litellm.aembedding`
  as `dimensions` through a dedicated service (Decision 9);
  `tests/unit/lib/test_tool_initializer.py::TestEmbeddingDimensionsForwarding` covers factory forwarding,
  distinct services for different overrides, the shared default, single-tool init, provider size
  mismatch, and provider failure with the cause preserved.
- [ ] Re-establish grounded ingest/query acceptance for OpenAI, Azure, and Ollama. Earlier Azure
  seam smoke evidence does not prove the full three-provider criterion.

**Checkpoint A:** Embedding cutover, error boundary, and dimension wiring present; three-provider
live validation remains open (035 T10). There is no fallback branch to test.

### Task 4 — Rewire contextual retrieval

- [x] Generator accepts `LiteLLMModelSpec` and calls `litellm.acompletion` for response text.
- [x] Committed `tests/unit/lib/test_llm_context_generator.py` covers retry/rate-limit behavior,
  graceful-empty failure, truncation, token counting, and batch/concurrency behavior.
- [x] Focused generator tests passed in the parent audit; see the [verification record](reconciliation.md#verification-in-this-audit).
- [ ] Re-establish type checks.

### Task 5 — Wire context resolution and remove old plumbing

- [x] `_resolve_context_generator` resolves a configured context model through LiteLLM.
- [x] Caller-provided generator has priority; Anthropic fallback still constructs
  `ClaudeSDKContextGenerator`. Resolution-priority tests exist in `test_tool_initializer.py`.
- [x] `_create_chat_service_from_config` and `chat_service=` references are absent from `src/`.
- [ ] Re-establish real hierarchical-document contextual ingest and Claude regression validation.

**Checkpoint B:** Contextual-inference replacement implemented; original live-ingest/full-suite gate not rerun.

### Task 6 — LiteLLM telemetry and redaction

- [x] `enable_litellm_telemetry` registers an idempotent OTel callback after tracer-provider setup.
- [x] `RedactingSpanProcessor` covers `gen_ai.input.messages`, `gen_ai.output.messages`,
  `gen_ai.system_instructions`, and legacy prompt/completion prefixes; regression tests exist
  in `tests/unit/lib/backends/test_otel_redaction.py`.
- [x] Callback configuration uses `SPAN_ONLY` when `traces.capture_content` is enabled and
  `NO_CONTENT` otherwise; configuration tests exist in observability instrumentation tests.
- [x] **Semconv resolved (T2, 2026-09-06):** no pin; the default shape is the contract and the opt-in is
  documented and tested (Decision 6).
- [x] **Emission proven at the exporter (T2, 2026-09-06):** `tests/unit/lib/observability/test_litellm_spans.py`
  (9 tests) drives real `litellm.aembedding` / `LLMContextGenerator` → `litellm.acompletion` with
  `mock_response` through `enable_litellm_telemetry` and `RedactingSpanProcessor`: span names, request
  type, token attributes, content absent and no raw span under `NO_CONTENT`, redacted prompt/completion/raw
  payload under `SPAN_ONLY`, the opt-in shape with and without capture, and a redacted failure span.
- [ ] Collector smoke against a live exporter when authorized (035 T10).

### Task 7 — Confirm the hard cutover

- [x] No SK embedding/chat imports or rollback flag remain in `tool_initializer.py`.
- [x] `llm_context_generator.py` has no SK import.
- [x] Remaining `semantic_kernel` references under `src/holodeck/lib/` are vector-store code,
  text chunking, observability setup/exports, logger suppression, and bridge filters.
- [ ] Verify installed SK import, original format/lint/type/security/full-suite gates, and real
  RAG/contextual-ingest acceptance. None is inferred from source searches.

**Checkpoint C:** SK removed from production RAG inference. Connectors/chunking and SK configuration
hooks remain. Original full-suite acceptance is not re-established by this reconciliation.

### Task 8 — Documentation

- [x] This plan documents the implemented mapping and retained SK dependency.
- [x] `docs/guides/vector-stores.md` describes the retained SK connector layer.
- [x] Chunker replacement is recorded as a follow-up below.
- [x] **Documentation (T2, 2026-09-06):** `docs/guides/tools.md` gained an "Embedding inference" section
  (provider mapping, error boundary, dimensions) and the `embedding_dimensions` field row;
  `docs/guides/observability.md` gained "RAG inference spans (LiteLLM)"; changelog updated.
- [x] `make harness-check` and `uv run mkdocs build --strict` run with the T2 documentation.

**Checkpoint Complete:** Not met. Core migration, focused contracts, and documentation are present;
the live three-provider ingest/query, contextual-ingest, and collector gates above remain with 035 T10. User review
recorded by an earlier session must not be inferred from the implemented status.

## Remaining work and full SK decommission

The remaining open items above (live provider, contextual-ingest, collector smoke, and full-suite
gates) belong to 035 T10/T11; the Task 2/3/6/8 contract items were closed by 035 T2.
The following work is separate from this inference swap:

- [ ] Replace SK vector-store record definitions, collection contracts, 11 provider mappings,
  and `PostgresSettings` in `lib/vector_store.py`, preserving search/upsert behavior.
- [ ] Replace `split_plaintext_paragraph` in `lib/text_chunker.py` with verified local splitting.
- [ ] Migrate the direct `OllamaTextEmbedding` use in `tests/integration/test_structured_vectorstore.py`
  and remaining SK fixtures/mocks. Production inference migration did not remove every test import.
- [ ] Reassess/remove SK telemetry exports/setup, suppression/filter defaults (including
  `models/observability.py`), and related tests/documentation after connector migration.
- [ ] Remove `semantic-kernel` requirements and lock entries only after runtime and test imports
  are gone; verify package installation and representative retrieval/provider regressions.

See the [SK inventory](plan-sk-decouple.md#remaining-sk-inventory-and-follow-ups) for file-level scope.
A dedicated final-decommission spec/plan is still needed; neither plan authorizes silently replacing
all connectors during this documentation reconciliation.
