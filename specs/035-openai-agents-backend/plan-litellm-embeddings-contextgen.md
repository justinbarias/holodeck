# Implementation Plan: Replace SK Embedding + Chat Services with LiteLLM (keep SK vector-store abstractions)

**Spec context:** `specs/035-openai-agents-backend/spec.md` (RAG layer) — this is the follow-up
`plan-sk-decouple.md` tracked as out-of-scope: *"Full removal of SK from the
embedding/context-gen layer."*
**Scope:** Replace Semantic Kernel's `*TextEmbedding` (embedding) and `*ChatCompletion` +
`LLMContextGenerator` (contextual-retrieval chat) with **LiteLLM**, while **keeping every SK
vector-store abstraction** (`@vectorstoremodel`, `VectorStoreField`, the `data.vector` connectors).
**Status:** Draft for review

---

## Overview

The SK embedding/chat services are only *loosely* coupled into the RAG path — they sit behind two
clean seams:

- **Embeddings:** produced by one factory (`create_embedding_service`), injected via
  `EmbeddingServiceMixin.set_embedding_service()`, and consumed through a **single method**,
  `await service.generate_embeddings(list[str])`, at 3 call sites. The SK vector-store collections
  never see the embedding object — they store/search **pre-computed `list[float]`** only (no
  `VectorStoreField(embedding_generator=…)` anywhere).
- **Chat (contextual retrieval only):** produced by `_create_chat_service_from_config`, consumed
  **only** by `LLMContextGenerator` via `chat_service.get_chat_message_contents(...)`. The generator
  already conforms to the backend-agnostic `ContextGenerator` protocol (`base.py:260`) — the same
  protocol `ClaudeSDKContextGenerator` (non-SK) implements.

So this is a localized swap: replace what produces vectors and chat completions; leave the
connectors, record models, and the `ContextGenerator`/embedding-mixin seams intact.

## Coupling map (evidence)

| Surface | Where | SK symbol | Consumer seam |
|---|---|---|---|
| Embedding factory | `tool_initializer.py:101` `create_embedding_service` | `OpenAITextEmbedding` / `AzureTextEmbedding` / `OllamaTextEmbedding` | returns service object |
| Embedding consume | `vectorstore_tool.py:423`, `hierarchical_document_tool.py:832,990` | `.generate_embeddings()` | `EmbeddingServiceMixin._embedding_service` (`base_tool.py:50`) |
| Vectors into store | `vectorstore_tool.py:515` upsert / `:921` search | — | raw `list[float]` (no generator on `VectorStoreField`) |
| Chat factory | `tool_initializer.py:425` `_create_chat_service_from_config` | `OpenAIChatCompletion` / `AzureChatCompletion` / `AnthropicChatCompletion` / `OllamaChatCompletion` | returns service object |
| Chat consume | `llm_context_generator.py:215,222` | `get_chat_message_contents` + `ChatHistory` + `OpenAIChatPromptExecutionSettings` | `ContextGenerator` protocol (`base.py:269` `contextualize_batch`) |
| Context resolution | `tool_initializer.py:497` `_resolve_context_generator` (5-tier) | builds `LLMContextGenerator` | priorities 2/3 build from a chat service; 4 = Claude (keep) |

**Kept untouched:** all 11 connectors (`PostgresCollection`, `QdrantCollection`, `InMemoryCollection`,
`AzureAISearchCollection`, …), `@vectorstoremodel`, `VectorStoreField`, `DynamicDocumentRecord` /
`StructuredRecord`; `ClaudeSDKContextGenerator` (Anthropic, non-SK); the `semantic-kernel` pin
(connectors need it).

## Architecture decisions

1. **One LiteLLM provider-mapping resolver.** A new `holodeck/lib/litellm_support.py` exposes
   `resolve_litellm_model(model_config, kind)` → a small `LiteLLMModelSpec`
   (`model`, `api_key`, `api_base`, `api_version`, `dimensions`). Single place for HoloDeck
   `LLMProvider` → LiteLLM mapping: OpenAI `text-embedding-3-small`; Azure `azure/<deployment>` +
   `api_base`/`api_version`; Ollama `ollama/<model>` + `api_base`. Both embedding and chat go
   through it.
2. **Embeddings → thin shim that keeps the mixin contract.** `LiteLLMEmbeddingService` exposes the
   one method the tools call — `async generate_embeddings(list[str]) -> list[list[float]]` (LiteLLM
   `aembedding`, forwarding `dimensions`). `create_embedding_service` returns it. No call-site
   changes (the tools already normalize via `[list(emb) for emb in embeddings]`).
3. **Context gen → rewire `LLMContextGenerator` in place (don't fork).** The retry/concurrency/
   truncation/tiktoken logic is SK-independent and valuable. Only `__init__` (take a
   `LiteLLMModelSpec`, drop `chat_service`/`execution_settings`) and `_call_llm` (→
   `litellm.acompletion`) change. `_create_chat_service_from_config` is replaced by the resolver;
   `_resolve_context_generator` priorities 2/3 build the rewired generator from a spec. The
   `chat_service` param (internal-only plumbing) is removed.
4. **Single rollback flag for one release.** `HOLODECK_RAG_BACKEND = litellm | sk` (default
   `litellm`) gates BOTH swaps so a regression can fall back to SK without a revert. Removed in the
   cleanup phase once validated.
5. **Preserve telemetry.** SK currently emits GenAI spans for embedding/chat. Wire LiteLLM's OTel
   callback so those spans continue; `RedactingSpanProcessor` keeps scrubbing.
6. **Out of scope (this plan):** `text_chunker.split_plaintext_paragraph` (pure SK utility, no model
   call) and any connector changes. Both tracked as follow-ups.

## Dependency graph

```
T1 (litellm dep + resolver) ──┬── T2 (embedding shim) ── T3 (wire create_embedding_service + flag)
                              └── T4 (rewire LLMContextGenerator) ── T5 (wire resolver + drop chat_service)
T3, T5 ──► T6 (telemetry) ──► T7 (remove SK fallback + flag) ──► T8 (docs)
```

Phase 1 (embeddings) and Phase 2 (context gen) are independent after T1 — parallelizable. T7 is the
hard cutover and must come after both are validated.

---

## Task List

### Phase 0 — Dependency + shared resolver

#### Task 1: Add `litellm` + `LiteLLMModelSpec` resolver
**Description:** Add `litellm` to `pyproject.toml` (core dep — RAG needs it). Create
`src/holodeck/lib/litellm_support.py` with `LiteLLMModelSpec` and
`resolve_litellm_model(model_config, kind: Literal["embedding","chat"])` mapping a HoloDeck
`LLMProvider` to LiteLLM args for OpenAI / Azure / Ollama (chat additionally Anthropic). No callers
yet — pure addition.
**Acceptance criteria:**
- [ ] `uv sync` resolves with `litellm` pinned.
- [ ] `resolve_litellm_model` returns correct `model`/`api_base`/`api_version`/`api_key`/`dimensions`
      for each provider (Azure → `azure/<deployment>`, Ollama → `ollama/<model>`+host, OpenAI → bare).
- [ ] Unsupported provider raises `ToolInitializerError` with the existing message shape.
**Verification:** `tests/unit/lib/test_litellm_support.py` (per-provider mapping); `make type-check`.
**Dependencies:** None
**Files:** `pyproject.toml`, `src/holodeck/lib/litellm_support.py`,
`tests/unit/lib/test_litellm_support.py`
**Scope:** S

### Checkpoint 0 — Resolver
- [ ] `litellm` installs; resolver unit tests green; no behavior change yet.

---

### Phase 1 — Embeddings via LiteLLM

#### Task 2: `LiteLLMEmbeddingService` shim
**Description:** New shim exposing `async generate_embeddings(texts: list[str]) -> list[list[float]]`
backed by `litellm.aembedding`, forwarding `dimensions` from the resolver. Matches the mixin
contract so downstream `[list(emb) for emb in embeddings]` and dimension validation
(`vectorstore_tool.py:428`) are unaffected.
**Acceptance criteria:**
- [ ] `generate_embeddings(["a","b"])` returns two equal-length `list[float]` (mocked `aembedding`).
- [ ] `dimensions` is forwarded for OpenAI v3 models; absent for providers that reject it.
- [ ] Errors surface as `ToolInitializerError` (or the existing embedding-error type).
**Verification:** `tests/unit/lib/test_litellm_support.py` or a new `test_litellm_embedding.py`
(mock `litellm.aembedding`); assert dimension forwarding.
**Dependencies:** T1
**Files:** `src/holodeck/lib/litellm_support.py` (or `litellm_embedding.py`), tests
**Scope:** S

#### Task 3: Route `create_embedding_service` through the shim (flag-gated)
**Description:** Make `create_embedding_service(agent)` return `LiteLLMEmbeddingService` when
`HOLODECK_RAG_BACKEND != "sk"` (default litellm); keep the SK `*TextEmbedding` branch as the `sk`
fallback. No change to the 3 consuming call sites.
**Acceptance criteria:**
- [ ] Default path returns the LiteLLM shim; `HOLODECK_RAG_BACKEND=sk` returns the SK service.
- [ ] Dimension-mismatch validation still fires with a wrong `embedding_dimensions`.
- [ ] A vectorstore agent ingests + answers a grounded query on OpenAI/Azure/Ollama (creds-gated).
**Verification:** `tests/unit/lib/test_tool_initializer.py` (flag branch, mocked); creds-gated
RAG integration test unchanged and green; `make test-unit`.
**Dependencies:** T2
**Files:** `src/holodeck/lib/tool_initializer.py`, tests
**Scope:** M

### Checkpoint A — Embeddings on LiteLLM
- [ ] Grounded query works via LiteLLM embeddings; SK fallback works via flag; vector-store
      connectors + records untouched; tool-init endpoints green.

---

### Phase 2 — Contextual retrieval via LiteLLM

#### Task 4: Rewire `LLMContextGenerator` to LiteLLM
**Description:** Change `LLMContextGenerator.__init__` to take a `LiteLLMModelSpec` (drop the
SK `chat_service` + `execution_settings` params) and rewrite `_call_llm` to use
`litellm.acompletion` (system/user message → response text). Keep ALL existing logic: retry/backoff,
429 concurrency reduction, document truncation, tiktoken counting, graceful-empty degradation. Keep
protocol conformance (`contextualize_batch`).
**Acceptance criteria:**
- [ ] `generate_context(chunk, doc)` returns the model text (mocked `acompletion`).
- [ ] 429 path reduces concurrency and retries; final failure returns `""` (graceful).
- [ ] Document truncation + `contextualize_batch` ordering/concurrency unchanged.
**Verification:** `tests/unit/lib/test_llm_context_generator.py` updated to mock `acompletion`
(retry, truncation, batch order, graceful degradation); `make type-check`.
**Dependencies:** T1
**Files:** `src/holodeck/lib/llm_context_generator.py`,
`tests/unit/lib/test_llm_context_generator.py`
**Scope:** M

#### Task 5: Wire resolver into `_resolve_context_generator`; drop `chat_service` plumbing
**Description:** Replace `_create_chat_service_from_config` with a chat-kind call to
`resolve_litellm_model`. Update `_resolve_context_generator` priorities 2 & 3 to build the rewired
`LLMContextGenerator` from a spec. Remove the now-unused `chat_service` param from
`initialize_tools` / `initialize_hierarchical_doc_tools` (internal-only; external callers don't pass
it — confirmed). Leave priority 4 (`ClaudeSDKContextGenerator`, Anthropic) untouched.
**Acceptance criteria:**
- [ ] A hier-doc tool with `context_model` builds a LiteLLM-backed generator and contextualizes
      chunks (creds-gated).
- [ ] Anthropic agents still resolve `ClaudeSDKContextGenerator` (no regression).
- [ ] No remaining references to `_create_chat_service_from_config` or the `chat_service` param.
**Verification:** `grep -rn "_create_chat_service_from_config\|chat_service=" src` returns nothing;
`tests/unit/lib/test_tool_initializer.py` (resolution priorities); creds-gated hier-doc contextual
ingest test green.
**Dependencies:** T4
**Files:** `src/holodeck/lib/tool_initializer.py`, `tests/unit/lib/test_tool_initializer.py`
**Scope:** M

### Checkpoint B — Contextual retrieval on LiteLLM
- [ ] Contextual ingest works via LiteLLM; Claude context path unaffected; full suite green.

---

### Phase 3 — Telemetry + hard cutover

#### Task 6: LiteLLM OTel telemetry (+ close the redaction-prefix gap)
**Description:** Register LiteLLM's OTel callback so embedding + chat calls emit GenAI-semconv
`gen_ai.*` spans equivalent to the SK ones they replace, flowing through the existing
`RedactingSpanProcessor`. **LiteLLM emits message content under the *current* semconv names
`gen_ai.input.messages` / `gen_ai.output.messages` (NOT the older `gen_ai.prompt` /
`gen_ai.completion` our processor scrubs today) — so the redaction prefix list must be extended or
prompt/response credentials leak to the exporter.** Pin `OTEL_SEMCONV_STABILITY_OPT_IN` to a fixed
value so attribute names stay stable across LiteLLM upgrades (the GenAI semconv is still
Experimental and churning), and set the content-capture control
(`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` / `litellm.turn_off_message_logging`)
consistently with the existing `traces.capture_content` config.
**Acceptance criteria:**
- [ ] An embedding call and a context-gen call each emit a `gen_ai.*` span to the configured
      exporter (embeddings span op = `embeddings`, chat op = `chat`/`acompletion`).
- [ ] `RedactingSpanProcessor` scrubs credential-shaped values under `gen_ai.input.messages`,
      `gen_ai.output.messages`, and `gen_ai.system_instructions` (new prefixes) **and** the legacy
      `gen_ai.prompt` / `gen_ai.completion` (kept for back-compat). Regression test proves the new
      prefixes are covered.
- [ ] `gen_ai.usage.input_tokens` / `output_tokens` / `total_tokens` are present; the semconv
      stability opt-in is pinned so names don't drift on upgrade.
- [ ] Content capture honors `traces.capture_content` (off ⇒ no message content on spans).
**Verification:** `tests/unit/lib/observability/test_otel_redaction.py` adds cases for
`gen_ai.input.messages` / `gen_ai.output.messages`; span-emission test with a mock exporter; manual
OTel-collector check (creds-gated).
**Dependencies:** T3, T5
**Files:** `src/holodeck/lib/litellm_support.py` (callback registration + semconv pin),
`src/holodeck/lib/backends/otel_redaction.py` (extend redaction prefixes),
`src/holodeck/lib/observability/*`, `tests/unit/lib/backends/test_otel_redaction.py`
**Scope:** M

#### Task 7: Remove the SK embedding/chat path + flag
**Description:** Delete the SK `*TextEmbedding` and `*ChatCompletion` branches and the
`HOLODECK_RAG_BACKEND` flag; LiteLLM becomes the only path. Drop the now-dead SK imports from
`tool_initializer.py`. Re-evaluate the SK GenAI-telemetry hook for embeddings/chat (vector-store SK
telemetry stays). Confirm the only remaining `semantic_kernel` imports in the RAG inference path are
gone (connectors + `split_plaintext_paragraph` remain).
**Acceptance criteria:**
- [ ] `grep -rn "TextEmbedding\|ChatCompletion\|HOLODECK_RAG_BACKEND" src/holodeck/lib/tool_initializer.py`
      returns nothing.
- [ ] `python -c "import semantic_kernel"` still resolves (connectors kept).
- [ ] `grep -rn "semantic_kernel" src/holodeck/lib/` lists only vector_store, text_chunker,
      observability hooks, logging_config, otel_bridge — NOT tool_initializer or llm_context_generator.
**Verification:** greps above; `make format lint type-check security`; `make test` (parallel) green;
creds-gated RAG ingest+search + contextual ingest green.
**Dependencies:** T6
**Files:** `src/holodeck/lib/tool_initializer.py`, `src/holodeck/lib/llm_context_generator.py`,
`src/holodeck/lib/observability/*`, tests
**Scope:** M

### Checkpoint C — SK removed from RAG inference
- [ ] Full suite green; SK confined to connectors + chunker; LiteLLM is the only embedding/chat path.

---

### Phase 4 — Docs

#### Task 8: Documentation
**Description:** Update RAG/embedding docs to LiteLLM; document the provider→LiteLLM mapping and the
retained SK surface (connectors + `split_plaintext_paragraph`); note the `semantic-kernel` pin stays.
**Acceptance criteria:**
- [ ] Docs describe LiteLLM embeddings + contextual retrieval and the kept SK vector-store layer.
- [ ] `text_chunker` SK utility flagged as a remaining follow-up.
**Verification:** docs build / link check; grep for stale SK embedding/chat references in `docs/`.
**Dependencies:** T7
**Files:** `docs/guides/*`, `docs/api/*`, `AGENTS.md`
**Scope:** S

### Checkpoint Complete
- [ ] All acceptance criteria met; vector-store abstractions untouched; suite green; user review.

---

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LiteLLM embedding output shape differs from SK (ndarray vs list) | Med | Shim returns `list[list[float]]`; downstream already does `list(emb)` + dimension validation (T2/T3 tests) |
| OpenAI v3 `dimensions` param mismatch breaks dimension validation | Med | Resolver forwards `dimensions`; T3 asserts the validation path on OpenAI/Azure |
| Azure deployment/endpoint/api_version mapping wrong in LiteLLM | High | Centralized in `resolve_litellm_model`; per-provider unit tests (T1) + creds-gated Azure RAG run (T3) |
| Removing `chat_service` param breaks an external caller | Low | Confirmed external callers (claude_backend) don't pass it; grep gate in T5 |
| Loss of SK GenAI telemetry for embeddings/chat | Med | T6 wires LiteLLM OTel callback before the SK path is removed in T7 |
| Credential leak: LiteLLM emits content under `gen_ai.input.messages`/`output.messages`, which the current `RedactingSpanProcessor` does NOT scrub (it only knows `gen_ai.prompt`/`completion`) | High | T6 extends the redaction prefix list + adds a regression test asserting the new prefixes are scrubbed; gate content capture on `traces.capture_content` |
| GenAI semconv is Experimental and churning (v1.38.0 deprecated `gen_ai.prompt`/`completion`); attribute names drift on LiteLLM upgrade | Med | Pin `OTEL_SEMCONV_STABILITY_OPT_IN` to a fixed value (T6); keep legacy prefixes in the redactor for back-compat |
| Ollama embeddings/chat via LiteLLM differ from SK behavior | Med | `ollama/` prefix + `api_base`; creds-gated local Ollama smoke; flag fallback until validated |
| Hard cutover (T7) regresses a provider not covered by creds-gated tests | Med | Keep the `HOLODECK_RAG_BACKEND=sk` flag through Checkpoints A/B; only remove in T7 after both validate |

## Out of scope (follow-ups)

- **`text_chunker.split_plaintext_paragraph`** → replace SK utility with a local splitter (XS,
  separate change; no model calls, low value/risk).
- **Dropping the `semantic-kernel` pin** → only possible after the connectors are also replaced
  (large, separate spec). The pin stays.
- **Replacing the SK `data.vector` connectors** → explicitly kept; not this plan.

## Open questions

1. **Rollback flag vs hard cutover.** Plan assumes a one-release `HOLODECK_RAG_BACKEND` flag
   (Decision 4). If you'd rather cut over hard, drop T3/T7's flag handling and fold the SK removal
   into T3/T5 (smaller, riskier).
2. **`litellm` as core vs optional extra.** Plan adds it as a core dep (RAG always needs it). If RAG
   should stay optional, gate it as an extra like `openai-agents`.
3. **Ollama coverage in CI.** Creds-gated Azure/OpenAI smokes are straightforward; confirm whether a
   local Ollama smoke is run in CI or left manual.
