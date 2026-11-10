# Remove Semantic Kernel and revamp the dependency stack (feature 042)

**Status:** In progress since 2026-11-10. The maintainer approved the plan and delegated the ask-first decisions; each is recorded under "Progress and decisions".
**Product contract:** [Spec 042](../../../product-specs/042-dependency-stack-revamp/spec.md).
**Evidence:** [dependency inventory](../../../product-specs/042-dependency-stack-revamp/dependency-inventory.md), [selection evidence](../../../design-docs/042-dependency-stack-revamp/selection-evidence.md).
**Debt closed by this plan:** [H-008](../../tech-debt-tracker.md#h-008).

## Objective and observable acceptance criteria

Remove Semantic Kernel (SK) from HoloDeck and replace its vector-store connectors with HoloDeck-owned adapters on each store's native client. Replace SK's text splitter with a HoloDeck splitter. Remove Microsoft AI evaluation and Hugging Face `evaluate`. Split the install into feature profiles.

The plan is complete when spec acceptance criteria AC-01 to AC-13 pass. The observable results are:

- `uv pip install holodeck-ai` in a clean environment installs no `semantic-kernel`, `azure-ai-evaluation`, `evaluate`, `mkdocs-material` or `twine`.
- `make verify` and `make verify-live` pass with SK not installed.
- Existing Qdrant and Chroma collections created before this work return the same results as before, with no re-embedding.
- `mcp` and `pinecone` are no longer capped by SK.

## Scope and constraints

In scope: everything in spec sections 2 to 5. Out of scope: the items in the spec's "Excluded" list, the `ag-ui-protocol` 1.0 migration, and unfinished 035 features.

Constraints that apply to every task:

- Follow the invariants in [AGENTS.md](../../../../AGENTS.md): Pydantic models, no `Any` or `getattr` at new boundaries, no blocking I/O on async paths, LLM calls and I/O stay outside Temporal workflows.
- Tests first. Each behavior change has a regression test that fails before the change.
- No task may change stored record layout, IDs, vector names or score conventions unless the task says so (FR-015, FR-017).
- **Ask-first items** stop for a maintainer checkpoint before code is written. They are marked **ASK** below.
- Live databases and provider calls need the authorization rules in [Reliability](../../../RELIABILITY.md). Missing credentials leave an acceptance item open; they do not pass it.

## Code map

| Area | Files | Note |
| --- | --- | --- |
| SK storage | `src/holodeck/lib/vector_store.py` (1,263 lines) | Record classes built on SK decorators, 11 connector paths, collection factory |
| Native reads | `src/holodeck/lib/collection_filter.py` (345) | Filtered reads for postgres, qdrant, weaviate, chromadb, in-memory |
| Hybrid search | `src/holodeck/lib/keyword_search.py` (1,142) | Qdrant full-text RRF, BM25/OpenSearch fallback, one `collection.hybrid_search` call |
| Consumers | `src/holodeck/tools/vectorstore_tool.py` (1,113), `tools/hierarchical_document_tool.py` (1,371), `tools/base_tool.py` | Call `collection_exists`, `ensure_collection_exists`, `upsert`, `get`, `delete`, `search` on SK collections |
| Splitter | `src/holodeck/lib/text_chunker.py` (90) | Wraps SK `split_plaintext_paragraph` |
| SK names | `models/observability.py`, `lib/logging_config.py`, `lib/backends/otel_bridge.py`, `lib/observability/{instrumentation,providers,__init__}.py` | `enable_semantic_kernel_telemetry`, namespace defaults |
| Evaluation | `src/holodeck/lib/evaluators/azure_ai.py` (451), `nlp_metrics.py` (431), `lib/test_runner/executor.py`, `models/evaluation.py` | Legacy metric names, `evaluate.load` for ROUGE and METEOR |
| Parsing | `src/holodeck/lib/file_processor.py` (856) | `markitdown[all]` |
| SK in tests | 10 files under `tests/` | Listed in the inventory |

## Design decisions

- **D1. One adapter interface.** A new package `src/holodeck/lib/retrieval/` owns the contracts: record, filter, query, result, capability and error types, plus a `VectorStoreAdapter` protocol with the FR-013 operations. Tools depend on the protocol only.
- **D2. One contract test suite.** A parametrized suite runs the same operation tests against every adapter. An adapter is done when the suite passes for it. Unsupported operations must raise the capability error before ingestion, never return empty results (FR-014).
- **D3. Temporary SK bridge.** T3 wraps the existing SK collections in one bridge adapter so that tools switch to the interface once (T4). Each native adapter then replaces the bridge for its provider. T16 deletes the bridge. FR-019 forbids it at cutover.
- **D4. Build order follows evidence of use:** in-memory, qdrant, chromadb, postgres, pinecone, then weaviate, azure-ai-search, azure-cosmos-nosql.
- **D5. Existing collections stay readable.** Adapters keep the current layout. Only the splitter change (T14) alters chunk boundaries, and it is gated by the pipeline fingerprint (T15).
- **D6. Independent work ships as independent PRs.** Evaluation, parsing and cleanup tasks do not depend on the retrieval stack and target `main` directly.

## Model and effort per task

From the user-level model guidance. Escalate in order: add a self-check, raise effort one level, then move up a model at lower effort.

| Kind of task | Model | Effort |
| --- | --- | --- |
| Contract design, ask-first decisions, integration of the tools onto the interface | Fable (main session), inline | session default |
| Adapters, migration tooling, evaluator removal, packaging split | Opus 5.5 subagent | `medium`, then `high` |
| Well-scoped changes against a fixed contract: cleanup, splitter, metric swap, docs | Sonnet 5.5 subagent | `medium`, then `high` |
| Reading logs, test output and wide searches | Haiku 5.5 subagent | default |
| Adversarial review, once per PR layer | Codex gpt-6-astra `high`, or Opus 5.5 `high` | maintainer's choice |

## Ordered work and verification

Sizes: S is 1 to 2 files, M is 3 to 5, L is 5 to 8. Every task also runs focused Black, Ruff, MyPy and pytest (`-n auto`) on the files it changes.

### Phase 0: Baseline and quick wins

#### T0. Freeze golden fixtures and baseline measurements

Record current behavior before any replacement, so later tasks have a fixed target.

- [x] Retrieval fixtures (tracked under `tests/fixtures/`): expected IDs and order for definitions, headings, section IDs, contextual text, no-result and punctuation queries, on in-memory, Qdrant and Chroma.
- [x] Plain-chunk, parser (page, sheet, range, headings) and ROUGE/METEOR/BLEU score fixtures.
- [x] Baseline table ([baseline measurements](baseline-measurements.md)): package count, installed bytes, `holodeck --help` and `holodeck test` import time, for base and all-extras installs.

**Verify:** new fixture tests pass on unchanged `main`; `make verify-live` passes.
**Depends on:** none. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-008, AC-05, AC-07, AC-09.

#### T1. Remove unused declarations and the second HTTP client

- [x] Remove direct declarations of `aiofiles`, `python-dateutil`, `anthropic`, `ollama`, `absl-py`, `ml-dtypes`, and the stale `sentry-sdk` constraint. Keep any that a clean-install test shows is needed.
- [x] Move `lib/file_processor.py` and `services/mcp_registry.py` from Requests to HTTPX with `follow_redirects=True`; drop the direct `requests` declaration.
- [x] `make verify` passes; Claude, Ollama-routed and DeepEval-on-Anthropic checks pass without the removed packages.

**Verify:** `uv lock --check`, `make verify`, `make verify-live LIVE_ARGS="--backend claude"`.
**Depends on:** none. Independent PR. **Size:** M. **Model:** Sonnet `medium`. **Spec:** FR-001, FR-007.

### Phase 1: Contract

#### T2. Define the retrieval contracts — ASK

Design the `lib/retrieval/` types and the `VectorStoreAdapter` protocol. Present the interface to the maintainer before any adapter is written.

- [x] Typed records for the three record families, a portable filter type, query and result types, a capability declaration per adapter, and an error hierarchy that includes a missing-client error with the install command.
- [x] Score convention documented: similarity in `[0, 1]`; adapters convert distances.
- [x] Contract test suite skeleton, parametrized by adapter, covering each FR-013 operation.

**Verify:** MyPy strict on the new package; the suite collects and runs against a stub.
**Depends on:** T0. **Size:** M. **Model:** Fable inline. **Spec:** FR-012, FR-013, FR-015.

#### T3. In-memory adapter and the temporary SK bridge

- [x] Native in-memory adapter passes the full contract suite.
- [x] One SK bridge adapter exposes existing SK collections through the protocol for every provider.
- [x] The bridge is marked temporary in code and listed for deletion in T16.

**Verify:** contract suite green for `in-memory` and for the bridge on `in-memory`.
**Depends on:** T2. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-013, FR-019.

#### T4. Move the tools onto the interface

Switch `vectorstore_tool.py`, `hierarchical_document_tool.py`, `base_tool.py`, `collection_filter.py` and `keyword_search.py` to the protocol. Behavior must not change.

- [x] No tool imports `semantic_kernel` or calls an SK collection method.
- [x] T0 golden fixtures unchanged on in-memory, Qdrant and Chroma.
- [x] Unsupported operations raise the capability error before ingestion.

**Verify:** existing tool tests, T0 fixtures, `make verify-live`.
**Depends on:** T3. **Size:** L. Split by tool if the diff exceeds one review sitting. **Model:** Fable inline. **Spec:** FR-012, FR-014, AC-06.

#### Checkpoint A

- [ ] `make verify` and `make verify-live` pass.
- [ ] Maintainer reviews the interface in use before adapter work fans out.

### Phase 2: Core adapters

T5 to T8 each replace the bridge for one provider. After T4 they touch disjoint files and can run in parallel.

#### T5. Qdrant adapter

Consolidate the existing native Qdrant code into the adapter. This is the highest-risk adapter, so it goes first.

- [x] UUIDv5 IDs, the `embedding` vector name, payload layout, payload and full-text indexes, and async-client ownership are unchanged.
- [x] Dense plus full-text RRF matches T0 fixtures, including the punctuation-only fallback and candidate pool `max(4 * top_k, 20)`.
- [x] A collection created on `main` and a newly created collection both pass.

**Verify:** contract suite, T0 Qdrant fixtures, `make verify-live` (Qdrant in Docker).
**Depends on:** T4. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-017, FR-018, AC-05.

#### T6. Chroma adapter

- [x] Contract suite passes for HTTP and embedded modes; embedded calls run off the event loop.
- [x] A collection created on `main` reads correctly.

**Verify:** contract suite, T0 Chroma fixtures, `make verify-live` (Chroma in Docker).
**Depends on:** T4. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-020, AC-04.

#### T7. Postgres adapter

- [x] psycopg 3 async with `pgvector`; existing SK table layout is read and written unchanged.
- [x] A pgvector container joins `make verify-live`.

**Verify:** contract suite against the container; a table created by the SK connector reads correctly.
**Depends on:** T4. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-015, AC-04.

#### T8. Pinecone adapter on client 10

- [x] Adapter uses `pinecone` 10 async; the `pinecone` extra moves to `>=10,<11`.
- [x] The Dependabot ignore for Pinecone majors is removed.
- [x] Enumeration strategy for source replacement is documented; if the service cannot support it, the capability error applies.

**Verify:** contract suite against Pinecone Local in Docker; a live-index run if the maintainer supplies credentials, otherwise recorded as open.
**Depends on:** T4. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-014, AC-04.

#### Checkpoint B

- [x] Five providers run without the SK bridge.
- [x] `make verify-live` passes with Qdrant, Chroma and Postgres.
- [x] Adversarial review of the stack so far.

### Phase 3: Remaining providers

#### T9. Provider scope decision — ASK

`faiss`, `sql-server` and `azure-cosmos-mongo` have no install extra, sample or live test today.

- [x] Maintainer decides per provider: build an adapter, or retire.
- [x] A retired provider fails config validation with a migration message before any agent call, and has a changelog entry and a migration note.

**Verify:** config tests for each decision.
**Depends on:** T2. **Size:** S. **Model:** Fable inline. **Spec:** FR-014.

#### T10. Weaviate adapter

- [x] Contract suite passes against Weaviate in Docker, with native hybrid.
- [x] New `weaviate` extra.

**Depends on:** T4. **Size:** M. **Model:** Opus `medium`.

#### T11. Azure AI Search adapter

- [x] Contract suite passes with mocked transport; native hybrid mapped.
- [x] New `azure-ai-search` extra. Live run needs Azure credentials; otherwise recorded as open.

**Depends on:** T4. **Size:** M. **Model:** Opus `medium`.

#### T12. Azure Cosmos NoSQL adapter

- [x] Contract suite passes against the Cosmos emulator in Docker.
- [x] New `azure-cosmos` extra.

**Depends on:** T4. **Size:** M. **Model:** Opus `medium`.

#### T13. Adapters for providers kept by T9

One task per kept provider, same acceptance shape as T10. Skipped for retired providers.

**Depends on:** T9. **Size:** M each. **Model:** Opus `medium`.

### Phase 4: Splitter and stored data

#### T14. HoloDeck plain text splitter

- [ ] A tiktoken-based splitter honours chunk size, overlap and separators; unsupported combinations are rejected.
- [ ] `lib/text_chunker.py` no longer imports SK.
- [ ] Boundary changes against T0 fixtures are recorded.

**Verify:** new splitter tests; `structured_chunker` tests unchanged.
**Depends on:** T0. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-022.

#### T15. Pipeline fingerprint and migration dry run — ASK

The fingerprint adds stored metadata, so the layout is an ask-first schema decision.

- [ ] A fingerprint covers embedding model and dimensions, parser and splitter policy, record layout, vector names and retrieval mode.
- [ ] Writes to an incompatible collection are refused with a clear message. Collections with no fingerprint and the current layout count as compatible.
- [ ] A read-only dry run reports collection identity, record counts, compatibility and re-embedding need. Migration resumes after interruption and can roll back.

**Verify:** migration tests for dry run, interruption, resume and rollback.
**Depends on:** T5, T14. **Size:** L. Split fingerprint from migration tooling. **Model:** Opus `medium`. **Spec:** FR-023, AC-08.

### Phase 5: SK cutover

#### T16. Remove Semantic Kernel — ASK for the config default

`models/observability.py` defaults a public field to `["semantic_kernel"]`. Changing it changes the published schema.

- [ ] Delete the SK bridge, `lib/vector_store.py` SK code, `enable_semantic_kernel_telemetry` and SK namespace defaults; update the 10 test files.
- [ ] Remove `semantic-kernel` from `pyproject.toml` and relock. Set `tool.uv.prerelease` to the strictest value that resolves.
- [ ] A test fails if `semantic_kernel` is importable in the base install or referenced in `src/`.

**Verify:** `make schema` and `make schema-check`, `make verify`, `make verify-live`, a clean-venv install with SK absent.
**Depends on:** T5 to T8, T10 to T15. **Size:** L. **Model:** Opus `medium`. **Spec:** FR-003, AC-02.

#### T17. Lift the MCP cap

- [ ] `mcp` allows 2.x; the Claude and OpenAI backends, MCP bridges and `holodeck mcp` pass. If the Claude backend still fails on MCP 2, keep the cap and record the cause.

**Verify:** `make verify`, `make verify-live`.
**Depends on:** T16. Independent PR. **Size:** S. **Model:** Sonnet `medium`.

#### Checkpoint C

- [ ] SK is absent from every profile; H-008 evidence recorded.
- [ ] Integrated adversarial review over the full retrieval stack diff.

### Phase 6: Evaluation (independent of Phases 1 to 5)

#### T18. Remove Azure AI evaluation and legacy metrics — ASK

The retired metric names are part of the public YAML schema.

- [ ] Delete `lib/evaluators/azure_ai.py`, its exports, executor branches, tests and the `azure-ai-evaluation` dependency.
- [ ] YAML with a retired metric name fails before any agent or model call, with the DeepEval replacement named.
- [ ] Stored results that used a retired metric still load in the dashboard.

**Verify:** `make schema` and `make schema-check`; evaluator and dashboard tests; `make verify`.
**Depends on:** T0. Independent PR. **Size:** L. Split removal from the migration error if needed. **Model:** Opus `medium`. **Spec:** FR-030, FR-031, AC-13.

#### T19. Call ROUGE and METEOR directly

- [ ] `nlp_metrics.py` calls `rouge_score` and NLTK METEOR directly; scores equal the T0 fixtures.
- [ ] Remove `evaluate` and the `datasets` and `pyarrow` constraints if nothing else needs them. No network call at metric load.

**Verify:** NLP metric tests with network disabled.
**Depends on:** T0. Independent PR. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-025, AC-09.

### Phase 7: Parsing and packaging

#### T20. MarkItDown selective extras

- [ ] `markitdown[pdf,docx,pptx,xlsx]` replaces `[all]`; `openpyxl`, `pdfminer.six` and `python-pptx` are declared directly.
- [ ] T0 parser fixtures unchanged for PDF, Word, PowerPoint, Excel, CSV, text, images and URLs.

**Verify:** file-processor tests, T0 parser fixtures.
**Depends on:** T0. Independent PR. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-021, AC-07.

#### T21. Install profiles — ASK

Extra names and what leaves the base install are public packaging changes.

- [ ] Maintainer approves the extra names and the compatibility aggregate.
- [ ] Docs and release tooling leave runtime dependencies. DeepEval, NLP metrics, the optimizer and OpenSearch become feature extras.
- [ ] A command that needs a missing extra prints the install command; unrelated commands still start.

**Verify:** lazy-import guard tests extended per profile; `make verify`.
**Depends on:** T16, T18, T19, T20. **Size:** L. Split tooling move from feature extras. **Model:** Opus `medium`. **Spec:** FR-004, FR-005.

#### T22. Installed-wheel profile tests in CI

- [ ] CI builds the wheel and installs each profile in a clean environment, with SK and Microsoft evaluation packages blocked.
- [ ] Each profile runs its smoke check.

**Verify:** the new CI job passes on the PR.
**Depends on:** T21. **Size:** M. **Model:** Opus `medium`. **Spec:** AC-01, AC-02, AC-13.

#### T23. Reproducible images

- [ ] Generated Dockerfiles install a pinned HoloDeck version with the reviewed extras; no unpinned prerelease resolution.
- [ ] Image digest and package manifest recorded.

**Verify:** deploy tests; a local image build.
**Depends on:** T21. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-029, AC-11.

### Phase 8: Close

#### T24. Measure, document and close

- [ ] Repeat the T0 measurements and record old against new.
- [ ] Update architecture, guides, provider capability docs, the inventory and the changelog; close H-008.
- [ ] Fill the acceptance table below with evidence; move this plan to `completed/`.

**Verify:** `make harness-check`, `uv run mkdocs build --strict`, full `make verify` and `make verify-live`.
**Depends on:** all. **Size:** M. **Model:** Sonnet `medium`. **Spec:** AC-12.

## Dependencies and PR boundaries

```
T0 ──┬─ T1 (independent PR)
     ├─ T2 ─ T3 ─ T4 ─┬─ T5 ─┐
     │                ├─ T6  ├─ T15 ─┐
     │                ├─ T7  │       │
     │                ├─ T8  │       ├─ T16 ─ T17
     │                ├─ T10, T11, T12, T13
     ├─ T14 ──────────┘
     ├─ T18, T19, T20 (independent PRs)
     └─ T21 ─ T22, T23 ─ T24
```

- **Retrieval stack (`gh stack`):** contracts and in-memory (T2, T3) → tools on the interface (T4) → one PR per adapter (T5 to T8, T10 to T13) → splitter and fingerprint (T14, T15) → SK removal (T16).
- **Independent PRs to `main`:** T1, T17, T18, T19, T20. Merge one at a time; the self-hosted CI runner contends when runs overlap.
- **Packaging stack:** T21 → T22 → T23, after the retrieval stack and Phase 6 merge.
- One adversarial review per PR layer, and one integrated pass over the retrieval stack before it merges. Docs-only PRs skip it.

## Acceptance evidence

| ID | Proved by | Evidence |
| --- | --- | --- |
| AC-01 | T22, T24 | open |
| AC-02 | T16, T22 | open |
| AC-03 | T4, T16, T17 | open |
| AC-04 | T5 to T13 | open |
| AC-05 | T0, T5 | open |
| AC-06 | T4, T15 | open |
| AC-07 | T0, T14, T20 | open |
| AC-08 | T15 | open |
| AC-09 | T19, T21 | open |
| AC-10 | T16 | open |
| AC-11 | T22, T23, T24 | open |
| AC-12 | T24 | open |
| AC-13 | T18, T22 | open |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| An adapter changes ranking or stored layout | High | T0 fixtures frozen first; old and new collections tested in T5 to T7 |
| The interface is wrong and every adapter needs rework | High | T2 is ask-first; in-memory and the SK bridge prove it before native adapters start |
| Splitter change mixes old and new chunks | High | Fingerprint (T15) refuses incompatible writes |
| Long-tail stores have no local emulator or credentials | Medium | Mocked transport plus open acceptance; never recorded as passed |
| Adapter cost exceeds the estimate | Medium | Checkpoint B measures the first four; T9 can reduce scope |
| Profile split breaks existing installs | Medium | Compatibility aggregate extra; old-to-new commands in release notes |
| MCP 2 breaks the Claude backend | Low | T17 is separate and can keep the cap |

## Open questions for the maintainer

1. Provider scope (T9): keep or retire `faiss`, `sql-server`, `azure-cosmos-mongo`?
2. Credentials for live Pinecone and Azure AI Search runs, or accept mocked evidence with the item left open?
3. Extra names and the contents of the compatibility aggregate (T21).
4. Is the fingerprint stored per collection or per record (T15)?

## Progress and decisions

- 2026-11-10: T22 built, CI run pending. `scripts/verify/profiles.py` (`make verify-profiles`) builds the wheel, installs 16 profiles into clean Python 3.11 venvs from the lock, and runs each profile's checks from an empty directory; `.github/workflows/profiles.yml` runs it in four groups on hosted runners. All 16 pass locally. It found one packaging bug, fixed here: `.gitignore` excluded `.env.example.j2`, so the wheel had no `.env.example.j2` template and `holodeck init` failed on every installed copy (regression test: no tracked file under `src/holodeck` is gitignored). Open: a vector store tool whose client is missing logs a warning with the install command and falls back to in-memory storage instead of failing; the base profile accepts that as the documented behavior, but it hides a misconfiguration.
- 2026-11-10: The maintainer delegated the open questions. Decisions: (1) `faiss`, `sql-server` and `azure-cosmos-mongo` are retired with a config-time migration error (T9); (2) Pinecone and Azure AI Search use local or mocked evidence and their live items stay open; (3) the fingerprint is stored per collection (T15); (4) extra names are fixed in T21. Each phase is one PR on a single `gh stack`, with one Codex review round per phase.
- 2026-11-10: Phase 2 verify passed (`make verify`; `make verify-live` 31 passed, 0 failed, 0 skipped, with Qdrant, Chroma and Postgres tool cases on both backends). The Codex review gave ten findings; all are fixed with a regression test each. One part stays open for T15: `HierarchicalDocumentTool` cannot tell a first ingestion that stopped midway from a complete one without a stored chunk count or completion flag.
- 2026-11-10: T9 to T13 done. `faiss`, `sql-server` and `azure-cosmos-mongo` are retired (the maintainer confirmed `sql-server` on 2026-11-10); config naming one fails at validation with the migration text, and stored runs still load. Native adapters for Weaviate, Azure AI Search and Cosmos DB NoSQL are in. No provider uses the SK bridge any more. T13 has nothing to build.
  - Weaviate: all ten operations, verified on Weaviate 1.40 in Docker. `weaviate-client` needs `grpcio<1.80`, so the lock moves `grpcio` from 1.85 to 1.78.
  - Azure AI Search: all ten operations, verified on a fake service and on the real SDK over a stub transport. **Live evidence stays open.**
  - Cosmos DB NoSQL: verified on the Linux emulator except hybrid search, which the emulator cannot run; hybrid is opt-in (`full_text_search: true`) and verified on a fake only. **Live evidence stays open.**
  - The SK paths for Weaviate, Azure AI Search and Cosmos could not create a collection at all, so there is no stored data to migrate for them.
- 2026-11-10: T5 to T8 done. Native adapters for Qdrant, Chroma, Postgres and Pinecone replace the bridge for those providers.
  - Qdrant: layout, point IDs, payload indexes and hybrid ranking proven equal to the SK path in both directions (`tests/integration/retrieval/test_qdrant_compat.py`); goldens unchanged.
  - Chroma: writes are delete-then-add. Chroma 1.5.9 `upsert` and `update` drop a row from the vector index in about 5% of 36-chunk ingests (reproduced; `add` never did in 220 runs). A unit test forbids `upsert` and `update`. The sync client is used for HTTP too, off the event loop, because the async client has no public close.
  - Postgres: psycopg 3 and `pgvector`; DDL equal to SK's. The SK path was broken live in two ways (it ignored a non-default port and returned distance as the score), so Postgres scores and ports are fixed, not preserved. A pgvector container and a `postgres vectorstore tool` case join `make verify-live`.
  - Pinecone: client 10.3. Pinecone Local reached end of life on 2026-10-07 and cannot run the v10 control plane, so the contract suite runs the data plane on Local and the control plane on a fake client. Default collection names had underscores, which Pinecone rejects, so the SK path could never create an index; the adapter normalizes names. **Live Pinecone evidence stays open.**
- 2026-11-10: T2 to T4 done. The contract is `src/holodeck/lib/retrieval/` (types, errors, `VectorStoreAdapter`, registry) with the contract suite in `tests/contract/retrieval/`. Tools hold an adapter; `lib/collection_filter.py` is deleted. Two behavior changes: Chroma results now rank by similarity (SK returned distance), and `HierarchicalDocumentTool` raises a capability error before ingestion on providers with no exact-filter read (pinecone, azure-ai-search, azure-cosmos-nosql and the three retired names). Before, those providers re-ingested on every start and never deleted stale chunks. Native adapters in Phases 2 and 3 restore them.
- 2026-11-10: T0 done. Goldens under `tests/fixtures/golden/`, tests under `tests/integration/golden/` (14 pass on in-memory, Qdrant and Chroma). The goldens freeze six existing defects, listed in each expected file under `known_behavior`: Chroma semantic ranking is inverted (SK returns distance), Qdrant keyword mode returns nothing, Qdrant hybrid scores are raw RRF values, Qdrant hybrid order is not reproducible for multi-match queries, PDF and PowerPoint page selection is zero-based under a positive-only validator, and the plain splitter ignores overlap. A task that fixes one updates its golden on purpose.
- 2026-11-10: T1 done. Removed `aiofiles`, `python-dateutil`, `absl-py`, `ml-dtypes`, `requests` and the `sentry-sdk` constraint. `anthropic` and `ollama` stay: DeepEval's `AnthropicModel` and `OllamaModel` need them.
- 2026-11-10: Plan created after the second inventory and market scan. The retrieval direction changed from LlamaIndex to native adapters the same day; see the selection evidence. No implementation task has started.

## Validation results and remaining blockers

None yet.
