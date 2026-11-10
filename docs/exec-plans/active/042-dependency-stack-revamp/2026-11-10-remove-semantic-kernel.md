# Remove Semantic Kernel and revamp the dependency stack (feature 042)

**Status:** Implementation complete 2026-11-10; live evidence open. All 25 tasks are done, and the open items under "Validation results and remaining blockers" keep acceptance from being complete. [PLANS.md](../../../PLANS.md) moves a plan to `completed/` only when all acceptance criteria pass, so this plan stays in `active/` until those items close. The maintainer approved the plan and delegated the ask-first decisions; each is recorded under "Progress and decisions".
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

- [x] `make verify` and `make verify-live` pass.
- [x] Maintainer reviews the interface in use before adapter work fans out.

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

- [ ] Run against a hosted Pinecone index (`PINECONE_API_KEY`). **Open: needs credentials.** A verify case exists in `make verify-live` and skips without the key.
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
- [ ] Run against a hosted Azure AI Search service (`AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`). **Open: needs credentials.** A verify case exists in `make verify-live` and skips without them.

**Depends on:** T4. **Size:** M. **Model:** Opus `medium`.

#### T12. Azure Cosmos NoSQL adapter

- [x] Contract suite passes against the Cosmos emulator in Docker.
- [x] New `azure-cosmos` extra.
- [ ] Hybrid search (`full_text_search: true`) and the hosted Cosmos DB service. **Open: needs credentials.** The emulator cannot run hybrid search, which ran on a fake only.

**Depends on:** T4. **Size:** M. **Model:** Opus `medium`.

#### T13. Adapters for providers kept by T9

One task per kept provider, same acceptance shape as T10. Skipped for retired providers.

**Depends on:** T9. **Size:** M each. **Model:** Opus `medium`.

### Phase 4: Splitter and stored data

#### T14. HoloDeck plain text splitter

- [x] A tiktoken-based splitter honours chunk size, overlap and separators; unsupported combinations are rejected.
- [x] `lib/text_chunker.py` no longer imports SK.
- [x] Boundary changes against T0 fixtures are recorded.

**Verify:** new splitter tests; `structured_chunker` tests unchanged.
**Depends on:** T0. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-022.

#### T15. Pipeline fingerprint and migration dry run — ASK

The fingerprint adds stored metadata, so the layout is an ask-first schema decision.

- [x] A fingerprint covers embedding model and dimensions, parser and splitter policy, record layout, vector names and retrieval mode.
- [x] Writes to an incompatible collection are refused with a clear message. Collections with no fingerprint and the current layout count as compatible.
- [x] A read-only dry run reports collection identity, record counts, compatibility and re-embedding need. Migration resumes after interruption and can roll back.

**Verify:** migration tests for dry run, interruption, resume and rollback.
**Depends on:** T5, T14. **Size:** L. Split fingerprint from migration tooling. **Model:** Opus `medium`. **Spec:** FR-023, AC-08.

### Phase 5: SK cutover

#### T16. Remove Semantic Kernel — ASK for the config default

`models/observability.py` defaults a public field to `["semantic_kernel"]`. Changing it changes the published schema.

- [x] Delete the SK bridge, `lib/vector_store.py` SK code, `enable_semantic_kernel_telemetry` and SK namespace defaults; update the 10 test files.
- [x] Remove `semantic-kernel` from `pyproject.toml` and relock. Set `tool.uv.prerelease` to the strictest value that resolves.
- [x] A test fails if `semantic_kernel` is importable in the base install or referenced in `src/`.

**Verify:** `make schema` and `make schema-check`, `make verify`, `make verify-live`, a clean-venv install with SK absent.
**Depends on:** T5 to T8, T10 to T15. **Size:** L. **Model:** Opus `medium`. **Spec:** FR-003, AC-02.

#### T17. Lift the MCP cap

- [x] `mcp` allows 2.x; the Claude and OpenAI backends, MCP bridges and `holodeck mcp` pass. If the Claude backend still fails on MCP 2, keep the cap and record the cause.

**Verify:** `make verify`, `make verify-live`.
**Depends on:** T16. Independent PR. **Size:** S. **Model:** Sonnet `medium`.

#### Checkpoint C

- [x] SK is absent from every profile; H-008 evidence recorded.
- [x] Integrated adversarial review over the full retrieval stack diff.

### Phase 6: Evaluation (independent of Phases 1 to 5)

#### T18. Remove Azure AI evaluation and legacy metrics — ASK

The retired metric names are part of the public YAML schema.

- [x] Delete `lib/evaluators/azure_ai.py`, its exports, executor branches, tests and the `azure-ai-evaluation` dependency.
- [x] YAML with a retired metric name fails before any agent or model call, with the DeepEval replacement named.
- [x] Stored results that used a retired metric still load in the dashboard.

**Verify:** `make schema` and `make schema-check`; evaluator and dashboard tests; `make verify`.
**Depends on:** T0. Independent PR. **Size:** L. Split removal from the migration error if needed. **Model:** Opus `medium`. **Spec:** FR-030, FR-031, AC-13.

#### T19. Call ROUGE and METEOR directly

- [x] `nlp_metrics.py` calls `rouge_score` and NLTK METEOR directly; scores equal the T0 fixtures.
- [x] Remove `evaluate` and the `datasets` and `pyarrow` constraints if nothing else needs them. No network call at metric load.

**Verify:** NLP metric tests with network disabled.
**Depends on:** T0. Independent PR. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-025, AC-09.

### Phase 7: Parsing and packaging

#### T20. MarkItDown selective extras

- [x] `markitdown[pdf,docx,pptx,xlsx]` replaces `[all]`; `openpyxl`, `pdfminer.six` and `python-pptx` are declared directly.
- [x] T0 parser fixtures unchanged for PDF, Word, PowerPoint, Excel, CSV and text. Images and URLs have no golden fixture (`tests/integration/golden/test_parser_golden.py` excludes them), so they are not covered by this check.

**Verify:** file-processor tests, T0 parser fixtures.
**Depends on:** T0. Independent PR. **Size:** S. **Model:** Sonnet `medium`. **Spec:** FR-021, AC-07.

#### T21. Install profiles — ASK

Extra names and what leaves the base install are public packaging changes.

- [x] Maintainer approves the extra names and the compatibility aggregate.
- [x] Docs and release tooling leave runtime dependencies. DeepEval, NLP metrics, the optimizer and OpenSearch become feature extras.
- [x] A command that needs a missing extra prints the install command; unrelated commands still start.

**Verify:** lazy-import guard tests extended per profile; `make verify`.
**Depends on:** T16, T18, T19, T20. **Size:** L. Split tooling move from feature extras. **Model:** Opus `medium`. **Spec:** FR-004, FR-005.

#### T22. Installed-wheel profile tests in CI

- [x] CI builds the wheel and installs each profile in a clean environment, with SK and Microsoft evaluation packages blocked.
- [x] Each profile runs its smoke check.

**Verify:** the new CI job passes on the PR.
**Depends on:** T21. **Size:** M. **Model:** Opus `medium`. **Spec:** AC-01, AC-02, AC-13.

#### T23. Reproducible images

- [x] Generated Dockerfiles install a pinned HoloDeck version with the reviewed extras; no unpinned prerelease resolution.
- [x] Image digest and package manifest recorded.

**Verify:** deploy tests; a local image build.
**Depends on:** T21. **Size:** M. **Model:** Opus `medium`. **Spec:** FR-029, AC-11.

### Phase 8: Close

#### T24. Measure, document and close

- [x] Repeat the T0 measurements and record old against new.
- [x] Update architecture, guides, provider capability docs, the inventory and the changelog; close H-008.
- [x] Fill the acceptance table below with evidence; move this plan to `completed/`.
- [ ] Move this plan to `completed/`. **Open:** the plan stays in `active/` while the items under "Validation results and remaining blockers" are open.
- [ ] Third step of the live validation loop in [AGENTS.md](../../../../AGENTS.md): a CopilotKit browser flow against `holodeck serve`. **Open: user-driven.** `make verify-live` covers steps 1 and 2.
- [ ] FR-029: review the provenance of the Node.js install in the image template (NodeSource). **Open: not reviewed.**

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

Status values: `met`; `met locally; live evidence open` (proved by tests, emulators, fakes or local containers, with a named part not proved against the real service); `partial` (a named part of the criterion is not met). "Live evidence" means a run against a hosted service with credentials. The remaining items are listed under "Validation results and remaining blockers".

| ID | Proved by | Evidence | Status |
| --- | --- | --- | --- |
| AC-01 | T22, T24 | [Dependency inventory, "State after implementation"](../../../product-specs/042-dependency-stack-revamp/dependency-inventory.md) accounts for every declaration, extra and constraint from `pyproject.toml` and `uv.lock`; `uv lock --check` passes (329 entries); `tests/unit/test_verify_profiles.py` fails when an extra has no module list; `scripts/verify/profiles.py` checks that each profile imports its extras' modules (PR #442). | met |
| AC-02 | T16, T22 | `make verify-profiles`: 16 profiles installed from the built wheel into clean Python 3.11 venvs pass locally, and the four `profiles` groups of `.github/workflows/profiles.yml` pass on PR #442 (record of the hosted run under "Validation results"). `semantic-kernel`, `azure-ai-evaluation` and `evaluate` are absent by import and by distribution name in every profile. `tests/unit/test_no_semantic_kernel.py` scans `src/` and `uv.lock` (PR #440); it does not scan `tests/`, where adapter tests and legacy-layout fixtures still name Semantic Kernel. | met |
| AC-03 | T4, T16, T17 | `make verify` and `make verify-live` passed after every phase from 2 to 6: Claude and OpenAI backends through `holodeck test run` and `holodeck serve` AG-UI (2 threads, 2 turns), on MCP 2.3.0 from phase 5 (PR #440). Temporal: `tests/integration/temporal` passed 13 of 13 on a Temporal dev server with the live Claude backend (2026-11-10, credentials from `.verify/live.env`). Ollama: the smoke agent ran through `holodeck test run` on a local `qwen3.8` model through the Claude backend (tool call, tool result and code grader passed; the first run timed out at 60 s while the model loaded). Not run live: Azure OpenAI (no credential). | met; Azure OpenAI live run open |
| AC-04 | T5 to T13 | Operation matrix below. Contract suite `tests/contract/retrieval/test_adapter_contract.py` runs on all eight native adapters; `tests/unit/lib/retrieval/test_retired_providers.py` covers the three retired names. Real stores: Qdrant, Chroma, Postgres and Weaviate in Docker, Cosmos DB on the Linux emulator (except hybrid search). Fake or stub only: Pinecone (data plane on Pinecone Local, control plane on a fake), Azure AI Search (fake service, real SDK over a stub transport), Cosmos DB hybrid search. | met locally; live evidence open |
| AC-05 | T0, T5 | `tests/fixtures/golden/` and `tests/integration/golden/test_retrieval_golden.py` (in-memory, Qdrant, Chroma); `tests/integration/retrieval/test_qdrant_compat.py` proves layout, point IDs, payload indexes and hybrid ranking equal to the SK path in both directions; frozen legacy layouts in `tests/fixtures/legacy_layout/`. PRs #435, #436. | met |
| AC-06 | T4, T15 | The contract suite covers source replacement, shrinking files and capability errors (an unsupported operation raises before ingestion); `tests/integration/retrieval/test_migration.py` covers cold restart and resume; `make verify-live` checks thread isolation and that the real store holds records. Concurrent writers are narrowed, not eliminated, on stores with no compare-and-set (see debt). Source replacement on Pinecone and Azure AI Search ran on fakes. | met locally; live evidence open |
| AC-07 | T0, T14, T20 | `tests/integration/golden/test_parser_golden.py` unchanged by T20 (PR #442); `tests/integration/golden/test_text_chunker_golden.py` and the retrieval goldens re-baselined on purpose in T14 (PR #439); `tests/unit/lib/retrieval/test_fingerprint.py`; `tests/integration/retrieval/test_migration.py`. | met |
| AC-08 | T15 | `tests/integration/retrieval/test_migration.py`: dry run, interruption, resume, rollback and prune on a shared in-memory store and on Qdrant and pgvector containers (a container that is unreachable is skipped); `tests/unit/lib/retrieval/test_manifest.py`. Compatible and legacy collections need no re-embedding. Not run on Pinecone, Azure AI Search, Cosmos DB or Weaviate. PR #439. | met locally; live evidence open |
| AC-09 | T19, T21 | `tests/integration/golden/test_nlp_metrics_golden.py` unchanged at 1e-6; `tests/unit/lib/evaluators/test_nlp_metrics_offline.py` (no network, no model call); DeepEval evaluators in `tests/unit/lib/evaluators/test_deepeval/`; `holodeck test run` stops early with the install command when `eval` or `nlp` is missing (`tests/unit/test_install_profiles.py`); optimizer and Temporal determinism tests run in `make verify` (full suite). PRs #441, #442. | met |
| AC-10 | T16 | `tests/unit/lib/backends/test_otel_bridge.py`, `test_otel_redaction.py`, `test_openai_agents_tracing.py` and `tests/unit/lib/observability/` (in-memory exporters: routing, redaction, no duplicate spans); SK telemetry code and defaults are removed (PR #440). `make verify-live` sends spans to a local Aspire collector and reads them back through the dashboard API (`collector spans redacted` row, added in PR #443): each backend's `leak_probe` span from the run must be present and show `[REDACTED:github-token]`, never the fake token. Both backends passed on 2026-11-10. No hosted collector was used. | met |
| AC-11 | T22, T23, T24 | `make verify-deps` (pip-audit with expiring ignores); [baseline measurements](baseline-measurements.md#result-after-spec-042) compare footprint and startup on the same method; images install `holodeck-ai[extras]==X` with the shipped constraints, the Python image pinned by digest, a build manifest under `.holodeck/builds/` (`tests/unit/deploy/`, PR #442). Local image build and digest: see the T23 entry. FR-029 Node.js provenance and the license review: see "Supply-chain review" below. | met |
| AC-12 | T24 | Architecture, design, quality, reliability, changelog, upgrade guide, vector store guide and dependency inventory updated in this change; `make harness-check` and `uv run mkdocs build --strict` pass. The three copies of `agent.schema.json` under `src/holodeck/templates/*/schemas/` are now written by `make schema` and checked by `make schema-check`. H-008 is resolved (PR #440); the open live evidence is tracked in the debt tracker. | met |
| AC-13 | T18, T22 | `tests/unit/models/test_retired_metrics.py` (retired names fail at config validation with the DeepEval replacement, before any model call); `tests/unit/dashboard/test_load_all.py` (stored runs still load); `azure-ai-evaluation` absent in every profile (PR #441, #442); NLP metrics and code graders pass in `make verify`. | met |

### Operation matrix (AC-04)

Read from `Capabilities` in `src/holodeck/lib/retrieval/adapters/*.py` on 2026-11-10. "Core" means the eight operations every adapter supports: upsert, get by ID, delete by ID, exact metadata filtering, paginated enumeration, delete by metadata filter, record count and vector search. Where an adapter has no native lexical or hybrid search, HoloDeck uses its own BM25 index and fuses the results itself (the adapter's capability hint says so). The user-facing version is in the [vector stores guide](../../../guides/vector-stores.md).

| Provider | Core | Lexical search | Hybrid search | Verified against |
| --- | --- | --- | --- | --- |
| `in-memory` | yes | HoloDeck BM25 | HoloDeck fusion | Contract suite, goldens |
| `qdrant` | yes | no (full-text match is an unscored filter) | native (dense plus full-text, RRF) | Qdrant 1.17 in Docker |
| `chromadb` | yes | HoloDeck BM25 | HoloDeck fusion | Chroma 1.5.9 in Docker (HTTP) and embedded |
| `postgres` | yes | native | HoloDeck fusion | pgvector pg17 in Docker |
| `pinecone` | yes | HoloDeck BM25 | HoloDeck fusion | Pinecone Local (data plane) and a fake control plane. Not the hosted service |
| `weaviate` | yes | native (BM25) | native | Weaviate 1.40 in Docker |
| `azure-ai-search` | yes | native | native | A fake service and the real SDK over a stub transport. Not the hosted service |
| `azure-cosmos-nosql` | yes | HoloDeck BM25 | native only with `full_text_search: true`, otherwise HoloDeck fusion | Linux emulator for everything except hybrid search; hybrid on a fake |

| Retired name | Migration target | Behavior |
| --- | --- | --- |
| `faiss` | `chromadb` (embedded, local files) or `in-memory` | Config validation fails before any agent call, with the migration text |
| `sql-server` | `postgres` (pgvector) | Same |
| `azure-cosmos-mongo` | `azure-cosmos-nosql` | Same |

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

All four were resolved on 2026-11-10.

1. Provider scope (T9): `faiss`, `sql-server` and `azure-cosmos-mongo` are retired. The maintainer confirmed `sql-server` explicitly.
2. Credentials for live Pinecone and Azure AI Search runs: none supplied. Mocked and local evidence is accepted, with the live items left open (see "Validation results and remaining blockers").
3. Extra names (T21): `eval`, `nlp`, `optimizer`, `opensearch`, and the aggregate `all`.
4. Fingerprint storage (T15): per collection, in a companion manifest collection (`holodeck_manifest`), not per record.

## Progress and decisions

- 2026-11-10: Phase 8 (T24, measure, document and close) done in PR #443. Footprint: the base install falls from 264 to 143 packages and from 1,194 to 751 MiB, `holodeck --help` from 5.65 to 1.76 s and the `test` import from 4.8 to 1.2 s; the lock from 376 to 329 entries (table in [baseline measurements](baseline-measurements.md#result-after-spec-042)). The acceptance table is filled; AC-11 and AC-12 were `partial` at that point and were closed later the same day (see the table); AC-04, AC-06 and AC-08 stay `met locally` with hosted-store evidence open. The plan stays in `active/`: PLANS.md has no rule for a plan with open evidence, and it moves a plan to `completed/` only when all acceptance criteria pass. H-008 is resolved; new debt entries record the open items. Final run at the phase 8 tip, 2026-11-10: `make verify` passed (7,683 tests passed, 107 skipped; smoke 12 of 12) and `make verify-live` passed (57 passed, 0 failed, 4 skipped; the skips are the credential-gated cases). An earlier live run at this tip failed one case, the OpenAI AG-UI thread B turn 2 code-word recall, where the model refused to repeat the word; the same case passed in this run.
- 2026-11-10: Verification record and reviews. Every phase from 2 to 6 passed `make verify` and `make verify-live`. From phase 3 the live suite includes the provider matrix (Qdrant, Chroma, Postgres, Weaviate and the Cosmos emulator; the hierarchical tool on Qdrant, Postgres and Weaviate), with a check that the real store holds records: 57 passed, 0 failed, 4 skipped on the stack tip. The four skips are Pinecone and Azure AI Search on each backend (no credentials). One live run had a single flaky failure (the OpenAI model declined to repeat a code word in an AG-UI turn); the re-run passed. Each phase had one Codex (gpt-6-astra, high) adversarial review:

  | Phase | PR | Findings | Disposition |
  | --- | --- | --- | --- |
  | 0 | #434 | 6 | 4 fixed; 1 accepted (`NO_PROXY` CIDR matching differs on HTTPX); 1 resolved by T19 |
  | 1 | #435 | 6 | 5 fixed across phases 1 and 2; 1 accepted in the temporary bridge, since deleted |
  | 2 | #436 | 10 | 10 fixed |
  | 3 | #438 | 9 | 9 fixed |
  | 4 | #439 | 13 | 13 fixed |
  | 5 | #440 | 1 | 1 fixed |
  | 6 | #441 | 1 | 1 fixed |
  | 7 | #442 | 5 | 5 fixed, plus the hosted `pre-commit` MyPy failure in `scripts/verify/profile_probe.py`. `make verify` passed (7,658 tests), `make verify-profiles` passed 16 of 16 profiles, `make verify-live` 57 passed, 0 failed, 4 skipped; these ran before the review-fix commit, after which the full suite (7,683 passed) and the base, qdrant, all and stores profiles were run again. |

  The phases are one `gh stack`, one PR each. Phase 8 is in PR #443. Its Codex review gave 5 findings, all fixed: the schema generator skipped a deleted template copy; the upgrade guide and changelog promised an in-memory fallback for an unreachable store, which holds only for a failure while the adapter is created; and three evidence lines in this plan overstated coverage (image and URL parser fixtures, the collector read-back for AC-10, and the Semantic Kernel scan of `tests/`).
- 2026-11-10: T23 done (PR #442). Images install `holodeck-ai[extras]==X.Y.Z` on `holodeck-base:X.Y.Z`, with `--constraint` and `--override` files that the wheel ships (`src/holodeck/deploy/constraints.txt` and `overrides.txt`, generated from `uv.lock` by `make constraints`; `make constraints-check` runs in `make verify-fast`). No `--prerelease` flag is used. `holodeck.deploy.extras.required_extras(agent)` is the single rule for extras: serving images never get `eval`, `nlp` or `optimizer`. `deploy build` writes a build manifest to `.holodeck/builds/<tag>.json` (version, extras, base image reference and digest, image ID, package list), and each image keeps `/opt/holodeck/manifest.txt`. The Python base image and uv are pinned by digest, `Dockerfile.local` no longer pipes the uv installer into a shell, and the base-image workflow resolves the latest PyPI release instead of building `latest`. Local build evidence, 2026-11-10: image `sha256:06afd3bfda5d…` holds 146 packages with 0 version mismatches against the shipped constraints, and two builds gave identical package lists.
- 2026-11-10: T22 on GitHub: on PR #442 the four `profiles` groups (`base`, `features`, `stores`, `all`) of `profiles.yml` pass. The `pre-commit` job fails: its MyPy hook reports errors in `scripts/verify/profile_probe.py` (missing named arguments for `VectorstoreTool`, an unexpected `location` argument for `DatabaseConfig`, and a `None` passed to `NumericProposer.begin`), which the local run of the profile checks does not catch. The `test (3.11)` and `test (3.13)` jobs were still pending when this was written. This is a code fix for the owner of T22 (see the debt tracker).
- 2026-11-10: T20 and T21 done (PR #442). T20: `markitdown[pdf,docx,pptx,xlsx]` replaces `[all]`; the parser goldens are unchanged. T21: extras `eval`, `nlp`, `optimizer`, `opensearch` and the aggregate `all`; `mkdocs-material` and `twine` are dev-only. `MissingExtraError` stops `holodeck test run` before any agent call when a metric needs a missing extra. A missing vector store client or a retired provider is now an error, with no in-memory fallback (commit 7c61589); a store that cannot be reached still falls back (debt entry).
- 2026-11-10: T18 and T19 done (PR #441). T18: Azure AI evaluation is removed. The metric names `groundedness`, `relevance`, `coherence`, `fluency`, `similarity` and `safety` fail at config validation and name the DeepEval replacement; stored runs still load in the dashboard. T19: ROUGE runs through `rouge_score` and METEOR through NLTK, with no network call at metric load. METEOR needs `python -m nltk.downloader wordnet punkt_tab omw-1.4`, and CI downloads it. The goldens are unchanged at 1e-6. The phase 6 review gave one finding, fixed (METEOR names the NLTK download command when its data is incomplete).
- 2026-11-10: T16 and T17 done (PR #440). T16: Semantic Kernel is removed. The legacy Qdrant, Chroma and Postgres layouts are frozen as fixtures under `tests/fixtures/legacy_layout/`, so the compatibility proof no longer needs SK. `tool.uv.prerelease` is removed. The lock went from 378 to 347 entries at that point. `tests/unit/test_no_semantic_kernel.py` fails if SK is importable or named in `src/`. No deprecated SK config key had to be kept (the last one went in an earlier release), and the default of `observability.logs.filter_namespaces` is now empty. T17: `mcp` is `>=1.30.0,<3`, locked on 2.3.0, and verified on both real backends in `make verify-live`. The phase 5 review gave one finding, fixed.
- 2026-11-10: T14 and T15 done (PR #439). T14: a tiktoken `cl100k_base` splitter honours chunk size and overlap; the chunker and `VectorStoreTool` goldens were re-baselined on purpose. T15: a pipeline fingerprint and a companion manifest collection, `holodeck_manifest`, per store connection (the main layouts are unchanged); per-source completion markers; and `holodeck vectorstore inspect|migrate|rollback|prune`. The Codex review gave 13 findings, all fixed. Residual risk, accepted and documented in the vector stores guide: concurrent writers are narrowed, not eliminated, on stores with no compare-and-set.
- 2026-11-10: T22 built (the hosted run is in the entry above). `scripts/verify/profiles.py` (`make verify-profiles`) builds the wheel, installs 16 profiles into clean Python 3.11 venvs from the lock, and runs each profile's checks from an empty directory; `.github/workflows/profiles.yml` runs it in four groups on hosted runners. All 16 pass locally. It found one packaging bug, fixed here: `.gitignore` excluded `.env.example.j2`, so the wheel had no `.env.example.j2` template and `holodeck init` failed on every installed copy (regression test: no tracked file under `src/holodeck` is gitignored). That entry listed one open item: a vector store tool whose client is missing logged a warning and fell back to in-memory storage. Commit 7c61589 resolves it: a missing client is now an error.
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

Done and verified at the tip of the stack (`536cad9`, T23): every phase from 2 to 6 passed `make verify` and `make verify-live` (57 passed, 0 failed, 4 skipped on the stack tip). Install profiles: 16 pass locally on Python 3.11, and the four hosted groups pass on PR #442. Footprint: see [baseline measurements](baseline-measurements.md#result-after-spec-042). The review counts are under "Progress and decisions".

### Supply-chain review (2026-11-10, PR #443)

**Node.js in generated images (FR-029).** The image template piped the NodeSource setup script into a shell. It now adds the NodeSource apt repository with its signing key (`signed-by=/etc/apt/keyrings/nodesource.asc`) and installs `nodejs` with apt, which verifies the package signature; no remote script runs. A test build on the pinned `python:3.13-slim` image installed `nodejs 22.23.4-1nodesource1`; the key file's SHA-256 was `b42e0321dabdc24e892115da705cf061167eac12a317f23d329862d0aa0a271d`. The key is fetched over TLS and is not pinned by checksum, and the Node.js patch version is not pinned. Node.js is installed only when the agent needs it.

**Licenses.** Method: the license metadata of every distribution in `uv export --all-extras --no-dev` (327 names; 321 installed on macOS, the other 6 are platform-specific). Result: 309 permissive (MIT, BSD, Apache-2.0, ISC, PSF and similar). No GPL-only or AGPL distribution is a runtime dependency. Weak copyleft, all used unmodified: MPL-2.0 for `certifi`, `fqdn`, `pathspec`, `orjson` and `tqdm` (the last two dual or combined with MIT/Apache); LGPL-3.0 for `psycopg`, `psycopg-binary` and `psycopg-pool`, which only the `postgres` extra installs. `docutils` (public domain, BSD and GPL parts) comes only through `twine` in the `dev` extra. `bkflow-feel` declares no license in its metadata; its repository is MIT. This is a metadata review, not legal advice.

Hosted CI on PR #442, read with `gh pr checks 442` on 2026-11-10: `profiles` (base, features, stores, all) pass; CodeQL passes; `pre-commit` failed on MyPy errors in `scripts/verify/profile_probe.py` and passes after the phase 7 review-fix commit. Final state on 2026-11-10: all checks pass on #442 and on #443 (`test (3.11)`, `test (3.13)`, the four `profiles` jobs, `pre-commit`, CodeQL and `codecov/patch`), read before the last evidence commit on #443.

Remaining blockers for full acceptance. None of these is a failing test; each is evidence that does not exist yet.

1. **Pinecone.** The contract suite ran on Pinecone Local (end of life) for the data plane and on a fake control plane. No run against the hosted service. A credential-gated case exists in `make verify-live` and skips without `PINECONE_API_KEY`.
2. **Azure AI Search.** Verified on a fake service and on the real SDK over a stub transport only. A credential-gated case exists and skips without the endpoint and key.
3. **Azure Cosmos DB NoSQL.** The Linux emulator covers everything except hybrid search, which ran on a fake only. The hosted service has not been used.
4. **Manifest and fingerprint.** Not run live on Pinecone, Azure AI Search, Cosmos DB or Weaviate. They ran on Qdrant and pgvector containers and on a shared in-memory store.
5. **CopilotKit browser step.** Step 3 of the live validation loop in AGENTS.md was not run. It is user-driven.
6. **Hosted CI.** `test (3.11)` and `test (3.13)` on PR #442 were pending, and the `pre-commit` job fails on MyPy errors in `scripts/verify/profile_probe.py`. Both need a final read before the stack merges.
7. **FR-029.** The provenance of the Node.js install in the image template (NodeSource) has not been reviewed.
8. **Live backends beyond Claude and OpenAI.** Azure OpenAI, Ollama and a Temporal server were not part of `make verify-live` (AC-03).
9. **Template schemas.** Closed in PR #443: `make schema` writes the copies under `src/holodeck/templates/*/schemas/` and `make schema-check` checks them.

Items 1 to 5 and 7 to 9 are in the [technical debt tracker](../../tech-debt-tracker.md) with an owner and an exit criterion. The plan closes (moves to `completed/`) when each is met or the maintainer accepts it in writing.
