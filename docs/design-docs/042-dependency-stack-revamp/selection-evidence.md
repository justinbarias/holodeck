# Dependency stack selection evidence

**Reviewed:** 2026-11-10 at `00d1ed4` (second round). First round: 2026-09-06 at `3e59e1d`.
**Owner:** Retrieval and packaging maintainers.
**Product contract:** [Spec 042](../../product-specs/042-dependency-stack-revamp/spec.md).

This record separates inspected implementation from upstream claims and from the proposed selection. Package facts come from the PyPI JSON API and released source archives on 2026-11-10. No candidate was installed, benchmarked or run against a live database for this round.

## Second-round result

The first round preferred LlamaIndex as the retrieval foundation. The second round does not support that choice. The evidence now favours HoloDeck-owned native adapters, with no retrieval framework. The maintainer accepted this on 2026-11-10 and the spec was revised the same day. See [Recommendation and decision](#recommendation-and-decision).

Three findings drive the change:

1. LlamaIndex connectors do not provide the operations the spec requires (FR-013, FR-020). See [operation coverage](#llamaindex-operation-coverage).
2. Every framework lags the store clients. That is the same problem SK causes today. See [pin lag](#pin-lag-across-frameworks).
3. HoloDeck already owns native code for the stores it documents and tests. See the [inventory provider table](../../product-specs/042-dependency-stack-revamp/dependency-inventory.md#vector-store-providers-what-works-today).

## Semantic Kernel status and urgency

- Latest release is [`semantic-kernel` 1.47.0](https://pypi.org/project/semantic-kernel/1.47.0/) (2026-11-05). It requires Python `>=3.11`.
- From 1.44, `mcp<2.0` is a core requirement. MCP is at [2.3.0](https://pypi.org/project/mcp/). Both native agent SDKs HoloDeck uses allow `mcp<3`, so SK is the only package that holds MCP back. HoloDeck pins MCP for security fixes.
- SK also caps `pydantic<2.14`, `pymongo<4.17`, `chromadb<1.6` and `pinecone~=7.0`. Pinecone is at [10.3.0](https://pypi.org/project/pinecone/). SK's Pinecone connector fails to import with 8+ (verified locally in #431).
- The [SK repository](https://github.com/microsoft/semantic-kernel) names Microsoft Agent Framework as its successor. SK 1.x receives critical fixes. No end-of-support date is announced. Microsoft's stated minimum is one year after Agent Framework general availability (`agent-framework-core` 1.0.0, 2026-04-02), which gives about April 2027 by inference.
- In HoloDeck's base install SK alone brings 26 packages that nothing else needs.

## Candidate frameworks

Dated PyPI metadata, 2026-11-10. Package presence is not proof of operation parity.

| Candidate | Version (date) | Python | Base weight | Finding |
| --- | --- | --- | --- | --- |
| [llama-index-core](https://pypi.org/project/llama-index-core/0.14.25/) | `0.14.25` (2026-09-21) | `>=3.10,<4.0` | 29 required packages, including nltk, sqlalchemy, `llama-index-workflows`, networkx, pillow | No release in 7 weeks. No slim core. `SentenceSplitter` downloads NLTK data at run time. MIT. |
| [agent-framework-core](https://pypi.org/project/agent-framework-core/) | `1.25.0` (2026-11-07) | `>=3.11` | 7 required packages | New since round one. Has an SK-style vector-store API with precomputed vectors, paged reads without vectors and keyword-hybrid search. The API is marked experimental. It lives in the agent runtime package; no separate vector package exists. |
| [langchain-core](https://pypi.org/project/langchain-core/) + store packages | current | `>=3.10` | varies | `langchain-pinecone` requires `pinecone<8` and Python `<3.14`. `langchain-postgres` requires `pgvector<0.4`. Azure packages pull langgraph. Adds a second document contract. |
| [haystack-ai](https://pypi.org/project/haystack-ai/) | `3.4.1` (2026-10-21) | `>=3.11` | 19 required packages, including posthog | Pipeline runtime and telemetry. No Cosmos or SQL Server store. Collections created outside Haystack generally need migration. |
| pydantic-ai | current | `>=3.11` | n/a | No vector-store abstraction. |

Not credible for this purpose after review: txtai, cognee, mem0, lancedb, vecs, autogen-ext memory. Each is a store, an application framework, or too immature to be an abstraction over eleven stores.

### LlamaIndex connector versions

| Package | Version | Change since 2026-09-06 | Client requirement |
| --- | --- | --- | --- |
| [qdrant](https://pypi.org/project/llama-index-vector-stores-qdrant/0.10.4/) | `0.10.4` | New release; the Python `<3.14` bound is gone | `qdrant-client>=1.16.0` |
| [postgres](https://pypi.org/project/llama-index-vector-stores-postgres/0.9.0/) | `0.9.0` | None | `asyncpg`, `psycopg2-binary`, `pgvector`, `sqlalchemy<2.1` |
| [pinecone](https://pypi.org/project/llama-index-vector-stores-pinecone/0.9.0/) | `0.9.0` | None | `pinecone>=7,<10` |
| [chroma](https://pypi.org/project/llama-index-vector-stores-chroma/0.6.0/) | `0.6.0` | None | `chromadb>=0.5.17` |
| [azureaisearch](https://pypi.org/project/llama-index-vector-stores-azureaisearch/0.6.0/) | `0.6.0` | None | `azure-search-documents<12` (12.0.0 is released) |
| weaviate `1.7.1`, faiss `0.7.0`, azurecosmosmongo `0.9.0`, azurecosmosnosql `1.6.0` | unchanged | None | no conflict found |
| SQL Server | none | Five likely package names return HTTP 404 | n/a |

Conflicts with HoloDeck's lock:

- The Postgres connector requires `sqlalchemy<2.1`. HoloDeck locks sqlalchemy 2.1.8 through Optuna. This does not resolve.
- The Postgres connector imports both asyncpg and psycopg2; HoloDeck uses psycopg 3. It creates its own `data_<name>` table with a fixed schema, so existing SK tables need migration.
- The Pinecone connector allows 7–9, not 10.

### LlamaIndex operation coverage

Read from the released source archives, not from documentation. In `llama-index-core` 0.14.25 the base class's default async methods call the sync method on the event loop, and `get_nodes`/`delete_nodes` raise `NotImplementedError` unless a connector overrides them.

| Connector | Get by ID | Delete by ID | Real async | Metadata filters | Paged read without vectors |
| --- | --- | --- | --- | --- | --- |
| qdrant | Yes | Yes | Yes | Yes | No: scrolls with vectors, limit 9999, no cursor |
| postgres | Yes | Yes | Yes | Yes | No: selects the embedding, no limit |
| chroma | Yes | Yes | No: blocks the event loop | Yes | Partial: limit, no offset |
| pinecone | No: raises `ValueError`; filter-only through a zero-vector query | Yes | No | Yes | No |
| weaviate | No | Yes | Yes | Yes | No |
| azureaisearch | Yes | Yes | Yes | Yes | Not verified |
| azurecosmosnosql | No | No | No | No: raw filter strings only; `delete()` formats the document ID into SQL | No |
| azurecosmosmongo | No | No | No | No | No |
| faiss | No | No (`FaissVectorStore`) | No | No: raises on filters | No |
| SimpleVectorStore | No | Yes | No | Yes | No |

No connector provides paged enumeration without vectors. HoloDeck needs it for incremental ingestion, source deletion and cold-start reload (FR-013, FR-016). Every store would still need native code inside the adapter. Qdrant hybrid in LlamaIndex is dense plus sparse and needs FastEmbed or a supplied sparse function; it is not HoloDeck's dense plus full-text path (FR-018).

### Microsoft Agent Framework connector status

| Store | Package status (2026-11-07 builds) | Limit that matters |
| --- | --- | --- |
| qdrant | Alpha | No keyword-hybrid; HoloDeck already does this natively |
| postgres | Alpha; uses psycopg 3 and pgvector 0.5 | No full-text or hybrid |
| sql-server | Alpha; `mssql-python` on an executor | No keyword-hybrid |
| azure-cosmos (NoSQL) | Beta | `/id` must be the only partition key |
| azure-ai-search | Beta | Native hybrid available |
| mongodb / azure-documentdb | Alpha | Mapping to `azure-cosmos-mongo` not verified |
| in-memory | In core, experimental | n/a |
| chroma, weaviate, pinecone | `0.0.0a1` placeholders with no code | Not usable |
| faiss | No package | Not usable |

Agent Framework is the natural successor to SK's vector API and matches HoloDeck's drivers. It is not ready: connectors are date-stamped alphas, and two of the five stores HoloDeck documents (chroma and pinecone) have no connector.

### Pin lag across frameworks

| Client | Latest | SK 1.47 | LlamaIndex | LangChain | Agent Framework |
| --- | --- | --- | --- | --- | --- |
| pinecone | 10.3.0 | `~=7.0` | `<10` | `<8` | no connector |
| azure-search-documents | 12.0.0 | `<13` | `<12` | n/a | `>=12,<13` |
| pgvector | 0.5.1 | n/a | `<1.0` | `<0.4` | `>=0.5,<0.6` |
| mcp | 2.3.0 | `<2.0` | n/a | n/a | n/a |

## Native client option

All clients below are current and install on Python 3.14. "Exists" means HoloDeck code already calls the native client for that operation today.

| Store | Client (version) | Real async | Native hybrid | Existing HoloDeck native code | Adapter size (estimate) |
| --- | --- | --- | --- | --- | --- |
| qdrant | [qdrant-client](https://pypi.org/project/qdrant-client/) 1.20.0 | Yes | Yes | Client ownership, scroll, payload indexes, full-text RRF | Small |
| postgres | [psycopg](https://pypi.org/project/psycopg/) 3.3.8 + [pgvector](https://pypi.org/project/pgvector/) 0.5.1 | Yes | By SQL (tsvector + vector + RRF) | Filtered reads | Small to medium |
| chromadb | [chromadb](https://pypi.org/project/chromadb/) 1.5.9 | HTTP client only; embedded mode needs a thread | Cloud only | Client setup, filtered reads | Small |
| in-memory | none | n/a | n/a | Filtered reads | Small |
| pinecone | pinecone 10.3.0 | Yes | Sparse-dense | None | Medium |
| weaviate | [weaviate-client](https://pypi.org/project/weaviate-client/) 4.24.1 | Yes | Yes | Filtered reads | Medium |
| azure-ai-search | [azure-search-documents](https://pypi.org/project/azure-search-documents/) 12.0.0 | Yes | Yes | None | Medium |
| azure-cosmos-nosql | [azure-cosmos](https://pypi.org/project/azure-cosmos/) 4.17.1 | Yes | Yes | None | Medium |
| azure-cosmos-mongo | [pymongo](https://pypi.org/project/pymongo/) 4.19.0 | Yes | Not verified | None | Medium |
| faiss | [faiss-cpu](https://pypi.org/project/faiss-cpu/) 1.15.1 | No; in-process | No | None | Medium: needs an ID map, a metadata sidecar and persistence |
| sql-server | [mssql-python](https://pypi.org/project/mssql-python/) 1.17.0 | No; needs an executor | Manual RRF | None | Medium to large |

Adapter sizes are estimates from the required operation list, not measurements. The cost of this option is ownership: HoloDeck maintains each adapter and its live tests. The benefit is that client versions, Python support and security floors no longer wait on a framework.

## Current storage capability evidence

Sources: [collection factory](../../../src/holodeck/lib/vector_store.py), [filter helpers](../../../src/holodeck/lib/collection_filter.py), [keyword routing](../../../src/holodeck/lib/keyword_search.py), [hierarchical tool](../../../src/holodeck/tools/hierarchical_document_tool.py). “Native” describes inspected routing, not a live test result.

| Provider | Current collection owner | Hybrid route | Native filtered enumeration / full scan | 2026-09 candidate direction (superseded by the 2026-11 scan below) |
| --- | --- | --- | --- | --- |
| `postgres` | SK | BM25/OpenSearch + application RRF | Implemented | LlamaIndex Postgres; audit schema, psycopg/asyncpg and enumeration |
| `azure-ai-search` | SK | SK native hybrid | Unsupported by HoloDeck helper | LlamaIndex Azure AI Search plus required operation adapter |
| `qdrant` | SK collection shell + native client | HoloDeck dense/full-text RRF | Implemented | LlamaIndex/native compatibility adapter; preserve existing data |
| `weaviate` | SK | SK native hybrid | Implemented | LlamaIndex Weaviate with native operation coverage |
| `chromadb` | SK + HoloDeck client setup | BM25/OpenSearch + application RRF | Implemented | LlamaIndex Chroma; audit blocking methods |
| `faiss` | SK | BM25/OpenSearch + application RRF | Unsupported by HoloDeck helper | LlamaIndex FAISS plus metadata/persistence support as required |
| `azure-cosmos-nosql` | SK | SK native hybrid | Unsupported by HoloDeck helper | LlamaIndex Cosmos NoSQL plus capability verification |
| `azure-cosmos-mongo` | SK | Falls through to BM25 fallback | Unsupported by HoloDeck helper | LlamaIndex Cosmos MongoDB; resolve routing/name drift |
| `sql-server` | SK | BM25/OpenSearch + application RRF | Unsupported by HoloDeck helper | No verified LlamaIndex candidate; native adapter or evaluated integration required |
| `pinecone` | SK | BM25/OpenSearch + application RRF | Unsupported by HoloDeck helper | LlamaIndex Pinecone; source manifest/enumeration strategy required |
| `in-memory` | SK | BM25 + application RRF | Implemented | LlamaIndex simple store or small typed adapter; same behavior contract |

There is naming drift: `mongodb` appears in native-hybrid capability sets but is not a collection-factory provider. `azure-cosmos-mongo` is absent from both explicit native/fallback sets and reaches fallback through the selector default.

Unsupported reads affect incremental ingestion, source deletion, and cold-start reload. Some callers catch errors into reingest or empty results. These are existing gaps to resolve or reject clearly, not behavior to preserve as successful parity. The new matrix must include existence/create, all record families, upsert, ID read/delete, metadata predicates, pagination, source replacement, restart, vector/lexical/hybrid search, score semantics, installability, and client ownership.

## Qdrant compatibility details

- [Client construction](../../../src/holodeck/lib/vector_store.py) creates an async client explicitly because SK otherwise closes a client across collection contexts.
- [Filtered reads](../../../src/holodeck/lib/collection_filter.py) use native paginated `scroll` and omit vectors.
- [Hierarchical ingestion](../../../src/holodeck/tools/hierarchical_document_tool.py) derives UUIDv5 IDs from tool/chunk identity. It creates keyword indexes for section/definition/source fields and a WORD full-text index for `searchable_text`, with lowercase tokens and lengths 2–20.
- Searchable text includes contextual body, ancestry and section ID twice, the defined term three times, cross references, and filename.
- [Hybrid retrieval](../../../src/holodeck/lib/keyword_search.py) uses dense vector `embedding` plus tokenized `MatchText` OR filters, candidate count `max(4 * top_k, 20)`, and RRF fusion. Empty token queries use dense-only candidates. Payload decoding supplies source/context without loading the full corpus.
- Native-hybrid providers skip unused fallback BM25/OpenSearch setup and full-corpus loading.

A connector default that introduces sparse embeddings, renames vectors, nests payload metadata differently, or reassigns IDs changes this contract. It needs a deliberate migration and retrieval-quality evidence.

## Parsing and persistence evidence

The three record families in `vector_store.py` carry plain, structured, and hierarchical documents. Dynamic metadata, source keys, dimensions, score interpretation, and JSON-string relationship fields need explicit conversion rules. Current vector results clamp similarity to `[0, 1]`; a replacement must distinguish distances from similarities before normalization.

[TextChunker](../../../src/holodeck/lib/text_chunker.py) wraps SK and currently ignores accepted overlap/separator options. [StructuredChunker](../../../src/holodeck/lib/structured_chunker.py) is already HoloDeck code using tiktoken. It must not be replaced merely because LlamaIndex has a splitter.

[FileProcessor](../../../src/holodeck/lib/file_processor.py) serves ingestion and multimodal inputs, including selected pages, Excel sheet/range processing, headings, URLs, and caching. A generic reader listing is not evidence that those extraction semantics are preserved. Reader selection needs format fixtures and license/optional-dependency review.

## Text splitting

`lib/text_chunker.py` is the only non-storage SK import. SK's splitter counts tokens as `len(text) // 4`, and HoloDeck's wrapper accepts overlap and separator options that it ignores.

| Candidate | Version | Weight | Finding |
| --- | --- | --- | --- |
| In-house splitter on tiktoken | n/a | No new package; tiktoken is already required | Estimated 80–120 lines plus tests. Honours size, overlap and separators. |
| [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/) | `0.33.1` | No dependencies (Rust wheel) | Cannot honour custom separators. |
| [chonkie](https://pypi.org/project/chonkie/) | `1.7.0` | 6 required packages | Workable; more than the need. |
| `langchain-text-splitters`, `llama-index-core` | current | Pull their framework core | Not justified for one function. |

Any replacement that counts real tokens changes chunk boundaries. FR-022 and FR-023 apply: new pipeline fingerprint and controlled reindexing.

## Document parsing

| Candidate | Version | Finding |
| --- | --- | --- |
| [markitdown](https://pypi.org/project/markitdown/) | `0.1.9` (2026-10-21), stable | Granular extras exist (`pdf`, `docx`, `pptx`, `xlsx`, `xls`, `outlook`, audio and others). The base package still requires magika, which pulls onnxruntime. `[all]` requires a package that has only prereleases. |
| docling | current | Pulls torch. Too heavy for the base host. |
| unstructured | `0.27.x` | Requires Python `<3.14`. |
| pymupdf4llm | current | AGPL. |
| kreuzberg, LiteParse | current | Light and permissively licensed. Not evaluated against fixtures; watch only. |

## NLP metrics and sparse search

- `evaluate` 0.4.6 has had no release since 2025-09. Only ROUGE and METEOR go through it; `evaluate.load` fetches metric scripts from the Hugging Face Hub at first use. BLEU already calls sacrebleu directly.
- Direct calls reproduce the wrappers: `rouge_scorer.RougeScorer(...).score(...)` and NLTK `single_meteor_score(alpha=0.9, beta=3, gamma=0.5)`. Score equality on HoloDeck fixtures is not yet proven.
- `rouge-score` 0.1.2 (2022) requires NLTK, so NLTK and its scoped security exception stay. No NLTK-free METEOR was found.
- `rank-bm25` 0.2.2 (2022) still works on Python 3.14. [bm25s](https://pypi.org/project/bm25s/) 0.3.16 is active, but no mode reproduces BM25Okapi's idf floor (epsilon times average idf), so rankings change.
- DeepEval [4.3.3](https://pypi.org/project/deepeval/4.3.3/) still requires pytest, four pytest plugins and posthog at run time and has no slim extra.

## Recommendation and decision

Ranked options for replacing SK storage:

1. **Native adapters behind a HoloDeck-owned interface.** Build qdrant, postgres, chromadb and in-memory first; most of that code exists. Add pinecone (which unlocks client 10), then weaviate, azure-ai-search and azure-cosmos-nosql. Decide separately whether faiss, sql-server and azure-cosmos-mongo are kept: none has an install extra, a sample or a live test today.
2. **Microsoft Agent Framework vector stores.** Revisit when the API leaves experimental status and chroma, pinecone and weaviate connectors exist. The HoloDeck interface from option 1 keeps this open.
3. **LlamaIndex.** Heaviest core, a second workflow runtime in the graph, a resolver conflict on Postgres, and no connector that meets the enumeration requirement.
4. **LangChain or Haystack.** Same pin lag as SK, plus a second document contract or pipeline runtime.

Supporting selections: an in-house tiktoken splitter; MarkItDown with selective extras; direct ROUGE/METEOR calls without `evaluate`; keep rank-bm25.

Evidence that would change the ranking: Agent Framework vector stores reaching general availability with the missing connectors; a LlamaIndex release that adds paged enumeration and drops the workflows requirement; measured adapter cost far above the estimates here.

**Decision (2026-11-10):** option 1 is adopted. The spec's objective, working assumptions, section 3, FR-019, FR-021, FR-022 and open-decision table were revised to match. Provider scope for faiss, sql-server and azure-cosmos-mongo remains open.

## Required next evidence

- A maintainer decision on provider scope for faiss, sql-server and azure-cosmos-mongo (FR-014).
- The operation matrix for each kept store, with live tests.
- Frozen retrieval, parser and metric fixtures before any replacement.
- Installed-profile measurements (packages, bytes, import time) for old and new graphs (FR-008).
- A stored-data migration and rollback design (FR-023).

This record does not authorize installing dependencies or migrating production data.
