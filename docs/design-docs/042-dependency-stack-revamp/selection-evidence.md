# Dependency stack selection evidence

**Reviewed:** 2026-09-06. **Code:** `3e59e1d`.
**Owner:** Retrieval and packaging maintainers.
**Product contract:** [Spec 042](../../product-specs/042-dependency-stack-revamp/spec.md).

This record separates inspected implementation from upstream capability claims and proposed selection. No LlamaIndex installation, benchmark, database migration, or provider certification was performed for this draft.

## Selection direction

Use LlamaIndex core and selected vector-store integrations as the preferred retrieval foundation. Keep HoloDeck-owned public records and capability contracts. Keep native Claude/OpenAI execution, LiteLLM inference, MCP, OpenTelemetry, and Temporal in their existing roles.

The [LlamaIndex installation guide](https://developers.llamaindex.ai/python/framework/getting_started/installation/) documents modular packages. The starter bundle also installs OpenAI LLM/embedding integrations; selective installation avoids adding a second default inference configuration. Package metadata still shows substantial core dependencies, so this is a responsibility-consolidation proposal, not a measured footprint reduction.

The [vector-store API](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/) provides node/query/result/filter abstractions. An available abstraction does not imply every connector implements filtering, enumeration, hybrid search, or true asynchronous I/O. Inspect and test the selected package versions before acceptance.

The [Qdrant integration](https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/qdrant/) exposes async clients, payload-index settings, named vectors, and dense/sparse hybrid search. Its documented hybrid behavior does not establish equivalence with HoloDeck's dense/full-text path. The [native Qdrant client](https://github.com/qdrant/qdrant-client) supplies typed sync/async APIs and local mode and remains an allowed adapter dependency.

Other candidates were considered. [LangChain Qdrant](https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant) supports filters, existing collections, and dense/sparse hybrid retrieval, but introduces another document contract. [Haystack Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store) explicitly cautions that collections created outside Haystack generally require migration. Neither is selected by this proposal.

Microsoft's [SK repository](https://github.com/microsoft/semantic-kernel) identifies Microsoft Agent Framework as its successor. That supports an explicit exit strategy; it does not justify introducing another agent runtime to replace the remaining storage/text dependencies.

## Current storage capability evidence

Sources: [collection factory](../../../src/holodeck/lib/vector_store.py), [filter helpers](../../../src/holodeck/lib/collection_filter.py), [keyword routing](../../../src/holodeck/lib/keyword_search.py), [hierarchical tool](../../../src/holodeck/tools/hierarchical_document_tool.py). “Native” describes inspected routing, not a live test result.

| Provider | Current collection owner | Hybrid route | Native filtered enumeration / full scan | Candidate direction |
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

## Candidate package metadata

The following table is a dated PyPI metadata snapshot, not a resolved version set or installation recommendation. Versions must be pinned and tested together during implementation planning. Package presence is not proof of operation parity.

| Package | Observed version | Requires Python |
| --- | --- | --- |
| [llama-index-core](https://pypi.org/project/llama-index-core/0.14.24/) | `0.14.24` | `<4.0,>=3.10` |
| [llama-index-vector-stores-qdrant](https://pypi.org/project/llama-index-vector-stores-qdrant/0.10.3/) | `0.10.3` | `<3.14,>=3.10` |
| [llama-index-vector-stores-postgres](https://pypi.org/project/llama-index-vector-stores-postgres/0.9.0/) | `0.9.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-azureaisearch](https://pypi.org/project/llama-index-vector-stores-azureaisearch/0.6.0/) | `0.6.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-weaviate](https://pypi.org/project/llama-index-vector-stores-weaviate/1.7.1/) | `1.7.1` | `<4.0,>=3.10` |
| [llama-index-vector-stores-chroma](https://pypi.org/project/llama-index-vector-stores-chroma/0.6.0/) | `0.6.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-faiss](https://pypi.org/project/llama-index-vector-stores-faiss/0.7.0/) | `0.7.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-azurecosmosmongo](https://pypi.org/project/llama-index-vector-stores-azurecosmosmongo/0.9.0/) | `0.9.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-azurecosmosnosql](https://pypi.org/project/llama-index-vector-stores-azurecosmosnosql/1.6.0/) | `1.6.0` | `<4.0,>=3.10` |
| [llama-index-vector-stores-pinecone](https://pypi.org/project/llama-index-vector-stores-pinecone/0.9.0/) | `0.9.0` | `<4.0,>=3.10` |

Important implications:

- Core currently depends on NLTK, NumPy, SQLAlchemy, HTTP clients, tokenization packages, and `llama-index-workflows`. Keep workflow execution on Temporal and review the existing scoped NLTK security exception for any newly introduced usage.
- The Qdrant candidate advertises Python `<3.14`; HoloDeck currently declares `>=3.10` without an upper bound. Resolve that constraint before adopting it; do not silently narrow support.
- The Postgres candidate uses `asyncpg` and `psycopg2-binary`; HoloDeck currently declares psycopg 3. Driver duplication and compatibility must be measured and resolved.
- A lookup for `llama-index-vector-stores-mssql` returned HTTP 404. This establishes no verified candidate under that name, not proof that no third-party integration exists.
- Candidate package metadata is not a substitute for license/security review or clean installed-wheel tests. No claim of full eleven-provider replacement is made here.

## Required next evidence

The reviewed implementation plan must select a compatible version set, complete the operation matrix, freeze retrieval/parser fixtures, measure identical installation profiles, and specify stored-data migration/rollback. Owners and release acceptance are defined in the product spec. This record alone does not authorize installing dependencies or migrating production data.
