# Implementation Plan: HierarchicalDocumentTool

**Branch**: `020-structured-document-tool` | **Date**: 2026-01-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/020-structured-document-tool/spec.md`

## Summary

Implement a HierarchicalDocumentTool that extends HoloDeck's vectorstore capabilities with:
1. **Structure-aware document parsing** - Preserve hierarchical structure (headings → parent chains)
2. **Hybrid search** - Combine dense embeddings, sparse keyword index, and exact match
3. **Contextual embeddings** - Prepend structural context before embedding (per Anthropic's research)
4. **Reciprocal Rank Fusion** - Merge results from multiple search modalities
5. **Definition/cross-reference extraction** - Auto-detect and index definitions and references

This builds on existing `VectorStoreTool` patterns while adding markdown structure parsing, tiered keyword indexing (native or configurable fallback), and RRF-based result fusion.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Semantic Kernel (existing) - vector store abstraction, embeddings
- opensearch-py - OpenSearch client for production sparse keyword index service
- rank_bm25 - BM25Okapi implementation for in-memory sparse fallback (dev/local)
- markitdown (existing) - document-to-markdown conversion
- Pydantic v2 (existing) - configuration models

**Storage**: In-memory by default (via Semantic Kernel), configurable to postgres, qdrant, weaviate, chromadb, faiss, pinecone, azure-ai-search, etc.

**Native Hybrid Search Support by Provider** (via Semantic Kernel `collection.hybrid_search()`):
| Provider | Native Hybrid | Strategy |
|----------|---------------|----------|
| azure-ai-search | ✅ Yes | Use `collection.hybrid_search()` |
| weaviate | ✅ Yes | Use `collection.hybrid_search()` |
| qdrant | ✅ Yes | Use `collection.hybrid_search()` |
| mongodb | ✅ Yes | Use `collection.hybrid_search()` (MongoDB Atlas) |
| azure-cosmos-nosql | ✅ Yes | Use `collection.hybrid_search()` |
| postgres | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |
| pinecone | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |
| chromadb | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |
| faiss | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |
| in-memory | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |
| sql-server | ❌ No | Use configurable sparse fallback (OpenSearch endpoint preferred, in-memory BM25 optional) + app-level RRF |

**Note**: MongoDB Atlas requires the `MongoDBAtlasStore` connector (not `azure-cosmos-mongo`). Azure Cosmos MongoDB vCore (`azure-cosmos-mongo`) is EXCLUDED from both lists as it does NOT support hybrid search natively and requires a different integration pattern.

**Testing**: pytest with markers (`@pytest.mark.unit`, `@pytest.mark.integration`)

**Target Platform**: Linux server (same as existing HoloDeck)

**Project Type**: Single project - extends existing `src/holodeck/` structure

**Performance Goals**:
- Ingestion: <30 seconds per 100 pages
- Search: <2 seconds for up to 10,000 chunks
- 800 tokens default chunk size (Anthropic baseline)

**Constraints**:
- Default 10 results, max 100 per query
- Must integrate with existing tool framework (ToolUnion discriminated union)
- YAML-only configuration (no code required for users)

**Scale/Scope**: Up to 10,000 chunks per corpus

## Preprocessing Pipeline (Anthropic Contextual Retrieval)

The preprocessing pipeline follows Anthropic's contextual retrieval approach, using LLM-generated context for each chunk.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 1: DOCUMENT CONVERSION (existing markitdown)                     │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  Input: PDF, Word, HTML, etc.                                          │ │
│  │  Output: Markdown text                                                 │ │
│  │  Tool: markitdown (existing integration)                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 2: STRUCTURE PARSING (StructuredChunker)                         │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  • Parse markdown headings → hierarchical tree                         │ │
│  │  • Extract parent_chain for each section (e.g., ["Title I", "Ch 2"])   │ │
│  │  • Detect definitions (heading keywords + "X means Y" patterns)        │ │
│  │  • Detect cross-references (§4.2, "see Section 5", etc.)               │ │
│  │  • Classify chunk types (definition, requirement, content, header)     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 3: CHUNKING (structure-aware, max 800 tokens)                    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  • Split at structural boundaries (headings, paragraphs)               │ │
│  │  • Respect max_chunk_tokens (default 800 per Anthropic)                │ │
│  │  • Preserve parent_chain metadata per chunk                            │ │
│  │  • Generate chunk IDs: "{source_path}_chunk_{index}"                   │ │
│  │  • Handle overflow: split long sections at sentence boundaries         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 4: LLM CONTEXT GENERATION (Anthropic approach)                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  For each chunk:                                                       │ │
│  │    1. Send WHOLE document + chunk to Claude Haiku                      │ │
│  │    2. Generate 50-100 token context explaining the chunk               │ │
│  │    3. Prepend context to chunk: "{context}\n\n{chunk.content}"         │ │
│  │                                                                        │ │
│  │  Prompt:                                                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ <document>{WHOLE_DOCUMENT}</document>                            │ │ │
│  │  │ Here is the chunk we want to situate within the whole document   │ │ │
│  │  │ <chunk>{CHUNK_CONTENT}</chunk>                                   │ │ │
│  │  │ Please give a short succinct context to situate this chunk       │ │ │
│  │  │ within the overall document for the purposes of improving        │ │ │
│  │  │ search retrieval of the chunk. Answer only with the succinct     │ │ │
│  │  │ context and nothing else.                                        │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  Concurrency: 10 parallel LLM calls (configurable)                     │ │
│  │  Cost: ~$0.03 per 100-page document (using Haiku)                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 5: EMBEDDING GENERATION                                          │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  • Input: Contextualized text (context + original chunk)               │ │
│  │  • Model: Configured embedding model (e.g., text-embedding-3-small)    │ │
│  │  • Output: Dense vector (e.g., 1536 dimensions)                        │ │
│  │  • Batch processing for efficiency                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 6: MULTI-INDEX CONSTRUCTION                                      │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │ │
│  │  │  DENSE INDEX    │  │  KEYWORD INDEX  │  │  EXACT INDEX    │        │ │
│  │  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤        │ │
│  │  │ Contextualized  │  │ Contextualized  │  │ section_id →    │        │ │
│  │  │ embeddings      │  │ text (native or │  │ chunk_id        │        │ │
│  │  │                 │  │ sparse fallback)│  │                 │        │ │
│  │  │ → Vector Store  │  │ → Native/Sparse │  │ → Dict lookup   │        │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │ │
│  │                                                                        │ │
│  │  + Definition Index: term_normalized → DefinitionEntry                 │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STAGE 7: PERSISTENCE                                                   │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  Dense Index + Metadata → Vector Store (postgres, qdrant, etc.)        │ │
│  │  Keyword Index:                                                        │ │
│  │    • Native hybrid providers → Persisted by provider                   │ │
│  │    • Fallback (opensearch) → Persisted in OpenSearch endpoint           │ │
│  │    • Optional in-memory BM25 fallback → Rebuilt from chunks on startup  │ │
│  │  Exact/Definition Index → Rebuilt from stored chunks on startup        │ │
│  │  Original + Contextualized content → Stored in vector store records    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Both embeddings AND sparse keyword index use the contextualized text. This allows keyword searches to match on the LLM-generated context, not just the original chunk content.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | ✅ PASS | Tool configured entirely via YAML |
| II. MCP for API Integrations | ✅ N/A | No external API integrations (internal tool) |
| III. Test-First with Multimodal Support | ✅ PASS | Supports PDF, Word, text via markitdown |
| IV. OpenTelemetry-Native Observability | ⚠️ DEFERRED | Logging via existing patterns; OTel traces future work |
| V. Evaluation Flexibility | ✅ N/A | Not an evaluation feature |

**Architecture Constraints Check**:
- ✅ Agent Engine integration (tool type)
- ✅ Independently testable
- ✅ Well-defined contracts (Pydantic models)

## Project Structure

### Documentation (this feature)

```text
specs/020-structured-document-tool/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── hierarchical_document_tool_config.yaml
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── tool.py                      # Add HierarchicalDocumentTool config model
├── tools/
│   ├── vectorstore_tool.py          # Existing (reference patterns)
│   └── hierarchical_document_tool.py  # NEW: Main tool implementation
├── lib/
│   ├── vector_store.py              # Existing (reuse patterns)
│   ├── text_chunker.py              # Existing (extend for structure-aware)
│   ├── structured_chunker.py        # NEW: Markdown structure parser
│   ├── keyword_search.py            # NEW: Tiered keyword search (native/OpenSearch/in-memory)
│   ├── hybrid_search.py             # NEW: RRF fusion + provider-aware hybrid orchestration
│   ├── definition_extractor.py      # NEW: Definition/cross-ref extraction
│   └── tool_filter/
│       └── index.py                 # REFACTOR: Use shared keyword_search.py & hybrid_search.py

tests/
├── unit/
│   ├── lib/
│   │   ├── test_structured_chunker.py
│   │   ├── test_keyword_search.py       # Tests for OpenSearch + in-memory fallback + provider detection
│   │   ├── test_hybrid_search.py        # Tests for RRF fusion logic
│   │   └── test_definition_extractor.py
│   └── tools/
│       └── test_hierarchical_document_tool.py
├── integration/
│   └── tools/
│       └── test_hierarchical_document_integration.py
└── fixtures/
    └── hierarchical_documents/
        ├── sample_legislation.md
        ├── sample_technical_doc.md
        └── sample_flat_text.txt
```

**Structure Decision**: Single project extending existing `src/holodeck/` layout. New modules in `lib/` for reusable components, new tool in `tools/`, configuration model in `models/tool.py`.

## Complexity Tracking

No constitution violations requiring justification.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HierarchicalDocumentTool                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐                     │
│  │   Document Input    │    │   Configuration     │                     │
│  │   (via markitdown)  │    │   (YAML → Pydantic) │                     │
│  └──────────┬──────────┘    └──────────┬──────────┘                     │
│             │                          │                                 │
│             ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              StructuredChunker                               │        │
│  │  • Parse markdown headings → hierarchical tree               │        │
│  │  • Chunk by structure (max 800 tokens)                       │        │
│  │  • Preserve parent_chain metadata                            │        │
│  │  • Extract definitions & cross-references                    │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                            │
│             ┌───────────────┼───────────────┐                           │
│             ▼               ▼               ▼                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ Dense Index  │  │ Keyword Index│  │ Exact Index  │                   │
│  │ (Embeddings) │  │ (see below)  │  │ (dict lookup)│                   │
│  │              │  │              │  │              │                   │
│  │ Contextual   │  │ Tiered:      │  │ Section IDs  │                   │
│  │ embedding    │  │ Native/Sparse│  │ + phrases    │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                 │                            │
│         └─────────────────┼─────────────────┘                           │
│                           ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              HybridSearch (Provider-Aware)                   │        │
│  │  • Native hybrid: azure-ai-search, weaviate, qdrant, mongodb │        │
│  │  • Fallback: OpenSearch endpoint (or in-memory BM25) + RRF   │        │
│  │  • Configurable weights per modality                         │        │
│  │  • Optional reranking                                        │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              SearchResult                                    │        │
│  │  • content, score, source_path, parent_chain                 │        │
│  │  • Per-modality scores (semantic, keyword, exact)            │        │
│  │  • Definitions context (if applicable)                       │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Keyword Index Strategy (Tiered)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Provider Detection                                   │
│                            │                                             │
│              ┌─────────────┴─────────────┐                              │
│              ▼                           ▼                              │
│  ┌───────────────────────┐   ┌───────────────────────┐                 │
│  │    Native Hybrid      │   │  Sparse Fallback      │                 │
│  │   (SK hybrid_search)  │   │ (OpenSearch/BM25)     │                 │
│  ├───────────────────────┤   ├───────────────────────┤                 │
│  │ • azure-ai-search     │   │ • postgres            │                 │
│  │ • weaviate            │   │ • pinecone            │                 │
│  │ • qdrant              │   │ • chromadb            │                 │
│  │ • mongodb (Atlas)     │   │ • faiss               │                 │
│  │ • azure-cosmos-nosql  │   │ • in-memory           │                 │
│  │                       │   │ • sql-server          │                 │
│  ├───────────────────────┤   ├───────────────────────┤                 │
│  │ collection.hybrid_    │   │ Query OpenSearch text │                 │
│  │ search(query,         │   │ index (preferred) or  │                 │
│  │   additional_property │   │ local BM25, then fuse │                 │
│  │   _name="content")    │   │ with app-level RRF    │                 │
│  │                       │   │ (k=60)                │                 │
│  └───────────────────────┘   └───────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. LLM-Based Contextual Embedding (per Anthropic)

**Decision**: Use LLM to generate context for each chunk before embedding (full Anthropic approach).

**Rationale**:
- Anthropic's research shows 49% reduction in retrieval failures (67% with reranking)
- LLM understands semantic meaning, not just structural position
- Context explains what the chunk is about within the document
- Both embeddings AND sparse keyword index use the contextualized text

**Prompt Template** (from Anthropic):

```python
CONTEXT_PROMPT_TEMPLATE = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
```

**Implementation**:

```python
class LLMContextGenerator:
    """Generate context for chunks using LLM (Anthropic approach).

    Calls the configured LLM (default: Claude Haiku) for each chunk,
    providing the whole document + chunk to generate a 50-100 token
    context that explains what the chunk is about.
    """

    def __init__(self, llm_service: Any, max_context_tokens: int = 100):
        self._llm = llm_service
        self._max_tokens = max_context_tokens

    async def generate_context(self, chunk_text: str, document_text: str) -> str:
        """Generate context for a single chunk.

        Args:
            chunk_text: The chunk content to contextualize
            document_text: The full document for context

        Returns:
            Generated context (50-100 tokens typically)
        """
        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            document=document_text,
            chunk=chunk_text
        )
        context = await self._llm.complete(prompt, max_tokens=self._max_tokens)
        return context.strip()

    async def contextualize_chunk(self, chunk: DocumentChunk, document_text: str) -> str:
        """Prepend LLM-generated context to chunk content.

        Returns:
            Contextualized text: "{context}\n\n{chunk.content}"
        """
        context = await self.generate_context(chunk.content, document_text)
        return f"{context}\n\n{chunk.content}"

    async def contextualize_batch(
        self,
        chunks: list[DocumentChunk],
        document_text: str,
        concurrency: int = 10
    ) -> list[str]:
        """Contextualize multiple chunks with controlled concurrency.

        For a 100-page document with ~125 chunks, this makes ~125 LLM calls.
        Using Haiku at ~$0.25/1M input tokens, cost is roughly:
        - 800 tokens/chunk * 125 chunks = 100K tokens input
        - 100 tokens/context * 125 chunks = 12.5K tokens output
        - Total cost: ~$0.03 per 100-page document
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def contextualize_with_limit(chunk: DocumentChunk) -> str:
            async with semaphore:
                return await self.contextualize_chunk(chunk, document_text)

        tasks = [contextualize_with_limit(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
```

**Configuration**:

```yaml
# agent.yaml
tools:
  - name: legal_docs
    type: hierarchical_document
    source: ./docs

    # Context generation (required for contextual embeddings)
    contextual_embeddings: true
    context_model:
      provider: anthropic
      name: claude-3-haiku-20240307  # Recommended: fast & cheap
      temperature: 0.0
    context_max_tokens: 100
    context_concurrency: 10  # Parallel LLM calls during ingestion
```

**Cost Estimate** (per Anthropic):
- ~$1.02 per 10M tokens of source material
- For a typical 100-page document (~125 chunks): ~$0.03

**What Gets Indexed**:
- **Dense Index**: Embed the contextualized text (context + chunk)
- **Sparse Keyword Index**: Index the contextualized text (OpenSearch endpoint or in-memory BM25)
- **Exact Index**: Uses original chunk.section_id for fast lookup

### 2. Tiered Keyword Search Strategy

**Decision**: Use Semantic Kernel's native `hybrid_search()` when provider supports it. For providers without native hybrid, use a configurable sparse keyword backend: OpenSearch endpoint (`opensearch`) for production, or in-memory BM25 (`in-memory`) for local/dev.

**Rationale**:
- Native hybrid search is more efficient (single query, provider handles fusion)
- Semantic Kernel's `hybrid_search()` requires `is_full_text_indexed=True` on the content field
- OpenSearch endpoint gives durable sparse indexing that survives restarts and scales for serve/deploy
- In-memory BM25 keeps local/dev workflows simple and dependency-light

**Semantic Kernel Hybrid Search API**:

```python
# Native hybrid search via Semantic Kernel (for supported providers)
search_results = await collection.hybrid_search(
    query="search text",                    # Text query for keyword search
    vector_property_name="embedding",       # Vector field name
    additional_property_name="content",     # Full-text indexed field name
    top=10,
    # Optional: supply pre-generated embedding
    vector=query_embedding,
)

# Process results
async for result in search_results.results:
    record = result.record
    score = result.score
    # ...
```

**Requirements for hybrid_search()**:
- Vector field on the record
- String field with `is_full_text_indexed=True` (our `content` field has this)
- Provider must implement hybrid search interface

**Configuration Model (vectorstore-style)**:

```python
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KeywordIndexProvider(str, Enum):
    IN_MEMORY = "in-memory"
    OPENSEARCH = "opensearch"


class KeywordIndexConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: Literal["in-memory", "opensearch"] = Field(default="in-memory")
    endpoint: str | None = Field(default=None)
    index_name: str | None = Field(default=None)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    verify_certs: bool = Field(default=True)
    timeout_seconds: int = Field(default=10, ge=1, le=120)
```

**Implementation**:

```python
from enum import Enum
from typing import Any

class KeywordSearchStrategy(str, Enum):
    """Keyword search strategy based on provider capabilities."""
    NATIVE_HYBRID = "native_hybrid"    # Use collection.hybrid_search()
    FALLBACK_BM25 = "fallback_bm25"    # Use sparse fallback + app-level RRF


# Provider capability mapping (based on Semantic Kernel support)
NATIVE_HYBRID_PROVIDERS: set[str] = {
    "azure-ai-search",
    "weaviate",
    "qdrant",
    "mongodb",           # MongoDB Atlas via MongoDBAtlasStore
    "azure-cosmos-nosql",
}

FALLBACK_BM25_PROVIDERS: set[str] = {
    "postgres",
    "pinecone",
    "chromadb",
    "faiss",
    "in-memory",
    "sql-server",
}
# NOTE: azure-cosmos-mongo is EXCLUDED - it does NOT support hybrid search.
# MongoDB vCore uses a different API and requires MongoDBAtlasStore for native hybrid.


def get_keyword_search_strategy(provider: str) -> KeywordSearchStrategy:
    """Determine search strategy based on provider."""
    if provider in NATIVE_HYBRID_PROVIDERS:
        return KeywordSearchStrategy.NATIVE_HYBRID
    return KeywordSearchStrategy.FALLBACK_BM25


class HybridSearchExecutor:
    """Executes hybrid search using appropriate strategy for provider."""

    def __init__(
        self,
        provider: str,
        collection: Any,
        keyword_index_config: KeywordIndexConfig,
    ):
        self.provider = provider
        self.collection = collection
        self.strategy = get_keyword_search_strategy(provider)
        self.keyword_index_config = keyword_index_config
        self._bm25_index: InMemoryBM25KeywordProvider | None = None
        self._opensearch_index: OpenSearchKeywordProvider | None = None

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Execute hybrid search and return (chunk_id, score) tuples."""

        if self.strategy == KeywordSearchStrategy.NATIVE_HYBRID:
            return await self._native_hybrid_search(query, query_embedding, top_k)
        else:
            return await self._fallback_hybrid_search(query, query_embedding, top_k)

    async def _native_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Use Semantic Kernel's native hybrid_search()."""
        async with self.collection as coll:
            results = await coll.hybrid_search(
                query,
                vector=query_embedding,
                vector_property_name="embedding",
                additional_property_name="content",  # Must have is_full_text_indexed=True
                top=top_k,
            )
            return [
                (result.record.id, result.score)
                async for result in results.results
            ]

    async def _fallback_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Run vector + sparse keyword separately, fuse with RRF."""
        # Vector search
        async with self.collection as coll:
            vector_results = await coll.search(vector=query_embedding, top=top_k)
            vector_ranked = [
                (result.record.id, result.score)
                async for result in vector_results.results
            ]

        # Sparse keyword search (OpenSearch preferred, BM25 optional)
        if self.keyword_index_config.provider == "opensearch":
            bm25_ranked = self._opensearch_index.search(query, top_k)
        elif self._bm25_index:
            bm25_ranked = self._bm25_index.search(query, top_k)
        else:
            bm25_ranked = []

        # Fuse with RRF
        return reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=60)


class InMemoryBM25KeywordProvider:
    """In-memory BM25 fallback implementation using rank_bm25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: Any = None  # BM25Okapi
        self._doc_ids: list[str] = []

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Build index from (doc_id, contextualized_text) tuples."""
        from rank_bm25 import BM25Okapi
        self._doc_ids = [doc_id for doc_id, _ in documents]
        tokenized = [self._tokenize(text) for _, text in documents]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(self._doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"[a-zA-Z0-9]+", text.lower())


class OpenSearchKeywordProvider:
    """Sparse keyword provider backed by OpenSearch endpoint."""

    def __init__(self, endpoint: str, index_name: str, **kwargs: Any):
        from opensearchpy import OpenSearch

        self._client = OpenSearch(hosts=[endpoint], **kwargs)
        self._index_name = index_name

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Bulk upsert contextualized chunks into OpenSearch index."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Run BM25-style text search via OpenSearch query DSL."""
        ...
```

**Persistence**:
- **Native hybrid providers**: Full-text index persisted by the provider automatically
- **Fallback (`keyword_index.provider=opensearch`)**: Persisted in OpenSearch endpoint; recommended for serve/deploy
- **Fallback (`keyword_index.provider=in-memory`)**: Rebuilt from stored chunks on startup

**Sources**:
- [Semantic Kernel Hybrid Search](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/hybrid-search)
- [Semantic Kernel MongoDB Connector](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/out-of-the-box-connectors/mongodb-connector)
- [OpenSearch Python Client](https://opensearch.org/docs/latest/clients/python-low-level/)

### 2.1 Observability (Constitution Principle IV)

All search operations MUST emit OpenTelemetry spans per Constitution Principle IV:

```python
from opentelemetry import trace

tracer = trace.get_tracer("holodeck.hierarchical_document")

class HybridSearchExecutor:
    """Executes hybrid search with OpenTelemetry instrumentation."""

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Execute hybrid search with tracing."""
        with tracer.start_as_current_span(
            "hierarchical_document.search",
            attributes={
                "search.mode": self.strategy.value,
                "search.provider": self.provider,
                "search.top_k": top_k,
            }
        ) as span:
            results = await self._execute_search(query, query_embedding, top_k)
            span.set_attribute("search.result_count", len(results))
            return results


class InMemoryBM25KeywordProvider:
    """In-memory BM25 fallback with OpenTelemetry instrumentation."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search with tracing."""
        with tracer.start_as_current_span(
            "keyword.search.in_memory_bm25",
            attributes={
                "search.query": query,
                "bm25.query_tokens": len(self._tokenize(query)),
                "bm25.index_size": len(self._doc_ids),
            }
        ) as span:
            results = self._search_internal(query, top_k)
            span.set_attribute("search.result_count", len(results))
            return results


class OpenSearchKeywordProvider:
    """OpenSearch keyword fallback with OpenTelemetry instrumentation."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        with tracer.start_as_current_span(
            "keyword.search.opensearch",
            attributes={
                "search.query": query,
                "search.top_k": top_k,
                "keyword.provider": "opensearch",
            },
        ) as span:
            results = self._search_internal(query, top_k)
            span.set_attribute("search.result_count", len(results))
            return results
```

### 2.2 Graceful Degradation

When sparse keyword index initialization/search fails, the system MUST degrade gracefully to semantic-only search:

```python
class HybridSearchExecutor:
    def _build_keyword_index(self, documents: list[tuple[str, str]]) -> None:
        """Build sparse keyword index with graceful degradation."""
        try:
            if self.keyword_index_config.provider == "opensearch":
                self._opensearch_index = OpenSearchKeywordProvider(
                    endpoint=self.keyword_index_config.endpoint,
                    index_name=self.keyword_index_config.index_name,
                )
                self._opensearch_index.build(documents)
            else:
                self._bm25_index = InMemoryBM25KeywordProvider()
                self._bm25_index.build(documents)
        except Exception as e:
            import logging

            logging.warning(
                f"Keyword index unavailable, falling back to semantic-only: {e}"
            )
            self._opensearch_index = None
            self._bm25_index = None
```

### 3. Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.

    Formula: score(d) = Σ weight_i / (k + rank_i(d))
    """
    scores: dict[str, float] = {}
    weights = weights or [1.0] * len(ranked_lists)

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += weight / (k + rank)

    # Sort by fused score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 3.1 Exact Match Boosting (SC-002 Compliance)

To achieve 99% top-result accuracy for exact matches (SC-002), apply a score boost BEFORE RRF fusion:

```python
EXACT_MATCH_BOOST = 10.0  # Large boost to ensure top position

def apply_exact_match_boost(
    exact_results: list[tuple[str, float]],
    other_results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Boost exact matches to ensure they appear first.

    When a user queries for an exact section ID (e.g., "Section 203(a)(1)"),
    the exact match MUST appear at position 1 in results.

    Args:
        exact_results: Results from exact match index
        other_results: Results from semantic/keyword search

    Returns:
        Combined results with exact matches boosted to top
    """
    boosted = [(doc_id, score + EXACT_MATCH_BOOST) for doc_id, score in exact_results]
    return boosted + other_results


def hybrid_search_with_exact_boost(
    exact_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Hybrid search with exact match boosting for SC-002 compliance.

    Ensures exact section lookups (e.g., "Section 203(a)(1)") always
    appear at position 1.
    """
    # Apply boost to exact matches before RRF
    boosted_exact = apply_exact_match_boost(exact_results, [])

    # Run RRF on semantic + keyword
    rrf_results = reciprocal_rank_fusion(
        [semantic_results, keyword_results], k=k
    )

    # Merge boosted exact matches with RRF results
    return apply_exact_match_boost(boosted_exact, rrf_results)
```

### 4. Definition Detection and Persistence

**Decision**: Detect definitions during parsing, store definition metadata on chunks for persistence, and rebuild in-memory definition index on startup.

**Detection Strategy**:
1. **Section-based detection**: Headings containing keywords like "Definitions", "Glossary", "Terms", "Interpretation"
2. **Pattern-based detection**: Regex patterns within content:
   - `"X" means/shall mean Y` → term="X", definition_text=Y
   - `X: Y` at start of paragraph → term="X", definition_text=Y
   - `X - Y` with capitalized X → term="X", definition_text=Y

**Persistence Strategy**:
- Store `defined_term` and `defined_term_normalized` on each DocumentChunk in the vector store
- Only populated when `chunk_type == "definition"`
- On startup, rebuild definitions index by querying chunks where `defined_term != ""`

**Implementation**:

```python
class DefinitionExtractor:
    """Extract definitions from document chunks."""

    SECTION_KEYWORDS = {"definitions", "glossary", "terms", "interpretation", "defined terms"}

    # Regex patterns for common definition formats
    DEFINITION_PATTERNS = [
        # "X" means Y / "X" shall mean Y
        re.compile(r'"([^"]+)"\s+(?:means?|shall mean|is defined as)\s+(.+)', re.IGNORECASE),
        # X: Y (at paragraph start)
        re.compile(r'^([A-Z][a-zA-Z\s]+):\s+(.+)', re.MULTILINE),
        # X - Y (term with dash separator)
        re.compile(r'^([A-Z][a-zA-Z\s]+)\s+-\s+(.+)', re.MULTILINE),
    ]

    def is_definitions_section(self, heading: str) -> bool:
        """Check if heading indicates a definitions section."""
        return any(kw in heading.lower() for kw in self.SECTION_KEYWORDS)

    def extract_definitions(self, chunk: DocumentChunk) -> list[DefinitionEntry]:
        """Extract definitions from chunk content."""
        definitions = []

        # Check if in definitions section (via parent_chain)
        in_def_section = any(
            self.is_definitions_section(h) for h in chunk.parent_chain
        )

        # Apply regex patterns
        for pattern in self.DEFINITION_PATTERNS:
            for match in pattern.finditer(chunk.content):
                term = match.group(1).strip()
                definition_text = match.group(2).strip()

                definitions.append(DefinitionEntry(
                    id=f"{chunk.source_path}_def_{self._normalize(term)}",
                    source_path=chunk.source_path,
                    term=term,
                    term_normalized=self._normalize(term),
                    definition_text=definition_text,
                    source_section=chunk.section_id,
                ))

        return definitions

    def _normalize(self, term: str) -> str:
        """Normalize term for lookup: lowercase, replace spaces with underscores."""
        return re.sub(r'\s+', '_', term.lower().strip())
```

**Startup Rebuild**:

```python
async def rebuild_definitions_index(collection: VectorStoreCollection) -> dict[str, DefinitionEntry]:
    """Rebuild definitions index from stored chunks on startup."""
    definitions: dict[str, DefinitionEntry] = {}

    # Query all chunks with defined_term set
    async with collection as coll:
        # Filter for chunks where defined_term is not empty
        results = await coll.search(
            filter=lambda r: r.defined_term != "",
            top=10000,  # Get all definition chunks
        )

        async for result in results.results:
            record = result.record
            entry = DefinitionEntry(
                id=record.id,
                source_path=record.source_path,
                term=record.defined_term,
                term_normalized=record.defined_term_normalized,
                definition_text=record.content,
                source_section=record.section_id,
            )
            definitions[record.defined_term_normalized] = entry

    return definitions
```

### 5. HierarchicalDocumentChunk Record Class

Extends existing DocumentRecord pattern:

```python
@vectorstoremodel(collection_name=f"structured_docs_dim{dimensions}")
@dataclass
class HierarchicalDocumentChunk:
    id: Annotated[str, VectorStoreField("key")]
    source_path: Annotated[str, VectorStoreField("data", is_indexed=True)]
    chunk_index: Annotated[int, VectorStoreField("data", is_indexed=True)]
    content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    embedding: Annotated[list[float] | None, VectorStoreField("vector", ...)]

    # NEW: Structure-aware fields
    parent_chain: Annotated[str, VectorStoreField("data")]  # JSON array
    section_id: Annotated[str, VectorStoreField("data", is_indexed=True)]
    chunk_type: Annotated[str, VectorStoreField("data")]  # definition, requirement, etc.
    cross_references: Annotated[str, VectorStoreField("data")]  # JSON array

    # Existing
    mtime: Annotated[float, VectorStoreField("data")]
    file_type: Annotated[str, VectorStoreField("data")]
```

## Implementation Phases

### Phase 1: Foundation (P1 Stories)
1. StructuredChunker - markdown parsing, parent chain extraction
2. HierarchicalDocumentTool config model
3. Dense index integration (reuse existing patterns)
4. Basic search with source attribution

### Phase 2: Hybrid Search (P2 Stories)
1. **Tiered keyword search module** (`keyword_search.py`):
   - Provider capability detection
   - Configurable sparse fallback provider (`opensearch` or `in-memory`)
   - `OpenSearchKeywordProvider` using opensearch-py endpoint
   - `InMemoryBM25KeywordProvider` using rank_bm25
   - `NativeHybridProvider` wrapper for azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql
   - `NativeFullTextProvider` wrapper for postgres tsvector
2. Exact match index (dict-based)
3. RRF fusion logic (shared `hybrid_search.py` module)
4. Contextual embeddings
5. YAML configuration integration
6. **Refactor tool_filter to use shared keyword_search/hybrid_search modules** (see below)

### Phase 3: Advanced Features (P3 Stories)
1. Definition extraction
2. Cross-reference detection
3. Optional reranking
4. Incremental updates

## Refactoring: Unify tool_filter with Shared Modules

### Background

The existing `src/holodeck/lib/tool_filter/index.py` contains a custom BM25 implementation (lines 207-428) and RRF fusion (lines 430-501). This should be refactored to use:
1. The shared `keyword_search.py` module (with `InMemoryBM25KeywordProvider`)
2. The shared `hybrid_search.py` module for RRF fusion

Note: tool_filter is always in-memory (tools loaded from kernel), so it will always use the BM25 fallback strategy.

### Current Custom Implementation (to be replaced)

```python
# Current: Manual BM25 in tool_filter/index.py
class ToolIndex:
    _BM25_K1 = 1.5
    _BM25_B = 0.75

    def _build_bm25_index(self, documents):
        # Manual IDF calculation, tokenization
        ...

    def _bm25_score_single(self, query, tool):
        # Manual BM25 formula implementation
        ...
```

### Target Architecture

```python
# Updated tool_filter/index.py uses shared modules
from holodeck.lib.keyword_search import InMemoryBM25KeywordProvider
from holodeck.lib.hybrid_search import reciprocal_rank_fusion

class ToolIndex:
    def __init__(self):
        self.tools: dict[str, ToolMetadata] = {}
        self._keyword_provider = InMemoryBM25KeywordProvider(k1=1.5, b=0.75)

    def _build_bm25_index(self, documents: list[tuple[str, str]]) -> None:
        self._keyword_provider.build(documents)

    def _bm25_search(self, query: str) -> list[tuple[ToolMetadata, float]]:
        results = self._keyword_provider.search(query, top_k=len(self.tools))
        # Map IDs back to ToolMetadata
        return [(self.tools[id], score) for id, score in results if id in self.tools]

    async def _hybrid_search(self, query: str, embedding_service) -> list[tuple[ToolMetadata, float]]:
        semantic_results = await self._semantic_search(query, embedding_service)
        bm25_results = self._bm25_search(query)

        # Convert to ranked lists for RRF
        semantic_ranked = [(t.full_name, s) for t, s in sorted(semantic_results, key=lambda x: x[1], reverse=True)]
        bm25_ranked = [(t.full_name, s) for t, s in sorted(bm25_results, key=lambda x: x[1], reverse=True)]

        # Use shared RRF implementation
        fused = reciprocal_rank_fusion([semantic_ranked, bm25_ranked], k=60)

        # Map back to ToolMetadata
        return [(self.tools[id], score) for id, score in fused if id in self.tools]
```

### Benefits

1. **Code reuse**: Shared keyword search abstractions across tool filtering and document search
2. **Consistency**: Same algorithm behavior across features
3. **Maintainability**: OpenSearch endpoint for prod + rank_bm25 fallback for local keeps complexity explicit
4. **Performance**: rank_bm25 uses NumPy for efficient scoring
5. **Extensibility**: If tool_filter ever moves to a persistent store, it can leverage native hybrid search

### Migration Tasks

1. Create `src/holodeck/lib/keyword_search.py` with:
   - `KeywordSearchStrategy` enum (`NATIVE_HYBRID`, `FALLBACK_BM25`)
   - `NATIVE_HYBRID_PROVIDERS` set (azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql)
   - `FALLBACK_BM25_PROVIDERS` set (postgres, pinecone, chromadb, faiss, in-memory, etc.)
   - `InMemoryBM25KeywordProvider` class (using rank_bm25)
   - `OpenSearchKeywordProvider` class (using opensearch-py endpoint)
   - `HybridSearchExecutor` class (routes to native `hybrid_search()` or configured sparse fallback)
   - `get_keyword_search_strategy()` factory function
2. Create `src/holodeck/lib/hybrid_search.py` with:
   - `reciprocal_rank_fusion()` function
   - Logic to merge exact match results with hybrid results
3. Update `src/holodeck/lib/tool_filter/index.py` to use shared modules
4. Remove deprecated manual BM25 methods from ToolIndex
5. Update tests to verify equivalent behavior
6. Add `mongodb` provider support to `vector_store.py` (MongoDBAtlasStore/MongoDBAtlasCollection)

## Dependencies to Add

```toml
# pyproject.toml
[project.dependencies]
opensearch-py = "^3.1.0"  # Production sparse keyword endpoint
rank-bm25 = "^0.2.2"  # In-memory sparse fallback (dev/local)
```

**Note**: For providers without native hybrid search, `keyword_index.provider=opensearch` is recommended for serve/deploy. `rank-bm25` remains available via `keyword_index.provider=in-memory` for local development and tests.

## Test Strategy

1. **Unit tests**: Each lib module (chunker, bm25, hybrid_search, definition_extractor)
2. **Integration tests**: Full ingestion → search flow with sample documents
3. **Fixtures**: Sample legislation, technical doc, flat text for edge cases
4. **Metrics**: Verify RRF improves over semantic-only (SC-003: 20% improvement)
