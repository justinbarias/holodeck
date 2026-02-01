# Research: HierarchicalDocumentTool

**Feature**: 020-structured-document-tool
**Date**: 2026-01-29

## Research Questions Resolved

### 1. LLM-Based Contextual Embeddings (per Anthropic)

**Decision**: Use Claude (Haiku recommended) to generate 50-100 token context per chunk, prepended before embedding and BM25 indexing.

**Rationale**: Anthropic's [contextual retrieval research](https://www.anthropic.com/news/contextual-retrieval) shows:
- 35% reduction in retrieval failure rate with contextual embeddings alone
- 49% reduction when combined with contextual BM25
- 67% reduction with reranking added

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

**Example Output**:
```
Input chunk: "The Administrator shall submit a report to Congress within 90 days."

LLM-generated context: "This chunk describes the reporting requirements under Section 203
of the Environmental Protection Act, specifically the timeline for Congressional reports."

Contextualized text (for embedding + BM25):
"This chunk describes the reporting requirements under Section 203 of the Environmental
Protection Act, specifically the timeline for Congressional reports.

The Administrator shall submit a report to Congress within 90 days."
```

**Cost Analysis** (using Claude 3 Haiku):
- Input: ~800 tokens/chunk (document) + ~100 tokens (chunk) = ~900 tokens
- Output: ~75 tokens (context)
- For 100-page document (~125 chunks):
  - Input: 125 × 900 = 112.5K tokens → ~$0.028
  - Output: 125 × 75 = 9.4K tokens → ~$0.005
  - **Total: ~$0.03 per 100-page document**
- At scale: ~$1.02 per 10M tokens of source material

**Implementation Considerations**:
- Use concurrency control (default: 10 parallel calls) to avoid rate limits
- Cache generated contexts to avoid regeneration on re-index
- Store both original content and contextualized content in vector store

**Alternatives Considered**:
- Rule-based structural context (cheaper but lower quality, ~35% worse than LLM)
- No context (baseline, significantly worse retrieval)

**Preprocessing Pipeline Overview**:

```
Document (PDF/Word/etc)
        │
        ▼
┌───────────────────┐
│ markitdown        │  → Markdown
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ StructuredChunker │  → Chunks with parent_chain, section_id, chunk_type
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ LLMContextGen     │  → For each chunk: call Claude Haiku
│ (Anthropic)       │     Input: whole document + chunk
└───────────────────┘     Output: 50-100 token context
        │
        ▼
┌───────────────────┐
│ Contextualized    │  → "{context}\n\n{original_chunk}"
│ Text              │
└───────────────────┘
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Dense Index   │      │ Keyword Index │      │ Exact Index   │
│ (embeddings)  │      │ (BM25/native) │      │ (dict lookup) │
└───────────────┘      └───────────────┘      └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                    ┌───────────────────┐
                    │ Hybrid Search     │
                    │ (RRF Fusion k=60) │
                    └───────────────────┘
```

**Sources**:
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

### 2. Keyword Search Strategy (Tiered)

**Decision**: Use Semantic Kernel's native `collection.hybrid_search()` when provider supports it; fall back to `rank_bm25` otherwise

**Provider Capabilities** (based on Semantic Kernel Python support):
| Provider | Native Hybrid | Strategy |
|----------|---------------|----------|
| azure-ai-search | ✅ Yes | Use `collection.hybrid_search()` |
| weaviate | ✅ Yes | Use `collection.hybrid_search()` |
| qdrant | ✅ Yes | Use `collection.hybrid_search()` |
| mongodb | ✅ Yes | Use `collection.hybrid_search()` (MongoDB Atlas) |
| azure-cosmos-nosql | ✅ Yes | Use `collection.hybrid_search()` |
| postgres | ❌ No | Use rank_bm25 fallback + app-level RRF |
| pinecone | ❌ No | Use rank_bm25 fallback + app-level RRF |
| chromadb | ❌ No | Use rank_bm25 fallback + app-level RRF |
| faiss | ❌ No | Use rank_bm25 fallback + app-level RRF |
| in-memory | ❌ No | Use rank_bm25 fallback + app-level RRF |
| sql-server | ❌ No | Use rank_bm25 fallback + app-level RRF |

**Note**: Azure Cosmos MongoDB vCore (`azure-cosmos-mongo`) is **EXCLUDED** from both provider lists. It does NOT support hybrid search natively. MongoDB vCore uses a different API pattern and requires separate vector + text search API calls, which is not compatible with either the native hybrid strategy or the standard BM25 fallback approach. Use `mongodb` (MongoDB Atlas via MongoDBAtlasStore) for native hybrid search support.

**Rationale**:
- Native hybrid search is more efficient (single query, provider handles fusion)
- Avoids maintaining separate BM25 index when provider already does it
- Semantic Kernel's `is_full_text_indexed=True` field annotation enables native full-text on supporting providers
- `rank_bm25` provides consistent fallback for providers without native support
- **Unifies with existing codebase**: `tool_filter/index.py` has a custom BM25 implementation that will be refactored to use the shared `BM25FallbackProvider`

**Fallback Implementation** (for providers without native support):

**Implementation**:
```python
from rank_bm25 import BM25Okapi

class BM25Index:
    """Sparse index using BM25Okapi for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Document length normalization
        self._bm25: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._chunks = chunks

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        if self._bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        # Get top_k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(self._chunks[i], scores[i]) for i in top_indices if scores[i] > 0]

    def _tokenize(self, text: str) -> list[str]:
        # Simple whitespace + lowercase tokenization
        # Could be enhanced with stemming, stopword removal
        return text.lower().split()
```

**Semantic Kernel Native Hybrid Search API**:

```python
# Native hybrid search via Semantic Kernel's hybrid_search() method
# Works with: azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql

async def native_hybrid_search(
    collection,
    query: str,
    query_embedding: list[float],
    top_k: int = 10
) -> list[tuple[str, float]]:
    """Execute hybrid search using Semantic Kernel's native API.

    Requirements:
    - Vector field on the record
    - String field with is_full_text_indexed=True (e.g., 'content')
    - Provider must support hybrid search interface
    """
    async with collection as coll:
        # hybrid_search() combines vector + keyword search internally
        results = await coll.hybrid_search(
            query,                                   # Text query for keyword matching
            vector=query_embedding,                  # Pre-computed embedding (optional)
            vector_property_name="embedding",        # Name of vector field
            additional_property_name="content",      # Name of full-text indexed field
            top=top_k,
        )

        # Process results
        return [
            (result.record.id, result.score)
            async for result in results.results
        ]


# Example with Azure AI Search
from semantic_kernel.connectors.azure_ai_search import AzureAISearchCollection

collection = store.get_collection(HierarchicalDocumentRecord, "my_collection")
results = await native_hybrid_search(collection, "reporting requirements", embedding, top_k=10)


# Example with MongoDB Atlas
from semantic_kernel.connectors.mongodb import MongoDBAtlasCollection

collection = store.get_collection(HierarchicalDocumentRecord, "my_collection")
results = await native_hybrid_search(collection, "reporting requirements", embedding, top_k=10)
```

**Persistence**:
- **Native hybrid providers** (azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql): Full-text index persisted by provider automatically
- **Fallback (rank_bm25)**: In-memory only, rebuilt on startup from stored chunks

**Alternatives Considered**:
- Elasticsearch/OpenSearch (overkill for embedded use case, but would provide native hybrid)
- BM25S (faster but more complex dependencies)
- Whoosh (full-text search engine, heavier)

**Sources**:
- [rank_bm25 PyPI](https://pypi.org/project/rank-bm25/)
- [rank_bm25 GitHub](https://github.com/dorianbrown/rank_bm25)
- [Azure AI Search Hybrid](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)
- [Pinecone Hybrid Search](https://docs.pinecone.io/guides/data/understanding-hybrid-search)

---

### 3. Reciprocal Rank Fusion (RRF) Algorithm

**Decision**: Implement RRF with k=60 (industry standard), configurable weights

**When RRF is Used**:
- **Native hybrid providers** (azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql): Provider handles fusion internally via `hybrid_search()`, no app-level RRF needed for vector+keyword
- **Fallback providers** (postgres, pinecone, chromadb, faiss, in-memory, sql-server): App-level RRF to merge vector + BM25 results
- **Exact match fusion**: Always app-level RRF to merge exact match results with hybrid results (both native and fallback)

**Rationale**:
- Rank-based fusion avoids score normalization issues
- k=60 is the standard value used by Elasticsearch, Azure AI Search, OpenSearch
- Weights allow tuning semantic vs keyword emphasis
- **Shared module**: `tool_filter/index.py` already implements RRF inline; will be refactored to use shared `hybrid_search.py` module

**Formula**:
```
score(d) = Σ (weight_i / (k + rank_i(d)))
```

Where:
- k = 60 (ranking constant, dampens top-rank dominance)
- weight_i = per-modality weight (default 1.0)
- rank_i(d) = document's rank in result list i (starting from 1)

**Implementation**:
```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked results, each as [(doc_id, score), ...]
        k: Ranking constant (default 60, per Elasticsearch/Azure)
        weights: Optional weights per list (default all 1.0)

    Returns:
        Fused results sorted by RRF score descending
    """
    if not ranked_lists:
        return []

    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Effect of k value**:
- k=60: Rank 1 → 1/61 ≈ 0.0164, Rank 10 → 1/70 ≈ 0.0143
- Lower k values give more weight to top ranks
- Higher k values smooth out rank differences

**Alternatives Considered**:
- Min-max score normalization (sensitive to outliers)
- CombSUM/CombMNZ (simpler but less robust)
- Learned fusion (requires training data)

**Sources**:
- [Azure AI Search Hybrid Scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [OpenSearch RRF](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)
- [Elastic RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

---

### 4. Document Record Model Pattern (from existing codebase)

**Decision**: Extend existing `create_document_record_class()` pattern with structure fields

**Rationale**: Maintains consistency with existing VectorStoreTool implementation

**Existing Pattern** (from `src/holodeck/lib/vector_store.py`):
```python
def create_document_record_class(dimensions: int = 1536) -> type[Any]:
    @vectorstoremodel(collection_name=f"documents_dim{dimensions}")
    @dataclass
    class DynamicDocumentRecord:
        id: Annotated[str, VectorStoreField("key")]
        source_path: Annotated[str, VectorStoreField("data", is_indexed=True)]
        chunk_index: Annotated[int, VectorStoreField("data", is_indexed=True)]
        content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
        embedding: Annotated[list[float] | None, VectorStoreField("vector", ...)]
        mtime: Annotated[float, VectorStoreField("data")]
        file_type: Annotated[str, VectorStoreField("data")]
        file_size_bytes: Annotated[int, VectorStoreField("data")]
    return DynamicDocumentRecord
```

**Extended Pattern** for HierarchicalDocumentTool:
```python
def create_hierarchical_document_record_class(dimensions: int = 1536) -> type[Any]:
    @vectorstoremodel(collection_name=f"structured_docs_dim{dimensions}")
    @dataclass
    class HierarchicalDocumentRecord:
        # Existing fields
        id: Annotated[str, VectorStoreField("key")]
        source_path: Annotated[str, VectorStoreField("data", is_indexed=True)]
        chunk_index: Annotated[int, VectorStoreField("data", is_indexed=True)]
        content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
        embedding: Annotated[list[float] | None, VectorStoreField("vector", ...)]
        mtime: Annotated[float, VectorStoreField("data")]
        file_type: Annotated[str, VectorStoreField("data")]

        # NEW: Structure-aware fields
        parent_chain: Annotated[str, VectorStoreField("data")]  # JSON: ["Title", "Section 1"]
        section_id: Annotated[str, VectorStoreField("data", is_indexed=True)]  # "sec_1_2_a"
        chunk_type: Annotated[str, VectorStoreField("data")]  # "definition", "requirement", etc.
        cross_references: Annotated[str, VectorStoreField("data")]  # JSON: ["sec_2_3", "sec_4_1"]
        doc_summary: Annotated[str, VectorStoreField("data")]  # For contextual embedding
    return HierarchicalDocumentRecord
```

---

### 5. Markdown Structure Parsing

**Decision**: Use regex-based heading detection with token-aware chunking

**Rationale**:
- Markdown headings (`#`, `##`, etc.) reliably indicate structure
- Token-based max size (800 per Anthropic) ensures embedding compatibility
- Sentence boundary splitting preserves readability

**Implementation Approach**:
```python
import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

@dataclass
class StructuredChunk:
    text: str
    parent_chain: list[str]
    section_id: str
    heading_level: int
    chunk_type: str  # "content", "definition", "requirement"

class StructuredChunker:
    def __init__(self, max_tokens: int = 800):
        self.max_tokens = max_tokens

    def parse(self, markdown: str, source_path: str) -> list[StructuredChunk]:
        """Parse markdown into structure-aware chunks."""
        # 1. Extract heading hierarchy
        headings = self._extract_headings(markdown)

        # 2. Split by headings into sections
        sections = self._split_by_headings(markdown, headings)

        # 3. Chunk sections respecting max_tokens
        chunks = []
        for section in sections:
            if self._count_tokens(section.text) > self.max_tokens:
                sub_chunks = self._split_section(section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)

        return chunks

    def _extract_headings(self, markdown: str) -> list[tuple[int, str, int]]:
        """Extract (level, title, position) tuples."""
        return [(len(m.group(1)), m.group(2), m.start())
                for m in HEADING_PATTERN.finditer(markdown)]
```

---

### 6. Definition Detection Patterns

**Decision**: Heading keywords + inline pattern matching

**Rationale**: Covers both explicit definition sections and inline definitions

**Patterns**:
```python
# Heading keywords (case-insensitive)
DEFINITION_HEADINGS = {"definitions", "glossary", "terms", "terminology", "key terms"}

# Inline definition patterns
DEFINITION_PATTERNS = [
    re.compile(r'"([^"]+)"\s+means\s+', re.IGNORECASE),  # "Term" means...
    re.compile(r'"([^"]+)"\s+refers\s+to\s+', re.IGNORECASE),  # "Term" refers to...
    re.compile(r'([A-Z][a-zA-Z\s]+)\s*:\s*(?:means|is defined as)', re.IGNORECASE),
]

# Cross-reference patterns
XREF_PATTERNS = [
    re.compile(r'[Ss]ection\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)', re.IGNORECASE),  # Section 203(a)
    re.compile(r'§\s*(\d+(?:\.\d+)*)', re.IGNORECASE),  # §4.2
    re.compile(r'(?:see|as defined in|pursuant to)\s+([A-Z][a-zA-Z\s]+)', re.IGNORECASE),
]
```

**Definition Persistence Strategy**:

Definitions are persisted via the vector store chunk records and rebuilt on startup:

1. **During ingestion**:
   - When a definition is detected (via heading keywords or inline patterns), the chunk is marked with `chunk_type="definition"`
   - The `defined_term` field stores the term being defined (e.g., "Force Majeure")
   - The `defined_term_normalized` field stores the normalized lookup key (e.g., "force_majeure")
   - These fields are stored in the vector store record alongside other chunk metadata

2. **On startup/reconnection**:
   - Query the vector store for all chunks where `defined_term != ""`
   - Rebuild the in-memory definitions index from these records
   - This avoids re-running the definition detection logic on every startup

3. **Benefits**:
   - Definitions survive vector store restarts (persistent storage)
   - Fast rebuilds from indexed fields vs. re-parsing documents
   - Consistent with BM25 fallback rebuild strategy (both use stored chunk fields)

---

### 7. Supported Vector Store Providers

**Decision**: Reuse existing DatabaseConfig providers from VectorstoreTool

**Available Providers** (from `src/holodeck/models/tool.py`):
- `postgres`
- `azure-ai-search`
- `qdrant`
- `weaviate`
- `chromadb`
- `faiss`
- `azure-cosmos-mongo`
- `azure-cosmos-nosql`
- `sql-server`
- `pinecone`
- `in-memory` (default)

**Implementation**: Reuse `get_collection_factory()` pattern from `src/holodeck/lib/vector_store.py`

---

## Technology Stack Summary

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Dense Index | Semantic Kernel VectorStore | Existing pattern, multi-provider |
| Sparse Index | rank_bm25 (BM25Okapi) | Simple, efficient, well-tested |
| Exact Match | Python dict lookup | Fastest for identifier matching |
| Fusion | RRF (k=60) | Industry standard, robust |
| Chunking | Custom StructuredChunker | Structure-aware, token-limited |
| Config | Pydantic v2 | Existing pattern, validation |
| Testing | pytest | Existing pattern |

## Open Questions (Deferred)

1. **Reranking models**: Which specific reranker models to support beyond cross-encoder? (User-provided via config for now)
2. **Async BM25**: rank_bm25 is sync; may need thread pool for large corpora (>10K chunks)
3. **Multi-language support**: Should we add language detection and model selection? (Defer to future iteration)

## Resolved Decisions

1. **LLM-based contextual embedding**: ✅ RESOLVED - Using Claude Haiku (or configurable model) to generate 50-100 token context per chunk. Cost: ~$0.03 per 100-page document. Benefit: 49% retrieval improvement per Anthropic research.
2. **Keyword search strategy**: ✅ RESOLVED - Tiered approach using native `hybrid_search()` for supported providers, BM25 fallback for others.
