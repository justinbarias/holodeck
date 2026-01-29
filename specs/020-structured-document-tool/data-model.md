# Data Model: HierarchicalDocumentTool

**Feature**: 020-structured-document-tool
**Date**: 2026-01-29

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HierarchicalDocument                                │
│  (Represents an ingested document)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  source_path: str (PK)         # Absolute path to source file           │
│  doc_summary: str              # Auto-generated or first paragraph      │
│  language: str                 # Detected language (e.g., "en")         │
│  total_chunks: int             # Number of chunks after parsing         │
│  mtime: float                  # File modification time                 │
│  file_type: str                # Extension (.md, .pdf, .docx)           │
│  ingested_at: datetime         # When document was ingested             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ 1:N
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocumentChunk                                     │
│  (A parsed section with structure metadata)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  id: str (PK)                  # "{source_path}_chunk_{index}"          │
│  source_path: str (FK)         # Reference to parent document           │
│  chunk_index: int              # Sequential index within document       │
│  content: str                  # Raw chunk text (original)              │
│  contextualized_content: str   # LLM-generated context + content        │
│                                # Format: "{context}\n\n{content}"       │
│                                # Used for BOTH embedding AND BM25       │
│  embedding: list[float]        # Dense vector of contextualized_content │
│  parent_chain: list[str]       # ["Title I", "Chapter 2", "Section 3"]  │
│  section_id: str               # Normalized ID (e.g., "sec_1_2_3")      │
│  chunk_type: ChunkType         # definition | requirement | content     │
│  cross_references: list[str]   # ["sec_2_1", "sec_4_3_a"]               │
│  heading_level: int            # 1-6 for H1-H6, 0 for body              │
│  mtime: float                  # From parent document                   │
│                                                                         │
│  # Definition fields (only populated when chunk_type="definition")      │
│  defined_term: str             # "Force Majeure" (term being defined)   │
│  defined_term_normalized: str  # "force_majeure" (for index lookup)     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ N:M (via cross_references)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DefinitionEntry                                   │
│  (Extracted definition from definitions section or inline)              │
├─────────────────────────────────────────────────────────────────────────┤
│  id: str (PK)                  # "{source_path}_def_{term_normalized}"  │
│  source_path: str (FK)         # Document containing definition         │
│  term: str                     # The defined term                       │
│  term_normalized: str          # Lowercase, no spaces (for lookup)      │
│  definition_text: str          # The definition content                 │
│  source_section: str           # Section ID where defined               │
│  exceptions: list[str]         # Any noted exceptions                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        HybridIndex                                       │
│  (Container for multi-modal search indices)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  name: str (PK)                # Index name (from tool config)          │
│  dense_index: VectorCollection # Embeddings of CONTEXTUALIZED text      │
│  sparse_index: BM25Index|Native# BM25/native on CONTEXTUALIZED text     │
│  exact_index: dict[str, list]  # Section ID → chunk IDs mapping         │
│  definitions: dict[str, Def]   # term_normalized → DefinitionEntry      │
│  chunk_count: int              # Total chunks indexed                   │
│  last_updated: datetime        # Last index update time                 │
│                                                                         │
│  NOTE: Both dense and sparse indices use LLM-contextualized text,       │
│  enabling keyword search on the generated context (Anthropic approach)  │
│                                                                         │
│  PERSISTENCE STRATEGY:                                                  │
│  • dense_index: Persisted in vector store                               │
│  • sparse_index: Native providers persist automatically; BM25 fallback  │
│    rebuilt from stored contextualized_content on startup                │
│  • exact_index: Rebuilt from stored chunk.section_id on startup         │
│  • definitions: Rebuilt from chunks where defined_term != "" on startup │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        SearchResult                                      │
│  (A single result from hybrid search)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  chunk_id: str                 # Reference to DocumentChunk.id          │
│  content: str                  # Chunk content                          │
│  fused_score: float            # Combined RRF score                     │
│  semantic_score: float | None  # Dense index score (if enabled)         │
│  keyword_score: float | None   # BM25 score (if enabled)                │
│  exact_match: bool             # True if exact match found              │
│  source_path: str              # Document path                          │
│  parent_chain: list[str]       # Structural location                    │
│  section_id: str               # Section identifier                     │
│  definitions_context: list     # Relevant definitions for terms used    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pydantic Models

### Configuration Models (in `src/holodeck/models/tool.py`)

```python
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from holodeck.models.llm import LLMProvider


class SearchMode(str, Enum):
    """Search modality options."""
    SEMANTIC = "semantic"      # Dense embeddings only
    KEYWORD = "keyword"        # BM25 only
    EXACT = "exact"            # Exact match only
    HYBRID = "hybrid"          # All modalities with RRF fusion


class ChunkingStrategy(str, Enum):
    """Document chunking approach."""
    STRUCTURE = "structure"    # Structure-aware (headings, sections)
    TOKEN = "token"            # Token-based fallback


class HierarchicalDocumentToolConfig(BaseModel):
    """Configuration for HierarchicalDocumentTool in agent.yaml."""

    model_config = ConfigDict(extra="forbid")

    # Required fields
    name: str = Field(
        ...,
        pattern=r"^[0-9A-Za-z_]+$",
        description="Tool identifier (alphanumeric and underscores)"
    )
    description: str = Field(
        ...,
        description="Human-readable tool description"
    )
    type: Literal["hierarchical_document"] = Field(
        default="hierarchical_document",
        description="Tool type discriminator"
    )
    source: str = Field(
        ...,
        description="Path to file or directory to index"
    )

    # Chunking configuration
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.STRUCTURE,
        description="How to chunk documents"
    )
    max_chunk_tokens: int = Field(
        default=800,
        ge=100,
        le=2000,
        description="Maximum tokens per chunk (default 800 per Anthropic)"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Token overlap between chunks"
    )

    # Search configuration
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Which search modalities to use"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold"
    )

    # Hybrid search weights (must sum to 1.0 if provided)
    semantic_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid mode"
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid mode"
    )
    exact_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for exact match in hybrid mode"
    )

    # RRF configuration
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="RRF ranking constant (default 60)"
    )

    # Embedding configuration
    embedding_model: str | None = Field(
        default=None,
        description="Custom embedding model (defaults to provider default)"
    )
    embedding_dimensions: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Embedding dimensions (auto-detected if not specified)"
    )
    contextual_embeddings: bool = Field(
        default=True,
        description="Prepend structural context before embedding"
    )

    # LLM Context Generation (Anthropic approach)
    contextual_embeddings: bool = Field(
        default=True,
        description="Use LLM to generate context for each chunk before embedding"
    )
    context_model: LLMProvider | None = Field(
        default=None,
        description=(
            "LLM model for context generation. If None, uses agent's default model. "
            "Recommended: Claude Haiku for cost efficiency."
        )
    )
    context_max_tokens: int = Field(
        default=100,
        ge=50,
        le=200,
        description="Maximum tokens for LLM-generated context (default 100)"
    )
    context_concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Parallel LLM calls during context generation (default 10)"
    )

    # Feature toggles
    extract_definitions: bool = Field(
        default=True,
        description="Auto-extract definitions from documents"
    )
    extract_cross_references: bool = Field(
        default=True,
        description="Auto-extract cross-references"
    )
    enable_reranking: bool = Field(
        default=False,
        description="Enable optional reranking of results"
    )
    reranker_model: str | None = Field(
        default=None,
        description="Reranker model (required if enable_reranking=True)"
    )

    # Storage configuration
    database: DatabaseConfig | str | None = Field(
        default=None,
        description="Vector database configuration (in-memory if None)"
    )
    defer_loading: bool = Field(
        default=True,
        description="Defer tool loading until first use"
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source must be a non-empty path")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "HierarchicalDocumentToolConfig":
        """Warn if hybrid weights don't sum to 1.0."""
        if self.search_mode == SearchMode.HYBRID:
            total = self.semantic_weight + self.keyword_weight + self.exact_weight
            if abs(total - 1.0) > 0.01:
                import logging
                logging.warning(
                    f"Hybrid search weights sum to {total}, not 1.0. "
                    "Results may be skewed."
                )
        return self

    @model_validator(mode="after")
    def validate_reranker(self) -> "HierarchicalDocumentToolConfig":
        """Ensure reranker_model is set if enable_reranking is True."""
        if self.enable_reranking and not self.reranker_model:
            raise ValueError(
                "reranker_model is required when enable_reranking=True"
            )
        return self
```

### Runtime Models (in `src/holodeck/lib/`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    """Classification of chunk content."""
    CONTENT = "content"
    DEFINITION = "definition"
    REQUIREMENT = "requirement"
    REFERENCE = "reference"
    HEADER = "header"


@dataclass
class DocumentChunk:
    """A parsed section of a document with structure metadata."""

    id: str
    source_path: str
    chunk_index: int
    content: str
    parent_chain: list[str] = field(default_factory=list)
    section_id: str = ""
    chunk_type: ChunkType = ChunkType.CONTENT
    cross_references: list[str] = field(default_factory=list)
    heading_level: int = 0
    embedding: list[float] | None = None
    contextualized_content: str = ""
    mtime: float = 0.0

    # Definition fields (only populated when chunk_type == DEFINITION)
    defined_term: str = ""              # The term being defined (e.g., "Force Majeure")
    defined_term_normalized: str = ""   # Normalized for lookup (e.g., "force_majeure")

    def to_record_dict(self) -> dict[str, Any]:
        """Convert to dict for vector store record creation."""
        import json
        return {
            "id": self.id,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding": self.embedding,
            "parent_chain": json.dumps(self.parent_chain),
            "section_id": self.section_id,
            "chunk_type": self.chunk_type.value,
            "cross_references": json.dumps(self.cross_references),
            "mtime": self.mtime,
            "defined_term": self.defined_term,
            "defined_term_normalized": self.defined_term_normalized,
        }


@dataclass
class DefinitionEntry:
    """An extracted definition."""

    id: str
    source_path: str
    term: str
    term_normalized: str
    definition_text: str
    source_section: str
    exceptions: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single result from hybrid search."""

    chunk_id: str
    content: str
    fused_score: float
    source_path: str
    parent_chain: list[str]
    section_id: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    exact_match: bool = False
    definitions_context: list[DefinitionEntry] = field(default_factory=list)

    def format(self) -> str:
        """Format result for agent consumption."""
        location = " > ".join(self.parent_chain) if self.parent_chain else "Root"
        lines = [
            f"Score: {self.fused_score:.3f} | Source: {self.source_path}",
            f"Location: {location}",
            f"Section: {self.section_id}" if self.section_id else "",
            "",
            self.content,
        ]
        if self.definitions_context:
            lines.append("")
            lines.append("Relevant definitions:")
            for defn in self.definitions_context:
                lines.append(f"  • {defn.term}: {defn.definition_text[:100]}...")
        return "\n".join(line for line in lines if line)
```

## Vector Store Record Schema

Following the existing pattern from `src/holodeck/lib/vector_store.py`:

```python
from dataclasses import dataclass, field
from typing import Annotated, Any, cast
from uuid import uuid4

from semantic_kernel.data import (
    DistanceFunction,
    VectorStoreField,
    vectorstoremodel,
)


def create_hierarchical_document_record_class(dimensions: int = 1536) -> type[Any]:
    """Create a HierarchicalDocumentRecord class with specified embedding dimensions.

    Follows the pattern from create_document_record_class() in vector_store.py.
    """
    if not 1 <= dimensions <= 10000:
        raise ValueError(f"dimensions must be 1-10000, got {dimensions}")

    @vectorstoremodel(collection_name=f"structured_docs_dim{dimensions}")
    @dataclass
    class HierarchicalDocumentRecord:
        """Vector store record for structured document chunks."""

        # Key field
        id: Annotated[str, VectorStoreField("key")] = field(
            default_factory=lambda: str(uuid4())
        )

        # Indexed data fields
        source_path: Annotated[str, VectorStoreField("data", is_indexed=True)] = field(
            default=""
        )
        chunk_index: Annotated[int, VectorStoreField("data", is_indexed=True)] = field(
            default=0
        )
        section_id: Annotated[str, VectorStoreField("data", is_indexed=True)] = field(
            default=""
        )

        # Full-text indexed content
        content: Annotated[
            str, VectorStoreField("data", is_full_text_indexed=True)
        ] = field(default="")

        # Vector embedding
        embedding: Annotated[
            list[float] | None,
            VectorStoreField(
                "vector",
                dimensions=dimensions,
                distance_function=DistanceFunction.COSINE_SIMILARITY,
            ),
        ] = field(default=None)

        # Structure metadata (stored as JSON strings)
        parent_chain: Annotated[str, VectorStoreField("data")] = field(default="[]")
        chunk_type: Annotated[str, VectorStoreField("data")] = field(default="content")
        cross_references: Annotated[str, VectorStoreField("data")] = field(default="[]")
        doc_summary: Annotated[str, VectorStoreField("data")] = field(default="")

        # File metadata
        mtime: Annotated[float, VectorStoreField("data")] = field(default=0.0)
        file_type: Annotated[str, VectorStoreField("data")] = field(default="")

        # Definition fields (only populated when chunk_type="definition")
        # Used to rebuild the definitions index on startup
        defined_term: Annotated[str, VectorStoreField("data", is_indexed=True)] = field(
            default=""
        )
        defined_term_normalized: Annotated[str, VectorStoreField("data", is_indexed=True)] = field(
            default=""
        )

    return cast(type[Any], HierarchicalDocumentRecord)


# Pre-create default instance for common embedding dimension
HierarchicalDocumentRecord = create_hierarchical_document_record_class(1536)
```

## State Transitions

### Document Lifecycle (Preprocessing Pipeline)

The preprocessing pipeline follows Anthropic's contextual retrieval approach:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT PREPROCESSING                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐                                                            │
│  │   PENDING    │  File discovered, not yet processed                        │
│  └──────┬───────┘                                                            │
│         │ convert() [markitdown]                                             │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │  CONVERTING  │  PDF/Word/HTML → Markdown                                  │
│  └──────┬───────┘                                                            │
│         │ parse() [StructuredChunker]                                        │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │   PARSING    │  Extract headings, parent_chain, definitions, cross-refs   │
│  └──────┬───────┘                                                            │
│         │ chunk() [max 800 tokens, structure-aware]                          │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │   CHUNKING   │  Split at structural boundaries, preserve metadata         │
│  └──────┬───────┘                                                            │
│         │ contextualize() [LLM call per chunk - Anthropic approach]          │
│         ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     CONTEXTUALIZING (LLM)                               │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  For each chunk:                                                        │ │
│  │    • Send whole document + chunk to Claude Haiku                        │ │
│  │    • Generate 50-100 token context                                      │ │
│  │    • Prepend: "{context}\n\n{chunk.content}"                            │ │
│  │                                                                         │ │
│  │  Prompt:                                                                │ │
│  │  <document>{WHOLE_DOC}</document>                                       │ │
│  │  <chunk>{CHUNK}</chunk>                                                 │ │
│  │  Please give a short succinct context to situate this chunk...          │ │
│  │                                                                         │ │
│  │  Concurrency: 10 parallel calls | Cost: ~$0.03 per 100 pages            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         │                                                                    │
│         │ embed() [contextualized text]                                      │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │  EMBEDDING   │  Generate dense vectors from contextualized text           │
│  └──────┬───────┘                                                            │
│         │ index() [dense + keyword + exact]                                  │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │   INDEXING   │  Build all indices from contextualized chunks              │
│  │              │  • Dense: embed contextualized text                        │
│  │              │  • Keyword: BM25/native on contextualized text             │
│  │              │  • Exact: section_id → chunk_id mapping                    │
│  └──────┬───────┘                                                            │
│         │ persist()                                                          │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │   INDEXED    │  All indices populated, ready for search                   │
│  └──────┬───────┘                                                            │
│         │ needs_update?                                                      │
│         ▼                                                                    │
│  ┌──────────────┴──────────────┐                                            │
│  │                             │                                             │
│  ▼                             ▼                                             │
│  ┌───────────┐          ┌─────────────┐                                     │
│  │   STALE   │          │   CURRENT   │                                     │
│  │ (mtime    │          │ (up to date)│                                     │
│  │  changed) │          └─────────────┘                                     │
│  └─────┬─────┘                                                              │
│        │ reindex()                                                           │
│        └─────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Through Preprocessing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA TRANSFORMATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT                         STAGE                      OUTPUT             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  document.pdf            →  markitdown  →           document.md              │
│  (binary)                   (convert)               (markdown text)          │
│                                                                              │
│  document.md             →  StructuredChunker  →    List[DocumentChunk]      │
│  (markdown text)            (parse + chunk)         with parent_chain,       │
│                                                     section_id, chunk_type   │
│                                                                              │
│  DocumentChunk           →  LLMContextGenerator →   DocumentChunk            │
│  .content = "The Admin      (contextualize)         .content = "The Admin    │
│   shall submit..."                                   shall submit..."        │
│                                                     .contextualized_content  │
│                                                      = "This chunk describes │
│                                                      reporting requirements  │
│                                                      under Section 203...\n\n│
│                                                      The Admin shall submit" │
│                                                                              │
│  DocumentChunk           →  EmbeddingService  →     DocumentChunk            │
│  .contextualized_content    (embed)                 .embedding = [0.1, ...]  │
│                                                                              │
│  List[DocumentChunk]     →  IndexBuilder  →         HybridIndex              │
│  (with embeddings)          (index)                 .dense_index             │
│                                                     .keyword_index           │
│                                                     .exact_index             │
│                                                     .definitions             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Search Flow

```
Query Input
    │
    ▼
┌───────────────────────────────────────┐
│           Query Analysis              │
│  • Detect exact match patterns        │
│  • Tokenize for BM25                  │
│  • Generate query embedding           │
└─────────────────┬─────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Semantic│ │ Keyword │ │  Exact  │
│ Search  │ │ Search  │ │  Match  │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 ▼
         ┌─────────────┐
         │ RRF Fusion  │
         │  (k=60)     │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │  Reranking  │  (optional)
         │             │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │  Enrich w/  │
         │ Definitions │
         └──────┬──────┘
                │
                ▼
         SearchResult[]
```

## Validation Rules

### DocumentChunk Validation

| Field | Rule | Error |
|-------|------|-------|
| id | Must match `{source_path}_chunk_{index}` | InvalidChunkId |
| content | Non-empty after whitespace strip | EmptyChunkContent |
| chunk_index | >= 0 | InvalidChunkIndex |
| parent_chain | Valid JSON array | InvalidParentChain |
| embedding | Correct dimensions if present | DimensionMismatch |

### Configuration Validation

| Field | Rule | Error |
|-------|------|-------|
| source | Non-empty path | EmptySource |
| max_chunk_tokens | 100-2000 | InvalidChunkSize |
| top_k | 1-100 | InvalidTopK |
| weights | Sum to ~1.0 (warning) | WeightsSumWarning |
| reranker_model | Required if enable_reranking | MissingReranker |
