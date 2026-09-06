# Data Model: Unstructured Vector Ingestion and Search

**Feature**: 008-unstructured-vector-ingestion-search
**Date**: 2025-11-23
**Purpose**: Define entities, relationships, and validation rules

## Entity Overview

```
┌─────────────────────┐
│  VectorStoreConfig  │  (YAML configuration)
│  - type: vectorstore│
│  - source           │
│  - embedding_model  │
│  - database         │
│  - top_k            │
│  - min_similarity   │
└──────────┬──────────┘
           │ configures
           ▼
┌─────────────────────┐      ingests       ┌──────────────────┐
│  VectorStoreTool    │◄──────────────────│  SourceFile      │
│  - config           │                    │  - path          │
│  - vector_store     │                    │  - mtime         │
│  - embedding_svc    │                    │  - content       │
└──────────┬──────────┘                    └──────────────────┘
           │ stores/retrieves
           ▼
┌─────────────────────┐      contains      ┌──────────────────┐
│  VectorStore        │◄──────────────────│  DocumentRecord  │
│  - backend (Redis/  │                    │  - id            │
│    in-memory)       │                    │  - content       │
│  - collection_name  │                    │  - source_path   │
└─────────────────────┘                    │  - chunk_index   │
                                           │  - mtime         │
                                           │  - embedding     │
                                           └──────────────────┘
           │ executes
           ▼
┌─────────────────────┐      returns       ┌──────────────────┐
│  SearchQuery        │◄──────────────────│  QueryResult     │
│  - query_text       │                    │  - content       │
│  - top_k            │                    │  - score         │
│  - min_score        │                    │  - source_path   │
└─────────────────────┘                    │  - chunk_index   │
                                           └──────────────────┘
```

## Core Entities

### 1. VectorStoreConfig (Configuration Model)

**Purpose**: YAML-based configuration for vectorstore tools

**Source**: Defined in agent.yaml, parsed by ConfigLoader

**Fields**:
```python
@dataclass
class VectorStoreConfig:
    """Configuration for vectorstore tool.

    Defined in agent.yaml under tools section:
    tools:
      - type: vectorstore
        name: knowledge_base
        source: data/docs/  # or data/faq.md
        embedding_model: text-embedding-3-small  # optional
        database:  # optional
          provider: redis
          connection_string: redis://localhost:6379
          index_name: holodeck_vectors
        top_k: 5  # optional, default 5
        min_similarity_score: 0.7  # optional, default None
    """
    type: Literal["vectorstore"]
    name: str
    source: str  # File path or directory path
    embedding_model: str | None = None  # Optional: defaults to provider default
    database: DatabaseConfig | None = None  # Optional: defaults to in-memory
    top_k: int = 5  # Number of results to return
    min_similarity_score: float | None = None  # Minimum relevance threshold

@dataclass
class DatabaseConfig:
    """Vector database connection configuration."""
    provider: Literal["redis"]  # Future: "postgres", "azure_search"
    connection_string: str  # Redis: "redis://host:port" or "rediss://..." for SSL
    index_name: str = "holodeck_vectors"  # Collection/index name
    vector_algorithm: Literal["HNSW", "FLAT"] = "HNSW"
    distance_metric: Literal["COSINE", "L2", "IP"] = "COSINE"
```

**Validation Rules**:
- `type` MUST be `"vectorstore"`
- `source` MUST be non-empty string pointing to valid file or directory path
- `embedding_model` if specified MUST be supported by configured LLM provider
- `top_k` MUST be positive integer (default: 5)
- `min_similarity_score` if specified MUST be float between 0.0 and 1.0
- `database.provider` currently only supports `"redis"` (in-memory fallback automatic)
- `database.connection_string` MUST be valid Redis URL format if provider is `"redis"`

**State Transitions**: N/A (immutable configuration)

**Related Requirements**: FR-001, FR-002, FR-006, FR-008, FR-010

---

### 2. VectorStoreTool (Runtime Tool Instance)

**Purpose**: Executable tool instance that performs semantic search operations

**Lifecycle**:
1. **Initialization**: Created from VectorStoreConfig during agent startup
2. **Ingestion**: Loads source files, chunks content, generates embeddings, stores in vector database
3. **Query**: Accepts search queries, generates query embedding, retrieves top-k results
4. **Cleanup**: Releases resources on agent shutdown

**Fields**:
```python
class VectorStoreTool:
    """Vectorstore tool for semantic search over unstructured data."""

    config: VectorStoreConfig  # Configuration from YAML
    vector_store: VectorStore  # Backend storage (Redis or in-memory)
    embedding_service: TextEmbedding  # Embedding generation service
    file_processor: FileProcessor  # Existing file-to-markdown converter
    text_chunker: RecursiveCharacterTextSplitter  # Text chunking

    # State
    is_initialized: bool = False
    last_ingest_time: datetime | None = None
    document_count: int = 0
```

**Methods**:
```python
async def initialize(self, force_ingest: bool = False) -> None:
    """Initialize tool and ingest source files if needed."""

async def search(self, query: str) -> list[QueryResult]:
    """Execute semantic search and return results."""

async def _ingest_source(self, force: bool) -> None:
    """Ingest files from configured source."""

async def _process_file(self, file_path: Path) -> list[DocumentRecord]:
    """Process single file: convert to markdown, chunk, embed."""

async def _needs_reingest(self, file_path: Path) -> bool:
    """Check if file needs re-ingestion based on mtime."""
```

**Validation Rules**:
- Tool MUST call `initialize()` before first `search()` call
- `search()` query MUST be non-empty string
- Source files MUST exist at initialization time (FileNotFoundError if missing)

**State Transitions**:
```
[Created] --initialize()--> [Initialized] --search()--> [Ready]
                                    ↑            │
                                    └────────────┘
                                   (continuous queries)
```

**Related Requirements**: FR-001, FR-003, FR-004, FR-005, FR-006, FR-012

---

### 3. DocumentRecord (Vector Store Record)

**Purpose**: Represents a single document chunk with embedding stored in vector database

**Storage**: Persisted in Redis (production) or in-memory (development)

**Fields**:
```python
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data import (
    VectorStoreRecordKeyField,
    VectorStoreRecordDataField,
    VectorStoreRecordVectorField
)

@dataclass
class DocumentRecord:
    """Vector store record for document chunks.

    Stored in Redis or in-memory vector store. Each document file is split
    into multiple chunks, each with its own embedding.
    """

    # Primary key: {source_path}_chunk_{index}
    id: Annotated[str, VectorStoreRecordKeyField()] = field(default="")

    # Chunk content (markdown text)
    content: Annotated[str, VectorStoreRecordDataField()] = field(default="")

    # Original source file path
    source_path: Annotated[str, VectorStoreRecordDataField()] = field(default="")

    # Chunk index within document (0-indexed)
    chunk_index: Annotated[int, VectorStoreRecordDataField()] = field(default=0)

    # File modification time (for change detection)
    mtime: Annotated[float, VectorStoreRecordDataField()] = field(default=0.0)

    # Embedding vector (dimensions vary by model)
    embedding: Annotated[list[float], VectorStoreRecordVectorField(
        dimensions=1536,  # text-embedding-3-small default
        distance_function="cosine"
    )] = field(default_factory=list)

    # Optional metadata
    file_type: Annotated[str, VectorStoreRecordDataField()] = field(default="")
    file_size_bytes: Annotated[int, VectorStoreRecordDataField()] = field(default=0)
```

**Validation Rules**:
- `id` MUST be unique across all records in collection
- `id` MUST follow format: `{source_path}_chunk_{chunk_index}`
- `content` MUST be non-empty string (empty chunks skipped during ingestion)
- `source_path` MUST be valid file path string
- `chunk_index` MUST be non-negative integer
- `mtime` MUST be valid Unix timestamp (float)
- `embedding` length MUST match configured embedding model dimensions

**Relationships**:
- Multiple DocumentRecords → One SourceFile (one-to-many: file split into chunks)
- DocumentRecord → VectorStore collection (stored in)

**Related Requirements**: FR-004, FR-005, FR-007, FR-021

---

### 4. SourceFile (Input Entity)

**Purpose**: Represents a source file to be ingested (file or directory entry)

**Lifecycle**: Temporary entity during ingestion, not persisted

**Fields**:
```python
@dataclass
class SourceFile:
    """Source file to be ingested into vector store."""

    path: Path  # Absolute file path
    content: str = ""  # Markdown content (populated via FileProcessor)
    mtime: float = 0.0  # File modification time
    size_bytes: int = 0  # File size
    file_type: str = ""  # File extension (.txt, .md, .pdf, etc.)
    chunks: list[str] = field(default_factory=list)  # Text chunks (populated by chunker)
```

**Validation Rules**:
- `path` MUST exist and be readable (FR-020)
- `path` MUST have supported file extension: .txt, .md, .pdf, .csv, .json (FR-003)
- `size_bytes` SHOULD be ≤ 100MB (warning logged if exceeded, FR-016)
- `content` MUST be non-empty after processing (empty files skipped with warning, FR-018)

**State Transitions**:
```
[Discovered] --FileProcessor--> [Converted to Markdown]
      ▼
[Chunked] --EmbeddingService--> [Embedded] --VectorStore--> [Stored]
```

**Related Requirements**: FR-003, FR-004, FR-014, FR-015, FR-016, FR-018, FR-019, FR-020

---

### 5. QueryResult (Output Entity)

**Purpose**: Represents a single search result returned from vector store query

**Lifecycle**: Created during search operation, returned to agent LLM

**Fields**:
```python
@dataclass
class QueryResult:
    """Search result from vector store query."""

    content: str  # Matched document chunk content
    score: float  # Relevance/similarity score (0.0-1.0, higher is better)
    source_path: str  # Original source file path
    chunk_index: int  # Chunk index within source file
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional metadata
```

**Validation Rules**:
- `score` MUST be float between 0.0 and 1.0
- `score` MUST be ≥ `min_similarity_score` if configured (FR-006, FR-013)
- Results MUST be ordered by descending score (FR-013)
- Result count MUST be ≤ `top_k` (FR-006, FR-013)

**Relationships**:
- QueryResult ← DocumentRecord (derived from)
- Multiple QueryResults → One SearchQuery (returned from)

**Related Requirements**: FR-006, FR-007, FR-013

---

### 6. SearchQuery (Input Entity)

**Purpose**: Represents a semantic search query from the agent

**Lifecycle**: Created when agent calls vectorstore tool, discarded after results returned

**Fields**:
```python
@dataclass
class SearchQuery:
    """Semantic search query."""

    query_text: str  # Natural language query
    top_k: int = 5  # Number of results to return
    min_similarity_score: float | None = None  # Minimum score threshold

    # Internal (populated during execution)
    query_embedding: list[float] = field(default_factory=list)
```

**Validation Rules**:
- `query_text` MUST be non-empty string
- `top_k` MUST be positive integer (default from config or 5)
- `min_similarity_score` if specified MUST be float between 0.0 and 1.0

**Related Requirements**: FR-006, FR-013

---

### 7. VectorStore (Storage Abstraction)

**Purpose**: Abstraction layer over Redis and in-memory vector storage backends

**Implementation**: Wrapper around Semantic Kernel's VectorStore connectors

**Fields**:
```python
class VectorStore:
    """Vector store abstraction (Redis or in-memory)."""

    backend: RedisStore | VolatileVectorStore  # Semantic Kernel connector
    collection: VectorStoreRecordCollection[str, DocumentRecord]
    collection_name: str
```

**Methods**:
```python
async def upsert(self, record: DocumentRecord) -> None:
    """Insert or update document record."""

async def get(self, record_id: str) -> DocumentRecord | None:
    """Retrieve record by ID."""

async def delete(self, record_id: str) -> None:
    """Delete record by ID."""

async def search(self, query_embedding: list[float], top_k: int, min_score: float | None) -> list[QueryResult]:
    """Perform similarity search."""

async def delete_by_source(self, source_path: str) -> None:
    """Delete all records for a given source file."""
```

**Backends**:
- **Redis** (production): Persistent, multi-process, scalable (10k+ documents)
- **In-memory** (development): Ephemeral, single-process, fast for small datasets (<1000 documents)

**Related Requirements**: FR-010, FR-011, FR-017

---

## Entity Relationships

### Cardinality

```
VectorStoreConfig (1) ─── creates ───> (1) VectorStoreTool
VectorStoreTool (1) ─── uses ───> (1) VectorStore
VectorStoreTool (1) ─── processes ───> (0..*) SourceFile
SourceFile (1) ─── generates ───> (1..*) DocumentRecord
VectorStore (1) ─── contains ───> (0..*) DocumentRecord
SearchQuery (1) ─── returns ───> (0..top_k) QueryResult
QueryResult (1) ─── derived_from ───> (1) DocumentRecord
```

### Key Constraints

1. **Source Path Uniqueness**: Multiple DocumentRecords can share the same `source_path` (one per chunk)
2. **ID Uniqueness**: Each DocumentRecord MUST have unique `id` across entire collection
3. **Embedding Dimensions**: All DocumentRecords in a collection MUST have same embedding dimensions
4. **Search Ordering**: QueryResults MUST be ordered by descending score (FR-013)
5. **Modification Tracking**: DocumentRecord `mtime` MUST match source file `mtime` or trigger re-ingestion

---

## Data Flow Diagrams

### Ingestion Flow

```
┌─────────────┐
│ agent.yaml  │
│   source:   │
│   data/docs/│
└──────┬──────┘
       │ parse config
       ▼
┌─────────────────┐
│ VectorStoreTool │
│  .initialize()  │
└────────┬────────┘
         │ discover files
         ▼
┌──────────────────────┐
│ Directory Scanner    │
│ (*.txt, *.md, *.pdf) │
└─────────┬────────────┘
          │ for each file
          ▼
┌──────────────────────┐
│ FileProcessor        │
│ (convert to markdown)│
└─────────┬────────────┘
          │ markdown content
          ▼
┌──────────────────────┐
│ TextChunker          │
│ (512 tokens, overlap)│
└─────────┬────────────┘
          │ chunks
          ▼
┌──────────────────────┐
│ EmbeddingService     │
│ (batch embed chunks) │
└─────────┬────────────┘
          │ embeddings
          ▼
┌──────────────────────┐
│ VectorStore          │
│ .upsert(DocumentRec) │
└──────────────────────┘
```

### Query Flow

```
┌─────────────┐
│ Agent LLM   │
│ calls tool: │
│ search(q)   │
└──────┬──────┘
       │ query text
       ▼
┌─────────────────┐
│ VectorStoreTool │
│  .search(query) │
└────────┬────────┘
         │ embed query
         ▼
┌──────────────────────┐
│ EmbeddingService     │
│ (generate embedding) │
└─────────┬────────────┘
          │ query_embedding
          ▼
┌──────────────────────┐
│ VectorStore          │
│ .search(embedding,   │
│   top_k=5,           │
│   min_score=0.7)     │
└─────────┬────────────┘
          │ ranked results
          ▼
┌──────────────────────┐
│ QueryResult[]        │
│ (content + score +   │
│  source_path)        │
└─────────┬────────────┘
          │ return to LLM
          ▼
┌─────────────┐
│ Agent LLM   │
│ uses context│
└─────────────┘
```

---

## Validation Matrix

| Entity | Field | Validation Rule | Error Type | Related FR |
|--------|-------|----------------|------------|-----------|
| VectorStoreConfig | type | Must be "vectorstore" | ConfigError | FR-001 |
| VectorStoreConfig | source | Non-empty, valid path | ConfigError | FR-002 |
| VectorStoreConfig | top_k | Positive integer | ConfigError | FR-006 |
| VectorStoreConfig | min_similarity_score | 0.0 ≤ x ≤ 1.0 or None | ConfigError | FR-006 |
| SourceFile | path | Must exist, readable | FileNotFoundError | FR-020 |
| SourceFile | file_type | In [.txt, .md, .pdf, .csv, .json] | ValidationError | FR-003 |
| SourceFile | size_bytes | Warn if > 100MB | Warning (logged) | FR-016 |
| SourceFile | content | Non-empty after processing | Warning (skip file) | FR-018 |
| DocumentRecord | id | Unique, follows format | RuntimeError | Internal |
| DocumentRecord | content | Non-empty | ValidationError | FR-004 |
| DocumentRecord | embedding | Length matches model dims | ValidationError | FR-005 |
| SearchQuery | query_text | Non-empty string | ValueError | FR-006 |
| QueryResult | score | 0.0 ≤ x ≤ 1.0 | RuntimeError | FR-013 |

---

## Schema Evolution Strategy

**Embedding Dimension Changes**:
- If `embedding_model` changes in config → User MUST run `--force-ingest`
- Tool detects dimension mismatch and raises clear error with remediation steps

**New Fields**:
- Adding optional fields to DocumentRecord → Backward compatible (default values)
- Adding required fields → Requires migration or `--force-ingest`

**Backend Migration**:
- Redis → In-memory: No migration needed (re-ingest on startup)
- In-memory → Redis: Automatic on first startup (ephemeral data lost)

**Version Tracking** (Future):
- Store schema version in vector store metadata
- Detect version mismatches and trigger migrations

---

## Summary

This data model defines 7 core entities supporting the vectorstore tool feature:

1. **VectorStoreConfig**: YAML configuration (immutable)
2. **VectorStoreTool**: Runtime tool instance (stateful)
3. **DocumentRecord**: Persistent vector store record (stored in Redis/in-memory)
4. **SourceFile**: Temporary ingestion entity (ephemeral)
5. **QueryResult**: Search result output (ephemeral)
6. **SearchQuery**: Query input (ephemeral)
7. **VectorStore**: Storage abstraction (stateful)

All entities follow validation rules from functional requirements (FR-001 through FR-023) and support the three user stories (P1-P3) defined in the feature specification.
