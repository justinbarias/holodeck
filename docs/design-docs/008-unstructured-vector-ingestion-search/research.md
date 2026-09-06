# Research: Unstructured Vector Ingestion and Search

**Feature**: 008-unstructured-vector-ingestion-search
**Date**: 2025-11-23
**Purpose**: Resolve technical unknowns and establish implementation patterns

## Research Areas

### 1. Semantic Kernel Vector Store Abstractions

**Question**: How does Semantic Kernel structure vector store operations, and what abstractions should we use?

**Decision**: Use Semantic Kernel's `VectorStore` and `VectorStoreRecordCollection` abstractions

**Rationale**:
- Semantic Kernel provides unified abstractions (`VectorStore`, `VectorStoreRecordCollection`) that work across Redis, Azure AI Search, Qdrant, Weaviate, and in-memory stores
- The `VectorStoreRecordCollection` interface provides standard methods: `upsert`, `get`, `delete`, `search` with similarity scoring
- Data classes use `@dataclass` with `VectorStoreRecordKeyField`, `VectorStoreRecordDataField`, and `VectorStoreRecordVectorField` decorators
- Collections are strongly typed and handle serialization/deserialization automatically
- This abstraction enables switching between Redis (production) and in-memory (development/testing) without code changes

**Alternatives considered**:
- Direct Redis vector client: Rejected - no abstraction, hard to test, vendor lock-in
- LangChain vector stores: Rejected - introduces additional dependency when Semantic Kernel already used
- Custom abstraction layer: Rejected - reinventing wheel, Semantic Kernel provides battle-tested implementation

**Implementation approach**:
```python
from semantic_kernel.connectors.memory.redis import RedisStore
from semantic_kernel.connectors.memory.volatile import VolatileVectorStore
from semantic_kernel.data import VectorStoreRecordCollection

# Define document record schema
@dataclass
class DocumentRecord:
    id: Annotated[str, VectorStoreRecordKeyField()]
    content: Annotated[str, VectorStoreRecordDataField()]
    source_path: Annotated[str, VectorStoreRecordDataField()]
    chunk_index: Annotated[int, VectorStoreRecordDataField()]
    embedding: Annotated[list[float], VectorStoreRecordVectorField(dimensions=1536)]

# Use Redis or in-memory based on config
store = RedisStore(...) if config else VolatileVectorStore()
collection = VectorStoreRecordCollection[str, DocumentRecord](store, "documents")
```

**References**:
- Semantic Kernel Python docs: Vector Store Abstractions
- Example: `semantic_kernel/connectors/memory/redis/redis_store.py`

---

### 2. Text Chunking with Semantic Kernel

**Question**: How should we chunk documents for embedding, and what chunking strategies does Semantic Kernel provide?

**Decision**: Use `RecursiveCharacterTextSplitter` from Semantic Kernel with token-based chunking

**Rationale**:
- Semantic Kernel provides `RecursiveCharacterTextSplitter` which splits text hierarchically (\n\n → \n → space → char) to preserve semantic units
- Token-based chunking (via `from_tiktoken_encoder`) ensures chunks don't exceed model token limits
- Supports overlap between chunks to preserve context across boundaries
- Handles markdown structure intelligently by prioritizing section breaks
- Default chunk size: 512 tokens with 50 token overlap (balances context vs. granularity)

**Alternatives considered**:
- Fixed character chunking: Rejected - breaks mid-sentence, poor semantic preservation
- Sentence-based chunking: Rejected - varying sizes can exceed token limits
- Custom markdown-aware chunker: Rejected - Semantic Kernel's recursive splitter handles markdown well

**Implementation approach**:
```python
from semantic_kernel.text import RecursiveCharacterTextSplitter

# Create chunker with token-based sizing
chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4, text-embedding-3-*
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Hierarchical splitting
)

# Chunk markdown content from FileProcessor
chunks = chunker.split_text(markdown_content)
```

**References**:
- Semantic Kernel text splitting: `semantic_kernel/text/text_splitter.py`
- Token encoding: tiktoken library (cl100k_base for OpenAI embeddings)

---

### 3. Embedding Generation Best Practices

**Question**: How should we generate embeddings efficiently, handle different embedding models, and manage rate limits?

**Decision**: Use Semantic Kernel's `TextEmbedding` service with model-specific configuration and batch processing

**Rationale**:
- Semantic Kernel's `TextEmbedding` abstraction supports OpenAI, Azure OpenAI, Hugging Face, and custom models
- Provider-specific defaults align with LLM provider (e.g., OpenAI LLM → `text-embedding-3-small` by default)
- Batch embedding generation (10-100 texts per API call) reduces latency and API costs
- Built-in retry logic with exponential backoff handles transient failures
- Embedding dimensions vary by model (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)

**Alternatives considered**:
- Direct OpenAI API calls: Rejected - no abstraction, harder to test, manual retry logic
- Pre-compute all embeddings: Rejected - inflexible, requires re-embedding on model change
- Stream embeddings one-by-one: Rejected - excessive API calls, slow for large corpora

**Implementation approach**:
```python
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

# Initialize embedding service based on config
embedding_service = OpenAITextEmbedding(
    ai_model_id=config.embedding_model or "text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Batch embed chunks (10-100 at a time)
batch_size = 20
embeddings = []
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_embeddings = await embedding_service.generate_embeddings(batch)
    embeddings.extend(batch_embeddings)
```

**Default embedding models by provider**:
- OpenAI: `text-embedding-3-small` (1536 dims, $0.02/1M tokens)
- Azure OpenAI: `text-embedding-ada-002` (1536 dims)
- Anthropic: No native embeddings - use OpenAI or Voyage AI
- Custom: User-specified model

**References**:
- Semantic Kernel embeddings: `semantic_kernel/connectors/ai/embeddings/`
- OpenAI embedding models: https://platform.openai.com/docs/guides/embeddings

---

### 4. Redis Vector Store Integration

**Question**: How should we configure Redis for vector storage, and what are the connection/schema requirements?

**Decision**: Use Semantic Kernel's `RedisStore` connector with RediSearch module for vector similarity search

**Rationale**:
- RediSearch (Redis module) provides native vector search with HNSW and FLAT index algorithms
- Semantic Kernel's `RedisStore` handles schema creation, indexing, and search queries automatically
- Connection via redis-py with connection pooling for performance
- Supports both Redis Cloud and self-hosted Redis Stack (includes RediSearch module)
- Index auto-creation on first collection access (no manual schema management)

**Alternatives considered**:
- Redis JSON + manual vector search: Rejected - requires manual index management
- Separate vector database (Pinecone, Weaviate): Rejected - adds deployment complexity
- PostgreSQL with pgvector: Rejected - not supported by Semantic Kernel connectors

**Implementation approach**:
```python
from semantic_kernel.connectors.memory.redis import RedisStore
import redis.asyncio as redis

# Connection configuration from YAML
redis_config = VectorStoreConfig(
    provider="redis",
    connection_string="redis://localhost:6379",  # or Redis Cloud URL
    index_name="holodeck_vectors",
    vector_algorithm="HNSW",  # or "FLAT"
    distance_metric="COSINE"  # or "L2", "IP"
)

# Initialize Redis store
redis_client = await redis.from_url(redis_config.connection_string)
store = RedisStore(redis_client)

# Collection auto-creates index on first access
collection = VectorStoreRecordCollection[str, DocumentRecord](
    store,
    collection_name=redis_config.index_name
)
```

**Configuration defaults**:
- Index algorithm: HNSW (Hierarchical Navigable Small World) - best for large datasets
- Distance metric: COSINE - standard for text embeddings
- Index name: `holodeck_vectors_{agent_name}`

**References**:
- Redis vector search: https://redis.io/docs/interact/search-and-query/search/vectors/
- Semantic Kernel Redis connector: `semantic_kernel/connectors/memory/redis/`

---

### 5. File Modification Tracking Strategy

**Question**: How should we track file modifications to avoid unnecessary re-ingestion and re-embedding?

**Decision**: Use file modification timestamps (mtime) with metadata storage in vector database

**Rationale**:
- File `mtime` (modification time) is reliable cross-platform indicator of changes
- Store `mtime` in vector record metadata alongside embeddings
- On agent startup, compare current file `mtime` with stored `mtime` to detect changes
- If changed, delete old records for that file and re-ingest
- `--force-ingest` flag bypasses timestamp check and forces full re-ingestion

**Alternatives considered**:
- File content hashing (MD5/SHA256): Rejected - requires reading entire file to detect changes (slower)
- Manual versioning in config: Rejected - error-prone, requires user action
- Always re-ingest: Rejected - wasteful for large corpora, slow startup

**Implementation approach**:
```python
import os
from pathlib import Path

# Enhanced DocumentRecord with modification tracking
@dataclass
class DocumentRecord:
    id: Annotated[str, VectorStoreRecordKeyField()]
    content: Annotated[str, VectorStoreRecordDataField()]
    source_path: Annotated[str, VectorStoreRecordDataField()]
    mtime: Annotated[float, VectorStoreRecordDataField()]  # File modification time
    embedding: Annotated[list[float], VectorStoreRecordVectorField(dimensions=1536)]

# Check if re-ingestion needed
async def needs_reingest(file_path: Path, collection: VectorStoreRecordCollection) -> bool:
    current_mtime = file_path.stat().st_mtime

    # Query existing records for this file
    existing = await collection.get(f"{file_path}_chunk_0")

    if not existing:
        return True  # Never ingested

    if existing.mtime < current_mtime:
        return True  # File modified since last ingestion

    return False  # Up to date

# Force re-ingestion via CLI flag
# holodeck chat --force-ingest
# holodeck test --force-ingest
```

**Edge cases handled**:
- File deleted: Remove corresponding records from vector store on next run
- File renamed: Treated as delete + new file (mtime unchanged but path changed)
- Embedding model changed: Requires `--force-ingest` (detected via config change)

**References**:
- Python os.stat: https://docs.python.org/3/library/os.html#os.stat_result
- File timestamp tracking pattern: Common in build systems (Make, Bazel)

---

### 6. In-Memory Fallback Strategy

**Question**: How should the in-memory store work when no Redis is configured, and what are the limitations?

**Decision**: Use Semantic Kernel's `VolatileVectorStore` with session-only persistence

**Rationale**:
- `VolatileVectorStore` provides identical interface to `RedisStore` (VectorStore abstraction)
- Stores embeddings in Python dict/list structures (no external dependencies)
- Automatically cleared when agent process exits (ephemeral)
- Fast for development and small-scale testing (<1000 documents)
- No configuration required - works out of the box

**Alternatives considered**:
- SQLite with vector extension: Rejected - adds dependency, Semantic Kernel doesn't provide connector
- Pickle file persistence: Rejected - fragile, version compatibility issues
- Always require Redis: Rejected - raises barrier for development/testing

**Implementation approach**:
```python
from semantic_kernel.connectors.memory.volatile import VolatileVectorStore

# Determine store type from config
if vectorstore_config.database:
    # Production: Redis
    store = RedisStore(redis_client)
else:
    # Development/testing: In-memory
    logger.info("No vector database configured - using in-memory storage (session only)")
    store = VolatileVectorStore()

# Identical usage regardless of store type
collection = VectorStoreRecordCollection[str, DocumentRecord](store, "documents")
await collection.upsert(record)  # Works with both Redis and in-memory
results = await collection.search(query_embedding, top=5)
```

**Limitations documented**:
- No persistence across agent restarts
- Memory usage grows with corpus size (risk of OOM for large datasets)
- Single-process only (no multi-agent sharing)
- Recommended limit: 1000 documents or 10MB total embeddings

**References**:
- Semantic Kernel VolatileVectorStore: `semantic_kernel/connectors/memory/volatile/`

---

## Summary of Technical Decisions

| Area | Technology Choice | Key Rationale |
|------|------------------|---------------|
| Vector Store Abstraction | Semantic Kernel VectorStore | Unified interface, multi-backend support, strongly typed |
| Text Chunking | RecursiveCharacterTextSplitter (token-based) | Semantic preservation, token limit compliance, overlap support |
| Embedding Service | Semantic Kernel TextEmbedding | Multi-provider, batch processing, retry logic |
| Production Storage | Redis with RediSearch | Native vector search, Semantic Kernel connector, production-ready |
| Development Storage | VolatileVectorStore | Zero config, fast, identical interface to Redis |
| Modification Tracking | File mtime + metadata storage | Reliable, cross-platform, avoids content hashing overhead |
| File Processing | Existing FileProcessor (markitdown) | Already implemented, multimodal support, markdown output |

## Implementation Sequence

1. **Phase 1.1**: Implement `VectorStore` abstraction wrapper (lib/vector_store.py)
   - Support Redis and in-memory backends via Semantic Kernel connectors
   - Define `DocumentRecord` schema with mtime tracking

2. **Phase 1.2**: Implement text chunking (lib/text_chunker.py)
   - Wrap Semantic Kernel's `RecursiveCharacterTextSplitter`
   - Configure token-based chunking with overlap

3. **Phase 1.3**: Implement `VectorStoreTool` (tools/vectorstore_tool.py)
   - Integrate FileProcessor for file-to-markdown conversion
   - Generate embeddings via Semantic Kernel TextEmbedding
   - Store/retrieve from vector store with mtime tracking

4. **Phase 1.4**: Extend CLI commands (cli/commands/chat.py, test.py)
   - Add `--force-ingest` flag to bypass mtime checks

5. **Phase 2**: Testing
   - Unit tests: text chunking, vector store operations, mtime tracking
   - Integration tests: end-to-end ingestion and search with Redis/in-memory

## Open Questions (Resolved)

All technical unknowns from plan.md Technical Context have been resolved through this research.
