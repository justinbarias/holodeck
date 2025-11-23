"""Vector store abstractions using Semantic Kernel collection types.

This module provides a unified interface for working with various vector storage
backends (Redis, PostgreSQL, Azure AI Search, Qdrant, Weaviate, etc.) through
Semantic Kernel's VectorStoreCollection abstractions.

The DocumentRecord model is compatible with all supported backends, allowing
seamless switching between providers via configuration.

Supported Providers:
- redis-hashset: Redis with Hashset storage
- redis-json: Redis with JSON storage
- postgres: PostgreSQL with pgvector extension
- azure-ai-search: Azure AI Search (Cognitive Search)
- qdrant: Qdrant vector database
- weaviate: Weaviate vector database
- chromadb: ChromaDB (local or server)
- faiss: FAISS (in-memory or file-based)
- azure-cosmos-mongo: Azure Cosmos DB (MongoDB API)
- azure-cosmos-nosql: Azure Cosmos DB (NoSQL API)
- sql-server: SQL Server with vector support
- pinecone: Pinecone serverless vector database
- in-memory: Simple in-memory storage (development only)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from semantic_kernel.connectors.memory import (
    AzureAISearchCollection,
    ChromaCollection,
    CosmosMongoCollection,
    CosmosNoSqlCollection,
    FaissCollection,
    InMemoryCollection,
    PineconeCollection,
    PostgresCollection,
    QdrantCollection,
    RedisHashsetCollection,
    RedisJsonCollection,
    SqlServerCollection,
    WeaviateCollection,
)
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

logger = logging.getLogger(__name__)


@vectorstoremodel(collection_name="documents")
@dataclass
class DocumentRecord:  # type: ignore[misc]
    """Vector store record for document chunks with embeddings.

    Each document file is split into multiple chunks, each with its own embedding.
    This record is compatible with all Semantic Kernel vector store backends.

    The @vectorstoremodel decorator enables automatic schema generation for the
    underlying vector database, supporting all major vector store providers.

    Attributes:
        id: Unique identifier (key field) following format:
            {source_path}_chunk_{chunk_index}
        source_path: Original source file path (indexed for filtering)
        chunk_index: Chunk index within document (0-indexed, indexed)
        content: Chunk content for semantic search (full-text indexed)
        embedding: Vector embedding (1536 dimensions for text-embedding-3-small)
        mtime: File modification time (Unix timestamp) for change detection
        file_type: Source file extension (.txt, .md, .pdf, etc.)
        file_size_bytes: Original file size in bytes
    """

    id: Annotated[str, VectorStoreField("key")] = field(
        default_factory=lambda: str(uuid4())
    )
    source_path: Annotated[str, VectorStoreField("data", is_indexed=True)] = field(
        default=""
    )
    chunk_index: Annotated[int, VectorStoreField("data", is_indexed=True)] = field(
        default=0
    )
    content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)] = (
        field(default="")
    )
    embedding: Annotated[
        list[float] | None, VectorStoreField("vector", dimensions=1536)
    ] = None
    mtime: Annotated[float, VectorStoreField("data")] = field(default=0.0)
    file_type: Annotated[str, VectorStoreField("data")] = field(default="")
    file_size_bytes: Annotated[int, VectorStoreField("data")] = field(default=0)


@dataclass
class QueryResult:
    """Search result from vector store query.

    Represents a single match returned from semantic search operations.

    Attributes:
        content: Matched document chunk content
        score: Relevance/similarity score (0.0-1.0, higher is better)
        source_path: Original source file path
        chunk_index: Chunk index within source file
        metadata: Additional metadata (file_type, file_size, mtime, etc.)
    """

    content: str
    score: float
    source_path: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


def get_collection_factory(
    provider: str,
    **connection_kwargs: Any,
) -> Callable[[], Any]:
    """Get a vector store collection factory for the specified provider.

    Returns a callable that lazily initializes the appropriate Semantic Kernel
    collection type based on the provider name and connection parameters.

    Args:
        provider: Vector store provider name (redis-hashset, postgres, etc.)
        **connection_kwargs: Provider-specific connection parameters

    Returns:
        Callable that returns an async context manager for the collection

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> factory = get_collection_factory("postgres",
        ...     connection_string="postgresql://user:pass@localhost/db")
        >>> async with factory() as collection:
        ...     await collection.upsert([record])
    """
    factories: dict[str, type[Any]] = {
        "redis-hashset": RedisHashsetCollection,
        "redis-json": RedisJsonCollection,
        "postgres": PostgresCollection,
        "azure-ai-search": AzureAISearchCollection,
        "qdrant": QdrantCollection,
        "weaviate": WeaviateCollection,
        "chromadb": ChromaCollection,
        "faiss": FaissCollection,
        "azure-cosmos-mongo": CosmosMongoCollection,
        "azure-cosmos-nosql": CosmosNoSqlCollection,
        "sql-server": SqlServerCollection,
        "pinecone": PineconeCollection,
        "in-memory": InMemoryCollection,
    }

    if provider not in factories:
        raise ValueError(
            f"Unsupported vector store provider: {provider}. "
            f"Supported providers: {', '.join(sorted(factories.keys()))}"
        )

    collection_class = factories[provider]

    def factory() -> Any:
        """Return async context manager for the collection."""
        return collection_class[str, DocumentRecord](
            record_type=DocumentRecord,
            **connection_kwargs,
        )

    return factory


async def convert_document_to_query_result(
    record: DocumentRecord,
    score: float,
) -> QueryResult:
    """Convert a DocumentRecord search result to QueryResult.

    Args:
        record: DocumentRecord from vector search
        score: Relevance/similarity score (0.0-1.0)

    Returns:
        QueryResult with metadata extracted from the record

    Raises:
        ValueError: If score is outside valid range
    """
    return QueryResult(
        content=record.content,
        score=score,
        source_path=record.source_path,
        chunk_index=record.chunk_index,
        metadata={
            "file_type": record.file_type,
            "file_size_bytes": record.file_size_bytes,
            "mtime": record.mtime,
        },
    )
