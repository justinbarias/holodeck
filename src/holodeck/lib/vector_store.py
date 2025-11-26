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

import logging
from collections.abc import Callable
from dataclasses import field
from typing import Annotated, Any, cast
from uuid import uuid4

from pydantic.dataclasses import dataclass

# Vector store connectors are imported lazily in _get_collection_class()
# to avoid import errors when optional dependencies are not installed.
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

logger = logging.getLogger(__name__)


def create_document_record_class(dimensions: int = 1536) -> type[Any]:
    """Create a DocumentRecord class with specified embedding dimensions.

    This factory creates a new DocumentRecord dataclass with custom dimensions.
    Each collection can have its own DocumentRecord type.

    Args:
        dimensions: Embedding vector dimensions

    Returns:
        DocumentRecord class configured for the specified dimensions

    Raises:
        ValueError: If dimensions is invalid
    """
    if dimensions <= 0 or dimensions > 10000:
        raise ValueError(f"Invalid dimensions: {dimensions}")

    @vectorstoremodel(collection_name=f"documents_dim{dimensions}")
    @dataclass
    class DynamicDocumentRecord:  # type: ignore[misc]
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
            embedding: Vector embedding
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
            list[float] | None, VectorStoreField("vector", dimensions=dimensions)
        ] = field(default=None)
        mtime: Annotated[float, VectorStoreField("data")] = field(default=0.0)
        file_type: Annotated[str, VectorStoreField("data")] = field(default="")
        file_size_bytes: Annotated[int, VectorStoreField("data")] = field(default=0)

    return cast(type[Any], DynamicDocumentRecord)


# Keep original DocumentRecord for backward compatibility (1536 dimensions)
DocumentRecord = create_document_record_class(1536)


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


def get_collection_class(provider: str) -> type[Any]:
    """Lazily import and return the collection class for a provider.

    This function imports connector classes on-demand to avoid import errors
    when optional dependencies are not installed.

    Args:
        provider: Vector store provider name

    Returns:
        The collection class for the specified provider

    Raises:
        ValueError: If provider is not supported
        ImportError: If required dependencies for the provider are not installed
    """
    # Map providers to their import paths and class names
    provider_imports: dict[str, tuple[str, str]] = {
        "redis-hashset": (
            "semantic_kernel.connectors.redis",
            "RedisHashsetCollection",
        ),
        "redis-json": ("semantic_kernel.connectors.redis", "RedisJsonCollection"),
        "postgres": ("semantic_kernel.connectors.postgres", "PostgresCollection"),
        "azure-ai-search": (
            "semantic_kernel.connectors.azure_ai_search",
            "AzureAISearchCollection",
        ),
        "qdrant": ("semantic_kernel.connectors.qdrant", "QdrantCollection"),
        "weaviate": ("semantic_kernel.connectors.weaviate", "WeaviateCollection"),
        "chromadb": ("semantic_kernel.connectors.chroma", "ChromaCollection"),
        "faiss": ("semantic_kernel.connectors.faiss", "FaissCollection"),
        "azure-cosmos-mongo": (
            "semantic_kernel.connectors.azure_cosmos_db",
            "CosmosMongoCollection",
        ),
        "azure-cosmos-nosql": (
            "semantic_kernel.connectors.azure_cosmos_db",
            "CosmosNoSqlCollection",
        ),
        "sql-server": ("semantic_kernel.connectors.sql_server", "SqlServerCollection"),
        "pinecone": ("semantic_kernel.connectors.pinecone", "PineconeCollection"),
        "in-memory": ("semantic_kernel.connectors.in_memory", "InMemoryCollection"),
    }

    if provider not in provider_imports:
        raise ValueError(
            f"Unsupported vector store provider: {provider}. "
            f"Supported providers: {', '.join(sorted(provider_imports.keys()))}"
        )

    module_path, class_name = provider_imports[provider]

    try:
        import importlib

        module = importlib.import_module(module_path)
        return cast(type[Any], getattr(module, class_name))
    except ImportError as e:
        # Provide helpful error message about missing dependencies
        dep_hints: dict[str, str] = {
            "redis-hashset": "redis[hiredis]",
            "redis-json": "redis[hiredis]",
            "postgres": "psycopg[binary,pool]",
            "azure-ai-search": "azure-search-documents",
            "qdrant": "qdrant-client",
            "weaviate": "weaviate-client",
            "chromadb": "chromadb",
            "faiss": "faiss-cpu",
            "azure-cosmos-mongo": "pymongo",
            "azure-cosmos-nosql": "azure-cosmos",
            "sql-server": "pyodbc",
            "pinecone": "pinecone-client",
        }
        hint = dep_hints.get(provider, "")
        install_msg = f" Try: pip install {hint}" if hint else ""
        raise ImportError(
            f"Missing dependencies for vector store provider '{provider}'.{install_msg}"
        ) from e


def get_collection_factory(
    provider: str,
    dimensions: int = 1536,
    **connection_kwargs: Any,
) -> Callable[[], Any]:
    """Get a vector store collection factory for the specified provider.

    Returns a callable that lazily initializes the appropriate Semantic Kernel
    collection type based on the provider name and connection parameters.

    Args:
        provider: Vector store provider name (redis-hashset, postgres, etc.)
        dimensions: Embedding vector dimensions (default: 1536)
        **connection_kwargs: Provider-specific connection parameters

    Returns:
        Callable that returns an async context manager for the collection

    Raises:
        ValueError: If provider or dimensions are invalid
        ImportError: If required dependencies for the provider are not installed

    Example:
        >>> factory = get_collection_factory("postgres",
        ...     dimensions=1536,
        ...     connection_string="postgresql://user:pass@localhost/db")
        >>> async with factory() as collection:
        ...     await collection.upsert([record])
    """
    supported_providers = [
        "redis-hashset",
        "redis-json",
        "postgres",
        "azure-ai-search",
        "qdrant",
        "weaviate",
        "chromadb",
        "faiss",
        "azure-cosmos-mongo",
        "azure-cosmos-nosql",
        "sql-server",
        "pinecone",
        "in-memory",
    ]

    # Validate dimensions
    if dimensions <= 0 or dimensions > 10000:
        raise ValueError(f"Invalid dimensions: {dimensions}")

    if provider not in supported_providers:
        raise ValueError(
            f"Unsupported vector store provider: {provider}. "
            f"Supported providers: {', '.join(sorted(supported_providers))}"
        )

    # Create DocumentRecord class for these dimensions
    record_class = create_document_record_class(dimensions)

    def factory() -> Any:
        """Return async context manager for the collection."""
        # Lazy import at factory call time
        collection_class = get_collection_class(provider)
        return collection_class[str, record_class](
            record_type=record_class,
            **connection_kwargs,
        )

    return factory


async def convert_document_to_query_result(
    record: Any,
    score: float,
) -> QueryResult:
    """Convert a DocumentRecord search result to QueryResult.

    Args:
        record: DocumentRecord from vector search (dynamically created)
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
