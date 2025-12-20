"""Vector store abstractions using Semantic Kernel collection types.

This module provides a unified interface for working with various vector storage
backends (PostgreSQL, Azure AI Search, Qdrant, Weaviate, etc.) through
Semantic Kernel's VectorStoreCollection abstractions.

The DocumentRecord model is compatible with all supported backends, allowing
seamless switching between providers via configuration.

Supported Providers:
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
from typing import Annotated, Any, TypedDict, cast
from urllib.parse import urlparse
from uuid import uuid4

from pydantic.dataclasses import dataclass

# Vector store connectors are imported lazily in _get_collection_class()
# to avoid import errors when optional dependencies are not installed.
from semantic_kernel.data.vector import (
    DistanceFunction,
    VectorStoreCollectionDefinition,
    VectorStoreField,
    vectorstoremodel,
)

logger = logging.getLogger(__name__)


class ChromaConnectionParams(TypedDict):
    """Parameters for ChromaDB connection.

    Attributes:
        host: Server hostname (e.g., 'localhost')
        port: Server port (e.g., 8000)
        ssl: Whether to use HTTPS
    """

    host: str
    port: int
    ssl: bool


class QdrantConnectionParams(TypedDict, total=False):
    """Parameters for Qdrant connection.

    All fields are optional since Qdrant defaults to in-memory when no params provided.

    Attributes:
        url: Full URL to Qdrant server (e.g., 'https://qdrant.example.com:6333')
        api_key: API key for authentication
        host: Server hostname
        port: HTTP port (default: 6333)
        grpc_port: gRPC port (default: 6334)
        prefer_grpc: Whether to prefer gRPC over HTTP
        location: Special location string (e.g., ':memory:' for in-memory)
        path: Path for persistent local storage
    """

    url: str | None
    api_key: str | None
    host: str | None
    port: int | None
    grpc_port: int | None
    prefer_grpc: bool
    location: str | None
    path: str | None


class PineconeConnectionParams(TypedDict, total=False):
    """Parameters for Pinecone connection.

    Attributes:
        api_key: Pinecone API key (required)
        namespace: Namespace for the index (optional)
        use_grpc: Whether to use gRPC client (default: False)
    """

    api_key: str | None
    namespace: str | None
    use_grpc: bool


class PostgresConnectionParams(TypedDict, total=False):
    """Parameters for PostgreSQL connection.

    Attributes:
        connection_string: Full PostgreSQL connection string
        host: Database host
        port: Database port
        dbname: Database name
        user: Database user
        password: Database password
        sslmode: SSL mode (disable, allow, prefer, require, verify-ca, verify-full)
        db_schema: PostgreSQL schema (default: 'public')
    """

    connection_string: str | None
    host: str | None
    port: int | None
    dbname: str | None
    user: str | None
    password: str | None
    sslmode: str | None
    db_schema: str | None


def parse_chromadb_connection_string(connection_string: str) -> ChromaConnectionParams:
    """Parse a ChromaDB connection string into connection parameters.

    Supports URL format: http[s]://[host][:port][/path]

    The connection string follows standard URL conventions:
    - Scheme (http/https) determines SSL setting
    - Host defaults to 'localhost' if not specified
    - Port defaults to 8000 for HTTP, 443 for HTTPS

    Args:
        connection_string: URL-style connection string for ChromaDB server

    Returns:
        ChromaConnectionParams with host, port, and ssl values

    Raises:
        ValueError: If connection string is empty or uses unsupported scheme

    Examples:
        >>> parse_chromadb_connection_string("http://localhost:8000")
        {'host': 'localhost', 'port': 8000, 'ssl': False}

        >>> parse_chromadb_connection_string("https://chroma.example.com")
        {'host': 'chroma.example.com', 'port': 443, 'ssl': True}

        >>> parse_chromadb_connection_string("http://localhost")
        {'host': 'localhost', 'port': 8000, 'ssl': False}

        >>> parse_chromadb_connection_string("https://chroma.internal:9000")
        {'host': 'chroma.internal', 'port': 9000, 'ssl': True}
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")

    parsed = urlparse(connection_string)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}'. ChromaDB connection string must use "
            "http:// or https:// scheme"
        )

    ssl = parsed.scheme == "https"
    host = parsed.hostname or "localhost"

    # Default ports based on scheme
    port = parsed.port or (443 if ssl else 8000)

    params: ChromaConnectionParams = {
        "host": host,
        "port": port,
        "ssl": ssl,
    }

    return params


def parse_qdrant_connection_string(connection_string: str) -> QdrantConnectionParams:
    """Parse a Qdrant connection string into connection parameters.

    Supports multiple formats for flexibility:
    - Standard URL: https://host:port or http://localhost:6333
    - With API key in userinfo: https://api_key@host:port
    - gRPC preference: qdrant+grpc://host:port
    - In-memory: :memory:
    - Local path: /path/to/qdrant/data or file:///path/to/data

    The connection string is parsed and mapped to QdrantCollection parameters.

    Args:
        connection_string: Connection string in one of the supported formats

    Returns:
        QdrantConnectionParams with appropriate fields set

    Raises:
        ValueError: If connection string is empty or uses unsupported scheme

    Examples:
        >>> parse_qdrant_connection_string("https://qdrant.example.com:6333")
        {'url': 'https://qdrant.example.com:6333'}

        >>> parse_qdrant_connection_string("http://localhost:6333")
        {'host': 'localhost', 'port': 6333}

        >>> parse_qdrant_connection_string(":memory:")
        {'location': ':memory:'}

        >>> parse_qdrant_connection_string("qdrant+grpc://localhost:6334")
        {'host': 'localhost', 'grpc_port': 6334, 'prefer_grpc': True}
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")

    # Handle special in-memory location
    if connection_string == ":memory:":
        return {"location": ":memory:"}

    # Handle local file path (for persistent local storage)
    if connection_string.startswith("/") or connection_string.startswith("file://"):
        path = connection_string.replace("file://", "")
        return {"path": path}

    parsed = urlparse(connection_string)

    # Handle gRPC preference via scheme
    prefer_grpc = parsed.scheme in ("qdrant+grpc", "grpc")

    # Normalize scheme for URL construction
    if parsed.scheme in ("qdrant", "qdrant+grpc", "grpc"):
        # Convert custom schemes to http/https
        actual_scheme = "https" if parsed.port == 443 else "http"
    elif parsed.scheme in ("http", "https"):
        actual_scheme = parsed.scheme
    else:
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}'. Qdrant connection string must use "
            "http://, https://, qdrant://, or qdrant+grpc:// scheme"
        )

    host = parsed.hostname or "localhost"

    # Default ports based on transport
    if prefer_grpc:
        grpc_port = parsed.port or 6334
        params: QdrantConnectionParams = {
            "host": host,
            "grpc_port": grpc_port,
            "prefer_grpc": True,
        }
    else:
        port = parsed.port or 6333
        # For remote servers, pass full URL; for localhost use host/port
        if host == "localhost" or host == "127.0.0.1":
            params = {
                "host": host,
                "port": port,
            }
        else:
            # Use full URL for remote servers
            url = f"{actual_scheme}://{host}"
            if parsed.port:
                url += f":{parsed.port}"
            params = {"url": url}

    # Extract API key from userinfo if present
    if parsed.username:
        params["api_key"] = parsed.username

    return params


def parse_pinecone_connection_string(
    connection_string: str,
) -> PineconeConnectionParams:
    """Parse a Pinecone connection string into connection parameters.

    Pinecone primarily uses API key authentication. The connection string
    can be the API key directly or a URL-like format for consistency.

    Supported formats:
    - Direct API key: "pc-abc123..." (starts with 'pc-')
    - URL format: pinecone://api_key or pinecone://api_key@namespace

    Args:
        connection_string: Connection string with API key

    Returns:
        PineconeConnectionParams with api_key and optional namespace

    Raises:
        ValueError: If connection string is empty

    Examples:
        >>> parse_pinecone_connection_string("pc-abc123def456")
        {'api_key': 'pc-abc123def456'}

        >>> parse_pinecone_connection_string("pinecone://pc-abc123")
        {'api_key': 'pc-abc123'}

        >>> parse_pinecone_connection_string("pinecone://pc-abc123@my-namespace")
        {'api_key': 'pc-abc123', 'namespace': 'my-namespace'}
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")

    # Direct API key format (Pinecone keys typically start with 'pc-')
    if connection_string.startswith("pc-") or not connection_string.startswith(
        "pinecone://"
    ):
        return {"api_key": connection_string}

    # URL-like format: pinecone://api_key[@namespace]
    parsed = urlparse(connection_string)

    if parsed.scheme != "pinecone":
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}'. "
            "Use 'pinecone://' or provide API key directly"
        )

    params: PineconeConnectionParams = {}

    # API key can be in username position or hostname
    if parsed.username:
        params["api_key"] = parsed.username
        if parsed.hostname:
            params["namespace"] = parsed.hostname
    elif parsed.hostname:
        params["api_key"] = parsed.hostname

    # Check for namespace in path
    if parsed.path and parsed.path.startswith("/"):
        namespace = parsed.path[1:]  # Remove leading slash
        if namespace:
            params["namespace"] = namespace

    return params


def create_chromadb_client(
    connection_string: str | None = None,
    persist_directory: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a ChromaDB client from connection parameters.

    This function handles the complexity of creating the appropriate ChromaDB
    client type based on the provided parameters. It abstracts away the
    differences between ChromaDB's HttpClient, PersistentClient, and
    EphemeralClient.

    Client Selection Logic:
        1. If connection_string is provided: Creates HttpClient for remote server
        2. If persist_directory is provided: Creates PersistentClient for local storage
        3. Otherwise: Creates EphemeralClient (in-memory, data lost on exit)

    Args:
        connection_string: URL for remote ChromaDB server.
            Format: "http[s]://host[:port]"
            Examples: "http://localhost:8000", "https://chroma.prod.example.com"
        persist_directory: Local directory path for persistent storage.
            The directory will be created if it doesn't exist.
            Example: "/var/data/chromadb", "./data/vectors"
        **kwargs: Additional parameters passed to the client:
            - headers (dict[str, str]): HTTP headers for authentication
              (only used with connection_string)
            - tenant (str): Tenant name (default: 'default_tenant')
            - database (str): Database name (default: 'default_database')

    Returns:
        ChromaDB ClientAPI instance (HttpClient, PersistentClient, or EphemeralClient)

    Raises:
        ValueError: If connection_string format is invalid
        ImportError: If chromadb package is not installed

    Examples:
        >>> # Connect to remote ChromaDB server
        >>> client = create_chromadb_client(
        ...     connection_string="http://localhost:8000"
        ... )

        >>> # Connect to remote server with authentication
        >>> client = create_chromadb_client(
        ...     connection_string="https://chroma.example.com",
        ...     headers={"Authorization": "Bearer token123"}
        ... )

        >>> # Create persistent local database
        >>> client = create_chromadb_client(
        ...     persist_directory="/path/to/data"
        ... )

        >>> # Create ephemeral in-memory database (for testing/development)
        >>> client = create_chromadb_client()
    """
    try:
        import chromadb
    except ImportError as e:
        raise ImportError(
            "ChromaDB is not installed. Install with: pip install chromadb"
        ) from e

    # Extract known kwargs
    headers = kwargs.pop("headers", None)
    tenant = kwargs.pop("tenant", "default_tenant")
    database = kwargs.pop("database", "default_database")

    if connection_string:
        # Remote server mode - parse connection string and create HttpClient
        params = parse_chromadb_connection_string(connection_string)
        return chromadb.HttpClient(
            host=params["host"],
            port=params["port"],
            ssl=params["ssl"],
            headers=headers,
            tenant=tenant,
            database=database,
        )
    elif persist_directory:
        # Persistent local mode
        return chromadb.PersistentClient(
            path=persist_directory,
            tenant=tenant,
            database=database,
        )
    else:
        # Ephemeral in-memory mode
        return chromadb.EphemeralClient(
            tenant=tenant,
            database=database,
        )


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
            list[float] | None,
            VectorStoreField(
                "vector",
                dimensions=dimensions,
                distance_function=DistanceFunction.COSINE_SIMILARITY,
            ),
        ] = field(default=None)
        mtime: Annotated[float, VectorStoreField("data")] = field(default=0.0)
        file_type: Annotated[str, VectorStoreField("data")] = field(default="")
        file_size_bytes: Annotated[int, VectorStoreField("data")] = field(default=0)

    return cast(type[Any], DynamicDocumentRecord)


def create_structured_record_class(
    dimensions: int = 1536,
    metadata_field_names: list[str] | None = None,
    collection_name: str = "structured_records",
) -> tuple[type[Any], VectorStoreCollectionDefinition]:
    """Create a StructuredRecord class and definition for structured data.

    This factory creates a new StructuredRecord dataclass AND a matching
    VectorStoreCollectionDefinition. Both are needed for proper persistence
    of dynamic metadata fields to Semantic Kernel vector stores.

    Unlike DocumentRecord which is for unstructured documents, this is
    designed for structured data (CSV, JSON, JSONL) with user-defined
    metadata fields.

    Args:
        dimensions: Embedding vector dimensions (default: 1536)
        metadata_field_names: List of metadata field names to include
        collection_name: Vector store collection name

    Returns:
        Tuple of (record_class, definition) for collection creation

    Raises:
        ValueError: If dimensions is invalid (<=0 or >10000)
        ValueError: If metadata field name is not a valid Python identifier

    Example:
        >>> RecordClass, definition = create_structured_record_class(
        ...     dimensions=768,
        ...     metadata_field_names=["title", "category", "price"],
        ...     collection_name="products",
        ... )
        >>> record = RecordClass(
        ...     id="P001",
        ...     content="Product description",
        ...     embedding=[...],
        ...     source_file="products.csv",
        ...     title="Widget Pro",
        ...     category="Electronics",
        ...     price="99.99",
        ... )
    """
    if dimensions <= 0 or dimensions > 10000:
        raise ValueError(f"Invalid dimensions: {dimensions}")

    # Validate metadata field names are valid Python identifiers
    if metadata_field_names:
        for field_name in metadata_field_names:
            if not field_name.isidentifier():
                raise ValueError(
                    f"Invalid field name: '{field_name}' "
                    "(must be valid Python identifier)"
                )

    # 1. Create simple dataclass WITHOUT @vectorstoremodel decorator
    # The definition is built separately and passed explicitly to collections
    @dataclass
    class DynamicStructuredRecord:
        """Vector store record for structured data with embeddings.

        Each record represents a row/document from structured data
        (CSV, JSON, JSONL). Compatible with all Semantic Kernel vector
        store backends.

        Attributes:
            id: Unique identifier from the source data's id_field (key)
            content: Concatenated vector field content (full-text indexed)
            embedding: Vector embedding
            source_file: Original source file path (indexed for filtering)
        """

        id: str = ""
        content: str = ""
        embedding: list[float] | None = None
        source_file: str = ""

    # 2. Build VectorStoreField list programmatically
    fields: list[VectorStoreField] = [
        VectorStoreField("key", name="id", type="str"),
        VectorStoreField("data", name="content", type="str", is_full_text_indexed=True),
        VectorStoreField(
            "vector",
            name="embedding",
            type="float",
            dimensions=dimensions,
            distance_function=DistanceFunction.COSINE_SIMILARITY,
        ),
        VectorStoreField("data", name="source_file", type="str", is_indexed=True),
    ]

    # 3. Add dynamic metadata fields to BOTH class and definition
    if metadata_field_names:
        # Add fields to the definition
        for field_name in metadata_field_names:
            fields.append(
                VectorStoreField("data", name=field_name, type="str", is_indexed=True)
            )

        # Store metadata config in a dedicated namespace (explicit and introspectable)
        DynamicStructuredRecord.__metadata_config__ = {  # type: ignore[attr-defined]
            "field_names": metadata_field_names
        }

        # Register the metadata fields in __annotations__ for introspection
        for field_name in metadata_field_names:
            DynamicStructuredRecord.__annotations__[field_name] = str

        # Override __init__ to accept metadata fields as keyword arguments
        original_init = DynamicStructuredRecord.__init__

        def new_init(
            self: Any,
            id: str = "",
            content: str = "",
            embedding: list[float] | None = None,
            source_file: str = "",
            **kwargs: Any,
        ) -> None:
            # Call original with keyword args (not positional) for dataclass compat
            original_init(
                self,
                id=id,
                content=content,
                embedding=embedding,
                source_file=source_file,
            )
            # Store metadata fields as instance attributes
            config = DynamicStructuredRecord.__metadata_config__  # type: ignore[attr-defined]
            for fname in config["field_names"]:
                setattr(self, fname, kwargs.get(fname, ""))

        DynamicStructuredRecord.__init__ = new_init  # type: ignore[method-assign]

    # 4. Build the VectorStoreCollectionDefinition
    # Note: We don't need to_dict/from_dict for individual records -
    # SK will use the field definitions directly for serialization
    definition = VectorStoreCollectionDefinition(
        collection_name=collection_name,
        fields=fields,
    )

    return cast(type[Any], DynamicStructuredRecord), definition


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


@dataclass
class StructuredQueryResult:
    """Search result from structured data vector store query.

    Represents a single match returned from semantic search over structured data
    (CSV, JSON, JSONL files). Unlike QueryResult which uses chunk_index, this uses
    the original record ID from the source data.

    Attributes:
        id: Original record identifier from the id_field in source data
        content: Concatenated vector field content that was embedded
        score: Relevance/similarity score (0.0-1.0, higher is better)
        source_file: Original source file path (e.g., "products.csv")
        metadata: Dictionary of metadata field values from the source record
    """

    id: str
    content: str
    score: float
    source_file: str
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
        # Use holodeck-ai[extra] format for optional dependencies we provide
        dep_hints: dict[str, str] = {
            "postgres": "holodeck-ai[postgres]",
            "azure-ai-search": "azure-search-documents",
            "qdrant": "holodeck-ai[qdrant]",
            "weaviate": "weaviate-client",
            "chromadb": "holodeck-ai[chromadb]",
            "faiss": "faiss-cpu",
            "azure-cosmos-mongo": "pymongo",
            "azure-cosmos-nosql": "azure-cosmos",
            "sql-server": "pyodbc",
            "pinecone": "holodeck-ai[pinecone]",
        }
        hint = dep_hints.get(provider, "")
        if hint:
            if hint.startswith("holodeck-ai["):
                install_msg = f" Install with: uv add {hint}"
            else:
                install_msg = f" Install with: pip install {hint}"
        else:
            install_msg = ""
        raise ImportError(
            f"Missing dependencies for vector store provider '{provider}'.{install_msg}"
        ) from e


def get_collection_factory(
    provider: str,
    dimensions: int = 1536,
    record_class: type[Any] | None = None,
    definition: VectorStoreCollectionDefinition | None = None,
    **connection_kwargs: Any,
) -> Callable[[], Any]:
    """Get a vector store collection factory for the specified provider.

    Returns a callable that lazily initializes the appropriate Semantic Kernel
    collection type based on the provider name and connection parameters.

    Args:
        provider: Vector store provider name. Supported providers:
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

        dimensions: Embedding vector dimensions (default: 1536).
            Must be between 1 and 10000.

        definition: Optional VectorStoreCollectionDefinition for structured
            data with dynamic metadata fields. When provided, this definition
            is passed to the collection constructor for proper field handling.

        **connection_kwargs: Provider-specific connection parameters.

            For chromadb provider:
                - connection_string (str): URL for remote ChromaDB server.
                  Format: "http[s]://host[:port]"
                  Examples: "http://localhost:8000", "https://chroma.example.com"
                - persist_directory (str): Local directory for persistent storage.
                  If provided, creates a PersistentClient instead of HttpClient.
                - headers (dict[str, str]): HTTP headers for authentication
                  (only used with connection_string)
                - tenant (str): Tenant name (default: 'default_tenant')
                - database (str): Database name (default: 'default_database')

                Note: If neither connection_string nor persist_directory is
                provided, an ephemeral in-memory client is created.

            For other providers:
                Refer to Semantic Kernel documentation for provider-specific
                connection parameters (e.g., connection_string for postgres).

    Returns:
        Callable that returns a Semantic Kernel VectorStoreCollection instance

    Raises:
        ValueError: If provider is not supported or dimensions are invalid
        ImportError: If required dependencies for the provider are not installed

    Examples:
        >>> # PostgreSQL with connection string
        >>> factory = get_collection_factory(
        ...     "postgres",
        ...     dimensions=1536,
        ...     connection_string="postgresql://user:pass@localhost/db"
        ... )
        >>> async with factory() as collection:
        ...     await collection.upsert([record])

        >>> # ChromaDB - Connect to remote server
        >>> factory = get_collection_factory(
        ...     "chromadb",
        ...     dimensions=1536,
        ...     connection_string="http://localhost:8000"
        ... )

        >>> # ChromaDB - Connect with authentication headers
        >>> factory = get_collection_factory(
        ...     "chromadb",
        ...     dimensions=1536,
        ...     connection_string="https://chroma.example.com",
        ...     headers={"Authorization": "Bearer token123"}
        ... )

        >>> # ChromaDB - Persistent local storage
        >>> factory = get_collection_factory(
        ...     "chromadb",
        ...     dimensions=1536,
        ...     persist_directory="/var/data/vectors"
        ... )

        >>> # ChromaDB - Ephemeral in-memory (for testing)
        >>> factory = get_collection_factory("chromadb", dimensions=768)

        >>> # In-memory provider (development only)
        >>> factory = get_collection_factory("in-memory", dimensions=1536)
    """
    supported_providers = [
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

    # Use provided record_class or create default DocumentRecord
    if record_class is None:
        record_class = create_document_record_class(dimensions)

    # Pre-process provider-specific kwargs to avoid mutating original dict in factory
    # Each provider may need connection_string parsed into specific parameters

    # ChromaDB handling
    if provider == "chromadb":
        chromadb_connection_string = connection_kwargs.pop("connection_string", None)
        chromadb_persist_directory = connection_kwargs.pop("persist_directory", None)
        chromadb_extra_kwargs = connection_kwargs.copy()
    else:
        chromadb_connection_string = None
        chromadb_persist_directory = None
        chromadb_extra_kwargs = {}

    # Qdrant handling - parse connection_string into Qdrant-specific params
    if provider == "qdrant":
        qdrant_params: QdrantConnectionParams = {}
        if "connection_string" in connection_kwargs:
            qdrant_params = parse_qdrant_connection_string(
                connection_kwargs.pop("connection_string")
            )
        # Merge any explicit kwargs (they override parsed values)
        qdrant_params.update(connection_kwargs)  # type: ignore[typeddict-item]
    else:
        qdrant_params = {}

    # Pinecone handling - parse connection_string or use api_key directly
    if provider == "pinecone":
        pinecone_params: PineconeConnectionParams = {}
        if "connection_string" in connection_kwargs:
            pinecone_params = parse_pinecone_connection_string(
                connection_kwargs.pop("connection_string")
            )
        # Merge any explicit kwargs (api_key, namespace, use_grpc)
        if "api_key" in connection_kwargs:
            pinecone_params["api_key"] = connection_kwargs.pop("api_key")
        if "namespace" in connection_kwargs:
            pinecone_params["namespace"] = connection_kwargs.pop("namespace")
        if "use_grpc" in connection_kwargs:
            pinecone_params["use_grpc"] = connection_kwargs.pop("use_grpc")
        pinecone_extra_kwargs = connection_kwargs.copy()
    else:
        pinecone_params = {}
        pinecone_extra_kwargs = {}

    # PostgreSQL handling - connection_string is passed directly to SK
    # SK's PostgresSettings handles parsing internally
    if provider == "postgres":
        postgres_params: PostgresConnectionParams = {}
        if "connection_string" in connection_kwargs:
            # Pass connection_string to SK which will parse it via PostgresSettings
            postgres_params["connection_string"] = connection_kwargs.pop(
                "connection_string"
            )
        # Handle db_schema separately
        if "db_schema" in connection_kwargs:
            postgres_params["db_schema"] = connection_kwargs.pop("db_schema")
        postgres_extra_kwargs = connection_kwargs.copy()
    else:
        postgres_params = {}
        postgres_extra_kwargs = {}

    def factory() -> Any:
        """Return async context manager for the collection."""
        # Lazy import at factory call time
        collection_class = get_collection_class(provider)

        # Build base kwargs with optional definition for structured data
        base_kwargs: dict[str, Any] = {"record_type": record_class}
        if definition is not None:
            base_kwargs["definition"] = definition

        # ChromaDB requires special handling for connection_string
        if provider == "chromadb":
            # Create the appropriate client
            client = create_chromadb_client(
                connection_string=chromadb_connection_string,
                persist_directory=chromadb_persist_directory,
                **chromadb_extra_kwargs,
            )

            # Pass the pre-configured client to ChromaCollection
            return collection_class[str, record_class](
                client=client,
                **base_kwargs,
            )

        # Qdrant - pass parsed parameters directly to QdrantCollection
        if provider == "qdrant":
            return collection_class[str, record_class](
                **base_kwargs,
                **qdrant_params,
            )

        # Pinecone - pass parsed parameters to PineconeCollection
        if provider == "pinecone":
            return collection_class[str, record_class](
                **base_kwargs,
                **pinecone_params,
                **pinecone_extra_kwargs,
            )

        # PostgreSQL - SK handles connection_string parsing via PostgresSettings
        if provider == "postgres":
            # PostgresCollection accepts settings or individual connection params
            # We pass connection_string which PostgresSettings will parse
            kwargs_for_postgres: dict[str, Any] = base_kwargs.copy()
            conn_str = postgres_params.get("connection_string")
            if conn_str:
                # Create PostgresSettings with connection_string
                # SK will handle the parsing internally
                from pydantic import SecretStr
                from semantic_kernel.connectors.postgres import PostgresSettings

                settings = PostgresSettings(connection_string=SecretStr(conn_str))
                kwargs_for_postgres["settings"] = settings
            db_schema = postgres_params.get("db_schema")
            if db_schema:
                kwargs_for_postgres["db_schema"] = db_schema
            kwargs_for_postgres.update(postgres_extra_kwargs)
            return collection_class[str, record_class](**kwargs_for_postgres)

        # Default handling for other providers
        return collection_class[str, record_class](
            **base_kwargs,
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
