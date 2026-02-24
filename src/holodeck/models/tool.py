"""Tool models for agent configuration.

This module defines the Tool data models used in agent.yaml configuration.
Tools enable agents to interact with external systems and data sources.

Tool types:
- VectorstoreTool: Semantic search over document collections
- FunctionTool: Call Python functions
- MCPTool: Model Context Protocol integrations
- PromptTool: AI-powered semantic functions
"""

import math
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from holodeck.models.llm import LLMProvider


class TransportType(str, Enum):
    """Supported MCP transport types.

    Defines the communication protocol used to connect to MCP servers:
    - STDIO: Local process via stdin/stdout (default, most common)
    - SSE: Server-Sent Events over HTTP
    - WEBSOCKET: WebSocket for bidirectional communication
    - HTTP: Streamable HTTP transport
    """

    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"
    HTTP = "http"


class CommandType(str, Enum):
    """Allowed stdio commands for MCP servers (security constraint).

    Only these commands are permitted for spawning MCP server processes
    to prevent command injection attacks:
    - NPX: Node.js/npm package runner
    - NODE: Direct Node.js script runner
    - UVX: Python/uv package runner
    - DOCKER: Docker container runner
    """

    NPX = "npx"
    NODE = "node"
    UVX = "uvx"
    DOCKER = "docker"


class SearchMode(str, Enum):
    """Search modality options for HierarchicalDocumentTool.

    Defines which search indices are used during query:
    - SEMANTIC: Dense embeddings only (conceptual similarity)
    - KEYWORD: BM25 sparse index only (term frequency)
    - HYBRID: Keyword + semantic combined with RRF fusion
    """

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class ChunkingStrategy(str, Enum):
    """Document chunking approach for HierarchicalDocumentTool.

    Defines how documents are split into chunks:
    - STRUCTURE: Parse markdown headings and split at structural boundaries
    - TOKEN: Fixed token-based splitting with overlap (fallback for flat text)
    """

    STRUCTURE = "structure"
    TOKEN = "token"  # noqa: S105  # nosec B105 - Not a password, token-based chunking


class DocumentDomain(str, Enum):
    """Predefined document structure patterns for subsection detection.

    Enables YAML-based configuration of domain-specific parsing rules
    for hierarchical document tools. Each domain defines subsection patterns
    that recognize implicit headings in documents.

    Attributes:
        NONE: No subsection patterns (default, backward compatible).
        US_LEGISLATIVE: US Code style: (a), (1), (A), (i) hierarchy.
        AU_LEGISLATIVE: Australian style: (1), (a), (i), (A) hierarchy.
        ACADEMIC: Academic papers: 1., 1.1, 1.1.1 numbered sections.
        TECHNICAL: Technical manuals: Step 1, 1.1, Note:, Warning:.
        LEGAL_CONTRACT: Legal contracts: Article I, Section 1, (a) clauses.
    """

    NONE = "none"
    US_LEGISLATIVE = "us_legislative"
    AU_LEGISLATIVE = "au_legislative"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LEGAL_CONTRACT = "legal_contract"


class KeywordIndexProvider(str, Enum):
    """Keyword index backend provider for sparse/BM25 search."""

    IN_MEMORY = "in-memory"
    OPENSEARCH = "opensearch"


class Tool(BaseModel):
    """Base tool model with discriminated union for subtypes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        pattern=r"^[0-9A-Za-z_]+$",
        description="Tool identifier (alphanumeric and underscores only)",
    )
    description: str = Field(..., description="Human-readable tool description")
    type: str = Field(
        ..., description="Tool type: vectorstore, function, mcp, or prompt"
    )
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools that should "
            "always be available."
        ),
    )


class DatabaseConfig(BaseModel):
    """Vector database connection configuration.

    Supports all Semantic Kernel vector store providers including PostgreSQL,
    Azure AI Search, Qdrant, Weaviate, ChromaDB, FAISS, Pinecone, and more.

    Provider-specific parameters are passed via the config dict:
    - postgres: connection_string
    - azure-ai-search: connection_string, api_key
    - qdrant: url, api_key (optional)
    - weaviate: url, api_key (optional)
    - chromadb: path or host
    - faiss: path
    - pinecone: api_key, index_name
    - And more...
    """

    model_config = ConfigDict(extra="allow")  # Allow provider-specific parameters

    provider: Literal[
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
    ] = Field(
        ...,
        description=(
            "Vector database provider: postgres, azure-ai-search, qdrant, "
            "weaviate, chromadb, faiss, azure-cosmos-mongo, azure-cosmos-nosql, "
            "sql-server, pinecone, in-memory"
        ),
    )
    connection_string: str | None = Field(
        None,
        description=(
            "Database connection string (format depends on provider). "
            "Examples: postgresql://user:pass@host/db, "
            "https://search-service.search.windows.net"
        ),
    )

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str | None) -> str | None:
        """Validate connection string is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("connection_string must be non-empty if provided")
        return v


class KeywordIndexConfig(BaseModel):
    """Keyword index configuration for sparse/BM25 search backend.

    Two providers:
    - in-memory: rank_bm25 in-process (default, dev/local)
    - opensearch: OpenSearch endpoint (production)

    When provider='opensearch', endpoint and index_name are required
    (validated at model construction time).
    """

    model_config = ConfigDict(extra="allow")

    provider: Literal["in-memory", "opensearch"] = Field(
        default="in-memory",
        description="Keyword index backend: 'in-memory' or 'opensearch'",
    )
    endpoint: str | None = Field(default=None, description="OpenSearch endpoint URL")
    index_name: str | None = Field(default=None, description="OpenSearch index name")
    username: str | None = Field(default=None, description="Basic auth username")
    password: str | None = Field(default=None, description="Basic auth password")
    api_key: str | None = Field(default=None, description="API key auth (alt to basic)")
    verify_certs: bool = Field(default=True, description="Verify TLS certificates")
    timeout_seconds: int = Field(
        default=10, ge=1, le=120, description="Connection timeout (1-120s)"
    )

    @model_validator(mode="after")
    def validate_opensearch_fields(self) -> "KeywordIndexConfig":
        """Validate that endpoint and index_name are set for opensearch provider."""
        if self.provider == "opensearch":
            if not self.endpoint:
                raise ValueError("endpoint is required when provider is 'opensearch'")
            if not self.index_name:
                raise ValueError("index_name is required when provider is 'opensearch'")
        return self


class VectorstoreTool(BaseModel):
    """Vectorstore tool for semantic search over documents."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[0-9A-Za-z_]+$", description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["vectorstore"] = Field(default="vectorstore", description="Tool type")
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools."
        ),
    )
    source: str = Field(..., description="Path to data file or directory")
    vector_field: str | list[str] | None = Field(
        None, description="Field(s) to vectorize"
    )
    meta_fields: list[str] | None = Field(None, description="Metadata fields")
    chunk_size: int | None = Field(None, description="Text chunk size for splitting")
    chunk_overlap: int | None = Field(None, description="Chunk overlap size")
    embedding_model: str | None = Field(
        None, description="Custom embedding model (defaults to provider default)"
    )
    embedding_dimensions: int | None = Field(
        None,
        description=(
            "Embedding vector dimensions (auto-detected from model if not specified)"
        ),
    )
    database: DatabaseConfig | str | None = Field(
        None,
        description=(
            "Vector database configuration. Can be:\n"
            "- DatabaseConfig object with provider and connection details\n"
            "- String reference to a named vectorstore in global config\n"
            "- None for in-memory storage"
        ),
    )
    top_k: int = Field(
        default=5, description="Number of top results to return from search"
    )
    min_similarity_score: float | None = Field(
        None, description="Minimum similarity score threshold for results (0.0-1.0)"
    )
    record_path: str | None = Field(None, description="Path to array in JSON")
    record_prefix: str | None = Field(None, description="Record field prefix")
    meta_prefix: str | None = Field(None, description="Metadata field prefix")

    # Structured data fields
    id_field: str | None = Field(
        None,
        description=(
            "Unique record identifier field (required for structured data mode). "
            "Must be set when vector_field is specified."
        ),
    )
    field_separator: str = Field(
        default="\n",
        description=(
            "Separator for concatenating multiple vector_fields (default: newline). "
            "Used when vector_field is a list."
        ),
    )
    delimiter: str | None = Field(
        None,
        description=(
            "CSV delimiter character (auto-detect if None). "
            "Only applicable for CSV files."
        ),
    )

    @model_validator(mode="after")
    def validate_structured_config(self) -> "VectorstoreTool":
        """Validate structured data configuration.

        When vector_field is set (structured data mode), id_field becomes required
        to enable record identification for upserts and deduplication.
        """
        if self.vector_field is not None and self.id_field is None:
            raise ValueError(
                "id_field is required when vector_field is specified "
                "(structured data mode)"
            )
        return self

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source is not empty."""
        if not v or not v.strip():
            raise ValueError("source must be a non-empty path")
        return v

    @field_validator("database")
    @classmethod
    def validate_database(
        cls, v: DatabaseConfig | str | None
    ) -> DatabaseConfig | str | None:
        """Validate database is not empty string if provided as string."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("database reference must be a non-empty string")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int | None) -> int | None:
        """Validate chunk_size is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int | None) -> int | None:
        """Validate chunk_overlap is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is a positive integer."""
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        if v > 100:
            raise ValueError("top_k should not exceed 100")
        return v

    @field_validator("min_similarity_score")
    @classmethod
    def validate_min_similarity_score(cls, v: float | None) -> float | None:
        """Validate min_similarity_score is between 0.0 and 1.0 if provided."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_similarity_score must be between 0.0 and 1.0")
        return v

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, v: int | None) -> int | None:
        """Validate embedding_dimensions is positive and reasonable if provided."""
        if v is not None:
            if v <= 0:
                raise ValueError("embedding_dimensions must be positive")
            if v > 10000:
                raise ValueError("embedding_dimensions unreasonably large (max 10000)")
        return v


class FunctionTool(BaseModel):
    """Function tool for calling Python functions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[0-9A-Za-z_]+$", description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["function"] = Field(default="function", description="Tool type")
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools."
        ),
    )
    file: str = Field(..., description="Path to Python file")
    function: str = Field(..., description="Function name")
    parameters: dict[str, dict[str, Any]] | None = Field(
        None, description="Parameter schema"
    )

    @field_validator("file")
    @classmethod
    def validate_file(cls, v: str) -> str:
        """Validate file is not empty."""
        if not v or not v.strip():
            raise ValueError("file must be a non-empty path")
        return v

    @field_validator("function")
    @classmethod
    def validate_function(cls, v: str) -> str:
        """Validate function is not empty."""
        if not v or not v.strip():
            raise ValueError("function must be a non-empty identifier")
        return v


class MCPTool(BaseModel):
    """MCP (Model Context Protocol) tool for standardized integrations.

    Supports four transport types:
    - stdio (default): Local MCP servers via subprocess
    - sse: Remote servers via Server-Sent Events
    - websocket: Bidirectional WebSocket communication
    - http: Streamable HTTP transport

    For stdio transport, only npx, uvx, or docker commands are allowed
    for security reasons.
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    name: str = Field(..., pattern=r"^[0-9A-Za-z_]+$", description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["mcp"] = Field(default="mcp", description="Tool type")
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools."
        ),
    )

    # Transport configuration
    transport: TransportType = Field(
        default=TransportType.STDIO, description="Transport type"
    )

    # Stdio transport fields
    command: CommandType | None = Field(
        None, description="Command to run (required for stdio: npx, uvx, or docker)"
    )
    args: list[str] | None = Field(None, description="Command arguments")
    env: dict[str, str] | None = Field(None, description="Environment variables")
    env_file: str | None = Field(None, description="Path to .env file")
    encoding: str | None = Field(None, description="Stream encoding (default: utf-8)")

    # HTTP/SSE/WebSocket transport fields
    url: str | None = Field(None, description="Server URL for HTTP/SSE/WebSocket")
    headers: dict[str, str] | None = Field(None, description="HTTP headers")
    timeout: float | None = Field(None, description="Connection timeout (seconds)")
    sse_read_timeout: float | None = Field(None, description="SSE read timeout")
    terminate_on_close: bool | None = Field(
        None, description="Terminate HTTP connection on close"
    )

    # Common optional fields
    config: dict[str, Any] | None = Field(
        None, description="Server-specific configuration"
    )
    load_tools: bool = Field(True, description="Auto-discover tools from server")
    load_prompts: bool = Field(True, description="Auto-discover prompts from server")
    request_timeout: int = Field(60, description="Operation timeout (seconds)")

    # RAG evaluation fields
    is_retrieval: bool = Field(
        default=False,
        description=(
            "Mark this MCP tool as a retrieval tool for RAG evaluation metrics. "
            "When True, tool results contribute to retrieval_context for metrics "
            "like Faithfulness, ContextualRelevancy, ContextualPrecision, and "
            "ContextualRecall."
        ),
    )
    registry_name: str | None = Field(
        None,
        description=(
            "Full reverse-DNS name from MCP registry (e.g., 'io.github.user/server'). "
            "Used for duplicate detection and origin tracking. "
            "None for manually configured MCP servers."
        ),
    )

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str | None) -> str | None:
        """Validate URL scheme for HTTP-based transports.

        Allows http:// only for localhost, requires https:// for remote URLs.
        WebSocket URLs can use wss:// or ws://.
        """
        if v is None:
            return v
        # Allow http:// for localhost, require https:// otherwise
        if v.startswith("http://"):
            localhost_prefixes = (
                "http://localhost",
                "http://127.0.0.1",
                "http://[::1]",
            )
            if not any(v.startswith(prefix) for prefix in localhost_prefixes):
                raise ValueError("'url' must use https:// (or http:// for localhost)")
        elif not v.startswith(("https://", "wss://", "ws://")):
            raise ValueError("'url' must use https://, wss://, or ws:// scheme")
        return v

    @field_validator("request_timeout")
    @classmethod
    def validate_request_timeout(cls, v: int) -> int:
        """Validate request_timeout is positive."""
        if v <= 0:
            raise ValueError("request_timeout must be positive")
        return v

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPTool":
        """Validate transport-specific required fields.

        - stdio transport requires 'command'
        - sse, websocket, http transports require 'url'
        """
        if self.transport == TransportType.STDIO:
            if self.command is None:
                raise ValueError("'command' is required for stdio transport")
        elif (
            self.transport
            in (TransportType.SSE, TransportType.WEBSOCKET, TransportType.HTTP)
            and self.url is None
        ):
            raise ValueError(f"'url' is required for {self.transport.value} transport")
        return self


class PromptTool(BaseModel):
    """Prompt-based tool for AI-powered semantic functions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[0-9A-Za-z_]+$", description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["prompt"] = Field(default="prompt", description="Tool type")
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools."
        ),
    )
    template: str | None = Field(None, description="Inline prompt template")
    file: str | None = Field(None, description="Path to prompt file")
    parameters: dict[str, dict[str, Any]] = Field(
        ..., description="Parameter definitions (required)"
    )
    model: dict[str, Any] | None = Field(None, description="Model config override")

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str | None) -> str | None:
        """Validate template is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("template must be non-empty if provided")
        return v

    @field_validator("file")
    @classmethod
    def validate_file(cls, v: str | None) -> str | None:
        """Validate file is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("file must be non-empty if provided")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, v: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Validate parameters is not empty."""
        if not v:
            raise ValueError("parameters must have at least one parameter")
        return v

    @field_validator("parameters", mode="before")
    @classmethod
    def check_template_or_file(cls, v: Any, info: Any) -> Any:
        """Validate that exactly one of template or file is provided."""
        data = info.data
        template = data.get("template")
        file_path = data.get("file")

        if not template and not file_path:
            raise ValueError("Exactly one of 'template' or 'file' must be provided")
        if template and file_path:
            raise ValueError("Cannot provide both 'template' and 'file'")

        return v


class HierarchicalDocumentToolConfig(BaseModel):
    """Configuration for the HierarchicalDocumentTool.

    A specialized document search tool that preserves document structure,
    supports multiple search modalities (semantic, keyword, exact), and
    enables context-aware embeddings for improved retrieval quality.

    Features:
    - Structure-aware chunking that respects markdown headings
    - Hybrid search combining semantic, keyword, and exact matching
    - Contextual embeddings that include document/section context
    - Automatic definition and cross-reference extraction
    - Optional reranking for improved result quality
    """

    model_config = ConfigDict(extra="forbid")

    # Required fields
    name: str = Field(
        ...,
        pattern=r"^[0-9A-Za-z_]+$",
        description="Tool identifier (alphanumeric and underscores only)",
    )
    description: str = Field(..., description="Human-readable tool description")
    type: Literal["hierarchical_document"] = Field(
        default="hierarchical_document", description="Tool type"
    )
    source: str = Field(
        ...,
        description=(
            "Path to markdown document or directory containing markdown files. "
            "Supports glob patterns for multiple files."
        ),
    )

    # Chunking configuration
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.STRUCTURE,
        description=(
            "Document chunking approach: 'structure' parses markdown headings, "
            "'token' uses fixed token-based splitting"
        ),
    )
    max_chunk_tokens: int = Field(
        default=800,
        ge=100,
        le=2000,
        description="Maximum tokens per chunk (100-2000, default 800)",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Token overlap between chunks (0-200, default 50)",
    )

    # Document domain configuration for subsection detection
    document_domain: DocumentDomain = Field(
        default=DocumentDomain.NONE,
        description=(
            "Document structure domain for subsection detection. "
            "Options: 'none' (default), 'us_legislative', 'au_legislative', "
            "'academic', 'technical', 'legal_contract'. When set to a domain, "
            "enables recognition of implicit headings like (a), (1), Step 1, etc."
        ),
    )
    max_subsection_depth: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum subsection nesting depth. If None (default), uses all "
            "patterns defined for the document_domain. Set to limit depth, e.g., "
            "2 to only recognize Title and Chapter but not Section and below."
        ),
    )

    # Search configuration
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description=(
            "Search modality: 'semantic' (embeddings), 'keyword' (BM25), "
            "'exact' (ID/phrase match), 'hybrid' (all combined)"
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return (1-100, default 10)",
    )
    min_score: float | None = Field(
        default=None,
        description="Minimum similarity score threshold for results (0.0-1.0)",
    )
    semantic_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid mode (default 0.5)",
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid mode (default 0.3)",
    )
    exact_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for exact search in hybrid mode (default 0.2)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "Reciprocal Rank Fusion constant for hybrid search (default 60). "
            "Higher values give more weight to lower-ranked results."
        ),
    )

    # Contextual embeddings configuration
    contextual_embeddings: bool = Field(
        default=True,
        description=(
            "Enable contextual embeddings that include document/section context "
            "for improved semantic understanding"
        ),
    )
    context_max_tokens: int = Field(
        default=100,
        ge=50,
        le=200,
        description="Maximum tokens for context summary (50-200, default 100)",
    )
    context_concurrency: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Concurrent context generation requests (1-50, default 10)",
    )

    # Feature extraction
    extract_definitions: bool = Field(
        default=True,
        description="Extract and index term definitions from documents",
    )
    extract_cross_references: bool = Field(
        default=True,
        description="Extract and resolve cross-references between sections",
    )

    # Reranking configuration
    enable_reranking: bool = Field(
        default=False,
        description="Enable LLM-based reranking of search results",
    )
    reranker_model: LLMProvider | None = Field(
        default=None,
        description=("LLM for reranking results. Required when enable_reranking=True."),
    )
    context_model: LLMProvider | None = Field(
        default=None,
        description=(
            "LLM for contextual embedding generation. "
            "Overrides the default model used by LLMContextGenerator."
        ),
    )
    embedding_model: str | None = Field(
        default=None,
        description="Custom embedding model (defaults to provider default)",
    )

    # Storage configuration
    database: DatabaseConfig | str | None = Field(
        default=None,
        description=(
            "Vector database configuration. Can be:\n"
            "- DatabaseConfig object with provider and connection details\n"
            "- String reference to a named vectorstore in global config\n"
            "- None for in-memory storage"
        ),
    )
    keyword_index: KeywordIndexConfig | None = Field(
        default=None,
        description=(
            "Keyword index configuration for sparse/BM25 search. "
            "If None, defaults to in-memory BM25."
        ),
    )
    defer_loading: bool = Field(
        default=True,
        description=(
            "If True, tool is excluded from initial context and loaded on-demand "
            "via semantic search. Set to False for critical tools."
        ),
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source is not empty."""
        if not v or not v.strip():
            raise ValueError("source must be a non-empty path")
        return v

    @field_validator("min_score")
    @classmethod
    def validate_min_score(cls, v: float | None) -> float | None:
        """Validate min_score is between 0.0 and 1.0 if provided."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_score must be between 0.0 and 1.0")
        return v

    @field_validator("semantic_weight", "keyword_weight", "exact_weight")
    @classmethod
    def validate_finite_weights(cls, v: float) -> float:
        """Validate search weights are finite numeric values."""
        if not math.isfinite(v):
            raise ValueError("search weights must be finite numbers")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "HierarchicalDocumentToolConfig":
        """Validate hybrid weights are well-formed for deterministic fusion."""
        if self.search_mode == SearchMode.HYBRID:
            total = self.semantic_weight + self.keyword_weight + self.exact_weight
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Hybrid search weights must sum to 1.0, got {total:.6f} "
                    "(semantic_weight + keyword_weight + exact_weight)"
                )

            if (self.semantic_weight + self.keyword_weight) <= 0:
                raise ValueError(
                    "Hybrid search requires semantic_weight + keyword_weight > 0"
                )
        return self

    @model_validator(mode="after")
    def validate_reranker(self) -> "HierarchicalDocumentToolConfig":
        """Validate reranker_model is provided when enable_reranking=True."""
        if self.enable_reranking and self.reranker_model is None:
            raise ValueError("reranker_model is required when enable_reranking=True")
        return self


def _get_tool_type(v: Any) -> str:
    """Extract tool type from dict or model for discrimination.

    Args:
        v: Tool data as dict (from YAML) or model instance

    Returns:
        Tool type string for discriminator matching
    """
    if isinstance(v, dict):
        tool_type: str = v.get("type", "")
        return tool_type
    result: str = getattr(v, "type", "")
    return result


# Discriminated union for all tool types - enables type-safe tool handling
# Each tool model is tagged with its type value for Pydantic discriminator
ToolUnion = Annotated[
    Annotated[VectorstoreTool, Tag("vectorstore")]
    | Annotated[FunctionTool, Tag("function")]
    | Annotated[MCPTool, Tag("mcp")]
    | Annotated[PromptTool, Tag("prompt")]
    | Annotated[HierarchicalDocumentToolConfig, Tag("hierarchical_document")],
    Discriminator(_get_tool_type),
]
