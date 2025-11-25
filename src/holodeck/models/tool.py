"""Tool models for agent configuration.

This module defines the Tool data models used in agent.yaml configuration.
Tools enable agents to interact with external systems and data sources.

Tool types:
- VectorstoreTool: Semantic search over document collections
- FunctionTool: Call Python functions
- MCPTool: Model Context Protocol integrations
- PromptTool: AI-powered semantic functions
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator


class Tool(BaseModel):
    """Base tool model with discriminated union for subtypes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier, unique within agent")
    description: str = Field(..., description="Human-readable tool description")
    type: str = Field(
        ..., description="Tool type: vectorstore, function, mcp, or prompt"
    )


class DatabaseConfig(BaseModel):
    """Vector database connection configuration.

    Supports all Semantic Kernel vector store providers including Redis, PostgreSQL,
    Azure AI Search, Qdrant, Weaviate, ChromaDB, FAISS, Pinecone, and more.

    Provider-specific parameters are passed via the config dict:
    - redis-hashset/redis-json: connection_string
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
    ] = Field(
        ...,
        description=(
            "Vector database provider: redis-hashset, redis-json, postgres, "
            "azure-ai-search, qdrant, weaviate, chromadb, faiss, "
            "azure-cosmos-mongo, azure-cosmos-nosql, sql-server, pinecone, in-memory"
        ),
    )
    connection_string: str | None = Field(
        None,
        description=(
            "Database connection string (format depends on provider). "
            "Examples: redis://localhost:6379, postgresql://user:pass@host/db, "
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


class VectorstoreTool(BaseModel):
    """Vectorstore tool for semantic search over documents."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["vectorstore"] = Field(default="vectorstore", description="Tool type")
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


class FunctionTool(BaseModel):
    """Function tool for calling Python functions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["function"] = Field(default="function", description="Tool type")
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
    """MCP (Model Context Protocol) tool for standardized integrations."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["mcp"] = Field(default="mcp", description="Tool type")
    server: str = Field(..., description="MCP server identifier")
    config: dict[str, Any] | None = Field(None, description="MCP configuration")

    @field_validator("server")
    @classmethod
    def validate_server(cls, v: str) -> str:
        """Validate server is not empty."""
        if not v or not v.strip():
            raise ValueError("server must be a non-empty identifier")
        return v


class PromptTool(BaseModel):
    """Prompt-based tool for AI-powered semantic functions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: Literal["prompt"] = Field(default="prompt", description="Tool type")
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
    | Annotated[PromptTool, Tag("prompt")],
    Discriminator(_get_tool_type),
]
