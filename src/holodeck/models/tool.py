"""Tool models for agent configuration.

This module defines the Tool data models used in agent.yaml configuration.
Tools enable agents to interact with external systems and data sources.

Tool types:
- VectorstoreTool: Semantic search over document collections
- FunctionTool: Call Python functions
- MCPTool: Model Context Protocol integrations
- PromptTool: AI-powered semantic functions
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tool(BaseModel):
    """Base tool model with discriminated union for subtypes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier, unique within agent")
    description: str = Field(..., description="Human-readable tool description")
    type: str = Field(
        ..., description="Tool type: vectorstore, function, mcp, or prompt"
    )


class VectorstoreTool(BaseModel):
    """Vectorstore tool for semantic search over documents."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: str = Field(default="vectorstore", description="Tool type")
    source: str = Field(..., description="Path to data file or directory")
    vector_field: str | list[str] | None = Field(
        None, description="Field(s) to vectorize"
    )
    meta_fields: list[str] | None = Field(None, description="Metadata fields")
    chunk_size: int | None = Field(None, description="Text chunk size for splitting")
    chunk_overlap: int | None = Field(None, description="Chunk overlap size")
    embedding_model: str | None = Field(None, description="Embedding model name")
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


class FunctionTool(BaseModel):
    """Function tool for calling Python functions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Tool description")
    type: str = Field(default="function", description="Tool type")
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
    type: str = Field(default="mcp", description="Tool type")
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
    type: str = Field(default="prompt", description="Tool type")
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
