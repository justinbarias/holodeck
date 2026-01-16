"""Data models for tool filtering.

This module defines the Pydantic models used for tool filtering configuration
and metadata tracking. These models enable the Anthropic Tool Search pattern
for reducing token usage by dynamically filtering tools per request.

Key models:
- ToolMetadata: Metadata about a single tool including embedding and usage stats
- ToolFilterConfig: Configuration for tool filtering behavior
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolMetadata(BaseModel):
    """Metadata for a single tool used in semantic search and filtering.

    Stores information about tools extracted from the Semantic Kernel,
    including embeddings for semantic search and usage statistics for
    adaptive optimization.

    Attributes:
        name: Tool function name (e.g., "search", "get_user").
        plugin_name: Plugin namespace (e.g., "vectorstore", "mcp_weather").
        full_name: Combined identifier as "plugin_name-function_name".
        description: Human-readable description for semantic search.
        parameters: List of parameter descriptions for enhanced matching.
        defer_loading: If True, exclude from initial context (load on-demand).
        embedding: Pre-computed embedding vector for semantic search.
        usage_count: Number of times this tool has been invoked.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool function name")
    plugin_name: str = Field(default="", description="Plugin namespace")
    full_name: str = Field(..., description="Combined plugin_name-function_name")
    description: str = Field(..., description="Tool description for semantic search")
    parameters: list[str] = Field(
        default_factory=list, description="Parameter descriptions"
    )
    defer_loading: bool = Field(
        default=True, description="Exclude from initial context if True"
    )
    embedding: list[float] | None = Field(
        default=None, description="Pre-computed embedding vector"
    )
    usage_count: int = Field(default=0, description="Number of times tool was invoked")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        """Validate full_name is not empty."""
        if not v or not v.strip():
            raise ValueError("full_name must be a non-empty string")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty."""
        if not v or not v.strip():
            raise ValueError("description must be a non-empty string")
        return v

    @field_validator("usage_count")
    @classmethod
    def validate_usage_count(cls, v: int) -> int:
        """Validate usage_count is non-negative."""
        if v < 0:
            raise ValueError("usage_count must be non-negative")
        return v


class ToolFilterConfig(BaseModel):
    """Configuration for automatic tool filtering.

    Defines how tools are filtered per request to reduce token usage.
    When enabled, only the most relevant tools (based on semantic similarity
    to the user query) are included in each LLM call.

    Attributes:
        enabled: Enable or disable tool filtering globally.
        top_k: Maximum number of tools to include per request.
        similarity_threshold: Minimum similarity score for tool inclusion.
        always_include: Tool names that are always included regardless of score.
        always_include_top_n_used: Number of most-used tools to always include.
        search_method: Method for tool search (semantic, bm25, or hybrid).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable tool filtering")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum tools per request (1-50)",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for inclusion (0.0-1.0)",
    )
    always_include: list[str] = Field(
        default_factory=list,
        description="Tool names always included in context",
    )
    always_include_top_n_used: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of most-used tools to always include (0-20)",
    )
    search_method: Literal["semantic", "bm25", "hybrid"] = Field(
        default="semantic",
        description="Tool search method: semantic, bm25, or hybrid",
    )

    @field_validator("always_include")
    @classmethod
    def validate_always_include(cls, v: list[str]) -> list[str]:
        """Validate always_include entries are non-empty strings."""
        for tool_name in v:
            if not tool_name or not tool_name.strip():
                raise ValueError("always_include entries must be non-empty strings")
        return v
