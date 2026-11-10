"""Search results that the vectorstore tool returns to its callers."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


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
