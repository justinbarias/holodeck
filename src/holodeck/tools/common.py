"""Common utilities for HoloDeck tools.

This module provides shared constants and pure utility functions used by
VectorStoreTool, HierarchicalDocumentTool, and other tool implementations.

The utilities extracted here follow the DRY principle and provide:
- File extension constants and type mappings
- Path resolution with context variable support
- Recursive file discovery with extension filtering
- Placeholder embedding generation for testing

Usage:
    from holodeck.tools.common import (
        SUPPORTED_EXTENSIONS,
        FILE_TYPE_MAPPING,
        get_file_type,
        resolve_source_path,
        discover_files,
        generate_placeholder_embeddings,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Supported file extensions for ingestion
# Used by VectorStoreTool and HierarchicalDocumentTool
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".pdf", ".csv", ".json"}
)


# Mapping from file extensions to FileInput type values
# Used for file processing with FileProcessor
FILE_TYPE_MAPPING: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".csv": "csv",
    ".json": "text",
}


def get_file_type(path: str | Path) -> str:
    """Get FileInput type from file extension.

    Maps file extensions to the appropriate type value for FileProcessor.
    Defaults to "text" for unknown extensions.

    Args:
        path: File path (string or Path object)

    Returns:
        FileInput type string ("text", "pdf", "csv", etc.)

    Example:
        >>> get_file_type("document.pdf")
        'pdf'
        >>> get_file_type(Path("data/file.csv"))
        'csv'
        >>> get_file_type("unknown.xyz")
        'text'
    """
    if isinstance(path, str):
        path = Path(path)
    extension = path.suffix.lower()
    return FILE_TYPE_MAPPING.get(extension, "text")


def resolve_source_path(source: str, base_dir: str | None = None) -> Path:
    """Resolve a source path relative to a base directory.

    This function handles path resolution in priority order:
    1. If source is absolute, return as-is
    2. If base_dir is provided, resolve relative to base_dir
    3. Try agent_base_dir context variable
    4. Fall back to current working directory

    Args:
        source: Source path to resolve (from tool config)
        base_dir: Optional base directory for relative path resolution

    Returns:
        Resolved absolute Path to the source

    Example:
        >>> resolve_source_path("/absolute/path/file.txt")
        PosixPath('/absolute/path/file.txt')
        >>> resolve_source_path("relative/file.txt", "/base")
        PosixPath('/base/relative/file.txt')
    """
    source_path = Path(source)

    # If path is absolute, use it directly
    if source_path.is_absolute():
        return source_path

    # Resolve relative to base directory
    # Priority: explicit base_dir > context var > cwd
    effective_base = base_dir
    if effective_base is None:
        # Try to get from context variable
        from holodeck.config.context import agent_base_dir

        effective_base = agent_base_dir.get()

    if effective_base:
        return (Path(effective_base) / source).resolve()

    return source_path.resolve()


def discover_files(
    source_path: Path,
    extensions: frozenset[str] | None = None,
) -> list[Path]:
    """Discover files to ingest from a source path.

    Recursively traverses directories and filters by supported extensions.
    For single files, validates the extension is supported.

    Args:
        source_path: Resolved path (file or directory) to discover from
        extensions: Set of supported extensions (default: SUPPORTED_EXTENSIONS)

    Returns:
        List of Path objects for files to process, sorted for deterministic order

    Note:
        This function does not validate file existence - that should be
        checked before calling this function.

    Example:
        >>> discover_files(Path("/docs"))
        [PosixPath('/docs/file1.md'), PosixPath('/docs/subdir/file2.txt')]
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    if source_path.is_file():
        # Single file - check if supported
        if source_path.suffix.lower() in extensions:
            return [source_path]
        logger.warning(
            f"File {source_path} has unsupported extension "
            f"{source_path.suffix}. Supported: {extensions}"
        )
        return []

    if source_path.is_dir():
        # Directory - recursively find all supported files
        discovered: list[Path] = []
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in extensions:
                    discovered.append(file_path)
                else:
                    logger.debug(
                        f"Skipping unsupported file: {file_path} "
                        f"(extension: {file_path.suffix})"
                    )
        # Sort for deterministic ordering
        return sorted(discovered)

    # Path doesn't exist - return empty list
    return []


def generate_placeholder_embeddings(
    count: int,
    dimensions: int = 1536,
) -> list[list[float]]:
    """Generate placeholder embedding vectors for testing.

    Creates zero-valued embedding vectors when no embedding service is available.
    Useful for testing and development without LLM API access.

    Args:
        count: Number of embeddings to generate
        dimensions: Embedding vector dimensions (default: 1536)

    Returns:
        List of zero-valued embedding vectors

    Example:
        >>> embeddings = generate_placeholder_embeddings(3, 768)
        >>> len(embeddings)
        3
        >>> len(embeddings[0])
        768
    """
    logger.debug(f"Generated {count} placeholder embeddings, dim={dimensions}")
    return [[0.0] * dimensions for _ in range(count)]
