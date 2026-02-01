"""Tools module for HoloDeck agent capabilities.

This module provides tool implementations that extend agent capabilities:
- VectorStoreTool: Semantic search over unstructured documents
- HierarchicalDocumentTool: Structure-aware document retrieval

Also provides shared utilities for tool development:
- common: Shared constants and utility functions
- base_tool: Mixin classes for common functionality
"""

from holodeck.tools.base_tool import DatabaseConfigMixin, EmbeddingServiceMixin
from holodeck.tools.common import (
    FILE_TYPE_MAPPING,
    SUPPORTED_EXTENSIONS,
    discover_files,
    generate_placeholder_embeddings,
    get_file_type,
    resolve_source_path,
)
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool

__all__ = [
    # Tools
    "VectorStoreTool",
    "HierarchicalDocumentTool",
    # Mixins
    "EmbeddingServiceMixin",
    "DatabaseConfigMixin",
    # Utilities
    "SUPPORTED_EXTENSIONS",
    "FILE_TYPE_MAPPING",
    "get_file_type",
    "resolve_source_path",
    "discover_files",
    "generate_placeholder_embeddings",
]
