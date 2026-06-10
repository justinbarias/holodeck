# Tools

The `holodeck.tools` package provides tool implementations that extend agent capabilities with semantic search, hierarchical document retrieval, and MCP server integration.

## Module Overview

| Module | Description |
|--------|-------------|
| `holodeck.tools` | Package exports for tools, mixins, and utilities |
| `holodeck.tools.base_tool` | Mixin classes for embedding and database configuration |
| `holodeck.tools.common` | Shared constants and pure utility functions |
| `holodeck.tools.vectorstore_tool` | Semantic search over unstructured and structured data |
| `holodeck.tools.hierarchical_document_tool` | Structure-aware document retrieval with hybrid search |

---

## `holodeck.tools.base_tool`

Mixin classes that encapsulate common functionality shared between `VectorStoreTool` and `HierarchicalDocumentTool`. Mixins are used instead of inheritance because the tools have fundamentally different record types and core behaviors.

### `EmbeddingServiceMixin`

::: holodeck.tools.base_tool.EmbeddingServiceMixin
    options:
      docstring_style: google
      show_source: true
      members:
        - set_embedding_service

### `DatabaseConfigMixin`

::: holodeck.tools.base_tool.DatabaseConfigMixin
    options:
      docstring_style: google
      show_source: true
      members:
        - _resolve_database_config
        - _create_collection_with_fallback

---

## `holodeck.tools.common`

Shared constants and pure utility functions used by `VectorStoreTool`, `HierarchicalDocumentTool`, and other tool implementations. Follows the DRY principle for file handling, path resolution, and embedding generation.

### Constants

#### `SUPPORTED_EXTENSIONS`

::: holodeck.tools.common.SUPPORTED_EXTENSIONS
    options:
      docstring_style: google
      show_source: true

#### `FILE_TYPE_MAPPING`

::: holodeck.tools.common.FILE_TYPE_MAPPING
    options:
      docstring_style: google
      show_source: true

### Functions

#### `get_file_type`

::: holodeck.tools.common.get_file_type
    options:
      docstring_style: google
      show_source: true

#### `resolve_source_path`

::: holodeck.tools.common.resolve_source_path
    options:
      docstring_style: google
      show_source: true

#### `discover_files`

::: holodeck.tools.common.discover_files
    options:
      docstring_style: google
      show_source: true

#### `generate_placeholder_embeddings`

::: holodeck.tools.common.generate_placeholder_embeddings
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.tools.vectorstore_tool`

Provides semantic search over files and directories containing text data or structured data (CSV, JSON, JSONL files with field mapping). Supports automatic file discovery, text chunking, embedding generation, vector storage, and modification-time tracking for incremental ingestion.

### `VectorStoreTool`

::: holodeck.tools.vectorstore_tool.VectorStoreTool
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - initialize
        - search
        - set_embedding_service

---

## `holodeck.tools.hierarchical_document_tool`

Provides intelligent document search that understands document structure, extracts definitions, and generates optimized context for LLM consumption. Supports semantic, keyword (BM25), and hybrid search modes with configurable weights.

### `HierarchicalDocumentTool`

::: holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - initialize
        - search
        - get_context
        - get_definition
        - set_embedding_service
        - set_context_generator
