# GraphRAG Engine Contract

**Feature**: 016-graphrag-integration
**Date**: 2025-12-27
**Type**: Internal Python API

---

## 1. Overview

This contract defines the internal Python API for the GraphRAG engine integration. It specifies the interfaces between HoloDeck components and the GraphRAG library.

---

## 2. GraphRAGEngine Class

The main interface for GraphRAG operations, matching the existing VectorStoreTool pattern.

### 2.1 Class Definition

```python
class GraphRAGEngine:
    """GraphRAG engine for knowledge graph-based retrieval.

    Provides initialization, indexing, and search capabilities
    for GraphRAG-based vectorstore tools.
    """

    def __init__(
        self,
        config: VectorstoreTool,
        base_dir: str | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize GraphRAG engine.

        Args:
            config: VectorstoreTool configuration with engine="graphrag"
            base_dir: Base directory for resolving paths
            execution_config: Execution settings (for interface compat)

        Raises:
            ValueError: If config.engine != "graphrag"
            ImportError: If graphrag package not installed
        """
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the engine has been initialized."""
        ...

    @property
    def index_stats(self) -> dict[str, int]:
        """Statistics about the built index.

        Returns:
            Dict with keys: entities, relationships, communities, files
        """
        ...

    async def initialize(
        self,
        force_reindex: bool = False,
    ) -> None:
        """Initialize the GraphRAG engine.

        Steps:
        1. Resolve source path
        2. Set up storage directories
        3. Build or load index
        4. Initialize search engine

        Args:
            force_reindex: Force rebuild even if index is current

        Raises:
            FileNotFoundError: If source path doesn't exist
            IndexingError: If GraphRAG indexing fails
        """
        ...

    async def search(self, query: str) -> str:
        """Search the knowledge graph.

        Args:
            query: The search query

        Returns:
            Formatted search results as string

        Raises:
            RuntimeError: If engine not initialized
        """
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
```

---

## 3. GraphRAGStorage Class

Manages artifact storage and index metadata.

### 3.1 Class Definition

```python
class GraphRAGStorage:
    """Manages GraphRAG artifact storage and index metadata.

    Storage layout:
        .holodeck/graphrag/{tool_name}/
        ├── output/           # GraphRAG parquet artifacts
        ├── cache/            # LLM call cache
        ├── logs/             # Pipeline logs
        ├── lancedb/          # Entity embeddings
        └── index.meta        # Index metadata
    """

    REQUIRED_TABLES: ClassVar[list[str]] = [
        "create_final_entities",
        "create_final_relationships",
        "create_final_communities",
        "create_final_community_reports",
        "create_final_text_units",
    ]

    def __init__(
        self,
        tool_config: VectorstoreTool,
        base_dir: Path | None = None,
    ) -> None:
        """Initialize storage manager."""
        ...

    @property
    def storage_dir(self) -> Path:
        """Root storage directory for this tool."""
        ...

    @property
    def output_dir(self) -> Path:
        """Directory for GraphRAG parquet output."""
        ...

    def ensure_directories(self) -> None:
        """Create storage directories if needed."""
        ...

    def has_valid_index(self) -> bool:
        """Check if all required output tables exist."""
        ...

    def get_metadata(self) -> IndexMetadata | None:
        """Load index metadata if exists."""
        ...

    def save_metadata(self, metadata: IndexMetadata) -> None:
        """Save index metadata to disk."""
        ...

    def compute_source_hash(self, source_path: Path) -> str:
        """Compute hash of source files for change detection."""
        ...

    def needs_reindex(self, source_path: Path) -> bool:
        """Check if source files changed since last index."""
        ...

    def clear_index(self) -> None:
        """Delete all index artifacts."""
        ...

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a parquet table as DataFrame."""
        ...
```

---

## 4. GraphRAGIndexer Class

Orchestrates the indexing pipeline.

### 4.1 Class Definition

```python
class GraphRAGIndexer:
    """Orchestrates GraphRAG indexing pipeline."""

    def __init__(
        self,
        tool_config: VectorstoreTool,
        storage: GraphRAGStorage,
    ) -> None:
        """Initialize indexer."""
        ...

    async def build_index(
        self,
        source_path: Path,
        force: bool = False,
    ) -> IndexMetadata:
        """Build or update the GraphRAG index.

        Args:
            source_path: Path to source documents
            force: Force rebuild even if current

        Returns:
            Index metadata

        Raises:
            IndexingError: If indexing fails
        """
        ...
```

---

## 5. Config Generation Function

### 5.1 Function Definition

```python
def build_graphrag_config(
    tool_config: VectorstoreTool,
    storage_dir: Path,
    input_dir: Path,
) -> GraphRagConfig:
    """Build GraphRagConfig from HoloDeck configuration.

    Maps HoloDeck VectorstoreTool settings to GraphRAG's
    native GraphRagConfig format.

    Args:
        tool_config: VectorstoreTool with engine="graphrag"
        storage_dir: Directory for GraphRAG artifacts
        input_dir: Directory containing source documents

    Returns:
        GraphRagConfig ready for indexing/search

    Raises:
        ValueError: If required configuration missing
    """
    ...
```

---

## 6. Error Types

### 6.1 Custom Exceptions

```python
class IndexingError(Exception):
    """Raised when GraphRAG indexing fails."""
    pass
```

---

## 7. Integration Points

### 7.1 VectorStoreTool Modifications

```python
# In vectorstore_tool.py

class VectorStoreTool:
    _graphrag_engine: GraphRAGEngine | None = None

    async def initialize(
        self,
        force_ingest: bool = False,
        provider_type: str | None = None,
    ) -> None:
        # NEW: Branch on engine type
        if self.config.engine == "graphrag":
            await self._initialize_graphrag(force_ingest)
            return

        # Existing code for default engine...

    async def _initialize_graphrag(
        self,
        force_ingest: bool = False,
    ) -> None:
        """Initialize GraphRAG engine."""
        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                "GraphRAG requires: pip install holodeck[graphrag]"
            )

        from holodeck.lib.graphrag.engine import GraphRAGEngine

        self._graphrag_engine = GraphRAGEngine(
            config=self.config,
            base_dir=self._base_dir,
        )
        await self._graphrag_engine.initialize(force_reindex=force_ingest)
        self.is_initialized = True

    async def search(self, query: str) -> str:
        # NEW: Delegate to GraphRAG engine
        if self.config.engine == "graphrag":
            if self._graphrag_engine is None:
                raise RuntimeError("GraphRAG engine not initialized")
            return await self._graphrag_engine.search(query)

        # Existing search code...
```

---

## 8. YAML Configuration Schema

### 8.1 Minimal Configuration

```yaml
tools:
  - name: knowledge_graph
    type: vectorstore
    engine: graphrag
    source: data/documents/
    description: Search knowledge graph
```

### 8.2 Full Configuration

```yaml
tools:
  - name: entity_search
    type: vectorstore
    engine: graphrag
    source: data/company_docs/
    description: Search for entities and relationships

    graphrag:
      search_mode: local
      community_level: 2

      indexing_model:
        provider: openai
        name: gpt-4o-mini
        temperature: 0.0

      search_model:
        provider: openai
        name: gpt-4o
        temperature: 0.0

      entity_types:
        - organization
        - person
        - product
        - technology

      chunk_size: 300
      chunk_overlap: 100
      max_gleanings: 1
      skip_claim_extraction: true
```

---

## 9. Response Format

### 9.1 Search Response

```python
def _format_result(result: SearchResult) -> str:
    """Format GraphRAG search result for agent consumption."""

    lines = [
        f"**GraphRAG {result.search_mode.title()} Search Result**",
        "",
        result.response,
        "",
        "---",
        f"*Mode: {result.search_mode} | "
        f"Entities: {result.entities_used} | "
        f"Communities: {result.communities_used}*",
    ]

    if result.sources:
        lines.append("")
        lines.append("**Sources:**")
        for source in result.sources[:5]:
            lines.append(f"- {source}")

    return "\n".join(lines)
```

---

## 10. Thread Safety

All public methods are async and designed for single-threaded async execution within an agent loop. No special thread-safety considerations required.

---

## 11. Version Compatibility

| Component | Version Requirement |
|-----------|---------------------|
| graphrag | >=0.5.0,<1.0.0 |
| Python | >=3.10 |
| pandas | >=2.0 |
| pydantic | >=2.0 |
