# GraphRAG Integration Plan for HoloDeck

## Overview

Integrate Microsoft GraphRAG as an `engine: graphrag` option within HoloDeck's existing vectorstore tool type. This enables knowledge graph-based retrieval with entity extraction, community detection, and hierarchical summarization.

**GraphRAG vs Traditional RAG**:
| Aspect | Traditional RAG (current) | GraphRAG |
|--------|---------------------------|----------|
| Data Model | Flat chunks with embeddings | Entities, Relationships, Communities, Reports |
| Indexing | Embedding generation only | Multi-step LLM pipeline (entity extraction, community detection, summarization) |
| Query | Vector similarity | Graph traversal + LLM synthesis |
| Cost | Low (embedding API only) | High (LLM calls for indexing + querying) |
| Best For | Specific factual queries | Holistic analysis, relationship understanding |

---

## User Requirements

- **Integration**: Engine option in vectorstore (`engine: graphrag`)
- **Indexing**: On-demand during tool initialization
- **Query Modes**: Local + Global only (no DRIFT or Basic)
- **Reranking**: Keep separate from existing reranking extension

---

## Configuration Schema

### New Pydantic Models

Add to `src/holodeck/models/tool.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal


class GraphRAGModelConfig(BaseModel):
    """LLM configuration for GraphRAG operations."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "azure_openai", "ollama"] = Field(
        default="openai",
        description="LLM provider"
    )
    name: str = Field(
        default="gpt-4o-mini",
        description="Model name"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for response"
    )
    # Azure-specific
    api_base: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL"
    )
    api_version: str | None = Field(
        default=None,
        description="Azure API version"
    )
    deployment_name: str | None = Field(
        default=None,
        description="Azure deployment name"
    )


class GraphRAGConfig(BaseModel):
    """Configuration for GraphRAG engine."""

    model_config = ConfigDict(extra="forbid")

    search_mode: Literal["local", "global"] = Field(
        default="local",
        description=(
            "Search mode: 'local' for entity-centric queries, "
            "'global' for dataset-wide analytical queries"
        )
    )
    community_level: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Community hierarchy level for global search. "
            "Lower = more detailed communities, higher = broader summaries"
        )
    )
    indexing_model: GraphRAGModelConfig | None = Field(
        default=None,
        description=(
            "LLM for indexing (entity extraction, summarization). "
            "Recommend cheaper model like gpt-4o-mini. "
            "Defaults to agent's model if not specified."
        )
    )
    search_model: GraphRAGModelConfig | None = Field(
        default=None,
        description=(
            "LLM for search queries. "
            "Defaults to agent's model if not specified."
        )
    )
    embedding_model: str | None = Field(
        default=None,
        description=(
            "Embedding model for vector search. "
            "Defaults to tool's embedding_model or provider default."
        )
    )
    storage_dir: str | None = Field(
        default=None,
        description=(
            "Directory for GraphRAG artifacts. "
            "Defaults to .holodeck/graphrag/{tool_name}/"
        )
    )
    chunk_size: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Token count per text unit for indexing"
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Token overlap between text units"
    )
    entity_types: list[str] | None = Field(
        default=None,
        description=(
            "Entity types to extract. "
            "Defaults to ['organization', 'person', 'location', 'event', 'concept']"
        )
    )
    max_gleanings: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Number of additional LLM passes for entity extraction"
    )
    skip_claim_extraction: bool = Field(
        default=True,
        description="Skip claim/covariate extraction (faster indexing)"
    )


# Modify VectorstoreTool class to add:
class VectorstoreTool(BaseModel):
    """Vectorstore tool for semantic search over documents."""

    # ... existing fields ...

    # NEW FIELDS:
    engine: Literal["default", "graphrag"] = Field(
        default="default",
        description=(
            "Search engine: 'default' for traditional vector RAG, "
            "'graphrag' for knowledge graph-based RAG"
        )
    )
    graphrag: GraphRAGConfig | None = Field(
        default=None,
        description="GraphRAG-specific configuration (required when engine='graphrag')"
    )

    @model_validator(mode="after")
    def validate_graphrag_config(self) -> "VectorstoreTool":
        """Validate GraphRAG configuration."""
        if self.engine == "graphrag" and self.graphrag is None:
            # Create default config
            object.__setattr__(self, 'graphrag', GraphRAGConfig())
        if self.engine != "graphrag" and self.graphrag is not None:
            raise ValueError(
                "graphrag configuration is only valid when engine='graphrag'"
            )
        return self
```

### YAML Configuration Examples

**Local Search (Entity-Centric)**:
```yaml
tools:
  - name: entity_search
    type: vectorstore
    engine: graphrag
    source: data/company_docs/
    description: "Search for specific entities and their relationships"

    graphrag:
      search_mode: local
      indexing_model:
        provider: openai
        name: gpt-4o-mini    # Cheaper model for indexing
        temperature: 0.0
      search_model:
        provider: openai
        name: gpt-4o         # Better model for search
      entity_types:
        - organization
        - person
        - product
        - technology
```

**Global Search (Analytical)**:
```yaml
tools:
  - name: theme_analyzer
    type: vectorstore
    engine: graphrag
    source: data/research_papers/
    description: "Analyze themes and patterns across the dataset"

    graphrag:
      search_mode: global
      community_level: 1     # More detailed communities
      indexing_model:
        provider: azure_openai
        name: gpt-4o-mini
        api_base: ${AZURE_OPENAI_ENDPOINT}
        deployment_name: gpt-4o-mini
```

---

## File Structure

### New Files to Create

```
src/holodeck/lib/graphrag/
├── __init__.py           # Package exports + dependency check
├── config.py             # HoloDeck config → GraphRagConfig conversion
├── storage.py            # Artifact storage & index metadata management
├── indexer.py            # Indexing pipeline orchestration
├── search.py             # Local/Global search engine wrappers
└── engine.py             # Main GraphRAGEngine class
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/holodeck/models/tool.py` | Add `engine`, `GraphRAGConfig`, `GraphRAGModelConfig` |
| `src/holodeck/tools/vectorstore_tool.py` | Add engine branching in `initialize()` and `search()` |
| `src/holodeck/lib/test_runner/agent_factory.py` | Handle GraphRAG chat model injection |
| `pyproject.toml` | Add optional `graphrag` dependency group |

---

## Implementation Details

### File 1: `src/holodeck/lib/graphrag/__init__.py`

```python
"""GraphRAG integration for HoloDeck vectorstore tools.

This module provides knowledge graph-based retrieval augmented generation
using Microsoft's GraphRAG library.

The integration supports:
- Local Search: Entity-centric queries with relationship traversal
- Global Search: Dataset-wide analytical queries using community hierarchies

Example:
    >>> from holodeck.lib.graphrag import GraphRAGEngine, GRAPHRAG_AVAILABLE
    >>> if GRAPHRAG_AVAILABLE:
    ...     engine = GraphRAGEngine(config)
    ...     await engine.initialize()
    ...     result = await engine.search("What are the main themes?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


def _check_graphrag_available() -> bool:
    """Check if graphrag package is installed."""
    try:
        import graphrag  # noqa: F401
        return True
    except ImportError:
        return False


GRAPHRAG_AVAILABLE = _check_graphrag_available()
"""Whether the graphrag package is available for import."""


def require_graphrag() -> None:
    """Raise ImportError if graphrag is not installed.

    Raises:
        ImportError: If graphrag package is not installed.
    """
    if not GRAPHRAG_AVAILABLE:
        raise ImportError(
            "GraphRAG engine requires the 'graphrag' package. "
            "Install with: pip install holodeck[graphrag] "
            "or: pip install graphrag"
        )


if TYPE_CHECKING:
    from holodeck.lib.graphrag.engine import GraphRAGEngine
    from holodeck.lib.graphrag.config import generate_graphrag_config
    from holodeck.lib.graphrag.storage import GraphRAGStorage
    from holodeck.lib.graphrag.indexer import GraphRAGIndexer
    from holodeck.lib.graphrag.search import LocalSearchEngine, GlobalSearchEngine


__all__ = [
    "GRAPHRAG_AVAILABLE",
    "require_graphrag",
    "GraphRAGEngine",
    "GraphRAGStorage",
    "GraphRAGIndexer",
    "LocalSearchEngine",
    "GlobalSearchEngine",
    "generate_graphrag_config",
]


def __getattr__(name: str):
    """Lazy import GraphRAG components."""
    if name in __all__:
        require_graphrag()
        if name == "GraphRAGEngine":
            from holodeck.lib.graphrag.engine import GraphRAGEngine
            return GraphRAGEngine
        elif name == "GraphRAGStorage":
            from holodeck.lib.graphrag.storage import GraphRAGStorage
            return GraphRAGStorage
        elif name == "GraphRAGIndexer":
            from holodeck.lib.graphrag.indexer import GraphRAGIndexer
            return GraphRAGIndexer
        elif name == "LocalSearchEngine":
            from holodeck.lib.graphrag.search import LocalSearchEngine
            return LocalSearchEngine
        elif name == "GlobalSearchEngine":
            from holodeck.lib.graphrag.search import GlobalSearchEngine
            return GlobalSearchEngine
        elif name == "generate_graphrag_config":
            from holodeck.lib.graphrag.config import generate_graphrag_config
            return generate_graphrag_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

### File 2: `src/holodeck/lib/graphrag/config.py`

```python
"""GraphRAG configuration generation and management.

This module handles conversion between HoloDeck configuration and
GraphRAG's native GraphRagConfig format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.models.tool import VectorstoreTool, GraphRAGConfig, GraphRAGModelConfig

import logging

logger = logging.getLogger(__name__)


# Default entity types for extraction
DEFAULT_ENTITY_TYPES = [
    "organization",
    "person",
    "location",
    "event",
    "concept",
]


def _resolve_api_key(provider: str) -> str | None:
    """Resolve API key from environment based on provider.

    Args:
        provider: LLM provider name

    Returns:
        API key string or None if not found
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "ollama": None,  # Ollama doesn't need API key
    }
    env_var = env_vars.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def _build_model_config(
    model_cfg: GraphRAGModelConfig | None,
    fallback_provider: str = "openai",
    fallback_model: str = "gpt-4o-mini",
    is_embedding: bool = False,
) -> dict[str, Any]:
    """Build GraphRAG model configuration dictionary.

    Args:
        model_cfg: HoloDeck model configuration
        fallback_provider: Default provider if not specified
        fallback_model: Default model name if not specified
        is_embedding: Whether this is for embedding model

    Returns:
        Configuration dict for GraphRAG
    """
    if model_cfg is None:
        model_cfg_dict = {
            "provider": fallback_provider,
            "name": fallback_model,
        }
    else:
        model_cfg_dict = {
            "provider": model_cfg.provider,
            "name": model_cfg.name,
            "temperature": model_cfg.temperature,
        }
        if model_cfg.max_tokens:
            model_cfg_dict["max_tokens"] = model_cfg.max_tokens

    # Handle Azure-specific settings
    if model_cfg_dict.get("provider") == "azure_openai":
        if model_cfg and model_cfg.api_base:
            model_cfg_dict["api_base"] = model_cfg.api_base
        else:
            model_cfg_dict["api_base"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

        if model_cfg and model_cfg.api_version:
            model_cfg_dict["api_version"] = model_cfg.api_version
        else:
            model_cfg_dict["api_version"] = os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
            )

        if model_cfg and model_cfg.deployment_name:
            model_cfg_dict["deployment_name"] = model_cfg.deployment_name
        else:
            model_cfg_dict["deployment_name"] = model_cfg_dict["name"]

    # Add API key
    api_key = _resolve_api_key(model_cfg_dict.get("provider", "openai"))
    if api_key:
        model_cfg_dict["api_key"] = api_key

    return model_cfg_dict


def generate_graphrag_config(
    tool_config: VectorstoreTool,
    storage_dir: Path,
    input_dir: Path,
) -> dict[str, Any]:
    """Generate GraphRAG configuration from HoloDeck tool config.

    Args:
        tool_config: VectorstoreTool configuration
        storage_dir: Directory for GraphRAG output artifacts
        input_dir: Directory containing input documents

    Returns:
        Configuration dictionary for GraphRAG

    Raises:
        ValueError: If required configuration is missing
    """
    graphrag_cfg = tool_config.graphrag
    if graphrag_cfg is None:
        raise ValueError("GraphRAG configuration is required")

    # Build chat model configs
    indexing_model = _build_model_config(
        graphrag_cfg.indexing_model,
        fallback_model="gpt-4o-mini",
    )
    search_model = _build_model_config(
        graphrag_cfg.search_model,
        fallback_model="gpt-4o",
    )

    # Build embedding model config
    embedding_model_name = (
        graphrag_cfg.embedding_model
        or tool_config.embedding_model
        or "text-embedding-3-small"
    )
    embedding_model = _build_model_config(
        None,
        fallback_model=embedding_model_name,
        is_embedding=True,
    )
    embedding_model["type"] = "openai_embedding"

    # Entity types
    entity_types = graphrag_cfg.entity_types or DEFAULT_ENTITY_TYPES

    # Build the full config
    config = {
        "root_dir": str(storage_dir),
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": str(input_dir),
            "file_pattern": ".*\\.(txt|md|pdf)$",
            "encoding": "utf-8",
        },
        "cache": {
            "type": "file",
            "base_dir": str(storage_dir / "cache"),
        },
        "storage": {
            "type": "file",
            "base_dir": str(storage_dir / "output"),
        },
        "reporting": {
            "type": "file",
            "base_dir": str(storage_dir / "logs"),
        },
        "models": {
            "default_chat_model": indexing_model,
            "default_embedding_model": embedding_model,
        },
        "chunks": {
            "size": graphrag_cfg.chunk_size,
            "overlap": graphrag_cfg.chunk_overlap,
            "group_by_columns": ["id"],
            "encoding_model": "cl100k_base",
        },
        "extract_graph": {
            "enabled": True,
            "prompt": None,  # Use default
            "entity_types": entity_types,
            "max_gleanings": graphrag_cfg.max_gleanings,
        },
        "cluster_graph": {
            "enabled": True,
            "max_cluster_size": 10,
            "strategy": {"type": "leiden"},
        },
        "embed_graph": {
            "enabled": True,
            "strategy": {"type": "node2vec"},
        },
        "community_reports": {
            "enabled": True,
            "prompt": None,  # Use default
            "max_length": 2000,
            "max_input_length": 8000,
        },
        "claim_extraction": {
            "enabled": not graphrag_cfg.skip_claim_extraction,
        },
        "local_search": {
            "chat_model": search_model,
            "embedding_model": embedding_model,
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "top_k_entities": 10,
            "top_k_relationships": 10,
            "max_tokens": 12000,
        },
        "global_search": {
            "chat_model": search_model,
            "map_max_tokens": 1000,
            "reduce_max_tokens": 2000,
            "concurrency": 32,
            "dynamic_community_selection": False,
            "dynamic_community_selection_kwargs": {
                "community_level": graphrag_cfg.community_level,
                "max_level": 5,
                "min_level": 0,
            },
        },
    }

    return config


def save_graphrag_config(config: dict[str, Any], path: Path) -> None:
    """Save GraphRAG configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved GraphRAG configuration to {path}")


def load_graphrag_config(path: Path) -> dict[str, Any]:
    """Load GraphRAG configuration from YAML file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"GraphRAG configuration not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)
```

---

### File 3: `src/holodeck/lib/graphrag/storage.py`

```python
"""GraphRAG artifact storage management.

Handles storage of GraphRAG index artifacts, metadata tracking,
and incremental update detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.models.tool import VectorstoreTool

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """Metadata about a GraphRAG index."""

    tool_name: str
    source_path: str
    source_hash: str
    indexed_at: str  # ISO format
    file_count: int
    entity_count: int = 0
    relationship_count: int = 0
    community_count: int = 0
    graphrag_version: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexMetadata":
        """Create from dictionary."""
        return cls(**data)


class GraphRAGStorage:
    """Manages GraphRAG artifact storage and index metadata.

    Storage layout:
        .holodeck/graphrag/{tool_name}/
        ├── output/                    # GraphRAG parquet artifacts
        │   ├── create_final_entities/
        │   ├── create_final_relationships/
        │   ├── create_final_communities/
        │   ├── create_final_community_reports/
        │   └── create_final_text_units/
        ├── cache/                     # LLM call cache
        ├── logs/                      # Pipeline logs
        ├── settings.yaml              # Generated GraphRAG config
        └── index.meta                 # Index metadata

    Attributes:
        storage_dir: Root directory for this tool's GraphRAG artifacts
        tool_name: Name of the vectorstore tool
    """

    METADATA_FILE = "index.meta"
    CONFIG_FILE = "settings.yaml"
    OUTPUT_DIR = "output"
    CACHE_DIR = "cache"
    LOGS_DIR = "logs"

    # Required output tables for a valid index
    REQUIRED_TABLES = [
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
        """Initialize GraphRAG storage manager.

        Args:
            tool_config: VectorstoreTool configuration
            base_dir: Base directory for HoloDeck artifacts.
                Defaults to current working directory.
        """
        self.tool_name = tool_config.name
        self._tool_config = tool_config

        # Resolve storage directory
        if tool_config.graphrag and tool_config.graphrag.storage_dir:
            self.storage_dir = Path(tool_config.graphrag.storage_dir)
        else:
            base = base_dir or Path.cwd()
            self.storage_dir = base / ".holodeck" / "graphrag" / self.tool_name

    @property
    def output_dir(self) -> Path:
        """Directory for GraphRAG output artifacts."""
        return self.storage_dir / self.OUTPUT_DIR

    @property
    def cache_dir(self) -> Path:
        """Directory for LLM call cache."""
        return self.storage_dir / self.CACHE_DIR

    @property
    def logs_dir(self) -> Path:
        """Directory for pipeline logs."""
        return self.storage_dir / self.LOGS_DIR

    @property
    def config_path(self) -> Path:
        """Path to GraphRAG configuration file."""
        return self.storage_dir / self.CONFIG_FILE

    @property
    def metadata_path(self) -> Path:
        """Path to index metadata file."""
        return self.storage_dir / self.METADATA_FILE

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        logger.debug(f"Ensured GraphRAG storage directories at {self.storage_dir}")

    def has_valid_index(self) -> bool:
        """Check if a valid index exists.

        Returns:
            True if all required output tables exist
        """
        if not self.output_dir.exists():
            return False

        for table_name in self.REQUIRED_TABLES:
            table_dir = self.output_dir / table_name
            if not table_dir.exists():
                logger.debug(f"Missing required table: {table_name}")
                return False
            # Check for parquet files
            parquet_files = list(table_dir.glob("*.parquet"))
            if not parquet_files:
                logger.debug(f"No parquet files in {table_name}")
                return False

        return True

    def get_metadata(self) -> IndexMetadata | None:
        """Load index metadata if it exists.

        Returns:
            IndexMetadata or None if not found
        """
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path) as f:
                data = json.load(f)
            return IndexMetadata.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return None

    def save_metadata(self, metadata: IndexMetadata) -> None:
        """Save index metadata.

        Args:
            metadata: Index metadata to save
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.debug(f"Saved index metadata to {self.metadata_path}")

    def compute_source_hash(self, source_path: Path) -> str:
        """Compute hash of source files for change detection.

        Args:
            source_path: Path to source file or directory

        Returns:
            SHA256 hash of source content metadata
        """
        hasher = hashlib.sha256()

        if source_path.is_file():
            files = [source_path]
        else:
            files = sorted(source_path.rglob("*"))

        for file_path in files:
            if file_path.is_file():
                # Hash file path, size, and mtime
                stat = file_path.stat()
                file_info = f"{file_path.relative_to(source_path)}:{stat.st_size}:{stat.st_mtime}"
                hasher.update(file_info.encode())

        return hasher.hexdigest()[:16]

    def needs_reindex(self, source_path: Path) -> bool:
        """Check if source files have changed since last index.

        Args:
            source_path: Path to source file or directory

        Returns:
            True if reindexing is needed
        """
        if not self.has_valid_index():
            logger.info("No valid index found, indexing required")
            return True

        metadata = self.get_metadata()
        if metadata is None:
            logger.info("No index metadata found, indexing required")
            return True

        current_hash = self.compute_source_hash(source_path)
        if current_hash != metadata.source_hash:
            logger.info(
                f"Source files changed (hash: {current_hash} vs {metadata.source_hash})"
            )
            return True

        logger.info(f"Index is current (indexed at {metadata.indexed_at})")
        return False

    def clear_index(self) -> None:
        """Clear all index artifacts.

        Use with caution - this deletes all GraphRAG data for this tool.
        """
        import shutil

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleared index artifacts at {self.output_dir}")

        if self.metadata_path.exists():
            self.metadata_path.unlink()

        # Recreate directories
        self.ensure_directories()

    def get_table_path(self, table_name: str) -> Path:
        """Get path to a specific output table.

        Args:
            table_name: Name of the table (e.g., 'create_final_entities')

        Returns:
            Path to table directory
        """
        return self.output_dir / table_name

    def load_table(self, table_name: str) -> "pd.DataFrame":
        """Load a parquet table as DataFrame.

        Args:
            table_name: Name of the table to load

        Returns:
            pandas DataFrame

        Raises:
            FileNotFoundError: If table doesn't exist
        """
        import pandas as pd

        table_path = self.get_table_path(table_name)
        if not table_path.exists():
            raise FileNotFoundError(f"Table not found: {table_name}")

        parquet_files = list(table_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in: {table_name}")

        # Load all parquet files and concatenate
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
```

---

### File 4: `src/holodeck/lib/graphrag/indexer.py`

```python
"""GraphRAG indexing pipeline orchestration.

Wraps GraphRAG's build_index API with HoloDeck-specific handling
for configuration, progress tracking, and error management.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.models.tool import VectorstoreTool
    from holodeck.lib.graphrag.storage import GraphRAGStorage, IndexMetadata

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when GraphRAG indexing fails."""

    pass


class IndexingCallbacks:
    """Callbacks for GraphRAG indexing progress tracking."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.current_workflow: str | None = None
        self.workflows_completed: list[str] = []

    def on_workflow_start(self, name: str, instance: Any = None) -> None:
        """Called when a workflow starts."""
        self.current_workflow = name
        logger.info(f"[{self.tool_name}] Starting workflow: {name}")

    def on_workflow_end(
        self, name: str, instance: Any = None, result: Any = None
    ) -> None:
        """Called when a workflow completes."""
        self.workflows_completed.append(name)
        logger.info(f"[{self.tool_name}] Completed workflow: {name}")

    def on_step_start(self, step: str, info: dict | None = None) -> None:
        """Called when a workflow step starts."""
        logger.debug(f"[{self.tool_name}] Step: {step}")

    def on_step_end(
        self, step: str, info: dict | None = None, result: Any = None
    ) -> None:
        """Called when a workflow step completes."""
        pass

    def on_error(
        self, message: str, cause: BaseException | None = None, stack: str | None = None
    ) -> None:
        """Called on error."""
        logger.error(f"[{self.tool_name}] Error: {message}")
        if cause:
            logger.error(f"[{self.tool_name}] Cause: {cause}")

    def on_warning(self, message: str) -> None:
        """Called on warning."""
        logger.warning(f"[{self.tool_name}] Warning: {message}")

    def on_log(self, message: str) -> None:
        """Called for general log messages."""
        logger.debug(f"[{self.tool_name}] {message}")


class GraphRAGIndexer:
    """Orchestrates GraphRAG indexing pipeline.

    Handles:
    - Configuration generation and validation
    - Index building via graphrag.api.build_index
    - Progress tracking and logging
    - Error handling and recovery
    - Metadata tracking for incremental updates

    Example:
        >>> indexer = GraphRAGIndexer(tool_config, storage)
        >>> await indexer.build_index(source_path, force=False)
    """

    def __init__(
        self,
        tool_config: VectorstoreTool,
        storage: GraphRAGStorage,
    ) -> None:
        """Initialize the indexer.

        Args:
            tool_config: VectorstoreTool configuration
            storage: GraphRAG storage manager
        """
        self.tool_config = tool_config
        self.storage = storage
        self._callbacks = IndexingCallbacks(tool_config.name)

    async def build_index(
        self,
        source_path: Path,
        force: bool = False,
    ) -> IndexMetadata:
        """Build or update the GraphRAG index.

        Args:
            source_path: Path to source documents
            force: Force rebuild even if index is current

        Returns:
            Index metadata

        Raises:
            IndexingError: If indexing fails
        """
        from holodeck.lib.graphrag.config import (
            generate_graphrag_config,
            save_graphrag_config,
        )
        from holodeck.lib.graphrag.storage import IndexMetadata

        # Check if reindexing is needed
        if not force and not self.storage.needs_reindex(source_path):
            metadata = self.storage.get_metadata()
            if metadata:
                logger.info(
                    f"Using existing index for {self.tool_config.name} "
                    f"(indexed at {metadata.indexed_at})"
                )
                return metadata

        logger.info(f"Building GraphRAG index for {self.tool_config.name}")

        # Ensure storage directories exist
        self.storage.ensure_directories()

        # Generate and save configuration
        config_dict = generate_graphrag_config(
            self.tool_config,
            self.storage.storage_dir,
            source_path,
        )
        save_graphrag_config(config_dict, self.storage.config_path)

        # Import GraphRAG components
        try:
            from graphrag.api import build_index
            from graphrag.config import load_config
        except ImportError as e:
            raise IndexingError(
                "GraphRAG package not found. "
                "Install with: pip install holodeck[graphrag]"
            ) from e

        # Load the generated config
        try:
            config = load_config(str(self.storage.storage_dir))
        except Exception as e:
            raise IndexingError(f"Failed to load GraphRAG config: {e}") from e

        # Build the index
        try:
            logger.info(f"Starting GraphRAG indexing pipeline for {source_path}")
            start_time = datetime.now()

            # Run the indexing pipeline
            results = await build_index(
                config=config,
                is_update_run=False,  # Full rebuild for now
                memory_profile=False,
                callbacks=[self._create_workflow_callbacks()],
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"GraphRAG indexing completed in {elapsed:.1f}s "
                f"({len(results)} workflows)"
            )

        except Exception as e:
            raise IndexingError(f"GraphRAG indexing failed: {e}") from e

        # Load stats from output
        stats = self._load_index_stats()

        # Create and save metadata
        metadata = IndexMetadata(
            tool_name=self.tool_config.name,
            source_path=str(source_path),
            source_hash=self.storage.compute_source_hash(source_path),
            indexed_at=datetime.now().isoformat(),
            file_count=self._count_source_files(source_path),
            entity_count=stats.get("entities", 0),
            relationship_count=stats.get("relationships", 0),
            community_count=stats.get("communities", 0),
        )
        self.storage.save_metadata(metadata)

        return metadata

    def _create_workflow_callbacks(self) -> Any:
        """Create GraphRAG workflow callbacks.

        Returns:
            WorkflowCallbacks-compatible object
        """
        from graphrag.callbacks import WorkflowCallbacks

        class HoloDeckCallbacks(WorkflowCallbacks):
            def __init__(inner_self):
                inner_self._parent = self._callbacks

            def on_workflow_start(inner_self, name, instance=None):
                self._callbacks.on_workflow_start(name, instance)

            def on_workflow_end(inner_self, name, instance=None, result=None):
                self._callbacks.on_workflow_end(name, instance, result)

            def on_step_start(inner_self, step, details=None):
                self._callbacks.on_step_start(step, details)

            def on_step_end(inner_self, step, details=None, result=None):
                self._callbacks.on_step_end(step, details, result)

            def on_error(inner_self, message, cause=None, stack=None, details=None):
                self._callbacks.on_error(message, cause, stack)

            def on_warning(inner_self, message, details=None):
                self._callbacks.on_warning(message)

            def on_log(inner_self, message, details=None):
                self._callbacks.on_log(message)

        return HoloDeckCallbacks()

    def _count_source_files(self, source_path: Path) -> int:
        """Count source files."""
        if source_path.is_file():
            return 1
        return len(list(source_path.rglob("*")))

    def _load_index_stats(self) -> dict[str, int]:
        """Load statistics from indexed output tables.

        Returns:
            Dict with entity, relationship, community counts
        """
        stats = {}

        try:
            entities_df = self.storage.load_table("create_final_entities")
            stats["entities"] = len(entities_df)
        except FileNotFoundError:
            stats["entities"] = 0

        try:
            rels_df = self.storage.load_table("create_final_relationships")
            stats["relationships"] = len(rels_df)
        except FileNotFoundError:
            stats["relationships"] = 0

        try:
            communities_df = self.storage.load_table("create_final_communities")
            stats["communities"] = len(communities_df)
        except FileNotFoundError:
            stats["communities"] = 0

        return stats
```

---

### File 5: `src/holodeck/lib/graphrag/search.py`

```python
"""GraphRAG search engines for Local and Global search modes.

Wraps GraphRAG's search APIs with HoloDeck-specific handling
for result formatting and error management.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.models.tool import VectorstoreTool
    from holodeck.lib.graphrag.storage import GraphRAGStorage
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a GraphRAG search query."""

    response: str
    """The generated response text."""

    sources: list[str]
    """List of source references used."""

    entities_used: int
    """Number of entities used in context."""

    communities_used: int
    """Number of communities used in context."""

    search_mode: str
    """Search mode used ('local' or 'global')."""

    def __str__(self) -> str:
        return self.response


class BaseSearchEngine(ABC):
    """Abstract base class for GraphRAG search engines."""

    def __init__(
        self,
        tool_config: VectorstoreTool,
        storage: GraphRAGStorage,
    ) -> None:
        """Initialize search engine.

        Args:
            tool_config: VectorstoreTool configuration
            storage: GraphRAG storage manager
        """
        self.tool_config = tool_config
        self.storage = storage
        self._is_initialized = False

    @abstractmethod
    async def search(self, query: str) -> SearchResult:
        """Execute a search query.

        Args:
            query: The search query

        Returns:
            SearchResult with response and metadata

        Raises:
            RuntimeError: If engine not initialized
            SearchError: If search fails
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the search engine by loading index data."""
        pass

    def _ensure_initialized(self) -> None:
        """Raise if not initialized."""
        if not self._is_initialized:
            raise RuntimeError(
                f"Search engine for {self.tool_config.name} not initialized. "
                "Call initialize() first."
            )


class LocalSearchEngine(BaseSearchEngine):
    """Local search engine for entity-centric queries.

    Local search finds entities semantically related to the query,
    then traverses relationships to build rich context.

    Best for:
    - "Who is [person] and what do they do?"
    - "What companies are related to [organization]?"
    - "Explain the relationship between X and Y"
    """

    def __init__(
        self,
        tool_config: VectorstoreTool,
        storage: GraphRAGStorage,
    ) -> None:
        super().__init__(tool_config, storage)
        self._entities: list[Any] = []
        self._relationships: list[Any] = []
        self._text_units: list[Any] = []
        self._community_reports: list[Any] = []
        self._communities: list[Any] = []
        self._covariates: dict[str, list[Any]] = {}
        self._description_embedding_store: Any = None

    async def initialize(self) -> None:
        """Load index data for local search."""
        from graphrag.query.indexer_adapters import (
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_text_units,
            read_indexer_reports,
            read_indexer_communities,
        )
        from graphrag.vector_stores import VectorStoreFactory

        logger.info(f"Initializing local search for {self.tool_config.name}")

        # Load data tables
        entities_df = self.storage.load_table("create_final_entities")
        relationships_df = self.storage.load_table("create_final_relationships")
        text_units_df = self.storage.load_table("create_final_text_units")
        communities_df = self.storage.load_table("create_final_communities")
        reports_df = self.storage.load_table("create_final_community_reports")

        # Convert to domain models
        self._communities = read_indexer_communities(communities_df, reports_df)
        self._entities = read_indexer_entities(entities_df, communities_df)
        self._relationships = read_indexer_relationships(relationships_df)
        self._text_units = read_indexer_text_units(text_units_df)
        self._community_reports = read_indexer_reports(reports_df, communities_df)

        # Initialize vector store for entity embeddings
        self._description_embedding_store = VectorStoreFactory.create_vector_store(
            vector_store_type="lancedb",
            collection_name="entity_description_embeddings",
        )
        self._description_embedding_store.connect(
            db_uri=str(self.storage.storage_dir / "lancedb")
        )

        # Load entity embeddings if available
        if "description_embedding" in entities_df.columns:
            from graphrag.vector_stores import VectorStoreDocument

            documents = [
                VectorStoreDocument(
                    id=row["id"],
                    text=row.get("description", ""),
                    vector=row["description_embedding"],
                    attributes={"title": row.get("title", "")},
                )
                for _, row in entities_df.iterrows()
                if row.get("description_embedding") is not None
            ]
            if documents:
                self._description_embedding_store.load_documents(documents)

        self._is_initialized = True
        logger.info(
            f"Local search initialized: {len(self._entities)} entities, "
            f"{len(self._relationships)} relationships"
        )

    async def search(self, query: str) -> SearchResult:
        """Execute local search.

        Args:
            query: Search query

        Returns:
            SearchResult with entity-focused response
        """
        self._ensure_initialized()

        from graphrag.query.factory import get_local_search_engine
        from graphrag.config import load_config

        # Load config for search parameters
        config = load_config(str(self.storage.storage_dir))

        # Get search engine
        search_engine = get_local_search_engine(
            config=config,
            reports=self._community_reports,
            text_units=self._text_units,
            entities=self._entities,
            relationships=self._relationships,
            covariates=self._covariates,
            description_embedding_store=self._description_embedding_store,
        )

        # Execute search
        result = await search_engine.search(query)

        # Format result
        return SearchResult(
            response=str(result.response),
            sources=self._extract_sources(result.context_data),
            entities_used=len(self._entities),  # Approximate
            communities_used=0,
            search_mode="local",
        )

    def _extract_sources(self, context_data: Any) -> list[str]:
        """Extract source references from context data."""
        sources = []
        if hasattr(context_data, "get"):
            if "sources" in context_data:
                sources = context_data["sources"]
        return sources


class GlobalSearchEngine(BaseSearchEngine):
    """Global search engine for dataset-wide analytical queries.

    Global search uses map-reduce over community reports to synthesize
    insights across the entire dataset.

    Best for:
    - "What are the main themes in this dataset?"
    - "What are the key patterns across all documents?"
    - "Summarize the major findings"
    """

    def __init__(
        self,
        tool_config: VectorstoreTool,
        storage: GraphRAGStorage,
    ) -> None:
        super().__init__(tool_config, storage)
        self._community_reports: list[Any] = []
        self._entities: list[Any] = []
        self._communities: list[Any] = []

    async def initialize(self) -> None:
        """Load index data for global search."""
        from graphrag.query.indexer_adapters import (
            read_indexer_entities,
            read_indexer_reports,
            read_indexer_communities,
        )

        logger.info(f"Initializing global search for {self.tool_config.name}")

        # Load data tables
        entities_df = self.storage.load_table("create_final_entities")
        communities_df = self.storage.load_table("create_final_communities")
        reports_df = self.storage.load_table("create_final_community_reports")

        # Convert to domain models
        self._communities = read_indexer_communities(communities_df, reports_df)
        self._entities = read_indexer_entities(entities_df, communities_df)
        self._community_reports = read_indexer_reports(reports_df, communities_df)

        # Filter reports by community level
        graphrag_cfg = self.tool_config.graphrag
        target_level = graphrag_cfg.community_level if graphrag_cfg else 2

        self._community_reports = [
            r for r in self._community_reports
            if getattr(r, "level", 0) <= target_level
        ]

        self._is_initialized = True
        logger.info(
            f"Global search initialized: {len(self._community_reports)} reports "
            f"at level <= {target_level}"
        )

    async def search(self, query: str) -> SearchResult:
        """Execute global search.

        Args:
            query: Search query

        Returns:
            SearchResult with dataset-wide insights
        """
        self._ensure_initialized()

        from graphrag.query.factory import get_global_search_engine
        from graphrag.config import load_config

        # Load config for search parameters
        config = load_config(str(self.storage.storage_dir))

        # Get search engine
        search_engine = get_global_search_engine(
            config=config,
            reports=self._community_reports,
            entities=self._entities,
            communities=self._communities,
        )

        # Execute search
        result = await search_engine.search(query)

        return SearchResult(
            response=str(result.response),
            sources=[],
            entities_used=len(self._entities),
            communities_used=len(self._community_reports),
            search_mode="global",
        )


def create_search_engine(
    tool_config: VectorstoreTool,
    storage: GraphRAGStorage,
) -> BaseSearchEngine:
    """Factory function to create appropriate search engine.

    Args:
        tool_config: VectorstoreTool configuration
        storage: GraphRAG storage manager

    Returns:
        LocalSearchEngine or GlobalSearchEngine based on config
    """
    graphrag_cfg = tool_config.graphrag
    search_mode = graphrag_cfg.search_mode if graphrag_cfg else "local"

    if search_mode == "global":
        return GlobalSearchEngine(tool_config, storage)
    else:
        return LocalSearchEngine(tool_config, storage)
```

---

### File 6: `src/holodeck/lib/graphrag/engine.py`

```python
"""Main GraphRAG engine class for HoloDeck integration.

Provides a unified interface for GraphRAG initialization and search,
matching the VectorStoreTool interface pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.models.tool import VectorstoreTool
    from holodeck.models.config import ExecutionConfig

logger = logging.getLogger(__name__)


class GraphRAGEngine:
    """GraphRAG engine for knowledge graph-based retrieval.

    This engine provides:
    - Automatic index building on first initialization
    - Incremental updates when source files change
    - Local search for entity-centric queries
    - Global search for dataset-wide analytical queries

    The engine matches VectorStoreTool's interface pattern for
    seamless integration.

    Attributes:
        config: VectorstoreTool configuration
        is_initialized: Whether the engine has been initialized
        index_stats: Statistics about the built index

    Example:
        >>> config = VectorstoreTool(
        ...     name="knowledge_graph",
        ...     type="vectorstore",
        ...     engine="graphrag",
        ...     source="data/docs/",
        ...     graphrag=GraphRAGConfig(search_mode="local"),
        ... )
        >>> engine = GraphRAGEngine(config)
        >>> await engine.initialize()
        >>> result = await engine.search("Who is the CEO?")
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
            base_dir: Base directory for resolving relative paths
            execution_config: Execution configuration (unused, for interface compat)
        """
        from holodeck.lib.graphrag import require_graphrag
        require_graphrag()

        if config.engine != "graphrag":
            raise ValueError(
                f"GraphRAGEngine requires engine='graphrag', got '{config.engine}'"
            )

        self.config = config
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._execution_config = execution_config

        # Lazy-initialized components
        self._storage: GraphRAGStorage | None = None
        self._indexer: GraphRAGIndexer | None = None
        self._search_engine: BaseSearchEngine | None = None
        self._is_initialized = False
        self._index_metadata: IndexMetadata | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether the engine has been initialized."""
        return self._is_initialized

    @property
    def index_stats(self) -> dict[str, int]:
        """Statistics about the built index."""
        if self._index_metadata is None:
            return {}
        return {
            "entities": self._index_metadata.entity_count,
            "relationships": self._index_metadata.relationship_count,
            "communities": self._index_metadata.community_count,
            "files": self._index_metadata.file_count,
        }

    async def initialize(
        self,
        force_reindex: bool = False,
    ) -> None:
        """Initialize the GraphRAG engine.

        This method:
        1. Resolves source path
        2. Sets up storage
        3. Builds or loads index
        4. Initializes search engine

        Args:
            force_reindex: Force rebuild of index even if current
        """
        from holodeck.lib.graphrag.storage import GraphRAGStorage
        from holodeck.lib.graphrag.indexer import GraphRAGIndexer
        from holodeck.lib.graphrag.search import create_search_engine

        if self._is_initialized:
            logger.debug(f"GraphRAG engine {self.config.name} already initialized")
            return

        logger.info(f"Initializing GraphRAG engine: {self.config.name}")

        # Resolve source path
        source_path = Path(self.config.source)
        if not source_path.is_absolute():
            source_path = self._base_dir / source_path

        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Initialize storage
        self._storage = GraphRAGStorage(self.config, self._base_dir)

        # Initialize indexer and build/load index
        self._indexer = GraphRAGIndexer(self.config, self._storage)
        self._index_metadata = await self._indexer.build_index(
            source_path,
            force=force_reindex,
        )

        # Initialize search engine
        self._search_engine = create_search_engine(self.config, self._storage)
        await self._search_engine.initialize()

        self._is_initialized = True
        logger.info(
            f"GraphRAG engine initialized: {self._index_metadata.entity_count} entities, "
            f"{self._index_metadata.relationship_count} relationships, "
            f"{self._index_metadata.community_count} communities"
        )

    async def search(self, query: str) -> str:
        """Search the knowledge graph.

        Args:
            query: The search query

        Returns:
            Formatted search results as string

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self._is_initialized or self._search_engine is None:
            raise RuntimeError(
                f"GraphRAG engine {self.config.name} not initialized. "
                "Call initialize() first."
            )

        logger.debug(f"GraphRAG search [{self.config.name}]: {query}")

        result = await self._search_engine.search(query)

        # Format result for agent consumption
        return self._format_result(result)

    def _format_result(self, result: SearchResult) -> str:
        """Format search result for agent consumption.

        Args:
            result: SearchResult from search engine

        Returns:
            Formatted string
        """
        from holodeck.lib.graphrag.search import SearchResult

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
            for source in result.sources[:5]:  # Limit to 5 sources
                lines.append(f"- {source}")

        return "\n".join(lines)

    async def close(self) -> None:
        """Clean up resources.

        Call this when done with the engine to release any
        open connections or file handles.
        """
        # Currently no cleanup needed, but provided for interface compat
        self._is_initialized = False
        logger.debug(f"GraphRAG engine {self.config.name} closed")


# Import types for type checking
if TYPE_CHECKING:
    from holodeck.lib.graphrag.storage import GraphRAGStorage, IndexMetadata
    from holodeck.lib.graphrag.indexer import GraphRAGIndexer
    from holodeck.lib.graphrag.search import BaseSearchEngine, SearchResult
```

---

## Modifications to Existing Files

### 1. `src/holodeck/tools/vectorstore_tool.py`

Add engine branching in `__init__`, `initialize()`, and `search()`:

```python
# At the top of the file, add import guard:
from holodeck.lib.graphrag import GRAPHRAG_AVAILABLE

# In VectorStoreTool.__init__(), add:
self._graphrag_engine: GraphRAGEngine | None = None

# Add new method:
async def _initialize_graphrag(
    self,
    force_ingest: bool = False,
) -> None:
    """Initialize GraphRAG engine.

    Args:
        force_ingest: Force rebuild of index
    """
    if not GRAPHRAG_AVAILABLE:
        raise ImportError(
            "GraphRAG engine requires the 'graphrag' package. "
            "Install with: pip install holodeck[graphrag]"
        )

    from holodeck.lib.graphrag.engine import GraphRAGEngine

    self._graphrag_engine = GraphRAGEngine(
        config=self.config,
        base_dir=self._base_dir,
        execution_config=self._execution_config,
    )
    await self._graphrag_engine.initialize(force_reindex=force_ingest)
    self.is_initialized = True

# Modify initialize() to add engine branching:
async def initialize(
    self,
    force_ingest: bool = False,
    provider_type: str = "openai",
) -> None:
    """Initialize the tool by ingesting files and preparing for search."""
    if self.is_initialized:
        return

    # NEW: Branch on engine type
    if self.config.engine == "graphrag":
        await self._initialize_graphrag(force_ingest)
        return

    # Existing code for default engine...
    if self._is_structured_mode():
        await self._initialize_structured(force_ingest, provider_type)
    else:
        # ... existing unstructured initialization ...

# Modify search() to delegate to GraphRAG:
async def search(self, query: str) -> str:
    """Search for relevant content.

    Args:
        query: Search query

    Returns:
        Formatted search results
    """
    if not self.is_initialized:
        raise RuntimeError("Tool not initialized")

    # NEW: Delegate to GraphRAG engine
    if self.config.engine == "graphrag" and self._graphrag_engine is not None:
        return await self._graphrag_engine.search(query)

    # Existing search code...
    query_embedding = await self._generate_embedding(query)
    # ...
```

### 2. `src/holodeck/lib/test_runner/agent_factory.py`

Handle GraphRAG tools (no changes to embedding service, but add chat model availability):

```python
# In _register_vectorstore_tools(), add check for GraphRAG:
async def _register_vectorstore_tools(self) -> None:
    """Discover, initialize, and register vectorstore tools."""
    for tool_config in self.agent_config.tools:
        if not isinstance(tool_config, VectorstoreTool):
            continue

        # Create tool instance
        tool = VectorStoreTool(
            tool_config,
            base_dir=str(self._project_dir),
            execution_config=self._execution_config,
        )

        # GraphRAG tools don't need embedding service injection
        # (they handle their own embeddings internally)
        if tool_config.engine != "graphrag":
            tool.set_embedding_service(self._embedding_service)

        # Initialize
        await tool.initialize(
            force_ingest=self._force_ingest,
            provider_type=provider_type,
        )

        # ... rest of registration code ...
```

### 3. `pyproject.toml`

Add optional dependency:

```toml
[project.optional-dependencies]
graphrag = [
    "graphrag>=0.5.0,<1.0.0",
]

# Update the "all" extras to include graphrag
all = [
    "holodeck[graphrag]",
    # ... other extras ...
]
```

---

## Testing Plan

### Unit Tests

Create `tests/unit/lib/graphrag/test_config.py`:

```python
import pytest
from holodeck.models.tool import VectorstoreTool, GraphRAGConfig


@pytest.mark.unit
def test_graphrag_config_defaults():
    """Test GraphRAGConfig default values."""
    config = GraphRAGConfig()
    assert config.search_mode == "local"
    assert config.community_level == 2
    assert config.indexing_model is None


@pytest.mark.unit
def test_vectorstore_tool_with_graphrag_engine():
    """Test VectorstoreTool with graphrag engine."""
    tool = VectorstoreTool(
        name="test_graphrag",
        type="vectorstore",
        source="data/docs/",
        description="Test GraphRAG tool",
        engine="graphrag",
    )
    assert tool.engine == "graphrag"
    assert tool.graphrag is not None  # Auto-created default


@pytest.mark.unit
def test_vectorstore_tool_graphrag_requires_engine():
    """Test that graphrag config without engine fails."""
    with pytest.raises(ValueError, match="only valid when engine='graphrag'"):
        VectorstoreTool(
            name="test",
            type="vectorstore",
            source="data/",
            description="Test",
            engine="default",
            graphrag=GraphRAGConfig(),  # Not allowed with default engine
        )
```

Create `tests/unit/lib/graphrag/test_storage.py`:

```python
import pytest
from pathlib import Path
from holodeck.lib.graphrag.storage import GraphRAGStorage, IndexMetadata


@pytest.mark.unit
def test_storage_directory_structure(tmp_path):
    """Test storage creates correct directory structure."""
    # Create mock config
    from holodeck.models.tool import VectorstoreTool, GraphRAGConfig

    config = VectorstoreTool(
        name="test_storage",
        type="vectorstore",
        source="data/",
        description="Test",
        engine="graphrag",
        graphrag=GraphRAGConfig(storage_dir=str(tmp_path / "graphrag")),
    )

    storage = GraphRAGStorage(config)
    storage.ensure_directories()

    assert storage.output_dir.exists()
    assert storage.cache_dir.exists()
    assert storage.logs_dir.exists()


@pytest.mark.unit
def test_index_metadata_serialization():
    """Test IndexMetadata to/from dict."""
    metadata = IndexMetadata(
        tool_name="test",
        source_path="/data/docs",
        source_hash="abc123",
        indexed_at="2024-01-01T00:00:00",
        file_count=10,
        entity_count=100,
    )

    data = metadata.to_dict()
    restored = IndexMetadata.from_dict(data)

    assert restored.tool_name == "test"
    assert restored.entity_count == 100
```

### Integration Tests

Create `tests/integration/test_graphrag_integration.py`:

```python
import pytest
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("graphrag", reason="graphrag not installed"),
    reason="graphrag package required",
)
async def test_graphrag_local_search(tmp_path):
    """Test GraphRAG local search end-to-end."""
    from holodeck.models.tool import VectorstoreTool, GraphRAGConfig
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.txt").write_text(
        "John Smith is the CEO of Acme Corp. "
        "Acme Corp is a technology company based in San Francisco."
    )

    # Create tool config
    config = VectorstoreTool(
        name="test_graphrag",
        type="vectorstore",
        source=str(docs_dir),
        description="Test GraphRAG",
        engine="graphrag",
        graphrag=GraphRAGConfig(
            search_mode="local",
            storage_dir=str(tmp_path / "graphrag"),
        ),
    )

    # Initialize and search
    tool = VectorStoreTool(config)
    await tool.initialize()

    result = await tool.search("Who is the CEO of Acme Corp?")
    assert "John Smith" in result or "CEO" in result
```

---

## Error Handling Matrix

| Error | Location | Behavior |
|-------|----------|----------|
| GraphRAG not installed | `__init__.py` | ImportError with install instructions |
| Invalid engine value | `tool.py` validator | ValidationError |
| Missing source path | `engine.py` | FileNotFoundError |
| LLM API key missing | `indexer.py` | IndexingError with message |
| Indexing pipeline fails | `indexer.py` | IndexingError with cause |
| Search before initialize | `engine.py` | RuntimeError |
| Invalid community level | `tool.py` validator | ValidationError |
| Parquet load fails | `storage.py` | FileNotFoundError |

---

## Performance Considerations

1. **Indexing Cost**
   - GraphRAG indexing is LLM-intensive (~75% of cost is entity extraction)
   - Recommend `gpt-4o-mini` for indexing, `gpt-4o` for search
   - Use LLM caching (enabled by default in storage/cache/)

2. **Index Persistence**
   - Index artifacts stored in `.holodeck/graphrag/`
   - Incremental updates based on source file hash
   - Clear with `storage.clear_index()` if needed

3. **Search Latency**
   - Local search: Fast (entity vector lookup + relationship traversal)
   - Global search: Slower (map-reduce over community reports)
   - `community_level` affects global search speed (lower = more reports)

4. **Memory Usage**
   - Entities, relationships, and reports loaded into memory
   - For large datasets (>100k entities), consider external vector store

---

## Summary

This plan provides complete implementation details for integrating GraphRAG as an engine option in HoloDeck's vectorstore tool. The implementation:

1. **Extends existing configuration** with `engine: graphrag` and nested `graphrag` config
2. **Creates isolated module** at `lib/graphrag/` with 6 files
3. **Matches existing patterns** for initialization and search
4. **Handles optional dependency** gracefully
5. **Supports both search modes** (local and global)
6. **Tracks index metadata** for incremental updates

Total files to create: 6
Total files to modify: 4
Estimated implementation effort: 13 tasks across 4 phases
