# GraphRAG Integration Research

**Feature**: 016-graphrag-integration
**Date**: 2025-12-27
**Status**: Complete

## Executive Summary

This research document captures the findings from analyzing the Microsoft GraphRAG library (v0.5+) to understand the exact API integration requirements for HoloDeck. The integration will add `engine: graphrag` as an option in the existing `VectorstoreTool` configuration.

---

## 1. GraphRAG API Architecture

### 1.1 Main API Entry Points

**Location**: `graphrag/api/__init__.py`

```python
# Indexing
from graphrag.api.index import build_index

# Query (all search modes)
from graphrag.api.query import (
    local_search, local_search_streaming,
    global_search, global_search_streaming,
    # NOT implementing: drift_search, basic_search (per spec)
)
```

**Decision**: Use `build_index` for indexing and direct `local_search`/`global_search` API functions (not factory pattern)

**Rationale**: The direct API functions handle all data loading internally and provide cleaner error handling than the factory approach used in the original claude-plan.md.

**Alternatives considered**:
- Factory pattern (`get_local_search_engine`, `get_global_search_engine`) - More complex, requires manual data loading
- CLI wrapper - Too much overhead, not suitable for programmatic integration

---

## 2. Build Index API

**Location**: `graphrag/api/index.py:29-96`

### 2.1 Function Signature

```python
async def build_index(
    config: GraphRagConfig,
    method: IndexingMethod | str = IndexingMethod.Standard,
    is_update_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    additional_context: dict[str, Any] | None = None,
    verbose: bool = False,
    input_documents: pd.DataFrame | None = None,
) -> list[PipelineRunResult]:
```

### 2.2 Key Parameters

| Parameter | Usage | HoloDeck Mapping |
|-----------|-------|------------------|
| `config` | GraphRagConfig instance | Generated from HoloDeck config |
| `method` | `IndexingMethod.Standard` or `IndexingMethod.Fast` | Default to Standard |
| `is_update_run` | Incremental update mode | `storage.needs_reindex()` check |
| `callbacks` | Progress tracking | Custom `WorkflowCallbacks` impl |
| `input_documents` | Override document loading | None (use file-based input) |

### 2.3 IndexingMethod Enum

**Location**: `graphrag/config/enums.py`

```python
class IndexingMethod(str, Enum):
    Standard = "standard"  # Full LLM-based extraction
    Fast = "fast"          # NLP + lightweight LLM (faster, cheaper)
```

**Decision**: Use `Standard` for accurate entity extraction; consider `Fast` as future optimization

---

## 3. Search APIs

### 3.1 Local Search

**Location**: `graphrag/api/query.py:342-406`

```python
async def local_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    text_units: pd.DataFrame,
    relationships: pd.DataFrame,
    covariates: pd.DataFrame | None,
    community_level: int,
    response_type: str,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[str | dict | list[dict], str | list[pd.DataFrame] | dict[str, pd.DataFrame]]:
```

### 3.2 Global Search

**Location**: `graphrag/api/query.py:64-124`

```python
async def global_search(
    config: GraphRagConfig,
    entities: pd.DataFrame,
    communities: pd.DataFrame,
    community_reports: pd.DataFrame,
    community_level: int | None,
    dynamic_community_selection: bool,
    response_type: str,
    query: str,
    callbacks: list[QueryCallbacks] | None = None,
    verbose: bool = False,
) -> tuple[str | dict | list[dict], str | list[pd.DataFrame] | dict[str, pd.DataFrame]]:
```

### 3.3 Data Loading for Search

**Location**: `graphrag/query/indexer_adapters.py`

Required DataFrames must be loaded from parquet files:

```python
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_text_units,
    read_indexer_reports,
    read_indexer_communities,
    read_indexer_covariates,  # Optional, for claims
)
```

**Decision**: Load parquet files directly using pandas, then convert via adapters

---

## 4. Configuration System

### 4.1 GraphRagConfig

**Location**: `graphrag/config/models/graph_rag_config.py`

The main configuration class contains:

```python
class GraphRagConfig(BaseModel):
    root_dir: str  # Root directory for all paths
    models: dict[str, LanguageModelConfig]  # Must include "default_chat_model" and "default_embedding_model"
    input: InputConfig
    chunks: ChunkingConfig
    output: StorageConfig
    cache: CacheConfig
    reporting: ReportingConfig
    vector_store: dict[str, VectorStoreConfig]
    extract_graph: ExtractGraphConfig
    community_reports: CommunityReportsConfig
    local_search: LocalSearchConfig
    global_search: GlobalSearchConfig
    # ... more optional configs
```

### 4.2 LanguageModelConfig

**Location**: `graphrag/config/models/language_model_config.py`

```python
class LanguageModelConfig(BaseModel):
    type: ModelType | str  # "Chat", "Embedding", or legacy types
    model: str  # Model name (e.g., "gpt-4o", "gpt-4o-mini")
    model_provider: str | None  # "openai", "azure", etc.
    api_key: str | None
    api_base: str | None  # Required for Azure
    api_version: str | None  # Required for Azure
    deployment_name: str | None  # For Azure
    temperature: float = 0.0
    max_tokens: int | None = None
    # Rate limiting
    tokens_per_minute: int | None
    requests_per_minute: int | None
    concurrent_requests: int = 25
    # Retry
    max_retries: int = 10
    retry_strategy: str = "exponential_backoff"
```

### 4.3 Configuration Loading

**Location**: `graphrag/config/load_config.py`

```python
def load_config(
    root_dir: Path,
    config_filepath: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> GraphRagConfig:
```

Also available programmatically:

```python
from graphrag.config.create_graphrag_config import create_graphrag_config

config = create_graphrag_config(values=config_dict, root_dir=str(root_path))
```

**Decision**: Use `create_graphrag_config()` to build config from HoloDeck settings

---

## 5. Storage & Output Structure

### 5.1 Output Directory Layout

After indexing, GraphRAG produces parquet files in the output directory:

```
{root_dir}/output/
├── create_final_entities.parquet
├── create_final_relationships.parquet
├── create_final_communities.parquet
├── create_final_community_reports.parquet
├── create_final_text_units.parquet
├── create_final_covariates.parquet  # If claims enabled
└── create_final_documents.parquet
```

### 5.2 Vector Store Configuration

**Location**: `graphrag/config/models/vector_store_config.py`

```python
class VectorStoreConfig(BaseModel):
    type: str  # "lancedb", "azure_ai_search", "cosmos_db"
    db_uri: str | None  # For LanceDB
    url: str | None  # For Azure AI Search
    container_name: str
    overwrite: bool = True
```

**Decision**: Use LanceDB as default vector store for entity embeddings (lightweight, file-based)

---

## 6. Data Models

### 6.1 Entity

**Location**: `graphrag/data_model/entity.py`

```python
@dataclass
class Entity:
    id: str
    title: str
    type: str | None
    description: str | None
    description_embedding: list[float] | None
    community_ids: list[str] | None
    text_unit_ids: list[str] | None
    rank: int | None
    attributes: dict[str, Any] | None
```

### 6.2 Relationship

**Location**: `graphrag/data_model/relationship.py`

```python
@dataclass
class Relationship:
    id: str
    source: str  # Source entity name
    target: str  # Target entity name
    weight: float | None
    description: str | None
    text_unit_ids: list[str] | None
    rank: int | None
```

### 6.3 Community

**Location**: `graphrag/data_model/community.py`

```python
@dataclass
class Community:
    id: str
    title: str
    level: str  # Hierarchy level (0 = most detailed)
    parent: str
    children: list[str]
    entity_ids: list[str] | None
    size: int | None
```

### 6.4 CommunityReport

**Location**: `graphrag/data_model/community_report.py`

```python
@dataclass
class CommunityReport:
    id: str
    title: str
    community_id: str
    summary: str
    full_content: str
    rank: float | None
    full_content_embedding: list[float] | None
```

---

## 7. Callbacks System

### 7.1 WorkflowCallbacks (Indexing)

**Location**: `graphrag/callbacks/workflow_callbacks.py`

```python
class WorkflowCallbacks(Protocol):
    def pipeline_start(self, names: list[str]) -> None: ...
    def pipeline_end(self, results: list[PipelineRunResult]) -> None: ...
    def workflow_start(self, name: str, instance: object) -> None: ...
    def workflow_end(self, name: str, instance: object) -> None: ...
    def progress(self, progress: Progress) -> None: ...
```

### 7.2 QueryCallbacks (Search)

**Location**: `graphrag/callbacks/query_callbacks.py`

```python
class QueryCallbacks:
    def on_context(self, context: Any) -> None: ...
    def on_map_response_start(self, contexts: list[str]) -> None: ...
    def on_map_response_end(self, outputs: list[SearchResult]) -> None: ...
    def on_reduce_response_start(self, context: str | dict) -> None: ...
    def on_reduce_response_end(self, output: str) -> None: ...
    def on_llm_new_token(self, token) -> None: ...
```

**Decision**: Implement custom callbacks for progress logging; no streaming support initially

---

## 8. Integration Architecture

### 8.1 HoloDeck Integration Points

Based on analysis of `src/holodeck/tools/vectorstore_tool.py`:

1. **Configuration** (`src/holodeck/models/tool.py`):
   - Add `engine: Literal["default", "graphrag"]` to `VectorstoreTool`
   - Add `graphrag: GraphRAGConfig | None` for GraphRAG-specific settings

2. **Engine Branching** (`src/holodeck/tools/vectorstore_tool.py`):
   - In `initialize()`: Branch on `config.engine == "graphrag"`
   - In `search()`: Delegate to GraphRAG engine

3. **New Module** (`src/holodeck/lib/graphrag/`):
   - `__init__.py`: Lazy imports, availability check
   - `config.py`: HoloDeck → GraphRagConfig conversion
   - `storage.py`: Index metadata and artifact management
   - `indexer.py`: Build index orchestration
   - `search.py`: Local/Global search wrappers
   - `engine.py`: Main GraphRAGEngine class

### 8.2 Revised Module Structure

```
src/holodeck/lib/graphrag/
├── __init__.py           # GRAPHRAG_AVAILABLE, require_graphrag(), lazy exports
├── config.py             # build_graphrag_config() function
├── storage.py            # GraphRAGStorage class
├── indexer.py            # GraphRAGIndexer class
├── search.py             # LocalSearch, GlobalSearch wrappers
└── engine.py             # GraphRAGEngine (main interface)
```

---

## 9. Key Implementation Decisions

### 9.1 Config Generation vs File-Based

**Decision**: Generate `GraphRagConfig` programmatically, not from YAML file

**Rationale**:
- Avoids filesystem overhead of writing settings.yaml
- Enables dynamic configuration based on HoloDeck settings
- GraphRAG's `create_graphrag_config()` supports dict input

### 9.2 Parquet Storage Location

**Decision**: Store in `.holodeck/graphrag/{tool_name}/output/`

**Rationale**:
- Keeps GraphRAG artifacts separate from traditional vectorstore
- `.holodeck/` is already the project's artifact directory
- Per-tool isolation prevents conflicts

### 9.3 Model Configuration Mapping

**Decision**: Map HoloDeck's `graphrag.indexing_model` and `graphrag.search_model` to GraphRAG's model system

| HoloDeck Config | GraphRAG Config |
|-----------------|-----------------|
| `graphrag.indexing_model.provider` | `models.default_chat_model.model_provider` |
| `graphrag.indexing_model.name` | `models.default_chat_model.model` |
| `graphrag.search_model.provider` | Used in search API calls |
| `embedding_model` | `models.default_embedding_model.model` |

### 9.4 Search Result Format

**Decision**: Return markdown-formatted response from GraphRAG

```python
def _format_result(self, result: SearchResult) -> str:
    return f"""**GraphRAG {result.search_mode.title()} Search Result**

{result.response}

---
*Mode: {result.search_mode} | Entities: {result.entities_used} | Communities: {result.communities_used}*
"""
```

**Rationale**: Matches existing VectorStoreTool output format

---

## 10. Dependency Management

### 10.1 Optional Dependency

**Decision**: Make `graphrag` an optional dependency

```toml
# pyproject.toml
[project.optional-dependencies]
graphrag = ["graphrag>=0.5.0,<1.0.0"]
```

### 10.2 Import Guard

```python
def _check_graphrag_available() -> bool:
    try:
        import graphrag
        return True
    except ImportError:
        return False

GRAPHRAG_AVAILABLE = _check_graphrag_available()
```

---

## 11. Error Handling Matrix

| Error Scenario | Detection Point | User Message |
|----------------|-----------------|--------------|
| GraphRAG not installed | `require_graphrag()` | "Install with: pip install holodeck[graphrag]" |
| Invalid engine value | Pydantic validator | "engine must be 'default' or 'graphrag'" |
| Missing source path | `engine.initialize()` | "Source path not found: {path}" |
| LLM API key missing | GraphRagConfig validation | "API key required for {provider}" |
| Indexing fails | `build_index()` exception | "GraphRAG indexing failed: {error}" |
| Search before init | `engine.search()` | "Call initialize() first" |
| Invalid community_level | Pydantic validator | "community_level must be 0-5" |

---

## 12. Testing Strategy

### 12.1 Unit Tests

- `test_config.py`: Config generation and validation
- `test_storage.py`: Index metadata persistence
- `test_engine.py`: Mock-based engine tests

### 12.2 Integration Tests

- `test_graphrag_local_search.py`: End-to-end local search
- `test_graphrag_global_search.py`: End-to-end global search
- `test_graphrag_incremental.py`: Index update detection

### 12.3 Test Fixtures

```python
@pytest.fixture
def sample_corpus(tmp_path):
    """Create minimal test corpus."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "test.txt").write_text(
        "John Smith is CEO of Acme Corp. "
        "Acme Corp is based in San Francisco."
    )
    return docs
```

---

## 13. Performance Considerations

### 13.1 Indexing Costs

- ~75% of cost is entity extraction LLM calls
- Recommend `gpt-4o-mini` for indexing (~$0.15/1M input tokens)
- Cache enabled by default in `.holodeck/graphrag/{tool}/cache/`

### 13.2 Search Latency

- Local search: ~1-3s (entity vector lookup + LLM synthesis)
- Global search: ~5-15s (map-reduce over community reports)
- `community_level` affects global search (lower = more reports = slower)

### 13.3 Memory Usage

- Entities/relationships loaded as DataFrames
- For >100k entities, consider chunked loading (future optimization)

---

## 14. Breaking Changes from Claude-Plan

Deviations from the original `claude-plan.md`:

1. **No Factory Pattern**: Using direct API functions instead of `get_local_search_engine` factory
2. **No settings.yaml**: Generating config programmatically
3. **Simplified Callbacks**: Logging-only callbacks, no streaming progress UI
4. **Single Engine Class**: Merged indexer/search into `GraphRAGEngine`
5. **Updated Model Config**: Using new LiteLLM-based config (GraphRAG v0.5+)

---

## References

- GraphRAG Repository: https://github.com/microsoft/graphrag
- GraphRAG Documentation: https://microsoft.github.io/graphrag/
- Explored Source: `/tmp/graphrag-explore/graphrag/`
