# Data Model: GraphRAG Engine Integration

**Feature**: 016-graphrag-integration
**Date**: 2025-12-27
**Source**: Feature specification + GraphRAG API research

---

## 1. Configuration Entities

### 1.1 GraphRAGModelConfig

Settings for LLM model used in GraphRAG operations.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| provider | Literal["openai", "azure_openai"] | No | "openai" | LLM provider |
| name | str | No | "gpt-4o-mini" | Model name |
| temperature | float | No | 0.0 | Sampling temperature (0.0-2.0) |
| max_tokens | int \| None | No | None | Maximum tokens for response |
| api_base | str \| None | No | None | Azure OpenAI endpoint URL |
| api_version | str \| None | No | None | Azure API version |
| deployment_name | str \| None | No | None | Azure deployment name |

**Validation Rules**:
- `temperature` must be between 0.0 and 2.0
- `api_base`, `api_version`, `deployment_name` required when `provider` is "azure_openai"

---

### 1.2 GraphRAGConfig

GraphRAG-specific configuration nested within VectorstoreTool.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| search_mode | Literal["local", "global"] | No | "local" | Search strategy |
| community_level | int | No | 2 | Community hierarchy level (0-5) |
| indexing_model | GraphRAGModelConfig \| None | No | None | LLM for indexing |
| search_model | GraphRAGModelConfig \| None | No | None | LLM for search |
| embedding_model | str \| None | No | None | Embedding model name |
| storage_dir | str \| None | No | None | Custom storage path |
| chunk_size | int | No | 300 | Tokens per text unit |
| chunk_overlap | int | No | 100 | Token overlap between units |
| entity_types | list[str] \| None | No | None | Entity types to extract |
| max_gleanings | int | No | 1 | Additional extraction passes |
| skip_claim_extraction | bool | No | True | Skip covariate extraction |

**Validation Rules**:
- `community_level` must be between 0 and 5
- `chunk_size` must be between 50 and 2000
- `chunk_overlap` must be >= 0
- `max_gleanings` must be between 0 and 5

**Default Entity Types** (when `entity_types` is None):
- organization
- person
- location
- event
- concept

---

### 1.3 VectorstoreTool (Extended)

Existing VectorstoreTool model with new GraphRAG fields.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| *existing fields* | ... | ... | ... | ... |
| engine | Literal["default", "graphrag"] | No | "default" | Search engine type |
| graphrag | GraphRAGConfig \| None | No | None | GraphRAG settings |

**Validation Rules**:
- When `engine == "graphrag"`, create default `GraphRAGConfig` if `graphrag` is None
- When `engine != "graphrag"`, `graphrag` must be None (raise error if provided)

---

## 2. Runtime Entities

### 2.1 IndexMetadata

Metadata about a built GraphRAG index.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| tool_name | str | Yes | - | Vectorstore tool name |
| source_path | str | Yes | - | Absolute path to source |
| source_hash | str | Yes | - | SHA256 hash of source files |
| indexed_at | str | Yes | - | ISO 8601 timestamp |
| file_count | int | Yes | - | Number of source files |
| entity_count | int | No | 0 | Entities extracted |
| relationship_count | int | No | 0 | Relationships extracted |
| community_count | int | No | 0 | Communities detected |
| graphrag_version | str | No | "" | GraphRAG package version |
| config_hash | str | No | "" | Hash of GraphRAG config |

**State Transitions**:
- Created after successful indexing
- Updated on re-indexing
- Deleted when index is cleared

---

### 2.2 SearchResult

Result from a GraphRAG search query.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| response | str | Yes | - | Generated response text |
| sources | list[str] | No | [] | Source references |
| entities_used | int | No | 0 | Entities in context |
| communities_used | int | No | 0 | Communities in context |
| search_mode | str | Yes | - | "local" or "global" |

---

## 3. Domain Entities (From GraphRAG)

These entities are stored as parquet files and loaded for search operations.

### 3.1 Entity

A person, organization, location, event, or concept extracted from documents.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| title | str | Entity name |
| type | str \| None | Entity type |
| description | str \| None | Entity description |
| description_embedding | list[float] \| None | Semantic embedding |
| community_ids | list[str] \| None | Associated communities |
| text_unit_ids | list[str] \| None | Source text units |
| rank | int \| None | Importance rank |

---

### 3.2 Relationship

A connection between two entities.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| source | str | Source entity name |
| target | str | Target entity name |
| weight | float \| None | Edge weight |
| description | str \| None | Relationship description |
| text_unit_ids | list[str] \| None | Source text units |
| rank | int \| None | Importance rank |

---

### 3.3 Community

A cluster of related entities detected via Leiden algorithm.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| title | str | Community title |
| level | str | Hierarchy level |
| parent | str | Parent community ID |
| children | list[str] | Child community IDs |
| entity_ids | list[str] \| None | Member entity IDs |
| size | int \| None | Community size |

---

### 3.4 CommunityReport

LLM-generated summary of a community.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| title | str | Report title |
| community_id | str | Associated community |
| summary | str | Brief summary |
| full_content | str | Full report text |
| rank | float \| None | Importance rank |
| full_content_embedding | list[float] \| None | Semantic embedding |

---

### 3.5 TextUnit

A chunk of source document text.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier |
| text | str | Text content |
| entity_ids | list[str] \| None | Extracted entities |
| relationship_ids | list[str] \| None | Extracted relationships |
| n_tokens | int \| None | Token count |
| document_ids | list[str] \| None | Source documents |

---

## 4. Storage Layout

### 4.1 Directory Structure

```
.holodeck/graphrag/{tool_name}/
├── output/                           # GraphRAG parquet artifacts
│   ├── create_final_entities.parquet
│   ├── create_final_relationships.parquet
│   ├── create_final_communities.parquet
│   ├── create_final_community_reports.parquet
│   ├── create_final_text_units.parquet
│   └── create_final_documents.parquet
├── cache/                            # LLM response cache
├── logs/                             # Pipeline logs
├── lancedb/                          # Vector store for entity embeddings
├── settings.yaml                     # Generated GraphRAG config (optional)
└── index.meta                        # IndexMetadata JSON
```

### 4.2 Required Parquet Tables

These tables must exist for a valid index:

1. `create_final_entities` - Entity data
2. `create_final_relationships` - Relationship data
3. `create_final_communities` - Community hierarchy
4. `create_final_community_reports` - Community summaries
5. `create_final_text_units` - Source text chunks

---

## 5. Entity Relationships

```
┌─────────────────┐     ┌─────────────────┐
│ VectorstoreTool │────>│  GraphRAGConfig │
│     (YAML)      │1   1│    (nested)     │
└─────────────────┘     └─────────────────┘
         │                      │
         │ initialize()         │ generates
         ▼                      ▼
┌─────────────────┐     ┌─────────────────┐
│ GraphRAGEngine  │────>│  GraphRagConfig │
│    (runtime)    │1   1│   (graphrag)    │
└─────────────────┘     └─────────────────┘
         │
         │ builds/loads
         ▼
┌─────────────────┐
│  IndexMetadata  │
│   (persisted)   │
└─────────────────┘
         │
         │ tracks
         ▼
┌─────────────────┐     ┌─────────────────┐
│     Entity      │◆───◆│  Relationship   │
└─────────────────┘     └─────────────────┘
         │                      │
         │ belongs_to           │ connects
         ▼                      │
┌─────────────────┐             │
│   Community     │◄────────────┘
└─────────────────┘
         │
         │ has_report
         ▼
┌─────────────────┐
│CommunityReport  │
└─────────────────┘
```

---

## 6. Pydantic Model Definitions

### 6.1 GraphRAGModelConfig

```python
class GraphRAGModelConfig(BaseModel):
    """LLM configuration for GraphRAG operations."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "azure_openai"] = Field(
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
    api_base: str | None = Field(default=None)
    api_version: str | None = Field(default=None)
    deployment_name: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_azure_fields(self) -> "GraphRAGModelConfig":
        if self.provider == "azure_openai":
            if not self.api_base:
                raise ValueError("api_base required for azure_openai")
        return self
```

### 6.2 GraphRAGConfig

```python
class GraphRAGConfig(BaseModel):
    """Configuration for GraphRAG engine."""

    model_config = ConfigDict(extra="forbid")

    search_mode: Literal["local", "global"] = Field(
        default="local",
        description="Search mode"
    )
    community_level: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Community hierarchy level"
    )
    indexing_model: GraphRAGModelConfig | None = Field(default=None)
    search_model: GraphRAGModelConfig | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    storage_dir: str | None = Field(default=None)
    chunk_size: int = Field(default=300, ge=50, le=2000)
    chunk_overlap: int = Field(default=100, ge=0)
    entity_types: list[str] | None = Field(default=None)
    max_gleanings: int = Field(default=1, ge=0, le=5)
    skip_claim_extraction: bool = Field(default=True)
```

---

## 7. Migration Notes

### 7.1 New Fields on VectorstoreTool

- `engine`: New field with default "default" (backward compatible)
- `graphrag`: New optional nested config (ignored when engine != "graphrag")

### 7.2 No Breaking Changes

Existing VectorstoreTool configurations continue to work unchanged. The new fields have defaults that preserve existing behavior.
