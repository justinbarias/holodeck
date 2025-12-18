# Data Model: Structured Data Field Mapping and Ingestion

**Feature**: 014-structured-data-ingestion
**Date**: 2025-12-18
**Status**: Complete

## Overview

This document defines the data models for structured data ingestion into vector stores. The key innovation is the `DynamicStructuredRecord` class factory that generates Semantic Kernel-compatible records with user-defined metadata fields.

## Entity Relationship Diagram

```
┌─────────────────────────┐     ┌─────────────────────────┐
│    VectorstoreTool      │     │   StructuredDataLoader  │
│   (Configuration)       │     │   (Runtime Processing)  │
├─────────────────────────┤     ├─────────────────────────┤
│ name: str               │     │ source_path: str        │
│ source: str             │────▶│ file_type: FileType     │
│ id_field: str           │     │ id_field: str           │
│ vector_field: str|list  │     │ vector_fields: list[str]│
│ metadata_fields: list   │     │ metadata_fields: list   │
│ field_separator: str    │     │ field_separator: str    │
│ delimiter: str (CSV)    │     │ delimiter: str          │
│ database: DatabaseConfig│     └───────────┬─────────────┘
└─────────────────────────┘                 │
                                            │ generates
                                            ▼
                          ┌─────────────────────────────────┐
                          │    DynamicStructuredRecord      │
                          │    (Vector Store Record)        │
                          ├─────────────────────────────────┤
                          │ id: str (key)                   │
                          │ content: str (full-text)        │
                          │ embedding: list[float] (vector) │
                          │ source_file: str (indexed)      │
                          │ [dynamic metadata fields...]    │
                          └─────────────────────────────────┘
                                            │
                                            │ stored in
                                            ▼
                          ┌─────────────────────────────────┐
                          │    Vector Store Collection      │
                          │ (ChromaDB, Qdrant, Postgres...) │
                          └─────────────────────────────────┘
```

## Core Entities

### 1. VectorstoreTool Configuration (Extended)

The existing `VectorstoreTool` model in `src/holodeck/models/tool.py` already has many fields. For structured data, we use/extend:

```python
class VectorstoreTool(BaseModel):
    """Vectorstore tool configuration - extended for structured data."""

    # Existing fields (unchanged)
    name: str
    description: str
    type: Literal["vectorstore"] = "vectorstore"
    source: str  # File path or directory
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    database: DatabaseConfig | str | None = None
    top_k: int = 5
    min_similarity_score: float | None = None

    # Structured data fields (existing but now required for structured mode)
    vector_field: str | list[str] | None = None  # Field(s) to embed
    meta_fields: list[str] | None = None  # Metadata fields to include

    # NEW fields for structured data
    id_field: str | None = None  # Required for structured data
    field_separator: str = "\n"  # Separator for multiple vector_fields
    delimiter: str | None = None  # CSV delimiter (auto-detect if None)
    record_path: str | None = None  # JSON array path (e.g., "data.items")

    @model_validator(mode="after")
    def validate_structured_config(self) -> "VectorstoreTool":
        """Validate structured data configuration."""
        # If any structured-specific field is set, id_field becomes required
        if self.vector_field is not None and self.id_field is None:
            raise ValueError(
                "id_field is required when vector_field is specified "
                "(structured data mode)"
            )
        return self
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id_field` | str | Yes (structured) | Field containing unique record identifier |
| `vector_field` | str \| list[str] | Yes (structured) | Field(s) to embed for semantic search |
| `meta_fields` | list[str] | No | Fields to include in results (default: all) |
| `field_separator` | str | No | Separator when concatenating multiple vector_fields (default: newline) |
| `delimiter` | str | No | CSV delimiter (auto-detect if not specified) |
| `record_path` | str | No | JSON path to array of records (e.g., "data.items") |

### 2. DynamicStructuredRecord

A runtime-generated Semantic Kernel vector store record class.

```python
# Factory function signature
def create_structured_record_class(
    dimensions: int = 1536,
    metadata_field_names: list[str] | None = None,
    collection_name: str = "structured_records",
) -> type[Any]:
    """
    Create a StructuredRecord class compatible with Semantic Kernel vector stores.

    Args:
        dimensions: Embedding vector dimensions
        metadata_field_names: Names of metadata fields to include
        collection_name: Vector store collection name

    Returns:
        A dataclass type with @vectorstoremodel decorator applied
    """
```

**Generated Class Structure**:

```python
@vectorstoremodel(collection_name="structured_records")
@dataclass
class DynamicStructuredRecord:
    # Fixed fields (always present)
    id: Annotated[str, VectorStoreField("key")]
    content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    embedding: Annotated[list[float] | None, VectorStoreField("vector", dimensions=N)]
    source_file: Annotated[str, VectorStoreField("data", is_indexed=True)]

    # Dynamic metadata fields (generated based on configuration)
    # Example: if metadata_field_names = ["title", "category", "price"]
    title: Annotated[str, VectorStoreField("data", is_indexed=True)]
    category: Annotated[str, VectorStoreField("data", is_indexed=True)]
    price: Annotated[str, VectorStoreField("data")]  # Not indexed by default
```

**Field Type Mapping**:

| Source Type | Record Field Type | Notes |
|-------------|-------------------|-------|
| String | str | Direct mapping |
| Integer | str | Converted to string |
| Float | str | Converted to string |
| Boolean | str | "true" or "false" |
| List/Array | str | JSON-serialized |
| Object/Dict | str | JSON-serialized |
| Null | str | Empty string "" |

### 3. StructuredDataLoader

Responsible for loading and validating structured data from various sources.

```python
class StructuredDataLoader:
    """Load and iterate over records from structured data sources."""

    def __init__(
        self,
        source_path: str,
        id_field: str,
        vector_fields: list[str],
        metadata_fields: list[str] | None = None,
        field_separator: str = "\n",
        delimiter: str | None = None,
        record_path: str | None = None,
        batch_size: int = 10_000,
    ):
        """
        Initialize the loader.

        Args:
            source_path: Path to data file
            id_field: Field containing unique record ID
            vector_fields: Fields to embed
            metadata_fields: Fields to include as metadata (None = all)
            field_separator: Separator for concatenating vector fields
            delimiter: CSV delimiter (auto-detect if None)
            record_path: JSON path to records array
            batch_size: Number of records per batch for large files
        """

    def validate_schema(self) -> list[str]:
        """
        Validate configuration against source schema.

        Returns:
            List of available field names

        Raises:
            ConfigError: If configured fields don't exist in source
        """

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over records from the source.

        Yields:
            Dict with keys: id, content, metadata (dict of field values)
        """

    def iter_batches(self) -> Iterator[list[dict[str, Any]]]:
        """
        Iterate over batches of records.

        Yields:
            List of record dicts (up to batch_size)
        """
```

### 4. StructuredQueryResult

Search result with dynamic metadata fields.

```python
@dataclass
class StructuredQueryResult:
    """Search result from structured data vector store query."""

    id: str  # Original record ID from id_field
    content: str  # Concatenated vector field content
    score: float  # Similarity score (0.0-1.0)
    source_file: str  # Original source file path
    metadata: dict[str, Any]  # All metadata field values

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
```

## State Transitions

### Record Ingestion States

```
┌─────────────┐
│   Source    │
│   File      │
└──────┬──────┘
       │ load
       ▼
┌─────────────┐
│  Validated  │  ← Schema validation (fields exist)
│   Records   │
└──────┬──────┘
       │ transform
       ▼
┌─────────────┐
│ Structured  │  ← ID extracted, content concatenated
│   Records   │
└──────┬──────┘
       │ embed
       ▼
┌─────────────┐
│  Embedded   │  ← Embeddings generated
│   Records   │
└──────┬──────┘
       │ upsert
       ▼
┌─────────────┐
│   Stored    │  ← In vector database
│   Records   │
└─────────────┘
```

## Validation Rules

### Configuration Validation

| Rule | Condition | Error Message |
|------|-----------|---------------|
| V1 | `id_field` required when `vector_field` set | "id_field is required for structured data mode" |
| V2 | `vector_field` XOR `vector_fields` | "Cannot specify both vector_field and vector_fields" |
| V3 | Fields must exist in source | "Field '{name}' not found. Available: {fields}" |
| V4 | `id_field` must have unique values | "Duplicate ID found: {id}" |
| V5 | Vector field(s) cannot be empty for all records | "No non-empty values found for vector field(s)" |

### Runtime Validation

| Rule | Condition | Behavior |
|------|-----------|----------|
| R1 | Record has null/empty vector fields | Skip with warning log |
| R2 | Record has duplicate ID | Update existing (upsert) |
| R3 | Field value is non-string | Convert to string |
| R4 | Nested field path invalid | Return empty string |

## Example Data

### CSV Example (`products.csv`)

```csv
id,title,description,category,price
1,Widget Pro,"Advanced widget with AI features",Electronics,99.99
2,Super Gadget,"Multi-purpose gadget for daily use",Electronics,149.99
3,Home Helper,"Smart home automation device",Smart Home,199.99
```

**Configuration**:
```yaml
tools:
  - name: product_search
    type: vectorstore
    source: data/products.csv
    id_field: id
    vector_field: description
    meta_fields: [title, category, price]
```

**Generated Records**:
```python
[
    DynamicStructuredRecord(
        id="1",
        content="Advanced widget with AI features",
        embedding=[...],
        source_file="data/products.csv",
        title="Widget Pro",
        category="Electronics",
        price="99.99"
    ),
    # ... more records
]
```

### JSON Example (`faqs.json`)

```json
{
  "data": {
    "items": [
      {
        "faq_id": "FAQ001",
        "question": "How do I reset my password?",
        "answer": "Go to Settings > Security > Reset Password",
        "category": "Account"
      }
    ]
  }
}
```

**Configuration**:
```yaml
tools:
  - name: faq_search
    type: vectorstore
    source: data/faqs.json
    record_path: data.items
    id_field: faq_id
    vector_fields: [question, answer]
    field_separator: "\n\n"
    meta_fields: [category]
```

**Generated Content** (concatenated):
```
How do I reset my password?

Go to Settings > Security > Reset Password
```

## Integration Points

### With Existing VectorstoreTool

The structured data loader integrates with the existing vectorstore tool execution:

```python
# In vectorstore_tool.py execution flow
async def execute_vectorstore_tool(tool_config: VectorstoreTool, query: str):
    # Detect structured vs unstructured mode
    if tool_config.vector_field is not None:
        # Structured mode
        loader = StructuredDataLoader(
            source_path=tool_config.source,
            id_field=tool_config.id_field,
            vector_fields=normalize_vector_fields(tool_config.vector_field),
            metadata_fields=tool_config.meta_fields,
            field_separator=tool_config.field_separator,
            delimiter=tool_config.delimiter,
            record_path=tool_config.record_path,
        )
        record_class = create_structured_record_class(
            dimensions=tool_config.embedding_dimensions or 1536,
            metadata_field_names=loader.get_metadata_field_names(),
        )
        # ... proceed with structured ingestion
    else:
        # Existing unstructured mode
        # ... existing DocumentRecord flow
```

### With Vector Store Collections

Uses the same `get_collection_factory()` from `vector_store.py`:

```python
factory = get_collection_factory(
    provider=database_config.provider,
    dimensions=embedding_dimensions,
    **connection_kwargs,
)

async with factory() as collection:
    # Upsert structured records
    await collection.upsert(structured_records)
```
