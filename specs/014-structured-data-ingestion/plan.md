# Implementation Plan: Structured Data Field Mapping and Ingestion

**Branch**: `014-structured-data-ingestion` | **Date**: 2025-12-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/014-structured-data-ingestion/spec.md`
**User Guidance**: Create a new record type, DynamicStructuredRecord patterned on DynamicDocumentRecord in vector_store.py

## Summary

This feature extends the vectorstore tool to support structured data sources (CSV, JSON, JSONL, databases) with explicit field mapping. Unlike unstructured documents that embed full content chunks, structured data allows users to specify which fields to embed (`vector_field`/`vector_fields`), which fields to include as metadata (`metadata_fields`), and a unique record identifier (`id_field`) for incremental re-ingestion.

**Technical Approach**: Create a `DynamicStructuredRecord` class factory (similar to `create_document_record_class`) that generates Semantic Kernel-compatible vector store records with dynamic metadata fields. The existing `VectorstoreTool` model already has fields for `vector_field`, `meta_fields`, etc., which will be enhanced for structured data support.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Pydantic v2 (data validation)
- Semantic Kernel (vector store abstractions supporting 11+ providers: ChromaDB, Qdrant, PostgreSQL, Pinecone, Redis, etc.)
- Built-in csv, json modules (primary)
- pandas (CSV/JSON loading, optional for enhanced performance)

**Storage**: Semantic Kernel vector stores (ChromaDB, Qdrant, PostgreSQL, Pinecone, in-memory, etc.)
**Testing**: pytest with markers (@pytest.mark.unit, @pytest.mark.integration)
**Target Platform**: Linux server, macOS, Windows
**Project Type**: Single Python package (src/holodeck)
**Performance Goals**:
- Ingest up to 100,000 records within 2 minutes
- Search queries return within 2 seconds
**Constraints**:
- Process files up to 1M rows in batches without memory errors
- Memory-efficient streaming for large files
**Scale/Scope**: Support files with 1M+ rows, databases with 1M+ records

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. No-Code-First Agent Definition | ✅ PASS | All configuration via YAML (source, vector_field, metadata_fields, id_field) |
| II. MCP for API Integrations | ✅ PASS | Not an API integration - extends existing vectorstore tool |
| III. Test-First with Multimodal Support | ✅ PASS | CSV/JSON are multimodal inputs; test cases will validate field mapping |
| IV. OpenTelemetry-Native Observability | ⚠️ DEFERRED | Tracing for ingestion/search operations deferred to future sprint |
| V. Evaluation Flexibility | ✅ N/A | Not evaluation-related |

**Architecture Constraints**:
- ✅ Changes confined to Agent Engine (vector store operations)
- ✅ No coupling to Evaluation Framework or Deployment Engine

## Project Structure

### Documentation (this feature)

```text
specs/014-structured-data-ingestion/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (internal API contracts)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/holodeck/
├── models/
│   └── tool.py              # Extend VectorstoreTool for structured data fields
├── lib/
│   ├── vector_store.py      # Add DynamicStructuredRecord factory
│   └── structured_loader.py # NEW: CSV/JSON/JSONL/database record loading
└── tools/
    └── vectorstore_tool.py  # Extend for structured data ingestion

tests/
├── unit/
│   ├── test_structured_record.py  # DynamicStructuredRecord tests
│   └── test_structured_loader.py  # Loader tests
├── integration/
│   └── test_structured_vectorstore.py  # End-to-end structured data tests
└── fixtures/
    ├── structured/
    │   ├── products.csv
    │   ├── faqs.json
    │   └── nested_data.json
```

**Structure Decision**: Single project structure. New code extends existing modules (`vector_store.py`, `tool.py`) with minimal new files (`structured_loader.py`).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Key Design Decisions

### 1. DynamicStructuredRecord Pattern

Following the user's guidance and the existing `create_document_record_class` pattern:

```python
def create_structured_record_class(
    dimensions: int = 1536,
    metadata_fields: list[str] | None = None,
) -> type[Any]:
    """Create a StructuredRecord class with specified dimensions and metadata fields.

    Unlike DocumentRecord (which has fixed metadata: source_path, chunk_index, etc.),
    StructuredRecord has dynamic metadata fields based on the source data schema.
    """
```

### 2. Field Mapping Strategy

- **id_field** (required): Maps to the record's `id` key field in vector store
- **vector_field/vector_fields**: Content to embed (concatenated if multiple)
- **metadata_fields**: All other fields to store (default: all non-vector fields)

### 3. Source Type Detection

The existing `VectorstoreTool.source` field will be used. Detection logic:
- `.csv` extension → CSV loader
- `.json`/`.jsonl` extension → JSON/JSONL loader
- Database sources → Use Semantic Kernel's existing vector store providers (P3)
- Otherwise → Fallback to existing unstructured document loading

### 4. Integration with Existing Infrastructure

- Reuse `get_collection_factory()` for vector database connections
- Reuse embedding infrastructure from existing vectorstore tool
- Extend `VectorstoreTool` Pydantic model (already has `vector_field`, `meta_fields`)

### 5. Deep Nested JSON `record_path` Design

The `record_path` parameter enables extraction of record arrays from deeply nested JSON structures. This is critical for handling API responses and complex data files.

**Supported Path Syntax:**

| Syntax | Example | Description |
|--------|---------|-------------|
| Dot notation | `data.items` | Navigate nested objects |
| Array index | `[0]` | Access specific array element |
| Combined | `data.results[0].items` | Mix of object and array access |
| Bracket key | `["special-key"]` | Keys with special characters |

**Path Resolution Algorithm:**

```python
def resolve_record_path(data: dict, record_path: str) -> tuple[list[dict], list[str]]:
    """
    Navigate JSON structure using dot notation with array indexing.

    Args:
        data: Root JSON object
        record_path: Path like "data.results.items" or "batches[0].records"

    Returns:
        Tuple of (records_list, traversed_path_segments)

    Raises:
        RecordPathError: With available keys at failure point
    """
```

**Example Paths:**

| JSON Structure | record_path | Result |
|----------------|-------------|--------|
| `{"data": {"items": [...]}}` | `data.items` | Array at items |
| `{"response": {"data": [{"entries": [...]}]}}` | `response.data[0].entries` | entries from first data element |
| `{"batches": [[...], [...]]}` | `batches[1]` | Second batch array |
| `{"results": {"page-1": [...]}}` | `results["page-1"]` | Array at special key |

**Error Handling:**

When path resolution fails, provide actionable error messages:

```
RecordPathError: Key 'items' not found at path 'data.results'.
  Traversed: data → results
  Available keys at 'results': ['records', 'metadata', 'pagination']
  Did you mean: 'records'?
```

## Implementation Phases

### Phase 1: Core Record & Loader (P1 - MVP)
1. `create_structured_record_class()` factory in `vector_store.py`
2. `StructuredDataLoader` class in `structured_loader.py`
3. CSV and JSON file loading with field validation
4. Integration with existing vectorstore tool execution

### Phase 2: Multi-Field & Enhanced Features (P2)
1. Multiple `vector_fields` concatenation with separator
2. Dot notation for nested JSON fields
3. Auto-detection of CSV delimiters
4. Batch processing for large files

### Phase 3: Database Sources (P3)
1. Leverage Semantic Kernel's existing database vector store connectors (PostgreSQL, SQL Server, etc.)
2. No additional ORM libraries required - SK abstractions handle connection management
3. Incremental re-ingestion based on modification tracking
