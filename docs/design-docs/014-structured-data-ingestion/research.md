# Research: Structured Data Field Mapping and Ingestion

**Feature**: 014-structured-data-ingestion
**Date**: 2025-12-18
**Status**: Complete

## Research Questions

### 1. How should DynamicStructuredRecord differ from DocumentRecord?

**Decision**: Create a new factory function `create_structured_record_class()` that generates records with:
- Fixed fields: `id` (key), `content` (combined vector fields text), `embedding` (vector)
- Dynamic metadata fields: Generated based on `metadata_fields` configuration

**Rationale**: DocumentRecord has fixed schema (source_path, chunk_index, mtime, file_type, file_size_bytes). Structured data needs arbitrary metadata fields from the source schema. A factory approach allows runtime schema generation while maintaining Semantic Kernel compatibility.

**Alternatives considered**:
1. Single generic record with JSON metadata blob → Rejected: Loses type safety, breaks filtering
2. Extend DocumentRecord with extra fields → Rejected: Fields are fixed at class definition time
3. Multiple predefined record types → Rejected: Can't anticipate all schemas

### 2. How to handle Semantic Kernel's @vectorstoremodel decorator with dynamic fields?

**Decision**: Use `dataclasses.make_dataclass()` with `VectorStoreField` annotations to dynamically create compatible classes.

**Rationale**: The `@vectorstoremodel` decorator processes type annotations. We can build these programmatically:

```python
from dataclasses import make_dataclass, field
from typing import Annotated
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

def create_structured_record_class(
    dimensions: int,
    metadata_field_names: list[str],
    collection_name: str = "structured_records",
) -> type:
    """Dynamically create a vector store record class."""

    # Build field definitions
    fields = [
        ("id", Annotated[str, VectorStoreField("key")], field(default_factory=lambda: str(uuid4()))),
        ("content", Annotated[str, VectorStoreField("data", is_full_text_indexed=True)], field(default="")),
        ("embedding", Annotated[list[float] | None, VectorStoreField("vector", dimensions=dimensions)], field(default=None)),
    ]

    # Add dynamic metadata fields
    for name in metadata_field_names:
        fields.append(
            (name, Annotated[str, VectorStoreField("data", is_indexed=True)], field(default=""))
        )

    # Create the dataclass
    cls = make_dataclass("DynamicStructuredRecord", fields)

    # Apply vectorstoremodel decorator
    return vectorstoremodel(collection_name=collection_name)(cls)
```

**Alternatives considered**:
1. Use `type()` with `__annotations__` → More complex, less readable
2. Pre-define common schemas → Too limiting for arbitrary structured data

### 3. Best practices for CSV/JSON loading at scale

**Decision**: Use streaming/chunked loading with configurable batch size (default: 10,000 records).

**Rationale**:
- For CSV: Use `csv.DictReader` for streaming (memory-efficient)
- For JSON: Use `ijson` for streaming large JSON arrays, or load in chunks for JSONL
- For small files (<10MB): Load entire file for simplicity

**Implementation approach**:
```python
def load_structured_records(
    source: str,
    id_field: str,
    vector_fields: list[str],
    metadata_fields: list[str] | None,
    batch_size: int = 10_000,
) -> Iterator[dict[str, Any]]:
    """Stream records from structured source."""
```

**Alternatives considered**:
1. pandas for all loading → Rejected: Heavy dependency, memory issues with large files
2. Load entire file always → Rejected: Memory issues with 1M+ rows
3. Required pandas → Rejected: Keep as optional dependency

### 4. How to detect source type (structured vs unstructured)?

**Decision**: Use file extension and explicit `source_type` parameter:
- `.csv` → CSV loader
- `.json` → JSON loader (detect array vs objects)
- `.jsonl`, `.ndjson` → JSONL loader (newline-delimited)
- If `vector_field` or `id_field` is specified → Structured mode
- Otherwise → Unstructured mode (existing behavior)

**Rationale**: Extension-based detection is simple and reliable. The presence of `vector_field`/`id_field` configuration clearly indicates structured data intent.

**Alternatives considered**:
1. Content sniffing → Rejected: Unreliable, adds complexity
2. New tool type `structured_vectorstore` → Rejected: Violates DRY, user confusion
3. Always structured for JSON/CSV → Rejected: Some users may want full-text embedding

### 5. How to validate field names against source schema?

**Decision**: Validate fields on first record read:
1. Load first record/row
2. Extract available field names
3. Validate id_field, vector_field(s), metadata_fields against available fields
4. Raise `ConfigError` with available fields list if validation fails

**Rationale**: Early validation prevents silent failures and provides actionable error messages.

**Error message format**:
```
ConfigError: Field 'descriptin' not found in data source.
Available fields: id, title, description, category, price
Did you mean: 'description'?
```

### 6. How to handle nested JSON fields?

**Decision**: Support dot notation (e.g., `"details.description"`) using a recursive field accessor.

**Implementation**:
```python
def get_nested_value(record: dict, field_path: str) -> Any:
    """Get value from nested dict using dot notation."""
    keys = field_path.split(".")
    value = record
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value
```

**Rationale**: Dot notation is intuitive and widely used (e.g., MongoDB, jq, Python dict access patterns).

**Alternatives considered**:
1. JSONPath syntax → Rejected: Overkill, adds dependency
2. Bracket notation only → Rejected: Less readable for simple cases
3. Flatten all nested objects → Rejected: Loses structure, naming conflicts

### 7. CSV delimiter detection strategy

**Decision**: Use Python's `csv.Sniffer` for auto-detection, with fallback to comma. Allow explicit `delimiter` parameter override.

**Implementation**:
```python
def detect_csv_delimiter(file_path: str, sample_size: int = 8192) -> str:
    """Auto-detect CSV delimiter."""
    with open(file_path, "r") as f:
        sample = f.read(sample_size)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","  # Default to comma
```

**Rationale**: Built-in `csv.Sniffer` handles common cases well. Explicit parameter provides escape hatch.

### 8. Deep nested JSON `record_path` resolution

**Decision**: Support a comprehensive `record_path` parameter with dot notation and array indexing for extracting records from deeply nested JSON structures.

**Rationale**: Real-world JSON data (especially API responses) frequently nests record arrays deeply. Users need a flexible path syntax that covers common patterns without requiring preprocessing.

**Supported Syntax:**

1. **Dot notation**: `data.results.items` - Navigate nested objects
2. **Array indexing**: `[0]`, `[1]`, `[-1]` - Access specific array elements
3. **Combined**: `data.results[0].items` - Mix object and array access
4. **Bracket notation**: `["special-key"]` - Keys with special characters (dots, dashes, spaces)

**Implementation Approach:**

```python
import re

def parse_path_segments(record_path: str) -> list[str | int]:
    """
    Parse path into segments: 'data.results[0].items' -> ['data', 'results', 0, 'items']

    Handles:
    - Dot notation: data.items
    - Array indices: [0], [-1]
    - Bracket notation: ["special-key"]
    """
    segments = []
    # Pattern matches: .key, [0], ["key"], ['key']
    pattern = r'\.([^.\[\]]+)|\[(\d+)\]|\[(["\'])([^"\']+)\3\]|^([^.\[\]]+)'
    for match in re.finditer(pattern, record_path):
        if match.group(1):  # .key
            segments.append(match.group(1))
        elif match.group(2):  # [0]
            segments.append(int(match.group(2)))
        elif match.group(4):  # ["key"] or ['key']
            segments.append(match.group(4))
        elif match.group(5):  # leading key without dot
            segments.append(match.group(5))
    return segments

def resolve_record_path(
    data: dict,
    record_path: str | None,
) -> tuple[list[dict], list[str]]:
    """
    Navigate JSON structure to extract record array.

    Returns:
        (records_list, traversed_path) for success
        Raises RecordPathError with available keys on failure
    """
    if not record_path:
        # No path specified - expect data to be list of records
        if isinstance(data, list):
            return data, []
        raise RecordPathError("Expected array at root", available_keys=list(data.keys()))

    segments = parse_path_segments(record_path)
    current = data
    traversed = []

    for segment in segments:
        if isinstance(segment, int):
            # Array index access
            if not isinstance(current, list):
                raise RecordPathError(
                    f"Expected array for index [{segment}]",
                    traversed_path=traversed,
                    available_keys=list(current.keys()) if isinstance(current, dict) else None,
                )
            if segment < 0 or segment >= len(current):
                raise RecordPathError(
                    f"Array index [{segment}] out of bounds (length: {len(current)})",
                    traversed_path=traversed,
                )
            current = current[segment]
            traversed.append(f"[{segment}]")
        else:
            # Object key access
            if not isinstance(current, dict):
                raise RecordPathError(
                    f"Expected object for key '{segment}'",
                    traversed_path=traversed,
                )
            if segment not in current:
                raise RecordPathError(
                    f"Key '{segment}' not found",
                    traversed_path=traversed,
                    available_keys=list(current.keys()),
                    full_path=record_path,
                )
            current = current[segment]
            traversed.append(segment)

    if not isinstance(current, list):
        raise RecordPathError(
            f"Path '{record_path}' does not resolve to an array",
            traversed_path=traversed,
            available_keys=list(current.keys()) if isinstance(current, dict) else None,
        )

    return current, traversed
```

**Error Message Examples:**

```
RecordPathError: Key 'items' not found at path 'data.results'.
  Traversed: data → results
  Available keys: ['records', 'metadata', 'pagination']
  Did you mean: 'records'?
```

```
RecordPathError: Array index [5] out of bounds (length: 3) at path 'batches[5]'.
  Traversed: batches
```

**Alternatives considered:**
1. JSONPath syntax (full spec) → Rejected: Overkill, adds `jsonpath-ng` dependency
2. Simple dot notation only → Rejected: Can't handle arrays or special characters
3. JMESPath → Rejected: Different semantics, learning curve

## Technology Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Record class generation | `make_dataclass()` + `@vectorstoremodel` | SK compatibility, dynamic fields |
| CSV loading | `csv.DictReader` (streaming) | Memory-efficient, built-in |
| JSON loading | Built-in `json` + chunking | No extra dependencies |
| Large file handling | Batch processing (10K records) | Balance memory/performance |
| Field validation | On first record read | Early failure, good errors |
| Nested fields | Dot notation accessor | Intuitive, no dependencies |
| Delimiter detection | `csv.Sniffer` + fallback | Built-in, reliable |
| record_path resolution | Dot notation + array indexing | Covers real-world JSON patterns, no dependencies |
| Database integration | Semantic Kernel providers | No additional ORM, 11+ providers already supported |

## Dependencies

**Required** (already in project):
- `semantic-kernel` (vector store abstractions)
- `pydantic` (validation)

**Optional** (for enhanced performance):
- `pandas` (faster CSV/JSON for moderate files)
- `ijson` (streaming JSON for very large files)

**Note**: No additional ORM libraries (like SQLAlchemy) are required. Semantic Kernel's vector store abstractions already support 11+ providers including PostgreSQL and SQL Server for database integration.

## Next Steps

1. Proceed to data-model.md (Phase 1)
2. Define `StructuredRecord` schema
3. Define `StructuredDataLoader` API
4. Create test fixtures
