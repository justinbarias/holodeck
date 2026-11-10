"""Collection schemas and record converters for HoloDeck's retrieval tools.

Three record families exist, one per tool mode:

* document: ``VectorStoreTool`` over unstructured files;
* structured: ``VectorStoreTool`` over CSV/JSON rows (``vector_field`` set);
* hierarchical: ``HierarchicalDocumentTool`` chunks.

Each schema has exactly the fields, types and index flags of the Semantic
Kernel record factory it replaces (``create_document_record_class``,
``create_structured_record_class`` and
``create_hierarchical_document_record_class`` in ``holodeck.lib.vector_store``),
and the same collection name, so collections written before the move stay
readable. ``tests/unit/lib/retrieval/test_schemas.py`` checks the equivalence.

Collection names
----------------
* document: ``documents_dim{dimensions}``
* structured: the tool name
* hierarchical: ``hierarchical_docs_{tool_name}_dim{dimensions}``

A ``collection_name`` key in the tool's database settings replaces the
default, as it did when those settings went straight to the SK collection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from holodeck.lib.retrieval.errors import SchemaError
from holodeck.lib.retrieval.types import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    Record,
)
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk

_STRING = FieldType.STRING
_INTEGER = FieldType.INTEGER
_FLOAT = FieldType.FLOAT

DOCUMENT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("source_path", _STRING, indexed=True),
    FieldSpec("chunk_index", _INTEGER, indexed=True),
    FieldSpec("content", _STRING, full_text=True),
    FieldSpec("mtime", _FLOAT),
    FieldSpec("file_type", _STRING),
    FieldSpec("file_size_bytes", _INTEGER),
    FieldSpec("content_hash", _STRING),
)

STRUCTURED_BASE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("content", _STRING, full_text=True),
    FieldSpec("source_file", _STRING, indexed=True),
)

HIERARCHICAL_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("source_path", _STRING, indexed=True),
    FieldSpec("chunk_index", _INTEGER, indexed=True),
    FieldSpec("section_id", _STRING, indexed=True),
    FieldSpec("content", _STRING),
    FieldSpec("parent_chain", _STRING),
    FieldSpec("chunk_type", _STRING),
    FieldSpec("cross_references", _STRING),
    FieldSpec("contextualized_content", _STRING),
    FieldSpec("searchable_text", _STRING, full_text=True),
    FieldSpec("mtime", _FLOAT),
    FieldSpec("file_type", _STRING),
    FieldSpec("defined_term", _STRING, indexed=True),
    FieldSpec("defined_term_normalized", _STRING, indexed=True),
    FieldSpec("subsection_ids", _STRING),
    FieldSpec("content_hash", _STRING),
)

HIERARCHICAL_TEXT_FIELD = "searchable_text"
"""Full-text field that the hierarchical tool's keyword leg searches."""


def collection_name_override(connection: Mapping[str, object]) -> str | None:
    """Return the ``collection_name`` set in the database settings, if any."""
    value = connection.get("collection_name")
    return value if isinstance(value, str) and value else None


def document_schema(
    dimensions: int, connection: Mapping[str, object] | None = None
) -> CollectionSchema:
    """Return the schema of ``VectorStoreTool`` document chunks."""
    name = collection_name_override(connection or {}) or f"documents_dim{dimensions}"
    return CollectionSchema(name, dimensions, DOCUMENT_FIELDS)


def structured_schema(
    dimensions: int,
    metadata_field_names: Sequence[str] | None,
    tool_name: str,
    connection: Mapping[str, object] | None = None,
) -> CollectionSchema:
    """Return the schema of ``VectorStoreTool`` structured records.

    Raises:
        SchemaError: A metadata field name is not a Python identifier.
    """
    fields = list(STRUCTURED_BASE_FIELDS)
    for field_name in metadata_field_names or ():
        if not field_name.isidentifier():
            raise SchemaError(
                f"Invalid field name: '{field_name}' (must be valid Python identifier)"
            )
        fields.append(FieldSpec(field_name, _STRING, indexed=True))
    name = collection_name_override(connection or {}) or tool_name
    return CollectionSchema(name, dimensions, tuple(fields))


def hierarchical_schema(
    dimensions: int, tool_name: str, connection: Mapping[str, object] | None = None
) -> CollectionSchema:
    """Return the schema of ``HierarchicalDocumentTool`` chunks."""
    name = (
        collection_name_override(connection or {})
        or f"hierarchical_docs_{tool_name}_dim{dimensions}"
    )
    return CollectionSchema(name, dimensions, HIERARCHICAL_FIELDS)


# -- typed field access --------------------------------------------------------


def text_field(record: Record, name: str, default: str = "") -> str:
    """Return a string field, or ``default`` when it is absent or empty."""
    value = record.fields.get(name)
    if value is None or value == "":
        return default
    return value if isinstance(value, str) else str(value)


def int_field(record: Record, name: str, default: int = 0) -> int:
    """Return an integer field, or ``default`` when it is absent."""
    value = record.fields.get(name)
    if isinstance(value, bool) or value is None or value == "":
        return default
    return int(value)


def float_field(record: Record, name: str, default: float = 0.0) -> float:
    """Return a float field, or ``default`` when it is absent or empty."""
    value = record.fields.get(name)
    if isinstance(value, bool) or value is None or value == "":
        return default
    return float(value)


def _json_list(record: Record, name: str) -> list[str]:
    try:
        loaded = json.loads(text_field(record, name, "[]"))
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


# -- document records ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocumentChunkRecord:
    """One chunk of an unstructured file, as ``VectorStoreTool`` stores it."""

    id: str
    source_path: str
    chunk_index: int
    content: str
    embedding: Sequence[float]
    mtime: float
    file_type: str
    file_size_bytes: int
    content_hash: str = ""

    def to_record(self) -> Record:
        """Return the stored form."""
        return Record(
            id=self.id,
            fields={
                "source_path": self.source_path,
                "chunk_index": self.chunk_index,
                "content": self.content,
                "mtime": self.mtime,
                "file_type": self.file_type,
                "file_size_bytes": self.file_size_bytes,
                "content_hash": self.content_hash,
            },
            vector=list(self.embedding),
        )


def structured_record(
    record_id: str,
    content: str,
    embedding: Sequence[float],
    source_file: str,
    metadata: Mapping[str, str],
    metadata_field_names: Sequence[str] | None,
) -> Record:
    """Return the stored form of one structured row.

    Every configured metadata field is written; a field missing from
    ``metadata`` is stored as an empty string.
    """
    fields: dict[str, str] = {"content": content, "source_file": source_file}
    for name in metadata_field_names or ():
        fields[name] = metadata.get(name, "")
    return Record(id=record_id, fields=fields, vector=list(embedding))


# -- hierarchical chunks -----------------------------------------------------------


def chunk_to_record(
    chunk: DocumentChunk, *, searchable_text: str, content_hash: str = ""
) -> Record:
    """Return the stored form of a hierarchical chunk."""
    return Record(
        id=chunk.id,
        fields={
            "source_path": chunk.source_path,
            "chunk_index": chunk.chunk_index,
            "section_id": chunk.section_id,
            "content": chunk.content,
            "parent_chain": json.dumps(chunk.parent_chain),
            "chunk_type": chunk.chunk_type.value,
            "cross_references": json.dumps(chunk.cross_references),
            "contextualized_content": chunk.contextualized_content or "",
            "searchable_text": searchable_text,
            "mtime": chunk.mtime,
            "file_type": "",
            "defined_term": chunk.defined_term or "",
            "defined_term_normalized": chunk.defined_term_normalized or "",
            "subsection_ids": json.dumps(chunk.subsection_ids),
            "content_hash": content_hash,
        },
        vector=list(chunk.embedding or ()),
    )


def chunk_from_record(record: Record) -> DocumentChunk:
    """Rebuild a hierarchical chunk from its stored form (without vector).

    Malformed JSON lists decode as empty lists and an unknown chunk type as
    ``ChunkType.CONTENT``.
    """
    try:
        chunk_type = ChunkType(text_field(record, "chunk_type"))
    except ValueError:
        chunk_type = ChunkType.CONTENT
    return DocumentChunk(
        id=record.id,
        source_path=text_field(record, "source_path"),
        chunk_index=int_field(record, "chunk_index"),
        content=text_field(record, "content"),
        parent_chain=_json_list(record, "parent_chain"),
        section_id=text_field(record, "section_id"),
        chunk_type=chunk_type,
        cross_references=_json_list(record, "cross_references"),
        heading_level=0,
        contextualized_content=text_field(record, "contextualized_content"),
        mtime=float_field(record, "mtime"),
        defined_term=text_field(record, "defined_term"),
        defined_term_normalized=text_field(record, "defined_term_normalized"),
        subsection_ids=_json_list(record, "subsection_ids"),
    )
