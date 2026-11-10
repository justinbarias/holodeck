"""Schemas match the SK record factories they replace; converters round-trip."""

from __future__ import annotations

from typing import Any

import pytest
from semantic_kernel.data.vector import VectorStoreCollectionDefinition

from holodeck.lib.retrieval import CollectionSchema, Record, SchemaError
from holodeck.lib.retrieval.adapters.sk_bridge import build_definition
from holodeck.lib.retrieval.schemas import (
    DocumentChunkRecord,
    chunk_from_record,
    chunk_to_record,
    document_schema,
    hierarchical_schema,
    structured_record,
    structured_schema,
)
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk
from holodeck.lib.vector_store import (
    create_document_record_class,
    create_hierarchical_document_record_class,
    create_structured_record_class,
)

DIMENSIONS = 8


def _described(definition: VectorStoreCollectionDefinition) -> dict[str, Any]:
    return {
        field.name: (
            field.field_type,
            field.type_,
            field.is_indexed,
            field.is_full_text_indexed,
            field.dimensions,
            field.distance_function,
        )
        for field in definition.fields
    }


@pytest.mark.parametrize(
    ("schema", "definition"),
    [
        (
            document_schema(DIMENSIONS),
            create_document_record_class(
                DIMENSIONS
            ).__kernel_vectorstoremodel_definition__,
        ),
        (
            hierarchical_schema(DIMENSIONS, "my_tool"),
            create_hierarchical_document_record_class(
                DIMENSIONS, "my_tool"
            ).__kernel_vectorstoremodel_definition__,
        ),
        (
            structured_schema(DIMENSIONS, ["title", "category"], "products"),
            create_structured_record_class(
                DIMENSIONS, ["title", "category"], "products"
            )[1],
        ),
    ],
    ids=["document", "hierarchical", "structured"],
)
def test_schema_matches_the_record_factory_layout(
    schema: CollectionSchema, definition: VectorStoreCollectionDefinition
) -> None:
    built = build_definition(schema)

    assert schema.name == definition.collection_name
    assert _described(built) == _described(definition)


def test_collection_names_used_today() -> None:
    assert document_schema(768).name == "documents_dim768"
    assert hierarchical_schema(1536, "kb").name == "hierarchical_docs_kb_dim1536"
    assert structured_schema(1536, None, "products").name == "products"


def test_collection_name_from_database_settings_wins() -> None:
    connection = {"collection_name": "custom"}

    assert document_schema(8, connection).name == "custom"
    assert hierarchical_schema(8, "kb", connection).name == "custom"
    assert structured_schema(8, None, "products", connection).name == "custom"


def test_structured_metadata_names_must_be_identifiers() -> None:
    with pytest.raises(SchemaError, match="valid Python identifier"):
        structured_schema(8, ["not-valid"], "products")


def test_structured_record_fills_missing_metadata_with_empty_strings() -> None:
    record = structured_record(
        "P1",
        content="Widget",
        embedding=[0.1] * 8,
        source_file="products.csv",
        metadata={"title": "Widget Pro"},
        metadata_field_names=["title", "category"],
    )

    assert record.fields == {
        "content": "Widget",
        "source_file": "products.csv",
        "title": "Widget Pro",
        "category": "",
    }


def test_document_chunk_record_fields() -> None:
    record = DocumentChunkRecord(
        id="a_chunk_0",
        source_path="/a.md",
        chunk_index=0,
        content="text",
        embedding=[0.0] * 8,
        mtime=1.5,
        file_type=".md",
        file_size_bytes=10,
    ).to_record()

    assert set(record.fields) == set(document_schema(8).field_names)
    assert record.fields["content_hash"] == ""


def test_hierarchical_chunk_round_trips() -> None:
    chunk = DocumentChunk(
        id="doc_chunk_3",
        source_path="/docs/policy.md",
        chunk_index=3,
        content="Force Majeure means any event beyond control.",
        parent_chain=["Chapter 1", "Definitions"],
        section_id="1.2",
        chunk_type=ChunkType.DEFINITION,
        cross_references=["2.1"],
        embedding=[0.5] * 8,
        contextualized_content="ctx",
        mtime=12.25,
        defined_term="Force Majeure",
        defined_term_normalized="force majeure",
        subsection_ids=["1.2.a"],
    )

    record = chunk_to_record(chunk, searchable_text="search me", content_hash="h")
    restored = chunk_from_record(Record(id=record.id, fields=record.fields))

    assert set(record.fields) == set(hierarchical_schema(8, "t").field_names)
    assert record.fields["searchable_text"] == "search me"
    assert record.fields["file_type"] == ""
    restored.embedding = chunk.embedding
    assert restored == chunk


def test_chunk_from_record_tolerates_bad_stored_values() -> None:
    chunk = chunk_from_record(
        Record(
            id="x",
            fields={
                "parent_chain": "not json",
                "subsection_ids": '{"a": 1}',
                "chunk_type": "unknown",
                "mtime": None,
            },
        )
    )

    assert chunk.parent_chain == []
    assert chunk.subsection_ids == []
    assert chunk.chunk_type is ChunkType.CONTENT
    assert chunk.mtime == 0.0
    assert chunk.source_path == ""
