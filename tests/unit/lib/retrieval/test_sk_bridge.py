"""Tests for the temporary Semantic Kernel bridge (removed in task T16)."""

from __future__ import annotations

import importlib
import tomllib
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from semantic_kernel.data.vector import (
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    MissingClientError,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    UnsupportedOperationError,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import sk_bridge
from holodeck.lib.retrieval.adapters.sk_bridge import (
    SKBridgeAdapter,
    SKPineconeAdapter,
    SKQdrantAdapter,
    build_definition,
)
from holodeck.lib.retrieval.registry import PROVIDERS
from holodeck.lib.vector_store import (
    create_document_record_class,
    create_hierarchical_document_record_class,
    create_structured_record_class,
)

DIMENSIONS = 8
_FIELD_TYPES = {
    "str": FieldType.STRING,
    "int": FieldType.INTEGER,
    "float": FieldType.FLOAT,
    "bool": FieldType.BOOLEAN,
}

BRIDGED = [
    "postgres",
    "azure-ai-search",
    "qdrant",
    "weaviate",
    "chromadb",
    "faiss",
    "azure-cosmos-mongo",
    "azure-cosmos-nosql",
    "sql-server",
    "pinecone",
]


def _schema_from(definition: VectorStoreCollectionDefinition) -> CollectionSchema:
    """Rebuild the schema that describes an existing SK definition."""
    specs = tuple(
        FieldSpec(
            field.name,
            _FIELD_TYPES[field.type_ or "str"],
            indexed=bool(field.is_indexed),
            full_text=bool(field.is_full_text_indexed),
        )
        for field in definition.fields
        if field.field_type.value == "data"
    )
    return CollectionSchema(definition.collection_name, DIMENSIONS, specs)


def _described(
    definition: VectorStoreCollectionDefinition,
) -> dict[str, tuple[Any, ...]]:
    def describe(field: VectorStoreField) -> tuple[Any, ...]:
        return (
            field.field_type,
            field.type_,
            field.is_indexed,
            field.is_full_text_indexed,
            field.dimensions,
            field.index_kind,
            field.distance_function,
        )

    return {field.name: describe(field) for field in definition.fields}


def _old_definitions() -> list[tuple[str, VectorStoreCollectionDefinition]]:
    document = create_document_record_class(DIMENSIONS)
    hierarchical = create_hierarchical_document_record_class(DIMENSIONS, "tool")
    _, structured = create_structured_record_class(
        DIMENSIONS, ["title", "category"], "products"
    )
    return [
        ("document", document.__kernel_vectorstoremodel_definition__),
        ("hierarchical", hierarchical.__kernel_vectorstoremodel_definition__),
        ("structured", structured),
    ]


@pytest.mark.parametrize(
    "definition",
    [definition for _, definition in _old_definitions()],
    ids=[name for name, _ in _old_definitions()],
)
def test_bridge_definition_matches_the_existing_record_factories(
    definition: VectorStoreCollectionDefinition,
) -> None:
    schema = _schema_from(definition)

    built = build_definition(schema)

    assert built.collection_name == definition.collection_name
    assert _described(built) == _described(definition)


def test_definition_has_one_key_and_one_cosine_vector_field() -> None:
    schema = CollectionSchema(
        "docs", DIMENSIONS, (FieldSpec("content", full_text=True),)
    )

    definition = build_definition(schema)

    assert [f.name for f in definition.fields if f.field_type.value == "key"] == ["id"]
    (vector,) = definition.vector_fields
    assert vector.name == "embedding"
    assert vector.dimensions == DIMENSIONS
    assert vector.distance_function.value == "cosine_similarity"


def test_registry_maps_every_bridged_provider() -> None:
    assert PROVIDERS["in-memory"].class_name == "InMemoryAdapter"
    for provider in BRIDGED:
        entry = PROVIDERS[provider]
        assert entry.module == "holodeck.lib.retrieval.adapters.sk_bridge"
        adapter_class = getattr(importlib.import_module(entry.module), entry.class_name)
        assert issubclass(adapter_class, SKBridgeAdapter)
        assert provider == adapter_class.PROVIDER
        assert entry.extra == adapter_class.EXTRA


def test_registry_extras_exist_in_pyproject() -> None:
    root = Path(__file__).resolve().parents[4]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert {e.extra for e in PROVIDERS.values() if e.extra} <= set(extras)
    for provider in ("postgres", "qdrant", "chromadb", "pinecone"):
        assert PROVIDERS[provider].extra == provider


def _schema() -> CollectionSchema:
    return CollectionSchema("docs", 4, (FieldSpec("content", full_text=True),))


def test_missing_client_with_an_extra_names_the_extra() -> None:
    with (
        patch(
            "holodeck.lib.vector_store.get_collection_class",
            side_effect=ImportError("no qdrant"),
        ),
        pytest.raises(MissingClientError) as raised,
    ):
        create_adapter("qdrant", _schema(), {})

    assert raised.value.extra == "qdrant"
    assert "pip install 'holodeck-ai[qdrant]'" in str(raised.value)


def test_missing_client_without_an_extra_keeps_the_package_hint() -> None:
    with (
        patch(
            "holodeck.lib.vector_store.get_collection_class",
            side_effect=ImportError("no weaviate"),
        ),
        pytest.raises(MissingClientError) as raised,
    ):
        create_adapter("weaviate", _schema(), {})

    assert "pip install weaviate-client" in str(raised.value)
    assert raised.value.provider == "weaviate"


# provider -> operations the bridge supports today
_FILTERS = {Operation.FIND, Operation.LIST, Operation.DELETE_WHERE, Operation.COUNT}
_BASE = {Operation.UPSERT, Operation.GET, Operation.DELETE, Operation.VECTOR_SEARCH}
_HYBRID = {Operation.HYBRID_SEARCH}
EXPECTED: dict[str, set[Operation]] = {
    "postgres": _BASE | _FILTERS,
    "azure-ai-search": _BASE | _HYBRID,
    "qdrant": _BASE | _FILTERS | _HYBRID,
    "weaviate": _BASE | _FILTERS | _HYBRID,
    "chromadb": _BASE | _FILTERS,
    "faiss": _BASE,
    "azure-cosmos-mongo": _BASE,
    "azure-cosmos-nosql": _BASE | _HYBRID,
    "sql-server": _BASE,
    "pinecone": _BASE,
}


@pytest.mark.parametrize("provider", BRIDGED)
def test_capabilities_match_what_works_today(provider: str) -> None:
    entry = PROVIDERS[provider]
    adapter_class: type[SKBridgeAdapter] = getattr(
        importlib.import_module(entry.module), entry.class_name
    )

    capabilities = adapter_class.declared_capabilities()

    assert capabilities.provider == provider
    assert set(capabilities.supported) == EXPECTED[provider]
    assert not capabilities.supports(Operation.LEXICAL_SEARCH)


def test_hybrid_providers_match_the_keyword_search_native_set() -> None:
    from holodeck.lib.keyword_search import NATIVE_HYBRID_PROVIDERS

    hybrid = {p for p, ops in EXPECTED.items() if Operation.HYBRID_SEARCH in ops}
    assert hybrid == NATIVE_HYBRID_PROVIDERS & set(BRIDGED)


def _fake_adapter(adapter_class: type[SKBridgeAdapter]) -> tuple[Any, MagicMock]:
    collection = MagicMock(name="sk_collection")
    with patch.object(
        SKBridgeAdapter, "_make_factory", return_value=lambda: collection
    ):
        return adapter_class(_schema_with_filters(), {}), collection


def _schema_with_filters() -> CollectionSchema:
    return CollectionSchema(
        "docs",
        4,
        (
            FieldSpec("source_path", indexed=True),
            FieldSpec("content", full_text=True),
        ),
    )


@pytest.mark.asyncio
async def test_unsupported_operations_raise_before_the_store_is_touched() -> None:
    adapter, collection = _fake_adapter(SKPineconeAdapter)

    with pytest.raises(UnsupportedOperationError):
        await adapter.find(RecordFilter.where(source_path="a.md"))
    with pytest.raises(UnsupportedOperationError):
        await adapter.list_page()
    with pytest.raises(UnsupportedOperationError):
        await adapter.count()
    with pytest.raises(UnsupportedOperationError):
        await adapter.delete_where(RecordFilter.where(source_path="a.md"))

    assert collection.method_calls == []
    assert collection.__aenter__.call_count == 0


def test_field_names_must_be_identifiers() -> None:
    schema = CollectionSchema("docs", 4, (FieldSpec("not-valid"),))
    with pytest.raises(SchemaError), patch.object(SKBridgeAdapter, "_make_factory"):
        SKPineconeAdapter(schema, {})


def test_qdrant_ids_keep_uuids_and_hash_everything_else() -> None:
    adapter, _ = _fake_adapter(SKQdrantAdapter)
    native = str(uuid.uuid4())

    assert adapter._store_id(native) == native
    mapped = adapter._store_id("a_chunk_0")
    assert uuid.UUID(mapped).version == 5
    assert adapter._store_id("a_chunk_0") == mapped
    assert adapter._store_id("42") != "42"
    assert adapter._store_id("a_chunk_1") != mapped


def test_qdrant_reserves_its_id_payload_key() -> None:
    schema = CollectionSchema("docs", 4, (FieldSpec(sk_bridge._QDRANT_ID_KEY),))
    with pytest.raises(SchemaError), patch.object(SKBridgeAdapter, "_make_factory"):
        SKQdrantAdapter(schema, {})


def test_missing_fields_are_stored_with_the_type_default() -> None:
    adapter, _ = _fake_adapter(SKQdrantAdapter)

    row = adapter._to_store(Record("a", {"content": "x"}, [1.0, 0.0, 0.0, 0.0]))

    assert row == {
        "id": "a",
        "source_path": "",
        "content": "x",
        "embedding": [1.0, 0.0, 0.0, 0.0],
    }


def test_filters_become_sk_lambda_strings() -> None:
    adapter, _ = _fake_adapter(SKQdrantAdapter)

    expression = adapter._filter_expression(
        RecordFilter.where(source_path="a.md", content="it's")
    )

    assert expression == "lambda x: x.source_path == 'a.md' and x.content == \"it's\""
    assert adapter._filter_expression(RecordFilter()) is None
    with pytest.raises(SchemaError):
        adapter._filter_expression(RecordFilter.where(nope="x"))
