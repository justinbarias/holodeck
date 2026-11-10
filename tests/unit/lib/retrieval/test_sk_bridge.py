"""Tests for the temporary Semantic Kernel bridge (removed in task T16)."""

from __future__ import annotations

import importlib
import tomllib
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from semantic_kernel.data.vector import (
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
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


# Providers that moved to a native adapter; their bridge class stays until T16.
NATIVE = {"qdrant": "QdrantAdapter"}
BRIDGE_CLASSES: dict[str, type[SKBridgeAdapter]] = {
    "qdrant": SKQdrantAdapter,
}


def test_registry_maps_every_bridged_provider() -> None:
    assert PROVIDERS["in-memory"].class_name == "InMemoryAdapter"
    assert PROVIDERS["postgres"].class_name == "PostgresAdapter"
    for provider, class_name in NATIVE.items():
        assert PROVIDERS[provider].class_name == class_name
        assert PROVIDERS[provider].module.endswith(f".adapters.{provider}")
    for provider in (p for p in BRIDGED if p not in NATIVE):
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
    adapter_class: type[SKBridgeAdapter] = BRIDGE_CLASSES.get(provider) or getattr(
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


# -- Qdrant native hybrid (moved from keyword_search in 042 T4) ----------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Abiomed ABMD", ["abiomed", "abmd"]),
        ("A pile of foo", ["pile", "of", "foo"]),  # one-char tokens dropped
        ("... ?!", []),
        ("Apple, Inc. 2024", ["apple", "inc", "2024"]),
        ("apple apple AAPL apple", ["apple", "aapl"]),
        ("", []),
    ],
)
def test_qdrant_query_tokens(text: str, tokens: list[str]) -> None:
    assert sk_bridge._tokenize(text) == tokens


def _qdrant_with_points(points: list[Any]) -> tuple[SKQdrantAdapter, MagicMock]:
    adapter, collection = _fake_adapter(SKQdrantAdapter)
    client = MagicMock()
    client.query_points = AsyncMock(return_value=MagicMock(points=points))
    collection.qdrant_client = client
    collection.collection_name = "docs"
    return adapter, client


@pytest.mark.asyncio
async def test_qdrant_hybrid_uses_match_text_per_token_and_rrf() -> None:
    from qdrant_client.http import models as qm

    point = MagicMock(id="chunk-1", score=0.9, payload={"content": "x"})
    adapter, client = _qdrant_with_points([point])

    hits = await adapter.hybrid_search(
        HybridQuery("Abiomed ABMD", [0.1] * 4, top_k=5, text_field="content")
    )

    assert [(hit.record.id, hit.score) for hit in hits] == [("chunk-1", 0.9)]
    kwargs = client.query_points.await_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["limit"] == 5
    vector_leg, keyword_leg = kwargs["prefetch"]
    assert vector_leg.using == "embedding"
    assert vector_leg.limit >= 20  # wider than top_k
    should = keyword_leg.filter.should
    assert [c.match.text for c in should] == ["abiomed", "abmd"]
    assert all(isinstance(c.match, qm.MatchText) for c in should)
    assert all(c.key == "content" for c in should)
    assert kwargs["query"].fusion == qm.Fusion.RRF


@pytest.mark.asyncio
async def test_qdrant_hybrid_without_tokens_searches_by_vector_only() -> None:
    adapter, client = _qdrant_with_points([])

    await adapter.hybrid_search(
        HybridQuery("... ?!", [0.1] * 4, top_k=5, text_field="content")
    )

    assert len(client.query_points.await_args.kwargs["prefetch"]) == 1


# -- Qdrant payload indexes (moved from HierarchicalDocumentTool in 042 T4) ----


def _qdrant_for_indexes(
    schema: CollectionSchema, side_effect: Exception | None = None
) -> tuple[SKQdrantAdapter, MagicMock, MagicMock]:
    collection = MagicMock(name="sk_collection")
    collection.collection_exists = AsyncMock(return_value=True)
    collection.collection_name = schema.name
    client = MagicMock()
    client.create_payload_index = AsyncMock(side_effect=side_effect)
    collection.qdrant_client = client
    with patch.object(
        SKBridgeAdapter, "_make_factory", return_value=lambda: collection
    ):
        return SKQdrantAdapter(schema, {}), collection, client


@pytest.mark.asyncio
async def test_qdrant_ensure_collection_indexes_the_hierarchical_fields() -> None:
    from qdrant_client.http import models as qm

    from holodeck.lib.retrieval.schemas import hierarchical_schema

    adapter, _, client = _qdrant_for_indexes(hierarchical_schema(4, "tool"))

    await adapter.ensure_collection()

    schemas = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.await_args_list
    }
    keyword = {
        "source_path",
        "section_id",
        "defined_term",
        "defined_term_normalized",
    }
    assert set(schemas) == keyword | {"chunk_index", "searchable_text"}
    for name in keyword:
        assert isinstance(schemas[name], qm.KeywordIndexParams)
    assert isinstance(schemas["chunk_index"], qm.IntegerIndexParams)
    text = schemas["searchable_text"]
    assert isinstance(text, qm.TextIndexParams)
    assert (text.min_token_len, text.max_token_len, text.lowercase) == (2, 20, True)


@pytest.mark.asyncio
async def test_qdrant_payload_indexes_are_created_once_per_adapter() -> None:
    adapter, _, client = _qdrant_for_indexes(_schema_with_filters())

    await adapter.ensure_collection()
    calls = client.create_payload_index.await_count
    await adapter.ensure_collection()

    assert calls == 2
    assert client.create_payload_index.await_count == calls


@pytest.mark.asyncio
async def test_qdrant_index_failure_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter, _, client = _qdrant_for_indexes(
        _schema_with_filters(), side_effect=RuntimeError("boom")
    )

    await adapter.ensure_collection()

    assert client.create_payload_index.await_count == 2
    assert "Qdrant payload index failed for 'source_path'" in caplog.text
