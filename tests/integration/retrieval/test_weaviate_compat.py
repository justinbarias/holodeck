"""The native Weaviate adapter and Semantic Kernel's Weaviate collection layout.

042 T10 replaces ``SKWeaviateAdapter``. That bridge class has never worked
against a Weaviate server (verified against Weaviate 1.40, client 4.24):

* ``ensure_collection`` fails. The bridge defines the vector with
  ``DistanceFunction.COSINE_SIMILARITY``; SK's Weaviate connector accepts only
  ``COSINE_DISTANCE`` ("Distance function ... is not supported by Weaviate");
* the first ``upsert`` then lets Weaviate create the collection from the data
  (default unnamed vector), and the write fails because the ``embedding`` vector
  is not configured;
* SK passes the HoloDeck ID to Weaviate as the object UUID, so the chunk IDs the
  tools use (``/srv/doc.md_chunk_3``) fail at the server.

No HoloDeck collection written through that bridge exists, so there is nothing
to migrate. These tests build the collection with SK's own
``WeaviateCollection`` and the one change that makes it work
(``COSINE_DISTANCE``), which is the layout SK's connector would produce, and
prove the native adapter creates the same layout except where it is stricter
(field tokenization on filter fields), and refuses the SK layout for that reason.
The first test pins the bridge defect; it goes away with the bridge (task T16).

Each test uses its own collections on the Weaviate server the live suite uses
(``127.0.0.1:8089`` HTTP, ``50059`` gRPC) and deletes them afterwards.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio
import weaviate
from semantic_kernel.data.vector import DistanceFunction
from semantic_kernel.exceptions import VectorStoreOperationException
from weaviate import WeaviateAsyncClient
from weaviate.classes.config import Configure, DataType, Property, Tokenization

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    IncompatibleCollectionError,
    LexicalQuery,
    Record,
    RecordFilter,
    VectorQuery,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.adapters.sk_bridge import (
    SKWeaviateAdapter,
    build_definition,
)
from holodeck.lib.retrieval.adapters.weaviate import (
    WeaviateAdapter,
    weaviate_class_name,
)
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from holodeck.lib.vector_store import get_collection_factory

CONNECTION = {"local_host": "127.0.0.1", "local_port": 8089, "local_grpc_port": 50059}


def _reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8089), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason="Weaviate not reachable at :8089"),
]

QUOTED_SOURCE = 'it\'s "b" \\ dir/b.md'


def uid(name: str) -> str:
    """A UUID ID: the only kind SK's connector stores."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"compat/{name}"))


def make_schema(name: str) -> CollectionSchema:
    """One field of every type, every index kind."""
    return CollectionSchema(
        name=name,
        dimensions=4,
        fields=(
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT, indexed=True),
            FieldSpec("pinned", FieldType.BOOLEAN, indexed=True),
            FieldSpec("note", FieldType.STRING),
        ),
    )


def rec(
    record_id: str,
    vector: list[float],
    source: str,
    index: int,
    content: str,
    *,
    mtime: float = 1.5,
    pinned: bool = False,
    note: str = "",
) -> Record:
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": index,
            "content": content,
            "mtime": mtime,
            "pinned": pinned,
            "note": note,
        },
        vector=vector,
    )


CORPUS: list[Record] = [
    rec(uid("a0"), [1.0, 0.0, 0.0, 0.0], "a.md", 0, "alpha apple orchard"),
    rec(uid("a1"), [0.9, 0.1, 0.0, 0.0], "a.md", 1, "alpha banana grove"),
    rec(
        uid("q0"),
        [0.0, 1.0, 0.0, 0.0],
        QUOTED_SOURCE,
        0,
        "beta cherry tree",
        mtime=2.25,
        pinned=True,
        note="n",
    ),
    rec(uid("q1"), [0.0, 0.0, 1.0, 0.0], QUOTED_SOURCE, 1, "beta zeppelin hangar"),
    rec(uid("c0"), [0.0, 0.0, 0.2, 1.0], "c.md", 0, "gamma delta"),
    rec(uid("d3"), [0.5, 0.0, 0.5, 0.0], "docs/d.md", 3, "kestrel dawn"),
]
IDS = [record.id for record in CORPUS] + [uid("missing")]

VECTOR = [0.8, 0.2, 0.1, 0.0]
HYBRID_TEXTS = ["zeppelin", "kestrel", "zzqx plugh", "?!?", ""]


# -- the SK collection --------------------------------------------------------------


@asynccontextmanager
async def sk_collection(schema: CollectionSchema) -> AsyncIterator[Any]:
    """SK's ``WeaviateCollection`` for ``schema``, with a cosine distance SK accepts."""
    definition = build_definition(schema)
    for definition_field in definition.fields:
        if definition_field.field_type == "vector":
            definition_field.distance_function = DistanceFunction.COSINE_DISTANCE
    factory = get_collection_factory(
        "weaviate",
        schema.dimensions,
        record_class=dict,
        definition=definition,
        **CONNECTION,
    )
    collection = factory()
    async with collection:
        yield collection


def sk_rows(records: list[Record]) -> list[dict[str, Any]]:
    return [
        {"id": r.id, **r.fields, "embedding": list(r.vector or ())} for r in records
    ]


@pytest_asyncio.fixture
async def raw() -> AsyncIterator[WeaviateAsyncClient]:
    client = weaviate.use_async_with_local(host="127.0.0.1", port=8089, grpc_port=50059)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def names(raw: WeaviateAsyncClient) -> AsyncIterator[list[str]]:
    """Unique collection names; every one is deleted afterwards."""
    created: list[str] = []
    try:
        yield created
    finally:
        for name in created:
            await raw.collections.delete(weaviate_class_name(name))


def unique(names: list[str]) -> str:
    name = f"Compat_{uuid.uuid4().hex[:12]}"  # valid class name: used unchanged
    names.append(name)
    return name


def native(schema: CollectionSchema) -> WeaviateAdapter:
    return WeaviateAdapter.create(schema, CONNECTION)


# -- comparison helpers -----------------------------------------------------------


def _row(record: Record) -> tuple[str, dict[str, Any]]:
    return record.id, dict(record.fields)


async def _walk(adapter: VectorStoreAdapter, limit: int) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    cursor: str | None = None
    for _ in range(50):
        page = await adapter.list_page(cursor=cursor, limit=limit)
        rows.extend(_row(record) for record in page.records)
        cursor = page.next_cursor
        if cursor is None:
            return sorted(rows)
    raise AssertionError("list_page did not finish")


async def snapshot(adapter: VectorStoreAdapter) -> dict[str, Any]:
    """Every read the tools use, in comparable form."""
    got = await adapter.get(IDS, include_vectors=True)
    filters = [
        RecordFilter.where(source_path=QUOTED_SOURCE),
        RecordFilter.where(source_path="a.md", chunk_index=1),
        RecordFilter.where(pinned=True),
        RecordFilter.where(mtime=2.25),
        RecordFilter.where(source_path="docs"),
        RecordFilter.where(source_path="nope"),
    ]
    found = [sorted(_row(r) for r in await adapter.find(f)) for f in filters]
    searches = []
    for record_filter in (None, RecordFilter.where(source_path=QUOTED_SOURCE)):
        hits = await adapter.vector_search(VectorQuery(VECTOR, 5, record_filter))
        searches.append([(h.record.id, dict(h.record.fields), h.score) for h in hits])
    hybrid = []
    for text in HYBRID_TEXTS:
        hits = await adapter.hybrid_search(HybridQuery(text, VECTOR, 3, "content"))
        hybrid.append(sorted((h.record.id, round(h.score, 6), h.kind) for h in hits))
    lexical = [
        sorted(
            (h.record.id, round(h.score, 6))
            for h in await adapter.lexical_search(LexicalQuery(text, 3, "content"))
        )
        for text in ("zeppelin", "kestrel", "nomatch")
    ]
    return {
        "get": [_row(r) for r in got],
        "vectors": [list(r.vector or ()) for r in got],
        "find": found,
        "list": await _walk(adapter, 2),
        "count": await adapter.count(),
        "vector_search": searches,
        "hybrid": hybrid,
        "lexical": lexical,
    }


def assert_same(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["get"] == right["get"]
    assert len(left["get"]) == len(CORPUS)
    for a, b in zip(left["vectors"], right["vectors"], strict=True):
        assert a == pytest.approx(b, abs=1e-6)
    assert left["find"] == right["find"]
    assert [len(rows) for rows in left["find"]] == [2, 1, 1, 1, 0, 0]
    assert left["list"] == right["list"]
    assert left["count"] == right["count"] == len(CORPUS)
    for a_hits, b_hits in zip(
        left["vector_search"], right["vector_search"], strict=True
    ):
        assert [(i, f) for i, f, _ in a_hits] == [(i, f) for i, f, _ in b_hits]
        assert [s for *_, s in a_hits] == pytest.approx(
            [s for *_, s in b_hits], abs=1e-6
        )
    assert left["hybrid"] == right["hybrid"]
    assert left["lexical"] == right["lexical"]
    assert [i for i, _ in left["lexical"][0]] == [uid("q1")]


async def _close(*adapters: VectorStoreAdapter) -> None:
    for adapter in adapters:
        await adapter.close()


# -- the bridge ---------------------------------------------------------------------


async def test_the_sk_bridge_cannot_create_a_collection_today(
    names: list[str],
) -> None:
    schema = make_schema(unique(names))
    bridge = SKWeaviateAdapter.create(schema, CONNECTION)
    try:
        with pytest.raises(VectorStoreOperationException, match="COSINE_SIMILARITY"):
            await bridge.ensure_collection()
        assert await bridge.collection_exists() is False
    finally:
        await bridge.close()


# -- SK layout, native adapter --------------------------------------------------------


async def test_native_refuses_an_sk_collection_with_word_tokenized_filters(
    names: list[str],
) -> None:
    """SK tokenizes every string by word, so exact-match filters are not exact.

    The native adapter reads and writes only the layout it creates (field
    tokenization on filter fields). No SK-written Weaviate data exists to keep.
    """
    schema = make_schema(unique(names))
    adapter = native(schema)
    try:
        async with sk_collection(schema) as collection:
            await collection.ensure_collection_exists()
            await collection.upsert(sk_rows(CORPUS))
            with pytest.raises(IncompatibleCollectionError, match="tokenization"):
                await adapter.ensure_collection()
    finally:
        await adapter.close()


async def test_sk_reads_a_native_collection(names: list[str]) -> None:
    schema = make_schema(unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(CORPUS)
        expected = await adapter.get(IDS, include_vectors=True)
        expected_hits = await adapter.vector_search(VectorQuery(VECTOR, 4))
        async with sk_collection(schema) as collection:
            rows = await collection.get(IDS, include_vectors=True)
            results = await collection.search(vector=VECTOR, top=4)
            sk_hits = [(str(r.record["id"]), r.score) async for r in results.results]
    finally:
        await adapter.close()

    by_id = {str(row["id"]): row for row in rows}
    assert sorted(by_id) == sorted(record.id for record in expected)
    for record in expected:
        row = by_id[record.id]
        assert {k: row[k] for k in record.fields} == dict(record.fields)
        assert row["embedding"] == pytest.approx(record.vector, abs=1e-6)
    assert [i for i, _ in sk_hits] == [hit.record.id for hit in expected_hits]
    # SK returns the cosine distance; the adapter returns 1 - distance, clamped.
    assert [1.0 - d for _, d in sk_hits] == pytest.approx(
        [hit.score for hit in expected_hits], abs=1e-5
    )


@pytest.mark.parametrize(
    "build",
    [
        make_schema,
        lambda name: document_schema(4, {"collection_name": name}),
        lambda name: hierarchical_schema(4, "tool", {"collection_name": name}),
    ],
    ids=["all-types", "document", "hierarchical"],
)
async def test_collection_config_matches_what_sk_creates(
    names: list[str], raw: WeaviateAsyncClient, build: Any
) -> None:
    sk_schema: CollectionSchema = build(unique(names))
    own_schema: CollectionSchema = build(unique(names))
    adapter = native(own_schema)
    try:
        async with sk_collection(sk_schema) as collection:
            await collection.ensure_collection_exists()
        await adapter.ensure_collection()
        sk_config = await raw.collections.get(
            weaviate_class_name(sk_schema.name)
        ).config.get()
        config = await raw.collections.get(
            weaviate_class_name(own_schema.name)
        ).config.get()
    finally:
        await adapter.close()

    assert config.vector_config == sk_config.vector_config
    assert config.inverted_index_config == sk_config.inverted_index_config
    assert config.multi_tenancy_config == sk_config.multi_tenancy_config
    assert config.replication_config == sk_config.replication_config

    sk_props = {p.name: p for p in sk_config.properties}
    props = {p.name: p for p in config.properties}
    # The one extra property keeps the original HoloDeck ID of non-UUID IDs.
    assert set(props) - set(sk_props) == {"holodeck_id"}
    for name, sk_prop in sk_props.items():
        assert props[name].data_type == sk_prop.data_type
        spec = own_schema.field(name)
        text = spec.type is FieldType.STRING
        # SK leaves every property filterable and searchable; the adapter builds
        # the indexes the FieldSpec flags ask for (documented difference).
        assert sk_prop.index_filterable is True
        assert props[name].index_filterable is (spec.indexed and not spec.full_text)
        if spec.full_text:
            assert props[name].index_searchable is True
            assert props[name].tokenization == sk_prop.tokenization == Tokenization.WORD
        elif text:
            assert props[name].index_searchable is False
            if spec.indexed:
                # Exact-match filters need field tokenization.
                assert props[name].tokenization == Tokenization.FIELD
                assert sk_prop.tokenization == Tokenization.WORD
    assert props["holodeck_id"].data_type == DataType.TEXT


async def test_stored_objects_are_identical(
    names: list[str], raw: WeaviateAsyncClient
) -> None:
    """Same collection name, written once by SK and once by the adapter."""
    schema = make_schema(unique(names))
    collection_name = weaviate_class_name(schema.name)

    async def objects() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        response = await raw.collections.get(collection_name).query.fetch_objects(
            limit=100, include_vector=True
        )
        # An unset holodeck_id reads back as None; SK's collection has no such property.
        return {
            str(o.uuid): (
                {
                    k: v
                    for k, v in o.properties.items()
                    if not (k == "holodeck_id" and v is None)
                },
                dict(o.vector),
            )
            for o in response.objects
        }

    async with sk_collection(schema) as collection:
        await collection.ensure_collection_exists()
        await collection.upsert(sk_rows(CORPUS))
        by_sk = await objects()
        await collection.ensure_collection_deleted()
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(CORPUS)
        by_native = await objects()
    finally:
        await adapter.close()

    assert by_native.keys() == by_sk.keys() == {record.id for record in CORPUS}
    for key, (properties, vectors) in by_native.items():
        sk_properties, sk_vectors = by_sk[key]
        assert properties == sk_properties
        assert {k: type(v) for k, v in properties.items()} == {
            k: type(v) for k, v in sk_properties.items()
        }
        assert vectors.keys() == sk_vectors.keys() == {"embedding"}
        assert vectors["embedding"] == pytest.approx(sk_vectors["embedding"], abs=1e-7)


async def test_ensure_collection_on_a_native_collection_is_idempotent(
    names: list[str], raw: WeaviateAsyncClient
) -> None:
    schema = make_schema(unique(names))
    configs = []
    for _ in range(3):
        adapter = native(schema)
        try:
            await adapter.ensure_collection()
            if not configs:
                await adapter.upsert(CORPUS)
            configs.append(
                await raw.collections.get(weaviate_class_name(schema.name)).config.get()
            )
            assert await adapter.count() == len(CORPUS)
        finally:
            await adapter.close()
    assert configs[0] == configs[1] == configs[2]


# -- collections the adapter must refuse ----------------------------------------------


async def test_a_collection_without_the_named_vector_is_refused(
    names: list[str], raw: WeaviateAsyncClient
) -> None:
    schema = make_schema(unique(names))
    await raw.collections.create(
        name=weaviate_class_name(schema.name),
        properties=[Property(name="content", data_type=DataType.TEXT)],
        vector_config=Configure.Vectors.self_provided(),
    )
    adapter = native(schema)
    try:
        with pytest.raises(IncompatibleCollectionError, match="named vector"):
            await adapter.ensure_collection()
    finally:
        await adapter.close()


async def test_a_property_of_another_type_is_refused(
    names: list[str], raw: WeaviateAsyncClient
) -> None:
    schema = make_schema(unique(names))
    await raw.collections.create(
        name=weaviate_class_name(schema.name),
        properties=[Property(name="chunk_index", data_type=DataType.TEXT)],
        vector_config=Configure.Vectors.self_provided(name="embedding"),
    )
    adapter = native(schema)
    try:
        with pytest.raises(IncompatibleCollectionError, match="chunk_index"):
            await adapter.ensure_collection()
    finally:
        await adapter.close()


# -- native only ----------------------------------------------------------------------


async def test_ids_round_trip_and_replace_by_store_id(names: list[str]) -> None:
    schema = make_schema(unique(names))
    adapter = native(schema)
    upper = "8F14E45F-CEEA-467F-A0E6-0C0B6C0B6C0B"
    ids = ['it\'s "quoted" id', "42", "/srv/x/doc.md_chunk_3", "", "é ü/\\", upper]
    try:
        await adapter.ensure_collection()
        records = [
            rec(
                i or "empty-string-neighbour", [1.0, 0.0, 0.0, float(n)], "m.md", n, "x"
            )
            for n, i in enumerate(ids)
        ]
        await adapter.upsert(records)
        got = await adapter.get([r.id for r in records])
        assert [r.id for r in got] == [r.id for r in records]
        # The upper-case UUID is one object whichever way it is spelled.
        await adapter.upsert([rec(upper.lower(), [0.0, 1.0, 0.0, 0.0], "m.md", 9, "y")])
        assert await adapter.count() == len(records)
        (same,) = await adapter.get([upper])
        assert same.id == upper.lower()
        source = RecordFilter.where(source_path="m.md")
        await adapter.replace_source(source, [records[-1]])
        assert [r.id for r in await adapter.find(source)] == [upper]
    finally:
        await adapter.close()


async def test_hostile_filter_values_match_only_themselves(names: list[str]) -> None:
    schema = make_schema(unique(names))
    adapter = native(schema)
    hostile = [
        QUOTED_SOURCE,
        "a",
        "",
        "*",
        "docs/*",
        "a.md OR 1=1",
        'x"}) { Delete }',
        "é中\n\t ",
        "A.MD",
        "a.md ",
    ]
    try:
        await adapter.ensure_collection()
        records = [
            rec(f"id{n}", [1.0, float(n), 0.0, 0.0], value, n, "x")
            for n, value in enumerate(hostile)
        ]
        await adapter.upsert(records)
        for n, value in enumerate(hostile):
            clause = RecordFilter.where(source_path=value)
            assert [r.id for r in await adapter.find(clause)] == [f"id{n}"], value
            hits = await adapter.vector_search(VectorQuery(VECTOR, 20, clause))
            assert [h.record.id for h in hits] == [f"id{n}"], value
        assert await adapter.find(RecordFilter.where(source_path="a.m")) == []
        assert await adapter.delete_where(RecordFilter.where(source_path="*")) == 1
        assert await adapter.count() == len(hostile) - 1
    finally:
        await adapter.close()


async def test_the_client_is_rebuilt_after_close(names: list[str]) -> None:
    schema = make_schema(unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(CORPUS[:2])
        await adapter.close()
        await adapter.close()
        assert await adapter.count() == 2
    finally:
        await adapter.close()
