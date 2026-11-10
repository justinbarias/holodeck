"""The native Weaviate adapter: stored layout, refusals and ID handling (042 T10).

The adapter replaced ``SKWeaviateAdapter``, which never worked against a
Weaviate server (its ``COSINE_SIMILARITY`` vector was refused by SK's
connector, and SK passed non-UUID chunk IDs to the server as object UUIDs).
No HoloDeck collection was ever written through that path, so there is no
legacy Weaviate layout to keep readable. Semantic Kernel is gone (042 T16);
the tests that built collections with SK's own ``WeaviateCollection`` went
with it, and the native-only tests stay.

Each test uses its own collections on the Weaviate server the live suite uses
(``127.0.0.1:8089`` HTTP, ``50059`` gRPC) and deletes them afterwards.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.classes.config import Configure, DataType, Property

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    IncompatibleCollectionError,
    Record,
    RecordFilter,
    VectorQuery,
)
from holodeck.lib.retrieval.adapters.weaviate import (
    WeaviateAdapter,
    weaviate_class_name,
)

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
    name = f"compat_{uuid.uuid4().hex[:12]}"
    names.append(name)
    return name


def native(schema: CollectionSchema) -> WeaviateAdapter:
    return WeaviateAdapter.create(schema, CONNECTION)


async def test_ensure_collection_on_an_existing_collection_is_idempotent(
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
