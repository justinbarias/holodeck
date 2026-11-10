"""The native Qdrant adapter and the SK bridge read and write the same layout.

Collections created today by the SK bridge (``SKQdrantAdapter``) must stay
readable and writable after the move to ``QdrantAdapter`` (042 T5, FR-017),
and the reverse. Each test uses its own collection on the Qdrant server the
live suite uses (``http://127.0.0.1:6333``) and deletes it afterwards.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    Record,
    RecordFilter,
    VectorQuery,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.adapters.qdrant import QdrantAdapter
from holodeck.lib.retrieval.adapters.sk_bridge import SKQdrantAdapter
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema

QDRANT_URL = "http://127.0.0.1:6333"


def _reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 6333), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _reachable(), reason=f"Qdrant not reachable at {QDRANT_URL}"
    ),
]

QUOTED_SOURCE = 'it\'s "b" \\ dir/b.md'
UUID_ID = str(uuid.UUID("0f8fad5b-d9cb-469f-a165-70867728950e"))
UPPER_UUID_ID = "7C9E6679-7425-40DE-944B-E07FC1F90AE7"


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
    note: str | None = None,
) -> Record:
    fields: dict[str, str | int | float | bool | None] = {
        "source_path": source,
        "chunk_index": index,
        "content": content,
        "mtime": mtime,
        "pinned": pinned,
    }
    if note is not None:
        fields["note"] = note
    return Record(id=record_id, fields=fields, vector=vector)


CORPUS: list[Record] = [
    rec("a_chunk_0", [1.0, 0.0, 0.0, 0.0], "a.md", 0, "alpha apple orchard"),
    rec("a_chunk_1", [0.9, 0.1, 0.0, 0.0], "a.md", 1, "alpha banana grove"),
    rec(
        'it\'s "quoted" id',
        [0.0, 1.0, 0.0, 0.0],
        QUOTED_SOURCE,
        0,
        "beta cherry tree",
        mtime=2.25,
        pinned=True,
        note="n",
    ),
    rec("42", [0.0, 0.0, 1.0, 0.0], QUOTED_SOURCE, 1, "beta zeppelin hangar"),
    rec(UUID_ID, [0.0, 0.0, 0.2, 1.0], "c.md", 0, "gamma delta"),
    rec(UPPER_UUID_ID, [0.3, 0.3, 0.3, 0.3], "c.md", 1, "gamma epsilon, ok?"),
    rec("/srv/x/doc.md_chunk_3", [0.5, 0.0, 0.5, 0.0], "d.md", 3, "kestrel dawn"),
]
IDS = [record.id for record in CORPUS] + ["missing"]

VECTOR = [0.8, 0.2, 0.1, 0.0]
# Single-match, no-match and punctuation-only queries; multi-match queries
# are unordered on Qdrant (see the retrieval goldens) and are not compared.
HYBRID_TEXTS = ["zeppelin", "kestrel", "zzqx plugh", "?!?", ""]


def bridge(schema: CollectionSchema) -> VectorStoreAdapter:
    return SKQdrantAdapter.create(schema, {"connection_string": QDRANT_URL})


def native(schema: CollectionSchema) -> VectorStoreAdapter:
    return QdrantAdapter.create(schema, {"connection_string": QDRANT_URL})


@pytest_asyncio.fixture
async def raw() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=QDRANT_URL)
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def names(raw: AsyncQdrantClient) -> AsyncIterator[list[str]]:
    """Unique collection names; every one is deleted afterwards."""
    created: list[str] = []
    try:
        yield created
    finally:
        for name in created:
            if await raw.collection_exists(name):
                await raw.delete_collection(name)


def unique(names: list[str]) -> str:
    name = f"compat_{uuid.uuid4().hex[:12]}"
    names.append(name)
    return name


# -- comparison helpers ----------------------------------------------------------


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
            return rows
    raise AssertionError("list_page did not finish")


async def snapshot(adapter: VectorStoreAdapter) -> dict[str, Any]:
    """Every read the tools use, in comparable form."""
    got = await adapter.get(IDS, include_vectors=True)
    filters = [
        RecordFilter.where(source_path=QUOTED_SOURCE),
        RecordFilter.where(source_path="a.md", chunk_index=1),
        RecordFilter.where(pinned=True),
        RecordFilter.where(mtime=2.25),
        RecordFilter.where(source_path="nope"),
    ]
    found = [sorted(_row(r) for r in await adapter.find(f)) for f in filters]
    searches = []
    for record_filter in (None, RecordFilter.where(source_path=QUOTED_SOURCE)):
        hits = await adapter.vector_search(VectorQuery(VECTOR, 5, record_filter))
        searches.append([(h.record.id, dict(h.record.fields), h.score) for h in hits])
    hybrid = []
    for text in HYBRID_TEXTS:
        for record_filter in (None, RecordFilter.where(source_path="a.md")):
            hits = await adapter.hybrid_search(
                HybridQuery(text, VECTOR, 3, "content", record_filter)
            )
            hybrid.append([(h.record.id, h.score, h.kind) for h in hits])
    return {
        "get": [_row(r) for r in got],
        "vectors": [list(r.vector or ()) for r in got],
        "find": found,
        "find_limit": len(
            await adapter.find(RecordFilter.where(source_path="a.md"), limit=1)
        ),
        "list": await _walk(adapter, 2),
        "count": await adapter.count(),
        "vector_search": searches,
        "hybrid": hybrid,
    }


def assert_same(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["get"] == right["get"]
    assert len(left["get"]) == len(CORPUS)
    for a, b in zip(left["vectors"], right["vectors"], strict=True):
        assert a == pytest.approx(b, abs=1e-6)
    assert left["find"] == right["find"]
    assert left["find_limit"] == right["find_limit"] == 1
    assert left["list"] == right["list"]
    assert left["count"] == right["count"] == len(CORPUS)
    for a_hits, b_hits in zip(
        left["vector_search"], right["vector_search"], strict=True
    ):
        assert [(i, f) for i, f, _ in a_hits] == [(i, f) for i, f, _ in b_hits]
        assert [s for *_, s in a_hits] == pytest.approx(
            [s for *_, s in b_hits], abs=1e-6
        )
    for a_hits, b_hits in zip(left["hybrid"], right["hybrid"], strict=True):
        assert [(i, k) for i, _, k in a_hits] == [(i, k) for i, _, k in b_hits]
        assert [s for _, s, _ in a_hits] == pytest.approx(
            [s for _, s, _ in b_hits], abs=1e-6
        )


async def _close(*adapters: VectorStoreAdapter) -> None:
    for adapter in adapters:
        await adapter.close()


# -- tests -----------------------------------------------------------------------


async def test_native_reads_a_bridge_collection_identically(names: list[str]) -> None:
    schema = make_schema(unique(names))
    old, new = bridge(schema), native(schema)
    try:
        await old.ensure_collection()
        await old.upsert(CORPUS)
        expected = await snapshot(old)
        assert_same(await snapshot(new), expected)
        # Sanity: the single-match hybrid query finds its chunk.
        ids = {hit[0] for hit in expected["hybrid"][0]}
        assert "42" in ids
    finally:
        await _close(old, new)


async def test_bridge_reads_a_native_collection_identically(names: list[str]) -> None:
    schema = make_schema(unique(names))
    old, new = bridge(schema), native(schema)
    try:
        await new.ensure_collection()
        await new.upsert(CORPUS)
        assert_same(await snapshot(old), await snapshot(new))
    finally:
        await _close(old, new)


@pytest.mark.parametrize(
    "build",
    [
        make_schema,
        lambda name: document_schema(4, {"collection_name": name}),
        lambda name: hierarchical_schema(4, "tool", {"collection_name": name}),
    ],
    ids=["all-types", "document", "hierarchical"],
)
async def test_collection_config_and_indexes_are_equal(
    names: list[str], raw: AsyncQdrantClient, build: Any
) -> None:
    old_schema: CollectionSchema = build(unique(names))
    new_schema: CollectionSchema = build(unique(names))
    old, new = bridge(old_schema), native(new_schema)
    try:
        await old.ensure_collection()
        await new.ensure_collection()
        old_info = await raw.get_collection(old_schema.name)
        new_info = await raw.get_collection(new_schema.name)
    finally:
        await _close(old, new)

    assert new_info.config.params == old_info.config.params
    assert new_info.config.params.vectors == old_info.config.params.vectors
    assert new_info.config.hnsw_config == old_info.config.hnsw_config
    assert new_info.config.optimizer_config == old_info.config.optimizer_config
    assert new_info.config.quantization_config == old_info.config.quantization_config

    def indexes(info: Any) -> dict[str, tuple[Any, Any]]:
        return {
            name: (index.data_type, index.params)
            for name, index in info.payload_schema.items()
        }

    assert indexes(new_info) == indexes(old_info)
    expected = {
        spec.name for spec in new_schema.fields if spec.indexed or spec.full_text
    }
    assert set(indexes(new_info)) == expected


async def test_ensure_collection_on_an_existing_bridge_collection_is_idempotent(
    names: list[str], raw: AsyncQdrantClient
) -> None:
    schema = make_schema(unique(names))
    old = bridge(schema)
    try:
        await old.ensure_collection()
        await old.upsert(CORPUS)
        before = await raw.get_collection(schema.name)
        for _ in range(2):
            new = native(schema)
            await new.ensure_collection()
            await new.close()
        after = await raw.get_collection(schema.name)
    finally:
        await old.close()

    assert after.config == before.config
    assert {k: (v.data_type, v.params) for k, v in after.payload_schema.items()} == {
        k: (v.data_type, v.params) for k, v in before.payload_schema.items()
    }
    assert after.points_count == len(CORPUS)


async def _raw_points(
    raw: AsyncQdrantClient, name: str
) -> dict[str, tuple[dict[str, Any], Sequence[float]]]:
    points, _ = await raw.scroll(name, limit=100, with_payload=True, with_vectors=True)
    result: dict[str, tuple[dict[str, Any], Sequence[float]]] = {}
    for point in points:
        assert isinstance(point.vector, dict)
        vector = point.vector["embedding"]
        assert isinstance(vector, list)
        result[str(point.id)] = (dict(point.payload or {}), vector)
    return result


async def test_stored_points_are_identical(
    names: list[str], raw: AsyncQdrantClient
) -> None:
    """Same collection name, written once by each adapter: same points."""
    schema = make_schema(unique(names))
    old = bridge(schema)
    try:
        await old.ensure_collection()
        await old.upsert(CORPUS)
        written_by_bridge = await _raw_points(raw, schema.name)
        await old.delete_collection()
    finally:
        await old.close()
    new = native(schema)
    try:
        await new.ensure_collection()
        await new.upsert(CORPUS)
        written_by_native = await _raw_points(raw, schema.name)
    finally:
        await new.close()

    assert written_by_native.keys() == written_by_bridge.keys()
    for point_id, (payload, vector) in written_by_native.items():
        old_payload, old_vector = written_by_bridge[point_id]
        assert payload == old_payload
        assert {k: type(v) for k, v in payload.items()} == {
            k: type(v) for k, v in old_payload.items()
        }
        assert vector == pytest.approx(old_vector, abs=1e-7)
    # UUID IDs are stored as canonical UUIDs; other IDs keep the original.
    assert UUID_ID in written_by_native
    assert "holodeck_id" not in written_by_native[UUID_ID][0]
    upper = written_by_native[UPPER_UUID_ID.lower()][0]
    assert upper["holodeck_id"] == UPPER_UUID_ID
    assert sum("holodeck_id" in p for p, _ in written_by_native.values()) == 6
    # Missing fields get the type default.
    assert written_by_native[UUID_ID][0]["note"] == ""


async def test_native_writes_to_a_bridge_collection_stay_consistent(
    names: list[str],
) -> None:
    schema = make_schema(unique(names))
    old, new = bridge(schema), native(schema)
    try:
        await old.ensure_collection()
        await old.upsert(CORPUS)

        await new.ensure_collection()
        await new.upsert(
            [
                rec("a_chunk_0", [0.0, 0.0, 0.0, 1.0], "a.md", 0, "rewritten"),
                rec("e_chunk_0", [0.1, 0.2, 0.3, 0.4], "e.md", 0, "new zeppelin"),
            ]
        )
        await new.delete(["/srv/x/doc.md_chunk_3", "missing"])
        await new.replace_source(
            RecordFilter.where(source_path="a.md"),
            [rec("a_chunk_0", [0.5, 0.5, 0.0, 0.0], "a.md", 0, "shorter")],
        )
        assert await new.delete_where(RecordFilter.where(source_path="c.md")) == 2

        expected_ids = ["a_chunk_0", 'it\'s "quoted" id', "42", "e_chunk_0"]
        for adapter in (old, new):
            got = await adapter.get(IDS + ["e_chunk_0"])
            assert [r.id for r in got] == expected_ids
            assert got[0].fields["content"] == "shorter"
            assert await adapter.count() == 4
            assert [
                r.id for r in await adapter.find(RecordFilter.where(source_path="a.md"))
            ] == ["a_chunk_0"]
        assert await snapshot_after_writes(old) == await snapshot_after_writes(new)
    finally:
        await _close(old, new)


async def snapshot_after_writes(adapter: VectorStoreAdapter) -> dict[str, Any]:
    hits = await adapter.vector_search(VectorQuery(VECTOR, 10))
    hybrid = await adapter.hybrid_search(HybridQuery("hangar", VECTOR, 3, "content"))
    return {
        "list": await _walk(adapter, 3),
        "vector": [(h.record.id, round(h.score, 6)) for h in hits],
        "hybrid": [(h.record.id, round(h.score, 6)) for h in hybrid],
    }
