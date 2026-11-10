"""The native Qdrant adapter reads and writes the layout the SK path stored.

Collections created by HoloDeck through Semantic Kernel must stay readable and
writable through ``QdrantAdapter`` (042 T5, FR-017, AC-05). SK is gone (T16),
so the SK layout is frozen under ``tests/fixtures/legacy_layout/qdrant/``:
collection config, payload indexes, every point, and what SK's path returned
for each read. Each test rebuilds a legacy collection from those files with
the raw Qdrant client, or creates a fresh one natively, on the Qdrant server
the live suite uses (``http://127.0.0.1:6333``), and deletes it afterwards.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import AsyncIterator, Callable
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
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from tests.integration.retrieval.legacy_layout import (
    assert_close,
    basic_snapshot,
    load,
    plain,
    qdrant_layout,
    qdrant_point_id,
    qdrant_rebuild,
    qdrant_rows,
    qdrant_rows_for,
    sample_records,
    schema_facts,
)

QDRANT_URL = "http://127.0.0.1:6333"
STORE = "qdrant"


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

SCHEMAS: dict[str, Callable[[str], CollectionSchema]] = {
    "all_types": make_schema,
    "document": lambda name: document_schema(4, {"collection_name": name}),
    "hierarchical": lambda name: hierarchical_schema(
        4, "tool", {"collection_name": name}
    ),
}


def corpus(kind: str, schema: CollectionSchema) -> list[Record]:
    return CORPUS if kind == "all_types" else sample_records(schema)


def native(schema: CollectionSchema) -> VectorStoreAdapter:
    return QdrantAdapter.create(schema, {"connection_string": QDRANT_URL})


# -- reads compared with what the SK path returned -----------------------------------


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
            # Pages follow point IDs, which depend on the collection name.
            return sorted(rows, key=lambda row: row[0])
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
    return plain(
        {
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
    )


async def read_all(
    kind: str, adapter: VectorStoreAdapter, schema: CollectionSchema
) -> dict[str, Any]:
    if kind == "all_types":
        return await snapshot(adapter)
    return await basic_snapshot(adapter, schema)


async def native_writes(adapter: VectorStoreAdapter) -> None:
    """The writes the tools make, applied to a collection holding ``CORPUS``."""
    await adapter.ensure_collection()
    await adapter.upsert(
        [
            rec("a_chunk_0", [0.0, 0.0, 0.0, 1.0], "a.md", 0, "rewritten"),
            rec("e_chunk_0", [0.1, 0.2, 0.3, 0.4], "e.md", 0, "new zeppelin"),
        ]
    )
    await adapter.delete(["/srv/x/doc.md_chunk_3", "missing"])
    await adapter.replace_source(
        RecordFilter.where(source_path="a.md"),
        [rec("a_chunk_0", [0.5, 0.5, 0.0, 0.0], "a.md", 0, "shorter")],
    )
    assert await adapter.delete_where(RecordFilter.where(source_path="c.md")) == 2


async def snapshot_after_writes(adapter: VectorStoreAdapter) -> dict[str, Any]:
    got = await adapter.get(IDS + ["e_chunk_0"])
    hits = await adapter.vector_search(VectorQuery(VECTOR, 10))
    hybrid = await adapter.hybrid_search(HybridQuery("hangar", VECTOR, 3, "content"))
    return plain(
        {
            "get": [_row(r) for r in got],
            "count": await adapter.count(),
            "find_a": [
                r.id for r in await adapter.find(RecordFilter.where(source_path="a.md"))
            ],
            "list": await _walk(adapter, 3),
            "vector": [(h.record.id, h.score) for h in hits],
            "hybrid": [(h.record.id, h.score) for h in hybrid],
        }
    )


# -- fixtures ------------------------------------------------------------------------


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


async def legacy(
    raw: AsyncQdrantClient, names: list[str], kind: str
) -> tuple[CollectionSchema, dict[str, Any]]:
    """A legacy collection rebuilt from the frozen SK layout."""
    fixture = load(STORE, kind)
    schema = SCHEMAS[kind](unique(names))
    await qdrant_rebuild(raw, schema.name, fixture)
    return schema, fixture


KINDS = pytest.mark.parametrize("kind", list(SCHEMAS))


# -- tests ---------------------------------------------------------------------------


@KINDS
def test_the_frozen_schema_is_the_current_schema(kind: str) -> None:
    assert schema_facts(SCHEMAS[kind]("x")) == load(STORE, kind)["schema"]


@KINDS
def test_frozen_point_ids_follow_the_documented_rule(kind: str) -> None:
    fixture = load(STORE, kind)
    for row in fixture["rows"]:
        holodeck_id = row["payload"].get("holodeck_id", row["id"])
        assert row["id"] == qdrant_point_id(fixture["collection"], holodeck_id)
    assert qdrant_rows_for(fixture, fixture["collection"]) == fixture["rows"]


@KINDS
async def test_the_rebuilt_collection_is_the_frozen_one(
    raw: AsyncQdrantClient, names: list[str], kind: str
) -> None:
    schema, fixture = await legacy(raw, names, kind)
    assert await qdrant_layout(raw, schema.name) == fixture["layout"]
    assert_close(
        await qdrant_rows(raw, schema.name),
        qdrant_rows_for(fixture, schema.name),
        tol=1e-7,
    )


@KINDS
async def test_native_reads_a_legacy_collection_as_the_sk_path_did(
    raw: AsyncQdrantClient, names: list[str], kind: str
) -> None:
    schema, fixture = await legacy(raw, names, kind)
    adapter = native(schema)
    try:
        got = await read_all(kind, adapter, schema)
    finally:
        await adapter.close()
    assert_close(got, fixture["expected"], tol=1e-6)
    if kind == "all_types":
        assert len(got["get"]) == len(CORPUS)
        assert got["find_limit"] == 1
        assert got["count"] == len(CORPUS)
        # Sanity: the single-match hybrid query finds its chunk.
        assert "42" in {hit[0] for hit in got["hybrid"][0]}


@KINDS
async def test_native_creates_the_legacy_config_and_indexes(
    raw: AsyncQdrantClient, names: list[str], kind: str
) -> None:
    schema = SCHEMAS[kind](unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
    finally:
        await adapter.close()
    layout = await qdrant_layout(raw, schema.name)
    assert layout == load(STORE, kind)["layout"]
    expected = {spec.name for spec in schema.fields if spec.indexed or spec.full_text}
    assert set(layout["payload_schema"]) == expected


@KINDS
async def test_native_stores_the_legacy_points(
    raw: AsyncQdrantClient, names: list[str], kind: str
) -> None:
    """A fresh collection written natively holds exactly the points SK wrote."""
    schema = SCHEMAS[kind](unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(corpus(kind, schema))
    finally:
        await adapter.close()
    rows = await qdrant_rows(raw, schema.name)
    assert_close(rows, qdrant_rows_for(load(STORE, kind), schema.name), tol=1e-7)
    if kind != "all_types":
        return
    by_id = {row["id"]: row["payload"] for row in rows}
    # UUID IDs are stored as canonical UUIDs; other IDs keep the original.
    assert UUID_ID in by_id
    assert "holodeck_id" not in by_id[UUID_ID]
    assert by_id[UPPER_UUID_ID.lower()]["holodeck_id"] == UPPER_UUID_ID
    assert sum("holodeck_id" in payload for payload in by_id.values()) == 6
    # Missing fields get the type default.
    assert by_id[UUID_ID]["note"] == ""


async def test_ensure_collection_on_a_legacy_collection_is_idempotent(
    raw: AsyncQdrantClient, names: list[str]
) -> None:
    schema, _ = await legacy(raw, names, "all_types")
    before = await raw.get_collection(schema.name)
    for _ in range(2):
        adapter = native(schema)
        await adapter.ensure_collection()
        await adapter.close()
    after = await raw.get_collection(schema.name)

    assert after.config == before.config
    assert {k: (v.data_type, v.params) for k, v in after.payload_schema.items()} == {
        k: (v.data_type, v.params) for k, v in before.payload_schema.items()
    }
    assert after.points_count == len(CORPUS)


async def test_native_writes_to_a_legacy_collection_stay_consistent(
    raw: AsyncQdrantClient, names: list[str]
) -> None:
    """The same writes on an SK collection read back as SK's path read them."""
    schema, fixture = await legacy(raw, names, "all_types")
    adapter = native(schema)
    try:
        await native_writes(adapter)
        got = await snapshot_after_writes(adapter)
    finally:
        await adapter.close()
    assert [row[0] for row in got["get"]] == [
        "a_chunk_0",
        'it\'s "quoted" id',
        "42",
        "e_chunk_0",
    ]
    assert got["get"][0][1]["content"] == "shorter"
    assert got["count"] == 4
    assert got["find_a"] == ["a_chunk_0"]
    assert_close(got, fixture["after_writes"], tol=1e-6)


async def test_replace_source_keeps_a_record_whose_uuid_spelling_changed(
    names: list[str],
) -> None:
    schema = make_schema(unique(names))
    adapter = native(schema)
    upper = "8F14E45F-CEEA-467F-A0E6-0C0B6C0B6C0B"
    try:
        await adapter.ensure_collection()
        first = rec(upper, [1.0, 0.0, 0.0, 0.0], "u.md", 0, "uuid spelling")
        await adapter.upsert([first])
        source = RecordFilter.where(source_path=str(first.fields["source_path"]))
        replacement = Record(id=upper.lower(), fields=first.fields, vector=first.vector)
        await adapter.replace_source(source, [replacement])
        assert await adapter.count() == 1
    finally:
        await adapter.close()
