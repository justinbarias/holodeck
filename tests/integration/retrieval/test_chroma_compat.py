"""The native Chroma adapter reads and writes the layout the SK path stored (042 T6).

Collections written by HoloDeck through Semantic Kernel must stay readable and
writable through ``ChromaAdapter``. SK is gone (T16), so the SK layout is
frozen under ``tests/fixtures/legacy_layout/chroma/``: collection
configuration, every row, and what SK's path returned for each read, taken on
a Chroma server. Each test rebuilds a legacy collection from those files with
the raw ``chromadb`` client, on the server and in an embedded persistent
directory, or creates a fresh one natively.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
import pytest

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    Record,
    RecordFilter,
    VectorQuery,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.adapters.chroma import ChromaAdapter
from holodeck.lib.retrieval.schemas import document_schema, hierarchical_schema
from tests.integration.retrieval.legacy_layout import (
    assert_close,
    basic_snapshot,
    chroma_layout,
    chroma_rebuild,
    chroma_rows,
    load,
    plain,
    sample_records,
    schema_facts,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8765
CHROMA_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"
DIMENSIONS = 4
STORE = "chroma"


def _server_reachable() -> bool:
    try:
        with socket.create_connection((CHROMA_HOST, CHROMA_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def make_schema(name: str) -> CollectionSchema:
    return CollectionSchema(
        name=name,
        dimensions=DIMENSIONS,
        fields=(
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def _record(
    record_id: str,
    vector: list[float],
    *,
    source: str = "a.md",
    index: int = 0,
    content: str = "",
    mtime: float = 1.5,
    pinned: bool = False,
) -> Record:
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": index,
            "content": content,
            "mtime": mtime,
            "pinned": pinned,
        },
        vector=vector,
    )


CORPUS = [
    _record("a_0", [1.0, 0.0, 0.0, 0.0], content="alpha apple"),
    _record("a_1", [0.9, 0.1, 0.0, 0.0], index=1, content="alpha banana"),
    _record("b_0", [0.0, 1.0, 0.0, 0.0], source="b.md", content="beta", mtime=2.25),
    _record("b_1", [0.0, 0.5, 0.5, 0.0], source="b.md", index=1, pinned=True),
    _record("c_0", [0.0, 0.0, 0.0, 1.0], source="c.md", content="gamma"),
]

SCHEMAS: dict[str, Callable[[str], CollectionSchema]] = {
    "all_types": make_schema,
    "document": lambda name: document_schema(DIMENSIONS, {"collection_name": name}),
    "hierarchical": lambda name: hierarchical_schema(
        DIMENSIONS, "tool", {"collection_name": name}
    ),
}
KINDS = pytest.mark.parametrize("kind", list(SCHEMAS))


def corpus(kind: str, schema: CollectionSchema) -> list[Record]:
    return CORPUS if kind == "all_types" else sample_records(schema)


# -- reads compared with what the SK path returned -----------------------------------


def _fields(record: Record) -> dict[str, object]:
    return dict(record.fields)


async def snapshot(adapter: VectorStoreAdapter) -> dict[str, Any]:
    """Every read the tools use, in comparable form."""
    ids = [record.id for record in CORPUS] + ["missing"]
    plain_get = await adapter.get(ids)
    with_vectors = await adapter.get(ids, include_vectors=True)
    found = []
    for where in (
        RecordFilter.where(source_path="b.md"),
        RecordFilter.where(source_path="a.md", chunk_index=1),
        RecordFilter.where(pinned=True),
        RecordFilter.where(mtime=2.25),
        RecordFilter(),
    ):
        rows = sorted(await adapter.find(where), key=lambda r: r.id)
        found.append([(r.id, _fields(r)) for r in rows])
    pages: list[str] = []
    cursor: str | None = None
    while True:
        page = await adapter.list_page(cursor=cursor, limit=2)
        pages.extend(r.id for r in page.records)
        cursor = page.next_cursor
        if cursor is None:
            break
    searches = []
    for query in (
        # No tie at the top-k boundary, so the hits do not depend on the index.
        VectorQuery([1.0, 0.0, 0.0, 0.2], top_k=3),
        VectorQuery([0.0, 1.0, 1.0, 0.0], top_k=5),
        VectorQuery(
            [1.0, 0.0, 0.0, 0.0], top_k=5, filter=RecordFilter.where(source_path="b.md")
        ),
    ):
        hits = await adapter.vector_search(query)
        # Equal scores come back in index order; compare them by ID.
        ordered = sorted(hits, key=lambda h: (-round(h.score, 5), h.record.id))
        searches.append([(h.record.id, _fields(h.record), h.score) for h in ordered])
    return plain(
        {
            "get": [(r.id, _fields(r)) for r in plain_get],
            "get_has_vectors": [r.vector is not None for r in plain_get],
            "vectors": [(r.id, list(r.vector or ())) for r in with_vectors],
            "find": found,
            "list": sorted(pages),
            "count": await adapter.count(),
            "vector_search": searches,
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
    # upsert replaces an existing ID and adds a new one.
    await adapter.upsert(
        [
            _record("a_0", [0.0, 0.0, 1.0, 0.0], content="rewritten", mtime=9.5),
            _record("d_0", [0.5, 0.5, 0.5, 0.5], source="d.md", content="delta"),
        ]
    )
    # replace_source shrinks a source: the stale chunk goes.
    await adapter.replace_source(
        RecordFilter.where(source_path="b.md"),
        [_record("b_0", [0.0, 1.0, 0.0, 0.0], source="b.md", content="beta v2")],
    )
    await adapter.delete(["c_0", "missing"])
    assert await adapter.delete_where(RecordFilter.where(source_path="d.md")) == 1


async def snapshot_after_writes(adapter: VectorStoreAdapter) -> dict[str, Any]:
    (replaced,) = await adapter.get(["a_0"], include_vectors=True)
    remaining = await adapter.find(RecordFilter.where(source_path="b.md"))
    searches = []
    for vector in ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]):
        hits = await adapter.vector_search(VectorQuery(vector, top_k=5))
        ordered = sorted(hits, key=lambda h: (-round(h.score, 5), h.record.id))
        searches.append([(h.record.id, h.score) for h in ordered])
    return plain(
        {
            "a_0": (_fields(replaced), list(replaced.vector or ())),
            "b": sorted((r.id, r.fields["content"]) for r in remaining),
            "gone": [r.id for r in await adapter.get(["c_0", "d_0"])],
            "count": await adapter.count(),
            "vector_search": searches,
        }
    )


# -- where the legacy collection lives ------------------------------------------------


@dataclass
class Store:
    """A Chroma server or a persistent directory, seen by both clients."""

    name: str
    connection: Mapping[str, object]
    created: list[str] = field(default_factory=list)

    def client(self) -> Any:
        if "connection_string" in self.connection:
            return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        return chromadb.PersistentClient(path=str(self.connection["persist_directory"]))

    def native(self, schema: CollectionSchema) -> ChromaAdapter:
        return ChromaAdapter.create(schema, self.connection)


@pytest.fixture(
    params=[
        pytest.param("server", id="server"),
        pytest.param("embedded", id="embedded"),
    ]
)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Store]:
    if request.param == "server":
        if not _server_reachable():
            pytest.skip(f"Chroma is not reachable at {CHROMA_URL}")
        chosen = Store("server", {"connection_string": CHROMA_URL})
    else:
        chosen = Store("embedded", {"persist_directory": str(tmp_path / "chroma")})
    yield chosen
    client = chosen.client()
    for name in chosen.created:
        if name in {c.name for c in client.list_collections()}:
            client.delete_collection(name)


def unique(store: Store) -> str:
    name = f"compat_{uuid4().hex[:12]}"
    store.created.append(name)
    return name


def legacy(store: Store, kind: str) -> tuple[CollectionSchema, dict[str, Any]]:
    """A legacy collection rebuilt from the frozen SK layout."""
    fixture = load(STORE, kind)
    schema = SCHEMAS[kind](unique(store))
    chroma_rebuild(store.client(), schema.name, fixture)
    return schema, fixture


def raw_collection(store: Store, name: str) -> chromadb.Collection:
    return store.client().get_collection(name, embedding_function=None)


# -- tests ---------------------------------------------------------------------------


@KINDS
def test_the_frozen_schema_is_the_current_schema(kind: str) -> None:
    assert schema_facts(SCHEMAS[kind]("x")) == load(STORE, kind)["schema"]


def test_the_sk_path_layout_is_the_documented_one() -> None:
    """What the SK path stored is what the native adapter's docstring says."""
    fixture = load(STORE, "all_types")
    assert fixture["layout"]["hnsw"]["space"] == "cosine"
    rows = {row["id"]: row for row in fixture["rows"]}
    assert sorted(rows) == sorted(r.id for r in CORPUS)
    assert all(row["document"] is None for row in rows.values())
    first = rows["a_0"]["metadata"]
    assert first == {
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "alpha apple",
        "mtime": 1.5,
        "pinned": False,
    }
    assert isinstance(first["chunk_index"], int)
    assert isinstance(first["pinned"], bool)


@KINDS
async def test_the_rebuilt_collection_is_the_frozen_one(
    store: Store, kind: str
) -> None:
    schema, fixture = legacy(store, kind)
    collection = raw_collection(store, schema.name)
    assert chroma_layout(collection) == fixture["layout"]
    assert_close(chroma_rows(collection), fixture["rows"])


@KINDS
async def test_native_reads_a_legacy_collection_as_the_sk_path_did(
    store: Store, kind: str
) -> None:
    schema, fixture = legacy(store, kind)
    adapter = store.native(schema)
    try:
        assert await adapter.collection_exists() is True
        got = await read_all(kind, adapter, schema)
    finally:
        await adapter.close()
    assert_close(got, fixture["expected"])
    if kind == "all_types":
        assert not any(got["get_has_vectors"])
        assert got["count"] == len(CORPUS)


@KINDS
async def test_native_creates_and_stores_the_legacy_layout(
    store: Store, kind: str
) -> None:
    """A fresh collection written natively has SK's configuration and rows."""
    schema = SCHEMAS[kind](unique(store))
    adapter = store.native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(corpus(kind, schema))
    finally:
        await adapter.close()
    fixture = load(STORE, kind)
    collection = raw_collection(store, schema.name)
    assert chroma_layout(collection) == fixture["layout"]
    rows = chroma_rows(collection)
    assert_close(rows, fixture["rows"])
    for row, record in zip(
        rows, sorted(corpus(kind, schema), key=lambda r: r.id), strict=True
    ):
        assert row["metadata"] == dict(record.fields)
        assert {k: type(v) for k, v in row["metadata"].items()} == {
            k: type(v) for k, v in record.fields.items()
        }


async def test_native_writes_on_a_legacy_collection_stay_consistent(
    store: Store,
) -> None:
    """The same writes on an SK collection read back as SK's path read them."""
    schema, fixture = legacy(store, "all_types")
    adapter = store.native(schema)
    try:
        await native_writes(adapter)
        got = await snapshot_after_writes(adapter)
    finally:
        await adapter.close()
    assert got["a_0"][0]["content"] == "rewritten"
    assert got["a_0"][0]["mtime"] == 9.5
    assert got["a_0"][1] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-5)
    assert got["b"] == [["b_0", "beta v2"]]
    assert got["gone"] == []
    assert got["count"] == 3
    assert_close(got, fixture["after_writes"])


async def test_an_embedded_legacy_directory_is_reopened(tmp_path: Path) -> None:
    store = Store("embedded", {"persist_directory": str(tmp_path / "persisted")})
    schema, _ = legacy(store, "all_types")

    reopened = store.native(schema)
    try:
        assert await reopened.collection_exists() is True
        assert await reopened.count() == len(CORPUS)
        (record,) = await reopened.get(["b_0"], include_vectors=True)
        assert record.fields["mtime"] == 2.25
        hits = await reopened.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], top_k=2))
        assert [h.record.id for h in hits] == ["a_0", "a_1"]
        await reopened.upsert([_record("e_0", [0.0, 0.0, 0.0, 1.0], source="e.md")])
        await reopened.delete(["a_1"])
    finally:
        await reopened.close()

    again = raw_collection(store, schema.name)
    assert again.count() == len(CORPUS)
    assert again.get(ids=["e_0", "a_1"])["ids"] == ["e_0"]
