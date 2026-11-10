"""The native Chroma adapter and the Semantic Kernel path share one layout (042 T6).

A collection written by either side must read back identically through the
other, on a Chroma server and in an embedded persistent directory. The SK path
is ``SKChromaAdapter`` in the temporary bridge (removed in task T16); these
tests go with it.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
import pytest_asyncio

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
from holodeck.lib.retrieval.adapters.sk_bridge import SKChromaAdapter

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8765
CHROMA_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"
DIMENSIONS = 4


def _server_reachable() -> bool:
    try:
        with socket.create_connection((CHROMA_HOST, CHROMA_PORT), timeout=0.5):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class Store:
    """Where the two adapters meet: a server URL or a persistent directory."""

    name: str
    connection: Mapping[str, object]


def _schema() -> CollectionSchema:
    return CollectionSchema(
        name=f"compat_{uuid4().hex[:12]}",
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


@pytest.fixture(
    params=[
        pytest.param("server", id="server"),
        pytest.param("embedded", id="embedded"),
    ]
)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    if request.param == "server":
        if not _server_reachable():
            pytest.skip(f"Chroma is not reachable at {CHROMA_URL}")
        return Store("server", {"connection_string": CHROMA_URL})
    return Store("embedded", {"persist_directory": str(tmp_path / "chroma")})


@dataclass
class Pair:
    """Both adapters on one collection."""

    schema: CollectionSchema
    store: Store
    make_native: Callable[[], ChromaAdapter]
    make_bridge: Callable[[], SKChromaAdapter]


@pytest_asyncio.fixture
async def pair(store: Store) -> AsyncIterator[Pair]:
    schema = _schema()
    opened: list[VectorStoreAdapter] = []

    def make_native() -> ChromaAdapter:
        adapter = ChromaAdapter.create(schema, store.connection)
        opened.append(adapter)
        return adapter

    def make_bridge() -> SKChromaAdapter:
        adapter = SKChromaAdapter.create(schema, store.connection)
        opened.append(adapter)
        return adapter

    try:
        yield Pair(schema, store, make_native, make_bridge)
    finally:
        cleaner = ChromaAdapter.create(schema, store.connection)
        try:
            await cleaner.delete_collection()
        finally:
            await cleaner.close()
        for adapter in opened:
            await adapter.close()


def _fields(record: Record) -> dict[str, object]:
    return dict(record.fields)


async def _assert_same_reads(
    left: VectorStoreAdapter, right: VectorStoreAdapter
) -> None:
    """Both adapters return identical data for every read operation."""
    ids = [record.id for record in CORPUS] + ["missing"]

    left_get = await left.get(ids)
    right_get = await right.get(ids)
    assert [r.id for r in left_get] == [r.id for r in right_get]
    assert [_fields(r) for r in left_get] == [_fields(r) for r in right_get]
    assert all(r.vector is None for r in left_get + right_get)

    left_vec = await left.get(ids, include_vectors=True)
    right_vec = await right.get(ids, include_vectors=True)
    for a, b in zip(left_vec, right_vec, strict=True):
        assert a.vector is not None
        assert b.vector is not None
        assert list(a.vector) == pytest.approx(list(b.vector), abs=1e-5)

    for where in (
        RecordFilter.where(source_path="b.md"),
        RecordFilter.where(source_path="a.md", chunk_index=1),
        RecordFilter.where(pinned=True),
        RecordFilter.where(mtime=2.25),
        RecordFilter(),
    ):
        left_found = sorted(await left.find(where), key=lambda r: r.id)
        right_found = sorted(await right.find(where), key=lambda r: r.id)
        assert [r.id for r in left_found] == [r.id for r in right_found]
        assert [_fields(r) for r in left_found] == [_fields(r) for r in right_found]

    left_pages: list[str] = []
    right_pages: list[str] = []
    for adapter, seen in ((left, left_pages), (right, right_pages)):
        cursor: str | None = None
        while True:
            page = await adapter.list_page(cursor=cursor, limit=2)
            seen.extend(r.id for r in page.records)
            cursor = page.next_cursor
            if cursor is None:
                break
    assert sorted(left_pages) == sorted(right_pages)

    assert await left.count() == await right.count()

    for query in (
        VectorQuery([1.0, 0.0, 0.0, 0.0], top_k=3),
        VectorQuery([0.0, 1.0, 1.0, 0.0], top_k=5),
        VectorQuery(
            [1.0, 0.0, 0.0, 0.0], top_k=5, filter=RecordFilter.where(source_path="b.md")
        ),
    ):
        left_hits = await left.vector_search(query)
        right_hits = await right.vector_search(query)
        assert [h.record.id for h in left_hits] == [h.record.id for h in right_hits]
        assert [_fields(h.record) for h in left_hits] == [
            _fields(h.record) for h in right_hits
        ]
        assert [h.score for h in left_hits] == pytest.approx(
            [h.score for h in right_hits], abs=1e-5
        )


def _raw_collection(pair: Pair) -> chromadb.Collection:
    if "connection_string" in pair.store.connection:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    else:
        client = chromadb.PersistentClient(
            path=str(pair.store.connection["persist_directory"])
        )
    return client.get_collection(pair.schema.name, embedding_function=None)


# -- (a) the SK path writes, the native adapter reads ----------------------------


async def test_native_reads_a_collection_written_by_the_sk_path(pair: Pair) -> None:
    bridge = pair.make_bridge()
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS)

    await _assert_same_reads(pair.make_native(), bridge)


async def test_the_sk_path_layout_is_the_documented_one(pair: Pair) -> None:
    """What the SK path stores is what the native adapter's docstring says."""
    bridge = pair.make_bridge()
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS[:2])
    raw = _raw_collection(pair)
    hnsw = raw.configuration.get("hnsw")
    assert hnsw is not None
    assert hnsw.get("space") == "cosine"
    stored = raw.get(include=["metadatas", "documents", "embeddings"])
    assert sorted(stored["ids"]) == ["a_0", "a_1"]
    assert stored["documents"] == [None, None]
    assert stored["metadatas"] is not None
    first = dict(stored["metadatas"][stored["ids"].index("a_0")])
    assert first == {
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "alpha apple",
        "mtime": 1.5,
        "pinned": False,
    }
    assert isinstance(first["chunk_index"], int)
    assert isinstance(first["pinned"], bool)


async def test_the_native_layout_equals_the_sk_path_layout(pair: Pair) -> None:
    """Both adapters write the same collection configuration and rows."""
    native = pair.make_native()
    await native.ensure_collection()
    await native.upsert(CORPUS)
    raw = _raw_collection(pair)
    hnsw = raw.configuration.get("hnsw")
    assert hnsw is not None
    assert hnsw.get("space") == "cosine"
    stored = raw.get(include=["metadatas", "documents", "embeddings"])
    assert sorted(stored["ids"]) == sorted(r.id for r in CORPUS)
    assert all(document is None for document in (stored["documents"] or []))
    metadatas = stored["metadatas"] or []
    by_id = dict(zip(stored["ids"], metadatas, strict=True))
    for record in CORPUS:
        assert dict(by_id[record.id]) == dict(record.fields)
        assert {k: type(v) for k, v in by_id[record.id].items()} == {
            k: type(v) for k, v in record.fields.items()
        }

    await _assert_same_reads(native, pair.make_bridge())


# -- (b) the native adapter writes, the SK path reads ----------------------------


async def test_the_sk_path_reads_a_collection_written_natively(pair: Pair) -> None:
    native = pair.make_native()
    await native.ensure_collection()
    await native.upsert(CORPUS)

    await _assert_same_reads(native, pair.make_bridge())


# -- (c) native writes on a collection the SK path created ----------------------


async def test_native_writes_on_an_sk_collection_are_read_back_by_the_sk_path(
    pair: Pair,
) -> None:
    bridge = pair.make_bridge()
    native = pair.make_native()
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS)

    # upsert replaces an existing ID and adds a new one.
    await native.upsert(
        [
            _record("a_0", [0.0, 0.0, 1.0, 0.0], content="rewritten", mtime=9.5),
            _record("d_0", [0.5, 0.5, 0.5, 0.5], source="d.md", content="delta"),
        ]
    )
    (replaced,) = await bridge.get(["a_0"], include_vectors=True)
    assert replaced.fields["content"] == "rewritten"
    assert replaced.fields["mtime"] == 9.5
    assert replaced.vector is not None
    assert list(replaced.vector) == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-5)
    assert await bridge.count() == len(CORPUS) + 1

    # replace_source shrinks a source: the stale chunk goes.
    await native.replace_source(
        RecordFilter.where(source_path="b.md"),
        [_record("b_0", [0.0, 1.0, 0.0, 0.0], source="b.md", content="beta v2")],
    )
    remaining = await bridge.find(RecordFilter.where(source_path="b.md"))
    assert [(r.id, r.fields["content"]) for r in remaining] == [("b_0", "beta v2")]

    # delete and delete_where.
    await native.delete(["c_0", "missing"])
    assert await bridge.get(["c_0"]) == []
    assert await native.delete_where(RecordFilter.where(source_path="d.md")) == 1
    assert await bridge.get(["d_0"]) == []

    await _assert_same_reads_after_edits(native, bridge)


async def _assert_same_reads_after_edits(
    native: VectorStoreAdapter, bridge: VectorStoreAdapter
) -> None:
    assert await native.count() == await bridge.count() == 3
    for vector in ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]):
        left = await native.vector_search(VectorQuery(vector, top_k=5))
        right = await bridge.vector_search(VectorQuery(vector, top_k=5))
        assert [h.record.id for h in left] == [h.record.id for h in right]
        assert [h.score for h in left] == pytest.approx(
            [h.score for h in right], abs=1e-5
        )


# -- (d) an embedded directory written by the SK path, reopened natively ---------


async def test_an_embedded_directory_written_by_the_sk_path_is_reopened(
    tmp_path: Path,
) -> None:
    schema = _schema()
    connection = {"persist_directory": str(tmp_path / "persisted")}
    bridge = SKChromaAdapter.create(schema, connection)
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS)
    await bridge.close()

    reopened = ChromaAdapter.create(schema, connection)
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

    again = SKChromaAdapter.create(schema, connection)
    try:
        assert await again.count() == len(CORPUS)
        assert [r.id for r in await again.get(["e_0", "a_1"])] == ["e_0"]
    finally:
        await again.delete_collection()
        await again.close()
