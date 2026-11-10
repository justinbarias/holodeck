"""Tests for the native ChromaDB adapter (042 T6).

Connection parsing, client construction and filter building run without a
server. Behavior tests use an embedded persistent database in ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import chromadb
import pytest
import pytest_asyncio
from chromadb.api.models.Collection import Collection

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    IncompatibleCollectionError,
    MissingClientError,
    Operation,
    Record,
    RecordFilter,
    RetrievalError,
    SchemaError,
    StoreConnectionError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import chroma
from holodeck.lib.retrieval.adapters.chroma import (
    ChromaAdapter,
    ChromaSettings,
    parse_chromadb_connection_string,
    parse_connection,
)
from holodeck.lib.retrieval.registry import PROVIDERS

DIMENSIONS = 4


def _schema(name: str = "chroma_unit") -> CollectionSchema:
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


def _record(record_id: str, vector: list[float], **fields: object) -> Record:
    values: dict[str, str | int | float | bool | None] = {
        "source_path": "a.md",
        "chunk_index": 0,
        "content": "text",
        "mtime": 1.5,
        "pinned": False,
    }
    values.update(cast("dict[str, str | int | float | bool | None]", fields))
    return Record(id=record_id, fields=values, vector=vector)


@pytest.fixture
def directory(tmp_path: Path) -> str:
    return str(tmp_path / "chroma")


@pytest_asyncio.fixture
async def adapter(directory: str) -> AsyncIterator[ChromaAdapter]:
    instance = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    try:
        yield instance
    finally:
        await instance.close()


# -- connection string -------------------------------------------------------


class TestParseConnectionString:
    @pytest.mark.parametrize(
        ("url", "host", "port", "ssl"),
        [
            ("http://localhost:8000", "localhost", 8000, False),
            ("https://chroma.example.com", "chroma.example.com", 443, True),
            ("http://localhost", "localhost", 8000, False),
            ("https://chroma.internal:9000", "chroma.internal", 9000, True),
            ("http://localhost:8000/api/v1", "localhost", 8000, False),
            ("http://192.168.1.100:8000", "192.168.1.100", 8000, False),
            (
                "https://chroma.prod.example.com:443",
                "chroma.prod.example.com",
                443,
                True,
            ),
            ("http://:8000", "localhost", 8000, False),
        ],
    )
    def test_parses_host_port_and_ssl(
        self, url: str, host: str, port: int, ssl: bool
    ) -> None:
        assert parse_chromadb_connection_string(url) == {
            "host": host,
            "port": port,
            "ssl": ssl,
        }

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_chromadb_connection_string("")

    @pytest.mark.parametrize(
        "url", ["ftp://localhost:8000", "localhost:8000", "tcp://localhost:8000"]
    )
    def test_other_schemes_are_rejected(self, url: str) -> None:
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_chromadb_connection_string(url)


# -- settings and client construction -----------------------------------------


class TestParseConnection:
    def test_defaults_to_an_ephemeral_database(self) -> None:
        assert parse_connection({}) == ChromaSettings()

    def test_reads_every_known_key(self) -> None:
        settings = parse_connection(
            {
                "connection_string": "https://chroma.example.com",
                "persist_directory": "/data",
                "headers": {"Authorization": "Bearer x"},
                "tenant": "t",
                "database": "d",
            }
        )
        assert settings == ChromaSettings(
            connection_string="https://chroma.example.com",
            persist_directory="/data",
            headers={"Authorization": "Bearer x"},
            tenant="t",
            database="d",
        )

    def test_empty_strings_count_as_unset(self) -> None:
        assert parse_connection({"connection_string": "", "tenant": ""}) == (
            ChromaSettings()
        )

    def test_a_path_is_a_valid_persist_directory(self, tmp_path: Path) -> None:
        settings = parse_connection({"persist_directory": tmp_path})
        assert settings.persist_directory == str(tmp_path)

    def test_unknown_keys_are_ignored_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            settings = parse_connection({"collection_name": "x", "tenant": "t"})
        assert settings.tenant == "t"
        assert "collection_name" in caplog.text

    @pytest.mark.parametrize(
        "connection",
        [
            {"connection_string": "ftp://host"},
            {"connection_string": 8000},
            {"headers": ["a"]},
            {"headers": {"a": 1}},
            {"tenant": 3},
        ],
    )
    def test_invalid_values_are_rejected(self, connection: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            parse_connection(connection)

    def test_create_turns_a_bad_setting_into_a_retrieval_error(self) -> None:
        with pytest.raises(RetrievalError, match="Invalid ChromaDB connection"):
            ChromaAdapter.create(_schema(), {"connection_string": "ftp://host"})


class TestCreateClient:
    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        module = MagicMock(name="chromadb")
        monkeypatch.setattr(chroma, "chromadb", module)
        return module

    def test_http_client_for_a_connection_string(self, fake: MagicMock) -> None:
        client = chroma.create_chromadb_client(
            ChromaSettings(connection_string="http://localhost:8000")
        )
        fake.HttpClient.assert_called_once_with(
            host="localhost",
            port=8000,
            ssl=False,
            headers=None,
            tenant="default_tenant",
            database="default_database",
        )
        assert client is fake.HttpClient.return_value

    def test_https_client_with_headers_tenant_and_database(
        self, fake: MagicMock
    ) -> None:
        chroma.create_chromadb_client(
            ChromaSettings(
                connection_string="https://chroma.example.com",
                headers={"Authorization": "Bearer token123"},
                tenant="my_tenant",
                database="my_database",
            )
        )
        fake.HttpClient.assert_called_once_with(
            host="chroma.example.com",
            port=443,
            ssl=True,
            headers={"Authorization": "Bearer token123"},
            tenant="my_tenant",
            database="my_database",
        )

    def test_persistent_client_for_a_directory(self, fake: MagicMock) -> None:
        client = chroma.create_chromadb_client(
            ChromaSettings(
                persist_directory="/var/data/chromadb", tenant="c", database="d"
            )
        )
        fake.PersistentClient.assert_called_once_with(
            path="/var/data/chromadb", tenant="c", database="d"
        )
        assert client is fake.PersistentClient.return_value

    def test_ephemeral_client_without_either(self, fake: MagicMock) -> None:
        client = chroma.create_chromadb_client(ChromaSettings(tenant="t"))
        fake.EphemeralClient.assert_called_once_with(
            tenant="t", database="default_database"
        )
        assert client is fake.EphemeralClient.return_value

    def test_connection_string_wins_over_a_directory(self, fake: MagicMock) -> None:
        chroma.create_chromadb_client(
            ChromaSettings(
                connection_string="http://localhost:8000", persist_directory="/d"
            )
        )
        fake.HttpClient.assert_called_once()
        fake.PersistentClient.assert_not_called()


class TestRegistry:
    def test_chromadb_is_native(self) -> None:
        entry = PROVIDERS["chromadb"]
        assert entry.module == "holodeck.lib.retrieval.adapters.chroma"
        assert entry.class_name == "ChromaAdapter"
        assert entry.extra == "chromadb"

    def test_create_adapter_builds_it(self, directory: str) -> None:
        built = create_adapter("chromadb", _schema(), {"persist_directory": directory})
        assert isinstance(built, ChromaAdapter)

    def test_a_missing_client_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "holodeck.lib.retrieval.adapters.chroma", None)
        with pytest.raises(MissingClientError) as raised:
            create_adapter("chromadb", _schema(), {})
        assert raised.value.extra == "chromadb"
        assert "holodeck-ai[chromadb]" in str(raised.value)

    def test_capabilities(self) -> None:
        adapter = ChromaAdapter.create(_schema(), {})
        assert set(adapter.capabilities.supported) == {
            Operation.UPSERT,
            Operation.GET,
            Operation.DELETE,
            Operation.FIND,
            Operation.LIST,
            Operation.DELETE_WHERE,
            Operation.COUNT,
            Operation.VECTOR_SEARCH,
        }


# -- filters and rows -----------------------------------------------------------


class TestWhere:
    @pytest.fixture
    def built(self) -> ChromaAdapter:
        return ChromaAdapter.create(_schema(), {})

    def test_no_filter_is_no_where(self, built: ChromaAdapter) -> None:
        assert built._where(None) is None
        assert built._where(RecordFilter()) is None

    def test_one_clause_is_a_plain_equality(self, built: ChromaAdapter) -> None:
        assert built._where(RecordFilter.where(source_path="a.md")) == {
            "source_path": "a.md"
        }

    def test_several_clauses_use_and(self, built: ChromaAdapter) -> None:
        where = built._where(
            RecordFilter.where(source_path="a.md", chunk_index=1, pinned=True)
        )
        assert where == {
            "$and": [{"source_path": "a.md"}, {"chunk_index": 1}, {"pinned": True}]
        }

    def test_unknown_field_is_a_schema_error(self, built: ChromaAdapter) -> None:
        with pytest.raises(SchemaError):
            built._where(RecordFilter.where(nope="x"))

    def test_none_cannot_be_filtered(self, built: ChromaAdapter) -> None:
        with pytest.raises(SchemaError, match="None"):
            built._where(RecordFilter.where(source_path=None))


class TestBatch:
    def test_none_values_become_the_type_default(self) -> None:
        built = ChromaAdapter.create(_schema(), {})
        batch = built._batch([_record("a", [1, 0, 0, 0], content=None, mtime=None)])
        assert batch.metadatas == [
            {
                "source_path": "a.md",
                "chunk_index": 0,
                "content": "",
                "mtime": 0.0,
                "pinned": False,
            }
        ]

    def test_values_are_stored_with_the_schema_type(self) -> None:
        built = ChromaAdapter.create(_schema(), {})
        batch = built._batch([_record("a", [1, 0, 0, 0], mtime=2, chunk_index=True)])
        assert batch.metadatas is not None
        assert isinstance(batch.metadatas[0]["mtime"], float)
        assert type(batch.metadatas[0]["chunk_index"]) is int

    def test_duplicate_ids_keep_the_last_record(self) -> None:
        built = ChromaAdapter.create(_schema(), {})
        batch = built._batch(
            [
                _record("a", [1, 0, 0, 0], content="old"),
                _record("b", [0, 1, 0, 0]),
                _record("a", [0, 0, 1, 0], content="new"),
            ]
        )
        assert batch.ids == ["a", "b"]
        assert batch.vectors[0] == [0.0, 0.0, 1.0, 0.0]
        assert batch.metadatas is not None
        assert batch.metadatas[0]["content"] == "new"

    def test_a_schema_without_fields_has_no_metadata(self) -> None:
        built = ChromaAdapter.create(CollectionSchema("bare", 2, ()), {})
        batch = built._batch([Record("a", {}, [1.0, 0.0])])
        assert batch.metadatas is None


# -- behavior against an embedded database ---------------------------------------


@pytest.mark.asyncio
async def test_records_round_trip_through_an_embedded_database(
    adapter: ChromaAdapter,
) -> None:
    await adapter.upsert(
        [
            _record("a", [1, 0, 0, 0], mtime=2.25, pinned=True, chunk_index=3),
            _record("b", [0, 1, 0, 0], source_path="b.md"),
        ]
    )
    records = await adapter.get(["b", "missing", "a"])
    assert [r.id for r in records] == ["b", "a"]
    assert dict(records[1].fields) == {
        "source_path": "a.md",
        "chunk_index": 3,
        "content": "text",
        "mtime": 2.25,
        "pinned": True,
    }
    assert await adapter.count() == 2


@pytest.mark.asyncio
async def test_get_keeps_request_order_for_repeated_ids(adapter: ChromaAdapter) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0]), _record("b", [0, 1, 0, 0])])
    records = await adapter.get(["b", "a", "b"])
    assert [r.id for r in records] == ["b", "a", "b"]


@pytest.mark.asyncio
async def test_vectors_come_back_only_on_request(adapter: ChromaAdapter) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    (plain,) = await adapter.get(["a"])
    (with_vector,) = await adapter.get(["a"], include_vectors=True)
    assert plain.vector is None
    assert with_vector.vector == pytest.approx([1.0, 0.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_reads_on_a_missing_collection_are_empty(adapter: ChromaAdapter) -> None:
    assert await adapter.collection_exists() is False
    assert await adapter.count() == 0
    assert await adapter.get(["a"]) == []
    assert await adapter.find(RecordFilter()) == []
    assert (await adapter.list_page()).records == ()
    assert await adapter.delete_where(RecordFilter()) == 0
    assert await adapter.vector_search(VectorQuery([1, 0, 0, 0], top_k=3)) == []
    await adapter.delete(["a"])
    assert await adapter.collection_exists() is False


@pytest.mark.asyncio
async def test_upsert_creates_a_missing_collection(adapter: ChromaAdapter) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    assert await adapter.collection_exists() is True


@pytest.mark.asyncio
async def test_vector_search_returns_what_exists_when_top_k_is_larger(
    adapter: ChromaAdapter,
) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0]), _record("b", [0, 1, 0, 0])])
    hits = await adapter.vector_search(VectorQuery([1, 0, 0, 0], top_k=50))
    assert [h.record.id for h in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)
    assert await adapter.vector_search(VectorQuery([1, 0, 0, 0], top_k=0)) == []


@pytest.mark.asyncio
async def test_delete_where_with_no_clause_removes_everything(
    adapter: ChromaAdapter,
) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0]), _record("b", [0, 1, 0, 0])])
    assert await adapter.delete_where(RecordFilter()) == 2
    assert await adapter.count() == 0


@pytest.mark.asyncio
async def test_pages_follow_offsets(adapter: ChromaAdapter) -> None:
    await adapter.upsert([_record(f"r{i}", [1, 0, 0, 0]) for i in range(5)])
    first = await adapter.list_page(limit=2)
    assert first.next_cursor == "2"
    last = await adapter.list_page(cursor="4", limit=2)
    assert len(last.records) == 1
    assert last.next_cursor is None
    with pytest.raises(SchemaError):
        await adapter.list_page(cursor="x")
    with pytest.raises(SchemaError):
        await adapter.list_page(cursor="-1")
    with pytest.raises(SchemaError):
        await adapter.list_page(limit=0)


@pytest.mark.asyncio
async def test_large_writes_are_split_into_batches(
    adapter: ChromaAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chroma, "_WRITE_BATCH", 3)
    await adapter.upsert([_record(f"r{i}", [1, 0, 0, 0]) for i in range(8)])
    assert await adapter.count() == 8
    assert await adapter.delete_where(RecordFilter()) == 8


@pytest.mark.asyncio
async def test_the_adapter_works_again_after_close(adapter: ChromaAdapter) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    await adapter.close()
    await adapter.close()
    assert await adapter.count() == 1
    await adapter.upsert([_record("b", [0, 1, 0, 0])])
    assert await adapter.count() == 2


@pytest.mark.asyncio
async def test_close_before_any_call_does_nothing(adapter: ChromaAdapter) -> None:
    await adapter.close()


@pytest.mark.asyncio
async def test_a_collection_recreated_by_someone_else_is_found_again(
    directory: str,
) -> None:
    first = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    second = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    try:
        await first.upsert([_record("a", [1, 0, 0, 0])])
        assert await first.count() == 1
        await second.delete_collection()
        await second.upsert([_record("b", [0, 1, 0, 0]), _record("c", [0, 0, 1, 0])])
        assert await first.count() == 2
        assert [r.id for r in await first.get(["b"])] == ["b"]
    finally:
        await first.close()
        await second.close()


# -- compatibility with collections other code created ------------------------------


@pytest.mark.asyncio
async def test_a_collection_with_the_legacy_space_metadata_is_read(
    directory: str,
) -> None:
    client = chromadb.PersistentClient(path=directory)
    legacy = client.create_collection(
        "chroma_unit", metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    legacy.add(
        ids=["a"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        metadatas=[
            {
                "source_path": "a.md",
                "chunk_index": 0,
                "content": "x",
                "mtime": 1.0,
                "pinned": False,
            }
        ],
    )
    client.close()
    built = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    try:
        (hit,) = await built.vector_search(VectorQuery([1, 0, 0, 0], top_k=1))
        assert hit.record.id == "a"
        assert hit.score == pytest.approx(1.0, abs=1e-5)
    finally:
        await built.close()


@pytest.mark.asyncio
async def test_a_collection_with_another_space_is_refused(directory: str) -> None:
    client = chromadb.PersistentClient(path=directory)
    client.create_collection("chroma_unit", embedding_function=None)  # l2 by default
    client.close()
    built = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    try:
        with pytest.raises(IncompatibleCollectionError, match="l2"):
            await built.ensure_collection()
        assert await built.collection_exists() is True
        await built.delete_collection()
        assert await built.collection_exists() is False
    finally:
        await built.close()


@pytest.mark.asyncio
async def test_a_collection_of_another_size_is_refused(directory: str) -> None:
    other = ChromaAdapter.create(
        CollectionSchema("chroma_unit", 2, ()), {"persist_directory": directory}
    )
    built = ChromaAdapter.create(_schema(), {"persist_directory": directory})
    try:
        await other.upsert([Record("a", {}, [1.0, 0.0])])
        with pytest.raises(IncompatibleCollectionError):
            await built.upsert([_record("b", [1, 0, 0, 0])])
    finally:
        await other.close()
        await built.close()


@pytest.mark.asyncio
async def test_an_unreachable_server_is_a_connection_error() -> None:
    built = ChromaAdapter.create(_schema(), {"connection_string": "http://127.0.0.1:9"})
    try:
        with pytest.raises(StoreConnectionError, match="chromadb"):
            await built.collection_exists()
        with pytest.raises(StoreConnectionError):
            await built.upsert([_record("a", [1, 0, 0, 0])])
    finally:
        await built.close()


# -- event loop ---------------------------------------------------------------------


def _slow(name: str, seconds: float, hook: Callable[[], None] | None = None) -> None:
    """Make ``Collection.<name>`` sleep, as a slow embedded call would."""
    original = getattr(Collection, name)

    def slow(self: Collection, *args: object, **kwargs: object) -> object:
        if hook is not None:
            hook()
        time.sleep(seconds)
        return original(self, *args, **kwargs)

    setattr(Collection, name, slow)


@pytest.mark.asyncio
async def test_a_slow_embedded_call_does_not_block_the_event_loop(
    adapter: ChromaAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    monkeypatch.setattr(Collection, "count", Collection.count)  # restored on exit
    _slow("count", 0.3)

    ticks = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    started = time.monotonic()
    count = await adapter.count()
    elapsed = time.monotonic() - started
    done.set()
    await task

    assert count == 1
    assert elapsed >= 0.3
    assert ticks >= 10


@pytest.mark.asyncio
async def test_calls_on_one_adapter_never_overlap_on_the_client(
    adapter: ChromaAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    monkeypatch.setattr(Collection, "count", Collection.count)
    state = {"running": 0, "peak": 0}
    guard = threading.Lock()

    def enter() -> None:
        with guard:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])

    original = Collection.count

    def slow_count(self: Collection, *args: object, **kwargs: object) -> int:
        enter()
        try:
            time.sleep(0.05)
            return original(self)
        finally:
            with guard:
                state["running"] -= 1

    monkeypatch.setattr(Collection, "count", slow_count)

    counts = await asyncio.gather(*(adapter.count() for _ in range(6)))

    assert counts == [1] * 6
    assert state["peak"] == 1


@pytest.mark.asyncio
async def test_a_cancelled_call_does_not_free_the_client_for_another(
    adapter: ChromaAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    await adapter.upsert([_record("a", [1, 0, 0, 0])])
    original = Collection.count
    state = {"running": 0, "peak": 0}
    guard = threading.Lock()

    def slow_count(self: Collection, *args: object, **kwargs: object) -> int:
        with guard:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        try:
            time.sleep(0.2)
            return original(self)
        finally:
            with guard:
                state["running"] -= 1

    monkeypatch.setattr(Collection, "count", slow_count)

    first = asyncio.create_task(adapter.count())
    await asyncio.sleep(0.05)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await adapter.count() == 1
    assert state["peak"] == 1


@pytest.mark.asyncio
async def test_writes_never_use_chroma_upsert_or_update(
    adapter: ChromaAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chroma 1.5.9 loses rows from the vector index on upsert and update."""

    def forbidden(self: Collection, *args: object, **kwargs: object) -> None:
        raise AssertionError("upsert and update are not safe on Chroma 1.5.9")

    monkeypatch.setattr(Collection, "upsert", forbidden)
    monkeypatch.setattr(Collection, "update", forbidden)

    await adapter.upsert([_record("a", [1, 0, 0, 0], content="old")])
    await adapter.upsert([_record("a", [0, 1, 0, 0], content="new")])

    (record,) = await adapter.get(["a"], include_vectors=True)
    assert record.fields["content"] == "new"
    assert record.vector == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert await adapter.count() == 1
