"""Unit tests for the native Weaviate adapter, on a fake asynchronous client.

The fake returns the client's own result types, so a renamed field in the client
shows up here. ``tests/contract/retrieval`` runs the same adapter against a
Weaviate server.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest
from weaviate import WeaviateAsyncClient
from weaviate.classes.config import DataType, Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.collections.classes.batch import (
    BatchObject,
    BatchObjectReturn,
    ErrorObject,
)
from weaviate.collections.classes.internal import MetadataReturn, Object
from weaviate.collections.classes.types import WeaviateField
from weaviate.exceptions import WeaviateConnectionError

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    LexicalQuery,
    Operation,
    Record,
    RecordFilter,
    RetrievalError,
    SchemaError,
    ScoreKind,
    StoreConnectionError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import weaviate as module
from holodeck.lib.retrieval.adapters.weaviate import (
    ID_PROPERTY,
    WeaviateAdapter,
    WeaviateSettings,
    property_for,
    settings_from_connection,
    weaviate_class_name,
)

API_KEY = "wcd-SECRET-key-123"
LOCAL = {"local_host": "127.0.0.1", "local_port": 8089, "local_grpc_port": 50059}
DIMENSIONS = 4


def schema(name: str = "contract_docs") -> CollectionSchema:
    return CollectionSchema(
        name,
        DIMENSIONS,
        (
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT, indexed=True),
            FieldSpec("pinned", FieldType.BOOLEAN, indexed=True),
            FieldSpec("note", FieldType.STRING),
        ),
    )


def record(
    record_id: str, *, source: str = "a.md", index: int = 0, content: str = "text"
) -> Record:
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": index,
            "content": content,
            "mtime": 1.5,
            "pinned": False,
        },
        vector=[1.0, 0.0, 0.0, 0.0],
    )


# -- fakes ---------------------------------------------------------------------------


def stored_object(
    key: uuid.UUID,
    properties: Mapping[str, WeaviateField],
    *,
    distance: float | None = None,
    score: float | None = None,
    vector: list[float] | None = None,
) -> Object[Mapping[str, WeaviateField], None]:
    return Object(
        uuid=key,
        metadata=MetadataReturn(distance=distance, score=score),
        properties=properties,
        references=None,
        vector={} if vector is None else {"embedding": cast(list[float], vector)},
        collection="Contract_docs",
    )


def props(
    source: str = "a.md", index: int = 0, content: str = "text"
) -> dict[str, WeaviateField]:
    return {
        "source_path": source,
        "chunk_index": index,
        "content": content,
        "mtime": 1.5,
        "pinned": False,
    }


@dataclass
class Response:
    objects: list[Object[Mapping[str, WeaviateField], None]]


@dataclass
class Aggregate:
    total_count: int | None


class Delete:
    def __init__(self, failed: int = 0) -> None:
        self.failed = failed
        self.matches = 0
        self.successful = 0


@dataclass
class FakeCollection:
    """One collection that logs its calls and serves scripted responses."""

    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    insert_errors: dict[int, ErrorObject] = field(default_factory=dict)
    delete_failed: int = 0
    total: int | None = 0

    def __post_init__(self) -> None:
        self.data = self
        self.query = self
        self.aggregate = self

    def names(self, name: str) -> list[dict[str, object]]:
        return [args for called, args in self.calls if called == name]

    def _next(self) -> Response:
        return self.responses.pop(0) if self.responses else Response([])

    async def insert_many(self, objects: Sequence[DataObject]) -> BatchObjectReturn:
        self.calls.append(("insert_many", {"objects": list(objects)}))
        return BatchObjectReturn(errors=self.insert_errors)

    async def delete_many(self, *, where: object) -> Delete:
        self.calls.append(("delete_many", {"where": where}))
        return Delete(self.delete_failed)

    async def fetch_objects(self, **arguments: object) -> Response:
        self.calls.append(("fetch_objects", arguments))
        return self._next()

    async def near_vector(self, **arguments: object) -> Response:
        self.calls.append(("near_vector", arguments))
        return self._next()

    async def bm25(self, **arguments: object) -> Response:
        self.calls.append(("bm25", arguments))
        return self._next()

    async def hybrid(self, **arguments: object) -> Response:
        self.calls.append(("hybrid", arguments))
        return self._next()

    async def over_all(self, **arguments: object) -> Aggregate:
        self.calls.append(("over_all", arguments))
        return Aggregate(self.total)


@dataclass
class FakeClient:
    """A ``WeaviateAsyncClient`` stand-in."""

    collection: FakeCollection = field(default_factory=FakeCollection)
    exists: bool = True
    connect_failure: Exception | None = None
    connected: int = 0
    closed: int = 0
    created: list[dict[str, object]] = field(default_factory=list)
    collections: _Collections = field(init=False)

    def __post_init__(self) -> None:
        self.collections = _Collections(self)

    async def connect(self) -> None:
        self.connected += 1
        if self.connect_failure is not None:
            raise self.connect_failure

    async def close(self) -> None:
        self.closed += 1

    def get(self, name: str) -> FakeCollection:
        self.collection.calls.append(("get", {"name": name}))
        return self.collection

    async def create(self, **arguments: object) -> None:
        self.created.append(arguments)
        self.exists = True

    async def delete(self, name: str) -> None:
        self.collection.calls.append(("delete_collection", {"name": name}))
        self.exists = False


class _Collections:
    """``client.collections`` with ``exists`` as a method, as the real one has."""

    def __init__(self, client: FakeClient) -> None:
        self._client = client

    async def exists(self, name: str) -> bool:
        return self._client.exists

    def get(self, name: str) -> FakeCollection:
        return self._client.get(name)

    async def create(self, **arguments: object) -> None:
        await self._client.create(**arguments)

    async def delete(self, name: str) -> None:
        await self._client.delete(name)


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeClient]]:
    """Every client the adapter builds, in order."""
    built: list[FakeClient] = []

    def factory(settings: WeaviateSettings) -> WeaviateAsyncClient:
        client = FakeClient()
        built.append(client)
        return cast(WeaviateAsyncClient, client)

    monkeypatch.setattr(module, "create_weaviate_client", factory)
    yield built


def adapter(**settings: object) -> WeaviateAdapter:
    return WeaviateAdapter.create(schema(), {**LOCAL, **settings})


# -- connection settings ---------------------------------------------------------------


def test_local_settings_and_defaults() -> None:
    assert settings_from_connection(LOCAL) == WeaviateSettings(
        host="127.0.0.1", port=8089, grpc_port=50059
    )
    assert settings_from_connection({"local_host": "weaviate"}) == WeaviateSettings(
        host="weaviate"
    )
    assert settings_from_connection({"local_host": "h", "local_port": "9000"}) == (
        WeaviateSettings(host="h", port=9000)
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://abc.weaviate.cloud",
        "abc.weaviate.cloud",
        "https://abc.weaviate.cloud:443/v1",
    ],
)
def test_cloud_url_forms(url: str) -> None:
    settings = settings_from_connection({"url": url, "api_key": API_KEY})
    assert settings.cluster == "abc.weaviate.cloud"
    assert settings.host is None
    assert API_KEY not in repr(settings)


def test_connection_string_forms() -> None:
    cloud = settings_from_connection(
        {"connection_string": "https://abc.weaviate.cloud", "api_key": API_KEY}
    )
    assert cloud.cluster == "abc.weaviate.cloud"
    local = settings_from_connection(
        {"connection_string": "http://weaviate:9090", "local_grpc_port": 50052}
    )
    assert (local.host, local.port, local.grpc_port) == ("weaviate", 9090, 50052)
    # Explicit keys win over the connection string.
    both = settings_from_connection(
        {"connection_string": "http://weaviate:9090", "local_port": 9091}
    )
    assert both.port == 9091


def test_environment_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVIATE_URL", "https://env.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", API_KEY)
    settings = settings_from_connection({})
    assert (settings.cluster, settings.api_key) == ("env.weaviate.cloud", API_KEY)
    # An explicit target ignores the environment.
    assert settings_from_connection({"local_host": "h"}).cluster is None
    monkeypatch.delenv("WEAVIATE_URL")
    monkeypatch.setenv("WEAVIATE_LOCAL_HOST", "envhost")
    monkeypatch.setenv("WEAVIATE_LOCAL_PORT", "9999")
    local = settings_from_connection({})
    assert (local.host, local.port) == ("envhost", 9999)


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        ({}, "No Weaviate server"),
        ({"bogus": 1, "local_host": "h"}, "Unknown connection settings"),
        ({"url": "https://x.weaviate.cloud"}, "No API key"),
        ({"url": "http://x", "api_key": "k"}, "https URL"),
        ({"url": "https://x:8443", "api_key": "k"}, "port 443"),
        ({"url": "https://x", "api_key": "k", "local_host": "h"}, "not both"),
        ({"url": "https://x", "api_key": "k", "local_port": 80}, "apply to local_host"),
        ({"local_host": "h", "local_port": 0}, "port from 1 to 65535"),
        ({"local_host": "h", "local_port": True}, "port from 1 to 65535"),
        ({"local_host": "h", "local_port": "abc"}, "port from 1 to 65535"),
        ({"local_host": "h", "local_port": 70000}, "port from 1 to 65535"),
        ({"local_host": "h", "local_port": 5, "local_grpc_port": 5}, "must differ"),
        ({"local_host": 5}, "must be a string"),
        ({"connection_string": "ftp://h"}, "connection_string must be"),
    ],
)
def test_invalid_connection_settings(
    connection: dict[str, object], message: str
) -> None:
    with pytest.raises(StoreConnectionError, match=message):
        WeaviateAdapter.create(schema(), connection)


def test_a_wrong_key_does_not_leak_the_api_key_in_the_error() -> None:
    with pytest.raises(StoreConnectionError) as raised:
        WeaviateAdapter.create(schema(), {"url": "http://x", "api_key": API_KEY})
    assert API_KEY not in str(raised.value)


def test_registry_builds_the_native_adapter() -> None:
    built = create_adapter("weaviate", schema(), LOCAL)
    assert isinstance(built, WeaviateAdapter)
    assert built.provider == "weaviate"


def test_every_operation_is_supported() -> None:
    built = adapter()
    for operation in Operation:
        assert built.capabilities.supports(operation)


# -- names, properties -----------------------------------------------------------------


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Documents_dim1536", "Documents_dim1536"),
        ("Already", "Already"),
        ("A", "A"),
        ("X" * 255, "X" * 255),
        ("documents_dim1536", "Documents_dim1536_{}"),
        ("hierarchical_docs_my-tool_dim768", "Hierarchical_docs_my_tool_dim768_{}"),
        ("a.b c", "A_b_c_{}"),
        ("1docs", "Collection_1docs_{}"),
        ("_x", "Collection__x_{}"),
        ("é", "Collection___{}"),
    ],
)
def test_class_names(name: str, expected: str) -> None:
    """A valid name is unchanged; any other change appends a hash of the original."""
    assert weaviate_class_name(name) == expected.format(_digest(name))


def test_names_that_clean_to_the_same_text_get_different_collections() -> None:
    names = ["docs", "Docs", "a-b", "a_b", "a.b", "A-b", "a b"]
    classes = {weaviate_class_name(name) for name in names}
    assert len(classes) == len(names)
    # Weaviate upper-cases the first letter itself, so only the hash separates
    # "docs" from "Docs".
    assert weaviate_class_name("Docs") == "Docs"
    assert weaviate_class_name("docs") == f"Docs_{_digest('docs')}"


def test_a_long_class_name_keeps_a_hash() -> None:
    long = weaviate_class_name("x" * 400)
    assert len(long) == 255
    assert long.endswith(f"_{_digest('x' * 400)}")
    assert long != weaviate_class_name("x" * 401)
    assert len(weaviate_class_name("X" * 256)) == 255


def test_property_definitions() -> None:
    exact = property_for(FieldSpec("source_path", FieldType.STRING, indexed=True))
    assert exact.dataType == DataType.TEXT
    assert exact.tokenization == Tokenization.FIELD
    assert exact.indexFilterable is True
    assert exact.indexSearchable is False
    plain = property_for(FieldSpec("note", FieldType.STRING))
    assert (plain.indexFilterable, plain.indexSearchable) == (False, False)
    text = property_for(FieldSpec("content", FieldType.STRING, full_text=True))
    assert (text.indexSearchable, text.indexFilterable) == (True, False)
    assert text.tokenization is None
    # full_text wins when a field is also indexed.
    both = property_for(FieldSpec("b", FieldType.STRING, indexed=True, full_text=True))
    assert (both.indexSearchable, both.indexFilterable) == (True, False)
    unindexed = property_for(FieldSpec("n", FieldType.INTEGER))
    assert unindexed.indexFilterable is False
    for kind, data_type in [
        (FieldType.INTEGER, DataType.INT),
        (FieldType.FLOAT, DataType.NUMBER),
        (FieldType.BOOLEAN, DataType.BOOL),
    ]:
        number = property_for(FieldSpec("n", kind, indexed=True))
        assert number.dataType == data_type
        assert number.indexFilterable is True
        assert number.tokenization is None


@pytest.mark.parametrize(
    "name", ["vector", ID_PROPERTY, "has-dash", "1x", "a b", "_id"]
)
def test_invalid_property_names_are_rejected(name: str) -> None:
    bad = CollectionSchema("c", DIMENSIONS, (FieldSpec(name, FieldType.STRING),))
    with pytest.raises(SchemaError):
        WeaviateAdapter.create(bad, LOCAL)


# -- filters ---------------------------------------------------------------------------

HOSTILE = [
    "plain.md",
    'it\'s "q" \\ x',
    "",
    "*",
    "a.md OR 1=1",
    'x"}) { Delete }',
    "é中\n\t ",
    "a" * 100_000,
]


@pytest.mark.parametrize("value", HOSTILE)
def test_filter_values_are_passed_verbatim(value: str) -> None:
    built = adapter().build_filter(RecordFilter.where(source_path=value))
    assert built == Filter.by_property("source_path").equal(value)


def test_multiple_clauses_are_a_conjunction_with_typed_values() -> None:
    built = adapter().build_filter(
        RecordFilter.where(source_path="a.md", chunk_index=2, mtime=2, pinned=True)
    )
    assert built is not None
    clauses = vars(built)["filters"]
    assert clauses == [
        Filter.by_property("source_path").equal("a.md"),
        Filter.by_property("chunk_index").equal(2),
        Filter.by_property("mtime").equal(2.0),
        Filter.by_property("pinned").equal(True),
    ]
    assert isinstance(vars(clauses[2])["value"], float)


def test_no_clauses_means_no_filter() -> None:
    assert adapter().build_filter(None) is None
    assert adapter().build_filter(RecordFilter()) is None


@pytest.mark.parametrize(
    ("clause", "message"),
    [
        (RecordFilter.where(nope="x"), "no field"),
        (RecordFilter.where(content="alpha"), "full-text"),
        (RecordFilter.where(source_path=None), "never null"),
        (RecordFilter.where(source_path=3), "has another type"),
        (RecordFilter.where(chunk_index="1"), "has another type"),
        (RecordFilter.where(chunk_index=True), "boolean is not a number"),
        (RecordFilter.where(chunk_index=1.5), "has another type"),
        (RecordFilter.where(mtime="1.5"), "has another type"),
        (RecordFilter.where(pinned=1), "has another type"),
        (RecordFilter.where(note="x"), "not indexed"),
    ],
)
def test_filters_that_cannot_match_exactly_are_errors(
    clause: RecordFilter, message: str
) -> None:
    with pytest.raises(SchemaError, match=message):
        adapter().build_filter(clause)


# -- ID mapping and conversion ---------------------------------------------------------


def test_store_ids() -> None:
    built = adapter()
    canonical = "8f14e45f-ceea-467f-a0e6-0c0b6c0b6c0b"
    assert built.store_id(canonical) == uuid.UUID(canonical)
    assert built.store_id(canonical.upper()) == uuid.UUID(canonical)
    other = built.store_id("/srv/x/doc.md_chunk_3")
    assert other == built.store_id("/srv/x/doc.md_chunk_3")
    assert other.version == 5
    assert other != built.store_id("/srv/x/doc.md_chunk_4")
    # The namespace is the collection: the same ID maps differently elsewhere.
    elsewhere = WeaviateAdapter.create(schema("other"), LOCAL)
    assert elsewhere.store_id("/srv/x/doc.md_chunk_3") != other


@pytest.mark.asyncio
async def test_the_original_id_is_stored_only_when_it_is_not_the_uuid(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    canonical = "8f14e45f-ceea-467f-a0e6-0c0b6c0b6c0b"
    await built.upsert(
        [
            record(canonical),
            record(canonical.upper(), index=1),
            record("free text id", index=2),
        ]
    )
    (call,) = clients[0].collection.names("insert_many")
    objects = cast(list[DataObject], call["objects"])
    # The two spellings are one object; the last one wins.
    assert [str(o.uuid) for o in objects][0] == canonical
    assert len(objects) == 2
    assert objects[0].properties[ID_PROPERTY] == canonical.upper()
    assert objects[1].properties[ID_PROPERTY] == "free text id"
    assert objects[1].properties["chunk_index"] == 2
    assert objects[1].vector == {"embedding": [1.0, 0.0, 0.0, 0.0]}


@pytest.mark.asyncio
async def test_missing_values_are_stored_as_type_defaults(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    key = "8f14e45f-ceea-467f-a0e6-0c0b6c0b6c0b"
    await built.upsert([Record(key, {"source_path": "a"}, [1.0, 0.0, 0.0, 0.0])])
    (call,) = clients[0].collection.names("insert_many")
    (obj,) = cast(list[DataObject], call["objects"])
    assert obj.properties == {
        "source_path": "a",
        "chunk_index": 0,
        "content": "",
        "mtime": 0.0,
        "pinned": False,
        "note": "",
    }


@pytest.mark.asyncio
async def test_a_value_of_the_wrong_type_is_a_schema_error(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    bad = Record("x", {"chunk_index": "abc"}, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(SchemaError, match="chunk_index"):
        await built.upsert([bad])
    assert clients == []  # rejected before any connection


@pytest.mark.asyncio
async def test_upsert_batches_and_surfaces_rejected_records(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.upsert([record(f"id{n}") for n in range(1200)])
    sizes = [
        len(cast(list[object], c["objects"]))
        for c in clients[0].collection.names("insert_many")
    ]
    assert sizes == [500, 500, 200]

    rejected = ErrorObject(message="bad vector", object_=cast(BatchObject, None))
    clients[0].collection.insert_errors = {0: rejected}
    with pytest.raises(RetrievalError, match="bad vector"):
        await built.upsert([record("x")])


@pytest.mark.asyncio
async def test_get_keeps_request_order_and_skips_missing(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    first, second = built.store_id("one"), built.store_id("two")
    served = [
        stored_object(second, {**props(), ID_PROPERTY: "two"}),
        stored_object(first, {**props(index=1), ID_PROPERTY: "one"}),
    ]
    await built.collection_exists()  # builds the client
    clients[0].collection.responses = [Response(served)]
    got = await built.get(["one", "missing", "two", "one"])
    assert [r.id for r in got] == ["one", "two", "one"]
    assert got[0].fields["chunk_index"] == 1
    (call,) = clients[0].collection.names("fetch_objects")
    assert call["limit"] == 3  # one, missing, two
    assert call["include_vector"] is False


@pytest.mark.asyncio
async def test_get_returns_vectors_on_request(clients: list[FakeClient]) -> None:
    built = adapter()
    key = built.store_id("k")
    await built.collection_exists()
    clients[0].collection.responses = [
        Response(
            [stored_object(key, {**props(), ID_PROPERTY: "k"}, vector=[1.0, 2, 3, 4])]
        )
    ]
    (got,) = await built.get(["k"], include_vectors=True)
    assert got.vector == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_ids_without_the_original_id_property_read_as_the_uuid(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    key = uuid.uuid4()
    await built.collection_exists()
    clients[0].collection.responses = [Response([stored_object(key, props())])]
    (got,) = await built.get([str(key)])
    assert got.id == str(key)
    assert got.fields["chunk_index"] == 0
    assert isinstance(got.fields["chunk_index"], int)


# -- reads -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_checks_every_hit_against_the_filter(
    clients: list[FakeClient],
) -> None:
    """A collection with word tokenization returns supersets; find trims them."""
    built = adapter()
    await built.collection_exists()
    keys = [uuid.uuid4() for _ in range(3)]
    clients[0].collection.responses = [
        Response(
            [
                stored_object(keys[0], props(source="docs/a.md")),
                stored_object(keys[1], props(source="docs/a.md.bak")),
                stored_object(keys[2], props(source="a docs")),
            ]
        )
    ]
    found = await built.find(RecordFilter.where(source_path="docs/a.md"))
    assert [r.id for r in found] == [str(keys[0])]


@pytest.mark.asyncio
async def test_find_pages_until_a_short_page_and_honours_limit(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    page = [stored_object(uuid.uuid4(), props()) for _ in range(512)]
    clients[0].collection.responses = [Response(page), Response(page[:3])]
    found = await built.find(RecordFilter.where(source_path="a.md"))
    assert len(found) == 515
    calls = clients[0].collection.names("fetch_objects")
    assert [(c["offset"], c["limit"]) for c in calls] == [(0, 512), (512, 512)]

    clients[0].collection.responses = [Response(page)]
    limited = await built.find(RecordFilter.where(source_path="a.md"), limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_find_switches_to_cursor_enumeration_at_the_query_ceiling(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "QUERY_MAXIMUM_RESULTS", 1024)
    built = adapter()
    await built.collection_exists()
    page = [stored_object(uuid.uuid4(), props()) for _ in range(512)]
    other = [stored_object(uuid.uuid4(), props(source="b.md")) for _ in range(512)]
    tail = [stored_object(uuid.uuid4(), props()) for _ in range(3)]
    clients[0].collection.responses = [
        Response(page),
        Response(page),
        # Past the ceiling: the scan starts over and reads the whole collection.
        Response(other),
        Response(page),
        Response(tail),
    ]
    found = await built.find(RecordFilter.where(source_path="a.md"))
    assert len(found) == 512 + 3
    calls = clients[0].collection.names("fetch_objects")
    assert [(c.get("offset"), c.get("after")) for c in calls] == [
        (0, None),
        (512, None),
        (None, None),
        (None, other[-1].uuid),
        (None, page[-1].uuid),
    ]
    assert all(c.get("filters") is None for c in calls[2:])


@pytest.mark.asyncio
async def test_list_page_cursor_is_the_last_uuid(clients: list[FakeClient]) -> None:
    built = adapter()
    await built.collection_exists()
    keys = [uuid.uuid4() for _ in range(3)]
    clients[0].collection.responses = [
        Response([stored_object(k, props()) for k in keys])
    ]
    page = await built.list_page(limit=2)
    assert [r.id for r in page.records] == [str(keys[0]), str(keys[1])]
    assert page.next_cursor == str(keys[1])
    assert all(r.vector is None for r in page.records)

    clients[0].collection.responses = [Response([stored_object(keys[2], props())])]
    last = await built.list_page(cursor=page.next_cursor, limit=2)
    assert last.next_cursor is None
    calls = clients[0].collection.names("fetch_objects")
    assert (calls[0]["after"], calls[0]["limit"]) == (None, 3)
    assert calls[1]["after"] == keys[1]


@pytest.mark.asyncio
async def test_list_page_rejects_a_bad_cursor_and_limit(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    with pytest.raises(SchemaError, match="Invalid cursor"):
        await built.list_page(cursor="not-a-uuid")
    with pytest.raises(SchemaError, match="positive"):
        await built.list_page(limit=0)
    assert clients == []


@pytest.mark.asyncio
async def test_count_is_zero_when_the_collection_is_absent(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    clients[0].exists = False
    assert await built.count() == 0
    clients[0].exists = True
    clients[0].collection.total = 7
    assert await built.count() == 7
    clients[0].collection.total = None
    assert await built.count() == 0


@pytest.mark.asyncio
async def test_delete_removes_by_uuid_and_reports_failures(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.delete(["a", "a", "b"])
    (call,) = clients[0].collection.names("delete_many")
    assert call["where"] == Filter.by_id().contains_any(
        [built.store_id("a"), built.store_id("b")]
    )
    clients[0].collection.delete_failed = 1
    with pytest.raises(RetrievalError, match="failed to delete 1"):
        await built.delete(["c"])
    await built.delete([])  # no call


# -- search ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_search_converts_distance_to_similarity(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    keys = [uuid.uuid4() for _ in range(3)]
    clients[0].collection.responses = [
        Response(
            [
                stored_object(keys[0], props(), distance=0.0),
                stored_object(keys[1], props(), distance=0.25),
                stored_object(keys[2], props(), distance=2.0),
            ]
        )
    ]
    hits = await built.vector_search(
        VectorQuery([1.0, 0.0, 0.0, 0.0], 3, RecordFilter.where(source_path="a.md"))
    )
    assert [h.score for h in hits] == [1.0, 0.75, 0.0]
    assert all(h.kind is ScoreKind.SIMILARITY for h in hits)
    (call,) = clients[0].collection.names("near_vector")
    assert call["target_vector"] == "embedding"
    assert call["filters"] == Filter.by_property("source_path").equal("a.md")


@pytest.mark.asyncio
async def test_a_hit_without_a_distance_is_an_error(clients: list[FakeClient]) -> None:
    built = adapter()
    await built.collection_exists()
    clients[0].collection.responses = [Response([stored_object(uuid.uuid4(), props())])]
    with pytest.raises(RetrievalError, match="without a distance"):
        await built.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 1))


@pytest.mark.asyncio
async def test_lexical_search_uses_bm25_on_the_text_field(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    clients[0].collection.responses = [
        Response([stored_object(uuid.uuid4(), props(), score=1.25)])
    ]
    (hit,) = await built.lexical_search(LexicalQuery("zeppelin", 3, "content"))
    assert (hit.score, hit.kind) == (1.25, ScoreKind.LEXICAL)
    (call,) = clients[0].collection.names("bm25")
    assert call["query_properties"] == ["content"]
    assert await built.lexical_search(LexicalQuery("   ", 3, "content")) == []
    assert await built.lexical_search(LexicalQuery("x", 0, "content")) == []
    assert len(clients[0].collection.names("bm25")) == 1


@pytest.mark.asyncio
async def test_text_search_needs_a_full_text_field(clients: list[FakeClient]) -> None:
    built = adapter()
    with pytest.raises(SchemaError, match="not a full-text field"):
        await built.lexical_search(LexicalQuery("x", 3, "source_path"))
    with pytest.raises(SchemaError, match="not a full-text field"):
        await built.hybrid_search(
            HybridQuery("x", [1.0, 0.0, 0.0, 0.0], 3, "chunk_index")
        )
    with pytest.raises(SchemaError, match="no field"):
        await built.lexical_search(LexicalQuery("x", 3, "nope"))
    assert clients == []


@pytest.mark.asyncio
async def test_hybrid_search_uses_the_native_hybrid_query(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    clients[0].collection.responses = [
        Response([stored_object(uuid.uuid4(), props(), score=0.8)])
    ]
    (hit,) = await built.hybrid_search(
        HybridQuery("zeppelin", [1.0, 0.0, 0.0, 0.0], 3, "content")
    )
    assert (hit.score, hit.kind) == (0.8, ScoreKind.FUSED)
    (call,) = clients[0].collection.names("hybrid")
    assert call["query"] == "zeppelin"
    assert call["query_properties"] == ["content"]
    assert call["alpha"] == 0.5
    assert call["target_vector"] == "embedding"
    assert call["limit"] == 3


@pytest.mark.asyncio
async def test_search_rejects_a_wrong_sized_vector(clients: list[FakeClient]) -> None:
    built = adapter()
    with pytest.raises(SchemaError, match="dimensions"):
        await built.vector_search(VectorQuery([1.0] * 5, 1))
    with pytest.raises(SchemaError, match="dimensions"):
        await built.hybrid_search(HybridQuery("x", [1.0] * 5, 1, "content"))
    assert clients == []


# -- collection ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_creates_the_sk_layout(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    await built.collection_exists()
    clients[0].exists = False
    await built.ensure_collection()
    (call,) = clients[0].created
    assert call["name"] == weaviate_class_name("contract_docs")
    properties = cast(list[object], call["properties"])
    names = [vars(p)["name"] for p in properties]
    assert names == [*schema().field_names, ID_PROPERTY]
    vector_config = vars(call["vector_config"])
    assert vector_config["name"] == "embedding"


# -- lifecycle and errors --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_client_is_lazy_shared_and_rebuilt_after_close(
    clients: list[FakeClient],
) -> None:
    built = adapter()
    assert clients == []
    await built.collection_exists()
    await built.count()
    await built.upsert([record("a")])
    assert len(clients) == 1
    assert clients[0].connected == 1

    await built.close()
    await built.close()
    assert clients[0].closed == 1

    await built.collection_exists()
    assert len(clients) == 2
    await built.close()
    assert clients[1].closed == 1


@pytest.mark.asyncio
async def test_a_close_that_never_connected_is_a_noop(
    clients: list[FakeClient],
) -> None:
    await adapter().close()
    assert clients == []


@pytest.mark.asyncio
async def test_connect_failures_become_store_connection_errors_without_the_key(
    clients: list[FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = WeaviateConnectionError(f"http://x?apikey={API_KEY} refused")

    def factory(settings: WeaviateSettings) -> WeaviateAsyncClient:
        client = FakeClient(connect_failure=failure)
        clients.append(client)
        return cast(WeaviateAsyncClient, client)

    monkeypatch.setattr(module, "create_weaviate_client", factory)
    built = WeaviateAdapter.create(
        schema(), {"url": "https://x.weaviate.cloud", "api_key": API_KEY}
    )
    with pytest.raises(StoreConnectionError) as raised:
        await built.collection_exists()
    assert API_KEY not in str(raised.value)
    assert raised.value.__cause__ is None
    assert clients[0].closed == 1  # the half-open client is released

    with pytest.raises(StoreConnectionError):
        await built.count()
    assert len(clients) == 2  # the next call tries again


@pytest.mark.asyncio
async def test_a_client_that_rejects_its_settings_is_a_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(settings: WeaviateSettings) -> WeaviateAsyncClient:
        raise ValueError("http.port and grpc.port must be different")

    monkeypatch.setattr(module, "create_weaviate_client", factory)
    with pytest.raises(StoreConnectionError, match="must be different"):
        await adapter().collection_exists()
