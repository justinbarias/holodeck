"""Unit tests for the native Qdrant adapter (042 T5).

Server-free: connection parsing, ID mapping, filters, index parameters,
request shapes against a mocked client, and lifecycle against the in-process
``:memory:`` store.
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

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
    StoreConnectionError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import qdrant
from holodeck.lib.retrieval.adapters.qdrant import (
    QdrantAdapter,
    index_params,
    parse_qdrant_connection_string,
    qdrant_connection_params,
    tokenize,
)
from holodeck.lib.retrieval.schemas import hierarchical_schema


def _schema(name: str = "docs") -> CollectionSchema:
    return CollectionSchema(
        name,
        4,
        (
            FieldSpec("source_path", indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def _memory(schema: CollectionSchema | None = None) -> QdrantAdapter:
    return QdrantAdapter.create(schema or _schema(), {"location": ":memory:"})


# -- connection string (copied from the vector_store tests) ----------------------


class TestParseQdrantConnectionString:
    def test_localhost_url(self) -> None:
        result = parse_qdrant_connection_string("http://localhost:6333")
        assert result == {"host": "localhost", "port": 6333}

    def test_https_remote_url(self) -> None:
        result = parse_qdrant_connection_string("https://qdrant.example.com:6333")
        assert result == {"url": "https://qdrant.example.com:6333"}

    def test_default_port(self) -> None:
        result = parse_qdrant_connection_string("http://localhost")
        assert result["port"] == 6333

    def test_in_memory(self) -> None:
        assert parse_qdrant_connection_string(":memory:") == {"location": ":memory:"}

    def test_local_path(self) -> None:
        assert parse_qdrant_connection_string("/var/data/qdrant") == {
            "path": "/var/data/qdrant"
        }

    def test_file_url(self) -> None:
        assert parse_qdrant_connection_string("file:///var/data/qdrant") == {
            "path": "/var/data/qdrant"
        }

    @pytest.mark.parametrize("scheme", ["qdrant+grpc", "grpc"])
    def test_grpc(self, scheme: str) -> None:
        result = parse_qdrant_connection_string(f"{scheme}://localhost:6334")
        assert result == {"host": "localhost", "grpc_port": 6334, "prefer_grpc": True}

    def test_api_key_in_userinfo(self) -> None:
        result = parse_qdrant_connection_string("https://my-api-key@qdrant.example.com")
        assert result == {"url": "https://qdrant.example.com", "api_key": "my-api-key"}

    def test_loopback_ip(self) -> None:
        result = parse_qdrant_connection_string("http://127.0.0.1:6333")
        assert result == {"host": "127.0.0.1", "port": 6333}

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_qdrant_connection_string("")

    @pytest.mark.parametrize("value", ["ftp://localhost:6333", "tcp://localhost:6333"])
    def test_invalid_scheme(self, value: str) -> None:
        with pytest.raises(ValueError, match="Invalid scheme"):
            parse_qdrant_connection_string(value)

    def test_custom_scheme(self) -> None:
        result = parse_qdrant_connection_string("qdrant://localhost:6333")
        assert result == {"host": "localhost", "port": 6333}

    def test_remote_without_port(self) -> None:
        result = parse_qdrant_connection_string("https://qdrant.example.com")
        assert result == {"url": "https://qdrant.example.com"}


class TestConnectionParams:
    def test_explicit_keys_override_the_connection_string(self) -> None:
        params = qdrant_connection_params(
            {
                "connection_string": "https://key@qdrant.example.com",
                "api_key": "explicit",
                "timeout": 30,
                "prefix": "service",
            }
        )
        assert params == {
            "url": "https://qdrant.example.com",
            "api_key": "explicit",
            "timeout": 30,
            "prefix": "service",
        }

    def test_collection_name_is_not_a_client_setting(self) -> None:
        params = qdrant_connection_params(
            {"url": "http://q:6333", "collection_name": "x"}
        )
        assert params == {"url": "http://q:6333"}

    def test_typed_keys(self) -> None:
        params = qdrant_connection_params(
            {
                "host": "q",
                "port": "6333",
                "grpc_port": 6334,
                "prefer_grpc": True,
                "https": False,
                "check_compatibility": False,
                "headers": {"X-Tenant": "a"},
                "api_key": None,
            }
        )
        assert params == {
            "host": "q",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": True,
            "https": False,
            "check_compatibility": False,
            "headers": {"X-Tenant": "a"},
        }

    @pytest.mark.parametrize(
        ("connection", "message"),
        [
            ({"bogus": 1}, "Unknown Qdrant settings"),
            ({"port": "abc"}, "must be an integer"),
            ({"port": True}, "must be an integer"),
            ({"prefer_grpc": "yes"}, "must be true or false"),
            ({"url": 5}, "must be a string"),
            ({"headers": {"a": 1}}, "header names"),
            ({"url": "http://a", "host": "b"}, "Only one of"),
            ({"connection_string": ":memory:", "path": "/x"}, "Only one of"),
            ({"connection_string": "ftp://x"}, "Invalid scheme"),
        ],
    )
    def test_invalid_settings(
        self, connection: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            qdrant_connection_params(connection)

    def test_create_reports_bad_settings_as_a_connection_error(self) -> None:
        with pytest.raises(StoreConnectionError, match="Unknown Qdrant settings"):
            QdrantAdapter.create(_schema(), {"bogus": 1})


# -- registry --------------------------------------------------------------------


def test_registry_builds_the_native_adapter() -> None:
    adapter = create_adapter("qdrant", _schema(), {"location": ":memory:"})
    assert isinstance(adapter, QdrantAdapter)
    assert adapter.provider == "qdrant"
    assert set(adapter.capabilities.supported) == {
        Operation.UPSERT,
        Operation.GET,
        Operation.DELETE,
        Operation.FIND,
        Operation.LIST,
        Operation.DELETE_WHERE,
        Operation.COUNT,
        Operation.VECTOR_SEARCH,
        Operation.HYBRID_SEARCH,
    }


def test_missing_qdrant_client_names_the_extra() -> None:
    with (
        patch(
            "holodeck.lib.retrieval.registry.importlib.import_module",
            side_effect=ImportError("No module named 'qdrant_client'"),
        ),
        pytest.raises(MissingClientError) as raised,
    ):
        create_adapter("qdrant", _schema(), {})
    assert raised.value.extra == "qdrant"
    assert "pip install 'holodeck-ai[qdrant]'" in str(raised.value)


def test_adapter_module_does_not_import_semantic_kernel_or_vector_store() -> None:
    assert qdrant.__file__ is not None
    text = Path(qdrant.__file__).read_text()
    assert "semantic_kernel" not in text
    assert "holodeck.lib.vector_store" not in text


# -- IDs -------------------------------------------------------------------------


def test_ids_keep_uuids_and_hash_everything_else() -> None:
    adapter = _memory()
    native = str(uuid.uuid4())
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, "holodeck/retrieval/docs")

    assert adapter.store_id(native) == native
    assert adapter.store_id(native.upper()) == native
    assert adapter.store_id("a_chunk_0") == str(uuid.uuid5(namespace, "a_chunk_0"))
    assert adapter.store_id("42") == str(uuid.uuid5(namespace, "42"))
    assert _memory(_schema("other")).store_id("a_chunk_0") != adapter.store_id(
        "a_chunk_0"
    )


def test_ids_match_the_bridge() -> None:
    from holodeck.lib.retrieval.adapters.sk_bridge import SKQdrantAdapter

    bridge = SKQdrantAdapter.create(_schema(), {"location": ":memory:"})
    adapter = _memory()
    for record_id in ["a_chunk_0", "42", 'it\'s "q"', str(uuid.uuid4()).upper()]:
        assert adapter.store_id(record_id) == bridge._store_id(record_id)


def test_holodeck_id_is_a_reserved_field() -> None:
    schema = CollectionSchema("docs", 4, (FieldSpec("holodeck_id"),))
    with pytest.raises(SchemaError):
        QdrantAdapter.create(schema, {})


# -- filters, tokens, indexes ----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["it's", 'say "hi"', "back\\slash", "x' or '1'='1", "ünïcode ✓", "a == b"],
)
def test_filter_values_are_passed_as_data(value: str) -> None:
    built = _memory().build_filter(RecordFilter.where(source_path=value))
    assert built == qm.Filter(
        must=[qm.FieldCondition(key="source_path", match=qm.MatchValue(value=value))]
    )


def test_filter_clause_types() -> None:
    built = _memory().build_filter(
        RecordFilter.where(chunk_index=3, mtime=2.5, pinned=True, content=None)
    )
    assert built is not None
    assert built.must == [
        qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=3)),
        qm.FieldCondition(key="mtime", range=qm.Range(gte=2.5, lte=2.5)),
        qm.FieldCondition(key="pinned", match=qm.MatchValue(value=True)),
        qm.IsNullCondition(is_null=qm.PayloadField(key="content")),
    ]


def test_a_number_filter_follows_the_field_type_not_the_python_type() -> None:
    adapter = _memory()

    float_field = adapter.build_filter(RecordFilter.where(mtime=2))
    integer_field = adapter.build_filter(RecordFilter.where(chunk_index=3.0))

    assert float_field == qm.Filter(
        must=[qm.FieldCondition(key="mtime", range=qm.Range(gte=2.0, lte=2.0))]
    )
    assert integer_field == qm.Filter(
        must=[qm.FieldCondition(key="chunk_index", match=qm.MatchValue(value=3))]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"chunk_index": 2.5},
        {"chunk_index": "2"},
        {"mtime": "2.5"},
        {"source_path": 2},
        {"pinned": 1},
    ],
)
def test_a_filter_value_that_cannot_equal_a_stored_value_is_rejected(
    bad: dict[str, Any],
) -> None:
    with pytest.raises(SchemaError, match="another type"):
        _memory().build_filter(RecordFilter.where(**bad))


@pytest.mark.parametrize("field", ["chunk_index", "mtime"])
def test_a_boolean_is_never_a_number_in_a_filter(field: str) -> None:
    with pytest.raises(SchemaError, match="boolean"):
        _memory().build_filter(RecordFilter.where(**{field: True}))


def test_empty_filter_and_unknown_field() -> None:
    adapter = _memory()
    assert adapter.build_filter(None) is None
    assert adapter.build_filter(RecordFilter()) is None
    with pytest.raises(SchemaError):
        adapter.build_filter(RecordFilter.where(nope="x"))


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Abiomed ABMD", ["abiomed", "abmd"]),
        ("A pile of foo", ["pile", "of", "foo"]),
        ("... ?!", []),
        ("Apple, Inc. 2024", ["apple", "inc", "2024"]),
        ("apple apple AAPL apple", ["apple", "aapl"]),
        ("", []),
    ],
)
def test_query_tokens(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_index_params_for_the_hierarchical_schema() -> None:
    schema = hierarchical_schema(4, "tool")
    built = {
        spec.name: index_params(
            spec.type, indexed=spec.indexed, full_text=spec.full_text
        )
        for spec in schema.fields
    }
    indexed = {name: params for name, params in built.items() if params is not None}
    keyword = {"source_path", "section_id", "defined_term", "defined_term_normalized"}
    assert set(indexed) == keyword | {"chunk_index", "searchable_text"}
    for name in keyword:
        assert isinstance(indexed[name], qm.KeywordIndexParams)
    assert isinstance(indexed["chunk_index"], qm.IntegerIndexParams)
    text = indexed["searchable_text"]
    assert isinstance(text, qm.TextIndexParams)
    assert text.tokenizer == qm.TokenizerType.WORD
    assert (text.min_token_len, text.max_token_len, text.lowercase) == (2, 20, True)
    assert isinstance(
        index_params(FieldType.FLOAT, indexed=True, full_text=False),
        qm.FloatIndexParams,
    )
    assert isinstance(
        index_params(FieldType.BOOLEAN, indexed=True, full_text=False),
        qm.BoolIndexParams,
    )


# -- requests against a mocked client --------------------------------------------


def _mocked(points: list[Any] | None = None) -> tuple[QdrantAdapter, MagicMock]:
    client = MagicMock(spec=AsyncQdrantClient)
    client.query_points = AsyncMock(return_value=MagicMock(points=points or []))
    client.collection_exists = AsyncMock(return_value=True)
    client.create_payload_index = AsyncMock()
    client.close = AsyncMock()
    adapter = QdrantAdapter.create(_schema(), {"url": "http://qdrant.test:6333"})
    return adapter, client


async def _prime(adapter: QdrantAdapter, client: MagicMock) -> None:
    with patch.object(qdrant, "create_qdrant_client", return_value=client):
        await adapter._open()


@pytest.mark.asyncio
async def test_hybrid_uses_match_text_per_token_and_rrf() -> None:
    point = MagicMock(id="chunk-1", score=0.9, payload={"content": "x"})
    adapter, client = _mocked([point])
    await _prime(adapter, client)

    hits = await adapter.hybrid_search(
        HybridQuery(
            "Abiomed ABMD",
            [0.1] * 4,
            top_k=5,
            text_field="content",
            filter=RecordFilter.where(source_path="a.md"),
        )
    )

    assert [(hit.record.id, hit.score) for hit in hits] == [("chunk-1", 0.9)]
    kwargs = client.query_points.await_args.kwargs
    assert kwargs["collection_name"] == "docs"
    assert kwargs["limit"] == 5
    assert kwargs["query"] == qm.FusionQuery(fusion=qm.Fusion.RRF)
    vector_leg, keyword_leg = kwargs["prefetch"]
    base = qm.Filter(
        must=[qm.FieldCondition(key="source_path", match=qm.MatchValue(value="a.md"))]
    )
    assert vector_leg == qm.Prefetch(
        query=[0.1] * 4, using="embedding", filter=base, limit=20
    )
    text = qm.Filter(
        should=[
            qm.FieldCondition(key="content", match=qm.MatchText(text="abiomed")),
            qm.FieldCondition(key="content", match=qm.MatchText(text="abmd")),
        ]
    )
    assert keyword_leg == qm.Prefetch(filter=qm.Filter(must=[base, text]), limit=20)


@pytest.mark.asyncio
async def test_hybrid_candidate_pool_and_punctuation_fallback() -> None:
    adapter, client = _mocked()
    await _prime(adapter, client)

    await adapter.hybrid_search(HybridQuery("?!?", [0.1] * 4, 7, "content"))
    (leg,) = client.query_points.await_args.kwargs["prefetch"]
    assert leg.limit == 28
    assert leg.filter is None


@pytest.mark.asyncio
async def test_vector_search_clamps_and_maps_ids() -> None:
    points = [
        MagicMock(id="u1", score=1.0000002, payload={"holodeck_id": "a", "pinned": 1}),
        MagicMock(id="u2", score=-0.2, payload={"chunk_index": 2.0}),
    ]
    adapter, client = _mocked(points)
    await _prime(adapter, client)

    hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 2))

    assert [(h.record.id, h.score) for h in hits] == [("a", 1.0), ("u2", 0.0)]
    assert hits[0].record.fields == {"pinned": True}
    assert hits[1].record.fields == {"chunk_index": 2}
    kwargs = client.query_points.await_args.kwargs
    assert kwargs["using"] == "embedding"
    assert kwargs["with_vectors"] is False


@pytest.mark.asyncio
async def test_payload_indexes_are_created_once_and_failures_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from qdrant_client.http.exceptions import UnexpectedResponse

    adapter, client = _mocked()
    client.create_payload_index.side_effect = UnexpectedResponse(
        400, "Bad Request", b"{}", MagicMock()
    )
    await _prime(adapter, client)

    await adapter.ensure_collection()
    await adapter.ensure_collection()

    names = [
        c.kwargs["field_name"] for c in client.create_payload_index.await_args_list
    ]
    assert names == ["source_path", "chunk_index", "content"]
    assert "Qdrant payload index failed for 'source_path'" in caplog.text


# -- lifecycle and errors --------------------------------------------------------


@pytest.mark.asyncio
async def test_one_client_is_reused_and_close_is_idempotent() -> None:
    created: list[AsyncQdrantClient] = []
    real = qdrant.create_qdrant_client

    def tracking(params: Any) -> AsyncQdrantClient:
        client = real(params)
        created.append(client)
        return client

    adapter = _memory()
    with patch.object(qdrant, "create_qdrant_client", side_effect=tracking):
        await adapter.ensure_collection()
        await adapter.upsert(
            [Record("a", {"source_path": "a.md"}, [1.0, 0.0, 0.0, 0.0])]
        )
        assert [r.id for r in await adapter.get(["a"])] == ["a"]
        assert await adapter.count() == 1
        assert len(created) == 1

        await adapter.close()
        await adapter.close()

        # Usable again: a new client (the in-process store starts empty).
        assert await adapter.collection_exists() is False
        assert len(created) == 2
        await adapter.close()


@pytest.mark.asyncio
async def test_in_memory_round_trip_with_special_ids_and_values() -> None:
    adapter = _memory()
    weird_id = 'it\'s "x" \\ id'
    weird_source = 'dir/it\'s "b".md'
    try:
        await adapter.ensure_collection()
        await adapter.upsert(
            [
                Record(weird_id, {"source_path": weird_source}, [1.0, 0.0, 0.0, 0.0]),
                Record("b", {"source_path": "b.md"}, [0.0, 1.0, 0.0, 0.0]),
            ]
        )
        (record,) = await adapter.get([weird_id], include_vectors=True)
        assert record.id == weird_id
        assert record.fields == {
            "source_path": weird_source,
            "chunk_index": 0,
            "content": "",
            "mtime": 0.0,
            "pinned": False,
        }
        assert record.vector == pytest.approx([1.0, 0.0, 0.0, 0.0])
        found = await adapter.find(RecordFilter.where(source_path=weird_source))
        assert [r.id for r in found] == [weird_id]
        assert await adapter.delete_where(RecordFilter.where(source_path="b.md")) == 1
        assert await adapter.count() == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_count_is_zero_without_a_collection() -> None:
    adapter = _memory()
    try:
        assert await adapter.count() == 0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_unreachable_server_raises_store_connection_error() -> None:
    adapter = QdrantAdapter.create(
        _schema(), {"url": "http://127.0.0.1:9", "check_compatibility": False}
    )
    try:
        with pytest.raises(StoreConnectionError, match="qdrant"):
            await adapter.collection_exists()
    finally:
        await adapter.close()


# -- secrets stay out of errors --------------------------------------------------

KEY = "s3cr3t-api-key-value"


def _formatted(error: BaseException) -> str:
    return "".join(traceback.format_exception(error))


@pytest.mark.parametrize("bad", [f"{KEY}\n", f"{KEY}\r\n", f"a\tb{KEY}"])
def test_a_credential_with_a_control_character_is_rejected_without_echoing_it(
    bad: str,
) -> None:
    for connection in ({"api_key": bad}, {"headers": {"X-Token": bad}}):
        with pytest.raises(StoreConnectionError, match="control character") as raised:
            QdrantAdapter.create(_schema(), connection)
        assert KEY not in _formatted(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("echoed", [KEY, f"b'{KEY}\\n'", f"Illegal header value {KEY}"])
async def test_an_error_that_repeats_the_api_key_is_masked_and_not_chained(
    echoed: str,
) -> None:
    from qdrant_client.http.exceptions import ResponseHandlingException

    adapter = QdrantAdapter.create(
        _schema(), {"url": "http://qdrant.test:6333", "api_key": KEY}
    )
    client = MagicMock(spec=AsyncQdrantClient)
    client.collection_exists = AsyncMock(
        side_effect=ResponseHandlingException(Exception(echoed))
    )
    await _prime(adapter, client)

    with pytest.raises(StoreConnectionError) as raised:
        await adapter.collection_exists()

    assert KEY not in str(raised.value)
    assert "***" in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert KEY not in _formatted(raised.value)


@pytest.mark.asyncio
async def test_a_header_value_and_a_keyed_connection_string_are_masked() -> None:
    from qdrant_client.http.exceptions import UnexpectedResponse

    url = f"https://{KEY}@qdrant.test:6333"
    adapter = QdrantAdapter.create(
        _schema(), {"connection_string": url, "headers": {"X-Token": "tok-123456"}}
    )
    client = MagicMock(spec=AsyncQdrantClient)
    client.collection_exists = AsyncMock(
        side_effect=UnexpectedResponse(
            401, "Unauthorized", f"{url} tok-123456".encode(), MagicMock()
        )
    )
    await _prime(adapter, client)

    with pytest.raises(StoreConnectionError) as raised:
        await adapter.collection_exists()

    assert KEY not in _formatted(raised.value)
    assert "tok-123456" not in _formatted(raised.value)


@pytest.mark.asyncio
async def test_an_error_without_a_secret_keeps_its_chain() -> None:
    from qdrant_client.http.exceptions import ResponseHandlingException

    adapter = QdrantAdapter.create(
        _schema(), {"url": "http://qdrant.test:6333", "api_key": KEY}
    )
    original = ResponseHandlingException(Exception("connection refused"))
    client = MagicMock(spec=AsyncQdrantClient)
    client.collection_exists = AsyncMock(side_effect=original)
    await _prime(adapter, client)

    with pytest.raises(StoreConnectionError) as raised:
        await adapter.collection_exists()

    assert raised.value.__cause__ is original
    assert "connection refused" in str(raised.value)


# -- ID collisions (review fix 1) ---------------------------------------------


def _rec(record_id: str, content: str = "") -> Record:
    return Record(
        record_id,
        {"source_path": "s.md", "content": content},
        [1.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.asyncio
async def test_two_ids_sharing_a_point_id_are_rejected_in_one_call() -> None:
    adapter = _memory()
    try:
        await adapter.ensure_collection()
        clash = adapter.store_id("invoice-7")
        with pytest.raises(SchemaError, match="same store ID"):
            await adapter.upsert([_rec("invoice-7"), _rec(clash)])
        with pytest.raises(SchemaError, match="same store ID"):
            await adapter.replace_source(
                RecordFilter.where(source_path="s.md"),
                [_rec(clash), _rec("invoice-7")],
            )
        assert await adapter.count() == 0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_two_spellings_of_one_uuid_are_one_id_not_a_collision() -> None:
    adapter = _memory()
    native = str(uuid.uuid4())
    try:
        await adapter.ensure_collection()
        await adapter.upsert([_rec(native, "lower"), _rec(native.upper(), "upper")])
        assert await adapter.count() == 1
        await adapter.upsert([_rec(native.upper(), "again")])
        assert await adapter.count() == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("hashed_first", [True, False])
async def test_an_id_cannot_overwrite_a_record_with_another_id(
    hashed_first: bool,
) -> None:
    adapter = _memory()
    try:
        await adapter.ensure_collection()
        clash = adapter.store_id("invoice-7")
        first, second = ("invoice-7", clash) if hashed_first else (clash, "invoice-7")
        await adapter.upsert([_rec(first, "original")])
        with pytest.raises(SchemaError, match="already holds"):
            await adapter.upsert([_rec("fresh"), _rec(second, "intruder")])
        with pytest.raises(SchemaError, match="already holds"):
            await adapter.replace_source(
                RecordFilter.where(source_path="s.md"), [_rec(second, "intruder")]
            )
        (kept,) = await adapter.get([first])
        assert kept.id == first
        assert kept.fields["content"] == "original"
        assert await adapter.count() == 1
        # The owner may still rewrite its own record.
        await adapter.upsert([_rec(first, "rewritten")])
        (kept,) = await adapter.get([first])
        assert kept.fields["content"] == "rewritten"
    finally:
        await adapter.close()
