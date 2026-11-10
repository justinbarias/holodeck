"""Unit tests for the native Azure Cosmos DB NoSQL adapter.

They run on ``tests/contract/retrieval/cosmos_fake.py``, which answers only the
SQL shapes the adapter emits. ``tests/contract/retrieval`` runs the contract
suite on the fake and on the Linux emulator. Nothing here ran against the
Cosmos DB service.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest
from azure.core.exceptions import AzureError
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    HybridQuery,
    IncompatibleCollectionError,
    LexicalQuery,
    MissingClientError,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    ScoreKind,
    StoreConnectionError,
    UnsupportedOperationError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import cosmos_nosql as module
from holodeck.lib.retrieval.adapters.cosmos_nosql import (
    CosmosNoSqlAdapter,
    CosmosSettings,
    decode_id,
    encode_id,
    parse_cosmos_connection_string,
    resolve_vector_index,
    settings_from_connection,
    tokenize,
)
from tests.contract.retrieval.conftest import CORPUS, make_record, make_schema
from tests.contract.retrieval.cosmos_fake import (
    FakeAccount,
    FakeCosmosAdapter,
    FakeCosmosClient,
)

KEY = "SECRET_account_key_0123456789=="
URL = "https://holodeck.documents.azure.com:443/"
HOSTILE = "x') OR 1=1 --; DROP c \"\\ '"


def _schema(dimensions: int = 4) -> CollectionSchema:
    return CollectionSchema(
        name="docs",
        dimensions=dimensions,
        fields=(
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def _settings(**overrides: object) -> CosmosSettings:
    values: dict[str, object] = {
        "endpoint": URL,
        "key": KEY,
        "database_name": "holodeck",
        "create_database": True,
    }
    values.update(overrides)
    return CosmosSettings(**values)  # type: ignore[arg-type]


def _adapter(
    schema: CollectionSchema | None = None, **overrides: object
) -> FakeCosmosAdapter:
    return FakeCosmosAdapter(schema or make_schema(), _settings(**overrides))


# -- connection settings ---------------------------------------------------------


class TestConnectionSettings:
    def test_reads_url_key_and_database(self) -> None:
        settings = settings_from_connection(
            {"url": URL, "key": KEY, "database_name": "kb"}
        )

        assert settings == CosmosSettings(URL, KEY, "kb")

    def test_endpoint_is_an_alias_of_url(self) -> None:
        settings = settings_from_connection(
            {"endpoint": URL, "key": KEY, "database_name": "kb"}
        )
        assert settings.endpoint == URL

    def test_url_and_endpoint_must_agree(self) -> None:
        with pytest.raises(StoreConnectionError, match="differ"):
            settings_from_connection(
                {
                    "url": URL,
                    "endpoint": "https://other.documents.azure.com/",
                    "key": KEY,
                    "database_name": "kb",
                }
            )

    def test_portal_connection_string(self) -> None:
        settings = settings_from_connection(
            {
                "connection_string": f"AccountEndpoint={URL};AccountKey={KEY};",
                "database_name": "kb",
            }
        )

        assert (settings.endpoint, settings.key) == (URL, KEY)

    def test_explicit_settings_win_over_the_connection_string(self) -> None:
        settings = settings_from_connection(
            {
                "connection_string": "AccountEndpoint=https://a.example.com/;"
                "AccountKey=from-string;",
                "url": URL,
                "key": KEY,
                "database_name": "kb",
            }
        )
        assert (settings.endpoint, settings.key) == (URL, KEY)

    def test_connection_string_parsing(self) -> None:
        assert parse_cosmos_connection_string("https://x.example.com/") == (
            "https://x.example.com/",
            None,
        )
        assert parse_cosmos_connection_string(
            "AccountKey=a=b==; AccountEndpoint=https://x.example.com/"
        ) == ("https://x.example.com/", "a=b==")
        with pytest.raises(ValueError, match="AccountEndpoint"):
            parse_cosmos_connection_string("AccountKey=abc")
        with pytest.raises(ValueError, match="empty"):
            parse_cosmos_connection_string(" ")

    def test_environment_fallbacks_are_the_ones_semantic_kernel_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_COSMOS_DB_NO_SQL_URL", URL)
        monkeypatch.setenv("AZURE_COSMOS_DB_NO_SQL_KEY", KEY)
        monkeypatch.setenv("AZURE_COSMOS_DB_NO_SQL_DATABASE_NAME", "from_env")

        assert settings_from_connection({}) == CosmosSettings(URL, KEY, "from_env")

    @pytest.mark.parametrize(
        ("connection", "message"),
        [
            ({"key": KEY, "database_name": "d"}, "No account URL"),
            ({"url": URL, "database_name": "d"}, "No account key"),
            ({"url": URL, "key": KEY}, "No database name"),
            ({"url": "ftp://x", "key": KEY, "database_name": "d"}, "must start with"),
            ({"url": URL, "key": KEY, "database_name": "d", "bogus": 1}, "bogus"),
            ({"url": URL, "key": 7, "database_name": "d"}, "'key' must be a string"),
            (
                {
                    "url": URL,
                    "key": KEY,
                    "database_name": "d",
                    "create_database": "yes",
                },
                "true or false",
            ),
            (
                {"url": URL, "key": KEY, "database_name": "d", "vector_index": "hnsw"},
                "vector_index must be one of",
            ),
            (
                {"url": URL, "key": KEY, "database_name": "d", "partition_key": "/pk"},
                "Only the partition key /id",
            ),
        ],
    )
    def test_invalid_settings_are_errors(
        self,
        connection: dict[str, object],
        message: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for name in (
            "AZURE_COSMOS_DB_NO_SQL_URL",
            "AZURE_COSMOS_DB_NO_SQL_KEY",
            "AZURE_COSMOS_DB_NO_SQL_DATABASE_NAME",
        ):
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(StoreConnectionError, match=message):
            settings_from_connection(connection)

    def test_the_partition_key_the_sk_default_used_is_accepted(self) -> None:
        for value in ("id", "/id"):
            settings_from_connection(
                {"url": URL, "key": KEY, "database_name": "d", "partition_key": value}
            )

    def test_unknown_key_error_names_the_accepted_keys(self) -> None:
        with pytest.raises(StoreConnectionError) as raised:
            settings_from_connection(
                {"url": URL, "key": KEY, "database_name": "d", "cosmos_client": 1}
            )
        assert "cosmos_client" in str(raised.value)
        assert "database_name" in str(raised.value)

    def test_the_key_is_not_in_the_settings_repr(self) -> None:
        assert KEY not in repr(_settings())

    def test_collection_name_is_accepted_because_the_schema_carries_it(self) -> None:
        settings_from_connection(
            {"url": URL, "key": KEY, "database_name": "d", "collection_name": "c"}
        )

    def test_create_builds_an_adapter_without_a_request(self) -> None:
        adapter = CosmosNoSqlAdapter.create(
            _schema(),
            {"url": URL, "key": KEY, "database_name": "d"},
        )

        assert adapter.provider == "azure-cosmos-nosql"
        assert adapter._client is None


# -- registry ----------------------------------------------------------------------


def test_registry_builds_the_native_adapter() -> None:
    adapter = create_adapter(
        "azure-cosmos-nosql",
        _schema(),
        {"url": URL, "key": KEY, "database_name": "d"},
    )
    assert type(adapter) is CosmosNoSqlAdapter


def test_missing_client_library_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, module.__name__, raising=False)
    monkeypatch.setitem(sys.modules, "azure.cosmos", None)

    with pytest.raises(MissingClientError) as raised:
        create_adapter("azure-cosmos-nosql", _schema(), {})

    assert raised.value.extra == "azure-cosmos"
    assert "holodeck-ai[azure-cosmos]" in str(raised.value)


def test_adapter_has_no_untyped_escape_hatches() -> None:
    tree = ast.parse(Path(module.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "Any" not in names
    assert "getattr" not in names
    assert "typing.Any" not in imported


# -- capabilities ------------------------------------------------------------------


def test_capabilities_without_full_text_search() -> None:
    adapter = _adapter(full_text_search=False)

    assert adapter.capabilities.supported == frozenset(
        {
            Operation.UPSERT,
            Operation.GET,
            Operation.DELETE,
            Operation.FIND,
            Operation.LIST,
            Operation.DELETE_WHERE,
            Operation.COUNT,
            Operation.VECTOR_SEARCH,
        }
    )
    assert (
        "full_text_search: true" in adapter.capabilities.hints[Operation.HYBRID_SEARCH]
    )


def test_capabilities_with_full_text_search_add_hybrid() -> None:
    adapter = _adapter(full_text_search=True)
    assert adapter.capabilities.supports(Operation.HYBRID_SEARCH)
    assert not adapter.capabilities.supports(Operation.LEXICAL_SEARCH)


def test_full_text_search_without_a_full_text_field_offers_no_hybrid() -> None:
    schema = CollectionSchema("plain", 4, (FieldSpec("title"),))
    adapter = FakeCosmosAdapter(schema, _settings(full_text_search=True))
    assert not adapter.capabilities.supports(Operation.HYBRID_SEARCH)
    assert adapter.full_text_policy() is None


@pytest.mark.asyncio
async def test_hybrid_search_is_refused_before_any_request_when_off() -> None:
    adapter = _adapter(full_text_search=False)
    with pytest.raises(UnsupportedOperationError):
        await adapter.hybrid_search(
            HybridQuery("alpha", [1.0, 0.0, 0.0, 0.0], 3, "content")
        )
    with pytest.raises(UnsupportedOperationError):
        await adapter.lexical_search(LexicalQuery("alpha", 3, "content"))
    assert adapter.account.clients_opened == 0


# -- item IDs ----------------------------------------------------------------------

HOSTILE_IDS = [
    "docs/guide.md_chunk_0",
    "C:\\docs\\a.md_chunk_1",
    "what?_chunk_2",
    "frag#ment",
    "~hd~already-looks-encoded",
    " leading",
    "trailing ",
    "tab\there",
    "nul\x00byte",
    "ünïcode/路径/文件.md_chunk_3",
    "emoji-\U0001f600/x",
    "a/b#c?d\\e",
]


class TestIds:
    @pytest.mark.parametrize("record_id", HOSTILE_IDS)
    def test_round_trip(self, record_id: str) -> None:
        stored = encode_id(record_id)

        assert not set("/\\?#") & set(stored)
        assert decode_id(stored) == record_id

    @pytest.mark.parametrize(
        "record_id",
        ["abc", "0b1f2d3c-aaaa-5bbb-8ccc-1234567890ab", "doc.md_chunk_12", "a%2Fb"],
    )
    def test_valid_ids_are_stored_unchanged(self, record_id: str) -> None:
        assert encode_id(record_id) == record_id
        assert decode_id(record_id) == record_id

    def test_the_mapping_is_one_to_one(self) -> None:
        # A raw ID that already looks like an encoded one must not collide
        # with the encoding of the ID it would decode to.
        encoded = encode_id("a/b")
        assert encode_id(encoded) != encoded
        assert decode_id(encode_id(encoded)) == encoded
        stored = {encode_id(record_id) for record_id in [*HOSTILE_IDS, encoded]}
        assert len(stored) == len(HOSTILE_IDS) + 1

    def test_an_undecodable_prefixed_id_is_returned_as_stored(self) -> None:
        assert decode_id("~hd~!!!") == "~hd~!!!"

    def test_empty_and_oversized_ids_are_schema_errors(self) -> None:
        with pytest.raises(SchemaError, match="empty"):
            encode_id("")
        with pytest.raises(SchemaError, match="too long"):
            encode_id("a" * 1024)
        with pytest.raises(SchemaError, match="too long"):
            encode_id("/" * 800)  # 800 bytes become 1070 base64 characters

    def test_lone_surrogates_are_a_schema_error(self) -> None:
        with pytest.raises(SchemaError, match="UTF-8"):
            encode_id("bad\ud800")


@pytest.mark.asyncio
async def test_path_ids_are_stored_encoded_with_the_original_alongside() -> None:
    adapter = _adapter()
    await adapter.ensure_collection()
    record = make_record("docs/guide.md_chunk_0", [1.0, 0.0, 0.0, 0.0])

    await adapter.upsert([record, make_record("plain", [0.0, 1.0, 0.0, 0.0])])

    stored = adapter.account.databases["holodeck"][adapter.schema.name].documents
    encoded = encode_id("docs/guide.md_chunk_0")
    assert set(stored) == {encoded, "plain"}
    assert stored[encoded]["holodeck_id"] == "docs/guide.md_chunk_0"
    assert "holodeck_id" not in stored["plain"]
    (read,) = await adapter.get(["docs/guide.md_chunk_0"])
    assert read.id == "docs/guide.md_chunk_0"
    hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], top_k=1))
    assert hits[0].record.id == "docs/guide.md_chunk_0"
    await adapter.delete(["docs/guide.md_chunk_0"])
    assert await adapter.get(["docs/guide.md_chunk_0"]) == []


@pytest.mark.asyncio
async def test_documents_written_by_semantic_kernel_read_back() -> None:
    # SK stored the key as `id`, data fields by name, `embedding` as a list, and
    # no holodeck_id. A field it never wrote, and a null, read as None.
    adapter = _adapter()
    await adapter.ensure_collection()
    container = adapter.account.databases["holodeck"][adapter.schema.name]
    container.documents["sk-written"] = {
        "id": "sk-written",
        "source_path": "a.md",
        "chunk_index": 3,
        "content": "hello",
        "mtime": None,
        "embedding": [1, 0, 0, 0],
    }

    (record,) = await adapter.get(["sk-written"], include_vectors=True)

    assert record.id == "sk-written"
    assert dict(record.fields) == {
        "source_path": "a.md",
        "chunk_index": 3,
        "content": "hello",
        "mtime": None,
        "pinned": None,
    }
    assert record.vector == [1.0, 0.0, 0.0, 0.0]


# -- container definition --------------------------------------------------------


class TestContainerDefinition:
    def test_policies_are_the_literal_layout(self) -> None:
        adapter = FakeCosmosAdapter(_schema(), _settings())

        assert adapter.vector_embedding_policy() == {
            "vectorEmbeddings": [
                {
                    "path": '/"embedding"',
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 4,
                }
            ]
        }
        assert adapter.indexing_policy() == {
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": '/"_etag"/?'},
                {"path": '/"mtime"/*'},
                {"path": '/"pinned"/*'},
                {"path": '/"embedding"/*'},
            ],
            "vectorIndexes": [{"path": '/"embedding"', "type": "flat"}],
        }
        assert adapter.full_text_policy() is None

    def test_full_text_search_adds_the_policy_and_the_index(self) -> None:
        adapter = FakeCosmosAdapter(_schema(), _settings(full_text_search=True))

        assert adapter.full_text_policy() == {
            "defaultLanguage": "en-US",
            "fullTextPaths": [{"path": '/"content"', "language": "en-US"}],
        }
        assert adapter.indexing_policy()["fullTextIndexes"] == [{"path": '/"content"'}]

    @pytest.mark.parametrize(
        ("dimensions", "requested", "expected"),
        [
            (4, None, "flat"),
            (505, None, "flat"),
            (506, None, "quantizedFlat"),
            (1536, None, "quantizedFlat"),
            (4096, None, "quantizedFlat"),
            (1536, "diskANN", "diskANN"),
            (4, "diskANN", "diskANN"),
            (4, "quantizedFlat", "quantizedFlat"),
        ],
    )
    def test_vector_index_choice(
        self, dimensions: int, requested: str | None, expected: str
    ) -> None:
        assert resolve_vector_index(dimensions, requested) == expected

    def test_dimensions_the_service_cannot_index_are_schema_errors(self) -> None:
        with pytest.raises(SchemaError, match="4096"):
            resolve_vector_index(4097, None)
        with pytest.raises(SchemaError, match="505"):
            resolve_vector_index(1536, "flat")
        with pytest.raises(SchemaError, match="505"):
            FakeCosmosAdapter(_schema(1536), _settings(vector_index="flat"))

    @pytest.mark.parametrize(
        "name", ["has space", "a-b", 'q"uote', "holodeck_id", "_etag", "1abc", ""]
    )
    def test_field_names_that_cannot_be_queried_safely_are_rejected(
        self, name: str
    ) -> None:
        schema = CollectionSchema("c", 4, (FieldSpec(name),))
        with pytest.raises(SchemaError, match="Cosmos DB"):
            FakeCosmosAdapter(schema, _settings())

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_what_the_policies_describe(self) -> None:
        adapter = FakeCosmosAdapter(_schema(), _settings(full_text_search=True))

        await adapter.ensure_collection()

        properties = adapter.account.databases["holodeck"]["docs"].properties
        assert properties["partitionKey"] == {
            "paths": ["/id"],
            "kind": "Hash",
            "version": 2,
        }
        assert properties["indexingPolicy"] == adapter.indexing_policy()
        assert properties["vectorEmbeddingPolicy"] == adapter.vector_embedding_policy()
        assert properties["fullTextPolicy"] == adapter.full_text_policy()


# -- statements --------------------------------------------------------------------

COLUMNS = (
    'c.id, c["source_path"], c["chunk_index"], c["content"], c["mtime"], c["pinned"]'
)


class TestStatements:
    def setup_method(self) -> None:
        self.adapter = FakeCosmosAdapter(_schema(), _settings(full_text_search=True))

    def test_select_all(self) -> None:
        statement = self.adapter.select_statement()
        assert statement.text == f"SELECT {COLUMNS} FROM c"  # noqa: S608
        assert statement.parameters == ()

    def test_select_with_filter_and_limit(self) -> None:
        statement = self.adapter.select_statement(
            RecordFilter.where(source_path="a.md", chunk_index=2, pinned=True),
            limit=5,
        )

        assert statement.text == (
            f'SELECT TOP @limit {COLUMNS} FROM c WHERE c["source_path"] = @f0 '  # noqa: S608
            'AND c["chunk_index"] = @f1 AND c["pinned"] = @f2'
        )
        assert statement.parameters == (
            ("@f0", "a.md"),
            ("@f1", 2),
            ("@f2", True),
            ("@limit", 5),
        )

    def test_a_null_filter_value_is_is_null_without_a_parameter(self) -> None:
        statement = self.adapter.select_statement(
            RecordFilter.where(source_path=None, chunk_index=1)
        )
        assert statement.text.endswith(
            'WHERE IS_NULL(c["source_path"]) AND c["chunk_index"] = @f0'
        )
        assert statement.parameters == (("@f0", 1),)

    def test_ids_only(self) -> None:
        statement = self.adapter.select_statement(
            RecordFilter.where(source_path="a.md"), ids_only=True
        )
        assert statement.text == 'SELECT c.id FROM c WHERE c["source_path"] = @f0'

    def test_count(self) -> None:
        assert self.adapter.count_statement().text == "SELECT VALUE COUNT(1) FROM c"

    def test_vector_statement(self) -> None:
        statement = self.adapter.vector_statement(
            VectorQuery(
                [1, 0, 0, 0],
                top_k=3,
                filter=RecordFilter.where(source_path="b.md"),
            )
        )

        distance = 'VectorDistance(c["embedding"], @vector)'
        assert statement.text == (
            f"SELECT TOP @top {COLUMNS}, {distance} AS score "  # noqa: S608
            f'FROM c WHERE c["source_path"] = @f0 ORDER BY {distance}'
        )
        assert statement.parameters == (
            ("@top", 3),
            ("@vector", [1.0, 0.0, 0.0, 0.0]),
            ("@f0", "b.md"),
        )

    def test_hybrid_statement(self) -> None:
        statement = self.adapter.hybrid_statement(
            HybridQuery(
                "alpha zeppelin",
                [1, 0, 0, 0],
                top_k=4,
                text_field="content",
                filter=RecordFilter.where(chunk_index=0),
            ),
            ["alpha", "zeppelin"],
        )

        assert statement.text == (
            f'SELECT TOP @top {COLUMNS} FROM c WHERE c["chunk_index"] = @f0 '  # noqa: S608
            'ORDER BY RANK RRF(VectorDistance(c["embedding"], @vector), '
            'FullTextScore(c["content"], @t0, @t1))'
        )
        assert statement.parameters == (
            ("@top", 4),
            ("@vector", [1.0, 0.0, 0.0, 0.0]),
            ("@f0", 0),
            ("@t0", "alpha"),
            ("@t1", "zeppelin"),
        )

    def test_the_sdk_parameter_form(self) -> None:
        statement = self.adapter.select_statement(
            RecordFilter.where(source_path="a.md")
        )
        assert statement.sdk_parameters() == [{"name": "@f0", "value": "a.md"}]

    def test_hostile_values_are_parameters_never_text(self) -> None:
        statements = [
            self.adapter.select_statement(RecordFilter.where(source_path=HOSTILE)),
            self.adapter.vector_statement(
                VectorQuery([1, 0, 0, 0], 2, RecordFilter.where(source_path=HOSTILE))
            ),
            self.adapter.hybrid_statement(
                HybridQuery(
                    HOSTILE,
                    [1, 0, 0, 0],
                    2,
                    "content",
                    RecordFilter.where(source_path=HOSTILE),
                ),
                tokenize(HOSTILE) or ["x"],
            ),
        ]

        for statement in statements:
            assert HOSTILE not in statement.text
            assert "DROP" not in statement.text
            assert "'" not in statement.text
            assert any(value == HOSTILE for _, value in statement.parameters)

    @pytest.mark.parametrize(
        "name",
        ['source_path"] = 1 OR c["x', "nope", "id", "embedding", "c.id; --", "*"],
    )
    def test_unknown_or_hostile_field_names_are_rejected(self, name: str) -> None:
        with pytest.raises(SchemaError):
            self.adapter.select_statement(RecordFilter.where(**{name: "v"}))
        with pytest.raises(SchemaError):
            self.adapter.vector_statement(
                VectorQuery([1, 0, 0, 0], 1, RecordFilter.where(**{name: "v"}))
            )

    @pytest.mark.asyncio
    async def test_the_text_field_must_be_a_full_text_schema_field(self) -> None:
        for name in ("nope", "source_path", 'content"] OR 1=1 --'):
            with pytest.raises(SchemaError):
                await self.adapter.hybrid_search(
                    HybridQuery("alpha", [1.0, 0.0, 0.0, 0.0], 3, name)
                )
        assert self.adapter.account.clients_opened == 0

    def test_tokenize(self) -> None:
        assert tokenize("The Zeppelin, the zeppelin! a x-ray 42") == [
            "the",
            "zeppelin",
            "ray",
            "42",
        ]
        assert tokenize("  ?!-- ") == []
        assert len(tokenize(" ".join(f"word{i}" for i in range(100)))) == 32

    def test_numbers_and_floats_are_typed_in_parameters(self) -> None:
        statement = self.adapter.select_statement(RecordFilter.where(mtime=2))
        assert statement.parameters == (("@f0", 2.0),)
        assert isinstance(statement.parameters[0][1], float)


# -- behavior on the fake --------------------------------------------------------


async def _loaded(**overrides: object) -> FakeCosmosAdapter:
    adapter = _adapter(**overrides)
    await adapter.ensure_collection()
    await adapter.upsert(CORPUS)
    return adapter


class TestBehavior:
    @pytest.mark.asyncio
    async def test_the_client_is_lazy_and_close_is_repeatable_and_reusable(
        self,
    ) -> None:
        adapter = _adapter()
        assert adapter.account.clients_opened == 0

        assert await adapter.count() == 0
        assert adapter.account.clients_opened == 1
        await adapter.close()
        await adapter.close()
        assert adapter.account.clients_closed == 1

        await adapter.ensure_collection()
        await adapter.upsert([make_record("a", [1.0, 0.0, 0.0, 0.0])])
        assert await adapter.count() == 1
        assert adapter.account.clients_opened == 2
        await adapter.close()
        assert adapter.account.clients_closed == 2

    @pytest.mark.asyncio
    async def test_one_client_serves_every_operation(self) -> None:
        adapter = await _loaded()
        await adapter.get(["a_chunk_0"])
        await adapter.find(RecordFilter.where(source_path="a.md"))
        await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 1))
        assert adapter.account.clients_opened == 1

    @pytest.mark.asyncio
    async def test_reads_on_a_missing_container_return_nothing(self) -> None:
        adapter = _adapter()
        assert await adapter.collection_exists() is False
        assert await adapter.count() == 0
        assert await adapter.get(["x"]) == []
        assert await adapter.find(RecordFilter.where(source_path="a.md")) == []
        assert (await adapter.list_page()).records == ()
        assert await adapter.delete_where(RecordFilter.where(source_path="a")) == 0
        await adapter.delete(["x"])
        assert await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 3)) == []

    @pytest.mark.asyncio
    async def test_upsert_creates_the_container_when_it_is_missing(self) -> None:
        adapter = _adapter()

        await adapter.upsert([make_record("a", [1.0, 0.0, 0.0, 0.0])])

        assert await adapter.collection_exists() is True
        assert await adapter.count() == 1

    @pytest.mark.asyncio
    async def test_upsert_recovers_when_the_container_vanishes_underneath(
        self,
    ) -> None:
        adapter = _adapter()
        await adapter.ensure_collection()
        await adapter.count()  # verified and cached
        del adapter.account.databases["holodeck"][adapter.schema.name]

        await adapter.upsert([make_record("a", [1.0, 0.0, 0.0, 0.0])])

        assert await adapter.count() == 1

    @pytest.mark.asyncio
    async def test_a_missing_database_is_a_clear_error_unless_creation_is_on(
        self,
    ) -> None:
        adapter = _adapter(create_database=False)
        with pytest.raises(StoreConnectionError, match="create_database: true"):
            await adapter.ensure_collection()

        creating = _adapter(create_database=True)
        await creating.ensure_collection()
        assert "holodeck" in creating.account.databases

    @pytest.mark.asyncio
    async def test_writes_are_bounded_and_deduplicated(self) -> None:
        adapter = _adapter()
        records = [
            make_record(f"r{i % 40}", [1.0, 0.0, 0.0, 0.0], content=f"v{i}")
            for i in range(80)
        ]

        await adapter.upsert(records)

        assert await adapter.count() == 40
        assert 1 < adapter.account.peak_in_flight <= module.WRITE_CONCURRENCY
        (last,) = await adapter.get(["r39"])
        assert last.fields["content"] == "v79"  # the last write of a repeated ID wins

    @pytest.mark.asyncio
    async def test_a_failed_write_leaves_no_task_running(self) -> None:
        adapter = _adapter()
        await adapter.ensure_collection()
        bad = make_record("ok", [1.0, 0.0, 0.0, 0.0])
        nan = make_record("nan", [float("nan"), 0.0, 0.0, 0.0])

        with pytest.raises(SchemaError, match="non-finite"):
            await adapter.upsert([bad, nan])

        assert await adapter.count() == 0
        assert adapter.account.in_flight == 0

    @pytest.mark.asyncio
    async def test_get_keeps_request_order_and_repeats(self) -> None:
        adapter = await _loaded()
        records = await adapter.get(["b_chunk_0", "missing", "a_chunk_0", "b_chunk_0"])
        assert [r.id for r in records] == ["b_chunk_0", "a_chunk_0", "b_chunk_0"]

    @pytest.mark.asyncio
    async def test_get_with_no_ids_sends_nothing(self) -> None:
        adapter = _adapter()
        assert await adapter.get([]) == []
        assert adapter.account.clients_opened == 0

    @pytest.mark.asyncio
    async def test_field_values_come_back_in_their_schema_types(self) -> None:
        adapter = _adapter()
        await adapter.upsert(
            [make_record("a", [1.0, 0.0, 0.0, 0.0], mtime=2.0, index=7, pinned=True)]
        )
        stored = adapter.account.databases["holodeck"][adapter.schema.name].documents
        assert stored["a"]["mtime"] == 2  # the service stores 2.0 as 2

        (record,) = await adapter.get(["a"])

        assert record.fields["mtime"] == 2.0
        assert isinstance(record.fields["mtime"], float)
        assert isinstance(record.fields["chunk_index"], int)
        assert record.fields["pinned"] is True

    @pytest.mark.asyncio
    async def test_a_boolean_filter_does_not_match_a_number(self) -> None:
        adapter = await _loaded()
        pinned = await adapter.find(RecordFilter.where(pinned=True))
        assert [r.id for r in pinned] == ["b_chunk_0"]

    @pytest.mark.asyncio
    async def test_list_page_passes_the_continuation_token_as_the_cursor(
        self,
    ) -> None:
        adapter = await _loaded()
        first = await adapter.list_page(limit=2)
        assert len(first.records) == 2
        assert first.next_cursor is not None
        queries_before = len(adapter.account.queries)

        second = await adapter.list_page(cursor=first.next_cursor, limit=2)

        assert len(adapter.account.queries) == queries_before + 1
        assert not {r.id for r in first.records} & {r.id for r in second.records}

        ids = [r.id for r in first.records] + [r.id for r in second.records]
        cursor = second.next_cursor
        while cursor is not None:
            page = await adapter.list_page(cursor=cursor, limit=2)
            ids.extend(r.id for r in page.records)
            cursor = page.next_cursor
        assert sorted(ids) == sorted(r.id for r in CORPUS)

    @pytest.mark.asyncio
    async def test_an_invalid_cursor_is_a_connection_error(self) -> None:
        adapter = await _loaded()
        with pytest.raises(StoreConnectionError):
            await adapter.list_page(cursor="garbage", limit=2)

    @pytest.mark.asyncio
    async def test_a_page_limit_must_be_positive(self) -> None:
        adapter = _adapter()
        with pytest.raises(SchemaError):
            await adapter.list_page(limit=0)

    @pytest.mark.asyncio
    async def test_a_scan_larger_than_one_service_page_is_fully_read(self) -> None:
        adapter = _adapter()
        await adapter.upsert(
            [make_record(f"r{i}", [1.0, 0.0, 0.0, 0.0]) for i in range(250)]
        )
        found = await adapter.find(RecordFilter())
        assert len(found) == 250

    @pytest.mark.asyncio
    async def test_delete_where_deletes_by_the_ids_it_selected(self) -> None:
        adapter = await _loaded()
        removed = await adapter.delete_where(RecordFilter.where(source_path="b.md"))
        assert removed == 2
        text, parameters = adapter.account.queries[-1]
        assert text == 'SELECT c.id FROM c WHERE c["source_path"] = @f0'
        assert parameters == [{"name": "@f0", "value": "b.md"}]
        assert await adapter.count() == len(CORPUS) - 2

    @pytest.mark.asyncio
    async def test_replace_source_removes_encoded_ids_too(self) -> None:
        adapter = _adapter()
        await adapter.upsert(
            [
                make_record("dir/a.md_chunk_0", [1.0, 0.0, 0.0, 0.0]),
                make_record("dir/a.md_chunk_1", [0.0, 1.0, 0.0, 0.0], index=1),
            ]
        )

        await adapter.replace_source(
            RecordFilter.where(source_path="a.md"),
            [make_record("dir/a.md_chunk_0", [0.5, 0.5, 0.0, 0.0], content="new")],
        )

        remaining = await adapter.find(RecordFilter.where(source_path="a.md"))
        assert [r.id for r in remaining] == ["dir/a.md_chunk_0"]
        assert remaining[0].fields["content"] == "new"

    @pytest.mark.asyncio
    async def test_vector_search_clamps_similarity_and_keeps_the_service_order(
        self,
    ) -> None:
        adapter = _adapter()
        await adapter.upsert(
            [
                make_record("near", [1.0, 0.1, 0.0, 0.0]),
                make_record("opposite", [-1.0, 0.0, 0.0, 0.0]),
            ]
        )

        hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], 5))

        assert [h.record.id for h in hits] == ["near", "opposite"]
        assert hits[1].score == 0.0
        assert 0.99 < hits[0].score <= 1.0
        assert all(h.kind is ScoreKind.SIMILARITY for h in hits)
        text, parameters = adapter.account.queries[-1]
        assert text.startswith("SELECT TOP @top ")
        assert {p["name"] for p in parameters} == {"@top", "@vector"}

    @pytest.mark.asyncio
    async def test_vector_search_checks_the_query_before_any_request(self) -> None:
        adapter = _adapter()
        with pytest.raises(SchemaError):
            await adapter.vector_search(VectorQuery([1.0] * 5, 1))
        with pytest.raises(SchemaError, match="non-finite"):
            await adapter.vector_search(VectorQuery([float("inf")] * 4, 1))
        assert await adapter.vector_search(VectorQuery([1.0] * 4, 0)) == []
        assert adapter.account.clients_opened == 0

    @pytest.mark.asyncio
    async def test_hybrid_search_sends_the_rank_query_with_token_parameters(
        self,
    ) -> None:
        adapter = await _loaded(full_text_search=True)

        hits = await adapter.hybrid_search(
            HybridQuery("Zeppelin, zeppelin!", [1.0, 0.0, 0.0, 0.0], 5, "content")
        )

        text, parameters = adapter.account.queries[-1]
        assert "ORDER BY RANK RRF(" in text
        assert 'FullTextScore(c["content"], @t0)' in text
        assert {"name": "@t0", "value": "zeppelin"} in parameters
        assert "zeppelin" not in text
        assert all(h.kind is ScoreKind.FUSED for h in hits)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6) or scores[0] > 0
        assert {"a_chunk_0", "b_chunk_1"} <= {h.record.id for h in hits[:2]}

    @pytest.mark.asyncio
    async def test_hybrid_search_without_tokens_runs_the_vector_leg_alone(
        self,
    ) -> None:
        adapter = await _loaded(full_text_search=True)

        hits = await adapter.hybrid_search(
            HybridQuery("?! -- #", [1.0, 0.0, 0.0, 0.0], 2, "content")
        )

        text, _ = adapter.account.queries[-1]
        assert "RRF" not in text and "FullTextScore" not in text
        assert [h.record.id for h in hits] == ["a_chunk_0", "a_chunk_1"]
        assert all(h.kind is ScoreKind.FUSED for h in hits)

    @pytest.mark.asyncio
    async def test_hybrid_search_applies_the_filter(self) -> None:
        adapter = await _loaded(full_text_search=True)
        hits = await adapter.hybrid_search(
            HybridQuery(
                "alpha",
                [1.0, 0.0, 0.0, 0.0],
                5,
                "content",
                RecordFilter.where(source_path="b.md"),
            )
        )
        assert {h.record.id for h in hits} == {"b_chunk_0", "b_chunk_1"}

    @pytest.mark.asyncio
    async def test_hostile_text_never_reaches_the_query_text(self) -> None:
        adapter = await _loaded(full_text_search=True)

        await adapter.hybrid_search(
            HybridQuery(HOSTILE, [1.0, 0.0, 0.0, 0.0], 2, "content")
        )
        await adapter.find(RecordFilter.where(source_path=HOSTILE))
        await adapter.delete_where(RecordFilter.where(source_path=HOSTILE))

        for text, _ in adapter.account.queries:
            assert "DROP" not in text and "OR 1=1" not in text and "'" not in text


# -- compatibility with an existing container ------------------------------------


class TestExistingContainers:
    async def _existing(
        self, adapter: FakeCosmosAdapter, **changes: object
    ) -> FakeCosmosAdapter:
        await adapter.ensure_collection()
        properties = adapter.account.databases["holodeck"][adapter.schema.name]
        properties.properties.update(changes)
        adapter._verified = False
        return adapter

    @pytest.mark.asyncio
    async def test_wrong_dimensions(self) -> None:
        adapter = _adapter()
        await adapter.ensure_collection()
        container = adapter.account.databases["holodeck"][adapter.schema.name]
        container.properties["vectorEmbeddingPolicy"] = {
            "vectorEmbeddings": [
                {
                    "path": '/"embedding"',
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536,
                }
            ]
        }
        adapter._verified = False

        with pytest.raises(IncompatibleCollectionError, match="1536"):
            await adapter.ensure_collection()
        with pytest.raises(IncompatibleCollectionError, match="1536"):
            await adapter.upsert([make_record("a", [1.0, 0.0, 0.0, 0.0])])

    @pytest.mark.asyncio
    async def test_wrong_distance(self) -> None:
        adapter = _adapter()
        await adapter.ensure_collection()
        container = adapter.account.databases["holodeck"][adapter.schema.name]
        policy = container.properties["vectorEmbeddingPolicy"]
        assert isinstance(policy, dict)
        policy["vectorEmbeddings"][0]["distanceFunction"] = "euclidean"
        adapter._verified = False

        with pytest.raises(IncompatibleCollectionError, match="euclidean"):
            await adapter.count()

    @pytest.mark.asyncio
    async def test_wrong_partition_key(self) -> None:
        adapter = await self._existing(
            _adapter(), partitionKey={"paths": ["/source_path"], "kind": "Hash"}
        )
        with pytest.raises(IncompatibleCollectionError, match="partition key"):
            await adapter.delete(["a"])

    @pytest.mark.asyncio
    async def test_a_container_without_the_vector_path(self) -> None:
        adapter = await self._existing(
            _adapter(),
            vectorEmbeddingPolicy={
                "vectorEmbeddings": [
                    {"path": '/"other"', "dimensions": 4, "distanceFunction": "cosine"}
                ]
            },
        )
        with pytest.raises(IncompatibleCollectionError, match="/embedding"):
            await adapter.count()

    @pytest.mark.asyncio
    async def test_a_service_that_does_not_report_the_policy_is_accepted(
        self,
    ) -> None:
        # The Linux emulator omits vectorEmbeddingPolicy from a container read.
        adapter = _adapter()
        await adapter.ensure_collection()
        container = adapter.account.databases["holodeck"][adapter.schema.name]
        del container.properties["vectorEmbeddingPolicy"]
        adapter._verified = False
        assert await adapter.count() == 0

    @pytest.mark.asyncio
    async def test_full_text_search_needs_a_full_text_index(self) -> None:
        # What a container made by Semantic Kernel looks like.
        plain = _adapter(full_text_search=False)
        await plain.ensure_collection()
        account = plain.account

        hybrid = FakeCosmosAdapter(
            plain.schema, _settings(full_text_search=True), account=account
        )
        with pytest.raises(IncompatibleCollectionError, match="full-text index"):
            await hybrid.hybrid_search(
                HybridQuery("alpha", [1.0, 0.0, 0.0, 0.0], 3, "content")
            )
        with pytest.raises(IncompatibleCollectionError, match="full_text_search"):
            await hybrid.ensure_collection()

    @pytest.mark.asyncio
    async def test_ensure_collection_does_not_touch_an_existing_container(
        self,
    ) -> None:
        adapter = await _loaded()
        policies = dict(
            adapter.account.databases["holodeck"][adapter.schema.name].properties
        )

        await adapter.ensure_collection()

        container = adapter.account.databases["holodeck"][adapter.schema.name]
        assert container.properties == policies
        assert len(container.documents) == len(CORPUS)


# -- errors --------------------------------------------------------------------------


class LeakyAdapter(FakeCosmosAdapter):
    """A client whose errors carry the account key, as a proxy might echo it."""

    async def _connect(self) -> module._Client:
        client = await super()._connect()

        async def explode(*_args: object, **_kwargs: object) -> object:
            raise AzureError(f"401 for AccountKey={KEY} at {URL}")

        client.create_database_if_not_exists = explode  # type: ignore[method-assign,assignment]
        return client


class TestErrors:
    @pytest.mark.asyncio
    async def test_sdk_errors_become_connection_errors_without_the_key(self) -> None:
        adapter = LeakyAdapter(make_schema(), _settings())

        with pytest.raises(StoreConnectionError) as raised:
            await adapter.ensure_collection()

        assert KEY not in str(raised.value)
        assert "***" in str(raised.value)
        assert raised.value.__cause__ is None
        assert "azure-cosmos-nosql" in str(raised.value)

    @pytest.mark.asyncio
    async def test_retrieval_errors_pass_through(self) -> None:
        adapter = _adapter()
        with pytest.raises(SchemaError):
            await adapter.upsert([make_record("", [1.0, 0.0, 0.0, 0.0])])

    @pytest.mark.asyncio
    async def test_a_service_error_that_is_not_missing_data_is_not_swallowed(
        self,
    ) -> None:
        adapter = await _loaded()

        async def boom(*_args: object, **_kwargs: object) -> object:
            raise AzureError("503 service unavailable")

        container = (
            FakeCosmosClient(adapter.account)
            .get_database_client("holodeck")
            .get_container_client(adapter.schema.name)
        )
        assert container is not None
        adapter._verified = True
        original = adapter._handle

        async def broken() -> module._Container:
            handle = await original()
            handle.read_item = boom  # type: ignore[method-assign,assignment]
            return handle

        adapter._handle = broken  # type: ignore[method-assign]
        with pytest.raises(StoreConnectionError, match="503"):
            await adapter.get(["a_chunk_0"])

    @pytest.mark.asyncio
    async def test_not_found_on_delete_is_ignored_but_counted(self) -> None:
        adapter = await _loaded()
        assert await adapter.delete_where(RecordFilter.where(source_path="a.md")) == 2
        assert await adapter.delete_where(RecordFilter.where(source_path="a.md")) == 0

    def test_the_fake_raises_the_sdks_not_found_error(self) -> None:
        assert issubclass(CosmosResourceNotFoundError, AzureError)


def test_the_fake_account_starts_empty() -> None:
    assert FakeAccount().databases == {}


# -- review fixes (042 phase 3) ----------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"chunk_index": 1.9},
        {"chunk_index": "1"},
        {"mtime": "2.25"},
        {"source_path": 3},
        {"pinned": 1},
    ],
)
def test_a_filter_value_that_cannot_equal_a_stored_value_is_rejected(
    bad: dict[str, Any],
) -> None:
    """``where(chunk_index=1.9)`` must not become ``chunk_index = 1``."""
    with pytest.raises(SchemaError, match="another type"):
        _adapter().select_statement(RecordFilter.where(**bad))


def test_an_integral_float_still_filters_an_integer_field() -> None:
    statement = _adapter().select_statement(RecordFilter.where(chunk_index=1.0))
    assert statement.parameters == (("@f0", 1),)
    assert isinstance(statement.parameters[0][1], int)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        make_record("a_chunk_0", [float("nan"), 0.0, 0.0, 0.0]),
        make_record("a" * 1024, [1.0, 0.0, 0.0, 0.0]),
        make_record("/" * 800, [1.0, 0.0, 0.0, 0.0]),
        Record("a_chunk_0", {"chunk_index": "abc"}, [1.0, 0.0, 0.0, 0.0]),
    ],
    ids=["nan-vector", "long-id", "long-encoded-id", "string-in-integer-field"],
)
async def test_replace_source_converts_every_record_before_deleting(
    bad: Record,
) -> None:
    adapter = _adapter()
    await adapter.ensure_collection()
    await adapter.upsert(CORPUS)
    with pytest.raises(SchemaError):
        await adapter.replace_source(RecordFilter.where(source_path="a.md"), [bad])
    kept = await adapter.find(RecordFilter.where(source_path="a.md"))
    assert sorted(record.id for record in kept) == ["a_chunk_0", "a_chunk_1"]
    assert await adapter.count() == len(CORPUS)
