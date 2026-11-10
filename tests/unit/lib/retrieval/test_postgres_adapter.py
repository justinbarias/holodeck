"""Offline tests of the native Postgres adapter: settings, SQL text, type mapping.

Behavior against a server is covered by ``tests/contract/retrieval`` and
``tests/integration/retrieval/test_postgres_compat.py``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest
from psycopg import sql
from pydantic import SecretStr

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    LexicalQuery,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    StoreConnectionError,
    VectorQuery,
    create_adapter,
)
from holodeck.lib.retrieval.adapters import postgres
from holodeck.lib.retrieval.adapters.postgres import (
    PostgresAdapter,
    parse_connection,
)
from holodeck.lib.retrieval.registry import PROVIDERS

EMPTY_ENV: Mapping[str, str] = {}
URL = "postgresql://alice:s3cret@db.example.com:6543/vault?sslmode=require"


def _schema(name: str = "docs", dimensions: int = 3) -> CollectionSchema:
    return CollectionSchema(
        name,
        dimensions,
        (
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def _adapter(
    schema: CollectionSchema | None = None,
    connection: Mapping[str, object] | None = None,
) -> PostgresAdapter:
    return PostgresAdapter.create(schema or _schema(), connection or {})


def _sql(statement: sql.Composable) -> str:
    return statement.as_string()


# -- registry and capabilities -----------------------------


def test_registry_points_at_the_native_adapter() -> None:
    entry = PROVIDERS["postgres"]
    assert entry.module == "holodeck.lib.retrieval.adapters.postgres"
    assert entry.class_name == "PostgresAdapter"
    assert entry.extra == "postgres"
    assert isinstance(create_adapter("postgres", _schema(), {}), PostgresAdapter)


def test_capabilities() -> None:
    capabilities = _adapter().capabilities
    assert capabilities.provider == "postgres"
    assert set(capabilities.supported) == {
        Operation.UPSERT,
        Operation.GET,
        Operation.DELETE,
        Operation.FIND,
        Operation.LIST,
        Operation.DELETE_WHERE,
        Operation.COUNT,
        Operation.VECTOR_SEARCH,
        Operation.LEXICAL_SEARCH,
    }
    assert not capabilities.supports(Operation.HYBRID_SEARCH)


def test_the_adapter_imports_no_sqlalchemy() -> None:
    path = Path(postgres.__file__)
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if name.split(".")[0] == "sqlalchemy"}


# -- connection settings -----------------------------------


def test_connection_string_is_parsed_and_its_port_kept() -> None:
    settings = parse_connection({"connection_string": URL}, EMPTY_ENV)
    assert settings.connect_args["host"] == "db.example.com"
    assert settings.connect_args["port"] == 6543
    assert settings.connect_args["dbname"] == "vault"
    assert settings.connect_args["user"] == "alice"
    assert settings.connect_args["sslmode"] == "require"
    assert settings.db_schema == "public"
    assert (settings.min_pool, settings.max_pool) == (1, 5)


def test_a_secret_string_connection_string_is_accepted() -> None:
    settings = parse_connection({"connection_string": SecretStr(URL)}, EMPTY_ENV)
    assert settings.connect_args["host"] == "db.example.com"


def test_individual_keys_override_the_connection_string() -> None:
    settings = parse_connection(
        {
            "connection_string": URL,
            "host": "other",
            "port": "7000",
            "dbname": "d2",
            "user": "bob",
            "password": SecretStr("pw"),
            "sslmode": "disable",
            "db_schema": "rag",
            "min_pool": 2,
            "max_pool": "9",
        },
        EMPTY_ENV,
    )
    assert (
        dict(settings.connect_args).items()
        >= {
            "host": "other",
            "port": 7000,
            "dbname": "d2",
            "user": "bob",
            "password": "pw",
            "sslmode": "disable",
        }.items()
    )
    assert settings.db_schema == "rag"
    assert (settings.min_pool, settings.max_pool) == (2, 9)


def test_environment_fills_only_what_the_settings_leave_out() -> None:
    env = {
        "POSTGRES_CONNECTION_STRING": "postgresql://u@envhost:1111/envdb",
        "PGSSL_MODE": "require",
        "POSTGRES_MIN_POOL": "3",
        "POSTGRES_MAX_POOL": "4",
    }
    settings = parse_connection({}, env)
    assert settings.connect_args["host"] == "envhost"
    assert settings.connect_args["sslmode"] == "require"
    assert (settings.min_pool, settings.max_pool) == (3, 4)

    explicit = parse_connection({"connection_string": URL}, env)
    assert explicit.connect_args["host"] == "db.example.com"
    assert (
        explicit.connect_args["sslmode"] == "require"
    )  # from the string, not PGSSL_MODE


def test_connect_timeout_is_bounded_unless_set() -> None:
    assert parse_connection({}, EMPTY_ENV).connect_args["connect_timeout"] == 10
    given = parse_connection(
        {"connection_string": "postgresql://h/d?connect_timeout=3"}, EMPTY_ENV
    )
    assert given.connect_args["connect_timeout"] == "3"


def test_unknown_keys_are_ignored_as_before() -> None:
    settings = parse_connection({"collection_name": "x", "whatever": 1}, EMPTY_ENV)
    assert "whatever" not in settings.connect_args


@pytest.mark.parametrize(
    "connection",
    [
        {"port": "not-a-number"},
        {"min_pool": "x"},
        {"min_pool": 5, "max_pool": 2},
        {"max_pool": 0},
        {"connection_string": "postgresql://u:topsecret@h:99999999999/d?bad=="},
    ],
    ids=["port", "min-pool", "pool-order", "max-pool", "dsn"],
)
def test_malformed_settings_raise_without_echoing_values(
    connection: Mapping[str, object],
) -> None:
    with pytest.raises(StoreConnectionError) as raised:
        parse_connection(connection, EMPTY_ENV)
    assert "topsecret" not in str(raised.value)
    assert "not-a-number" not in str(raised.value)


def test_the_password_stays_out_of_repr_and_messages() -> None:
    settings = parse_connection({"connection_string": URL}, EMPTY_ENV)
    assert "s3cret" not in repr(settings)
    assert settings.redact("FATAL: password s3cret rejected") == (
        "FATAL: password *** rejected"
    )
    assert "s3cret" not in settings.target


def test_the_connection_string_and_escaped_passwords_are_masked_too() -> None:
    settings = parse_connection({"connection_string": URL}, EMPTY_ENV)

    masked = settings.redact(f"bad dsn {URL}; password b's3cret\\n' and s3cret")

    assert "s3cret" not in masked
    assert masked.startswith("bad dsn ***;")


@pytest.mark.asyncio
async def test_a_failed_connection_raises_without_the_password() -> None:
    adapter = _adapter(
        connection={
            "host": "127.0.0.1",
            "port": 1,
            "user": "u",
            "password": "hunter2-secret",
            "connect_timeout": 2,
        }
    )
    with pytest.raises(StoreConnectionError) as raised:
        await adapter.count()
    assert "hunter2-secret" not in str(raised.value)
    assert "postgres" in str(raised.value)
    await adapter.close()
    await adapter.close()


# -- schema checks -----------------------------------------


def test_more_than_2000_dimensions_are_refused_like_semantic_kernel_did() -> None:
    assert _adapter(_schema(dimensions=2000))
    with pytest.raises(SchemaError, match="2000"):
        _adapter(_schema(dimensions=2001))


@pytest.mark.parametrize(
    "name", ["", "x" * 64, "bad\x00name"], ids=["empty", "too-long", "nul"]
)
def test_unstorable_names_are_refused(name: str) -> None:
    with pytest.raises(SchemaError):
        _adapter(CollectionSchema(name or "n", 3, (FieldSpec(name or "\x00"),)))
    if name:
        with pytest.raises(SchemaError):
            _adapter(CollectionSchema(name, 3, (FieldSpec("f"),)))
    with pytest.raises(SchemaError):
        _adapter(connection={"db_schema": "x" * 64})


def test_an_empty_db_schema_means_public() -> None:
    adapter = _adapter(connection={"db_schema": ""})
    assert _sql(adapter._table) == '"public"."docs"'


# -- SQL text ----------------------------------------------


def test_create_table_matches_the_layout_semantic_kernel_creates() -> None:
    adapter = _adapter()
    assert _sql(adapter.create_table_statement()) == (
        'CREATE TABLE IF NOT EXISTS "public"."docs" ('
        '"id" TEXT PRIMARY KEY, "source_path" TEXT, "chunk_index" INTEGER, '
        '"content" TEXT, "mtime" DOUBLE PRECISION, "pinned" BOOLEAN, '
        '"embedding" VECTOR(3))'
    )
    assert _sql(adapter.create_index_statement()) == (
        'CREATE INDEX IF NOT EXISTS "docs_embedding_idx" ON "public"."docs" '
        'USING hnsw ("embedding" vector_cosine_ops)'
    )


def test_a_non_default_db_schema_qualifies_every_statement() -> None:
    adapter = _adapter(connection={"db_schema": "rag"})
    assert '"rag"."docs"' in _sql(adapter.create_table_statement())
    assert '"rag"."docs"' in _sql(adapter.create_index_statement())
    assert '"rag"."docs"' in _sql(adapter.upsert_statement())


def test_upsert_is_insert_on_conflict_do_update() -> None:
    assert _sql(_adapter().upsert_statement()) == (
        'INSERT INTO "public"."docs" ("id", "source_path", "chunk_index", "content", '
        '"mtime", "pinned", "embedding") VALUES (%s, %s, %s, %s, %s, %s, %s) '
        'ON CONFLICT ("id") DO UPDATE SET "source_path" = EXCLUDED."source_path", '
        '"chunk_index" = EXCLUDED."chunk_index", "content" = EXCLUDED."content", '
        '"mtime" = EXCLUDED."mtime", "pinned" = EXCLUDED."pinned", '
        '"embedding" = EXCLUDED."embedding"'
    )


HOSTILE = 'x"; DROP TABLE users; --'


def test_hostile_identifiers_are_quoted_not_executed() -> None:
    schema = CollectionSchema(
        HOSTILE, 3, (FieldSpec('a"b'), FieldSpec("semi;colon", FieldType.INTEGER))
    )
    adapter = _adapter(schema, {"db_schema": 'sch"ema'})
    quoted_table = '"sch""ema"."x""; DROP TABLE users; --"'
    for statement in (
        adapter.create_table_statement(),
        adapter.create_index_statement(),
        adapter.upsert_statement(),
        adapter.select_statement(RecordFilter.where(**{'a"b': "v"}))[0],
    ):
        text = _sql(statement)
        assert quoted_table in text
        # Every quote inside a name is doubled, so no name can end early.
        assert text.replace(quoted_table, "").count('"') % 2 == 0
    create = _sql(adapter.create_table_statement())
    assert '"a""b" TEXT' in create
    assert '"semi;colon" INTEGER' in create


def test_hostile_values_travel_as_parameters() -> None:
    adapter = _adapter()
    statement, params = adapter.select_statement(
        RecordFilter.where(source_path=HOSTILE, chunk_index=2), limit=7
    )
    text = _sql(statement)
    assert "DROP" not in text
    assert text == (
        'SELECT "id", "source_path", "chunk_index", "content", "mtime", "pinned" '
        'FROM "public"."docs" WHERE "source_path" = %s AND "chunk_index" = %s '
        'ORDER BY "id" LIMIT %s'
    )
    assert params == [HOSTILE, 2, 7]


def test_a_none_filter_value_becomes_is_null() -> None:
    statement, params = _adapter().select_statement(
        RecordFilter.where(source_path=None)
    )
    assert 'WHERE "source_path" IS NULL' in _sql(statement)
    assert params == []


def test_keyset_paging_uses_the_cursor_as_a_parameter() -> None:
    statement, params = _adapter().select_statement(None, after=HOSTILE, limit=3)
    assert 'WHERE "id" > %s ORDER BY "id" LIMIT %s' in _sql(statement)
    assert params == [HOSTILE, 3]


def test_an_unknown_filter_field_is_a_schema_error() -> None:
    with pytest.raises(SchemaError):
        _adapter().select_statement(RecordFilter.where(**{HOSTILE: 1}))


def test_vector_search_orders_by_cosine_distance_with_the_filter_pushed_down() -> None:
    statement, params = _adapter().vector_statement(
        VectorQuery([1.0, 0.0, 0.0], 4, filter=RecordFilter.where(source_path=HOSTILE))
    )
    text = _sql(statement)
    assert '"embedding" <=> %s AS "__holodeck_distance"' in text
    assert 'WHERE "source_path" = %s ORDER BY "__holodeck_distance" LIMIT %s' in text
    assert "DROP" not in text
    assert len(params) == 3
    assert params[1:] == [HOSTILE, 4]


def test_lexical_search_pushes_filter_and_tokens_as_parameters() -> None:
    adapter = _adapter()
    statement, params = adapter.lexical_statement(
        LexicalQuery(
            "x'); DROP", 5, "content", filter=RecordFilter.where(source_path="a")
        ),
        "x | drop",
    )
    text = _sql(statement)
    assert "to_tsvector('simple', \"content\") @@ to_tsquery('simple', %s)" in text
    assert 'AND "source_path" = %s' in text
    assert text.endswith('ORDER BY "__holodeck_rank" DESC, "id" LIMIT %s')
    assert params == ["x | drop", "x | drop", "a", 5]


def test_lexical_search_needs_a_text_field() -> None:
    adapter = _adapter()
    with pytest.raises(SchemaError):
        adapter.lexical_statement(LexicalQuery("a", 1, "chunk_index"), "a")
    with pytest.raises(SchemaError):
        adapter.lexical_statement(LexicalQuery("a", 1, "nope"), "a")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Zeppelin hangar", "zeppelin | hangar"),
        ("a a b", "a | b"),
        ("O'Brien's  cat!", "o | brien | s | cat"),
        ("?!?", ""),
        ("naïve café", "naïve | café"),
    ],
)
def test_lexical_query_ors_distinct_word_tokens(text: str, expected: str) -> None:
    assert postgres._lexical_query(text) == expected


# -- type mapping ------------------------------------------


@pytest.mark.parametrize(
    ("field_type", "ddl"),
    [
        (FieldType.STRING, "TEXT"),
        (FieldType.INTEGER, "INTEGER"),
        (FieldType.FLOAT, "DOUBLE PRECISION"),
        (FieldType.BOOLEAN, "BOOLEAN"),
    ],
)
def test_field_types_map_to_the_semantic_kernel_column_types(
    field_type: FieldType, ddl: str
) -> None:
    adapter = _adapter(CollectionSchema("t", 2, (FieldSpec("f", field_type),)))
    assert f'"f" {ddl}, "embedding" VECTOR(2)' in _sql(adapter.create_table_statement())


def test_values_are_checked_against_the_field_type_on_write() -> None:
    string, integer, number, flag = (
        _schema().fields[0],
        _schema().fields[1],
        _schema().fields[3],
        _schema().fields[4],
    )
    assert postgres._write_value(string, "x") == "x"
    assert postgres._write_value(integer, 3) == 3
    assert postgres._write_value(number, 2) == 2.0
    assert isinstance(postgres._write_value(number, 2), float)
    assert postgres._write_value(flag, True) is True
    assert postgres._write_value(integer, None) is None
    for spec, bad in (
        (string, 1),
        (integer, "1"),
        (integer, True),
        (integer, 1.5),
        (number, True),
        (number, "1.0"),
        (flag, 1),
    ):
        with pytest.raises(SchemaError):
            postgres._write_value(spec, bad)


def test_column_values_are_read_as_the_field_type() -> None:
    string, integer, number, flag = (
        _schema().fields[0],
        _schema().fields[1],
        _schema().fields[3],
        _schema().fields[4],
    )
    assert postgres._read_value(string, "x") == "x"
    assert postgres._read_value(string, 5) == "5"
    assert postgres._read_value(integer, 7) == 7
    assert postgres._read_value(integer, Decimal("7")) == 7
    assert postgres._read_value(number, Decimal("1.5")) == 1.5
    assert postgres._read_value(flag, True) is True
    assert postgres._read_value(integer, None) is None


def test_missing_fields_are_written_as_the_type_default() -> None:
    adapter = _adapter()
    row = adapter._row(Record("r", {"source_path": "a.md"}, [1.0, 0.0, 0.0]))
    assert row[:6] == ["r", "a.md", 0, "", 0.0, False]
    assert row[6].to_list() == [1.0, 0.0, 0.0]


def test_number_filters_follow_the_field_type() -> None:
    adapter = _adapter()

    _, params = adapter.select_statement(RecordFilter.where(mtime=2, chunk_index=3.0))

    assert params == [2.0, 3]
    assert [type(value) for value in params] == [float, int]
    with pytest.raises(SchemaError):
        adapter.select_statement(RecordFilter.where(chunk_index=True))
