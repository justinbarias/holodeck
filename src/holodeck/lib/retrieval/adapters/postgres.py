"""Native PostgreSQL adapter: psycopg 3 (async) and the pgvector extension.

Stored layout
-------------
The adapter reads and writes the table that Semantic Kernel's
``PostgresCollection`` created for the same collection, and creates the same
table on a fresh database. For schema ``public`` and collection ``name``::

    CREATE TABLE public."name" (
        "id" TEXT PRIMARY KEY,
        "<field>" TEXT | INTEGER | DOUBLE PRECISION | BOOLEAN,   -- schema order
        "embedding" VECTOR(<dimensions>)
    );
    CREATE INDEX "name_embedding_idx" ON public."name"
        USING hnsw ("embedding" vector_cosine_ops);

No other index exists. Table and column names are quoted, so case is kept.
SK refuses more than 2000 dimensions (the pgvector HNSW limit); so does this
adapter.

Differences from the SK path, all deliberate:

* the connection string's port is honoured. SK's ``PostgresSettings`` replaces
  it with its default, 5432;
* ``host``, ``port``, ``dbname``, ``user``, ``password`` and ``sslmode`` in the
  database settings are used. SK ignored them (``PostgresCollection`` swallows
  unknown keyword arguments);
* the ``PG*`` environment variables only fill what the settings leave out, as
  libpq does. SK let them override the connection string;
* ``ensure_collection`` creates table and index in one transaction and runs
  ``CREATE EXTENSION IF NOT EXISTS vector`` when the extension is missing;
* cosine scores are similarities. SK's cosine search returns the raw distance.

Connection settings
-------------------
``connection_string`` (or ``POSTGRES_CONNECTION_STRING``), ``host``, ``port``,
``dbname``, ``user``, ``password``, ``sslmode`` (or ``PGSSL_MODE``),
``db_schema`` (default ``public``), ``min_pool`` and ``max_pool`` (or
``POSTGRES_MIN_POOL`` and ``POSTGRES_MAX_POOL``; default 1 and 5).
``collection_name`` is applied by the schema builders. Unknown keys are
ignored, as they were on the SK path.

Lifecycle
---------
One ``AsyncConnectionPool`` per adapter, opened on first use and closed by
``close``. The adapter can be used again after ``close``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TypeVar

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, errors, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import SecretStr

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    IncompatibleCollectionError,
    SchemaError,
    StoreConnectionError,
)
from holodeck.lib.retrieval.redaction import redact, secret_values
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    FieldSpec,
    FieldType,
    FieldValue,
    LexicalQuery,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
    checked_filter_value,
    clamp_similarity,
)

logger = logging.getLogger(__name__)

PROVIDER = "postgres"
DEFAULT_DB_SCHEMA = "public"
MAX_DIMENSIONS = 2000
"""pgvector's limit for HNSW indexes on ``vector`` columns."""
MAX_IDENTIFIER_BYTES = 63
RESERVED_PREFIX = "__holodeck_"
"""Prefix of the computed columns of a search; no schema field may start with it."""
DISTANCE_ALIAS = RESERVED_PREFIX + "distance"
RANK_ALIAS = RESERVED_PREFIX + "rank"
UPSERT_BATCH_SIZE = 1000
DEFAULT_MIN_POOL = 1
DEFAULT_MAX_POOL = 5
CONNECT_TIMEOUT_SECONDS = 10
POOL_TIMEOUT_SECONDS = 30.0
MAX_EF_SEARCH = 1000
MIN_EF_SEARCH = 40
LEXICAL_CONFIG = "simple"

_T = TypeVar("_T")
_ConnectArgs = dict[str, str | int]

_CAPABILITIES = Capabilities(
    provider=PROVIDER,
    supported=frozenset(
        {
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
    ),
    hints={
        Operation.HYBRID_SEARCH: "HoloDeck fuses vector and BM25 results itself.",
    },
)

_PG_TYPES: dict[FieldType, sql.SQL] = {
    FieldType.STRING: sql.SQL("TEXT"),
    FieldType.INTEGER: sql.SQL("INTEGER"),
    FieldType.FLOAT: sql.SQL("DOUBLE PRECISION"),
    FieldType.BOOLEAN: sql.SQL("BOOLEAN"),
}

_DEFAULTS: dict[FieldType, FieldValue] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

# format_type() names a column may have and still hold a field of that type.
_COMPATIBLE_TYPES: dict[FieldType, frozenset[str]] = {
    FieldType.STRING: frozenset({"text", "character varying"}),
    FieldType.INTEGER: frozenset({"smallint", "integer", "bigint"}),
    FieldType.FLOAT: frozenset({"real", "double precision", "numeric"}),
    FieldType.BOOLEAN: frozenset({"boolean"}),
}

_VECTOR_TYPE = re.compile(r"^vector\((\d+)\)$")
_WORD = re.compile(r"\w+", re.UNICODE)


# -- connection settings ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PostgresConnectionSettings:
    """Resolved connection settings.

    ``connect_args`` are libpq keyword arguments and include the password, so
    they stay out of ``repr``.
    """

    connect_args: Mapping[str, str | int] = field(repr=False)
    db_schema: str = DEFAULT_DB_SCHEMA
    min_pool: int = DEFAULT_MIN_POOL
    max_pool: int = DEFAULT_MAX_POOL
    dsn: str | None = field(default=None, repr=False)

    def redact(self, text: str) -> str:
        """Return ``text`` with the password and the connection string removed."""
        return redact(
            text, secret_values(self.dsn, str(self.connect_args.get("password", "")))
        )

    @property
    def target(self) -> str:
        """Return ``host:port/dbname`` for messages. It holds no secret."""
        host = self.connect_args.get("host", "localhost")
        port = self.connect_args.get("port", 5432)
        dbname = self.connect_args.get("dbname", "")
        return f"{host}:{port}/{dbname}"


def _text(value: object) -> str | None:
    """Return a non-empty string from a setting, unwrapping ``SecretStr``."""
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _positive_int(name: str, value: str | None, default: int, minimum: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        raise StoreConnectionError(PROVIDER, f"'{name}' must be an integer.") from None
    if number < minimum:
        raise StoreConnectionError(PROVIDER, f"'{name}' must be at least {minimum}.")
    return number


def parse_connection(
    connection: Mapping[str, object], environ: Mapping[str, str] | None = None
) -> PostgresConnectionSettings:
    """Resolve the database settings of a tool into connection settings.

    Raises:
        StoreConnectionError: A setting is malformed. The message never
            repeats a setting value.
    """
    env = os.environ if environ is None else environ
    args: _ConnectArgs = {}

    dsn = _text(connection.get("connection_string")) or env.get(
        "POSTGRES_CONNECTION_STRING"
    )
    if dsn:
        try:
            parsed = conninfo_to_dict(dsn)
        except psycopg.ProgrammingError:
            raise StoreConnectionError(PROVIDER, "invalid connection string.") from None
        for key, value in parsed.items():
            args[key] = value if isinstance(value, str | int) else str(value)

    for key in ("host", "port", "dbname", "user", "password", "sslmode"):
        value = _text(connection.get(key))
        if value is not None:
            args[key] = value
    if "port" in args:
        try:
            args["port"] = int(args["port"])
        except ValueError:
            raise StoreConnectionError(PROVIDER, "'port' must be an integer.") from None
    if "sslmode" not in args and env.get("PGSSL_MODE"):
        args["sslmode"] = env["PGSSL_MODE"]
    args.setdefault("connect_timeout", CONNECT_TIMEOUT_SECONDS)

    min_pool = _positive_int(
        "min_pool",
        _text(connection.get("min_pool")) or env.get("POSTGRES_MIN_POOL"),
        DEFAULT_MIN_POOL,
        0,
    )
    max_pool = _positive_int(
        "max_pool",
        _text(connection.get("max_pool")) or env.get("POSTGRES_MAX_POOL"),
        DEFAULT_MAX_POOL,
        1,
    )
    if max_pool < min_pool:
        raise StoreConnectionError(PROVIDER, "'max_pool' is below 'min_pool'.")
    return PostgresConnectionSettings(
        connect_args=args,
        db_schema=_text(connection.get("db_schema")) or DEFAULT_DB_SCHEMA,
        min_pool=min_pool,
        max_pool=max_pool,
        dsn=dsn or None,
    )


# -- value conversion -------------------------------------------------------------


def _write_value(spec: FieldSpec, value: FieldValue) -> FieldValue:
    """Return ``value`` as the column stores it, or raise ``SchemaError``."""
    if value is None:
        return None
    if spec.type is FieldType.STRING and isinstance(value, str):
        return value
    if spec.type is FieldType.BOOLEAN and isinstance(value, bool):
        return value
    if not isinstance(value, bool):
        if spec.type is FieldType.INTEGER and isinstance(value, int):
            return value
        if spec.type is FieldType.FLOAT and isinstance(value, int | float):
            return float(value)
    raise SchemaError(
        f"Field '{spec.name}' is {spec.type.value}; got {type(value).__name__}."
    )


def _read_value(spec: FieldSpec, value: object) -> FieldValue:
    """Return a column value as a ``FieldValue`` of the field's type."""
    if value is None:
        return None
    if spec.type is FieldType.BOOLEAN:
        return bool(value)
    if isinstance(value, bool | int | float | Decimal | str):
        if spec.type is FieldType.INTEGER:
            return int(value)
        if spec.type is FieldType.FLOAT:
            return float(value)
    return value if isinstance(value, str) else str(value)


def _vector_list(value: object) -> list[float]:
    if isinstance(value, Vector):
        return [float(item) for item in value.to_list()]
    if isinstance(value, str):
        return [float(item) for item in Vector.from_text(value).to_list()]
    raise SchemaError(f"Unexpected vector value of type {type(value).__name__}.")


def _lexical_query(text: str) -> str:
    """Return a ``to_tsquery`` string that ORs the distinct word tokens."""
    seen: dict[str, None] = {}
    for token in _WORD.findall(text.lower()):
        seen.setdefault(token)
    return " | ".join(seen)


def _check_identifier(kind: str, name: str) -> None:
    if not name or "\x00" in name:
        raise SchemaError(f"Invalid {kind} name: {name!r}.")
    if len(name.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise SchemaError(
            f"The {kind} name '{name}' is longer than {MAX_IDENTIFIER_BYTES} bytes; "
            "PostgreSQL would truncate it."
        )


# -- adapter ----------------------------------------------------------------------


class PostgresAdapter(BaseVectorStoreAdapter):
    """One PostgreSQL table with a pgvector column."""

    def __init__(
        self, schema: CollectionSchema, settings: PostgresConnectionSettings
    ) -> None:
        super().__init__(schema, _CAPABILITIES)
        if schema.dimensions > MAX_DIMENSIONS:
            raise SchemaError(
                f"Dimensionality of {schema.dimensions} exceeds the maximum of "
                f"{MAX_DIMENSIONS} that a pgvector HNSW index supports."
            )
        _check_identifier("schema", settings.db_schema)
        _check_identifier("collection", schema.name)
        for name in (schema.key_field, schema.vector_field, *schema.field_names):
            _check_identifier("column", name)
        for name in schema.field_names:
            if name.startswith(RESERVED_PREFIX):
                raise SchemaError(
                    f"Field name '{name}' starts with '{RESERVED_PREFIX}', which "
                    "the Postgres adapter reserves for computed columns."
                )
        self._settings = settings
        self._pool: AsyncConnectionPool[AsyncConnection[TupleRow]] | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> PostgresAdapter:
        """Build the adapter. Nothing connects until the first operation."""
        return cls(schema, parse_connection(connection))

    # -- names ---------------------------------------------------------------

    @property
    def _table(self) -> sql.Identifier:
        return sql.Identifier(self._settings.db_schema, self._schema.name)

    @property
    def _key(self) -> sql.Identifier:
        return sql.Identifier(self._schema.key_field)

    @property
    def _vector(self) -> sql.Identifier:
        return sql.Identifier(self._schema.vector_field)

    def _columns(self, *, vector: bool = False) -> sql.Composable:
        names = [self._schema.key_field, *self._schema.field_names]
        if vector:
            names.append(self._schema.vector_field)
        return sql.SQL(", ").join(sql.Identifier(name) for name in names)

    # -- statements ------------------------------------------------------------

    def create_table_statement(self) -> sql.Composed:
        """Return ``CREATE TABLE IF NOT EXISTS`` for the collection."""
        columns: list[sql.Composable] = [
            sql.SQL("{} TEXT PRIMARY KEY").format(self._key)
        ]
        columns.extend(
            sql.SQL("{} {}").format(sql.Identifier(spec.name), _PG_TYPES[spec.type])
            for spec in self._schema.fields
        )
        columns.append(
            sql.SQL("{} VECTOR({})").format(
                self._vector, sql.Literal(self._schema.dimensions)
            )
        )
        return sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            self._table, sql.SQL(", ").join(columns)
        )

    def create_index_statement(self) -> sql.Composed:
        """Return ``CREATE INDEX IF NOT EXISTS`` for the vector index."""
        index = f"{self._schema.name}_{self._schema.vector_field}_idx"
        return sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw ({} vector_cosine_ops)"
        ).format(sql.Identifier(index), self._table, self._vector)

    def upsert_statement(self) -> sql.Composed:
        """Return the ``INSERT ... ON CONFLICT`` statement for one record."""
        names = [self._schema.key_field, *self._schema.field_names]
        names.append(self._schema.vector_field)
        updates = sql.SQL(", ").join(
            sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(name))
            for name in names[1:]
        )
        return sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}"
        ).format(
            self._table,
            sql.SQL(", ").join(sql.Identifier(name) for name in names),
            sql.SQL(", ").join(sql.Placeholder() * len(names)),
            self._key,
            updates,
        )

    def _where(
        self, record_filter: RecordFilter | None
    ) -> tuple[list[sql.Composable], list[FieldValue]]:
        """Return the conditions and parameters for an exact-match filter."""
        conditions: list[sql.Composable] = []
        params: list[FieldValue] = []
        if record_filter is None:
            return conditions, params
        for clause in record_filter.all_of:
            spec = self._schema.field(clause.field)
            value = _write_value(spec, checked_filter_value(spec, clause.value))
            if value is None:
                conditions.append(
                    sql.SQL("{} IS NULL").format(sql.Identifier(spec.name))
                )
            else:
                conditions.append(sql.SQL("{} = %s").format(sql.Identifier(spec.name)))
                params.append(value)
        return conditions, params

    @staticmethod
    def _where_sql(conditions: Sequence[sql.Composable]) -> sql.Composable:
        if not conditions:
            return sql.SQL("")
        return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

    def select_statement(
        self,
        record_filter: RecordFilter | None = None,
        *,
        after: str | None = None,
        limit: int | None = None,
    ) -> tuple[sql.Composed, list[FieldValue]]:
        """Return a filtered, key-ordered ``SELECT`` and its parameters."""
        conditions, params = self._where(record_filter)
        if after is not None:
            conditions.append(sql.SQL("{} > %s").format(self._key))
            params.append(after)
        query = sql.SQL("SELECT {} FROM {}{} ORDER BY {}").format(
            self._columns(), self._table, self._where_sql(conditions), self._key
        )
        if limit is not None:
            query += sql.SQL(" LIMIT %s")
            params.append(limit)
        return query, params

    def vector_statement(self, query: VectorQuery) -> tuple[sql.Composed, list[object]]:
        """Return the nearest-neighbour ``SELECT`` and its parameters.

        The last selected column is the cosine distance. ``ORDER BY`` uses its
        alias, which keeps the HNSW index usable. The alias starts with the
        reserved prefix, so it cannot match a field name.
        """
        conditions, filter_params = self._where(query.filter)
        alias = sql.Identifier(DISTANCE_ALIAS)
        statement = sql.SQL(
            "SELECT {cols}, {vector} <=> %s AS {alias} FROM {table}{where} "
            "ORDER BY {alias} LIMIT %s"
        ).format(
            cols=self._columns(),
            vector=self._vector,
            alias=alias,
            table=self._table,
            where=self._where_sql(conditions),
        )
        params: list[object] = [
            Vector([float(value) for value in query.vector]),
            *filter_params,
            query.top_k,
        ]
        return statement, params

    def lexical_statement(
        self, query: LexicalQuery, tsquery: str
    ) -> tuple[sql.Composed, list[object]]:
        """Return the full-text ``SELECT`` and its parameters.

        The last selected column is the rank.
        """
        spec = self._schema.field(query.text_field)
        if spec.type is not FieldType.STRING:
            raise SchemaError(f"Field '{spec.name}' is not a text field.")
        column = sql.Identifier(spec.name)
        conditions, filter_params = self._where(query.filter)
        conditions.insert(
            0,
            sql.SQL("to_tsvector({}, {}) @@ to_tsquery({}, %s)").format(
                sql.Literal(LEXICAL_CONFIG), column, sql.Literal(LEXICAL_CONFIG)
            ),
        )
        statement = sql.SQL(
            "SELECT {cols}, ts_rank(to_tsvector({cfg}, {col}), "
            "to_tsquery({cfg}, %s)) AS {alias} FROM {table}{where} "
            "ORDER BY {alias} DESC, {key} LIMIT %s"
        ).format(
            alias=sql.Identifier(RANK_ALIAS),
            cols=self._columns(),
            cfg=sql.Literal(LEXICAL_CONFIG),
            col=column,
            table=self._table,
            where=self._where_sql(conditions),
            key=self._key,
        )
        params: list[object] = [tsquery, tsquery, *filter_params, query.top_k]
        return statement, params

    # -- pool ------------------------------------------------------------------

    async def _ensure_pool(self) -> AsyncConnectionPool[AsyncConnection[TupleRow]]:
        async with self._lock:
            if self._pool is None:
                self._pool = await self._open_pool()
            return self._pool

    async def _open_pool(self) -> AsyncConnectionPool[AsyncConnection[TupleRow]]:
        settings = self._settings
        conninfo = make_conninfo("", **settings.connect_args)
        try:
            # Fail fast with the server's own reason; a pool only reports
            # that it timed out.
            probe = await AsyncConnection.connect(conninfo)
        except psycopg.Error as error:
            raise StoreConnectionError(
                PROVIDER,
                f"could not connect to {settings.target}: "
                f"{settings.redact(str(error).strip())}",
            ) from None
        await probe.close()
        pool: AsyncConnectionPool[AsyncConnection[TupleRow]] = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=settings.min_pool,
            max_size=settings.max_pool,
            timeout=POOL_TIMEOUT_SECONDS,
            open=False,
            configure=_configure,
        )
        try:
            await pool.open(wait=True, timeout=POOL_TIMEOUT_SECONDS)
        except BaseException as error:
            # Any unsuccessful open, a cancellation included, must not leave
            # the pool's worker tasks running with nothing to close them.
            await asyncio.shield(pool.close())
            if isinstance(error, PoolTimeout):
                raise StoreConnectionError(
                    PROVIDER, f"timed out opening a pool to {settings.target}."
                ) from None
            raise
        return pool

    async def _discard_pool(self) -> None:
        async with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection[TupleRow]]:
        pool = await self._ensure_pool()
        async with pool.connection() as connection:
            yield connection

    async def close(self) -> None:
        await self._discard_pool()

    async def _after_create(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``operation``; create the collection and retry if it is absent."""
        try:
            return await operation()
        except errors.UndefinedTable:
            await self.ensure_collection()
            return await operation()

    # -- collection ----------------------------------------------------------------

    async def collection_exists(self) -> bool:
        async with self._connection() as connection:
            cursor = await connection.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (self._settings.db_schema, self._schema.name),
            )
            return await cursor.fetchone() is not None

    async def ensure_collection(self) -> None:
        extension_created = False
        async with self._connection() as connection:
            extension_created = await self._ensure_extension(connection)
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{self._settings.db_schema}.{self._schema.name}",),
                )
                await connection.execute(self.create_table_statement())
                await self._check_layout(connection)
                await connection.execute(self.create_index_statement())
        if extension_created:
            # Pooled connections opened before the extension existed have no
            # vector adapters.
            await self._discard_pool()

    async def _ensure_extension(self, connection: AsyncConnection[TupleRow]) -> bool:
        cursor = await connection.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        if await cursor.fetchone() is not None:
            return False
        try:
            async with connection.transaction():
                await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error as error:
            raise StoreConnectionError(
                PROVIDER,
                "the pgvector extension is not installed in this database and "
                "could not be created "
                f"({self._settings.redact(str(error).strip())}). "
                "Run CREATE EXTENSION vector as a superuser.",
            ) from None
        return True

    async def _check_layout(self, connection: AsyncConnection[TupleRow]) -> None:
        """Raise ``IncompatibleCollectionError`` if the table cannot hold the schema."""
        cursor = await connection.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s "
            "AND a.attnum > 0 AND NOT a.attisdropped",
            (self._settings.db_schema, self._schema.name),
        )
        existing: dict[str, str] = {
            str(name): str(type_name) for name, type_name in await cursor.fetchall()
        }
        problems: list[str] = []
        if self._schema.key_field not in existing:
            problems.append(f"missing key column '{self._schema.key_field}'")
        for spec in self._schema.fields:
            found = existing.get(spec.name)
            if found is None:
                problems.append(f"missing column '{spec.name}'")
            elif found.split("(")[0] not in _COMPATIBLE_TYPES[spec.type]:
                problems.append(
                    f"column '{spec.name}' is {found}, not {spec.type.value}"
                )
        found_vector = existing.get(self._schema.vector_field)
        match = _VECTOR_TYPE.match(found_vector) if found_vector else None
        if match is None:
            problems.append(f"'{self._schema.vector_field}' is not a vector(N) column")
        elif int(match.group(1)) != self._schema.dimensions:
            problems.append(
                f"'{self._schema.vector_field}' has {match.group(1)} dimensions, "
                f"the schema needs {self._schema.dimensions}"
            )
        if problems:
            raise IncompatibleCollectionError(
                f"Table {self._settings.db_schema}.{self._schema.name} does not match "
                f"the collection schema: {'; '.join(problems)}."
            )

    async def delete_collection(self) -> None:
        async with self._connection() as connection:
            await connection.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(self._table)
            )

    # -- reads -----------------------------------------------------------------------

    def _record(self, row: Sequence[object], *, vector: bool = False) -> Record:
        fields = {
            spec.name: _read_value(spec, row[index])
            for index, spec in enumerate(self._schema.fields, start=1)
        }
        return Record(
            id=str(row[0]),
            fields=fields,
            vector=(
                _vector_list(row[len(self._schema.fields) + 1])
                if vector and row[len(self._schema.fields) + 1] is not None
                else None
            ),
        )

    async def count(self) -> int:
        try:
            async with self._connection() as connection:
                cursor = await connection.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(self._table)
                )
                row = await cursor.fetchone()
        except errors.UndefinedTable:
            return 0
        return int(row[0]) if row else 0

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        if not ids:
            return []
        statement = sql.SQL("SELECT {} FROM {} WHERE {} = ANY(%s)").format(
            self._columns(vector=include_vectors), self._table, self._key
        )
        try:
            async with self._connection() as connection:
                cursor = await connection.execute(statement, (list(ids),))
                rows = await cursor.fetchall()
        except errors.UndefinedTable:
            return []
        by_id = {
            record.id: record
            for record in (self._record(row, vector=include_vectors) for row in rows)
        }
        return [by_id[key] for key in ids if key in by_id]

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        statement, params = self.select_statement(record_filter, limit=limit)
        try:
            async with self._connection() as connection:
                cursor = await connection.execute(statement, params)
                rows = await cursor.fetchall()
        except errors.UndefinedTable:
            return []
        return [self._record(row) for row in rows]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        """Return one page ordered by key. The cursor is the last key returned."""
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        statement, params = self.select_statement(None, after=cursor, limit=limit + 1)
        try:
            async with self._connection() as connection:
                result = await connection.execute(statement, params)
                rows = await result.fetchall()
        except errors.UndefinedTable:
            return Page(records=())
        records = [self._record(row) for row in rows[:limit]]
        more = len(rows) > limit
        return Page(
            records=tuple(records), next_cursor=records[-1].id if more else None
        )

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        if len(query.vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(query.vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )
        if query.top_k <= 0:
            return []
        statement, params = self.vector_statement(query)
        try:
            async with self._connection() as connection, connection.transaction():
                await self._tune_index_scan(connection, query.top_k)
                cursor = await connection.execute(statement, params)
                rows = await cursor.fetchall()
        except errors.UndefinedTable:
            return []
        hits: list[ScoredRecord] = []
        for row in rows:
            distance = float(row[-1]) if isinstance(row[-1], int | float) else math.nan
            similarity = (
                0.0 if math.isnan(distance) else clamp_similarity(1.0 - distance)
            )
            hits.append(
                ScoredRecord(self._record(row), similarity, ScoreKind.SIMILARITY)
            )
        return hits

    @staticmethod
    async def _tune_index_scan(
        connection: AsyncConnection[TupleRow], top_k: int
    ) -> None:
        """Let the HNSW scan return ``top_k`` rows, and filtered ones.

        pgvector stops an HNSW scan after ``hnsw.ef_search`` (40) candidates
        and applies the filter afterwards. A larger ``top_k`` or a selective
        filter then returns too few rows. The settings last for the
        transaction. A server without ``hnsw.iterative_scan`` (pgvector below
        0.8) keeps its defaults.
        """
        ef_search = min(MAX_EF_SEARCH, max(MIN_EF_SEARCH, top_k))
        try:
            async with connection.transaction():
                await connection.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true), "
                    "set_config('hnsw.iterative_scan', 'strict_order', true)",
                    (str(ef_search),),
                )
        except psycopg.Error:
            logger.debug("pgvector index scan settings are not available.")

    async def lexical_search(self, query: LexicalQuery) -> list[ScoredRecord]:
        tsquery = _lexical_query(query.text)
        statement, params = self.lexical_statement(query, tsquery)
        if not tsquery or query.top_k <= 0:
            return []
        try:
            async with self._connection() as connection:
                cursor = await connection.execute(statement, params)
                rows = await cursor.fetchall()
        except errors.UndefinedTable:
            return []
        return [
            ScoredRecord(
                self._record(row),
                float(row[-1]) if isinstance(row[-1], int | float) else 0.0,
                ScoreKind.LEXICAL,
            )
            for row in rows
        ]

    # -- writes ----------------------------------------------------------------------

    def _row(self, record: Record) -> list[object]:
        values: list[object] = [record.id]
        for spec in self._schema.fields:
            value = record.fields.get(spec.name, _DEFAULTS[spec.type])
            values.append(_write_value(spec, value))
        values.append(Vector([float(item) for item in record.vector or ()]))
        return values

    def _source_lock_key(self, record_filter: RecordFilter) -> str:
        """Return the lock key for a table and a canonical form of the filter."""
        clauses: list[tuple[str, str]] = []
        for clause in record_filter.all_of:
            spec = self._schema.field(clause.field)
            value = _write_value(spec, checked_filter_value(spec, clause.value))
            clauses.append((clause.field, repr(value)))
        clauses.sort()
        return f"{self._settings.db_schema}.{self._schema.name}:{clauses!r}"

    async def _write(
        self, records: Sequence[Record], stale: RecordFilter | None
    ) -> None:
        """Upsert ``records`` in one transaction, then optionally delete stale rows.

        ``stale`` selects the records to remove when their key is not in
        ``records``.
        """
        rows = [self._row(record) for record in records]
        statement = self.upsert_statement()
        async with self._connection() as connection, connection.transaction():
            if stale is not None:
                # Two replacements of one source would each miss the other's
                # uncommitted rows and both keep their own. The lock, held to
                # the end of this transaction, makes the second one wait.
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (self._source_lock_key(stale),),
                )
            async with connection.cursor() as cursor:
                for start in range(0, len(rows), UPSERT_BATCH_SIZE):
                    await cursor.executemany(
                        statement, rows[start : start + UPSERT_BATCH_SIZE]
                    )
            if stale is not None:
                conditions, params = self._where(stale)
                conditions.append(sql.SQL("{} <> ALL(%s)").format(self._key))
                await connection.execute(
                    sql.SQL("DELETE FROM {}{}").format(
                        self._table, self._where_sql(conditions)
                    ),
                    [*params, [record.id for record in records]],
                )

    async def upsert(self, records: Sequence[Record]) -> None:
        self._preflight(records)
        if not records:
            return
        await self._after_create(lambda: self._write(records, None))

    async def replace_source(
        self, record_filter: RecordFilter, records: Sequence[Record]
    ) -> None:
        """Upsert ``records`` and delete the rest of the source in one transaction."""
        self._preflight(records)
        self._where(record_filter)  # reject unknown fields before any write
        await self._after_create(lambda: self._write(records, record_filter))

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        statement = sql.SQL("DELETE FROM {} WHERE {} = ANY(%s)").format(
            self._table, self._key
        )
        try:
            async with self._connection() as connection:
                await connection.execute(statement, (list(ids),))
        except errors.UndefinedTable:
            return

    async def delete_where(self, record_filter: RecordFilter) -> int:
        conditions, params = self._where(record_filter)
        statement = sql.SQL("DELETE FROM {}{}").format(
            self._table, self._where_sql(conditions)
        )
        try:
            async with self._connection() as connection:
                cursor = await connection.execute(statement, params)
                return max(int(cursor.rowcount), 0)
        except errors.UndefinedTable:
            return 0


async def _configure(connection: AsyncConnection[TupleRow]) -> None:
    """Register the vector type on a new pooled connection.

    A database without the extension yet is not an error: ``ensure_collection``
    creates it and then replaces the pool.
    """
    try:
        await register_vector_async(connection)
    except psycopg.ProgrammingError:
        logger.debug("pgvector is not installed in this database yet.")
    finally:
        await connection.commit()
