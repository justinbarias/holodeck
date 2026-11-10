"""Native ChromaDB adapter.

Stored layout
-------------
The layout matches the collections the Semantic Kernel path wrote, so a
collection written by either side is read by the other:

* Collection name is ``schema.name``.
* The collection is created with HNSW space ``cosine`` in its configuration.
  Collections created with the older ``hnsw:space`` metadata report the same
  space in their configuration and are read as they are.
* Record IDs are stored as given.
* Every schema field is one Chroma metadata key with the schema's type. A
  ``None`` value is stored as the type's default, because Chroma rejects
  ``None`` metadata values.
* The vector is the Chroma embedding. No Chroma ``document`` is stored.

Connection settings
-------------------
``connection_string`` (``http[s]://host[:port]``) selects a Chroma server.
Otherwise ``persist_directory`` selects an embedded persistent database, and
with neither the database is embedded and in memory. ``headers`` (server
only), ``tenant`` and ``database`` are optional. Other keys are ignored with a
warning, as before.

Execution
---------
The Chroma client is synchronous. Every client call runs on one worker thread
of the adapter, so the event loop never waits on Chroma and one adapter never
runs two calls on its client at once. The worker thread starts on the first
call and ends in ``close``. The adapter is usable again after ``close``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, TypedDict, TypeVar, cast
from urllib.parse import urlparse

import chromadb
import httpx
from chromadb.api import ClientAPI
from chromadb.api.client import Client
from chromadb.api.collection_configuration import (
    CreateCollectionConfiguration,
    CreateHNSWConfiguration,
)
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Include, Metadata, Where
from chromadb.errors import (
    ChromaError,
    InvalidArgumentError,
    InvalidDimensionException,
    NotFoundError,
)

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    IncompatibleCollectionError,
    RetrievalError,
    SchemaError,
    StoreConnectionError,
)
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    FieldType,
    FieldValue,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
    clamp_similarity,
)

logger = logging.getLogger(__name__)

PROVIDER = "chromadb"

DEFAULT_TENANT = "default_tenant"
DEFAULT_DATABASE = "default_database"

_SPACE: Literal["cosine"] = "cosine"
_WRITE_BATCH = 1000

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
        }
    ),
    hints={
        Operation.LEXICAL_SEARCH: "HoloDeck uses its BM25 index for this provider.",
        Operation.HYBRID_SEARCH: "HoloDeck fuses vector and BM25 results itself.",
    },
)

_DEFAULTS: dict[FieldType, str | int | float | bool] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

_KNOWN_KEYS = frozenset(
    {"connection_string", "persist_directory", "headers", "tenant", "database"}
)

T = TypeVar("T")


# -- connection settings -----------------------------------------------------


class ChromaConnectionParams(TypedDict):
    """Parameters for a ChromaDB server connection.

    Attributes:
        host: Server hostname (for example ``localhost``).
        port: Server port (for example 8000).
        ssl: Whether to use HTTPS.
    """

    host: str
    port: int
    ssl: bool


def parse_chromadb_connection_string(connection_string: str) -> ChromaConnectionParams:
    """Parse a ChromaDB connection string into connection parameters.

    The format is ``http[s]://[host][:port]``. The scheme sets SSL. The host
    defaults to ``localhost`` and the port to 8000 (HTTP) or 443 (HTTPS).

    Raises:
        ValueError: The string is empty or its scheme is not http or https.

    Examples:
        >>> parse_chromadb_connection_string("http://localhost:8000")
        {'host': 'localhost', 'port': 8000, 'ssl': False}
        >>> parse_chromadb_connection_string("https://chroma.example.com")
        {'host': 'chroma.example.com', 'port': 443, 'ssl': True}
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")

    parsed = urlparse(connection_string)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}'. ChromaDB connection string must use "
            "http:// or https:// scheme"
        )

    ssl = parsed.scheme == "https"
    return ChromaConnectionParams(
        host=parsed.hostname or "localhost",
        port=parsed.port or (443 if ssl else 8000),
        ssl=ssl,
    )


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    """Validated connection settings.

    Attributes:
        connection_string: Server URL, or None for an embedded database.
        persist_directory: Directory of an embedded persistent database.
        headers: HTTP headers for the server.
        tenant: Chroma tenant.
        database: Chroma database.
    """

    connection_string: str | None = None
    persist_directory: str | None = None
    headers: Mapping[str, str] | None = None
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE


def _optional_text(connection: Mapping[str, object], key: str) -> str | None:
    value = connection.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if key == "persist_directory" and isinstance(value, os.PathLike):
        return str(value)
    raise ValueError(f"ChromaDB setting '{key}' must be a string.")


def parse_connection(connection: Mapping[str, object]) -> ChromaSettings:
    """Validate the connection mapping of an agent's ``database`` block.

    Raises:
        ValueError: A setting has the wrong type or the URL is invalid.
    """
    headers_raw = connection.get("headers")
    headers: dict[str, str] | None = None
    if headers_raw is not None:
        if not isinstance(headers_raw, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers_raw.items()
        ):
            raise ValueError("ChromaDB setting 'headers' must map strings to strings.")
        headers = {str(k): str(v) for k, v in headers_raw.items()}

    connection_string = _optional_text(connection, "connection_string")
    if connection_string is not None:
        parse_chromadb_connection_string(connection_string)

    ignored = sorted(set(connection) - _KNOWN_KEYS)
    if ignored:
        logger.warning("ChromaDB ignores these connection settings: %s", ignored)

    return ChromaSettings(
        connection_string=connection_string,
        persist_directory=_optional_text(connection, "persist_directory"),
        headers=headers,
        tenant=_optional_text(connection, "tenant") or DEFAULT_TENANT,
        database=_optional_text(connection, "database") or DEFAULT_DATABASE,
    )


def create_chromadb_client(settings: ChromaSettings) -> ClientAPI:
    """Create the synchronous Chroma client for ``settings``.

    A connection string wins over a persist directory. With neither, the
    client is ephemeral and the data is lost when the process ends.
    """
    if settings.connection_string:
        params = parse_chromadb_connection_string(settings.connection_string)
        return chromadb.HttpClient(
            host=params["host"],
            port=params["port"],
            ssl=params["ssl"],
            headers=dict(settings.headers) if settings.headers else None,
            tenant=settings.tenant,
            database=settings.database,
        )
    if settings.persist_directory:
        return chromadb.PersistentClient(
            path=settings.persist_directory,
            tenant=settings.tenant,
            database=settings.database,
        )
    return chromadb.EphemeralClient(tenant=settings.tenant, database=settings.database)


# -- the synchronous store ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Row:
    """One stored row as read back from Chroma."""

    id: str
    metadata: Mapping[str, FieldValue]
    vector: list[float] | None = None
    distance: float | None = None


@dataclass(frozen=True, slots=True)
class _Batch:
    """Rows to write, in Chroma's columnar form."""

    ids: list[str]
    vectors: list[list[float]]
    metadatas: list[dict[str, FieldValue]] | None


def _scalars(raw: Mapping[str, object] | None) -> dict[str, FieldValue]:
    """Keep the scalar metadata values; Chroma may also hold sparse vectors."""
    if not raw:
        return {}
    return {
        key: value
        for key, value in raw.items()
        if value is None or isinstance(value, str | int | float | bool)
    }


def _collection_space(collection: Collection) -> str | None:
    hnsw = collection.configuration.get("hnsw")
    if hnsw is not None:
        space = hnsw.get("space")
        if space is not None:
            return str(space)
    metadata = collection.metadata
    if metadata is not None and "hnsw:space" in metadata:
        return str(metadata["hnsw:space"])
    return None


class _ChromaStore:
    """One collection through the synchronous Chroma client.

    Not thread-safe: the adapter calls it from one worker thread. It holds
    the client and the collection handle and opens them on first use, so a
    store is usable again after ``close``.
    """

    def __init__(self, settings: ChromaSettings, name: str) -> None:
        self._settings = settings
        self._name = name
        self._client: ClientAPI | None = None
        self._handle: Collection | None = None

    def _connect(self) -> ClientAPI:
        if self._client is None:
            try:
                self._client = create_chromadb_client(self._settings)
            except ValueError as error:
                # Chroma reports an unreachable server, or a tenant or
                # database it does not have, as a ValueError at construction.
                if self._settings.connection_string:
                    raise StoreConnectionError(PROVIDER, str(error)) from error
                raise
        return self._client

    def _lookup(self) -> Collection | None:
        try:
            return self._connect().get_collection(
                name=self._name, embedding_function=None
            )
        except NotFoundError:
            return None

    def _create(self) -> Collection:
        return self._connect().create_collection(
            name=self._name,
            configuration=CreateCollectionConfiguration(
                hnsw=CreateHNSWConfiguration(space=_SPACE)
            ),
            embedding_function=None,
            get_or_create=True,
        )

    def _resolve(self, *, create: bool) -> Collection | None:
        if self._handle is None:
            handle = self._lookup()
            if handle is None and create:
                handle = self._create()
            if handle is not None:
                space = _collection_space(handle)
                if space != _SPACE:
                    raise IncompatibleCollectionError(
                        f"Chroma collection '{self._name}' uses distance space "
                        f"'{space}'; HoloDeck needs '{_SPACE}'."
                    )
            self._handle = handle
        return self._handle

    def _with(
        self,
        operation: Callable[[Collection], T],
        *,
        absent: T,
        create: bool = False,
    ) -> T:
        """Run ``operation`` on the collection, or return ``absent`` when missing.

        A handle that went stale (the collection was deleted or recreated by
        someone else) is dropped and resolved once more.
        """
        for attempt in (1, 2):
            handle = self._resolve(create=create)
            if handle is None:
                return absent
            try:
                return operation(handle)
            except NotFoundError:
                self._handle = None
                if attempt == 2:
                    return absent
        return absent

    # -- collection ----------------------------------------------------------

    def exists(self) -> bool:
        return self._lookup() is not None

    def ensure(self) -> None:
        self._resolve(create=True)

    def drop(self) -> None:
        self._handle = None
        try:
            self._connect().delete_collection(name=self._name)
        except NotFoundError:
            return

    def close(self) -> None:
        client, self._client, self._handle = self._client, None, None
        if isinstance(client, Client):
            client.close()

    # -- records -------------------------------------------------------------

    def count(self) -> int:
        return self._with(lambda collection: collection.count(), absent=0)

    def write(self, batch: _Batch) -> None:
        """Insert the rows, replacing any that exist, by delete then add.

        Chroma's ``upsert`` and ``update`` are not used. Against Chroma
        1.5.9 an upsert of new IDs dropped exactly one row from the vector
        index in 5 to 7 percent of 36-row ingests: the row stayed out of every
        vector query for good, although ``get`` and ``count`` saw it, and
        writing it again fixed it. ``update`` showed it too (1 in 100).
        ``add`` and ``delete`` never did (0 in 220).

        A batch is not atomic: a failure between the delete and the add
        leaves those rows missing, which the callers' source replacement and
        re-ingestion repair.
        """

        def operation(collection: Collection) -> None:
            for start in range(0, len(batch.ids), _WRITE_BATCH):
                stop = start + _WRITE_BATCH
                ids = batch.ids[start:stop]
                existing = collection.get(ids=ids, include=[])["ids"]
                if existing:
                    collection.delete(ids=existing)
                collection.add(
                    ids=ids,
                    embeddings=cast("list[Sequence[float]]", batch.vectors[start:stop]),
                    metadatas=(
                        None
                        if batch.metadatas is None
                        else cast("list[Metadata]", batch.metadatas)[start:stop]
                    ),
                )

        self._with(operation, absent=None, create=True)

    def delete_ids(self, ids: Sequence[str]) -> None:
        def operation(collection: Collection) -> None:
            for start in range(0, len(ids), _WRITE_BATCH):
                collection.delete(ids=list(ids[start : start + _WRITE_BATCH]))

        self._with(operation, absent=None)

    def delete_where(self, where: Where | None) -> int:
        def operation(collection: Collection) -> int:
            ids = collection.get(where=where, include=[])["ids"]
            if not ids:
                return 0
            if where is None:
                for start in range(0, len(ids), _WRITE_BATCH):
                    collection.delete(ids=ids[start : start + _WRITE_BATCH])
            else:
                collection.delete(where=where)
            return len(ids)

        return self._with(operation, absent=0)

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Where | None = None,
        limit: int | None = None,
        offset: int | None = None,
        vectors: bool = False,
    ) -> list[_Row]:
        include: Include = ["metadatas", "embeddings"] if vectors else ["metadatas"]

        def operation(collection: Collection) -> list[_Row]:
            result = collection.get(
                ids=None if ids is None else list(ids),
                where=where,
                limit=limit,
                offset=offset,
                include=include,
            )
            metadatas = result["metadatas"] or []
            embeddings = result["embeddings"]
            rows: list[_Row] = []
            for index, record_id in enumerate(result["ids"]):
                vector = (
                    [float(v) for v in embeddings[index]]
                    if embeddings is not None
                    else None
                )
                rows.append(
                    _Row(
                        id=record_id,
                        metadata=_scalars(metadatas[index] if metadatas else None),
                        vector=vector,
                    )
                )
            return rows

        return self._with(operation, absent=[])

    def query(
        self, vector: Sequence[float], top_k: int, where: Where | None
    ) -> list[_Row]:
        embeddings: list[Sequence[float]] = [list(vector)]

        def operation(collection: Collection) -> list[_Row]:
            result = collection.query(
                query_embeddings=embeddings,
                n_results=top_k,
                where=where,
                include=["metadatas", "distances"],
            )
            ids = result["ids"][0]
            metadatas = (result["metadatas"] or [[]])[0]
            distances = (result["distances"] or [[]])[0]
            return [
                _Row(
                    id=record_id,
                    metadata=_scalars(metadatas[index] if metadatas else None),
                    distance=float(distances[index]),
                )
                for index, record_id in enumerate(ids)
            ]

        return self._with(operation, absent=[])


# -- the adapter -------------------------------------------------------------


def _coerce(field_type: FieldType, value: FieldValue) -> FieldValue:
    if value is None:
        return None
    if field_type is FieldType.INTEGER:
        return int(value)
    if field_type is FieldType.FLOAT:
        return float(value)
    if field_type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


class ChromaAdapter(BaseVectorStoreAdapter):
    """One ChromaDB collection, on a server or embedded."""

    def __init__(self, schema: CollectionSchema, settings: ChromaSettings) -> None:
        super().__init__(schema, _CAPABILITIES)
        self._settings = settings
        self._store = _ChromaStore(settings, schema.name)
        self._pool: ThreadPoolExecutor | None = None

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> ChromaAdapter:
        """Build the adapter. No connection is opened until the first call.

        Raises:
            RetrievalError: A connection setting is invalid.
        """
        try:
            settings = parse_connection(connection)
        except ValueError as error:
            raise RetrievalError(
                f"Invalid ChromaDB connection settings: {error}"
            ) from error
        return cls(schema, settings)

    # -- execution -----------------------------------------------------------

    async def _run(self, function: Callable[[], T]) -> T:
        """Run one store call on the adapter's worker thread."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"holodeck-chroma-{self._schema.name}"
            )
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._pool, function)
        except RetrievalError:
            raise
        except httpx.TransportError as error:
            raise StoreConnectionError(PROVIDER, str(error)) from error
        except InvalidDimensionException as error:
            raise IncompatibleCollectionError(str(error)) from error
        except InvalidArgumentError as error:
            if "dimension" in str(error):
                raise IncompatibleCollectionError(str(error)) from error
            raise RetrievalError(f"ChromaDB error: {error}") from error
        except ChromaError as error:
            raise RetrievalError(f"ChromaDB error: {error}") from error

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(pool, self._store.close)
        except httpx.HTTPError:
            logger.debug("Chroma client close failed", exc_info=True)
        finally:
            pool.shutdown(wait=False)

    # -- collection ----------------------------------------------------------

    async def collection_exists(self) -> bool:
        return await self._run(self._store.exists)

    async def ensure_collection(self) -> None:
        await self._run(self._store.ensure)

    async def delete_collection(self) -> None:
        await self._run(self._store.drop)

    async def count(self) -> int:
        return await self._run(self._store.count)

    # -- conversion ----------------------------------------------------------

    def _to_record(self, row: _Row) -> Record:
        fields = {
            spec.name: _coerce(spec.type, row.metadata[spec.name])
            for spec in self._schema.fields
            if spec.name in row.metadata
        }
        return Record(id=row.id, fields=fields, vector=row.vector)

    def _batch(self, records: Sequence[Record]) -> _Batch:
        latest = {record.id: record for record in records}
        ids = list(latest)
        metadatas: list[dict[str, FieldValue]] = []
        for record in latest.values():
            metadata: dict[str, FieldValue] = {}
            for spec in self._schema.fields:
                value = record.fields.get(spec.name)
                metadata[spec.name] = (
                    _DEFAULTS[spec.type] if value is None else _coerce(spec.type, value)
                )
            metadatas.append(metadata)
        return _Batch(
            ids=ids,
            vectors=[[float(v) for v in latest[i].vector or ()] for i in ids],
            metadatas=metadatas if self._schema.fields else None,
        )

    def _where(self, record_filter: RecordFilter | None) -> Where | None:
        if record_filter is None or not record_filter.all_of:
            return None
        clauses: list[dict[str, FieldValue]] = []
        for clause in record_filter.all_of:
            self._schema.field(clause.field)
            if clause.value is None:
                raise SchemaError(
                    f"Cannot filter on None for field '{clause.field}': "
                    "ChromaDB stores no null values."
                )
            clauses.append({clause.field: clause.value})
        return cast(Where, clauses[0] if len(clauses) == 1 else {"$and": clauses})

    # -- records -------------------------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        if not records:
            return
        batch = self._batch(records)
        await self._run(lambda: self._store.write(batch))

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        if not ids:
            return []
        wanted = list(dict.fromkeys(ids))
        rows = await self._run(
            lambda: self._store.get(ids=wanted, vectors=include_vectors)
        )
        by_id = {row.id: self._to_record(row) for row in rows}
        return [by_id[key] for key in ids if key in by_id]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        unique = list(dict.fromkeys(ids))
        await self._run(lambda: self._store.delete_ids(unique))

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        where = self._where(record_filter)
        rows = await self._run(lambda: self._store.get(where=where, limit=limit))
        return [self._to_record(row) for row in rows]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        if offset < 0:
            raise SchemaError(f"Invalid cursor: {cursor!r}")
        rows = await self._run(lambda: self._store.get(limit=limit + 1, offset=offset))
        more = len(rows) > limit
        return Page(
            records=tuple(self._to_record(row) for row in rows[:limit]),
            next_cursor=str(offset + limit) if more else None,
        )

    async def delete_where(self, record_filter: RecordFilter) -> int:
        where = self._where(record_filter)
        return await self._run(lambda: self._store.delete_where(where))

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        if len(query.vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(query.vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )
        where = self._where(query.filter)
        if query.top_k <= 0:
            return []
        vector = [float(v) for v in query.vector]
        rows = await self._run(lambda: self._store.query(vector, query.top_k, where))
        hits = [
            ScoredRecord(
                record=self._to_record(row),
                score=clamp_similarity(1.0 - (row.distance or 0.0)),
                kind=ScoreKind.SIMILARITY,
            )
            for row in rows
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: query.top_k]
