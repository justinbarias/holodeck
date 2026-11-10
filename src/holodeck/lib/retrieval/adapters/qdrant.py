"""Native Qdrant adapter on ``qdrant_client.AsyncQdrantClient``.

Stored layout
-------------
The layout is the one Semantic Kernel's ``QdrantCollection`` and the SK
bridge wrote, so existing collections stay readable and writable:

* one named dense vector, ``schema.vector_field`` (``embedding``), cosine
  distance, ``schema.dimensions``, float32;
* point IDs: a HoloDeck ID that is a UUID is stored as the canonical UUID
  string. Any other ID is stored as ``uuid5(namespace, id)`` with
  ``namespace = uuid5(NAMESPACE_URL, "holodeck/retrieval/<collection>")``
  and the original ID in the ``holodeck_id`` payload key. Reads return the
  HoloDeck ID;
* payload: one key per data field, with the type default for a missing value;
* payload indexes from the ``FieldSpec`` flags: keyword, integer, float or
  bool for ``indexed`` fields, and a word-tokenized, lowercased full-text
  index (token length 2 to 20) for ``full_text`` fields that are not also
  ``indexed``.

Hybrid search
-------------
A dense leg and a full-text leg (one ``MatchText`` clause per query token,
OR-combined) are prefetched with ``max(4 * top_k, 20)`` candidates each and
fused by the server with RRF. Scores are the raw RRF values. A query with no
tokens of two or more characters searches by vector only.

Lifecycle
---------
The adapter owns one client. It is built on first use and closed in
``close``; the next call after ``close`` builds a new one.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TypedDict
from urllib.parse import urlparse

from grpc import StatusCode
from grpc.aio import AioRpcError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    RetrievalError,
    SchemaError,
    StoreConnectionError,
)
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    FieldType,
    FieldValue,
    HybridQuery,
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

PROVIDER = "qdrant"

ID_PAYLOAD_KEY = "holodeck_id"
"""Payload key that keeps the HoloDeck ID when it is not a UUID."""

MIN_TOKEN_LEN = 2
"""Shortest query token used by the full-text leg of hybrid search."""

CAPABILITIES = Capabilities(
    PROVIDER,
    frozenset(
        {
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
    ),
    {
        Operation.LEXICAL_SEARCH: (
            "Qdrant full-text matching is an unscored filter; use hybrid search."
        ),
    },
)

_DEFAULTS: dict[FieldType, str | int | float | bool] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

_CONNECTION_REFUSED = frozenset(
    {
        StatusCode.UNAVAILABLE,
        StatusCode.UNAUTHENTICATED,
        StatusCode.PERMISSION_DENIED,
        StatusCode.DEADLINE_EXCEEDED,
    }
)


# -- connection settings -------------------------------------------------------


class QdrantConnectionParams(TypedDict, total=False):
    """Keyword arguments for ``AsyncQdrantClient``.

    All fields are optional; with none set the client connects to
    ``localhost:6333``.

    Attributes:
        url: Full URL of the server, e.g. ``https://qdrant.example.com:6333``.
        api_key: API key for authentication.
        host: Server hostname.
        port: HTTP port (client default 6333).
        grpc_port: gRPC port (client default 6334).
        prefer_grpc: Use gRPC instead of HTTP.
        https: Use HTTPS with ``host``.
        prefix: URL path prefix.
        timeout: Request timeout in seconds.
        location: ``:memory:`` for an in-process store, or a URL.
        path: Directory of an in-process persistent store.
        check_compatibility: Check the client and server versions.
        force_disable_check_same_thread: Local mode SQLite thread check.
        pool_size: Connection pool size.
        headers: Extra HTTP headers.
    """

    url: str
    api_key: str
    host: str
    port: int
    grpc_port: int
    prefer_grpc: bool
    https: bool
    prefix: str
    timeout: int
    location: str
    path: str
    check_compatibility: bool
    force_disable_check_same_thread: bool
    pool_size: int
    headers: dict[str, str]


CONNECTION_KEYS = frozenset(QdrantConnectionParams.__annotations__) | {
    "connection_string",
    "collection_name",
}
"""Connection keys the adapter accepts.

``collection_name`` is read by the tool schemas and only names the collection.
"""


def parse_qdrant_connection_string(connection_string: str) -> QdrantConnectionParams:
    """Parse a Qdrant connection string into client parameters.

    Supported formats:

    - URL: ``https://host:port`` or ``http://localhost:6333``
    - API key in userinfo: ``https://api_key@host:port``
    - gRPC: ``qdrant+grpc://host:port`` or ``grpc://host:port``
    - In-memory: ``:memory:``
    - Local path: ``/path/to/data`` or ``file:///path/to/data``

    Raises:
        ValueError: The string is empty or uses an unsupported scheme.

    Examples:
        >>> parse_qdrant_connection_string("https://qdrant.example.com:6333")
        {'url': 'https://qdrant.example.com:6333'}
        >>> parse_qdrant_connection_string("http://localhost:6333")
        {'host': 'localhost', 'port': 6333}
        >>> parse_qdrant_connection_string(":memory:")
        {'location': ':memory:'}
        >>> parse_qdrant_connection_string("qdrant+grpc://localhost:6334")
        {'host': 'localhost', 'grpc_port': 6334, 'prefer_grpc': True}
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")
    if connection_string == ":memory:":
        return {"location": ":memory:"}
    if connection_string.startswith("/") or connection_string.startswith("file://"):
        return {"path": connection_string.replace("file://", "")}

    parsed = urlparse(connection_string)
    prefer_grpc = parsed.scheme in ("qdrant+grpc", "grpc")
    if parsed.scheme in ("qdrant", "qdrant+grpc", "grpc"):
        actual_scheme = "https" if parsed.port == 443 else "http"
    elif parsed.scheme in ("http", "https"):
        actual_scheme = parsed.scheme
    else:
        raise ValueError(
            f"Invalid scheme '{parsed.scheme}'. Qdrant connection string must use "
            "http://, https://, qdrant://, or qdrant+grpc:// scheme"
        )

    host = parsed.hostname or "localhost"
    params: QdrantConnectionParams
    if prefer_grpc:
        params = {
            "host": host,
            "grpc_port": parsed.port or 6334,
            "prefer_grpc": True,
        }
    elif host in ("localhost", "127.0.0.1"):
        params = {"host": host, "port": parsed.port or 6333}
    else:
        url = f"{actual_scheme}://{host}"
        if parsed.port:
            url += f":{parsed.port}"
        params = {"url": url}

    if parsed.username:
        params["api_key"] = parsed.username
    return params


def _as_str(key: str, value: object) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"Qdrant setting '{key}' must be a string.")


def _as_int(key: str, value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"Qdrant setting '{key}' must be an integer.")


def _as_bool(key: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Qdrant setting '{key}' must be true or false.")


def _as_headers(key: str, value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        headers: dict[str, str] = {}
        for name, item in value.items():
            if not isinstance(name, str) or not isinstance(item, str):
                break
            headers[name] = item
        else:
            return headers
    raise ValueError(f"Qdrant setting '{key}' must map header names to strings.")


def qdrant_connection_params(
    connection: Mapping[str, object],
) -> QdrantConnectionParams:
    """Build client parameters from tool database settings.

    ``connection_string`` is parsed first; explicit keys override it.
    ``collection_name`` is ignored here (the schema carries it).

    Raises:
        ValueError: An unknown key, a value of the wrong type, a bad
            connection string, or more than one of url/host/location/path.
    """
    unknown = sorted(set(connection).difference(CONNECTION_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown Qdrant settings {unknown}. "
            f"Accepted settings: {sorted(CONNECTION_KEYS)}"
        )
    params: QdrantConnectionParams = {}
    raw = connection.get("connection_string")
    if raw is not None:
        params = parse_qdrant_connection_string(_as_str("connection_string", raw))
    for key, value in connection.items():
        if value is None or key in ("connection_string", "collection_name"):
            continue
        if key == "url":
            params["url"] = _as_str(key, value)
        elif key == "api_key":
            params["api_key"] = _as_str(key, value)
        elif key == "host":
            params["host"] = _as_str(key, value)
        elif key == "prefix":
            params["prefix"] = _as_str(key, value)
        elif key == "location":
            params["location"] = _as_str(key, value)
        elif key == "path":
            params["path"] = _as_str(key, value)
        elif key == "port":
            params["port"] = _as_int(key, value)
        elif key == "grpc_port":
            params["grpc_port"] = _as_int(key, value)
        elif key == "timeout":
            params["timeout"] = _as_int(key, value)
        elif key == "pool_size":
            params["pool_size"] = _as_int(key, value)
        elif key == "prefer_grpc":
            params["prefer_grpc"] = _as_bool(key, value)
        elif key == "https":
            params["https"] = _as_bool(key, value)
        elif key == "check_compatibility":
            params["check_compatibility"] = _as_bool(key, value)
        elif key == "force_disable_check_same_thread":
            params["force_disable_check_same_thread"] = _as_bool(key, value)
        elif key == "headers":
            params["headers"] = _as_headers(key, value)
    targets = [key for key in ("url", "host", "location", "path") if key in params]
    if len(targets) > 1:
        raise ValueError(
            f"Only one of url, host, location or path may be set; got {targets}."
        )
    return params


def create_qdrant_client(params: QdrantConnectionParams) -> AsyncQdrantClient:
    """Build the ``AsyncQdrantClient`` an adapter owns."""
    return AsyncQdrantClient(**params)


# -- conversions ---------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Split a query into lowercase tokens, dropping short and repeated ones."""
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in re.findall(r"\w+", text.lower(), re.UNICODE):
        if len(raw) < MIN_TOKEN_LEN or raw in seen:
            continue
        seen.add(raw)
        tokens.append(raw)
    return tokens


def _coerce(field_type: FieldType, value: object) -> FieldValue:
    """Convert a payload value to the schema type, as the SK bridge did."""
    if value is None:
        return None
    if field_type is FieldType.BOOLEAN:
        return bool(value)
    if field_type is FieldType.STRING:
        return value if isinstance(value, str) else str(value)
    if isinstance(value, int | float | str):
        number = float(value) if field_type is FieldType.FLOAT else int(value)
        return number
    raise SchemaError(f"Cannot read {value!r} as {field_type.value}.")


def _dense_vector(raw: object, name: str) -> list[float] | None:
    """Return the named dense vector of a point, or None."""
    vector = raw.get(name) if isinstance(raw, dict) else raw
    if isinstance(vector, list) and all(
        isinstance(item, int | float) for item in vector
    ):
        return [float(item) for item in vector]
    return None


def index_params(
    field_type: FieldType, *, indexed: bool, full_text: bool
) -> qm.PayloadSchemaParams | None:
    """Return the payload index for a field, or None when it has none."""
    if full_text and not indexed:
        return qm.TextIndexParams(
            type=qm.TextIndexType.TEXT,
            tokenizer=qm.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
    if not indexed:
        return None
    if field_type is FieldType.STRING:
        return qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD)
    if field_type is FieldType.INTEGER:
        return qm.IntegerIndexParams(type=qm.IntegerIndexType.INTEGER)
    if field_type is FieldType.FLOAT:
        return qm.FloatIndexParams(type=qm.FloatIndexType.FLOAT)
    return qm.BoolIndexParams(type=qm.BoolIndexType.BOOL)


# -- adapter -------------------------------------------------------------------


class QdrantAdapter(BaseVectorStoreAdapter):
    """One Qdrant collection through ``AsyncQdrantClient``."""

    def __init__(
        self, schema: CollectionSchema, params: QdrantConnectionParams
    ) -> None:
        if ID_PAYLOAD_KEY in schema.field_names:
            raise SchemaError(f"Field name '{ID_PAYLOAD_KEY}' is reserved for qdrant.")
        super().__init__(schema, CAPABILITIES)
        self._params = params
        self._local = "path" in params or params.get("location") == ":memory:"
        self._namespace = uuid.uuid5(
            uuid.NAMESPACE_URL, f"holodeck/retrieval/{schema.name}"
        )
        self._client: AsyncQdrantClient | None = None
        self._lock = asyncio.Lock()
        self._indexes_ensured = False

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> QdrantAdapter:
        """Build the adapter from tool database settings.

        Raises:
            StoreConnectionError: The settings are invalid.
        """
        try:
            params = qdrant_connection_params(connection)
        except ValueError as error:
            raise StoreConnectionError(PROVIDER, str(error)) from error
        return cls(schema, params)

    # -- lifecycle -----------------------------------------------------------

    async def _open(self) -> AsyncQdrantClient:
        async with self._lock:
            if self._client is None:
                with self._errors():
                    self._client = create_qdrant_client(self._params)
            return self._client

    async def close(self) -> None:
        """Close the client. Safe to call more than once."""
        async with self._lock:
            client, self._client = self._client, None
            if client is not None:
                await client.close()

    @contextmanager
    def _errors(self) -> Iterator[None]:
        """Turn vendor connection and response errors into retrieval errors."""
        try:
            yield
        except ResponseHandlingException as error:
            raise StoreConnectionError(PROVIDER, str(error)) from error
        except UnexpectedResponse as error:
            if error.status_code in (401, 403):
                raise StoreConnectionError(PROVIDER, str(error)) from error
            raise RetrievalError(f"Qdrant request failed: {error}") from error
        except AioRpcError as error:
            if error.code() in _CONNECTION_REFUSED:
                raise StoreConnectionError(PROVIDER, str(error.details())) from error
            raise RetrievalError(f"Qdrant request failed: {error.details()}") from error
        except (ConnectionError, TimeoutError) as error:
            raise StoreConnectionError(PROVIDER, str(error)) from error

    # -- collection ----------------------------------------------------------

    async def collection_exists(self) -> bool:
        client = await self._open()
        with self._errors():
            return bool(await client.collection_exists(self._schema.name))

    async def ensure_collection(self) -> None:
        client = await self._open()
        with self._errors():
            if not await client.collection_exists(self._schema.name):
                await client.create_collection(
                    collection_name=self._schema.name,
                    vectors_config={
                        self._schema.vector_field: qm.VectorParams(
                            size=self._schema.dimensions,
                            distance=qm.Distance.COSINE,
                            datatype=qm.Datatype.FLOAT32,
                        )
                    },
                )
        await self._ensure_payload_indexes(client)

    async def delete_collection(self) -> None:
        client = await self._open()
        with self._errors():
            if await client.collection_exists(self._schema.name):
                await client.delete_collection(self._schema.name)
        self._indexes_ensured = False

    async def _ensure_payload_indexes(self, client: AsyncQdrantClient) -> None:
        """Create the payload indexes for indexed and full-text fields.

        Qdrant treats a repeated request with the same parameters as a no-op.
        A rejected index is logged and does not block writes; a connection
        failure is raised.
        """
        if self._indexes_ensured:
            return
        if self._local:
            # The in-process store ignores payload indexes and warns about them.
            self._indexes_ensured = True
            return
        for spec in self._schema.fields:
            params = index_params(
                spec.type, indexed=spec.indexed, full_text=spec.full_text
            )
            if params is None:
                continue
            try:
                with self._errors():
                    await client.create_payload_index(
                        collection_name=self._schema.name,
                        field_name=spec.name,
                        field_schema=params,
                    )
            except StoreConnectionError:
                raise
            except RetrievalError as error:
                logger.warning(
                    "Qdrant payload index failed for '%s' on '%s': %s",
                    spec.name,
                    self._schema.name,
                    error,
                )
        self._indexes_ensured = True

    # -- IDs, payloads and filters -------------------------------------------

    def store_id(self, record_id: str) -> str:
        """Return the Qdrant point ID for a HoloDeck record ID."""
        try:
            return str(uuid.UUID(record_id))
        except ValueError:
            return str(uuid.uuid5(self._namespace, record_id))

    def _payload(self, record: Record, store_id: str) -> dict[str, FieldValue]:
        payload: dict[str, FieldValue] = {}
        for spec in self._schema.fields:
            value = record.fields.get(spec.name)
            payload[spec.name] = _DEFAULTS[spec.type] if value is None else value
        if store_id != record.id:
            payload[ID_PAYLOAD_KEY] = record.id
        return payload

    def _record(
        self,
        point_id: qm.ExtendedPointId,
        payload: Mapping[str, object] | None,
        vector: object = None,
        *,
        vectors: bool = False,
    ) -> Record:
        payload = payload or {}
        original = payload.get(ID_PAYLOAD_KEY)
        fields = {
            spec.name: _coerce(spec.type, payload[spec.name])
            for spec in self._schema.fields
            if spec.name in payload
        }
        return Record(
            id=str(original) if original is not None else str(point_id),
            fields=fields,
            vector=(
                _dense_vector(vector, self._schema.vector_field) if vectors else None
            ),
        )

    def _check_filter(self, record_filter: RecordFilter | None) -> None:
        if record_filter is not None:
            for clause in record_filter.all_of:
                self._schema.field(clause.field)

    def build_filter(self, record_filter: RecordFilter | None) -> qm.Filter | None:
        """Return the Qdrant filter for ``record_filter``, or None."""
        self._check_filter(record_filter)
        if record_filter is None or not record_filter.all_of:
            return None
        conditions: list[qm.Condition] = []
        for clause in record_filter.all_of:
            value = clause.value
            if value is None:
                conditions.append(
                    qm.IsNullCondition(is_null=qm.PayloadField(key=clause.field))
                )
            elif isinstance(value, float):
                conditions.append(
                    qm.FieldCondition(
                        key=clause.field, range=qm.Range(gte=value, lte=value)
                    )
                )
            else:
                conditions.append(
                    qm.FieldCondition(
                        key=clause.field, match=qm.MatchValue(value=value)
                    )
                )
        return qm.Filter(must=conditions)

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )

    # -- writes and point reads ----------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        if not records:
            return
        points: list[qm.PointStruct] = []
        for record in records:
            store_id = self.store_id(record.id)
            points.append(
                qm.PointStruct(
                    id=store_id,
                    vector={
                        self._schema.vector_field: [
                            float(v) for v in record.vector or ()
                        ]
                    },
                    payload=self._payload(record, store_id),
                )
            )
        client = await self._open()
        with self._errors():
            await client.upsert(collection_name=self._schema.name, points=points)

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        if not ids:
            return []
        client = await self._open()
        with self._errors():
            points = await client.retrieve(
                collection_name=self._schema.name,
                ids=[self.store_id(key) for key in ids],
                with_payload=True,
                with_vectors=include_vectors,
            )
        by_store_id = {
            str(point.id): self._record(
                point.id, point.payload, point.vector, vectors=include_vectors
            )
            for point in points
        }
        return [
            by_store_id[self.store_id(key)]
            for key in ids
            if self.store_id(key) in by_store_id
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        client = await self._open()
        with self._errors():
            await client.delete(
                collection_name=self._schema.name,
                points_selector=qm.PointIdsList(
                    points=[self.store_id(key) for key in ids]
                ),
            )

    # -- filter, page, count -------------------------------------------------

    async def _scroll(
        self,
        query_filter: qm.Filter | None,
        offset: int | str | None,
        limit: int,
    ) -> tuple[list[Record], int | str | None]:
        client = await self._open()
        with self._errors():
            points, next_offset = await client.scroll(
                collection_name=self._schema.name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        records = [self._record(point.id, point.payload) for point in points]
        if next_offset is None or isinstance(next_offset, int | str):
            return records, next_offset
        return records, str(next_offset)

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        query_filter = self.build_filter(record_filter)
        results: list[Record] = []
        offset: int | str | None = None
        while True:
            page_limit = DEFAULT_PAGE_SIZE
            if limit is not None:
                remaining = limit - len(results)
                if remaining <= 0:
                    break
                page_limit = min(page_limit, remaining)
            records, offset = await self._scroll(query_filter, offset, page_limit)
            results.extend(records)
            if offset is None:
                break
        return results

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        offset: int | str | None = None
        if cursor is not None:
            offset = int(cursor) if cursor.isdigit() else cursor
        records, next_offset = await self._scroll(None, offset, limit)
        return Page(
            records=tuple(records),
            next_cursor=None if next_offset is None else str(next_offset),
        )

    async def delete_where(self, record_filter: RecordFilter) -> int:
        query_filter = self.build_filter(record_filter) or qm.Filter()
        client = await self._open()
        with self._errors():
            matched = await client.count(
                collection_name=self._schema.name,
                count_filter=query_filter,
                exact=True,
            )
            if matched.count == 0:
                return 0
            await client.delete(
                collection_name=self._schema.name,
                points_selector=qm.FilterSelector(filter=query_filter),
            )
        return int(matched.count)

    async def count(self) -> int:
        if not await self.collection_exists():
            return 0
        client = await self._open()
        with self._errors():
            result = await client.count(collection_name=self._schema.name, exact=True)
        return int(result.count)

    # -- search --------------------------------------------------------------

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        self._check_vector(query.vector)
        if query.top_k <= 0:
            return []
        query_filter = self.build_filter(query.filter)
        client = await self._open()
        with self._errors():
            response = await client.query_points(
                collection_name=self._schema.name,
                query=[float(v) for v in query.vector],
                using=self._schema.vector_field,
                query_filter=query_filter,
                limit=query.top_k,
                with_payload=True,
                with_vectors=False,
            )
        return [
            ScoredRecord(
                record=self._record(point.id, point.payload),
                score=clamp_similarity(point.score),
                kind=ScoreKind.SIMILARITY,
            )
            for point in response.points
        ]

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Dense and ``MatchText`` legs fused by the server with RRF."""
        self._check_vector(query.vector)
        self._schema.field(query.text_field)
        if query.top_k <= 0:
            return []
        tokens = tokenize(query.text)
        prefetch_limit = max(query.top_k * 4, 20)
        base_filter = self.build_filter(query.filter)
        prefetch: list[qm.Prefetch] = [
            qm.Prefetch(
                query=[float(v) for v in query.vector],
                using=self._schema.vector_field,
                filter=base_filter,
                limit=prefetch_limit,
            )
        ]
        if tokens:
            should: list[qm.Condition] = [
                qm.FieldCondition(key=query.text_field, match=qm.MatchText(text=token))
                for token in tokens
            ]
            text_filter = qm.Filter(should=should)
            if base_filter is not None:
                text_filter = qm.Filter(must=[base_filter, text_filter])
            prefetch.append(qm.Prefetch(filter=text_filter, limit=prefetch_limit))
        client = await self._open()
        with self._errors():
            response = await client.query_points(
                collection_name=self._schema.name,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=query.top_k,
                with_payload=True,
                with_vectors=False,
            )
        return [
            ScoredRecord(
                record=self._record(point.id, point.payload),
                score=float(point.score),
                kind=ScoreKind.FUSED,
            )
            for point in response.points
        ]
