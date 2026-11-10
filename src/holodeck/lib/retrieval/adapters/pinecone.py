"""Pinecone adapter on the native ``pinecone`` client (10.x, asyncio).

Stored layout
-------------
The layout is what Semantic Kernel's ``PineconeCollection`` wrote, so indexes
created before the move stay readable:

* one index per collection, a serverless dense index with the ``cosine``
  metric and ``schema.dimensions`` dimensions (default cloud ``aws``, region
  ``us-east-1``);
* the HoloDeck record ID is the Pinecone vector ID, unchanged;
* every data field is a metadata key of the same name;
* one namespace per adapter (default: the default namespace).

Client 10 creates indexes from a field schema. The adapter declares the single
reserved dense field ``_values``, which Pinecone stores exactly like an index
made with the pre-10 ``dimension``/``metric`` arguments, so the vector
operations (``upsert``, ``fetch``, ``query``, ``list``, ``fetch_by_metadata``)
serve it and so do indexes created by older clients.

Index names
-----------
Pinecone index names are 1 to 45 characters of lowercase letters, digits and
hyphens, and cannot start or end with a hyphen. HoloDeck's default collection
names contain underscores (``documents_dim1536``), which Pinecone rejects, so
``normalize_index_name`` maps a name deterministically: a name Pinecone already
accepts is unchanged; otherwise it is lower-cased, every other character run
becomes one hyphen, and a name over 45 characters keeps its head plus a hash of
the full name. Names that differ only by ``_`` and ``-`` therefore share an
index.

Enumeration and filtering (FR-014)
----------------------------------
Pinecone has no scan API. The adapter uses three operations:

* ``find``: the SDK's ``fetch_by_metadata`` with an exact-match filter, paged
  until the filter is exhausted. Records come back with values and metadata, so
  the adapter drops the values. ``fetch_by_metadata`` is a serverless data
  plane route; when the endpoint answers 404 (Pinecone Local 1.0 and older
  emulators do not serve it) the adapter remembers that and falls back to the
  scan below. An empty filter also takes the scan, because the route rejects
  empty filters.
* scan: list IDs page by page (``list_paginated``, serverless only), ``fetch``
  each page and filter on the client. It is exact but reads the whole
  namespace, so it costs one list call per 100 IDs plus one fetch per batch.
  ``list_page`` uses the same list and fetch calls, one page at a time.
* ``delete_where``: the base-class default, which is ``find`` followed by delete
  by ID. Pinecone's delete-by-filter takes no count and not every serverless
  index accepts it, so the adapter does not push it down. The count returned is
  exact.

Record IDs carry no source prefix that the adapter could list on: the tools
choose them, and a prefix match can also match another source.

Consistency
-----------
Pinecone reads are eventually consistent. After a write the adapter polls
``fetch`` for the last record it wrote (up to ``VISIBILITY_TIMEOUT`` seconds,
then logs a warning and returns) so that ``get``, ``find`` and ``list_page``
issued next see it. Floats may come back with fewer digits (Pinecone Local
keeps about fifteen significant digits), so the wait compares numbers within a
relative 1e-9. Fetch visibility does not promise that ``query`` or
``describe_index_stats`` (``count``) have caught up; the service documents that
they can lag. Pinecone Local 1.0 is consistent at once, so the wait ends on its
first poll.

Null values
-----------
Pinecone metadata cannot hold null. A ``None`` field is not written, and a
field missing on read is restored to its type default (``""``, ``0``, ``0.0``,
``False``). Filters on ``None`` raise ``SchemaError``.

Not supported
-------------
Lexical and hybrid search: HoloDeck fuses vector and BM25 results itself.

Lifecycle
---------
One ``AsyncPinecone`` control client and one ``AsyncIndex`` data handle per
adapter, both built on first use and released by ``close``. ``close`` is safe
to repeat and the adapter is usable again afterwards.

Errors
------
SDK errors become ``StoreConnectionError`` (or ``SchemaError`` for rejected
input). The API key is removed from every message the adapter builds.

Testing seam
------------
``_describe``, ``_create_index`` and ``_drop_index`` are the only control plane
calls. Pinecone Local 1.0 speaks the pre-10 control plane, so the contract
suite subclasses the adapter and replaces those three; every data plane call
still goes through the SDK.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Protocol, TypedDict

from pinecone import (
    AsyncPinecone,
    DenseVectorField,
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    NotFoundError,
    PineconeError,
    QueryResponse,
    Vector,
)
from pinecone.errors import ConflictError, PineconeValueError

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
    FieldSpec,
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

PROVIDER = "pinecone"
EXTRA = "pinecone"

DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"

VECTOR_FIELD = "_values"
"""Reserved schema field that makes a document index a plain vector index."""

MAX_INDEX_NAME = 45
UPSERT_MAX_RECORDS = 1000
UPSERT_MAX_BYTES = 1_500_000
FETCH_MAX_IDS = 100
FETCH_MAX_ID_CHARS = 4000
DELETE_MAX_IDS = 1000
LIST_MAX_IDS = 100
QUERY_MAX_TOP_K = 10_000
FETCH_BY_METADATA_MAX = 10_000
MAX_ID_LENGTH = 512
READY_TIMEOUT = 300
DELETE_TIMEOUT = 300
VISIBILITY_TIMEOUT = 30.0
_POLL_START = 0.05
_POLL_MAX = 1.0
_DEFAULT_NAMESPACE_KEYS = ("", "__default__")

_VALID_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

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

_DEFAULTS: Final[dict[FieldType, FieldValue]] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

_ACCEPTED_KEYS = (
    "api_key",
    "connection_string",
    "namespace",
    "host",
    "index_host",
    "cloud",
    "region",
    "spec",
    "index_name",
    "collection_name",
    "use_grpc",
)


# -- connection settings -------------------------------------------------------------


class PineconeConnectionParams(TypedDict, total=False):
    """Parameters read from a Pinecone connection string.

    Attributes:
        api_key: Pinecone API key.
        namespace: Namespace for the records.
    """

    api_key: str | None
    namespace: str | None


def parse_pinecone_connection_string(
    connection_string: str,
) -> PineconeConnectionParams:
    """Parse a Pinecone connection string into connection parameters.

    Supported formats:

    * a bare API key: ``pc-abc123`` (anything not starting with ``pinecone://``)
    * ``pinecone://api_key``
    * ``pinecone://api_key@namespace``
    * ``pinecone://api_key/namespace`` (the path wins over ``@namespace``)

    The key keeps its case.

    Raises:
        ValueError: The connection string is empty.
    """
    if not connection_string:
        raise ValueError("Connection string cannot be empty")
    scheme = "pinecone://"
    if not connection_string.startswith(scheme):
        return {"api_key": connection_string}

    authority, _, path = connection_string[len(scheme) :].partition("/")
    params: PineconeConnectionParams = {}
    if "@" in authority:
        api_key, _, namespace = authority.rpartition("@")
        params["api_key"] = api_key
        if namespace:
            params["namespace"] = namespace
    elif authority:
        params["api_key"] = authority
    if path:
        params["namespace"] = path
    return params


@dataclass(frozen=True, slots=True)
class PineconeSettings:
    """Connection settings of one adapter.

    Attributes:
        api_key: Pinecone API key. Never shown in ``repr``.
        namespace: Namespace of the records. Empty for the default namespace.
        cloud: Cloud of a created serverless index.
        region: Region of a created serverless index.
        host: Control plane URL, for a gateway or a private endpoint.
        index_host: Data plane URL. Skips the host lookup of the index.
        index_name: Index name that replaces the collection name.
    """

    api_key: str = field(repr=False)
    namespace: str = ""
    cloud: str = DEFAULT_CLOUD
    region: str = DEFAULT_REGION
    host: str | None = None
    index_host: str | None = None
    index_name: str | None = None


def _text(connection: Mapping[str, object], key: str) -> str | None:
    value = connection.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{key}' must be a string."
        )
    return value or None


def _serverless_spec(connection: Mapping[str, object]) -> tuple[str, str]:
    """Return ``(cloud, region)`` from ``cloud``/``region`` or a serverless ``spec``."""
    cloud = _text(connection, "cloud")
    region = _text(connection, "region")
    spec = connection.get("spec")
    if spec is not None:
        serverless = spec.get("serverless") if isinstance(spec, Mapping) else None
        if not isinstance(serverless, Mapping):
            raise StoreConnectionError(
                PROVIDER,
                "Connection setting 'spec' must look like "
                "{serverless: {cloud: aws, region: us-east-1}}.",
            )
        spec_cloud, spec_region = serverless.get("cloud"), serverless.get("region")
        if isinstance(spec_cloud, str):
            cloud = cloud or spec_cloud
        if isinstance(spec_region, str):
            region = region or spec_region
    return cloud or DEFAULT_CLOUD, region or DEFAULT_REGION


def settings_from_connection(connection: Mapping[str, object]) -> PineconeSettings:
    """Build settings from the database settings of an agent.

    Accepted keys: ``connection_string`` (see ``parse_pinecone_connection_string``),
    ``api_key`` (falls back to ``PINECONE_API_KEY``), ``namespace``, ``host``,
    ``index_host``, ``cloud``, ``region``, ``spec``, ``index_name`` and
    ``collection_name`` (the schema already carries the latter), and
    ``use_grpc`` (ignored: the async client always uses HTTP).

    Raises:
        StoreConnectionError: A key is unknown or mistyped, or no API key is set.
    """
    unknown = sorted(set(connection).difference(_ACCEPTED_KEYS))
    if unknown:
        raise StoreConnectionError(
            PROVIDER,
            f"Unknown connection settings {unknown}. "
            f"Accepted: {', '.join(_ACCEPTED_KEYS)}.",
        )
    api_key: str | None = None
    namespace: str | None = None
    connection_string = _text(connection, "connection_string")
    if connection_string is not None:
        try:
            parsed = parse_pinecone_connection_string(connection_string)
        except ValueError as error:
            raise StoreConnectionError(PROVIDER, str(error)) from None
        api_key = parsed.get("api_key")
        namespace = parsed.get("namespace")
    api_key = (
        _text(connection, "api_key") or api_key or os.environ.get("PINECONE_API_KEY")
    )
    if not api_key:
        raise StoreConnectionError(
            PROVIDER,
            "No API key. Set api_key or connection_string in the database "
            "settings, or the PINECONE_API_KEY environment variable.",
        )
    namespace = _text(connection, "namespace") or namespace or ""
    if len(namespace) > MAX_ID_LENGTH or not namespace.isascii() or "\0" in namespace:
        raise StoreConnectionError(
            PROVIDER,
            "Namespace must be ASCII, without NUL, "
            f"at most {MAX_ID_LENGTH} characters.",
        )
    if connection.get("use_grpc"):
        logger.warning("Pinecone setting use_grpc is ignored; the client uses HTTP.")
    cloud, region = _serverless_spec(connection)
    return PineconeSettings(
        api_key=api_key,
        namespace=namespace,
        cloud=cloud,
        region=region,
        host=_text(connection, "host"),
        index_host=_text(connection, "index_host"),
        index_name=_text(connection, "index_name"),
    )


def normalize_index_name(name: str) -> str:
    """Return the Pinecone index name for a collection name. See the module docs."""
    if len(name) <= MAX_INDEX_NAME and _VALID_NAME.match(name):
        return name
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", name.lower())).strip("-")
    if slug and len(slug) <= MAX_INDEX_NAME:
        return slug
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    head = slug[: MAX_INDEX_NAME - len(digest) - 1].strip("-")
    return f"{head}-{digest}" if head else f"holodeck-{digest}"


# -- control and data plane seams ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexInfo:
    """What the adapter needs to know about an existing index."""

    host: str
    dimension: int | None = None
    metric: str | None = None


class _DataPlane(Protocol):
    """The ``AsyncIndex`` methods the adapter calls."""

    async def upsert(self, *, vectors: Sequence[Vector], namespace: str = "") -> object:
        """Write vectors."""
        ...

    async def fetch(self, *, ids: Sequence[str], namespace: str = "") -> FetchResponse:
        """Read vectors by ID."""
        ...

    async def fetch_by_metadata(
        self,
        *,
        filter: Mapping[str, object],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> FetchByMetadataResponse:
        """Read one page of the vectors that match a filter."""
        ...

    async def delete(self, *, ids: Sequence[str], namespace: str = "") -> None:
        """Delete vectors by ID."""
        ...

    async def list_paginated(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
    ) -> ListResponse:
        """List one page of vector IDs."""
        ...

    async def query(
        self,
        *,
        top_k: int,
        vector: Sequence[float],
        namespace: str = "",
        filter: Mapping[str, object] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> QueryResponse:
        """Return the nearest vectors."""
        ...

    async def describe_index_stats(self) -> DescribeIndexStatsResponse:
        """Return the vector counts."""
        ...

    async def close(self) -> None:
        """Release the HTTP client."""
        ...


# -- the adapter --------------------------------------------------------------------


def _valid_id(record_id: str) -> bool:
    return (
        0 < len(record_id) <= MAX_ID_LENGTH
        and record_id.isascii()
        and "\0" not in record_id
    )


def _coerce(field_type: FieldType, value: object) -> FieldValue:
    if isinstance(value, bool):
        return value if field_type is FieldType.BOOLEAN else _DEFAULTS[field_type]
    if field_type is FieldType.INTEGER:
        return int(value) if isinstance(value, (int, float, str)) else 0
    if field_type is FieldType.FLOAT:
        return float(value) if isinstance(value, (int, float, str)) else 0.0
    if field_type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


def _stored(spec: FieldSpec, value: FieldValue) -> str | int | float | bool:
    """Return ``value`` in the type Pinecone stores for ``spec``."""
    if spec.type is FieldType.INTEGER:
        return int(value) if value is not None else 0
    if spec.type is FieldType.FLOAT:
        return float(value) if value is not None else 0.0
    if spec.type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


class PineconeAdapter(BaseVectorStoreAdapter):
    """One Pinecone index (and namespace) as a HoloDeck collection."""

    def __init__(self, schema: CollectionSchema, settings: PineconeSettings) -> None:
        super().__init__(schema, _CAPABILITIES)
        for spec in schema.fields:
            if spec.name.startswith("$"):
                raise SchemaError(
                    f"Field name '{spec.name}' starts with '$', which Pinecone "
                    "reserves for filter operators."
                )
        self._settings = settings
        self._index_name = normalize_index_name(settings.index_name or schema.name)
        self._namespace = settings.namespace
        self._lock = asyncio.Lock()
        self._client: AsyncPinecone | None = None
        self._index: _DataPlane | None = None
        self._metadata_fetch = True

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> PineconeAdapter:
        """Build the adapter. No request is sent until first use."""
        return cls(schema, settings_from_connection(connection))

    @property
    def index_name(self) -> str:
        """Name of the Pinecone index that holds the collection."""
        return self._index_name

    # -- errors --------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        key = self._settings.api_key
        return text.replace(key, "***") if key else text

    @contextmanager
    def _translated(self) -> Iterator[None]:
        """Turn SDK errors into HoloDeck errors without the API key."""
        try:
            yield
        except RetrievalError:
            raise
        except PineconeValueError as error:
            text = str(error)
            cause: BaseException | None = (
                None if self._settings.api_key in text else error
            )
            raise SchemaError(self._scrub(text)) from cause
        except PineconeError as error:
            text = str(error)
            cause = None if self._settings.api_key in text else error
            raise StoreConnectionError(
                PROVIDER, self._scrub(f"{type(error).__name__}: {text}")
            ) from cause

    # -- control plane (the testing seam) -------------------------------------------

    def _pinecone(self) -> AsyncPinecone:
        if self._client is None:
            self._client = AsyncPinecone(
                api_key=self._settings.api_key, host=self._settings.host
            )
        return self._client

    async def _describe(self) -> IndexInfo | None:
        """Return the index, or None when it does not exist."""
        try:
            model = await self._pinecone().indexes.describe(self._index_name)
        except NotFoundError:
            return None
        if model.host is None:
            raise StoreConnectionError(
                PROVIDER, f"Index '{self._index_name}' has no host yet."
            )
        dense = [
            declared
            for declared in model.schema.fields.values()
            if isinstance(declared, DenseVectorField)
        ]
        only = dense[0] if len(dense) == 1 else None
        return IndexInfo(
            host=self._settings.index_host or model.host,
            dimension=only.dimension if only else None,
            metric=only.metric if only else None,
        )

    async def _create_index(self) -> None:
        """Create the serverless index and wait until it is ready."""
        try:
            await self._pinecone().indexes.create(
                name=self._index_name,
                schema={
                    "fields": {
                        VECTOR_FIELD: {
                            "type": "dense_vector",
                            "dimension": self._schema.dimensions,
                            "metric": self._schema.distance.value,
                        }
                    }
                },
                deployment={
                    "deployment_type": "managed",
                    "cloud": self._settings.cloud,
                    "region": self._settings.region,
                },
                timeout=READY_TIMEOUT,
            )
        except ConflictError:
            logger.debug("Index %s was created concurrently", self._index_name)

    async def _drop_index(self) -> None:
        """Delete the index and wait until it is gone."""
        try:
            await self._pinecone().indexes.delete(
                self._index_name, timeout=DELETE_TIMEOUT
            )
        except NotFoundError:
            return

    # -- lifecycle -----------------------------------------------------------------

    async def _open_index(self) -> _DataPlane | None:
        """Return the data plane handle, or None when the index does not exist."""
        async with self._lock:
            if self._index is not None:
                return self._index
            info = await self._describe()
            if info is None:
                return None
            self._index = await self._pinecone().index(host=info.host)
            return self._index

    async def _require_index(self) -> _DataPlane:
        index = await self._open_index()
        if index is None:
            await self.ensure_collection()
            index = await self._open_index()
        if index is None:
            raise StoreConnectionError(
                PROVIDER, f"Index '{self._index_name}' is not available."
            )
        return index

    async def _release_index(self) -> None:
        async with self._lock:
            index, self._index = self._index, None
        if index is not None:
            await index.close()

    async def collection_exists(self) -> bool:
        with self._translated():
            return await self._describe() is not None

    async def ensure_collection(self) -> None:
        with self._translated():
            info = await self._describe()
            if info is None:
                await self._create_index()
                info = await self._describe()
                if info is None:
                    raise StoreConnectionError(
                        PROVIDER,
                        f"Index '{self._index_name}' was created but is not visible.",
                    )
            self._check_compatible(info)

    def _check_compatible(self, info: IndexInfo) -> None:
        if info.dimension is not None and info.dimension != self._schema.dimensions:
            raise IncompatibleCollectionError(
                f"Pinecone index '{self._index_name}' has {info.dimension} "
                f"dimensions; collection '{self._schema.name}' needs "
                f"{self._schema.dimensions}."
            )
        if info.metric is not None and info.metric != self._schema.distance.value:
            raise IncompatibleCollectionError(
                f"Pinecone index '{self._index_name}' uses the {info.metric} "
                f"metric; collection '{self._schema.name}' needs "
                f"{self._schema.distance.value}."
            )

    async def delete_collection(self) -> None:
        with self._translated():
            await self._release_index()
            if await self._describe() is not None:
                await self._drop_index()

    async def close(self) -> None:
        async with self._lock:
            index, self._index = self._index, None
            client, self._client = self._client, None
        try:
            if index is not None:
                await index.close()
        except PineconeError as error:
            logger.debug(
                "Closing the Pinecone index failed: %s", self._scrub(str(error))
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except PineconeError as error:
                    logger.debug(
                        "Closing the Pinecone client failed: %s",
                        self._scrub(str(error)),
                    )

    # -- conversion ----------------------------------------------------------------

    def _metadata(self, record: Record) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {}
        for spec in self._schema.fields:
            value = record.fields.get(spec.name)
            if value is not None:
                metadata[spec.name] = _stored(spec, value)
        return metadata

    def _record(
        self,
        record_id: str,
        metadata: Mapping[str, object] | None,
        values: Sequence[float] | None,
        *,
        vector: bool = False,
    ) -> Record:
        stored = metadata or {}
        fields: dict[str, FieldValue] = {}
        for spec in self._schema.fields:
            raw = stored.get(spec.name)
            fields[spec.name] = (
                _DEFAULTS[spec.type] if raw is None else _coerce(spec.type, raw)
            )
        return Record(
            id=record_id,
            fields=fields,
            vector=[float(v) for v in values or ()] if vector else None,
        )

    def _filter_expression(
        self, record_filter: RecordFilter | None
    ) -> dict[str, object] | None:
        """Return a Pinecone filter for exact matches, or None for no clauses."""
        if record_filter is None or not record_filter.all_of:
            return None
        clauses: list[dict[str, object]] = []
        for clause in record_filter.all_of:
            spec = self._schema.field(clause.field)
            value = clause.value
            if value is None:
                raise SchemaError(
                    f"Pinecone cannot store null, so field '{clause.field}' cannot "
                    "be filtered on None."
                )
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            ok = {
                FieldType.STRING: isinstance(value, str),
                FieldType.INTEGER: isinstance(value, int)
                and not isinstance(value, bool),
                FieldType.FLOAT: numeric,
                FieldType.BOOLEAN: isinstance(value, bool),
            }[spec.type]
            if not ok:
                raise SchemaError(
                    f"Filter value {value!r} does not fit {spec.type.value} field "
                    f"'{clause.field}'."
                )
            clauses.append({clause.field: {"$eq": value}})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )

    # -- write and point reads -----------------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        for record in records:
            if not _valid_id(record.id):
                raise SchemaError(
                    f"Record ID {record.id[:60]!r} is not a valid Pinecone vector ID: "
                    f"IDs are ASCII, without NUL, 1 to {MAX_ID_LENGTH} characters."
                )
        if not records:
            return
        with self._translated():
            index = await self._require_index()
            vectors = [
                Vector(
                    id=record.id,
                    values=[float(v) for v in record.vector or ()],
                    metadata=self._metadata(record),
                )
                for record in records
            ]
            for batch in _upsert_batches(vectors, self._schema.dimensions):
                await index.upsert(vectors=batch, namespace=self._namespace)
            await self._await_written(index, vectors[-1])

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        usable = [key for key in ids if _valid_id(key)]
        if not usable:
            return []
        with self._translated():
            index = await self._open_index()
            if index is None:
                return []
            found = await self._fetch(index, usable)
        return [
            self._record(
                key, found[key].metadata, found[key].values, vector=include_vectors
            )
            for key in usable
            if key in found
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        usable = list(dict.fromkeys(key for key in ids if _valid_id(key)))
        if not usable:
            return
        with self._translated():
            index = await self._open_index()
            if index is None:
                return
            for start in range(0, len(usable), DELETE_MAX_IDS):
                await index.delete(
                    ids=usable[start : start + DELETE_MAX_IDS],
                    namespace=self._namespace,
                )
            await self._await_deleted(index, usable[-1])

    async def count(self) -> int:
        with self._translated():
            index = await self._open_index()
            if index is None:
                return 0
            stats = await index.describe_index_stats()
        for key in (
            _DEFAULT_NAMESPACE_KEYS if not self._namespace else (self._namespace,)
        ):
            summary = stats.namespaces.get(key)
            if summary is not None:
                return int(summary.vector_count)
        return 0

    # -- filtered reads and enumeration -------------------------------------------

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        expression = self._filter_expression(record_filter)
        if limit is not None and limit <= 0:
            return []
        with self._translated():
            index = await self._open_index()
            if index is None:
                return []
            if expression is not None and self._metadata_fetch:
                try:
                    return await self._find_by_metadata(index, expression, limit)
                except NotFoundError:
                    logger.info(
                        "Pinecone fetch_by_metadata is not served at this endpoint; "
                        "find() now lists IDs and filters on the client."
                    )
                    self._metadata_fetch = False
            return await self._scan(index, record_filter, limit)

    async def _find_by_metadata(
        self, index: _DataPlane, expression: Mapping[str, object], limit: int | None
    ) -> list[Record]:
        records: list[Record] = []
        token: str | None = None
        while True:
            page_size = FETCH_BY_METADATA_MAX
            if limit is not None:
                page_size = min(page_size, limit - len(records))
            page = await index.fetch_by_metadata(
                filter=expression,
                namespace=self._namespace,
                limit=page_size,
                pagination_token=token,
            )
            for key, vector in page.vectors.items():
                records.append(self._record(key, vector.metadata, None))
            token = page.pagination.next if page.pagination is not None else None
            if token is None or (limit is not None and len(records) >= limit):
                return records if limit is None else records[:limit]

    async def _scan(
        self, index: _DataPlane, record_filter: RecordFilter, limit: int | None
    ) -> list[Record]:
        records: list[Record] = []
        async for ids in self._id_pages(index):
            for record in await self._load(index, ids):
                if record_filter.matches(record):
                    records.append(record)
                    if limit is not None and len(records) >= limit:
                        return records
        return records

    async def _id_pages(self, index: _DataPlane) -> AsyncIterator[list[str]]:
        token: str | None = None
        while True:
            page = await index.list_paginated(
                limit=LIST_MAX_IDS,
                pagination_token=token,
                namespace=self._namespace,
            )
            ids = [item.id for item in page.vectors if item.id is not None]
            if ids:
                yield ids
            token = page.pagination.next if page.pagination is not None else None
            if token is None:
                return

    async def _load(self, index: _DataPlane, ids: Sequence[str]) -> list[Record]:
        found = await self._fetch(index, ids)
        return [
            self._record(key, found[key].metadata, None) for key in ids if key in found
        ]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        with self._translated():
            index = await self._open_index()
            if index is None:
                return Page(records=())
            ids: list[str] = []
            token = cursor
            while len(ids) < limit:
                page = await index.list_paginated(
                    limit=min(LIST_MAX_IDS, limit - len(ids)),
                    pagination_token=token,
                    namespace=self._namespace,
                )
                ids.extend(item.id for item in page.vectors if item.id is not None)
                token = page.pagination.next if page.pagination is not None else None
                if token is None:
                    break
            records = await self._load(index, ids)
        return Page(records=tuple(records), next_cursor=token)

    # -- search --------------------------------------------------------------------

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        self._check_vector(query.vector)
        expression = self._filter_expression(query.filter)
        if query.top_k <= 0:
            return []
        with self._translated():
            index = await self._open_index()
            if index is None:
                return []
            response = await index.query(
                top_k=min(query.top_k, QUERY_MAX_TOP_K),
                vector=[float(v) for v in query.vector],
                namespace=self._namespace,
                filter=expression,
                include_metadata=True,
                include_values=False,
            )
        return [
            ScoredRecord(
                record=self._record(match.id, match.metadata, None),
                score=clamp_similarity(match.score),
                kind=ScoreKind.SIMILARITY,
            )
            for match in response.matches
        ]

    # -- batching and consistency ----------------------------------------------------

    async def _fetch(self, index: _DataPlane, ids: Sequence[str]) -> dict[str, Vector]:
        """Fetch ``ids`` in requests that fit the URL, keyed by ID."""
        found: dict[str, Vector] = {}
        batch: list[str] = []
        chars = 0
        for key in dict.fromkeys(ids):
            if batch and (
                len(batch) >= FETCH_MAX_IDS or chars + len(key) > FETCH_MAX_ID_CHARS
            ):
                found.update(await self._fetch_batch(index, batch))
                batch, chars = [], 0
            batch.append(key)
            chars += len(key)
        if batch:
            found.update(await self._fetch_batch(index, batch))
        return found

    async def _fetch_batch(
        self, index: _DataPlane, batch: Sequence[str]
    ) -> dict[str, Vector]:
        response = await index.fetch(ids=batch, namespace=self._namespace)
        return dict(response.vectors)

    async def _await_written(self, index: _DataPlane, last: Vector) -> None:
        """Wait until ``fetch`` returns the last written vector, or give up."""
        expected = last.metadata or {}

        def visible(found: Mapping[str, Vector]) -> bool:
            stored = found.get(last.id)
            return stored is not None and _same_metadata(
                expected, stored.metadata or {}
            )

        await self._poll(index, last.id, visible)

    async def _await_deleted(self, index: _DataPlane, last_id: str) -> None:
        """Wait until ``fetch`` no longer returns the last deleted ID, or give up."""
        await self._poll(index, last_id, lambda found: last_id not in found)

    async def _poll(
        self,
        index: _DataPlane,
        probe: str,
        done: Callable[[Mapping[str, Vector]], bool],
    ) -> None:
        deadline = time.monotonic() + VISIBILITY_TIMEOUT
        delay = _POLL_START
        while True:
            found = await self._fetch_batch(index, [probe])
            if done(found):
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    "Pinecone index %s did not show a write within %.0f s; "
                    "reads may lag behind it.",
                    self._index_name,
                    VISIBILITY_TIMEOUT,
                )
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, _POLL_MAX)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _same_metadata(sent: Mapping[str, object], stored: Mapping[str, object]) -> bool:
    """Compare metadata as written with metadata as read back.

    The service may keep fewer digits of a float (Pinecone Local keeps about
    fifteen significant digits), so numbers match within a relative 1e-9.
    """
    if sent.keys() != stored.keys():
        return False
    for key, value in sent.items():
        other = stored[key]
        left, right = _number(value), _number(other)
        if left is not None and right is not None:
            if not math.isclose(left, right, rel_tol=1e-9):
                return False
        elif value != other or isinstance(value, bool) != isinstance(other, bool):
            return False
    return True


def _upsert_batches(
    vectors: Sequence[Vector], dimensions: int
) -> Iterator[list[Vector]]:
    """Split vectors into requests under the record count and size limits."""
    batch: list[Vector] = []
    size = 0
    for vector in vectors:
        estimate = (
            len(vector.id)
            + dimensions * 12
            + len(json.dumps(vector.metadata or {}))
            + 64
        )
        if batch and (
            len(batch) >= UPSERT_MAX_RECORDS or size + estimate > UPSERT_MAX_BYTES
        ):
            yield batch
            batch, size = [], 0
        batch.append(vector)
        size += estimate
    if batch:
        yield batch
