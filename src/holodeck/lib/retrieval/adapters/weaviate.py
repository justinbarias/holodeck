"""Native Weaviate adapter on the asynchronous ``weaviate-client`` v4.

Stored layout
-------------
The layout is the one Semantic Kernel's ``WeaviateCollection`` produces
(its ``ensure_collection_exists`` with a cosine named vector). The SK bridge
could never create that collection, see "Compatibility" below.

* one collection (Weaviate "class") per HoloDeck collection; the name is
  ``weaviate_class_name(schema.name)``: the first letter upper-cased, every
  character outside ``[0-9A-Za-z_]`` replaced by ``_``. SK only upper-cased the
  first letter and so rejected names with ``-`` or ``.``. Names that differ
  only in the case of the first letter share one collection;
* one self-provided named vector, ``schema.vector_field`` (``embedding``), cosine
  distance, a flat index, no vectorizer. SK's default index kind is flat too;
* one property per data field: ``text``, ``int``, ``number`` or ``boolean``;
* the HoloDeck record ID is the object UUID. A HoloDeck ID that is a UUID is
  stored as the canonical UUID. Any other ID is stored as
  ``uuid5(namespace, id)`` with
  ``namespace = uuid5(NAMESPACE_URL, "holodeck/retrieval/<schema.name>")``
  (the Qdrant adapter's mapping) and the original ID goes into the
  ``holodeck_id`` text property. Reads return the HoloDeck ID. SK passed the ID
  to Weaviate unchanged and so accepted only UUIDs.

Property indexes follow the ``FieldSpec`` flags:

* ``full_text`` strings are word-tokenized and BM25-searchable, and have no
  filter index (``full_text`` wins when a field is also ``indexed``);
* ``indexed`` strings use ``field`` tokenization and a filter index only, so a
  filter is an exact, case-sensitive match. With SK's default word
  tokenization a filter on ``docs/a.md`` also matches ``docs/a.md.bak`` and a
  value such as ``a`` (a stop word) is rejected by the server;
* ``indexed`` numbers and booleans get a filter index;
* every other field has no index. SK left all properties filterable and
  searchable.

Filters
-------
A filter is a conjunction of ``equal`` clauses built from typed values, never
from strings. A clause on a field that is not ``indexed``, ``None`` values and
full-text strings cannot be matched exactly and raise ``SchemaError``. ``find``
and ``delete_where`` check every returned record against the filter, so a
collection made elsewhere with word
tokenization still gets exact matches there. Vector and hybrid searches push
the filter to the server and rely on the field tokenization above.

Weaviate limits one query to ``offset + limit <= QUERY_MAXIMUM_RESULTS``
(10,000 by default), so ``find`` fails for a filter that matches more records.
``list_page`` uses the cursor API (``after``) and has no such limit.

Search
------
* vector: ``near_vector`` on the named vector. The score is
  ``clamp_similarity(1 - cosine distance)``;
* lexical: ``bm25`` on one full-text field, scored by the BM25 score;
* hybrid: Weaviate's native ``hybrid`` (vector plus BM25 on the full-text
  field, relative-score fusion, ``alpha`` 0.5). Scores are Weaviate's fused
  values. A query text with no tokens searches by vector only.

Compatibility
-------------
Against Weaviate 1.40 the SK path fails: ``SKWeaviateAdapter.ensure_collection``
raises ("Distance function COSINE_SIMILARITY is not supported by Weaviate"; SK's
connector accepts only ``COSINE_DISTANCE``), the first upsert then creates a
collection with a default, unnamed vector and fails again, and non-UUID IDs fail
at the server. No HoloDeck collection written through SK exists. With the
distance function corrected SK writes the layout above, which this adapter reads
and writes (``tests/integration/retrieval/test_weaviate_compat.py``).

Lifecycle
---------
The adapter owns one ``WeaviateAsyncClient``. It is built and connected on first
use and closed in ``close``; the next call after ``close`` builds a new one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlparse

from grpc.aio import AioRpcError
from weaviate import (
    WeaviateAsyncClient,
    use_async_with_local,
    use_async_with_weaviate_cloud,
)
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.data import DataObject
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, FilterReturn, MetadataQuery
from weaviate.collections.classes.config import (
    VectorIndexConfigFlat,
    VectorIndexConfigHNSW,
)
from weaviate.collections.classes.internal import Object
from weaviate.collections.classes.types import WeaviateField
from weaviate.collections.collection import CollectionAsync
from weaviate.exceptions import (
    AuthenticationFailedError,
    UnexpectedStatusCodeError,
    WeaviateBaseError,
    WeaviateClosedClientError,
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateInvalidInputError,
    WeaviateStartUpError,
    WeaviateTimeoutError,
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
    FieldSpec,
    FieldType,
    FieldValue,
    HybridQuery,
    LexicalQuery,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
    clamp_similarity,
    normalize_filter_value,
)

logger = logging.getLogger(__name__)

PROVIDER: Final = "weaviate"
EXTRA: Final = "weaviate"

ID_PROPERTY: Final = "holodeck_id"
"""Text property that keeps the HoloDeck ID when it is not a UUID."""

UPSERT_BATCH: Final = 500
GET_BATCH: Final = 500
DELETE_BATCH: Final = 500
MAX_CLASS_NAME: Final = 255
HYBRID_ALPHA: Final = 0.5
"""Weight of the vector leg in hybrid search; 0.5 weighs both legs equally."""

DEFAULT_LOCAL_PORT: Final = 8080
DEFAULT_LOCAL_GRPC_PORT: Final = 50051
_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
_RESERVED_PROPERTIES: Final = frozenset({"id", "_id", "vector", "_additional"})
_PROPERTY_NAME = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")

CAPABILITIES: Final = Capabilities(
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
            Operation.LEXICAL_SEARCH,
            Operation.HYBRID_SEARCH,
        }
    ),
)

_DEFAULTS: Final[dict[FieldType, str | int | float | bool]] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

_DATA_TYPES: Final[dict[FieldType, DataType]] = {
    FieldType.STRING: DataType.TEXT,
    FieldType.INTEGER: DataType.INT,
    FieldType.FLOAT: DataType.NUMBER,
    FieldType.BOOLEAN: DataType.BOOL,
}

_CONNECTION_KEYS: Final = (
    "url",
    "api_key",
    "connection_string",
    "local_host",
    "local_port",
    "local_grpc_port",
    "collection_name",
)
"""Connection keys the adapter accepts.

``collection_name`` is read by the tool schemas and only names the collection.
"""


# -- connection settings -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeaviateSettings:
    """Where and how one adapter connects.

    Exactly one of ``cluster`` (Weaviate Cloud) and ``host`` (a server reached
    over plain HTTP and gRPC) is set.

    Attributes:
        cluster: Weaviate Cloud cluster host name.
        host: Server host name.
        port: HTTP port of ``host``.
        grpc_port: gRPC port of ``host``.
        api_key: API key. Never shown in ``repr``.
    """

    cluster: str | None = None
    host: str | None = None
    port: int = DEFAULT_LOCAL_PORT
    grpc_port: int = DEFAULT_LOCAL_GRPC_PORT
    api_key: str | None = field(default=None, repr=False)


def _text(connection: Mapping[str, object], key: str) -> str | None:
    value = connection.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{key}' must be a string."
        )
    return value.strip() or None


def _port(key: str, value: object) -> int:
    if isinstance(value, bool):
        number = -1
    elif isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value)
    else:
        number = -1
    if not 1 <= number <= 65535:
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{key}' must be a port from 1 to 65535."
        )
    return number


def _cluster_host(url: str) -> str:
    """Return the Weaviate Cloud host name of a cluster URL or bare host name."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme != "https":
        raise StoreConnectionError(
            PROVIDER,
            "Weaviate Cloud needs an https URL. For a server on plain HTTP set "
            "local_host, or connection_string to an http:// URL.",
        )
    if parsed.port not in (None, 443) or not parsed.hostname:
        raise StoreConnectionError(
            PROVIDER,
            "Weaviate Cloud URLs use port 443 and need a host name. For another "
            "port set local_host and local_port.",
        )
    return parsed.hostname


def _from_connection_string(
    connection_string: str,
) -> tuple[str | None, str | None, int | None]:
    """Return ``(cluster, host, port)`` for a connection string URL."""
    parsed = urlparse(connection_string)
    if parsed.scheme == "https":
        return _cluster_host(connection_string), None, None
    if parsed.scheme == "http" and parsed.hostname:
        return None, parsed.hostname, parsed.port
    raise StoreConnectionError(
        PROVIDER,
        "connection_string must be an https:// Weaviate Cloud URL or an "
        "http://host[:port] URL.",
    )


def settings_from_connection(connection: Mapping[str, object]) -> WeaviateSettings:
    """Build settings from the database settings of an agent.

    Accepted keys: ``url`` (Weaviate Cloud, ``https://`` or a bare host name),
    ``api_key``, ``local_host``, ``local_port``, ``local_grpc_port``,
    ``connection_string`` (an ``https://`` cloud URL, or ``http://host[:port]``
    for a server) and ``collection_name`` (the schema already carries it).
    When none of ``url``, ``local_host`` and ``connection_string`` is set, the
    environment variables ``WEAVIATE_URL`` or ``WEAVIATE_LOCAL_HOST``,
    ``WEAVIATE_LOCAL_PORT`` and ``WEAVIATE_LOCAL_GRPC_PORT`` decide.
    ``WEAVIATE_API_KEY`` is the fallback for ``api_key``. Semantic Kernel read
    the same variables.

    Raises:
        StoreConnectionError: A key is unknown or mistyped, the target is
            missing or ambiguous, or a Weaviate Cloud URL has no API key.
    """
    unknown = sorted(set(connection).difference(_CONNECTION_KEYS))
    if unknown:
        raise StoreConnectionError(
            PROVIDER,
            f"Unknown connection settings {unknown}. "
            f"Accepted: {', '.join(_CONNECTION_KEYS)}.",
        )
    url = _text(connection, "url")
    local_host = _text(connection, "local_host")
    connection_string = _text(connection, "connection_string")
    port_value = connection.get("local_port")
    grpc_value = connection.get("local_grpc_port")

    if url is None and local_host is None and connection_string is None:
        url = os.environ.get("WEAVIATE_URL") or None
        local_host = os.environ.get("WEAVIATE_LOCAL_HOST") or None
        if port_value is None:
            port_value = os.environ.get("WEAVIATE_LOCAL_PORT") or None
        if grpc_value is None:
            grpc_value = os.environ.get("WEAVIATE_LOCAL_GRPC_PORT") or None

    cluster: str | None = None
    host: str | None = None
    port: int | None = None
    if connection_string is not None:
        cluster, host, port = _from_connection_string(connection_string)
    if url is not None:
        cluster = _cluster_host(url)
    if local_host is not None:
        host = local_host
    if port_value is not None:
        port = _port("local_port", port_value)
    grpc_port = (
        _port("local_grpc_port", grpc_value)
        if grpc_value is not None
        else DEFAULT_LOCAL_GRPC_PORT
    )

    if cluster is not None and host is not None:
        raise StoreConnectionError(
            PROVIDER,
            "Set either a Weaviate Cloud url or local_host, not both.",
        )
    if cluster is None and host is None:
        raise StoreConnectionError(
            PROVIDER,
            "No Weaviate server. Set url (Weaviate Cloud) or local_host "
            "(a server), or the WEAVIATE_URL or WEAVIATE_LOCAL_HOST "
            "environment variable.",
        )
    api_key = _text(connection, "api_key") or os.environ.get("WEAVIATE_API_KEY") or None
    if cluster is not None:
        if not api_key:
            raise StoreConnectionError(
                PROVIDER,
                "No API key. Set api_key in the database settings, or the "
                "WEAVIATE_API_KEY environment variable.",
            )
        if port is not None or grpc_value is not None:
            raise StoreConnectionError(
                PROVIDER,
                "local_port and local_grpc_port apply to local_host, not to "
                "a Weaviate Cloud url.",
            )
        return WeaviateSettings(cluster=cluster, api_key=api_key)
    port = port or DEFAULT_LOCAL_PORT
    if port == grpc_port:
        raise StoreConnectionError(
            PROVIDER,
            f"local_port and local_grpc_port must differ; both are {port}.",
        )
    return WeaviateSettings(host=host, port=port, grpc_port=grpc_port, api_key=api_key)


def create_weaviate_client(settings: WeaviateSettings) -> WeaviateAsyncClient:
    """Build the async client an adapter owns. It is not connected yet."""
    if settings.cluster is not None and settings.api_key:
        return use_async_with_weaviate_cloud(
            cluster_url=settings.cluster,
            auth_credentials=Auth.api_key(settings.api_key),
        )
    return use_async_with_local(
        host=settings.host or "localhost",
        port=settings.port,
        grpc_port=settings.grpc_port,
        auth_credentials=settings.api_key,
    )


# -- names and conversions -----------------------------------------------------


def weaviate_class_name(name: str) -> str:
    """Return the Weaviate collection name for a HoloDeck collection name.

    Weaviate names match ``[A-Z][_0-9A-Za-z]*`` and have at most 255
    characters. The mapping is deterministic and leaves a valid name that
    starts with an upper-case letter unchanged.
    """
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not cleaned[:1].isalpha():
        cleaned = f"Collection_{cleaned}"
    cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > MAX_CLASS_NAME:
        digest = hashlib.sha256(name.encode()).hexdigest()[:8]
        cleaned = f"{cleaned[: MAX_CLASS_NAME - len(digest) - 1]}_{digest}"
    return cleaned


def property_for(spec: FieldSpec) -> Property:
    """Return the Weaviate property that stores ``spec``. See the module docs."""
    data_type = _DATA_TYPES[spec.type]
    if spec.type is not FieldType.STRING:
        return Property(
            name=spec.name, data_type=data_type, index_filterable=spec.indexed
        )
    if spec.full_text:
        return Property(
            name=spec.name,
            data_type=data_type,
            index_filterable=False,
            index_searchable=True,
        )
    if spec.indexed:
        return Property(
            name=spec.name,
            data_type=data_type,
            index_filterable=True,
            index_searchable=False,
            tokenization=Tokenization.FIELD,
        )
    return Property(
        name=spec.name,
        data_type=data_type,
        index_filterable=False,
        index_searchable=False,
    )


def _id_property() -> Property:
    return Property(
        name=ID_PROPERTY,
        data_type=DataType.TEXT,
        index_filterable=False,
        index_searchable=False,
    )


def _coerce(field_type: FieldType, value: object) -> FieldValue:
    """Convert a stored property to the schema type."""
    if value is None:
        return None
    if field_type is FieldType.BOOLEAN:
        return bool(value)
    if field_type is FieldType.STRING:
        return value if isinstance(value, str) else str(value)
    if isinstance(value, int | float | str):
        return float(value) if field_type is FieldType.FLOAT else int(value)
    raise SchemaError(f"Cannot read {value!r} as {field_type.value}.")


def _stored(spec: FieldSpec, value: FieldValue) -> str | int | float | bool:
    """Return ``value`` in the type Weaviate stores for ``spec``."""
    if value is None:
        return _DEFAULTS[spec.type]
    try:
        if spec.type is FieldType.INTEGER:
            return int(value)
        if spec.type is FieldType.FLOAT:
            return float(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(
            f"Field '{spec.name}' needs {spec.type.value}, got {value!r}."
        ) from error
    if spec.type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


def _dense_vector(vectors: Mapping[str, object], name: str) -> list[float] | None:
    """Return the named dense vector of an object, or None."""
    vector = vectors.get(name)
    if isinstance(vector, list) and all(
        isinstance(item, int | float) for item in vector
    ):
        return [float(item) for item in vector]
    return None


# -- adapter --------------------------------------------------------------------


class WeaviateAdapter(BaseVectorStoreAdapter):
    """One Weaviate collection through ``WeaviateAsyncClient``."""

    def __init__(self, schema: CollectionSchema, settings: WeaviateSettings) -> None:
        for spec in schema.fields:
            if (
                not _PROPERTY_NAME.match(spec.name)
                or spec.name in _RESERVED_PROPERTIES
                or spec.name == ID_PROPERTY
            ):
                raise SchemaError(
                    f"Field name '{spec.name}' is not a valid Weaviate property "
                    f"name (letters, digits and '_'; '{ID_PROPERTY}', 'id' and "
                    "'vector' are reserved)."
                )
        super().__init__(schema, CAPABILITIES)
        self._settings = settings
        self._class = weaviate_class_name(schema.name)
        self._namespace = uuid.uuid5(
            uuid.NAMESPACE_URL, f"holodeck/retrieval/{schema.name}"
        )
        self._client: WeaviateAsyncClient | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> WeaviateAdapter:
        """Build the adapter. No request is sent until first use.

        Raises:
            StoreConnectionError: The settings are invalid.
        """
        return cls(schema, settings_from_connection(connection))

    @property
    def collection_name(self) -> str:
        """Name of the Weaviate collection that holds the records."""
        return self._class

    # -- errors and lifecycle ---------------------------------------------------

    def _scrub(self, text: str) -> str:
        key = self._settings.api_key
        return text.replace(key, "***") if key else text

    @contextmanager
    def _errors(self) -> Iterator[None]:
        """Turn client errors into retrieval errors without the API key."""
        try:
            yield
        except RetrievalError:
            raise
        except (
            WeaviateConnectionError,
            WeaviateStartUpError,
            WeaviateTimeoutError,
            WeaviateGRPCUnavailableError,
            WeaviateClosedClientError,
            AuthenticationFailedError,
        ) as error:
            raise StoreConnectionError(
                PROVIDER, self._scrub(f"{type(error).__name__}: {error}")
            ) from None
        except UnexpectedStatusCodeError as error:
            text = self._scrub(f"{type(error).__name__}: {error}")
            if error.status_code in (401, 403):
                raise StoreConnectionError(PROVIDER, text) from None
            raise RetrievalError(f"Weaviate request failed. {text}") from None
        except WeaviateInvalidInputError as error:
            raise SchemaError(self._scrub(str(error))) from None
        except WeaviateBaseError as error:
            raise RetrievalError(
                self._scrub(f"Weaviate request failed. {type(error).__name__}: {error}")
            ) from None
        except AioRpcError as error:
            raise StoreConnectionError(
                PROVIDER, self._scrub(str(error.details()))
            ) from None
        except (ConnectionError, TimeoutError) as error:
            raise StoreConnectionError(PROVIDER, self._scrub(str(error))) from None

    async def _open(self) -> WeaviateAsyncClient:
        """Return the connected client, building it on first use."""
        async with self._lock:
            if self._client is not None:
                return self._client
            client: WeaviateAsyncClient | None = None
            try:
                try:
                    client = create_weaviate_client(self._settings)
                except ValueError as error:
                    # The client validates hosts and ports with pydantic.
                    raise StoreConnectionError(
                        PROVIDER, self._scrub(str(error))
                    ) from None
                with self._errors():
                    await client.connect()
            except BaseException:
                if client is not None:
                    await self._discard(client)
                raise
            self._client = client
            return client

    async def _discard(self, client: WeaviateAsyncClient) -> None:
        try:
            await client.close()
        except (WeaviateBaseError, OSError) as error:
            logger.debug(
                "Closing the Weaviate client failed: %s", self._scrub(str(error))
            )

    async def close(self) -> None:
        """Close the client. Safe to call more than once."""
        async with self._lock:
            client, self._client = self._client, None
        if client is not None:
            await self._discard(client)

    async def _collection(self) -> CollectionAsync:
        client = await self._open()
        return client.collections.get(self._class)

    # -- collection ------------------------------------------------------------

    async def collection_exists(self) -> bool:
        client = await self._open()
        with self._errors():
            return bool(await client.collections.exists(self._class))

    async def ensure_collection(self) -> None:
        client = await self._open()
        with self._errors():
            if not await client.collections.exists(self._class):
                try:
                    await client.collections.create(
                        name=self._class,
                        properties=[
                            *(property_for(spec) for spec in self._schema.fields),
                            _id_property(),
                        ],
                        vector_config=Configure.Vectors.self_provided(
                            name=self._schema.vector_field,
                            vector_index_config=Configure.VectorIndex.flat(
                                distance_metric=VectorDistances.COSINE
                            ),
                        ),
                    )
                    return
                except UnexpectedStatusCodeError:
                    # Another process may have created it between the two calls.
                    if not await client.collections.exists(self._class):
                        raise
            await self._check_existing(client.collections.get(self._class))

    async def _check_existing(self, collection: CollectionAsync) -> None:
        """Check that an existing collection holds the schema; add what is missing.

        Raises:
            IncompatibleCollectionError: The named vector is missing or not
                cosine, or a property has another type.
        """
        config = await collection.config.get()
        vectors = config.vector_config or {}
        named = vectors.get(self._schema.vector_field)
        if named is None:
            raise IncompatibleCollectionError(
                f"Weaviate collection '{self._class}' has no named vector "
                f"'{self._schema.vector_field}' (found {sorted(vectors) or 'none'}). "
                "Delete it or use another collection_name."
            )
        index = named.vector_index_config
        distance = (
            index.distance_metric
            if isinstance(index, VectorIndexConfigFlat | VectorIndexConfigHNSW)
            else None
        )
        if distance is not None and distance != VectorDistances.COSINE:
            raise IncompatibleCollectionError(
                f"Weaviate collection '{self._class}' uses the {distance.value} "
                f"distance; collection '{self._schema.name}' needs cosine."
            )
        existing = {prop.name: prop for prop in config.properties}
        for spec in self._schema.fields:
            found = existing.get(spec.name)
            if found is None:
                continue
            if found.data_type != _DATA_TYPES[spec.type]:
                raise IncompatibleCollectionError(
                    f"Weaviate collection '{self._class}' stores '{spec.name}' as "
                    f"{found.data_type.value}; collection '{self._schema.name}' "
                    f"needs {_DATA_TYPES[spec.type].value}."
                )
            if spec.full_text and found.index_searchable is False:
                raise IncompatibleCollectionError(
                    f"Weaviate property '{spec.name}' of '{self._class}' has no "
                    "search index; lexical and hybrid search need one."
                )
            if spec.indexed and not spec.full_text and found.index_filterable is False:
                raise IncompatibleCollectionError(
                    f"Weaviate property '{spec.name}' of '{self._class}' has no "
                    "filter index; exact-match filters need one."
                )
        for spec in self._schema.fields:
            if spec.name not in existing:
                await collection.config.add_property(property_for(spec))
        if ID_PROPERTY not in existing:
            await collection.config.add_property(_id_property())

    async def delete_collection(self) -> None:
        client = await self._open()
        with self._errors():
            if await client.collections.exists(self._class):
                await client.collections.delete(self._class)

    # -- IDs, properties and filters ---------------------------------------------

    def store_id(self, record_id: str) -> uuid.UUID:
        """Return the object UUID for a HoloDeck record ID."""
        try:
            return uuid.UUID(record_id)
        except ValueError:
            return uuid.uuid5(self._namespace, record_id)

    def _properties(
        self, record: Record, store_id: uuid.UUID
    ) -> dict[str, str | int | float | bool]:
        properties: dict[str, str | int | float | bool] = {
            spec.name: _stored(spec, record.fields.get(spec.name))
            for spec in self._schema.fields
        }
        if str(store_id) != record.id:
            properties[ID_PROPERTY] = record.id
        return properties

    def _record(
        self, obj: Object[Mapping[str, WeaviateField], None], *, vectors: bool
    ) -> Record:
        original = obj.properties.get(ID_PROPERTY)
        fields = {
            spec.name: _coerce(spec.type, obj.properties[spec.name])
            for spec in self._schema.fields
            if spec.name in obj.properties
        }
        return Record(
            id=original if isinstance(original, str) and original else str(obj.uuid),
            fields=fields,
            vector=(
                _dense_vector(obj.vector, self._schema.vector_field)
                if vectors
                else None
            ),
        )

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )

    def _filter_value(
        self, spec: FieldSpec, value: FieldValue
    ) -> str | int | float | bool:
        if value is None:
            raise SchemaError(
                f"Field '{spec.name}': Weaviate stores type defaults, never null, "
                "so a filter on None cannot match."
            )
        if spec.type is FieldType.STRING and isinstance(value, str):
            return value
        if spec.type is FieldType.BOOLEAN and isinstance(value, bool):
            return value
        if (
            spec.type is FieldType.INTEGER
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return value
        if (
            spec.type is FieldType.FLOAT
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            return float(value)
        raise SchemaError(
            f"Field '{spec.name}' is {spec.type.value}; the filter value "
            f"{value!r} has another type."
        )

    def build_filter(self, record_filter: RecordFilter | None) -> FilterReturn | None:
        """Return the Weaviate filter for ``record_filter``, or None.

        Raises:
            SchemaError: A clause names a field that is not in the schema,
                matches a full-text field or None, or has a value of the
                wrong type.
        """
        if record_filter is None or not record_filter.all_of:
            return None
        clauses: list[FilterReturn] = []
        for clause in record_filter.all_of:
            spec = self._schema.field(clause.field)
            if spec.type is FieldType.STRING and spec.full_text:
                raise SchemaError(
                    f"Field '{spec.name}' is full-text indexed; Weaviate matches "
                    "it by token, not exactly. Filter on an exact-match field."
                )
            if not spec.indexed:
                raise SchemaError(
                    f"Field '{spec.name}' is not indexed; Weaviate filters need "
                    "FieldSpec(indexed=True)."
                )
            clauses.append(
                Filter.by_property(spec.name).equal(
                    self._filter_value(spec, normalize_filter_value(spec, clause.value))
                )
            )
        return clauses[0] if len(clauses) == 1 else Filter.all_of(clauses)

    async def replace_source(
        self, record_filter: RecordFilter, records: Sequence[Record]
    ) -> None:
        """Replace the records that match the filter with ``records``.

        Stale records are found by object UUID, not by HoloDeck ID: two
        spellings of one UUID are the same object, and deleting the old
        spelling would delete the replacement.
        """
        self.capabilities.require(Operation.FIND, Operation.DELETE, Operation.UPSERT)
        self._validate(records)
        existing = await self.find(record_filter)
        keep = {self.store_id(record.id) for record in records}
        stale = [
            record.id for record in existing if self.store_id(record.id) not in keep
        ]
        if stale:
            await self.delete(stale)
        if records:
            await self.upsert(records)

    # -- writes and point reads -------------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        if not records:
            return
        objects: dict[
            uuid.UUID, DataObject[dict[str, str | int | float | bool], None]
        ] = {}
        for record in records:
            store_id = self.store_id(record.id)
            objects.pop(store_id, None)  # the last record of an ID wins
            objects[store_id] = DataObject(
                properties=self._properties(record, store_id),
                uuid=store_id,
                vector={
                    self._schema.vector_field: [float(v) for v in record.vector or ()]
                },
            )
        batch = list(objects.values())
        collection = await self._collection()
        with self._errors():
            for start in range(0, len(batch), UPSERT_BATCH):
                result = await collection.data.insert_many(
                    batch[start : start + UPSERT_BATCH]
                )
                if result.has_errors:
                    first = next(iter(result.errors.values()))
                    raise RetrievalError(
                        f"Weaviate rejected {len(result.errors)} record(s): "
                        f"{first.message}"
                    )

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        if not ids:
            return []
        wanted = [self.store_id(key) for key in ids]
        unique = list(dict.fromkeys(wanted))
        collection = await self._collection()
        found: dict[uuid.UUID, Record] = {}
        with self._errors():
            for start in range(0, len(unique), GET_BATCH):
                chunk = unique[start : start + GET_BATCH]
                response = await collection.query.fetch_objects(
                    filters=Filter.by_id().contains_any(chunk),
                    limit=len(chunk),
                    include_vector=include_vectors,
                )
                for obj in response.objects:
                    found[obj.uuid] = self._record(obj, vectors=include_vectors)
        return [found[key] for key in wanted if key in found]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        unique = list(dict.fromkeys(self.store_id(key) for key in ids))
        collection = await self._collection()
        with self._errors():
            for start in range(0, len(unique), DELETE_BATCH):
                result = await collection.data.delete_many(
                    where=Filter.by_id().contains_any(
                        unique[start : start + DELETE_BATCH]
                    )
                )
                if result.failed:
                    raise RetrievalError(
                        f"Weaviate failed to delete {result.failed} record(s)."
                    )

    # -- filter, page, count ---------------------------------------------------

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        query_filter = self.build_filter(record_filter)
        collection = await self._collection()
        results: list[Record] = []
        offset = 0
        with self._errors():
            while True:
                response = await collection.query.fetch_objects(
                    filters=query_filter,
                    limit=DEFAULT_PAGE_SIZE,
                    offset=offset,
                )
                for obj in response.objects:
                    record = self._record(obj, vectors=False)
                    # Exact match, also on collections with word tokenization.
                    if record_filter.matches(record):
                        results.append(record)
                        if limit is not None and len(results) >= limit:
                            return results
                if len(response.objects) < DEFAULT_PAGE_SIZE:
                    return results
                offset += DEFAULT_PAGE_SIZE

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        after: uuid.UUID | None = None
        if cursor is not None:
            try:
                after = uuid.UUID(cursor)
            except ValueError as error:
                raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        collection = await self._collection()
        with self._errors():
            response = await collection.query.fetch_objects(
                limit=limit + 1, after=after
            )
        objects = response.objects
        more = len(objects) > limit
        page = objects[:limit]
        return Page(
            records=tuple(self._record(obj, vectors=False) for obj in page),
            next_cursor=str(page[-1].uuid) if more else None,
        )

    async def count(self) -> int:
        if not await self.collection_exists():
            return 0
        collection = await self._collection()
        with self._errors():
            result = await collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    # -- search ----------------------------------------------------------------

    def _text_field(self, name: str) -> str:
        spec = self._schema.field(name)
        if spec.type is not FieldType.STRING or not spec.full_text:
            raise SchemaError(
                f"Field '{name}' of schema '{self._schema.name}' is not a "
                "full-text field."
            )
        return spec.name

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        self._check_vector(query.vector)
        query_filter = self.build_filter(query.filter)
        if query.top_k <= 0:
            return []
        collection = await self._collection()
        with self._errors():
            response = await collection.query.near_vector(
                near_vector=[float(v) for v in query.vector],
                target_vector=self._schema.vector_field,
                limit=query.top_k,
                filters=query_filter,
                return_metadata=MetadataQuery(distance=True),
            )
        hits: list[ScoredRecord] = []
        for obj in response.objects:
            distance = obj.metadata.distance
            if distance is None:
                raise RetrievalError("Weaviate returned a hit without a distance.")
            hits.append(
                ScoredRecord(
                    record=self._record(obj, vectors=False),
                    score=clamp_similarity(1.0 - distance),
                    kind=ScoreKind.SIMILARITY,
                )
            )
        return hits

    async def lexical_search(self, query: LexicalQuery) -> list[ScoredRecord]:
        text_field = self._text_field(query.text_field)
        query_filter = self.build_filter(query.filter)
        if query.top_k <= 0 or not query.text.strip():
            return []
        collection = await self._collection()
        with self._errors():
            response = await collection.query.bm25(
                query=query.text,
                query_properties=[text_field],
                limit=query.top_k,
                filters=query_filter,
                return_metadata=MetadataQuery(score=True),
            )
        return [
            ScoredRecord(
                record=self._record(obj, vectors=False),
                score=float(obj.metadata.score or 0.0),
                kind=ScoreKind.LEXICAL,
            )
            for obj in response.objects
        ]

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Weaviate's native hybrid search: vector plus BM25 on the text field."""
        self._check_vector(query.vector)
        text_field = self._text_field(query.text_field)
        query_filter = self.build_filter(query.filter)
        if query.top_k <= 0:
            return []
        collection = await self._collection()
        with self._errors():
            response = await collection.query.hybrid(
                query=query.text,
                vector=[float(v) for v in query.vector],
                query_properties=[text_field],
                alpha=HYBRID_ALPHA,
                target_vector=self._schema.vector_field,
                limit=query.top_k,
                filters=query_filter,
                return_metadata=MetadataQuery(score=True),
            )
        return [
            ScoredRecord(
                record=self._record(obj, vectors=False),
                score=float(obj.metadata.score or 0.0),
                kind=ScoreKind.FUSED,
            )
            for obj in response.objects
        ]
