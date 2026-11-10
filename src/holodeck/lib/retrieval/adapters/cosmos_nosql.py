"""Azure Cosmos DB for NoSQL adapter on the native ``azure-cosmos`` async client.

Stored layout
-------------
The layout is what Semantic Kernel's ``CosmosNoSqlCollection`` wrote, so
containers created before the move stay readable:

* one container per collection, named after the collection, in the database
  ``database_name``;
* partition key ``/id``. Every document is its own logical partition, so
  ``VectorDistance`` and filter queries run across partitions. The adapter
  only supports this key (see "Partition key");
* a document is ``{"id": ..., <data fields>..., "embedding": [...]}``;
* vector embedding policy: path ``/"embedding"``, ``float32``, ``cosine``,
  ``schema.dimensions`` dimensions;
* indexing policy: automatic, ``/*`` included, ``/"_etag"/?`` excluded, every
  data field that is neither ``indexed`` nor ``full_text`` excluded (the same
  rule SK applied), one entry in ``vectorIndexes`` for ``/"embedding"``, and
  ``/"embedding"/*`` excluded so the vector is not indexed twice.

Vector index type
-----------------
SK always used ``flat``, which Cosmos limits to 505 dimensions, so SK could not
create a container for HoloDeck's default 1536-dimension embeddings. The
adapter uses ``flat`` up to 505 dimensions and ``quantizedFlat`` above (up to
4096, the service limit). The ``vector_index`` setting selects ``flat``,
``quantizedFlat`` or ``diskANN`` explicitly. The setting only matters when the
adapter creates the container: an existing container keeps its index. Cosmos
cannot change a vector index in place.

Record IDs
----------
Cosmos rejects item IDs that contain ``/``, ``\\``, ``?`` or ``#`` (and control
characters, and leading or trailing whitespace). HoloDeck IDs such as
``docs/guide.md_chunk_0`` contain paths. SK stored the ID unchanged, so those
writes failed. The adapter stores an ID that Cosmos accepts unchanged and
encodes any other: ``~hd~`` followed by the unpadded URL-safe base64 of the
UTF-8 ID. An ID that itself starts with ``~hd~`` is encoded too, so the mapping
is one-to-one. The original ID is also written to the ``holodeck_id`` field of
an encoded document, for people who query the container directly. Reads decode
the stored ID, so documents SK wrote with a plain ID read back unchanged. An
encoded ID must stay within 1023 bytes (the Cosmos limit).

Partition key
-------------
Only ``/id`` is supported. SK's delete passed the item ID as the partition key
value, so a container with another partition key never deleted anything and
HoloDeck's source replacement failed on it. A ``partition_key`` setting other
than ``id`` or ``/id`` is an error, and so is an existing container whose
partition key is not ``/id`` (``IncompatibleCollectionError``).

Queries
-------
Every query is parameterized. Field names come only from the collection schema
(unknown names raise ``SchemaError``) and must be plain identifiers; they are
written as ``c["name"]``. Values, ``TOP`` counts, vectors and search terms are
always query parameters. Queries pass ``enable_scan_in_query`` so that a filter
on a field excluded from the index (a non-``indexed`` field) still runs.

* ``vector_search``: ``SELECT TOP @top ..., VectorDistance(c["embedding"],
  @vector) AS score ... ORDER BY VectorDistance(...)``. For ``cosine`` Cosmos
  returns the similarity (``-1`` to ``1``, best first), which the adapter
  clamps to ``[0, 1]``.
* ``hybrid_search``: ``ORDER BY RANK RRF(VectorDistance(...),
  FullTextScore(c["field"], @t0, @t1, ...))``. The service does not return the
  fused score, so the adapter reports ``1 / (60 + rank)`` with ``ScoreKind.FUSED``
  (only the order is meaningful). A query without word tokens runs the vector
  leg alone, as the Qdrant adapter does.
* ``list_page``: the cursor is the service continuation token.

Hybrid search needs a full-text policy and index, and a Cosmos account with the
full-text search capability. The adapter creates them only when the connection
sets ``full_text_search: true``; SK never created them, so a container made by
SK cannot serve hybrid search and the adapter raises
``IncompatibleCollectionError`` when ``full_text_search`` is on and the
container has no full-text index. With the default (off) the adapter does not
declare ``Operation.HYBRID_SEARCH`` and HoloDeck fuses its own BM25 results.

Not supported
-------------
Lexical search alone: HoloDeck uses its BM25 index.

Lifecycle
---------
One ``CosmosClient`` (aio) per adapter, built on first use and closed by
``close``. ``close`` is safe to repeat and the adapter is usable again
afterwards. Writes and deletes run concurrently, at most ``WRITE_CONCURRENCY``
requests at a time; the SDK retries throttled (429) requests itself.

Errors
------
SDK errors become ``StoreConnectionError`` (or ``SchemaError`` for rejected
input). The account key is removed from every message the adapter builds.

Testing seam
------------
``_connect`` builds the client. The contract suite replaces it with a fake of
the async container API (``tests/contract/retrieval/cosmos_fake.py``) because
the Linux emulator lacks full-text search.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import os
import re
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Protocol, TypeVar, cast
from urllib.parse import urlparse

from azure.core.exceptions import AzureError
from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import (
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
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

PROVIDER = "azure-cosmos-nosql"
EXTRA = "azure-cosmos"

ENV_URL = "AZURE_COSMOS_DB_NO_SQL_URL"
ENV_KEY = "AZURE_COSMOS_DB_NO_SQL_KEY"
ENV_DATABASE = "AZURE_COSMOS_DB_NO_SQL_DATABASE_NAME"

ID_FIELD: Final = "id"
VECTOR_FIELD: Final = "embedding"
ORIGINAL_ID_FIELD: Final = "holodeck_id"
ID_PREFIX: Final = "~hd~"
MAX_ID_BYTES = 1023
PARTITION_KEY_PATH: Final = "/id"

FLAT_MAX_DIMENSIONS = 505
MAX_DIMENSIONS = 4096
VECTOR_INDEX_TYPES: Final = ("flat", "quantizedFlat", "diskANN")
DEFAULT_LANGUAGE: Final = "en-US"

WRITE_CONCURRENCY = 8
GET_CONCURRENCY = 8
MAX_QUERY_TERMS = 32
MIN_TOKEN_LEN = 2
RRF_K = 60

_SYSTEM_FIELDS: Final = frozenset(
    {"_rid", "_self", "_etag", "_attachments", "_ts", ORIGINAL_ID_FIELD}
)
_FIELD_NAME = re.compile(r"^[^\W\d]\w*$")
_ILLEGAL_ID_CHARS: Final = frozenset("/\\?#")

_CAPABILITIES = frozenset(
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

_ACCEPTED_KEYS = (
    "url",
    "endpoint",
    "key",
    "connection_string",
    "database_name",
    "collection_name",
    "partition_key",
    "create_database",
    "vector_index",
    "full_text_search",
)

_T = TypeVar("_T")


def _capabilities(full_text: bool) -> Capabilities:
    supported = _CAPABILITIES | ({Operation.HYBRID_SEARCH} if full_text else set())
    hints = {
        Operation.LEXICAL_SEARCH: "HoloDeck uses its BM25 index for this provider."
    }
    if not full_text:
        hints[Operation.HYBRID_SEARCH] = (
            "HoloDeck fuses vector and BM25 results itself. Set "
            "full_text_search: true on a container that has a full-text index "
            "to use the service's hybrid search."
        )
    return Capabilities(provider=PROVIDER, supported=frozenset(supported), hints=hints)


# -- connection settings -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CosmosSettings:
    """Connection settings of one adapter.

    Attributes:
        endpoint: Account URL.
        key: Account key. Never shown in ``repr``.
        database_name: Database that holds the container.
        create_database: Create the database when it is missing.
        vector_index: Vector index type of a created container, or None to
            choose by dimensions.
        full_text_search: Create (and require) a full-text index and offer
            native hybrid search.
    """

    endpoint: str
    key: str = field(repr=False)
    database_name: str
    create_database: bool = False
    vector_index: str | None = None
    full_text_search: bool = False


def parse_cosmos_connection_string(connection_string: str) -> tuple[str, str | None]:
    """Return ``(endpoint, key)`` from a Cosmos account connection string.

    Formats: ``AccountEndpoint=https://...;AccountKey=...;`` (the Azure portal
    form) or a bare ``https://...`` URL, which carries no key.

    Raises:
        ValueError: The string has no endpoint.
    """
    text = connection_string.strip()
    if not text:
        raise ValueError("Connection string cannot be empty")
    if "=" not in text:
        return text, None
    parts: dict[str, str] = {}
    for item in text.split(";"):
        name, separator, value = item.strip().partition("=")
        if separator:
            parts[name.strip().lower()] = value.strip()
    endpoint = parts.get("accountendpoint")
    if not endpoint:
        raise ValueError("Connection string has no AccountEndpoint")
    return endpoint, parts.get("accountkey") or None


def _text(connection: Mapping[str, object], name: str) -> str | None:
    value = connection.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{name}' must be a string."
        )
    return value.strip() or None


def _flag(connection: Mapping[str, object], name: str) -> bool:
    value = connection.get(name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{name}' must be true or false."
        )
    return value


def settings_from_connection(connection: Mapping[str, object]) -> CosmosSettings:
    """Build settings from the database settings of an agent.

    Accepted keys: ``url`` or ``endpoint``, ``key``, ``connection_string`` (the
    portal form), ``database_name``, ``collection_name`` (the schema already
    carries it), ``partition_key`` (only ``id`` or ``/id``), ``create_database``,
    ``vector_index`` and ``full_text_search``. ``url``, ``key`` and
    ``database_name`` fall back to ``AZURE_COSMOS_DB_NO_SQL_URL``,
    ``AZURE_COSMOS_DB_NO_SQL_KEY`` and ``AZURE_COSMOS_DB_NO_SQL_DATABASE_NAME``,
    which SK read.

    Raises:
        StoreConnectionError: A key is unknown or mistyped, or a required
            setting is missing.
    """
    unknown = sorted(set(connection).difference(_ACCEPTED_KEYS))
    if unknown:
        raise StoreConnectionError(
            PROVIDER,
            f"Unknown connection settings {unknown}. "
            f"Accepted: {', '.join(_ACCEPTED_KEYS)}.",
        )
    endpoint = _text(connection, "url")
    alias = _text(connection, "endpoint")
    if endpoint and alias and endpoint != alias:
        raise StoreConnectionError(
            PROVIDER, "Connection settings 'url' and 'endpoint' differ; set one."
        )
    endpoint = endpoint or alias
    key = _text(connection, "key")
    connection_string = _text(connection, "connection_string")
    if connection_string is not None:
        try:
            parsed_endpoint, parsed_key = parse_cosmos_connection_string(
                connection_string
            )
        except ValueError as error:
            raise StoreConnectionError(PROVIDER, str(error)) from None
        endpoint = endpoint or parsed_endpoint
        key = key or parsed_key
    endpoint = endpoint or os.environ.get(ENV_URL) or None
    key = key or os.environ.get(ENV_KEY) or None
    if not endpoint:
        raise StoreConnectionError(
            PROVIDER,
            f"No account URL. Set url in the database settings or {ENV_URL}.",
        )
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise StoreConnectionError(
            PROVIDER, "The account URL must start with https:// (or http://)."
        )
    if parsed.scheme == "http" and parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        logger.warning("The Cosmos DB account URL uses http; the key is not encrypted.")
    if not key:
        raise StoreConnectionError(
            PROVIDER,
            f"No account key. Set key in the database settings or {ENV_KEY}. "
            "Token credentials cannot be set from YAML.",
        )
    database = _text(connection, "database_name") or os.environ.get(ENV_DATABASE)
    if not database:
        raise StoreConnectionError(
            PROVIDER,
            "No database name. Set database_name in the database settings "
            f"or {ENV_DATABASE}.",
        )
    partition_key = _text(connection, "partition_key")
    if partition_key is not None and partition_key.strip("/") != ID_FIELD:
        raise StoreConnectionError(
            PROVIDER,
            "Only the partition key /id is supported; "
            f"got '{partition_key}'. Delete by ID needs it.",
        )
    vector_index = _text(connection, "vector_index")
    if vector_index is not None and vector_index not in VECTOR_INDEX_TYPES:
        raise StoreConnectionError(
            PROVIDER,
            f"vector_index must be one of {', '.join(VECTOR_INDEX_TYPES)}; "
            f"got '{vector_index}'.",
        )
    return CosmosSettings(
        endpoint=endpoint,
        key=key,
        database_name=database,
        create_database=_flag(connection, "create_database"),
        vector_index=vector_index,
        full_text_search=_flag(connection, "full_text_search"),
    )


# -- item IDs ------------------------------------------------------------------------


def _needs_encoding(record_id: str) -> bool:
    return (
        record_id.startswith(ID_PREFIX)
        or record_id != record_id.strip()
        or any(
            char in _ILLEGAL_ID_CHARS or ord(char) < 0x20 or ord(char) == 0x7F
            for char in record_id
        )
    )


def encode_id(record_id: str) -> str:
    """Return the Cosmos item ID for a HoloDeck record ID. See the module docs.

    Raises:
        SchemaError: The ID is empty, is not valid text, or is too long.
    """
    if not record_id:
        raise SchemaError("Record ID must not be empty.")
    try:
        raw = record_id.encode("utf-8")
    except UnicodeEncodeError:
        raise SchemaError("Record ID is not valid UTF-8 text.") from None
    stored = record_id
    if _needs_encoding(record_id):
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        stored = f"{ID_PREFIX}{token}"
    if len(stored.encode("utf-8")) > MAX_ID_BYTES:
        raise SchemaError(
            f"Record ID is too long for Cosmos DB: {MAX_ID_BYTES} bytes at most "
            "after encoding."
        )
    return stored


def decode_id(stored_id: str) -> str:
    """Return the HoloDeck record ID of a stored item ID."""
    if not stored_id.startswith(ID_PREFIX):
        return stored_id
    token = stored_id[len(ID_PREFIX) :]
    try:
        raw = base64.b64decode(
            token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
        )
        return raw.decode("utf-8") or stored_id
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return stored_id


# -- value conversion ----------------------------------------------------------------


def _stored(spec: FieldSpec, value: FieldValue) -> FieldValue:
    """Return ``value`` in the JSON type Cosmos stores for ``spec``."""
    if value is None:
        return None
    if spec.type is FieldType.INTEGER:
        return int(value)
    if spec.type is FieldType.FLOAT:
        number = float(value)
        if not math.isfinite(number):
            raise SchemaError(f"Field '{spec.name}' must be a finite number.")
        return number
    if spec.type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


def _coerce(spec: FieldSpec, value: object) -> FieldValue:
    """Convert a stored JSON value to the schema type."""
    if value is None:
        return None
    if spec.type is FieldType.BOOLEAN:
        return bool(value)
    if spec.type is FieldType.INTEGER:
        if isinstance(value, bool):
            return int(value)
        return int(value) if isinstance(value, (int, float, str)) else None
    if spec.type is FieldType.FLOAT:
        if isinstance(value, bool):
            return float(value)
        return float(value) if isinstance(value, (int, float, str)) else None
    return value if isinstance(value, str) else str(value)


def _vector_list(value: object) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        numbers.append(float(item))
    return numbers


def tokenize(text: str) -> list[str]:
    """Split a query into lowercase word tokens, dropping short and repeated ones."""
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in re.findall(r"\w+", text.lower(), re.UNICODE):
        if len(raw) < MIN_TOKEN_LEN or raw in seen:
            continue
        seen.add(raw)
        tokens.append(raw)
        if len(tokens) == MAX_QUERY_TERMS:
            break
    return tokens


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> list[object]:
    return (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, str)
        else []
    )


def _plain_path(path: object) -> str:
    """Return a policy path without the quoting Cosmos adds (``/"a"`` to ``/a``)."""
    return str(path).replace('"', "")


# -- statements ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Statement:
    """A parameterized Cosmos SQL query.

    Attributes:
        text: Query text. It holds no caller-supplied value.
        parameters: ``(name, value)`` pairs; every name appears in ``text``.
    """

    text: str
    parameters: tuple[tuple[str, object], ...] = ()

    def sdk_parameters(self) -> list[dict[str, object]]:
        """Return the parameters in the SDK's ``name``/``value`` form."""
        return [{"name": name, "value": value} for name, value in self.parameters]


# -- the SDK seam --------------------------------------------------------------------


class _Pager(Protocol):
    """The page iterator of ``query_items(...).by_page()``."""

    continuation_token: str | None

    async def __anext__(self) -> AsyncIterator[object]:
        """Fetch the next page."""
        ...


class _Query(Protocol):
    """The result of ``query_items``."""

    def by_page(self, continuation_token: str | None = None) -> _Pager:
        """Iterate by page, optionally from a continuation token."""
        ...

    def __aiter__(self) -> AsyncIterator[object]:
        """Iterate over every item."""
        ...


class _Container(Protocol):
    """The ``ContainerProxy`` methods the adapter calls."""

    async def read(self) -> Mapping[str, object]:
        """Return the container properties."""
        ...

    async def upsert_item(self, body: Mapping[str, object]) -> object:
        """Write one item."""
        ...

    async def read_item(self, item: str, partition_key: str) -> Mapping[str, object]:
        """Read one item."""
        ...

    async def delete_item(self, item: str, partition_key: str) -> None:
        """Delete one item."""
        ...

    def query_items(
        self,
        query: str,
        *,
        parameters: Sequence[Mapping[str, object]],
        max_item_count: int | None = None,
        enable_scan_in_query: bool = True,
    ) -> _Query:
        """Run a query."""
        ...


class _Database(Protocol):
    """The ``DatabaseProxy`` methods the adapter calls."""

    async def create_container_if_not_exists(
        self,
        *,
        id: str,
        partition_key: PartitionKey,
        indexing_policy: Mapping[str, object],
        vector_embedding_policy: Mapping[str, object],
        full_text_policy: Mapping[str, object] | None = None,
    ) -> object:
        """Create the container unless it exists."""
        ...

    async def delete_container(self, container: str) -> None:
        """Delete a container."""
        ...

    def get_container_client(self, container: str) -> _Container:
        """Return a handle on a container. Sends no request."""
        ...


class _Client(Protocol):
    """The ``CosmosClient`` methods the adapter calls."""

    async def create_database_if_not_exists(self, id: str) -> _Database:
        """Create the database unless it exists."""
        ...

    def get_database_client(self, database: str) -> _Database:
        """Return a handle on a database. Sends no request."""
        ...

    async def close(self) -> None:
        """Release the HTTP session."""
        ...


# -- the adapter ---------------------------------------------------------------------


def resolve_vector_index(dimensions: int, requested: str | None) -> str:
    """Return the vector index type for a created container.

    Raises:
        SchemaError: The dimensions do not fit the index type.
    """
    if dimensions > MAX_DIMENSIONS:
        raise SchemaError(
            f"Cosmos DB supports at most {MAX_DIMENSIONS} vector dimensions; "
            f"the collection needs {dimensions}."
        )
    index = requested or (
        "flat" if dimensions <= FLAT_MAX_DIMENSIONS else "quantizedFlat"
    )
    if index == "flat" and dimensions > FLAT_MAX_DIMENSIONS:
        raise SchemaError(
            f"A flat vector index holds at most {FLAT_MAX_DIMENSIONS} dimensions; "
            f"the collection needs {dimensions}. Use quantizedFlat or diskANN."
        )
    return index


class CosmosNoSqlAdapter(BaseVectorStoreAdapter):
    """One Azure Cosmos DB NoSQL container as a HoloDeck collection."""

    def __init__(self, schema: CollectionSchema, settings: CosmosSettings) -> None:
        super().__init__(
            schema,
            _capabilities(settings.full_text_search and bool(schema.full_text_fields)),
        )
        for spec in schema.fields:
            if not _FIELD_NAME.match(spec.name) or spec.name in _SYSTEM_FIELDS:
                raise SchemaError(
                    f"Field name '{spec.name}' cannot be used in Cosmos DB: use a "
                    "plain identifier that is not a system property name."
                )
        if schema.key_field != ID_FIELD or schema.vector_field != VECTOR_FIELD:
            raise SchemaError(
                f"Cosmos DB layout needs the key field '{ID_FIELD}' and the "
                f"vector field '{VECTOR_FIELD}'."
            )
        self._vector_index = resolve_vector_index(
            schema.dimensions, settings.vector_index
        )
        self._settings = settings
        self._lock = asyncio.Lock()
        self._client: _Client | None = None
        self._verified = False
        self._columns = ", ".join(
            [f"c.{ID_FIELD}", *(self._ref(spec.name) for spec in schema.fields)]
        )

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> CosmosNoSqlAdapter:
        """Build the adapter. No request is sent until first use."""
        return cls(schema, settings_from_connection(connection))

    # -- errors --------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        key = self._settings.key
        return text.replace(key, "***") if key else text

    @contextmanager
    def _translated(self) -> Iterator[None]:
        """Turn SDK errors into HoloDeck errors without the account key."""
        try:
            yield
        except RetrievalError:
            raise
        except AzureError as error:
            text = str(error)
            cause = None if self._settings.key in text else error
            raise StoreConnectionError(
                PROVIDER, self._scrub(f"{type(error).__name__}: {text}")
            ) from cause

    # -- container definition -----------------------------------------------------

    def _ref(self, name: str) -> str:
        """Return ``c["name"]`` for a schema field."""
        self._schema.field(name)
        return f'c["{name}"]'

    @staticmethod
    def _path(name: str) -> str:
        return f'/"{name}"'

    def vector_embedding_policy(self) -> dict[str, object]:
        """Return the vector embedding policy of a created container."""
        return {
            "vectorEmbeddings": [
                {
                    "path": self._path(VECTOR_FIELD),
                    "dataType": "float32",
                    "distanceFunction": self._schema.distance.value,
                    "dimensions": self._schema.dimensions,
                }
            ]
        }

    def indexing_policy(self) -> dict[str, object]:
        """Return the indexing policy of a created container."""
        excluded: list[dict[str, str]] = [{"path": '/"_etag"/?'}]
        for spec in self._schema.fields:
            if not spec.indexed and not spec.full_text:
                excluded.append({"path": f"{self._path(spec.name)}/*"})
        excluded.append({"path": f"{self._path(VECTOR_FIELD)}/*"})
        policy: dict[str, object] = {
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": excluded,
            "vectorIndexes": [
                {"path": self._path(VECTOR_FIELD), "type": self._vector_index}
            ],
        }
        if self._settings.full_text_search:
            policy["fullTextIndexes"] = [
                {"path": self._path(name)} for name in self._schema.full_text_fields
            ]
        return policy

    def full_text_policy(self) -> dict[str, object] | None:
        """Return the full-text policy of a created container, or None."""
        names = self._schema.full_text_fields
        if not self._settings.full_text_search or not names:
            return None
        return {
            "defaultLanguage": DEFAULT_LANGUAGE,
            "fullTextPaths": [
                {"path": self._path(name), "language": DEFAULT_LANGUAGE}
                for name in names
            ],
        }

    def _check_properties(self, properties: Mapping[str, object]) -> None:
        """Raise when an existing container cannot hold this collection."""
        name = self._schema.name
        paths = [
            _plain_path(path)
            for path in _items(_mapping(properties.get("partitionKey")).get("paths"))
        ]
        if paths and paths != [PARTITION_KEY_PATH]:
            raise IncompatibleCollectionError(
                f"Cosmos container '{name}' has the partition key {paths}; "
                f"HoloDeck needs {PARTITION_KEY_PATH}."
            )
        embeddings = _items(
            _mapping(properties.get("vectorEmbeddingPolicy")).get("vectorEmbeddings")
        )
        if isinstance(properties.get("vectorEmbeddingPolicy"), Mapping):
            ours = [
                _mapping(item)
                for item in embeddings
                if _plain_path(_mapping(item).get("path")) == f"/{VECTOR_FIELD}"
            ]
            if not ours:
                raise IncompatibleCollectionError(
                    f"Cosmos container '{name}' has no vector embedding policy for "
                    f"/{VECTOR_FIELD}."
                )
            policy = ours[0]
            if policy.get("dimensions") != self._schema.dimensions:
                raise IncompatibleCollectionError(
                    f"Cosmos container '{name}' holds vectors of "
                    f"{policy.get('dimensions')} dimensions; the collection needs "
                    f"{self._schema.dimensions}."
                )
            if policy.get("distanceFunction") != self._schema.distance.value:
                raise IncompatibleCollectionError(
                    f"Cosmos container '{name}' uses the "
                    f"{policy.get('distanceFunction')} distance; the collection "
                    f"needs {self._schema.distance.value}."
                )
        if self.capabilities.supports(Operation.HYBRID_SEARCH):
            indexed = {
                _plain_path(_mapping(item).get("path"))
                for item in _items(
                    _mapping(properties.get("indexingPolicy")).get("fullTextIndexes")
                )
            }
            missing = [
                field_name
                for field_name in self._schema.full_text_fields
                if f"/{field_name}" not in indexed
            ]
            if missing:
                raise IncompatibleCollectionError(
                    f"Cosmos container '{name}' has no full-text index for {missing}, "
                    "which full_text_search needs. Cosmos cannot add a full-text "
                    "policy to an existing container: create a new container or "
                    "set full_text_search to false."
                )

    # -- lifecycle -----------------------------------------------------------------

    async def _connect(self) -> _Client:
        """Build the client. The contract suite replaces this."""
        return cast(
            _Client,
            CosmosClient(self._settings.endpoint, credential=self._settings.key),
        )

    async def _open(self) -> _Client:
        async with self._lock:
            if self._client is None:
                self._client = await self._connect()
            return self._client

    async def _handle(self) -> _Container:
        client = await self._open()
        database = client.get_database_client(self._settings.database_name)
        return database.get_container_client(self._schema.name)

    async def _existing(self) -> _Container | None:
        """Return the container, or None when it does not exist."""
        container = await self._handle()
        if not self._verified:
            try:
                properties = await container.read()
            except CosmosResourceNotFoundError:
                return None
            self._check_properties(properties)
            self._verified = True
        return container

    async def _writable(self) -> _Container:
        container = await self._existing()
        if container is None:
            await self.ensure_collection()
            container = await self._existing()
        if container is None:
            raise StoreConnectionError(
                PROVIDER, f"Container '{self._schema.name}' is not available."
            )
        return container

    async def collection_exists(self) -> bool:
        with self._translated():
            return await self._existing() is not None

    async def ensure_collection(self) -> None:
        with self._translated():
            client = await self._open()
            name = self._settings.database_name
            database = client.get_database_client(name)
            if self._settings.create_database:
                # A concurrent creator makes the create fail with a conflict.
                with suppress(CosmosResourceExistsError):
                    database = await client.create_database_if_not_exists(id=name)
            try:
                with suppress(CosmosResourceExistsError):
                    await database.create_container_if_not_exists(
                        id=self._schema.name,
                        partition_key=PartitionKey(path=PARTITION_KEY_PATH),
                        indexing_policy=self.indexing_policy(),
                        vector_embedding_policy=self.vector_embedding_policy(),
                        full_text_policy=self.full_text_policy(),
                    )
            except CosmosResourceNotFoundError:
                raise StoreConnectionError(
                    PROVIDER,
                    f"Database '{name}' does not exist. Create it, or set "
                    "create_database: true.",
                ) from None
            self._verified = False
            if await self._existing() is None:
                raise StoreConnectionError(
                    PROVIDER,
                    f"Container '{self._schema.name}' was created but is not visible.",
                )

    async def delete_collection(self) -> None:
        with self._translated():
            client = await self._open()
            database = client.get_database_client(self._settings.database_name)
            with suppress(CosmosResourceNotFoundError):
                await database.delete_container(self._schema.name)
            self._verified = False

    async def close(self) -> None:
        async with self._lock:
            client, self._client = self._client, None
            self._verified = False
        if client is None:
            return
        try:
            await client.close()
        except AzureError as error:
            logger.debug(
                "Closing the Cosmos DB client failed: %s", self._scrub(str(error))
            )

    # -- statements ----------------------------------------------------------------

    def _where(
        self, record_filter: RecordFilter | None
    ) -> tuple[str, list[tuple[str, object]]]:
        """Return the ``WHERE`` clause (with a leading space) and its parameters."""
        if record_filter is None or not record_filter.all_of:
            return "", []
        clauses: list[str] = []
        parameters: list[tuple[str, object]] = []
        for clause in record_filter.all_of:
            spec = self._schema.field(clause.field)
            reference = self._ref(spec.name)
            value = _stored(spec, clause.value)
            if value is None:
                clauses.append(f"IS_NULL({reference})")
            else:
                name = f"@f{len(parameters)}"
                clauses.append(f"{reference} = {name}")
                parameters.append((name, value))
        return " WHERE " + " AND ".join(clauses), parameters

    def select_statement(
        self,
        record_filter: RecordFilter | None = None,
        *,
        limit: int | None = None,
        ids_only: bool = False,
    ) -> Statement:
        """Return a filtered ``SELECT`` of records (or of their IDs)."""
        where, parameters = self._where(record_filter)
        columns = f"c.{ID_FIELD}" if ids_only else self._columns
        top = ""
        if limit is not None:
            top = "TOP @limit "
            parameters.append(("@limit", limit))
        return Statement(
            f"SELECT {top}{columns} FROM c{where}",  # noqa: S608  # nosec B608
            tuple(parameters),
        )

    def count_statement(self) -> Statement:
        """Return the ``SELECT`` that counts every document."""
        return Statement("SELECT VALUE COUNT(1) FROM c")

    def vector_statement(self, query: VectorQuery) -> Statement:
        """Return the nearest-neighbour ``SELECT``; ``score`` is the similarity."""
        where, parameters = self._where(query.filter)
        distance = f'VectorDistance(c["{VECTOR_FIELD}"], @vector)'
        return Statement(
            f"SELECT TOP @top {self._columns}, {distance} AS score "  # noqa: S608  # nosec B608
            f"FROM c{where} ORDER BY {distance}",
            (
                ("@top", query.top_k),
                ("@vector", [float(value) for value in query.vector]),
                *parameters,
            ),
        )

    def hybrid_statement(self, query: HybridQuery, tokens: Sequence[str]) -> Statement:
        """Return the ``ORDER BY RANK RRF`` ``SELECT`` for non-empty ``tokens``."""
        text_reference = self._ref(query.text_field)
        where, parameters = self._where(query.filter)
        term_names = [f"@t{index}" for index in range(len(tokens))]
        rank = (
            f'RRF(VectorDistance(c["{VECTOR_FIELD}"], @vector), '
            f"FullTextScore({text_reference}, {', '.join(term_names)}))"
        )
        return Statement(
            f"SELECT TOP @top {self._columns} FROM c{where} ORDER BY RANK {rank}",  # noqa: S608  # nosec B608
            (
                ("@top", query.top_k),
                ("@vector", [float(value) for value in query.vector]),
                *parameters,
                *zip(term_names, tokens, strict=True),
            ),
        )

    # -- row conversion ------------------------------------------------------------

    def _record(self, row: Mapping[str, object], *, vector: bool = False) -> Record:
        stored = row.get(ID_FIELD)
        return Record(
            id=decode_id(stored if isinstance(stored, str) else ""),
            fields={
                spec.name: _coerce(spec, row.get(spec.name))
                for spec in self._schema.fields
            },
            vector=_vector_list(row.get(VECTOR_FIELD)) if vector else None,
        )

    def _document(self, record: Record) -> dict[str, object]:
        stored = encode_id(record.id)
        document: dict[str, object] = {ID_FIELD: stored}
        if stored != record.id:
            document[ORIGINAL_ID_FIELD] = record.id
        for spec in self._schema.fields:
            document[spec.name] = _stored(spec, record.fields.get(spec.name))
        vector = [float(value) for value in record.vector or ()]
        if not all(math.isfinite(value) for value in vector):
            raise SchemaError(
                f"Record '{record.id}' has a vector with a non-finite value."
            )
        document[VECTOR_FIELD] = vector
        return document

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; collection "
                f"'{self._schema.name}' needs {self._schema.dimensions}."
            )
        if not all(math.isfinite(value) for value in vector):
            raise SchemaError("Query vector has a non-finite value.")

    # -- running -------------------------------------------------------------------

    async def _bounded(
        self, calls: Sequence[Callable[[], Awaitable[_T]]], limit: int
    ) -> list[_T]:
        """Run ``calls`` with at most ``limit`` in flight; raise the first error."""
        semaphore = asyncio.Semaphore(limit)

        async def run(call: Callable[[], Awaitable[_T]]) -> _T:
            async with semaphore:
                return await call()

        results = await asyncio.gather(
            *(run(call) for call in calls), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return cast(list[_T], results)

    async def _rows(
        self,
        container: _Container,
        statement: Statement,
        *,
        max_item_count: int | None = None,
    ) -> list[object] | None:
        """Run a query and return its rows, or None when the container is gone."""
        try:
            return [
                row
                async for row in container.query_items(
                    statement.text,
                    parameters=statement.sdk_parameters(),
                    max_item_count=max_item_count,
                    enable_scan_in_query=True,
                )
            ]
        except CosmosResourceNotFoundError:
            self._verified = False
            return None

    # -- operations ----------------------------------------------------------------

    async def count(self) -> int:
        with self._translated():
            container = await self._existing()
            if container is None:
                return 0
            rows = await self._rows(container, self.count_statement())
        total = 0
        for row in rows or ():
            if isinstance(row, (int, float)) and not isinstance(row, bool):
                total += int(row)
        return total

    @staticmethod
    def _mappings(rows: Sequence[object] | None) -> list[Mapping[str, object]]:
        return [row for row in rows or () if isinstance(row, Mapping)]

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        documents: dict[str, dict[str, object]] = {}
        for record in records:
            document = self._document(record)
            documents[cast(str, document[ID_FIELD])] = document
        if not documents:
            return
        with self._translated():
            container = await self._writable()
            try:
                await self._write(container, list(documents.values()))
            except CosmosResourceNotFoundError:
                self._verified = False
                container = await self._writable()
                await self._write(container, list(documents.values()))

    async def _write(
        self, container: _Container, documents: Sequence[Mapping[str, object]]
    ) -> None:
        await self._bounded(
            [partial(container.upsert_item, document) for document in documents],
            WRITE_CONCURRENCY,
        )

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        if not ids:
            return []
        stored_ids = [encode_id(record_id) for record_id in ids]
        with self._translated():
            container = await self._existing()
            if container is None:
                return []

            async def read(stored: str) -> Mapping[str, object] | None:
                try:
                    return await container.read_item(stored, partition_key=stored)
                except CosmosResourceNotFoundError:
                    return None

            unique = list(dict.fromkeys(stored_ids))
            rows = await self._bounded(
                [partial(read, stored) for stored in unique], GET_CONCURRENCY
            )
        by_id = {
            stored: self._record(row, vector=include_vectors)
            for stored, row in zip(unique, rows, strict=True)
            if row is not None
        }
        return [by_id[stored] for stored in stored_ids if stored in by_id]

    async def delete(self, ids: Sequence[str]) -> None:
        stored_ids = list(dict.fromkeys(encode_id(record_id) for record_id in ids))
        if not stored_ids:
            return
        with self._translated():
            container = await self._existing()
            if container is not None:
                await self._delete_stored(container, stored_ids)

    async def _delete_stored(
        self, container: _Container, stored_ids: Sequence[str]
    ) -> int:
        """Delete items by stored ID. Return how many existed."""

        async def remove(stored: str) -> int:
            try:
                await container.delete_item(stored, partition_key=stored)
            except CosmosResourceNotFoundError:
                return 0
            return 1

        counts = await self._bounded(
            [partial(remove, stored) for stored in stored_ids], WRITE_CONCURRENCY
        )
        return sum(counts)

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        if limit is not None and limit <= 0:
            return []
        statement = self.select_statement(record_filter, limit=limit)
        with self._translated():
            container = await self._existing()
            rows = await self._rows(container, statement) if container else None
        return [self._record(row) for row in self._mappings(rows)]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        """Return one page. The cursor is the service continuation token."""
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        statement = self.select_statement()
        with self._translated():
            container = await self._existing()
            if container is None:
                return Page(records=())
            pager = container.query_items(
                statement.text,
                parameters=statement.sdk_parameters(),
                max_item_count=limit,
                enable_scan_in_query=True,
            ).by_page(cursor)
            try:
                page = await pager.__anext__()
                rows = self._mappings([row async for row in page])
            except StopAsyncIteration:
                return Page(records=())
            except CosmosResourceNotFoundError:
                self._verified = False
                return Page(records=())
        records = tuple(self._record(row) for row in rows[:limit])
        return Page(records=records, next_cursor=pager.continuation_token or None)

    async def delete_where(self, record_filter: RecordFilter) -> int:
        statement = self.select_statement(record_filter, ids_only=True)
        with self._translated():
            container = await self._existing()
            rows = await self._rows(container, statement) if container else None
            stored_ids = [
                stored
                for row in self._mappings(rows)
                if isinstance(stored := row.get(ID_FIELD), str)
            ]
            if container is None or not stored_ids:
                return 0
            return await self._delete_stored(container, stored_ids)

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        """Nearest records by cosine similarity, best first."""
        self._check_vector(query.vector)
        if query.top_k <= 0:
            return []
        statement = self.vector_statement(query)
        with self._translated():
            container = await self._existing()
            rows = await self._rows(container, statement) if container else None
        hits: list[ScoredRecord] = []
        for row in self._mappings(rows):
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            hits.append(
                ScoredRecord(
                    record=self._record(row),
                    score=clamp_similarity(score),
                    kind=ScoreKind.SIMILARITY,
                )
            )
        return hits

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Vector and full-text legs fused by the service with RRF."""
        self._capabilities.require(Operation.HYBRID_SEARCH)
        self._check_vector(query.vector)
        spec = self._schema.field(query.text_field)
        if not spec.full_text:
            raise SchemaError(
                f"Field '{spec.name}' is not a full-text field of "
                f"'{self._schema.name}'."
            )
        if query.top_k <= 0:
            return []
        tokens = tokenize(query.text)
        if tokens:
            statement = self.hybrid_statement(query, tokens)
        else:
            statement = self.vector_statement(
                VectorQuery(query.vector, query.top_k, query.filter)
            )
        with self._translated():
            container = await self._existing()
            rows = await self._rows(container, statement) if container else None
        return [
            ScoredRecord(
                record=self._record(row),
                score=1.0 / (RRF_K + rank),
                kind=ScoreKind.FUSED,
            )
            for rank, row in enumerate(self._mappings(rows), start=1)
        ]
