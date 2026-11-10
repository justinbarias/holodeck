"""Azure AI Search adapter on the native ``azure-search-documents`` client (12.0).

Only the asynchronous clients are used: ``azure.search.documents.aio.SearchClient``
for documents and ``azure.search.documents.indexes.aio.SearchIndexClient`` for
index management. Nothing here blocks the event loop.

Verification status
-------------------
No Azure AI Search emulator exists and no Azure credentials were available when
this adapter was written. Every check runs against an in-memory fake of the two
clients (``tests/contract/retrieval/azure_search_fake.py``) or against the real
SDK with its HTTP transport stubbed (``tests/unit/lib/retrieval/
test_azure_ai_search_requests.py``). Behavior that depends on the live service
(keyset ordering of string keys, eventual consistency, score values) has not been
observed. The live run stays open in the 042 exec plan.

Stored layout
-------------
Compatible with what Semantic Kernel's ``AzureAISearchCollection`` creates:

* one index per collection, named after the collection (see "Index names");
* key field ``id`` (``Edm.String``, key), vector field ``embedding``
  (``Collection(Edm.Single)``, ``schema.dimensions`` dimensions, profile
  ``embedding_profile`` over the HNSW configuration ``embedding_algorithm`` with
  the ``cosine`` metric);
* one field per data field: ``Edm.String``, ``Edm.Int64``, ``Edm.Double`` or
  ``Edm.Boolean``, the types SK uses.

Differences from the SK index, all on the safe side: ``filterable`` is true only
for ``FieldSpec.indexed`` fields (SK also made full-text fields filterable, which
Azure limits to 32 KB terms), ``searchable`` is true only for
``FieldSpec.full_text`` fields (SK made every string searchable), and the key
field is ``sortable`` because keyset paging orders by it. A key field cannot be
made sortable after the index exists, so an index created by SK, whose key is not
sortable, is rejected with ``IncompatibleCollectionError`` instead of failing
inside a query.

Record IDs
----------
Azure keys hold letters, digits, ``_``, ``-`` and ``=``, at most 1024
characters, and cannot start with ``_``. HoloDeck IDs are chunk IDs built from
source paths (``docs/a.md_chunk_0``), and Semantic Kernel wrote them unchanged,
which Azure rejects for any path with a ``/`` or ``.``. ``encode_key`` therefore
keeps an ID that is already a valid key (so records written by the SK path stay
readable) and stores every other ID as ``b64-`` plus its URL-safe Base64 text.
An ID that is valid but starts with ``b64-`` is encoded too, and ``decode_key``
only decodes text that re-encodes to itself, so the mapping is one to one and a
raw key such as ``b64-foo`` still reads as itself. The original ID is
recoverable from the key, so the index needs no extra field. IDs longer than 759
UTF-8 bytes do not fit and raise ``SchemaError``.

Index names
-----------
Azure index names are 2 to 128 characters: lowercase letters, digits, ``-`` and
``_``, starting with a letter or digit, no two ``-``/``_`` side by side. HoloDeck's
default names (``documents_dim1536``) are valid and are used unchanged, so they
match the index SK would create. ``normalize_index_name`` maps any other name
deterministically: lower-cased, every run of other characters becomes one
hyphen, and a name that is too short or too long gets a hash of the full name.
Names that differ only in invalid characters share an index.

Filters
-------
Exact-match filters become OData ``$filter`` text. String literals double every
single quote. A field must be ``indexed`` (filterable) or the adapter raises
``SchemaError`` before the request. The filter goes in the POST body, so no URL
length limit applies.

Enumeration
-----------
``find``, ``list_page`` (and so ``delete_where``) search ``*`` with the filter,
order by the key and page with a range filter on the last key (``id gt '<key>'``),
1000 documents per request. ``$skip`` is not used because the service caps it at
100,000. The cursor of ``list_page`` is the last encoded key. This is the method
the service documentation recommends for scanning past the ``$skip`` limit.

Search
------
* ``vector_search``: a ``VectorizedQuery`` on ``embedding`` with ``k = top``,
  pre-filtered (``vectorFilterMode=preFilter``). The service caps results at 1000.
* ``lexical_search``: ``search_text`` on the full-text field only
  (``search_fields``), simple query syntax, ``search_mode=any``. The text is
  reduced to word tokens first, so ``+ - | " * ( )`` in a question cannot act as
  operators. Scores are BM25 and only their order is portable (``LEXICAL``).
* ``hybrid_search``: the same text query and one vector query in one request;
  the service fuses the two ranked lists with Reciprocal Rank Fusion (``FUSED``).
  The vector side asks for ``max(4 * top, 20)`` neighbours, as the Qdrant path
  does for its candidate pool.

Score conversion
----------------
For the ``cosine`` metric Azure does not return the cosine. The documented
transformation (https://learn.microsoft.com/azure/search/vector-search-ranking,
"Scores in a vector search results") is ``@search.score = 1 / (1 + d)`` with the
cosine distance ``d = 1 - cos``. Solving for the cosine gives ``cos = 2 - 1 /
score``, which is the same as the service's own ``ScoreToSimilarity`` example.
The adapter clamps it to ``[0, 1]``. The service orders hits by ``@search.score``,
which is monotonic in the cosine, so negative similarities rank correctly before
the clamp.

Consistency
-----------
Indexing is eventually consistent: a document can be missing from search for a
moment after an upload, or still present after a delete. ``upsert`` and
``delete`` therefore poll one search for the last document of the call (present
with the written values, or gone) for up to ``VISIBILITY_TIMEOUT`` seconds, then
log a warning and continue. This keeps ``find`` after ``replace_source`` and a
re-ingest right after an ingest correct without a sleep. ``count`` is the
service's document count, which the service documents as approximate while
documents are being indexed.

``upsert`` is ``merge_or_upload``. Because merge keeps fields that a document
leaves out, every schema field is written (``None`` as JSON null), so an upsert
replaces the record as the contract requires.

Authentication
--------------
``api_key`` (or ``AZURE_AI_SEARCH_API_KEY``) uses key authentication. With no key
the adapter uses ``azure.identity.aio.DefaultAzureCredential`` (Microsoft Entra
ID), which the ``azure-ai-search`` extra installs.

Lifecycle
---------
Both clients are built on first use and released by ``close``. ``close`` is safe
to repeat and the adapter is usable again afterwards.

Errors
------
SDK errors become ``StoreConnectionError`` (``SchemaError`` for HTTP 400). The
API key is removed from every message the adapter builds.

Testing seam
------------
``AzureAISearchAdapter(schema, settings, client_factory=...)`` takes a callable
that builds the ``ClientPair``. Tests pass a fake pair, or a real pair on a stub
transport. The default factory builds the SDK clients.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Protocol
from urllib.parse import urlsplit

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import IndexingResult, VectorizedQuery, VectorQuery

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    IncompatibleCollectionError,
    MissingClientError,
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
    clamp_similarity,
)
from holodeck.lib.retrieval.types import (
    VectorQuery as HoloVectorQuery,
)

logger = logging.getLogger(__name__)

PROVIDER = "azure-ai-search"
EXTRA = "azure-ai-search"

ENDPOINT_ENV = "AZURE_AI_SEARCH_ENDPOINT"
API_KEY_ENV = "AZURE_AI_SEARCH_API_KEY"

KEY_PREFIX: Final = "b64-"
MAX_KEY_LENGTH = 1024
MAX_INDEX_NAME = 128
MIN_INDEX_NAME = 2
UPLOAD_MAX_DOCUMENTS = 1000
UPLOAD_MAX_BYTES = 10_000_000
"""Service limit is 16 MB per request; the estimate leaves room for the envelope."""
DELETE_MAX_DOCUMENTS = 1000
SEARCH_MAX_TOP = 1000
SCAN_PAGE_SIZE = 1000
GET_CONCURRENCY = 8
HYBRID_MIN_NEIGHBOURS = 20
HYBRID_POOL_FACTOR = 4
VISIBILITY_TIMEOUT = 10.0
_POLL_START = 0.1
_POLL_MAX = 1.0

VECTOR_PROFILE_SUFFIX = "_profile"
VECTOR_ALGORITHM_SUFFIX = "_algorithm"

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9=-][A-Za-z0-9_=-]*$")
_INDEX_NAME_PATTERN = re.compile(r"^[a-z0-9]+([-_][a-z0-9]+)*$")
_WORD = re.compile(r"\w+", re.UNICODE)

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
            Operation.HYBRID_SEARCH,
        }
    ),
)

_EDM_TYPES: Final[dict[FieldType, SearchFieldDataType]] = {
    FieldType.STRING: SearchFieldDataType.STRING,
    FieldType.INTEGER: SearchFieldDataType.INT64,
    FieldType.FLOAT: SearchFieldDataType.DOUBLE,
    FieldType.BOOLEAN: SearchFieldDataType.BOOLEAN,
}

_ACCEPTED_KEYS = (
    "connection_string",
    "endpoint",
    "search_endpoint",
    "api_key",
    "index_name",
    "collection_name",
)


# -- connection settings -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AzureSearchSettings:
    """Connection settings of one adapter.

    Attributes:
        endpoint: Service URL, ``https://<service>.search.windows.net``.
        api_key: Admin or query key. ``None`` selects Microsoft Entra ID. Never
            shown in ``repr``.
        index_name: Index name that replaces the collection name.
    """

    endpoint: str
    api_key: str | None = field(default=None, repr=False)
    index_name: str | None = None


def _text(connection: Mapping[str, object], key: str) -> str | None:
    value = connection.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StoreConnectionError(
            PROVIDER, f"Connection setting '{key}' must be a string."
        )
    return value.strip() or None


def _clean_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme != "https" or not parts.hostname:
        raise StoreConnectionError(
            PROVIDER,
            "The endpoint must be an https URL such as "
            "https://<service>.search.windows.net.",
        )
    return f"https://{parts.netloc}".rstrip("/")


def settings_from_connection(connection: Mapping[str, object]) -> AzureSearchSettings:
    """Build settings from the database settings of an agent.

    Accepted keys: ``connection_string`` (the endpoint URL), ``endpoint`` and
    ``search_endpoint`` (the same, the second is the Semantic Kernel name),
    ``api_key``, ``index_name`` and ``collection_name`` (the schema already
    carries the latter). The endpoint falls back to ``AZURE_AI_SEARCH_ENDPOINT``
    and the key to ``AZURE_AI_SEARCH_API_KEY``. With no key, Microsoft Entra ID
    (``DefaultAzureCredential``) authenticates.

    Raises:
        StoreConnectionError: A key is unknown or mistyped, or no valid https
            endpoint is set.
    """
    unknown = sorted(set(connection).difference(_ACCEPTED_KEYS))
    if unknown:
        raise StoreConnectionError(
            PROVIDER,
            f"Unknown connection settings {unknown}. "
            f"Accepted: {', '.join(_ACCEPTED_KEYS)}.",
        )
    endpoint = (
        _text(connection, "endpoint")
        or _text(connection, "search_endpoint")
        or _text(connection, "connection_string")
        or os.environ.get(ENDPOINT_ENV)
    )
    if not endpoint:
        raise StoreConnectionError(
            PROVIDER,
            "No endpoint. Set connection_string (or endpoint) in the database "
            f"settings, or the {ENDPOINT_ENV} environment variable.",
        )
    api_key = _text(connection, "api_key") or os.environ.get(API_KEY_ENV) or None
    return AzureSearchSettings(
        endpoint=_clean_endpoint(endpoint),
        api_key=api_key,
        index_name=_text(connection, "index_name"),
    )


# -- names and keys ------------------------------------------------------------------


def normalize_index_name(name: str) -> str:
    """Return the Azure index name for a collection name. See the module docs."""
    if MIN_INDEX_NAME <= len(name) <= MAX_INDEX_NAME and _INDEX_NAME_PATTERN.match(
        name
    ):
        return name
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if MIN_INDEX_NAME <= len(slug) <= MAX_INDEX_NAME:
        return slug
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    head = slug[: MAX_INDEX_NAME - len(digest) - 1].strip("-")
    return f"{head or 'holodeck'}-{digest}"


def _is_raw_key(record_id: str) -> bool:
    return (
        len(record_id) <= MAX_KEY_LENGTH
        and bool(_KEY_PATTERN.match(record_id))
        and not record_id.startswith(KEY_PREFIX)
    )


def encode_key(record_id: str) -> str:
    """Return the Azure document key of a HoloDeck record ID.

    Raises:
        SchemaError: The ID is empty or does not fit in a key.
    """
    if not record_id:
        raise SchemaError("Record ID must not be empty.")
    if _is_raw_key(record_id):
        return record_id
    encoded = KEY_PREFIX + base64.urlsafe_b64encode(record_id.encode()).decode()
    if len(encoded) > MAX_KEY_LENGTH:
        raise SchemaError(
            f"Record ID {record_id[:60]!r}... is too long for an Azure AI Search "
            f"key: it needs at most {(MAX_KEY_LENGTH - len(KEY_PREFIX)) // 4 * 3} "
            "UTF-8 bytes."
        )
    return encoded


def decode_key(key: str) -> str:
    """Return the record ID that ``encode_key`` turned into ``key``.

    A key that is not the encoding of any ID (a key written by another tool)
    is its own ID.
    """
    if not key.startswith(KEY_PREFIX):
        return key
    try:
        decoded = base64.urlsafe_b64decode(key[len(KEY_PREFIX) :]).decode()
    except (binascii.Error, ValueError):
        return key
    try:
        return decoded if encode_key(decoded) == key else key
    except SchemaError:
        return key


# -- OData ---------------------------------------------------------------------------


def odata_string(value: str) -> str:
    """Return ``value`` as an OData string literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


def _odata_literal(spec: FieldSpec, value: FieldValue) -> str:
    if value is None:
        return "null"
    ok = False
    if spec.type is FieldType.STRING:
        if isinstance(value, str):
            return odata_string(value)
    elif spec.type is FieldType.BOOLEAN:
        if isinstance(value, bool):
            return "true" if value else "false"
    elif not isinstance(value, (bool, str)):
        number = value
        ok = spec.type is FieldType.FLOAT or isinstance(number, int)
        if ok and isinstance(number, float) and not math.isfinite(number):
            ok = False
        if ok:
            return repr(number)
    raise SchemaError(
        f"Filter value {value!r} does not fit {spec.type.value} field '{spec.name}'."
    )


# -- client seams --------------------------------------------------------------------


class SearchDocumentsClient(Protocol):
    """The ``azure.search.documents.aio.SearchClient`` methods the adapter calls."""

    def search(
        self,
        search_text: str | None = None,
        *,
        filter: str | None = None,
        order_by: list[str] | None = None,
        query_type: str | None = None,
        search_fields: list[str] | None = None,
        search_mode: str | None = None,
        select: list[str] | None = None,
        top: int | None = None,
        vector_queries: list[VectorQuery] | None = None,
        vector_filter_mode: str | None = None,
    ) -> Awaitable[AsyncIterator[Mapping[str, object]]]:
        """Return the matching documents with their ``@search.*`` metadata."""
        ...

    def get_document(
        self, key: str, *, selected_fields: list[str] | None = None
    ) -> Awaitable[Mapping[str, object]]:
        """Return one document by key."""
        ...

    def get_document_count(self) -> Awaitable[int]:
        """Return the number of documents."""
        ...

    def merge_or_upload_documents(
        self, documents: list[dict[str, object]]
    ) -> Awaitable[Sequence[IndexingResult]]:
        """Merge documents into the index, inserting those that are new."""
        ...

    def delete_documents(
        self, documents: list[dict[str, object]]
    ) -> Awaitable[Sequence[IndexingResult]]:
        """Delete documents by key."""
        ...

    def close(self) -> Awaitable[None]:
        """Release the HTTP client."""
        ...


class SearchIndexesClient(Protocol):
    """The ``azure.search.documents.indexes.aio.SearchIndexClient`` methods used."""

    def get_index(self, name: str) -> Awaitable[SearchIndex]:
        """Return an index definition."""
        ...

    def create_index(self, index: SearchIndex) -> Awaitable[SearchIndex]:
        """Create an index."""
        ...

    def delete_index(self, index: str) -> Awaitable[None]:
        """Delete an index and its documents."""
        ...

    def close(self) -> Awaitable[None]:
        """Release the HTTP client."""
        ...


class _Closable(Protocol):
    def close(self) -> Awaitable[None]: ...


@dataclass(frozen=True, slots=True)
class ClientPair:
    """The two clients of one adapter and the credential they share, if owned."""

    documents: SearchDocumentsClient
    indexes: SearchIndexesClient
    credential: _Closable | None = None


ClientFactory = Callable[[AzureSearchSettings, str], ClientPair]
"""Builds the clients for ``(settings, index_name)``. Called on first use."""


def default_client_factory(
    settings: AzureSearchSettings, index_name: str
) -> ClientPair:
    """Build the SDK clients. The one place that touches the real SDK clients."""
    credential: _Closable | None = None
    if settings.api_key:
        auth: AzureKeyCredential | AsyncTokenCredential = AzureKeyCredential(
            settings.api_key
        )
    else:
        try:
            from azure.identity.aio import DefaultAzureCredential
        except ImportError as error:
            raise MissingClientError(PROVIDER, EXTRA) from error
        entra = DefaultAzureCredential()
        credential = entra
        auth = entra
    documents = SearchClient(settings.endpoint, index_name, auth)
    indexes = SearchIndexClient(settings.endpoint, auth)
    return ClientPair(documents=documents, indexes=indexes, credential=credential)


# -- index definition ----------------------------------------------------------------


def build_index(schema: CollectionSchema, index_name: str) -> SearchIndex:
    """Return the index definition for ``schema``. See the module docs."""
    fields: list[SearchField] = [
        SearchField(
            name=schema.key_field,
            type=SearchFieldDataType.STRING,
            key=True,
            searchable=False,
            filterable=True,
            sortable=True,
            facetable=False,
            retrievable=True,
        )
    ]
    for spec in schema.fields:
        fields.append(
            SearchField(
                name=spec.name,
                type=_EDM_TYPES[spec.type],
                searchable=spec.full_text,
                filterable=spec.indexed,
                sortable=False,
                facetable=False,
                retrievable=True,
            )
        )
    profile = f"{schema.vector_field}{VECTOR_PROFILE_SUFFIX}"
    algorithm = f"{schema.vector_field}{VECTOR_ALGORITHM_SUFFIX}"
    fields.append(
        SearchField(
            name=schema.vector_field,
            type="Collection(Edm.Single)",
            searchable=True,
            retrievable=True,
            vector_search_dimensions=schema.dimensions,
            vector_search_profile_name=profile,
        )
    )
    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name=profile, algorithm_configuration_name=algorithm
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=algorithm,
                    parameters=HnswParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    ),
                )
            ],
        ),
    )


def similarity_from_score(score: float) -> float:
    """Return the cosine similarity in ``[0, 1]`` for an Azure ``@search.score``.

    For the cosine metric the service returns ``1 / (1 + (1 - cos))``; see the
    "Score conversion" section of the module docs.
    """
    if not score > 0.0:
        return 0.0
    return clamp_similarity(2.0 - 1.0 / score)


# -- conversion helpers --------------------------------------------------------------


def _stored(spec: FieldSpec, value: FieldValue) -> str | int | float | bool | None:
    """Return ``value`` in the JSON type Azure stores for ``spec``."""
    if value is None:
        return None
    try:
        if spec.type is FieldType.INTEGER:
            return int(value)
        if spec.type is FieldType.FLOAT:
            number = float(value)
            if not math.isfinite(number):
                raise SchemaError(f"Field '{spec.name}' must be a finite number.")
            return number
        if spec.type is FieldType.BOOLEAN:
            return bool(value)
    except (TypeError, ValueError):
        raise SchemaError(
            f"Value {value!r} does not fit {spec.type.value} field '{spec.name}'."
        ) from None
    return value if isinstance(value, str) else str(value)


def _read(spec: FieldSpec, raw: object) -> FieldValue:
    """Return a stored value as a ``FieldValue``; ``None`` stays ``None``."""
    if raw is None:
        return None
    try:
        if spec.type is FieldType.INTEGER:
            return int(raw) if isinstance(raw, (int, float, str)) else None
        if spec.type is FieldType.FLOAT:
            return float(raw) if isinstance(raw, (int, float, str)) else None
        if spec.type is FieldType.BOOLEAN:
            return raw if isinstance(raw, bool) else bool(raw)
    except ValueError:
        raise IncompatibleCollectionError(
            f"Stored value {raw!r} of field '{spec.name}' is not a "
            f"{spec.type.value}."
        ) from None
    return raw if isinstance(raw, str) else str(raw)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _same_fields(
    schema: CollectionSchema, sent: Mapping[str, object], stored: Mapping[str, object]
) -> bool:
    """Compare a document as sent with the document as read back.

    Doubles may come back with fewer digits, so numbers match within 1e-9.
    """
    for spec in schema.fields:
        want, got = sent.get(spec.name), stored.get(spec.name)
        left, right = _number(want), _number(got)
        if left is not None and right is not None:
            if not math.isclose(left, right, rel_tol=1e-9):
                return False
        elif want != got or isinstance(want, bool) != isinstance(got, bool):
            return False
    return True


def _upload_batches(
    documents: Sequence[dict[str, object]],
) -> Iterator[list[dict[str, object]]]:
    """Split documents into requests under the count and byte limits."""
    batch: list[dict[str, object]] = []
    size = 0
    for document in documents:
        estimate = len(json.dumps(document, separators=(",", ":"))) + 32
        if batch and (
            len(batch) >= UPLOAD_MAX_DOCUMENTS or size + estimate > UPLOAD_MAX_BYTES
        ):
            yield batch
            batch, size = [], 0
        batch.append(document)
        size += estimate
    if batch:
        yield batch


# -- the adapter ---------------------------------------------------------------------


class AzureAISearchAdapter(BaseVectorStoreAdapter):
    """One Azure AI Search index as a HoloDeck collection."""

    def __init__(
        self,
        schema: CollectionSchema,
        settings: AzureSearchSettings,
        *,
        client_factory: ClientFactory = default_client_factory,
    ) -> None:
        super().__init__(schema, _CAPABILITIES)
        for spec in schema.fields:
            if spec.full_text and spec.type is not FieldType.STRING:
                raise SchemaError(
                    f"Field '{spec.name}' is {spec.type.value}; only string fields "
                    "can be full-text searched."
                )
        self._settings = settings
        self._index_name = normalize_index_name(settings.index_name or schema.name)
        self._client_factory = client_factory
        self._lock = asyncio.Lock()
        self._clients: ClientPair | None = None
        self._ready = False

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> AzureAISearchAdapter:
        """Build the adapter. No request is sent until first use."""
        return cls(schema, settings_from_connection(connection))

    @property
    def index_name(self) -> str:
        """Name of the Azure AI Search index that holds the collection."""
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
        except AzureError as error:
            text = f"{type(error).__name__}: {error.message or error}"
            if isinstance(error, ClientAuthenticationError):
                text = f"{text} Check the API key or the Microsoft Entra ID role."
            key = self._settings.api_key
            cause: BaseException | None = None if key and key in text else error
            status = error.status_code if isinstance(error, HttpResponseError) else None
            if status == 400:
                raise SchemaError(self._scrub(text)) from cause
            raise StoreConnectionError(PROVIDER, self._scrub(text)) from cause

    # -- clients ---------------------------------------------------------------------

    async def _pair(self) -> ClientPair:
        async with self._lock:
            if self._clients is None:
                self._clients = self._client_factory(self._settings, self._index_name)
            return self._clients

    async def _documents(self) -> SearchDocumentsClient:
        return (await self._pair()).documents

    async def _indexes(self) -> SearchIndexesClient:
        return (await self._pair()).indexes

    # -- lifecycle -------------------------------------------------------------------

    async def _describe(self) -> SearchIndex | None:
        try:
            return await (await self._indexes()).get_index(self._index_name)
        except ResourceNotFoundError:
            return None

    async def collection_exists(self) -> bool:
        with self._translated():
            return await self._describe() is not None

    async def ensure_collection(self) -> None:
        with self._translated():
            index = await self._describe()
            if index is None:
                try:
                    await (await self._indexes()).create_index(
                        build_index(self._schema, self._index_name)
                    )
                except ResourceExistsError:
                    logger.debug("Index %s was created concurrently", self._index_name)
                index = await self._describe()
                if index is None:
                    raise StoreConnectionError(
                        PROVIDER,
                        f"Index '{self._index_name}' was created but is not visible.",
                    )
            self._check_compatible(index)
            self._ready = True

    def _check_compatible(self, index: SearchIndex) -> None:
        by_name = {declared.name: declared for declared in index.fields}
        where = f"Index '{self._index_name}' (collection '{self._schema.name}')"
        key = by_name.get(self._schema.key_field)
        if key is None or not key.key:
            raise IncompatibleCollectionError(
                f"{where} has no key field '{self._schema.key_field}'."
            )
        if key.sortable is False or key.filterable is False:
            raise IncompatibleCollectionError(
                f"{where} has a key field that is not sortable and filterable, "
                "which enumeration needs. A key cannot be changed on an existing "
                "index: create the index with the adapter or delete it."
            )
        vector = by_name.get(self._schema.vector_field)
        if vector is None:
            raise IncompatibleCollectionError(
                f"{where} has no vector field '{self._schema.vector_field}'."
            )
        if (
            vector.vector_search_dimensions is not None
            and vector.vector_search_dimensions != self._schema.dimensions
        ):
            raise IncompatibleCollectionError(
                f"{where} has {vector.vector_search_dimensions} dimensions; "
                f"{self._schema.dimensions} are needed."
            )
        for spec in self._schema.fields:
            declared = by_name.get(spec.name)
            if declared is None:
                raise IncompatibleCollectionError(
                    f"{where} has no field '{spec.name}'."
                )
            if spec.indexed and declared.filterable is False:
                raise IncompatibleCollectionError(
                    f"{where}: field '{spec.name}' is not filterable."
                )
            if spec.full_text and declared.searchable is False:
                raise IncompatibleCollectionError(
                    f"{where}: field '{spec.name}' is not searchable."
                )

    async def delete_collection(self) -> None:
        self._ready = False
        with self._translated():
            try:
                await (await self._indexes()).delete_index(self._index_name)
            except ResourceNotFoundError:
                return
            deadline = time.monotonic() + VISIBILITY_TIMEOUT
            delay = _POLL_START
            while await self._describe() is not None:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Azure AI Search index %s is still listed after a delete.",
                        self._index_name,
                    )
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, _POLL_MAX)

    async def close(self) -> None:
        async with self._lock:
            pair, self._clients = self._clients, None
            self._ready = False
        if pair is None:
            return
        for closable in (pair.documents, pair.indexes, pair.credential):
            if closable is None:
                continue
            try:
                await closable.close()
            except AzureError as error:
                logger.debug(
                    "Closing an Azure AI Search client failed: %s",
                    self._scrub(str(error)),
                )

    # -- conversion --------------------------------------------------------------------

    def _select(self, *, vector: bool = False) -> list[str]:
        names = [self._schema.key_field, *self._schema.field_names]
        if vector:
            names.append(self._schema.vector_field)
        return names

    def _document(self, record: Record) -> dict[str, object]:
        document: dict[str, object] = {self._schema.key_field: encode_key(record.id)}
        for spec in self._schema.fields:
            document[spec.name] = _stored(spec, record.fields.get(spec.name))
        document[self._schema.vector_field] = _vector(record.vector or ())
        return document

    def _record(
        self, document: Mapping[str, object], *, vector: bool = False
    ) -> Record:
        raw_key = document.get(self._schema.key_field)
        if not isinstance(raw_key, str):
            raise StoreConnectionError(
                PROVIDER, f"A result from index '{self._index_name}' has no key."
            )
        fields = {
            spec.name: _read(spec, document.get(spec.name))
            for spec in self._schema.fields
        }
        embedding: list[float] | None = None
        if vector:
            stored = document.get(self._schema.vector_field)
            if isinstance(stored, Sequence) and not isinstance(stored, (str, bytes)):
                embedding = [
                    float(item) for item in stored if _number(item) is not None
                ]
        return Record(id=decode_key(raw_key), fields=fields, vector=embedding)

    def _filter_expression(
        self, record_filter: RecordFilter | None, *extra: str
    ) -> str | None:
        """Return an OData ``$filter`` for the exact matches plus ``extra`` clauses."""
        clauses: list[str] = []
        if record_filter is not None:
            for clause in record_filter.all_of:
                spec = self._schema.field(clause.field)
                if not spec.indexed:
                    raise SchemaError(
                        f"Field '{clause.field}' is not indexed, so Azure AI Search "
                        "cannot filter on it."
                    )
                clauses.append(f"{spec.name} eq {_odata_literal(spec, clause.value)}")
        clauses.extend(extra)
        return " and ".join(clauses) or None

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )

    def _text_field(self, name: str) -> str:
        spec = self._schema.field(name)
        if not spec.full_text:
            raise SchemaError(f"Field '{name}' is not full-text indexed.")
        return spec.name

    # -- write and point reads ---------------------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        with self._translated():
            documents = [self._document(record) for record in records]
            if not documents:
                return
            if not self._ready:
                await self.ensure_collection()
            client = await self._documents()
            for batch in _upload_batches(documents):
                try:
                    results = await client.merge_or_upload_documents(batch)
                except ResourceNotFoundError:
                    # The index was deleted behind this adapter: make it again.
                    await self.ensure_collection()
                    results = await client.merge_or_upload_documents(batch)
                self._check_results(results, "upload")
            last = documents[-1]
            await self._await_visible(str(last[self._schema.key_field]), expected=last)

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        keys: list[str] = []
        for record_id in dict.fromkeys(ids):
            try:
                keys.append(encode_key(record_id))
            except SchemaError:
                continue  # No such key can exist.
        if not keys:
            return []
        select = self._select(vector=include_vectors)
        gate = asyncio.Semaphore(GET_CONCURRENCY)
        with self._translated():
            client = await self._documents()

            async def fetch(key: str) -> Mapping[str, object] | None:
                async with gate:
                    try:
                        return await client.get_document(key, selected_fields=select)
                    except ResourceNotFoundError:
                        return None

            found = await asyncio.gather(*(fetch(key) for key in keys))
        return [
            self._record(document, vector=include_vectors)
            for document in found
            if document is not None
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        keys: list[str] = []
        for record_id in dict.fromkeys(ids):
            try:
                keys.append(encode_key(record_id))
            except SchemaError:
                continue
        if not keys:
            return
        with self._translated():
            client = await self._documents()
            try:
                for start in range(0, len(keys), DELETE_MAX_DOCUMENTS):
                    batch = keys[start : start + DELETE_MAX_DOCUMENTS]
                    results = await client.delete_documents(
                        [{self._schema.key_field: key} for key in batch]
                    )
                    self._check_results(results, "delete")
            except ResourceNotFoundError:
                return
            await self._await_visible(keys[-1], expected=None)

    async def count(self) -> int:
        with self._translated():
            try:
                return int(await (await self._documents()).get_document_count())
            except ResourceNotFoundError:
                return 0

    # -- filtered reads and enumeration ---------------------------------------------

    async def _scan(
        self, expression: str | None, limit: int | None, after: str | None = None
    ) -> list[Mapping[str, object]]:
        """Return documents in key order, at most ``limit``, after key ``after``."""
        key_field = self._schema.key_field
        rows: list[Mapping[str, object]] = []
        cursor = after
        client = await self._documents()
        while limit is None or len(rows) < limit:
            size = (
                SCAN_PAGE_SIZE
                if limit is None
                else min(SCAN_PAGE_SIZE, limit - len(rows))
            )
            clauses = [expression] if expression else []
            if cursor is not None:
                clauses.append(f"{key_field} gt {odata_string(cursor)}")
            try:
                results = await client.search(
                    "*",
                    filter=" and ".join(clauses) or None,
                    order_by=[f"{key_field} asc"],
                    select=self._select(),
                    top=size,
                )
                page = [document async for document in results]
            except ResourceNotFoundError:
                return rows
            rows.extend(page)
            if len(page) < size:
                break
            last = page[-1].get(key_field)
            if not isinstance(last, str) or (cursor is not None and last <= cursor):
                raise StoreConnectionError(
                    PROVIDER,
                    f"Index '{self._index_name}' did not return its keys in "
                    "ascending order, so enumeration cannot continue.",
                )
            cursor = last
        return rows

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        expression = self._filter_expression(record_filter)
        if limit is not None and limit <= 0:
            return []
        with self._translated():
            rows = await self._scan(expression, limit)
        return [self._record(row) for row in rows]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        if cursor is not None and not _KEY_PATTERN.match(cursor):
            raise SchemaError(f"Invalid cursor: {cursor!r}")
        with self._translated():
            rows = await self._scan(None, limit + 1, after=cursor)
        page = rows[:limit]
        next_cursor: str | None = None
        if len(rows) > limit:
            last = page[-1].get(self._schema.key_field)
            next_cursor = last if isinstance(last, str) else None
        return Page(
            records=tuple(self._record(row) for row in page), next_cursor=next_cursor
        )

    # -- search ------------------------------------------------------------------------

    def _hits(
        self, results: Sequence[Mapping[str, object]], kind: ScoreKind
    ) -> list[ScoredRecord]:
        hits: list[ScoredRecord] = []
        for document in results:
            raw = _number(document.get("@search.score")) or 0.0
            score = similarity_from_score(raw) if kind is ScoreKind.SIMILARITY else raw
            hits.append(
                ScoredRecord(record=self._record(document), score=score, kind=kind)
            )
        return hits

    async def _search(
        self,
        text: str | None,
        *,
        vector: Sequence[float] | None,
        neighbours: int,
        text_field: str | None,
        top: int,
        expression: str | None,
        kind: ScoreKind,
    ) -> list[ScoredRecord]:
        queries: list[VectorQuery] | None = None
        if vector is not None:
            queries = [
                VectorizedQuery(
                    vector=_vector(vector),
                    k_nearest_neighbors=neighbours,
                    fields=self._schema.vector_field,
                )
            ]
        with self._translated():
            client = await self._documents()
            try:
                results = await client.search(
                    text,
                    filter=expression,
                    query_type="simple" if text is not None else None,
                    search_fields=[text_field] if text_field else None,
                    search_mode="any" if text is not None else None,
                    select=self._select(),
                    top=top,
                    vector_queries=queries,
                    vector_filter_mode="preFilter" if queries else None,
                )
                documents = [document async for document in results]
            except ResourceNotFoundError:
                return []
        return self._hits(documents, kind)

    async def vector_search(self, query: HoloVectorQuery) -> list[ScoredRecord]:
        self._check_vector(query.vector)
        expression = self._filter_expression(query.filter)
        if query.top_k <= 0:
            return []
        top = min(query.top_k, SEARCH_MAX_TOP)
        return await self._search(
            None,
            vector=query.vector,
            neighbours=top,
            text_field=None,
            top=top,
            expression=expression,
            kind=ScoreKind.SIMILARITY,
        )

    async def lexical_search(self, query: LexicalQuery) -> list[ScoredRecord]:
        text_field = self._text_field(query.text_field)
        expression = self._filter_expression(query.filter)
        text = " ".join(_WORD.findall(query.text))
        if not text or query.top_k <= 0:
            return []
        return await self._search(
            text,
            vector=None,
            neighbours=0,
            text_field=text_field,
            top=min(query.top_k, SEARCH_MAX_TOP),
            expression=expression,
            kind=ScoreKind.LEXICAL,
        )

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        self._check_vector(query.vector)
        text_field = self._text_field(query.text_field)
        expression = self._filter_expression(query.filter)
        text = " ".join(_WORD.findall(query.text))
        if query.top_k <= 0:
            return []
        top = min(query.top_k, SEARCH_MAX_TOP)
        if not text:
            # No usable words: the vector side alone, still reported as fused.
            hits = await self._search(
                None,
                vector=query.vector,
                neighbours=top,
                text_field=None,
                top=top,
                expression=expression,
                kind=ScoreKind.SIMILARITY,
            )
            return [
                ScoredRecord(hit.record, hit.score, ScoreKind.FUSED) for hit in hits
            ]
        return await self._search(
            text,
            vector=query.vector,
            neighbours=min(
                max(HYBRID_POOL_FACTOR * query.top_k, HYBRID_MIN_NEIGHBOURS),
                SEARCH_MAX_TOP,
            ),
            text_field=text_field,
            top=top,
            expression=expression,
            kind=ScoreKind.FUSED,
        )

    # -- results and consistency ----------------------------------------------------

    def _check_results(self, results: Sequence[IndexingResult], action: str) -> None:
        failed = [result for result in results if not result.succeeded]
        if not failed:
            return
        first = failed[0]
        detail = (
            f"{action} failed for {len(failed)} of {len(results)} documents; first "
            f"{decode_key(first.key or '')!r}: HTTP {first.status_code} "
            f"{(first.error_message or '')[:200]}"
        )
        if all(result.status_code == 400 for result in failed):
            raise SchemaError(self._scrub(detail))
        raise StoreConnectionError(PROVIDER, self._scrub(detail))

    async def _await_visible(
        self, key: str, *, expected: Mapping[str, object] | None
    ) -> None:
        """Wait until search shows ``key`` as written, or gone if ``expected`` is None.

        Gives up after ``VISIBILITY_TIMEOUT`` seconds with a warning.
        """
        client = await self._documents()
        deadline = time.monotonic() + VISIBILITY_TIMEOUT
        delay = _POLL_START
        while True:
            results = await client.search(
                "*",
                filter=f"{self._schema.key_field} eq {odata_string(key)}",
                select=self._select(),
                top=1,
            )
            found = [document async for document in results]
            if expected is None:
                if not found:
                    return
            elif found and _same_fields(self._schema, expected, found[0]):
                return
            if time.monotonic() >= deadline:
                logger.warning(
                    "Azure AI Search index %s did not show a write within %.0f s; "
                    "reads may lag behind it.",
                    self._index_name,
                    VISIBILITY_TIMEOUT,
                )
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, _POLL_MAX)


def _vector(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise SchemaError("Vectors must hold finite numbers.")
    return vector
