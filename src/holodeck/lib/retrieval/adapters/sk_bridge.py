"""TEMPORARY bridge from Semantic Kernel collections to the retrieval contract.

This module is deleted in task T16 of the 042 exec plan, when every provider
has a native adapter. Until then it lets the tools move onto
``VectorStoreAdapter`` without a flag day: each class here wraps the SK
collection that ``holodeck.lib.vector_store.get_collection_factory`` builds
today and exposes it through the new protocol.

It is the only module in ``holodeck.lib.retrieval`` that may import
``semantic_kernel`` or use ``Any``/``getattr``. Do not add a provider here.
Write a native adapter instead.

Stored layout
-------------
``build_definition`` produces the SK collection definition from a
``CollectionSchema``. For the fields of the three record factories in
``vector_store.py`` it is equivalent to the definition they build: key field
``id``, vector field ``embedding`` (cosine similarity, ``schema.dimensions``),
and each data field with the same type and index flags. Field order differs
(the vector field is last), which does not change what a store holds.
Records cross SK as plain dicts, not as generated record classes.

Native operations
-----------------
SK cannot filter, page, count or hybrid-search the way the tools need, so the
provider classes below hold that logic (it used to live in the tools,
``collection_filter.py`` and ``keyword_search.py``; T4 removed it there).
Capabilities list what works today and nothing more.

Lifecycle
---------
The tools used ``async with collection`` per call. Leaving that block closes
the client of providers that manage their own (postgres pool, weaviate,
pinecone, azure) and, for qdrant, used to close a shared client. The bridge
enters the SK collection once, on first use, and leaves it in ``close``.
After ``close`` the next call builds a fresh collection from the factory, so
cleanup after close (a test fixture deleting its collection) still works.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from semantic_kernel.data.vector import (
    DistanceFunction,
    VectorStoreCollectionDefinition,
    VectorStoreField,
)

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    MissingClientError,
    RetrievalError,
    SchemaError,
    UnsupportedOperationError,
)
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    FieldType,
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

Row = dict[str, Any]

_SK_TYPES: dict[FieldType, str] = {
    FieldType.STRING: "str",
    FieldType.INTEGER: "int",
    FieldType.FLOAT: "float",
    FieldType.BOOLEAN: "bool",
}

_DEFAULTS: dict[FieldType, str | int | float | bool] = {
    FieldType.STRING: "",
    FieldType.INTEGER: 0,
    FieldType.FLOAT: 0.0,
    FieldType.BOOLEAN: False,
}

_BASE_OPERATIONS = frozenset(
    {Operation.UPSERT, Operation.GET, Operation.DELETE, Operation.VECTOR_SEARCH}
)
_FILTER_OPERATIONS = frozenset(
    {Operation.FIND, Operation.LIST, Operation.DELETE_WHERE, Operation.COUNT}
)

_NO_FILTER_HINT = (
    "The Semantic Kernel bridge has no native filter API for this provider."
)
_LEXICAL_HINT = "HoloDeck uses its BM25 index for this provider."
_HYBRID_HINT = "HoloDeck fuses vector and BM25 results itself."

_QDRANT_MIN_TOKEN_LEN = 2


class MissingPackageError(MissingClientError):
    """The client package of a provider without an install extra is missing."""

    def __init__(self, provider: str, package: str) -> None:
        super().__init__(provider, package)
        self.install_command = f"pip install {package}"
        self.args = (
            f"Vector store provider '{provider}' needs a client library that is "
            f"not installed. Install it with: {self.install_command}",
        )


def build_definition(schema: CollectionSchema) -> VectorStoreCollectionDefinition:
    """Return the SK definition for ``schema``."""
    fields: list[VectorStoreField] = [
        VectorStoreField("key", name=schema.key_field, type="str")
    ]
    for spec in schema.fields:
        fields.append(
            VectorStoreField(
                "data",
                name=spec.name,
                type=_SK_TYPES[spec.type],
                is_indexed=True if spec.indexed else None,
                is_full_text_indexed=True if spec.full_text else None,
            )
        )
    fields.append(
        VectorStoreField(
            "vector",
            name=schema.vector_field,
            type="float",
            dimensions=schema.dimensions,
            distance_function=DistanceFunction.COSINE_SIMILARITY,
        )
    )
    return VectorStoreCollectionDefinition(collection_name=schema.name, fields=fields)


def _coerce(field_type: FieldType, value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if field_type is FieldType.INTEGER:
        return int(value)
    if field_type is FieldType.FLOAT:
        return float(value)
    if field_type is FieldType.BOOLEAN:
        return bool(value)
    return value if isinstance(value, str) else str(value)


def _tokenize(text: str) -> list[str]:
    """Split a query into lowercase tokens, dropping short and repeated ones."""
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in re.findall(r"\w+", text.lower(), re.UNICODE):
        if len(raw) < _QDRANT_MIN_TOKEN_LEN or raw in seen:
            continue
        seen.add(raw)
        tokens.append(raw)
    return tokens


class SKBridgeAdapter(BaseVectorStoreAdapter):
    """One SK collection exposed through ``VectorStoreAdapter``.

    Subclasses set the class variables and, where the provider has a native
    API, override the ``_*_rows`` hooks. A provider whose capabilities lack
    FIND/LIST/DELETE_WHERE/COUNT needs no hooks.
    """

    PROVIDER: ClassVar[str]
    EXTRA: ClassVar[str | None] = None
    PACKAGE: ClassVar[str | None] = None
    OPERATIONS: ClassVar[frozenset[Operation]] = _BASE_OPERATIONS
    DROP_ON_CLOSE: ClassVar[bool] = True
    SCORE_IS_DISTANCE: ClassVar[bool] = False

    def __init__(
        self, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> None:
        super().__init__(schema, self.declared_capabilities())
        for spec in schema.fields:
            if not spec.name.isidentifier():
                raise SchemaError(
                    f"Field name '{spec.name}' is not a valid identifier; "
                    "the Semantic Kernel bridge builds filters from field names."
                )
        self._definition = build_definition(schema)
        self._factory = self._make_factory(dict(connection))
        self._collection: Any = self._factory()
        self._entered = False
        self._lock = asyncio.Lock()

    @classmethod
    def declared_capabilities(cls) -> Capabilities:
        """Return what this provider supports through the bridge today."""
        hints: dict[Operation, str] = {Operation.LEXICAL_SEARCH: _LEXICAL_HINT}
        if Operation.HYBRID_SEARCH not in cls.OPERATIONS:
            hints[Operation.HYBRID_SEARCH] = _HYBRID_HINT
        for operation in _FILTER_OPERATIONS:
            if operation not in cls.OPERATIONS:
                hints[operation] = _NO_FILTER_HINT
        return Capabilities(cls.PROVIDER, cls.OPERATIONS, hints)

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> SKBridgeAdapter:
        """Build the adapter, or raise ``MissingClientError``."""
        try:
            return cls(schema, connection)
        except ImportError as error:
            if cls.EXTRA is not None:
                raise MissingClientError(cls.PROVIDER, cls.EXTRA) from error
            if cls.PACKAGE is not None:
                raise MissingPackageError(cls.PROVIDER, cls.PACKAGE) from error
            raise

    def _make_factory(self, connection: dict[str, Any]) -> Callable[[], Any]:
        from holodeck.lib.vector_store import get_collection_factory

        return get_collection_factory(
            self.PROVIDER,
            self._schema.dimensions,
            record_class=dict,
            definition=self._definition,
            **connection,
        )

    # -- lifecycle ---------------------------------------------------------

    async def _open(self) -> Any:
        async with self._lock:
            if self._collection is None:
                self._collection = self._factory()
            if not self._entered:
                await self._collection.__aenter__()
                self._entered = True
            return self._collection

    async def _release(self, collection: Any) -> None:
        await collection.__aexit__(None, None, None)

    async def close(self) -> None:
        async with self._lock:
            collection = self._collection
            if collection is None or not self._entered:
                return
            self._entered = False
            try:
                await self._release(collection)
            finally:
                if self.DROP_ON_CLOSE:
                    self._collection = None

    async def collection_exists(self) -> bool:
        collection = await self._open()
        return bool(await collection.collection_exists())

    async def ensure_collection(self) -> None:
        collection = await self._open()
        if not await collection.collection_exists():
            await collection.ensure_collection_exists()

    async def delete_collection(self) -> None:
        collection = await self._open()
        if await collection.collection_exists():
            await collection.ensure_collection_deleted()

    # -- record conversion -------------------------------------------------

    def _to_store(self, record: Record) -> Row:
        row: Row = {self._schema.key_field: record.id}
        for spec in self._schema.fields:
            value = record.fields.get(spec.name)
            row[spec.name] = _DEFAULTS[spec.type] if value is None else value
        row[self._schema.vector_field] = [float(v) for v in record.vector or ()]
        return row

    def _to_record(self, row: Mapping[str, Any], *, vectors: bool = False) -> Record:
        fields = {
            spec.name: _coerce(spec.type, row[spec.name])
            for spec in self._schema.fields
            if spec.name in row
        }
        vector: list[float] | None = None
        if vectors:
            raw = row.get(self._schema.vector_field)
            if raw is not None:
                vector = [float(v) for v in raw]
        return Record(id=str(row[self._schema.key_field]), fields=fields, vector=vector)

    def _check_filter(self, record_filter: RecordFilter | None) -> None:
        if record_filter is not None:
            for clause in record_filter.all_of:
                self._schema.field(clause.field)

    def _filter_expression(self, record_filter: RecordFilter | None) -> str | None:
        """Return an SK filter string such as ``lambda x: x.a == 'b'``."""
        self._check_filter(record_filter)
        if record_filter is None or not record_filter.all_of:
            return None
        clauses = [f"x.{c.field} == {c.value!r}" for c in record_filter.all_of]
        return "lambda x: " + " and ".join(clauses)

    def _similarity(self, raw: float) -> float:
        return clamp_similarity(1.0 - raw if self.SCORE_IS_DISTANCE else raw)

    def _check_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(vector)} dimensions; "
                f"schema '{self._schema.name}' needs {self._schema.dimensions}."
            )

    # -- write and point reads ---------------------------------------------

    async def upsert(self, records: Sequence[Record]) -> None:
        self._capabilities.require(Operation.UPSERT)
        self._validate(records)
        if not records:
            return
        collection = await self._open()
        await collection.upsert([self._to_store(record) for record in records])

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        self._capabilities.require(Operation.GET)
        if not ids:
            return []
        collection = await self._open()
        raw = await collection.get(list(ids), include_vectors=include_vectors)
        by_id = {
            record.id: record
            for record in (
                self._to_record(row, vectors=include_vectors) for row in raw or []
            )
        }
        return [by_id[key] for key in ids if key in by_id]

    async def delete(self, ids: Sequence[str]) -> None:
        self._capabilities.require(Operation.DELETE)
        if not ids:
            return
        collection = await self._open()
        await collection.delete(list(ids))

    # -- filter, page, count -----------------------------------------------

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        raise UnsupportedOperationError(self.provider, Operation.FIND.value)

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        raise UnsupportedOperationError(self.provider, Operation.LIST.value)

    async def _count_rows(self) -> int:
        raise UnsupportedOperationError(self.provider, Operation.COUNT.value)

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        self._capabilities.require(Operation.FIND)
        self._check_filter(record_filter)
        rows = await self._find_rows(record_filter, limit)
        return [self._to_record(row) for row in rows]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        self._capabilities.require(Operation.LIST)
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        rows, next_cursor = await self._page_rows(cursor, limit)
        return Page(
            records=tuple(self._to_record(row) for row in rows),
            next_cursor=next_cursor,
        )

    async def count(self) -> int:
        self._capabilities.require(Operation.COUNT)
        if not await self.collection_exists():
            return 0
        return await self._count_rows()

    # -- search --------------------------------------------------------------

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        self._capabilities.require(Operation.VECTOR_SEARCH)
        self._check_vector(query.vector)
        if query.top_k <= 0:
            return []
        expression = self._filter_expression(query.filter)
        collection = await self._open()
        options: dict[str, Any] = {}
        if expression is not None:
            options["filter"] = expression
        try:
            results = await collection.search(
                vector=[float(v) for v in query.vector], top=query.top_k, **options
            )
        except NotImplementedError as error:
            raise UnsupportedOperationError(
                self.provider, "filtered vector search", str(error)
            ) from error
        hits: list[ScoredRecord] = []
        async for result in results.results:
            hits.append(
                ScoredRecord(
                    record=self._to_record(result.record),
                    score=self._similarity(float(result.score or 0.0)),
                    kind=ScoreKind.SIMILARITY,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: query.top_k]

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        self._capabilities.require(Operation.HYBRID_SEARCH)
        self._check_vector(query.vector)
        self._schema.field(query.text_field)
        if query.top_k <= 0:
            return []
        expression = self._filter_expression(query.filter)
        collection = await self._open()
        options: dict[str, Any] = {}
        if expression is not None:
            options["filter"] = expression
        results = await collection.hybrid_search(
            query.text,
            vector=[float(v) for v in query.vector],
            vector_property_name=self._schema.vector_field,
            additional_property_name=query.text_field,
            top=query.top_k,
            **options,
        )
        hits: list[ScoredRecord] = []
        async for result in results.results:
            hits.append(
                ScoredRecord(
                    record=self._to_record(result.record),
                    score=float(result.score or 0.0),
                    kind=ScoreKind.FUSED,
                )
            )
        return hits[: query.top_k]


# -- in-memory (SK's collection; test target and development) ----------------


class SKInMemoryAdapter(SKBridgeAdapter):
    """SK's in-memory collection. Not registered: ``in-memory`` is native."""

    PROVIDER = "in-memory"
    OPERATIONS = _BASE_OPERATIONS | _FILTER_OPERATIONS
    DROP_ON_CLOSE = False

    def __init__(
        self, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> None:
        super().__init__(schema, connection)
        # SK's collection always reports that it exists.
        self._exists = False

    async def collection_exists(self) -> bool:
        return self._exists

    async def ensure_collection(self) -> None:
        self._exists = True

    async def delete_collection(self) -> None:
        collection = await self._open()
        await collection.ensure_collection_deleted()
        self._exists = False

    async def upsert(self, records: Sequence[Record]) -> None:
        await super().upsert(records)
        self._exists = True

    async def _all_rows(self) -> list[Row]:
        collection = await self._open()
        return [dict(row) for row in collection.inner_storage.values()]

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        rows = []
        for row in await self._all_rows():
            if record_filter is None or record_filter.matches(self._to_record(row)):
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        try:
            start = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        rows = await self._all_rows()
        end = start + limit
        return rows[start:end], str(end) if end < len(rows) else None

    async def _count_rows(self) -> int:
        return len(await self._all_rows())


# -- qdrant -------------------------------------------------------------------

# A key in the payload that keeps the HoloDeck ID when it is not a UUID.
_QDRANT_ID_KEY = "holodeck_id"


class SKQdrantAdapter(SKBridgeAdapter):
    """Qdrant through SK's collection and its ``AsyncQdrantClient``.

    SK serializes records to points correctly, but Qdrant accepts only UUID or
    integer point IDs and HoloDeck IDs are free text. Writes and reads that
    carry IDs therefore use the client directly: a UUID ID is stored as it is,
    anything else is stored as a UUIDv5 and the original ID goes into the
    ``holodeck_id`` payload key. Both forms use the same point layout as SK.
    """

    PROVIDER = "qdrant"
    EXTRA = "qdrant"
    OPERATIONS = _BASE_OPERATIONS | _FILTER_OPERATIONS | {Operation.HYBRID_SEARCH}

    def __init__(
        self, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> None:
        if _QDRANT_ID_KEY in schema.field_names:
            raise SchemaError(f"Field name '{_QDRANT_ID_KEY}' is reserved for qdrant.")
        super().__init__(schema, connection)
        self._namespace = uuid.uuid5(
            uuid.NAMESPACE_URL, f"holodeck/retrieval/{schema.name}"
        )
        self._indexes_ensured = False

    async def _release(self, collection: Any) -> None:
        try:
            await collection.__aexit__(None, None, None)
        finally:
            await collection.qdrant_client.close()

    # IDs

    def _store_id(self, record_id: str) -> str:
        try:
            return str(uuid.UUID(record_id))
        except ValueError:
            return str(uuid.uuid5(self._namespace, record_id))

    def _row(self, point: Any, *, vectors: bool = False) -> Row:
        payload = dict(point.payload or {})
        original = payload.pop(_QDRANT_ID_KEY, None)
        row: Row = {
            self._schema.key_field: original if original is not None else str(point.id),
            **payload,
        }
        if vectors and point.vector:
            vector = point.vector
            row[self._schema.vector_field] = (
                vector[self._schema.vector_field]
                if isinstance(vector, dict)
                else vector
            )
        return row

    def _qdrant_filter(self, record_filter: RecordFilter | None) -> Any:
        from qdrant_client.http import models as qm

        self._check_filter(record_filter)
        if record_filter is None or not record_filter.all_of:
            return None
        conditions: list[Any] = []
        for clause in record_filter.all_of:
            if clause.value is None:
                conditions.append(
                    qm.IsNullCondition(is_null=qm.PayloadField(key=clause.field))
                )
            elif isinstance(clause.value, float):
                conditions.append(
                    qm.FieldCondition(
                        key=clause.field,
                        range=qm.Range(gte=clause.value, lte=clause.value),
                    )
                )
            else:
                conditions.append(
                    qm.FieldCondition(
                        key=clause.field, match=qm.MatchValue(value=clause.value)
                    )
                )
        return qm.Filter(must=conditions)

    # collection

    async def ensure_collection(self) -> None:
        collection = await self._open()
        if not await collection.collection_exists():
            await collection.ensure_collection_exists()
        await self._ensure_payload_indexes(collection)

    async def delete_collection(self) -> None:
        await super().delete_collection()
        self._indexes_ensured = False

    async def _ensure_payload_indexes(self, collection: Any) -> None:
        """Create the payload indexes that SK's connector never creates.

        Qdrant 1.18 and later reject a filter on an unindexed field. The call
        is idempotent for matching parameters.
        """
        if self._indexes_ensured:
            return
        from qdrant_client.http import models as qm

        client = collection.qdrant_client
        for spec in self._schema.fields:
            schema: Any
            if spec.full_text and not spec.indexed:
                schema = qm.TextIndexParams(
                    type=qm.TextIndexType.TEXT,
                    tokenizer=qm.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                )
            elif spec.indexed and spec.type is FieldType.STRING:
                schema = qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD)
            elif spec.indexed and spec.type is FieldType.INTEGER:
                schema = qm.IntegerIndexParams(type=qm.IntegerIndexType.INTEGER)
            elif spec.indexed and spec.type is FieldType.FLOAT:
                schema = qm.FloatIndexParams(type=qm.FloatIndexType.FLOAT)
            elif spec.indexed and spec.type is FieldType.BOOLEAN:
                schema = qm.BoolIndexParams(type=qm.BoolIndexType.BOOL)
            else:
                continue
            try:
                await client.create_payload_index(
                    collection_name=collection.collection_name,
                    field_name=spec.name,
                    field_schema=schema,
                )
            except Exception as error:  # an index failure must not block writes
                logger.warning(
                    "Qdrant payload index failed for '%s' on '%s': %s",
                    spec.name,
                    collection.collection_name,
                    error,
                )
        self._indexes_ensured = True

    # records

    async def upsert(self, records: Sequence[Record]) -> None:
        self._capabilities.require(Operation.UPSERT)
        self._validate(records)
        if not records:
            return
        from qdrant_client.http import models as qm

        collection = await self._open()
        points = []
        for record in records:
            store_id = self._store_id(record.id)
            payload = {
                spec.name: (
                    _DEFAULTS[spec.type]
                    if record.fields.get(spec.name) is None
                    else record.fields[spec.name]
                )
                for spec in self._schema.fields
            }
            if store_id != record.id:
                payload[_QDRANT_ID_KEY] = record.id
            points.append(
                qm.PointStruct(
                    id=store_id,
                    vector={
                        self._schema.vector_field: [
                            float(v) for v in record.vector or ()
                        ]
                    },
                    payload=payload,
                )
            )
        await collection.qdrant_client.upsert(
            collection_name=collection.collection_name, points=points
        )

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        self._capabilities.require(Operation.GET)
        if not ids:
            return []
        collection = await self._open()
        points = await collection.qdrant_client.retrieve(
            collection_name=collection.collection_name,
            ids=[self._store_id(key) for key in ids],
            with_payload=True,
            with_vectors=include_vectors,
        )
        by_store_id = {
            str(point.id): self._to_record(
                self._row(point, vectors=include_vectors), vectors=include_vectors
            )
            for point in points
        }
        return [
            by_store_id[self._store_id(key)]
            for key in ids
            if self._store_id(key) in by_store_id
        ]

    async def delete(self, ids: Sequence[str]) -> None:
        self._capabilities.require(Operation.DELETE)
        if not ids:
            return
        from qdrant_client.http import models as qm

        collection = await self._open()
        await collection.qdrant_client.delete(
            collection_name=collection.collection_name,
            points_selector=qm.PointIdsList(
                points=[self._store_id(key) for key in ids]
            ),
        )

    async def _scroll(
        self, record_filter: RecordFilter | None, offset: Any, limit: int
    ) -> tuple[list[Row], Any]:
        collection = await self._open()
        points, next_offset = await collection.qdrant_client.scroll(
            collection_name=collection.collection_name,
            scroll_filter=self._qdrant_filter(record_filter),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [self._row(point) for point in points], next_offset

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        results: list[Row] = []
        offset: Any = None
        while True:
            page_limit = DEFAULT_PAGE_SIZE
            if limit is not None:
                remaining = limit - len(results)
                if remaining <= 0:
                    break
                page_limit = min(page_limit, remaining)
            rows, offset = await self._scroll(record_filter, offset, page_limit)
            results.extend(rows)
            if offset is None:
                break
        return results

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        offset: Any = None
        if cursor is not None:
            offset = int(cursor) if cursor.isdigit() else cursor
        rows, next_offset = await self._scroll(None, offset, limit)
        return rows, None if next_offset is None else str(next_offset)

    async def _count_rows(self) -> int:
        collection = await self._open()
        result = await collection.qdrant_client.count(
            collection_name=collection.collection_name, exact=True
        )
        return int(result.count)

    # search

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        self._capabilities.require(Operation.VECTOR_SEARCH)
        self._check_vector(query.vector)
        if query.top_k <= 0:
            return []
        collection = await self._open()
        response = await collection.qdrant_client.query_points(
            collection_name=collection.collection_name,
            query=[float(v) for v in query.vector],
            using=self._schema.vector_field,
            query_filter=self._qdrant_filter(query.filter),
            limit=query.top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            ScoredRecord(
                record=self._to_record(self._row(point)),
                score=clamp_similarity(point.score),
                kind=ScoreKind.SIMILARITY,
            )
            for point in response.points
        ]

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Vector and ``MatchText`` legs fused with RRF.

        SK's own keyword leg matches whole values and finds nothing in a
        free-text field, so this builds the query on the client. Each token
        of the query becomes a ``MatchText`` clause, OR-combined. Both legs
        prefetch ``max(4 * top_k, 20)`` candidates so the fusion has
        something to merge. A query without tokens searches by vector only.
        """
        self._capabilities.require(Operation.HYBRID_SEARCH)
        self._check_vector(query.vector)
        self._schema.field(query.text_field)
        if query.top_k <= 0:
            return []
        from qdrant_client.http import models as qm

        tokens = _tokenize(query.text)
        prefetch_limit = max(query.top_k * 4, 20)
        base_filter = self._qdrant_filter(query.filter)
        prefetch: list[qm.Prefetch] = [
            qm.Prefetch(
                query=[float(v) for v in query.vector],
                using=self._schema.vector_field,
                filter=base_filter,
                limit=prefetch_limit,
            )
        ]
        if tokens:
            text_filter = qm.Filter(
                should=[
                    qm.FieldCondition(
                        key=query.text_field, match=qm.MatchText(text=token)
                    )
                    for token in tokens
                ]
            )
            if base_filter is not None:
                text_filter = qm.Filter(must=[base_filter, text_filter])
            prefetch.append(qm.Prefetch(filter=text_filter, limit=prefetch_limit))
        collection = await self._open()
        response = await collection.qdrant_client.query_points(
            collection_name=collection.collection_name,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=query.top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            ScoredRecord(
                record=self._to_record(self._row(point)),
                score=float(point.score),
                kind=ScoreKind.FUSED,
            )
            for point in response.points
        ]


# -- chromadb -----------------------------------------------------------------


class SKChromaAdapter(SKBridgeAdapter):
    """ChromaDB through SK's collection and the chroma client it wraps.

    SK writes with ``add``, which ignores an existing ID, so upsert goes to
    the chroma collection directly: existing IDs are deleted, then all
    records are added. SK returns the cosine distance as the score, which
    becomes ``1 - distance`` here.
    """

    PROVIDER = "chromadb"
    EXTRA = "chromadb"
    OPERATIONS = _BASE_OPERATIONS | _FILTER_OPERATIONS
    DROP_ON_CLOSE = False
    SCORE_IS_DISTANCE = True

    def _native(self, collection: Any) -> Any:
        return collection._get_collection()

    def _where(self, record_filter: RecordFilter | None) -> dict[str, Any] | None:
        self._check_filter(record_filter)
        if record_filter is None or not record_filter.all_of:
            return None
        clauses = [{c.field: c.value} for c in record_filter.all_of]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _rows(self, raw: Mapping[str, Any]) -> list[Row]:
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        rows: list[Row] = []
        for index, record_id in enumerate(ids):
            row: Row = {self._schema.key_field: record_id}
            if index < len(metadatas) and isinstance(metadatas[index], dict):
                row.update(metadatas[index])
            rows.append(row)
        return rows

    async def upsert(self, records: Sequence[Record]) -> None:
        self._capabilities.require(Operation.UPSERT)
        self._validate(records)
        if not records:
            return
        collection = await self._open()
        rows = [self._to_store(record) for record in records]
        vector_field = self._schema.vector_field
        key_field = self._schema.key_field
        metadatas = [
            {k: v for k, v in row.items() if k not in (key_field, vector_field)}
            for row in rows
        ]
        ids = [row[key_field] for row in rows]
        native = self._native(collection)
        # Replace with delete + add, not chroma's upsert: against the Chroma
        # 1.0 server, records written with upsert were intermittently missing
        # from query results (about 1 run in 30 of the retrieval goldens;
        # 0 in 60 with add). See the 042 exec plan, T4.
        existing = native.get(ids=ids, include=[]).get("ids") or []
        if existing:
            native.delete(ids=list(existing))
        native.add(
            ids=ids,
            embeddings=[row[vector_field] for row in rows],
            metadatas=metadatas if metadatas[0] else None,
        )

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        collection = await self._open()
        args: dict[str, Any] = {"include": ["metadatas"]}
        where = self._where(record_filter)
        if where is not None:
            args["where"] = where
        if limit is not None:
            args["limit"] = limit
        return self._rows(self._native(collection).get(**args))

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        collection = await self._open()
        raw = self._native(collection).get(
            include=["metadatas"], limit=limit + 1, offset=offset
        )
        rows = self._rows(raw)
        if len(rows) > limit:
            return rows[:limit], str(offset + limit)
        return rows, None

    async def _count_rows(self) -> int:
        collection = await self._open()
        return int(self._native(collection).count())


# -- postgres -----------------------------------------------------------------


class SKPostgresAdapter(SKBridgeAdapter):
    """PostgreSQL (pgvector) through SK's collection and its connection pool."""

    PROVIDER = "postgres"
    EXTRA = "postgres"
    OPERATIONS = _BASE_OPERATIONS | _FILTER_OPERATIONS

    async def _select(
        self,
        record_filter: RecordFilter | None,
        *,
        order: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Row]:
        from psycopg import sql

        collection = await self._open()
        pool = collection.connection_pool
        if pool is None:
            raise RetrievalError("postgres connection pool is not initialised.")
        definition = collection.definition
        key_name = definition.key_field_storage_name
        columns = [key_name] + [
            f.storage_name or f.name for f in definition.data_fields
        ]
        query = sql.SQL("SELECT {cols} FROM {schema}.{table}").format(
            cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            schema=sql.Identifier(collection.db_schema),
            table=sql.Identifier(collection.collection_name),
        )
        params: list[Any] = []
        if record_filter is not None and record_filter.all_of:
            conditions = []
            for clause in record_filter.all_of:
                if clause.value is None:
                    conditions.append(
                        sql.SQL("{f} IS NULL").format(f=sql.Identifier(clause.field))
                    )
                else:
                    conditions.append(
                        sql.SQL("{f} = %s").format(f=sql.Identifier(clause.field))
                    )
                    params.append(clause.value)
            query = sql.SQL("{q} WHERE {w}").format(
                q=query, w=sql.SQL(" AND ").join(conditions)
            )
        if order:
            query = sql.SQL("{q} ORDER BY {k}").format(
                q=query, k=sql.Identifier(key_name)
            )
        if limit is not None:
            query = sql.SQL("{q} LIMIT %s").format(q=query)
            params.append(limit)
        if offset is not None:
            query = sql.SQL("{q} OFFSET %s").format(q=query)
            params.append(offset)
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(query, tuple(params))
            fetched = await cur.fetchall()
        rows: list[Row] = []
        for values in fetched:
            row = dict(zip(columns, values, strict=True))
            row[self._schema.key_field] = row[key_name]
            rows.append(row)
        return rows

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        return await self._select(record_filter, limit=limit)

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        rows = await self._select(None, order=True, limit=limit + 1, offset=offset)
        if len(rows) > limit:
            return rows[:limit], str(offset + limit)
        return rows, None

    async def _count_rows(self) -> int:
        from psycopg import sql

        collection = await self._open()
        query = sql.SQL("SELECT count(*) FROM {schema}.{table}").format(
            schema=sql.Identifier(collection.db_schema),
            table=sql.Identifier(collection.collection_name),
        )
        async with (
            collection.connection_pool.connection() as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(query)
            result = await cur.fetchone()
        return int(result[0]) if result else 0


# -- weaviate -----------------------------------------------------------------


class SKWeaviateAdapter(SKBridgeAdapter):
    """Weaviate through SK's collection and its async client."""

    PROVIDER = "weaviate"
    OPERATIONS = _BASE_OPERATIONS | _FILTER_OPERATIONS | {Operation.HYBRID_SEARCH}
    PACKAGE = "weaviate-client"

    def _weaviate_filter(self, record_filter: RecordFilter | None) -> Any:
        from weaviate.classes.query import Filter

        if record_filter is None or not record_filter.all_of:
            return None
        clauses = [
            Filter.by_property(c.field).equal(c.value) for c in record_filter.all_of
        ]
        return clauses[0] if len(clauses) == 1 else Filter.all_of(clauses)

    async def _fetch(
        self, record_filter: RecordFilter | None, limit: int, offset: int
    ) -> list[Row]:
        collection = await self._open()
        wcoll = collection.async_client.collections.get(collection.collection_name)
        response = await wcoll.query.fetch_objects(
            filters=self._weaviate_filter(record_filter),
            limit=limit,
            offset=offset,
            include_vector=False,
        )
        return [
            {self._schema.key_field: str(obj.uuid), **dict(obj.properties or {})}
            for obj in response.objects or []
        ]

    async def _find_rows(
        self, record_filter: RecordFilter | None, limit: int | None
    ) -> list[Row]:
        results: list[Row] = []
        offset = 0
        while True:
            page_limit = DEFAULT_PAGE_SIZE
            if limit is not None:
                remaining = limit - len(results)
                if remaining <= 0:
                    break
                page_limit = min(page_limit, remaining)
            rows = await self._fetch(record_filter, page_limit, offset)
            results.extend(rows)
            if len(rows) < page_limit:
                break
            offset += page_limit
        return results

    async def _page_rows(
        self, cursor: str | None, limit: int
    ) -> tuple[list[Row], str | None]:
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        rows = await self._fetch(None, limit + 1, offset)
        if len(rows) > limit:
            return rows[:limit], str(offset + limit)
        return rows, None

    async def _count_rows(self) -> int:
        collection = await self._open()
        wcoll = collection.async_client.collections.get(collection.collection_name)
        result = await wcoll.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)


# -- providers without native filter operations --------------------------------


class SKAzureAISearchAdapter(SKBridgeAdapter):
    """Azure AI Search through SK's collection."""

    PROVIDER = "azure-ai-search"
    OPERATIONS = _BASE_OPERATIONS | {Operation.HYBRID_SEARCH}
    PACKAGE = "azure-search-documents"


class SKFaissAdapter(SKBridgeAdapter):
    """FAISS through SK's collection."""

    PROVIDER = "faiss"
    PACKAGE = "faiss-cpu"


class SKCosmosMongoAdapter(SKBridgeAdapter):
    """Azure Cosmos DB (MongoDB API) through SK's collection."""

    PROVIDER = "azure-cosmos-mongo"
    PACKAGE = "pymongo"


class SKCosmosNoSqlAdapter(SKBridgeAdapter):
    """Azure Cosmos DB (NoSQL API) through SK's collection."""

    PROVIDER = "azure-cosmos-nosql"
    OPERATIONS = _BASE_OPERATIONS | {Operation.HYBRID_SEARCH}
    PACKAGE = "azure-cosmos"


class SKSqlServerAdapter(SKBridgeAdapter):
    """SQL Server through SK's collection."""

    PROVIDER = "sql-server"
    PACKAGE = "pyodbc"


class SKPineconeAdapter(SKBridgeAdapter):
    """Pinecone through SK's collection. Its query API requires a vector."""

    PROVIDER = "pinecone"
    EXTRA = "pinecone"
