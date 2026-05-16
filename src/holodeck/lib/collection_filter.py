"""Backend-aware filter helpers for SK vector store collections.

SK's :meth:`collection.get` raises ``NotImplementedError("Get without keys is
not yet implemented.")`` on qdrant, postgres, weaviate, pinecone, mongodb,
in-memory, and others whenever called with a payload filter but no explicit
keys. Chroma silently ignores the filter and returns the first N records,
which is worse. Only Oracle implements the operation properly.

The :class:`holodeck.tools.hierarchical_document_tool.HierarchicalDocumentTool`
needs three filter-based operations to make incremental ingestion work:
look up records by ``source_path`` (cache check), delete records by
``source_path``, and load every record (chunk reload after a skip).

This module drops down to each backend's native client to implement those
operations. Records are returned as plain ``dict`` payloads so callers don't
have to round-trip through SK's record_class. Vectors are intentionally
omitted — the call sites never need them and skipping them keeps qdrant
scrolls and postgres SELECTs cheap.

Supported providers: ``qdrant``, ``chromadb``, ``postgres``, ``weaviate``,
``in-memory``. ``pinecone`` is unsupported because the pinecone query API
requires a vector — use a sidecar manifest if you need cache invalidation
on pinecone.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel page size for scroll / SELECT-style pagination. 512 keeps the
# round-trips bounded for large collections without blowing up memory on
# typical chunked-document workloads (<~50k chunks per source).
_PAGE_SIZE = 512

# Providers this module handles natively. Anything else gets a clear error
# from :func:`_unsupported` so callers know to add a branch.
_SUPPORTED = frozenset({"qdrant", "chromadb", "postgres", "weaviate", "in-memory"})


def _unsupported(operation: str, provider: str) -> NotImplementedError:
    return NotImplementedError(
        f"collection_filter.{operation} does not support provider '{provider}'. "
        f"Supported providers: {sorted(_SUPPORTED)}."
    )


async def find_records_by_field(
    provider: str,
    collection: Any,
    field: str,
    value: Any,
) -> list[dict[str, Any]]:
    """Return all records whose ``field`` equals ``value`` as payload dicts.

    Each returned dict contains the record id under the key ``"id"`` plus
    every payload/property field (vectors excluded). For backends whose
    native API doesn't yield the id alongside the payload (chromadb,
    weaviate), the helper injects it.

    Args:
        provider: One of ``qdrant``, ``chromadb``, ``postgres``, ``weaviate``,
            ``in-memory``. Provider names match
            :func:`holodeck.lib.vector_store.get_collection_factory`.
        collection: The SK collection instance (already entered as an async
            context if the backend requires it). The helper accesses the
            backend's native client attribute on this object.
        field: Payload field name to filter on (e.g. ``"source_path"``).
        value: Exact value to match.

    Returns:
        List of dicts in insertion order. Empty list if nothing matches.

    Raises:
        NotImplementedError: If ``provider`` is not in the supported set.
    """
    if provider == "qdrant":
        return await _find_qdrant(collection, field, value)
    if provider == "chromadb":
        return _find_chromadb(collection, field, value)
    if provider == "postgres":
        return await _find_postgres(collection, field, value)
    if provider == "weaviate":
        return await _find_weaviate(collection, field, value)
    if provider == "in-memory":
        return _find_in_memory(collection, field, value)
    raise _unsupported("find_records_by_field", provider)


async def find_all_records(
    provider: str,
    collection: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return every record in the collection as payload dicts.

    Args:
        provider: See :func:`find_records_by_field`.
        collection: See :func:`find_records_by_field`.
        limit: Optional cap on returned records. ``None`` means "all".

    Returns:
        List of dicts in insertion order (or backend-native order — qdrant
        scroll order, postgres unordered, etc.).

    Raises:
        NotImplementedError: If ``provider`` is not in the supported set.
    """
    if provider == "qdrant":
        return await _find_qdrant(collection, field=None, value=None, limit=limit)
    if provider == "chromadb":
        return _find_chromadb(collection, field=None, value=None, limit=limit)
    if provider == "postgres":
        return await _find_postgres(collection, field=None, value=None, limit=limit)
    if provider == "weaviate":
        return await _find_weaviate(collection, field=None, value=None, limit=limit)
    if provider == "in-memory":
        return _find_in_memory(collection, field=None, value=None, limit=limit)
    raise _unsupported("find_all_records", provider)


async def _find_qdrant(
    collection: Any,
    field: str | None,
    value: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Scroll a qdrant collection optionally filtered by a payload field.

    Uses ``qdrant_client.scroll`` directly because SK's ``_inner_get``
    raises NotImplementedError for keyless calls.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    scroll_filter: Filter | None = None
    if field is not None:
        scroll_filter = Filter(
            must=[FieldCondition(key=field, match=MatchValue(value=value))]
        )

    results: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        page_limit = _PAGE_SIZE
        if limit is not None:
            remaining = limit - len(results)
            if remaining <= 0:
                break
            page_limit = min(_PAGE_SIZE, remaining)

        points, offset = await collection.qdrant_client.scroll(
            collection_name=collection.collection_name,
            scroll_filter=scroll_filter,
            limit=page_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            payload["id"] = str(point.id)
            results.append(payload)
        if offset is None:
            break
    return results


def _find_chromadb(
    collection: Any,
    field: str | None,
    value: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch from a chromadb collection using its native ``get(where=...)``.

    SK's ``_inner_get`` ignores filter options, so we use the chroma client
    directly. Chroma stores all non-vector payload in ``metadatas``.
    """
    args: dict[str, Any] = {"include": ["metadatas", "documents"]}
    if field is not None:
        args["where"] = {field: value}
    if limit is not None:
        args["limit"] = limit

    raw = collection._get_collection().get(**args)
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    results: list[dict[str, Any]] = []
    for i, record_id in enumerate(ids):
        record: dict[str, Any] = {"id": record_id}
        if i < len(metadatas) and isinstance(metadatas[i], dict):
            record.update(metadatas[i])
        # Chroma's "documents" field is our record's `content`-equivalent
        # text payload; not all schemas use it but expose it when present.
        if i < len(documents) and documents[i] is not None:
            record.setdefault("document", documents[i])
        results.append(record)
    return results


async def _find_postgres(
    collection: Any,
    field: str | None,
    value: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """SELECT from the SK postgres connector's underlying pool."""
    pool = collection.connection_pool
    if pool is None:
        raise RuntimeError(
            "postgres connection_pool is not initialised; the collection must "
            "be entered as an async context manager before calling collection_filter."
        )

    from psycopg import sql

    schema = collection.db_schema
    table = collection.collection_name
    fields = collection.definition.fields
    non_vector_cols = [
        f.storage_name or f.name
        for f in fields
        # Skip vector fields — they're huge and unused by callers.
        if getattr(f, "field_type", None) != "vector"
        and getattr(getattr(f, "field_type_", None), "value", None) != "vector"
        and "Vector" not in type(f).__name__
    ]
    if not non_vector_cols:
        # Fall back to all columns if heuristic misses; postgres will still
        # honor the rest of the query.
        non_vector_cols = [f.storage_name or f.name for f in fields]

    col_idents = sql.SQL(", ").join(sql.Identifier(c) for c in non_vector_cols)
    base = sql.SQL("SELECT {cols} FROM {schema}.{table}").format(
        cols=col_idents,
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
    )
    params: tuple[Any, ...] = ()
    if field is not None:
        base = sql.SQL("{base} WHERE {field} = %s").format(
            base=base, field=sql.Identifier(field)
        )
        params = (value,)
    if limit is not None:
        base = sql.SQL("{base} LIMIT %s").format(base=base)
        params = (*params, limit)

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(base, params)
        rows = await cur.fetchall()
        col_names = [d.name for d in cur.description] if cur.description else []

    key_name = collection.definition.key_field_storage_name
    results: list[dict[str, Any]] = []
    for row in rows:
        record = dict(zip(col_names, row, strict=False))
        # Normalise the key column to `id` so callers don't need to know
        # the SK key-field naming.
        if key_name in record:
            record["id"] = record[key_name]
        results.append(record)
    return results


async def _find_weaviate(
    collection: Any,
    field: str | None,
    value: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch from weaviate via ``fetch_objects`` with an optional filter."""
    from weaviate.classes.query import Filter as WFilter

    wcoll = collection.async_client.collections.get(collection.collection_name)

    filters = None
    if field is not None:
        filters = WFilter.by_property(field).equal(value)

    # Weaviate's fetch_objects has its own paging via offset/limit; pull in
    # pages of _PAGE_SIZE to bound memory.
    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_limit = _PAGE_SIZE
        if limit is not None:
            remaining = limit - len(results)
            if remaining <= 0:
                break
            page_limit = min(_PAGE_SIZE, remaining)

        resp = await wcoll.query.fetch_objects(
            filters=filters,
            limit=page_limit,
            offset=offset,
            include_vector=False,
        )
        objects = resp.objects or []
        for obj in objects:
            record: dict[str, Any] = {"id": str(obj.uuid)}
            record.update(dict(obj.properties or {}))
            results.append(record)
        if len(objects) < page_limit:
            break
        offset += page_limit
    return results


def _find_in_memory(
    collection: Any,
    field: str | None,
    value: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Iterate the SK in-memory store and filter by attribute equality."""
    import dataclasses

    results: list[dict[str, Any]] = []
    for record in collection.inner_storage.values():
        if field is not None and getattr(record, field, None) != value:
            continue
        if dataclasses.is_dataclass(record):
            payload = {
                f.name: getattr(record, f.name)
                for f in dataclasses.fields(record)
                if f.name != "embedding"
            }
        else:
            # Best-effort attribute walk for non-dataclass records.
            payload = {
                name: getattr(record, name)
                for name in dir(record)
                if not name.startswith("_")
                and not callable(getattr(record, name))
                and name != "embedding"
            }
        results.append(payload)
        if limit is not None and len(results) >= limit:
            break
    return results
