"""Frozen store layouts written by the Semantic Kernel path (042 T16).

Before T16 removed Semantic Kernel, ``scripts/freeze_legacy_layouts.py``
created collections through the SK bridge on Qdrant, Chroma and Postgres and
saved the raw store state under ``tests/fixtures/legacy_layout/<store>/``.
This module reads those files, rebuilds a legacy collection from them with
only the store's own client, and dumps a live collection in the same form, so
the compatibility tests compare the native adapters against what SK stored.
See ``tests/fixtures/legacy_layout/README.md``.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import chromadb
import psycopg
from psycopg import sql
from qdrant_client import AsyncQdrantClient, models

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldType,
    Record,
    RecordFilter,
    VectorQuery,
    VectorStoreAdapter,
)

_T = TypeVar("_T")

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_layout"
TABLE = "<table>"
"""Placeholder for the table name in frozen Postgres DDL."""


# -- fixture files ---------------------------------------------------------------


def fixture_path(store: str, kind: str) -> Path:
    return FIXTURES / store / f"{kind}.json"


def load(store: str, kind: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(fixture_path(store, kind).read_text())
    return data


def save(store: str, kind: str, data: Mapping[str, Any]) -> None:
    path = fixture_path(store, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def plain(value: _T) -> _T:
    """``value`` as it reads back from a fixture file (tuples become lists)."""
    return cast(_T, json.loads(json.dumps(value)))


def schema_facts(schema: CollectionSchema) -> dict[str, Any]:
    """The schema without its name, to detect drift from the frozen one."""
    return {
        "dimensions": schema.dimensions,
        "fields": [
            [spec.name, spec.type.value, spec.indexed, spec.full_text]
            for spec in schema.fields
        ],
    }


def assert_close(actual: Any, expected: Any, *, tol: float = 1e-5) -> None:
    """Deep equality with floats compared to ``tol`` and bools kept distinct."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected and type(actual) is type(expected), (
            actual,
            expected,
        )
    elif isinstance(expected, float):
        assert isinstance(actual, float), (actual, expected)
        assert abs(actual - expected) <= tol, (actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), (actual, expected)
        assert actual.keys() == expected.keys(), (actual, expected)
        for key in expected:
            assert_close(actual[key], expected[key], tol=tol)
    elif isinstance(expected, list):
        assert isinstance(actual, list), (actual, expected)
        assert len(actual) == len(expected), (actual, expected)
        for left, right in zip(actual, expected, strict=True):
            assert_close(left, right, tol=tol)
    else:
        assert actual == expected and type(actual) is type(expected), (
            actual,
            expected,
        )


# -- synthetic records for the tool schemas ------------------------------------------


def sample_records(schema: CollectionSchema, count: int = 4) -> list[Record]:
    """Deterministic records that fill every field of ``schema``."""
    records = []
    for n in range(count):
        fields: dict[str, str | int | float | bool | None] = {}
        for spec in schema.fields:
            if spec.name in {"source_path", "source_file"}:
                fields[spec.name] = f"doc{n % 2}.md"
            elif spec.type is FieldType.STRING:
                fields[spec.name] = f"{spec.name} {n} kestrel" if n == 1 else f"v{n}"
            elif spec.type is FieldType.INTEGER:
                fields[spec.name] = n
            elif spec.type is FieldType.FLOAT:
                fields[spec.name] = n + 0.5
            else:
                fields[spec.name] = n % 2 == 0
        vector = [0.0] * schema.dimensions
        vector[n % schema.dimensions] = 1.0
        vector[(n + 1) % schema.dimensions] = 0.25 * n
        records.append(Record(id=f"rec_{n}", fields=fields, vector=vector))
    return records


def source_field(schema: CollectionSchema) -> str:
    names = {spec.name for spec in schema.fields}
    return "source_path" if "source_path" in names else "source_file"


def _row(record: Record) -> list[Any]:
    return [record.id, dict(record.fields)]


async def walk(adapter: VectorStoreAdapter, limit: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    cursor: str | None = None
    for _ in range(50):
        page = await adapter.list_page(cursor=cursor, limit=limit)
        rows.extend(_row(record) for record in page.records)
        cursor = page.next_cursor
        if cursor is None:
            return sorted(rows, key=lambda row: str(row[0]))
    raise AssertionError("list_page did not finish")


async def basic_snapshot(
    adapter: VectorStoreAdapter, schema: CollectionSchema
) -> dict[str, Any]:
    """The reads the tools use, for a collection holding ``sample_records``."""
    ids = [record.id for record in sample_records(schema)] + ["missing"]
    got = await adapter.get(ids, include_vectors=True)
    where = RecordFilter.where(**{source_field(schema): "doc1.md"})
    query = [1.0] + [0.1] * (schema.dimensions - 1)
    hits = await adapter.vector_search(VectorQuery(query, 3))
    return plain(
        {
            "get": [_row(r) for r in got],
            "vectors": [list(r.vector or ()) for r in got],
            "find": sorted((_row(r) for r in await adapter.find(where)), key=str),
            "list": await walk(adapter, 3),
            "count": await adapter.count(),
            "vector_search": [[h.record.id, h.score] for h in hits],
        }
    )


# -- Qdrant ------------------------------------------------------------------------

_QDRANT_INDEX_PARAMS: dict[str, type[Any]] = {
    "text": models.TextIndexParams,
    "keyword": models.KeywordIndexParams,
    "integer": models.IntegerIndexParams,
    "float": models.FloatIndexParams,
    "bool": models.BoolIndexParams,
}


async def qdrant_layout(raw: AsyncQdrantClient, name: str) -> dict[str, Any]:
    """Collection config and payload indexes, as the old compat tests compared."""
    info = await raw.get_collection(name)
    config = info.config
    return plain(
        {
            "params": config.params.model_dump(mode="json"),
            "hnsw_config": config.hnsw_config.model_dump(mode="json"),
            "optimizer_config": config.optimizer_config.model_dump(mode="json"),
            "quantization_config": (
                None
                if config.quantization_config is None
                else config.quantization_config.model_dump(mode="json")
            ),
            "payload_schema": {
                field: {
                    "data_type": str(index.data_type.value),
                    "params": (
                        None
                        if index.params is None
                        else index.params.model_dump(mode="json")
                    ),
                }
                for field, index in sorted(info.payload_schema.items())
            },
        }
    )


async def qdrant_rows(raw: AsyncQdrantClient, name: str) -> list[dict[str, Any]]:
    points, _ = await raw.scroll(name, limit=1000, with_payload=True, with_vectors=True)
    rows = []
    for point in points:
        assert isinstance(point.vector, dict)
        rows.append(
            {
                "id": str(point.id),
                "payload": dict(point.payload or {}),
                "vector": list(point.vector["embedding"]),  # type: ignore[arg-type]
            }
        )
    return plain(sorted(rows, key=lambda row: row["id"]))


def qdrant_point_id(collection: str, holodeck_id: str) -> str:
    """The point ID SK's path gave a HoloDeck ID in ``collection``.

    A UUID is stored as its canonical form; any other ID as
    ``uuid5(uuid5(NAMESPACE_URL, "holodeck/retrieval/<collection>"), id)``.
    """
    try:
        return str(uuid.UUID(holodeck_id))
    except ValueError:
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"holodeck/retrieval/{collection}")
        return str(uuid.uuid5(namespace, holodeck_id))


def qdrant_rows_for(fixture: Mapping[str, Any], collection: str) -> list[Any]:
    """The frozen points with their IDs derived for ``collection``.

    Point IDs depend on the collection name, so a rebuilt collection under a
    new name gets the IDs SK's path would have given there.
    """
    rows = []
    for row in fixture["rows"]:
        holodeck_id = row["payload"].get("holodeck_id")
        point_id = row["id"]
        if holodeck_id is not None:
            point_id = qdrant_point_id(collection, holodeck_id)
        rows.append({**row, "id": point_id})
    return sorted(rows, key=lambda row: row["id"])


async def qdrant_rebuild(
    raw: AsyncQdrantClient, name: str, fixture: Mapping[str, Any]
) -> None:
    """Recreate a legacy collection from its frozen layout and points."""
    layout = fixture["layout"]
    vectors = layout["params"]["vectors"]["embedding"]
    await raw.create_collection(
        name,
        vectors_config={
            "embedding": models.VectorParams(
                size=vectors["size"],
                distance=models.Distance(vectors["distance"]),
                datatype=models.Datatype(vectors["datatype"]),
            )
        },
    )
    for field, index in layout["payload_schema"].items():
        params_type = _QDRANT_INDEX_PARAMS[index["data_type"]]
        params = {k: v for k, v in index["params"].items() if v is not None}
        await raw.create_payload_index(
            name, field, field_schema=params_type(**params), wait=True
        )
    await raw.upsert(
        name,
        points=[
            models.PointStruct(
                id=row["id"],
                payload=row["payload"],
                vector={"embedding": row["vector"]},
            )
            for row in qdrant_rows_for(fixture, name)
        ],
        wait=True,
    )


# -- Chroma ------------------------------------------------------------------------


def chroma_layout(collection: chromadb.Collection) -> dict[str, Any]:
    configuration = dict(collection.configuration)
    return plain(
        {
            "hnsw": configuration.get("hnsw"),
            "spann": configuration.get("spann"),
            "embedding_function": configuration.get("embedding_function"),
            "metadata": collection.metadata,
        }
    )


def chroma_rows(collection: chromadb.Collection) -> list[dict[str, Any]]:
    stored = collection.get(include=["metadatas", "documents", "embeddings"])
    embeddings = stored["embeddings"]
    assert embeddings is not None
    metadatas = stored["metadatas"] or [None] * len(stored["ids"])
    documents = stored["documents"] or [None] * len(stored["ids"])
    rows = [
        {
            "id": record_id,
            "metadata": None if metadata is None else dict(metadata),
            "document": document,
            "embedding": [float(x) for x in embedding],
        }
        for record_id, metadata, document, embedding in zip(
            stored["ids"], metadatas, documents, embeddings, strict=True
        )
    ]
    return plain(sorted(rows, key=lambda row: row["id"]))


def chroma_rebuild(
    client: Any, name: str, fixture: Mapping[str, Any]
) -> chromadb.Collection:
    """Recreate a legacy collection with ``client`` (HTTP or persistent)."""
    layout = fixture["layout"]
    collection = client.create_collection(
        name,
        configuration={"hnsw": {"space": layout["hnsw"]["space"]}},
        metadata=layout["metadata"],
        embedding_function=None,
    )
    rows = fixture["rows"]
    collection.add(
        ids=[row["id"] for row in rows],
        embeddings=[row["embedding"] for row in rows],
        metadatas=[row["metadata"] for row in rows],
        documents=[row["document"] for row in rows],
    )
    return collection


# -- Postgres ----------------------------------------------------------------------


def postgres_layout(url: str, table: str) -> dict[str, Any]:
    """Columns, types, indexes and constraints, with the table name masked."""
    with psycopg.connect(url) as connection:
        columns = connection.execute(
            "SELECT column_name, data_type, udt_name, is_nullable, column_default, "
            "character_maximum_length, numeric_precision, ordinal_position "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,),
        ).fetchall()
        types = connection.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a WHERE a.attrelid = %s::regclass "
            "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
            (f"public.{table}",),
        ).fetchall()
        indexes = connection.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = %s ORDER BY indexname",
            (table,),
        ).fetchall()
        constraints = connection.execute(
            "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = %s::regclass ORDER BY contype",
            (f"public.{table}",),
        ).fetchall()
    mask = re.compile(re.escape(table))
    return plain(
        {
            "columns": columns,
            "types": types,
            "indexes": [
                (mask.sub(TABLE, name), mask.sub(TABLE, ddl)) for name, ddl in indexes
            ],
            "constraints": [(kind, mask.sub(TABLE, ddl)) for kind, ddl in constraints],
        }
    )


def postgres_rows(url: str, table: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as connection:
        cursor = connection.execute(
            sql.SQL(
                'SELECT *, embedding::text AS "<embedding>" FROM {} ORDER BY id'
            ).format(sql.Identifier("public", table))
        )
        names = [column.name for column in cursor.description or ()]
        rows = []
        for values in cursor.fetchall():
            row = dict(zip(names, values, strict=True))
            text = row.pop("<embedding>")
            row["embedding"] = None if text is None else json.loads(text)
            rows.append(row)
    return plain(rows)


def postgres_rebuild(url: str, table: str, fixture: Mapping[str, Any]) -> None:
    """Recreate a legacy table from its frozen DDL facts and rows."""
    layout = fixture["layout"]
    nullable = {row[0]: row[3] == "YES" for row in layout["columns"]}
    columns = [
        f'"{name}" {sql_type}' + ("" if nullable[name] else " NOT NULL")
        for name, sql_type in layout["types"]
    ]
    columns += [ddl for _kind, ddl in layout["constraints"]]
    quoted = sql.Identifier("public", table)
    with psycopg.connect(url) as connection:
        connection.execute(
            sql.SQL("CREATE TABLE {} ({})").format(quoted, sql.SQL(", ".join(columns)))
        )
        for name, ddl in layout["indexes"]:
            if name.endswith("_pkey"):
                continue  # created by the PRIMARY KEY constraint
            connection.execute(ddl.replace(TABLE, table))
        for row in fixture["rows"]:
            names = list(row)
            placeholders = [
                sql.SQL("%s::vector" if name == "embedding" else "%s") for name in names
            ]
            values: Sequence[Any] = [
                json.dumps(row[name]) if name == "embedding" else row[name]
                for name in names
            ]
            connection.execute(
                sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    quoted,
                    sql.SQL(", ").join(sql.Identifier(name) for name in names),
                    sql.SQL(", ").join(placeholders),
                ),
                values,
            )
