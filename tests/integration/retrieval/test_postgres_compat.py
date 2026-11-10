"""The native Postgres adapter and Semantic Kernel's collection share one table.

Collections written by HoloDeck before the native adapter (through the SK
bridge) must stay readable, and the reverse. The bridge's Postgres path needs
two workarounds, both bugs in the SK connector it wraps:

* ``PostgresSettings`` replaces the port of a connection string with 5432, so
  the bridge here gets the port from ``PGPORT``;
* SK's cosine search returns the raw distance as the score, so the bridge
  below treats scores as distances (``SCORE_IS_DISTANCE``).

Needs the pgvector container on 127.0.0.1:54329 (see ``docs/RELIABILITY.md``).
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldType,
    LexicalQuery,
    Record,
    RecordFilter,
    VectorQuery,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.adapters.postgres import PostgresAdapter
from holodeck.lib.retrieval.adapters.sk_bridge import SKPostgresAdapter
from holodeck.lib.retrieval.schemas import (
    document_schema,
    hierarchical_schema,
    structured_schema,
)
from tests.contract.retrieval.conftest import CORPUS, make_record, make_schema
from tests.contract.retrieval.targets import POSTGRES_URL, port_open

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 54329)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))


class FixedScoreBridge(SKPostgresAdapter):
    """The SK bridge with the score of SK's cosine search read as a distance."""

    SCORE_IS_DISTANCE = True


@pytest.fixture(autouse=True)
def bridge_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPORT", "54329")


@pytest.fixture
def schema() -> CollectionSchema:
    return make_schema()


@pytest_asyncio.fixture
async def native(schema: CollectionSchema) -> AsyncIterator[PostgresAdapter]:
    adapter = PostgresAdapter.create(schema, {"connection_string": POSTGRES_URL})
    yield adapter
    await adapter.delete_collection()
    await adapter.close()


@pytest_asyncio.fixture
async def bridge(schema: CollectionSchema) -> AsyncIterator[FixedScoreBridge]:
    adapter = FixedScoreBridge.create(schema, {"connection_string": POSTGRES_URL})
    yield adapter
    await adapter.close()


async def _drain(adapter: VectorStoreAdapter) -> list[Record]:
    records: list[Record] = []
    cursor: str | None = None
    while True:
        page = await adapter.list_page(cursor=cursor, limit=2)
        records.extend(page.records)
        cursor = page.next_cursor
        if cursor is None:
            return records


def _view(record: Record) -> tuple[str, dict[str, Any]]:
    return record.id, dict(record.fields)


async def _snapshot(adapter: VectorStoreAdapter) -> dict[str, Any]:
    """Everything the tools read, in a form that compares with ``==``."""
    ids = ["b_chunk_0", "missing", "a_chunk_0", "c_chunk_0"]
    with_vectors = await adapter.get(ids, include_vectors=True)
    hits = await adapter.vector_search(VectorQuery([1.0, 0.1, 0.0, 0.0], top_k=4))
    filtered = await adapter.vector_search(
        VectorQuery(
            [1.0, 0.0, 0.0, 0.0], 5, filter=RecordFilter.where(source_path="b.md")
        )
    )
    return {
        "count": await adapter.count(),
        "get": [_view(r) for r in await adapter.get(ids)],
        "vectors": {r.id: list(r.vector or ()) for r in with_vectors},
        "find": sorted(
            _view(r) for r in await adapter.find(RecordFilter.where(source_path="b.md"))
        ),
        "find_two": [
            _view(r)
            for r in await adapter.find(
                RecordFilter.where(source_path="b.md", chunk_index=1)
            )
        ],
        "pages": [_view(r) for r in await _drain(adapter)],
        "search": [(h.record.id, dict(h.record.fields), h.score) for h in hits],
        "filtered": [(h.record.id, h.score) for h in filtered],
    }


def _assert_same(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if key == "vectors":
            assert left[key].keys() == right[key].keys()
            for name in left[key]:
                assert left[key][name] == pytest.approx(right[key][name], abs=1e-6)
        elif key in {"search", "filtered"}:
            assert len(left[key]) == len(right[key])
            for mine, theirs in zip(left[key], right[key], strict=True):
                assert mine[:-1] == theirs[:-1], key
                assert mine[-1] == pytest.approx(theirs[-1], abs=1e-6), key
        else:
            assert left[key] == right[key], key


async def test_native_reads_a_table_the_bridge_created(
    bridge: FixedScoreBridge, native: PostgresAdapter
) -> None:
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS)

    assert await native.collection_exists() is True
    await native.ensure_collection()  # an existing SK table is accepted as it is
    _assert_same(await _snapshot(native), await _snapshot(bridge))
    assert (await _snapshot(native))["count"] == len(CORPUS)


async def test_bridge_reads_a_table_the_native_adapter_created(
    bridge: FixedScoreBridge, native: PostgresAdapter
) -> None:
    await native.ensure_collection()
    await native.upsert(CORPUS)

    assert await bridge.collection_exists() is True
    _assert_same(await _snapshot(bridge), await _snapshot(native))


async def test_native_writes_to_a_bridge_table_and_the_bridge_reads_them(
    bridge: FixedScoreBridge, native: PostgresAdapter
) -> None:
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS[:3])

    await native.upsert(
        [
            make_record("a_chunk_0", [0.0, 0.0, 0.0, 1.0], content="rewritten"),
            make_record("d_chunk_0", [0.5, 0.5, 0.0, 0.0], source="d.md", index=3),
        ]
    )
    await native.replace_source(
        RecordFilter.where(source_path="b.md"),
        [make_record("b_chunk_9", [0.0, 1.0, 1.0, 0.0], source="b.md", index=9)],
    )
    await native.delete(["a_chunk_1"])

    expected = {"a_chunk_0", "b_chunk_9", "d_chunk_0"}
    assert {r.id for r in await _drain(bridge)} == expected
    assert await bridge.count() == 3
    (rewritten,) = await bridge.get(["a_chunk_0"], include_vectors=True)
    assert rewritten.fields["content"] == "rewritten"
    assert list(rewritten.vector or ()) == pytest.approx([0, 0, 0, 1], abs=1e-6)
    (added,) = await bridge.get(["d_chunk_0"])
    assert added.fields["chunk_index"] == 3
    assert added.fields["pinned"] is False
    _assert_same(await _snapshot(bridge), await _snapshot(native))


async def test_bridge_writes_to_a_native_table_and_the_native_adapter_reads_them(
    bridge: FixedScoreBridge, native: PostgresAdapter
) -> None:
    await native.ensure_collection()
    await native.upsert(CORPUS[:2])
    await bridge.upsert(CORPUS[2:])

    assert sorted(r.id for r in await _drain(native)) == sorted(r.id for r in CORPUS)
    _assert_same(await _snapshot(native), await _snapshot(bridge))


# -- stored layout -------------------------------------------------------------------


def _layout(table: str) -> dict[str, list[Any]]:
    """Columns and indexes of ``public.<table>``, with the table name masked."""
    with psycopg.connect(POSTGRES_URL) as connection:
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
    return {
        "columns": columns,
        "types": types,
        "indexes": [(mask.sub("T", name), mask.sub("T", ddl)) for name, ddl in indexes],
        "constraints": [(kind, mask.sub("T", ddl)) for kind, ddl in constraints],
    }


def _unique() -> str:
    return f"compat_{uuid4().hex[:12]}"


SCHEMAS: dict[str, Callable[[], CollectionSchema]] = {
    "contract": make_schema,
    "document": lambda: document_schema(8, {"collection_name": _unique()}),
    "structured": lambda: structured_schema(8, ["region", "year"], _unique()),
    "hierarchical": lambda: hierarchical_schema(8, _unique()),
}


@pytest.mark.parametrize("kind", SCHEMAS)
async def test_native_and_bridge_create_the_same_table(kind: str) -> None:
    bridge_schema = SCHEMAS[kind]()
    native_schema = SCHEMAS[kind]()
    bridge = FixedScoreBridge.create(bridge_schema, {"connection_string": POSTGRES_URL})
    native = PostgresAdapter.create(native_schema, {"connection_string": POSTGRES_URL})
    try:
        await bridge.ensure_collection()
        await native.ensure_collection()
        mine = _layout(native_schema.name)
        theirs = _layout(bridge_schema.name)
    finally:
        await bridge.delete_collection()
        await bridge.close()
        await native.delete_collection()
        await native.close()

    assert mine == theirs
    expected = {
        "id": "text",
        "embedding": "vector(8)" if kind != "contract" else "vector(4)",
    }
    for spec in native_schema.fields:
        expected[spec.name] = {
            FieldType.STRING: "text",
            FieldType.INTEGER: "integer",
            FieldType.FLOAT: "double precision",
            FieldType.BOOLEAN: "boolean",
        }[spec.type]
    assert dict(mine["types"]) == expected
    assert [name for name, _ in mine["indexes"]] == ["T_embedding_idx", "T_pkey"]
    assert dict(mine["indexes"])["T_embedding_idx"].endswith(
        "USING hnsw (embedding vector_cosine_ops)"
    )


async def test_ensure_collection_on_an_existing_table_changes_nothing(
    native: PostgresAdapter, schema: CollectionSchema
) -> None:
    await native.ensure_collection()
    before = _layout(schema.name)
    await native.ensure_collection()
    assert _layout(schema.name) == before


# -- lexical search (native only) ----------------------------------------------------


async def test_lexical_search_on_a_bridge_table(
    bridge: FixedScoreBridge, native: PostgresAdapter
) -> None:
    await bridge.ensure_collection()
    await bridge.upsert(CORPUS)

    hits = await native.lexical_search(
        LexicalQuery("zeppelin or cherry", top_k=5, text_field="content")
    )
    assert sorted(h.record.id for h in hits) == ["b_chunk_0", "b_chunk_1"]
    assert all(math.isfinite(h.score) and h.score > 0 for h in hits)
