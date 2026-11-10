"""The native Postgres adapter reads and writes the table the SK path created.

Tables written by HoloDeck through Semantic Kernel's Postgres collection must
stay readable and writable through ``PostgresAdapter`` (042 T7). SK is gone
(T16), so the SK layout is frozen under
``tests/fixtures/legacy_layout/postgres/``: columns, types, indexes,
constraints, every row, and what SK's path returned for each read. Each test
rebuilds a legacy table from those files with plain ``psycopg``, or creates a
fresh one natively.

The SK path's scores were read through the bridge with SK's cosine score
treated as a distance (``SCORE_IS_DISTANCE``), the fix the bridge needed for
SK's connector; the frozen scores are therefore similarities.

Needs the pgvector container on 127.0.0.1:54329 (see ``docs/RELIABILITY.md``).
"""

from __future__ import annotations

import math
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
    ScoredRecord,
    VectorQuery,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.adapters.postgres import PostgresAdapter
from holodeck.lib.retrieval.schemas import (
    document_schema,
    hierarchical_schema,
    structured_schema,
)
from tests.contract.retrieval.conftest import CORPUS, make_record, make_schema
from tests.contract.retrieval.targets import POSTGRES_URL, port_open
from tests.integration.retrieval.legacy_layout import (
    assert_close,
    basic_snapshot,
    load,
    plain,
    postgres_layout,
    postgres_rebuild,
    postgres_rows,
    sample_records,
    schema_facts,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 54329)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))

STORE = "postgres"


def _all_types(name: str) -> CollectionSchema:
    base = make_schema()
    return CollectionSchema(name, base.dimensions, base.fields)


SCHEMAS: dict[str, Callable[[str], CollectionSchema]] = {
    "all_types": _all_types,
    "document": lambda name: document_schema(8, {"collection_name": name}),
    "structured": lambda name: structured_schema(8, ["region", "year"], name),
    "hierarchical": lambda name: hierarchical_schema(
        8, "tool", {"collection_name": name}
    ),
}
KINDS = pytest.mark.parametrize("kind", list(SCHEMAS))


def corpus(kind: str, schema: CollectionSchema) -> list[Record]:
    return CORPUS if kind == "all_types" else sample_records(schema)


# -- reads compared with what the SK path returned -----------------------------------


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


def _ordered(hits: list[ScoredRecord]) -> list[ScoredRecord]:
    """Equal scores come back in index order; compare them by ID."""
    scores = [round(hit.score, 5) for hit in hits]
    assert scores == sorted(scores, reverse=True), f"not ranked best first: {scores}"
    return sorted(hits, key=lambda h: (-round(h.score, 5), h.record.id))


async def snapshot(adapter: VectorStoreAdapter) -> dict[str, Any]:
    """Everything the tools read, in a form that compares with ``==``."""
    ids = ["b_chunk_0", "missing", "a_chunk_0", "c_chunk_0"]
    with_vectors = await adapter.get(ids, include_vectors=True)
    # No tie at the top-k boundary, so the hits do not depend on the index.
    hits = _ordered(
        await adapter.vector_search(VectorQuery([1.0, 0.1, 0.0, 0.05], top_k=4))
    )
    filtered = _ordered(
        await adapter.vector_search(
            VectorQuery(
                [1.0, 0.0, 0.0, 0.0], 5, filter=RecordFilter.where(source_path="b.md")
            )
        )
    )
    return plain(
        {
            "count": await adapter.count(),
            "get": [_view(r) for r in await adapter.get(ids)],
            "vectors": {r.id: list(r.vector or ()) for r in with_vectors},
            "find": sorted(
                _view(r)
                for r in await adapter.find(RecordFilter.where(source_path="b.md"))
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
    )


async def read_all(
    kind: str, adapter: VectorStoreAdapter, schema: CollectionSchema
) -> dict[str, Any]:
    if kind == "all_types":
        return await snapshot(adapter)
    return await basic_snapshot(adapter, schema)


async def native_writes(adapter: VectorStoreAdapter) -> None:
    """The writes the tools make, applied to a table holding ``CORPUS``."""
    await adapter.ensure_collection()  # an existing SK table is accepted as it is
    await adapter.upsert(
        [
            make_record("a_chunk_0", [0.0, 0.0, 0.0, 1.0], content="rewritten"),
            make_record("d_chunk_0", [0.5, 0.5, 0.0, 0.0], source="d.md", index=3),
        ]
    )
    await adapter.replace_source(
        RecordFilter.where(source_path="b.md"),
        [make_record("b_chunk_9", [0.0, 1.0, 1.0, 0.0], source="b.md", index=9)],
    )
    await adapter.delete(["a_chunk_1"])


async def snapshot_after_writes(adapter: VectorStoreAdapter) -> dict[str, Any]:
    (rewritten,) = await adapter.get(["a_chunk_0"], include_vectors=True)
    (added,) = await adapter.get(["d_chunk_0"])
    return plain(
        {
            "ids": sorted(r.id for r in await _drain(adapter)),
            "rewritten": (_view(rewritten), list(rewritten.vector or ())),
            "added": _view(added),
            "snapshot": await snapshot(adapter),
        }
    )


# -- fixtures ------------------------------------------------------------------------


@pytest_asyncio.fixture
async def names() -> AsyncIterator[list[str]]:
    """Unique table names; every one is dropped afterwards."""
    created: list[str] = []
    yield created
    with psycopg.connect(POSTGRES_URL) as connection:
        for name in created:
            connection.execute(f'DROP TABLE IF EXISTS public."{name}"')


def unique(names: list[str]) -> str:
    name = f"compat_{uuid4().hex[:12]}"
    names.append(name)
    return name


def legacy(names: list[str], kind: str) -> tuple[CollectionSchema, dict[str, Any]]:
    """A legacy table rebuilt from the frozen SK layout."""
    fixture = load(STORE, kind)
    schema = SCHEMAS[kind](unique(names))
    postgres_rebuild(POSTGRES_URL, schema.name, fixture)
    return schema, fixture


def native(schema: CollectionSchema) -> PostgresAdapter:
    return PostgresAdapter.create(schema, {"connection_string": POSTGRES_URL})


# -- tests ---------------------------------------------------------------------------


@KINDS
def test_the_frozen_schema_is_the_current_schema(kind: str) -> None:
    assert schema_facts(SCHEMAS[kind]("x")) == load(STORE, kind)["schema"]


@KINDS
async def test_the_rebuilt_table_is_the_frozen_one(names: list[str], kind: str) -> None:
    schema, fixture = legacy(names, kind)
    assert postgres_layout(POSTGRES_URL, schema.name) == fixture["layout"]
    assert postgres_rows(POSTGRES_URL, schema.name) == fixture["rows"]


@KINDS
async def test_native_reads_a_legacy_table_as_the_sk_path_did(
    names: list[str], kind: str
) -> None:
    schema, fixture = legacy(names, kind)
    adapter = native(schema)
    try:
        assert await adapter.collection_exists() is True
        await adapter.ensure_collection()  # an existing SK table is accepted
        got = await read_all(kind, adapter, schema)
    finally:
        await adapter.close()
    assert_close(got, fixture["expected"], tol=1e-6)
    assert postgres_layout(POSTGRES_URL, schema.name) == fixture["layout"]
    if kind == "all_types":
        assert got["count"] == len(CORPUS)


@KINDS
async def test_native_creates_the_legacy_table(names: list[str], kind: str) -> None:
    schema = SCHEMAS[kind](unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        await adapter.upsert(corpus(kind, schema))
    finally:
        await adapter.close()
    fixture = load(STORE, kind)
    mine = postgres_layout(POSTGRES_URL, schema.name)
    assert mine == fixture["layout"]
    assert postgres_rows(POSTGRES_URL, schema.name) == fixture["rows"]

    expected = {"id": "text", "embedding": f"vector({schema.dimensions})"}
    for spec in schema.fields:
        expected[spec.name] = {
            FieldType.STRING: "text",
            FieldType.INTEGER: "integer",
            FieldType.FLOAT: "double precision",
            FieldType.BOOLEAN: "boolean",
        }[spec.type]
    assert dict(mine["types"]) == expected
    index_names = [name for name, _ in mine["indexes"]]
    assert index_names == ["<table>_embedding_idx", "<table>_pkey"]
    assert dict(mine["indexes"])["<table>_embedding_idx"].endswith(
        "USING hnsw (embedding vector_cosine_ops)"
    )


async def test_native_writes_to_a_legacy_table_stay_consistent(
    names: list[str],
) -> None:
    """The same writes on an SK table read back as SK's path read them."""
    schema, fixture = legacy(names, "all_types")
    adapter = native(schema)
    try:
        await native_writes(adapter)
        got = await snapshot_after_writes(adapter)
    finally:
        await adapter.close()
    assert got["ids"] == ["a_chunk_0", "b_chunk_9", "c_chunk_0", "d_chunk_0"]
    assert got["rewritten"][0][1]["content"] == "rewritten"
    assert got["rewritten"][1] == pytest.approx([0, 0, 0, 1], abs=1e-6)
    assert got["added"][1]["chunk_index"] == 3
    assert got["added"][1]["pinned"] is False
    assert_close(got, fixture["after_writes"], tol=1e-6)


async def test_ensure_collection_on_an_existing_table_changes_nothing(
    names: list[str],
) -> None:
    schema = _all_types(unique(names))
    adapter = native(schema)
    try:
        await adapter.ensure_collection()
        before = postgres_layout(POSTGRES_URL, schema.name)
        await adapter.ensure_collection()
        assert postgres_layout(POSTGRES_URL, schema.name) == before
    finally:
        await adapter.close()


async def test_lexical_search_on_a_legacy_table(names: list[str]) -> None:
    schema, _ = legacy(names, "all_types")
    adapter = native(schema)
    try:
        hits = await adapter.lexical_search(
            LexicalQuery("zeppelin or cherry", top_k=5, text_field="content")
        )
    finally:
        await adapter.close()
    assert sorted(h.record.id for h in hits) == ["b_chunk_0", "b_chunk_1"]
    assert all(math.isfinite(h.score) and h.score > 0 for h in hits)
