"""Regression tests for the phase 2 review findings on the Postgres adapter.

Needs the pgvector container on 127.0.0.1:54329 (see ``docs/RELIABILITY.md``).
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    LexicalQuery,
    Record,
    RecordFilter,
    SchemaError,
    StoreConnectionError,
    VectorQuery,
)
from holodeck.lib.retrieval.adapters import postgres
from holodeck.lib.retrieval.adapters.postgres import PostgresAdapter
from tests.contract.retrieval.targets import POSTGRES_URL, port_open

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 54329)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))


def _name() -> str:
    return f"review_{uuid4().hex[:12]}"


def _adapter(schema: CollectionSchema) -> PostgresAdapter:
    return PostgresAdapter.create(schema, {"connection_string": POSTGRES_URL})


@pytest_asyncio.fixture
async def source_schema() -> AsyncIterator[CollectionSchema]:
    schema = CollectionSchema(
        _name(),
        3,
        (
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
        ),
    )
    yield schema
    adapter = _adapter(schema)
    await adapter.delete_collection()
    await adapter.close()


# -- 5: computed columns cannot collide with field names -----------------------------


async def test_fields_named_distance_and_rank_do_not_break_searches() -> None:
    schema = CollectionSchema(
        _name(),
        3,
        (
            FieldSpec("distance", FieldType.FLOAT),
            FieldSpec("rank", FieldType.INTEGER),
            FieldSpec("content", FieldType.STRING, full_text=True),
        ),
    )
    adapter = _adapter(schema)
    try:
        await adapter.upsert(
            [
                Record(
                    "a",
                    {"distance": 9.0, "rank": 7, "content": "red apple"},
                    [1, 0, 0],
                ),
                Record(
                    "b",
                    {"distance": 1.0, "rank": 8, "content": "green pear"},
                    [0, 1, 0],
                ),
            ]
        )

        nearest, *_ = await adapter.vector_search(VectorQuery([1, 0, 0], top_k=2))
        assert nearest.record.id == "a"
        assert nearest.record.fields["distance"] == 9.0
        assert nearest.score == pytest.approx(1.0, abs=1e-6)

        (lexical,) = await adapter.lexical_search(
            LexicalQuery("apple", top_k=2, text_field="content")
        )
        assert lexical.record.id == "a"
        assert lexical.record.fields["rank"] == 7
    finally:
        await adapter.delete_collection()
        await adapter.close()


@pytest.mark.parametrize(
    "name", ["__holodeck_distance", "__holodeck_rank", "__holodeck_x"]
)
async def test_field_names_with_the_computed_column_prefix_are_rejected(
    name: str,
) -> None:
    schema = CollectionSchema("docs", 3, (FieldSpec(name),))

    with pytest.raises(SchemaError, match="__holodeck_"):
        PostgresAdapter.create(schema, {})


# -- 6: concurrent replacements of one source serialize ------------------------------


async def test_concurrent_replacements_of_one_source_leave_exactly_one_set(
    source_schema: CollectionSchema,
) -> None:
    adapter = _adapter(source_schema)
    try:
        await adapter.ensure_collection()
        where = RecordFilter.where(source_path="doc.md")

        def records(prefix: str) -> list[Record]:
            return [
                Record(
                    f"{prefix}{n}",
                    {"source_path": "doc.md", "content": prefix},
                    [1, 0, 0],
                )
                for n in range(150)
            ]

        for _ in range(4):
            await adapter.delete_where(where)
            await asyncio.gather(
                adapter.replace_source(where, records("a")),
                adapter.replace_source(where, records("b")),
            )
            stored = {record.id[0] for record in await adapter.find(where)}
            assert len(stored) == 1, f"both replacements survived: {stored}"
            assert await adapter.count() == 150
    finally:
        await adapter.close()


# -- 7: a cancelled first use leaves no pool behind ----------------------------------


def _pool_workers() -> list[str]:
    tasks = [t.get_name() for t in asyncio.all_tasks() if "pool" in t.get_name()]
    threads = [t.name for t in threading.enumerate() if "pool" in t.name.lower()]
    return [*tasks, *threads]


async def test_a_first_use_cancelled_while_the_pool_opens_closes_the_pool(
    source_schema: CollectionSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(source_schema)
    original_open = postgres.AsyncConnectionPool.open

    async def slow_open(self: object, *args: object, **kwargs: object) -> None:
        await original_open(self, *args, **kwargs)  # type: ignore[arg-type]
        await asyncio.sleep(30)

    monkeypatch.setattr(postgres.AsyncConnectionPool, "open", slow_open)
    try:
        first = asyncio.create_task(adapter.count())
        await asyncio.sleep(1.0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.sleep(0.2)

        assert adapter._pool is None
        assert _pool_workers() == []

        monkeypatch.setattr(postgres.AsyncConnectionPool, "open", original_open)
        assert await adapter.count() == 0  # the adapter is usable again
    finally:
        await adapter.close()


# -- 10: secrets stay out of errors ---------------------------------------------------


async def test_a_rejected_password_is_not_in_the_error(
    source_schema: CollectionSchema,
) -> None:
    wrong = POSTGRES_URL.replace(":holodeck@", ":wr0ng-p4ss@")
    adapter = PostgresAdapter.create(source_schema, {"connection_string": wrong})
    try:
        with pytest.raises(StoreConnectionError) as raised:
            await adapter.count()
    finally:
        await adapter.close()

    text = "".join(traceback.format_exception(raised.value))
    assert "wr0ng-p4ss" not in text
    assert wrong not in text
