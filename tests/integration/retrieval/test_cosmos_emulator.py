"""Cosmos DB adapter behavior that only the real service parser can show.

Runs on the Linux emulator (``tests/contract/retrieval/cosmos_emulator.py``);
skipped when it is not running. The contract suite already covers the
operations; these tests cover what the fake cannot judge: that the emulator's
SQL parser accepts the adapter's statements with hostile values, that path-like
IDs reach the service in a form it accepts, that results spanning several
service pages are read in full, and which full-text claims the emulator can
honor. Nothing here ran against the Cosmos DB service.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from holodeck.lib.retrieval import (
    CollectionSchema,
    HybridQuery,
    IncompatibleCollectionError,
    RecordFilter,
    StoreConnectionError,
    VectorQuery,
)
from holodeck.lib.retrieval.adapters.cosmos_nosql import CosmosNoSqlAdapter
from tests.contract.retrieval import cosmos_emulator
from tests.contract.retrieval.conftest import make_record, make_schema

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = cosmos_emulator.skip_reason()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))

HOSTILE = "x') OR 1=1 --; DROP c \"\\ '"


def _build(schema: CollectionSchema, **overrides: object) -> CosmosNoSqlAdapter:
    return CosmosNoSqlAdapter.create(
        schema, {**cosmos_emulator.connection(), **overrides}
    )


@pytest_asyncio.fixture
async def adapter() -> AsyncIterator[CosmosNoSqlAdapter]:
    instance = _build(make_schema())
    await instance.ensure_collection()
    try:
        yield instance
    finally:
        await instance.delete_collection()
        await instance.close()


async def test_hostile_values_are_accepted_as_parameters(
    adapter: CosmosNoSqlAdapter,
) -> None:
    await adapter.upsert(
        [
            make_record("a", [1.0, 0.0, 0.0, 0.0], source=HOSTILE, content=HOSTILE),
            make_record("b", [0.0, 1.0, 0.0, 0.0], source="safe.md"),
        ]
    )

    found = await adapter.find(RecordFilter.where(source_path=HOSTILE))
    assert [r.id for r in found] == ["a"]
    assert found[0].fields["content"] == HOSTILE
    hits = await adapter.vector_search(
        VectorQuery([1.0, 0.0, 0.0, 0.0], 5, RecordFilter.where(source_path=HOSTILE))
    )
    assert [h.record.id for h in hits] == ["a"]
    assert await adapter.find(RecordFilter.where(source_path="b' OR '1'='1")) == []
    assert await adapter.delete_where(RecordFilter.where(source_path=HOSTILE)) == 1
    assert await adapter.count() == 1


async def test_ids_with_reserved_characters_round_trip_on_the_service(
    adapter: CosmosNoSqlAdapter,
) -> None:
    ids = ["docs/guide.md_chunk_0", "a#b?c\\d", "~hd~x", " padded ", "ünï/路径"]
    await adapter.upsert(
        [
            make_record(record_id, [1.0, float(i), 0.0, 0.0], source="p.md", index=i)
            for i, record_id in enumerate(ids)
        ]
    )

    got = await adapter.get(list(reversed(ids)))
    assert [r.id for r in got] == list(reversed(ids))
    listed = await adapter.list_page(limit=10)
    assert sorted(r.id for r in listed.records) == sorted(ids)

    await adapter.replace_source(
        RecordFilter.where(source_path="p.md"),
        [make_record(ids[0], [0.0, 0.0, 1.0, 0.0], source="p.md")],
    )
    assert [
        r.id for r in await adapter.find(RecordFilter.where(source_path="p.md"))
    ] == [ids[0]]
    await adapter.delete([ids[0]])
    assert await adapter.count() == 0


async def test_results_larger_than_one_service_page_are_read_in_full(
    adapter: CosmosNoSqlAdapter,
) -> None:
    await adapter.upsert(
        [make_record(f"r{i:03d}", [1.0, float(i % 7), 0.0, 0.0]) for i in range(230)]
    )

    assert await adapter.count() == 230
    assert len(await adapter.find(RecordFilter())) == 230
    cursor: str | None = None
    seen = 0
    while True:
        page = await adapter.list_page(cursor=cursor, limit=100)
        seen += len(page.records)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == 230


async def test_a_filter_on_a_field_excluded_from_the_index_still_runs(
    adapter: CosmosNoSqlAdapter,
) -> None:
    await adapter.upsert(
        [
            make_record("a", [1.0, 0.0, 0.0, 0.0], mtime=1.5),
            make_record("b", [0.0, 1.0, 0.0, 0.0], mtime=2.5, pinned=True),
        ]
    )
    # pinned is neither indexed nor full-text, so the container excludes its path.
    assert [r.id for r in await adapter.find(RecordFilter.where(pinned=True))] == ["b"]


async def test_the_container_is_created_with_the_adapters_policies(
    adapter: CosmosNoSqlAdapter,
) -> None:
    client = await adapter._open()
    container = client.get_database_client(
        cosmos_emulator.DATABASE
    ).get_container_client(adapter.schema.name)
    properties = await container.read()

    assert properties["partitionKey"] == {
        "kind": "Hash",
        "paths": ["/id"],
        "version": 2,
    }
    indexing = properties["indexingPolicy"]
    assert isinstance(indexing, dict)
    assert indexing["vectorIndexes"] == adapter.indexing_policy()["vectorIndexes"]
    excluded = {item["path"] for item in indexing["excludedPaths"]}
    assert {'/"pinned"/*', '/"embedding"/*'} <= excluded


async def test_the_client_can_be_reused_after_close(
    adapter: CosmosNoSqlAdapter,
) -> None:
    await adapter.upsert([make_record("a", [1.0, 0.0, 0.0, 0.0])])
    await adapter.close()
    await adapter.close()

    assert [r.id for r in await adapter.get(["a"])] == ["a"]


async def test_a_missing_database_is_reported_when_creation_is_off() -> None:
    missing = _build(
        make_schema(), database_name=f"absent_{uuid4().hex[:8]}", create_database=False
    )
    try:
        with pytest.raises(StoreConnectionError, match="create_database"):
            await missing.ensure_collection()
    finally:
        await missing.close()


async def test_full_text_search_is_refused_where_the_service_drops_the_index() -> None:
    # The emulator accepts a full-text policy and index, then does not keep
    # them, and does not know FullTextScore. The adapter must say so instead of
    # sending a hybrid query that fails inside the service.
    unsupported = _build(make_schema(), full_text_search=True)
    try:
        with pytest.raises(IncompatibleCollectionError, match="full-text index"):
            await unsupported.ensure_collection()
        with pytest.raises(IncompatibleCollectionError, match="full-text index"):
            await unsupported.hybrid_search(
                HybridQuery("alpha", [1.0, 0.0, 0.0, 0.0], 3, "content")
            )
    finally:
        await unsupported.delete_collection()
        await unsupported.close()
