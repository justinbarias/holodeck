"""The same behavior tests for every vector store adapter.

An adapter is complete when this suite passes for it. A provider that
cannot do an operation must raise ``UnsupportedOperationError``; it must
never return an empty result in its place.
"""

from __future__ import annotations

import math

import pytest

from holodeck.lib.retrieval import (
    HybridQuery,
    LexicalQuery,
    Operation,
    Record,
    RecordFilter,
    SchemaError,
    ScoreKind,
    UnsupportedOperationError,
    VectorQuery,
    VectorStoreAdapter,
)
from tests.contract.retrieval.conftest import CORPUS, DIMENSIONS, make_record

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def needs(adapter: VectorStoreAdapter, *operations: Operation) -> None:
    """Skip a behavior test when the adapter declares it unsupported."""
    for operation in operations:
        if not adapter.capabilities.supports(operation):
            pytest.skip(f"{adapter.provider} does not support {operation.value}")


async def test_adapter_satisfies_the_protocol(adapter: VectorStoreAdapter) -> None:
    assert isinstance(adapter, VectorStoreAdapter)
    assert adapter.capabilities.provider == adapter.provider


async def test_collection_lifecycle(adapter: VectorStoreAdapter) -> None:
    assert await adapter.collection_exists() is False
    await adapter.ensure_collection()
    assert await adapter.collection_exists() is True
    await adapter.ensure_collection()
    await adapter.delete_collection()
    assert await adapter.collection_exists() is False
    await adapter.delete_collection()


async def test_close_is_idempotent(adapter: VectorStoreAdapter) -> None:
    await adapter.close()
    await adapter.close()


async def test_get_returns_existing_records_in_request_order(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.GET)
    records = await loaded.get(["b_chunk_0", "missing", "a_chunk_0"])
    assert [record.id for record in records] == ["b_chunk_0", "a_chunk_0"]
    assert all(record.vector is None for record in records)


async def test_field_types_round_trip(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.GET)
    (record,) = await loaded.get(["b_chunk_0"])
    assert dict(record.fields) == {
        "source_path": "b.md",
        "chunk_index": 0,
        "content": "beta cherry tree",
        "mtime": 2.25,
        "pinned": True,
    }
    assert isinstance(record.fields["chunk_index"], int)
    assert isinstance(record.fields["pinned"], bool)


async def test_get_includes_vectors_on_request(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.GET)
    (record,) = await loaded.get(["a_chunk_1"], include_vectors=True)
    assert record.vector is not None
    # Stores with cosine collections may return the vector normalized.
    norm = math.sqrt(sum(value * value for value in record.vector))
    unit = [value / norm for value in record.vector]
    assert unit == pytest.approx([0.993884, 0.110432, 0.0, 0.0], abs=1e-5)


async def test_upsert_replaces_a_record_with_the_same_id(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.GET, Operation.COUNT)
    await loaded.upsert(
        [make_record("a_chunk_0", [0.0, 0.0, 0.0, 1.0], content="rewritten")]
    )
    (record,) = await loaded.get(["a_chunk_0"])
    assert record.fields["content"] == "rewritten"
    assert await loaded.count() == len(CORPUS)


@pytest.mark.parametrize(
    "bad",
    [
        Record(id="x", fields={"source_path": "a.md"}, vector=[1.0, 0.0]),
        Record(id="x", fields={"source_path": "a.md"}, vector=None),
        Record(id="x", fields={"not_in_schema": "v"}, vector=[1.0, 0.0, 0.0, 0.0]),
        Record(id="", fields={}, vector=[1.0, 0.0, 0.0, 0.0]),
    ],
    ids=["wrong-dimensions", "no-vector", "unknown-field", "empty-id"],
)
async def test_upsert_rejects_a_record_that_does_not_fit_the_schema(
    loaded: VectorStoreAdapter, bad: Record
) -> None:
    needs(loaded, Operation.COUNT)
    with pytest.raises(SchemaError):
        await loaded.upsert([make_record("ok", [1.0, 0.0, 0.0, 0.0]), bad])
    assert await loaded.count() == len(CORPUS)


async def test_delete_ignores_missing_ids(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.DELETE, Operation.GET)
    await loaded.delete(["a_chunk_0", "missing"])
    assert await loaded.get(["a_chunk_0"]) == []
    assert [r.id for r in await loaded.get(["a_chunk_1"])] == ["a_chunk_1"]


async def test_count(adapter: VectorStoreAdapter) -> None:
    needs(adapter, Operation.COUNT)
    assert await adapter.count() == 0
    await adapter.ensure_collection()
    assert await adapter.count() == 0
    await adapter.upsert(CORPUS)
    assert await adapter.count() == len(CORPUS)


async def test_find_matches_exact_field_values(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.FIND)
    by_source = await loaded.find(RecordFilter.where(source_path="b.md"))
    assert sorted(record.id for record in by_source) == ["b_chunk_0", "b_chunk_1"]
    assert all(record.vector is None for record in by_source)

    both = await loaded.find(RecordFilter.where(source_path="b.md", chunk_index=1))
    assert [record.id for record in both] == ["b_chunk_1"]

    assert await loaded.find(RecordFilter.where(source_path="b")) == []
    assert len(await loaded.find(RecordFilter.where(source_path="b.md"), limit=1)) == 1


async def test_list_page_returns_every_record_once(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.LIST)
    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = await loaded.list_page(cursor=cursor, limit=2)
        assert len(page.records) <= 2
        assert all(record.vector is None for record in page.records)
        seen.extend(record.id for record in page.records)
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break
        assert pages < 10
    assert sorted(seen) == sorted(record.id for record in CORPUS)


async def test_delete_where_removes_only_matching_records(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.DELETE_WHERE, Operation.FIND, Operation.COUNT)
    removed = await loaded.delete_where(RecordFilter.where(source_path="b.md"))
    assert removed == 2
    assert await loaded.find(RecordFilter.where(source_path="b.md")) == []
    assert await loaded.count() == len(CORPUS) - 2
    assert await loaded.delete_where(RecordFilter.where(source_path="b.md")) == 0


async def test_replace_source_removes_stale_chunks_when_a_source_shrinks(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.FIND, Operation.DELETE, Operation.UPSERT)
    replacement = [make_record("a_chunk_0", [0.5, 0.5, 0.0, 0.0], content="shorter")]
    await loaded.replace_source(RecordFilter.where(source_path="a.md"), replacement)
    remaining = await loaded.find(RecordFilter.where(source_path="a.md"))
    assert [record.id for record in remaining] == ["a_chunk_0"]
    assert remaining[0].fields["content"] == "shorter"
    assert len(await loaded.find(RecordFilter.where(source_path="b.md"))) == 2


async def test_vector_search_orders_by_similarity(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.VECTOR_SEARCH)
    hits = await loaded.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], top_k=3))
    assert [hit.record.id for hit in hits[:2]] == ["a_chunk_0", "a_chunk_1"]
    assert len(hits) == 3
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[1].score == pytest.approx(0.993884, abs=1e-4)
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    assert all(hit.kind is ScoreKind.SIMILARITY for hit in hits)
    assert all(hit.record.vector is None for hit in hits)
    assert hits[0].record.fields["content"] == "alpha apple orchard"


async def test_vector_search_applies_the_filter(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.VECTOR_SEARCH)
    hits = await loaded.vector_search(
        VectorQuery(
            [1.0, 0.0, 0.0, 0.0], top_k=5, filter=RecordFilter.where(source_path="b.md")
        )
    )
    assert sorted(hit.record.id for hit in hits) == ["b_chunk_0", "b_chunk_1"]


async def test_vector_search_rejects_a_wrong_sized_query(
    loaded: VectorStoreAdapter,
) -> None:
    needs(loaded, Operation.VECTOR_SEARCH)
    with pytest.raises(SchemaError):
        await loaded.vector_search(VectorQuery([1.0] * (DIMENSIONS + 1), top_k=1))


async def test_lexical_search_matches_tokens(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.LEXICAL_SEARCH)
    hits = await loaded.lexical_search(
        LexicalQuery("zeppelin", top_k=3, text_field="content")
    )
    assert [hit.record.id for hit in hits] == ["b_chunk_1"]
    assert hits[0].kind is not ScoreKind.SIMILARITY
    assert (
        await loaded.lexical_search(
            LexicalQuery("nomatchtoken", top_k=3, text_field="content")
        )
        == []
    )


async def test_hybrid_search_uses_both_signals(loaded: VectorStoreAdapter) -> None:
    needs(loaded, Operation.HYBRID_SEARCH)
    hits = await loaded.hybrid_search(
        HybridQuery("zeppelin", [1.0, 0.0, 0.0, 0.0], top_k=5, text_field="content")
    )
    ids = [hit.record.id for hit in hits]
    assert "b_chunk_1" in ids[:2]
    assert "a_chunk_0" in ids[:2]
    assert all(hit.kind is ScoreKind.FUSED for hit in hits)


@pytest.mark.parametrize(
    "operation",
    [Operation.LEXICAL_SEARCH, Operation.HYBRID_SEARCH, Operation.FIND, Operation.LIST],
    ids=lambda operation: operation.name.lower(),
)
async def test_unsupported_operations_raise_instead_of_returning_nothing(
    loaded: VectorStoreAdapter, operation: Operation
) -> None:
    if loaded.capabilities.supports(operation):
        pytest.skip(f"{loaded.provider} supports {operation.value}")
    vector = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(UnsupportedOperationError) as raised:
        if operation is Operation.LEXICAL_SEARCH:
            await loaded.lexical_search(LexicalQuery("alpha", 3, "content"))
        elif operation is Operation.HYBRID_SEARCH:
            await loaded.hybrid_search(HybridQuery("alpha", vector, 3, "content"))
        elif operation is Operation.FIND:
            await loaded.find(RecordFilter.where(source_path="a.md"))
        else:
            await loaded.list_page()
    assert loaded.provider in str(raised.value)


async def test_vector_search_ranks_negative_similarities_before_clamping(
    adapter: VectorStoreAdapter,
) -> None:
    needs(adapter, Operation.VECTOR_SEARCH)
    await adapter.ensure_collection()
    await adapter.upsert(
        [
            make_record("opposite", [-1.0, 0.0, 0.0, 0.0]),
            make_record("nearly-orthogonal", [-0.1, 1.0, 0.0, 0.0]),
        ]
    )
    hits = await adapter.vector_search(VectorQuery([1.0, 0.0, 0.0, 0.0], top_k=1))
    assert [hit.record.id for hit in hits] == ["nearly-orthogonal"]
    assert hits[0].score == 0.0
