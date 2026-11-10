"""Regression tests for the phase 3 review findings on the Weaviate adapter.

Needs the Weaviate container on 127.0.0.1:8089 (gRPC 50059).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    IncompatibleCollectionError,
    Record,
    RecordFilter,
    SchemaError,
)
from holodeck.lib.retrieval.adapters import weaviate as module
from holodeck.lib.retrieval.adapters.weaviate import (
    WeaviateAdapter,
    weaviate_class_name,
)
from tests.contract.retrieval.targets import WEAVIATE_CONNECTION, port_open

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_reason = port_open("127.0.0.1", 8089)()
if _reason is not None:
    pytestmark.append(pytest.mark.skip(reason=_reason))


def _schema() -> CollectionSchema:
    return CollectionSchema(
        name=f"Review_{uuid.uuid4().hex[:12]}",
        dimensions=3,
        fields=(
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
        ),
    )


def _record(record_id: str, source: str = "a.md", index: int = 0) -> Record:
    return Record(
        record_id, {"source_path": source, "chunk_index": index}, [1.0, 0.0, 0.0]
    )


@pytest_asyncio.fixture
async def raw() -> AsyncIterator[WeaviateAsyncClient]:
    client = weaviate.use_async_with_local(host="127.0.0.1", port=8089, grpc_port=50059)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def adapter() -> AsyncIterator[WeaviateAdapter]:
    built = WeaviateAdapter.create(_schema(), WEAVIATE_CONNECTION)
    await built.ensure_collection()
    try:
        yield built
    finally:
        await built.delete_collection()
        await built.close()


# -- 1: IDs that share an object UUID ------------------------------------------------


async def test_two_ids_that_share_an_object_uuid_are_rejected_in_one_call(
    adapter: WeaviateAdapter,
) -> None:
    clash = str(adapter.store_id("invoice-7"))
    with pytest.raises(SchemaError, match="same store ID"):
        await adapter.upsert([_record("invoice-7"), _record(clash)])
    with pytest.raises(SchemaError, match="same store ID"):
        await adapter.replace_source(
            RecordFilter.where(source_path="a.md"),
            [_record(clash), _record("invoice-7")],
        )
    assert await adapter.count() == 0


async def test_two_spellings_of_one_uuid_are_one_id(adapter: WeaviateAdapter) -> None:
    native = str(uuid.uuid4())
    await adapter.upsert([_record(native), _record(native.upper())])
    assert await adapter.count() == 1


@pytest.mark.parametrize("hashed_first", [True, False])
async def test_an_id_cannot_overwrite_a_record_with_another_id(
    adapter: WeaviateAdapter, hashed_first: bool
) -> None:
    clash = str(adapter.store_id("invoice-7"))
    first, second = ("invoice-7", clash) if hashed_first else (clash, "invoice-7")
    await adapter.upsert([_record(first, index=1)])
    with pytest.raises(SchemaError, match="already holds"):
        await adapter.upsert([_record("fresh"), _record(second, index=2)])
    with pytest.raises(SchemaError, match="already holds"):
        await adapter.replace_source(
            RecordFilter.where(source_path="a.md"), [_record(second, index=2)]
        )
    (kept,) = await adapter.get([first])
    assert (kept.id, kept.fields["chunk_index"]) == (first, 1)
    assert await adapter.count() == 1
    await adapter.upsert([_record(first, index=3)])  # the owner may rewrite it
    (kept,) = await adapter.get([first])
    assert kept.fields["chunk_index"] == 3


# -- 6: more matches than the server returns for one query ---------------------------


async def test_find_delete_where_and_replace_source_pass_the_query_ceiling(
    adapter: WeaviateAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "QUERY_MAXIMUM_RESULTS", 20)
    scans: list[int] = []
    original = WeaviateAdapter._scan

    async def counting_scan(*args: Any, **kwargs: Any) -> list[Record]:
        scans.append(1)
        return await original(*args, **kwargs)

    monkeypatch.setattr(WeaviateAdapter, "_scan", counting_scan)
    await adapter.upsert(
        [_record(f"big_{i}", "big.md", i) for i in range(45)]
        + [_record(f"other_{i}", "other.md", i) for i in range(5)]
    )
    big = RecordFilter.where(source_path="big.md")

    found = await adapter.find(big)
    assert sorted(r.id for r in found) == sorted(f"big_{i}" for i in range(45))
    assert len(await adapter.find(big, limit=7)) == 7
    assert len(await adapter.find(RecordFilter.where(source_path="other.md"))) == 5

    replacement = [_record("big_0", "big.md", 0), _record("big_1", "big.md", 1)]
    await adapter.replace_source(big, replacement)
    assert sorted(r.id for r in await adapter.find(big)) == ["big_0", "big_1"]

    await adapter.upsert([_record(f"more_{i}", "big.md", i) for i in range(45)])
    assert await adapter.delete_where(big) == 47
    assert await adapter.find(big) == []
    assert await adapter.count() == 5
    # find, replace_source and delete_where each enumerated by cursor
    assert len(scans) == 3


# -- 7: tokenization of an existing collection ---------------------------------------


async def test_an_existing_collection_with_word_tokenized_filter_fields_is_refused(
    raw: WeaviateAsyncClient,
) -> None:
    schema = _schema()
    name = weaviate_class_name(schema.name)
    await raw.collections.create(
        name=name,
        properties=[
            Property(
                name="source_path",
                data_type=DataType.TEXT,
                index_filterable=True,
                tokenization=Tokenization.WORD,
            ),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
        vector_config=Configure.Vectors.self_provided(
            name="embedding",
            vector_index_config=Configure.VectorIndex.flat(
                distance_metric=VectorDistances.COSINE
            ),
        ),
    )
    built = WeaviateAdapter.create(schema, WEAVIATE_CONNECTION)
    try:
        with pytest.raises(IncompatibleCollectionError, match="source_path"):
            await built.ensure_collection()
    finally:
        await built.close()
        await raw.collections.delete(name)


# -- 9: the distance of an existing collection ---------------------------------------


async def test_an_existing_collection_with_another_distance_is_refused(
    raw: WeaviateAsyncClient,
) -> None:
    schema = _schema()
    name = weaviate_class_name(schema.name)
    await raw.collections.create(
        name=name,
        properties=[
            Property(
                name="source_path",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
        ],
        vector_config=Configure.Vectors.self_provided(
            name="embedding",
            vector_index_config=Configure.VectorIndex.flat(
                distance_metric=VectorDistances.DOT
            ),
        ),
    )
    built = WeaviateAdapter.create(schema, WEAVIATE_CONNECTION)
    try:
        with pytest.raises(IncompatibleCollectionError, match="cosine"):
            await built.ensure_collection()
    finally:
        await built.close()
        await raw.collections.delete(name)
