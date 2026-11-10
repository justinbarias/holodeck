"""Fixtures for the vector store adapter contract suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio

from holodeck.lib.retrieval import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    Record,
    VectorStoreAdapter,
)
from tests.contract.retrieval.targets import TARGETS, ContractTarget

DIMENSIONS = 4


def make_schema() -> CollectionSchema:
    """Return a schema with one field of every type and a unique name."""
    return CollectionSchema(
        name=f"contract_{uuid4().hex[:12]}",
        dimensions=DIMENSIONS,
        fields=(
            FieldSpec("source_path", FieldType.STRING, indexed=True),
            FieldSpec("chunk_index", FieldType.INTEGER, indexed=True),
            FieldSpec("content", FieldType.STRING, full_text=True),
            FieldSpec("mtime", FieldType.FLOAT),
            FieldSpec("pinned", FieldType.BOOLEAN),
        ),
    )


def make_record(
    record_id: str,
    vector: list[float],
    *,
    source: str = "a.md",
    index: int = 0,
    content: str = "",
    mtime: float = 1.5,
    pinned: bool = False,
) -> Record:
    """Return a record that fits ``make_schema``."""
    return Record(
        id=record_id,
        fields={
            "source_path": source,
            "chunk_index": index,
            "content": content,
            "mtime": mtime,
            "pinned": pinned,
        },
        vector=vector,
    )


CORPUS: list[Record] = [
    make_record("a_chunk_0", [1.0, 0.0, 0.0, 0.0], content="alpha apple orchard"),
    make_record(
        "a_chunk_1", [0.9, 0.1, 0.0, 0.0], index=1, content="alpha banana grove"
    ),
    make_record(
        "b_chunk_0",
        [0.0, 1.0, 0.0, 0.0],
        source="b.md",
        content="beta cherry tree",
        mtime=2.25,
        pinned=True,
    ),
    make_record(
        "b_chunk_1",
        [0.0, 0.0, 1.0, 0.0],
        source="b.md",
        index=1,
        content="beta zeppelin hangar",
    ),
    make_record(
        "c_chunk_0", [0.0, 0.0, 0.0, 1.0], source="c.md", content="gamma delta"
    ),
]


@pytest.fixture(params=TARGETS, ids=lambda target: target.name)
def target(request: pytest.FixtureRequest) -> ContractTarget:
    """One adapter target; skipped when its store is not reachable."""
    chosen: ContractTarget = request.param
    reason = chosen.available()
    if reason is not None:
        pytest.skip(reason)
    return chosen


@pytest_asyncio.fixture
async def adapter(target: ContractTarget) -> AsyncIterator[VectorStoreAdapter]:
    """An adapter on an empty collection that does not exist yet."""
    instance = target.build(make_schema())
    try:
        yield instance
    finally:
        try:
            await instance.delete_collection()
        finally:
            await instance.close()


@pytest_asyncio.fixture
async def loaded(adapter: VectorStoreAdapter) -> VectorStoreAdapter:
    """An adapter whose collection holds ``CORPUS``."""
    await adapter.ensure_collection()
    await adapter.upsert(CORPUS)
    return adapter
