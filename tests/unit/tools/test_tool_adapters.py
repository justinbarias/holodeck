"""Tools on the adapter contract: capability checks, source replacement, close.

Spec 042: FR-014 (unsupported operations fail before ingestion), stale chunks
of a shrunken file are removed, and a tool releases its adapter on close.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from holodeck.lib.retrieval import (
    CollectionSchema,
    Operation,
    Record,
    UnsupportedOperationError,
)
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.unit.tools.fakes import ALL_OPERATIONS, FakeAdapter


class SharedAdapters:
    """Patch ``create_adapter`` so every tool built in a test shares one store."""

    def __init__(self, operations: frozenset[Operation] = ALL_OPERATIONS) -> None:
        self.operations = operations
        self.adapter: FakeAdapter | None = None

    def create(
        self,
        provider: str,
        schema: CollectionSchema,
        connection: Mapping[str, object] | None = None,
    ) -> FakeAdapter:
        if self.adapter is None:
            self.adapter = FakeAdapter(
                schema, provider=provider, operations=self.operations
            )
        return self.adapter


@pytest.fixture
def shared() -> Iterator[SharedAdapters]:
    adapters = SharedAdapters()
    with patch("holodeck.lib.retrieval.create_adapter", side_effect=adapters.create):
        yield adapters


def _without(*operations: Operation) -> frozenset[Operation]:
    return ALL_OPERATIONS - set(operations)


def _touch_later(path: Path) -> None:
    """Move the file's mtime forward so incremental ingestion sees a change."""
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))


def _vector_tool(source: Path) -> VectorStoreTool:
    return VectorStoreTool(
        VectorstoreToolConfig(
            name="kb",
            description="Knowledge base",
            source=str(source),
            chunk_size=40,
            chunk_overlap=0,
            embedding_dimensions=8,
        )
    )


def _hierarchical_tool(source: Path) -> HierarchicalDocumentTool:
    return HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name="docs",
            description="Documents",
            source=str(source),
            search_mode=SearchMode.SEMANTIC,
            contextual_embeddings=False,
        )
    )


def _long_text(paragraphs: int) -> str:
    return "\n\n".join(
        f"Paragraph {n} talks about subject {n} in enough words to fill a chunk."
        for n in range(paragraphs)
    )


def _sections(count: int) -> str:
    return "\n\n".join(
        f"## Section {n}\n\nBody of section {n} with its own distinct content."
        for n in range(count)
    )


# -- FR-014: capability checks before ingestion --------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    [Operation.GET, Operation.DELETE, Operation.UPSERT, Operation.VECTOR_SEARCH],
)
async def test_vectorstore_tool_rejects_a_provider_without_a_needed_operation(
    tmp_path: Path, missing: Operation
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_long_text(3))
    adapters = SharedAdapters(_without(missing))
    tool = _vector_tool(source)

    with (
        patch("holodeck.lib.retrieval.create_adapter", side_effect=adapters.create),
        pytest.raises(UnsupportedOperationError, match=missing.value),
    ):
        await tool.initialize()

    assert adapters.adapter is not None
    assert adapters.adapter.stored() == []  # nothing ingested


@pytest.mark.asyncio
async def test_vectorstore_tool_needs_no_filter_operations(
    tmp_path: Path,
) -> None:
    """Providers without FIND/LIST keep working: chunk IDs are walked by key."""
    source = tmp_path / "doc.md"
    source.write_text(_long_text(3))
    filters = (Operation.FIND, Operation.LIST, Operation.DELETE_WHERE)
    adapters = SharedAdapters(_without(*filters, Operation.COUNT))
    tool = _vector_tool(source)

    with patch("holodeck.lib.retrieval.create_adapter", side_effect=adapters.create):
        await tool.initialize(force_ingest=True)

    assert tool.document_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    [Operation.FIND, Operation.DELETE, Operation.UPSERT, Operation.VECTOR_SEARCH],
)
async def test_hierarchical_tool_rejects_a_provider_without_a_needed_operation(
    tmp_path: Path, missing: Operation
) -> None:
    """Without FIND a changed file would be re-ingested forever (FR-014)."""
    source = tmp_path / "doc.md"
    source.write_text(_sections(2))
    adapters = SharedAdapters(_without(missing))
    tool = _hierarchical_tool(source)

    with (
        patch("holodeck.lib.retrieval.create_adapter", side_effect=adapters.create),
        pytest.raises(UnsupportedOperationError, match=missing.value),
    ):
        await tool.initialize(force_ingest=False)

    assert adapters.adapter is not None
    assert adapters.adapter.stored() == []
    assert adapters.adapter.deleted == []


# -- stale chunks of a shrunken file -------------------------------------------


@pytest.mark.asyncio
async def test_vectorstore_tool_removes_chunks_a_shrunken_file_no_longer_has(
    tmp_path: Path, shared: SharedAdapters
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_long_text(6))
    await _vector_tool(source).initialize()
    assert shared.adapter is not None
    before = len(shared.adapter.stored())

    source.write_text(_long_text(2))
    _touch_later(source)
    tool = _vector_tool(source)
    await tool.initialize(force_ingest=False)

    after = shared.adapter.stored()
    assert 0 < len(after) < before
    assert len(after) == tool.document_count
    assert sorted(int(r.fields["chunk_index"]) for r in after) == list(
        range(len(after))
    )
    assert all("Paragraph 5" not in str(r.fields["content"]) for r in after)


@pytest.mark.asyncio
async def test_hierarchical_tool_removes_chunks_a_shrunken_file_no_longer_has(
    tmp_path: Path, shared: SharedAdapters
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(5))
    await _hierarchical_tool(source).initialize(force_ingest=False)
    assert shared.adapter is not None
    before = len(shared.adapter.stored())

    source.write_text(_sections(2))
    _touch_later(source)
    tool = _hierarchical_tool(source)
    await tool.initialize(force_ingest=False)

    after = shared.adapter.stored()
    assert 0 < len(after) < before
    assert len(after) == len(tool._chunks)
    assert all("section 4" not in str(r.fields["content"]) for r in after)


@pytest.mark.asyncio
async def test_hierarchical_tool_skips_an_unchanged_file(
    tmp_path: Path, shared: SharedAdapters
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(3))
    await _hierarchical_tool(source).initialize(force_ingest=False)
    assert shared.adapter is not None
    stored = len(shared.adapter.stored())

    tool = _hierarchical_tool(source)
    tool._setup_collection("openai")
    skipped = await tool._ingest_documents(force_ingest=False)

    assert skipped == 1
    assert len(shared.adapter.stored()) == stored


# -- completion marker ---------------------------------------------------------


class BatchFailingAdapter(FakeAdapter):
    """Fake adapter that writes in batches and fails on one chosen batch.

    Models a store that splits a large upsert into several requests, one of
    which fails after earlier ones were applied.
    """

    batch_size = 2

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.fail_on_batch: int | None = None
        self.batches = 0

    async def upsert(self, records: Sequence[Record]) -> None:
        for start in range(0, len(records), self.batch_size):
            self.batches += 1
            if self.fail_on_batch is not None and self.batches >= self.fail_on_batch:
                raise RuntimeError("simulated failure of a write batch")
            await super().upsert(records[start : start + self.batch_size])


@pytest.mark.asyncio
async def test_vectorstore_tool_publishes_chunk_zero_only_after_the_other_chunks(
    tmp_path: Path,
) -> None:
    """A write that fails midway leaves the old chunk 0, so the next start retries."""
    source = tmp_path / "doc.md"
    source.write_text(_long_text(10))
    holder: dict[str, BatchFailingAdapter] = {}

    def create(
        provider: str,
        schema: CollectionSchema,
        connection: Mapping[str, object] | None = None,
    ) -> BatchFailingAdapter:
        if "adapter" not in holder:
            holder["adapter"] = BatchFailingAdapter(schema, provider=provider)
        return holder["adapter"]

    with patch("holodeck.lib.retrieval.create_adapter", side_effect=create):
        await _vector_tool(source).initialize()
        adapter = holder["adapter"]
        chunk_zero = next(r for r in adapter.stored() if r.fields["chunk_index"] == 0)
        old_mtime = chunk_zero.fields["mtime"]

        source.write_text(_long_text(10) + "\n\nAn appended paragraph.")
        _touch_later(source)
        tool = _vector_tool(source)
        adapter.fail_on_batch = adapter.batches + 2  # fails after one batch applied

        with pytest.raises(RuntimeError, match="simulated"):
            await tool.initialize(force_ingest=False)

        stored_zero = next(r for r in adapter.stored() if r.fields["chunk_index"] == 0)
        assert stored_zero.fields["mtime"] == old_mtime
        assert await tool._needs_reingest(source) is True


@pytest.mark.asyncio
async def test_hierarchical_tool_retries_a_source_whose_rewrite_stopped_midway(
    tmp_path: Path,
) -> None:
    """One stored chunk with an old mtime is enough to ingest the source again."""
    source = tmp_path / "doc.md"
    source.write_text(_sections(6))
    holder: dict[str, BatchFailingAdapter] = {}

    def create(
        provider: str,
        schema: CollectionSchema,
        connection: Mapping[str, object] | None = None,
    ) -> BatchFailingAdapter:
        if "adapter" not in holder:
            holder["adapter"] = BatchFailingAdapter(schema, provider=provider)
        return holder["adapter"]

    with patch("holodeck.lib.retrieval.create_adapter", side_effect=create):
        await _hierarchical_tool(source).initialize(force_ingest=False)
        adapter = holder["adapter"]

        source.write_text(_sections(6) + "\n\n## Appendix\n\nExtra closing words.")
        _touch_later(source)
        tool = _hierarchical_tool(source)
        adapter.fail_on_batch = adapter.batches + 2  # fails after one batch applied

        with pytest.raises(RuntimeError, match="simulated"):
            await tool.initialize(force_ingest=False)

        # Chunks of the first batch carry the new mtime, the rest the old one.
        assert len({r.fields["mtime"] for r in adapter.stored()}) == 2
        assert await tool._needs_reingest(source) is True

        adapter.fail_on_batch = None
        await _hierarchical_tool(source).initialize(force_ingest=False)
        assert len({r.fields["mtime"] for r in adapter.stored()}) == 1


# -- lifecycle -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_vectorstore_tool_close_closes_its_adapter(
    tmp_path: Path, shared: SharedAdapters
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_long_text(2))
    tool = _vector_tool(source)
    await tool.initialize()

    await tool.close()

    assert shared.adapter is not None
    assert shared.adapter.close_calls == 1


@pytest.mark.asyncio
async def test_hierarchical_tool_cleanup_closes_its_adapter(
    tmp_path: Path, shared: SharedAdapters
) -> None:
    """Backends call ``cleanup()`` on owned tools at teardown."""
    source = tmp_path / "doc.md"
    source.write_text(_sections(2))
    tool = _hierarchical_tool(source)
    await tool.initialize()

    await tool.cleanup()

    assert shared.adapter is not None
    assert shared.adapter.close_calls == 1
