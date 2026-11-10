"""Tools on the pipeline fingerprint and completion markers (042 T15).

Spec 042: FR-016 (interrupted ingestion is recoverable, cold restart does not
re-embed unchanged sources), FR-023 (incompatible collections are detected
before writes), AC-06, AC-07, AC-08.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from holodeck.lib.retrieval import IncompatibleCollectionError, Operation
from holodeck.lib.retrieval.manifest import (
    MANIFEST_COLLECTION,
    CollectionEntry,
    SourceEntry,
    collection_entry_id,
    source_entry_id,
)
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.unit.tools.fakes import (
    ALL_OPERATIONS,
    CountingEmbeddings,
    FailingWriteError,
    StoreRegistry,
)

DIMS = 8
VEC_COLLECTION = f"documents_dim{DIMS}"
HIER_DIMS = 768  # ollama default dimensions
HIER_COLLECTION = f"hierarchical_docs_docs_dim{HIER_DIMS}"


@pytest.fixture
def store() -> Iterator[StoreRegistry]:
    registry = StoreRegistry()
    with (
        patch("holodeck.lib.retrieval.create_adapter", side_effect=registry.create),
        patch(
            "holodeck.lib.retrieval.registry.create_adapter",
            side_effect=registry.create,
        ),
    ):
        yield registry


def _touch_later(path: Path, seconds: float = 10) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds))


def _text(paragraphs: int, word: str = "subject") -> str:
    return "\n\n".join(
        f"Paragraph {n} talks about {word} {n} in enough words to fill a chunk."
        for n in range(paragraphs)
    )


def _sections(count: int) -> str:
    return "\n\n".join(
        f"## Section {n}\n\nBody of section {n} with its own distinct content."
        for n in range(count)
    )


def _vector_tool(
    source: Path, embeddings: CountingEmbeddings, chunk_size: int = 40
) -> VectorStoreTool:
    tool = VectorStoreTool(
        VectorstoreToolConfig(
            name="kb",
            description="Knowledge base",
            source=str(source),
            chunk_size=chunk_size,
            chunk_overlap=0,
            embedding_dimensions=DIMS,
            top_k=3,
        )
    )
    tool.set_embedding_service(embeddings)
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    return tool


def _hier_tool(
    source: Path, embeddings: CountingEmbeddings
) -> HierarchicalDocumentTool:
    tool = HierarchicalDocumentTool(
        HierarchicalDocumentToolConfig(
            name="docs",
            description="Documents",
            source=str(source),
            search_mode=SearchMode.SEMANTIC,
            contextual_embeddings=False,
        )
    )
    tool.set_embedding_service(embeddings)
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    return tool


def _collection_entry(store: StoreRegistry, logical: str) -> CollectionEntry:
    records = store.collections[MANIFEST_COLLECTION]
    return CollectionEntry.from_record(records[collection_entry_id(logical)])


def _source_entry(store: StoreRegistry, physical: str, key: str) -> SourceEntry:
    records = store.collections[MANIFEST_COLLECTION]
    return SourceEntry.from_record(records[source_entry_id(physical, key)])


def _data_records(store: StoreRegistry, name: str) -> int:
    return len(store.records(name))


# -- fingerprint rules --


@pytest.mark.asyncio
async def test_new_collection_records_the_current_fingerprint(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    tool = _vector_tool(source, CountingEmbeddings(DIMS))
    await tool.initialize(provider_type="ollama")

    entry = _collection_entry(store, VEC_COLLECTION)
    assert entry.fingerprint == tool.pipeline_inputs().fingerprint
    assert entry.active_collection == VEC_COLLECTION
    assert entry.state == "active"
    assert '"legacy"' not in entry.fingerprint_inputs
    assert '"chunker.policy":"plain-tiktoken-cl100k/1"' in entry.fingerprint_inputs
    assert _source_entry(store, VEC_COLLECTION, str(source)).complete


@pytest.mark.asyncio
async def test_legacy_collection_is_adopted_without_reembedding(
    tmp_path: Path, store: StoreRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    """A collection written before fingerprints stays usable (AC-08)."""
    unchanged = tmp_path / "a.md"
    changed = tmp_path / "b.md"
    unchanged.write_text(_text(3, "alpha"))
    changed.write_text(_text(3, "beta"))
    first = _vector_tool(tmp_path, CountingEmbeddings(DIMS))
    await first.initialize(provider_type="ollama")
    await first.close()
    # Simulate a pre-fingerprint store: drop the whole manifest.
    store.collections[MANIFEST_COLLECTION].clear()
    store.existing.discard(MANIFEST_COLLECTION)
    _touch_later(changed)

    embeddings = CountingEmbeddings(DIMS)
    second = _vector_tool(tmp_path, embeddings)
    with caplog.at_level(logging.INFO):
        await second.initialize(provider_type="ollama")

    entry = _collection_entry(store, VEC_COLLECTION)
    assert '"legacy":true' in entry.fingerprint_inputs
    assert entry.fingerprint == second.pipeline_inputs().fingerprint
    assert "predates pipeline fingerprints" in caplog.text
    assert "holodeck vectorstore migrate" in caplog.text
    # Only the changed source was re-embedded, as a whole.
    assert embeddings.texts
    assert all("beta" in text for text in embeddings.texts)
    assert "Found" in await second.search("alpha")


@pytest.mark.asyncio
async def test_changed_pipeline_refuses_writes_and_keeps_reads(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(6))
    first = _vector_tool(source, CountingEmbeddings(DIMS), chunk_size=40)
    await first.initialize(provider_type="ollama")
    stored = _data_records(store, VEC_COLLECTION)

    # Same sources, new chunk size: no write is needed, so start-up works and
    # searches read the existing records.
    reader = _vector_tool(source, CountingEmbeddings(DIMS), chunk_size=60)
    await reader.initialize(provider_type="ollama")
    assert "Found" in await reader.search("subject")

    # A changed source needs a write: refused, with the differing inputs.
    _touch_later(source)
    writer = _vector_tool(source, CountingEmbeddings(DIMS), chunk_size=60)
    with pytest.raises(IncompatibleCollectionError) as raised:
        await writer.initialize(provider_type="ollama")
    message = str(raised.value)
    assert "chunker.chunk_size: 40 -> 60" in message
    assert "holodeck vectorstore inspect AGENT_YAML --tool kb" in message
    assert "holodeck vectorstore migrate" in message
    assert _data_records(store, VEC_COLLECTION) == stored
    assert _collection_entry(store, VEC_COLLECTION).fingerprint == (
        first.pipeline_inputs().fingerprint
    )


@pytest.mark.asyncio
async def test_force_ingest_does_not_bypass_the_fingerprint(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    await _vector_tool(source, CountingEmbeddings(DIMS)).initialize(
        provider_type="ollama"
    )
    stored = _data_records(store, VEC_COLLECTION)
    embeddings = CountingEmbeddings(DIMS)
    other = _vector_tool(source, embeddings, chunk_size=60)
    with pytest.raises(IncompatibleCollectionError):
        await other.initialize(force_ingest=True, provider_type="ollama")
    assert _data_records(store, VEC_COLLECTION) == stored
    assert embeddings.calls == 0


@pytest.mark.asyncio
async def test_changed_embedding_model_is_refused_for_the_hierarchical_tool(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(3))
    await _hier_tool(source, CountingEmbeddings(HIER_DIMS)).initialize(
        force_ingest=False, provider_type="ollama"
    )
    other = _hier_tool(source, CountingEmbeddings(HIER_DIMS))
    other.set_embedding_identity("ollama", "mxbai-embed-large")
    with pytest.raises(IncompatibleCollectionError, match="embedding.model"):
        await other.initialize(force_ingest=True, provider_type="ollama")


@pytest.mark.asyncio
async def test_changed_pipeline_on_an_empty_collection_is_adopted(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    await _vector_tool(source, CountingEmbeddings(DIMS)).initialize(
        provider_type="ollama"
    )
    store.collections[VEC_COLLECTION].clear()  # collection emptied by hand

    tool = _vector_tool(source, CountingEmbeddings(DIMS), chunk_size=60)
    await tool.initialize(provider_type="ollama")
    assert _collection_entry(store, VEC_COLLECTION).fingerprint == (
        tool.pipeline_inputs().fingerprint
    )
    assert _data_records(store, VEC_COLLECTION) > 0


# -- completion markers --


@pytest.mark.asyncio
async def test_vectorstore_interrupted_first_ingestion_is_redone(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(8))
    store.fail_after = 2
    first = _vector_tool(source, CountingEmbeddings(DIMS))
    with pytest.raises(FailingWriteError):
        await first.initialize(provider_type="ollama")
    entry = _source_entry(store, VEC_COLLECTION, str(source))
    assert not entry.complete
    assert entry.chunk_count > 2

    store.fail_after = None
    embeddings = CountingEmbeddings(DIMS)
    await _vector_tool(source, embeddings).initialize(provider_type="ollama")
    assert embeddings.calls == 1
    assert _source_entry(store, VEC_COLLECTION, str(source)).complete
    assert _data_records(store, VEC_COLLECTION) == entry.chunk_count


@pytest.mark.asyncio
async def test_hierarchical_interrupted_first_ingestion_is_redone(
    tmp_path: Path, store: StoreRegistry
) -> None:
    """Without the marker the partial chunks all carry the current mtime and
    the source looked complete."""
    source = tmp_path / "doc.md"
    source.write_text(_sections(6))
    store.fail_after = 2
    with pytest.raises(FailingWriteError):
        await _hier_tool(source, CountingEmbeddings(HIER_DIMS)).initialize(
            force_ingest=False, provider_type="ollama"
        )
    assert _data_records(store, HIER_COLLECTION) == 2
    assert not _source_entry(store, HIER_COLLECTION, str(source)).complete

    store.fail_after = None
    embeddings = CountingEmbeddings(HIER_DIMS)
    await _hier_tool(source, embeddings).initialize(
        force_ingest=False, provider_type="ollama"
    )
    assert embeddings.calls == 1
    entry = _source_entry(store, HIER_COLLECTION, str(source))
    assert entry.complete
    assert _data_records(store, HIER_COLLECTION) == entry.chunk_count == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["vectorstore", "hierarchical"])
async def test_completed_source_is_not_reembedded_after_a_cold_restart(
    tmp_path: Path, store: StoreRegistry, kind: str
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(4) if kind == "hierarchical" else _text(4))

    async def start(embeddings: CountingEmbeddings) -> None:
        tool: VectorStoreTool | HierarchicalDocumentTool
        if kind == "hierarchical":
            tool = _hier_tool(source, embeddings)
            await tool.initialize(force_ingest=False, provider_type="ollama")
        else:
            tool = _vector_tool(source, embeddings)
            await tool.initialize(provider_type="ollama")
        await tool.close()

    dims = HIER_DIMS if kind == "hierarchical" else DIMS
    first = CountingEmbeddings(dims)
    await start(first)
    assert first.calls == 1

    restart = CountingEmbeddings(dims)
    await start(restart)
    assert restart.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["vectorstore", "hierarchical"])
async def test_source_marked_incomplete_is_redone_even_if_records_look_current(
    tmp_path: Path, store: StoreRegistry, kind: str
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(3) if kind == "hierarchical" else _text(3))
    dims = HIER_DIMS if kind == "hierarchical" else DIMS
    physical = HIER_COLLECTION if kind == "hierarchical" else VEC_COLLECTION

    def build(
        embeddings: CountingEmbeddings,
    ) -> VectorStoreTool | HierarchicalDocumentTool:
        if kind == "hierarchical":
            return _hier_tool(source, embeddings)
        return _vector_tool(source, embeddings)

    first = build(CountingEmbeddings(dims))
    await first.initialize(force_ingest=False, provider_type="ollama")
    entry = _source_entry(store, physical, str(source))
    manifest = store.collections[MANIFEST_COLLECTION]
    record = manifest[source_entry_id(physical, str(source))]
    manifest[record.id] = SourceEntry(
        physical, entry.source_key, entry.chunk_count, False, entry.content_marker
    ).to_record()

    embeddings = CountingEmbeddings(dims)
    await build(embeddings).initialize(force_ingest=False, provider_type="ollama")
    assert embeddings.calls == 1
    assert _source_entry(store, physical, str(source)).complete


@pytest.mark.asyncio
async def test_deleting_a_source_removes_its_manifest_entry(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    tool = _vector_tool(source, CountingEmbeddings(DIMS))
    await tool.initialize(provider_type="ollama")
    assert await tool._delete_file_records(source) > 0
    manifest = store.collections[MANIFEST_COLLECTION]
    assert source_entry_id(VEC_COLLECTION, str(source)) not in manifest


# -- logical and physical collection names --


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["vectorstore", "hierarchical"])
async def test_tool_uses_the_physical_collection_named_by_the_manifest(
    tmp_path: Path, store: StoreRegistry, kind: str
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_sections(3) if kind == "hierarchical" else _text(3))
    tool: VectorStoreTool | HierarchicalDocumentTool
    if kind == "hierarchical":
        tool = _hier_tool(source, CountingEmbeddings(HIER_DIMS))
        logical = HIER_COLLECTION
    else:
        tool = _vector_tool(source, CountingEmbeddings(DIMS))
        logical = VEC_COLLECTION
    physical = f"{logical}__0123abcd"
    # A finished migration points the logical name at a new collection.
    probe = store.create("in-memory", _manifest_schema())
    await probe.upsert([CollectionEntry(logical, "", "{}", physical).to_record()])

    await tool.initialize(force_ingest=False, provider_type="ollama")
    assert tool.logical_collection == logical
    assert tool.physical_collection == physical
    assert _data_records(store, physical) > 0
    assert _data_records(store, logical) == 0


def _manifest_schema():  # type: ignore[no-untyped-def]
    from holodeck.lib.retrieval.manifest import MANIFEST_SCHEMA

    return MANIFEST_SCHEMA


# -- providers without a manifest --


@pytest.mark.asyncio
async def test_provider_without_manifest_operations_keeps_old_behavior(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry = StoreRegistry(
        manifest_operations=ALL_OPERATIONS - {Operation.FIND},
    )
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    with (
        patch("holodeck.lib.retrieval.create_adapter", side_effect=registry.create),
        patch(
            "holodeck.lib.retrieval.registry.create_adapter",
            side_effect=registry.create,
        ),
        patch("holodeck.lib.retrieval.manifest._warned", set()),
        caplog.at_level(logging.WARNING),
    ):
        await _vector_tool(source, CountingEmbeddings(DIMS)).initialize(
            provider_type="ollama"
        )
        # A changed pipeline is not detected without a manifest: old behavior.
        _touch_later(source)
        await _vector_tool(source, CountingEmbeddings(DIMS), 60).initialize(
            provider_type="ollama"
        )
    assert "no pipeline manifest" in caplog.text
    assert caplog.text.count("no pipeline manifest") == 1
    assert registry.records(MANIFEST_COLLECTION) == []
    assert registry.records(VEC_COLLECTION)
