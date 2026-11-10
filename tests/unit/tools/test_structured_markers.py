"""Structured (CSV/JSON/JSONL) ingestion writes and honors completion markers.

Spec 042 FR-016 (cold restart does not re-embed unchanged sources, an
interrupted ingestion is redone) for the structured mode of the vectorstore
tool (042 phase 4 review, item 8).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from holodeck.lib.retrieval.manifest import (
    MANIFEST_COLLECTION,
    SourceEntry,
    source_entry_id,
)
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.unit.tools.fakes import CountingEmbeddings, FailingWriteError, StoreRegistry

DIMS = 8


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


def _rows(count: int, word: str = "row") -> str:
    lines = ["id,title,body"]
    lines += [f"{n},title {n},{word} body number {n}" for n in range(count)]
    return "\n".join(lines) + "\n"


def _tool(source: Path, embeddings: CountingEmbeddings) -> VectorStoreTool:
    tool = VectorStoreTool(
        VectorstoreToolConfig(
            name="rows",
            description="Rows",
            source=str(source),
            vector_field="body",
            id_field="id",
            meta_fields=["title"],
            embedding_dimensions=DIMS,
        )
    )
    tool.set_embedding_service(embeddings)
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    return tool


def _marker(store: StoreRegistry, source: Path) -> SourceEntry:
    record = store.collections[MANIFEST_COLLECTION][
        source_entry_id("rows", str(source))
    ]
    return SourceEntry.from_record(record)


@pytest.mark.asyncio
async def test_structured_source_gets_a_marker_and_is_not_reembedded(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "rows.csv"
    source.write_text(_rows(5))
    first = _tool(source, CountingEmbeddings(DIMS))
    await first.initialize(provider_type="ollama")
    marker = _marker(store, source)
    assert marker.complete and marker.chunk_count == 5
    assert len(store.records("rows")) == 5

    embeddings = CountingEmbeddings(DIMS)
    second = _tool(source, embeddings)
    await second.initialize(provider_type="ollama")
    assert embeddings.calls == 0
    assert second.document_count == 5
    assert "Found" in await second.search("row body")

    source.write_text(_rows(6, "changed"))
    changed = CountingEmbeddings(DIMS)
    await _tool(source, changed).initialize(provider_type="ollama")
    assert changed.texts and all("changed" in text for text in changed.texts)
    assert _marker(store, source).chunk_count == 6


@pytest.mark.asyncio
async def test_interrupted_structured_ingestion_is_redone(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "rows.csv"
    source.write_text(_rows(5))
    store.fail_after = 2
    with pytest.raises(FailingWriteError):
        await _tool(source, CountingEmbeddings(DIMS)).initialize(provider_type="ollama")
    assert not _marker(store, source).complete
    store.fail_after = None
    embeddings = CountingEmbeddings(DIMS)
    await _tool(source, embeddings).initialize(provider_type="ollama")
    assert len(embeddings.texts) == 5
    assert _marker(store, source).complete


@pytest.mark.asyncio
async def test_force_ingest_reembeds_a_complete_structured_source(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "rows.csv"
    source.write_text(_rows(3))
    await _tool(source, CountingEmbeddings(DIMS)).initialize(provider_type="ollama")
    embeddings = CountingEmbeddings(DIMS)
    await _tool(source, embeddings).initialize(
        force_ingest=True, provider_type="ollama"
    )
    assert len(embeddings.texts) == 3
