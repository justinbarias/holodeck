"""Tests for the HierarchicalDocumentTool -> ChunkSource adapter (spec 043).

The tool runs for real on the in-memory store with placeholder embeddings:
no network, no LLM and no embedding call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from holodeck.lib.backends.base import BackendSessionError
from holodeck.lib.hybrid_search import SearchResult
from holodeck.lib.memory.chunk_source import (
    UNKNOWN_ORDINAL,
    ChunkSource,
    SourceDocument,
)
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk
from holodeck.models.tool import HierarchicalDocumentToolConfig, SearchMode
from holodeck.tools.hierarchical_chunk_source import HierarchicalChunkSource
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

# The first chunk carries the front matter and body text together.
DATED = (
    "---\neffective_date: 2022-07-01\n---\n"
    "Opening clause about zebra payments.\n\n"
    "# Part 1\n\nRate rules for zebra payments.\n\n"
    "## Division 2\n\nIndexation of rates.\n"
)
PLAIN = "# Guide\n\nGeneral guidance text.\n"


def _tool(source: Path, **overrides: object) -> HierarchicalDocumentTool:
    config = HierarchicalDocumentToolConfig(
        name="acts",
        description="Acts",
        source=str(source),
        search_mode=SearchMode.KEYWORD,
        **overrides,
    )
    return HierarchicalDocumentTool(config)


@pytest_asyncio.fixture
async def source(tmp_path: Path) -> HierarchicalChunkSource:
    (tmp_path / "a_dated.md").write_text(DATED)
    (tmp_path / "b_plain.md").write_text(PLAIN)
    tool = _tool(tmp_path)
    await tool.initialize()
    return HierarchicalChunkSource(tool)


async def test_adapter_is_a_chunk_source(source: HierarchicalChunkSource) -> None:
    assert isinstance(source, ChunkSource)
    assert source.name == "acts"


async def test_documents_are_local_files_with_reliable_mtime(
    source: HierarchicalChunkSource, tmp_path: Path
) -> None:
    documents = await source.documents()

    assert documents == [
        SourceDocument("acts", str(tmp_path / name), tmp_path / name, True)
        for name in ("a_dated.md", "b_plain.md")
    ]


async def test_a6_front_matter_removed_from_chunk_text(
    source: HierarchicalChunkSource,
) -> None:
    dated = (await source.documents())[0]

    first = (await source.chunks(dated))[0]

    assert first.text == "Opening clause about zebra payments."
    assert "effective_date" not in first.text


async def test_a6_stored_chunk_text_unchanged(source: HierarchicalChunkSource) -> None:
    dated = (await source.documents())[0]
    stored = await source._tool.document_chunks()

    assert stored[dated.source_uri][0].content.startswith("---\neffective_date:")


async def test_ret02_find_text_matches_chunks_text(
    source: HierarchicalChunkSource,
) -> None:
    dated = (await source.documents())[0]
    listed = {ref.chunk_id: ref for ref in await source.chunks(dated)}

    matches = await source.find("opening clause zebra", 5)

    first = next(m for m in matches if m.chunk.chunk_id in listed)
    same = listed[first.chunk.chunk_id]
    assert (first.chunk.text, first.chunk.section_id, first.chunk.source_uri) == (
        same.text,
        same.section_id,
        same.source_uri,
    )
    assert first.chunk.text == "Opening clause about zebra payments."
    assert first.chunk.ordinal == UNKNOWN_ORDINAL


async def test_ret02_find_does_not_load_the_corpus(
    source: HierarchicalChunkSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail() -> dict[str, list[object]]:
        raise AssertionError("find() must not read stored chunks")

    monkeypatch.setattr(source._tool, "document_chunks", fail)

    assert await source.find("indexation of rates", 3)


async def test_ret02_find_keeps_tool_order_and_scores(
    source: HierarchicalChunkSource,
) -> None:
    results = await source._tool.search("indexation of rates", 3)

    matches = await source.find("indexation of rates", 3)

    assert results, "the query must match, or the comparison proves nothing"
    assert [(m.chunk.chunk_id, m.score) for m in matches] == [
        (r.chunk_id, r.fused_score) for r in results
    ]


async def test_find_drops_hits_that_are_only_front_matter(
    source: HierarchicalChunkSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hit(chunk_id: str, content: str, score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            content=content,
            fused_score=score,
            source_path="/docs/a.md",
            parent_chain=[],
            section_id="1",
        )

    results = [
        hit("a", "---\neffective_date: 2019-07-01\n---", 0.9),
        hit("b", "# T\n\nbody", 0.5),
    ]
    monkeypatch.setattr(source._tool, "search", AsyncMock(return_value=results))

    matches = await source.find("query", 5)

    assert [(m.chunk.chunk_id, m.chunk.text, m.score) for m in matches] == [
        ("b", "# T\n\nbody", 0.5)
    ]


async def test_chunks_ascending_ordinal_with_section_ids(
    source: HierarchicalChunkSource,
) -> None:
    dated = (await source.documents())[0]

    refs = await source.chunks(dated)

    ordinals = [ref.ordinal for ref in refs]
    assert ordinals == sorted(ordinals)
    assert len(refs) == 3
    assert all(ref.source_uri == dated.source_uri for ref in refs)
    assert all(ref.section_id for ref in refs)


async def test_chunk_text_is_raw_content_not_contextual(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text(PLAIN)
    tool = _tool(tmp_path)
    await tool.initialize()
    stored = await tool.document_chunks()
    for chunks in stored.values():
        for chunk in chunks:
            chunk.contextualized_content = "CONTEXT " + chunk.content
    source = HierarchicalChunkSource(tool)
    source._stored = stored
    document = SourceDocument("acts", next(iter(stored)), tmp_path / "doc.md", True)

    refs = await source.chunks(document)

    assert [ref.text for ref in refs] == ["# Guide\n\nGeneral guidance text."]


async def test_heading_only_chunk_left_out(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "unused.md")
    tool._initialized = True
    source = HierarchicalChunkSource(tool)
    key = str(tmp_path / "doc.md")
    source._stored = {
        key: [
            DocumentChunk("h", key, 0, "# Title", chunk_type=ChunkType.HEADER),
            DocumentChunk("c", key, 1, "Body text", section_id="s1"),
        ]
    }

    refs = await source.chunks(SourceDocument("acts", key, Path(key), True))

    assert [ref.chunk_id for ref in refs] == ["c"]


async def test_remote_source_mtime_unreliable(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "doc.md").write_text(PLAIN)
    tool = _tool(tmp_path)
    tool.set_source_context(source_root=tmp_path, is_remote=True)
    await tool.initialize()

    documents = await HierarchicalChunkSource(tool).documents()

    assert documents == [
        SourceDocument("acts", "docs/doc.md", tmp_path / "docs" / "doc.md", False)
    ]


async def test_uninitialized_tool_raises(tmp_path: Path) -> None:
    source = HierarchicalChunkSource(_tool(tmp_path))

    with pytest.raises(BackendSessionError, match="'acts' is not initialized"):
        await source.documents()
    with pytest.raises(BackendSessionError, match="not initialized"):
        await source.find("query", 3)


async def _listing(
    source: HierarchicalChunkSource,
) -> list[tuple[SourceDocument, list[tuple[str, str, int, str, str]]]]:
    return [
        (
            document,
            [
                (ref.source_uri, ref.section_id, ref.ordinal, ref.text, ref.chunk_id)
                for ref in await source.chunks(document)
            ],
        )
        for document in await source.documents()
    ]


async def test_ing_08_preview_lists_what_ingestion_stores(
    source: HierarchicalChunkSource, tmp_path: Path
) -> None:
    preview = HierarchicalChunkSource(_tool(tmp_path), preview=True)

    listed = await _listing(preview)

    assert listed == await _listing(source)
    assert sum(len(chunks) for _, chunks in listed) == 4


async def test_ing_08_preview_of_a_remote_source_lists_what_ingestion_stores(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "doc.md").write_text(DATED)
    stored_tool = _tool(tmp_path)
    stored_tool.set_source_context(source_root=tmp_path, is_remote=True)
    await stored_tool.initialize()
    preview_tool = _tool(tmp_path)
    preview_tool.set_source_context(source_root=tmp_path, is_remote=True)

    listed = await _listing(HierarchicalChunkSource(preview_tool, preview=True))

    assert listed == await _listing(HierarchicalChunkSource(stored_tool))
    assert [document.source_uri for document, _ in listed] == ["docs/doc.md"]


async def test_ing_08_preview_does_not_initialize_embed_or_store(
    tmp_path: Path,
) -> None:
    (tmp_path / "a_dated.md").write_text(DATED)
    tool = _tool(tmp_path)
    embedding = AsyncMock()
    tool.set_embedding_service(embedding)
    preview = HierarchicalChunkSource(tool, preview=True)

    documents = await preview.documents()
    await preview.chunks(documents[0])

    assert not tool._initialized
    assert tool._adapter is None
    assert embedding.mock_calls == []


async def test_ing_08_preview_reads_the_files_again_on_each_listing(
    tmp_path: Path,
) -> None:
    (tmp_path / "a_dated.md").write_text(DATED)
    preview = HierarchicalChunkSource(_tool(tmp_path), preview=True)
    assert len(await preview.documents()) == 1

    (tmp_path / "b_plain.md").write_text(PLAIN)

    assert len(await preview.documents()) == 2


async def test_preview_source_cannot_search(tmp_path: Path) -> None:
    (tmp_path / "a_dated.md").write_text(DATED)
    preview = HierarchicalChunkSource(_tool(tmp_path), preview=True)

    with pytest.raises(BackendSessionError, match="preview source and cannot search"):
        await preview.find("zebra", 3)


async def _fail_store_read() -> dict[str, list[object]]:
    raise AssertionError("document() must not read the store")


async def test_document_returns_what_the_listing_returns_for_that_uri(
    source: HierarchicalChunkSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = await source.documents()
    monkeypatch.setattr(source._tool, "document_chunks", _fail_store_read)

    found = [await source.document(document.source_uri) for document in listed]

    assert found == listed


async def test_document_of_a_remote_source_matches_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "doc.md").write_text(DATED)
    tool = _tool(tmp_path)
    tool.set_source_context(source_root=tmp_path, is_remote=True)
    await tool.initialize()
    remote = HierarchicalChunkSource(tool)
    [listed] = await remote.documents()
    monkeypatch.setattr(tool, "document_chunks", _fail_store_read)

    found = await remote.document("docs/doc.md")

    assert found == listed
    assert found is not None
    assert found.mtime_reliable is False


async def test_document_is_none_for_a_deleted_file(
    source: HierarchicalChunkSource, tmp_path: Path
) -> None:
    (tmp_path / "b_plain.md").unlink()

    assert await source.document(str(tmp_path / "b_plain.md")) is None
    assert await source.document(str(tmp_path / "a_dated.md")) is not None


async def test_document_of_a_preview_source_matches_its_listing(
    tmp_path: Path,
) -> None:
    (tmp_path / "a_dated.md").write_text(DATED)
    preview = HierarchicalChunkSource(_tool(tmp_path), preview=True)
    [listed] = await preview.documents()

    assert await preview.document(listed.source_uri) == listed
    assert not preview._tool._initialized
