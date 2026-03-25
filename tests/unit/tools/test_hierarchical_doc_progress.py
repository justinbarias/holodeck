"""Tests for HierarchicalDocumentTool progress_callback support (T015)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.models.tool import HierarchicalDocumentToolConfig
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool


def _make_tool(tmp_path: Path) -> HierarchicalDocumentTool:
    """Create a HierarchicalDocumentTool with a minimal config."""
    source_dir = tmp_path / "docs"
    source_dir.mkdir(exist_ok=True)

    config = HierarchicalDocumentToolConfig(
        name="test_hdoc",
        description="test hierarchical doc tool",
        type="hierarchical_document",
        source=str(source_dir),
    )
    return HierarchicalDocumentTool(config, base_dir=str(tmp_path))


class TestHierarchicalDocProgressCallback:
    """Test that initialize() invokes the progress_callback per file."""

    @pytest.mark.asyncio
    async def test_initialize_calls_progress_callback_per_file(
        self, tmp_path: Path
    ) -> None:
        """Callback receives (1,3), (2,3), (3,3) for three files."""
        tool = _make_tool(tmp_path)

        fake_files = [
            tmp_path / "docs" / "a.md",
            tmp_path / "docs" / "b.md",
            tmp_path / "docs" / "c.md",
        ]
        # Create real files so stat() works
        for f in fake_files:
            f.write_text("# Heading\nSome content here.")

        # Mock internals of _ingest_documents
        tool._resolve_source_path = MagicMock(return_value=tmp_path / "docs")
        tool._discover_files = MagicMock(return_value=fake_files)
        tool._setup_collection = MagicMock()
        tool._needs_reingest = AsyncMock(return_value=True)
        tool._delete_file_records = AsyncMock()
        tool._convert_to_markdown = AsyncMock(return_value="# Heading\nContent")

        # Mock the chunker
        fake_chunk = MagicMock()
        fake_chunk.chunk_type = MagicMock()
        fake_chunk.chunk_type.value = "section"
        fake_chunk.content = "Content"
        fake_chunk.contextualized_content = "Content"

        # Make chunk_type != ChunkType.HEADER
        from holodeck.lib.structured_chunker import ChunkType

        fake_chunk.chunk_type = ChunkType.CONTENT

        tool._chunker = MagicMock()
        tool._chunker.parse = MagicMock(return_value=[fake_chunk])

        tool._embed_chunks = AsyncMock()
        tool._store_chunks = AsyncMock()
        tool._build_hybrid_indices = AsyncMock()

        callback = MagicMock()

        await tool.initialize(
            force_ingest=True,
            provider_type="openai",
            progress_callback=callback,
        )

        assert callback.call_count == 3
        callback.assert_any_call(1, 3)
        callback.assert_any_call(2, 3)
        callback.assert_any_call(3, 3)

    @pytest.mark.asyncio
    async def test_initialize_no_callback_works(self, tmp_path: Path) -> None:
        """initialize() with no progress_callback (default None) succeeds."""
        tool = _make_tool(tmp_path)

        fake_files = [tmp_path / "docs" / "a.md"]
        for f in fake_files:
            f.write_text("# Heading\nContent")

        tool._resolve_source_path = MagicMock(return_value=tmp_path / "docs")
        tool._discover_files = MagicMock(return_value=fake_files)
        tool._setup_collection = MagicMock()
        tool._needs_reingest = AsyncMock(return_value=True)
        tool._delete_file_records = AsyncMock()
        tool._convert_to_markdown = AsyncMock(return_value="# Heading\nContent")

        from holodeck.lib.structured_chunker import ChunkType

        fake_chunk = MagicMock()
        fake_chunk.chunk_type = ChunkType.CONTENT
        fake_chunk.content = "Content"
        fake_chunk.contextualized_content = "Content"

        tool._chunker = MagicMock()
        tool._chunker.parse = MagicMock(return_value=[fake_chunk])

        tool._embed_chunks = AsyncMock()
        tool._store_chunks = AsyncMock()
        tool._build_hybrid_indices = AsyncMock()

        # Should not raise
        await tool.initialize(force_ingest=True, provider_type="openai")
