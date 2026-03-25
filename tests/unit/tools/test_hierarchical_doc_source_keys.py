"""Tests for HierarchicalDocumentTool source key infrastructure.

Tests stable record keys and content-hash change detection for remote sources.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock HierarchicalDocumentToolConfig."""
    config = MagicMock()
    config.name = "test_doc_tool"
    config.description = "Test hierarchical doc tool"
    config.source = "./docs"
    config.search_mode = MagicMock()
    config.search_mode.value = "hybrid"
    config.semantic_weight = 0.5
    config.keyword_weight = 0.3
    config.exact_match_weight = 0.2
    config.contextual_embeddings = False
    config.domain = None
    config.top_k = 5
    config.max_tokens = None
    config.subsection_patterns = None
    return config


@pytest.fixture
def tool(mock_config: MagicMock) -> HierarchicalDocumentTool:
    """Create a HierarchicalDocumentTool instance."""
    return HierarchicalDocumentTool(mock_config)


class TestSetSourceContext:
    """Tests for set_source_context method."""

    def test_sets_source_root_and_is_remote(
        self, tool: HierarchicalDocumentTool
    ) -> None:
        root = Path("/tmp/holodeck-init-abc")  # noqa: S108
        tool.set_source_context(source_root=root, is_remote=True)
        assert tool._source_root == root
        assert tool._is_remote is True

    def test_defaults_are_local(self, tool: HierarchicalDocumentTool) -> None:
        assert tool._source_root is None
        assert tool._is_remote is False


class TestComputeSourceKey:
    """Tests for _compute_source_key method."""

    def test_local_returns_absolute_path(self, tool: HierarchicalDocumentTool) -> None:
        file_path = Path("/absolute/path/to/file.txt")
        assert tool._compute_source_key(file_path) == str(file_path)

    def test_remote_returns_relative_path(self, tool: HierarchicalDocumentTool) -> None:
        root = Path("/tmp/holodeck-init-abc")  # noqa: S108
        tool.set_source_context(source_root=root, is_remote=True)
        file_path = root / "data" / "file.txt"
        assert tool._compute_source_key(file_path) == "data/file.txt"


class TestComputeContentHash:
    """Tests for _compute_content_hash method."""

    def test_local_returns_empty_string(self, tool: HierarchicalDocumentTool) -> None:
        assert tool._compute_content_hash(Path("/any/path")) == ""

    def test_remote_returns_sha256(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = tool._compute_content_hash(test_file)
        assert len(result) == 64
        assert result.isalnum()

    def test_remote_same_content_same_hash(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("identical")
        f2.write_text("identical")
        assert tool._compute_content_hash(f1) == tool._compute_content_hash(f2)


class TestRemapChunkKeys:
    """Tests for _remap_chunk_keys method."""

    def test_remote_remaps_source_path_and_id(
        self, tool: HierarchicalDocumentTool
    ) -> None:
        root = Path("/tmp/holodeck-init-abc")  # noqa: S108
        tool.set_source_context(source_root=root, is_remote=True)

        chunk = MagicMock()
        chunk.source_path = "/tmp/holodeck-init-abc/data/file.txt"  # noqa: S108
        chunk.id = "_tmp_holodeck-init-abc_data_file.txt_chunk_0"

        tool._remap_chunk_keys([chunk], root / "data" / "file.txt")

        assert chunk.source_path == "data/file.txt"
        assert chunk.id == "data_file.txt_chunk_0"

    def test_local_is_noop(self, tool: HierarchicalDocumentTool) -> None:
        chunk = MagicMock()
        chunk.source_path = "/local/path/file.txt"
        chunk.id = "_local_path_file.txt_chunk_0"

        tool._remap_chunk_keys([chunk], Path("/local/path/file.txt"))

        assert chunk.source_path == "/local/path/file.txt"
        assert chunk.id == "_local_path_file.txt_chunk_0"


class TestNeedsReingestRemote:
    """Tests for _needs_reingest with remote sources."""

    @pytest.mark.asyncio
    async def test_remote_unchanged_content_skips(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("hello")
        content_hash = tool._compute_content_hash(test_file)

        mock_record = MagicMock()
        mock_record.content_hash = content_hash

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=[mock_record])
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_remote_changed_content_reingests(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("new content")

        mock_record = MagicMock()
        mock_record.content_hash = "old_hash"

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=[mock_record])
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_remote_no_stored_hash_reingests(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        mock_record = MagicMock()
        mock_record.content_hash = ""

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=[mock_record])
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True


class TestNeedsReingestLocal:
    """Tests for _needs_reingest with local sources (backward compat)."""

    @pytest.mark.asyncio
    async def test_local_unchanged_mtime_skips(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")
        mtime = test_file.stat().st_mtime

        mock_record = MagicMock()
        mock_record.mtime = mtime

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=[mock_record])
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_local_newer_mtime_reingests(
        self, tool: HierarchicalDocumentTool, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        mock_record = MagicMock()
        mock_record.mtime = 1000.0

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=[mock_record])
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True
