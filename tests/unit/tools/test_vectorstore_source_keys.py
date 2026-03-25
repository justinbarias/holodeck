"""Tests for VectorStoreTool source key infrastructure.

Tests stable record keys and content-hash change detection for remote sources.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.tools.vectorstore_tool import VectorStoreTool


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock VectorstoreToolConfig."""
    config = MagicMock()
    config.name = "test_tool"
    config.description = "Test tool"
    config.source = "./data"
    config.top_k = 5
    config.min_similarity_score = None
    config.chunk_size = None
    config.chunk_overlap = None
    config.vector_field = None
    return config


@pytest.fixture
def tool(mock_config: MagicMock) -> VectorStoreTool:
    """Create a VectorStoreTool instance."""
    return VectorStoreTool(mock_config)


class TestSetSourceContext:
    """Tests for set_source_context method."""

    def test_sets_source_root_and_is_remote(self, tool: VectorStoreTool) -> None:
        root = Path("/tmp/holodeck-init-abc")  # noqa: S108
        tool.set_source_context(source_root=root, is_remote=True)
        assert tool._source_root == root
        assert tool._is_remote is True

    def test_defaults_are_local(self, tool: VectorStoreTool) -> None:
        assert tool._source_root is None
        assert tool._is_remote is False


class TestComputeSourceKey:
    """Tests for _compute_source_key method."""

    def test_local_returns_absolute_path(self, tool: VectorStoreTool) -> None:
        file_path = Path("/absolute/path/to/file.txt")
        assert tool._compute_source_key(file_path) == str(file_path)

    def test_remote_returns_relative_path(self, tool: VectorStoreTool) -> None:
        root = Path("/tmp/holodeck-init-abc")  # noqa: S108
        tool.set_source_context(source_root=root, is_remote=True)
        file_path = root / "data" / "file.txt"
        assert tool._compute_source_key(file_path) == "data/file.txt"

    def test_remote_without_root_returns_absolute(self, tool: VectorStoreTool) -> None:
        """If _is_remote but no root set (shouldn't happen), falls back to absolute."""
        tool._is_remote = True
        file_path = Path("/some/path/file.txt")
        assert tool._compute_source_key(file_path) == str(file_path)


class TestComputeContentHash:
    """Tests for _compute_content_hash method."""

    def test_local_returns_empty_string(self, tool: VectorStoreTool) -> None:
        assert tool._compute_content_hash(Path("/any/path")) == ""

    def test_remote_returns_sha256(self, tool: VectorStoreTool, tmp_path: Path) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = tool._compute_content_hash(test_file)
        assert len(result) == 64  # SHA-256 hex digest
        assert result.isalnum()

    def test_remote_same_content_same_hash(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("identical content")
        f2.write_text("identical content")
        assert tool._compute_content_hash(f1) == tool._compute_content_hash(f2)

    def test_remote_different_content_different_hash(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert tool._compute_content_hash(f1) != tool._compute_content_hash(f2)


class TestNeedsReingestRemote:
    """Tests for _needs_reingest with remote sources."""

    @pytest.mark.asyncio
    async def test_remote_unchanged_content_skips(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        """Same content hash -> returns False (skip reingest)."""
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("hello")
        content_hash = tool._compute_content_hash(test_file)

        mock_record = MagicMock()
        mock_record.content_hash = content_hash

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_remote_changed_content_reingests(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        """Different content hash -> returns True (reingest)."""
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("new content")

        mock_record = MagicMock()
        mock_record.content_hash = "old_hash_value"

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_remote_no_stored_hash_reingests(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        """Old record without content_hash -> returns True."""
        tool.set_source_context(source_root=tmp_path, is_remote=True)
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        mock_record = MagicMock()
        mock_record.content_hash = ""

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True

    @pytest.mark.asyncio
    async def test_remote_uses_relative_key_for_lookup(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        """Verify the record ID uses the relative source key, not absolute temp path."""
        root = tmp_path / "source"
        root.mkdir()
        test_file = root / "data" / "file.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")

        tool.set_source_context(source_root=root, is_remote=True)

        mock_record = MagicMock()
        mock_record.content_hash = tool._compute_content_hash(test_file)

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        await tool._needs_reingest(test_file)

        # Verify the lookup used relative key
        mock_collection.get.assert_called_once_with("data/file.txt_chunk_0")


class TestNeedsReingestLocal:
    """Tests for _needs_reingest with local sources (backward compat)."""

    @pytest.mark.asyncio
    async def test_local_unchanged_mtime_skips(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")
        mtime = round(test_file.stat().st_mtime, 6)

        mock_record = MagicMock()
        mock_record.mtime = mtime

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is False

    @pytest.mark.asyncio
    async def test_local_newer_mtime_reingests(
        self, tool: VectorStoreTool, tmp_path: Path
    ) -> None:
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        mock_record = MagicMock()
        mock_record.mtime = 1000.0  # old mtime

        mock_collection = AsyncMock()
        mock_collection.get = AsyncMock(return_value=mock_record)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_collection)
        mock_collection.__aexit__ = AsyncMock(return_value=False)
        tool._collection = mock_collection

        result = await tool._needs_reingest(test_file)
        assert result is True
