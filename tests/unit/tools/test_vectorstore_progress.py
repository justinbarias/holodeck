"""Tests for VectorStoreTool progress_callback support (T014)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.vectorstore_tool import VectorStoreTool


def _make_tool(tmp_path: Path) -> VectorStoreTool:
    """Create a VectorStoreTool with a minimal config pointing at tmp_path."""
    source_dir = tmp_path / "data"
    source_dir.mkdir(exist_ok=True)

    config = VectorstoreToolConfig(
        name="test_tool",
        description="test tool",
        type="vectorstore",
        source=str(source_dir),
    )
    return VectorStoreTool(config, base_dir=str(tmp_path))


class TestVectorStoreProgressCallback:
    """Test that initialize() invokes the progress_callback per file."""

    @pytest.mark.asyncio
    async def test_initialize_calls_progress_callback_per_file(
        self, tmp_path: Path
    ) -> None:
        """Callback receives (1,3), (2,3), (3,3) for three files."""
        tool = _make_tool(tmp_path)

        fake_files = [
            tmp_path / "data" / "a.txt",
            tmp_path / "data" / "b.txt",
            tmp_path / "data" / "c.txt",
        ]

        tool._resolve_source_path = MagicMock(return_value=tmp_path / "data")
        tool._discover_files = MagicMock(return_value=fake_files)
        tool._setup_collection = MagicMock()
        tool._needs_reingest = AsyncMock(return_value=True)
        tool._process_file = AsyncMock(return_value=MagicMock(chunks=["chunk1"]))
        tool._embed_chunks = AsyncMock(return_value=[[0.1, 0.2]])
        tool._store_chunks = AsyncMock(return_value=1)

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

        fake_files = [tmp_path / "data" / "a.txt"]

        tool._resolve_source_path = MagicMock(return_value=tmp_path / "data")
        tool._discover_files = MagicMock(return_value=fake_files)
        tool._setup_collection = MagicMock()
        tool._needs_reingest = AsyncMock(return_value=True)
        tool._process_file = AsyncMock(return_value=MagicMock(chunks=["chunk1"]))
        tool._embed_chunks = AsyncMock(return_value=[[0.1, 0.2]])
        tool._store_chunks = AsyncMock(return_value=1)

        # Should not raise
        await tool.initialize(force_ingest=True, provider_type="openai")

    @pytest.mark.asyncio
    async def test_initialize_callback_with_skipped_files(self, tmp_path: Path) -> None:
        """Callback is invoked even for skipped (up-to-date) files."""
        tool = _make_tool(tmp_path)

        fake_files = [
            tmp_path / "data" / "a.txt",
            tmp_path / "data" / "b.txt",
            tmp_path / "data" / "c.txt",
        ]

        tool._resolve_source_path = MagicMock(return_value=tmp_path / "data")
        tool._discover_files = MagicMock(return_value=fake_files)
        tool._setup_collection = MagicMock()

        # Second file is up-to-date → skipped
        tool._needs_reingest = AsyncMock(side_effect=[True, False, True])
        tool._process_file = AsyncMock(return_value=MagicMock(chunks=["chunk1"]))
        tool._embed_chunks = AsyncMock(return_value=[[0.1, 0.2]])
        tool._store_chunks = AsyncMock(return_value=1)

        callback = MagicMock()

        await tool.initialize(
            force_ingest=False,
            provider_type="openai",
            progress_callback=callback,
        )

        assert callback.call_count == 3
        callback.assert_any_call(1, 3)
        callback.assert_any_call(2, 3)
        callback.assert_any_call(3, 3)
