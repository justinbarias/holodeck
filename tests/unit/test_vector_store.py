"""Unit tests for VectorStore module.

Tests for vector store abstractions, connection handling, and Redis fallback.

Test IDs:
- T055: Redis connection fallback to in-memory on connection failure
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock semantic_kernel modules BEFORE importing holodeck modules
for module_name in [
    "semantic_kernel",
    "semantic_kernel.connectors",
    "semantic_kernel.connectors.memory",
    "semantic_kernel.connectors.ai",
    "semantic_kernel.contents",
    "semantic_kernel.data",
    "semantic_kernel.data.vector",
    "semantic_kernel.text",
    "semantic_kernel.connectors.in_memory",
    "semantic_kernel.connectors.redis",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

# Set up specific mock attributes
mock_memory = sys.modules["semantic_kernel.connectors.memory"]
mock_memory.InMemoryCollection = MagicMock()
mock_memory.RedisHashsetCollection = MagicMock()
mock_memory.RedisJsonCollection = MagicMock()

mock_vector = sys.modules["semantic_kernel.data.vector"]
mock_vector.VectorStoreField = MagicMock()
mock_vector.vectorstoremodel = lambda **kwargs: lambda cls: cls

mock_in_memory = sys.modules["semantic_kernel.connectors.in_memory"]
mock_in_memory.InMemoryCollection = MagicMock()

mock_redis = sys.modules["semantic_kernel.connectors.redis"]
mock_redis.RedisHashsetCollection = MagicMock()
mock_redis.RedisJsonCollection = MagicMock()


class TestRedisConnectionFallback:
    """T055: Tests for Redis connection fallback to in-memory."""

    def test_fallback_on_redis_connection_error(self, tmp_path: Path) -> None:
        """Test fallback to in-memory when Redis connection fails."""
        from holodeck.models.tool import DatabaseConfig, VectorstoreTool

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(
                provider="redis-json",
                connection_string="redis://invalid:6379",
            ),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_class to raise ConnectionError for redis
        def mock_get_collection_class(provider):
            if provider.startswith("redis"):
                raise ConnectionError("Failed to connect to Redis")
            return MagicMock(return_value=MagicMock())

        with patch(
            "holodeck.tools.vectorstore_tool.get_collection_class",
            side_effect=mock_get_collection_class,
        ):
            tool._setup_collection()

        # Should have fallen back to in-memory
        assert tool._provider == "in-memory"
        assert tool._collection is not None

    def test_fallback_logs_warning(self, tmp_path: Path) -> None:
        """Test that fallback logs appropriate warning message."""
        from holodeck.models.tool import DatabaseConfig, VectorstoreTool

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(
                provider="redis-json",
                connection_string="redis://invalid:6379",
            ),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_class to raise ConnectionError for redis
        def mock_get_collection_class(provider):
            if provider.startswith("redis"):
                raise ConnectionError("Failed to connect to Redis")
            return MagicMock(return_value=MagicMock())

        # Mock the logger to verify warning is called
        mock_logger = MagicMock()
        with (
            patch(
                "holodeck.tools.vectorstore_tool.get_collection_class",
                side_effect=mock_get_collection_class,
            ),
            patch("holodeck.tools.vectorstore_tool.logger", mock_logger),
        ):
            tool._setup_collection()

        # Should have logged a warning about fallback
        mock_logger.warning.assert_called()
        # Check that any warning call contains "Falling back"
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("falling back" in call.lower() for call in warning_calls)

    def test_no_fallback_when_redis_connects(self, tmp_path: Path) -> None:
        """Test no fallback when Redis connects successfully."""
        from holodeck.models.tool import DatabaseConfig, VectorstoreTool

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(
                provider="redis-json",
                connection_string="redis://localhost:6379",
            ),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_class to succeed for redis
        mock_collection = MagicMock()
        mock_collection_class = MagicMock(return_value=mock_collection)

        with patch(
            "holodeck.tools.vectorstore_tool.get_collection_class",
            return_value=mock_collection_class,
        ):
            tool._setup_collection()

        # Should NOT have fallen back
        assert tool._provider == "redis-json"
        assert tool._collection is mock_collection

    def test_fallback_on_import_error(self, tmp_path: Path) -> None:
        """Test fallback when Redis dependency is not installed."""
        from holodeck.models.tool import DatabaseConfig, VectorstoreTool

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            database=DatabaseConfig(
                provider="redis-hashset",
            ),
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_class to raise ImportError for redis
        def mock_get_collection_class(provider):
            if provider.startswith("redis"):
                raise ImportError("Missing redis dependency")
            return MagicMock(return_value=MagicMock())

        with patch(
            "holodeck.tools.vectorstore_tool.get_collection_class",
            side_effect=mock_get_collection_class,
        ):
            tool._setup_collection()

        # Should have fallen back to in-memory
        assert tool._provider == "in-memory"

    def test_no_fallback_for_in_memory_errors(self, tmp_path: Path) -> None:
        """Test that errors during in-memory setup are raised, not swallowed."""
        from holodeck.models.tool import VectorstoreTool

        source_file = tmp_path / "test.md"
        source_file.write_text("# Test")

        config = VectorstoreTool(
            name="test_vectorstore",
            description="Test tool",
            source=str(source_file),
            # No database config = in-memory by default
        )

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        tool = VectorStoreTool(config)

        # Mock get_collection_class to raise for in-memory
        def mock_get_collection_class(provider):
            raise RuntimeError("In-memory collection failed")

        with (
            patch(
                "holodeck.tools.vectorstore_tool.get_collection_class",
                side_effect=mock_get_collection_class,
            ),
            pytest.raises(RuntimeError, match="In-memory collection failed"),
        ):
            tool._setup_collection()


class TestGetCollectionClass:
    """Tests for get_collection_class factory function."""

    def test_unsupported_provider_raises_value_error(self) -> None:
        """Test that unsupported provider raises ValueError."""
        from holodeck.lib.vector_store import get_collection_class

        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            get_collection_class("unsupported-provider")

    def test_supported_providers_list(self) -> None:
        """Test that all documented providers are recognized."""
        from holodeck.lib.vector_store import get_collection_class

        supported = [
            "redis-hashset",
            "redis-json",
            "postgres",
            "azure-ai-search",
            "qdrant",
            "weaviate",
            "chromadb",
            "faiss",
            "azure-cosmos-mongo",
            "azure-cosmos-nosql",
            "sql-server",
            "pinecone",
            "in-memory",
        ]

        # Should not raise for any supported provider
        # (may raise ImportError for missing deps, but not ValueError)
        for provider in supported:
            try:
                get_collection_class(provider)
            except ImportError:
                pass  # Expected when optional deps not installed
            except ValueError:
                pytest.fail(f"Provider {provider} raised ValueError unexpectedly")


class TestDocumentRecordMtime:
    """Tests for DocumentRecord mtime field."""

    def test_document_record_has_mtime_field(self) -> None:
        """Test that DocumentRecord has mtime field."""
        from holodeck.lib.vector_store import DocumentRecord

        # Create a record with mtime
        record = DocumentRecord(
            id="test_id",
            source_path="/path/to/file.md",
            chunk_index=0,
            content="Test content",
            embedding=[0.1] * 1536,
            mtime=1234567890.5,
            file_type=".md",
            file_size_bytes=100,
        )

        assert record.mtime == 1234567890.5

    def test_document_record_mtime_defaults_to_zero(self) -> None:
        """Test that DocumentRecord mtime defaults to 0.0."""
        from holodeck.lib.vector_store import DocumentRecord

        record = DocumentRecord(
            id="test_id",
            source_path="/path/to/file.md",
            chunk_index=0,
            content="Test content",
        )

        assert record.mtime == 0.0
