"""Unit tests for holodeck.tools.base_tool mixins.

Tests for mixin classes that provide common functionality for tools.

Test IDs:
- TBT001: EmbeddingServiceMixin.set_embedding_service sets service
- TBT002: DatabaseConfigMixin._resolve_database_config handles None
- TBT003: DatabaseConfigMixin._resolve_database_config handles string ref
- TBT004: DatabaseConfigMixin._resolve_database_config handles DatabaseConfig
- TBT005: DatabaseConfigMixin._create_collection_with_fallback creates collection
- TBT006: DatabaseConfigMixin._create_collection_with_fallback falls back on error
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from holodeck.tools.base_tool import DatabaseConfigMixin, EmbeddingServiceMixin


class MockConfig:
    """Mock tool configuration for testing."""

    def __init__(self, name: str = "test_tool", database: Any = None) -> None:
        self.name = name
        self.database = database


class EmbeddingTestTool(EmbeddingServiceMixin):
    """Test class using EmbeddingServiceMixin."""

    def __init__(self, name: str = "test_tool") -> None:
        self.config = MockConfig(name=name)
        self._embedding_service: Any = None


class DatabaseTestTool(DatabaseConfigMixin):
    """Test class using DatabaseConfigMixin."""

    def __init__(self, name: str = "test_tool") -> None:
        self.config = MockConfig(name=name)
        self._provider: str = "in-memory"
        self._collection: Any = None
        self._embedding_dimensions: int | None = None


class TestEmbeddingServiceMixin:
    """Tests for EmbeddingServiceMixin."""

    def test_set_embedding_service_sets_service(self) -> None:
        """TBT001: set_embedding_service sets the service attribute."""
        tool = EmbeddingTestTool()
        mock_service = MagicMock()

        tool.set_embedding_service(mock_service)

        assert tool._embedding_service is mock_service

    def test_set_embedding_service_logs_tool_name(self) -> None:
        """set_embedding_service logs the tool name."""
        tool = EmbeddingTestTool(name="my_tool")
        mock_service = MagicMock()

        with patch("holodeck.tools.base_tool.logger") as mock_logger:
            tool.set_embedding_service(mock_service)
            mock_logger.debug.assert_called_once()
            assert "my_tool" in str(mock_logger.debug.call_args)


class MockDatabaseConfig:
    """Mock DatabaseConfig for testing."""

    def __init__(
        self,
        provider: str = "postgres",
        connection_string: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.connection_string = connection_string
        self.model_extra = extra or {}


class TestDatabaseConfigMixin:
    """Tests for DatabaseConfigMixin."""

    def test_resolve_database_config_handles_none(self) -> None:
        """TBT002: _resolve_database_config handles None."""
        tool = DatabaseTestTool()

        provider, kwargs = tool._resolve_database_config(None)

        assert provider == "in-memory"
        assert kwargs == {}

    def test_resolve_database_config_handles_string_ref(self) -> None:
        """TBT003: _resolve_database_config handles string reference."""
        tool = DatabaseTestTool()

        with patch("holodeck.tools.base_tool.logger") as mock_logger:
            provider, kwargs = tool._resolve_database_config("db_reference")

            assert provider == "in-memory"
            assert kwargs == {}
            mock_logger.warning.assert_called_once()

    def test_resolve_database_config_handles_database_config(self) -> None:
        """TBT004: _resolve_database_config handles DatabaseConfig object."""
        tool = DatabaseTestTool()
        db_config = MockDatabaseConfig(
            provider="postgres",
            connection_string="postgresql://localhost/db",
            extra={"db_schema": "public"},
        )

        provider, kwargs = tool._resolve_database_config(db_config)  # type: ignore[arg-type]

        assert provider == "postgres"
        assert kwargs["connection_string"] == "postgresql://localhost/db"
        assert kwargs["db_schema"] == "public"

    def test_resolve_database_config_handles_no_connection_string(self) -> None:
        """_resolve_database_config handles DatabaseConfig without connection_string."""
        tool = DatabaseTestTool()
        db_config = MockDatabaseConfig(provider="qdrant", connection_string=None)

        provider, kwargs = tool._resolve_database_config(db_config)  # type: ignore[arg-type]

        assert provider == "qdrant"
        assert "connection_string" not in kwargs

    def test_create_collection_with_fallback_creates_collection(self) -> None:
        """TBT005: _create_collection_with_fallback creates collection."""
        tool = DatabaseTestTool()
        mock_collection = MagicMock()

        def mock_get_collection_factory(**kwargs: Any) -> MagicMock:
            return MagicMock(return_value=mock_collection)

        with patch(
            "holodeck.lib.vector_store.get_collection_factory",
            side_effect=mock_get_collection_factory,
        ):
            result = tool._create_collection_with_fallback(
                provider="in-memory",
                dimensions=1536,
                connection_kwargs={},
            )

            assert result is mock_collection
            assert tool._provider == "in-memory"

    def test_create_collection_with_fallback_falls_back_on_error(self) -> None:
        """TBT006: _create_collection_with_fallback falls back on error."""
        tool = DatabaseTestTool()
        mock_collection = MagicMock()
        call_count = 0

        def factory_side_effect(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (postgres) - return failing factory
                def failing_factory() -> None:
                    raise ConnectionError("Cannot connect")

                return MagicMock(side_effect=failing_factory)
            else:
                # Second call (in-memory) - return working factory
                return MagicMock(return_value=mock_collection)

        with patch(
            "holodeck.lib.vector_store.get_collection_factory",
            side_effect=factory_side_effect,
        ):
            result = tool._create_collection_with_fallback(
                provider="postgres",
                dimensions=1536,
                connection_kwargs={"connection_string": "postgresql://localhost"},
            )

            assert result is mock_collection
            assert tool._provider == "in-memory"

    def test_create_collection_raises_for_in_memory_failure(self) -> None:
        """_create_collection_with_fallback raises for in-memory failures."""
        tool = DatabaseTestTool()

        def failing_factory() -> None:
            raise ValueError("Unexpected error")

        def mock_get_collection_factory(**kwargs: Any) -> MagicMock:
            return MagicMock(side_effect=failing_factory)

        with (
            patch(
                "holodeck.lib.vector_store.get_collection_factory",
                side_effect=mock_get_collection_factory,
            ),
            pytest.raises(ValueError, match="Unexpected error"),
        ):
            tool._create_collection_with_fallback(
                provider="in-memory",
                dimensions=1536,
                connection_kwargs={},
            )

    def test_create_collection_passes_record_class_and_definition(self) -> None:
        """_create_collection_with_fallback passes record_class and definition."""
        tool = DatabaseTestTool()
        mock_collection = MagicMock()
        mock_record_class = MagicMock()
        mock_definition = MagicMock()

        with patch(
            "holodeck.lib.vector_store.get_collection_factory",
        ) as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_collection)

            tool._create_collection_with_fallback(
                provider="in-memory",
                dimensions=768,
                connection_kwargs={},
                record_class=mock_record_class,
                definition=mock_definition,
            )

            mock_factory.assert_called_once_with(
                provider="in-memory",
                dimensions=768,
                record_class=mock_record_class,
                definition=mock_definition,
            )
