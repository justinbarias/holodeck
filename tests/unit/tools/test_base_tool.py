"""Unit tests for holodeck.tools.base_tool mixins.

Tests for mixin classes that provide common functionality for tools.

Test IDs:
- TBT001: EmbeddingServiceMixin.set_embedding_service sets service
- TBT002: DatabaseConfigMixin._resolve_database_config handles None
- TBT003: DatabaseConfigMixin._resolve_database_config handles string ref
- TBT004: DatabaseConfigMixin._resolve_database_config handles DatabaseConfig
- TBT005: DatabaseConfigMixin._create_adapter_with_fallback creates the adapter
- TBT006: DatabaseConfigMixin._create_adapter_with_fallback falls back on error
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from holodeck.lib.retrieval import CollectionSchema, FieldSpec, VectorStoreAdapter
from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter
from holodeck.tools.base_tool import DatabaseConfigMixin, EmbeddingServiceMixin
from tests.unit.tools.fakes import FakeAdapter


def _schema() -> CollectionSchema:
    return CollectionSchema("docs", 4, (FieldSpec("content", full_text=True),))


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
        self._adapter: VectorStoreAdapter | None = None
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
    """Mock DatabaseConfig for testing.

    Mirrors the real `DatabaseConfig.connection_string: SecretStr | None` shape
    so the mixin's `.get_secret_value()` reads work identically under test.
    """

    def __init__(
        self,
        provider: str = "postgres",
        connection_string: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.connection_string: SecretStr | None = (
            SecretStr(connection_string) if connection_string is not None else None
        )
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

    def test_create_adapter_with_fallback_creates_adapter(self) -> None:
        """TBT005: _create_adapter_with_fallback creates the provider's adapter."""
        tool = DatabaseTestTool()

        result = tool._create_adapter_with_fallback(
            provider="in-memory", schema=_schema(), connection_kwargs={}
        )

        assert isinstance(result, InMemoryAdapter)
        assert result.schema == _schema()
        assert tool._provider == "in-memory"

    def test_create_adapter_with_fallback_falls_back_on_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TBT006: _create_adapter_with_fallback falls back on error."""
        tool = DatabaseTestTool()
        fallback = InMemoryAdapter(_schema())
        calls: list[tuple[str, dict[str, object]]] = []

        def create(
            provider: str,
            schema: CollectionSchema,
            connection: dict[str, object] | None = None,
        ) -> InMemoryAdapter:
            calls.append((provider, dict(connection or {})))
            if provider == "postgres":
                raise ConnectionError("Cannot connect")
            return fallback

        with patch("holodeck.lib.retrieval.create_adapter", side_effect=create):
            result = tool._create_adapter_with_fallback(
                provider="postgres",
                schema=_schema(),
                connection_kwargs={"connection_string": "postgresql://localhost"},
            )

        assert result is fallback
        assert tool._provider == "in-memory"
        assert calls == [
            ("postgres", {"connection_string": "postgresql://localhost"}),
            ("in-memory", {}),
        ]
        assert "Failed to connect to postgres: Cannot connect" in caplog.text

    def test_unknown_provider_falls_back_to_in_memory(self) -> None:
        """An unknown provider falls back, as get_collection_factory's error did."""
        tool = DatabaseTestTool()

        result = tool._create_adapter_with_fallback(
            provider="not-a-store", schema=_schema(), connection_kwargs={}
        )

        assert isinstance(result, InMemoryAdapter)
        assert tool._provider == "in-memory"

    def test_create_adapter_raises_for_in_memory_failure(self) -> None:
        """_create_adapter_with_fallback raises for in-memory failures."""
        tool = DatabaseTestTool()

        with (
            patch(
                "holodeck.lib.retrieval.create_adapter",
                side_effect=ValueError("Unexpected error"),
            ),
            pytest.raises(ValueError, match="Unexpected error"),
        ):
            tool._create_adapter_with_fallback(
                provider="in-memory", schema=_schema(), connection_kwargs={}
            )

    @pytest.mark.asyncio
    async def test_close_closes_the_adapter(self) -> None:
        """close() and the backends' cleanup() hook release the adapter."""
        tool = DatabaseTestTool()
        adapter = FakeAdapter(_schema())
        tool._adapter = adapter

        await tool.close()
        await tool.cleanup()

        assert adapter.close_calls == 2

    @pytest.mark.asyncio
    async def test_close_without_adapter_is_a_no_op(self) -> None:
        """close() before setup does nothing."""
        tool = DatabaseTestTool()

        await tool.close()

        assert tool._adapter is None
