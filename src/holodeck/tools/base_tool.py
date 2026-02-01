"""Base tool mixins for HoloDeck tools.

This module provides mixin classes that encapsulate common functionality
shared between VectorStoreTool and HierarchicalDocumentTool.

Mixins are used instead of inheritance because:
- Tools have different core behaviors (unstructured vs hierarchical)
- Record types are fundamentally different
- Allows selective reuse without forced inheritance hierarchy

Usage:
    from holodeck.tools.base_tool import (
        EmbeddingServiceMixin,
        DatabaseConfigMixin,
    )

    class MyTool(EmbeddingServiceMixin, DatabaseConfigMixin):
        def __init__(self, config):
            self.config = config
            self._embedding_service = None
            self._collection = None
            self._provider = "in-memory"
            self._embedding_dimensions = None
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.models.tool import DatabaseConfig

logger = logging.getLogger(__name__)


class EmbeddingServiceMixin:
    """Mixin for embedding service injection.

    Provides the set_embedding_service method used by AgentFactory to inject
    a Semantic Kernel TextEmbedding service for generating real embeddings.

    Required instance attributes (set by subclass __init__):
        _embedding_service: Any - stores the injected service
        config: Any - tool configuration with a .name attribute
    """

    _embedding_service: Any

    def set_embedding_service(self, service: Any) -> None:
        """Set the embedding service for generating embeddings.

        This method allows AgentFactory to inject a Semantic Kernel TextEmbedding
        service for generating real embeddings instead of placeholder zeros.

        Args:
            service: Semantic Kernel TextEmbedding service instance
                (OpenAITextEmbedding or AzureTextEmbedding).
        """
        self._embedding_service = service
        tool_name = getattr(getattr(self, "config", None), "name", "unknown")
        logger.debug(f"Embedding service set for tool: {tool_name}")


class DatabaseConfigMixin:
    """Mixin for database configuration resolution and collection creation.

    Provides methods for resolving database configuration from various formats
    (None, string reference, DatabaseConfig object) and creating vector store
    collections with automatic fallback to in-memory storage.

    Required instance attributes (set by subclass __init__):
        config: Any - tool configuration with .name and .database attributes
        _provider: str - stores the resolved provider name
        _collection: Any - stores the created collection instance
        _embedding_dimensions: int | None - embedding dimensions
    """

    _provider: str
    _collection: Any
    _embedding_dimensions: int | None

    def _resolve_database_config(
        self, database: DatabaseConfig | str | None
    ) -> tuple[str, dict[str, Any]]:
        """Resolve database configuration to provider and connection kwargs.

        Handles three types of database configuration:
        1. None - use in-memory storage
        2. String reference - unresolved reference, warn and use in-memory
        3. DatabaseConfig object - extract provider and connection parameters

        Args:
            database: Database configuration (from tool config)

        Returns:
            Tuple of (provider_name, connection_kwargs)

        Example:
            >>> provider, kwargs = self._resolve_database_config(None)
            >>> provider
            'in-memory'
            >>> kwargs
            {}
        """
        tool_name = getattr(getattr(self, "config", None), "name", "unknown")

        if isinstance(database, str):
            # Unresolved string reference - this shouldn't happen if merge_configs
            # was called, but fall back to in-memory with a warning
            logger.warning(
                f"Tool '{tool_name}' has unresolved database "
                f"reference '{database}'. Falling back to in-memory storage."
            )
            return "in-memory", {}

        if database is not None:
            # DatabaseConfig object - use its settings
            provider = database.provider
            connection_kwargs: dict[str, Any] = {}
            if database.connection_string:
                connection_kwargs["connection_string"] = database.connection_string
            # Add extra fields from DatabaseConfig (extra="allow")
            if hasattr(database, "model_extra"):
                extra_fields = database.model_extra or {}
                connection_kwargs.update(extra_fields)
            return provider, connection_kwargs

        # None - use in-memory
        return "in-memory", {}

    def _create_collection_with_fallback(
        self,
        provider: str,
        dimensions: int,
        connection_kwargs: dict[str, Any],
        record_class: type[Any] | None = None,
        definition: Any | None = None,
    ) -> Any:
        """Create a vector store collection with fallback to in-memory.

        Attempts to create a collection with the specified provider. If creation
        fails (e.g., database unreachable), falls back to in-memory storage.

        Args:
            provider: Vector store provider name
            dimensions: Embedding dimensions for the collection
            connection_kwargs: Provider-specific connection parameters
            record_class: Optional custom record class for the collection
            definition: Optional VectorStoreCollectionDefinition for structured data

        Returns:
            Created collection instance

        Raises:
            Exception: If in-memory provider also fails (shouldn't happen)
        """
        from holodeck.lib.vector_store import get_collection_factory

        try:
            factory = get_collection_factory(
                provider=provider,
                dimensions=dimensions,
                record_class=record_class,
                definition=definition,
                **connection_kwargs,
            )
            collection = factory()
            logger.info(
                f"Vector store connected: provider={provider}, "
                f"dimensions={dimensions}"
            )
            self._provider = provider
            return collection

        except (ImportError, ConnectionError, Exception) as e:
            # Fall back to in-memory storage for non-in-memory providers
            if provider != "in-memory":
                logger.warning(
                    f"Failed to connect to {provider}: {e}. "
                    "Falling back to in-memory storage."
                )
                factory = get_collection_factory(
                    provider="in-memory",
                    dimensions=dimensions,
                    record_class=record_class,
                    definition=definition,
                )
                collection = factory()
                logger.info("Using in-memory vector storage (fallback)")
                self._provider = "in-memory"
                return collection
            else:
                # Don't catch errors for in-memory provider
                raise
