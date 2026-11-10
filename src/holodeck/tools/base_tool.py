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
            self._adapter = None
            self._provider = "in-memory"
            self._embedding_dimensions = None
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holodeck.lib.retrieval import CollectionSchema, VectorStoreAdapter
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
    """Mixin for database configuration resolution and adapter creation.

    Provides methods for resolving database configuration from various formats
    (None, string reference, DatabaseConfig object), creating the vector store
    adapter with automatic fallback to in-memory storage, and closing it.

    Required instance attributes (set by subclass __init__):
        config: Any - tool configuration with .name and .database attributes
        _provider: str - stores the resolved provider name
        _adapter: VectorStoreAdapter | None - the tool's vector store adapter
        _embedding_dimensions: int | None - embedding dimensions
    """

    _provider: str
    _adapter: VectorStoreAdapter | None
    _embedding_dimensions: int | None
    _manifest_enabled: bool = True
    """False when the database settings set ``manifest: false``."""

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
            self._manifest_enabled = database.manifest
            provider = database.provider
            connection_kwargs: dict[str, Any] = {}
            if database.connection_string:
                connection_kwargs["connection_string"] = (
                    database.connection_string.get_secret_value()
                )
            # Add extra fields from DatabaseConfig (extra="allow")
            if hasattr(database, "model_extra"):
                extra_fields = database.model_extra or {}
                connection_kwargs.update(extra_fields)
            return provider, connection_kwargs

        # None - use in-memory
        return "in-memory", {}

    def _create_adapter_with_fallback(
        self,
        provider: str,
        schema: CollectionSchema,
        connection_kwargs: Mapping[str, object],
    ) -> VectorStoreAdapter:
        """Create the vector store adapter with fallback to in-memory.

        Attempts to create the adapter for the configured provider. If that
        fails because of invalid settings or an unreachable store, falls back
        to in-memory storage. A missing client library or a retired provider
        is raised: the configuration cannot work until it is fixed.

        Args:
            provider: Vector store provider name
            schema: Collection layout (fields, dimensions, collection name)
            connection_kwargs: Provider-specific connection parameters

        Returns:
            The adapter, which the tool owns until ``close``.

        Raises:
            MissingExtraError: The provider's client library is not installed.
            RetiredProviderError: The provider name was retired.
            Exception: If the in-memory adapter also fails (shouldn't happen)
        """
        from holodeck.lib.extras import MissingExtraError
        from holodeck.lib.retrieval import RetiredProviderError, create_adapter

        try:
            adapter = create_adapter(provider, schema, connection_kwargs)
            logger.info(
                f"Vector store connected: provider={provider}, "
                f"dimensions={schema.dimensions}"
            )
            self._provider = provider
            return adapter

        except (MissingExtraError, RetiredProviderError):
            # The configuration cannot work on this install. Storing data in
            # memory in its place would hide that from the user.
            raise
        except Exception as e:
            # Fall back to in-memory storage for non-in-memory providers
            if provider != "in-memory":
                logger.warning(
                    f"Failed to connect to {provider}: {e}. "
                    "Falling back to in-memory storage."
                )
                adapter = create_adapter("in-memory", schema)
                logger.info("Using in-memory vector storage (fallback)")
                self._provider = "in-memory"
                return adapter
            # Don't catch errors for in-memory provider
            raise

    async def close(self) -> None:
        """Release the vector store client. Safe to call more than once."""
        if self._adapter is not None:
            await self._adapter.close()

    async def cleanup(self) -> None:
        """Release resources; the hook backends call on teardown."""
        await self.close()
