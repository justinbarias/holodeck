"""Pipeline fingerprint and completion markers for the retrieval tools.

``PipelineGuardMixin`` binds a tool to the physical collection that serves its
logical collection name and to the ``IngestionGuard`` that decides whether the
tool may write (see ``holodeck.lib.retrieval.guard``). Both
``VectorStoreTool`` and ``HierarchicalDocumentTool`` use it, so the
logical-to-physical indirection lives in one place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from holodeck.lib.retrieval.fingerprint import PipelineInputs
from holodeck.lib.retrieval.guard import (
    IngestionGuard,
    bind_collection,
    physical_connection,
    physical_schema,
    resolve_physical,
)
from holodeck.lib.retrieval.manifest import ManifestStore

if TYPE_CHECKING:
    from holodeck.lib.retrieval import CollectionSchema, VectorStoreAdapter

logger = logging.getLogger(__name__)

UNSPECIFIED = "unspecified"
PLACEHOLDER = "placeholder"


class PipelineGuardMixin:
    """Fingerprint check, logical-to-physical names and completion markers.

    The tool sets ``_collection_schema`` and ``_connection`` when it builds its
    adapter, implements ``pipeline_inputs`` and calls ``_bind_pipeline``
    before its first write.
    """

    _adapter: VectorStoreAdapter | None
    _provider: str
    _embedding_service: object
    _collection_schema: CollectionSchema | None = None
    _connection: dict[str, object] | None = None
    _manifest: ManifestStore | None = None
    _guard: IngestionGuard | None = None
    _embedding_identity: tuple[str, str] | None = None
    _migration_target: str | None = None

    def set_embedding_identity(self, provider: str, model: str) -> None:
        """Record the embedding provider and model the injected service uses.

        Both are part of the pipeline fingerprint. ``tool_initializer`` sets
        them from the agent configuration.
        """
        self._embedding_identity = (provider, model)

    def set_migration_target(self, physical: str) -> None:
        """Ingest into ``physical`` for ``holodeck vectorstore migrate``.

        The collection entry is left to the migration; completion markers of
        ``physical`` still apply, so an interrupted migration resumes.
        """
        self._migration_target = physical

    def embedding_identity(self) -> tuple[str, str]:
        """Return ``(provider, model)`` of the embeddings this tool stores."""
        if self._embedding_identity is not None:
            return self._embedding_identity
        if self._embedding_service is None:
            return (PLACEHOLDER, PLACEHOLDER)
        return (UNSPECIFIED, UNSPECIFIED)

    def pipeline_inputs(self) -> PipelineInputs:
        """Return the inputs of the tool's current ingestion pipeline."""
        raise NotImplementedError

    @property
    def logical_collection(self) -> str:
        """Logical collection name (the name the tool's schema gives)."""
        if self._collection_schema is None:
            raise RuntimeError("Collection not initialized")
        return self._collection_schema.name

    @property
    def collection_provider(self) -> str:
        """Vector store provider the tool's adapter uses."""
        return self._provider

    @property
    def collection_connection(self) -> dict[str, object]:
        """Connection settings of the logical collection."""
        return dict(self._connection or {})

    @property
    def physical_collection(self) -> str:
        """Physical collection the tool reads and writes."""
        if self._adapter is None:
            raise RuntimeError("Collection not initialized")
        return self._adapter.schema.name

    def _pipeline_guard(self) -> IngestionGuard:
        """Return the guard. A tool used before ``_bind_pipeline`` gets one
        without a manifest, so it keeps the behavior it had before."""
        if self._guard is None:
            self._guard = IngestionGuard(
                manifest=None,
                logical=self._adapter.schema.name if self._adapter else "",
                physical=self._adapter.schema.name if self._adapter else "",
            )
        return self._guard

    async def _bind_pipeline(self, tool_name: str) -> IngestionGuard:
        """Resolve the physical collection and apply the fingerprint rules.

        Call after the adapter for the logical collection is built and
        before any write.
        """
        from holodeck.lib.retrieval import create_adapter

        adapter = self._adapter
        schema = self._collection_schema
        if adapter is None or schema is None:
            # No collection was set up (a stubbed setup): nothing to protect.
            self._guard = None
            return self._pipeline_guard()
        connection = self._connection or {}
        logical = schema.name

        if self._manifest is None:
            self._manifest = ManifestStore.open(self._provider, connection)
        manifest = self._manifest

        physical = self._migration_target or await resolve_physical(manifest, logical)
        if physical != adapter.schema.name:
            await adapter.close()
            adapter = create_adapter(
                self._provider,
                physical_schema(schema, physical),
                physical_connection(connection, logical, physical),
            )
            self._adapter = adapter
            logger.info(
                "Collection '%s' is served by physical collection '%s'",
                logical,
                physical,
            )

        self._guard = await bind_collection(
            manifest,
            adapter,
            logical,
            self.pipeline_inputs(),
            tool_name=tool_name,
            migration_target=self._migration_target is not None,
        )
        return self._guard

    async def close(self) -> None:
        """Release the manifest and vector store clients."""
        if self._manifest is not None:
            try:
                await self._manifest.close()
            finally:
                self._manifest = None
        if self._adapter is not None:
            await self._adapter.close()

    async def cleanup(self) -> None:
        """Release resources; the hook backends call on teardown."""
        await self.close()
