"""Pipeline checks a retrieval tool runs before it writes (FR-016, FR-023).

``bind_collection`` is the one entry point both tools use. It:

1. resolves the logical collection name to the physical collection that
   serves it (``resolve_physical``);
2. compares the stored pipeline fingerprint with the current one and decides
   whether the tool may write;
3. loads the per-source completion flags, so an ingestion that stopped midway
   is redone on the next start.

Rules, applied before any write:

* no manifest entry, collection absent or empty: record the current
  fingerprint and write;
* no manifest entry, collection holds records: a legacy collection, created
  before fingerprints. It stays readable and writable; the entry is recorded
  with the current inputs marked ``legacy``. Each source that changes is
  replaced as a whole, so a source never mixes old and new chunks;
* entry with the same fingerprint: write;
* entry with a different fingerprint while the collection holds records:
  reads work, every write raises ``IncompatibleCollectionError``;
* entry with a different fingerprint and an absent or empty collection:
  nothing can be mixed, so the entry is updated to the current fingerprint.

``force_ingest`` does not change these rules.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from holodeck.lib.retrieval.adapter import VectorStoreAdapter
from holodeck.lib.retrieval.errors import IncompatibleCollectionError
from holodeck.lib.retrieval.fingerprint import (
    PipelineInputs,
    diff_inputs,
    format_diff,
    is_legacy,
    legacy_inputs_json,
    parse_inputs,
)
from holodeck.lib.retrieval.manifest import (
    COLLECTION_NAME_KEYS,
    CollectionEntry,
    ManifestStore,
    SourceEntry,
)
from holodeck.lib.retrieval.types import CollectionSchema, Operation, RecordFilter

logger = logging.getLogger(__name__)

INSPECT_COMMAND = "holodeck vectorstore inspect"
MIGRATE_COMMAND = "holodeck vectorstore migrate"


async def resolve_physical(manifest: ManifestStore | None, logical: str) -> str:
    """Return the physical collection that serves ``logical``.

    Without a manifest entry the physical name is the logical name, which is
    the name collections had before migrations existed.
    """
    if manifest is None:
        return logical
    entry = await manifest.get_collection(logical)
    if entry is None or not entry.active_collection:
        return logical
    return entry.active_collection


def physical_schema(schema: CollectionSchema, physical: str) -> CollectionSchema:
    """Return ``schema`` named ``physical``."""
    if schema.name == physical:
        return schema
    return dataclasses.replace(schema, name=physical)


def physical_connection(
    connection: Mapping[str, object], logical: str, physical: str
) -> dict[str, object]:
    """Return the connection settings that address ``physical``.

    A ``collection_name`` or ``index_name`` setting names the logical
    collection. A migrated physical collection has its own name, so those
    keys are dropped for it.
    """
    if physical == logical:
        return dict(connection)
    return {
        key: value
        for key, value in connection.items()
        if key not in COLLECTION_NAME_KEYS
    }


async def collection_has_records(adapter: VectorStoreAdapter) -> bool:
    """Return True when the collection exists and holds at least one record."""
    if not await adapter.collection_exists():
        return False
    capabilities = adapter.capabilities
    if capabilities.supports(Operation.COUNT):
        return await adapter.count() > 0
    if capabilities.supports(Operation.LIST):
        return bool((await adapter.list_page(limit=1)).records)
    if capabilities.supports(Operation.FIND):
        return bool(await adapter.find(RecordFilter(), limit=1))
    return True  # Cannot tell: assume records, which is the cautious answer.


def refusal_message(
    logical: str,
    physical: str,
    stored_inputs: str,
    current: PipelineInputs,
    tool_hint: str = "",
) -> str:
    """Return the text of an ``IncompatibleCollectionError``."""
    differences = diff_inputs(parse_inputs(stored_inputs), current.as_dict())
    lines = format_diff(differences) or "  (stored inputs unavailable)"
    target = (
        f" AGENT_YAML --tool {tool_hint}" if tool_hint else " AGENT_YAML --tool NAME"
    )
    return (
        f"Collection '{physical}' (logical name '{logical}') was written by a "
        f"different ingestion pipeline. Writing to it would mix incompatible "
        f"records, so HoloDeck refuses every write; searches still read it.\n"
        f"Inputs that differ (stored -> current):\n{lines}\n"
        f"See what a migration needs with: {INSPECT_COMMAND}{target}\n"
        f"Rebuild with the current pipeline with: {MIGRATE_COMMAND}{target}"
    )


@dataclass
class IngestionGuard:
    """Write permission and per-source completion state for one collection.

    Attributes:
        manifest: The manifest store, or None when the provider has none.
        logical: Logical collection name.
        physical: Physical collection the tool reads and writes.
        refusal: Error text when writes are refused, else None.
        legacy: True when the collection predates fingerprints.
        sources: Source entries of ``physical`` by source key.
    """

    manifest: ManifestStore | None
    logical: str
    physical: str
    refusal: str | None = None
    legacy: bool = False
    sources: dict[str, SourceEntry] = field(default_factory=dict)

    @property
    def writable(self) -> bool:
        """True when the tool may write."""
        return self.refusal is None

    def require_writable(self) -> None:
        """Raise ``IncompatibleCollectionError`` when writes are refused."""
        if self.refusal is not None:
            raise IncompatibleCollectionError(self.refusal)

    def is_incomplete(self, source_key: str) -> bool:
        """True when an ingestion of the source started and did not finish."""
        entry = self.sources.get(source_key)
        return entry is not None and not entry.complete

    async def begin_source(
        self, source_key: str, chunk_count: int, content_marker: str
    ) -> None:
        """Mark the source incomplete before its chunks are written."""
        self.require_writable()
        await self._put(source_key, chunk_count, content_marker, complete=False)

    async def complete_source(
        self, source_key: str, chunk_count: int, content_marker: str
    ) -> None:
        """Mark the source complete after its last chunk was written."""
        await self._put(source_key, chunk_count, content_marker, complete=True)

    async def forget_source(self, source_key: str) -> None:
        """Remove the source's entry after its chunks were deleted."""
        self.sources.pop(source_key, None)
        if self.manifest is not None:
            await self.manifest.delete_source(self.physical, source_key)

    async def _put(
        self, source_key: str, chunk_count: int, marker: str, *, complete: bool
    ) -> None:
        entry = SourceEntry(
            physical=self.physical,
            source_key=source_key,
            chunk_count=chunk_count,
            complete=complete,
            content_marker=marker,
            updated_at=time.time(),
        )
        self.sources[source_key] = entry
        if self.manifest is not None:
            await self.manifest.put_source(entry)


async def bind_collection(
    manifest: ManifestStore | None,
    adapter: VectorStoreAdapter,
    logical: str,
    current: PipelineInputs,
    *,
    tool_name: str = "",
    migration_target: bool = False,
) -> IngestionGuard:
    """Apply the module rules for ``adapter`` and return its guard.

    Args:
        manifest: The manifest store, or None (no protection, no markers).
        adapter: Adapter of the physical collection the tool will use.
        logical: Logical collection name.
        current: Inputs of the tool's current pipeline.
        tool_name: Tool name, used in the refusal message.
        migration_target: True when ``adapter`` is a collection that
            ``holodeck vectorstore migrate`` is building. The collection
            entry is then left to the migration.
    """
    physical = adapter.schema.name
    guard = IngestionGuard(manifest=manifest, logical=logical, physical=physical)
    if manifest is None:
        return guard
    if migration_target:
        guard.sources = await manifest.sources(physical)
        return guard

    entry = await manifest.get_collection(logical)
    has_records = await collection_has_records(adapter)
    now = time.time()

    if entry is None:
        if has_records:
            guard.legacy = True
            await manifest.put_collection(
                CollectionEntry(
                    logical=logical,
                    fingerprint=current.fingerprint,
                    fingerprint_inputs=legacy_inputs_json(current),
                    active_collection=physical,
                    updated_at=now,
                )
            )
            logger.info(
                "Collection '%s' predates pipeline fingerprints. It stays "
                "readable and unchanged sources are not re-embedded. "
                "'%s AGENT_YAML --tool %s' rebuilds it with the current pipeline.",
                physical,
                MIGRATE_COMMAND,
                tool_name or "NAME",
            )
        else:
            await manifest.put_collection(
                CollectionEntry(
                    logical=logical,
                    fingerprint=current.fingerprint,
                    fingerprint_inputs=current.canonical_json(),
                    active_collection=physical,
                    updated_at=now,
                )
            )
        guard.sources = await manifest.sources(physical)
        return guard

    guard.legacy = is_legacy(entry.fingerprint_inputs)
    if entry.fingerprint != current.fingerprint:
        if has_records:
            guard.refusal = refusal_message(
                logical, physical, entry.fingerprint_inputs, current, tool_name
            )
            logger.warning(
                "Collection '%s' was written by a different ingestion pipeline; "
                "writes are refused and searches read the existing records. "
                "Run '%s AGENT_YAML --tool %s' for details.",
                physical,
                INSPECT_COMMAND,
                tool_name or "NAME",
            )
        else:
            # Nothing stored, nothing to mix: adopt the current pipeline.
            guard.legacy = False
            await manifest.delete_sources(physical)
            await manifest.put_collection(
                dataclasses.replace(
                    entry,
                    fingerprint=current.fingerprint,
                    fingerprint_inputs=current.canonical_json(),
                    updated_at=now,
                )
            )
    guard.sources = await manifest.sources(physical)
    return guard
