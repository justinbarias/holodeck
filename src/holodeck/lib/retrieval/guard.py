"""Pipeline checks a retrieval tool runs before it writes (FR-016, FR-023).

A tool binds in two steps, both from ONE read of its manifest entry
(``read_snapshot``), so the physical collection it uses and the fingerprint it
checks come from the same revision:

1. ``resolve_physical`` maps the logical collection name to the physical
   collection that serves it;
2. ``bind_collection`` compares the stored pipeline fingerprint with the
   current one, decides whether the tool may write, and loads the per-source
   completion flags, so an ingestion that stopped midway is redone on the
   next start.

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

Concurrent starts
-----------------
Stores have no portable compare-and-set, so two processes can both see "no
entry" and both write one. After writing an entry the tool reads it back and
writes only when the stored revision is its own; otherwise the other
writer's entry decides (same fingerprint: write; different: refuse). The
guard then checks the entry again immediately before the first write of each
source (one manifest read per source) and refuses when its fingerprint or its
physical collection changed. This narrows the race; on a store without
transactions it cannot close it.

Manifest failures
-----------------
If the manifest collection does not exist and cannot be created (quota,
permissions, a provider error), no pointer exists to honor: the tool runs as
before, with one warning that fingerprint protection is off. If the manifest
exists but cannot be read or written, an established pointer may be in it:
reads work and every write is refused.
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
    STATE_MIGRATING,
    CollectionEntry,
    ManifestStore,
    SourceEntry,
    warn_unprotected,
)
from holodeck.lib.retrieval.types import CollectionSchema, Operation, RecordFilter

logger = logging.getLogger(__name__)

INSPECT_COMMAND = "holodeck vectorstore inspect"
MIGRATE_COMMAND = "holodeck vectorstore migrate"

# Settings that replace the collection's name, most specific first. Azure AI
# Search and Pinecone read ``index_name`` before the schema name.
_NAME_OVERRIDES = ("index_name",)


def effective_collection_name(
    schema: CollectionSchema, connection: Mapping[str, object]
) -> str:
    """Return the logical collection name of a tool.

    It is the name the provider is asked for: ``index_name`` when set, else
    the schema name (which already carries ``collection_name``), before any
    provider-specific normalization. Two tools with the same effective name
    share one collection and one manifest entry.
    """
    for key in _NAME_OVERRIDES:
        value = connection.get(key)
        if isinstance(value, str) and value:
            return value
    return schema.name


@dataclass(frozen=True, slots=True)
class ManifestSnapshot:
    """One read of a logical collection's manifest entry.

    Attributes:
        entry: The entry, or None when there is none (or no manifest).
        manifest_exists: True when the manifest collection exists.
        error: Why the manifest could not be read, or None.
    """

    entry: CollectionEntry | None = None
    manifest_exists: bool = False
    error: str | None = None


async def read_snapshot(
    manifest: ManifestStore | None, logical: str
) -> ManifestSnapshot:
    """Read the entry of ``logical`` once. Never raises."""
    if manifest is None:
        return ManifestSnapshot()
    try:
        exists = await manifest.exists()
    except Exception as error:  # noqa: BLE001 - reported through the guard
        return ManifestSnapshot(error=_describe(error))
    if not exists:
        return ManifestSnapshot()
    try:
        entry = await manifest.get_collection(logical)
    except Exception as error:  # noqa: BLE001 - reported through the guard
        return ManifestSnapshot(manifest_exists=True, error=_describe(error))
    return ManifestSnapshot(entry=entry, manifest_exists=True)


def resolve_physical(snapshot: ManifestSnapshot, logical: str) -> str:
    """Return the physical collection that serves ``logical``.

    Without a manifest entry the physical name is the logical name, which is
    the name collections had before migrations existed.
    """
    entry = snapshot.entry
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

    Settings that address one collection (``collection_name``,
    ``index_name``, ``index_host``) address the logical collection. A
    migrated physical collection is another collection, so they are dropped
    for it.
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


def _manifest_refusal(logical: str, reason: str) -> str:
    return (
        f"The pipeline manifest of collection '{logical}' could not be read or "
        f"written ({reason}). It may hold a pointer or fingerprint this tool "
        f"must honor, so HoloDeck refuses every write; searches still read "
        f"the collection. Retry when the store is reachable."
    )


def _moved_refusal(logical: str, physical: str, entry: CollectionEntry) -> str:
    return (
        f"The manifest entry of collection '{logical}' changed while this tool "
        f"was starting: it now points at '{entry.active_collection}' (state "
        f"{entry.state}), not '{physical}'. Another process migrated, rolled "
        f"back or re-bound it, so writes are refused. Restart to use the new "
        f"collection."
    )


def _describe(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


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
        fingerprint: Fingerprint the tool writes with ("" without a manifest).
        revision: Revision of the manifest entry the guard was bound to.
        migration_target: True when ``physical`` is a migration's target.
    """

    manifest: ManifestStore | None
    logical: str
    physical: str
    refusal: str | None = None
    legacy: bool = False
    sources: dict[str, SourceEntry] = field(default_factory=dict)
    fingerprint: str = ""
    revision: str = ""
    migration_target: bool = False
    _validated: set[str] = field(default_factory=set, repr=False)

    @property
    def writable(self) -> bool:
        """True when the tool may write."""
        return self.refusal is None

    def require_writable(self) -> None:
        """Raise ``IncompatibleCollectionError`` when writes are refused."""
        if self.refusal is not None:
            raise IncompatibleCollectionError(self.refusal)

    async def validate_before_write(self, source_key: str) -> None:
        """Check the manifest entry again before the first write of a source.

        One manifest read per source. When the entry's revision changed since
        the bind, the guard re-binds once to the new entry if it still names
        this physical collection and fingerprint, and refuses otherwise.

        Raises:
            IncompatibleCollectionError: Writes are refused.
        """
        self.require_writable()
        if self.manifest is None or source_key in self._validated:
            return
        try:
            entry = await self.manifest.get_collection(self.logical)
        except Exception as error:  # noqa: BLE001 - fail closed for writes
            self.refusal = _manifest_refusal(self.logical, _describe(error))
            raise IncompatibleCollectionError(self.refusal) from error
        self._check_entry(entry)
        self._validated.add(source_key)

    def _check_entry(self, entry: CollectionEntry | None) -> None:
        if self.migration_target:
            if (
                entry is None
                or entry.state != STATE_MIGRATING
                or entry.migration_target != self.physical
                or entry.migration_fingerprint != self.fingerprint
            ):
                self.refusal = (
                    f"The migration of collection '{self.logical}' into "
                    f"'{self.physical}' was abandoned or replaced by another "
                    f"process; writes are refused."
                )
                raise IncompatibleCollectionError(self.refusal)
            return
        if entry is None or entry.revision == self.revision:
            return
        if entry.active_collection != self.physical:
            self.refusal = _moved_refusal(self.logical, self.physical, entry)
            raise IncompatibleCollectionError(self.refusal)
        if entry.fingerprint != self.fingerprint:
            self.refusal = (
                f"Another process recorded a different ingestion pipeline for "
                f"collection '{self.logical}' after this tool started; writes "
                f"are refused so the two pipelines do not mix. Run "
                f"'{INSPECT_COMMAND}' to compare them."
            )
            raise IncompatibleCollectionError(self.refusal)
        self.revision = entry.revision  # Same pointer and pipeline: re-bound.

    def is_incomplete(self, source_key: str) -> bool:
        """True when an ingestion of the source started and did not finish."""
        entry = self.sources.get(source_key)
        return entry is not None and not entry.complete

    def completed(self, source_key: str, content_marker: str) -> SourceEntry | None:
        """Return the source's entry when it is complete for ``content_marker``."""
        entry = self.sources.get(source_key)
        if entry is None or not entry.complete:
            return None
        return entry if entry.content_marker == content_marker else None

    async def begin_source(
        self, source_key: str, chunk_count: int, content_marker: str
    ) -> None:
        """Mark the source incomplete before its chunks are written."""
        await self.validate_before_write(source_key)
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
    snapshot: ManifestSnapshot | None = None,
    physical: str | None = None,
    tool_name: str = "",
    migration_target: bool = False,
) -> IngestionGuard:
    """Apply the module rules for ``adapter`` and return its guard.

    Args:
        manifest: The manifest store, or None (no protection, no markers).
        adapter: Adapter of the physical collection the tool will use.
        logical: Logical collection name.
        current: Inputs of the tool's current pipeline.
        snapshot: The manifest read the physical name was resolved from. Read
            here when not given.
        physical: Physical collection name; ``adapter.schema.name`` when not
            given.
        tool_name: Tool name, used in the refusal message.
        migration_target: True when ``adapter`` is a collection that
            ``holodeck vectorstore migrate`` is building. The collection
            entry is then left to the migration.
    """
    physical = physical or adapter.schema.name
    guard = IngestionGuard(
        manifest=manifest,
        logical=logical,
        physical=physical,
        fingerprint=current.fingerprint if manifest is not None else "",
        migration_target=migration_target,
    )
    if manifest is None:
        return guard
    if snapshot is None:
        snapshot = await read_snapshot(manifest, logical)
    if snapshot.error is not None:
        # The manifest may hold a pointer this tool cannot see: fail closed.
        guard.refusal = _manifest_refusal(logical, snapshot.error)
        logger.warning("Collection '%s': %s", physical, guard.refusal)
        return guard
    try:
        if migration_target:
            guard.sources = await manifest.sources(physical)
            return guard
        return await _bind(guard, manifest, adapter, snapshot, current, tool_name)
    except IncompatibleCollectionError:
        raise
    except Exception as error:  # noqa: BLE001 - see "Manifest failures"
        if snapshot.manifest_exists:
            guard.refusal = _manifest_refusal(logical, _describe(error))
            logger.warning("Collection '%s': %s", physical, guard.refusal)
            return guard
        warn_unprotected(
            f"{adapter.capabilities.provider}:{logical}",
            f"the manifest collection could not be created: {_describe(error)}",
        )
        return IngestionGuard(manifest=None, logical=logical, physical=physical)


async def _bind(
    guard: IngestionGuard,
    manifest: ManifestStore,
    adapter: VectorStoreAdapter,
    snapshot: ManifestSnapshot,
    current: PipelineInputs,
    tool_name: str,
) -> IngestionGuard:
    logical, physical = guard.logical, guard.physical
    entry = snapshot.entry
    has_records = await collection_has_records(adapter)
    now = time.time()

    if entry is None:
        if has_records:
            guard.legacy = True
            inputs = legacy_inputs_json(current)
        else:
            inputs = current.canonical_json()
        written = await manifest.put_collection(
            CollectionEntry(
                logical=logical,
                fingerprint=current.fingerprint,
                fingerprint_inputs=inputs,
                active_collection=physical,
                updated_at=now,
            )
        )
        if await _confirm(guard, manifest, written, current, tool_name) and (
            guard.legacy
        ):
            logger.info(
                "Collection '%s' predates pipeline fingerprints. It stays "
                "readable and unchanged sources are not re-embedded. "
                "'%s AGENT_YAML --tool %s' rebuilds it with the current pipeline.",
                physical,
                MIGRATE_COMMAND,
                tool_name or "NAME",
            )
        guard.sources = await manifest.sources(physical)
        return guard

    guard.revision = entry.revision
    guard.legacy = is_legacy(entry.fingerprint_inputs)
    if entry.fingerprint != current.fingerprint:
        if has_records:
            _refuse(guard, entry, current, tool_name)
        else:
            # Nothing stored, nothing to mix: adopt the current pipeline.
            guard.legacy = False
            await manifest.delete_sources(physical)
            written = await manifest.put_collection(
                dataclasses.replace(
                    entry,
                    fingerprint=current.fingerprint,
                    fingerprint_inputs=current.canonical_json(),
                    updated_at=now,
                )
            )
            await _confirm(guard, manifest, written, current, tool_name)
    guard.sources = await manifest.sources(physical)
    return guard


async def _confirm(
    guard: IngestionGuard,
    manifest: ManifestStore,
    written: CollectionEntry,
    current: PipelineInputs,
    tool_name: str,
) -> bool:
    """Read the entry back; return True when the stored one is ``written``.

    When another process wrote after this one, its entry decides: the same
    fingerprint and physical collection allow writes, anything else refuses.
    """
    stored = await manifest.get_collection(guard.logical)
    if stored is not None and stored.revision == written.revision:
        guard.revision = written.revision
        return True
    if stored is None:
        guard.refusal = _manifest_refusal(
            guard.logical, "the entry disappeared right after it was written"
        )
        return False
    guard.revision = stored.revision
    guard.legacy = is_legacy(stored.fingerprint_inputs)
    if stored.active_collection != guard.physical:
        guard.refusal = _moved_refusal(guard.logical, guard.physical, stored)
    elif stored.fingerprint != current.fingerprint:
        _refuse(guard, stored, current, tool_name)
    return False


def _refuse(
    guard: IngestionGuard,
    entry: CollectionEntry,
    current: PipelineInputs,
    tool_name: str,
) -> None:
    guard.refusal = refusal_message(
        guard.logical, guard.physical, entry.fingerprint_inputs, current, tool_name
    )
    logger.warning(
        "Collection '%s' was written by a different ingestion pipeline; "
        "writes are refused and searches read the existing records. "
        "Run '%s AGENT_YAML --tool %s' for details.",
        guard.physical,
        INSPECT_COMMAND,
        tool_name or "NAME",
    )
