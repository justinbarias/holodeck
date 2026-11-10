"""The ``holodeck_manifest`` companion collection (FR-016, FR-023).

The record layout of the tools' collections stays as it was. State about a
collection lives in one extra collection per store connection, written
through the same adapter contract:

* a ``collection`` entry per logical collection name: the pipeline
  fingerprint, the physical collection that serves the name, the previous
  physical collection kept for rollback, and migration progress;
* a ``source`` entry per ingested source of a physical collection: the
  expected chunk count and a completion flag. The flag is 0 while the
  source's chunks are written and 1 after the last write succeeded.

Every manifest record has the fixed vector ``[1.0, 0.0]``: the manifest is
only read by ID and by exact filter.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass

from holodeck.lib.retrieval import registry
from holodeck.lib.retrieval.adapter import VectorStoreAdapter
from holodeck.lib.retrieval.schemas import int_field, text_field
from holodeck.lib.retrieval.types import (
    CollectionSchema,
    FieldSpec,
    FieldType,
    Operation,
    Record,
    RecordFilter,
)

logger = logging.getLogger(__name__)

MANIFEST_COLLECTION = "holodeck_manifest"
MANIFEST_DIMENSIONS = 2
MANIFEST_VECTOR: tuple[float, float] = (1.0, 0.0)

KIND_COLLECTION = "collection"
KIND_SOURCE = "source"

STATE_ACTIVE = "active"
STATE_MIGRATING = "migrating"

REQUIRED_OPERATIONS = (
    Operation.UPSERT,
    Operation.GET,
    Operation.DELETE,
    Operation.FIND,
)

# Connection keys that name the collection itself. The manifest is its own
# collection, so it ignores them.
COLLECTION_NAME_KEYS = ("collection_name", "index_name")

_S = FieldType.STRING
MANIFEST_SCHEMA = CollectionSchema(
    name=MANIFEST_COLLECTION,
    dimensions=MANIFEST_DIMENSIONS,
    fields=(
        FieldSpec("kind", _S, indexed=True),
        FieldSpec("collection", _S, indexed=True),
        FieldSpec("fingerprint", _S),
        FieldSpec("fingerprint_inputs", _S),
        FieldSpec("active_collection", _S),
        FieldSpec("previous_collection", _S),
        FieldSpec("previous_fingerprint", _S),
        FieldSpec("previous_fingerprint_inputs", _S),
        FieldSpec("state", _S),
        FieldSpec("migration_target", _S),
        FieldSpec("migration_fingerprint", _S),
        FieldSpec("migration_fingerprint_inputs", _S),
        FieldSpec("source_key", _S),
        FieldSpec("chunk_count", FieldType.INTEGER),
        FieldSpec("complete", FieldType.INTEGER),
        FieldSpec("content_marker", _S),
        FieldSpec("updated_at", FieldType.FLOAT),
    ),
)

_warned_providers: set[str] = set()


def collection_entry_id(logical: str) -> str:
    """Return the manifest record ID of a logical collection."""
    return f"{KIND_COLLECTION}::{logical}"


def source_entry_id(physical: str, source_key: str) -> str:
    """Return the manifest record ID of one source of a physical collection."""
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:32]
    return f"{KIND_SOURCE}::{physical}::{digest}"


@dataclass(frozen=True, slots=True)
class CollectionEntry:
    """Manifest state of one logical collection.

    Attributes:
        logical: Logical collection name (the name the tool's schema gives).
        fingerprint: Fingerprint of the pipeline that wrote ``active``.
        fingerprint_inputs: Canonical JSON of those pipeline inputs.
        active_collection: Physical collection that serves the logical name.
        previous_collection: Physical collection kept for rollback, or "".
        previous_fingerprint: Fingerprint of ``previous_collection``.
        previous_fingerprint_inputs: Inputs of ``previous_collection``.
        state: ``active`` or ``migrating``.
        migration_target: Physical collection a migration is building, or "".
        migration_fingerprint: Fingerprint the migration builds.
        migration_fingerprint_inputs: Inputs the migration builds.
        updated_at: Seconds since the epoch of the last write.
    """

    logical: str
    fingerprint: str
    fingerprint_inputs: str
    active_collection: str
    previous_collection: str = ""
    previous_fingerprint: str = ""
    previous_fingerprint_inputs: str = ""
    state: str = STATE_ACTIVE
    migration_target: str = ""
    migration_fingerprint: str = ""
    migration_fingerprint_inputs: str = ""
    updated_at: float = 0.0

    def to_record(self) -> Record:
        """Return the stored form."""
        return Record(
            id=collection_entry_id(self.logical),
            fields={
                "kind": KIND_COLLECTION,
                "collection": self.logical,
                "fingerprint": self.fingerprint,
                "fingerprint_inputs": self.fingerprint_inputs,
                "active_collection": self.active_collection,
                "previous_collection": self.previous_collection,
                "previous_fingerprint": self.previous_fingerprint,
                "previous_fingerprint_inputs": self.previous_fingerprint_inputs,
                "state": self.state,
                "migration_target": self.migration_target,
                "migration_fingerprint": self.migration_fingerprint,
                "migration_fingerprint_inputs": self.migration_fingerprint_inputs,
                "source_key": "",
                "chunk_count": 0,
                "complete": 0,
                "content_marker": "",
                "updated_at": self.updated_at or time.time(),
            },
            vector=list(MANIFEST_VECTOR),
        )

    @classmethod
    def from_record(cls, record: Record) -> CollectionEntry:
        """Rebuild an entry from its stored form."""
        return cls(
            logical=text_field(record, "collection"),
            fingerprint=text_field(record, "fingerprint"),
            fingerprint_inputs=text_field(record, "fingerprint_inputs"),
            active_collection=text_field(record, "active_collection"),
            previous_collection=text_field(record, "previous_collection"),
            previous_fingerprint=text_field(record, "previous_fingerprint"),
            previous_fingerprint_inputs=text_field(
                record, "previous_fingerprint_inputs"
            ),
            state=text_field(record, "state", STATE_ACTIVE),
            migration_target=text_field(record, "migration_target"),
            migration_fingerprint=text_field(record, "migration_fingerprint"),
            migration_fingerprint_inputs=text_field(
                record, "migration_fingerprint_inputs"
            ),
            updated_at=_float(record, "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """Manifest state of one source of one physical collection.

    Attributes:
        physical: Physical collection that holds the source's chunks.
        source_key: The tool's source key (path or relative path).
        chunk_count: Number of chunks the ingestion writes.
        complete: True once every chunk was written.
        content_marker: The mtime or content hash the chunks were made from.
        updated_at: Seconds since the epoch of the last write.
    """

    physical: str
    source_key: str
    chunk_count: int
    complete: bool
    content_marker: str = ""
    updated_at: float = 0.0

    def to_record(self) -> Record:
        """Return the stored form."""
        return Record(
            id=source_entry_id(self.physical, self.source_key),
            fields={
                "kind": KIND_SOURCE,
                "collection": self.physical,
                "fingerprint": "",
                "fingerprint_inputs": "",
                "active_collection": "",
                "previous_collection": "",
                "previous_fingerprint": "",
                "previous_fingerprint_inputs": "",
                "state": "",
                "migration_target": "",
                "migration_fingerprint": "",
                "migration_fingerprint_inputs": "",
                "source_key": self.source_key,
                "chunk_count": self.chunk_count,
                "complete": 1 if self.complete else 0,
                "content_marker": self.content_marker,
                "updated_at": self.updated_at or time.time(),
            },
            vector=list(MANIFEST_VECTOR),
        )

    @classmethod
    def from_record(cls, record: Record) -> SourceEntry:
        """Rebuild an entry from its stored form."""
        return cls(
            physical=text_field(record, "collection"),
            source_key=text_field(record, "source_key"),
            chunk_count=int_field(record, "chunk_count"),
            complete=int_field(record, "complete") == 1,
            content_marker=text_field(record, "content_marker"),
            updated_at=_float(record, "updated_at"),
        )


def _float(record: Record, name: str) -> float:
    value = record.fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def manifest_connection(connection: Mapping[str, object]) -> dict[str, object]:
    """Return ``connection`` without the keys that name a tool's collection."""
    return {
        key: value
        for key, value in connection.items()
        if key not in COLLECTION_NAME_KEYS
    }


class ManifestStore:
    """Reads and writes manifest entries through one adapter.

    Reads never create the manifest collection. The first write creates it.
    """

    def __init__(self, adapter: VectorStoreAdapter) -> None:
        self._adapter = adapter
        self._ensured = False

    @property
    def adapter(self) -> VectorStoreAdapter:
        """The adapter of the manifest collection."""
        return self._adapter

    @classmethod
    def open(
        cls, provider: str, connection: Mapping[str, object]
    ) -> ManifestStore | None:
        """Return a store for ``provider``, or None when it cannot hold one.

        A provider without get, delete, upsert or exact filtering, or one
        whose adapter cannot be built, has no manifest. A warning is logged
        once per provider; callers then run without fingerprint protection.
        """
        try:
            adapter = registry.create_adapter(
                provider, MANIFEST_SCHEMA, manifest_connection(connection)
            )
        except Exception as error:  # noqa: BLE001 - degrade, never block startup
            _warn_once(provider, f"the manifest adapter could not be built: {error}")
            return None
        missing = [
            operation.value
            for operation in REQUIRED_OPERATIONS
            if not adapter.capabilities.supports(operation)
        ]
        if missing:
            _warn_once(provider, f"the provider does not support {', '.join(missing)}")
            return None
        return cls(adapter)

    async def close(self) -> None:
        """Release the manifest adapter's client."""
        await self._adapter.close()

    async def exists(self) -> bool:
        """Return True when the manifest collection exists."""
        return await self._adapter.collection_exists()

    async def _ensure(self) -> None:
        if self._ensured:
            return
        try:
            await self._adapter.ensure_collection()
        except Exception:
            # Another process may have created it between the check and the
            # create call. Re-raise only when it still does not exist.
            await asyncio.sleep(0.2)
            if not await self._adapter.collection_exists():
                raise
        self._ensured = True

    # -- collection entries ------------------------------------------------

    async def get_collection(self, logical: str) -> CollectionEntry | None:
        """Return the entry of ``logical``, or None."""
        if not self._ensured and not await self.exists():
            return None
        found = await self._adapter.get([collection_entry_id(logical)])
        return CollectionEntry.from_record(found[0]) if found else None

    async def put_collection(self, entry: CollectionEntry) -> None:
        """Insert or replace the entry of ``entry.logical`` in one write."""
        await self._ensure()
        await self._adapter.upsert([entry.to_record()])

    async def delete_collection_entry(self, logical: str) -> None:
        """Delete the entry of ``logical``. No error when it is absent."""
        if not self._ensured and not await self.exists():
            return
        await self._adapter.delete([collection_entry_id(logical)])

    # -- source entries ----------------------------------------------------

    async def sources(self, physical: str) -> dict[str, SourceEntry]:
        """Return the source entries of ``physical`` by source key."""
        if not self._ensured and not await self.exists():
            return {}
        records = await self._adapter.find(
            RecordFilter.where(kind=KIND_SOURCE, collection=physical)
        )
        entries = (SourceEntry.from_record(record) for record in records)
        return {entry.source_key: entry for entry in entries}

    async def put_source(self, entry: SourceEntry) -> None:
        """Insert or replace one source entry."""
        await self._ensure()
        await self._adapter.upsert([entry.to_record()])

    async def delete_source(self, physical: str, source_key: str) -> None:
        """Delete one source entry. No error when it is absent."""
        if not self._ensured and not await self.exists():
            return
        await self._adapter.delete([source_entry_id(physical, source_key)])

    async def delete_sources(self, physical: str) -> int:
        """Delete every source entry of ``physical``. Return the count."""
        if not self._ensured and not await self.exists():
            return 0
        records = await self._adapter.find(
            RecordFilter.where(kind=KIND_SOURCE, collection=physical)
        )
        if records:
            await self._adapter.delete([record.id for record in records])
        return len(records)


def _warn_once(provider: str, reason: str) -> None:
    if provider in _warned_providers:
        return
    _warned_providers.add(provider)
    logger.warning(
        "Vector store provider '%s' has no pipeline manifest (%s). Collections "
        "on it are not protected against pipeline changes and interrupted "
        "ingestion is detected from stored chunks only.",
        provider,
        reason,
    )
