"""The ``holodeck_manifest`` companion collection (FR-016, FR-023).

The record layout of the tools' collections stays as it was. State about a
collection lives in one extra collection per store connection, written
through the same adapter contract:

* a ``collection`` entry per logical collection name: the pipeline
  fingerprint, the physical collection that serves the name, the previous
  physical collection kept for rollback, and migration progress;
* a ``source`` entry per ingested source of a physical collection: the
  expected chunk count and a completion flag. The flag is 0 while the
  source's chunks are written and 1 after the last write succeeded;
* a ``lock`` record per logical collection while ``holodeck vectorstore
  migrate``, ``rollback`` or ``prune`` runs on it.

Stores have no portable compare-and-set. Each write of a collection entry
carries a new random ``revision``; a writer reads the entry back and acts only
when the stored revision is its own. The same write-then-read-back rule takes
a lock. This narrows races between processes; on a store without
transactions it cannot close them.

Every manifest record has the fixed vector ``[1.0, 0.0]``: the manifest is
only read by ID and by exact filter.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from holodeck.lib.errors import HoloDeckError
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
KIND_LOCK = "lock"

STATE_ACTIVE = "active"
STATE_MIGRATING = "migrating"

REQUIRED_OPERATIONS = (
    Operation.UPSERT,
    Operation.GET,
    Operation.DELETE,
    Operation.FIND,
)

# Connection keys that address one collection: its name (every provider's
# ``collection_name``; Azure AI Search and Pinecone ``index_name``) or its
# endpoint (Pinecone ``index_host``, the data plane URL of one index). The
# manifest and a migrated collection are other collections, so they drop
# them. Audited over the eight adapters: Chroma ``tenant``/``database``,
# Cosmos ``database_name``/``partition_key``, Postgres ``db_schema`` and
# Pinecone ``namespace`` scope a whole store or every collection in it, not
# one collection, so they are kept.
COLLECTION_NAME_KEYS = ("collection_name", "index_name", "index_host")

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
        FieldSpec("revision", _S),
        FieldSpec("pending_delete", _S),
        FieldSpec("owner", _S),
        FieldSpec("operation", _S),
        FieldSpec("acquired_at", FieldType.FLOAT),
        FieldSpec("expires_at", FieldType.FLOAT),
    ),
)

_EMPTY_FIELDS: dict[str, str | int | float] = {
    spec.name: (
        0
        if spec.type is FieldType.INTEGER
        else 0.0 if spec.type is FieldType.FLOAT else ""
    )
    for spec in MANIFEST_SCHEMA.fields
}

_warned: set[str] = set()


def collection_entry_id(logical: str) -> str:
    """Return the manifest record ID of a logical collection."""
    return f"{KIND_COLLECTION}::{logical}"


def source_entry_id(physical: str, source_key: str) -> str:
    """Return the manifest record ID of one source of a physical collection."""
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:32]
    return f"{KIND_SOURCE}::{physical}::{digest}"


def lock_entry_id(logical: str) -> str:
    """Return the manifest record ID of a logical collection's lock."""
    return f"{KIND_LOCK}::{logical}"


def new_token() -> str:
    """Return a random token for a revision or a lock owner."""
    return uuid.uuid4().hex


def _record(record_id: str, values: Mapping[str, str | int | float]) -> Record:
    fields = dict(_EMPTY_FIELDS)
    fields.update(values)
    return Record(id=record_id, fields=fields, vector=list(MANIFEST_VECTOR))


@dataclass(frozen=True, slots=True)
class CollectionEntry:
    """Manifest state of one logical collection.

    Attributes:
        logical: Logical collection name: the effective name the tool's
            settings give (``index_name``, else ``collection_name``, else the
            schema default), before provider-specific normalization.
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
        revision: Random token of the write that stored this entry. Every
            write sets a new one (see ``ManifestStore.put_collection``).
        pending_delete: A kept collection that ``prune`` reserved for
            deletion and has not finished deleting, or "".
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
    revision: str = ""
    pending_delete: str = ""

    def with_revision(self, revision: str) -> CollectionEntry:
        """Return this entry carrying ``revision``."""
        return dataclasses.replace(self, revision=revision)

    def to_record(self) -> Record:
        """Return the stored form."""
        return _record(
            collection_entry_id(self.logical),
            {
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
                "updated_at": self.updated_at or time.time(),
                "revision": self.revision,
                "pending_delete": self.pending_delete,
            },
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
            revision=text_field(record, "revision"),
            pending_delete=text_field(record, "pending_delete"),
        )


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """Manifest state of one source of one physical collection.

    Attributes:
        physical: Physical collection that holds the source's chunks.
        source_key: The tool's source key (path or relative path).
        chunk_count: Number of chunks (or structured rows) the ingestion
            writes.
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
        return _record(
            source_entry_id(self.physical, self.source_key),
            {
                "kind": KIND_SOURCE,
                "collection": self.physical,
                "source_key": self.source_key,
                "chunk_count": self.chunk_count,
                "complete": 1 if self.complete else 0,
                "content_marker": self.content_marker,
                "updated_at": self.updated_at or time.time(),
            },
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


@dataclass(frozen=True, slots=True)
class LockRecord:
    """The lifecycle lock of one logical collection.

    Attributes:
        logical: Logical collection name.
        owner: Random token of the process that holds the lock.
        operation: ``migrate``, ``rollback`` or ``prune``.
        acquired_at: Seconds since the epoch when it was taken.
        expires_at: Seconds since the epoch after which it is stale.
    """

    logical: str
    owner: str
    operation: str
    acquired_at: float
    expires_at: float

    @property
    def stale(self) -> bool:
        """True once ``expires_at`` has passed."""
        return time.time() >= self.expires_at

    def to_record(self) -> Record:
        """Return the stored form."""
        return _record(
            lock_entry_id(self.logical),
            {
                "kind": KIND_LOCK,
                "collection": self.logical,
                "owner": self.owner,
                "operation": self.operation,
                "acquired_at": self.acquired_at,
                "expires_at": self.expires_at,
                "updated_at": time.time(),
            },
        )

    @classmethod
    def from_record(cls, record: Record) -> LockRecord:
        """Rebuild a lock from its stored form."""
        return cls(
            logical=text_field(record, "collection"),
            owner=text_field(record, "owner"),
            operation=text_field(record, "operation"),
            acquired_at=_float(record, "acquired_at"),
            expires_at=_float(record, "expires_at"),
        )


class LockHeldError(HoloDeckError):
    """Another process holds the lifecycle lock of a logical collection."""

    def __init__(self, holder: LockRecord) -> None:
        self.holder = holder
        super().__init__(
            f"Collection '{holder.logical}' is locked by a running "
            f"'{holder.operation}' (owner {holder.owner[:12]}, since "
            f"{_when(holder.acquired_at)}; the lock is stale after "
            f"{_when(holder.expires_at)}). Wait for it to finish, then run the "
            f"command again."
        )


def _when(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


def _float(record: Record, name: str) -> float:
    value = record.fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def manifest_connection(connection: Mapping[str, object]) -> dict[str, object]:
    """Return ``connection`` without the keys that address a tool's collection."""
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
            warn_unprotected(
                provider, f"the manifest adapter could not be built: {error}"
            )
            return None
        missing = [
            operation.value
            for operation in REQUIRED_OPERATIONS
            if not adapter.capabilities.supports(operation)
        ]
        if missing:
            warn_unprotected(
                provider, f"the provider does not support {', '.join(missing)}"
            )
            return None
        return cls(adapter)

    async def close(self) -> None:
        """Release the manifest adapter's client."""
        await self._adapter.close()

    async def exists(self) -> bool:
        """Return True when the manifest collection exists."""
        if self._ensured:
            return True
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
        if not await self.exists():
            return None
        found = await self._adapter.get([collection_entry_id(logical)])
        return CollectionEntry.from_record(found[0]) if found else None

    async def put_collection(self, entry: CollectionEntry) -> CollectionEntry:
        """Insert or replace the entry of ``entry.logical`` in one write.

        The stored entry gets a new random revision, which is returned. Read
        the entry back and compare revisions to learn whether another process
        wrote after this one.
        """
        await self._ensure()
        written = dataclasses.replace(
            entry, revision=new_token(), updated_at=entry.updated_at or time.time()
        )
        await self._adapter.upsert([written.to_record()])
        return written

    async def delete_collection_entry(self, logical: str) -> None:
        """Delete the entry of ``logical``. No error when it is absent."""
        if not await self.exists():
            return
        await self._adapter.delete([collection_entry_id(logical)])

    # -- lifecycle locks ---------------------------------------------------

    async def get_lock(self, logical: str) -> LockRecord | None:
        """Return the lock of ``logical``, or None."""
        if not await self.exists():
            return None
        found = await self._adapter.get([lock_entry_id(logical)])
        return LockRecord.from_record(found[0]) if found else None

    async def acquire_lock(
        self, logical: str, operation: str, ttl: float
    ) -> LockRecord:
        """Take the lifecycle lock of ``logical`` for ``ttl`` seconds.

        The lock is written, read back and kept only when the stored owner is
        this caller. A stale lock (past ``expires_at``) is taken over.

        Raises:
            LockHeldError: Another owner holds a lock that is not stale.
        """
        held = await self.get_lock(logical)
        if held is not None and not held.stale:
            raise LockHeldError(held)
        now = time.time()
        lock = LockRecord(logical, new_token(), operation, now, now + ttl)
        await self._ensure()
        await self._adapter.upsert([lock.to_record()])
        stored = await self.get_lock(logical)
        if stored is None or stored.owner != lock.owner:
            if stored is not None:
                raise LockHeldError(stored)
            raise HoloDeckError(
                f"The lock of collection '{logical}' was removed while it was "
                f"being taken; run the command again."
            )
        return lock

    async def refresh_lock(self, lock: LockRecord, ttl: float) -> LockRecord:
        """Extend ``lock`` to ``ttl`` seconds from now.

        Raises:
            LockHeldError: Another owner took the lock over.
        """
        stored = await self.get_lock(lock.logical)
        if stored is not None and stored.owner != lock.owner:
            raise LockHeldError(stored)
        refreshed = dataclasses.replace(lock, expires_at=time.time() + ttl)
        await self._adapter.upsert([refreshed.to_record()])
        return refreshed

    async def release_lock(self, lock: LockRecord) -> None:
        """Remove ``lock`` if this caller still owns it."""
        stored = await self.get_lock(lock.logical)
        if stored is not None and stored.owner == lock.owner:
            await self._adapter.delete([lock_entry_id(lock.logical)])

    # -- source entries ----------------------------------------------------

    async def sources(self, physical: str) -> dict[str, SourceEntry]:
        """Return the source entries of ``physical`` by source key."""
        if not await self.exists():
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
        if not await self.exists():
            return
        await self._adapter.delete([source_entry_id(physical, source_key)])

    async def delete_sources(self, physical: str) -> int:
        """Delete every source entry of ``physical``. Return the count."""
        if not await self.exists():
            return 0
        records = await self._adapter.find(
            RecordFilter.where(kind=KIND_SOURCE, collection=physical)
        )
        if records:
            await self._adapter.delete([record.id for record in records])
        return len(records)


def warn_unprotected(subject: str, reason: str) -> None:
    """Log once per ``subject`` that collections run without a manifest."""
    if subject in _warned:
        return
    _warned.add(subject)
    logger.warning(
        "Vector store '%s' has no pipeline manifest (%s). Fingerprint "
        "protection is off: collections on it are not protected against "
        "pipeline changes and interrupted ingestion is detected from stored "
        "chunks only.",
        subject,
        reason,
    )
