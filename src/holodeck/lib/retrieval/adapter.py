"""The contract every vector store adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    HybridQuery,
    LexicalQuery,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    VectorQuery,
    validate_record,
)

DEFAULT_PAGE_SIZE = 512


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """One collection in one vector store.

    An adapter owns its client. ``close`` releases it. Every method that the
    provider cannot do raises ``UnsupportedOperationError`` before it touches
    the store. Reads never return vectors unless ``include_vectors`` is set.
    """

    @property
    def provider(self) -> str:
        """Provider name as written in agent YAML."""
        ...

    @property
    def schema(self) -> CollectionSchema:
        """Layout of the collection."""
        ...

    @property
    def capabilities(self) -> Capabilities:
        """Operations this adapter supports."""
        ...

    async def collection_exists(self) -> bool:
        """Return True when the collection exists in the store."""
        ...

    async def ensure_collection(self) -> None:
        """Create the collection and its indexes when it does not exist."""
        ...

    async def delete_collection(self) -> None:
        """Delete the collection. No error when it does not exist."""
        ...

    async def count(self) -> int:
        """Return the number of records. Zero when the collection is absent."""
        ...

    async def upsert(self, records: Sequence[Record]) -> None:
        """Insert or replace records by ID. Every record carries a vector."""
        ...

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        """Return the records that exist, in the order of ``ids``."""
        ...

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete records by ID. Missing IDs are ignored."""
        ...

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        """Return every record that matches the filter, without vectors."""
        ...

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        """Return one page of all records, without vectors."""
        ...

    async def delete_where(self, record_filter: RecordFilter) -> int:
        """Delete every record that matches the filter. Return the count."""
        ...

    async def replace_source(
        self, record_filter: RecordFilter, records: Sequence[Record]
    ) -> None:
        """Replace the records that match the filter with ``records``.

        Stale records are removed even when the new set is smaller.
        """
        ...

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        """Return the nearest records, best first, scored by similarity."""
        ...

    async def lexical_search(self, query: LexicalQuery) -> list[ScoredRecord]:
        """Return records that match the query tokens, best first."""
        ...

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Return fused vector and lexical results, best first."""
        ...

    async def close(self) -> None:
        """Release the client. Safe to call more than once."""
        ...


class BaseVectorStoreAdapter(ABC):
    """Shared behavior for adapters.

    Subclasses set ``capabilities`` and implement the operations they
    support. Operations left alone raise ``UnsupportedOperationError``.
    """

    def __init__(self, schema: CollectionSchema, capabilities: Capabilities) -> None:
        self._schema = schema
        self._capabilities = capabilities

    @property
    def provider(self) -> str:
        """Provider name as written in agent YAML."""
        return self._capabilities.provider

    @property
    def schema(self) -> CollectionSchema:
        """Layout of the collection."""
        return self._schema

    @property
    def capabilities(self) -> Capabilities:
        """Operations this adapter supports."""
        return self._capabilities

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def _validate(self, records: Sequence[Record]) -> None:
        for record in records:
            validate_record(self._schema, record)

    def _preflight(self, records: Sequence[Record]) -> None:
        """Reject records the store would refuse, before anything is changed.

        ``upsert`` and ``replace_source`` call this before any delete or
        write, so a bad record leaves the stored data intact. The default is
        the schema check. An adapter whose store has limits of its own (key
        length, numeric range) overrides it to run the full conversion.

        Raises:
            SchemaError: A record does not fit the schema or the store.
        """
        self._validate(records)

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Return True when the collection exists in the store."""

    @abstractmethod
    async def ensure_collection(self) -> None:
        """Create the collection and its indexes when it does not exist."""

    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the collection. No error when it does not exist."""

    @abstractmethod
    async def close(self) -> None:
        """Release the client. Safe to call more than once."""

    async def count(self) -> int:
        """Return the number of records. Zero when the collection is absent."""
        self._capabilities.require(Operation.COUNT)
        raise NotImplementedError

    async def upsert(self, records: Sequence[Record]) -> None:
        """Insert or replace records by ID."""
        self._capabilities.require(Operation.UPSERT)
        raise NotImplementedError

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        """Return the records that exist, in the order of ``ids``."""
        self._capabilities.require(Operation.GET)
        raise NotImplementedError

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete records by ID. Missing IDs are ignored."""
        self._capabilities.require(Operation.DELETE)
        raise NotImplementedError

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        """Return every record that matches the filter, without vectors."""
        self._capabilities.require(Operation.FIND)
        raise NotImplementedError

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        """Return one page of all records, without vectors."""
        self._capabilities.require(Operation.LIST)
        raise NotImplementedError

    async def delete_where(self, record_filter: RecordFilter) -> int:
        """Delete every record that matches the filter. Return the count.

        The default finds the matching IDs and deletes them. Adapters with a
        native delete-by-filter override it.
        """
        self._capabilities.require(Operation.DELETE_WHERE)
        matches = await self.find(record_filter)
        if matches:
            await self.delete([record.id for record in matches])
        return len(matches)

    async def replace_source(
        self, record_filter: RecordFilter, records: Sequence[Record]
    ) -> None:
        """Replace the records that match the filter with ``records``.

        Stale records are removed first, then the new set is written. A
        caller that detects changes from a stored record (mtime, content
        hash) therefore sees the old values until the write succeeds, and
        ingests the source again after an interrupted call. Adapters with
        transactions override this to make the replacement atomic.
        """
        self._capabilities.require(Operation.FIND, Operation.DELETE, Operation.UPSERT)
        self._preflight(records)
        existing = await self.find(record_filter)
        keep = {record.id for record in records}
        stale = [record.id for record in existing if record.id not in keep]
        if stale:
            await self.delete(stale)
        if records:
            await self.upsert(records)

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        """Return the nearest records, best first, scored by similarity."""
        self._capabilities.require(Operation.VECTOR_SEARCH)
        raise NotImplementedError

    async def lexical_search(self, query: LexicalQuery) -> list[ScoredRecord]:
        """Return records that match the query tokens, best first."""
        self._capabilities.require(Operation.LEXICAL_SEARCH)
        raise NotImplementedError

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        """Return fused vector and lexical results, best first."""
        self._capabilities.require(Operation.HYBRID_SEARCH)
        raise NotImplementedError
