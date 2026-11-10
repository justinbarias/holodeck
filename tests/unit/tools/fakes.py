"""A recording in-memory adapter for tool tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from holodeck.lib.retrieval import (
    Capabilities,
    CollectionSchema,
    HybridQuery,
    Operation,
    Record,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
)
from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter

ALL_OPERATIONS = frozenset(Operation) - {Operation.LEXICAL_SEARCH}


class FakeAdapter(InMemoryAdapter):
    """In-memory adapter with chosen capabilities that records calls.

    ``hybrid_hits`` are returned by ``hybrid_search`` (when the capabilities
    allow it), so tests can drive the native-hybrid path. Set
    ``vector_hits`` to script ``vector_search`` results in a given order.
    """

    def __init__(
        self,
        schema: CollectionSchema,
        *,
        provider: str = "fake",
        operations: Iterable[Operation] | None = None,
        hybrid_hits: Sequence[ScoredRecord] = (),
    ) -> None:
        super().__init__(schema)
        supported = frozenset(operations if operations is not None else ALL_OPERATIONS)
        self._capabilities = Capabilities(provider, supported)
        self.hybrid_hits = list(hybrid_hits)
        self.hybrid_queries: list[HybridQuery] = []
        self.vector_hits: list[ScoredRecord] | None = None
        self.vector_queries: list[VectorQuery] = []
        self.deleted: list[str] = []
        self.close_calls = 0

    async def delete(self, ids: Sequence[str]) -> None:
        self._capabilities.require(Operation.DELETE)
        self.deleted.extend(ids)
        await super().delete(ids)

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        """Return ``vector_hits`` as scripted, or search the stored records."""
        self.vector_queries.append(query)
        if self.vector_hits is not None:
            return self.vector_hits[: query.top_k]
        return await super().vector_search(query)

    def put(self, *records: Record) -> None:
        """Store records without validation (test setup)."""
        self._exists = True
        self._records.update({record.id: record for record in records})

    async def hybrid_search(self, query: HybridQuery) -> list[ScoredRecord]:
        self._capabilities.require(Operation.HYBRID_SEARCH)
        self.hybrid_queries.append(query)
        return self.hybrid_hits[: query.top_k]

    async def close(self) -> None:
        self.close_calls += 1

    def stored(self) -> list[Record]:
        """Return every stored record, in insertion order."""
        return list(self._records.values())


def fused(record: Record, score: float) -> ScoredRecord:
    """Return a native-hybrid hit for ``record``."""
    return ScoredRecord(
        record=Record(id=record.id, fields=record.fields),
        score=score,
        kind=ScoreKind.FUSED,
    )


class FailingWriteError(Exception):
    """Raised by ``StoreRegistry`` adapters to simulate an interrupted write."""


class PersistentAdapter(FakeAdapter):
    """A ``FakeAdapter`` whose records live in a shared registry by name.

    Closing it keeps the records, so a new tool instance sees what an
    earlier one wrote (a cold restart against a persisted store).
    """

    def __init__(
        self,
        schema: CollectionSchema,
        registry: StoreRegistry,
        *,
        provider: str,
        operations: Iterable[Operation] | None = None,
    ) -> None:
        super().__init__(schema, provider=provider, operations=operations)
        self._registry = registry
        state = registry.collections.setdefault(schema.name, {})
        self._records = state  # shared dict: survives tool instances
        self._exists = schema.name in registry.existing

    async def ensure_collection(self) -> None:
        self._registry.existing.add(self.schema.name)
        self._exists = True

    async def collection_exists(self) -> bool:
        return self.schema.name in self._registry.existing

    async def delete_collection(self) -> None:
        self._records.clear()
        self._registry.existing.discard(self.schema.name)
        self._exists = False

    async def upsert(self, records: Sequence[Record]) -> None:
        self._registry.writes.append((self.schema.name, "upsert", len(records)))
        budget = self._registry.fail_after
        if budget is not None and self.schema.name != "holodeck_manifest":
            allowed = max(0, budget - self._registry.upserted)
            if allowed < len(records):
                await super().upsert(records[:allowed])
                self._registry.upserted += allowed
                self._registry.existing.add(self.schema.name)
                raise FailingWriteError(f"write failed after {budget} records")
            self._registry.upserted += len(records)
        await super().upsert(records)
        self._registry.existing.add(self.schema.name)

    async def delete(self, ids: Sequence[str]) -> None:
        self._registry.writes.append((self.schema.name, "delete", len(ids)))
        await super().delete(ids)


class StoreRegistry:
    """Collections by name, shared by every adapter a test builds.

    Patch ``holodeck.lib.retrieval.create_adapter`` and
    ``holodeck.lib.retrieval.registry.create_adapter`` with ``create``.
    Set ``fail_after`` to make data upserts fail once that many records were
    written (manifest writes do not count).
    """

    def __init__(
        self,
        operations: frozenset[Operation] = ALL_OPERATIONS,
        manifest_operations: frozenset[Operation] | None = None,
    ) -> None:
        self.collections: dict[str, dict[str, Record]] = {}
        self.existing: set[str] = set()
        self.operations = operations
        self.manifest_operations = manifest_operations or operations
        self.fail_after: int | None = None
        self.upserted = 0
        self.writes: list[tuple[str, str, int]] = []
        self.created: list[str] = []

    def create(
        self,
        provider: str,
        schema: CollectionSchema,
        connection: object = None,
    ) -> PersistentAdapter:
        self.created.append(schema.name)
        operations = (
            self.manifest_operations
            if schema.name == "holodeck_manifest"
            else self.operations
        )
        return PersistentAdapter(schema, self, provider=provider, operations=operations)

    def records(self, name: str) -> list[Record]:
        return list(self.collections.get(name, {}).values())


class CountingEmbeddings:
    """Embedding service that counts the texts it embeds."""

    def __init__(self, dimensions: int, *, fail_on_call: int | None = None) -> None:
        self.dimensions = dimensions
        self.calls = 0
        self.texts: list[str] = []
        self.fail_on_call = fail_on_call

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_on_call is not None and self.calls >= self.fail_on_call:
            raise RuntimeError("embedding service unavailable")
        self.texts.extend(texts)
        vectors = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) or 1
            vectors.append(
                [((seed * (i + 1)) % 97 + 1) / 97.0 for i in range(self.dimensions)]
            )
        return vectors
