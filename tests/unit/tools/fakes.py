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
