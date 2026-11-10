"""In-memory adapter for development and tests. Data is lost on close."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from holodeck.lib.retrieval.adapter import DEFAULT_PAGE_SIZE, BaseVectorStoreAdapter
from holodeck.lib.retrieval.errors import SchemaError
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
    clamp_similarity,
)

PROVIDER = "in-memory"

_CAPABILITIES = Capabilities(
    provider=PROVIDER,
    supported=frozenset(
        {
            Operation.UPSERT,
            Operation.GET,
            Operation.DELETE,
            Operation.FIND,
            Operation.LIST,
            Operation.DELETE_WHERE,
            Operation.COUNT,
            Operation.VECTOR_SEARCH,
        }
    ),
    hints={
        Operation.LEXICAL_SEARCH: "HoloDeck uses its BM25 index for this provider.",
        Operation.HYBRID_SEARCH: "HoloDeck fuses vector and BM25 results itself.",
    },
)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(math.fsum(a * a for a in left)) * math.sqrt(
        math.fsum(b * b for b in right)
    )
    return dot / norm if norm else 0.0


def _without_vector(record: Record) -> Record:
    return Record(id=record.id, fields=record.fields)


class InMemoryAdapter(BaseVectorStoreAdapter):
    """Keeps records in a dict in insertion order."""

    def __init__(self, schema: CollectionSchema) -> None:
        super().__init__(schema, _CAPABILITIES)
        self._records: dict[str, Record] = {}
        self._exists = False

    @classmethod
    def create(
        cls, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> InMemoryAdapter:
        """Build the adapter. The in-memory store takes no connection settings."""
        return cls(schema)

    async def collection_exists(self) -> bool:
        return self._exists

    async def ensure_collection(self) -> None:
        self._exists = True

    async def delete_collection(self) -> None:
        self._records.clear()
        self._exists = False

    async def close(self) -> None:
        return None

    async def count(self) -> int:
        return len(self._records)

    async def upsert(self, records: Sequence[Record]) -> None:
        self._validate(records)
        self._exists = True
        for record in records:
            vector = tuple(float(value) for value in record.vector or ())
            self._records[record.id] = Record(
                id=record.id, fields=dict(record.fields), vector=vector
            )

    async def get(
        self, ids: Sequence[str], *, include_vectors: bool = False
    ) -> list[Record]:
        found = [self._records[key] for key in ids if key in self._records]
        return found if include_vectors else [_without_vector(r) for r in found]

    async def delete(self, ids: Sequence[str]) -> None:
        for key in ids:
            self._records.pop(key, None)

    async def find(
        self, record_filter: RecordFilter, *, limit: int | None = None
    ) -> list[Record]:
        matches = [
            _without_vector(record)
            for record in self._records.values()
            if record_filter.matches(record)
        ]
        return matches if limit is None else matches[:limit]

    async def list_page(
        self, *, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> Page:
        if limit <= 0:
            raise SchemaError("Page limit must be positive.")
        try:
            start = int(cursor) if cursor is not None else 0
        except ValueError as error:
            raise SchemaError(f"Invalid cursor: {cursor!r}") from error
        records = list(self._records.values())
        end = start + limit
        return Page(
            records=tuple(_without_vector(r) for r in records[start:end]),
            next_cursor=str(end) if end < len(records) else None,
        )

    async def vector_search(self, query: VectorQuery) -> list[ScoredRecord]:
        if len(query.vector) != self.schema.dimensions:
            raise SchemaError(
                f"Query vector has {len(query.vector)} dimensions; "
                f"schema '{self.schema.name}' needs {self.schema.dimensions}."
            )
        # Rank on the raw cosine, then clamp: clamping first would tie every
        # negative similarity at zero and pick by insertion order.
        ranked = sorted(
            (
                (_cosine(query.vector, record.vector or ()), record)
                for record in self._records.values()
                if query.filter is None or query.filter.matches(record)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            ScoredRecord(
                record=_without_vector(record),
                score=clamp_similarity(raw),
                kind=ScoreKind.SIMILARITY,
            )
            for raw, record in ranked[: query.top_k]
        ]
