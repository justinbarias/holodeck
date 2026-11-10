"""Record, schema, query, result and capability types for vector stores.

These types are the only shapes that cross the adapter boundary. Vendor
record, filter and client types stay inside each adapter.

Score convention
----------------
``ScoredRecord.score`` is always "higher is better".

* ``ScoreKind.SIMILARITY``: cosine similarity clamped to ``[0.0, 1.0]``.
  Adapters whose store returns a distance convert it before returning.
* ``ScoreKind.LEXICAL``: the store's own lexical relevance. Only the order
  is portable; the magnitude is not comparable between providers.
* ``ScoreKind.FUSED``: a rank-fusion score. Only the order is portable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from holodeck.lib.retrieval.errors import SchemaError, UnsupportedOperationError

FieldValue = str | int | float | bool | None
"""A stored field value. Lists and nested objects are stored as JSON text."""


class FieldType(str, Enum):
    """Storage type of a data field."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class Distance(str, Enum):
    """Vector comparison used by a collection."""

    COSINE = "cosine"


class ScoreKind(str, Enum):
    """What a result score means. See the module docstring."""

    SIMILARITY = "similarity"
    LEXICAL = "lexical"
    FUSED = "fused"


class Operation(str, Enum):
    """An adapter operation that a provider may or may not support."""

    UPSERT = "upsert"
    GET = "get by ID"
    DELETE = "delete by ID"
    FIND = "exact metadata filtering"
    LIST = "paginated enumeration"
    DELETE_WHERE = "delete by metadata filter"
    COUNT = "record count"
    VECTOR_SEARCH = "vector search"
    LEXICAL_SEARCH = "lexical search"
    HYBRID_SEARCH = "native hybrid search"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One data field of a collection.

    Attributes:
        name: Field name as stored.
        type: Storage type.
        indexed: True when exact-match filters use this field. Filter only
            on indexed fields: some stores reject a filter on any other.
        full_text: True when lexical search uses this field.
    """

    name: str
    type: FieldType = FieldType.STRING
    indexed: bool = False
    full_text: bool = False


@dataclass(frozen=True, slots=True)
class CollectionSchema:
    """Layout of one collection.

    The key field is always named ``id`` and the vector field ``embedding``.
    Both names are part of the stored layout of existing collections.
    """

    name: str
    dimensions: int
    fields: tuple[FieldSpec, ...]
    key_field: str = "id"
    vector_field: str = "embedding"
    distance: Distance = Distance.COSINE

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("Collection name must not be empty.")
        if self.dimensions <= 0 or self.dimensions > 10000:
            raise SchemaError(f"Invalid dimensions: {self.dimensions}")
        names = [spec.name for spec in self.fields]
        if len(set(names)) != len(names):
            raise SchemaError(f"Duplicate field names in schema '{self.name}'.")
        reserved = {self.key_field, self.vector_field}
        clash = reserved.intersection(names)
        if clash:
            raise SchemaError(
                f"Field names {sorted(clash)} are reserved in schema '{self.name}'."
            )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Names of the data fields, in schema order."""
        return tuple(spec.name for spec in self.fields)

    @property
    def full_text_fields(self) -> tuple[str, ...]:
        """Names of the fields that lexical search uses."""
        return tuple(spec.name for spec in self.fields if spec.full_text)

    def field(self, name: str) -> FieldSpec:
        """Return the spec for ``name`` or raise ``SchemaError``."""
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise SchemaError(f"Schema '{self.name}' has no field '{name}'.")


@dataclass(frozen=True, slots=True)
class Record:
    """One stored record.

    Attributes:
        id: HoloDeck record ID. Adapters map it to the store's key type.
        fields: Data field values by name.
        vector: Embedding, or None when the read did not ask for vectors.
            A store with cosine collections may return it normalized.
    """

    id: str
    fields: Mapping[str, FieldValue] = field(default_factory=dict)
    vector: Sequence[float] | None = None


@dataclass(frozen=True, slots=True)
class FieldEquals:
    """Exact match of one data field."""

    field: str
    value: FieldValue


@dataclass(frozen=True, slots=True)
class RecordFilter:
    """Conjunction of exact matches. An empty filter matches every record."""

    all_of: tuple[FieldEquals, ...] = ()

    @classmethod
    def where(cls, **equals: FieldValue) -> RecordFilter:
        """Build a filter from keyword arguments."""
        return cls(tuple(FieldEquals(name, value) for name, value in equals.items()))

    def matches(self, record: Record) -> bool:
        """Return True when ``record`` satisfies every clause."""
        return all(
            clause.field in record.fields
            and record.fields[clause.field] == clause.value
            for clause in self.all_of
        )


@dataclass(frozen=True, slots=True)
class VectorQuery:
    """Nearest-neighbour query."""

    vector: Sequence[float]
    top_k: int
    filter: RecordFilter | None = None


@dataclass(frozen=True, slots=True)
class LexicalQuery:
    """Token query against one full-text field."""

    text: str
    top_k: int
    text_field: str
    filter: RecordFilter | None = None


@dataclass(frozen=True, slots=True)
class HybridQuery:
    """Vector plus lexical query fused by the store or the adapter."""

    text: str
    vector: Sequence[float]
    top_k: int
    text_field: str
    filter: RecordFilter | None = None


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    """A search hit. ``record.vector`` is None."""

    record: Record
    score: float
    kind: ScoreKind = ScoreKind.SIMILARITY


@dataclass(frozen=True, slots=True)
class Page:
    """One page of an enumeration. ``next_cursor`` is None on the last page."""

    records: tuple[Record, ...]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Operations a provider supports."""

    provider: str
    supported: frozenset[Operation]
    hints: Mapping[Operation, str] = field(default_factory=dict)

    def supports(self, operation: Operation) -> bool:
        """Return True when the provider supports ``operation``."""
        return operation in self.supported

    def require(self, *operations: Operation) -> None:
        """Raise ``UnsupportedOperationError`` for the first missing operation."""
        for operation in operations:
            if operation not in self.supported:
                raise UnsupportedOperationError(
                    self.provider, operation.value, self.hints.get(operation, "")
                )


def clamp_similarity(raw: float) -> float:
    """Clamp a cosine similarity to ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(raw)))


def normalize_filter_value(spec: FieldSpec, value: FieldValue) -> FieldValue:
    """Return ``value`` in the form that matches stored values of ``spec``'s type.

    The filter form follows the field's type, not the Python type of the
    argument: an ``int`` on a ``FLOAT`` field becomes a float (``mtime=2``
    finds ``2.0``) and an integral ``float`` on an ``INTEGER`` field becomes an
    int. Other values pass through unchanged.

    Raises:
        SchemaError: A ``bool`` is given for a numeric field. ``True`` is not
            the number 1 here.
    """
    if value is None:
        return None
    if spec.type in (FieldType.INTEGER, FieldType.FLOAT) and isinstance(value, bool):
        raise SchemaError(
            f"Filter value {value!r} does not fit {spec.type.value} field "
            f"'{spec.name}': a boolean is not a number."
        )
    if spec.type is FieldType.FLOAT and isinstance(value, int):
        return float(value)
    if (
        spec.type is FieldType.INTEGER
        and isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)
    return value


def checked_filter_value(spec: FieldSpec, value: FieldValue) -> FieldValue:
    """Return ``value`` in the stored form of ``spec``, or raise ``SchemaError``.

    A filter value that cannot equal any stored value of the field's type is
    an error, not an empty result and not a lossy conversion:
    ``where(chunk_index=1.9)`` on an ``INTEGER`` field never becomes ``1``.
    ``None`` passes through; each adapter decides what a null filter means.

    Raises:
        SchemaError: The value does not fit the field type.
    """
    normalized = normalize_filter_value(spec, value)
    if normalized is None:
        return None
    fits = {
        FieldType.STRING: isinstance(normalized, str),
        FieldType.INTEGER: isinstance(normalized, int)
        and not isinstance(normalized, bool),
        FieldType.FLOAT: isinstance(normalized, float),
        FieldType.BOOLEAN: isinstance(normalized, bool),
    }[spec.type]
    if not fits:
        raise SchemaError(
            f"Filter value {value!r} does not fit {spec.type.value} field "
            f"'{spec.name}': it has another type and cannot equal a stored value."
        )
    return normalized


def check_filter(
    schema: CollectionSchema, record_filter: RecordFilter | None
) -> RecordFilter:
    """Return ``record_filter`` with every value checked and normalized.

    Raises:
        SchemaError: A clause names an unknown field or its value does not fit.
    """
    if record_filter is None:
        return RecordFilter()
    return RecordFilter(
        tuple(
            FieldEquals(
                clause.field,
                checked_filter_value(schema.field(clause.field), clause.value),
            )
            for clause in record_filter.all_of
        )
    )


def validate_record(schema: CollectionSchema, record: Record) -> None:
    """Raise ``SchemaError`` when ``record`` cannot be written to ``schema``."""
    if not record.id:
        raise SchemaError("Record ID must not be empty.")
    unknown = set(record.fields).difference(schema.field_names)
    if unknown:
        raise SchemaError(
            f"Record '{record.id}' has fields not in schema '{schema.name}': "
            f"{sorted(unknown)}"
        )
    if record.vector is None:
        raise SchemaError(f"Record '{record.id}' has no vector.")
    if len(record.vector) != schema.dimensions:
        raise SchemaError(
            f"Record '{record.id}' has a vector of {len(record.vector)} dimensions; "
            f"schema '{schema.name}' needs {schema.dimensions}."
        )
    if not all(math.isfinite(component) for component in record.vector):
        raise SchemaError(
            f"Record '{record.id}' has a vector with a non-finite (NaN or "
            "infinite) value."
        )
