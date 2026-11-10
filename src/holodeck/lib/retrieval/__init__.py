"""HoloDeck's vector store contract.

Tools depend on ``VectorStoreAdapter`` and the types here. Each store's
client stays inside its adapter under ``holodeck.lib.retrieval.adapters``.
"""

from holodeck.lib.retrieval.adapter import (
    DEFAULT_PAGE_SIZE,
    BaseVectorStoreAdapter,
    VectorStoreAdapter,
)
from holodeck.lib.retrieval.errors import (
    IncompatibleCollectionError,
    MissingClientError,
    RetiredProviderError,
    RetrievalError,
    SchemaError,
    StoreConnectionError,
    UnknownProviderError,
    UnsupportedOperationError,
)
from holodeck.lib.retrieval.registry import create_adapter, supported_providers
from holodeck.lib.retrieval.types import (
    Capabilities,
    CollectionSchema,
    Distance,
    FieldEquals,
    FieldSpec,
    FieldType,
    FieldValue,
    HybridQuery,
    LexicalQuery,
    Operation,
    Page,
    Record,
    RecordFilter,
    ScoredRecord,
    ScoreKind,
    VectorQuery,
    clamp_similarity,
    validate_record,
)

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "BaseVectorStoreAdapter",
    "Capabilities",
    "CollectionSchema",
    "Distance",
    "FieldEquals",
    "FieldSpec",
    "FieldType",
    "FieldValue",
    "HybridQuery",
    "IncompatibleCollectionError",
    "LexicalQuery",
    "MissingClientError",
    "Operation",
    "Page",
    "Record",
    "RecordFilter",
    "RetiredProviderError",
    "RetrievalError",
    "SchemaError",
    "ScoreKind",
    "ScoredRecord",
    "StoreConnectionError",
    "UnknownProviderError",
    "UnsupportedOperationError",
    "VectorQuery",
    "VectorStoreAdapter",
    "clamp_similarity",
    "create_adapter",
    "supported_providers",
    "validate_record",
]
