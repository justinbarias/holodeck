"""Maps provider names to adapters and imports each adapter on first use."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from holodeck.lib.retired_providers import RETIRED_PROVIDERS
from holodeck.lib.retrieval.adapter import VectorStoreAdapter
from holodeck.lib.retrieval.errors import (
    MissingClientError,
    RetiredProviderError,
    UnknownProviderError,
)
from holodeck.lib.retrieval.types import CollectionSchema


class AdapterFactory(Protocol):
    """Class-level constructor every adapter provides."""

    def create(
        self, schema: CollectionSchema, connection: Mapping[str, object]
    ) -> VectorStoreAdapter:
        """Build an adapter from a schema and connection settings."""
        ...


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """Where an adapter lives and which install extra carries its client.

    ``extra`` is None when HoloDeck has no install extra for the client. The
    adapter then names the package in its own ``MissingClientError``.
    """

    module: str
    class_name: str
    extra: str | None = None


_ADAPTERS = "holodeck.lib.retrieval.adapters"


PROVIDERS: dict[str, ProviderEntry] = {
    "in-memory": ProviderEntry(f"{_ADAPTERS}.in_memory", "InMemoryAdapter"),
    "qdrant": ProviderEntry(f"{_ADAPTERS}.qdrant", "QdrantAdapter", extra="qdrant"),
    "chromadb": ProviderEntry(f"{_ADAPTERS}.chroma", "ChromaAdapter", extra="chromadb"),
    "postgres": ProviderEntry(
        f"{_ADAPTERS}.postgres", "PostgresAdapter", extra="postgres"
    ),
    "pinecone": ProviderEntry(
        f"{_ADAPTERS}.pinecone", "PineconeAdapter", extra="pinecone"
    ),
    "weaviate": ProviderEntry(
        f"{_ADAPTERS}.weaviate", "WeaviateAdapter", extra="weaviate"
    ),
    "azure-ai-search": ProviderEntry(
        f"{_ADAPTERS}.azure_ai_search", "AzureAISearchAdapter", extra="azure-ai-search"
    ),
    "azure-cosmos-nosql": ProviderEntry(
        f"{_ADAPTERS}.cosmos_nosql", "CosmosNoSqlAdapter", extra="azure-cosmos"
    ),
}


def supported_providers() -> list[str]:
    """Return the provider names that have an adapter, sorted."""
    return sorted(PROVIDERS)


def create_adapter(
    provider: str,
    schema: CollectionSchema,
    connection: Mapping[str, object] | None = None,
) -> VectorStoreAdapter:
    """Build the adapter for ``provider``.

    Raises:
        RetiredProviderError: The provider name was retired.
        UnknownProviderError: The provider name is not known.
        MissingClientError: The provider's client library is not installed.
    """
    if provider in RETIRED_PROVIDERS:
        raise RetiredProviderError(provider, RETIRED_PROVIDERS[provider])
    entry = PROVIDERS.get(provider)
    if entry is None:
        raise UnknownProviderError(provider, supported_providers())
    try:
        module = importlib.import_module(entry.module)
    except ImportError as error:
        if entry.extra is None:
            raise
        raise MissingClientError(provider, entry.extra) from error
    factory = cast(AdapterFactory, vars(module)[entry.class_name])
    try:
        return factory.create(schema, dict(connection or {}))
    except ImportError as error:
        # An adapter may import its client when it is built, not at module load.
        if entry.extra is None:
            raise
        raise MissingClientError(provider, entry.extra) from error
