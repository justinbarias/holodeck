"""Maps provider names to adapters and imports each adapter on first use."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

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

_BRIDGE = f"{_ADAPTERS}.sk_bridge"

PROVIDERS: dict[str, ProviderEntry] = {
    "in-memory": ProviderEntry(f"{_ADAPTERS}.in_memory", "InMemoryAdapter"),
    # Temporary Semantic Kernel bridge; each entry moves to a native adapter
    # and the bridge is deleted in task T16 of the 042 exec plan.
    "postgres": ProviderEntry(_BRIDGE, "SKPostgresAdapter", extra="postgres"),
    "azure-ai-search": ProviderEntry(_BRIDGE, "SKAzureAISearchAdapter"),
    "qdrant": ProviderEntry(_BRIDGE, "SKQdrantAdapter", extra="qdrant"),
    "weaviate": ProviderEntry(_BRIDGE, "SKWeaviateAdapter"),
    "chromadb": ProviderEntry(_BRIDGE, "SKChromaAdapter", extra="chromadb"),
    "faiss": ProviderEntry(_BRIDGE, "SKFaissAdapter"),
    "azure-cosmos-mongo": ProviderEntry(_BRIDGE, "SKCosmosMongoAdapter"),
    "azure-cosmos-nosql": ProviderEntry(_BRIDGE, "SKCosmosNoSqlAdapter"),
    "sql-server": ProviderEntry(_BRIDGE, "SKSqlServerAdapter"),
    "pinecone": ProviderEntry(_BRIDGE, "SKPineconeAdapter", extra="pinecone"),
}

RETIRED_PROVIDERS: dict[str, str] = {}


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
