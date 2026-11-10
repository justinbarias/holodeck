"""Adapters that the contract suite runs against.

Add one ``ContractTarget`` per adapter. ``available`` returns a skip reason
when the store cannot be reached, or None when the target can run.
"""

from __future__ import annotations

import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

from holodeck.lib.retrieval import CollectionSchema, VectorStoreAdapter
from holodeck.lib.retrieval.adapters.azure_ai_search import (
    AzureAISearchAdapter,
    AzureSearchSettings,
)
from holodeck.lib.retrieval.adapters.chroma import ChromaAdapter
from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter
from holodeck.lib.retrieval.adapters.postgres import PostgresAdapter
from holodeck.lib.retrieval.adapters.qdrant import QdrantAdapter
from holodeck.lib.retrieval.adapters.sk_bridge import (
    SKChromaAdapter,
    SKInMemoryAdapter,
    SKQdrantAdapter,
)
from holodeck.lib.retrieval.adapters.weaviate import WeaviateAdapter
from tests.contract.retrieval import pinecone_local
from tests.contract.retrieval.azure_search_fake import FakeSearchService

QDRANT_URL = "http://127.0.0.1:6333"
CHROMA_URL = "http://127.0.0.1:8765"
POSTGRES_URL = "postgresql://postgres:holodeck@127.0.0.1:54329/holodeck"
WEAVIATE_CONNECTION = {
    "local_host": "127.0.0.1",
    "local_port": 8089,
    "local_grpc_port": 50059,
}


def build_azure_fake(schema: CollectionSchema) -> AzureAISearchAdapter:
    """Return the Azure AI Search adapter on the in-memory fake clients.

    No emulator or credentials exist for this store, so this target is the only
    contract run for it. It does not prove behavior of the live service.
    """
    return AzureAISearchAdapter(
        schema,
        AzureSearchSettings(endpoint="https://fake.search.windows.net", api_key="fake"),
        client_factory=FakeSearchService().pair,
    )


@dataclass(frozen=True)
class ContractTarget:
    """One adapter under test."""

    name: str
    build: Callable[[CollectionSchema], VectorStoreAdapter]
    available: Callable[[], str | None] = lambda: None


def port_open(host: str, port: int) -> Callable[[], str | None]:
    """Return a probe that gives a skip reason when ``host:port`` is closed."""

    def probe() -> str | None:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return None
        except OSError:
            return f"{host}:{port} is not reachable"

    return probe


_EMBEDDED_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []


def _embedded_chroma(schema: CollectionSchema) -> VectorStoreAdapter:
    """A native Chroma adapter on its own persistent directory."""
    directory = tempfile.TemporaryDirectory(prefix="holodeck-chroma-contract-")
    _EMBEDDED_DIRECTORIES.append(directory)  # removed when the process ends
    return ChromaAdapter.create(schema, {"persist_directory": directory.name})


TARGETS: list[ContractTarget] = [
    ContractTarget(name="in-memory", build=InMemoryAdapter),
    ContractTarget(
        name="qdrant",
        build=lambda schema: QdrantAdapter.create(
            schema, {"connection_string": QDRANT_URL}
        ),
        available=port_open("127.0.0.1", 6333),
    ),
    ContractTarget(
        name="qdrant:memory",
        build=lambda schema: QdrantAdapter.create(schema, {"location": ":memory:"}),
    ),
    ContractTarget(
        name="chromadb",
        build=lambda schema: ChromaAdapter.create(
            schema, {"connection_string": CHROMA_URL}
        ),
        available=port_open("127.0.0.1", 8765),
    ),
    ContractTarget(name="chromadb-embedded", build=_embedded_chroma),
    ContractTarget(
        name="sk-bridge:in-memory",
        build=lambda schema: SKInMemoryAdapter.create(schema, {}),
    ),
    ContractTarget(
        name="sk-bridge:qdrant",
        build=lambda schema: SKQdrantAdapter.create(
            schema, {"connection_string": QDRANT_URL}
        ),
        available=port_open("127.0.0.1", 6333),
    ),
    ContractTarget(
        name="sk-bridge:chromadb",
        build=lambda schema: SKChromaAdapter.create(
            schema, {"connection_string": CHROMA_URL}
        ),
        available=port_open("127.0.0.1", 8765),
    ),
    ContractTarget(
        name="postgres",
        build=lambda schema: PostgresAdapter.create(
            schema, {"connection_string": POSTGRES_URL}
        ),
        available=port_open("127.0.0.1", 54329),
    ),
    ContractTarget(
        name="weaviate",
        build=lambda schema: WeaviateAdapter.create(schema, WEAVIATE_CONNECTION),
        available=port_open("127.0.0.1", 8089),
    ),
    ContractTarget(
        name="pinecone-local",
        build=pinecone_local.LocalPineconeAdapter.build,
        available=pinecone_local.skip_reason,
    ),
    ContractTarget(name="azure-ai-search:fake", build=build_azure_fake),
]
