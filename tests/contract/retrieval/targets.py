"""Adapters that the contract suite runs against.

Add one ``ContractTarget`` per adapter. ``available`` returns a skip reason
when the store cannot be reached, or None when the target can run.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from dataclasses import dataclass

from holodeck.lib.retrieval import CollectionSchema, VectorStoreAdapter
from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter
from holodeck.lib.retrieval.adapters.sk_bridge import (
    SKChromaAdapter,
    SKInMemoryAdapter,
    SKQdrantAdapter,
)

QDRANT_URL = "http://127.0.0.1:6333"
CHROMA_URL = "http://127.0.0.1:8765"


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


TARGETS: list[ContractTarget] = [
    ContractTarget(name="in-memory", build=InMemoryAdapter),
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
]
