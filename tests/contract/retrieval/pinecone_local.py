"""Pinecone Local support for tests.

The ``pinecone-io/pinecone-local`` image (``latest`` is 1.0.0-rc0 of February
2025; the project reached end of life on 2026-10-07) is the only stateful
emulator, and it serves the pre-10 control plane: ``create`` takes
``dimension`` and ``spec``, and ``describe`` has no ``schema``. The 10.x client
cannot create, describe or delete an index there. Its data plane (upsert,
fetch, list, query, stats) works with the 10.x client unchanged, except
``fetch_by_metadata`` (404) and delete by filter (400).

``LocalPineconeAdapter`` therefore replaces only the three control plane
methods of ``PineconeAdapter`` with raw HTTP calls. Everything else, which is
all of the data logic, is the production code on the 10.x client.

Start it with::

    docker run -d --name holodeck-verify-pinecone -e PORT=5081 \\
      -e PINECONE_HOST=localhost -p 127.0.0.1:5081-5090:5081-5090 \\
      --platform linux/amd64 ghcr.io/pinecone-io/pinecone-local:latest
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import httpx

from holodeck.lib.retrieval import CollectionSchema, StoreConnectionError
from holodeck.lib.retrieval.adapters.pinecone import (
    IndexInfo,
    PineconeAdapter,
    PineconeSettings,
)

LOCAL_URL = os.environ.get("HOLODECK_PINECONE_LOCAL_URL", "http://127.0.0.1:5081")
LOCAL_API_KEY = "pclocal"  # Pinecone Local accepts any key


def local_reachable() -> bool:
    """Return True when the Pinecone Local control plane port accepts connections."""
    parsed = urlparse(LOCAL_URL)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 5081), timeout=0.5
        ):
            return True
    except OSError:
        return False


def skip_reason() -> str | None:
    """Return a skip reason when Pinecone Local is not running."""
    return None if local_reachable() else f"Pinecone Local not reachable at {LOCAL_URL}"


class LocalPineconeAdapter(PineconeAdapter):
    """``PineconeAdapter`` with a control plane for Pinecone Local 1.0."""

    @classmethod
    def build(cls, schema: CollectionSchema) -> LocalPineconeAdapter:
        """Return an adapter on the Pinecone Local started by the docs above."""
        return cls(schema, PineconeSettings(api_key=LOCAL_API_KEY))

    @classmethod
    def create(  # type: ignore[override]
        cls, schema: CollectionSchema, connection: object
    ) -> LocalPineconeAdapter:
        """Registry entry point; the connection settings are ignored."""
        return cls.build(schema)

    async def _describe(self) -> IndexInfo | None:
        async with httpx.AsyncClient(headers={"Api-Key": LOCAL_API_KEY}) as http:
            response = await http.get(f"{LOCAL_URL}/indexes/{self.index_name}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        return IndexInfo(
            host=f"http://{body['host']}",
            dimension=int(body["dimension"]),
            metric=str(body["metric"]),
        )

    async def _create_index(self) -> None:
        async with httpx.AsyncClient(headers={"Api-Key": LOCAL_API_KEY}) as http:
            response = await http.post(
                f"{LOCAL_URL}/indexes",
                json={
                    "name": self.index_name,
                    "dimension": self.schema.dimensions,
                    "metric": self.schema.distance.value,
                    "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
                },
            )
        if response.status_code not in (201, 409):
            raise StoreConnectionError(
                "pinecone", f"Pinecone Local refused the index: {response.text}"
            )

    async def _drop_index(self) -> None:
        async with httpx.AsyncClient(headers={"Api-Key": LOCAL_API_KEY}) as http:
            await http.delete(f"{LOCAL_URL}/indexes/{self.index_name}")
        for _ in range(50):
            if await self._describe() is None:
                return
            await asyncio.sleep(0.1)
