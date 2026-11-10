"""Cosmos DB emulator support for tests.

The Linux ``vnext-preview`` emulator runs vector search, filters, point reads
and paging. It lacks full-text search (see ``cosmos_fake.py``), so the contract
target on the emulator uses ``full_text_search: false`` and the ``diskANN``
index, the only vector index it accepts.

Start it with (the key below is the published emulator key, not a secret)::

    docker run -d --name holodeck-verify-cosmos \\
      -p 127.0.0.1:8081:8081 -p 127.0.0.1:1234:1234 \\
      mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview \\
      --protocol http
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import cache
from urllib.parse import urlparse
from uuid import uuid4

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosResourceExistsError

EMULATOR_URL = os.environ.get("HOLODECK_COSMOS_EMULATOR_URL", "http://127.0.0.1:8081")
# The documented, public key of every Cosmos DB emulator.
EMULATOR_KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="  # noqa: E501
DATABASE = "holodeck_contract"


def connection() -> Mapping[str, object]:
    """Return database settings for the emulator."""
    return {
        "url": EMULATOR_URL,
        "key": EMULATOR_KEY,
        "database_name": DATABASE,
        "create_database": True,
        "vector_index": "diskANN",
    }


def _reachable() -> bool:
    parsed = urlparse(EMULATOR_URL)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 8081), timeout=0.5
        ):
            return True
    except OSError:
        return False


async def _vector_search_works() -> str | None:
    """Return None when the emulator can create a vector container."""
    try:
        async with CosmosClient(EMULATOR_URL, credential=EMULATOR_KEY) as client:
            database = client.get_database_client(DATABASE)
            with suppress(CosmosResourceExistsError):
                await client.create_database_if_not_exists(DATABASE)
            name = f"probe_{uuid4().hex[:12]}"
            await database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path="/id"),
                indexing_policy={
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [{"path": '/"embedding"/*'}],
                    "vectorIndexes": [{"path": '/"embedding"', "type": "diskANN"}],
                },
                vector_embedding_policy={
                    "vectorEmbeddings": [
                        {
                            "path": '/"embedding"',
                            "dataType": "float32",
                            "distanceFunction": "cosine",
                            "dimensions": 4,
                        }
                    ]
                },
            )
            await database.delete_container(name)
    except Exception as error:  # any failure means the target cannot run
        return f"Cosmos emulator at {EMULATOR_URL} lacks vector search: {error}"
    return None


@cache
def skip_reason() -> str | None:
    """Return a skip reason when the emulator is down or lacks vector search."""
    if not _reachable():
        return f"Cosmos emulator not reachable at {EMULATOR_URL}"
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _vector_search_works()).result(timeout=60)
