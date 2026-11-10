"""Freeze the Qdrant, Chroma and Postgres layouts written by the SK path (042 T16).

One-off generator, run while Semantic Kernel and the SK bridge
(``holodeck.lib.retrieval.adapters.sk_bridge``) still existed. For each store
and each schema in the store's compat test module it creates a collection
through the SK bridge, dumps the raw store state with the store's own client,
and records what the bridge returned for every read the tools make. For the
all-types schema it also applies the native adapter's writes to a second SK
collection and records what the bridge read back.

Output: ``tests/fixtures/legacy_layout/<store>/<schema>.json``. Needs the live
containers (Qdrant :6333, Chroma :8765, Postgres :54329). Run from the
repository root::

    uv run python -m scripts.freeze_legacy_layouts
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from typing import Any

import chromadb
import psycopg
from qdrant_client import AsyncQdrantClient

from holodeck.lib.retrieval.adapters.sk_bridge import (
    SKChromaAdapter,
    SKPostgresAdapter,
    SKQdrantAdapter,
)

# The schemas, corpora and reads live in the compat tests, so the fixtures and
# the tests that check them cannot drift apart.
ll: Any = import_module("tests.integration.retrieval.legacy_layout")
chroma_tests: Any = import_module("tests.integration.retrieval.test_chroma_compat")
postgres_tests: Any = import_module("tests.integration.retrieval.test_postgres_compat")
qdrant_tests: Any = import_module("tests.integration.retrieval.test_qdrant_compat")


class FixedScoreBridge(SKPostgresAdapter):
    """SK's Postgres cosine search returns a distance; read it as one."""

    SCORE_IS_DISTANCE = True


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - the developer's git
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _name(kind: str) -> str:
    return f"legacy_{kind}_{uuid.uuid4().hex[:8]}"


def _meta(store: str, server: str, client: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).date().isoformat(),
        "generator": "scripts/freeze_legacy_layouts.py",
        "base_commit": _commit(),
        "semantic_kernel": version("semantic-kernel"),
        "store": store,
        "server_version": server,
        "client": client,
    }


async def freeze_qdrant() -> None:
    url = qdrant_tests.QDRANT_URL
    raw = AsyncQdrantClient(url=url)
    info = await raw.info()
    meta = _meta("qdrant", info.version, f"qdrant-client {version('qdrant-client')}")
    connection = {"connection_string": url}
    try:
        for kind, build in qdrant_tests.SCHEMAS.items():
            schema = build(_name(kind))
            bridge = SKQdrantAdapter.create(schema, connection)
            try:
                await bridge.ensure_collection()
                await bridge.upsert(qdrant_tests.corpus(kind, schema))
                data: dict[str, Any] = {
                    "meta": meta,
                    "collection": schema.name,
                    "schema": ll.schema_facts(schema),
                    "layout": await ll.qdrant_layout(raw, schema.name),
                    "rows": await ll.qdrant_rows(raw, schema.name),
                    "expected": await qdrant_tests.read_all(kind, bridge, schema),
                }
            finally:
                await bridge.delete_collection()
                await bridge.close()
            if kind == "all_types":
                data["after_writes"] = await _qdrant_after_writes(build, connection)
            ll.save("qdrant", kind, data)
    finally:
        await raw.close()


async def _qdrant_after_writes(build: Any, connection: dict[str, str]) -> Any:
    schema = build(_name("writes"))
    bridge = SKQdrantAdapter.create(schema, connection)
    native = qdrant_tests.QdrantAdapter.create(schema, connection)
    try:
        await bridge.ensure_collection()
        await bridge.upsert(qdrant_tests.CORPUS)
        await qdrant_tests.native_writes(native)
        return await qdrant_tests.snapshot_after_writes(bridge)
    finally:
        await native.close()
        await bridge.delete_collection()
        await bridge.close()


async def freeze_chroma() -> None:
    connection = {"connection_string": chroma_tests.CHROMA_URL}
    client = chromadb.HttpClient(
        host=chroma_tests.CHROMA_HOST, port=chroma_tests.CHROMA_PORT
    )
    meta = _meta(
        "chroma (HTTP)", client.get_version(), f"chromadb {version('chromadb')}"
    )
    for kind, build in chroma_tests.SCHEMAS.items():
        schema = build(_name(kind))
        bridge = SKChromaAdapter.create(schema, connection)
        try:
            await bridge.ensure_collection()
            await bridge.upsert(chroma_tests.corpus(kind, schema))
            collection = client.get_collection(schema.name, embedding_function=None)
            data: dict[str, Any] = {
                "meta": meta,
                "collection": schema.name,
                "schema": ll.schema_facts(schema),
                "layout": ll.chroma_layout(collection),
                "rows": ll.chroma_rows(collection),
                "expected": await chroma_tests.read_all(kind, bridge, schema),
            }
        finally:
            await bridge.delete_collection()
            await bridge.close()
        if kind == "all_types":
            schema = build(_name("writes"))
            bridge = SKChromaAdapter.create(schema, connection)
            native = chroma_tests.ChromaAdapter.create(schema, connection)
            try:
                await bridge.ensure_collection()
                await bridge.upsert(chroma_tests.CORPUS)
                await chroma_tests.native_writes(native)
                data["after_writes"] = await chroma_tests.snapshot_after_writes(bridge)
            finally:
                await native.close()
                await bridge.delete_collection()
                await bridge.close()
        ll.save("chroma", kind, data)


async def freeze_postgres() -> None:
    # SK's PostgresSettings replaces the port of a connection string with 5432.
    os.environ["PGPORT"] = "54329"
    url = postgres_tests.POSTGRES_URL
    connection = {"connection_string": url}
    with psycopg.connect(url) as db:
        server = db.execute("SHOW server_version").fetchone()
        extension = db.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if server is None or extension is None:
        raise RuntimeError("pgvector is not installed in the Postgres container")
    meta = _meta(
        "postgres",
        f"PostgreSQL {server[0]}, pgvector {extension[0]}",
        f"psycopg {version('psycopg')}",
    )
    for kind, build in postgres_tests.SCHEMAS.items():
        schema = build(_name(kind))
        bridge = FixedScoreBridge.create(schema, connection)
        try:
            await bridge.ensure_collection()
            await bridge.upsert(postgres_tests.corpus(kind, schema))
            data: dict[str, Any] = {
                "meta": meta,
                "collection": schema.name,
                "schema": ll.schema_facts(schema),
                "layout": ll.postgres_layout(url, schema.name),
                "rows": ll.postgres_rows(url, schema.name),
                "expected": await postgres_tests.read_all(kind, bridge, schema),
            }
        finally:
            await bridge.delete_collection()
            await bridge.close()
        if kind == "all_types":
            schema = build(_name("writes"))
            bridge = FixedScoreBridge.create(schema, connection)
            native = postgres_tests.native(schema)
            try:
                await bridge.ensure_collection()
                await bridge.upsert(postgres_tests.CORPUS)
                await postgres_tests.native_writes(native)
                data["after_writes"] = await postgres_tests.snapshot_after_writes(
                    bridge
                )
            finally:
                await native.close()
                await bridge.delete_collection()
                await bridge.close()
        ll.save("postgres", kind, data)


async def main() -> None:
    await freeze_qdrant()
    await freeze_chroma()
    await freeze_postgres()


if __name__ == "__main__":
    asyncio.run(main())
