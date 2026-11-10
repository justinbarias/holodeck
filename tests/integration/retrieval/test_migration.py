"""Collection migration: dry run, interruption, resume, rollback, prune (AC-08).

Spec 042 FR-023. Runs on a persistent in-memory store (a shared registry of
fake adapters) and on the Qdrant, pgvector, Chroma, Weaviate and Cosmos DB
emulator containers; a container that is not reachable is skipped. The
offline hash embedding stands in for the model.

The old collection is written with ``chunk_size=40``; the agent then asks for
``chunk_size=60``, so its pipeline fingerprint differs and a migration is
needed.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from holodeck.lib.retrieval import CollectionSchema, VectorStoreAdapter, registry
from holodeck.lib.retrieval.manifest import MANIFEST_COLLECTION, ManifestStore
from holodeck.lib.retrieval.schemas import document_schema
from holodeck.lib.tool_initializer import close_tools, initialize_tools
from holodeck.lib.vectorstore_migration import (
    MigrationError,
    inspect_tool,
    migrate_tool,
    prune_tool,
    retrieval_tools,
    rollback_tool,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import DatabaseConfig
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from tests.contract.retrieval import cosmos_emulator
from tests.contract.retrieval.targets import (
    CHROMA_URL,
    POSTGRES_URL,
    QDRANT_URL,
    WEAVIATE_CONNECTION,
    port_open,
)
from tests.integration.golden._golden import RETRIEVAL_ROOT, HashEmbeddingService
from tests.unit.tools.fakes import StoreRegistry

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

DIMS = 256
CORPUS = RETRIEVAL_ROOT / "corpus"
TARGETS = [
    "memory",
    "qdrant",
    "postgres",
    "chromadb",
    "weaviate",
    "azure-cosmos-nosql",
]
WRITE_METHODS = (
    "upsert",
    "delete",
    "delete_where",
    "replace_source",
    "ensure_collection",
    "delete_collection",
)


class SpyEmbeddings(HashEmbeddingService):
    """Hash embedding that counts calls and can fail after some calls."""

    def __init__(self, dimensions: int = DIMS) -> None:
        super().__init__(dimensions)
        self.calls = 0
        self.texts: list[str] = []
        self.fail_after: int | None = None

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("embedding service unavailable")
        self.calls += 1
        self.texts.extend(texts)
        return await super().generate_embeddings(texts)


class Store:
    """One test target: provider settings, patches, spies and cleanup."""

    def __init__(self, target: str, collection: str) -> None:
        self.target = target
        self.collection = collection
        self.embeddings = SpyEmbeddings()
        self.writes: list[tuple[str, str]] = []
        self._memory = StoreRegistry() if target == "memory" else None
        self._real_create = registry.create_adapter

    @property
    def provider(self) -> str:
        return {"memory": "in-memory"}.get(self.target, self.target)

    def database(self) -> DatabaseConfig:
        if self.target == "memory":
            return DatabaseConfig(provider="in-memory", collection_name=self.collection)
        if self.target == "azure-cosmos-nosql":
            return DatabaseConfig(
                provider=self.target,
                collection_name=self.collection,
                **cosmos_emulator.connection(),
            )
        if self.target == "weaviate":
            return DatabaseConfig(
                provider=self.target,
                collection_name=self.collection,
                **WEAVIATE_CONNECTION,
            )
        urls = {"qdrant": QDRANT_URL, "postgres": POSTGRES_URL, "chromadb": CHROMA_URL}
        return DatabaseConfig(
            provider=self.target,
            connection_string=SecretStr(urls[self.target]),
            collection_name=self.collection,
        )

    def connection(self) -> dict[str, object]:
        database = self.database()
        settings: dict[str, object] = dict(database.model_extra or {})
        if database.connection_string is not None:
            settings["connection_string"] = (
                database.connection_string.get_secret_value()
            )
        return settings

    def create(
        self,
        provider: str,
        schema: CollectionSchema,
        connection: Any = None,
    ) -> VectorStoreAdapter:
        """Build an adapter that records every write call."""
        adapter: VectorStoreAdapter
        if self._memory is not None:
            adapter = self._memory.create(provider, schema, connection)
        else:
            adapter = self._real_create(provider, schema, connection)
        for name in WRITE_METHODS:
            original = getattr(adapter, name)

            def recorder(*args: Any, _o: Any = original, _n: str = name, **kw: Any):  # type: ignore[no-untyped-def]
                self.writes.append((schema.name, _n))
                return _o(*args, **kw)

            setattr(adapter, name, recorder)
        return adapter

    def patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch("holodeck.lib.retrieval.create_adapter", side_effect=self.create)
        )
        stack.enter_context(
            patch(
                "holodeck.lib.retrieval.registry.create_adapter",
                side_effect=self.create,
            )
        )
        stack.enter_context(
            patch(
                "holodeck.lib.tool_initializer.create_embedding_service",
                side_effect=lambda *a, **k: self.embeddings,
            )
        )
        return stack

    def adapter(self, name: str) -> VectorStoreAdapter:
        """An adapter for collection ``name`` (no collection-name setting)."""
        return self.create(
            self.provider,
            document_schema(DIMS, {"collection_name": name}),
            self.connection(),
        )

    async def manifest(self) -> ManifestStore:
        store = ManifestStore.open(self.provider, self.connection())
        assert store is not None
        return store

    async def count(self, name: str) -> int:
        adapter = self.adapter(name)
        try:
            return await adapter.count() if await adapter.collection_exists() else 0
        finally:
            await adapter.close()

    async def exists(self, name: str) -> bool:
        adapter = self.adapter(name)
        try:
            return await adapter.collection_exists()
        finally:
            await adapter.close()

    async def cleanup(self, names: Sequence[str]) -> None:
        manifest = await self.manifest()
        try:
            await manifest.delete_collection_entry(self.collection)
            for name in names:
                await manifest.delete_sources(name)
        finally:
            await manifest.close()
        for name in names:
            adapter = self.adapter(name)
            try:
                await adapter.delete_collection()
            finally:
                await adapter.close()


def _skip_reason(target: str) -> str | None:
    if target == "qdrant":
        return port_open("127.0.0.1", 6333)()
    if target == "postgres":
        return port_open("127.0.0.1", 54329)()
    if target == "chromadb":
        return port_open("127.0.0.1", 8765)()
    if target == "weaviate":
        return port_open("127.0.0.1", 8089)()
    if target == "azure-cosmos-nosql":
        return cosmos_emulator.skip_reason()
    return None


@pytest.fixture(params=TARGETS)
def store(request: pytest.FixtureRequest) -> Iterator[Store]:
    reason = _skip_reason(request.param)
    if reason is not None:
        pytest.skip(reason)
    collection = f"mig_{uuid.uuid4().hex[:10]}"
    target = Store(request.param, collection)
    with target.patches():
        yield target


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    target = tmp_path / "corpus"
    shutil.copytree(CORPUS, target)
    return target


def _agent(store: Store, corpus: Path, chunk_size: int) -> Agent:
    return Agent(
        name="migration_agent",
        model=LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://127.0.0.1:11434",
        ),
        instructions=Instructions(inline="Test agent."),
        tools=[
            VectorstoreToolConfig(
                name="kb",
                description="Knowledge base",
                source=str(corpus),
                chunk_size=chunk_size,
                chunk_overlap=0,
                embedding_dimensions=DIMS,
                top_k=3,
                database=store.database(),
            )
        ],
    )


async def _search(store: Store, agent: Agent, query: str) -> str:
    tools = await initialize_tools(agent)
    try:
        return str(await tools["kb"].search(query))
    finally:
        await close_tools(tools.values())


async def _ingest_old(store: Store, corpus: Path) -> tuple[Agent, Agent, int]:
    old = _agent(store, corpus, chunk_size=40)
    new = _agent(store, corpus, chunk_size=60)
    tools = await initialize_tools(old)
    await close_tools(tools.values())
    return old, new, await store.count(store.collection)


async def test_full_migration_cycle(store: Store, corpus: Path) -> None:
    old, new, old_count = await _ingest_old(store, corpus)
    (tool_config,) = retrieval_tools(new, "kb")
    logical = store.collection
    try:
        assert old_count > 0
        old_answer = await _search(store, old, "kestrel dawn")
        assert "Found" in old_answer

        # -- dry run: read-only, no model call ------------------------------
        manifest = await store.manifest()
        before = await manifest.get_collection(logical)
        await manifest.close()
        store.writes.clear()
        store.embeddings.calls = 0
        report = await inspect_tool(new, tool_config)
        assert store.writes == []
        assert store.embeddings.calls == 0
        assert report.compatible == "no"
        assert report.needs_reembedding
        assert report.exists and report.record_count == old_count
        assert ("chunker.chunk_size", 40, 60) in report.differences
        assert report.source_files == 3
        assert report.chunks > 0 and report.estimated_tokens > report.chunks
        manifest = await store.manifest()
        assert await manifest.get_collection(logical) == before
        await manifest.close()

        # -- writes with the new pipeline are refused --
        target = f"{logical}__{report.current_fingerprint[:8]}"
        changed = corpus / "legislation.md"
        stat = changed.stat()
        os.utime(changed, (stat.st_atime, stat.st_mtime + 10))
        with pytest.raises(Exception, match="refuses every write"):
            await initialize_tools(new)
        os.utime(changed, (stat.st_atime, stat.st_mtime))

        # -- interrupted migration --
        store.embeddings.calls = 0
        store.embeddings.texts.clear()
        store.embeddings.fail_after = 1
        with pytest.raises(RuntimeError, match="embedding service unavailable"):
            await migrate_tool(new, tool_config)
        first_texts = list(store.embeddings.texts)
        assert first_texts
        manifest = await store.manifest()
        entry = await manifest.get_collection(logical)
        sources = await manifest.sources(target)
        await manifest.close()
        assert entry is not None and entry.state == "migrating"
        assert entry.active_collection == logical
        assert entry.migration_target == target
        assert [s.complete for s in sources.values()] == [True]
        assert await store.count(logical) == old_count

        # -- resume: the finished source is not embedded again --
        store.embeddings.fail_after = None
        store.embeddings.texts.clear()
        result = await migrate_tool(new, tool_config)
        assert not set(first_texts) & set(store.embeddings.texts)
        assert result.active_collection == target
        assert result.previous_collection == logical
        assert result.sources == 3 and result.embedded_sources == 2
        assert await store.count(target) == result.record_count
        assert await store.count(logical) == old_count  # old one untouched

        # -- the migrated collection is compatible: no re-embedding on start ---------
        store.embeddings.calls = 0
        new_answer = await _search(store, new, "kestrel dawn")
        assert store.embeddings.calls == 1  # the query only
        assert "Found" in new_answer
        assert (await inspect_tool(new, tool_config)).compatible == "yes"

        # -- rollback restores the old collection and its results --
        switched = await rollback_tool(new, tool_config)
        assert switched.active_collection == logical
        assert switched.previous_collection == target
        store.embeddings.calls = 0
        assert await _search(store, old, "kestrel dawn") == old_answer
        assert store.embeddings.calls == 1

        # -- prune deletes only the kept collection --
        pruned = await prune_tool(new, tool_config)
        assert pruned.deleted_collection == target
        assert not await store.exists(target)
        assert await store.count(logical) == old_count
        manifest = await store.manifest()
        assert await manifest.sources(target) == {}
        final = await manifest.get_collection(logical)
        await manifest.close()
        assert final is not None and final.previous_collection == ""
        with pytest.raises(MigrationError, match="no kept previous collection"):
            await rollback_tool(new, tool_config)
    finally:
        names = [logical]
        report_fp = (await inspect_tool(new, tool_config)).current_fingerprint
        names.append(f"{logical}__{report_fp[:8]}")
        await store.cleanup(names)


async def test_migrating_a_compatible_collection_is_a_no_op(
    store: Store, corpus: Path
) -> None:
    agent = _agent(store, corpus, chunk_size=60)
    tools = await initialize_tools(agent)
    await close_tools(tools.values())
    (tool_config,) = retrieval_tools(agent, "kb")
    try:
        store.embeddings.calls = 0
        result = await migrate_tool(agent, tool_config)
        assert result.skipped
        assert "already matches" in result.message
        assert store.embeddings.calls == 0
        # A restart of a compatible collection embeds nothing either.
        tools = await initialize_tools(agent)
        await close_tools(tools.values())
        assert store.embeddings.calls == 0
    finally:
        await store.cleanup([store.collection])


async def test_migration_for_another_pipeline_in_progress_needs_restart(
    store: Store, corpus: Path
) -> None:
    old, new, _ = await _ingest_old(store, corpus)
    other = _agent(store, corpus, chunk_size=80)
    (new_tool,) = retrieval_tools(new, "kb")
    (other_tool,) = retrieval_tools(other, "kb")
    names = [store.collection]
    try:
        store.embeddings.fail_after = 0
        with pytest.raises(RuntimeError):
            await migrate_tool(new, new_tool)
        store.embeddings.fail_after = None
        with pytest.raises(MigrationError, match="--restart"):
            await migrate_tool(other, other_tool)
        result = await migrate_tool(other, other_tool, restart=True)
        names.append(result.active_collection)
        names.append(
            f"{store.collection}__"
            f"{(await inspect_tool(new, new_tool)).current_fingerprint[:8]}"
        )
        assert result.active_collection.startswith(f"{store.collection}__")
    finally:
        await store.cleanup(names)


async def test_manifest_collection_is_shared_per_connection(
    store: Store, corpus: Path
) -> None:
    agent = _agent(store, corpus, chunk_size=60)
    tools = await initialize_tools(agent)
    await close_tools(tools.values())
    try:
        assert store.create  # manifest lives next to the tool's collection
        manifest = await store.manifest()
        try:
            assert await manifest.exists()
            assert manifest.adapter.schema.name == MANIFEST_COLLECTION
            entry = await manifest.get_collection(store.collection)
            assert entry is not None and entry.active_collection == store.collection
        finally:
            await manifest.close()
    finally:
        await store.cleanup([store.collection])
