"""Migration lifecycle on a shared in-memory store (042 phase 4 review).

Covers the lifecycle lock (item 3), whole-collection migration and source
inventory (item 5), structured migration (item 8) and a strictly read-only
``inspect`` (item 10). Spec 042 FR-016, FR-023, AC-08.
"""

from __future__ import annotations

import builtins
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from holodeck.lib.retrieval.manifest import (
    MANIFEST_COLLECTION,
    MANIFEST_SCHEMA,
    CollectionEntry,
    LockRecord,
    ManifestStore,
    collection_entry_id,
)
from holodeck.lib.retrieval.schemas import text_field
from holodeck.lib.tool_initializer import close_tools, initialize_tools
from holodeck.lib.vectorstore_migration import (
    MigrationError,
    inspect_tool,
    migrate_tool,
    plan_migration,
    prune_tool,
    retrieval_tools,
    rollback_tool,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import DatabaseConfig
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from tests.integration.golden._golden import HashEmbeddingService
from tests.unit.tools.fakes import StoreRegistry

pytestmark = pytest.mark.asyncio

DIMS = 256
SHARED = f"documents_dim{DIMS}"


class Env:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = StoreRegistry()
        self.embeddings = HashEmbeddingService(DIMS)
        self.embed_calls = 0
        for folder in ("a", "b"):
            (root / folder).mkdir()
            for n in range(2):
                (root / folder / f"{folder}{n}.md").write_text(
                    "\n\n".join(
                        f"Paragraph {p} of {folder}{n} explains topic {p} at length."
                        for p in range(6)
                    )
                )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return await HashEmbeddingService.generate_embeddings(self.embeddings, texts)

    def patches(self) -> ExitStack:
        stack = ExitStack()
        for target in (
            "holodeck.lib.retrieval.create_adapter",
            "holodeck.lib.retrieval.registry.create_adapter",
        ):
            stack.enter_context(patch(target, side_effect=self.registry.create))
        stack.enter_context(
            patch.object(
                self.embeddings, "generate_embeddings", side_effect=self._embed
            )
        )
        stack.enter_context(
            patch(
                "holodeck.lib.tool_initializer.create_embedding_service",
                side_effect=lambda *a, **k: self.embeddings,
            )
        )
        return stack

    def manifest(self) -> ManifestStore:
        return ManifestStore(self.registry.create("in-memory", MANIFEST_SCHEMA))

    def entry(self, logical: str) -> CollectionEntry:
        record = self.registry.collections[MANIFEST_COLLECTION][
            collection_entry_id(logical)
        ]
        return CollectionEntry.from_record(record)

    def sources_in(self, name: str, field: str = "source_path") -> set[str]:
        return {text_field(r, field) for r in self.registry.records(name)}


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    environment = Env(tmp_path)
    with environment.patches():
        yield environment


def _agent(*tools: VectorstoreToolConfig) -> Agent:
    return Agent(
        name="migration_agent",
        model=LLMProvider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://127.0.0.1:11434",
        ),
        instructions=Instructions(inline="Test agent."),
        tools=list(tools),
    )


def _doc_tool(
    name: str, source: Path, chunk_size: int = 40, **database: Any
) -> VectorstoreToolConfig:
    return VectorstoreToolConfig(
        name=name,
        description=name,
        source=str(source),
        chunk_size=chunk_size,
        chunk_overlap=0,
        embedding_dimensions=DIMS,
        database=DatabaseConfig(provider="in-memory", **database),
    )


async def _start(agent: Agent) -> None:
    tools = await initialize_tools(agent)
    await close_tools(tools.values())


# -- item 5: the whole logical collection is migrated --


async def test_migration_rebuilds_every_tool_bound_to_the_collection(
    env: Env,
) -> None:
    old = _agent(_doc_tool("a", env.root / "a"), _doc_tool("b", env.root / "b"))
    await _start(old)
    assert len(env.sources_in(SHARED)) == 4
    new = _agent(_doc_tool("a", env.root / "a", 60), _doc_tool("b", env.root / "b", 60))
    tool_a, _ = retrieval_tools(new)

    plan = await plan_migration(new, tool_a)
    assert plan.logical_collection == SHARED and plan.tools == ["a", "b"]
    result = await migrate_tool(new, tool_a)

    assert result.tools == ["a", "b"]
    assert result.sources == 4
    assert env.sources_in(result.active_collection) == env.sources_in(SHARED)
    assert env.entry(SHARED).active_collection == result.active_collection


async def test_tools_sharing_a_collection_with_different_pipelines_are_refused(
    env: Env,
) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a"), _doc_tool("b", env.root / "b")))
    new = _agent(_doc_tool("a", env.root / "a", 60), _doc_tool("b", env.root / "b", 80))
    with pytest.raises(MigrationError, match="different ingestion pipelines"):
        await migrate_tool(new, retrieval_tools(new, "a")[0])
    assert env.entry(SHARED).active_collection == SHARED


async def test_sources_written_by_another_agent_block_the_switch(env: Env) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a"), _doc_tool("b", env.root / "b")))
    # This agent file only knows tool "a"; "b"'s sources would be dropped.
    only_a = _agent(_doc_tool("a", env.root / "a", 60))
    (tool_a,) = retrieval_tools(only_a)
    env.embed_calls = 0
    with pytest.raises(MigrationError, match="b0.md") as raised:
        await migrate_tool(only_a, tool_a)
    assert "--drop-unlisted-sources" in str(raised.value)
    assert env.embed_calls == 0  # refused before anything was embedded
    assert env.entry(SHARED).active_collection == SHARED

    result = await migrate_tool(only_a, tool_a, drop_unlisted_sources=True)
    assert sorted(Path(key).name for key in result.dropped_sources) == [
        "b0.md",
        "b1.md",
    ]
    assert env.entry(SHARED).active_collection == result.active_collection


async def test_a_source_that_fails_to_parse_fails_the_migration(env: Env) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a")))
    new = _agent(_doc_tool("a", env.root / "a", 60))
    (tool_a,) = retrieval_tools(new)
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    original = VectorStoreTool._process_file

    async def flaky(self: VectorStoreTool, path: Path) -> Any:
        return None if path.name == "a1.md" else await original(self, path)

    with (
        patch.object(VectorStoreTool, "_process_file", flaky),
        pytest.raises(MigrationError, match="a1.md"),
    ):
        await migrate_tool(new, tool_a)
    entry = env.entry(SHARED)
    assert entry.active_collection == SHARED and entry.state == "migrating"
    # Resumes once the source parses again; the finished one is not redone.
    env.embed_calls = 0
    result = await migrate_tool(new, tool_a)
    assert result.embedded_sources == 1
    assert env.entry(SHARED).active_collection == result.active_collection


async def test_a_removed_record_fails_verification(env: Env) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a")))
    new = _agent(_doc_tool("a", env.root / "a", 60))
    (tool_a,) = retrieval_tools(new)
    from holodeck.lib.vectorstore_migration import _verify as verify

    async def tampered(adapter: Any, *args: Any) -> None:
        first = (await adapter.list_page(limit=1)).records[0]
        await adapter.delete([first.id])
        await verify(adapter, *args)

    with (
        patch("holodeck.lib.vectorstore_migration._verify", tampered),
        pytest.raises(MigrationError, match="expected"),
    ):
        await migrate_tool(new, tool_a)
    assert env.entry(SHARED).active_collection == SHARED


# -- item 8: structured tools migrate, verify and resume --


async def test_structured_tool_migrates_with_row_markers(env: Env) -> None:
    rows = env.root / "rows.csv"
    rows.write_text(
        "id,title,body\n" + "".join(f"{n},t{n},body {n}\n" for n in range(7))
    )

    def structured(separator: str) -> VectorstoreToolConfig:
        return VectorstoreToolConfig(
            name="rows",
            description="rows",
            source=str(rows),
            vector_field=["title", "body"],
            field_separator=separator,
            id_field="id",
            embedding_dimensions=DIMS,
            database=DatabaseConfig(provider="in-memory"),
        )

    await _start(_agent(structured("\n")))
    new = _agent(structured(" | "))
    (config,) = retrieval_tools(new)
    result = await migrate_tool(new, config)
    assert result.record_count == 7 and result.sources == 1
    manifest = env.manifest()
    marker = (await manifest.sources(result.active_collection))[str(rows)]
    assert marker.complete and marker.chunk_count == 7
    # A cold start of the migrated collection embeds nothing.
    env.embed_calls = 0
    await _start(new)
    assert env.embed_calls == 0


# -- item 3: the lifecycle lock --


async def _seed_migrated(env: Env) -> tuple[Agent, VectorstoreToolConfig]:
    await _start(_agent(_doc_tool("a", env.root / "a")))
    new = _agent(_doc_tool("a", env.root / "a", 60))
    (tool_a,) = retrieval_tools(new)
    await migrate_tool(new, tool_a)
    return new, tool_a


@pytest.mark.parametrize("command", ["migrate", "rollback", "prune"])
async def test_a_held_lock_blocks_every_lifecycle_command(
    env: Env, command: str
) -> None:
    new, tool_a = await _seed_migrated(env)
    manifest = env.manifest()
    held = await manifest.acquire_lock(SHARED, "migrate", ttl=7200)
    run = {"migrate": migrate_tool, "rollback": rollback_tool, "prune": prune_tool}
    with pytest.raises(MigrationError, match="locked by a running 'migrate'"):
        await run[command](new, tool_a)
    assert await manifest.get_lock(SHARED) == held
    await manifest.release_lock(held)
    await run[command](new, tool_a)
    assert await manifest.get_lock(SHARED) is None  # released afterwards


async def test_a_stale_lock_is_taken_over(env: Env) -> None:
    new, tool_a = await _seed_migrated(env)
    manifest = env.manifest()
    stale = LockRecord(SHARED, "dead-owner", "migrate", 1.0, 2.0)
    await manifest.adapter.upsert([stale.to_record()])
    switched = await rollback_tool(new, tool_a)
    assert switched.active_collection == SHARED


async def test_the_lock_is_released_when_a_command_fails(env: Env) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a")))
    new = _agent(_doc_tool("a", env.root / "a", 60))
    (tool_a,) = retrieval_tools(new)
    with pytest.raises(MigrationError):
        await rollback_tool(new, tool_a)
    assert await env.manifest().get_lock(SHARED) is None


async def test_prune_reserves_the_collection_before_deleting_it(env: Env) -> None:
    new, tool_a = await _seed_migrated(env)
    kept = env.entry(SHARED).previous_collection
    assert kept == SHARED
    from tests.unit.tools.fakes import PersistentAdapter

    original = PersistentAdapter.delete_collection

    async def failing(self: PersistentAdapter) -> None:
        raise RuntimeError("delete failed midway")

    with (
        patch.object(PersistentAdapter, "delete_collection", failing),
        pytest.raises(RuntimeError, match="midway"),
    ):
        await prune_tool(new, tool_a)
    entry = env.entry(SHARED)
    assert entry.previous_collection == "" and entry.pending_delete == kept
    with pytest.raises(MigrationError, match="no kept previous collection"):
        await rollback_tool(new, tool_a)  # never points at a doomed collection

    with patch.object(PersistentAdapter, "delete_collection", original):
        result = await prune_tool(new, tool_a)
    assert result.deleted_collection == kept
    assert env.entry(SHARED).pending_delete == ""
    assert kept not in env.registry.existing


async def test_rollback_checks_the_kept_collection_under_the_lock(env: Env) -> None:
    new, tool_a = await _seed_migrated(env)
    kept = env.entry(SHARED).previous_collection
    env.registry.existing.discard(kept)
    with pytest.raises(MigrationError, match="no longer exists"):
        await rollback_tool(new, tool_a)
    assert env.entry(SHARED).active_collection != kept


# -- item 10: inspect is strictly read-only --


async def test_inspect_writes_downloads_and_embeds_nothing(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _start(_agent(_doc_tool("a", env.root / "a")))
    remote = VectorstoreToolConfig(
        name="remote",
        description="remote",
        source="https://example.invalid/docs.md",
        chunk_size=60,
        chunk_overlap=0,
        embedding_dimensions=DIMS,
        database=DatabaseConfig(provider="in-memory", collection_name="remote_kb"),
    )
    agent = _agent(_doc_tool("a", env.root / "a", 60), remote)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    import tiktoken

    tiktoken.get_encoding("cl100k_base")  # loaded before the spies

    env.registry.writes.clear()
    env.embed_calls = 0
    created_before = list(env.registry.created)
    existing_before = set(env.registry.existing)
    fs_writes: list[str] = []
    real_open = builtins.open

    def spy_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            fs_writes.append(f"open {file} {mode}")
        return real_open(file, mode, *args, **kwargs)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        fs_writes.append(f"write {args[:1]}")
        raise AssertionError("inspect wrote to the filesystem")

    def no_http(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("inspect made an HTTP call")

    with (
        patch("builtins.open", spy_open),
        patch.object(Path, "mkdir", refuse),
        patch.object(Path, "write_text", refuse),
        patch.object(Path, "write_bytes", refuse),
        patch.object(os, "makedirs", refuse),
        patch.object(os, "mkdir", refuse),
        patch.object(tempfile, "mkdtemp", refuse),
        patch.object(tempfile, "NamedTemporaryFile", refuse),
        patch.object(shutil, "copytree", refuse),
        patch.object(httpx.Client, "send", no_http),
        patch.object(httpx.AsyncClient, "send", no_http),
    ):
        reports = [await inspect_tool(agent, tool) for tool in retrieval_tools(agent)]

    assert fs_writes == []
    assert env.registry.writes == []
    assert env.registry.existing == existing_before
    assert env.embed_calls == 0
    assert not (workdir / ".holodeck").exists()
    new_adapters = env.registry.created[len(created_before) :]
    assert MANIFEST_COLLECTION in new_adapters  # built, never written
    local, remote_report = reports
    assert local.source_files == 2 and local.chunks > 0
    assert remote_report.remote_sources == ["https://example.invalid/docs.md"]
    assert remote_report.source_files == 0 and remote_report.chunks == 0
