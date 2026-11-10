"""Concurrent starts, single-snapshot binds, logical identity and manifest
failures of the pipeline guard (042 phase 4 review).

Spec 042 FR-016, FR-023, AC-06, AC-08.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from holodeck.lib.retrieval import IncompatibleCollectionError
from holodeck.lib.retrieval.manifest import (
    MANIFEST_COLLECTION,
    CollectionEntry,
    ManifestStore,
    collection_entry_id,
)
from holodeck.models.tool import DatabaseConfig
from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
from holodeck.tools.vectorstore_tool import VectorStoreTool
from tests.unit.tools.fakes import (
    CountingEmbeddings,
    PersistentAdapter,
    StoreRegistry,
)

DIMS = 8
COLLECTION = f"documents_dim{DIMS}"


@pytest.fixture
def store() -> Iterator[StoreRegistry]:
    registry = StoreRegistry()
    with (
        patch("holodeck.lib.retrieval.create_adapter", side_effect=registry.create),
        patch(
            "holodeck.lib.retrieval.registry.create_adapter",
            side_effect=registry.create,
        ),
        patch("holodeck.lib.retrieval.manifest._warned", set()),
    ):
        yield registry


def _text(paragraphs: int) -> str:
    return "\n\n".join(
        f"Paragraph {n} talks about subject {n} in enough words to fill a chunk."
        for n in range(paragraphs)
    )


def _tool(
    source: Path,
    chunk_size: int = 40,
    database: DatabaseConfig | None = None,
    embeddings: CountingEmbeddings | None = None,
) -> VectorStoreTool:
    tool = VectorStoreTool(
        VectorstoreToolConfig(
            name="kb",
            description="Knowledge base",
            source=str(source),
            chunk_size=chunk_size,
            chunk_overlap=0,
            embedding_dimensions=DIMS,
            database=database,
        )
    )
    tool.set_embedding_service(embeddings or CountingEmbeddings(DIMS))
    tool.set_embedding_identity("ollama", "nomic-embed-text")
    return tool


def _entry(store: StoreRegistry, logical: str) -> CollectionEntry:
    records = store.collections[MANIFEST_COLLECTION]
    return CollectionEntry.from_record(records[collection_entry_id(logical)])


def _fingerprint(source: Path, chunk_size: int) -> tuple[str, str]:
    tool = _tool(source, chunk_size)
    tool._setup_collection("ollama")
    inputs = tool.pipeline_inputs()
    return inputs.fingerprint, inputs.canonical_json()


# -- item 1: concurrent first starts --


@pytest.mark.asyncio
@pytest.mark.parametrize("same_pipeline", [False, True])
async def test_a_start_that_loses_the_entry_race_follows_the_winner(
    tmp_path: Path, store: StoreRegistry, same_pipeline: bool
) -> None:
    """Both processes see no entry and write one; the read-back decides."""
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    other_size = 40 if same_pipeline else 60
    other_fp, other_inputs = _fingerprint(source, other_size)
    original = ManifestStore.put_collection
    raced = False

    async def racing_put(self: ManifestStore, entry: CollectionEntry) -> Any:
        nonlocal raced
        written = await original(self, entry)
        if not raced:
            raced = True  # the other process writes right after this one
            await original(
                self,
                CollectionEntry(entry.logical, other_fp, other_inputs, COLLECTION),
            )
        return written

    tool = _tool(source)
    with patch.object(ManifestStore, "put_collection", racing_put):
        if same_pipeline:
            await tool.initialize(provider_type="ollama")
            assert store.records(COLLECTION)
        else:
            with pytest.raises(IncompatibleCollectionError, match="chunk_size"):
                await tool.initialize(provider_type="ollama")
            assert store.records(COLLECTION) == []
    assert _entry(store, COLLECTION).fingerprint == other_fp


@pytest.mark.asyncio
async def test_guard_rechecks_the_entry_before_the_first_write_of_each_source(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    tool = _tool(source)
    tool._setup_collection("ollama")
    guard = await tool._bind_pipeline("kb")
    assert guard.writable and guard.manifest is not None
    reads: list[Sequence[str]] = []
    adapter = guard.manifest.adapter
    original_get = adapter.get

    async def counting_get(ids: Sequence[str], **kwargs: Any) -> Any:
        reads.append(list(ids))
        return await original_get(ids, **kwargs)

    with patch.object(adapter, "get", counting_get):
        await guard.begin_source("a.md", 1, "m")
        await guard.begin_source("a.md", 1, "m")
        await guard.begin_source("b.md", 1, "m")
    assert reads == [[collection_entry_id(COLLECTION)]] * 2  # one read per source

    # Another process records a different pipeline after this bind.
    other_fp, other_inputs = _fingerprint(source, 60)
    await guard.manifest.put_collection(
        CollectionEntry(COLLECTION, other_fp, other_inputs, COLLECTION)
    )
    with pytest.raises(IncompatibleCollectionError, match="after this tool started"):
        await guard.begin_source("c.md", 1, "m")
    with pytest.raises(IncompatibleCollectionError):
        await guard.begin_source("a.md", 1, "m")  # refused from now on


# -- item 2: one snapshot; a changed revision re-binds once, then refuses --


@pytest.mark.asyncio
async def test_a_rewritten_entry_with_the_same_pointer_and_pipeline_rebinds(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    tool = _tool(source)
    tool._setup_collection("ollama")
    guard = await tool._bind_pipeline("kb")
    assert guard.manifest is not None
    entry = await guard.manifest.get_collection(COLLECTION)
    assert entry is not None and entry.revision == guard.revision
    rewritten = await guard.manifest.put_collection(entry)
    await guard.begin_source("a.md", 1, "m")
    assert guard.revision == rewritten.revision


@pytest.mark.asyncio
async def test_a_pointer_moved_between_bind_and_first_write_refuses(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(4))
    tool = _tool(source)
    tool._setup_collection("ollama")
    guard = await tool._bind_pipeline("kb")
    assert guard.manifest is not None
    entry = await guard.manifest.get_collection(COLLECTION)
    assert entry is not None
    await guard.manifest.put_collection(
        CollectionEntry(
            COLLECTION,
            entry.fingerprint,
            entry.fingerprint_inputs,
            f"{COLLECTION}__migrated",
        )
    )
    with pytest.raises(IncompatibleCollectionError, match="changed while this tool"):
        await guard.begin_source("a.md", 1, "m")


@pytest.mark.asyncio
async def test_physical_name_and_fingerprint_come_from_one_manifest_read(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    fingerprint, inputs = _fingerprint(source, 40)
    physical = f"{COLLECTION}__0123abcd"
    seed = ManifestStore(store.create("in-memory", _manifest_schema()))
    await seed.put_collection(
        CollectionEntry(COLLECTION, fingerprint, inputs, physical)
    )
    tool = _tool(source)
    tool._setup_collection("ollama")
    original = ManifestStore.get_collection
    reads: list[str] = []

    async def counting(self: ManifestStore, logical: str) -> Any:
        reads.append(logical)
        return await original(self, logical)

    with patch.object(ManifestStore, "get_collection", counting):
        guard = await tool._bind_pipeline("kb")
    assert guard.physical == physical and guard.writable
    assert tool.physical_collection == physical
    assert reads == [COLLECTION]


def _manifest_schema():  # type: ignore[no-untyped-def]
    from holodeck.lib.retrieval.manifest import MANIFEST_SCHEMA

    return MANIFEST_SCHEMA


# -- item 7: the logical identity is the effective provider name --


@pytest.mark.asyncio
async def test_name_overrides_give_separate_or_shared_entries(
    tmp_path: Path, store: StoreRegistry
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(3))

    def database(**names: str) -> DatabaseConfig:
        return DatabaseConfig(provider="in-memory", **names)

    first = _tool(source, database=database(index_name="idx-a"))
    second = _tool(source, database=database(index_name="idx-b"))
    third = _tool(source, chunk_size=60, database=database(index_name="idx-a"))
    await first.initialize(provider_type="ollama")
    await second.initialize(provider_type="ollama")
    assert first.logical_collection == "idx-a"
    assert second.logical_collection == "idx-b"
    manifest = store.collections[MANIFEST_COLLECTION]
    assert collection_entry_id("idx-a") in manifest
    assert collection_entry_id("idx-b") in manifest
    assert collection_entry_id(COLLECTION) not in manifest

    # Same effective name, other pipeline: the same entry, so writes refuse.
    (source).write_text(_text(5))
    with pytest.raises(IncompatibleCollectionError):
        await third.initialize(provider_type="ollama")

    named = _tool(source, database=database(collection_name="kb_docs"))
    await named.initialize(provider_type="ollama")
    assert named.logical_collection == "kb_docs"
    assert collection_entry_id("kb_docs") in manifest


# -- item 9: manifest provisioning failures and manifest: false --


class _BrokenManifest(StoreRegistry):
    def __init__(self, *, ensure: bool = False, get: bool = False) -> None:
        super().__init__()
        self.fail_ensure = ensure
        self.fail_get = get

    def create(
        self, provider: str, schema: Any, connection: object = None
    ) -> PersistentAdapter:
        adapter = super().create(provider, schema, connection)
        if schema.name != MANIFEST_COLLECTION:
            return adapter
        if self.fail_ensure:

            async def ensure() -> None:
                raise RuntimeError("quota exceeded: 5 indexes per project")

            adapter.ensure_collection = ensure  # type: ignore[method-assign]
        if self.fail_get:

            async def get(ids: Sequence[str], **kwargs: Any) -> Any:
                raise ConnectionError("read timed out")

            adapter.get = get  # type: ignore[method-assign]
        return adapter


def _patched(registry: StoreRegistry):  # type: ignore[no-untyped-def]
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("holodeck.lib.retrieval.create_adapter", side_effect=registry.create)
    )
    stack.enter_context(
        patch(
            "holodeck.lib.retrieval.registry.create_adapter",
            side_effect=registry.create,
        )
    )
    stack.enter_context(patch("holodeck.lib.retrieval.manifest._warned", set()))
    return stack


@pytest.mark.asyncio
async def test_a_manifest_that_cannot_be_created_falls_back_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry = _BrokenManifest(ensure=True)
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    with _patched(registry), caplog.at_level(logging.WARNING):
        await _tool(source).initialize(provider_type="ollama")
        await _tool(source).initialize(force_ingest=True, provider_type="ollama")
    assert registry.records(COLLECTION)
    assert caplog.text.count("Fingerprint protection is off") == 1
    assert "quota exceeded" in caplog.text


@pytest.mark.asyncio
async def test_an_unreadable_existing_manifest_fails_closed_for_writes(
    tmp_path: Path,
) -> None:
    registry = _BrokenManifest()
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    with _patched(registry):
        await _tool(source).initialize(provider_type="ollama")
    stored = len(registry.records(COLLECTION))
    registry.fail_get = True
    with _patched(registry):
        reader = _tool(source)  # nothing changed: no write, so it starts
        await reader.initialize(provider_type="ollama")
        assert "Found" in await reader.search("subject")
        source.write_text(_text(6))
        with pytest.raises(IncompatibleCollectionError, match="read timed out"):
            await _tool(source).initialize(provider_type="ollama")
    assert len(registry.records(COLLECTION)) == stored


@pytest.mark.asyncio
async def test_manifest_false_opts_out_without_a_warning(
    tmp_path: Path, store: StoreRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "doc.md"
    source.write_text(_text(3))
    database = DatabaseConfig(provider="in-memory", manifest=False)
    with caplog.at_level(logging.WARNING):
        tool = _tool(source, database=database)
        await tool.initialize(provider_type="ollama")
    assert MANIFEST_COLLECTION not in store.existing
    assert MANIFEST_COLLECTION not in store.created
    assert "no pipeline manifest" not in caplog.text
    assert store.records(COLLECTION)
    assert not tool.manifest_enabled


def test_manifest_setting_is_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DatabaseConfig(provider="in-memory", manifest="sometimes")  # type: ignore[arg-type]
