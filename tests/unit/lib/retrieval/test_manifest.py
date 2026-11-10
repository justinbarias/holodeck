"""Manifest collection round trips on the in-memory adapter (FR-016, FR-023)."""

from __future__ import annotations

import logging

import pytest

from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter
from holodeck.lib.retrieval.manifest import (
    MANIFEST_SCHEMA,
    CollectionEntry,
    ManifestStore,
    SourceEntry,
    collection_entry_id,
    manifest_connection,
    source_entry_id,
)


def _store() -> ManifestStore:
    return ManifestStore(InMemoryAdapter(MANIFEST_SCHEMA))


@pytest.mark.asyncio
async def test_reads_do_not_create_the_manifest() -> None:
    store = _store()
    assert await store.get_collection("docs") is None
    assert await store.sources("docs") == {}
    await store.delete_source("docs", "a.md")
    assert await store.delete_sources("docs") == 0
    assert not await store.exists()


@pytest.mark.asyncio
async def test_collection_entry_round_trip() -> None:
    store = _store()
    entry = CollectionEntry(
        logical="docs",
        fingerprint="f" * 64,
        fingerprint_inputs='{"a":1}',
        active_collection="docs__ffffffff",
        previous_collection="docs",
        previous_fingerprint="e" * 64,
        previous_fingerprint_inputs='{"a":0}',
        state="migrating",
        migration_target="docs__aaaaaaaa",
        migration_fingerprint="a" * 64,
        migration_fingerprint_inputs='{"a":2}',
        updated_at=12.5,
    )
    await store.put_collection(entry)
    assert await store.exists()
    assert await store.get_collection("docs") == entry
    await store.delete_collection_entry("docs")
    assert await store.get_collection("docs") is None


@pytest.mark.asyncio
async def test_source_entries_are_scoped_by_physical_collection() -> None:
    store = _store()
    first = SourceEntry("docs", "/data/a.md", 3, False, "1.5", 1.0)
    await store.put_source(first)
    await store.put_source(SourceEntry("docs", "/data/b.md", 1, True, "2.0", 1.0))
    await store.put_source(SourceEntry("other", "/data/a.md", 9, True, "x", 1.0))
    await store.put_collection(CollectionEntry("docs", "f", "{}", "docs"))

    sources = await store.sources("docs")
    assert sorted(sources) == ["/data/a.md", "/data/b.md"]
    assert sources["/data/a.md"] == first
    assert sources["/data/b.md"].complete

    await store.put_source(SourceEntry("docs", "/data/a.md", 3, True, "1.5", 2.0))
    assert (await store.sources("docs"))["/data/a.md"].complete

    await store.delete_source("docs", "/data/b.md")
    assert sorted(await store.sources("docs")) == ["/data/a.md"]
    assert await store.delete_sources("docs") == 1
    assert await store.sources("docs") == {}
    assert sorted(await store.sources("other")) == ["/data/a.md"]
    assert await store.get_collection("docs") is not None


def test_ids_are_stable_and_bounded() -> None:
    assert collection_entry_id("docs") == "collection::docs"
    long_key = "/very/long/" + "x" * 4000
    entry_id = source_entry_id("docs", long_key)
    assert entry_id == source_entry_id("docs", long_key)
    assert entry_id.startswith("source::docs::")
    assert len(entry_id.rsplit("::", 1)[1]) == 32
    assert source_entry_id("docs", "a") != source_entry_id("other", "a")


def test_manifest_ignores_collection_name_settings() -> None:
    connection = {"url": "x", "collection_name": "kb", "index_name": "idx"}
    assert manifest_connection(connection) == {"url": "x"}


def test_open_returns_none_and_warns_for_an_unknown_provider(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert ManifestStore.open("no-such-provider-xyz", {}) is None
    assert "no pipeline manifest" in caplog.text
