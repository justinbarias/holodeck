"""Manifest collection round trips on the in-memory adapter (FR-016, FR-023)."""

from __future__ import annotations

import logging

import pytest

from holodeck.lib.retrieval.adapters.in_memory import InMemoryAdapter
from holodeck.lib.retrieval.manifest import (
    MANIFEST_SCHEMA,
    CollectionEntry,
    LockHeldError,
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
    written = await store.put_collection(entry)
    assert await store.exists()
    assert written.revision and written == entry.with_revision(written.revision)
    assert await store.get_collection("docs") == written
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


def test_manifest_ignores_settings_that_pin_one_index() -> None:
    """A Pinecone ``index_host`` addresses the tool's index, not the manifest."""
    connection = {"api_key": "k", "index_name": "idx", "index_host": "https://h"}
    assert manifest_connection(connection) == {"api_key": "k"}


@pytest.mark.asyncio
async def test_every_write_of_an_entry_gets_a_new_revision() -> None:
    store = _store()
    entry = CollectionEntry("docs", "f", "{}", "docs")
    first = await store.put_collection(entry)
    second = await store.put_collection(first)
    assert first.revision and second.revision and first.revision != second.revision
    assert (await store.get_collection("docs")) == second


@pytest.mark.asyncio
async def test_lock_is_exclusive_until_released_or_stale() -> None:
    store = _store()
    lock = await store.acquire_lock("docs", "migrate", ttl=7200)
    assert lock.operation == "migrate" and lock.expires_at > lock.acquired_at
    with pytest.raises(LockHeldError, match="migrate") as raised:
        await store.acquire_lock("docs", "prune", ttl=7200)
    assert raised.value.holder.owner == lock.owner
    # Another logical collection has its own lock.
    other = await store.acquire_lock("kb", "prune", ttl=7200)
    await store.release_lock(other)

    refreshed = await store.refresh_lock(lock, ttl=7200)
    assert refreshed.expires_at >= lock.expires_at
    await store.release_lock(refreshed)
    again = await store.acquire_lock("docs", "rollback", ttl=7200)
    await store.release_lock(again)

    stale = await store.acquire_lock("docs", "migrate", ttl=-1)
    taken = await store.acquire_lock("docs", "prune", ttl=7200)
    assert taken.owner != stale.owner
    # Releasing a lock someone else took over does not remove theirs.
    await store.release_lock(stale)
    assert (await store.get_lock("docs")) == taken


@pytest.mark.asyncio
async def test_lock_records_are_not_collection_or_source_entries() -> None:
    store = _store()
    lock = await store.acquire_lock("docs", "migrate", ttl=60)
    assert await store.get_collection("docs") is None
    assert await store.sources("docs") == {}
    await store.release_lock(lock)
    assert await store.get_lock("docs") is None


def test_open_returns_none_and_warns_for_an_unknown_provider(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert ManifestStore.open("no-such-provider-xyz", {}) is None
    assert "no pipeline manifest" in caplog.text
