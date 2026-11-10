"""Inspect, migrate, roll back and prune retrieval collections (FR-023, AC-08).

``holodeck vectorstore`` calls these functions. Each one builds the tools the
agent configuration describes, so the pipeline fingerprint is computed by the
same code that ingests.

* ``inspect_tool`` is strictly read-only: it downloads nothing, creates no
  file, directory, collection or manifest entry, and calls no model. It
  parses and splits local sources to count chunks; a remote source is
  reported as not estimated.
* ``migrate_tool`` works on a logical collection. ``--tool`` selects it, and
  every retrieval tool of the agent bound to the same logical collection is
  rebuilt with it (they must share one fingerprint). It lists every source
  before it embeds, re-ingests them into a new physical collection
  ``<logical>__<fingerprint[:8]>``, verifies every source and the total
  count, checks that the old collection holds no source the migration did
  not rebuild, then points the logical name at the new collection in one
  manifest write. The old collection is kept for rollback. An interrupted
  migration resumes: completed sources are not embedded again.
* ``rollback_tool`` points the logical name back at the kept collection.
* ``prune_tool`` deletes the kept collection. Nothing else deletes one.

``migrate``, ``rollback`` and ``prune`` hold the manifest lock of the logical
collection while they run and read the collection entry again under it, so
two of them never act on one collection at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from holodeck.lib.errors import HoloDeckError
from holodeck.lib.retrieval.adapter import VectorStoreAdapter
from holodeck.lib.retrieval.fingerprint import (
    SettingValue,
    diff_inputs,
    format_diff,
    is_legacy,
    legacy_inputs_json,
    parse_inputs,
)
from holodeck.lib.retrieval.guard import (
    collection_has_records,
    physical_connection,
    physical_schema,
)
from holodeck.lib.retrieval.manifest import (
    STATE_ACTIVE,
    STATE_MIGRATING,
    CollectionEntry,
    LockHeldError,
    LockRecord,
    ManifestStore,
    SourceEntry,
    manifest_connection,
)
from holodeck.lib.retrieval.schemas import text_field
from holodeck.lib.retrieval.types import Operation, RecordFilter, VectorQuery

if TYPE_CHECKING:
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig
    from holodeck.models.tool import HierarchicalDocumentToolConfig
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    RetrievalTool = VectorStoreTool | HierarchicalDocumentTool
    RetrievalToolConfig = VectorstoreToolConfig | HierarchicalDocumentToolConfig

logger = logging.getLogger(__name__)

ESTIMATE_NOTE = (
    "Estimate: tiktoken cl100k_base tokens of the chunk text. The embedding "
    "model's tokenizer may count differently, and text that contextual "
    "embeddings add to each chunk is not counted."
)

LOCK_TTL = 2 * 60 * 60
"""Seconds after which a lifecycle lock that was not refreshed is stale."""

LOCK_REFRESH = 10 * 60
"""Seconds between refreshes of the lock while a command runs."""

_LIST_PAGE = 500


class MigrationError(HoloDeckError):
    """A vector store command cannot run as asked."""


# -- reports --


@dataclass
class InspectReport:
    """What ``holodeck vectorstore inspect`` found for one tool."""

    tool: str
    tool_type: str
    provider: str
    logical_collection: str
    physical_collection: str
    exists: bool
    record_count: int | None
    manifest_supported: bool
    manifest_state: str
    stored_fingerprint: str | None
    current_fingerprint: str
    differences: list[tuple[str, SettingValue, SettingValue]]
    compatible: str
    needs_reembedding: bool
    source_files: int
    chunks: int
    estimated_tokens: int
    previous_collection: str = ""
    migration_target: str = ""
    remote_sources: list[str] = field(default_factory=list)
    estimate_note: str = ESTIMATE_NOTE

    def to_dict(self) -> dict[str, object]:
        """Return the report as JSON-ready data."""
        data = dataclasses.asdict(self)
        data["differences"] = [
            {"input": name, "stored": old, "current": new}
            for name, old, new in self.differences
        ]
        return data


@dataclass
class MigrationPlan:
    """The logical collection a migration rebuilds and the tools bound to it."""

    logical_collection: str
    provider: str
    tools: list[str]


@dataclass
class MigrationResult:
    """Outcome of ``migrate_tool``."""

    tool: str
    logical_collection: str
    previous_collection: str
    active_collection: str
    record_count: int
    sources: int
    embedded_sources: int
    skipped: bool = False
    message: str = ""
    tools: list[str] = field(default_factory=list)
    dropped_sources: list[str] = field(default_factory=list)


@dataclass
class SwitchResult:
    """Outcome of ``rollback_tool`` and ``prune_tool``."""

    tool: str
    logical_collection: str
    active_collection: str
    previous_collection: str
    deleted_collection: str = ""
    deleted_sources: int = 0


@dataclass
class _Context:
    """A built, not yet initialized tool and its manifest."""

    tool: RetrievalTool
    config: RetrievalToolConfig
    provider_type: str
    manifest: ManifestStore | None
    remote_source: str | None = None
    extra_adapters: list[VectorStoreAdapter] = field(default_factory=list)

    @property
    def store_key(self) -> tuple[str, str, str]:
        """Identify the logical collection: provider, store and name."""
        connection = manifest_connection(self.tool.collection_connection)
        return (
            self.tool.collection_provider,
            repr(sorted((key, repr(value)) for key, value in connection.items())),
            self.tool.logical_collection,
        )


# -- tool selection and construction --


def retrieval_tools(agent: Agent, name: str | None = None) -> list[RetrievalToolConfig]:
    """Return the agent's vectorstore and hierarchical-document tool configs.

    Raises:
        MigrationError: ``name`` is given and names no retrieval tool.
    """
    from holodeck.models.tool import HierarchicalDocumentToolConfig
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig

    found: list[RetrievalToolConfig] = [
        tool
        for tool in agent.tools or []
        if isinstance(tool, VectorstoreToolConfig | HierarchicalDocumentToolConfig)
    ]
    if name is None:
        return found
    selected = [tool for tool in found if tool.name == name]
    if not selected:
        names = ", ".join(tool.name for tool in found) or "none"
        raise MigrationError(
            f"Agent has no vectorstore or hierarchical_document tool named "
            f"'{name}'. Retrieval tools: {names}"
        )
    return selected


@contextlib.asynccontextmanager
async def _built_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None,
    execution_config: ExecutionConfig | None,
    embeddings: bool,
    resolve_remote: bool = True,
) -> AsyncIterator[_Context]:
    """Build the tool as ``tool_initializer`` would, without initializing it.

    With ``embeddings`` False no embedding service is attached, so nothing
    can call the embedding model. The pipeline identity is still set from
    the configuration. With ``resolve_remote`` False a remote source is not
    downloaded; ``_Context.remote_source`` names it instead.
    """
    from holodeck.lib.source_resolver import SourceResolver
    from holodeck.lib.tool_initializer import (
        _embedding_service_for_tool,
        _is_remote_source,
        _resolve_context_generator,
        create_embedding_service,
        embedding_identity,
    )
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    identity = embedding_identity(agent)
    temp_dir: Path | None = None
    effective = tool_config
    remote_root: Path | None = None
    remote_source: str | None = None
    if _is_remote_source(tool_config.source):
        if resolve_remote:
            resolved = await SourceResolver.resolve(tool_config.source, base_dir)
            temp_dir = resolved.temp_dir
            remote_root = resolved.local_path
            effective = tool_config.model_copy(update={"source": str(remote_root)})
        else:
            remote_source = tool_config.source

    tool: RetrievalTool
    if isinstance(effective, VectorstoreToolConfig):
        tool = VectorStoreTool(
            effective, base_dir=base_dir, execution_config=execution_config
        )
        if embeddings:
            shared = create_embedding_service(agent)
            tool.set_embedding_service(
                _embedding_service_for_tool(agent, effective, shared)
            )
    else:
        tool = HierarchicalDocumentTool(
            effective, base_dir=base_dir, execution_config=execution_config
        )
        if embeddings:
            tool.set_embedding_service(create_embedding_service(agent))
        # Building a generator calls no model; it decides whether stored
        # text is contextual, which is a fingerprint input.
        generator = _resolve_context_generator(agent, effective)
        if generator is not None:
            tool.set_context_generator(generator)
    if remote_root is not None:
        tool.set_source_context(source_root=remote_root, is_remote=True)
    tool.set_embedding_identity(*identity)
    tool._setup_collection(identity[0])

    manifest = (
        ManifestStore.open(tool.collection_provider, tool.collection_connection)
        if tool.manifest_enabled
        else None
    )
    context = _Context(
        tool=tool,
        config=tool_config,
        provider_type=identity[0],
        manifest=manifest,
        remote_source=remote_source,
    )
    try:
        yield context
    finally:
        for adapter in context.extra_adapters:
            await adapter.close()
        if context.manifest is not None:
            await context.manifest.close()
        await tool.close()
        if temp_dir is not None:
            await SourceResolver.cleanup(temp_dir)


def _adapter_for(context: _Context, physical: str) -> VectorStoreAdapter:
    """Return an adapter for ``physical``; no store call is made."""
    from holodeck.lib.retrieval import create_adapter

    tool = context.tool
    logical = tool.logical_collection
    schema = tool._collection_schema
    if schema is None:
        raise RuntimeError("Collection not initialized")
    adapter = create_adapter(
        tool.collection_provider,
        physical_schema(schema, physical),
        physical_connection(tool.collection_connection, logical, physical),
    )
    context.extra_adapters.append(adapter)
    return adapter


def _target_name(logical: str, fingerprint: str) -> str:
    return f"{logical}__{fingerprint[:8]}"


def _require_manifest(context: _Context) -> ManifestStore:
    if context.manifest is None:
        reason = (
            "the database settings set manifest: false"
            if not context.tool.manifest_enabled
            else "the provider cannot hold the pipeline manifest"
        )
        raise MigrationError(
            f"Collections of tool '{context.config.name}' on provider "
            f"'{context.tool.collection_provider}' cannot be migrated: {reason}."
        )
    return context.manifest


# -- lifecycle lock --


class _HeldLock:
    """The lifecycle lock of one command, refreshed in the background."""

    def __init__(self, manifest: ManifestStore, lock: LockRecord) -> None:
        self.manifest = manifest
        self.lock = lock
        self.lost: LockHeldError | None = None

    async def refresh_forever(self) -> None:
        while True:
            await asyncio.sleep(LOCK_REFRESH)
            try:
                self.lock = await self.manifest.refresh_lock(self.lock, LOCK_TTL)
            except LockHeldError as error:
                self.lost = error
                return
            except Exception as error:  # noqa: BLE001 - retried next period
                logger.warning("Could not refresh the collection lock: %s", error)

    def require(self) -> None:
        """Raise when another process took the lock over."""
        if self.lost is not None:
            raise MigrationError(
                f"The lock of collection '{self.lock.logical}' was taken over "
                f"while this command ran: {self.lost}"
            )


@contextlib.asynccontextmanager
async def _lifecycle_lock(
    manifest: ManifestStore, logical: str, operation: str
) -> AsyncIterator[_HeldLock]:
    """Hold the manifest lock of ``logical`` for ``operation``.

    Raises:
        MigrationError: Another command holds the lock.
    """
    try:
        lock = await manifest.acquire_lock(logical, operation, LOCK_TTL)
    except LockHeldError as error:
        raise MigrationError(str(error)) from error
    held = _HeldLock(manifest, lock)
    refresher = asyncio.create_task(held.refresh_forever())
    try:
        yield held
    finally:
        refresher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresher
        try:
            await manifest.release_lock(held.lock)
        except Exception as error:  # noqa: BLE001 - the lock goes stale
            logger.warning(
                "Could not release the lock of collection '%s' (%s); it is "
                "stale after %d minutes.",
                logical,
                error,
                LOCK_TTL // 60,
            )


# -- inspect --


async def inspect_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
) -> InspectReport:
    """Report collection identity, compatibility and migration work. Read-only.

    Downloads nothing, writes no file, creates no collection or manifest
    entry and calls no model.
    """
    import tiktoken

    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=False,
        resolve_remote=False,
    ) as context:
        tool = context.tool
        manifest = context.manifest
        logical = tool.logical_collection
        current = tool.pipeline_inputs()
        entry = await manifest.get_collection(logical) if manifest else None
        physical = (
            entry.active_collection if entry and entry.active_collection else logical
        )
        adapter = _adapter_for(context, physical)
        exists = await adapter.collection_exists()
        record_count: int | None = None
        if exists and adapter.capabilities.supports(Operation.COUNT):
            record_count = await adapter.count()
        has_records = await collection_has_records(adapter) if exists else False

        differences: list[tuple[str, SettingValue, SettingValue]] = []
        if entry is None:
            compatible = "legacy" if has_records else "new"
            stored_fingerprint = None
        else:
            stored_fingerprint = entry.fingerprint
            differences = diff_inputs(
                parse_inputs(entry.fingerprint_inputs), current.as_dict()
            )
            if entry.fingerprint != current.fingerprint and has_records:
                compatible = "no"
            elif is_legacy(entry.fingerprint_inputs):
                compatible = "legacy"
            else:
                compatible = "yes"
        if manifest is None:
            compatible = "unknown"

        remote_sources: list[str] = []
        preview: list[tuple[str, list[str]]] = []
        if context.remote_source is not None:
            remote_sources.append(context.remote_source)
        else:
            tool.set_read_only()
            preview = await tool.preview_chunks()
        encoding = tiktoken.get_encoding("cl100k_base")
        chunks = sum(len(texts) for _, texts in preview)
        tokens = sum(
            len(encoding.encode(text, disallowed_special=()))
            for _, texts in preview
            for text in texts
        )
        return InspectReport(
            tool=tool_config.name,
            tool_type=tool_config.type,
            provider=tool.collection_provider,
            logical_collection=logical,
            physical_collection=physical,
            exists=exists,
            record_count=record_count,
            manifest_supported=manifest is not None,
            manifest_state=entry.state if entry else "none",
            stored_fingerprint=stored_fingerprint,
            current_fingerprint=current.fingerprint,
            differences=differences,
            compatible=compatible,
            needs_reembedding=compatible in ("no", "legacy"),
            source_files=len(preview),
            chunks=chunks,
            estimated_tokens=tokens,
            previous_collection=entry.previous_collection if entry else "",
            migration_target=entry.migration_target if entry else "",
            remote_sources=remote_sources,
        )


# -- migrate --


async def _members(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None,
    execution_config: ExecutionConfig | None,
) -> tuple[MigrationPlan, list[RetrievalToolConfig]]:
    """Return the logical collection of ``tool_config`` and the agent's tools
    bound to it. Builds each tool without downloads, models or store calls."""
    keys: dict[str, tuple[str, str, str]] = {}
    for config in retrieval_tools(agent):
        async with _built_tool(
            agent,
            config,
            base_dir=base_dir,
            execution_config=execution_config,
            embeddings=False,
            resolve_remote=False,
        ) as context:
            keys[config.name] = context.store_key
    selected = keys[tool_config.name]
    members = [
        config for config in retrieval_tools(agent) if keys[config.name] == selected
    ]
    plan = MigrationPlan(
        logical_collection=selected[2],
        provider=selected[0],
        tools=[config.name for config in members],
    )
    return plan, members


async def plan_migration(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
) -> MigrationPlan:
    """Return the logical collection a migration of ``tool_config`` rebuilds
    and every tool of the agent bound to it. Read-only."""
    plan, _ = await _members(
        agent, tool_config, base_dir=base_dir, execution_config=execution_config
    )
    return plan


def _check_one_pipeline(contexts: Sequence[_Context], logical: str) -> None:
    first = contexts[0]
    reference = first.tool.pipeline_inputs()
    lines: list[str] = []
    for context in contexts[1:]:
        inputs = context.tool.pipeline_inputs()
        if inputs.fingerprint != reference.fingerprint:
            differences = diff_inputs(reference.as_dict(), inputs.as_dict())
            lines.append(
                f"Tool '{context.config.name}' differs from tool "
                f"'{first.config.name}':\n{format_diff(differences)}"
            )
    if lines:
        raise MigrationError(
            f"Tools {[c.config.name for c in contexts]} share collection "
            f"'{logical}' but have different ingestion pipelines, so one "
            f"rebuilt collection cannot serve them all. Give them the same "
            f"settings, or give each its own collection_name.\n" + "\n".join(lines)
        )


async def _distinct_sources(adapter: VectorStoreAdapter, source_field: str) -> set[str]:
    """Return the distinct source keys stored in the collection."""
    if not await adapter.collection_exists():
        return set()
    found: set[str] = set()
    capabilities = adapter.capabilities
    if capabilities.supports(Operation.LIST):
        cursor: str | None = None
        while True:
            page = await adapter.list_page(cursor=cursor, limit=_LIST_PAGE)
            found.update(text_field(record, source_field) for record in page.records)
            cursor = page.next_cursor
            if not cursor or not page.records:
                break
    elif capabilities.supports(Operation.FIND):
        records = await adapter.find(RecordFilter())
        found.update(text_field(record, source_field) for record in records)
    else:
        raise MigrationError(
            f"Collection '{adapter.schema.name}' cannot be listed, so the "
            f"migration cannot check that it rebuilt every source in it. Pass "
            f"--drop-unlisted-sources to switch without that check."
        )
    found.discard("")
    return found


async def _check_unlisted(
    adapter: VectorStoreAdapter,
    source_field: str,
    inventory: set[str],
    drop: bool,
) -> list[str]:
    """Return sources of the old collection the migration does not rebuild.

    Raises:
        MigrationError: Some exist and ``drop`` is False.
    """
    if drop and not await adapter.collection_exists():
        return []
    try:
        stored = await _distinct_sources(adapter, source_field)
    except MigrationError:
        if drop:
            return []
        raise
    unlisted = sorted(stored - inventory)
    if unlisted and not drop:
        shown = "\n".join(f"  {key}" for key in unlisted[:50])
        more = f"\n  ... and {len(unlisted) - 50} more" if len(unlisted) > 50 else ""
        raise MigrationError(
            f"Collection '{adapter.schema.name}' holds {len(unlisted)} sources "
            f"that no tool of this agent ingests (another agent file may write "
            f"them), so the migration would drop them:\n{shown}{more}\n"
            f"Migrate from an agent file that lists every source, or pass "
            f"--drop-unlisted-sources to switch without them. The logical "
            f"name was not switched."
        )
    return unlisted


async def migrate_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
    restart: bool = False,
    drop_unlisted_sources: bool = False,
) -> MigrationResult:
    """Rebuild the logical collection of ``tool_config`` and switch to it.

    Every retrieval tool of the agent bound to the same logical collection
    is rebuilt with it.

    Raises:
        MigrationError: No manifest, the tools do not share one pipeline,
            another command holds the lock, another migration is in
            progress, a kept collection must be pruned first, a source was
            not ingested, verification failed, or the old collection holds
            sources the migration does not rebuild.
    """
    plan, configs = await _members(
        agent, tool_config, base_dir=base_dir, execution_config=execution_config
    )
    ordered = [tool_config] + [c for c in configs if c.name != tool_config.name]
    async with contextlib.AsyncExitStack() as stack:
        contexts = [
            await stack.enter_async_context(
                _built_tool(
                    agent,
                    config,
                    base_dir=base_dir,
                    execution_config=execution_config,
                    embeddings=True,
                )
            )
            for config in ordered
        ]
        primary = contexts[0]
        manifest = _require_manifest(primary)
        logical = primary.tool.logical_collection
        _check_one_pipeline(contexts, logical)
        async with _lifecycle_lock(manifest, logical, "migrate") as held:
            return await _migrate_locked(
                contexts,
                manifest,
                held,
                plan,
                restart=restart,
                drop_unlisted_sources=drop_unlisted_sources,
            )
    raise AssertionError("unreachable")  # pragma: no cover


async def _migrate_locked(
    contexts: Sequence[_Context],
    manifest: ManifestStore,
    held: _HeldLock,
    plan: MigrationPlan,
    *,
    restart: bool,
    drop_unlisted_sources: bool,
) -> MigrationResult:
    primary = contexts[0]
    tool = primary.tool
    name = primary.config.name
    logical = tool.logical_collection
    current = tool.pipeline_inputs()
    source_field = tool.source_field
    entry = await manifest.get_collection(logical)  # read again under the lock
    active = entry.active_collection if entry and entry.active_collection else logical
    active_adapter = _adapter_for(primary, active)
    active_has_records = await collection_has_records(active_adapter)

    def skipped(message: str, record_count: int = 0) -> MigrationResult:
        return MigrationResult(
            tool=name,
            logical_collection=logical,
            previous_collection=entry.previous_collection if entry else "",
            active_collection=active,
            record_count=record_count,
            sources=0,
            embedded_sources=0,
            skipped=True,
            message=message,
            tools=plan.tools,
        )

    if entry is None:
        if not active_has_records:
            return skipped(
                "Collection is empty or absent; the tools build it with the "
                "current pipeline on their next start."
            )
        entry = CollectionEntry(
            logical=logical,
            fingerprint=current.fingerprint,
            fingerprint_inputs=legacy_inputs_json(current),
            active_collection=active,
        )
    if entry.pending_delete:
        raise MigrationError(
            f"Prune of '{entry.pending_delete}' did not finish. Run "
            f"'holodeck vectorstore prune AGENT_YAML --tool {name} --yes' again "
            f"first."
        )
    legacy = is_legacy(entry.fingerprint_inputs)
    if (
        entry.fingerprint == current.fingerprint
        and not legacy
        and entry.state == STATE_ACTIVE
    ):
        count = (
            await active_adapter.count()
            if active_adapter.capabilities.supports(Operation.COUNT)
            else 0
        )
        return skipped("Collection already matches the current pipeline.", count)
    if (
        entry.state == STATE_MIGRATING
        and entry.migration_fingerprint
        and entry.migration_fingerprint != current.fingerprint
        and not restart
    ):
        raise MigrationError(
            f"A migration of '{logical}' to '{entry.migration_target}' for a "
            f"different pipeline is in progress. Re-run with --restart to "
            f"abandon it (its partial collection is kept, not deleted)."
        )
    target = _target_name(logical, current.fingerprint)
    if entry.previous_collection and entry.previous_collection not in (
        active,
        target,
    ):
        raise MigrationError(
            f"Collection '{entry.previous_collection}' is still kept from an "
            f"earlier migration of '{logical}'. Remove it first with: "
            f"holodeck vectorstore prune AGENT_YAML --tool {name} --yes"
        )
    if target == active:
        raise MigrationError(
            f"Collection '{active}' already has the target name; nothing to do."
        )

    # The full source inventory, before anything is embedded.
    inventory: set[str] = set()
    for context in contexts:
        inventory.update(context.tool.source_inventory())
    if not inventory:
        raise MigrationError(
            f"Tools {plan.tools} list no source files; refusing to replace "
            f"'{active}' with an empty collection."
        )
    if active_has_records:
        await _check_unlisted(
            active_adapter, source_field, inventory, drop_unlisted_sources
        )

    await manifest.put_collection(
        dataclasses.replace(
            entry,
            state=STATE_MIGRATING,
            migration_target=target,
            migration_fingerprint=current.fingerprint,
            migration_fingerprint_inputs=current.canonical_json(),
            updated_at=time.time(),
        )
    )

    target_adapter = _adapter_for(primary, target)
    before = await manifest.sources(target)
    for key in sorted(set(before) - inventory):
        # A source removed since an interrupted run: drop what it left behind.
        if await target_adapter.collection_exists():
            await target_adapter.delete_where(RecordFilter.where(**{source_field: key}))
        await manifest.delete_source(target, key)
    already_complete = {
        key for key, src in before.items() if src.complete and key in inventory
    }

    for context in contexts:
        held.require()
        member = context.tool
        await member.close()  # initialize builds its own adapter and manifest
        member.set_migration_target(target)
        await member.initialize(force_ingest=False, provider_type=context.provider_type)

    held.require()
    sources = await manifest.sources(target)
    unfinished = sorted(
        key for key in inventory if key not in sources or not sources[key].complete
    )
    if unfinished:
        raise MigrationError(
            f"These sources have no completion marker in '{target}' (they failed "
            f"to parse, produced no chunks, or their ingestion was interrupted): "
            f"{unfinished}. The logical name was not switched. Fix them and "
            f"re-run the migration to resume."
        )
    expected = sum(sources[key].chunk_count for key in inventory)
    await _verify(target_adapter, expected, source_field, sources, inventory)
    dropped: list[str] = []
    if active_has_records:
        dropped = await _check_unlisted(
            active_adapter, source_field, inventory, drop_unlisted_sources
        )

    latest = await manifest.get_collection(logical)
    if (
        latest is None
        or latest.state != STATE_MIGRATING
        or latest.migration_target != target
    ):
        raise MigrationError(
            f"The manifest entry of '{logical}' changed during the migration; "
            f"the logical name was not switched."
        )
    await manifest.put_collection(
        CollectionEntry(
            logical=logical,
            fingerprint=current.fingerprint,
            fingerprint_inputs=current.canonical_json(),
            active_collection=target,
            previous_collection=active,
            previous_fingerprint=entry.fingerprint,
            previous_fingerprint_inputs=entry.fingerprint_inputs,
            state=STATE_ACTIVE,
            updated_at=time.time(),
        )
    )
    return MigrationResult(
        tool=name,
        logical_collection=logical,
        previous_collection=active,
        active_collection=target,
        record_count=expected,
        sources=len(inventory),
        embedded_sources=len(inventory - already_complete),
        tools=plan.tools,
        dropped_sources=dropped,
    )


async def _verify(
    adapter: VectorStoreAdapter,
    expected: int,
    source_field: str,
    sources: dict[str, SourceEntry],
    inventory: set[str],
) -> None:
    """Check per-source and total counts and that a vector search answers."""
    adapter.capabilities.require(Operation.COUNT, Operation.VECTOR_SEARCH)
    if adapter.capabilities.supports(Operation.FIND):
        wrong: list[str] = []
        for key in sorted(inventory):
            marker = sources[key]
            stored = len(await adapter.find(RecordFilter.where(**{source_field: key})))
            if stored != marker.chunk_count:
                wrong.append(f"{key}: {stored} stored, {marker.chunk_count} expected")
        if wrong:
            raise MigrationError(
                f"Collection '{adapter.schema.name}' does not hold the records "
                f"its sources were ingested with: {wrong}. The logical name was "
                f"not switched. Re-run the migration to resume."
            )
    count = await adapter.count()
    if count != expected:
        raise MigrationError(
            f"Collection '{adapter.schema.name}' holds {count} records; the "
            f"ingested sources account for {expected}. The logical name was not "
            f"switched. Re-run the migration to resume."
        )
    if expected == 0:
        return
    sample = await _sample_vector(adapter)
    if sample is None:
        raise MigrationError(
            f"Could not read a vector back from '{adapter.schema.name}'."
        )
    hits = await adapter.vector_search(VectorQuery(vector=sample, top_k=1))
    if not hits:
        raise MigrationError(
            f"A vector search on '{adapter.schema.name}' returned no result; "
            f"the logical name was not switched."
        )


async def _sample_vector(adapter: VectorStoreAdapter) -> Sequence[float] | None:
    capabilities = adapter.capabilities
    if capabilities.supports(Operation.LIST):
        records = (await adapter.list_page(limit=1)).records
    elif capabilities.supports(Operation.FIND):
        records = tuple(await adapter.find(RecordFilter(), limit=1))
    else:
        return None
    if not records:
        return None
    found = await adapter.get([records[0].id], include_vectors=True)
    return found[0].vector if found and found[0].vector else None


# -- rollback and prune --


def _with_previous(
    entry: CollectionEntry | None, logical: str, tool_name: str, action: str
) -> CollectionEntry:
    if entry is None or not entry.previous_collection:
        raise MigrationError(
            f"Collection '{logical}' of tool '{tool_name}' has no kept previous "
            f"collection to {action}."
        )
    if entry.state == STATE_MIGRATING:
        raise MigrationError(
            f"A migration of '{logical}' is in progress; finish it with "
            f"'holodeck vectorstore migrate' before you {action}."
        )
    return entry


async def rollback_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
) -> SwitchResult:
    """Point the logical name back at the kept previous collection."""
    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=False,
        resolve_remote=False,
    ) as context:
        manifest = _require_manifest(context)
        logical = context.tool.logical_collection
        async with _lifecycle_lock(manifest, logical, "rollback"):
            entry = _with_previous(
                await manifest.get_collection(logical),
                logical,
                tool_config.name,
                "roll back to",
            )
            previous = entry.previous_collection
            # Checked under the lock, immediately before the pointer write.
            if not await _adapter_for(context, previous).collection_exists():
                raise MigrationError(
                    f"Kept collection '{previous}' no longer exists; cannot roll "
                    f"back."
                )
            await manifest.put_collection(
                dataclasses.replace(
                    entry,
                    fingerprint=entry.previous_fingerprint,
                    fingerprint_inputs=entry.previous_fingerprint_inputs,
                    active_collection=previous,
                    previous_collection=entry.active_collection,
                    previous_fingerprint=entry.fingerprint,
                    previous_fingerprint_inputs=entry.fingerprint_inputs,
                    updated_at=time.time(),
                )
            )
            return SwitchResult(
                tool=tool_config.name,
                logical_collection=entry.logical,
                active_collection=previous,
                previous_collection=entry.active_collection,
            )


async def prune_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
) -> SwitchResult:
    """Delete the kept previous collection and its manifest source entries.

    The manifest first stops naming the kept collection (it is marked
    ``pending_delete``); only then is the collection deleted, so a rollback
    can never point at a collection that is being deleted. An interrupted
    prune finishes on the next run.
    """
    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=False,
        resolve_remote=False,
    ) as context:
        manifest = _require_manifest(context)
        logical = context.tool.logical_collection
        async with _lifecycle_lock(manifest, logical, "prune"):
            entry = await manifest.get_collection(logical)
            if entry is None or not entry.pending_delete:
                entry = _with_previous(entry, logical, tool_config.name, "prune")
                doomed = entry.previous_collection
                if doomed == entry.active_collection:
                    raise MigrationError(
                        f"Kept collection '{doomed}' is the active collection; "
                        f"refusing to delete it."
                    )
                entry = await manifest.put_collection(
                    dataclasses.replace(
                        entry,
                        previous_collection="",
                        previous_fingerprint="",
                        previous_fingerprint_inputs="",
                        pending_delete=doomed,
                        updated_at=time.time(),
                    )
                )
            doomed = entry.pending_delete
            if doomed == entry.active_collection:
                raise MigrationError(
                    f"Collection '{doomed}' is reserved for deletion but is the "
                    f"active collection; refusing to delete it."
                )
            await _adapter_for(context, doomed).delete_collection()
            deleted_sources = await manifest.delete_sources(doomed)
            await manifest.put_collection(
                dataclasses.replace(entry, pending_delete="", updated_at=time.time())
            )
            return SwitchResult(
                tool=tool_config.name,
                logical_collection=entry.logical,
                active_collection=entry.active_collection,
                previous_collection="",
                deleted_collection=doomed,
                deleted_sources=deleted_sources,
            )
