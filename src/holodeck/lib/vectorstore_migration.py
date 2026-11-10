"""Inspect, migrate, roll back and prune retrieval collections (FR-023, AC-08).

``holodeck vectorstore`` calls these functions. Each one builds the tool the
agent configuration describes, so the pipeline fingerprint is computed by the
same code that ingests.

* ``inspect_tool`` is read-only: it creates no collection or manifest entry
  and calls no model. It parses and splits sources locally to count chunks.
* ``migrate_tool`` re-ingests every source into a new physical collection
  ``<logical>__<fingerprint[:8]>``, verifies it, then points the logical name
  at it in one manifest write. The old collection is kept for rollback. An
  interrupted migration resumes: completed sources are not embedded again.
* ``rollback_tool`` points the logical name back at the kept collection.
* ``prune_tool`` deletes the kept collection. Nothing else deletes one.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
    ManifestStore,
)
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

ESTIMATE_NOTE = (
    "Estimate: tiktoken cl100k_base tokens of the chunk text. The embedding "
    "model's tokenizer may count differently, and text that contextual "
    "embeddings add to each chunk is not counted."
)


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
    provider_type: str
    manifest: ManifestStore | None
    extra_adapters: list[VectorStoreAdapter] = field(default_factory=list)


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
) -> AsyncIterator[_Context]:
    """Build the tool as ``tool_initializer`` would, without initializing it.

    With ``embeddings`` False no embedding service is attached, so nothing
    can call the embedding model. The pipeline identity is still set from
    the configuration.
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
    if _is_remote_source(tool_config.source):
        resolved = await SourceResolver.resolve(tool_config.source, base_dir)
        temp_dir = resolved.temp_dir
        remote_root = resolved.local_path
        effective = tool_config.model_copy(update={"source": str(remote_root)})

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

    context = _Context(
        tool=tool,
        provider_type=identity[0],
        manifest=ManifestStore.open(
            tool.collection_provider, tool.collection_connection
        ),
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
    if tool._adapter is not None and tool._adapter.schema.name == physical:
        return tool._adapter
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
        raise MigrationError(
            f"Vector store provider '{context.tool.collection_provider}' cannot "
            f"hold the pipeline manifest, so collections on it cannot be migrated."
        )
    return context.manifest


# -- inspect --


async def inspect_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
) -> InspectReport:
    """Report collection identity, compatibility and migration work. Read-only."""
    import tiktoken

    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=False,
    ) as context:
        tool = context.tool
        manifest = context.manifest
        logical = tool.logical_collection
        current = tool.pipeline_inputs()
        entry = await manifest.get_collection(logical) if manifest else None
        physical = (
            entry.active_collection if entry and entry.active_collection else (logical)
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
        )


# -- migrate --


async def migrate_tool(
    agent: Agent,
    tool_config: RetrievalToolConfig,
    *,
    base_dir: str | None = None,
    execution_config: ExecutionConfig | None = None,
    restart: bool = False,
) -> MigrationResult:
    """Rebuild the tool's collection with the current pipeline and switch to it.

    Raises:
        MigrationError: No manifest, another migration is in progress, a kept
            collection must be pruned first, or verification failed.
    """
    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=True,
    ) as context:
        manifest = _require_manifest(context)
        tool = context.tool
        logical = tool.logical_collection
        current = tool.pipeline_inputs()
        entry = await manifest.get_collection(logical)
        active = (
            entry.active_collection if entry and entry.active_collection else (logical)
        )
        active_adapter = _adapter_for(context, active)
        active_has_records = await collection_has_records(active_adapter)

        if entry is None:
            if not active_has_records:
                return MigrationResult(
                    tool=tool_config.name,
                    logical_collection=logical,
                    previous_collection="",
                    active_collection=active,
                    record_count=0,
                    sources=0,
                    embedded_sources=0,
                    skipped=True,
                    message=(
                        "Collection is empty or absent; the tool builds it with "
                        "the current pipeline on its next start."
                    ),
                )
            entry = CollectionEntry(
                logical=logical,
                fingerprint=current.fingerprint,
                fingerprint_inputs=legacy_inputs_json(current),
                active_collection=active,
            )
        legacy = is_legacy(entry.fingerprint_inputs)
        if (
            entry.fingerprint == current.fingerprint
            and not legacy
            and entry.state == STATE_ACTIVE
        ):
            return MigrationResult(
                tool=tool_config.name,
                logical_collection=logical,
                previous_collection=entry.previous_collection,
                active_collection=active,
                record_count=(
                    await active_adapter.count()
                    if active_adapter.capabilities.supports(Operation.COUNT)
                    else 0
                ),
                sources=0,
                embedded_sources=0,
                skipped=True,
                message="Collection already matches the current pipeline.",
            )
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
                f"holodeck vectorstore prune AGENT_YAML --tool {tool_config.name} "
                f"--yes"
            )

        if target == active:
            raise MigrationError(
                f"Collection '{active}' already has the target name; nothing to do."
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

        before = await manifest.sources(target)
        already_complete = {key for key, src in before.items() if src.complete}
        await tool.close()  # initialize builds its own adapter and manifest
        tool.set_migration_target(target)
        await tool.initialize(force_ingest=False, provider_type=context.provider_type)

        target_adapter = tool._adapter
        if target_adapter is None:
            raise MigrationError("Migration target collection was not created.")
        sources = await manifest.sources(target)
        incomplete = sorted(key for key, src in sources.items() if not src.complete)
        if incomplete:
            raise MigrationError(
                f"Sources did not finish ingesting into '{target}': {incomplete}. "
                f"Re-run the migration to resume."
            )
        expected = sum(src.chunk_count for src in sources.values())
        await _verify(target_adapter, expected)

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
            tool=tool_config.name,
            logical_collection=logical,
            previous_collection=active,
            active_collection=target,
            record_count=expected,
            sources=len(sources),
            embedded_sources=len(set(sources) - already_complete),
        )


async def _verify(adapter: VectorStoreAdapter, expected: int) -> None:
    """Check the record count and that a vector search returns a result."""
    adapter.capabilities.require(Operation.COUNT, Operation.VECTOR_SEARCH)
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


async def _entry_with_previous(
    context: _Context, tool_name: str, action: str
) -> CollectionEntry:
    manifest = _require_manifest(context)
    logical = context.tool.logical_collection
    entry = await manifest.get_collection(logical)
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
    ) as context:
        entry = await _entry_with_previous(context, tool_config.name, "roll back to")
        previous = entry.previous_collection
        if not await _adapter_for(context, previous).collection_exists():
            raise MigrationError(
                f"Kept collection '{previous}' no longer exists; cannot roll back."
            )
        manifest = _require_manifest(context)
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
    """Delete the kept previous collection and its manifest source entries."""
    async with _built_tool(
        agent,
        tool_config,
        base_dir=base_dir,
        execution_config=execution_config,
        embeddings=False,
    ) as context:
        entry = await _entry_with_previous(context, tool_config.name, "prune")
        previous = entry.previous_collection
        if previous == entry.active_collection:
            raise MigrationError(
                f"Kept collection '{previous}' is the active collection; refusing "
                f"to delete it."
            )
        manifest = _require_manifest(context)
        await _adapter_for(context, previous).delete_collection()
        deleted_sources = await manifest.delete_sources(previous)
        await manifest.put_collection(
            dataclasses.replace(
                entry,
                previous_collection="",
                previous_fingerprint="",
                previous_fingerprint_inputs="",
                updated_at=time.time(),
            )
        )
        return SwitchResult(
            tool=tool_config.name,
            logical_collection=entry.logical,
            active_collection=entry.active_collection,
            previous_collection="",
            deleted_collection=previous,
            deleted_sources=deleted_sources,
        )
