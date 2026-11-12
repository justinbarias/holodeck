"""Compose a ``TemporalMemory`` from an agent's ``temporal_memory`` tool (spec 043).

``open_memory`` does the wiring that the CLI and the agent-facing tool share:
it initializes the agent's tools (phase A of ingestion, the source tools' own
vector ingestion), wraps the source tools as chunk sources, creates the backend
and builds the service. For a dry run it builds the source tools for reading
only and initializes none. It prints nothing and knows nothing about the CLI.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from holodeck.lib.memory.chunk_source import ChunkSource
from holodeck.lib.memory.errors import MemoryConfigError
from holodeck.lib.memory.ports import MemoryBackend
from holodeck.lib.memory.registry import create_backend
from holodeck.lib.memory.service import TemporalMemory
from holodeck.lib.source_resolver import SourceResolver
from holodeck.lib.tool_initializer import (
    _is_remote_source,
    close_tools,
    create_embedding_service,
    initialize_tools,
)
from holodeck.models.agent import Agent
from holodeck.models.claude_config import AuthProvider
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import ProviderEnum
from holodeck.models.tool import (
    HierarchicalDocumentToolConfig,
    TemporalMemoryToolConfig,
)
from holodeck.tools.hierarchical_chunk_source import HierarchicalChunkSource
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
from holodeck.tools.vectorstore_tool import VectorStoreTool

if TYPE_CHECKING:
    from holodeck.lib.litellm_support import LiteLLMEmbeddingService

AGENT_SDK_SECONDS_PER_EPISODE = 11.0
"""Middle of the 6.5 to 15.5 s measured per episode on the Agent SDK OAuth path."""


@dataclass(frozen=True, slots=True)
class MemorySession:
    """An open memory: the service, its backend and the tool configuration."""

    memory: TemporalMemory
    backend: MemoryBackend
    config: TemporalMemoryToolConfig


def find_memory_config(agent: Agent) -> TemporalMemoryToolConfig:
    """Return the agent's ``temporal_memory`` tool.

    Raises:
        MemoryConfigError: The agent has none.
    """
    for tool in agent.tools or []:
        if isinstance(tool, TemporalMemoryToolConfig):
            return tool
    raise MemoryConfigError("agent has no temporal_memory tool")


def _seconds_per_episode(
    config: TemporalMemoryToolConfig, agent: Agent
) -> float | None:
    model = config.extraction_model or agent.model
    if (
        model.provider == ProviderEnum.ANTHROPIC
        and model.auth_provider == AuthProvider.oauth_token
    ):
        return AGENT_SDK_SECONDS_PER_EPISODE
    return None


async def _preview_tool(
    agent: Agent,
    name: str,
    base_dir: str | None,
    execution_config: ExecutionConfig | None,
    temp_dirs: list[Path],
) -> HierarchicalDocumentTool:
    """Build source tool ``name`` for reading only. It is never initialized.

    No embedding service and no store are attached, so the tool cannot call
    a model or write. A remote source is downloaded to a temporary directory
    that is added to ``temp_dirs``.
    """
    tool_config = next(
        (
            tool
            for tool in agent.tools or []
            if isinstance(tool, HierarchicalDocumentToolConfig) and tool.name == name
        ),
        None,
    )
    if tool_config is None:
        raise MemoryConfigError(
            f"source tool '{name}' is not a hierarchical_document tool"
        )
    remote_root: Path | None = None
    if _is_remote_source(tool_config.source):
        resolved = await SourceResolver.resolve(tool_config.source, base_dir)
        if resolved.temp_dir:
            temp_dirs.append(resolved.temp_dir)
        remote_root = resolved.local_path
        tool_config = tool_config.model_copy(update={"source": str(remote_root)})
    tool = HierarchicalDocumentTool(
        tool_config, base_dir=base_dir, execution_config=execution_config
    )
    tool.set_read_only()
    if remote_root is not None:
        tool.set_source_context(source_root=remote_root, is_remote=True)
    return tool


def chunk_sources(
    config: TemporalMemoryToolConfig, tools: Mapping[str, object]
) -> list[ChunkSource]:
    """Wrap the memory's source tools as chunk sources, in ``source_tools`` order.

    Args:
        config: The memory tool.
        tools: Initialized tool instances keyed by tool name.

    Raises:
        MemoryConfigError: A source tool is not an initialized
            ``hierarchical_document`` tool.
    """
    sources: list[ChunkSource] = []
    for name in config.source_tools:
        tool = tools.get(name)
        if not isinstance(tool, HierarchicalDocumentTool):
            raise MemoryConfigError(
                f"source tool '{name}' is not an initialized "
                "hierarchical_document tool"
            )
        sources.append(HierarchicalChunkSource(tool))
    return sources


async def compose_memory(
    agent: Agent,
    config: TemporalMemoryToolConfig,
    sources: Sequence[ChunkSource],
    *,
    embedding_service: LiteLLMEmbeddingService | None = None,
    wrap_backend: Callable[[MemoryBackend], MemoryBackend] | None = None,
) -> MemorySession:
    """Create the backend and the service. Makes no store or model call.

    The CLI (``open_memory``) and the agent-facing tool both build their
    memory here.

    Args:
        agent: The agent that owns the memory tool.
        config: The memory tool.
        sources: Chunk sources, from ``chunk_sources`` or a preview.
        embedding_service: The agent's shared embedding service. A new one is
            created from the agent when None.
        wrap_backend: Wraps the created backend before the service gets it.
            The Temporal ingestion activity uses it to heartbeat per episode.
            The session's ``backend`` is the wrapped one.

    Raises:
        MemoryConfigError: The backend is not supported.
        MemoryValueError: A service setting is out of range.
    """
    backend = create_backend(
        config.backend,
        config.extraction_model or agent.model,
        (
            create_embedding_service(agent)
            if embedding_service is None
            else embedding_service
        ),
        config.schema_hints,
        config.extraction_reasoning_effort,
        config.close_on_text_change,
    )
    if wrap_backend is not None:
        backend = wrap_backend(backend)
    try:
        memory = TemporalMemory(
            namespace=config.resolved_namespace,
            backend=backend,
            sources=sources,
            effective_date_key=config.effective_date_key,
            instructions=config.extraction_instructions,
            seconds_per_episode=_seconds_per_episode(config, agent),
            top_k=config.top_k,
            hops=config.hops,
            weight_chunks=config.fusion_weights.chunks,
            weight_graph=config.fusion_weights.graph,
        )
    except BaseException:
        await backend.close()
        raise
    return MemorySession(memory=memory, backend=backend, config=config)


@asynccontextmanager
async def open_memory(
    agent: Agent,
    *,
    base_dir: str | None = None,
    force_ingest_sources: bool = False,
    initialize_sources: bool = True,
    preview_sources: bool = False,
    execution_config: ExecutionConfig | None = None,
) -> AsyncIterator[MemorySession]:
    """Open the agent's temporal memory and close everything on exit.

    Args:
        agent: Agent with one ``temporal_memory`` tool.
        base_dir: Directory that relative source paths resolve against.
        force_ingest_sources: Re-ingest the source tools' vector stores.
        initialize_sources: When False, no source tool is initialized and the
            service has no sources. For commands that only use the backend
            (status, purge).
        preview_sources: When True, no source tool is initialized: the
            sources list the chunks an ingestion would store, read from the
            source files, with no embedding call and no store write. For a
            dry run. ``initialize_sources=False`` takes precedence.
        execution_config: Execution settings forwarded to the source tools.

    Raises:
        MemoryConfigError: No memory tool, or a source tool is not an
            initialized ``hierarchical_document`` tool.
    """
    config = find_memory_config(agent)
    tools: dict[str, VectorStoreTool | HierarchicalDocumentTool] = {}
    temp_dirs: list[Path] = []
    backend: MemoryBackend | None = None
    try:
        sources: list[ChunkSource] = []
        if initialize_sources and preview_sources:
            for name in config.source_tools:
                preview = await _preview_tool(
                    agent, name, base_dir, execution_config, temp_dirs
                )
                tools[name] = preview
                sources.append(HierarchicalChunkSource(preview, preview=True))
        elif initialize_sources:
            tools = dict(
                await initialize_tools(
                    agent,
                    force_ingest=force_ingest_sources,
                    execution_config=execution_config,
                    base_dir=base_dir,
                    initialize_memory=False,
                )
            )
            sources = chunk_sources(config, tools)
        session = await compose_memory(agent, config, sources)
        backend = session.backend
        yield session
    finally:
        try:
            if backend is not None:
                await backend.close()
        finally:
            try:
                await close_tools(tools.values())
            finally:
                for temp_dir in temp_dirs:
                    await SourceResolver.cleanup(temp_dir)
