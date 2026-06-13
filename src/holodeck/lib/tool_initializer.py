"""Shared tool initialization for VectorStoreTool and HierarchicalDocumentTool.

Provider-agnostic: embedding and contextual-retrieval chat calls go through
LiteLLM (see ``holodeck.lib.litellm_support``); the SK vector-store
abstractions (connectors, record models) are unaffected.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from holodeck.lib.errors import HoloDeckError
from holodeck.lib.source_resolver import SourceResolver
from holodeck.models.llm import ProviderEnum

# Remote URI schemes that require SourceResolver
_REMOTE_SCHEMES = ("s3://", "az://", "https://", "http://")


def _is_remote_source(source: str) -> bool:
    """Check if a source string is a remote URI requiring SourceResolver."""
    return any(source.startswith(s) for s in _REMOTE_SCHEMES)


if TYPE_CHECKING:
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

logger = logging.getLogger(__name__)


class ToolInitializerError(HoloDeckError):
    """Raised when tool initialization fails."""

    pass


def resolve_embedding_model(agent: Agent) -> str:
    """Resolve embedding model name from agent config.

    Checks vectorstore and hierarchical-doc tool configs for explicit
    ``embedding_model`` values first. If explicit values conflict across tools,
    raises an error because embedding services are shared per agent. Falls back
    to provider defaults when no explicit value is configured.

    Args:
        agent: Agent configuration.

    Returns:
        Embedding model name string.

    Raises:
        ToolInitializerError: If explicit embedding_model values conflict.
    """
    from holodeck.models.tool import (
        HierarchicalDocumentToolConfig,
        VectorstoreTool,
    )

    explicit_models: dict[str, str] = {}
    if agent.tools:
        for tool in agent.tools:
            model_name: str | None = None
            if isinstance(tool, VectorstoreTool | HierarchicalDocumentToolConfig):
                model_name = tool.embedding_model

            if model_name:
                explicit_models[tool.name] = model_name

    if explicit_models:
        unique_models = sorted(set(explicit_models.values()))
        if len(unique_models) > 1:
            configured_models = ", ".join(
                f"{tool_name}={model_name}"
                for tool_name, model_name in sorted(explicit_models.items())
            )
            raise ToolInitializerError(
                "Conflicting embedding_model values detected across vectorstore/"
                "hierarchical_document tools. A single shared embedding service "
                "is used per agent, so explicit embedding_model values must match. "
                f"Found: {configured_models}"
            )
        return unique_models[0]

    # Determine which provider to use for defaults
    provider = _resolve_embedding_provider(agent)

    if provider == ProviderEnum.OLLAMA:
        return "nomic-embed-text:latest"
    # OpenAI / Azure OpenAI default
    return "text-embedding-3-small"


def create_embedding_service(agent: Agent) -> Any:
    """Create a LiteLLM-backed embedding service from agent config.

    For Anthropic provider: uses ``agent.embedding_provider`` config.
    For OpenAI/Azure/Ollama: uses ``agent.model`` config directly.

    Args:
        agent: Agent configuration.

    Returns:
        An initialized LiteLLMEmbeddingService instance.

    Raises:
        ToolInitializerError: If provider doesn't support embeddings.
    """
    from holodeck.lib.litellm_support import (
        LiteLLMEmbeddingService,
        resolve_litellm_model,
    )

    provider = _resolve_embedding_provider(agent)
    model_config = _resolve_embedding_model_config(agent)
    embedding_model = resolve_embedding_model(agent)

    logger.debug(
        "Creating embedding service: model=%s, provider=%s",
        embedding_model,
        provider,
    )

    spec = resolve_litellm_model(
        model_config, kind="embedding", model_name=embedding_model
    )
    return LiteLLMEmbeddingService(spec)


async def initialize_tools(
    agent: Agent,
    force_ingest: bool = False,
    execution_config: ExecutionConfig | None = None,
    base_dir: str | None = None,
    context_generator: Any | None = None,
) -> dict[str, Any]:
    """Initialize all vectorstore and hierarchical-doc tools for an agent.

    Creates embedding service, initializes each tool, returns dict keyed by
    tool config name. This is the main entry point for both backends.

    Args:
        agent: Agent configuration.
        force_ingest: Force re-ingestion of vector store source files.
        execution_config: Execution configuration for file processing.
        base_dir: Base directory for resolving relative source paths.
            If None, falls back to agent_base_dir context variable.
        context_generator: Optional pre-built ContextGenerator instance.
            When provided, takes highest priority for contextual embeddings.

    Returns:
        Dict mapping tool name to initialized tool instance.

    Raises:
        ToolInitializerError: On failure.
    """
    from holodeck.models.tool import (
        HierarchicalDocumentToolConfig,
    )
    from holodeck.models.tool import (
        VectorstoreTool as VectorstoreToolConfig,
    )

    if not agent.tools:
        return {}

    has_vs = any(isinstance(t, VectorstoreToolConfig) for t in agent.tools)
    has_hd = any(isinstance(t, HierarchicalDocumentToolConfig) for t in agent.tools)

    if not has_vs and not has_hd:
        return {}

    try:
        embedding_service = create_embedding_service(agent)
    except Exception as exc:
        raise ToolInitializerError(
            f"Failed to create embedding service: {exc}"
        ) from exc

    instances: dict[str, Any] = {}

    # Get provider type for dimension resolution
    provider_type = _resolve_embedding_provider(agent).value

    # Resolve base_dir: explicit > context var
    effective_base_dir = base_dir
    if effective_base_dir is None:
        from holodeck.config.context import agent_base_dir

        effective_base_dir = agent_base_dir.get()

    # Initialize vectorstore tools
    if has_vs:
        vs_instances = await _initialize_vectorstore_tools(
            agent=agent,
            embedding_service=embedding_service,
            force_ingest=force_ingest,
            execution_config=execution_config,
            provider_type=provider_type,
            base_dir=effective_base_dir,
        )
        instances.update(vs_instances)

    # Initialize hierarchical document tools
    if has_hd:
        hd_instances = await initialize_hierarchical_doc_tools(
            agent=agent,
            embedding_service=embedding_service,
            force_ingest=force_ingest,
            provider_type=provider_type,
            base_dir=effective_base_dir,
            context_generator=context_generator,
            execution_config=execution_config,
        )
        instances.update(hd_instances)

    return instances


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_embedding_provider(agent: Agent) -> ProviderEnum:
    """Determine which provider to use for embedding generation.

    For Anthropic main provider, falls back to ``agent.embedding_provider``.

    Args:
        agent: Agent configuration.

    Returns:
        The ProviderEnum to use for embeddings.

    Raises:
        ToolInitializerError: If Anthropic provider lacks embedding_provider.
    """
    if agent.model.provider == ProviderEnum.ANTHROPIC:
        if agent.embedding_provider is None:
            raise ToolInitializerError(
                "Anthropic provider does not support embeddings natively. "
                "Configure 'embedding_provider' in agent.yaml with an "
                "OpenAI, Azure OpenAI, or Ollama provider for vectorstore tools."
            )
        return agent.embedding_provider.provider
    return agent.model.provider


def _resolve_embedding_model_config(agent: Agent) -> Any:
    """Get the LLMProvider config to use for embedding credentials.

    For Anthropic agents, returns ``agent.embedding_provider``.
    For others, returns ``agent.model``.

    Args:
        agent: Agent configuration.

    Returns:
        LLMProvider instance for embedding credentials.
    """
    if (
        agent.model.provider == ProviderEnum.ANTHROPIC
        and agent.embedding_provider is not None
    ):
        return agent.embedding_provider
    return agent.model


async def _initialize_vectorstore_tools(
    agent: Agent,
    embedding_service: Any,
    force_ingest: bool,
    execution_config: ExecutionConfig | None,
    provider_type: str,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Initialize all vectorstore tools from agent config.

    Args:
        agent: Agent configuration.
        embedding_service: Embedding service (LiteLLMEmbeddingService).
        force_ingest: Force re-ingestion of source files.
        execution_config: Execution configuration.
        provider_type: Provider type string for dimension resolution.
        base_dir: Base directory for resolving relative source paths.

    Returns:
        Dict mapping tool name to initialized VectorStoreTool instance.

    Raises:
        ToolInitializerError: If any tool fails to initialize.
    """
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    instances: dict[str, Any] = {}
    cleanup_stack: list[Path] = []

    try:
        for tool_config in agent.tools or []:
            if not isinstance(tool_config, VectorstoreToolConfig):
                continue

            try:
                # Resolve remote source if needed
                effective_config = tool_config
                is_remote = _is_remote_source(tool_config.source)
                resolved_local_path: Path | None = None
                if is_remote:
                    resolved = await SourceResolver.resolve(
                        tool_config.source, base_dir
                    )
                    if resolved.temp_dir:
                        cleanup_stack.append(resolved.temp_dir)
                    resolved_local_path = resolved.local_path
                    effective_config = tool_config.model_copy(
                        update={"source": str(resolved_local_path)}
                    )

                tool = VectorStoreTool(
                    effective_config,
                    base_dir=base_dir,
                    execution_config=execution_config,
                )
                if is_remote and resolved_local_path is not None:
                    tool.set_source_context(
                        source_root=resolved_local_path,
                        is_remote=True,
                    )
                tool.set_embedding_service(embedding_service)
                await tool.initialize(
                    force_ingest=force_ingest, provider_type=provider_type
                )
                instances[tool_config.name] = tool
                logger.info("Initialized vectorstore tool: %s", tool_config.name)
            except Exception as exc:
                raise ToolInitializerError(
                    f"Failed to initialize vectorstore tool "
                    f"'{tool_config.name}': {exc}"
                ) from exc
    except Exception:
        for temp_dir in cleanup_stack:
            await SourceResolver.cleanup(temp_dir)
        raise

    return instances


def _resolve_context_model_config(
    agent: Agent,
    tool_config: Any,
) -> Any:
    """Resolve LLMProvider for contextual embedding generation.

    Resolution chain (highest to lowest priority):
    1. ``tool_config.context_model`` — explicit per-tool override
    2. ``agent.embedding_provider`` — for Anthropic agents
    3. ``agent.model`` — fallback to the main agent model

    Args:
        agent: Agent configuration.
        tool_config: HierarchicalDocumentToolConfig instance.

    Returns:
        LLMProvider instance for context generation.
    """
    from holodeck.models.llm import LLMProvider as LLMProviderModel

    context_model = getattr(tool_config, "context_model", None)
    if isinstance(context_model, LLMProviderModel) and context_model is not None:
        return context_model

    if agent.embedding_provider is not None:
        return agent.embedding_provider

    return agent.model


def _resolve_context_generator(
    agent: Agent,
    tool_config: Any,
    context_generator: Any | None = None,
) -> Any | None:
    """Resolve the ContextGenerator for a hierarchical document tool.

    Implements a 4-tier priority chain:

    1. Caller-provided ``context_generator`` (highest priority)
    2. ``tool_config.context_model`` → LiteLLM-backed LLMContextGenerator
    3. Anthropic agent provider → ClaudeSDKContextGenerator
    4. None (graceful degradation)

    Args:
        agent: Agent configuration.
        tool_config: HierarchicalDocumentToolConfig instance.
        context_generator: Pre-built ContextGenerator, if any.

    Returns:
        A ContextGenerator instance, or None if none could be resolved.
    """
    # Priority 1: caller-provided context generator
    if context_generator is not None:
        logger.debug(
            "Using caller-provided context generator for tool '%s'",
            tool_config.name,
        )
        return context_generator

    # Priority 2: tool_config.context_model → LiteLLM-backed LLMContextGenerator
    if tool_config.context_model is not None:
        from holodeck.lib.litellm_support import resolve_litellm_model
        from holodeck.lib.llm_context_generator import LLMContextGenerator

        context_llm = _resolve_context_model_config(agent, tool_config)
        spec = resolve_litellm_model(context_llm, kind="chat")
        logger.debug(
            "Created LLMContextGenerator from context_model for tool '%s': "
            "provider=%s, model=%s",
            tool_config.name,
            context_llm.provider,
            context_llm.name,
        )
        return LLMContextGenerator(
            model_spec=spec,
            max_context_tokens=tool_config.context_max_tokens,
            concurrency=tool_config.context_concurrency,
        )

    # Priority 3: Anthropic agent → ClaudeSDKContextGenerator
    if agent.model.provider == ProviderEnum.ANTHROPIC:
        from holodeck.lib.claude_context_generator import (
            ClaudeContextConfig,
            ClaudeSDKContextGenerator,
        )

        logger.debug(
            "Creating ClaudeSDKContextGenerator for Anthropic agent tool '%s'",
            tool_config.name,
        )
        return ClaudeSDKContextGenerator(
            config=ClaudeContextConfig(
                concurrency=tool_config.context_concurrency,
            ),
            max_context_tokens=tool_config.context_max_tokens,
        )

    # Priority 4: no generator available
    logger.debug(
        "No context generator available for tool '%s'",
        tool_config.name,
    )
    return None


async def initialize_hierarchical_doc_tools(
    agent: Agent,
    embedding_service: Any,
    force_ingest: bool,
    provider_type: str,
    base_dir: str | None = None,
    context_generator: Any | None = None,
    execution_config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Initialize all hierarchical document tools from agent config.

    Args:
        agent: Agent configuration.
        embedding_service: Embedding service (LiteLLMEmbeddingService).
        force_ingest: Force re-ingestion of source files.
        provider_type: Provider type string for dimension resolution.
        base_dir: Base directory for resolving relative source paths.
        context_generator: Optional pre-built ContextGenerator instance.
        execution_config: Execution configuration forwarded to each tool so
            ``execution.file_timeout`` (and download_timeout / cache_dir)
            from agent.yaml reaches the FileProcessor used during ingest.

    Returns:
        Dict mapping tool name to initialized HierarchicalDocumentTool instance.

    Raises:
        ToolInitializerError: If any tool fails to initialize.
    """
    from holodeck.models.tool import HierarchicalDocumentToolConfig
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

    instances: dict[str, Any] = {}
    cleanup_stack: list[Path] = []

    try:
        for tool_config in agent.tools or []:
            if not isinstance(tool_config, HierarchicalDocumentToolConfig):
                continue

            try:
                # Resolve remote source if needed
                effective_config = tool_config
                is_remote = _is_remote_source(tool_config.source)
                resolved_local_path: Path | None = None
                if is_remote:
                    resolved = await SourceResolver.resolve(
                        tool_config.source, base_dir
                    )
                    if resolved.temp_dir:
                        cleanup_stack.append(resolved.temp_dir)
                    resolved_local_path = resolved.local_path
                    effective_config = tool_config.model_copy(
                        update={"source": str(resolved_local_path)}
                    )

                tool = HierarchicalDocumentTool(
                    effective_config,
                    base_dir=base_dir,
                    execution_config=execution_config,
                )
                if is_remote and resolved_local_path is not None:
                    tool.set_source_context(
                        source_root=resolved_local_path,
                        is_remote=True,
                    )
                tool.set_embedding_service(embedding_service)

                # Resolve context generator via 4-tier priority chain
                resolved_generator = _resolve_context_generator(
                    agent=agent,
                    tool_config=tool_config,
                    context_generator=context_generator,
                )
                if resolved_generator is not None:
                    tool.set_context_generator(resolved_generator)

                await tool.initialize(
                    force_ingest=force_ingest, provider_type=provider_type
                )
                instances[tool_config.name] = tool
                logger.info(
                    "Initialized hierarchical document tool: %s",
                    tool_config.name,
                )
            except Exception as exc:
                raise ToolInitializerError(
                    f"Failed to initialize hierarchical document tool "
                    f"'{tool_config.name}': {exc}"
                ) from exc
    except Exception:
        for temp_dir in cleanup_stack:
            await SourceResolver.cleanup(temp_dir)
        raise

    return instances


async def initialize_single_tool(
    agent: Agent,
    tool_name: str,
    force_ingest: bool = False,
    progress_callback: Callable[[int, int | None], None] | None = None,
    source_override: Path | None = None,
    execution_config: ExecutionConfig | None = None,
) -> None:
    """Initialize a single tool by name.

    Finds the tool config in the agent, creates an embedding service,
    and initializes the tool. Used by ToolInitManager for async init jobs.

    Args:
        agent: Agent configuration containing tool definitions.
        tool_name: Name of the tool to initialize.
        force_ingest: If True, force re-ingestion of all source files.
        progress_callback: Optional callback for progress reporting.
        source_override: Optional path to use instead of the configured source.
        execution_config: Execution configuration forwarded to the tool so
            ``execution.file_timeout`` (and download_timeout / cache_dir)
            from agent.yaml reaches the FileProcessor used during ingest.

    Raises:
        ToolInitializerError: If tool not found, not initializable, or init fails.
    """
    from holodeck.models.tool import HierarchicalDocumentToolConfig
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
    from holodeck.tools.vectorstore_tool import VectorStoreTool

    # Find tool config by name
    tool_config = None
    for t in agent.tools or []:
        if t.name == tool_name:
            tool_config = t
            break

    if tool_config is None:
        raise ToolInitializerError(f"Tool not found: '{tool_name}'")

    # Type check — only vectorstore and hierarchical_document tools can be initialized
    is_vectorstore = isinstance(tool_config, VectorstoreToolConfig)
    is_hierarchical = isinstance(tool_config, HierarchicalDocumentToolConfig)

    if not (is_vectorstore or is_hierarchical):
        raise ToolInitializerError(
            f"Tool '{tool_name}' (type: {tool_config.type}) does not support "
            "initialization. Only vectorstore and hierarchical_document tools "
            "can be initialized."
        )

    # Determine if the original source is remote (before override).
    # Both VectorstoreTool and HierarchicalDocumentToolConfig have .source,
    # but mypy can't narrow the union after isinstance checks above.
    source_str: str = getattr(tool_config, "source", "")
    is_remote = _is_remote_source(source_str)

    # Source override: use Pydantic v2 model_copy to avoid mutating original config
    if source_override is not None:
        tool_config = tool_config.model_copy(update={"source": str(source_override)})

    # Create embedding service
    embedding_service = create_embedding_service(agent)

    # Resolve provider type for dimension resolution
    provider_type = _resolve_embedding_provider(agent).value

    if is_vectorstore:
        vs_tool = VectorStoreTool(
            cast(VectorstoreToolConfig, tool_config),
            execution_config=execution_config,
        )
        if is_remote and source_override is not None:
            vs_tool.set_source_context(source_root=source_override, is_remote=True)
        vs_tool.set_embedding_service(embedding_service)
        await vs_tool.initialize(
            force_ingest=force_ingest,
            provider_type=provider_type,
            progress_callback=progress_callback,
        )
    else:
        hd_tool = HierarchicalDocumentTool(
            cast(HierarchicalDocumentToolConfig, tool_config),
            execution_config=execution_config,
        )
        if is_remote and source_override is not None:
            hd_tool.set_source_context(source_root=source_override, is_remote=True)
        hd_tool.set_embedding_service(embedding_service)
        await hd_tool.initialize(
            force_ingest=force_ingest,
            provider_type=provider_type,
            progress_callback=progress_callback,
        )

    logger.info("Initialized single tool: %s", tool_name)
