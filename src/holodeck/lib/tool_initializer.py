"""Shared tool initialization for VectorStoreTool and HierarchicalDocumentTool.

Provider-agnostic: works for both SK (AgentFactory) and Claude (ClaudeBackend) paths.
The embedding service created here uses SK TextEmbedding classes — these are lightweight
wrappers around OpenAI/Azure/Ollama APIs and do NOT require the full SK kernel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from holodeck.lib.errors import HoloDeckError
from holodeck.models.llm import ProviderEnum

if TYPE_CHECKING:
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

logger = logging.getLogger(__name__)


class ToolInitializerError(HoloDeckError):
    """Raised when tool initialization fails."""

    pass


def resolve_embedding_model(agent: Agent) -> str:
    """Resolve embedding model name from agent config.

    Checks vectorstore tool configs for explicit ``embedding_model`` first,
    then falls back to provider defaults.

    Args:
        agent: Agent configuration.

    Returns:
        Embedding model name string.
    """
    from holodeck.models.tool import (
        HierarchicalDocumentToolConfig,
        VectorstoreTool,
    )

    # Check if any vectorstore/hierarchical-doc tool has explicit embedding_model
    if agent.tools:
        for tool in agent.tools:
            if isinstance(tool, VectorstoreTool) and tool.embedding_model:
                return tool.embedding_model
            if isinstance(tool, HierarchicalDocumentToolConfig):
                emb_model = getattr(tool, "embedding_model", None)
                if emb_model:
                    return str(emb_model)

    # Determine which provider to use for defaults
    provider = _resolve_embedding_provider(agent)

    if provider == ProviderEnum.OLLAMA:
        return "nomic-embed-text:latest"
    # OpenAI / Azure OpenAI default
    return "text-embedding-3-small"


def create_embedding_service(agent: Agent) -> Any:
    """Create an SK TextEmbedding service from agent config.

    For Anthropic provider: uses ``agent.embedding_provider`` config.
    For OpenAI/Azure/Ollama: uses ``agent.model`` config directly.

    Args:
        agent: Agent configuration.

    Returns:
        An initialized TextEmbedding service instance.

    Raises:
        ToolInitializerError: If provider doesn't support embeddings.
    """
    from semantic_kernel.connectors.ai.open_ai import (
        AzureTextEmbedding,
        OpenAITextEmbedding,
    )

    provider = _resolve_embedding_provider(agent)
    model_config = _resolve_embedding_model_config(agent)
    embedding_model = resolve_embedding_model(agent)

    logger.debug(
        "Creating embedding service: model=%s, provider=%s",
        embedding_model,
        provider,
    )

    if provider == ProviderEnum.OPENAI:
        return OpenAITextEmbedding(
            ai_model_id=embedding_model,
            api_key=model_config.api_key,
        )

    if provider == ProviderEnum.AZURE_OPENAI:
        return AzureTextEmbedding(
            deployment_name=embedding_model,
            endpoint=model_config.endpoint,
            api_key=model_config.api_key,
        )

    if provider == ProviderEnum.OLLAMA:
        try:
            from semantic_kernel.connectors.ai.ollama import OllamaTextEmbedding
        except ImportError as exc:
            raise ToolInitializerError(
                "Ollama provider requires 'ollama' package. "
                "Install with: pip install ollama"
            ) from exc

        return OllamaTextEmbedding(
            ai_model_id=embedding_model,
            host=model_config.endpoint if model_config.endpoint else None,
        )

    raise ToolInitializerError(
        f"Embedding service not supported for provider: {provider}. "
        "Vectorstore tools require OpenAI, Azure OpenAI, or Ollama provider."
    )


async def initialize_tools(
    agent: Agent,
    force_ingest: bool = False,
    execution_config: ExecutionConfig | None = None,
    chat_service: Any | None = None,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Initialize all vectorstore and hierarchical-doc tools for an agent.

    Creates embedding service, initializes each tool, returns dict keyed by
    tool config name. This is the main entry point for both backends.

    Args:
        agent: Agent configuration.
        force_ingest: Force re-ingestion of vector store source files.
        execution_config: Execution configuration for file processing.
        chat_service: Optional chat service for hierarchical doc tools.
        base_dir: Base directory for resolving relative source paths.
            If None, falls back to agent_base_dir context variable.

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
        hd_instances = await _initialize_hierarchical_doc_tools(
            agent=agent,
            embedding_service=embedding_service,
            chat_service=chat_service,
            force_ingest=force_ingest,
            provider_type=provider_type,
            base_dir=effective_base_dir,
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
        embedding_service: SK TextEmbedding service.
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

    for tool_config in agent.tools or []:
        if not isinstance(tool_config, VectorstoreToolConfig):
            continue

        try:
            tool = VectorStoreTool(
                tool_config, base_dir=base_dir, execution_config=execution_config
            )
            tool.set_embedding_service(embedding_service)
            await tool.initialize(
                force_ingest=force_ingest, provider_type=provider_type
            )
            instances[tool_config.name] = tool
            logger.info("Initialized vectorstore tool: %s", tool_config.name)
        except Exception as exc:
            raise ToolInitializerError(
                f"Failed to initialize vectorstore tool '{tool_config.name}': {exc}"
            ) from exc

    return instances


async def _initialize_hierarchical_doc_tools(
    agent: Agent,
    embedding_service: Any,
    chat_service: Any | None,
    force_ingest: bool,
    provider_type: str,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Initialize all hierarchical document tools from agent config.

    Args:
        agent: Agent configuration.
        embedding_service: SK TextEmbedding service.
        chat_service: Optional chat service for context generation.
        force_ingest: Force re-ingestion of source files.
        provider_type: Provider type string for dimension resolution.
        base_dir: Base directory for resolving relative source paths.

    Returns:
        Dict mapping tool name to initialized HierarchicalDocumentTool instance.

    Raises:
        ToolInitializerError: If any tool fails to initialize.
    """
    from holodeck.models.tool import HierarchicalDocumentToolConfig
    from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

    instances: dict[str, Any] = {}

    for tool_config in agent.tools or []:
        if not isinstance(tool_config, HierarchicalDocumentToolConfig):
            continue

        try:
            tool = HierarchicalDocumentTool(tool_config, base_dir=base_dir)
            tool.set_embedding_service(embedding_service)
            if chat_service is not None:
                tool.set_chat_service(chat_service)
            await tool.initialize(
                force_ingest=force_ingest, provider_type=provider_type
            )
            instances[tool_config.name] = tool
            logger.info("Initialized hierarchical document tool: %s", tool_config.name)
        except Exception as exc:
            raise ToolInitializerError(
                f"Failed to initialize hierarchical document tool "
                f"'{tool_config.name}': {exc}"
            ) from exc

    return instances
