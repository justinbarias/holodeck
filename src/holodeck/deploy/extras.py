"""Choose the ``holodeck-ai`` extras a deployed agent image needs.

A serving image installs only the extras its agent uses. This module holds the
one rule for that choice. Provider extras come from the retrieval registry, so
they cannot drift from the adapters that need them.

Evaluation extras (``eval``, ``nlp``, ``optimizer``) are never added: an image
runs ``holodeck serve``, not ``holodeck test``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holodeck.lib.retrieval.registry import PROVIDERS
from holodeck.models.llm import ProviderEnum
from holodeck.models.tool import (
    DatabaseConfig,
    HierarchicalDocumentToolConfig,
    VectorstoreTool,
)

if TYPE_CHECKING:
    from holodeck.models.agent import Agent

# Model providers served by the OpenAI Agents backend (BackendSelector).
_OPENAI_AGENTS_PROVIDERS = frozenset({ProviderEnum.OPENAI, ProviderEnum.AZURE_OPENAI})

# Model providers served by the Claude backend, which loads the Claude GenAI
# instrumentation when traces are on.
_CLAUDE_BACKEND_PROVIDERS = frozenset({ProviderEnum.ANTHROPIC, ProviderEnum.OLLAMA})

# Remote tool-source URI schemes and the extra whose client reads them.
_SOURCE_SCHEME_EXTRAS: dict[str, str] = {
    "az://": "azure-blob",
    "s3://": "s3",
}


def required_extras(agent: Agent, *, temporal_worker: bool = False) -> list[str]:
    """Return the ``holodeck-ai`` extras that ``agent`` needs to serve.

    Args:
        agent: Loaded agent configuration.
        temporal_worker: The image runs ``holodeck worker`` (Temporal), which
            needs the ``temporal`` extra. Nothing in an agent file says this,
            so the caller decides.

    Returns:
        Extra names, sorted and without duplicates. A database that the agent
        names by reference and the loader could not resolve adds nothing.
    """
    extras: set[str] = set()

    provider = agent.model.provider
    if provider in _OPENAI_AGENTS_PROVIDERS:
        extras.add("openai-agents")

    observability = agent.observability
    if observability is not None and observability.enabled:
        if provider in _CLAUDE_BACKEND_PROVIDERS and observability.traces.enabled:
            extras.add("claude-otel")
        exporters = observability.exporters
        if exporters.prometheus and exporters.prometheus.enabled:
            extras.add("otel-prometheus")
        if exporters.azure_monitor and exporters.azure_monitor.enabled:
            extras.add("otel-azmon")

    for tool in agent.tools or []:
        if not isinstance(tool, VectorstoreTool | HierarchicalDocumentToolConfig):
            continue
        if isinstance(tool.database, DatabaseConfig):
            entry = PROVIDERS.get(tool.database.provider)
            if entry is not None and entry.extra is not None:
                extras.add(entry.extra)
        for scheme, extra in _SOURCE_SCHEME_EXTRAS.items():
            if tool.source.startswith(scheme):
                extras.add(extra)
        if isinstance(tool, HierarchicalDocumentToolConfig):
            keyword_index = tool.keyword_index
            if keyword_index is not None and keyword_index.provider == "opensearch":
                extras.add("opensearch")

    if temporal_worker:
        extras.add("temporal")

    return sorted(extras)
