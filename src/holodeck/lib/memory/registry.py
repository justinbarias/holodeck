"""Create the memory backend that a ``temporal_memory`` tool names (spec 043).

Backend modules are imported only when one is created, so importing
``holodeck.lib.memory`` never imports a graph client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holodeck.lib.memory.errors import MemoryConfigError
from holodeck.lib.memory.ports import MemoryBackend
from holodeck.models.llm import LLMProvider
from holodeck.models.tool import GraphBackendConfig, MemorySchemaHints, ReasoningEffort

if TYPE_CHECKING:
    from holodeck.lib.memory.backends.graphiti import EmbeddingService


def create_backend(
    config: GraphBackendConfig,
    extraction_model: LLMProvider,
    embedding_service: EmbeddingService,
    schema_hints: MemorySchemaHints | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    close_on_text_change: bool = False,
) -> MemoryBackend:
    """Return the backend for ``config.provider``. Makes no I/O.

    Raises:
        MemoryConfigError: The provider has no backend.
    """
    provider: str = config.provider
    if provider == "graphiti":
        from holodeck.lib.memory.backends.graphiti import GraphitiBackend

        return GraphitiBackend(
            config,
            extraction_model,
            embedding_service,
            schema_hints,
            reasoning_effort,
            close_on_text_change,
        )
    raise MemoryConfigError(f"memory backend '{provider}' is not supported")
