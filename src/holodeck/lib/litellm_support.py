"""LiteLLM support for the RAG layer: provider mapping + embedding service.

Maps HoloDeck ``LLMProvider`` configs to LiteLLM call arguments and provides
the embedding service consumed through ``EmbeddingServiceMixin``. This module
replaces the Semantic Kernel ``*TextEmbedding`` / ``*ChatCompletion`` services
in the RAG inference path; the SK vector-store abstractions (connectors,
``@vectorstoremodel``, ``VectorStoreField``) are unaffected.

Provider-error boundary
-----------------------
``LiteLLMEmbeddingService.generate_embeddings`` is the boundary between the
provider SDK and HoloDeck's tools. Every failure raised while calling the
provider or decoding its response is re-raised as :class:`EmbeddingServiceError`
with the original exception chained as ``__cause__`` and named in the message,
so callers (``VectorStoreTool``, ``HierarchicalDocumentTool``, the tool
initializer's ``ToolInitializerError``) surface one stable type without losing
the provider's status code or message. Tools never substitute placeholder
vectors for a failed provider call: a failure during ingest aborts tool
initialization, and a failure at query time surfaces as the tool error.
Placeholder embeddings are used only when no embedding service is injected.

Output dimensions
-----------------
The service forwards an explicit ``dimensions`` value to the provider on every
request and omits the parameter when ``None``. The tool initializer builds one
shared service per agent (no ``dimensions``) and a dedicated service for each
tool that sets ``embedding_dimensions``, so a configured override reaches the
provider instead of only being checked against the returned vector length.
For Azure (``openai/<deployment>``) the parameter is allow-listed through
``allowed_openai_params`` because the deployment alias does not carry the
``text-embedding-3`` marker LiteLLM's client-side guard looks for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from holodeck.models.llm import ProviderEnum

if TYPE_CHECKING:
    from holodeck.models.llm import LLMProvider

logger = logging.getLogger(__name__)


def _azure_v1_base_url(endpoint: str) -> str:
    """Normalize an Azure endpoint to the OpenAI-compatible ``/openai/v1`` base.

    Mirrors the Azure routing of the OpenAI Agents backend: both
    ``*.openai.azure.com`` and Foundry ``*.services.ai.azure.com`` resources
    serve the OpenAI-compatible v1 surface there. A bare resource endpoint has
    ``/openai/v1`` appended; an endpoint already targeting the v1 surface is
    used as-is.
    """
    base = endpoint.rstrip("/")
    if base.endswith("/openai/v1"):
        return base
    return f"{base}/openai/v1"


@dataclass(frozen=True)
class LiteLLMModelSpec:
    """Resolved LiteLLM call arguments for a HoloDeck ``LLMProvider``.

    Attributes:
        model: LiteLLM model string, including provider prefix where needed
            (e.g. ``openai/<deployment>`` for Azure's OpenAI-compatible v1
            surface, ``ollama/<model>``).
        api_key: Raw API key, or None for providers without one (Ollama).
        api_base: API endpoint, or None to use the provider default.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None

    def call_kwargs(self) -> dict[str, Any]:
        """Return model + connection kwargs for litellm call functions.

        Returns:
            Dict with ``model`` and any non-None of ``api_key``, ``api_base``.
        """
        kwargs: dict[str, Any] = {"model": self.model}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        return kwargs


def resolve_litellm_model(
    model_config: LLMProvider,
    kind: Literal["embedding", "chat"],
    model_name: str | None = None,
) -> LiteLLMModelSpec:
    """Map a HoloDeck ``LLMProvider`` to LiteLLM call arguments.

    Single place for the provider → LiteLLM mapping used by both the
    embedding service and the contextual-retrieval chat path.

    Args:
        model_config: LLMProvider carrying provider, credentials, endpoint.
        kind: ``"embedding"`` or ``"chat"`` — Anthropic is chat-only.
        model_name: Override for the model/deployment name. Defaults to
            ``model_config.name``; the embedding path passes the resolved
            embedding model here because ``model_config.name`` is the chat
            model.

    Returns:
        A LiteLLMModelSpec ready for ``litellm.acompletion`` /
        ``litellm.aembedding``.

    Raises:
        ToolInitializerError: If the provider doesn't support the requested
            kind.
    """
    provider = model_config.provider
    name = model_name or model_config.name
    api_key = (
        model_config.api_key.get_secret_value()
        if model_config.api_key is not None
        else None
    )

    if provider == ProviderEnum.OPENAI:
        return LiteLLMModelSpec(model=name, api_key=api_key)

    if provider == ProviderEnum.AZURE_OPENAI:
        # Azure runs on the OpenAI-compatible /openai/v1 surface (same
        # convention as the OpenAI Agents backend), so LiteLLM's generic
        # ``openai/`` provider is used with the deployment as the model and
        # no api-version. The Azure key authenticates as a bearer token.
        return LiteLLMModelSpec(
            model=f"openai/{name}",
            api_key=api_key,
            api_base=_azure_v1_base_url(model_config.endpoint or ""),
        )

    if provider == ProviderEnum.OLLAMA:
        return LiteLLMModelSpec(
            model=f"ollama/{name}",
            api_base=model_config.endpoint if model_config.endpoint else None,
        )

    if provider == ProviderEnum.ANTHROPIC and kind == "chat":
        return LiteLLMModelSpec(model=f"anthropic/{name}", api_key=api_key)

    from holodeck.lib.tool_initializer import ToolInitializerError

    if kind == "embedding":
        raise ToolInitializerError(
            f"Embedding service not supported for provider: {provider}. "
            "Vectorstore tools require OpenAI, Azure OpenAI, or Ollama provider."
        )
    raise ToolInitializerError(
        f"Chat service not supported for provider: {provider}. "
        "Context model requires OpenAI, Azure OpenAI, Anthropic, or Ollama provider."
    )


class EmbeddingServiceError(Exception):
    """Raised when the embedding provider call fails or returns unusable data.

    The original provider exception is chained as ``__cause__`` and its type
    and message are included in this error's message.
    """


class LiteLLMEmbeddingService:
    """Embedding service backed by ``litellm.aembedding``.

    Exposes the single method the vectorstore / hierarchical-document tools
    call through ``EmbeddingServiceMixin``:
    ``await service.generate_embeddings(list[str])``.

    Attributes:
        _spec: Resolved LiteLLM call arguments.
        _dimensions: Optional output-dimension override, forwarded to the
            provider for models that support it (e.g. OpenAI v3 embeddings).
    """

    def __init__(self, spec: LiteLLMModelSpec, dimensions: int | None = None) -> None:
        """Initialize the embedding service.

        Args:
            spec: Resolved LiteLLM call arguments (embedding kind).
            dimensions: Optional output dimensions; omitted from the request
                when None so providers that reject the parameter are safe.
        """
        self._spec = spec
        self._dimensions = dimensions

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate one embedding vector per input text.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors, in input order.

        Raises:
            EmbeddingServiceError: If the provider call fails or the response
                cannot be decoded; the provider exception is chained.
        """
        if not texts:
            return []

        import litellm

        kwargs = self._spec.call_kwargs()
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
            if self._spec.model.startswith("openai/"):
                # Azure deployments run through LiteLLM's generic ``openai/``
                # provider under an arbitrary deployment alias. LiteLLM's
                # client-side guard only permits ``dimensions`` when the model
                # name contains ``text-embedding-3``, so the alias must be
                # allow-listed explicitly; the provider then validates it.
                kwargs["allowed_openai_params"] = ["dimensions"]

        try:
            response = await litellm.aembedding(input=texts, **kwargs)
            items = sorted(response.data, key=lambda item: item["index"])
            vectors = [[float(value) for value in item["embedding"]] for item in items]
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Embedding request failed for model '{self._spec.model}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if len(vectors) != len(texts):
            raise EmbeddingServiceError(
                f"Embedding response for model '{self._spec.model}' returned "
                f"{len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors
