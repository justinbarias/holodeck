"""Tests for litellm_support: provider mapping + LiteLLM embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.litellm_support import (
    EmbeddingServiceError,
    LiteLLMEmbeddingService,
    LiteLLMModelSpec,
    resolve_litellm_model,
)
from holodeck.lib.tool_initializer import ToolInitializerError
from holodeck.models.llm import LLMProvider, ProviderEnum


def make_provider(**overrides: object) -> LLMProvider:
    """Create an LLMProvider with test defaults."""
    defaults: dict[str, object] = {
        "provider": ProviderEnum.OPENAI,
        "name": "gpt-4o-mini",
        "api_key": "test-key",
    }
    defaults.update(overrides)
    return LLMProvider(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestResolveLiteLLMModel:
    """Tests for resolve_litellm_model provider mapping."""

    def test_openai_chat(self) -> None:
        spec = resolve_litellm_model(make_provider(), kind="chat")
        assert spec.model == "gpt-4o-mini"
        assert spec.api_key == "test-key"
        assert spec.api_base is None

    def test_openai_embedding_uses_model_name_override(self) -> None:
        spec = resolve_litellm_model(
            make_provider(), kind="embedding", model_name="text-embedding-3-small"
        )
        assert spec.model == "text-embedding-3-small"
        assert spec.api_key == "test-key"

    def test_azure_openai_uses_v1_surface(self) -> None:
        """Azure routes through the OpenAI-compatible /openai/v1 surface."""
        config = make_provider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="gpt-4o",
            endpoint="https://example.openai.azure.com",
        )
        spec = resolve_litellm_model(config, kind="chat")
        assert spec.model == "openai/gpt-4o"
        assert spec.api_base == "https://example.openai.azure.com/openai/v1"
        assert spec.api_key == "test-key"

    def test_azure_openai_v1_endpoint_not_doubled(self) -> None:
        """An endpoint already targeting /openai/v1 is used as-is."""
        config = make_provider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="embed-deploy",
            endpoint="https://example.services.ai.azure.com/openai/v1",
        )
        spec = resolve_litellm_model(
            config, kind="embedding", model_name="text-embedding-3-small"
        )
        assert spec.model == "openai/text-embedding-3-small"
        assert spec.api_base == "https://example.services.ai.azure.com/openai/v1"

    def test_ollama_prefixes_model_and_maps_endpoint(self) -> None:
        config = make_provider(
            provider=ProviderEnum.OLLAMA,
            name="llama3",
            endpoint="http://localhost:11434",
            api_key=None,
        )
        spec = resolve_litellm_model(config, kind="chat")
        assert spec.model == "ollama/llama3"
        assert spec.api_base == "http://localhost:11434"
        assert spec.api_key is None

    def test_ollama_without_endpoint(self) -> None:
        config = make_provider(
            provider=ProviderEnum.OLLAMA, name="nomic-embed-text", api_key=None
        )
        spec = resolve_litellm_model(
            config, kind="embedding", model_name="nomic-embed-text:latest"
        )
        assert spec.model == "ollama/nomic-embed-text:latest"
        assert spec.api_base is None

    def test_anthropic_chat_supported(self) -> None:
        config = make_provider(
            provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-5"
        )
        spec = resolve_litellm_model(config, kind="chat")
        assert spec.model == "anthropic/claude-sonnet-4-5"
        assert spec.api_key == "test-key"

    def test_anthropic_embedding_raises(self) -> None:
        config = make_provider(provider=ProviderEnum.ANTHROPIC, name="claude")
        with pytest.raises(ToolInitializerError, match="Embedding service not"):
            resolve_litellm_model(config, kind="embedding")


@pytest.mark.unit
class TestLiteLLMModelSpecCallKwargs:
    """Tests for LiteLLMModelSpec.call_kwargs None-filtering."""

    def test_full_spec(self) -> None:
        spec = LiteLLMModelSpec(
            model="openai/gpt-4o",
            api_key="k",
            api_base="https://e",
        )
        assert spec.call_kwargs() == {
            "model": "openai/gpt-4o",
            "api_key": "k",
            "api_base": "https://e",
        }

    def test_none_fields_omitted(self) -> None:
        spec = LiteLLMModelSpec(model="ollama/llama3")
        assert spec.call_kwargs() == {"model": "ollama/llama3"}


@pytest.mark.unit
class TestLiteLLMEmbeddingService:
    """Tests for the LiteLLM embedding shim."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_returns_float_lists(self) -> None:
        mock_response = type(
            "R",
            (),
            {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            },
        )()
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m", api_key="k"))
        with patch(
            "litellm.aembedding", new=AsyncMock(return_value=mock_response)
        ) as mock_embed:
            result = await service.generate_embeddings(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert all(
            isinstance(vec, list) and isinstance(v, float)
            for vec in result
            for v in vec
        )
        mock_embed.assert_awaited_once_with(input=["a", "b"], model="m", api_key="k")

    @pytest.mark.asyncio
    async def test_results_sorted_by_index(self) -> None:
        mock_response = type(
            "R",
            (),
            {
                "data": [
                    {"index": 1, "embedding": [0.3]},
                    {"index": 0, "embedding": [0.1]},
                ]
            },
        )()
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m"))
        with patch("litellm.aembedding", new=AsyncMock(return_value=mock_response)):
            result = await service.generate_embeddings(["a", "b"])

        assert result == [[0.1], [0.3]]

    @pytest.mark.asyncio
    async def test_dimensions_forwarded_when_set(self) -> None:
        mock_response = type("R", (), {"data": [{"index": 0, "embedding": [0.1]}]})()
        service = LiteLLMEmbeddingService(
            LiteLLMModelSpec(model="text-embedding-3-small"), dimensions=256
        )
        with patch(
            "litellm.aembedding", new=AsyncMock(return_value=mock_response)
        ) as mock_embed:
            await service.generate_embeddings(["a"])

        assert mock_embed.await_args.kwargs["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_azure_alias_allow_lists_dimensions(self) -> None:
        """An Azure deployment alias needs LiteLLM's guard bypassed."""
        mock_response = type("R", (), {"data": [{"index": 0, "embedding": [0.1]}]})()
        service = LiteLLMEmbeddingService(
            LiteLLMModelSpec(
                model="openai/embedding-prod",
                api_key="k",
                api_base="https://r.openai.azure.com/openai/v1",
            ),
            dimensions=512,
        )
        with patch(
            "litellm.aembedding", new=AsyncMock(return_value=mock_response)
        ) as mock_embed:
            await service.generate_embeddings(["a"])

        kwargs = mock_embed.await_args.kwargs
        assert kwargs["dimensions"] == 512
        assert kwargs["allowed_openai_params"] == ["dimensions"]

    @pytest.mark.asyncio
    async def test_openai_bare_model_has_no_allow_list(self) -> None:
        mock_response = type("R", (), {"data": [{"index": 0, "embedding": [0.1]}]})()
        service = LiteLLMEmbeddingService(
            LiteLLMModelSpec(model="text-embedding-3-small"), dimensions=256
        )
        with patch(
            "litellm.aembedding", new=AsyncMock(return_value=mock_response)
        ) as mock_embed:
            await service.generate_embeddings(["a"])

        assert "allowed_openai_params" not in mock_embed.await_args.kwargs

    @pytest.mark.asyncio
    async def test_dimensions_absent_when_none(self) -> None:
        mock_response = type("R", (), {"data": [{"index": 0, "embedding": [0.1]}]})()
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m"))
        with patch(
            "litellm.aembedding", new=AsyncMock(return_value=mock_response)
        ) as mock_embed:
            await service.generate_embeddings(["a"])

        assert "dimensions" not in mock_embed.await_args.kwargs

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self) -> None:
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m"))
        with patch("litellm.aembedding", new=AsyncMock()) as mock_embed:
            result = await service.generate_embeddings([])

        assert result == []
        mock_embed.assert_not_awaited()


class TestEmbeddingServiceErrorBoundary:
    """Provider failures are wrapped without losing the cause."""

    @pytest.mark.asyncio
    async def test_provider_exception_wrapped_with_cause(self) -> None:
        from openai import RateLimitError

        service = LiteLLMEmbeddingService(
            LiteLLMModelSpec(model="text-embedding-3-small", api_key="k")
        )
        upstream = RateLimitError(
            "rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        with (
            patch("litellm.aembedding", new=AsyncMock(side_effect=upstream)),
            pytest.raises(EmbeddingServiceError) as excinfo,
        ):
            await service.generate_embeddings(["a"])

        assert excinfo.value.__cause__ is upstream
        message = str(excinfo.value)
        assert "text-embedding-3-small" in message
        assert "RateLimitError" in message
        assert "rate limited" in message

    @pytest.mark.asyncio
    async def test_malformed_response_wrapped(self) -> None:
        bad = type("R", (), {"data": [{"index": 0}]})()
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m"))
        with (
            patch("litellm.aembedding", new=AsyncMock(return_value=bad)),
            pytest.raises(EmbeddingServiceError) as excinfo,
        ):
            await service.generate_embeddings(["a"])

        assert isinstance(excinfo.value.__cause__, KeyError)

    @pytest.mark.asyncio
    async def test_vector_count_mismatch_raises(self) -> None:
        short = type("R", (), {"data": [{"index": 0, "embedding": [0.1]}]})()
        service = LiteLLMEmbeddingService(LiteLLMModelSpec(model="m"))
        with (
            patch("litellm.aembedding", new=AsyncMock(return_value=short)),
            pytest.raises(EmbeddingServiceError, match="1 vectors for 2 inputs"),
        ):
            await service.generate_embeddings(["a", "b"])

    def test_error_is_plain_exception_subclass(self) -> None:
        assert issubclass(EmbeddingServiceError, Exception)
        assert not issubclass(EmbeddingServiceError, ValueError)
