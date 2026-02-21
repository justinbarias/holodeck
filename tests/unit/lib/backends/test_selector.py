"""Unit tests for holodeck.lib.backends.selector (Phase 4A TDD).

All tests fail with ImportError until Phase 4B creates selector.py.

Class:
    TestBackendSelector (T027): 5 tests for BackendSelector routing logic.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from holodeck.lib.backends.base import BackendInitError

# ImportError expected until Phase 4B creates this module.
from holodeck.lib.backends.selector import BackendSelector
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_agent(provider: ProviderEnum, **model_overrides: Any) -> Agent:
    """Create a minimal Agent for the given provider.

    Azure OpenAI requires an endpoint; all other providers use only name/provider.
    """
    model_kwargs: dict[str, Any] = {"provider": provider, "name": "gpt-4o"}
    if provider == ProviderEnum.AZURE_OPENAI:
        model_kwargs["endpoint"] = "https://example.openai.azure.com/"
    model_kwargs.update(model_overrides)
    return Agent(
        name="test-agent",
        model=LLMProvider(**model_kwargs),
        instructions=Instructions(inline="Be helpful."),
    )


# ---------------------------------------------------------------------------
# T027 — BackendSelector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackendSelector:
    """Tests for BackendSelector.select() provider routing (T027)."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.selector.SKBackend")
    async def test_openai_returns_sk_backend(self, mock_sk_cls: Any) -> None:
        """provider=openai routes to SKBackend; the class is instantiated once."""
        mock_backend = AsyncMock()
        mock_sk_cls.return_value = mock_backend

        backend = await BackendSelector.select(_make_agent(ProviderEnum.OPENAI))

        mock_sk_cls.assert_called_once()
        assert backend is mock_backend

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.selector.SKBackend")
    async def test_azure_openai_returns_sk_backend(self, mock_sk_cls: Any) -> None:
        """provider=azure_openai routes to SKBackend."""
        mock_backend = AsyncMock()
        mock_sk_cls.return_value = mock_backend

        backend = await BackendSelector.select(_make_agent(ProviderEnum.AZURE_OPENAI))

        mock_sk_cls.assert_called_once()
        assert backend is mock_backend

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.selector.SKBackend")
    async def test_ollama_returns_sk_backend(self, mock_sk_cls: Any) -> None:
        """provider=ollama routes to SKBackend."""
        mock_backend = AsyncMock()
        mock_sk_cls.return_value = mock_backend

        backend = await BackendSelector.select(_make_agent(ProviderEnum.OLLAMA))

        mock_sk_cls.assert_called_once()
        assert backend is mock_backend

    @pytest.mark.asyncio
    async def test_anthropic_raises_backend_init_error(self) -> None:
        """provider=anthropic raises BackendInitError (no SK backend support)."""
        agent = _make_agent(ProviderEnum.ANTHROPIC)

        with pytest.raises(BackendInitError):
            await BackendSelector.select(agent)

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.selector.SKBackend")
    async def test_initialize_awaited_on_returned_backend(
        self, mock_sk_cls: Any
    ) -> None:
        """select() awaits initialize() on the backend before returning."""
        mock_backend = AsyncMock()
        mock_sk_cls.return_value = mock_backend

        await BackendSelector.select(_make_agent(ProviderEnum.OPENAI))

        mock_backend.initialize.assert_awaited_once()
