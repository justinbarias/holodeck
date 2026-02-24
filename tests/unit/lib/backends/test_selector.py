"""Unit tests for holodeck.lib.backends.selector (Phase 4A + Phase 8B TDD).

Class:
    TestBackendSelector: 7 tests for BackendSelector routing logic.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.backends.base import BackendInitError
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
# TestBackendSelector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackendSelector:
    """Tests for BackendSelector.select() provider routing."""

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
    @patch("holodeck.lib.backends.selector.ClaudeBackend")
    async def test_anthropic_returns_claude_backend(self, mock_claude_cls: Any) -> None:
        """T023: provider=anthropic routes to ClaudeBackend."""
        mock_backend = AsyncMock()
        mock_claude_cls.return_value = mock_backend

        agent = _make_agent(ProviderEnum.ANTHROPIC)
        backend = await BackendSelector.select(agent)

        mock_claude_cls.assert_called_once()
        assert backend is mock_backend
        mock_backend.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.selector.ClaudeBackend")
    async def test_anthropic_passes_tool_instances_and_mode(
        self, mock_claude_cls: Any
    ) -> None:
        """T024: selector passes tool_instances/mode to ClaudeBackend."""
        mock_backend = AsyncMock()
        mock_claude_cls.return_value = mock_backend

        agent = _make_agent(ProviderEnum.ANTHROPIC)
        mock_tool = MagicMock()

        await BackendSelector.select(
            agent,
            tool_instances={"kb": mock_tool},
            mode="chat",
            allow_side_effects=True,
        )

        mock_claude_cls.assert_called_once_with(
            agent=agent,
            tool_instances={"kb": mock_tool},
            mode="chat",
            allow_side_effects=True,
        )

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises_backend_init_error(self) -> None:
        """T025: Unsupported provider raises BackendInitError."""
        # Use a known provider but manipulate to simulate unsupported
        agent = _make_agent(ProviderEnum.OPENAI)
        # Monkey-patch to simulate an unsupported provider
        agent.model.provider = "unsupported_provider"  # type: ignore[assignment]

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
