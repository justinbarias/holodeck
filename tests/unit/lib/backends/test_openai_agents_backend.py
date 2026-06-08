"""Unit tests for holodeck.lib.backends.openai_agents_backend.

All SDK interactions are mocked — no network calls and no real credentials are
required. The `openai-agents` package is installed (dev extra) so the lazy
imports resolve, but each SDK symbol is patched per-test to observe behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.backends.base import BackendInitError
from holodeck.lib.backends.openai_agents_backend import _build_model
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    provider: ProviderEnum = ProviderEnum.OPENAI,
    name: str = "gpt-4o-mini",
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
) -> Agent:
    """Build a minimal Agent for backend tests."""
    return Agent(
        name="test-agent",
        model=LLMProvider(
            provider=provider,
            name=name,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
        ),
        instructions=Instructions(inline="Be helpful."),
    )


# ---------------------------------------------------------------------------
# Task 2 — _build_model + credential pre-flight
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildModelOpenAI:
    """OpenAI provider returns the model-name string with a valid key."""

    def test_returns_model_name_with_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        with patch("agents.set_default_openai_key") as set_key:
            result = _build_model(agent)
        assert result == "gpt-4o-mini"
        set_key.assert_called_once_with("sk-test")

    def test_config_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o", api_key="sk-cfg")
        with patch("agents.set_default_openai_key") as set_key:
            result = _build_model(agent)
        assert result == "gpt-4o"
        set_key.assert_called_once_with("sk-cfg")

    def test_missing_key_raises_naming_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        with pytest.raises(BackendInitError, match="OPENAI_API_KEY"):
            _build_model(agent)


@pytest.mark.unit
class TestBuildModelAzure:
    """Azure provider wraps AsyncAzureOpenAI as an OpenAIChatCompletionsModel."""

    def test_builds_chat_completions_model_and_disables_tracing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI,
            "my-deployment",
            endpoint="https://x.openai.azure.com",
        )

        sentinel_client = MagicMock(name="async_azure_client")
        sentinel_model = MagicMock(name="chat_completions_model")
        with (
            patch(
                "openai.AsyncAzureOpenAI", return_value=sentinel_client
            ) as azure_ctor,
            patch(
                "agents.OpenAIChatCompletionsModel", return_value=sentinel_model
            ) as model_ctor,
            patch("agents.set_tracing_disabled") as disable_tracing,
        ):
            # endpoint comes from env since config endpoint is None
            result = _build_model(agent)

        assert result is sentinel_model
        disable_tracing.assert_called_once_with(True)
        azure_ctor.assert_called_once()
        _, kwargs = azure_ctor.call_args
        assert kwargs["api_key"] == "az-key"
        assert kwargs["azure_endpoint"] == "https://x.openai.azure.com"
        assert kwargs["api_version"]  # a GA default is applied
        # Deployment name is passed as the model.
        _, model_kwargs = model_ctor.call_args
        assert model_kwargs["model"] == "my-deployment"
        assert model_kwargs["openai_client"] is sentinel_client

    def test_config_api_version_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI,
            "dep",
            endpoint="https://x.openai.azure.com",
            api_version="2099-01-01",
        )
        with (
            patch("openai.AsyncAzureOpenAI") as azure_ctor,
            patch("agents.OpenAIChatCompletionsModel"),
            patch("agents.set_tracing_disabled"),
        ):
            _build_model(agent)
        _, kwargs = azure_ctor.call_args
        assert kwargs["api_version"] == "2099-01-01"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI, "dep", endpoint="https://x.openai.azure.com"
        )
        with pytest.raises(BackendInitError, match="AZURE_OPENAI_API_KEY"):
            _build_model(agent)

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Azure config requires an endpoint at model-validation time, so to reach
        # the backend's endpoint pre-flight we clear both env and the resolved
        # value by constructing the agent with an endpoint then dropping it.
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI, "dep", endpoint="https://placeholder"
        )
        agent.model.endpoint = None  # simulate unresolved endpoint
        with pytest.raises(BackendInitError, match="AZURE_OPENAI_ENDPOINT"):
            _build_model(agent)


@pytest.mark.unit
class TestBuildModelUnsupported:
    """Providers outside openai/azure are rejected by this backend."""

    @pytest.mark.parametrize("provider", [ProviderEnum.ANTHROPIC, ProviderEnum.OLLAMA])
    def test_unsupported_provider_raises(self, provider: ProviderEnum) -> None:
        agent = _make_agent(provider, "some-model")
        with pytest.raises(BackendInitError, match="does not support provider"):
            _build_model(agent)
