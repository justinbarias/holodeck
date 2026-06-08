"""Unit tests for holodeck.lib.backends.openai_agents_backend.

All SDK interactions are mocked — no network calls and no real credentials are
required. The `openai-agents` package is installed (dev extra) so the lazy
imports resolve, but each SDK symbol is patched per-test to observe behaviour.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendInitError,
    BackendSessionError,
)
from holodeck.lib.backends.openai_agents_backend import (
    OpenAIAgentsBackend,
    OpenAIAgentsSession,
    _build_model,
    _to_execution_result,
)
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


# ---------------------------------------------------------------------------
# Task 4 — _to_execution_result mapping
# ---------------------------------------------------------------------------


def _fake_run_result(
    *,
    final_output: str | None = "Final answer",
    tool_call: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    raw_response_count: int = 1,
) -> MagicMock:
    """Build a stand-in RunResult with the attributes the mapper reads."""
    from agents.items import ToolCallItem, ToolCallOutputItem
    from openai.types.responses import ResponseFunctionToolCall

    new_items: list = []
    if tool_call:
        raw_call = ResponseFunctionToolCall(
            call_id="call-1",
            name="add",
            arguments='{"a": 1, "b": 2}',
            type="function_call",
        )
        new_items.append(ToolCallItem(agent=MagicMock(), raw_item=raw_call))
        new_items.append(
            ToolCallOutputItem(
                agent=MagicMock(),
                raw_item={
                    "call_id": "call-1",
                    "output": "3",
                    "type": "function_call_output",
                },
                output="3",
            )
        )

    result = MagicMock()
    result.final_output = final_output
    result.new_items = new_items
    result.raw_responses = [MagicMock() for _ in range(raw_response_count)]
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens
    result.context_wrapper = MagicMock(usage=usage)
    return result


@pytest.mark.unit
class TestToExecutionResult:
    """Mapping an SDK run result onto the provider-agnostic ExecutionResult."""

    def test_plain_response(self) -> None:
        result = _to_execution_result(
            _fake_run_result(
                final_output="Hello there", input_tokens=10, output_tokens=5
            )
        )
        assert result.response == "Hello there"
        assert result.tool_calls == []
        assert result.tool_results == []
        assert result.token_usage.prompt_tokens == 10
        assert result.token_usage.completion_tokens == 5
        assert result.token_usage.total_tokens == 15  # derived when not given
        assert result.num_turns == 1
        assert result.thinking == ""
        assert result.is_error is False

    def test_none_final_output_becomes_empty_string(self) -> None:
        result = _to_execution_result(_fake_run_result(final_output=None))
        assert result.response == ""

    def test_tool_call_extraction(self) -> None:
        result = _to_execution_result(
            _fake_run_result(
                tool_call=True, input_tokens=20, output_tokens=8, total_tokens=28
            )
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[0]["arguments"] == {"a": 1, "b": 2}
        assert result.tool_calls[0]["call_id"] == "call-1"
        assert len(result.tool_results) == 1
        assert result.tool_results[0]["name"] == "add"  # matched via call_id
        assert result.tool_results[0]["result"] == "3"
        assert result.token_usage.total_tokens == 28

    def test_num_turns_from_raw_responses(self) -> None:
        result = _to_execution_result(_fake_run_result(raw_response_count=3))
        assert result.num_turns == 3


# ---------------------------------------------------------------------------
# Task 3 — OpenAIAgentsBackend + OpenAIAgentsSession
# ---------------------------------------------------------------------------


def _initialized_backend(monkeypatch: pytest.MonkeyPatch) -> OpenAIAgentsBackend:
    """Build a backend whose SDK agent is a stand-in (no real Agent build)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
    backend = OpenAIAgentsBackend(agent)
    backend._sdk_agent = MagicMock(name="sdk_agent")
    return backend


@pytest.mark.unit
class TestBackendProtocol:
    """Protocol conformance and lifecycle."""

    def test_backend_is_agent_backend(self) -> None:
        agent = _make_agent()
        assert isinstance(OpenAIAgentsBackend(agent), AgentBackend)

    def test_session_is_agent_session(self) -> None:
        assert isinstance(OpenAIAgentsSession(MagicMock(), MagicMock()), AgentSession)

    @pytest.mark.asyncio
    async def test_invoke_before_initialize_raises(self) -> None:
        agent = _make_agent()
        backend = OpenAIAgentsBackend(agent)
        with pytest.raises(BackendInitError, match="initialize"):
            await backend.invoke_once("hi")


@pytest.mark.unit
class TestBackendInitialize:
    """initialize() builds the SDK agent from config."""

    @pytest.mark.asyncio
    async def test_initialize_builds_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        backend = OpenAIAgentsBackend(agent)
        sdk_agent = MagicMock(name="sdk_agent")
        with (
            patch("agents.Agent", return_value=sdk_agent) as agent_ctor,
            patch("agents.ModelSettings") as settings_ctor,
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ) as build_tools,
        ):
            await backend.initialize()
        assert backend._sdk_agent is sdk_agent
        agent_ctor.assert_called_once()
        build_tools.assert_called_once()
        # model_settings carries the agent's sampling params
        _, settings_kwargs = settings_ctor.call_args
        assert settings_kwargs["temperature"] == 0.3
        assert settings_kwargs["max_tokens"] == 1000


@pytest.mark.unit
class TestBackendInvokeOnce:
    """Single-turn invoke_once via Runner.run."""

    @pytest.mark.asyncio
    async def test_invoke_once_returns_execution_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)
        fake_result = _fake_run_result(final_output="42")
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=fake_result)
            exec_result = await backend.invoke_once("what is 6*7?")
        assert exec_result.response == "42"
        assert exec_result.is_error is False
        runner.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_once_wraps_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(side_effect=RuntimeError("boom"))
            with pytest.raises(BackendSessionError, match="boom"):
                await backend.invoke_once("hi")


@pytest.mark.unit
class TestSession:
    """Multi-turn session behaviour."""

    @pytest.mark.asyncio
    async def test_create_session_returns_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)
        with patch("agents.SQLiteSession", return_value=MagicMock()) as sqlite_ctor:
            session = await backend.create_session()
        assert isinstance(session, OpenAIAgentsSession)
        sqlite_ctor.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_returns_execution_result(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock())
        fake_result = _fake_run_result(final_output="turn answer")
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=fake_result)
            result = await session.send("hello")
        assert result.response == "turn answer"
        # session= is threaded through so the SDK persists history
        _, kwargs = runner.run.call_args
        assert "session" in kwargs

    @pytest.mark.asyncio
    async def test_send_returns_error_result_on_failure(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock())
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(side_effect=RuntimeError("kaput"))
            result = await session.send("hello")
        assert result.is_error is True
        assert "kaput" in (result.error_reason or "")

    @pytest.mark.asyncio
    async def test_send_streaming_fallback_yields_response(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock())
        fake_result = _fake_run_result(final_output="streamed")
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=fake_result)
            chunks = [c async for c in session.send_streaming("hi")]
        assert "".join(chunks) == "streamed"
