"""Unit tests for holodeck.lib.backends.openai_agents_backend.

All SDK interactions are mocked — no network calls and no real credentials are
required. The `openai-agents` package is installed (dev extra) so the lazy
imports resolve, but each SDK symbol is patched per-test to observe behaviour.
"""

from types import SimpleNamespace
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
    _azure_v1_base_url,
    _build_model,
    _build_model_settings,
    _build_reasoning,
    _is_reasoning_model,
    _parse_tool_arguments,
    _to_execution_result,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.openai_config import OpenAIConfig

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
    """Azure provider wraps AsyncOpenAI (v1 surface) as an OpenAIResponsesModel."""

    def test_builds_responses_model_and_disables_tracing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI,
            "my-deployment",
            endpoint="https://x.openai.azure.com",
        )

        sentinel_client = MagicMock(name="async_openai_client")
        sentinel_model = MagicMock(name="responses_model")
        with (
            patch("openai.AsyncOpenAI", return_value=sentinel_client) as client_ctor,
            patch(
                "agents.OpenAIResponsesModel", return_value=sentinel_model
            ) as model_ctor,
            patch("agents.set_tracing_disabled") as disable_tracing,
        ):
            result = _build_model(agent)

        assert result is sentinel_model
        disable_tracing.assert_called_once_with(True)
        client_ctor.assert_called_once()
        _, kwargs = client_ctor.call_args
        assert kwargs["api_key"] == "az-key"
        # A bare resource endpoint is normalized to the /openai/v1 surface.
        assert kwargs["base_url"] == "https://x.openai.azure.com/openai/v1"
        # No api-version is forced when the config does not pin one.
        assert "default_query" not in kwargs
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
            patch("openai.AsyncOpenAI") as client_ctor,
            patch("agents.OpenAIResponsesModel"),
            patch("agents.set_tracing_disabled"),
        ):
            _build_model(agent)
        _, kwargs = client_ctor.call_args
        assert kwargs["default_query"] == {"api-version": "2099-01-01"}

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


@pytest.mark.unit
class TestBuildModelFallback:
    """``openai.fallback_model`` wraps the primary in a fallback Model (FR-033)."""

    def test_openai_no_fallback_returns_plain_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without fallback_model the OpenAI path returns the bare name."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o")
        agent.openai = OpenAIConfig()
        with patch("agents.set_default_openai_key"):
            result = _build_model(agent)
        assert result == "gpt-4o"

    def test_openai_fallback_wraps_via_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fallback_model the OpenAI path returns a wrapping Model built
        from two provider-resolved models sharing one client."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o")
        agent.openai = OpenAIConfig(fallback_model="gpt-4o-mini")

        primary_model = MagicMock(name="primary_model")
        fallback_model = MagicMock(name="fallback_model")
        provider = MagicMock(name="provider")
        provider.get_model.side_effect = [primary_model, fallback_model]
        wrapped = MagicMock(name="wrapped_model")

        with (
            patch("agents.set_default_openai_key"),
            patch(
                "agents.models.openai_provider.OpenAIProvider", return_value=provider
            ),
            patch(
                "holodeck.lib.backends.openai_agents_fallback.build_fallback_model",
                return_value=wrapped,
            ) as build,
        ):
            result = _build_model(agent)

        assert result is wrapped
        assert provider.get_model.call_args_list[0].args == ("gpt-4o",)
        assert provider.get_model.call_args_list[1].args == ("gpt-4o-mini",)
        build.assert_called_once_with(primary_model, fallback_model)

    def test_azure_fallback_reuses_same_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For Azure, the fallback deployment is built on the SAME client."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI,
            "primary-dep",
            endpoint="https://x.openai.azure.com",
        )
        agent.openai = OpenAIConfig(fallback_model="fallback-dep")

        sentinel_client = MagicMock(name="async_openai_client")
        primary_model = MagicMock(name="primary_responses_model")
        fallback_responses_model = MagicMock(name="fallback_responses_model")
        wrapped = MagicMock(name="wrapped_model")

        with (
            patch("openai.AsyncOpenAI", return_value=sentinel_client),
            patch(
                "agents.OpenAIResponsesModel",
                side_effect=[primary_model, fallback_responses_model],
            ) as model_ctor,
            patch("agents.set_tracing_disabled"),
            patch(
                "holodeck.lib.backends.openai_agents_fallback.build_fallback_model",
                return_value=wrapped,
            ) as build,
        ):
            result = _build_model(agent)

        assert result is wrapped
        # Both responses models share the one Azure client.
        assert model_ctor.call_args_list[0].kwargs["openai_client"] is sentinel_client
        assert model_ctor.call_args_list[1].kwargs["openai_client"] is sentinel_client
        assert model_ctor.call_args_list[0].kwargs["model"] == "primary-dep"
        assert model_ctor.call_args_list[1].kwargs["model"] == "fallback-dep"
        build.assert_called_once_with(primary_model, fallback_responses_model)


@pytest.mark.unit
class TestAzureV1BaseUrl:
    """Endpoint normalization to the OpenAI-compatible /openai/v1 surface."""

    def test_bare_endpoint_gets_v1_appended(self) -> None:
        assert (
            _azure_v1_base_url("https://x.openai.azure.com")
            == "https://x.openai.azure.com/openai/v1"
        )

    def test_trailing_slash_stripped_before_append(self) -> None:
        assert (
            _azure_v1_base_url("https://x.services.ai.azure.com/")
            == "https://x.services.ai.azure.com/openai/v1"
        )

    def test_v1_endpoint_left_as_is(self) -> None:
        assert (
            _azure_v1_base_url("https://x.services.ai.azure.com/openai/v1")
            == "https://x.services.ai.azure.com/openai/v1"
        )

    def test_v1_endpoint_with_trailing_slash_normalized(self) -> None:
        assert (
            _azure_v1_base_url("https://x.openai.azure.com/openai/v1/")
            == "https://x.openai.azure.com/openai/v1"
        )


@pytest.mark.unit
class TestIsReasoningModel:
    """Name-based heuristic for OpenAI reasoning models."""

    @pytest.mark.parametrize(
        "name",
        ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.4", "GPT-5"],
    )
    def test_reasoning_models(self, name: str) -> None:
        assert _is_reasoning_model(name) is True

    @pytest.mark.parametrize(
        "name",
        ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo", "my-deployment"],
    )
    def test_non_reasoning_models(self, name: str) -> None:
        assert _is_reasoning_model(name) is False


@pytest.mark.unit
class TestBuildModelSettings:
    """ModelSettings construction honoring reasoning-model constraints."""

    def test_non_reasoning_forwards_sampling_params(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o-mini",
            temperature=0.3,
            top_p=0.9,
            max_tokens=1000,
        )
        with patch("agents.ModelSettings") as settings_ctor:
            _build_model_settings(cfg)
        _, kwargs = settings_ctor.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_tokens"] == 1000

    def test_reasoning_omits_sampling_params(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.AZURE_OPENAI,
            name="gpt-5.4",
            endpoint="https://x.openai.azure.com",
            temperature=0.0,
            top_p=0.5,
            max_tokens=200,
        )
        with patch("agents.ModelSettings") as settings_ctor:
            _build_model_settings(cfg)
        _, kwargs = settings_ctor.call_args
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert kwargs["max_tokens"] == 200

    def test_effort_high_sets_reasoning_high(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-5",
            max_tokens=1000,
        )
        settings = _build_model_settings(cfg, OpenAIConfig(effort="high"))
        assert settings.reasoning is not None
        assert settings.reasoning.effort == "high"

    def test_effort_max_maps_to_xhigh(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-5",
            max_tokens=1000,
        )
        settings = _build_model_settings(cfg, OpenAIConfig(effort="max"))
        assert settings.reasoning is not None
        assert settings.reasoning.effort == "xhigh"

    def test_no_effort_leaves_reasoning_unset(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-5",
            max_tokens=1000,
        )
        settings = _build_model_settings(cfg, OpenAIConfig())
        assert settings.reasoning is None

    def test_no_openai_config_leaves_reasoning_unset(self) -> None:
        cfg = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-5",
            max_tokens=1000,
        )
        settings = _build_model_settings(cfg, None)
        assert settings.reasoning is None


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
    reasoning_summaries: list[str] | None = None,
) -> MagicMock:
    """Build a stand-in RunResult with the attributes the mapper reads."""
    from agents.items import ReasoningItem, ToolCallItem, ToolCallOutputItem
    from openai.types.responses import ResponseFunctionToolCall
    from openai.types.responses.response_reasoning_item import (
        ResponseReasoningItem,
        Summary,
    )

    new_items: list = []
    if reasoning_summaries is not None:
        raw_reasoning = ResponseReasoningItem(
            id="reason-1",
            type="reasoning",
            summary=[
                Summary(text=text, type="summary_text") for text in reasoning_summaries
            ],
        )
        new_items.append(ReasoningItem(agent=MagicMock(), raw_item=raw_reasoning))
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

    def test_no_usage_yields_zero_tokens(self) -> None:
        result = MagicMock()
        result.final_output = "no usage here"
        result.new_items = []
        result.raw_responses = [MagicMock()]
        result.context_wrapper = MagicMock(usage=None)
        mapped = _to_execution_result(result)
        assert mapped.token_usage.prompt_tokens == 0
        assert mapped.token_usage.completion_tokens == 0
        assert mapped.token_usage.total_tokens == 0


@pytest.mark.unit
class TestThinkingExtraction:
    """Populating ExecutionResult.thinking from ReasoningItem summaries (FR-004)."""

    def test_no_reasoning_items_leaves_thinking_empty(self) -> None:
        result = _to_execution_result(_fake_run_result())
        assert result.thinking == ""

    def test_reasoning_summaries_joined(self) -> None:
        result = _to_execution_result(
            _fake_run_result(reasoning_summaries=["First thought", "Second thought"])
        )
        assert result.thinking == "First thought\n\nSecond thought"

    def test_empty_summary_entries_ignored(self) -> None:
        result = _to_execution_result(
            _fake_run_result(reasoning_summaries=["", "Kept"])
        )
        assert result.thinking == "Kept"


@pytest.mark.unit
class TestStructuredOutput:
    """Populating ExecutionResult.structured_output from final_output (FR-004)."""

    def test_no_schema_leaves_structured_output_none(self) -> None:
        result = _to_execution_result(
            _fake_run_result(final_output='{"answer": "42"}'), structured=False
        )
        assert result.structured_output is None

    def test_dict_final_output_parsed(self) -> None:
        result = _to_execution_result(
            _fake_run_result(final_output=None), structured=True
        )
        # final_output None -> structured stays None.
        assert result.structured_output is None

    def test_dict_final_output(self) -> None:
        fake = _fake_run_result()
        fake.final_output = {"answer": "42"}
        result = _to_execution_result(fake, structured=True)
        assert result.structured_output == {"answer": "42"}

    def test_json_string_final_output_parsed(self) -> None:
        result = _to_execution_result(
            _fake_run_result(final_output='{"answer": "42"}'), structured=True
        )
        assert result.structured_output == {"answer": "42"}

    def test_pydantic_model_final_output_dumped(self) -> None:
        fake = _fake_run_result()
        fake.final_output = SimpleNamespace(model_dump=lambda: {"k": "v"})
        result = _to_execution_result(fake, structured=True)
        assert result.structured_output == {"k": "v"}


@pytest.mark.unit
class TestBuildReasoningSummary:
    """_build_reasoning requests summary='auto' so thinking has content."""

    def test_effort_sets_summary_auto(self) -> None:
        reasoning = _build_reasoning(OpenAIConfig(effort="high"))
        assert reasoning is not None
        assert reasoning.effort == "high"
        assert reasoning.summary == "auto"

    def test_no_effort_returns_none(self) -> None:
        assert _build_reasoning(OpenAIConfig()) is None
        assert _build_reasoning(None) is None


@pytest.mark.unit
class TestParseToolArguments:
    """Coercion of the SDK's raw tool-call ``arguments`` into a dict."""

    def test_dict_passthrough(self) -> None:
        assert _parse_tool_arguments({"a": 1}) == {"a": 1}

    def test_valid_json_string(self) -> None:
        assert _parse_tool_arguments('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_malformed_json_wrapped_as_raw(self) -> None:
        assert _parse_tool_arguments("not json") == {"raw": "not json"}

    def test_non_dict_json_wrapped_as_raw(self) -> None:
        assert _parse_tool_arguments("[1, 2, 3]") == {"raw": "[1, 2, 3]"}

    def test_other_type_yields_empty_dict(self) -> None:
        assert _parse_tool_arguments(None) == {}
        assert _parse_tool_arguments(42) == {}


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

    @pytest.mark.asyncio
    async def test_initialize_builds_azure_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        agent = _make_agent(
            ProviderEnum.AZURE_OPENAI,
            "my-deployment",
            endpoint="https://x.openai.azure.com",
        )
        backend = OpenAIAgentsBackend(agent)
        sdk_agent = MagicMock(name="sdk_agent")
        sentinel_model = MagicMock(name="responses_model")
        with (
            patch("agents.Agent", return_value=sdk_agent) as agent_ctor,
            patch("agents.ModelSettings"),
            patch("openai.AsyncOpenAI"),
            patch("agents.OpenAIResponsesModel", return_value=sentinel_model),
            patch("agents.set_tracing_disabled"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
        ):
            await backend.initialize()
        assert backend._sdk_agent is sdk_agent
        # The Azure client is wrapped as the SDK model argument.
        _, agent_kwargs = agent_ctor.call_args
        assert agent_kwargs["model"] is sentinel_model

    @pytest.mark.asyncio
    async def test_initialize_no_response_format_leaves_output_type_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        backend = OpenAIAgentsBackend(agent)
        with (
            patch("agents.Agent", return_value=MagicMock()) as agent_ctor,
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
        ):
            await backend.initialize()
        _, agent_kwargs = agent_ctor.call_args
        assert agent_kwargs["output_type"] is None
        assert backend._has_structured_output is False

    @pytest.mark.asyncio
    async def test_initialize_wires_response_format_to_output_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        agent.response_format = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }
        backend = OpenAIAgentsBackend(agent)
        with (
            patch("agents.Agent", return_value=MagicMock()) as agent_ctor,
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
        ):
            await backend.initialize()
        from agents.agent_output import AgentOutputSchemaBase

        _, agent_kwargs = agent_ctor.call_args
        assert isinstance(agent_kwargs["output_type"], AgentOutputSchemaBase)
        assert agent_kwargs["output_type"].json_schema() == agent.response_format
        assert backend._has_structured_output is True

    @pytest.mark.asyncio
    async def test_structured_output_surfaced_in_invoke_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")
        backend = OpenAIAgentsBackend(agent)
        backend._sdk_agent = MagicMock(name="sdk_agent")
        backend._has_structured_output = True
        fake = _fake_run_result()
        fake.final_output = {"answer": "42"}
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=fake)
            exec_result = await backend.invoke_once("q")
        assert exec_result.structured_output == {"answer": "42"}


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
    async def test_send_surfaces_structured_output(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock(), structured_output=True)
        fake = _fake_run_result()
        fake.final_output = {"answer": "42"}
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=fake)
            result = await session.send("hello")
        assert result.structured_output == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_send_returns_error_result_on_failure(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock())
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(side_effect=RuntimeError("kaput"))
            result = await session.send("hello")
        assert result.is_error is True
        assert "kaput" in (result.error_reason or "")

    @pytest.mark.asyncio
    async def test_send_streaming_yields_text_deltas(self) -> None:
        from openai.types.responses import ResponseTextDeltaEvent

        session = OpenAIAgentsSession(MagicMock(), MagicMock())

        def _delta_event(delta: str) -> MagicMock:
            data = MagicMock(spec=ResponseTextDeltaEvent)
            data.delta = delta
            return MagicMock(type="raw_response_event", data=data)

        # A non-text raw event and a non-raw lifecycle event must be skipped.
        other_raw = MagicMock(type="raw_response_event", data=MagicMock())
        lifecycle = MagicMock(type="run_item_stream_event", data=MagicMock())

        events = [
            _delta_event("Hel"),
            other_raw,
            _delta_event("lo"),
            lifecycle,
            _delta_event(" world"),
        ]

        async def _stream_events():  # type: ignore[no-untyped-def]
            for ev in events:
                yield ev

        streamed = MagicMock()
        streamed.stream_events = _stream_events
        with patch("agents.Runner") as runner:
            runner.run_streamed = MagicMock(return_value=streamed)
            chunks = [c async for c in session.send_streaming("hi")]
        assert "".join(chunks) == "Hello world"
        assert chunks == ["Hel", "lo", " world"]
        runner.run_streamed.assert_called_once()
        # session= is threaded so the SDK persists streamed-turn history
        _, kwargs = runner.run_streamed.call_args
        assert "session" in kwargs

    @pytest.mark.asyncio
    async def test_close_calls_sync_close(self) -> None:
        sqlite_session = MagicMock()
        sqlite_session.close = MagicMock()
        session = OpenAIAgentsSession(MagicMock(), sqlite_session)
        await session.close()
        sqlite_session.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_close_awaits_async_close(self) -> None:
        sqlite_session = MagicMock()
        sqlite_session.close = AsyncMock()
        session = OpenAIAgentsSession(MagicMock(), sqlite_session)
        await session.close()
        sqlite_session.close.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_close_method(self) -> None:
        sqlite_session = SimpleNamespace()  # no .close attribute
        session = OpenAIAgentsSession(MagicMock(), sqlite_session)
        await session.close()  # must not raise


# ---------------------------------------------------------------------------
# Task A2 — max_turns wiring + side-effect-free preflight
# Task A3 — RunConfig plumbing (workflow_name / group_id / trace sensitivity)
# ---------------------------------------------------------------------------


def _make_agent_with_openai(
    *, max_turns: int | None = None, capture_content: bool | None = None
) -> Agent:
    """Agent with an openai: block and optional observability tracing config."""
    kwargs: dict = {
        "name": "test-agent",
        "model": LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        "instructions": Instructions(inline="Be helpful."),
    }
    if max_turns is not None:
        kwargs["openai"] = {"max_turns": max_turns}
    if capture_content is not None:
        kwargs["observability"] = {"traces": {"capture_content": capture_content}}
    return Agent(**kwargs)


@pytest.mark.unit
class TestMaxTurnsWiring:
    """openai.max_turns reaches Runner.run; default 20 when unset."""

    @pytest.mark.asyncio
    async def test_invoke_once_default_max_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)  # no openai block
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await backend.invoke_once("hi")
        assert runner.run.call_args.kwargs["max_turns"] == 20

    @pytest.mark.asyncio
    async def test_invoke_once_configured_max_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_make_agent_with_openai(max_turns=7))
        backend._sdk_agent = MagicMock(name="sdk_agent")
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await backend.invoke_once("hi")
        assert runner.run.call_args.kwargs["max_turns"] == 7


@pytest.mark.unit
class TestRunConfigBuild:
    """A3 — _build_run_config maps identity + trace sensitivity."""

    def test_invoke_run_config_has_workflow_name_no_group(self) -> None:
        from holodeck.lib.backends.openai_agents_backend import _build_run_config

        rc = _build_run_config(_make_agent_with_openai())
        assert rc.workflow_name == "test-agent"
        assert rc.group_id is None

    def test_capture_content_false_by_default(self) -> None:
        from holodeck.lib.backends.openai_agents_backend import _build_run_config

        rc = _build_run_config(_make_agent_with_openai())
        assert rc.trace_include_sensitive_data is False

    def test_capture_content_true_includes_sensitive(self) -> None:
        from holodeck.lib.backends.openai_agents_backend import _build_run_config

        rc = _build_run_config(_make_agent_with_openai(capture_content=True))
        assert rc.trace_include_sensitive_data is True

    def test_session_run_config_carries_group_id(self) -> None:
        from holodeck.lib.backends.openai_agents_backend import _build_run_config

        rc = _build_run_config(_make_agent_with_openai(), group_id="sess-1")
        assert rc.group_id == "sess-1"

    @pytest.mark.asyncio
    async def test_invoke_once_passes_run_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await backend.invoke_once("hi")
        rc = runner.run.call_args.kwargs["run_config"]
        assert rc.workflow_name == "test-agent"

    @pytest.mark.asyncio
    async def test_session_send_passes_group_id_run_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)
        with patch("agents.SQLiteSession", return_value=MagicMock()):
            session = await backend.create_session()
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await session.send("hi")
        kwargs = runner.run.call_args.kwargs
        assert kwargs["max_turns"] == 20
        assert kwargs["run_config"].group_id is not None


# ---------------------------------------------------------------------------
# Task B1 — RAG tool-instance initialization + teardown cleanup
# ---------------------------------------------------------------------------


def _agent_with_vectorstore() -> Agent:
    from holodeck.models.tool import VectorstoreTool

    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
        tools=[VectorstoreTool(name="kb", description="knowledge", source=".")],
    )


@pytest.mark.unit
class TestRagInstanceInit:
    """initialize() builds RAG instances and threads them into build_sdk_tools."""

    @pytest.mark.asyncio
    async def test_initialize_threads_instances_and_validates_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_agent_with_vectorstore())
        fake_instances = {"kb": MagicMock(name="vs_instance")}
        with (
            patch("agents.Agent", return_value=MagicMock(name="sdk_agent")),
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new=AsyncMock(return_value=fake_instances),
            ) as init_tools,
            patch(
                "holodeck.lib.backends.validators.validate_embedding_provider"
            ) as validate_embeddings,
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ) as build_tools,
        ):
            await backend.initialize()
        init_tools.assert_awaited_once()
        validate_embeddings.assert_called_once()
        assert build_tools.call_args.kwargs["tool_instances"] == fake_instances

    @pytest.mark.asyncio
    async def test_initialize_skips_rag_init_without_rag_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        agent = _make_agent(ProviderEnum.OPENAI, "gpt-4o-mini")  # no tools
        backend = OpenAIAgentsBackend(agent)
        with (
            patch("agents.Agent", return_value=MagicMock()),
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.tool_initializer.initialize_tools",
                new=AsyncMock(return_value={}),
            ) as init_tools,
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
        ):
            await backend.initialize()
        init_tools.assert_not_awaited()


@pytest.mark.unit
class TestTeardownCleansOwnedTools:
    @pytest.mark.asyncio
    async def test_teardown_awaits_cleanup_and_clears(self) -> None:
        agent = _make_agent()
        backend = OpenAIAgentsBackend(agent)
        inst = MagicMock(name="owned_tool")
        inst.cleanup = AsyncMock()
        backend._owned_tools = [inst]
        backend._tool_instances = {"kb": inst}
        await backend.teardown()
        inst.cleanup.assert_awaited_once()
        assert backend._owned_tools == []
        assert backend._tool_instances == {}


# ---------------------------------------------------------------------------
# Task C1 — MCP server wiring + connect/cleanup lifecycle
# ---------------------------------------------------------------------------


def _agent_with_mcp() -> Agent:
    from holodeck.models.tool import MCPTool

    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
        tools=[
            MCPTool(
                name="files",
                description="filesystem server",
                transport="stdio",
                command="npx",
                args=["-y", "server"],
            )
        ],
    )


@pytest.mark.unit
class TestMcpServerWiring:
    """initialize() builds, connects, and threads MCP servers into the Agent."""

    @pytest.mark.asyncio
    async def test_mcp_servers_connected_and_passed_to_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_agent_with_mcp())
        server = MagicMock(name="mcp_server")
        server.connect = AsyncMock()
        server.cleanup = AsyncMock()
        with (
            patch(
                "agents.Agent", return_value=MagicMock(name="sdk_agent")
            ) as agent_ctor,
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
            patch(
                "holodeck.lib.backends.openai_agents_mcp.build_mcp_servers",
                return_value=[server],
            ) as build_mcp,
        ):
            await backend.initialize()
        build_mcp.assert_called_once()
        server.connect.assert_awaited_once()
        # The connected server reaches Agent(mcp_servers=...)
        assert agent_ctor.call_args.kwargs["mcp_servers"] == [server]
        assert backend._mcp_servers == [server]

    @pytest.mark.asyncio
    async def test_connect_failure_cleans_up_and_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_agent_with_mcp())
        good = MagicMock(name="good_server")
        good.connect = AsyncMock()
        good.cleanup = AsyncMock()
        bad = MagicMock(name="bad_server")
        bad.connect = AsyncMock(side_effect=RuntimeError("boom"))
        with (
            patch("agents.Agent", return_value=MagicMock()),
            patch("agents.ModelSettings"),
            patch("agents.set_default_openai_key"),
            patch(
                "holodeck.lib.backends.openai_agents_tool_adapters.build_sdk_tools",
                return_value=[],
            ),
            patch(
                "holodeck.lib.backends.openai_agents_mcp.build_mcp_servers",
                return_value=[good, bad],
            ),
            pytest.raises(BackendInitError, match="MCP server"),
        ):
            await backend.initialize()
        # The already-connected server is cleaned up; the failing one is not.
        good.connect.assert_awaited_once()
        good.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_teardown_cleans_up_mcp_servers(self) -> None:
        backend = OpenAIAgentsBackend(_make_agent())
        server = MagicMock(name="mcp_server")
        server.cleanup = AsyncMock()
        backend._mcp_servers = [server]
        await backend.teardown()
        server.cleanup.assert_awaited_once()
        assert backend._mcp_servers == []


# ---------------------------------------------------------------------------
# Task F3 — max_budget_usd cost-accountant RunHooks wiring
# ---------------------------------------------------------------------------


def _make_agent_with_budget(budget: float | None) -> Agent:
    """Agent with an openai: block carrying max_budget_usd (or omitting it)."""
    openai_block: dict = {}
    if budget is not None:
        openai_block["max_budget_usd"] = budget
    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
        openai=openai_block or None,
    )


@pytest.mark.unit
class TestBudgetHooksWiring:
    """openai.max_budget_usd attaches cost hooks; unset → no hooks (zero cost)."""

    @pytest.mark.asyncio
    async def test_no_budget_passes_no_hooks_invoke(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _initialized_backend(monkeypatch)  # no openai block
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await backend.invoke_once("hi")
        assert runner.run.call_args.kwargs["hooks"] is None

    @pytest.mark.asyncio
    async def test_budget_attaches_hooks_invoke(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_make_agent_with_budget(0.05))
        backend._sdk_agent = MagicMock(name="sdk_agent")
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await backend.invoke_once("hi")
        assert runner.run.call_args.kwargs["hooks"] is not None

    @pytest.mark.asyncio
    async def test_invoke_budget_exceeded_returns_error_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from holodeck.lib.backends.base import BackendBudgetExceededError

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = OpenAIAgentsBackend(_make_agent_with_budget(0.01))
        backend._sdk_agent = MagicMock(name="sdk_agent")
        exc = BackendBudgetExceededError(
            partial_response="half an answer",
            accumulated_cost_usd=0.5,
            budget_usd=0.01,
        )
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(side_effect=exc)
            result = await backend.invoke_once("expensive query")
        assert result.is_error is True
        assert result.response == "half an answer"
        assert "max_budget_usd exceeded" in (result.error_reason or "")
        assert "0.5" in (result.error_reason or "")

    @pytest.mark.asyncio
    async def test_session_budget_exceeded_returns_error_result(self) -> None:
        from holodeck.lib.backends.base import BackendBudgetExceededError

        session = OpenAIAgentsSession(MagicMock(), MagicMock(), budget_usd=0.01)
        exc = BackendBudgetExceededError(
            partial_response="partial turn",
            accumulated_cost_usd=0.2,
            budget_usd=0.01,
        )
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(side_effect=exc)
            result = await session.send("hi")
        assert result.is_error is True
        assert result.response == "partial turn"
        assert "max_budget_usd exceeded" in (result.error_reason or "")

    @pytest.mark.asyncio
    async def test_session_shares_one_accountant_across_turns(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock(), budget_usd=1.0)
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await session.send("turn one")
            first = session._accountant
            await session.send("turn two")
            second = session._accountant
        # The same accountant instance persists so the budget covers the session.
        assert first is not None
        assert first is second

    @pytest.mark.asyncio
    async def test_session_no_budget_passes_no_hooks(self) -> None:
        session = OpenAIAgentsSession(MagicMock(), MagicMock())  # no budget
        with patch("agents.Runner") as runner:
            runner.run = AsyncMock(return_value=_fake_run_result(final_output="ok"))
            await session.send("hi")
        assert runner.run.call_args.kwargs["hooks"] is None
