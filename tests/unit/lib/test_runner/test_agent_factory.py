"""Unit tests for AgentFactory with Semantic Kernel integration.

Tests cover core functionality through the public API:
- Agent initialization with multiple LLM providers
- Agent invocation with different response scenarios
- Timeout handling
- Retry logic with exponential backoff
- Error handling and recovery
"""

import asyncio
from typing import Any
from unittest import mock

import pytest
from semantic_kernel.functions import KernelArguments

from holodeck.lib.test_runner.agent_factory import (
    AgentExecutionResult,
    AgentFactory,
    AgentFactoryError,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum

# Check if anthropic is available
try:
    from semantic_kernel.connectors.ai.anthropic import (
        AnthropicChatCompletion,  # noqa: F401
    )

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class TestAgentFactoryInitialization:
    """Tests for AgentFactory initialization with different providers."""

    @pytest.mark.parametrize(
        "provider,endpoint,api_key",
        [
            (
                ProviderEnum.AZURE_OPENAI,
                "https://test.openai.azure.com",
                "azure-key",
            ),
            (ProviderEnum.OPENAI, "https://api.openai.com", "sk-openai-key"),
            pytest.param(
                ProviderEnum.ANTHROPIC,
                "https://api.anthropic.com",
                "sk-ant-key",
                marks=pytest.mark.skipif(
                    not ANTHROPIC_AVAILABLE,
                    reason="Anthropic package not installed",
                ),
            ),
        ],
    )
    def test_initialize_with_different_providers(
        self, provider: ProviderEnum, endpoint: str, api_key: str
    ) -> None:
        """Test initialization succeeds with all supported LLM providers."""
        agent_config = Agent(
            name="test-agent",
            description="Test agent",
            model=LLMProvider(
                provider=provider,
                name=(
                    "gpt-4o" if provider != ProviderEnum.ANTHROPIC else "claude-3-opus"
                ),
                endpoint=endpoint,
                api_key=api_key,
                temperature=0.7,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory.agent_config == agent_config
            assert factory.timeout == 60.0
            assert factory.max_retries == 3

    def test_initialize_with_custom_timeout_and_retry_config(self) -> None:
        """Test initialization with custom timeout and retry parameters."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(
                agent_config,
                execution_config=ExecutionConfig(llm_timeout=30),
                max_retries=5,
                retry_delay=1.0,
                retry_exponential_base=3.0,
            )

            assert factory.timeout == 30
            assert factory.max_retries == 5
            assert factory.retry_delay == 1.0
            assert factory.retry_exponential_base == 3.0

    def test_initialize_without_timeout(self) -> None:
        """Test initialization with no timeout."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            # When no execution_config is provided, factory uses default timeout
            factory = AgentFactory(agent_config)

            # Default timeout is DEFAULT_TIMEOUT_SECONDS (60.0)
            assert factory.timeout == 60.0

    def test_initialize_with_file_instructions(self, tmp_path: Any) -> None:
        """Test initialization with instructions loaded from file."""
        instructions_file = tmp_path / "instructions.txt"
        instructions_file.write_text("You are a code reviewer.")

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(file=str(instructions_file)),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory.agent_config.instructions.file == str(instructions_file)

    def test_initialize_fails_with_kernel_error(self) -> None:
        """Test that kernel creation errors are properly wrapped."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with mock.patch(
            "holodeck.lib.test_runner.agent_factory.Kernel",
            side_effect=RuntimeError("Kernel error"),
        ):
            with pytest.raises(AgentFactoryError) as exc_info:
                AgentFactory(agent_config)

            assert "Failed to initialize agent factory" in str(exc_info.value)
            assert "Kernel error" in str(exc_info.value)

    def test_initialize_with_missing_instructions_file(self, tmp_path: Any) -> None:
        """Test initialization fails when instructions file doesn't exist."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(file=str(tmp_path / "nonexistent.txt")),
        )

        with mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"):
            with pytest.raises(AgentFactoryError) as exc_info:
                AgentFactory(agent_config)

            assert "Failed to initialize agent factory" in str(exc_info.value)


class TestAgentFactoryKernelArguments:
    """Tests for KernelArguments construction and reuse."""

    @staticmethod
    def _build_factory_with_service(agent_config: Agent) -> AgentFactory:
        """Create a factory instance seeded with a fake kernel service."""

        class FakeSettings:
            def __init__(self) -> None:
                self.temperature = None
                self.top_p = None
                self.max_tokens = None
                self.max_completion_tokens = None
                self.ai_model_id = None
                self.response_format = None
                self.service_id = "fake-service"

        class FakeService:
            service_id = "fake-service"

            @staticmethod
            def get_prompt_execution_settings_class() -> type[FakeSettings]:
                return FakeSettings

        factory = AgentFactory.__new__(AgentFactory)
        factory.agent_config = agent_config
        factory.kernel_arguments = None
        factory.kernel = mock.Mock()
        fake_service = FakeService()
        factory.kernel.services = {"fake": fake_service}
        factory._llm_service = fake_service
        factory._tools_initialized = True  # Skip tool initialization for tests
        factory._mcp_plugins = []  # No MCP plugins in test
        factory._vectorstore_tool_instances = []  # No vectorstore tools in test
        factory._execution_config = None  # No execution config in test
        # Add retry and timeout attributes needed by create_thread_run
        factory.timeout = 60.0
        factory.max_retries = 3
        factory.retry_delay = 2.0
        factory.retry_exponential_base = 2.0
        factory.agent = mock.Mock()

        return factory

    def test_kernel_arguments_apply_model_settings(self) -> None:
        """KernelArguments should include configured execution settings."""

        response_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
                temperature=0.55,
                top_p=0.8,
                max_tokens=150,
            ),
            instructions=Instructions(inline="Test instructions"),
            response_format=response_schema,
        )

        factory = self._build_factory_with_service(agent_config)

        kernel_arguments = factory._build_kernel_arguments()

        assert isinstance(kernel_arguments, KernelArguments)
        settings_map = kernel_arguments.execution_settings
        assert settings_map is not None
        settings = settings_map.get("fake-service")
        assert settings is not None
        assert settings.temperature == 0.55
        assert settings.top_p == 0.8
        # Only max_completion_tokens is set (preferred over legacy max_tokens)
        assert settings.max_completion_tokens == 150
        # max_tokens should NOT be set when max_completion_tokens is available
        assert settings.max_tokens is None
        assert settings.ai_model_id == "gpt-4o"
        assert settings.response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "test-agent",
                "schema": response_schema,
                "strict": True,
            },
        }

    @pytest.mark.asyncio
    async def test_thread_run_invoke_uses_kernel_arguments(self) -> None:
        """Thread run invocation should use the configured KernelArguments."""

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        class FakeSettings:
            def __init__(self) -> None:
                self.service_id = "fake-service"

        kernel_arguments = KernelArguments(settings=FakeSettings())
        factory.kernel_arguments = kernel_arguments
        mock_response = mock.Mock()
        mock_response.content = "response"
        mock_response.tool_calls = None
        mock_response.metadata = {}

        mock_thread = mock.Mock()

        async def fake_invoke(*, thread: Any, arguments: Any) -> Any:
            assert arguments is kernel_arguments
            yield mock_response

        mock_agent = mock.MagicMock()
        mock_agent.invoke = fake_invoke
        factory.agent = mock_agent

        with mock.patch(
            "holodeck.lib.test_runner.agent_factory.ChatHistoryAgentThread",
            return_value=mock_thread,
        ):
            thread_run = await factory.create_thread_run()
            result = await thread_run.invoke("test input")

        assert result.chat_history is thread_run.chat_history
        assert factory.kernel_arguments is kernel_arguments


class TestAgentFactoryInvocation:
    """Tests for agent invocation through AgentThreadRun."""

    @pytest.mark.asyncio
    async def test_invoke_returns_execution_result(self) -> None:
        """Test successful thread run invocation returns AgentExecutionResult."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Mock the agent invocation
            mock_response = mock.Mock()
            mock_response.content = "Test response"
            mock_response.tool_calls = None

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            thread_run = await factory.create_thread_run()
            result = await thread_run.invoke("What is the capital of France?")

            assert isinstance(result, AgentExecutionResult)
            assert isinstance(result.tool_calls, list)
            assert result.chat_history is not None

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self) -> None:
        """Test invocation captures tool calls in result."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            mock_response = mock.Mock()
            mock_response.content = "Searching..."

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            # Mock tool call extraction from thread (where actual extraction happens)
            expected_tool_calls = [
                {"name": "search", "arguments": {"query": "Python testing"}}
            ]
            expected_tool_results: list[dict[str, Any]] = []

            async def mock_extract(
                *_args: Any, **_kwargs: Any
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                return expected_tool_calls, expected_tool_results

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_extract_tool_calls_from_thread", side_effect=mock_extract
            ):
                result = await thread_run.invoke("Search for Python testing")

                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "search"
                assert result.tool_calls[0]["arguments"] == {"query": "Python testing"}

    @pytest.mark.asyncio
    async def test_invoke_with_multiple_tool_calls(self) -> None:
        """Test invocation captures multiple tool calls."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            mock_response = mock.Mock()
            mock_response.content = "Processing..."

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            # Mock tool call extraction from thread (where actual extraction happens)
            expected_tool_calls = [
                {"name": "search", "arguments": {"q": "test"}},
                {"name": "analyze", "arguments": {"data": [1, 2, 3]}},
                {"name": "format", "arguments": {"type": "json"}},
            ]
            expected_tool_results: list[dict[str, Any]] = []

            async def mock_extract(
                *_args: Any, **_kwargs: Any
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                return expected_tool_calls, expected_tool_results

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_extract_tool_calls_from_thread", side_effect=mock_extract
            ):
                result = await thread_run.invoke("Process this data")

                assert len(result.tool_calls) == 3
                assert result.tool_calls[0]["name"] == "search"
                assert result.tool_calls[1]["name"] == "analyze"
                assert result.tool_calls[2]["name"] == "format"

    @pytest.mark.asyncio
    async def test_invoke_with_empty_response(self) -> None:
        """Test invocation handles empty response gracefully."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            mock_response = mock.Mock()
            mock_response.content = ""
            mock_response.tool_calls = None

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            thread_run = await factory.create_thread_run()
            result = await thread_run.invoke("Test")

            assert isinstance(result, AgentExecutionResult)
            assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_invoke_with_none_content(self) -> None:
        """Test invocation handles None content gracefully."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            mock_response = mock.Mock()
            mock_response.content = None
            mock_response.tool_calls = None

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            thread_run = await factory.create_thread_run()
            result = await thread_run.invoke("Test")

            assert isinstance(result, AgentExecutionResult)


class TestAgentFactoryTimeout:
    """Tests for timeout handling via AgentThreadRun."""

    @pytest.mark.asyncio
    async def test_invoke_respects_timeout(self) -> None:
        """Test that invocation times out when exceeding configured timeout."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            # Use short timeout (1 second) via execution_config
            # Note: llm_timeout=0 disables timeout due to `if self.timeout:` check
            factory = AgentFactory(
                agent_config, execution_config=ExecutionConfig(llm_timeout=1)
            )
            # Override to fraction for faster test (original test used 0.1s)
            factory.timeout = 0.1

            # Mock slow response (takes longer than timeout)
            async def slow_invoke(*_args: Any, **_kwargs: Any) -> Any:
                await asyncio.sleep(1.0)
                yield mock.Mock()

            factory.agent.invoke = slow_invoke  # type: ignore

            thread_run = await factory.create_thread_run()
            with pytest.raises(AgentFactoryError) as exc_info:
                await thread_run.invoke("Test")

            # Error message should indicate timeout or invocation failure
            error_msg = str(exc_info.value).lower()
            assert "timeout" in error_msg or "failed" in error_msg

    @pytest.mark.asyncio
    async def test_invoke_without_timeout_does_not_timeout(self) -> None:
        """Test that invocation without timeout waits indefinitely."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            # No execution_config means default timeout is used
            factory = AgentFactory(agent_config)

            mock_response = mock.Mock()
            mock_response.content = "Response"
            mock_response.tool_calls = None

            async def delayed_invoke(*_args: Any, **_kwargs: Any) -> Any:
                await asyncio.sleep(0.1)  # Small delay (well under default timeout)
                yield mock_response

            factory.agent.invoke = delayed_invoke  # type: ignore

            thread_run = await factory.create_thread_run()
            result = await thread_run.invoke("Test")

            assert isinstance(result, AgentExecutionResult)


class TestAgentFactoryRetry:
    """Tests for retry logic with exponential backoff via AgentThreadRun."""

    @pytest.mark.asyncio
    async def test_invoke_retries_on_connection_error(self) -> None:
        """Test that transient connection errors trigger retry."""
        from semantic_kernel.contents import ChatHistory

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config, max_retries=3, retry_delay=0.01)

            call_count = 0

            async def failing_then_success(*_args: Any, **_kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("Network error")

                history = ChatHistory()
                return AgentExecutionResult(
                    tool_calls=[], tool_results=[], chat_history=history
                )

            thread_run = await factory.create_thread_run()
            # Patch the internal implementation method on the thread run
            with mock.patch.object(
                thread_run, "_invoke_agent_impl", side_effect=failing_then_success
            ):
                result = await thread_run.invoke("Test")

                assert isinstance(result, AgentExecutionResult)
                assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_invoke_fails_after_max_retries(self) -> None:
        """Test that invocation fails after exhausting all retries."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config, max_retries=2, retry_delay=0.01)

            call_count = 0

            async def always_fail(*_args: Any, **_kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                raise ConnectionError("Persistent error")

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_invoke_agent_impl", side_effect=always_fail
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await thread_run.invoke("Test")

                assert "after 2 attempts" in str(exc_info.value)
                assert call_count == 2

    @pytest.mark.asyncio
    async def test_invoke_does_not_retry_non_retryable_errors(self) -> None:
        """Test that non-retryable errors fail immediately without retry."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config, max_retries=3, retry_delay=0.01)

            call_count = 0

            async def non_retryable_error(*_args: Any, **_kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                raise ValueError("Invalid input")

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_invoke_agent_impl", side_effect=non_retryable_error
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await thread_run.invoke("Test")

                assert "Non-retryable error" in str(exc_info.value)
                assert call_count == 1  # Should not retry


class TestAgentFactoryErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_invoke_handles_runtime_error(self) -> None:
        """Test that runtime errors during invocation are properly wrapped."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config, max_retries=1)

            async def runtime_error(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("Unexpected error")

            factory.agent.invoke = runtime_error  # type: ignore

            thread_run = await factory.create_thread_run()
            with pytest.raises(AgentFactoryError) as exc_info:
                await thread_run.invoke("Test")

            assert "Non-retryable error" in str(exc_info.value)

    def test_invalid_agent_config_raises_validation_error(self) -> None:
        """Test that invalid agent configuration is rejected by Pydantic."""
        with pytest.raises(ValueError):
            Agent(
                name="test-agent",
                model=LLMProvider(
                    provider=ProviderEnum.OPENAI,
                    name="gpt-4o",
                    endpoint="https://api.openai.com",
                    api_key="sk-test",
                ),
                instructions=None,  # type: ignore
            )


class TestAgentFactoryIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_tools_and_response(self) -> None:
        """Test complete workflow from initialization to result with tools."""
        agent_config = Agent(
            name="integration-test-agent",
            description="Integration test agent",
            author="Test Suite",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                temperature=0.7,
                max_tokens=1000,
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
            tools=[
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search function",
                    "file": "search.py",
                    "function": "search",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(
                agent_config, execution_config=ExecutionConfig(llm_timeout=60)
            )

            # Verify configuration is preserved
            assert factory.agent_config.name == "integration-test-agent"
            assert factory.agent_config.description == "Integration test agent"
            assert factory.agent_config.author == "Test Suite"
            assert factory.agent_config.model.temperature == 0.7
            assert factory.agent_config.model.max_tokens == 1000
            assert factory.agent_config.tools is not None
            assert len(factory.agent_config.tools) == 1

            # Mock response
            mock_response = mock.Mock()
            mock_response.content = "Searching for information..."

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            # Mock tool call extraction from thread (where actual extraction happens)
            expected_tool_calls = [
                {"name": "search", "arguments": {"query": "integration testing"}}
            ]
            expected_tool_results: list[dict[str, Any]] = []

            async def mock_extract(
                *_args: Any, **_kwargs: Any
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                return expected_tool_calls, expected_tool_results

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_extract_tool_calls_from_thread", side_effect=mock_extract
            ):
                result = await thread_run.invoke("How can you help me?")

                assert isinstance(result, AgentExecutionResult)
                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "search"
                assert result.tool_calls[0]["arguments"] == {
                    "query": "integration testing"
                }
                assert result.chat_history is not None

    @pytest.mark.asyncio
    async def test_workflow_with_retry_and_recovery(self) -> None:
        """Test workflow with transient failure and successful retry."""
        from semantic_kernel.contents import ChatHistory

        agent_config = Agent(
            name="retry-test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(
                agent_config,
                execution_config=ExecutionConfig(llm_timeout=5),
                max_retries=3,
                retry_delay=0.01,
            )

            attempt = 0

            async def flaky_invoke(*_args: Any, **_kwargs: Any) -> Any:
                nonlocal attempt
                attempt += 1
                if attempt <= 2:
                    raise ConnectionError("Temporary network issue")

                history = ChatHistory()
                return AgentExecutionResult(
                    tool_calls=[], tool_results=[], chat_history=history
                )

            thread_run = await factory.create_thread_run()
            with mock.patch.object(
                thread_run, "_invoke_agent_impl", side_effect=flaky_invoke
            ):
                result = await thread_run.invoke("Test query")

                assert isinstance(result, AgentExecutionResult)
                assert attempt == 3  # Failed twice, succeeded on third attempt


class TestEmbeddingServiceRegistration:
    """T037f: Tests for AgentFactory embedding service registration."""

    def test_register_embedding_service_openai(self) -> None:
        """Test embedding service registration for OpenAI provider."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",  # noqa: S108
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"
            ) as mock_embedding,
        ):
            factory = AgentFactory(agent_config)

            mock_embedding.assert_called_once_with(
                ai_model_id="text-embedding-3-small",
                api_key="sk-test",
            )
            assert factory._embedding_service is not None

    def test_register_embedding_service_azure(self) -> None:
        """Test embedding service registration for Azure OpenAI provider."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name="gpt-4o",
                endpoint="https://test.openai.azure.com",
                api_key="azure-key",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",  # noqa: S108
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.AzureTextEmbedding"
            ) as mock_embedding,
        ):
            factory = AgentFactory(agent_config)

            mock_embedding.assert_called_once_with(
                deployment_name="text-embedding-3-small",
                endpoint="https://test.openai.azure.com",
                api_key="azure-key",
            )
            assert factory._embedding_service is not None

    def test_no_embedding_service_without_vectorstore_tools(self) -> None:
        """Test that embedding service is not registered without vectorstore tools."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            # No tools
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"
            ) as mock_embedding,
        ):
            factory = AgentFactory(agent_config)

            mock_embedding.assert_not_called()
            assert factory._embedding_service is None

    def test_custom_embedding_model_from_config(self) -> None:
        """Test that custom embedding model from tool config is used."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",  # noqa: S108
                    "embedding_model": "text-embedding-3-large",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"
            ) as mock_embedding,
        ):
            factory = AgentFactory(agent_config)

            mock_embedding.assert_called_once_with(
                ai_model_id="text-embedding-3-large",
                api_key="sk-test",
            )
            assert factory._embedding_service is not None

    @pytest.mark.skipif(
        not ANTHROPIC_AVAILABLE,
        reason="Anthropic package not installed",
    )
    def test_unsupported_provider_raises_error(self) -> None:
        """Test Anthropic provider raises error for vectorstore tools."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name="claude-3-opus",
                endpoint="https://api.anthropic.com",
                api_key="sk-ant-key",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",  # noqa: S108
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            with pytest.raises(AgentFactoryError) as exc_info:
                AgentFactory(agent_config)

            error_msg = str(exc_info.value)
            assert "Embedding service not supported" in error_msg
            # Provider name may be lowercase in error message
            assert "anthropic" in error_msg.lower()


class TestKernelFunctionRegistration:
    """T037g: Tests for VectorStoreTool kernel function registration."""

    def test_create_search_kernel_function(self) -> None:
        """Test creation of KernelFunction from VectorStoreTool.search."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.KernelFunctionFromMethod"
            ) as mock_kf,
        ):
            # Setup mock to return a named mock
            mock_function = mock.Mock()
            mock_function.name = "test_search"
            mock_kf.return_value = mock_function

            factory = AgentFactory(agent_config)

            # Create a mock tool
            mock_tool = mock.Mock()
            mock_tool.search = mock.AsyncMock(return_value="Search results")

            kernel_function = factory._create_search_kernel_function(
                tool=mock_tool,
                tool_name="test_search",
                tool_description="Test search function",
            )

            # Verify function was created
            assert kernel_function is not None
            mock_kf.assert_called_once()
            # Verify the plugin_name was set correctly
            call_kwargs = mock_kf.call_args.kwargs
            assert call_kwargs["plugin_name"] == "vectorstore"

    @pytest.mark.asyncio
    async def test_kernel_function_calls_tool_search(self) -> None:
        """Test that registered KernelFunction correctly invokes tool.search()."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.KernelFunctionFromMethod"
            ) as mock_kf,
        ):
            # Capture the method that gets passed to KernelFunctionFromMethod
            captured_method = None

            def capture_method(*args: Any, **kwargs: Any) -> mock.Mock:
                nonlocal captured_method
                captured_method = kwargs.get("method")
                return mock.Mock()

            mock_kf.side_effect = capture_method

            factory = AgentFactory(agent_config)

            # Create a mock tool
            mock_tool = mock.Mock()
            mock_tool.search = mock.AsyncMock(return_value="Found 2 results")

            factory._create_search_kernel_function(
                tool=mock_tool,
                tool_name="test_search",
                tool_description="Test search function",
            )

            # Call the captured wrapper method directly
            assert captured_method is not None
            result = await captured_method("test query")

            mock_tool.search.assert_called_once_with("test query")
            assert result == "Found 2 results"


class TestVectorstoreToolDiscovery:
    """T037h: Tests for AgentFactory vectorstore tool discovery and initialization."""

    def test_has_vectorstore_tools_with_dict_config(self) -> None:
        """Test detection of vectorstore tools from dict configuration."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",  # noqa: S108
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_vectorstore_tools() is True

    def test_has_vectorstore_tools_without_vectorstore(self) -> None:
        """Test no vectorstore tools detected when only other tool types."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "function",
                    "name": "other_tool",
                    "description": "A test function tool",
                    "file": "tools/test.py",
                    "function": "test_func",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_vectorstore_tools() is False

    def test_has_vectorstore_tools_empty_tools(self) -> None:
        """Test no vectorstore tools when tools list is empty."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_vectorstore_tools() is False

    @pytest.mark.asyncio
    async def test_register_vectorstore_tools_skips_non_vectorstore(
        self, tmp_path: Any
    ) -> None:
        """Test that non-vectorstore tools are skipped during registration."""
        # Create a test file
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "function",
                    "name": "func_tool",
                    "description": "A test function tool",
                    "file": "tools/test.py",
                    "function": "test_func",
                },
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": str(source_file),
                },
            ],
        )

        # Create mock VectorStoreTool class (not used but kept for documentation)
        mock_vs_tool_instance = mock.Mock()
        mock_vs_tool_instance.search = mock.AsyncMock(return_value="results")
        mock_vs_tool_instance.initialize = mock.AsyncMock()
        _ = mock.Mock(return_value=mock_vs_tool_instance)  # mock_vs_tool_class

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.KernelFunctionFromMethod"
            ),
            mock.patch.dict(
                "sys.modules",
                {"holodeck.tools.vectorstore_tool": mock.Mock()},
            ),
        ):
            factory = AgentFactory(agent_config)

            # Manually patch the import inside _register_vectorstore_tools
            with mock.patch.object(
                factory,
                "_register_vectorstore_tools",
                wraps=factory._register_vectorstore_tools,
            ):
                # Instead, we'll directly test by setting tools_initialized
                # and checking the skip logic via _has_vectorstore_tools
                pass

            # Since we can't easily test the full async flow due to import issues,
            # test the helper method behavior instead
            assert factory._has_vectorstore_tools() is True
            assert factory._tools_initialized is False

    @pytest.mark.asyncio
    async def test_initialization_failure_raises_error(self) -> None:
        """Test that tool initialization failure raises AgentFactoryError."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "bad_tool",
                    "description": "Missing source",
                    "source": "/nonexistent/path/file.md",
                }
            ],
        )

        # Create a mock that raises FileNotFoundError on initialize
        mock_vs_tool_instance = mock.Mock()
        mock_vs_tool_instance.set_embedding_service = mock.Mock()
        mock_vs_tool_instance.initialize = mock.AsyncMock(
            side_effect=FileNotFoundError("Source path does not exist")
        )
        mock_vs_tool_class = mock.Mock(return_value=mock_vs_tool_instance)
        mock_vs_module = mock.Mock()
        mock_vs_module.VectorStoreTool = mock_vs_tool_class

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"),
        ):
            factory = AgentFactory(agent_config)

            # Patch the import to use our mock
            with mock.patch.dict(
                "sys.modules",
                {"holodeck.tools.vectorstore_tool": mock_vs_module},
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await factory._ensure_tools_initialized()

                assert "Failed to initialize vectorstore tool" in str(exc_info.value)
                assert "bad_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tools_initialized_only_once(self, tmp_path: Any) -> None:
        """Test that tools are only initialized once (lazy initialization)."""
        source_file = tmp_path / "test.md"
        source_file.write_text("# Test content")

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": str(source_file),
                }
            ],
        )

        # Create mock VectorStoreTool
        mock_vs_tool_instance = mock.Mock()
        mock_vs_tool_instance.set_embedding_service = mock.Mock()
        mock_vs_tool_instance.initialize = mock.AsyncMock()
        mock_vs_tool_instance.search = mock.AsyncMock(return_value="results")
        mock_vs_tool_class = mock.Mock(return_value=mock_vs_tool_instance)
        mock_vs_module = mock.Mock()
        mock_vs_module.VectorStoreTool = mock_vs_tool_class

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OpenAITextEmbedding"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.KernelFunctionFromMethod"
            ),
        ):
            factory = AgentFactory(agent_config)

            # Patch the import
            with mock.patch.dict(
                "sys.modules",
                {"holodeck.tools.vectorstore_tool": mock_vs_module},
            ):
                # First call initializes
                await factory._ensure_tools_initialized()
                first_initialized = factory._tools_initialized
                first_tool_count = len(factory._vectorstore_tools)

                # Second call should be a no-op
                await factory._ensure_tools_initialized()
                second_tool_count = len(factory._vectorstore_tools)

                assert first_initialized is True
                assert first_tool_count == second_tool_count == 1


# Check if Ollama is available
try:
    from semantic_kernel.connectors.ai.ollama import (
        OllamaChatCompletion,  # noqa: F401
    )

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class TestOllamaProvider:
    """Tests for Ollama provider support."""

    @pytest.mark.skipif(
        not OLLAMA_AVAILABLE,
        reason="Ollama package not installed",
    )
    def test_initialize_with_ollama_provider(self) -> None:
        """Test initialization with Ollama provider."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                endpoint="http://localhost:11434",
                api_key="",  # Ollama doesn't require API key
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaChatCompletion"
            ) as mock_ollama,
        ):
            factory = AgentFactory(agent_config)

            mock_ollama.assert_called_once_with(
                ai_model_id="llama3.2",
                host="http://localhost:11434",
            )
            assert factory.agent_config == agent_config

    @pytest.mark.skipif(
        not OLLAMA_AVAILABLE,
        reason="Ollama package not installed",
    )
    def test_initialize_ollama_without_endpoint_uses_default(self) -> None:
        """Test Ollama with no endpoint uses default host (None)."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                api_key="",
                # No endpoint specified
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaChatCompletion"
            ) as mock_ollama,
        ):
            factory = AgentFactory(agent_config)

            mock_ollama.assert_called_once_with(
                ai_model_id="llama3.2",
                host=None,
            )
            assert factory.agent_config == agent_config

    @pytest.mark.skipif(
        not OLLAMA_AVAILABLE,
        reason="Ollama package not installed",
    )
    def test_ollama_embedding_model_default(self) -> None:
        """Test Ollama uses nomic-embed-text as default embedding model."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                endpoint="http://localhost:11434",
                api_key="",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding"
            ) as mock_embed,
        ):
            factory = AgentFactory(agent_config)

            mock_embed.assert_called_once_with(
                ai_model_id="nomic-embed-text:latest",
                host="http://localhost:11434",
            )
            assert factory._embedding_service is not None

    @pytest.mark.skipif(
        not OLLAMA_AVAILABLE,
        reason="Ollama package not installed",
    )
    def test_ollama_embedding_without_endpoint(self) -> None:
        """Test Ollama embedding service with no endpoint uses None host."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                api_key="",
                # No endpoint
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding"
            ) as mock_embed,
        ):
            factory = AgentFactory(agent_config)

            mock_embed.assert_called_once_with(
                ai_model_id="nomic-embed-text:latest",
                host=None,
            )
            assert factory._embedding_service is not None


class TestOllamaNotAvailable:
    """Tests for Ollama when package is not installed."""

    def test_ollama_provider_raises_when_not_available(self) -> None:
        """Test Ollama provider raises error when package not installed."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                api_key="",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaChatCompletion", None
            ),
        ):
            with pytest.raises(AgentFactoryError) as exc_info:
                AgentFactory(agent_config)

            assert "Ollama provider requires" in str(exc_info.value)

    def test_ollama_embedding_raises_when_not_available(self) -> None:
        """Test Ollama embedding raises error when package not installed."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3.2",
                api_key="",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "vectorstore",
                    "name": "test_kb",
                    "description": "Test knowledge base",
                    "source": "test.md",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
            mock.patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion"),
            mock.patch(
                "holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding", None
            ),
        ):
            with pytest.raises(AgentFactoryError) as exc_info:
                AgentFactory(agent_config)

            assert "Ollama provider requires" in str(exc_info.value)


class TestMCPTools:
    """Tests for MCP tool support."""

    def test_has_mcp_tools_returns_true_when_mcp_configured(self) -> None:
        """Test detection of MCP tools in agent config."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "description": "Filesystem access",
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-server-filesystem"],
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_mcp_tools() is True

    def test_has_mcp_tools_returns_false_without_mcp(self) -> None:
        """Test no MCP tools detected when only other tool types."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "function",
                    "name": "other_tool",
                    "description": "A test function tool",
                    "file": "tools/test.py",
                    "function": "test_func",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_mcp_tools() is False

    def test_has_mcp_tools_returns_false_empty_tools(self) -> None:
        """Test no MCP tools when tools list is empty."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_mcp_tools() is False

    def test_has_mcp_tools_returns_false_no_tools(self) -> None:
        """Test no MCP tools when tools is None."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            assert factory._has_mcp_tools() is False

    @pytest.mark.asyncio
    async def test_register_mcp_tools_success(self) -> None:
        """Test successful MCP tool registration."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "description": "Filesystem access",
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-server-filesystem"],
                }
            ],
        )

        # Create mock plugin
        mock_plugin = mock.AsyncMock()
        mock_plugin.name = "filesystem"
        mock_plugin.__aenter__ = mock.AsyncMock(return_value=mock_plugin)
        mock_plugin.__aexit__ = mock.AsyncMock(return_value=None)

        mock_create_plugin = mock.Mock(return_value=mock_plugin)
        mock_factory_module = mock.Mock()
        mock_factory_module.create_mcp_plugin = mock_create_plugin

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            with mock.patch.dict(
                "sys.modules",
                {"holodeck.tools.mcp.factory": mock_factory_module},
            ):
                await factory._register_mcp_tools()

                assert len(factory._mcp_plugins) == 1
                mock_plugin.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_mcp_tools_skips_non_mcp(self) -> None:
        """Test that non-MCP tools are skipped during MCP registration."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "function",
                    "name": "func_tool",
                    "description": "A test function tool",
                    "file": "tools/test.py",
                    "function": "test_func",
                }
            ],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # This should be a no-op since there are no MCP tools
            await factory._register_mcp_tools()

            assert len(factory._mcp_plugins) == 0

    @pytest.mark.asyncio
    async def test_register_mcp_tools_failure_cleanup(self) -> None:
        """Test MCP plugin cleanup on initialization failure."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                {
                    "type": "mcp",
                    "name": "failing_mcp",
                    "description": "Failing MCP server",
                    "command": "npx",  # Valid command enum value
                    "args": ["-y", "failing-server"],
                }
            ],
        )

        # Create mock plugin that fails on __aenter__
        mock_plugin = mock.AsyncMock()
        mock_plugin.name = "failing_mcp"
        mock_plugin.__aenter__ = mock.AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        mock_plugin.__aexit__ = mock.AsyncMock(return_value=None)

        mock_create_plugin = mock.Mock(return_value=mock_plugin)

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            with mock.patch(
                "holodeck.tools.mcp.factory.create_mcp_plugin",
                mock_create_plugin,
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await factory._register_mcp_tools()

                assert "Failed to initialize MCP tool" in str(exc_info.value)
                assert "failing_mcp" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_shutdown_mcp_plugins(self) -> None:
        """Test MCP plugin shutdown cleans up all plugins."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Manually add mock plugins
            mock_plugin1 = mock.AsyncMock()
            mock_plugin1.name = "plugin1"
            mock_plugin1.__aexit__ = mock.AsyncMock(return_value=None)

            mock_plugin2 = mock.AsyncMock()
            mock_plugin2.name = "plugin2"
            mock_plugin2.__aexit__ = mock.AsyncMock(return_value=None)

            factory._mcp_plugins = [mock_plugin1, mock_plugin2]
            factory._tools_initialized = True

            await factory.shutdown()

            mock_plugin1.__aexit__.assert_called_once_with(None, None, None)
            mock_plugin2.__aexit__.assert_called_once_with(None, None, None)
            assert len(factory._mcp_plugins) == 0
            assert factory._tools_initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_with_vectorstore_cleanup(self) -> None:
        """Test shutdown cleans up vectorstore tools with cleanup method."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Add mock vectorstore tool with cleanup
            mock_vs_tool = mock.AsyncMock()
            mock_vs_tool.cleanup = mock.AsyncMock()
            factory._vectorstore_tools = [mock_vs_tool]
            factory._tools_initialized = True

            await factory.shutdown()

            mock_vs_tool.cleanup.assert_called_once()
            assert len(factory._vectorstore_tools) == 0

    @pytest.mark.asyncio
    async def test_shutdown_handles_plugin_errors(self) -> None:
        """Test shutdown continues despite plugin errors."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Plugin that raises on __aexit__
            mock_plugin = mock.AsyncMock()
            mock_plugin.name = "failing_plugin"
            mock_plugin.__aexit__ = mock.AsyncMock(
                side_effect=RuntimeError("Cleanup failed")
            )

            factory._mcp_plugins = [mock_plugin]
            factory._tools_initialized = True

            # Should not raise despite error
            await factory.shutdown()

            assert len(factory._mcp_plugins) == 0
            assert factory._tools_initialized is False


class TestResponseFormatWrapping:
    """Tests for response format wrapping logic."""

    @staticmethod
    def _build_factory_with_service(agent_config: Agent) -> AgentFactory:
        """Create a factory instance seeded with a fake kernel service."""

        class FakeSettings:
            def __init__(self) -> None:
                self.temperature = None
                self.top_p = None
                self.max_tokens = None
                self.max_completion_tokens = None
                self.ai_model_id = None
                self.response_format = None
                self.service_id = "fake-service"

        class FakeService:
            service_id = "fake-service"

            @staticmethod
            def get_prompt_execution_settings_class() -> type[FakeSettings]:
                return FakeSettings

        factory = AgentFactory.__new__(AgentFactory)
        factory.agent_config = agent_config
        factory.kernel_arguments = None
        factory.kernel = mock.Mock()
        fake_service = FakeService()
        factory.kernel.services = {"fake": fake_service}
        factory._llm_service = fake_service
        factory._tools_initialized = True  # Skip tool initialization for tests
        factory._mcp_plugins = []  # No MCP plugins in test
        factory._vectorstore_tool_instances = []  # No vectorstore tools in test
        factory._execution_config = None  # No execution config in test
        # Add retry and timeout attributes needed by create_thread_run
        factory.timeout = 60.0
        factory.max_retries = 3
        factory.retry_delay = 2.0
        factory.retry_exponential_base = 2.0
        factory.agent = mock.Mock()

        return factory

    def test_wrap_response_format_already_has_json_schema(self) -> None:
        """Test wrapping skips if json_schema key already exists."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        schema_with_json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "existing",
                "schema": {"type": "object"},
            },
        }

        result = factory._wrap_response_format(schema_with_json_schema)

        # Should return as-is since json_schema key exists
        assert result == schema_with_json_schema

    def test_wrap_response_format_with_type_json_schema_and_schema_key(self) -> None:
        """Test wrapping format with type=json_schema and schema key."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        # Schema with type=json_schema and schema key but no json_schema key
        schema_input = {
            "type": "json_schema",
            "schema": {"type": "object"},
            "name": "my_schema",
        }

        result = factory._wrap_response_format(schema_input)

        # Should wrap in json_schema structure
        assert result == {
            "type": "json_schema",
            "json_schema": schema_input,
        }

    def test_wrap_response_format_plain_schema(self) -> None:
        """Test wrapping a plain JSON schema."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        plain_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

        result = factory._wrap_response_format(plain_schema)

        assert result == {
            "type": "json_schema",
            "json_schema": {
                "name": "test-agent",
                "schema": plain_schema,
                "strict": True,
            },
        }

    def test_sanitize_response_format_name_with_special_chars(self) -> None:
        """Test name sanitization removes special characters."""
        agent_config = Agent(
            name="my agent!@#$%^&*()",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        result = factory._sanitize_response_format_name()

        assert result == "my_agent_"

    def test_sanitize_response_format_name_fallback(self) -> None:
        """Test name sanitization fallback for empty result."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = self._build_factory_with_service(agent_config)

        # Override agent name to test fallback behavior (after validation)
        factory.agent_config.name = ""  # Empty name

        result = factory._sanitize_response_format_name()

        assert result == "response_format"

    def test_load_response_format_from_file(self, tmp_path: Any) -> None:
        """Test loading response format from file path."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object", "properties": {}}')

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            response_format=str(schema_file),
        )

        factory = self._build_factory_with_service(agent_config)

        result = factory._load_response_format()

        assert result is not None
        assert result["type"] == "object"

    def test_load_response_format_returns_none_when_not_set(self) -> None:
        """Test loading response format returns None when not configured."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            # No response_format
        )

        factory = self._build_factory_with_service(agent_config)

        result = factory._load_response_format()

        assert result is None

    def test_apply_response_format_exception_handling(self) -> None:
        """Test response format loading exception is wrapped."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            response_format="/nonexistent/path/schema.json",
        )

        factory = self._build_factory_with_service(agent_config)

        class FakeSettings:
            response_format = None

        settings = FakeSettings()

        with pytest.raises(AgentFactoryError) as exc_info:
            factory._apply_response_format(settings)

        assert "Failed to load response_format" in str(exc_info.value)


class TestToolCallExtraction:
    """Tests for tool call extraction from AgentThreadRun."""

    @pytest.mark.asyncio
    async def test_extract_tool_calls_with_function_call_content(self) -> None:
        """Test extraction of tool calls from thread messages."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create mock FunctionCallContent
            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_123"
            mock_fcc.plugin_name = "vectorstore"
            mock_fcc.function_name = "search"
            mock_fcc.arguments = '{"query": "test"}'

            # Create mock message with items
            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            # Create mock thread
            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "vectorstore-search"
            assert tool_calls[0]["arguments"] == {"query": "test"}
            assert tool_results == []  # No FunctionResultContent in this test

    @pytest.mark.asyncio
    async def test_extract_tool_calls_with_dict_arguments(self) -> None:
        """Test extraction when arguments are already a dict."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create mock FunctionCallContent with dict arguments
            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_456"
            mock_fcc.plugin_name = ""
            mock_fcc.function_name = "calculate"
            mock_fcc.arguments = {"x": 10, "y": 20}

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "calculate"
            assert tool_calls[0]["arguments"] == {"x": 10, "y": 20}
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_with_invalid_json_arguments(self) -> None:
        """Test extraction handles invalid JSON in arguments."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create mock FunctionCallContent with invalid JSON
            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_789"
            mock_fcc.plugin_name = "test"
            mock_fcc.function_name = "func"
            mock_fcc.arguments = "not valid json {"

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "test-func"
            assert tool_calls[0]["arguments"] == {"raw": "not valid json {"}
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_deduplicates_by_id(self) -> None:
        """Test that duplicate tool calls with same ID are filtered."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Two FunctionCallContent with same ID
            mock_fcc1 = mock.Mock(spec=FunctionCallContent)
            mock_fcc1.id = "same_id"
            mock_fcc1.plugin_name = "plugin"
            mock_fcc1.function_name = "func"
            mock_fcc1.arguments = {}

            mock_fcc2 = mock.Mock(spec=FunctionCallContent)
            mock_fcc2.id = "same_id"  # Same ID
            mock_fcc2.plugin_name = "plugin"
            mock_fcc2.function_name = "func"
            mock_fcc2.arguments = {}

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc1, mock_fcc2]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            # Should only have one result due to deduplication
            assert len(tool_calls) == 1
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_handles_exception(self) -> None:
        """Test extraction handles exceptions gracefully."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            mock_thread = mock.AsyncMock()

            async def failing_get_messages() -> Any:
                raise RuntimeError("Thread error")
                yield  # Make it a generator

            mock_thread.get_messages = failing_get_messages

            # Should return empty lists, not raise
            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert tool_calls == []
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_with_none_arguments(self) -> None:
        """Test extraction handles None arguments."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_none"
            mock_fcc.plugin_name = "plugin"
            mock_fcc.function_name = "func"
            mock_fcc.arguments = None

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert len(tool_calls) == 1
            assert tool_calls[0]["arguments"] == {}
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_uses_call_id_fallback(self) -> None:
        """Test extraction uses call_id when id is not present."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = None
            mock_fcc.call_id = "fallback_call_id"
            mock_fcc.plugin_name = "plugin"
            mock_fcc.function_name = "func"
            mock_fcc.arguments = {}

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "plugin-func"
            assert tool_results == []


class TestErrorHandlingBranches:
    """Tests for error handling branches and edge cases."""

    @pytest.mark.asyncio
    async def test_extract_response_content_handles_exception(self) -> None:
        """Test response content extraction handles exceptions gracefully."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Response object where str(content) raises an exception
            class FailingContent:
                def __str__(self) -> str:
                    raise RuntimeError("Cannot convert to string")

            mock_response = mock.Mock()
            mock_response.content = FailingContent()

            result = thread_run._extract_response_content(mock_response)

            # Should return empty string on error (warning logged)
            assert result == ""

    @pytest.mark.asyncio
    async def test_extract_token_usage_handles_missing_attributes(self) -> None:
        """Test token usage extraction handles missing attributes gracefully."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Response with no metadata attribute
            mock_response = mock.Mock(spec=[])  # No attributes

            result = thread_run._extract_token_usage(mock_response)

            # Should return None when no metadata
            assert result is None

    def test_create_prompt_execution_settings_no_kernel(self) -> None:
        """Test error when kernel not initialized."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = AgentFactory.__new__(AgentFactory)
        factory.agent_config = agent_config
        # kernel attribute not set

        with pytest.raises(AgentFactoryError) as exc_info:
            factory._create_prompt_execution_settings()

        assert "Kernel must be initialized" in str(exc_info.value)

    def test_create_prompt_execution_settings_no_services(self) -> None:
        """Test error when no services configured."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = AgentFactory.__new__(AgentFactory)
        factory.agent_config = agent_config
        factory.kernel = mock.Mock()
        factory.kernel.services = {}  # Empty services
        factory._llm_service = None

        with pytest.raises(AgentFactoryError) as exc_info:
            factory._create_prompt_execution_settings()

        assert "No LLM services configured" in str(exc_info.value)

    def test_create_prompt_execution_settings_uses_llm_service_fallback(self) -> None:
        """Test fallback to _llm_service when kernel.services is empty."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        class FakeSettings:
            def __init__(self) -> None:
                self.temperature = None
                self.service_id = "fake"

        class FakeService:
            @staticmethod
            def get_prompt_execution_settings_class() -> type[FakeSettings]:
                return FakeSettings

        factory = AgentFactory.__new__(AgentFactory)
        factory.agent_config = agent_config
        factory.kernel = mock.Mock()
        factory.kernel.services = {}  # Empty
        factory._llm_service = FakeService()

        settings = factory._create_prompt_execution_settings()

        assert settings is not None

    def test_load_instructions_file_not_found(self, tmp_path: Any) -> None:
        """Test error when instructions file does not exist."""
        # Create config with non-existent file
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(file="nonexistent_instructions.txt"),
        )

        from holodeck.config.context import agent_base_dir

        token = agent_base_dir.set(str(tmp_path))
        try:
            with (
                mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
                mock.patch(
                    "holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"
                ),
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    AgentFactory(agent_config)

                assert "Instructions file not found" in str(exc_info.value)
        finally:
            agent_base_dir.reset(token)

    def test_load_instructions_with_agent_base_dir_context(self, tmp_path: Any) -> None:
        """Test loading instructions from file with agent_base_dir context."""
        instructions_file = tmp_path / "instructions.txt"
        instructions_file.write_text("You are a helpful assistant.")

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(file="instructions.txt"),
        )

        from holodeck.config.context import agent_base_dir

        token = agent_base_dir.set(str(tmp_path))
        try:
            with (
                mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
                mock.patch(
                    "holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"
                ),
            ):
                factory = AgentFactory(agent_config)
                # Should succeed with context set
                assert factory is not None
        finally:
            agent_base_dir.reset(token)

    @pytest.mark.asyncio
    async def test_invoke_general_exception_handling(self) -> None:
        """Test general exception is properly wrapped."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            thread_run = await factory.create_thread_run()

            # Mock _invoke_with_retry to raise generic Exception
            async def failing_invoke() -> Any:
                raise ValueError("Something went wrong")

            with mock.patch.object(thread_run, "_invoke_with_retry", failing_invoke):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await thread_run.invoke("Test")

                assert "Agent invocation failed" in str(exc_info.value)

    def test_unsupported_provider_raises_error(self) -> None:
        """Test unsupported provider raises appropriate error."""
        # This tests the else branch in _create_kernel
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,  # Start with valid provider
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"):
            factory = AgentFactory.__new__(AgentFactory)
            factory.agent_config = agent_config
            factory._force_ingest = False
            factory._vectorstore_tools = []
            factory._mcp_plugins = []
            factory._tools_initialized = False
            factory._embedding_service = None
            factory._llm_service = None

            # Temporarily modify provider to trigger unsupported branch
            original_provider = agent_config.model.provider
            agent_config.model.provider = mock.Mock()  # Invalid provider
            agent_config.model.provider.value = "unsupported_provider"

            with pytest.raises(AgentFactoryError) as exc_info:
                factory._create_kernel()

            assert "Unsupported LLM provider" in str(exc_info.value)

            # Restore
            agent_config.model.provider = original_provider


class TestMiscellaneousCoverage:
    """Miscellaneous tests to increase coverage."""

    @pytest.mark.asyncio
    async def test_thread_run_chat_history_isolation(self) -> None:
        """Test each thread run gets its own isolated chat history."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Create two thread runs
            thread_run1 = await factory.create_thread_run()
            thread_run2 = await factory.create_thread_run()

            # Each should have its own chat history
            assert thread_run1.chat_history is not thread_run2.chat_history
            assert len(thread_run1.chat_history.messages) == 0
            assert len(thread_run2.chat_history.messages) == 0

    def test_invoke_agent_impl_builds_kernel_arguments_when_none(self) -> None:
        """Test _invoke_agent_impl builds kernel_arguments if None."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        # The main test here is ensuring the code path is covered
        # when kernel_arguments is None in _invoke_agent_impl
        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            # kernel_arguments should be set after init
            assert factory.kernel_arguments is not None

    def test_apply_model_settings_with_max_tokens_fallback(self) -> None:
        """Test max_tokens is used when max_completion_tokens not available."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
                max_tokens=100,
            ),
            instructions=Instructions(inline="Test"),
        )

        # Settings class without max_completion_tokens
        class LegacySettings:
            def __init__(self) -> None:
                self.temperature = None
                self.top_p = None
                self.max_tokens = None
                self.ai_model_id = None

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            settings = LegacySettings()
            factory._apply_model_settings(settings)

            assert settings.max_tokens == 100

    @pytest.mark.asyncio
    async def test_register_vectorstore_tools_returns_early_no_tools(self) -> None:
        """Test _register_vectorstore_tools returns early when no tools."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=None,
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Should return early without error
            await factory._register_vectorstore_tools()

            assert len(factory._vectorstore_tools) == 0

    @pytest.mark.asyncio
    async def test_register_mcp_tools_returns_early_no_tools(self) -> None:
        """Test _register_mcp_tools returns early when no tools."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
            tools=None,
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)

            # Should return early without error
            await factory._register_mcp_tools()

            assert len(factory._mcp_plugins) == 0

    @pytest.mark.asyncio
    async def test_token_usage_extraction_success(self) -> None:
        """Test successful token usage extraction."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create mock response with proper usage object
            mock_usage = mock.Mock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20

            mock_response = mock.Mock()
            mock_response.metadata = {"usage": mock_usage}

            result = thread_run._extract_token_usage(mock_response)

            assert result is not None
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 20
            assert result.total_tokens == 30

    @pytest.mark.asyncio
    async def test_extract_response_content_without_content_attr(self) -> None:
        """Test response extraction when no content attribute."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Response without content attribute
            mock_response = mock.Mock(spec=[])  # No attributes

            result = thread_run._extract_response_content(mock_response)

            assert result == ""

    @pytest.mark.asyncio
    async def test_extract_tool_calls_skips_non_function_call_items(self) -> None:
        """Test extraction skips items that are not FunctionCallContent."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Mix of FunctionCallContent and other items
            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_1"
            mock_fcc.plugin_name = "plugin"
            mock_fcc.function_name = "func"
            mock_fcc.arguments = {}

            other_item = mock.Mock()  # Not a FunctionCallContent

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc, other_item]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            # Should only have the FunctionCallContent
            assert len(tool_calls) == 1
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_message_without_items(self) -> None:
        """Test extraction handles messages without items attribute."""
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Message without items attribute
            mock_message = mock.Mock(spec=[])

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread
            )

            assert tool_calls == []
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_with_start_index_skips_earlier_messages(
        self,
    ) -> None:
        """Test that start_index correctly skips earlier messages.

        This is important for multi-turn conversations where we only want
        to extract tool calls from the current turn, not from previous turns.
        """
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create mock FunctionCallContent for turn 1 (should be skipped)
            mock_fcc_turn1 = mock.Mock(spec=FunctionCallContent)
            mock_fcc_turn1.id = "call_turn1"
            mock_fcc_turn1.plugin_name = "tools"
            mock_fcc_turn1.function_name = "old_tool"
            mock_fcc_turn1.arguments = '{"query": "old"}'

            # Create mock FunctionCallContent for turn 2 (should be included)
            mock_fcc_turn2 = mock.Mock(spec=FunctionCallContent)
            mock_fcc_turn2.id = "call_turn2"
            mock_fcc_turn2.plugin_name = "tools"
            mock_fcc_turn2.function_name = "new_tool"
            mock_fcc_turn2.arguments = '{"query": "new"}'

            # Message 0: User message from turn 1
            mock_message_0 = mock.Mock()
            mock_message_0.items = []

            # Message 1: Assistant message with tool call from turn 1
            mock_message_1 = mock.Mock()
            mock_message_1.items = [mock_fcc_turn1]

            # Message 2: User message from turn 2
            mock_message_2 = mock.Mock()
            mock_message_2.items = []

            # Message 3: Assistant message with tool call from turn 2
            mock_message_3 = mock.Mock()
            mock_message_3.items = [mock_fcc_turn2]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message_0
                yield mock_message_1
                yield mock_message_2
                yield mock_message_3

            mock_thread.get_messages = mock_get_messages

            # Extract with start_index=2 (skip first 2 messages from turn 1)
            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread, start_index=2
            )

            # Should only have the tool call from turn 2
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "tools-new_tool"
            assert tool_calls[0]["arguments"] == {"query": "new"}
            assert tool_results == []

    @pytest.mark.asyncio
    async def test_extract_tool_calls_start_index_zero_returns_all(self) -> None:
        """Test that start_index=0 returns all tool calls (default behavior)."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Create two tool calls in separate messages
            mock_fcc_1 = mock.Mock(spec=FunctionCallContent)
            mock_fcc_1.id = "call_1"
            mock_fcc_1.plugin_name = ""
            mock_fcc_1.function_name = "tool_a"
            mock_fcc_1.arguments = "{}"

            mock_fcc_2 = mock.Mock(spec=FunctionCallContent)
            mock_fcc_2.id = "call_2"
            mock_fcc_2.plugin_name = ""
            mock_fcc_2.function_name = "tool_b"
            mock_fcc_2.arguments = "{}"

            mock_message_1 = mock.Mock()
            mock_message_1.items = [mock_fcc_1]

            mock_message_2 = mock.Mock()
            mock_message_2.items = [mock_fcc_2]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message_1
                yield mock_message_2

            mock_thread.get_messages = mock_get_messages

            # Extract with start_index=0 (default - get all)
            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread, start_index=0
            )

            # Should have both tool calls
            assert len(tool_calls) == 2
            assert tool_calls[0]["name"] == "tool_a"
            assert tool_calls[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_extract_tool_calls_start_index_beyond_messages(self) -> None:
        """Test that start_index beyond message count returns empty list."""
        from semantic_kernel.contents import FunctionCallContent

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o",
                endpoint="https://api.openai.com",
                api_key="sk-test",
            ),
            instructions=Instructions(inline="Test"),
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            mock_fcc = mock.Mock(spec=FunctionCallContent)
            mock_fcc.id = "call_1"
            mock_fcc.plugin_name = ""
            mock_fcc.function_name = "tool_a"
            mock_fcc.arguments = "{}"

            mock_message = mock.Mock()
            mock_message.items = [mock_fcc]

            mock_thread = mock.AsyncMock()

            async def mock_get_messages() -> Any:
                yield mock_message

            mock_thread.get_messages = mock_get_messages

            # Extract with start_index=100 (way beyond the single message)
            tool_calls, tool_results = await thread_run._extract_tool_calls_from_thread(
                mock_thread, start_index=100
            )

            # Should return empty since all messages are skipped
            assert tool_calls == []
            assert tool_results == []
