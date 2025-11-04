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

from holodeck.lib.test_runner.agent_factory import (
    AgentExecutionResult,
    AgentFactory,
    AgentFactoryError,
)
from holodeck.models.agent import Agent, Instructions
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
                timeout=30.0,
                max_retries=5,
                retry_delay=1.0,
                retry_exponential_base=3.0,
            )

            assert factory.timeout == 30.0
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
            factory = AgentFactory(agent_config, timeout=None)

            assert factory.timeout is None

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


class TestAgentFactoryInvocation:
    """Tests for agent invocation through the public API."""

    @pytest.mark.asyncio
    async def test_invoke_returns_execution_result(self) -> None:
        """Test successful agent invocation returns AgentExecutionResult."""
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

            result = await factory.invoke("What is the capital of France?")

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

            # Mock response with tool calls
            mock_tool_call = mock.Mock()
            mock_tool_call.name = "search"
            mock_tool_call.arguments = {"query": "Python testing"}

            mock_response = mock.Mock()
            mock_response.content = "Searching..."
            mock_response.tool_calls = [mock_tool_call]

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            result = await factory.invoke("Search for Python testing")

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

            # Mock response with multiple tool calls
            mock_tool_1 = mock.Mock()
            mock_tool_1.name = "search"
            mock_tool_1.arguments = {"q": "test"}

            mock_tool_2 = mock.Mock()
            mock_tool_2.name = "analyze"
            mock_tool_2.arguments = {"data": [1, 2, 3]}

            mock_tool_3 = mock.Mock()
            mock_tool_3.name = "format"
            mock_tool_3.arguments = {"type": "json"}

            mock_tools = [mock_tool_1, mock_tool_2, mock_tool_3]

            mock_response = mock.Mock()
            mock_response.content = "Processing..."
            mock_response.tool_calls = mock_tools

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            result = await factory.invoke("Process this data")

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

            result = await factory.invoke("Test")

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

            result = await factory.invoke("Test")

            assert isinstance(result, AgentExecutionResult)


class TestAgentFactoryTimeout:
    """Tests for timeout handling."""

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
            factory = AgentFactory(agent_config, timeout=0.1)

            # Mock slow response
            async def slow_invoke(*_args: Any, **_kwargs: Any) -> Any:
                await asyncio.sleep(1.0)
                yield mock.Mock()

            factory.agent.invoke = slow_invoke  # type: ignore

            with pytest.raises(AgentFactoryError) as exc_info:
                await factory.invoke("Test")

            assert "timeout" in str(exc_info.value).lower()

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
            factory = AgentFactory(agent_config, timeout=None)

            mock_response = mock.Mock()
            mock_response.content = "Response"
            mock_response.tool_calls = None

            async def delayed_invoke(*_args: Any, **_kwargs: Any) -> Any:
                await asyncio.sleep(0.1)  # Small delay
                yield mock_response

            factory.agent.invoke = delayed_invoke  # type: ignore

            result = await factory.invoke("Test")

            assert isinstance(result, AgentExecutionResult)


class TestAgentFactoryRetry:
    """Tests for retry logic with exponential backoff."""

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
                return AgentExecutionResult(tool_calls=[], chat_history=history)

            # Patch the internal implementation method
            with mock.patch.object(
                factory, "_invoke_agent_impl", side_effect=failing_then_success
            ):
                result = await factory.invoke("Test")

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

            with mock.patch.object(
                factory, "_invoke_agent_impl", side_effect=always_fail
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await factory.invoke("Test")

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

            with mock.patch.object(
                factory, "_invoke_agent_impl", side_effect=non_retryable_error
            ):
                with pytest.raises(AgentFactoryError) as exc_info:
                    await factory.invoke("Test")

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

            with pytest.raises(AgentFactoryError) as exc_info:
                await factory.invoke("Test")

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
            tools=[{"type": "function", "name": "search"}],
        )

        with (
            mock.patch("holodeck.lib.test_runner.agent_factory.Kernel"),
            mock.patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent"),
        ):
            factory = AgentFactory(agent_config, timeout=60.0)

            # Verify configuration is preserved
            assert factory.agent_config.name == "integration-test-agent"
            assert factory.agent_config.description == "Integration test agent"
            assert factory.agent_config.author == "Test Suite"
            assert factory.agent_config.model.temperature == 0.7
            assert factory.agent_config.model.max_tokens == 1000
            assert factory.agent_config.tools is not None
            assert len(factory.agent_config.tools) == 1

            # Mock response with tool call
            mock_tool = mock.Mock()
            mock_tool.name = "search"
            mock_tool.arguments = {"query": "integration testing"}

            mock_response = mock.Mock()
            mock_response.content = "Searching for information..."
            mock_response.tool_calls = [mock_tool]

            async def mock_invoke(*_args: Any, **_kwargs: Any) -> Any:
                yield mock_response

            factory.agent.invoke = mock_invoke  # type: ignore

            result = await factory.invoke("How can you help me?")

            assert isinstance(result, AgentExecutionResult)
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0]["name"] == "search"
            assert result.tool_calls[0]["arguments"] == {"query": "integration testing"}
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
                timeout=5.0,
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
                return AgentExecutionResult(tool_calls=[], chat_history=history)

            with mock.patch.object(
                factory, "_invoke_agent_impl", side_effect=flaky_invoke
            ):
                result = await factory.invoke("Test query")

                assert isinstance(result, AgentExecutionResult)
                assert attempt == 3  # Failed twice, succeeded on third attempt
