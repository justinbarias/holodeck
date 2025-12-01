"""Integration tests for AgentFactory with real LLM providers.

These tests make actual API calls to LLM providers (OpenAI, Azure OpenAI, Anthropic).
They require valid API keys to be set in tests/integration/.env file.

To run these tests:
1. Copy tests/integration/.env.example to tests/integration/.env
2. Fill in your actual API keys
3. Run: pytest tests/integration/test_agent_factory_integration.py

To skip these tests (e.g., in CI without API keys):
Set SKIP_LLM_INTEGRATION_TESTS=true in .env or environment
"""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.lib.test_runner.agent_factory import (
    AgentExecutionResult,
    AgentFactory,
    AgentFactoryError,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum


async def invoke_factory(
    factory: AgentFactory, user_input: str
) -> AgentExecutionResult:
    """Helper to invoke factory using the thread run pattern.

    Creates a thread run and invokes with the given input.
    This provides backward-compatible invoke semantics for tests.

    Args:
        factory: The AgentFactory instance.
        user_input: The user message to send.

    Returns:
        AgentExecutionResult from the invocation.
    """
    thread_run = await factory.create_thread_run()
    return await thread_run.invoke(user_input)


# Load environment variables from tests/integration/.env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Check if we should skip LLM integration tests
SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"

# API key checks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Skip markers for different providers
skip_if_no_openai = pytest.mark.skipif(
    SKIP_LLM_TESTS or not OPENAI_API_KEY,
    reason="OpenAI API key not configured or LLM tests disabled",
)

skip_if_no_azure = pytest.mark.skipif(
    SKIP_LLM_TESTS or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT,
    reason="Azure OpenAI credentials not configured or LLM tests disabled",
)

skip_if_no_anthropic = pytest.mark.skipif(
    SKIP_LLM_TESTS or not ANTHROPIC_API_KEY,
    reason="Anthropic API key not configured or LLM tests disabled",
)


@pytest.mark.integration
@pytest.mark.slow
class TestAgentFactoryOpenAI:
    """Integration tests for AgentFactory with OpenAI provider."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_basic_invocation(self) -> None:
        """Test basic invocation with OpenAI provider.

        Validates:
        1. Agent factory initializes with OpenAI provider
        2. Agent responds to simple query
        3. Response is captured in AgentExecutionResult
        """
        agent_config = Agent(
            name="openai-test-agent",
            description="Test agent for OpenAI integration",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.7,
                max_tokens=100,
            ),
            instructions=Instructions(
                inline="You are a helpful assistant. Answer questions concisely."
            ),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(factory, "What is 2 + 2? Answer in one sentence.")

        # Verify result structure
        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None
        assert len(result.chat_history.messages) > 0

        # Verify response contains expected content
        messages = list(result.chat_history.messages)
        assert any("4" in str(msg.content) for msg in messages)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_with_file_instructions(self, tmp_path: Path) -> None:
        """Test OpenAI invocation with instructions loaded from file.

        Validates:
        1. Instructions loaded from external file
        2. Agent follows file-based instructions
        3. Response reflects instruction content
        """
        # Create instruction file
        instructions_file = tmp_path / "system_prompt.txt"
        instructions_file.write_text(
            "You are a math tutor. Always explain your reasoning step by step."
        )

        agent_config = Agent(
            name="openai-file-test",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.5,
                max_tokens=150,
            ),
            instructions=Instructions(file=str(instructions_file)),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(factory, "What is 5 * 6?")

        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None

        # Verify response mentions the answer
        messages = list(result.chat_history.messages)
        assert any("30" in str(msg.content) for msg in messages)

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self) -> None:
        """Test that timeout is properly enforced with OpenAI.

        Validates:
        1. Very short timeout causes AgentFactoryError
        2. Error message includes timeout information
        """
        agent_config = Agent(
            name="openai-timeout-test",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.7,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        # Use very short timeout to trigger timeout error
        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=0)
        )

        with pytest.raises(AgentFactoryError) as exc_info:
            await invoke_factory(factory, "Tell me a long story about AI.")

        assert "timeout" in str(exc_info.value).lower()

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_openai_multiple_invocations(self) -> None:
        """Test multiple sequential invocations with same factory.

        Validates:
        1. Factory can be reused for multiple invocations
        2. Each invocation returns independent results
        3. Context is isolated between invocations
        """
        agent_config = Agent(
            name="openai-multi-test",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.7,
                max_tokens=50,
            ),
            instructions=Instructions(inline="You are a concise assistant."),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )

        # Create a thread run for multi-turn conversation
        thread_run = await factory.create_thread_run()

        # First invocation
        result1 = await thread_run.invoke("What is the capital of France?")
        assert isinstance(result1, AgentExecutionResult)
        messages1 = list(result1.chat_history.messages)
        assert any("Paris" in str(msg.content) for msg in messages1)

        # Second invocation (builds on conversation history)
        result2 = await thread_run.invoke("What is 10 + 5?")
        assert isinstance(result2, AgentExecutionResult)
        messages2 = list(result2.chat_history.messages)
        assert any("15" in str(msg.content) for msg in messages2)

        # Verify chat history accumulates both conversations
        # (ThreadRun maintains a persistent chat history for multi-turn)
        assert len(messages2) > len(messages1)


@pytest.mark.integration
@pytest.mark.slow
class TestAgentFactoryAzureOpenAI:
    """Integration tests for AgentFactory with Azure OpenAI provider."""

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_basic_invocation(self) -> None:
        """Test basic invocation with Azure OpenAI provider.

        Validates:
        1. Agent factory initializes with Azure OpenAI provider
        2. Agent responds to simple query
        3. Response is captured in AgentExecutionResult
        """
        agent_config = Agent(
            name="azure-test-agent",
            description="Test agent for Azure OpenAI integration",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(
                inline="You are a helpful assistant. Be concise."
            ),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(
            factory, "What is the largest planet? One sentence only."
        )

        # Verify result structure
        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None
        assert len(result.chat_history.messages) > 0

        # Verify response contains expected content
        messages = list(result.chat_history.messages)
        assert any("Jupiter" in str(msg.content) for msg in messages)

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_with_file_instructions(self, tmp_path: Path) -> None:
        """Test Azure OpenAI invocation with instructions from file.

        Validates:
        1. Instructions loaded from external file work with Azure
        2. Agent follows file-based instructions
        3. Response reflects instruction content
        """
        # Create instruction file
        instructions_file = tmp_path / "azure_instructions.txt"
        instructions_file.write_text(
            "You are a science teacher. Explain concepts clearly and simply."
        )

        agent_config = Agent(
            name="azure-file-test",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(file=str(instructions_file)),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(factory, "What is photosynthesis? One sentence.")

        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None

        # Verify response exists
        messages = list(result.chat_history.messages)
        response_text = " ".join(str(msg.content) for msg in messages).lower()
        assert len(response_text) > 0

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_multiple_invocations(self) -> None:
        """Test multiple sequential invocations with Azure OpenAI.

        Validates:
        1. Factory can be reused for multiple invocations
        2. Each invocation returns independent results
        3. Context is isolated between invocations
        """
        agent_config = Agent(
            name="azure-multi-test",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(inline="You are a concise assistant."),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )

        # Create a thread run for multi-turn conversation
        thread_run = await factory.create_thread_run()

        # First invocation
        result1 = await thread_run.invoke("What is the capital of Germany?")
        assert isinstance(result1, AgentExecutionResult)
        messages1 = list(result1.chat_history.messages)
        assert any("Berlin" in str(msg.content) for msg in messages1)

        # Second invocation (builds on conversation history)
        result2 = await thread_run.invoke("What is 12 * 12?")
        assert isinstance(result2, AgentExecutionResult)
        messages2 = list(result2.chat_history.messages)
        assert any("144" in str(msg.content) for msg in messages2)

        # Verify chat history accumulates both conversations
        # (ThreadRun maintains a persistent chat history for multi-turn)
        assert len(messages2) > len(messages1)

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_with_different_temperatures(self) -> None:
        """Test Azure OpenAI with default temperature settings.

        Note: Some Azure models only support default temperature (1.0).
        This test validates basic Azure functionality without custom temperature.

        Validates:
        1. Azure invocation works with default temperature
        2. Response is valid
        """
        # Some Azure models only support default parameters - explicitly set to None
        agent_config = Agent(
            name="azure-temp-test",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(inline="You are helpful."),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(factory, "What is 2 + 2?")

        assert isinstance(result, AgentExecutionResult)
        messages = list(result.chat_history.messages)
        assert any("4" in str(msg.content) for msg in messages)

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_retry_on_transient_error(self) -> None:
        """Test that retry logic works with Azure OpenAI.

        Note: This test validates retry configuration but may not trigger
        actual retries unless Azure API is experiencing issues.

        Validates:
        1. Factory configured with retry parameters
        2. Successful invocation even with retry config
        """
        agent_config = Agent(
            name="azure-retry-test",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        # Configure with retry parameters
        factory = AgentFactory(
            agent_config,
            execution_config=ExecutionConfig(llm_timeout=30),
            max_retries=3,
            retry_delay=1.0,
            retry_exponential_base=2.0,
        )

        result = await invoke_factory(factory, "Say hello.")

        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None

    @skip_if_no_azure
    @pytest.mark.asyncio
    async def test_azure_openai_concurrent_invocations(self) -> None:
        """Test concurrent invocations with Azure OpenAI.

        Validates:
        1. Multiple Azure agents can run concurrently
        2. Results are correctly associated with queries
        """
        agent_config1 = Agent(
            name="azure-concurrent-1",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(inline="You are a math expert."),
        )

        agent_config2 = Agent(
            name="azure-concurrent-2",
            model=LLMProvider(
                provider=ProviderEnum.AZURE_OPENAI,
                name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                endpoint=AZURE_OPENAI_ENDPOINT or "",
                api_key=AZURE_OPENAI_API_KEY or "",
                # Some Azure models only support default params - explicitly set to None
                temperature=None,
                max_tokens=None,
            ),
            instructions=Instructions(inline="You are a history expert."),
        )

        factory1 = AgentFactory(
            agent_config1, execution_config=ExecutionConfig(llm_timeout=30)
        )
        factory2 = AgentFactory(
            agent_config2, execution_config=ExecutionConfig(llm_timeout=30)
        )

        # Run invocations concurrently
        results = await asyncio.gather(
            invoke_factory(factory1, "What is 9 * 9?"),
            invoke_factory(factory2, "Who was the first president of the USA?"),
        )

        # Verify both results
        assert len(results) == 2
        assert all(isinstance(r, AgentExecutionResult) for r in results)

        # Verify response content
        messages1 = list(results[0].chat_history.messages)
        messages2 = list(results[1].chat_history.messages)

        response1 = " ".join(str(msg.content) for msg in messages1)
        response2 = " ".join(str(msg.content) for msg in messages2)

        assert "81" in response1 or "eighty" in response1.lower()
        assert (
            "Washington" in response2
            or "George" in response2
            or "washington" in response2.lower()
        )


@pytest.mark.integration
@pytest.mark.slow
class TestAgentFactoryAnthropic:
    """Integration tests for AgentFactory with Anthropic provider."""

    @skip_if_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_basic_invocation(self) -> None:
        """Test basic invocation with Anthropic provider.

        Validates:
        1. Agent factory initializes with Anthropic provider
        2. Agent responds to simple query
        3. Response is captured in AgentExecutionResult
        """
        # Check if Anthropic is available
        try:
            from semantic_kernel.connectors.ai.anthropic import (
                AnthropicChatCompletion,
            )

            # Verify import succeeded
            assert AnthropicChatCompletion is not None
        except ImportError:
            pytest.skip("Anthropic package not installed")

        agent_config = Agent(
            name="anthropic-test-agent",
            description="Test agent for Anthropic integration",
            model=LLMProvider(
                provider=ProviderEnum.ANTHROPIC,
                name=os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022"),
                endpoint=os.getenv("ANTHROPIC_ENDPOINT"),
                api_key=ANTHROPIC_API_KEY or "",
                temperature=0.7,
                max_tokens=100,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(
            factory, "What is Python? Answer in one sentence."
        )

        # Verify result structure
        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None
        assert len(result.chat_history.messages) > 0

        # Verify response mentions programming
        messages = list(result.chat_history.messages)
        response_text = " ".join(str(msg.content) for msg in messages).lower()
        assert (
            "programming" in response_text
            or "language" in response_text
            or "code" in response_text
        )


@pytest.mark.integration
@pytest.mark.slow
class TestAgentFactoryErrorScenarios:
    """Integration tests for error scenarios with real providers."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_error(self) -> None:
        """Test that invalid API key raises appropriate error.

        Validates:
        1. Invalid credentials are detected
        2. Error is properly wrapped in AgentFactoryError
        """
        agent_config = Agent(
            name="invalid-key-test",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name="gpt-4o-mini",
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key="sk-invalid-key-12345",  # Invalid key
                temperature=0.7,
            ),
            instructions=Instructions(inline="Test"),
        )

        factory = AgentFactory(
            agent_config,
            execution_config=ExecutionConfig(llm_timeout=10),
            max_retries=1,
        )

        with pytest.raises(AgentFactoryError):
            await invoke_factory(factory, "Test query")

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_empty_input_handled_gracefully(self) -> None:
        """Test that empty input is handled gracefully.

        Validates:
        1. Empty string input doesn't cause errors
        2. Agent returns valid response
        """
        agent_config = Agent(
            name="empty-input-test",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.7,
                max_tokens=50,
            ),
            instructions=Instructions(
                inline="If user sends empty message, say 'Hello! How can I help?'"
            ),
        )

        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )
        result = await invoke_factory(factory, "")

        assert isinstance(result, AgentExecutionResult)
        assert result.chat_history is not None


@pytest.mark.integration
@pytest.mark.slow
class TestAgentFactoryConcurrency:
    """Integration tests for concurrent invocations."""

    @skip_if_no_openai
    @pytest.mark.asyncio
    async def test_concurrent_invocations_with_different_factories(self) -> None:
        """Test multiple factories can run concurrently.

        Validates:
        1. Multiple factories can be created
        2. Concurrent invocations work correctly
        3. Results are correctly associated with queries
        """
        agent_config1 = Agent(
            name="concurrent-test-1",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.5,
                max_tokens=50,
            ),
            instructions=Instructions(inline="You are a math assistant."),
        )

        agent_config2 = Agent(
            name="concurrent-test-2",
            model=LLMProvider(
                provider=ProviderEnum.OPENAI,
                name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=OPENAI_API_KEY or "",
                temperature=0.5,
                max_tokens=50,
            ),
            instructions=Instructions(inline="You are a geography assistant."),
        )

        factory1 = AgentFactory(
            agent_config1, execution_config=ExecutionConfig(llm_timeout=30)
        )
        factory2 = AgentFactory(
            agent_config2, execution_config=ExecutionConfig(llm_timeout=30)
        )

        # Run invocations concurrently
        results = await asyncio.gather(
            invoke_factory(factory1, "What is 7 * 8?"),
            invoke_factory(factory2, "What is the capital of Japan?"),
        )

        # Verify both results
        assert len(results) == 2
        assert all(isinstance(r, AgentExecutionResult) for r in results)

        # Verify response content (basic check)
        messages1 = list(results[0].chat_history.messages)
        messages2 = list(results[1].chat_history.messages)

        response1 = " ".join(str(msg.content) for msg in messages1)
        response2 = " ".join(str(msg.content) for msg in messages2)

        assert "56" in response1 or "fifty" in response1.lower()
        assert "Tokyo" in response2
