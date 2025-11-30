"""Unit tests for AgentFactory Ollama provider integration.

This module tests Ollama-specific functionality in AgentFactory including:
- Kernel creation with Ollama service
- Connection error handling
- Model not found error handling
"""

from unittest.mock import Mock, patch

import pytest

from holodeck.lib.test_runner.agent_factory import (
    AgentFactory,
    AgentFactoryError,
)
from holodeck.models.agent import Agent, Instructions
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestOllamaKernelCreation:
    """T018 [P] [US1] - Tests for kernel creation with Ollama provider."""

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_create_kernel_ollama_local(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test kernel creation with Ollama provider (local endpoint)."""
        # Setup mock
        mock_service = Mock()
        mock_service.service_id = "ollama-service"
        mock_ollama_chat.return_value = mock_service

        # Create agent config with Ollama
        agent_config = Agent(
            name="test-agent",
            description="Test agent with Ollama",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
                temperature=0.7,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        # Create agent factory (triggers _create_kernel)
        factory = AgentFactory(agent_config)

        # Verify OllamaChatCompletion was called with correct parameters
        mock_ollama_chat.assert_called_once_with(
            ai_model_id="llama3",
            host="http://localhost:11434",
        )

        # Verify service was added to kernel
        assert factory.kernel is not None
        assert factory._llm_service is mock_service

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_create_kernel_ollama_remote(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test kernel creation with Ollama provider (remote endpoint with auth)."""
        # Setup mock
        mock_service = Mock()
        mock_service.service_id = "ollama-service"
        mock_ollama_chat.return_value = mock_service

        # Create agent config with remote Ollama + API key
        agent_config = Agent(
            name="test-agent",
            description="Test agent with remote Ollama",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="phi3",
                endpoint="http://192.168.1.100:11434",
                api_key="my-secret-key",
                temperature=0.5,
                max_tokens=2000,
                top_p=0.9,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        # Create agent factory
        factory = AgentFactory(agent_config)

        # Verify OllamaChatCompletion was called with correct parameters
        # Note: Ollama connector doesn't have api_key parameter
        mock_ollama_chat.assert_called_once_with(
            ai_model_id="phi3",
            host="http://192.168.1.100:11434",
        )

        # Verify factory initialized successfully
        assert factory.kernel is not None
        assert factory.agent_config.model.provider == ProviderEnum.OLLAMA

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_create_kernel_ollama_different_models(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test kernel creation with different Ollama model names."""
        model_names = ["llama3", "phi3", "mistral", "codellama", "gemma"]

        for model_name in model_names:
            # Reset mock
            mock_ollama_chat.reset_mock()

            # Create agent config
            agent_config = Agent(
                name=f"test-agent-{model_name}",
                model=LLMProvider(
                    provider=ProviderEnum.OLLAMA,
                    name=model_name,
                    endpoint="http://localhost:11434",
                ),
                instructions=Instructions(inline="Test"),
            )

            # Create factory
            factory = AgentFactory(agent_config)

            # Verify correct model name was passed
            mock_ollama_chat.assert_called_once()
            call_kwargs = mock_ollama_chat.call_args.kwargs
            assert call_kwargs["ai_model_id"] == model_name
            assert factory.agent_config.model.name == model_name


class TestOllamaConnectionErrorHandling:
    """T019 [P] [US1] - Tests for Ollama connection error handling."""

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    def test_connection_error_during_initialization(
        self, mock_ollama_chat: Mock
    ) -> None:
        """Test that connection failures raise OllamaConnectionError."""
        # Mock OllamaChatCompletion to raise ConnectionError
        mock_ollama_chat.side_effect = ConnectionError(
            "Failed to connect to http://localhost:11434"
        )

        # Create agent config
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
        )

        # Verify OllamaConnectionError is raised
        with pytest.raises(AgentFactoryError) as exc_info:
            AgentFactory(agent_config)

        # Verify error message contains helpful guidance
        error_message = str(exc_info.value)
        assert "failed" in error_message.lower() or "ollama" in error_message.lower()

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    def test_connection_error_message_contains_endpoint(
        self, mock_ollama_chat: Mock
    ) -> None:
        """Test that connection error includes the endpoint URL."""
        endpoint = "http://192.168.1.100:11434"

        # Mock connection failure
        mock_ollama_chat.side_effect = ConnectionError(
            f"Failed to connect to {endpoint}"
        )

        # Create agent config with remote endpoint
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint=endpoint,
            ),
            instructions=Instructions(inline="Test"),
        )

        # Verify error contains endpoint
        with pytest.raises(AgentFactoryError) as exc_info:
            AgentFactory(agent_config)

        error_message = str(exc_info.value)
        # The error should mention connection or initialization failure
        assert "failed" in error_message.lower() or "error" in error_message.lower()

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    def test_timeout_error_during_initialization(self, mock_ollama_chat: Mock) -> None:
        """Test that timeout errors during initialization are handled."""
        # Mock OllamaChatCompletion to raise TimeoutError
        mock_ollama_chat.side_effect = TimeoutError("Connection timeout")

        # Create agent config
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
        )

        # Verify error is raised (wrapped in AgentFactoryError)
        with pytest.raises(AgentFactoryError):
            AgentFactory(agent_config)


class TestOllamaModelNotFoundError:
    """T020 [P] [US1] - Tests for Ollama model not found error handling."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    async def test_model_not_found_on_invoke(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test that model-not-found errors are detected and wrapped on first invoke."""
        # Setup: Initialization succeeds, but invoke fails with model not found
        mock_service = Mock()
        mock_service.service_id = "ollama-service"
        mock_ollama_chat.return_value = mock_service

        # Create agent config
        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="nonexistent-model",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
        )

        # Create factory (should succeed - lazy validation)
        factory = AgentFactory(agent_config)

        # Create thread run and mock _invoke_agent_impl to raise model not found
        thread_run = await factory.create_thread_run()

        async def failing_impl() -> None:
            raise Exception("model 'nonexistent-model' not found")

        with patch.object(thread_run, "_invoke_agent_impl", side_effect=failing_impl):
            # Verify invoke raises AgentFactoryError
            with pytest.raises(AgentFactoryError) as exc_info:
                await thread_run.invoke("Test query")

            # Verify error message indicates the problem
            error_message = str(exc_info.value)
            assert "error" in error_message.lower() or "fail" in error_message.lower()

    @pytest.mark.asyncio
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    async def test_model_error_message_patterns(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test detection of various model not found error message patterns."""
        error_patterns = [
            "model 'phi3' not found",
            "model not loaded",
            "no such model: llama3",
            "Model 'mistral' not available",
        ]

        for error_msg in error_patterns:
            # Reset mocks
            mock_ollama_chat.reset_mock()

            # Setup
            mock_service = Mock()
            mock_service.service_id = "ollama-service"
            mock_ollama_chat.return_value = mock_service

            # Create agent config
            agent_config = Agent(
                name="test-agent",
                model=LLMProvider(
                    provider=ProviderEnum.OLLAMA,
                    name="test-model",
                    endpoint="http://localhost:11434",
                ),
                instructions=Instructions(inline="Test"),
            )

            # Create factory and thread run
            factory = AgentFactory(agent_config)
            thread_run = await factory.create_thread_run()

            # Mock invoke to raise model error
            # Capture error_msg in closure to avoid loop variable binding issue
            def make_failing_impl(msg: str):  # type: ignore
                async def failing_impl() -> None:
                    raise Exception(msg)

                return failing_impl

            failing_side_effect = make_failing_impl(error_msg)
            with (
                patch.object(
                    thread_run,
                    "_invoke_agent_impl",
                    side_effect=failing_side_effect,
                ),
                pytest.raises(AgentFactoryError),
            ):
                await thread_run.invoke("Test")


class TestOllamaEmbeddingService:
    """Tests for Ollama embedding service registration in vectorstore tools."""

    @patch("holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding")
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_register_embedding_service_ollama_local(
        self,
        mock_chat_agent: Mock,
        mock_ollama_chat: Mock,
        mock_ollama_embedding: Mock,
    ) -> None:
        """Test embedding service registration with local Ollama endpoint."""
        # Setup mocks
        mock_chat_service = Mock()
        mock_chat_service.service_id = "ollama-chat"
        mock_ollama_chat.return_value = mock_chat_service

        mock_embedding_service = Mock()
        mock_ollama_embedding.return_value = mock_embedding_service

        # Create agent config with vectorstore tool
        from holodeck.models.tool import VectorstoreTool

        agent_config = Agent(
            name="test-agent",
            description="Test agent with vectorstore",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="You are helpful."),
            tools=[
                VectorstoreTool(
                    name="knowledge_base",
                    description="Search docs",
                    source="data/docs/",
                    embedding_model="nomic-embed-text:latest",
                    top_k=5,
                )
            ],
        )

        # Create factory (triggers _register_embedding_service)
        factory = AgentFactory(agent_config)

        # Verify OllamaTextEmbedding was called with correct params
        mock_ollama_embedding.assert_called_once_with(
            ai_model_id="nomic-embed-text:latest",
            host="http://localhost:11434",
        )

        # Verify service was set
        assert factory._embedding_service is mock_embedding_service

    @patch("holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding")
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_register_embedding_service_ollama_uses_default_model(
        self,
        mock_chat_agent: Mock,
        mock_ollama_chat: Mock,
        mock_ollama_embedding: Mock,
    ) -> None:
        """Test that Ollama uses nomic-embed-text when embedding_model not specified."""
        # Setup mocks
        mock_chat_service = Mock()
        mock_chat_service.service_id = "ollama-chat"
        mock_ollama_chat.return_value = mock_chat_service

        mock_embedding_service = Mock()
        mock_ollama_embedding.return_value = mock_embedding_service

        # Create agent WITHOUT embedding_model specified in tool
        from holodeck.models.tool import VectorstoreTool

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                VectorstoreTool(
                    name="kb",
                    description="Search",
                    source="data/",
                    top_k=5,
                    # embedding_model NOT specified - should use default
                )
            ],
        )

        # Create factory
        AgentFactory(agent_config)

        # Verify default model was used
        mock_ollama_embedding.assert_called_once_with(
            ai_model_id="nomic-embed-text:latest",  # DEFAULT
            host="http://localhost:11434",
        )

    @patch("holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding")
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_register_embedding_service_ollama_custom_model(
        self,
        mock_chat_agent: Mock,
        mock_ollama_chat: Mock,
        mock_ollama_embedding: Mock,
    ) -> None:
        """Test Ollama embedding service with custom embedding model."""
        # Setup mocks
        mock_chat_service = Mock()
        mock_chat_service.service_id = "ollama-chat"
        mock_ollama_chat.return_value = mock_chat_service

        mock_embedding_service = Mock()
        mock_ollama_embedding.return_value = mock_embedding_service

        # Create agent with custom embedding model
        from holodeck.models.tool import VectorstoreTool

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                VectorstoreTool(
                    name="kb",
                    description="Search",
                    source="data/",
                    embedding_model="mxbai-embed-large",  # CUSTOM
                    top_k=5,
                )
            ],
        )

        # Create factory
        AgentFactory(agent_config)

        # Verify custom model was used
        mock_ollama_embedding.assert_called_once_with(
            ai_model_id="mxbai-embed-large",
            host="http://localhost:11434",
        )

    @patch("holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding", None)
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_embedding_service_ollama_package_missing(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test that helpful error is raised when ollama package is not installed."""
        # Setup mocks
        mock_chat_service = Mock()
        mock_chat_service.service_id = "ollama-chat"
        mock_ollama_chat.return_value = mock_chat_service

        # Create agent with Ollama and vectorstore
        from holodeck.models.tool import VectorstoreTool

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                VectorstoreTool(
                    name="kb",
                    description="Search",
                    source="data/",
                    top_k=5,
                )
            ],
        )

        # Verify AgentFactoryError with helpful message
        with pytest.raises(AgentFactoryError) as exc_info:
            AgentFactory(agent_config)

        error_msg = str(exc_info.value).lower()
        assert "ollama" in error_msg
        assert "package" in error_msg
        assert "pip install ollama" in error_msg

    @pytest.mark.parametrize(
        "model_name",
        [
            "nomic-embed-text:latest",
            "mxbai-embed-large",
            "all-minilm",
            "custom-embedding-model",
        ],
    )
    @patch("holodeck.lib.test_runner.agent_factory.OllamaTextEmbedding")
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_register_embedding_service_ollama_model_names(
        self,
        mock_chat_agent: Mock,
        mock_ollama_chat: Mock,
        mock_ollama_embedding: Mock,
        model_name: str,
    ) -> None:
        """Test Ollama embedding service with various model names."""
        # Setup mocks
        mock_chat_service = Mock()
        mock_chat_service.service_id = "ollama-chat"
        mock_ollama_chat.return_value = mock_chat_service

        mock_embedding_service = Mock()
        mock_ollama_embedding.return_value = mock_embedding_service

        # Create agent with specified model
        from holodeck.models.tool import VectorstoreTool

        agent_config = Agent(
            name="test-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
            tools=[
                VectorstoreTool(
                    name="kb",
                    description="Search",
                    source="data/",
                    embedding_model=model_name,
                    top_k=5,
                )
            ],
        )

        # Create factory
        AgentFactory(agent_config)

        # Verify correct model was passed
        call_kwargs = mock_ollama_embedding.call_args.kwargs
        assert call_kwargs["ai_model_id"] == model_name


class TestOllamaIntegration:
    """Integration tests for complete Ollama workflows."""

    @pytest.mark.asyncio
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    async def test_full_workflow_with_ollama(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test complete workflow from initialization to invocation with Ollama."""
        # Setup mocks
        mock_service = Mock()
        mock_service.service_id = "ollama-service"
        mock_ollama_chat.return_value = mock_service

        # Create agent config
        agent_config = Agent(
            name="ollama-integration-test",
            description="Integration test with Ollama",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
                temperature=0.7,
                max_tokens=1000,
            ),
            instructions=Instructions(inline="You are a helpful assistant."),
        )

        # Initialize factory
        factory = AgentFactory(
            agent_config, execution_config=ExecutionConfig(llm_timeout=30)
        )

        # Verify initialization
        assert factory.agent_config.name == "ollama-integration-test"
        assert factory.agent_config.model.provider == ProviderEnum.OLLAMA
        assert factory.agent_config.model.name == "llama3"
        assert factory.agent_config.model.temperature == 0.7

        # Mock successful response
        mock_response = Mock()
        mock_response.content = "Hello! I'm running on Ollama."
        mock_response.metadata = {}

        async def mock_invoke(*args, **kwargs):  # type: ignore
            yield mock_response

        factory.agent.invoke = mock_invoke  # type: ignore

        # Create thread run and invoke
        thread_run = await factory.create_thread_run()
        result = await thread_run.invoke("Hello")

        # Verify result
        assert result is not None
        assert result.chat_history is not None

    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    def test_ollama_with_custom_parameters(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test Ollama initialization with custom execution parameters."""
        # Create agent config with all custom parameters
        agent_config = Agent(
            name="custom-ollama-agent",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="mistral:7b-instruct",
                endpoint="https://ollama.example.com:11434",
                api_key="custom-api-key",
                temperature=0.8,
                max_tokens=2048,
                top_p=0.95,
            ),
            instructions=Instructions(inline="Test"),
        )

        # Create factory with custom timeout and retry settings
        factory = AgentFactory(
            agent_config,
            execution_config=ExecutionConfig(llm_timeout=45),
            max_retries=5,
            retry_delay=1.5,
        )

        # Verify all custom settings are preserved
        assert factory.agent_config.model.name == "mistral:7b-instruct"
        assert factory.agent_config.model.endpoint == "https://ollama.example.com:11434"
        assert factory.agent_config.model.api_key == "custom-api-key"
        assert factory.agent_config.model.temperature == 0.8
        assert factory.agent_config.model.max_tokens == 2048
        assert factory.agent_config.model.top_p == 0.95
        assert factory.timeout == 45.0
        assert factory.max_retries == 5
        assert factory.retry_delay == 1.5

        # Verify OllamaChatCompletion was called with correct parameters
        # Note: Ollama connector doesn't have api_key parameter
        mock_ollama_chat.assert_called_once_with(
            ai_model_id="mistral:7b-instruct",
            host="https://ollama.example.com:11434",
        )

    @pytest.mark.asyncio
    @patch("holodeck.lib.test_runner.agent_factory.OllamaChatCompletion")
    @patch("holodeck.lib.test_runner.agent_factory.ChatCompletionAgent")
    async def test_ollama_retry_on_transient_failure(
        self, mock_chat_agent: Mock, mock_ollama_chat: Mock
    ) -> None:
        """Test that transient Ollama errors trigger retry logic."""
        from semantic_kernel.contents import ChatHistory

        from holodeck.lib.test_runner.agent_factory import AgentExecutionResult

        # Setup
        mock_service = Mock()
        mock_service.service_id = "ollama-service"
        mock_ollama_chat.return_value = mock_service

        # Create agent config
        agent_config = Agent(
            name="retry-test",
            model=LLMProvider(
                provider=ProviderEnum.OLLAMA,
                name="llama3",
                endpoint="http://localhost:11434",
            ),
            instructions=Instructions(inline="Test"),
        )

        # Create factory with retry config
        factory = AgentFactory(agent_config, max_retries=3, retry_delay=0.01)
        thread_run = await factory.create_thread_run()

        # Mock: fail twice, then succeed
        call_count = 0

        async def flaky_invoke(*args, **kwargs) -> AgentExecutionResult:  # type: ignore
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary Ollama connection issue")

            history = ChatHistory()
            return AgentExecutionResult(
                tool_calls=[], tool_results=[], chat_history=history
            )

        with patch.object(thread_run, "_invoke_agent_impl", side_effect=flaky_invoke):
            result = await thread_run.invoke("Test query")

            # Verify retry logic worked
            assert result is not None
            assert call_count == 3  # Failed twice, succeeded on third attempt
