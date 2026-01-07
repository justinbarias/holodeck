"""Agent bridge for Semantic Kernel integration.

This module provides the AgentBridge class for executing agents
using Semantic Kernel with support for multiple LLM providers
(Azure OpenAI, OpenAI, Anthropic).

Key features:
- Kernel initialization and configuration
- Agent invocation with timeout and retry logic
- Response content and tool call extraction
- ChatHistory management for multi-turn conversations
"""

import asyncio
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.agents import Agent as SKAgent
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments, KernelFunction
from semantic_kernel.functions.kernel_function_from_method import (
    KernelFunctionFromMethod,
)

from holodeck.config.schema import SchemaValidator
from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_retry
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import ProviderEnum
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool import MCPTool, VectorstoreTool

# Default configuration constants for AgentFactory
DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_DELAY_SECONDS: float = 2.0
DEFAULT_RETRY_EXPONENTIAL_BASE: float = 2.0

# Try to import Anthropic support (optional dependency)
try:
    from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
except ImportError:
    AnthropicChatCompletion = None  # type: ignore[misc,assignment]

# Try to import Ollama support (optional dependency)
try:
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
except ImportError:
    OllamaChatCompletion = None  # type: ignore[misc,assignment]

# Try to import Ollama embedding support (optional dependency)
try:
    from semantic_kernel.connectors.ai.ollama import OllamaTextEmbedding
except ImportError:
    OllamaTextEmbedding = None  # type: ignore[misc,assignment]

logger = get_logger(__name__)


@dataclass
class AgentExecutionResult:
    """Result of agent execution containing tool calls and conversation history.

    Attributes:
        tool_calls: List of tool calls made by the agent during execution.
            Each dict contains 'name' and 'arguments' keys.
        tool_results: List of tool execution results for retrieval context.
            Each dict contains 'name' (tool name) and 'result' (execution output).
        chat_history: Complete conversation history including user inputs
            and agent responses
        token_usage: Token usage metadata if provided by LLM provider
    """

    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    chat_history: ChatHistory
    token_usage: TokenUsage | None = None


class AgentFactoryError(Exception):
    """Error raised during agent bridge operations."""

    pass


class AgentThreadRun:
    """Encapsulates a single agent execution thread with isolated chat history.

    Each instance maintains its own ChatHistory, ensuring test case isolation.
    Created by AgentFactory.create_thread_run().

    This class owns the invocation logic and response extraction methods,
    providing complete isolation between different test cases or chat sessions.
    """

    def __init__(
        self,
        agent: SKAgent,
        kernel: Kernel,
        kernel_arguments: KernelArguments,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
        retry_exponential_base: float = DEFAULT_RETRY_EXPONENTIAL_BASE,
        observability_enabled: bool = False,
    ) -> None:
        """Initialize an agent thread run with isolated chat history.

        Args:
            agent: Semantic Kernel agent instance.
            kernel: Configured Kernel instance.
            kernel_arguments: KernelArguments for agent invocation.
            timeout: Timeout in seconds for agent invocation.
            max_retries: Maximum retry attempts for transient failures.
            retry_delay: Base delay in seconds for exponential backoff.
            retry_exponential_base: Exponential base for backoff calculation.
            observability_enabled: Whether OTel tracing is enabled.
        """
        self.agent = agent
        self.kernel = kernel
        self.kernel_arguments = kernel_arguments
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential_base = retry_exponential_base
        self.observability_enabled = observability_enabled
        self.chat_history = ChatHistory()  # Fresh history per instance

        logger.debug(
            f"AgentThreadRun initialized: timeout={self.timeout}s, "
            f"max_retries={self.max_retries}, retry_delay={self.retry_delay}s"
        )

    async def invoke(self, user_input: str) -> AgentExecutionResult:
        """Invoke agent with user input.

        Args:
            user_input: User's input message.

        Returns:
            AgentExecutionResult with tool_calls and complete chat_history.

        Raises:
            AgentFactoryError: If invocation fails after retries.
        """
        # Create tracer span only if observability is enabled
        if self.observability_enabled:
            from holodeck.lib.observability import get_tracer

            tracer = get_tracer(__name__)
            span_context: Any = tracer.start_as_current_span("holodeck.agent.invoke")
        else:
            span_context = nullcontext()

        with span_context:
            try:
                # Add user input to chat history
                self.chat_history.add_user_message(user_input)

                # Invoke with timeout and retry logic
                if self.timeout:
                    logger.debug(
                        f"Invoking agent with timeout={self.timeout}s "
                        f"(input length: {len(user_input)} chars)"
                    )
                    result = await asyncio.wait_for(
                        self._invoke_with_retry(), timeout=self.timeout
                    )
                else:
                    logger.debug(
                        f"Invoking agent without timeout "
                        f"(input length: {len(user_input)} chars)"
                    )
                    result = await self._invoke_with_retry()

                return result

            except TimeoutError as e:
                raise AgentFactoryError(
                    f"Agent invocation timeout after {self.timeout}s"
                ) from e
            except AgentFactoryError:
                raise
            except Exception as e:
                raise AgentFactoryError(f"Agent invocation failed: {e}") from e

    async def _invoke_with_retry(self) -> AgentExecutionResult:
        """Invoke agent with retry logic for transient failures.

        Returns:
            AgentExecutionResult with tool_calls and complete chat_history.

        Raises:
            AgentFactoryError: If all retries are exhausted.
        """
        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Agent invocation attempt {attempt + 1}/{self.max_retries}"
                )
                result = await self._invoke_agent_impl()
                logger.debug(
                    f"Agent invocation succeeded on attempt {attempt + 1}, "
                    f"tool_calls={len(result.tool_calls)}"
                )
                return result

            except (ConnectionError, TimeoutError) as e:
                # Retryable error
                last_error = e
                if attempt < self.max_retries - 1:
                    log_retry(
                        logger,
                        "Agent invocation",
                        attempt=attempt + 1,
                        max_attempts=self.max_retries,
                        delay=delay,
                        error=e,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.retry_exponential_base, 60.0)  # Cap at 60s
                else:
                    logger.error(
                        f"All {self.max_retries} retries exhausted for agent invocation"
                    )

            except Exception as e:
                # Non-retryable error
                logger.error(
                    f"Non-retryable error during agent invocation: {e}", exc_info=True
                )
                raise AgentFactoryError(
                    f"Non-retryable error during agent invocation: {e}"
                ) from e

        # All retries exhausted
        logger.error(
            f"Agent invocation failed after {self.max_retries} attempts: {last_error}"
        )
        raise AgentFactoryError(
            f"Agent invocation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    async def _invoke_agent_impl(self) -> AgentExecutionResult:
        """Internal implementation of agent invocation.

        Returns:
            AgentExecutionResult with tool_calls, tool_results, and chat_history.
        """
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        token_usage: TokenUsage | None = None
        try:
            # Track message count before invoke to only extract current turn's
            # tool calls. This prevents returning accumulated tool calls from
            # previous turns in multi-turn conversations.
            history_length_before = len(self.chat_history.messages)

            # Invoke agent with chat history
            thread = ChatHistoryAgentThread(self.chat_history)
            arguments = self.kernel_arguments
            # Get the async generator and properly close it after extracting
            # first response. This avoids "Failed to detach context" errors
            # from OTel when breaking early from the generator.
            invoke_gen = self.agent.invoke(  # pyright: ignore[reportUnknownMemberType]
                thread=thread,
                arguments=arguments,
            )
            try:
                async for response in invoke_gen:
                    # Extract token usage from first response
                    token_usage = self._extract_token_usage(response)
                    break  # Only process first response
            finally:
                # Explicitly close the generator to ensure proper OTel context cleanup
                await invoke_gen.aclose()

            # Extract tool calls and results from thread's chat history
            # Only scan messages added during this turn (after history_length_before)
            tool_calls, tool_results = await self._extract_tool_calls_from_thread(
                thread, start_index=history_length_before
            )

            # Note: Assistant messages are automatically added to self.chat_history
            # by ChatHistoryAgentThread.on_new_message() during agent.invoke().
            # The thread shares the same ChatHistory object passed to its constructor.

            return AgentExecutionResult(
                tool_calls=tool_calls,
                tool_results=tool_results,
                chat_history=self.chat_history,
                token_usage=token_usage,
            )

        except Exception as e:
            raise AgentFactoryError(f"Agent execution failed: {e}") from e

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from agent response.

        Args:
            response: Response object from agent invocation.

        Returns:
            Extracted response text, or empty string if no content.
        """
        try:
            if hasattr(response, "content"):
                content = response.content
                return str(content) if content else ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return ""

    async def _extract_tool_calls_from_thread(
        self, thread: ChatHistoryAgentThread, start_index: int = 0
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract tool calls and results from ChatHistoryAgentThread.

        Scans the thread's message history for FunctionCallContent items to track
        which tools were invoked, and FunctionResultContent items to capture
        tool execution results for RAG evaluation metrics.

        Args:
            thread: ChatHistoryAgentThread containing the conversation history
                after agent invocation.
            start_index: Index to start scanning from. Use this to only extract
                tool calls from the current turn by passing the message count
                before the invoke. Defaults to 0 (scan all messages).

        Returns:
            Tuple of:
            - tool_calls: List of dicts with 'name' and 'arguments' keys
            - tool_results: List of dicts with 'name' and 'result' keys
        """
        from semantic_kernel.contents import FunctionCallContent, FunctionResultContent

        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        seen_call_ids: set[str] = set()  # Avoid duplicates
        seen_result_ids: set[str] = set()  # Avoid duplicate results

        # Map call_id -> tool name for matching results to calls
        call_id_to_name: dict[str, str] = {}

        try:
            # Iterate through thread messages (async generator)
            # Skip messages before start_index to only process current turn
            message_index = 0
            async for message in thread.get_messages():
                if message_index < start_index:
                    message_index += 1
                    continue
                message_index += 1
                if hasattr(message, "items") and message.items:
                    for item in message.items:
                        if isinstance(item, FunctionCallContent):
                            call_id = getattr(item, "id", None) or getattr(
                                item, "call_id", ""
                            )
                            if call_id and call_id in seen_call_ids:
                                continue
                            if call_id:
                                seen_call_ids.add(call_id)

                            # Build full function name from plugin_name + function_name
                            plugin_name = getattr(item, "plugin_name", "") or ""
                            function_name = getattr(
                                item, "function_name", ""
                            ) or getattr(item, "name", "")
                            full_name = (
                                f"{plugin_name}-{function_name}"
                                if plugin_name
                                else function_name
                            )

                            # Store mapping for result matching
                            if call_id:
                                call_id_to_name[call_id] = full_name

                            # Parse arguments (can be str, Mapping, or None)
                            raw_args = getattr(item, "arguments", None)
                            if isinstance(raw_args, str):
                                import json

                                try:
                                    arguments = json.loads(raw_args)
                                except json.JSONDecodeError:
                                    arguments = {"raw": raw_args}
                            elif isinstance(raw_args, dict):
                                arguments = raw_args
                            else:
                                arguments = dict(raw_args) if raw_args else {}

                            tool_calls.append(
                                {
                                    "name": full_name,
                                    "arguments": arguments,
                                }
                            )

                        elif isinstance(item, FunctionResultContent):
                            # Extract tool result for RAG evaluation
                            raw_result_id = getattr(item, "id", None) or getattr(
                                item, "call_id", ""
                            )
                            result_id = str(raw_result_id) if raw_result_id else ""
                            if result_id and result_id in seen_result_ids:
                                continue
                            if result_id:
                                seen_result_ids.add(result_id)

                            # Get result content
                            result_value = getattr(item, "result", None)
                            if result_value is None:
                                result_value = getattr(item, "content", "")

                            # Convert to string if needed
                            result_str = (
                                str(result_value) if result_value is not None else ""
                            )

                            # Match result to tool name via call_id
                            tool_name = call_id_to_name.get(result_id, "")
                            if not tool_name:
                                # Try to get name from FunctionResultContent
                                plugin_name = getattr(item, "plugin_name", "") or ""
                                function_name = getattr(
                                    item, "function_name", ""
                                ) or getattr(item, "name", "")
                                tool_name = (
                                    f"{plugin_name}-{function_name}"
                                    if plugin_name
                                    else function_name
                                )

                            if result_str:  # Only add non-empty results
                                tool_results.append(
                                    {
                                        "name": tool_name,
                                        "result": result_str,
                                    }
                                )

        except Exception as e:
            logger.warning(f"Failed to extract tool calls from thread: {e}")

        return tool_calls, tool_results

    def _extract_token_usage(self, response: Any) -> TokenUsage | None:
        """Extract token usage from agent response metadata.

        Accesses token usage from the response message's metadata dictionary,
        which is populated by OpenAI/Azure providers. Returns None if usage
        data is not available.

        Args:
            response: Response object (ChatMessageContent) from agent invocation.

        Returns:
            TokenUsage object if available, None otherwise.
        """
        try:
            # Check for metadata attribute (present on ChatMessageContent)
            if (
                hasattr(response, "metadata")
                and isinstance(response.metadata, dict)
                and "usage" in response.metadata
            ):
                usage_obj: Any = response.metadata["usage"]
                return TokenUsage(
                    prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0)),
                    completion_tokens=int(getattr(usage_obj, "completion_tokens", 0)),
                    total_tokens=int(getattr(usage_obj, "prompt_tokens", 0))
                    + int(getattr(usage_obj, "completion_tokens", 0)),
                )
        except Exception as e:
            logger.warning(f"Failed to extract token usage: {e}")
        return None


class AgentFactory:
    """Factory for creating and executing agents using Semantic Kernel.

    Handles Kernel creation, agent invocation, response extraction,
    and tool call handling with support for multiple LLM providers.
    """

    def __init__(
        self,
        agent_config: Agent,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
        retry_exponential_base: float = DEFAULT_RETRY_EXPONENTIAL_BASE,
        force_ingest: bool = False,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize agent factory with Semantic Kernel.

        Args:
            agent_config: Agent configuration with model and instructions
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Base delay in seconds for exponential backoff
            retry_exponential_base: Exponential base for backoff calculation
            force_ingest: Force re-ingestion of vector store source files
            execution_config: Execution configuration for timeouts and file processing

        Raises:
            AgentFactoryError: If kernel initialization fails
        """
        self.agent_config = agent_config
        self._execution_config = execution_config
        # Get timeout from execution_config or use default
        self.timeout: float | None = (
            execution_config.llm_timeout
            if execution_config
            else DEFAULT_TIMEOUT_SECONDS
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential_base = retry_exponential_base
        self._retry_count = 0
        self.kernel_arguments: KernelArguments | None = None
        self._llm_service: Any | None = None
        self._force_ingest = force_ingest

        # Vectorstore tool support
        self._tools_initialized = False
        self._vectorstore_tools: list[Any] = []
        self._embedding_service: Any = None

        # MCP tool support
        self._mcp_plugins: list[Any] = []

        logger.debug(
            f"Initializing AgentFactory: agent={agent_config.name}, "
            f"provider={agent_config.model.provider}, timeout={self.timeout}s, "
            f"max_retries={max_retries}"
        )

        try:
            self.kernel = self._create_kernel()

            # Register embedding service if vectorstore tools are configured
            if self._has_vectorstore_tools():
                self._register_embedding_service()

            self.agent = self._create_agent()
            logger.info(
                f"AgentFactory initialized successfully for agent: {agent_config.name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent factory: {e}", exc_info=True)
            raise AgentFactoryError(f"Failed to initialize agent factory: {e}") from e

    def _is_observability_enabled(self) -> bool:
        """Check if observability is enabled for this agent.

        Returns:
            True if observability is configured and enabled, False otherwise.
        """
        return (
            self.agent_config.observability is not None
            and self.agent_config.observability.enabled
        )

    def _create_kernel(self) -> Kernel:
        """Create and configure Semantic Kernel for LLM provider.

        Returns:
            Configured Kernel instance

        Raises:
            AgentFactoryError: If kernel creation fails
        """
        try:
            logger.debug("Creating Semantic Kernel")
            kernel = Kernel()

            model_config = self.agent_config.model

            # Add service based on provider type
            logger.debug(
                f"Configuring LLM service: provider={model_config.provider}, "
                f"model={model_config.name}"
            )
            service: Any
            if model_config.provider == ProviderEnum.AZURE_OPENAI:
                service = AzureChatCompletion(
                    deployment_name=model_config.name,
                    endpoint=model_config.endpoint,
                    api_key=model_config.api_key,
                )
            elif model_config.provider == ProviderEnum.OPENAI:
                service = OpenAIChatCompletion(
                    ai_model_id=model_config.name,
                    api_key=model_config.api_key,
                )
            elif model_config.provider == ProviderEnum.ANTHROPIC:
                if AnthropicChatCompletion is None:
                    raise AgentFactoryError(
                        "Anthropic provider requires 'anthropic' package. "
                        "Install with: pip install anthropic"
                    )
                service = AnthropicChatCompletion(
                    ai_model_id=model_config.name,
                    api_key=model_config.api_key,
                )
            elif model_config.provider == ProviderEnum.OLLAMA:
                if OllamaChatCompletion is None:
                    raise AgentFactoryError(
                        "Ollama provider requires 'ollama' package. "
                        "Install with: pip install ollama"
                    )
                # Use endpoint if provided, otherwise let Ollama use its default (http://127.0.0.1:11434)
                service = OllamaChatCompletion(
                    ai_model_id=model_config.name,
                    host=model_config.endpoint if model_config.endpoint else None,
                )
            else:
                raise AgentFactoryError(
                    f"Unsupported LLM provider: {model_config.provider}"
                )

            self._llm_service = service
            kernel.add_service(service)
            services = getattr(kernel, "services", None)
            if not isinstance(services, dict):
                kernel.services = {}
                services = kernel.services
            service_id = getattr(service, "service_id", "default")
            services[service_id] = service
            logger.debug("Kernel created and service added successfully")
            return kernel

        except Exception as e:
            logger.error(f"Kernel creation failed: {e}", exc_info=True)
            raise AgentFactoryError(f"Kernel creation failed: {e}") from e

    def _has_vectorstore_tools(self) -> bool:
        """Check if agent config contains any vectorstore tools.

        Returns:
            True if at least one vectorstore tool is configured.
        """
        if not self.agent_config.tools:
            return False

        for tool in self.agent_config.tools:
            if isinstance(tool, VectorstoreTool):
                return True
        return False

    def _get_embedding_model(self) -> str:
        """Get embedding model from first vectorstore tool or use provider default.

        Returns:
            Embedding model name to use for TextEmbedding service.
        """
        # Check if any vectorstore tool has explicit embedding_model
        if self.agent_config.tools:
            for tool in self.agent_config.tools:
                if isinstance(tool, VectorstoreTool) and tool.embedding_model:
                    return tool.embedding_model

        # Return provider-specific default
        if self.agent_config.model.provider == ProviderEnum.OLLAMA:
            return "nomic-embed-text:latest"
        else:
            # OpenAI/Azure OpenAI default
            return "text-embedding-3-small"

    def _register_embedding_service(self) -> None:
        """Register TextEmbedding service on kernel for vectorstore tools.

        Supports OpenAI and Azure OpenAI embedding providers. Uses the same
        credentials as the chat model configured in agent_config.model.

        Raises:
            AgentFactoryError: If provider doesn't support embeddings.
        """
        model_config = self.agent_config.model
        embedding_model = self._get_embedding_model()

        logger.debug(
            f"Registering embedding service: model={embedding_model}, "
            f"provider={model_config.provider}"
        )

        if model_config.provider == ProviderEnum.OPENAI:
            self._embedding_service = OpenAITextEmbedding(
                ai_model_id=embedding_model,
                api_key=model_config.api_key,
            )
        elif model_config.provider == ProviderEnum.AZURE_OPENAI:
            self._embedding_service = AzureTextEmbedding(
                deployment_name=embedding_model,
                endpoint=model_config.endpoint,
                api_key=model_config.api_key,
            )
        elif model_config.provider == ProviderEnum.OLLAMA:
            if OllamaTextEmbedding is None:
                raise AgentFactoryError(
                    "Ollama provider requires 'ollama' package. "
                    "Install with: pip install ollama"
                )
            self._embedding_service = OllamaTextEmbedding(
                ai_model_id=embedding_model,
                host=model_config.endpoint if model_config.endpoint else None,
            )
        else:
            raise AgentFactoryError(
                f"Embedding service not supported for provider: "
                f"{model_config.provider}. "
                "Vectorstore tools require OpenAI, Azure OpenAI, or Ollama provider."
            )

        self.kernel.add_service(self._embedding_service)
        logger.debug(f"Embedding service registered: {embedding_model}")

    def _create_search_kernel_function(
        self,
        tool: Any,
        tool_name: str,
        tool_description: str,
    ) -> KernelFunction:
        """Create a KernelFunction from VectorStoreTool.search method.

        Creates a wrapper function that calls tool.search() and registers it
        as a KernelFunction using the @kernel_function decorator.

        Args:
            tool: Initialized VectorStoreTool instance.
            tool_name: Name for the kernel function (from tool config).
            tool_description: Description for the function (from tool config).

        Returns:
            KernelFunction that can be registered on the kernel.
        """
        from semantic_kernel.functions.kernel_function_decorator import (
            kernel_function as kernel_function_decorator,
        )

        @kernel_function_decorator(name=tool_name, description=tool_description)
        async def search_wrapper(query: str) -> str:
            """Search the knowledge base for relevant content."""
            result = await tool.search(query)
            return str(result)

        return KernelFunctionFromMethod(
            method=search_wrapper,
            plugin_name="vectorstore",
        )

    async def _register_vectorstore_tools(self) -> None:
        """Discover, initialize, and register vectorstore tools from agent config.

        Iterates through agent_config.tools, finds vectorstore tools, creates
        VectorStoreTool instances, initializes them, and registers their search
        methods as KernelFunctions.

        Raises:
            AgentFactoryError: If tool initialization fails.
        """
        if not self.agent_config.tools:
            return

        # Check if there are any vectorstore tools before importing
        if not self._has_vectorstore_tools():
            return

        from holodeck.tools.vectorstore_tool import VectorStoreTool

        # Get provider type from agent config for dimension resolution
        provider_type = self.agent_config.model.provider.value

        for tool_config in self.agent_config.tools:
            # Only process vectorstore tools
            if not isinstance(tool_config, VectorstoreTool):
                continue

            config = tool_config

            try:
                # Create tool and inject embedding service
                # VectorStoreTool reads base_dir from agent_base_dir context var
                tool = VectorStoreTool(config, execution_config=self._execution_config)
                tool.set_embedding_service(self._embedding_service)

                # Initialize (async - ingests files, generates embeddings)
                # Pass provider type for dimension auto-detection
                await tool.initialize(
                    force_ingest=self._force_ingest, provider_type=provider_type
                )

                # Create and register KernelFunction
                kernel_function = self._create_search_kernel_function(
                    tool=tool,
                    tool_name=config.name,
                    tool_description=config.description,
                )

                self.kernel.add_function(
                    plugin_name="vectorstore", function=kernel_function
                )
                self._vectorstore_tools.append(tool)

                logger.info(f"Registered vectorstore tool: {config.name}")

            except Exception as e:
                logger.error(
                    f"Failed to initialize vectorstore tool {config.name}: {e}"
                )
                raise AgentFactoryError(
                    f"Failed to initialize vectorstore tool {config.name}: {e}"
                ) from e

    def _has_mcp_tools(self) -> bool:
        """Check if agent config contains any MCP tools.

        Returns:
            True if at least one MCP tool is configured.
        """
        if not self.agent_config.tools:
            return False

        return any(isinstance(tool, MCPTool) for tool in self.agent_config.tools)

    async def _register_mcp_tools(self) -> None:
        """Discover, initialize, and register MCP tools from agent config.

        Iterates through agent_config.tools, finds MCP tools, creates
        SK MCP plugins via the factory, connects them, and registers
        their discovered tools on the kernel.

        MCP plugins handle their own lifecycle via async context managers.
        SK will manage cleanup when plugins go out of scope.

        Raises:
            AgentFactoryError: If MCP plugin creation or connection fails.
        """
        if not self.agent_config.tools:
            return

        if not self._has_mcp_tools():
            return

        # Import factory lazily to avoid circular imports
        from holodeck.tools.mcp.factory import create_mcp_plugin

        for tool_config in self.agent_config.tools:
            # Only process MCP tools
            if not isinstance(tool_config, MCPTool):
                continue

            plugin = None
            try:
                # Create SK MCP plugin via factory
                plugin = create_mcp_plugin(tool_config)

                # Connect plugin (enters async context manager)
                await plugin.__aenter__()

                # Track plugin IMMEDIATELY after __aenter__ succeeds
                # This ensures cleanup happens even if add_plugin fails
                self._mcp_plugins.append(plugin)

                # Register the plugin on the kernel
                # SK MCP plugins auto-register their tools when connected
                self.kernel.add_plugin(plugin)

                logger.info(f"Registered MCP tool: {tool_config.name}")

            except Exception as e:
                # If plugin was entered but not yet tracked, clean it up
                if plugin is not None and plugin not in self._mcp_plugins:
                    try:
                        await plugin.__aexit__(None, None, None)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Error cleaning up MCP plugin {tool_config.name} "
                            f"after initialization failure: {cleanup_error}"
                        )

                logger.error(f"Failed to initialize MCP tool {tool_config.name}: {e}")
                raise AgentFactoryError(
                    f"Failed to initialize MCP tool {tool_config.name}: {e}"
                ) from e

    async def _ensure_tools_initialized(self) -> None:
        """Ensure all tools are initialized (called before first invoke).

        This implements lazy initialization - tools are initialized on the first
        call to invoke() rather than during __init__.
        """
        if self._tools_initialized:
            return
        await self._register_vectorstore_tools()
        await self._register_mcp_tools()
        self._tools_initialized = True

    async def shutdown(self) -> None:
        """Shutdown all MCP plugins and release resources.

        Must be called from the same task context where the factory was used.
        Properly exits all MCP plugin async context managers to avoid
        'Attempted to exit cancel scope in a different task' errors.
        """
        errors: list[Exception] = []

        # Shutdown MCP plugins in reverse order
        for plugin in reversed(self._mcp_plugins):
            try:
                plugin_name = getattr(plugin, "name", "unknown")
                logger.debug(f"Shutting down MCP plugin: {plugin_name}")
                await plugin.__aexit__(None, None, None)
                logger.info(f"MCP plugin shut down: {plugin_name}")
            except Exception as e:
                plugin_name = getattr(plugin, "name", "unknown")
                logger.warning(f"Error shutting down MCP plugin {plugin_name}: {e}")
                errors.append(e)

        self._mcp_plugins.clear()

        # Cleanup vectorstore tools if they have cleanup methods
        for tool in self._vectorstore_tools:
            try:
                if hasattr(tool, "cleanup"):
                    await tool.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up vectorstore tool: {e}")
                errors.append(e)

        self._vectorstore_tools.clear()
        self._tools_initialized = False

        if errors:
            logger.warning(f"Shutdown completed with {len(errors)} error(s)")

    def _create_agent(self) -> SKAgent:
        """Create Semantic Kernel Agent with configuration.

        Returns:
            Configured SKAgent instance

        Raises:
            AgentFactoryError: If agent creation fails
        """
        try:
            # Get instructions from config
            instructions = self._load_instructions()

            # Create agent with instructions
            kernel_arguments = self._build_kernel_arguments()

            agent = ChatCompletionAgent(
                name=self.agent_config.name,
                description=self.agent_config.description,
                kernel=self.kernel,
                instructions=instructions,
                arguments=kernel_arguments,
            )

            # Capture kernel arguments for invocation reuse
            self.kernel_arguments = kernel_arguments

            return agent

        except Exception as e:
            raise AgentFactoryError(f"Agent creation failed: {e}") from e

    def _load_instructions(self) -> str:
        """Load agent instructions from config.

        Uses agent_base_dir context variable to resolve relative file paths.
        If agent_base_dir is not set, falls back to current working directory.

        Returns:
            Instruction text for the agent

        Raises:
            AgentFactoryError: If instructions cannot be loaded
        """
        try:
            instructions = self.agent_config.instructions

            if instructions.inline:
                return instructions.inline
            elif instructions.file:
                from pathlib import Path

                from holodeck.config.context import agent_base_dir

                # Resolve file path relative to agent_base_dir context
                base_dir = agent_base_dir.get()
                if base_dir:
                    file_path = Path(base_dir) / instructions.file
                else:
                    file_path = Path(instructions.file)

                if not file_path.exists():
                    raise FileNotFoundError(f"Instructions file not found: {file_path}")
                return file_path.read_text()
            else:
                raise AgentFactoryError("No instructions provided (file or inline)")

        except Exception as e:
            raise AgentFactoryError(f"Failed to load instructions: {e}") from e

    async def create_thread_run(self) -> AgentThreadRun:
        """Create a new isolated agent thread run.

        Each thread run has its own ChatHistory, suitable for:
        - Individual test case execution
        - Isolated conversation sessions

        This method ensures tools are initialized before creating the run.

        Returns:
            A new AgentThreadRun instance with fresh chat history.
        """
        await self._ensure_tools_initialized()

        # Ensure kernel_arguments are built
        if self.kernel_arguments is None:
            self.kernel_arguments = self._build_kernel_arguments()

        exec_timeout = (
            self._execution_config.llm_timeout if self._execution_config else "N/A"
        )
        logger.debug(
            f"Creating AgentThreadRun with timeout={self.timeout}s "
            f"(from execution_config.llm_timeout={exec_timeout})"
        )

        return AgentThreadRun(
            agent=self.agent,
            kernel=self.kernel,
            kernel_arguments=self.kernel_arguments,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            retry_exponential_base=self.retry_exponential_base,
            observability_enabled=self._is_observability_enabled(),
        )

    def _build_kernel_arguments(self) -> KernelArguments:
        """Create kernel arguments with execution settings from configuration."""

        settings = self._create_prompt_execution_settings()
        return KernelArguments(settings=settings)

    def _create_prompt_execution_settings(self) -> PromptExecutionSettings:
        """Build provider-specific prompt execution settings."""

        if not hasattr(self, "kernel"):
            raise AgentFactoryError("Kernel must be initialized before settings.")

        service: Any | None = None
        if getattr(self.kernel, "services", None):
            try:
                service = next(iter(self.kernel.services.values()))
            except StopIteration:
                service = None

        if service is None:
            service = self._llm_service

        if service is None:
            raise AgentFactoryError("No LLM services configured on the kernel.")
        settings_cls = service.get_prompt_execution_settings_class()
        settings: PromptExecutionSettings = settings_cls()

        self._apply_model_settings(settings)
        self._apply_response_format(settings)

        return settings

    def _apply_model_settings(self, settings: PromptExecutionSettings) -> None:
        """Apply LLM model parameters to prompt execution settings."""

        model_config = self.agent_config.model

        if hasattr(settings, "temperature") and model_config.temperature is not None:
            settings.temperature = model_config.temperature

        if hasattr(settings, "top_p") and model_config.top_p is not None:
            settings.top_p = model_config.top_p

        if model_config.max_tokens is not None:
            # Prefer max_completion_tokens (newer OpenAI API) over max_tokens (legacy)
            # Some newer models reject max_tokens parameter entirely
            if hasattr(settings, "max_completion_tokens"):
                settings.max_completion_tokens = model_config.max_tokens
            elif hasattr(settings, "max_tokens"):
                settings.max_tokens = model_config.max_tokens

        if hasattr(settings, "ai_model_id"):
            settings.ai_model_id = model_config.name

    def _apply_response_format(self, settings: PromptExecutionSettings) -> None:
        """Attach response format schema to execution settings if provided."""

        try:
            response_format = self._load_response_format()
        except Exception as e:
            raise AgentFactoryError(f"Failed to load response_format: {e}") from e

        if response_format is not None and hasattr(settings, "response_format"):
            settings.response_format = self._wrap_response_format(response_format)

        if response_format is not None and hasattr(settings, "format"):
            settings.format = response_format

    def _wrap_response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Wrap JSON schema using the OpenAI response_format structure."""

        if "json_schema" in schema:
            return schema

        if schema.get("type") == "json_schema" and "schema" in schema:
            return {
                "type": "json_schema",
                "json_schema": schema,
            }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._sanitize_response_format_name(),
                "schema": schema,
                "strict": True,
            },
        }

    def _sanitize_response_format_name(self) -> str:
        """Generate a valid response format name from the agent configuration."""

        base_name = self.agent_config.name or "response_format"
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name)
        return safe_name or "response_format"

    def _load_response_format(self) -> dict[str, Any] | None:
        """Load response format schema from inline config or file path."""

        response_format = self.agent_config.response_format

        if response_format is None:
            return None

        if isinstance(response_format, dict):
            return SchemaValidator.validate_schema(response_format, "response_format")

        path = Path(response_format)
        return SchemaValidator.load_schema_from_file(path.as_posix())
