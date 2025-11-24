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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.agents import Agent as SKAgent
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from holodeck.lib.logging_config import get_logger
from holodeck.lib.logging_utils import log_retry
from holodeck.config.schema import SchemaValidator
from holodeck.models.agent import Agent
from holodeck.models.llm import ProviderEnum
from holodeck.models.token_usage import TokenUsage

# Try to import Anthropic support (optional dependency)
try:
    from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
except ImportError:
    AnthropicChatCompletion = None  # type: ignore[misc,assignment]

logger = get_logger(__name__)


@dataclass
class AgentExecutionResult:
    """Result of agent execution containing tool calls and conversation history.

    Attributes:
        tool_calls: List of tool calls made by the agent during execution
        chat_history: Complete conversation history including user inputs
            and agent responses
        token_usage: Token usage metadata if provided by LLM provider
    """

    tool_calls: list[dict[str, Any]]
    chat_history: ChatHistory
    token_usage: TokenUsage | None = None


class AgentFactoryError(Exception):
    """Error raised during agent bridge operations."""

    pass


class AgentFactory:
    """Factory for creating and executing agents using Semantic Kernel.

    Handles Kernel creation, agent invocation, response extraction,
    and tool call handling with support for multiple LLM providers.
    """

    def __init__(
        self,
        agent_config: Agent,
        timeout: float | None = 60.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        retry_exponential_base: float = 2.0,
    ) -> None:
        """Initialize agent factory with Semantic Kernel.

        Args:
            agent_config: Agent configuration with model and instructions
            timeout: Timeout for agent invocation in seconds (None for no timeout)
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Base delay in seconds for exponential backoff
            retry_exponential_base: Exponential base for backoff calculation

        Raises:
            AgentFactoryError: If kernel initialization fails
        """
        self.agent_config = agent_config
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential_base = retry_exponential_base
        self._retry_count = 0
        self.kernel_arguments: KernelArguments | None = None

        logger.debug(
            f"Initializing AgentFactory: agent={agent_config.name}, "
            f"provider={agent_config.model.provider}, timeout={timeout}s, "
            f"max_retries={max_retries}"
        )

        try:
            self.kernel = self._create_kernel()
            self.agent = self._create_agent()
            self.chat_history = self._create_chat_history()
            logger.info(
                f"AgentFactory initialized successfully for agent: {agent_config.name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent factory: {e}", exc_info=True)
            raise AgentFactoryError(f"Failed to initialize agent factory: {e}") from e

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
            else:
                raise AgentFactoryError(
                    f"Unsupported LLM provider: {model_config.provider}"
                )

            kernel.add_service(service)
            logger.debug("Kernel created and service added successfully")
            return kernel

        except Exception as e:
            logger.error(f"Kernel creation failed: {e}", exc_info=True)
            raise AgentFactoryError(f"Kernel creation failed: {e}") from e

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
            agent = ChatCompletionAgent(
                name=self.agent_config.name,
                description=self.agent_config.description,
                kernel=self.kernel,
                instructions=instructions,
                arguments=self._build_kernel_arguments(),
            )

            # Capture kernel arguments for invocation reuse
            self.kernel_arguments = agent.arguments

            return agent

        except Exception as e:
            raise AgentFactoryError(f"Agent creation failed: {e}") from e

    def _load_instructions(self) -> str:
        """Load agent instructions from config.

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

                file_path = Path(instructions.file)
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Instructions file not found: {instructions.file}"
                    )
                return file_path.read_text()
            else:
                raise AgentFactoryError("No instructions provided (file or inline)")

        except Exception as e:
            raise AgentFactoryError(f"Failed to load instructions: {e}") from e

    def _create_chat_history(self, user_input: str | None = None) -> ChatHistory:
        """Create ChatHistory with system instructions and optional user message.

        Args:
            user_input: User's input message (optional)

        Returns:
            Populated ChatHistory instance
        """
        history = ChatHistory()
        # Add user input if provided
        if user_input:
            history.add_user_message(user_input)

        return history

    async def invoke(self, user_input: str) -> AgentExecutionResult:
        """Invoke agent with timeout and retry logic.

        Args:
            user_input: User's input message

        Returns:
            AgentExecutionResult with tool_calls and complete chat_history

        Raises:
            AgentFactoryError: If invocation fails after retries
        """
        try:
            # Add user input to chat history
            self.chat_history.add_user_message(user_input)

            # Invoke with timeout and retry logic
            if self.timeout:
                result = await asyncio.wait_for(
                    self._invoke_with_retry(), timeout=self.timeout
                )
            else:
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
            AgentExecutionResult with tool_calls and complete chat_history

        Raises:
            AgentFactoryError: If all retries are exhausted
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
            AgentExecutionResult with tool_calls and complete chat_history
        """
        response_text = ""
        tool_calls: list[dict[str, Any]] = []
        token_usage: TokenUsage | None = None
        try:
            # Invoke agent with chat history
            thread = ChatHistoryAgentThread(self.chat_history)
            async for (
                response
            ) in self.agent.invoke(  # pyright: ignore[reportUnknownMemberType]
                thread=thread,
                arguments=self.kernel_arguments,
            ):
                # Extract response content
                response_text = self._extract_response_content(response)

                # Extract tool calls
                tool_calls = self._extract_tool_calls(response)

                # Extract token usage
                token_usage = self._extract_token_usage(response)
                break  # Only process first response

            # Add agent's response to chat history
            if response_text:
                self.chat_history.add_assistant_message(response_text)

            return AgentExecutionResult(
                tool_calls=tool_calls,
                chat_history=self.chat_history,
                token_usage=token_usage,
            )

        except Exception as e:
            raise AgentFactoryError(f"Agent execution failed: {e}") from e

    def _build_kernel_arguments(self) -> KernelArguments:
        """Create kernel arguments with execution settings from configuration."""

        settings = self._create_prompt_execution_settings()
        return KernelArguments(settings=settings)

    def _create_prompt_execution_settings(self) -> PromptExecutionSettings:
        """Build provider-specific prompt execution settings."""

        if not hasattr(self, "kernel"):
            raise AgentFactoryError("Kernel must be initialized before settings.")

        if not self.kernel.services:
            raise AgentFactoryError("No LLM services configured on the kernel.")

        # Use the first configured service (single-service kernel)
        service = next(iter(self.kernel.services.values()))
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

        if hasattr(settings, "max_tokens") and model_config.max_tokens is not None:
            settings.max_tokens = model_config.max_tokens

        if (
            hasattr(settings, "max_completion_tokens")
            and model_config.max_tokens is not None
        ):
            settings.max_completion_tokens = model_config.max_tokens

        if hasattr(settings, "ai_model_id"):
            settings.ai_model_id = model_config.name

    def _apply_response_format(self, settings: PromptExecutionSettings) -> None:
        """Attach response format schema to execution settings if provided."""

        try:
            response_format = self._load_response_format()
        except Exception as e:
            raise AgentFactoryError(f"Failed to load response_format: {e}") from e

        if response_format is not None and hasattr(settings, "response_format"):
            settings.response_format = response_format

    def _load_response_format(self) -> dict[str, Any] | None:
        """Load response format schema from inline config or file path."""

        response_format = self.agent_config.response_format

        if response_format is None:
            return None

        if isinstance(response_format, dict):
            return SchemaValidator.validate_schema(
                response_format, "response_format"
            )

        path = Path(response_format)
        return SchemaValidator.load_schema_from_file(path.as_posix())

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from agent response.

        Args:
            response: Response object from agent invocation

        Returns:
            Extracted response text, or empty string if no content
        """
        try:
            if hasattr(response, "content"):
                content = response.content
                return str(content) if content else ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return ""

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from agent response.

        Args:
            response: Response object from agent invocation

        Returns:
            List of tool call dictionaries with 'name' and 'arguments' keys
        """
        tool_calls: list[dict[str, Any]] = []

        try:
            # Try to extract from direct tool_calls attribute
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_calls.append(
                        {
                            "name": getattr(tool_call, "name", None),
                            "arguments": getattr(tool_call, "arguments", {}),
                        }
                    )

            # Try to extract from messages
            if hasattr(response, "messages") and response.messages:
                for message in response.messages:
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_calls.append(
                                {
                                    "name": getattr(tool_call, "name", None),
                                    "arguments": getattr(tool_call, "arguments", {}),
                                }
                            )

        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")

        return tool_calls

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
