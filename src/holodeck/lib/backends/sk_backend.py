"""Semantic Kernel backend implementation for HoloDeck agent execution.

Provides SKBackend (AgentBackend) and SKSession (AgentSession) that wrap
the existing AgentFactory / AgentThreadRun infrastructure behind the
provider-agnostic backend interfaces.
"""

from collections.abc import AsyncGenerator
from typing import Any

from semantic_kernel.contents import ChatHistory

from holodeck.lib.backends.base import AgentSession, ExecutionResult
from holodeck.lib.test_runner.agent_factory import AgentFactory
from holodeck.models.agent import Agent
from holodeck.models.token_usage import TokenUsage


def _extract_response(history: ChatHistory) -> str:
    """Extract the last assistant message content from a ChatHistory.

    Args:
        history: Semantic Kernel ChatHistory object.

    Returns:
        Content of the last assistant message, or empty string if not found.
    """
    if not history or not hasattr(history, "messages") or not history.messages:
        return ""

    for message in reversed(history.messages):
        if (
            hasattr(message, "role")
            and message.role == "assistant"
            and hasattr(message, "content")
        ):
            content = message.content
            return str(content) if content else ""
    return ""


class SKSession:
    """Stateful multi-turn session backed by an AgentThreadRun.

    Implements the AgentSession protocol by delegating to the underlying
    Semantic Kernel thread run for conversation management.
    """

    def __init__(self, thread_run: Any) -> None:
        """Initialize session with an AgentThreadRun.

        Args:
            thread_run: An AgentThreadRun instance from AgentFactory.
        """
        self._thread_run = thread_run

    async def send(self, message: str) -> ExecutionResult:
        """Send a message and receive a single-turn result.

        Args:
            message: The user message to send to the agent.

        Returns:
            ExecutionResult containing the agent response and metadata.
        """
        result = await self._thread_run.invoke(message)
        response = _extract_response(result.chat_history)
        token_usage = result.token_usage if result.token_usage else TokenUsage.zero()
        return ExecutionResult(
            response=response,
            tool_calls=result.tool_calls,
            tool_results=result.tool_results,
            token_usage=token_usage,
        )

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the response.

        Currently delegates to send() and yields the full response as a
        single chunk. True streaming will be implemented in a future phase.

        Args:
            message: The user message to send to the agent.

        Yields:
            String chunks of the agent response.
        """
        result = await self.send(message)
        yield result.response

    async def close(self) -> None:
        """Release session resources. No-op for SK sessions."""
        pass


class SKBackend:
    """Semantic Kernel backend implementing the AgentBackend protocol.

    Wraps AgentFactory to provide the provider-agnostic backend interface
    used by downstream consumers.
    """

    def __init__(self, agent_config: Agent) -> None:
        """Initialize the SK backend with agent configuration.

        Args:
            agent_config: Agent configuration with model and instructions.
        """
        self._factory = AgentFactory(agent_config=agent_config)

    async def initialize(self) -> None:
        """Prepare the backend for use by initializing tools."""
        await self._factory._ensure_tools_initialized()

    async def invoke_once(
        self,
        message: str,
        context: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute a single stateless agent turn.

        Args:
            message: The user message to send to the agent.
            context: Optional list of prior conversation turns (unused for now).

        Returns:
            ExecutionResult containing the agent response and metadata.
        """
        thread_run = await self._factory.create_thread_run()
        result = await thread_run.invoke(message)
        response = _extract_response(result.chat_history)
        token_usage = result.token_usage if result.token_usage else TokenUsage.zero()
        return ExecutionResult(
            response=response,
            tool_calls=result.tool_calls,
            tool_results=result.tool_results,
            token_usage=token_usage,
        )

    async def create_session(self) -> AgentSession:
        """Create a new stateful multi-turn session.

        Returns:
            An SKSession instance bound to a fresh thread run.
        """
        thread_run = await self._factory.create_thread_run()
        return SKSession(thread_run=thread_run)

    async def teardown(self) -> None:
        """Release all backend resources."""
        await self._factory.shutdown()
