"""Agent execution orchestrator for interactive chat."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from semantic_kernel.contents import ChatHistory

from holodeck.lib.logging_config import get_logger
from holodeck.lib.test_runner.agent_factory import AgentFactory, AgentThreadRun
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """Response from agent execution.

    Contains the agent's text response, any tool executions performed,
    token usage tracking, and execution timing information.
    """

    content: str
    tool_executions: list[ToolExecution]
    tokens_used: TokenUsage | None
    execution_time: float


class AgentExecutor:
    """Coordinates agent execution for chat sessions.

    Wraps AgentFactory to provide a clean interface for executing
    user messages and managing conversation history.
    """

    def __init__(
        self,
        agent_config: Agent,
        enable_observability: bool = False,
        timeout: float | None = 60.0,
        max_retries: int = 3,
        on_execution_start: Callable[[str], None] | None = None,
        on_execution_complete: Callable[[AgentResponse], None] | None = None,
        force_ingest: bool = False,
    ) -> None:
        """Initialize executor with agent configuration.

        Args:
            agent_config: Agent configuration with model and instructions.
            enable_observability: Enable OpenTelemetry tracing (TODO: Phase 5).
            timeout: Timeout for agent invocation in seconds.
            max_retries: Maximum retry attempts for transient failures.
            on_execution_start: Optional callback before agent execution.
            on_execution_complete: Optional callback after agent execution.
            force_ingest: Force re-ingestion of vector store source files.

        Raises:
            RuntimeError: If agent factory initialization fails.
        """
        self.agent_config = agent_config
        self._observability_enabled = enable_observability
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete

        try:
            # Create ExecutionConfig from timeout parameter for AgentFactory
            execution_config = ExecutionConfig(
                llm_timeout=int(timeout) if timeout else 60
            )
            self._factory = AgentFactory(
                agent_config=agent_config,
                max_retries=max_retries,
                force_ingest=force_ingest,
                execution_config=execution_config,
            )
            # Thread run is lazily initialized on first execute_turn()
            self._thread_run: AgentThreadRun | None = None
            logger.info(f"AgentExecutor initialized for agent: {agent_config.name}")
        except Exception as e:
            logger.error(f"Failed to initialize AgentExecutor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize agent: {e}") from e

    async def execute_turn(self, message: str) -> AgentResponse:
        """Execute a single turn of agent conversation.

        Sends a user message to the agent, captures the response,
        extracts tool calls, and tracks token usage.

        Args:
            message: User message to send to the agent.

        Returns:
            AgentResponse with content, tool executions, tokens, and timing.

        Raises:
            RuntimeError: If agent execution fails.
        """
        start_time = time.time()

        try:
            logger.debug(f"Executing turn for agent: {self.agent_config.name}")

            # Call pre-execution callback
            if self.on_execution_start:
                self.on_execution_start(message)

            # Lazy initialize thread run (preserves conversation history across turns)
            if self._thread_run is None:
                self._thread_run = await self._factory.create_thread_run()

            # Invoke agent using the persistent thread run
            result = await self._thread_run.invoke(message)
            elapsed = time.time() - start_time

            # Extract content from chat history
            content = self._extract_content(result.chat_history)

            # Convert tool calls to ToolExecution models
            tool_executions = self._convert_tool_calls(result.tool_calls)

            # Extract token usage from factory result
            tokens_used = result.token_usage

            logger.debug(
                f"Turn executed successfully: content={len(content)} chars, "
                f"tools={len(tool_executions)}, time={elapsed:.2f}s"
            )

            response = AgentResponse(
                content=content,
                tool_executions=tool_executions,
                tokens_used=tokens_used,
                execution_time=elapsed,
            )

            # Call post-execution callback
            if self.on_execution_complete:
                self.on_execution_complete(response)

            return response

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Agent execution failed: {e}") from e

    def get_history(self) -> ChatHistory:
        """Get current conversation history.

        Returns:
            Current ChatHistory from the thread run, or empty if not initialized.
        """
        if self._thread_run is not None:
            return self._thread_run.chat_history
        return ChatHistory()

    def clear_history(self) -> None:
        """Clear conversation history by discarding current thread run.

        Resets the agent's chat history to start fresh conversation.
        The next execute_turn() will create a new thread run with fresh history.
        """
        logger.debug("Clearing chat history by discarding thread run")
        self._thread_run = None

    async def shutdown(self) -> None:
        """Cleanup executor resources.

        Called when ending a chat session to release any held resources.
        Must be called from the same task context where the executor was used
        to properly cleanup MCP plugins.
        """
        try:
            logger.debug("AgentExecutor shutting down")
            # Shutdown the underlying factory (cleans up MCP plugins, vectorstores)
            await self._factory.shutdown()
            logger.debug("AgentExecutor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _extract_content(self, history: ChatHistory) -> str:
        """Extract last assistant message content from history.

        Args:
            history: ChatHistory to extract from.

        Returns:
            Content of the last assistant message, or empty string.
        """
        try:
            # ChatHistory messages are in order, get the last one
            if hasattr(history, "messages") and history.messages:
                # Get last message from history
                last_message = history.messages[-1]
                if hasattr(last_message, "content"):
                    content = last_message.content
                    return str(content) if content else ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract content from history: {e}")
            return ""

    def _convert_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[ToolExecution]:
        """Convert tool call dicts to ToolExecution models.

        Args:
            tool_calls: List of tool call dicts from AgentFactory.

        Returns:
            List of ToolExecution models.
        """
        executions: list[ToolExecution] = []
        try:
            for tool_call in tool_calls:
                execution = ToolExecution(
                    tool_name=tool_call.get("name", "unknown"),
                    parameters=tool_call.get("arguments", {}),
                    status=ToolStatus.SUCCESS,  # Assume success if it was executed
                )
                executions.append(execution)
            return executions
        except Exception as e:
            logger.warning(f"Failed to convert tool calls: {e}")
            return []
