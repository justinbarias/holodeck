"""Agent execution orchestrator for interactive chat."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
)
from holodeck.lib.backends.selector import BackendSelector
from holodeck.lib.logging_config import get_logger
from holodeck.models.agent import Agent
from holodeck.models.token_usage import TokenUsage
from holodeck.models.tool_execution import ToolExecution, ToolStatus

logger = get_logger(__name__)

_SENTINEL = object()


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


# ---------------------------------------------------------------------------
# _TaskBoundSession — actor wrapper for cross-task session safety
# ---------------------------------------------------------------------------


class _TaskBoundSession:
    """Wraps an AgentSession to run all calls in a dedicated background task.

    The Claude SDK's anyio task group binds to the async task that called
    ``connect()``.  HTTP servers spawn a new task per request, so the SDK
    client created in request N cannot be reused in request N+1.

    This wrapper starts a long-lived background task that owns the session.
    ``send()`` and ``send_streaming()`` delegate to that task via an
    ``asyncio.Queue``, keeping the SDK client in a single consistent task
    context while allowing callers from any task.
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._queue: asyncio.Queue[
            tuple[str, asyncio.Future[Any], asyncio.Queue[Any] | None] | None
        ] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background actor task."""
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        """Process messages sequentially in this task's context."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            message, future, chunk_queue = item
            try:
                if chunk_queue is not None:
                    # Streaming mode — push chunks to the caller's queue
                    try:
                        async for chunk in self._session.send_streaming(message):
                            await chunk_queue.put(chunk)
                        await chunk_queue.put(_SENTINEL)
                    except Exception as e:
                        await chunk_queue.put(e)
                    if not future.done():
                        future.set_result(None)
                else:
                    result = await self._session.send(message)
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def send(self, message: str) -> ExecutionResult:
        """Send a message via the actor task and await the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExecutionResult] = loop.create_future()
        await self._queue.put((message, future, None))
        return await future

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Stream a message via the actor task, yielding chunks."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        await self._queue.put((message, future, chunk_queue))
        while True:
            item = await chunk_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        """Shut down the actor and close the underlying session."""
        if self._task is not None and not self._task.done():
            await self._queue.put(None)
            await self._task
        await self._session.close()


class AgentExecutor:
    """Coordinates agent execution for chat sessions.

    Uses the provider-agnostic AgentBackend/AgentSession abstractions
    to execute user messages and manage conversation history.
    """

    def __init__(
        self,
        agent_config: Agent,
        backend: AgentBackend | None = None,
        on_execution_start: Callable[[str], None] | None = None,
        on_execution_complete: Callable[[AgentResponse], None] | None = None,
        release_transport_after_turn: bool = False,
    ) -> None:
        """Initialize executor with agent configuration.

        No I/O is performed during construction — backend and session
        are lazily created on the first ``execute_turn()`` call.

        Args:
            agent_config: Agent configuration with model and instructions.
            backend: Optional pre-initialized backend (bypasses BackendSelector).
            on_execution_start: Optional callback before agent execution.
            on_execution_complete: Optional callback after agent execution.
            release_transport_after_turn: If True, wrap the backend session
                in a ``_TaskBoundSession`` actor so the SDK client stays in
                a single background task.  Required for HTTP servers where
                each request runs in a different async task context.
        """
        self.agent_config = agent_config
        self._backend: AgentBackend | None = backend
        self._session: AgentSession | None = None
        self._history: list[dict[str, Any]] = []
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete
        self._use_task_bound_session = release_transport_after_turn

        logger.info(f"AgentExecutor initialized for agent: {agent_config.name}")

    async def _ensure_backend_and_session(self) -> None:
        """Lazily initialize backend and session on first use.

        If no backend was injected via the constructor, uses
        ``BackendSelector.select()`` to auto-select one based on the
        agent's LLM provider.

        Raises:
            BackendInitError: If backend selection or initialization fails.
            BackendSessionError: If session creation fails.
        """
        if self._session is not None:
            return
        if self._backend is None:
            self._backend = await BackendSelector.select(
                self.agent_config,
                tool_instances=None,
                mode="chat",
                allow_side_effects=False,
            )
        session = await self._backend.create_session()

        if self._use_task_bound_session:
            actor = _TaskBoundSession(session)
            await actor.start()
            self._session = actor
        else:
            self._session = session

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

            # Lazy initialize backend and session
            await self._ensure_backend_and_session()

            # Invoke agent via backend session
            result: ExecutionResult = await self._session.send(message)  # type: ignore[union-attr]
            elapsed = time.time() - start_time

            # Extract content from execution result
            content = result.response

            # Convert tool calls to ToolExecution models
            tool_executions = self._convert_tool_calls(result.tool_calls)

            # Extract token usage (always a TokenUsage, never None)
            tokens_used = result.token_usage

            logger.debug(
                f"Turn executed successfully: content={len(content)} chars, "
                f"tools={len(tool_executions)}, time={elapsed:.2f}s"
            )

            # Track history
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": content})

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

        except BackendSessionError:
            raise
        except BackendInitError as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Agent execution failed: {e}") from e
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Agent execution failed: {e}") from e

    async def execute_turn_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Stream agent response token by token.

        Args:
            message: User message to send to the agent.

        Yields:
            Successive string chunks of the agent response.

        Raises:
            RuntimeError: If agent execution fails.
        """
        try:
            await self._ensure_backend_and_session()
            collected: list[str] = []
            async for chunk in self._session.send_streaming(message):  # type: ignore[union-attr]
                collected.append(chunk)
                yield chunk
            # Update history after stream completes
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": "".join(collected)})
        except BackendSessionError:
            raise
        except BackendInitError as e:
            raise RuntimeError(f"Agent streaming failed: {e}") from e

    def get_history(self) -> list[dict[str, Any]]:
        """Get current conversation history.

        Returns:
            Serialized conversation history as a list of dicts, or empty list.
        """
        return list(self._history)

    async def clear_history(self) -> None:
        """Clear conversation history and close the current session.

        Resets the agent's chat history to start fresh conversation.
        The next ``execute_turn()`` will create a new session.
        """
        logger.debug("Clearing chat history and closing session")
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._history = []

    async def shutdown(self) -> None:
        """Cleanup executor resources.

        Called when ending a chat session to release any held resources.
        Closes the session and tears down the backend.
        """
        try:
            logger.debug("AgentExecutor shutting down")
            if self._session is not None:
                await self._session.close()
                self._session = None
            if self._backend is not None:
                await self._backend.teardown()
                self._backend = None
            logger.debug("AgentExecutor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _convert_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[ToolExecution]:
        """Convert tool call dicts to ToolExecution models.

        Args:
            tool_calls: List of tool call dicts from backend execution.

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
