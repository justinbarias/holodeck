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
    ToolEvent,
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
        self._ready: asyncio.Event = asyncio.Event()
        self._startup_error: BaseException | None = None

    async def start(self) -> None:
        """Start the actor and block until it has bound the SDK to its task.

        The actor's first job is to rebind the underlying session's
        transport (the SDK's anyio task group + ``_read_messages``
        background task) to this actor task. We block here until that's
        done so the caller can safely enqueue a ``send`` afterwards.
        """
        self._task = asyncio.create_task(self._loop())
        await self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    async def _loop(self) -> None:
        """Process messages sequentially in this task's context.

        The inner ``self._session.send(...)`` MUST be awaited directly in
        this task, not wrapped in a child task — the Claude SDK's anyio
        task group is bound to the task that called ``connect()``, and any
        subsequent call from a different task can deadlock when reading
        the shared anyio memory stream.

        Startup: if the inner session exposes ``_ensure_client`` (i.e.
        it's a ``ClaudeSession`` constructed via ``eager_connect=False``),
        call it here so the SDK's ``connect()`` runs in this task. That
        binds the SDK's anyio task group + ``_read_messages`` background
        reader to the actor — they live as long as the actor does, not
        as long as the HTTP request task that created the session.
        Without this, turn 2 hangs forever in ``receive_response()``
        while the CLI subprocess writes to a stdout nobody is reading,
        because the reader task was cancelled when turn 1's request
        task ended. SK / non-Claude sessions skip this cleanly.
        """
        ensure = getattr(self._session, "_ensure_client", None)
        if callable(ensure):
            try:
                logger.debug(
                    "[trace] _TaskBoundSession._loop: connecting inner session "
                    "in actor task"
                )
                await ensure()
            except BaseException as exc:
                self._startup_error = exc
                self._ready.set()
                return
        self._ready.set()

        turn_no = 0
        logger.debug("[trace] _TaskBoundSession._loop: started")
        while True:
            logger.debug(
                "[trace] _TaskBoundSession._loop: awaiting next queue item "
                "(completed=%d)",
                turn_no,
            )
            item = await self._queue.get()
            if item is None:
                logger.debug("[trace] _TaskBoundSession._loop: sentinel — exit")
                break
            turn_no += 1
            message, future, chunk_queue = item
            logger.debug(
                "[trace] _TaskBoundSession._loop turn=%d: dequeued, "
                "future_cancelled=%s, streaming=%s",
                turn_no,
                future.cancelled(),
                chunk_queue is not None,
            )

            # Caller already gave up (e.g. ``wait_for`` timed out) before we
            # picked this item up — don't burn an SDK turn whose result no
            # one will read. Otherwise the next request would queue behind
            # an abandoned turn and inherit its latency.
            if future.cancelled():
                logger.debug(
                    "[trace] _TaskBoundSession._loop turn=%d: skipping "
                    "(future already cancelled)",
                    turn_no,
                )
                continue

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
                    logger.debug(
                        "[trace] _TaskBoundSession._loop turn=%d: calling "
                        "inner send",
                        turn_no,
                    )
                    result = await self._session.send(message)
                    logger.debug(
                        "[trace] _TaskBoundSession._loop turn=%d: inner send "
                        "returned (future_done=%s)",
                        turn_no,
                        future.done(),
                    )
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                logger.debug(
                    "[trace] _TaskBoundSession._loop turn=%d: exception %s: %s",
                    turn_no,
                    type(e).__name__,
                    e,
                )
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

    @property
    def tool_events(self) -> asyncio.Queue[ToolEvent] | None:
        """Pass through the inner session's tool event queue."""
        return getattr(self._session, "tool_events", None)

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
        llm_timeout: int | float | None = None,
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
            llm_timeout: Per-turn LLM invocation timeout in seconds. When
                set, ``session.send`` / ``session.send_streaming`` are
                wrapped with ``asyncio.wait_for(timeout=llm_timeout)``.
                Mirrors the pattern used by ``TestExecutor``.
        """
        self.agent_config = agent_config
        self._backend: AgentBackend | None = backend
        self._session: AgentSession | None = None
        self._history: list[dict[str, Any]] = []
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete
        self._use_task_bound_session = release_transport_after_turn
        self._llm_timeout = float(llm_timeout) if llm_timeout else None

        logger.info(f"AgentExecutor initialized for agent: {agent_config.name}")

    @property
    def tool_event_queue(self) -> asyncio.Queue[ToolEvent] | None:
        """The tool event queue from the underlying session, if available."""
        if self._session is not None:
            return getattr(self._session, "tool_events", None)
        return None

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

        if self._use_task_bound_session:
            # Ask the backend NOT to eagerly connect — the actor task
            # must be the one that calls ``connect()`` so the SDK's
            # anyio task group binds to it and survives across HTTP
            # request tasks. Fall back to plain ``create_session()`` for
            # backends that don't expose the kwarg (only Claude needs it).
            try:
                session = await self._backend.create_session(  # type: ignore[call-arg]
                    eager_connect=False,
                )
            except TypeError:
                session = await self._backend.create_session()
            actor = _TaskBoundSession(session)
            await actor.start()
            self._session = actor
        else:
            session = await self._backend.create_session()
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

            # Invoke agent via backend session (timeout-wrapped when configured)
            if self._llm_timeout is not None:
                result: ExecutionResult = await asyncio.wait_for(
                    self._session.send(message),  # type: ignore[union-attr]
                    timeout=self._llm_timeout,
                )
            else:
                result = await self._session.send(message)  # type: ignore[union-attr]
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
        except asyncio.TimeoutError:
            # llm_timeout exceeded — let the caller distinguish timeout
            # from crash instead of wrapping it as a generic RuntimeError.
            logger.warning(
                "Agent invocation exceeded llm_timeout=%ss",
                self._llm_timeout,
            )
            raise
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
            stream = self._session.send_streaming(message)  # type: ignore[union-attr]
            # Enforce llm_timeout as a deadline across the whole stream —
            # each chunk must arrive within the remaining budget; otherwise
            # the next __anext__ is cancelled and TimeoutError propagates.
            deadline = (
                time.monotonic() + self._llm_timeout
                if self._llm_timeout is not None
                else None
            )
            while True:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError(
                            f"streaming response exceeded llm_timeout="
                            f"{self._llm_timeout}s"
                        )
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                else:
                    try:
                        chunk = await stream.__anext__()
                    except StopAsyncIteration:
                        break
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
