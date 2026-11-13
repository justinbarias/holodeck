"""Agent execution orchestrator for interactive chat."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

from holodeck.lib.backends.base import (
    AgentBackend,
    AgentSession,
    BackendBudgetExceededError,
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

if TYPE_CHECKING:
    from ag_ui.core import BaseEvent, RunAgentInput

logger = get_logger(__name__)

_SENTINEL = object()

_T = TypeVar("_T")

# One item of an ordered turn stream: a text delta, a tool event, or the
# completed turn's result (always last).
_TurnItem = str | ToolEvent | ExecutionResult


@runtime_checkable
class _AGUISession(Protocol):
    """Session capability for Claude-native AG-UI event streaming."""

    def send_agui(
        self,
        input_data: RunAgentInput,
        message_override: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream AG-UI events for one run."""
        ...


def _require_agui_session(session: AgentSession | None) -> _AGUISession:
    """Return *session* as AG-UI-capable or raise a clear runtime error."""
    if session is None or not isinstance(session, _AGUISession):
        raise AttributeError("Backend session does not support AG-UI streaming")
    return cast(_AGUISession, session)


@runtime_checkable
class _EventStreamSession(Protocol):
    """Session capability: one turn as ordered text deltas and tool events.

    Implemented by the OpenAI Agents session so the AG-UI bridge can emit tool
    cards and text live and correctly interleaved, without the bounded
    ``tool_events`` side queue.
    """

    def send_events(self, message: str) -> AsyncGenerator[_TurnItem, None]:
        """Stream text deltas and ``ToolEvent`` records in run order.

        The last item is the turn's ``ExecutionResult`` (final output, tool
        calls, token usage), built as the non-streaming ``send`` builds it.
        """
        ...


def _require_event_session(session: AgentSession | None) -> _EventStreamSession:
    """Return *session* as event-stream-capable or raise a clear error."""
    if session is None or not isinstance(session, _EventStreamSession):
        raise AttributeError("Backend session does not support event streaming")
    return cast(_EventStreamSession, session)


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
    thinking: str = ""


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

    def __init__(
        self,
        session: AgentSession | None = None,
        *,
        session_factory: Callable[[], Awaitable[AgentSession]] | None = None,
        teardown: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Wrap an existing session, or build one inside the actor task.

        Args:
            session: An already-constructed session. Its ``prepare()`` runs
                in the actor task.
            session_factory: Alternative to ``session``: an async factory
                awaited **inside the actor task** before ``prepare()``.
                Use it when constructing the session (or the backend behind
                it) binds task-local resources — the OpenAI Agents backend
                connects stdio MCP servers during ``initialize()``, and the
                anyio streams die with the task that opened them.
            teardown: Optional coroutine run inside the actor task after the
                session closes (for example the backend's ``teardown()``,
                whose MCP cleanup must exit its cancel scopes in the same
                task that entered them).

        Raises:
            ValueError: If neither or both of ``session``/``session_factory``
                are given.
        """
        if (session is None) == (session_factory is None):
            raise ValueError("pass exactly one of session or session_factory")
        self._session: AgentSession | None = session
        self._session_factory = session_factory
        self._teardown = teardown
        # (message, future, chunk_queue, events). ``chunk_queue`` is None for
        # ``send``; ``events`` selects ``send_events`` over ``send_streaming``.
        self._queue: asyncio.Queue[
            tuple[str, asyncio.Future[Any], asyncio.Queue[Any] | None, bool] | None
        ] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event = asyncio.Event()
        self._startup_error: BaseException | None = None
        self._closed: bool = False
        self._current: asyncio.Future[Any] | None = None

    def _require_session(self) -> AgentSession:
        if self._session is None:
            raise RuntimeError("_TaskBoundSession used before start()")
        return self._session

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

        Startup: invoke ``self._session.prepare()`` so any lazy connect
        runs in this task. For ``ClaudeSession`` (constructed with
        ``eager_connect=False``) this calls the SDK's ``connect()``,
        which binds the SDK's anyio task group + ``_read_messages``
        background reader to the actor — they live as long as the actor
        does, not the HTTP request task that created the session.
        Without this, turn 2 hangs forever in ``receive_response()``
        while the CLI subprocess writes to a stdout nobody is reading,
        because the reader task was cancelled when turn 1's request
        task ended. Sessions whose ``prepare()`` is a no-op (SK and the
        default protocol stub) skip the connect step cleanly.
        """
        try:
            if self._session is None and self._session_factory is not None:
                logger.debug(
                    "[trace] _TaskBoundSession._loop: building inner session "
                    "in actor task"
                )
                self._session = await self._session_factory()
            logger.debug(
                "[trace] _TaskBoundSession._loop: preparing inner session "
                "in actor task"
            )
            await self._require_session().prepare()
        except BaseException as exc:
            # The factory may already have allocated a backend (MCP servers,
            # SDK subprocess) before ``create_session``/``prepare`` failed.
            # Release it here, in the task that opened it, before reporting.
            await self._cleanup_failed_start()
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        session = self._require_session()

        turn_no = 0
        logger.debug("[trace] _TaskBoundSession._loop: started")
        try:
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
                await self._run_turn(session, turn_no, item)
        except asyncio.CancelledError:
            # ``close()`` was abandoned (for example the serve session store's
            # shutdown timeout) and cancelled us mid-turn. Fail the turn and
            # anything still queued so no caller waits forever, then still
            # close in this task via ``finally``.
            logger.debug("[trace] _TaskBoundSession._loop: cancelled")
            self._fail_pending(
                BackendSessionError("Agent session closed while a turn was in flight")
            )
            raise
        finally:
            await self._close_in_task(session)

    async def _run_turn(
        self,
        session: AgentSession,
        turn_no: int,
        item: tuple[str, asyncio.Future[Any], asyncio.Queue[Any] | None, bool],
    ) -> None:
        """Execute one queued turn inside the actor task."""
        message, future, chunk_queue, events = item
        self._current = future
        try:
            await self._execute_item(
                session, turn_no, message, future, chunk_queue, events=events
            )
        except asyncio.CancelledError:
            # The actor was cancelled mid-turn (see ``close``): the caller of
            # this turn must not wait forever.
            error = BackendSessionError(
                "Agent session closed while a turn was in flight"
            )
            if not future.done():
                future.set_exception(error)
            if chunk_queue is not None:
                chunk_queue.put_nowait(error)
            raise
        finally:
            self._current = None

    async def _execute_item(
        self,
        session: AgentSession,
        turn_no: int,
        message: str,
        future: asyncio.Future[Any],
        chunk_queue: asyncio.Queue[Any] | None,
        *,
        events: bool = False,
    ) -> None:
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
            return

        try:
            if chunk_queue is not None:
                # Streaming mode — push chunks (text, or text and tool events
                # in run order) to the caller's queue
                try:
                    if events:
                        await self._stream_events_turn(
                            session, turn_no, message, future, chunk_queue
                        )
                    else:
                        async for chunk in session.send_streaming(message):
                            await chunk_queue.put(chunk)
                        await chunk_queue.put(_SENTINEL)
                except Exception as e:
                    await chunk_queue.put(e)
                if not future.done():
                    future.set_result(None)
            else:
                logger.debug(
                    "[trace] _TaskBoundSession._loop turn=%d: calling " "inner send",
                    turn_no,
                )
                result = await session.send(message)
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

    async def _stream_events_turn(
        self,
        session: AgentSession,
        turn_no: int,
        message: str,
        future: asyncio.Future[Any],
        chunk_queue: asyncio.Queue[Any],
    ) -> None:
        """Run one event-stream turn; stop it if the caller abandons it.

        The caller cancels *future* when it stops reading (``llm_timeout``,
        client disconnect). That reschedules a timeout scope to "now", which
        cancels the inner stream in this task: the SDK cancels its run, and the
        actor stays alive for the next turn. A cancel scope from
        ``asyncio.timeout`` (not ``wait_for``) keeps the stream in this task,
        so cancel scopes the SDK entered here are exited here.
        """
        loop = asyncio.get_running_loop()
        scope: asyncio.Timeout | None = None
        # The callback runs via ``call_soon``, so it can fire after the turn
        # finished and the scope exited (removing it does not unschedule a
        # pending call); a finished ``Timeout`` cannot be rescheduled.
        in_scope = False

        def _on_abandoned(fut: asyncio.Future[Any]) -> None:
            if fut.cancelled() and in_scope and scope is not None:
                scope.reschedule(loop.time())

        future.add_done_callback(_on_abandoned)
        try:
            async with asyncio.timeout(None) as scope:
                in_scope = True
                async with contextlib.aclosing(
                    _require_event_session(session).send_events(message)
                ) as stream:
                    async for item in stream:
                        await chunk_queue.put(item)
                await chunk_queue.put(_SENTINEL)
        except TimeoutError:
            if scope is None or not scope.expired():
                raise
            logger.debug(
                "[trace] _TaskBoundSession._loop turn=%d: caller abandoned the "
                "turn; stream cancelled",
                turn_no,
            )
        finally:
            in_scope = False
            future.remove_done_callback(_on_abandoned)

    async def prepare(self) -> None:
        """No-op. The inner session is prepared during ``start()``."""
        return None

    async def send(self, message: str) -> ExecutionResult:
        """Send a message via the actor task and await the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExecutionResult] = loop.create_future()
        await self._queue.put((message, future, None, False))
        return await future

    async def send_streaming(self, message: str) -> AsyncGenerator[str, None]:
        """Stream a message via the actor task, yielding chunks."""
        async with contextlib.aclosing(
            self._stream_via_actor(message, events=False)
        ) as items:
            async for item in items:
                if isinstance(item, str):
                    yield item

    async def send_events(self, message: str) -> AsyncGenerator[_TurnItem, None]:
        """Stream text deltas and tool events via the actor task, in run order.

        Raises:
            AttributeError: If the wrapped session lacks ``send_events``
                (check ``supports_events`` first).
        """
        async with contextlib.aclosing(
            self._stream_via_actor(message, events=True)
        ) as items:
            async for item in items:
                yield item

    @property
    def supports_events(self) -> bool:
        """Whether the wrapped session implements ``send_events``."""
        return isinstance(self._session, _EventStreamSession)

    async def _stream_via_actor(
        self, message: str, *, events: bool
    ) -> AsyncGenerator[_TurnItem, None]:
        """Queue a streaming turn on the actor and yield what it produces.

        If the caller stops early (``llm_timeout``, client disconnect,
        ``aclose``), the turn's future is cancelled, as ``send`` does when its
        caller times out: a still-queued turn is skipped, and a running
        event-stream turn is cancelled (see ``_stream_events_turn``).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        await self._queue.put((message, future, chunk_queue, events))
        finished = False
        try:
            while True:
                item = await chunk_queue.get()
                if item is _SENTINEL:
                    finished = True
                    break
                if isinstance(item, Exception):
                    finished = True
                    raise item
                yield item
        finally:
            if not finished:
                future.cancel()

    async def send_agui(
        self,
        input_data: RunAgentInput,
        message_override: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream AG-UI events from the wrapped session when supported."""
        session = _require_agui_session(self._require_session())
        async for event in session.send_agui(input_data, message_override):
            yield event

    @property
    def tool_events(self) -> asyncio.Queue[ToolEvent] | None:
        """Pass through the inner session's tool event queue."""
        return getattr(self._session, "tool_events", None)

    async def _close_in_task(self, session: AgentSession) -> None:
        """Close the session and run the teardown hook inside the actor task."""
        if self._closed:
            return
        self._closed = True
        try:
            await session.close()
        finally:
            if self._teardown is not None:
                await self._teardown()

    async def _cleanup_failed_start(self) -> None:
        """Release whatever a failed start allocated, inside the actor task."""
        self._closed = True
        try:
            if self._session is not None:
                await self._session.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup after failure
            logger.debug("Session close after failed start raised", exc_info=True)
        finally:
            if self._teardown is not None:
                try:
                    await self._teardown()
                except Exception:  # noqa: BLE001 - best-effort cleanup after failure
                    logger.debug("Teardown after failed start raised", exc_info=True)

    def _fail_pending(self, error: BaseException) -> None:
        """Fail the in-flight turn and every queued turn with *error*."""
        current = self._current
        if current is not None and not current.done():
            current.set_exception(error)
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                continue
            _message, future, chunk_queue, _events = item
            if not future.done():
                future.set_exception(error)
            if chunk_queue is not None:
                chunk_queue.put_nowait(error)

    async def close(self) -> None:
        """Shut down the actor, closing the session and tearing down in-task.

        The close and teardown run inside the actor task (see
        ``_close_in_task``) so transports whose cancel scopes were entered
        there are exited there too. If the actor never started or already
        exited, the close runs in the caller's task as a fallback.

        If the caller stops waiting (``asyncio.wait_for`` timeout, request
        cancellation), the actor task is cancelled instead of being left
        with a stranded turn; its cancellation handler fails pending turns
        and still runs the close and teardown in the actor task.
        """
        task = self._task
        if task is not None and asyncio.current_task() is task:
            # Called from inside the actor (a hook or tool ending the
            # session). Close in place and let the loop exit after this turn.
            await self._close_in_task(self._require_session())
            self._queue.put_nowait(None)
            return
        if task is not None and not task.done():
            await self._queue.put(None)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                task.cancel()
                raise
            return
        if self._session is not None:
            await self._close_in_task(self._session)


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
        self._init_lock = asyncio.Lock()

        logger.info(f"AgentExecutor initialized for agent: {agent_config.name}")

    @property
    def tool_event_queue(self) -> asyncio.Queue[ToolEvent] | None:
        """The tool event queue from the underlying session, if available."""
        if self._session is not None:
            return getattr(self._session, "tool_events", None)
        return None

    @property
    def streams_turn_events(self) -> bool:
        """Whether ``execute_turn_events`` is available for the current session.

        ``False`` until the session exists (call ``_ensure_backend_and_session``
        first) or when the backend session lacks ``send_events``.
        """
        session = self._session
        if isinstance(session, _TaskBoundSession):
            return session.supports_events
        return isinstance(session, _EventStreamSession)

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
        async with self._init_lock:
            if self._session is None:
                await self._create_backend_and_session()

    async def _create_backend_and_session(self) -> None:
        """Create the backend and session; callers hold ``_init_lock``."""

        async def _select_backend() -> AgentBackend:
            if self._backend is None:
                self._backend = await BackendSelector.select(
                    self.agent_config,
                    tool_instances=None,
                    mode="chat",
                )
            return self._backend

        if self._use_task_bound_session:
            # Build the backend *and* the session inside the actor task, and
            # ask the backend NOT to eagerly connect. Anyio task-group state
            # (the Claude SDK's ``connect()``, the OpenAI Agents backend's
            # stdio MCP servers opened during ``initialize()``) binds to the
            # task that opened it and must survive across HTTP request
            # tasks; the actor invokes ``session.prepare()`` from inside its
            # own task as well.
            async def _build_session() -> AgentSession:
                backend = await _select_backend()
                return await backend.create_session(eager_connect=False)

            async def _teardown_backend() -> None:
                if self._backend is not None:
                    await self._backend.teardown()
                    self._backend = None

            actor = _TaskBoundSession(
                session_factory=_build_session, teardown=_teardown_backend
            )
            await actor.start()
            self._session = actor
        else:
            backend = await _select_backend()
            self._session = await backend.create_session()

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
            return self._complete_turn(message, result, time.time() - start_time)

        except BackendSessionError:
            raise
        except BackendInitError as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Agent execution failed: {e}") from e
        except TimeoutError:
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

    def _complete_turn(
        self, message: str, result: ExecutionResult, elapsed: float
    ) -> AgentResponse:
        """Record a finished turn in history and build its ``AgentResponse``.

        Shared by ``execute_turn`` and ``execute_turn_events`` so both record
        the same final answer and token usage.

        Raises:
            BackendSessionError: If the result is an error with no content.
        """
        content = result.response

        # Backends report runtime failures as ``is_error`` results rather
        # than raising (so multi-turn test runs can record them). A turn
        # with no content is a failed turn for chat/serve callers; a
        # partial response (budget cap, FR-032) is returned with the
        # reason logged so the caller still sees what the model produced.
        if result.is_error:
            reason = result.error_reason or "unknown backend error"
            if not content:
                raise BackendSessionError(f"Agent turn failed: {reason}")
            logger.warning(
                "Agent turn ended with error after partial response: %s", reason
            )

        tool_executions = self._convert_tool_calls(result.tool_calls)
        logger.debug(
            f"Turn executed successfully: content={len(content)} chars, "
            f"tools={len(tool_executions)}, time={elapsed:.2f}s"
        )

        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": content})

        response = AgentResponse(
            content=content,
            tool_executions=tool_executions,
            # Always a TokenUsage, never None.
            tokens_used=result.token_usage,
            execution_time=elapsed,
            thinking=result.thinking,
        )
        if self.on_execution_complete:
            self.on_execution_complete(response)
        return response

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
            async with contextlib.aclosing(
                self._with_deadline(
                    self._session.send_streaming(message)  # type: ignore[union-attr]
                )
            ) as chunks:
                async for chunk in chunks:
                    collected.append(chunk)
                    yield chunk
            # Update history after stream completes
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": "".join(collected)})
        except BackendSessionError:
            raise
        except BackendInitError as e:
            raise RuntimeError(f"Agent streaming failed: {e}") from e

    async def execute_turn_events(
        self, message: str
    ) -> AsyncGenerator[str | ToolEvent | AgentResponse, None]:
        """Stream one turn as text deltas and tool events, in run order.

        Requires ``streams_turn_events``. The final item is the turn's
        ``AgentResponse``, built and recorded in history exactly as
        ``execute_turn`` does (the run's final output, not the joined deltas,
        plus tool executions and token usage). Failures match
        ``execute_turn`` for serve callers: a budget trip with a partial
        response ends the turn with a warning; one with no response, or any
        backend failure, raises ``BackendSessionError``.

        Args:
            message: User message to send to the agent.

        Yields:
            ``str`` text deltas and ``ToolEvent`` records as the run produces
            them, then one ``AgentResponse``.

        Raises:
            BackendSessionError: If the turn fails.
            TimeoutError: If the turn exceeds ``llm_timeout``.
            RuntimeError: If backend initialization fails.
        """
        start_time = time.time()
        collected: list[str] = []
        result: ExecutionResult | None = None
        try:
            if self.on_execution_start:
                self.on_execution_start(message)
            await self._ensure_backend_and_session()
            session = self._session
            stream = (
                session.send_events(message)
                if isinstance(session, _TaskBoundSession)
                else _require_event_session(session).send_events(message)
            )
            async with contextlib.aclosing(self._with_deadline(stream)) as items:
                async for item in items:
                    if isinstance(item, ExecutionResult):
                        result = item
                        continue
                    if isinstance(item, str):
                        collected.append(item)
                    yield item
        except BackendBudgetExceededError as exc:
            # Mirrors the non-streaming ``send`` (FR-032): the partial
            # response becomes an error result that ``_complete_turn`` accepts
            # with a warning, or fails when nothing was produced.
            result = ExecutionResult(
                response=exc.partial_response or "".join(collected),
                is_error=True,
                error_reason=str(exc),
            )
        except (BackendSessionError, TimeoutError):
            raise
        except BackendInitError as e:
            raise RuntimeError(f"Agent execution failed: {e}") from e
        except Exception as e:
            # Backend runtime failures surface as BackendSessionError, as the
            # non-streaming ``send`` path does via ``is_error`` results.
            logger.warning("Agent turn failed: %s: %s", type(e).__name__, e)
            raise BackendSessionError(
                f"Agent turn failed: {type(e).__name__}: {e}"
            ) from e
        if result is None:
            # The session reported no final result; fall back to the text.
            result = ExecutionResult(response="".join(collected))
        yield self._complete_turn(message, result, time.time() - start_time)

    async def _with_deadline(
        self, stream: AsyncGenerator[_T, None]
    ) -> AsyncGenerator[_T, None]:
        """Yield from *stream*, raising ``TimeoutError`` past ``llm_timeout``.

        Enforces the timeout via a single deadline timer that cancels the
        current task. Per-chunk ``asyncio.wait_for`` would wrap each
        ``__anext__`` in a fresh task and break the SDK's anyio cancel scope
        ownership ("Attempted to exit cancel scope in a different task than it
        was entered in"). call_later runs on the current task, so the SDK's
        scope stays bound to a single task.

        Closing this generator disarms the timer first, then closes *stream*
        in the closing task, so no timer outlives the turn and cancels the
        caller's later work.
        """
        timed_out = False
        timer: asyncio.TimerHandle | None = None
        if self._llm_timeout is not None:
            loop = asyncio.get_running_loop()
            current = asyncio.current_task()

            def _on_deadline() -> None:
                nonlocal timed_out
                timed_out = True
                if current is not None:
                    current.cancel()

            timer = loop.call_later(self._llm_timeout, _on_deadline)
        try:
            async for item in stream:
                yield item
        except asyncio.CancelledError:
            if timed_out:
                raise TimeoutError(
                    f"streaming response exceeded llm_timeout={self._llm_timeout}s"
                ) from None
            raise
        finally:
            if timer is not None:
                timer.cancel()
            await stream.aclose()

    async def execute_turn_agui(
        self,
        input_data: RunAgentInput,
        message_override: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute an AG-UI turn through a backend session that supports it."""
        try:
            await self._ensure_backend_and_session()
            session = _require_agui_session(self._session)
            async for event in session.send_agui(input_data, message_override):
                yield event
        except BackendSessionError:
            raise
        except BackendInitError as e:
            raise RuntimeError(f"Agent AG-UI execution failed: {e}") from e

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
                # A task-bound actor tears the backend down inside its own
                # task (see ``_TaskBoundSession.close``) and clears
                # ``self._backend``; the plain path tears down here.
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
