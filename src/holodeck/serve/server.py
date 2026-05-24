"""Agent Local Server implementation.

Provides the FastAPI application factory and server lifecycle management
for exposing agents via HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from holodeck.lib.backends import BackendInitError, BackendSessionError
from holodeck.lib.backends.validators import validate_credentials, validate_nodejs
from holodeck.lib.errors import ConfigError
from holodeck.lib.logging_config import get_logger
from holodeck.lib.runtime import (
    derived_session_cap_from_memory,
    memory_limit_bytes,
)
from holodeck.models.llm import ProviderEnum
from holodeck.serve.middleware import ErrorHandlingMiddleware, LoggingMiddleware
from holodeck.serve.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProtocolType,
    ServerState,
)
from holodeck.serve.session_store import ServerSession, SessionStore
from holodeck.serve.tool_init_manager import ToolInitManager

if TYPE_CHECKING:
    from holodeck.chat.executor import AgentExecutor
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

logger = get_logger(__name__)

# Error message constants for contract-compliant responses
_BACKEND_ERROR_MSG = (
    "Claude Agent SDK subprocess terminated unexpectedly. "
    "Start a new session to retry."
)
_CAPACITY_MSG_TEMPLATE = (
    "Maximum concurrent active turns ({max}) reached. "
    "Retry after in-flight turns complete."
)
_CAPACITY_RETRY_AFTER_SECONDS = 5

# Fallback active-turn cap when neither an explicit
# claude.max_concurrent_sessions nor a cgroup memory limit is available
# (local dev, unconstrained Linux). Keep this conservative: this number is
# only reached when we can't introspect the runtime at all.
_FALLBACK_ACTIVE_TURN_CAP = 50

# Open-session ceiling. Under spec 034 P4, idle sessions cost ~30 MiB each
# (a Python object + JSONL transcript reference, no persistent subprocess),
# so the real binding constraint is concurrent active turns — not open
# session count. This cap exists only to bound disk usage from JSONL
# transcripts and to prevent unbounded dict growth.
_OPEN_SESSION_CEILING = 1000


class AgentServer:
    """HTTP server for exposing a single HoloDeck agent.

    The AgentServer wraps a FastAPI application and manages the server
    lifecycle, including session management and protocol handling.

    Attributes:
        agent_config: The agent configuration to serve.
        protocol: The protocol to use (AG-UI or REST).
        host: The hostname to bind to.
        port: The port to listen on.
        sessions: The session store for managing conversations.
        state: The current server state.
    """

    def __init__(
        self,
        agent_config: Agent,
        protocol: ProtocolType = ProtocolType.AG_UI,
        host: str = "127.0.0.1",
        port: int = 8000,
        cors_origins: list[str] | None = None,
        debug: bool = False,
        execution_config: ExecutionConfig | None = None,
        observability_enabled: bool = False,
        max_concurrent_init_jobs: int = 4,
    ) -> None:
        """Initialize the agent server.

        Args:
            agent_config: The agent configuration to serve.
            protocol: The protocol to use (default: AG-UI).
            host: The hostname to bind to (default: 127.0.0.1 for security).
                  Use 0.0.0.0 to expose to all network interfaces.
            port: The port to listen on (default: 8000).
            cors_origins: List of allowed CORS origins (default: ["*"]).
            debug: Enable debug logging (default: False).
            execution_config: Resolved execution configuration for timeouts.
            observability_enabled: Enable OpenTelemetry per-request tracing.
        """
        self.agent_config = agent_config
        self.protocol = protocol
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self.debug = debug
        self.execution_config = execution_config
        self.observability_enabled = observability_enabled

        # Warn if binding to all interfaces
        if host == "0.0.0.0":  # noqa: S104  # nosec B104
            logger.warning(
                "Server binding to 0.0.0.0 exposes it to all network interfaces. "
                "Use 127.0.0.1 for local-only access."
            )

        # Claude SDK's anyio task group binds to the task that called
        # connect(). HTTP requests run in different tasks, so the
        # executor must release transport after each turn and reconnect
        # on the next request (session_id preserves conversation state).
        self._release_transport = (
            self.agent_config.model.provider == ProviderEnum.ANTHROPIC
        )

        # Spec 034 P4 changed the binding constraint: the SDK subprocess
        # is now spawned per *turn* (not per session), so memory-bounded
        # capacity must gate concurrent active turns rather than open
        # session count. We derive the active-turn cap from the cgroup
        # memory limit using the same formula previously used for the
        # session cap — only the semantic shifted.
        self._active_turn_cap: int | None = None
        self._active_turn_semaphore: asyncio.BoundedSemaphore | None = None
        if self.agent_config.model.provider == ProviderEnum.ANTHROPIC:
            claude = self.agent_config.claude
            explicit_cap = claude.max_concurrent_sessions if claude else None
            per_turn_mib = claude.session_memory_estimate_mib if claude else 200
            per_turn_bytes = per_turn_mib * 1024 * 1024
            mem_bytes = memory_limit_bytes()
            logger.info(
                "Claude cgroup memory limit: %s",
                f"{mem_bytes} bytes" if mem_bytes is not None else "unbounded",
            )
            if explicit_cap is not None:
                self._active_turn_cap = explicit_cap
                logger.info(
                    "Claude active-turn cap: %d (explicit from "
                    "claude.max_concurrent_sessions)",
                    self._active_turn_cap,
                )
            elif mem_bytes is not None:
                self._active_turn_cap = derived_session_cap_from_memory(
                    mem_bytes, per_session_bytes=per_turn_bytes
                )
                logger.info(
                    "Claude active-turn cap: %d (derived from %d MiB memory "
                    "limit @ %d MiB/turn)",
                    self._active_turn_cap,
                    mem_bytes // (1024 * 1024),
                    per_turn_mib,
                )
            else:
                self._active_turn_cap = _FALLBACK_ACTIVE_TURN_CAP
                logger.info(
                    "Claude active-turn cap: %d (fallback — no cgroup memory "
                    "limit detected)",
                    self._active_turn_cap,
                )
            self._active_turn_semaphore = asyncio.BoundedSemaphore(
                self._active_turn_cap
            )
        self.sessions = SessionStore(max_sessions=_OPEN_SESSION_CEILING)
        self._tool_init_manager = ToolInitManager(
            agent=self.agent_config, max_concurrent=max_concurrent_init_jobs
        )
        self.state = ServerState.INITIALIZING
        self._app: FastAPI | None = None
        self._start_time: datetime | None = None

    @property
    def is_ready(self) -> bool:
        """Check if the server is ready to accept requests."""
        return self.state in (ServerState.READY, ServerState.RUNNING)

    @property
    def uptime_seconds(self) -> float:
        """Return server uptime in seconds."""
        if self._start_time is None:
            return 0.0
        delta = datetime.now(timezone.utc) - self._start_time
        return delta.total_seconds()

    async def _validate_backend_prerequisites(self) -> None:
        """Validate backend-specific prerequisites before serving.

        For Anthropic provider, validates Node.js availability/version
        and API credentials. Non-Anthropic providers skip validation.

        Raises:
            BackendInitError: If prerequisites are not met.
        """
        if self.agent_config.model.provider != ProviderEnum.ANTHROPIC:
            return

        try:
            validate_nodejs()
            validate_credentials(self.agent_config.model)
        except ConfigError as e:
            raise BackendInitError(str(e)) from e

        logger.info(
            "Backend prerequisites validated for %s provider",
            self.agent_config.model.provider.value,
        )

    def _get_timeout(self) -> float | None:
        """Return the configured LLM timeout, or None."""
        if self.execution_config and self.execution_config.llm_timeout:
            return float(self.execution_config.llm_timeout)
        return None

    def _create_executor(self) -> AgentExecutor:
        """Create an AgentExecutor for a new session."""
        from holodeck.chat.executor import (  # noqa: N814
            AgentExecutor as _Executor,
        )

        return _Executor(
            self.agent_config,
            release_transport_after_turn=self._release_transport,
            llm_timeout=self._get_timeout(),
        )

    def _get_or_create_session(
        self,
        session_id: str | None,
    ) -> ServerSession | JSONResponse:
        """Look up an existing session or create a new one.

        Returns:
            A ``ServerSession`` on success, or a ``JSONResponse``
            (503 capacity exceeded) that the caller should return directly.
        """
        if session_id:
            existing = self.sessions.get(session_id)
            if existing is not None:
                return existing

        timeout = self._get_timeout()
        logger.debug(
            "Creating AgentExecutor for session %s (timeout=%s)",
            session_id,
            timeout,
        )

        executor = self._create_executor()
        try:
            return self.sessions.create(executor, session_id=session_id)
        except RuntimeError:
            return self._capacity_exceeded_response()

    def _capacity_exceeded_response(self) -> JSONResponse:
        """Build a 429 JSON response for active-turn capacity exhaustion.

        Returns HTTP 429 with ``Retry-After`` so clients (and any fronting
        load balancer) recognise the backpressure and retry against
        another replica. Spec 034 P1a switched this from 503 because most
        HTTP client libraries treat 429 as a transient retryable signal
        out of the box, whereas 503 often surfaces as a hard error.

        Spec 034 P4 repurposed the cap from "open sessions" to "concurrent
        active turns" — the field stays ``max_sessions`` in the response
        body for client/dashboard compatibility, but reflects the
        active-turn cap.
        """
        cap = self._active_turn_cap or 0
        in_flight = self._active_turn_in_flight()
        return JSONResponse(
            status_code=429,
            content={
                "error": "capacity_exceeded",
                "message": _CAPACITY_MSG_TEMPLATE.format(max=cap),
                "active_sessions": in_flight,
                "max_sessions": cap,
            },
            headers={"Retry-After": str(_CAPACITY_RETRY_AFTER_SECONDS)},
        )

    def _active_turn_in_flight(self) -> int:
        """Return the count of currently-acquired turn slots.

        Reads ``BoundedSemaphore`` internals (``_value``) since the public
        API doesn't expose in-flight count. Safe because asyncio
        primitives are single-threaded.
        """
        sem = self._active_turn_semaphore
        if sem is None:
            return 0
        return self._active_turn_cap - sem._value  # type: ignore[operator]

    def _turn_capacity_exceeded(self) -> bool:
        """Return True iff acquiring a turn slot would block right now."""
        sem = self._active_turn_semaphore
        return sem is not None and sem.locked()

    def _try_acquire_turn_slot(self) -> bool:
        """Atomic non-blocking acquire.

        Returns True if a slot was acquired (or no gate is configured),
        False if at capacity. Atomic because asyncio is single-threaded
        and this method contains no awaits — the check + decrement
        happen in one tick.

        Pairs with ``_release_turn_slot()``. Callers MUST release exactly
        once for each successful acquire.
        """
        sem = self._active_turn_semaphore
        if sem is None:
            return True
        if sem._value <= 0:
            return False
        sem._value -= 1
        return True

    def _release_turn_slot(self) -> None:
        """Release one acquired turn slot. No-op when no gate is configured."""
        sem = self._active_turn_semaphore
        if sem is not None:
            sem.release()

    @contextlib.asynccontextmanager
    async def _acquire_turn_slot(self) -> AsyncGenerator[bool, None]:
        """Try-acquire one active-turn slot as a context manager.

        Yields True if the slot was acquired (or no gate is configured),
        False if at capacity. Callers MUST inspect the yielded value and
        short-circuit (e.g. return 429) when False.
        """
        acquired = self._try_acquire_turn_slot()
        try:
            yield acquired
        finally:
            if acquired:
                self._release_turn_slot()

    async def _turn_gated_stream(
        self,
        gen: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[bytes, None]:
        """Wrap a streaming response generator so it releases a turn slot.

        Assumes the caller has already acquired a slot via
        ``_try_acquire_turn_slot()`` before constructing the response —
        once HTTP headers have shipped, we can no longer return 429, so
        the gate must run synchronously in the endpoint. This wrapper
        owns the release.
        """
        try:
            async for chunk in gen:
                yield chunk
        finally:
            self._release_turn_slot()

    async def _handle_backend_session_error(
        self,
        session_id: str,
    ) -> JSONResponse:
        """Clean up a broken session and return a 502 response."""
        await self.sessions.delete(session_id)
        return JSONResponse(
            status_code=502,
            content={
                "error": "backend_error",
                "message": _BACKEND_ERROR_MSG,
                "retriable": True,
            },
        )

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application.

        Returns:
            Configured FastAPI application instance.
        """
        agent_name = self.agent_config.name
        protocol_name = self.protocol.value
        app = FastAPI(
            title=f"HoloDeck Agent: {agent_name}",
            description=f"Agent Local Server exposing {agent_name} via {protocol_name}",
            version="0.1.0",
            docs_url="/docs" if self.protocol == ProtocolType.REST else None,
            redoc_url="/redoc" if self.protocol == ProtocolType.REST else None,
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add custom middleware
        # Middleware order matters: Starlette executes in reverse order of addition.
        # Request flow:  Logging -> ErrorHandling -> CORS -> Handler
        # Response flow: Handler -> CORS -> ErrorHandling -> Logging
        #
        # This order ensures:
        # 1. LoggingMiddleware logs all requests/responses including error responses
        # 2. ErrorHandlingMiddleware catches handler exceptions and returns RFC 7807
        # 3. CORS headers are added to all responses including errors
        app.add_middleware(ErrorHandlingMiddleware, debug=self.debug)
        app.add_middleware(
            LoggingMiddleware,
            debug=self.debug,
            observability_enabled=self.observability_enabled,
        )

        # Register health endpoints
        self._register_health_endpoints(app)

        # Register tool init endpoints (protocol-agnostic)
        from holodeck.serve.tool_init_routes import router as tool_init_router

        app.state.tool_init_manager = self._tool_init_manager
        app.include_router(tool_init_router)

        # Register protocol-specific endpoints
        if self.protocol == ProtocolType.AG_UI:
            self._register_agui_endpoints(app)
        elif self.protocol == ProtocolType.REST:
            self._register_rest_endpoints(app)

        # Store reference
        self._app = app
        self.state = ServerState.READY

        logger.info(
            f"FastAPI app created for agent '{self.agent_config.name}' "
            f"with {self.protocol.value} protocol"
        )

        return app

    def _register_health_endpoints(self, app: FastAPI) -> None:
        """Register health check endpoints.

        Args:
            app: The FastAPI application.
        """

        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health() -> HealthResponse:
            """Basic health check endpoint."""
            return HealthResponse(
                status="healthy" if self.is_ready else "unhealthy",
                agent_name=self.agent_config.name,
                agent_ready=self.is_ready,
                active_sessions=self.sessions.active_count,
                uptime_seconds=self.uptime_seconds,
            )

        @app.get("/health/agent", response_model=HealthResponse, tags=["Health"])
        async def health_agent() -> HealthResponse:
            """Agent-specific health check endpoint."""
            return HealthResponse(
                status="healthy" if self.is_ready else "unhealthy",
                agent_name=self.agent_config.name,
                agent_ready=self.is_ready,
                active_sessions=self.sessions.active_count,
                uptime_seconds=self.uptime_seconds,
            )

        @app.get("/ready", tags=["Health"])
        async def ready() -> dict[str, bool]:
            """Readiness check endpoint for orchestrators."""
            return {"ready": self.is_ready}

    def _register_agui_endpoints(self, app: FastAPI) -> None:
        """Register AG-UI protocol endpoints.

        Args:
            app: The FastAPI application.
        """
        from ag_ui.core.events import RunAgentInput

        from holodeck.serve.protocols.agui import AGUIProtocol

        @app.post("/awp", tags=["AG-UI"])
        async def agui_endpoint(
            request: Request,
        ) -> StreamingResponse:
            """AG-UI protocol endpoint for agent interaction.

            Accepts RunAgentInput and streams AG-UI events back to the client.
            """
            from fastapi import HTTPException
            from pydantic import ValidationError

            # Parse request body manually to avoid FastAPI schema issues
            try:
                body = await request.json()
                input_data = RunAgentInput(**body)
            except ValidationError as e:
                raise HTTPException(status_code=422, detail=e.errors()) from e

            # Capacity exceeded surfaces as a real HTTP 429 (not an SSE
            # error frame inside a 200) so fronting load balancers and
            # client retry logic see the backpressure (spec 034 P1a).
            # Spec 034 P4: the cap is now concurrent active turns, not
            # open sessions. Acquire synchronously before constructing
            # the response — once headers ship, we can't switch to 429.
            # _turn_gated_stream owns the matching release.
            if not self._try_acquire_turn_slot():
                return self._capacity_exceeded_response()  # type: ignore[return-value]

            session_id = input_data.thread_id
            result = self._get_or_create_session(session_id)
            if isinstance(result, JSONResponse):
                self._release_turn_slot()
                return result  # type: ignore[return-value]
            session = result

            self.sessions.touch(session_id)
            session.message_count += 1

            # Create protocol with accept header for format negotiation
            accept_header = request.headers.get("accept")
            protocol = AGUIProtocol(
                accept_header=accept_header,
                execution_config=self.execution_config,
            )

            return StreamingResponse(
                self._turn_gated_stream(protocol.handle_request(input_data, session)),
                media_type=protocol.content_type,
            )

    def _register_rest_endpoints(self, app: FastAPI) -> None:
        """Register REST protocol endpoints.

        Endpoints:
        - POST /agent/{agent_name}/chat - Synchronous chat
        - POST /agent/{agent_name}/chat/stream - Streaming chat (SSE)
        - DELETE /sessions/{session_id} - Delete session

        Args:
            app: The FastAPI application.
        """
        from holodeck.serve.protocols.rest import RESTProtocol

        agent_name = self.agent_config.name

        @app.post(
            f"/agent/{agent_name}/chat",
            response_model=ChatResponse,
            tags=["Chat"],
        )
        async def chat_sync(request: ChatRequest) -> ChatResponse | JSONResponse:
            """Synchronous chat endpoint.

            Accepts a message, processes it through the agent, and returns
            the complete response as JSON.
            """
            async with self._acquire_turn_slot() as acquired:
                if not acquired:
                    return self._capacity_exceeded_response()

                result = self._get_or_create_session(request.session_id)
                if isinstance(result, JSONResponse):
                    return result
                session = result

                self.sessions.touch(session.session_id)
                session.message_count += 1

                protocol = RESTProtocol()
                try:
                    return await protocol.handle_sync_request(request, session)
                except BackendSessionError:
                    return await self._handle_backend_session_error(
                        session.session_id,
                    )

        @app.post(
            f"/agent/{agent_name}/chat/stream",
            tags=["Chat"],
        )
        async def chat_stream(request: ChatRequest) -> StreamingResponse:
            """Streaming chat endpoint with SSE.

            Accepts a message, processes it through the agent, and streams
            SSE events back to the client.
            """
            if not self._try_acquire_turn_slot():
                return self._capacity_exceeded_response()  # type: ignore[return-value]

            result = self._get_or_create_session(request.session_id)
            if isinstance(result, JSONResponse):
                self._release_turn_slot()
                return result  # type: ignore[return-value]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            return StreamingResponse(
                self._turn_gated_stream(protocol.handle_request(request, session)),
                media_type=protocol.content_type,
            )

        # Multipart form-data endpoints
        @app.post(
            f"/agent/{agent_name}/chat/multipart",
            response_model=ChatResponse,
            tags=["Chat"],
        )
        async def chat_sync_multipart(
            message: str = Form(..., min_length=1, max_length=10000),
            session_id: str | None = Form(default=None),
            files: list[UploadFile] = File(default=[]),  # noqa: B008
        ) -> ChatResponse | JSONResponse:
            """Synchronous chat endpoint with multipart file upload.

            Accepts a message and optional files via multipart form-data,
            processes them through the agent, and returns the complete
            response as JSON.
            """
            from holodeck.serve.protocols.rest import process_multipart_files

            # Validate file count
            if len(files) > 10:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum 10 files allowed per request",
                )

            # Convert multipart files to FileContent
            try:
                file_contents = await process_multipart_files(files)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            # Create ChatRequest with files
            chat_request = ChatRequest(
                message=message,
                session_id=session_id,
                files=file_contents if file_contents else None,
            )

            async with self._acquire_turn_slot() as acquired:
                if not acquired:
                    return self._capacity_exceeded_response()

                result = self._get_or_create_session(session_id)
                if isinstance(result, JSONResponse):
                    return result
                session = result

                self.sessions.touch(session.session_id)
                session.message_count += 1

                protocol = RESTProtocol()
                try:
                    return await protocol.handle_sync_request(chat_request, session)
                except BackendSessionError:
                    return await self._handle_backend_session_error(
                        session.session_id,
                    )

        @app.post(
            f"/agent/{agent_name}/chat/stream/multipart",
            tags=["Chat"],
        )
        async def chat_stream_multipart(
            message: str = Form(..., min_length=1, max_length=10000),
            session_id: str | None = Form(default=None),
            files: list[UploadFile] = File(default=[]),  # noqa: B008
        ) -> StreamingResponse:
            """Streaming chat endpoint with multipart file upload.

            Accepts a message and optional files via multipart form-data,
            processes them through the agent, and streams SSE events back
            to the client.
            """
            from holodeck.serve.protocols.rest import process_multipart_files

            # Validate file count
            if len(files) > 10:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum 10 files allowed per request",
                )

            # Convert multipart files to FileContent
            try:
                file_contents = await process_multipart_files(files)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            # Create ChatRequest with files
            chat_request = ChatRequest(
                message=message,
                session_id=session_id,
                files=file_contents if file_contents else None,
            )

            if not self._try_acquire_turn_slot():
                return self._capacity_exceeded_response()  # type: ignore[return-value]

            result = self._get_or_create_session(session_id)
            if isinstance(result, JSONResponse):
                self._release_turn_slot()
                return result  # type: ignore[return-value]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            return StreamingResponse(
                self._turn_gated_stream(protocol.handle_request(chat_request, session)),
                media_type=protocol.content_type,
            )

        @app.delete(
            "/sessions/{session_id}",
            status_code=204,
            tags=["Sessions"],
        )
        async def delete_session(session_id: str) -> Response:
            """Delete a session.

            Removes the session and its conversation history.
            Returns 204 No Content on success (idempotent).
            """
            await self.sessions.delete(session_id)
            logger.debug(f"Deleted session: {session_id}")
            return Response(status_code=204)

    async def start(self) -> None:
        """Start the server and begin accepting requests.

        This method should be called after create_app() to transition
        the server to the RUNNING state. Also starts the background
        session cleanup task.
        """
        if self._app is None:
            self.create_app()

        # Validate backend prerequisites before accepting requests
        await self._validate_backend_prerequisites()

        # Clean up orphaned temp directories from previous runs
        from holodeck.lib.source_resolver import SourceResolver

        orphans_removed = await SourceResolver.cleanup_orphans()
        if orphans_removed:
            logger.info(
                "Cleaned up %d orphaned init temp directories",
                orphans_removed,
            )

        self._start_time = datetime.now(timezone.utc)
        self.state = ServerState.RUNNING

        # Start automatic session cleanup
        await self.sessions.start_cleanup_task()

        logger.info(
            f"Agent server started at http://{self.host}:{self.port} "
            f"serving agent '{self.agent_config.name}'"
        )

    async def stop(self) -> None:
        """Stop the server gracefully.

        Transitions through SHUTTING_DOWN to STOPPED state,
        stopping the cleanup task and clearing all sessions.
        """
        self.state = ServerState.SHUTTING_DOWN

        # Shutdown tool init manager (cancel running jobs)
        await self._tool_init_manager.shutdown()

        # Stop cleanup task
        await self.sessions.stop_cleanup_task()

        # Shutdown all session executors (stops actor tasks, SDK subprocesses)
        session_count = self.sessions.active_count
        session_ids = list(self.sessions.sessions.keys())
        if session_ids:
            await asyncio.gather(
                *(self.sessions.delete(sid) for sid in session_ids),
                return_exceptions=True,
            )

        self.state = ServerState.STOPPED

        logger.info(f"Agent server stopped. Cleaned up {session_count} sessions.")
