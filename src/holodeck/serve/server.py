"""Agent Local Server implementation.

Provides the FastAPI application factory and server lifecycle management
for exposing agents via HTTP.
"""

from __future__ import annotations

import asyncio
import json
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
    "Maximum concurrent Claude sessions ({max}) reached. "
    "Retry after existing sessions complete."
)


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

        # Determine max sessions based on provider
        max_sessions = 1000  # Default for non-Claude providers
        if self.agent_config.model.provider == ProviderEnum.ANTHROPIC:
            if self.agent_config.claude is not None:
                max_sessions = self.agent_config.claude.max_concurrent_sessions or 10
            else:
                max_sessions = 10  # Default when claude section absent
        self.sessions = SessionStore(max_sessions=max_sessions)
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
        """Build a 503 JSON response for capacity-exceeded errors."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "capacity_exceeded",
                "message": _CAPACITY_MSG_TEMPLATE.format(
                    max=self.sessions.max_sessions,
                ),
                "active_sessions": self.sessions.active_count,
                "max_sessions": self.sessions.max_sessions,
            },
            headers={"Retry-After": "5"},
        )

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

            # Get session by thread_id or create new one
            session_id = input_data.thread_id
            result = self._get_or_create_session(session_id)
            if isinstance(result, JSONResponse):
                # Convert capacity error to AG-UI SSE format
                cap = self.sessions.max_sessions

                async def capacity_error_stream() -> AsyncGenerator[bytes, None]:
                    payload = json.dumps(
                        {
                            "type": "capacity_exceeded",
                            "message": (
                                f"Maximum concurrent Claude sessions "
                                f"({cap}) reached."
                            ),
                        }
                    )
                    yield f"event: error\ndata: {payload}\n\n".encode()

                return StreamingResponse(
                    capacity_error_stream(),
                    media_type="text/event-stream",
                )
            session = result

            self.sessions.touch(session_id)
            session.message_count += 1

            # Create protocol with accept header for format negotiation
            accept_header = request.headers.get("accept")
            protocol = AGUIProtocol(accept_header=accept_header)

            # Stream response
            return StreamingResponse(
                protocol.handle_request(input_data, session),
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
        async def chat_sync(request: ChatRequest) -> ChatResponse:
            """Synchronous chat endpoint.

            Accepts a message, processes it through the agent, and returns
            the complete response as JSON.
            """
            result = self._get_or_create_session(request.session_id)
            if isinstance(result, JSONResponse):
                return result  # type: ignore[return-value,no-any-return]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            try:
                return await protocol.handle_sync_request(request, session)
            except BackendSessionError:
                return await self._handle_backend_session_error(  # type: ignore[return-value,no-any-return]
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
            result = self._get_or_create_session(request.session_id)
            if isinstance(result, JSONResponse):
                return result  # type: ignore[return-value,no-any-return]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            return StreamingResponse(
                protocol.handle_request(request, session),
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
        ) -> ChatResponse:
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

            result = self._get_or_create_session(session_id)
            if isinstance(result, JSONResponse):
                return result  # type: ignore[return-value,no-any-return]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            try:
                return await protocol.handle_sync_request(chat_request, session)
            except BackendSessionError:
                return await self._handle_backend_session_error(  # type: ignore[return-value,no-any-return]
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

            result = self._get_or_create_session(session_id)
            if isinstance(result, JSONResponse):
                return result  # type: ignore[return-value,no-any-return]
            session = result

            self.sessions.touch(session.session_id)
            session.message_count += 1

            protocol = RESTProtocol()
            return StreamingResponse(
                protocol.handle_request(chat_request, session),
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
