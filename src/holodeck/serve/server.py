"""Agent Local Server implementation.

Provides the FastAPI application factory and server lifecycle management
for exposing agents via HTTP.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from holodeck.lib.logging_config import get_logger
from holodeck.serve.middleware import ErrorHandlingMiddleware, LoggingMiddleware
from holodeck.serve.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProtocolType,
    ServerState,
)
from holodeck.serve.session_store import SessionStore

if TYPE_CHECKING:
    from holodeck.models.agent import Agent
    from holodeck.models.config import ExecutionConfig

logger = get_logger(__name__)


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

        self.sessions = SessionStore()
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

        from holodeck.chat.executor import AgentExecutor
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
            session = self.sessions.get(session_id)

            if session is None:
                # Create new executor for this session
                # Use thread_id as session_id for AG-UI correlation
                # Pass timeout from execution_config if available
                timeout = (
                    float(self.execution_config.llm_timeout)
                    if self.execution_config and self.execution_config.llm_timeout
                    else None
                )
                logger.debug(
                    f"Creating AgentExecutor for session {session_id} "
                    f"with timeout={timeout}s"
                )
                executor = AgentExecutor(self.agent_config, timeout=timeout)
                session = self.sessions.create(executor, session_id=session_id)

            # Touch session to update last activity
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
        from holodeck.chat.executor import AgentExecutor
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
            # Get or create session
            session_id = request.session_id
            session = self.sessions.get(session_id) if session_id else None

            if session is None:
                # Create new executor for this session
                timeout = (
                    float(self.execution_config.llm_timeout)
                    if self.execution_config and self.execution_config.llm_timeout
                    else None
                )
                logger.debug(
                    f"Creating AgentExecutor for new session with timeout={timeout}s"
                )
                executor = AgentExecutor(self.agent_config, timeout=timeout)
                session = self.sessions.create(executor, session_id=session_id)

            # Touch session to update last activity
            self.sessions.touch(session.session_id)
            session.message_count += 1

            # Handle request synchronously
            protocol = RESTProtocol()
            return await protocol.handle_sync_request(request, session)

        @app.post(
            f"/agent/{agent_name}/chat/stream",
            tags=["Chat"],
        )
        async def chat_stream(request: ChatRequest) -> StreamingResponse:
            """Streaming chat endpoint with SSE.

            Accepts a message, processes it through the agent, and streams
            SSE events back to the client.
            """
            # Get or create session
            session_id = request.session_id
            session = self.sessions.get(session_id) if session_id else None

            if session is None:
                # Create new executor for this session
                timeout = (
                    float(self.execution_config.llm_timeout)
                    if self.execution_config and self.execution_config.llm_timeout
                    else None
                )
                logger.debug(
                    f"Creating AgentExecutor for new session with timeout={timeout}s"
                )
                executor = AgentExecutor(self.agent_config, timeout=timeout)
                session = self.sessions.create(executor, session_id=session_id)

            # Touch session to update last activity
            self.sessions.touch(session.session_id)
            session.message_count += 1

            # Handle request with streaming
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

            # Get or create session
            session = self.sessions.get(session_id) if session_id else None

            if session is None:
                timeout = (
                    float(self.execution_config.llm_timeout)
                    if self.execution_config and self.execution_config.llm_timeout
                    else None
                )
                logger.debug(
                    f"Creating AgentExecutor for new session with timeout={timeout}s"
                )
                executor = AgentExecutor(self.agent_config, timeout=timeout)
                session = self.sessions.create(executor, session_id=session_id)

            # Touch session to update last activity
            self.sessions.touch(session.session_id)
            session.message_count += 1

            # Handle request synchronously
            protocol = RESTProtocol()
            return await protocol.handle_sync_request(chat_request, session)

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

            # Get or create session
            session = self.sessions.get(session_id) if session_id else None

            if session is None:
                timeout = (
                    float(self.execution_config.llm_timeout)
                    if self.execution_config and self.execution_config.llm_timeout
                    else None
                )
                logger.debug(
                    f"Creating AgentExecutor for new session with timeout={timeout}s"
                )
                executor = AgentExecutor(self.agent_config, timeout=timeout)
                session = self.sessions.create(executor, session_id=session_id)

            # Touch session to update last activity
            self.sessions.touch(session.session_id)
            session.message_count += 1

            # Handle request with streaming
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
            self.sessions.delete(session_id)
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

        # Cleanup sessions
        session_count = self.sessions.active_count
        self.sessions.sessions.clear()

        self.state = ServerState.STOPPED

        logger.info(f"Agent server stopped. Cleaned up {session_count} sessions.")
