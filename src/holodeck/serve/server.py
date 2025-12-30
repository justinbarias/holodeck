"""Agent Local Server implementation.

Provides the FastAPI application factory and server lifecycle management
for exposing agents via HTTP.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from holodeck.lib.logging_config import get_logger
from holodeck.serve.middleware import ErrorHandlingMiddleware, LoggingMiddleware
from holodeck.serve.models import HealthResponse, ProtocolType, ServerState
from holodeck.serve.session_store import SessionStore

if TYPE_CHECKING:
    from holodeck.models.agent import Agent

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
        """
        self.agent_config = agent_config
        self.protocol = protocol
        self.host = host
        self.port = port
        self.cors_origins = cors_origins or ["*"]
        self.debug = debug

        # Warn if binding to all interfaces
        if host == "0.0.0.0":  # noqa: S104
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
        app.add_middleware(LoggingMiddleware, debug=self.debug)

        # Register health endpoints
        self._register_health_endpoints(app)

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
