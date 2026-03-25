"""Unit tests for AgentServer class.

Tests cover server initialization, FastAPI app creation, health endpoints,
lifecycle management, and state transitions.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from holodeck.lib.backends import BackendInitError
from holodeck.lib.errors import ConfigError
from holodeck.models.config import ExecutionConfig
from holodeck.models.llm import ProviderEnum
from holodeck.serve.models import ProtocolType, ServerState
from holodeck.serve.server import AgentServer


@pytest.fixture
def mock_agent_config() -> MagicMock:
    """Create a mock agent configuration."""
    agent = MagicMock()
    agent.name = "test-agent"
    return agent


@pytest.fixture
def mock_execution_config() -> ExecutionConfig:
    """Create a mock execution configuration."""
    return ExecutionConfig(
        llm_timeout=120,
        file_timeout=60,
        download_timeout=60,
        cache_enabled=True,
        cache_dir=".holodeck/cache",
        verbose=False,
        quiet=False,
    )


class TestAgentServerInit:
    """Tests for AgentServer initialization."""

    def test_init_with_defaults(self, mock_agent_config: MagicMock) -> None:
        """Test AgentServer initializes with default values."""
        server = AgentServer(agent_config=mock_agent_config)

        assert server.agent_config is mock_agent_config
        assert server.protocol == ProtocolType.AG_UI
        assert server.host == "127.0.0.1"  # Default is localhost for security
        assert server.port == 8000
        assert server.cors_origins == ["*"]
        assert server.debug is False
        assert server.execution_config is None
        assert server.state == ServerState.INITIALIZING
        assert server._app is None
        assert server._start_time is None

    def test_init_with_custom_values(self, mock_agent_config: MagicMock) -> None:
        """Test AgentServer with custom configuration."""
        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.REST,
            host="127.0.0.1",
            port=9000,
            cors_origins=["https://example.com"],
            debug=True,
        )

        assert server.protocol == ProtocolType.REST
        assert server.host == "127.0.0.1"
        assert server.port == 9000
        assert server.cors_origins == ["https://example.com"]
        assert server.debug is True

    def test_init_with_execution_config(
        self,
        mock_agent_config: MagicMock,
        mock_execution_config: ExecutionConfig,
    ) -> None:
        """Test AgentServer with execution configuration."""
        server = AgentServer(
            agent_config=mock_agent_config,
            execution_config=mock_execution_config,
        )

        assert server.execution_config is mock_execution_config
        assert server.execution_config.llm_timeout == 120
        assert server.execution_config.file_timeout == 60

    def test_init_warns_on_all_interfaces_binding(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test AgentServer warns when binding to all interfaces."""
        from unittest.mock import patch

        with patch("holodeck.serve.server.logger") as mock_logger:
            AgentServer(
                agent_config=mock_agent_config,
                host="0.0.0.0",  # noqa: S104
            )
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "0.0.0.0" in warning_msg  # noqa: S104
            assert "all network interfaces" in warning_msg

    def test_init_creates_session_store(self, mock_agent_config: MagicMock) -> None:
        """Test AgentServer creates a session store."""
        server = AgentServer(agent_config=mock_agent_config)

        assert server.sessions is not None
        assert server.sessions.active_count == 0

    @pytest.mark.unit
    def test_session_cap_anthropic_with_max_concurrent_sessions(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T015: Anthropic provider uses claude.max_concurrent_sessions."""
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        mock_agent_config.claude = MagicMock()
        mock_agent_config.claude.max_concurrent_sessions = 5

        server = AgentServer(agent_config=mock_agent_config)

        assert server.sessions.max_sessions == 5

    @pytest.mark.unit
    def test_session_cap_non_anthropic_defaults_to_1000(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T015: Non-Anthropic provider defaults to 1000 max sessions."""
        mock_agent_config.model.provider = ProviderEnum.OPENAI

        server = AgentServer(agent_config=mock_agent_config)

        assert server.sessions.max_sessions == 1000

    @pytest.mark.unit
    def test_session_cap_anthropic_without_claude_config_defaults_to_10(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T015: Anthropic provider without claude config defaults to 10."""
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        mock_agent_config.claude = None

        server = AgentServer(agent_config=mock_agent_config)

        assert server.sessions.max_sessions == 10


class TestAgentServerProperties:
    """Tests for AgentServer properties."""

    def test_is_ready_when_initializing(self, mock_agent_config: MagicMock) -> None:
        """Test is_ready returns False when initializing."""
        server = AgentServer(agent_config=mock_agent_config)
        assert server.state == ServerState.INITIALIZING
        assert server.is_ready is False

    def test_is_ready_when_ready(self, mock_agent_config: MagicMock) -> None:
        """Test is_ready returns True when ready."""
        server = AgentServer(agent_config=mock_agent_config)
        server.state = ServerState.READY
        assert server.is_ready is True

    def test_is_ready_when_running(self, mock_agent_config: MagicMock) -> None:
        """Test is_ready returns True when running."""
        server = AgentServer(agent_config=mock_agent_config)
        server.state = ServerState.RUNNING
        assert server.is_ready is True

    def test_is_ready_when_shutting_down(self, mock_agent_config: MagicMock) -> None:
        """Test is_ready returns False when shutting down."""
        server = AgentServer(agent_config=mock_agent_config)
        server.state = ServerState.SHUTTING_DOWN
        assert server.is_ready is False

    def test_is_ready_when_stopped(self, mock_agent_config: MagicMock) -> None:
        """Test is_ready returns False when stopped."""
        server = AgentServer(agent_config=mock_agent_config)
        server.state = ServerState.STOPPED
        assert server.is_ready is False

    def test_uptime_seconds_before_start(self, mock_agent_config: MagicMock) -> None:
        """Test uptime_seconds returns 0 before server starts."""
        server = AgentServer(agent_config=mock_agent_config)
        assert server.uptime_seconds == 0.0

    def test_uptime_seconds_after_start(self, mock_agent_config: MagicMock) -> None:
        """Test uptime_seconds calculates correctly after start."""
        server = AgentServer(agent_config=mock_agent_config)
        server._start_time = datetime.now(timezone.utc) - timedelta(seconds=100)

        uptime = server.uptime_seconds
        assert 99 < uptime < 101  # Allow small variance


class TestAgentServerCreateApp:
    """Tests for AgentServer.create_app() method."""

    def test_create_app_returns_fastapi_instance(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test create_app returns a FastAPI instance."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        from fastapi import FastAPI

        assert isinstance(app, FastAPI)
        assert server._app is app

    def test_create_app_sets_state_to_ready(self, mock_agent_config: MagicMock) -> None:
        """Test create_app transitions state to READY."""
        server = AgentServer(agent_config=mock_agent_config)
        assert server.state == ServerState.INITIALIZING

        server.create_app()

        assert server.state == ServerState.READY

    def test_create_app_ag_ui_protocol_no_docs(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test create_app with AG-UI protocol disables OpenAPI docs."""
        server = AgentServer(
            agent_config=mock_agent_config, protocol=ProtocolType.AG_UI
        )
        app = server.create_app()

        assert app.docs_url is None
        assert app.redoc_url is None

    def test_create_app_rest_protocol_has_docs(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test create_app with REST protocol enables OpenAPI docs."""
        server = AgentServer(agent_config=mock_agent_config, protocol=ProtocolType.REST)
        app = server.create_app()

        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_create_app_title_includes_agent_name(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test create_app sets title with agent name."""
        mock_agent_config.name = "my-custom-agent"
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        assert "my-custom-agent" in app.title


class TestAgentServerHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def server_client(self, mock_agent_config: MagicMock) -> TestClient:
        """Create a test client for the server."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()
        return TestClient(app)

    def test_health_endpoint_returns_200(self, server_client: TestClient) -> None:
        """Test /health endpoint returns 200."""
        response = server_client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, server_client: TestClient) -> None:
        """Test /health endpoint returns proper JSON."""
        response = server_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "agent_name" in data
        assert "agent_ready" in data
        assert "active_sessions" in data
        assert "uptime_seconds" in data

    def test_health_endpoint_shows_agent_name(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test /health endpoint shows correct agent name."""
        mock_agent_config.name = "special-agent"
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()
        client = TestClient(app)

        response = client.get("/health")
        data = response.json()

        assert data["agent_name"] == "special-agent"

    def test_health_agent_endpoint_returns_200(self, server_client: TestClient) -> None:
        """Test /health/agent endpoint returns 200."""
        response = server_client.get("/health/agent")
        assert response.status_code == 200

    def test_ready_endpoint_returns_200(self, server_client: TestClient) -> None:
        """Test /ready endpoint returns 200."""
        response = server_client.get("/ready")
        assert response.status_code == 200

    def test_ready_endpoint_returns_ready_status(
        self, server_client: TestClient
    ) -> None:
        """Test /ready endpoint returns ready status."""
        response = server_client.get("/ready")
        data = response.json()

        assert "ready" in data
        assert isinstance(data["ready"], bool)

    def test_health_shows_active_sessions(self, mock_agent_config: MagicMock) -> None:
        """Test health endpoint shows active session count."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        # Add some mock sessions
        mock_executor = MagicMock()
        server.sessions.create(mock_executor)
        server.sessions.create(mock_executor)

        client = TestClient(app)
        response = client.get("/health")
        data = response.json()

        assert data["active_sessions"] == 2


class TestAgentServerLifecycle:
    """Tests for server lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_creates_app_if_needed(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test start() creates app if not already created."""
        server = AgentServer(agent_config=mock_agent_config)
        assert server._app is None

        await server.start()

        assert server._app is not None

    @pytest.mark.asyncio
    async def test_start_sets_state_to_running(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test start() transitions state to RUNNING."""
        server = AgentServer(agent_config=mock_agent_config)

        await server.start()

        assert server.state == ServerState.RUNNING

    @pytest.mark.asyncio
    async def test_start_sets_start_time(self, mock_agent_config: MagicMock) -> None:
        """Test start() sets the start time."""
        server = AgentServer(agent_config=mock_agent_config)
        assert server._start_time is None

        await server.start()

        assert server._start_time is not None
        assert server._start_time.tzinfo is not None

    @pytest.mark.asyncio
    async def test_start_preserves_existing_app(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test start() preserves existing app if already created."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        await server.start()

        assert server._app is app

    @pytest.mark.asyncio
    async def test_stop_transitions_through_shutting_down(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test stop() transitions through SHUTTING_DOWN to STOPPED."""
        server = AgentServer(agent_config=mock_agent_config)
        await server.start()

        await server.stop()

        assert server.state == ServerState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_cleans_up_sessions(self, mock_agent_config: MagicMock) -> None:
        """Test stop() cleans up active sessions."""
        server = AgentServer(agent_config=mock_agent_config)
        await server.start()

        # Add some sessions
        mock_executor = MagicMock()
        server.sessions.create(mock_executor)
        server.sessions.create(mock_executor)
        assert server.sessions.active_count == 2

        await server.stop()

        assert server.sessions.active_count == 0


class TestAgentServerIntegration:
    """Integration tests for AgentServer."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_agent_config: MagicMock) -> None:
        """Test complete server lifecycle with health checks."""
        server = AgentServer(agent_config=mock_agent_config)

        # Initially not ready
        assert not server.is_ready
        assert server.state == ServerState.INITIALIZING

        # Create app
        app = server.create_app()
        assert server.is_ready
        assert server.state == ServerState.READY

        # Test health endpoint before start
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Start server
        await server.start()
        assert server.state == ServerState.RUNNING
        assert server.uptime_seconds > 0

        # Health still works
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

        # Stop server
        await server.stop()
        assert server.state == ServerState.STOPPED
        assert not server.is_ready

    def test_cors_middleware_configured(self, mock_agent_config: MagicMock) -> None:
        """Test CORS middleware is properly configured."""
        server = AgentServer(
            agent_config=mock_agent_config,
            cors_origins=["https://example.com"],
        )
        app = server.create_app()
        client = TestClient(app)

        response = client.options(
            "/health",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200


class TestAgentServerAGUIEndpoint:
    """Tests for AG-UI endpoint validation.

    Note: Full endpoint integration tests are in
    tests/integration/serve/test_server_agui.py.
    These unit tests focus on validation and endpoint registration.
    """

    def test_agui_endpoint_rejects_invalid_input(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test /awp endpoint returns 422 for invalid input."""
        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.AG_UI,
        )
        app = server.create_app()
        client = TestClient(app)

        # Missing required fields
        response = client.post(
            "/awp",
            json={
                "messages": [],
            },
        )
        assert response.status_code == 422

    def test_agui_endpoint_exists_on_ag_ui_protocol(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test /awp endpoint is registered with AG-UI protocol."""
        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.AG_UI,
        )
        app = server.create_app()

        # Check endpoint is registered
        routes = [route.path for route in app.routes]
        assert "/awp" in routes

    def test_agui_endpoint_not_present_on_rest_protocol(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test /awp endpoint is NOT registered with REST protocol."""
        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.REST,
        )
        app = server.create_app()

        # Check endpoint is NOT registered
        routes = [route.path for route in app.routes]
        assert "/awp" not in routes


class TestToolInitManagerWiring:
    """Tests for ToolInitManager wiring in AgentServer (T029)."""

    def test_init_creates_tool_init_manager(self, mock_agent_config: MagicMock) -> None:
        """T029: AgentServer creates a ToolInitManager on init."""
        from holodeck.serve.tool_init_manager import ToolInitManager

        server = AgentServer(agent_config=mock_agent_config)

        assert hasattr(server, "_tool_init_manager")
        assert isinstance(server._tool_init_manager, ToolInitManager)

    def test_init_max_concurrent_init_jobs_default(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T029: Default max_concurrent is 4."""
        server = AgentServer(agent_config=mock_agent_config)

        assert server._tool_init_manager._max_concurrent == 4

    def test_init_max_concurrent_init_jobs_custom(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T029: Custom max_concurrent_init_jobs is forwarded."""
        server = AgentServer(
            agent_config=mock_agent_config,
            max_concurrent_init_jobs=7,
        )

        assert server._tool_init_manager._max_concurrent == 7


class TestToolInitRouterRegistration:
    """Tests for tool init router registration in create_app (T028)."""

    def test_create_app_registers_tool_init_route(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T028: Tool init route is registered on the app."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        routes = [route.path for route in app.routes]
        assert "/tools/{tool_name}/init" in routes

    def test_app_state_has_tool_init_manager(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T028: app.state.tool_init_manager is set after create_app."""
        server = AgentServer(agent_config=mock_agent_config)
        app = server.create_app()

        assert hasattr(app.state, "tool_init_manager")
        assert app.state.tool_init_manager is server._tool_init_manager

    def test_tool_init_route_available_in_agui_protocol(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T028: Tool init endpoint available regardless of protocol (AG-UI)."""
        server = AgentServer(
            agent_config=mock_agent_config, protocol=ProtocolType.AG_UI
        )
        app = server.create_app()

        routes = [route.path for route in app.routes]
        assert "/tools/{tool_name}/init" in routes

    def test_tool_init_route_available_in_rest_protocol(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T028: Tool init endpoint available regardless of protocol (REST)."""
        server = AgentServer(agent_config=mock_agent_config, protocol=ProtocolType.REST)
        app = server.create_app()

        routes = [route.path for route in app.routes]
        assert "/tools/{tool_name}/init" in routes


class TestToolInitShutdownWiring:
    """Tests for tool init shutdown and cleanup wiring (T030)."""

    @pytest.mark.asyncio
    async def test_stop_calls_tool_init_manager_shutdown(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T030: stop() calls _tool_init_manager.shutdown()."""
        from unittest.mock import AsyncMock

        server = AgentServer(agent_config=mock_agent_config)
        await server.start()

        server._tool_init_manager.shutdown = AsyncMock()
        await server.stop()

        server._tool_init_manager.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_shutdown_before_session_cleanup(
        self, mock_agent_config: MagicMock
    ) -> None:
        """T030: Tool init shutdown happens before session cleanup."""

        call_order: list[str] = []

        server = AgentServer(agent_config=mock_agent_config)
        await server.start()

        async def track_shutdown() -> None:
            call_order.append("tool_init_shutdown")

        async def track_stop_cleanup() -> None:
            call_order.append("stop_cleanup_task")

        server._tool_init_manager.shutdown = track_shutdown
        server.sessions.stop_cleanup_task = track_stop_cleanup

        await server.stop()

        assert call_order.index("tool_init_shutdown") < call_order.index(
            "stop_cleanup_task"
        )

    @pytest.mark.asyncio
    @patch("holodeck.lib.source_resolver.SourceResolver.cleanup_orphans")
    async def test_start_calls_cleanup_orphans(
        self,
        mock_cleanup: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T030: start() calls SourceResolver.cleanup_orphans()."""

        mock_cleanup.return_value = 0

        server = AgentServer(agent_config=mock_agent_config)
        await server.start()

        mock_cleanup.assert_awaited_once()


class TestValidateBackendPrerequisites:
    """Tests for AgentServer._validate_backend_prerequisites().

    T014: Validates backend-specific prerequisites before serving.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("holodeck.serve.server.validate_credentials")
    @patch("holodeck.serve.server.validate_nodejs")
    async def test_valid_anthropic_no_exception(
        self,
        mock_validate_nodejs: MagicMock,
        mock_validate_credentials: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T014: Valid Anthropic provider passes validation without exception."""
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        server = AgentServer(agent_config=mock_agent_config)

        await server._validate_backend_prerequisites()

        mock_validate_nodejs.assert_called_once()
        mock_validate_credentials.assert_called_once_with(mock_agent_config.model)

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("holodeck.serve.server.validate_credentials")
    @patch("holodeck.serve.server.validate_nodejs")
    async def test_missing_nodejs_raises_backend_init_error(
        self,
        mock_validate_nodejs: MagicMock,
        mock_validate_credentials: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T014: Missing Node.js raises BackendInitError for Anthropic provider."""
        mock_validate_nodejs.side_effect = ConfigError("nodejs", "Node.js not found")
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        server = AgentServer(agent_config=mock_agent_config)

        with pytest.raises(BackendInitError):
            await server._validate_backend_prerequisites()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("holodeck.serve.server.validate_credentials")
    @patch("holodeck.serve.server.validate_nodejs")
    async def test_missing_credentials_raises_backend_init_error(
        self,
        mock_validate_nodejs: MagicMock,
        mock_validate_credentials: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T014: Missing credentials raises BackendInitError for Anthropic provider."""
        mock_validate_credentials.side_effect = ConfigError(
            "credentials", "API key not found"
        )
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        server = AgentServer(agent_config=mock_agent_config)

        with pytest.raises(BackendInitError):
            await server._validate_backend_prerequisites()

        mock_validate_nodejs.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("holodeck.serve.server.validate_credentials")
    @patch("holodeck.serve.server.validate_nodejs")
    async def test_non_anthropic_provider_skips_validation(
        self,
        mock_validate_nodejs: MagicMock,
        mock_validate_credentials: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T014: Non-Anthropic provider skips Node.js and credential validation."""
        mock_agent_config.model.provider = ProviderEnum.OPENAI
        server = AgentServer(agent_config=mock_agent_config)

        await server._validate_backend_prerequisites()

        mock_validate_nodejs.assert_not_called()
        mock_validate_credentials.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch("holodeck.serve.server.validate_credentials")
    @patch("holodeck.serve.server.validate_nodejs")
    async def test_old_nodejs_version_raises_backend_init_error(
        self,
        mock_validate_nodejs: MagicMock,
        mock_validate_credentials: MagicMock,
        mock_agent_config: MagicMock,
    ) -> None:
        """T014: Old Node.js version raises BackendInitError for Anthropic provider."""
        mock_validate_nodejs.side_effect = ConfigError(
            "nodejs", "Node.js version 16 found but >= 18 required"
        )
        mock_agent_config.model.provider = ProviderEnum.ANTHROPIC
        server = AgentServer(agent_config=mock_agent_config)

        with pytest.raises(BackendInitError):
            await server._validate_backend_prerequisites()
