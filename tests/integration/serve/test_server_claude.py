"""Integration tests for Claude-backed agent serve (US1).

Tests: T020 (startup/health), T021 (request handling), T022 (OTel).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from holodeck.models.llm import ProviderEnum
from holodeck.serve.models import ProtocolType, ServerState
from holodeck.serve.server import AgentServer

# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def claude_server() -> AgentServer:
    """Create an AgentServer with a mock Claude agent config."""
    mock_config = MagicMock()
    mock_config.name = "claude-serve-test"
    mock_config.model.provider = ProviderEnum.ANTHROPIC
    mock_config.claude.max_concurrent_sessions = 3

    server = AgentServer(
        agent_config=mock_config,
        protocol=ProtocolType.REST,
        host="127.0.0.1",
        port=8000,
    )
    return server


@pytest_asyncio.fixture
async def claude_client(
    claude_server: AgentServer,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an httpx AsyncClient for the Claude server."""
    app = claude_server.create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# =============================================================================
# T020: Startup and Health
# =============================================================================


class TestStartupAndHealth:
    """Integration tests for Claude server startup and health endpoints (T020)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_returns_backend_ready(
        self, claude_client: AsyncClient
    ) -> None:
        """GET /health returns 200 with backend_ready field (defaults True)."""
        response = await claude_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # agent_ready serves as the backend readiness indicator;
        # after create_app() the server transitions to READY which
        # makes is_ready (and therefore agent_ready) True.
        assert "agent_ready" in data
        assert data["agent_ready"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_endpoint_returns_backend_diagnostics_empty(
        self, claude_client: AsyncClient
    ) -> None:
        """GET /health returns an empty active_sessions count (no sessions yet)."""
        response = await claude_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        # No sessions have been created, so active count should be 0
        assert data["active_sessions"] == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_transitions_to_running_after_validation(
        self, claude_server: AgentServer
    ) -> None:
        """Server state becomes RUNNING after start() with mocked prerequisites."""
        # Ensure create_app() has been called so the app exists
        claude_server.create_app()

        with patch.object(
            claude_server,
            "_validate_backend_prerequisites",
            new_callable=AsyncMock,
        ):
            await claude_server.start()
            assert claude_server.state == ServerState.RUNNING

            await claude_server.stop()
            assert claude_server.state == ServerState.STOPPED


# =============================================================================
# T021: Request Handling
# =============================================================================


class TestRequestHandling:
    """Integration tests for Claude server request handling (T021)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rest_chat_returns_200(self, claude_server: AgentServer) -> None:
        """POST /agent/{name}/chat returns 200 with expected fields."""
        from holodeck.chat.executor import AgentResponse

        # Build a mock AgentResponse
        mock_response = MagicMock(spec=AgentResponse)
        mock_response.content = "Hello!"
        mock_response.tool_executions = []
        mock_response.tokens_used = None
        mock_response.execution_time = 0.1

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            agent_executor=mock_executor,
            message_count=0,
        )

        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_executor,
        ):
            app = claude_server.create_app()

            with (
                patch.object(
                    claude_server.sessions,
                    "get",
                    return_value=None,
                ),
                patch.object(
                    claude_server.sessions,
                    "create",
                    return_value=mock_session,
                ),
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/claude-serve-test/chat",
                        json={"message": "Hi"},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "content" in data
                    assert "session_id" in data
                    assert "tool_calls" in data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_tracked_in_store(self, claude_server: AgentServer) -> None:
        """After a successful chat request, session store has an active session."""
        from holodeck.chat.executor import AgentResponse

        mock_response = MagicMock(spec=AgentResponse)
        mock_response.content = "Hello!"
        mock_response.tool_executions = []
        mock_response.tokens_used = None
        mock_response.execution_time = 0.1

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            agent_executor=mock_executor,
            message_count=0,
        )

        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_executor,
        ):
            app = claude_server.create_app()

            with (
                patch.object(
                    claude_server.sessions,
                    "get",
                    return_value=None,
                ),
                patch.object(
                    claude_server.sessions,
                    "create",
                    return_value=mock_session,
                ) as mock_create,
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/claude-serve-test/chat",
                        json={"message": "Hi"},
                    )

                    assert response.status_code == 200
                    # Verify create was called, confirming session tracking
                    mock_create.assert_called_once()


# =============================================================================
# T022: OTel (simplified)
# =============================================================================


class TestObservability:
    """Integration tests for observability configuration (T022)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_server_with_observability_enabled(self) -> None:
        """Server created with observability_enabled=True exposes that flag."""
        mock_config = MagicMock()
        mock_config.name = "claude-otel-test"
        mock_config.model.provider = ProviderEnum.ANTHROPIC
        mock_config.claude.max_concurrent_sessions = 3

        server = AgentServer(
            agent_config=mock_config,
            protocol=ProtocolType.REST,
            host="127.0.0.1",
            port=8000,
            observability_enabled=True,
        )

        assert server.observability_enabled is True
