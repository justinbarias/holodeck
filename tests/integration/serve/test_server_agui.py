"""Integration tests for AG-UI protocol server.

Tests for:
- T017: AG-UI protocol endpoint (/awp)
- T018: AG-UI streaming response
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent_config() -> MagicMock:
    """Create a mock agent configuration."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.description = "A test agent"
    return agent


@pytest.fixture
def mock_agent_response() -> MagicMock:
    """Create a mock AgentResponse."""
    from holodeck.chat.executor import AgentResponse
    from holodeck.models.token_usage import TokenUsage

    response = MagicMock(spec=AgentResponse)
    response.content = "Hello! How can I help you today?"
    response.tool_executions = []
    response.tokens_used = TokenUsage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    response.execution_time = 0.5
    return response


@pytest.fixture
def mock_agent_executor(mock_agent_response: MagicMock) -> MagicMock:
    """Create a mock AgentExecutor."""
    executor = MagicMock()
    executor.execute_turn = AsyncMock(return_value=mock_agent_response)
    return executor


@pytest_asyncio.fixture
async def agui_client(
    mock_agent_config: MagicMock,
    mock_agent_executor: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with AG-UI protocol server."""
    from holodeck.serve.models import ProtocolType
    from holodeck.serve.server import AgentServer

    # Create server with AG-UI protocol
    server = AgentServer(
        agent_config=mock_agent_config,
        protocol=ProtocolType.AG_UI,
        host="127.0.0.1",
        port=8000,
    )
    app = server.create_app()

    # Create mock session
    mock_session = MagicMock(
        session_id="thread-123",
        agent_executor=mock_agent_executor,
        message_count=0,
    )

    # Patch session retrieval to return mock session
    # This avoids creating a real AgentExecutor
    with patch.object(
        server.sessions,
        "get",
        return_value=mock_session,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


# =============================================================================
# T017: Integration tests for AG-UI protocol endpoint
# =============================================================================


class TestAGUIEndpoint:
    """Integration tests for /awp AG-UI endpoint."""

    @pytest.mark.asyncio
    async def test_awp_endpoint_exists(self, agui_client: AsyncClient) -> None:
        """Test that /awp endpoint exists and accepts POST."""
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )
        # Should not return 404 or 405
        assert response.status_code != 404
        assert response.status_code != 405

    @pytest.mark.asyncio
    async def test_awp_endpoint_returns_streaming_response(
        self, agui_client: AsyncClient
    ) -> None:
        """Test POST /awp returns SSE streaming response."""
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )

        # Check response is streaming
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert (
            "text/event-stream" in content_type
            or "application/octet-stream" in content_type
        )

    @pytest.mark.asyncio
    async def test_awp_endpoint_accepts_run_agent_input(
        self, agui_client: AsyncClient
    ) -> None:
        """Test endpoint accepts RunAgentInput JSON body."""
        # Full RunAgentInput structure
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "system", "content": "You are helpful"},
                    {"id": "msg-2", "role": "user", "content": "What's the weather?"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_awp_endpoint_rejects_invalid_input(
        self, agui_client: AsyncClient
    ) -> None:
        """Test endpoint rejects invalid input with 422."""
        response = await agui_client.post(
            "/awp",
            json={
                # Missing required fields
                "messages": [],
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_awp_endpoint_content_type_negotiation(
        self, agui_client: AsyncClient
    ) -> None:
        """Test Accept header controls SSE vs binary format."""
        # Request SSE format
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


# =============================================================================
# T018: Integration tests for AG-UI streaming response
# =============================================================================


class TestAGUIStreaming:
    """Integration tests for AG-UI streaming responses."""

    @pytest.mark.asyncio
    async def test_streaming_includes_run_started_event(
        self, agui_client: AsyncClient
    ) -> None:
        """Test stream starts with RunStartedEvent."""
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )

        assert response.status_code == 200
        content = response.text

        # Should contain RUN_STARTED event
        assert "RUN_STARTED" in content or "run_started" in content.lower()

    @pytest.mark.asyncio
    async def test_streaming_includes_text_message_events(
        self, agui_client: AsyncClient
    ) -> None:
        """Test stream includes TextMessageStart/Content/End sequence."""
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )

        assert response.status_code == 200
        content = response.text

        # Should contain text message events
        assert (
            "TEXT_MESSAGE_START" in content or "text_message_start" in content.lower()
        )
        assert (
            "TEXT_MESSAGE_CONTENT" in content
            or "text_message_content" in content.lower()
        )
        assert "TEXT_MESSAGE_END" in content or "text_message_end" in content.lower()

    @pytest.mark.asyncio
    async def test_streaming_includes_run_finished_event(
        self, agui_client: AsyncClient
    ) -> None:
        """Test stream ends with RunFinishedEvent on success."""
        response = await agui_client.post(
            "/awp",
            json={
                "threadId": "thread-123",
                "runId": "run-456",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Hello"},
                ],
                "state": None,
                "tools": [],
                "context": [],
                "forwardedProps": None,
            },
        )

        assert response.status_code == 200
        content = response.text

        # Should contain RUN_FINISHED event
        assert "RUN_FINISHED" in content or "run_finished" in content.lower()

    @pytest.mark.asyncio
    async def test_streaming_includes_run_error_on_failure(
        self,
        mock_agent_config: MagicMock,
    ) -> None:
        """Test stream ends with RunErrorEvent on agent failure."""
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Create mock executor that raises an error
        failing_executor = MagicMock()
        failing_executor.execute_turn = AsyncMock(
            side_effect=RuntimeError("Agent execution failed")
        )

        # Create mock session with failing executor
        mock_session = MagicMock(
            session_id="thread-123",
            agent_executor=failing_executor,
            message_count=0,
        )

        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.AG_UI,
        )
        app = server.create_app()

        # Patch session retrieval to return mock session
        with patch.object(
            server.sessions,
            "get",
            return_value=mock_session,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/awp",
                    json={
                        "threadId": "thread-123",
                        "runId": "run-456",
                        "messages": [
                            {"id": "msg-1", "role": "user", "content": "Hello"},
                        ],
                        "state": None,
                        "tools": [],
                        "context": [],
                        "forwardedProps": None,
                    },
                )

                # Should still return 200 (streaming started) but contain error event
                content = response.text
                assert "RUN_ERROR" in content or "run_error" in content.lower()

    @pytest.mark.asyncio
    async def test_streaming_tool_call_events(
        self,
        mock_agent_config: MagicMock,
    ) -> None:
        """Test tool calls emit ToolCallStart/Args/End events."""
        from holodeck.chat.executor import AgentResponse
        from holodeck.models.token_usage import TokenUsage
        from holodeck.models.tool_execution import ToolExecution, ToolStatus
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Create mock response with tool execution
        tool_exec = ToolExecution(
            tool_name="search_knowledge_base",
            parameters={"query": "return policy"},
            status=ToolStatus.SUCCESS,
        )
        response_with_tools = MagicMock(spec=AgentResponse)
        response_with_tools.content = "Based on the search results..."
        response_with_tools.tool_executions = [tool_exec]
        response_with_tools.tokens_used = TokenUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        response_with_tools.execution_time = 0.5

        executor = MagicMock()
        executor.execute_turn = AsyncMock(return_value=response_with_tools)

        # Create mock session with executor that returns tool executions
        mock_session = MagicMock(
            session_id="thread-123",
            agent_executor=executor,
            message_count=0,
        )

        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.AG_UI,
        )
        app = server.create_app()

        # Patch session retrieval to return mock session
        with patch.object(
            server.sessions,
            "get",
            return_value=mock_session,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/awp",
                    json={
                        "threadId": "thread-123",
                        "runId": "run-456",
                        "messages": [
                            {
                                "id": "msg-1",
                                "role": "user",
                                "content": "What's the return policy?",
                            },
                        ],
                        "state": None,
                        "tools": [],
                        "context": [],
                        "forwardedProps": None,
                    },
                )

                assert response.status_code == 200
                content = response.text

                # Should contain tool call events
                assert (
                    "TOOL_CALL_START" in content or "tool_call_start" in content.lower()
                )
                assert (
                    "TOOL_CALL_ARGS" in content or "tool_call_args" in content.lower()
                )
                assert "TOOL_CALL_END" in content or "tool_call_end" in content.lower()


class TestAGUIHealthEndpoints:
    """Test health endpoints are accessible with AG-UI protocol."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, agui_client: AsyncClient) -> None:
        """Test /health endpoint works with AG-UI protocol."""
        response = await agui_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["agent_name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, agui_client: AsyncClient) -> None:
        """Test /ready endpoint works with AG-UI protocol."""
        response = await agui_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
