"""Integration tests for REST protocol server.

Tests for:
- T033: Synchronous chat endpoint
- T034: Streaming chat endpoint (SSE)
- T035: Multimodal file upload (base64 JSON)
- T036: Multimodal file upload (multipart form-data)
"""

from __future__ import annotations

import base64
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
async def rest_client(
    mock_agent_config: MagicMock,
    mock_agent_executor: MagicMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with REST protocol server."""
    from holodeck.serve.models import ProtocolType
    from holodeck.serve.server import AgentServer

    # Create mock session
    mock_session = MagicMock(
        session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        agent_executor=mock_agent_executor,
        message_count=0,
    )

    # Patch AgentExecutor BEFORE creating app to intercept imports
    with patch(
        "holodeck.chat.executor.AgentExecutor",
        return_value=mock_agent_executor,
    ):
        # Create server with REST protocol
        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.REST,
            host="127.0.0.1",
            port=8000,
        )
        app = server.create_app()

        # Patch session management
        # - get: return None to simulate no existing session (new session flow)
        # - create: return mock session
        with (
            patch.object(
                server.sessions,
                "get",
                return_value=None,
            ),
            patch.object(
                server.sessions,
                "create",
                return_value=mock_session,
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                yield client


# =============================================================================
# T033: Integration tests for synchronous chat endpoint
# =============================================================================


class TestChatSyncEndpoint:
    """Integration tests for POST /agent/{name}/chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_sync_basic_message(self, rest_client: AsyncClient) -> None:
        """Test basic synchronous chat request."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={"message": "Hello, world!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data
        assert "content" in data
        assert "session_id" in data
        assert "execution_time_ms" in data

    @pytest.mark.asyncio
    async def test_chat_sync_with_session_id(self, rest_client: AsyncClient) -> None:
        """Test synchronous chat with existing session_id."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Continue conversation",
                "session_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    @pytest.mark.asyncio
    async def test_chat_sync_creates_new_session(
        self, mock_agent_config: MagicMock, mock_agent_executor: MagicMock
    ) -> None:
        """Test that new session is created when session_id not provided."""
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Patch AgentExecutor BEFORE creating app
        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_agent_executor,
        ):
            server = AgentServer(
                agent_config=mock_agent_config,
                protocol=ProtocolType.REST,
            )
            app = server.create_app()

            mock_session = MagicMock(
                session_id="01NEW3NDEKTSV4RRFFQ69G5NEW",
                agent_executor=mock_agent_executor,
                message_count=0,
            )

            # Don't patch session.get - let it return None to trigger creation
            with (
                patch.object(server.sessions, "get", return_value=None),
                patch.object(
                    server.sessions, "create", return_value=mock_session
                ) as mock_create,
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/test-agent/chat",
                        json={"message": "Hello"},
                    )

                    assert response.status_code == 200
                    # create should have been called
                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_sync_invalid_request_400(
        self, rest_client: AsyncClient
    ) -> None:
        """Test 400 response for invalid request."""
        # Empty message should fail validation
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={"message": ""},
        )

        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_sync_missing_message_400(
        self, rest_client: AsyncClient
    ) -> None:
        """Test 400 response when message is missing."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={},
        )

        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_sync_includes_tool_calls(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test response includes tool_calls when agent uses tools."""
        from holodeck.models.tool_execution import ToolExecution, ToolStatus
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Create response with tool execution
        tool_exec = ToolExecution(
            tool_name="search_knowledge_base",
            parameters={"query": "return policy"},
            status=ToolStatus.SUCCESS,
        )
        mock_response = MagicMock()
        mock_response.content = "Based on the search results..."
        mock_response.tool_executions = [tool_exec]
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            agent_executor=mock_executor,
            message_count=0,
        )

        # Patch AgentExecutor BEFORE creating app
        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_executor,
        ):
            server = AgentServer(
                agent_config=mock_agent_config,
                protocol=ProtocolType.REST,
            )
            app = server.create_app()

            with (
                patch.object(server.sessions, "get", return_value=None),
                patch.object(server.sessions, "create", return_value=mock_session),
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/test-agent/chat",
                        json={"message": "What's the return policy?"},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "tool_calls" in data
                    assert len(data["tool_calls"]) == 1
                    assert data["tool_calls"][0]["name"] == "search_knowledge_base"


class TestChatSyncContentType:
    """Tests for content type handling."""

    @pytest.mark.asyncio
    async def test_chat_sync_returns_json(self, rest_client: AsyncClient) -> None:
        """Test sync endpoint returns application/json."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type


# =============================================================================
# T034: Integration tests for streaming chat endpoint (SSE)
# =============================================================================


class TestChatStreamEndpoint:
    """Integration tests for POST /agent/{name}/chat/stream endpoint."""

    @pytest.mark.asyncio
    async def test_chat_stream_returns_sse(self, rest_client: AsyncClient) -> None:
        """Test streaming endpoint returns text/event-stream."""
        response = await rest_client.post(
            "/agent/test-agent/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type

    @pytest.mark.asyncio
    async def test_chat_stream_receives_events(self, rest_client: AsyncClient) -> None:
        """Test streaming endpoint returns SSE events."""
        response = await rest_client.post(
            "/agent/test-agent/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        content = response.text

        # Should contain SSE events
        assert "event:" in content
        assert "data:" in content

    @pytest.mark.asyncio
    async def test_chat_stream_event_order(self, rest_client: AsyncClient) -> None:
        """Test SSE events are in correct order: stream_start, delta, stream_end."""
        response = await rest_client.post(
            "/agent/test-agent/chat/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        content = response.text

        # Find positions of events
        stream_start_pos = content.find("event: stream_start")
        message_delta_pos = content.find("event: message_delta")
        stream_end_pos = content.find("event: stream_end")

        # Verify order
        assert stream_start_pos >= 0
        assert stream_end_pos >= 0
        if message_delta_pos >= 0:  # May not have delta if content is empty
            assert stream_start_pos < message_delta_pos < stream_end_pos

    @pytest.mark.asyncio
    async def test_chat_stream_with_tool_calls(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test streaming includes tool call events."""
        from holodeck.models.tool_execution import ToolExecution, ToolStatus
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Create response with tool execution
        tool_exec = ToolExecution(
            tool_name="search",
            parameters={"query": "test"},
            status=ToolStatus.SUCCESS,
        )
        mock_response = MagicMock()
        mock_response.content = "Results found"
        mock_response.tool_executions = [tool_exec]
        mock_response.tokens_used = None

        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(return_value=mock_response)

        mock_session = MagicMock(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            agent_executor=mock_executor,
            message_count=0,
        )

        # Patch AgentExecutor BEFORE creating app
        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_executor,
        ):
            server = AgentServer(
                agent_config=mock_agent_config,
                protocol=ProtocolType.REST,
            )
            app = server.create_app()

            with (
                patch.object(server.sessions, "get", return_value=None),
                patch.object(server.sessions, "create", return_value=mock_session),
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/test-agent/chat/stream",
                        json={"message": "Search for info"},
                    )

                    assert response.status_code == 200
                    content = response.text

                    # Should contain tool call events
                    assert "event: tool_call_start" in content
                    assert "event: tool_call_end" in content

    @pytest.mark.asyncio
    async def test_chat_stream_error_event(self, mock_agent_config: MagicMock) -> None:
        """Test streaming emits error event on agent failure."""
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        # Mock executor that raises an error
        mock_executor = MagicMock()
        mock_executor.execute_turn = AsyncMock(
            side_effect=RuntimeError("Agent execution failed")
        )

        mock_session = MagicMock(
            session_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            agent_executor=mock_executor,
            message_count=0,
        )

        # Patch AgentExecutor BEFORE creating app
        with patch(
            "holodeck.chat.executor.AgentExecutor",
            return_value=mock_executor,
        ):
            server = AgentServer(
                agent_config=mock_agent_config,
                protocol=ProtocolType.REST,
            )
            app = server.create_app()

            with (
                patch.object(server.sessions, "get", return_value=None),
                patch.object(server.sessions, "create", return_value=mock_session),
            ):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/agent/test-agent/chat/stream",
                        json={"message": "Hello"},
                    )

                    # Should still return 200 (streaming started)
                    # but contain error event
                    content = response.text
                    assert "event: error" in content


# =============================================================================
# T035: Integration tests for multimodal file upload (base64 JSON)
# =============================================================================


class TestMultimodalBase64:
    """Integration tests for multimodal with base64-encoded files."""

    @pytest.mark.asyncio
    async def test_chat_with_base64_image(self, rest_client: AsyncClient) -> None:
        """Test chat with base64-encoded image file."""
        # 1x1 red PNG image
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Describe this image",
                "files": [
                    {
                        "content": base64.b64encode(png_bytes).decode(),
                        "mime_type": "image/png",
                        "filename": "test.png",
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    @pytest.mark.asyncio
    async def test_chat_with_base64_pdf(self, rest_client: AsyncClient) -> None:
        """Test chat with base64-encoded PDF file."""
        # Minimal PDF content
        pdf_content = b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"

        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Summarize this PDF",
                "files": [
                    {
                        "content": base64.b64encode(pdf_content).decode(),
                        "mime_type": "application/pdf",
                        "filename": "document.pdf",
                    }
                ],
            },
        )

        # Should succeed (even if PDF processing fails gracefully)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_with_invalid_base64_400(self, rest_client: AsyncClient) -> None:
        """Test 400 response for invalid base64 content."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Process this file",
                "files": [
                    {
                        "content": "not-valid-base64!!!",
                        "mime_type": "image/png",
                        "filename": "test.png",
                    }
                ],
            },
        )

        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_with_unsupported_mime_type_400(
        self, rest_client: AsyncClient
    ) -> None:
        """Test 400 response for unsupported MIME type."""
        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Process this file",
                "files": [
                    {
                        "content": base64.b64encode(b"test").decode(),
                        "mime_type": "application/x-unsupported",
                        "filename": "test.bin",
                    }
                ],
            },
        )

        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_with_multiple_files(self, rest_client: AsyncClient) -> None:
        """Test chat with multiple base64 files."""
        text_content = base64.b64encode(b"Hello, world!").decode()
        csv_content = base64.b64encode(b"a,b,c\n1,2,3").decode()

        response = await rest_client.post(
            "/agent/test-agent/chat",
            json={
                "message": "Analyze these files",
                "files": [
                    {
                        "content": text_content,
                        "mime_type": "text/plain",
                        "filename": "file1.txt",
                    },
                    {
                        "content": csv_content,
                        "mime_type": "text/csv",
                        "filename": "file2.csv",
                    },
                ],
            },
        )

        assert response.status_code == 200


# =============================================================================
# T036: Integration tests for multipart file upload
# =============================================================================


class TestMultimodalMultipart:
    """Integration tests for multimodal with multipart form-data."""

    @pytest.mark.asyncio
    async def test_chat_multipart_with_file(self, rest_client: AsyncClient) -> None:
        """Test chat with multipart file upload."""
        response = await rest_client.post(
            "/agent/test-agent/chat/multipart",
            data={"message": "Analyze this file"},
            files={"files": ("test.txt", b"Hello, world!", "text/plain")},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_multipart_multiple_files(
        self, rest_client: AsyncClient
    ) -> None:
        """Test chat with multiple multipart files."""
        response = await rest_client.post(
            "/agent/test-agent/chat/multipart",
            data={"message": "Analyze these files"},
            files=[
                ("files", ("file1.txt", b"Content 1", "text/plain")),
                ("files", ("file2.txt", b"Content 2", "text/plain")),
            ],
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_multipart_with_session_id(
        self, rest_client: AsyncClient
    ) -> None:
        """Test multipart with session_id in form data."""
        response = await rest_client.post(
            "/agent/test-agent/chat/multipart",
            data={
                "message": "Continue with file",
                "session_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            },
            files={"files": ("test.txt", b"Content", "text/plain")},
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_multipart_stream_endpoint(
        self, rest_client: AsyncClient
    ) -> None:
        """Test multipart upload to streaming endpoint."""
        response = await rest_client.post(
            "/agent/test-agent/chat/stream/multipart",
            data={"message": "Analyze with streaming"},
            files={"files": ("test.txt", b"Content", "text/plain")},
        )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type


# =============================================================================
# Health endpoints with REST protocol
# =============================================================================


class TestRESTHealthEndpoints:
    """Test health endpoints are accessible with REST protocol."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, rest_client: AsyncClient) -> None:
        """Test /health endpoint works with REST protocol."""
        response = await rest_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["agent_name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, rest_client: AsyncClient) -> None:
        """Test /ready endpoint works with REST protocol."""
        response = await rest_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    @pytest.mark.asyncio
    async def test_openapi_docs_available(self, rest_client: AsyncClient) -> None:
        """Test OpenAPI docs are available at /docs for REST protocol."""
        response = await rest_client.get("/docs")
        # /docs should redirect or return HTML
        assert response.status_code in (200, 307)


# =============================================================================
# Session management endpoints
# =============================================================================


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    @pytest.mark.asyncio
    async def test_delete_session_returns_204(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test DELETE /sessions/{session_id} returns 204."""
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.REST,
        )
        app = server.create_app()

        with patch.object(server.sessions, "delete", return_value=True):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.delete("/sessions/01ARZ3NDEKTSV4RRFFQ69G5FAV")
                assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_returns_204(
        self, mock_agent_config: MagicMock
    ) -> None:
        """Test DELETE returns 204 even for nonexistent session (idempotent)."""
        from holodeck.serve.models import ProtocolType
        from holodeck.serve.server import AgentServer

        server = AgentServer(
            agent_config=mock_agent_config,
            protocol=ProtocolType.REST,
        )
        app = server.create_app()

        with patch.object(server.sessions, "delete", return_value=False):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.delete("/sessions/nonexistent")
                # Should still return 204 (idempotent)
                assert response.status_code == 204
