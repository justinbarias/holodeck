"""Integration tests for tool init endpoints (POST and GET).

Tests the full HTTP path through AgentServer → tool_init_routes → ToolInitManager
using a real FastAPI app with realistic agent configurations. The actual tool
initialization background task is mocked to avoid needing real embedding providers.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

# Load environment variables from tests/integration/.env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_agent_with_tools() -> MagicMock:
    """Create a realistic mock agent with mixed tool types."""
    vs_tool = MagicMock()
    vs_tool.name = "knowledge_base"
    vs_tool.type = "vectorstore"
    vs_tool.source = "./data"

    hd_tool = MagicMock()
    hd_tool.name = "docs"
    hd_tool.type = "hierarchical_document"
    hd_tool.source = "./docs"

    fn_tool = MagicMock()
    fn_tool.name = "calculator"
    fn_tool.type = "function"

    mcp_tool = MagicMock()
    mcp_tool.name = "api"
    mcp_tool.type = "mcp"

    agent = MagicMock()
    agent.name = "test-agent"
    agent.tools = [vs_tool, hd_tool, fn_tool, mcp_tool]
    return agent


@pytest_asyncio.fixture
async def integration_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with a real AgentServer and ToolInitManager."""
    from holodeck.serve.server import AgentServer

    agent = _make_agent_with_tools()
    server = AgentServer(agent_config=agent)
    app = server.create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await server._tool_init_manager.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestToolInitEndpointIntegration:
    """Integration tests for the tool init endpoint through the full stack."""

    @pytest.mark.asyncio
    async def test_post_init_vectorstore_returns_201(
        self, integration_client: AsyncClient
    ) -> None:
        """POST /tools/{name}/init for vectorstore returns 201 with correct body."""
        with (
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("./data")
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            response = await integration_client.post("/tools/knowledge_base/init")

        assert response.status_code == 201
        assert response.headers["location"] == "/tools/knowledge_base/init"

        body = response.json()
        assert body["tool_name"] == "knowledge_base"
        assert body["state"] == "pending"
        assert body["href"] == "/tools/knowledge_base/init"
        assert body["force"] is False
        assert "created_at" in body

    @pytest.mark.asyncio
    async def test_post_init_hierarchical_doc_returns_201(
        self, integration_client: AsyncClient
    ) -> None:
        """POST /tools/{name}/init for hierarchical_document returns 201."""
        with (
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("./docs")
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            response = await integration_client.post("/tools/docs/init")

        assert response.status_code == 201
        assert response.json()["tool_name"] == "docs"

    @pytest.mark.asyncio
    async def test_post_init_nonexistent_tool_returns_404(
        self, integration_client: AsyncClient
    ) -> None:
        """POST /tools/{name}/init for unknown tool returns 404 ProblemDetail."""
        response = await integration_client.post("/tools/nonexistent/init")

        assert response.status_code == 404
        body = response.json()
        assert body["title"] == "Not Found"
        assert body["status"] == 404
        assert "application/problem+json" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_post_init_non_initializable_returns_400(
        self, integration_client: AsyncClient
    ) -> None:
        """POST /tools/{name}/init for function tool returns 400 ProblemDetail."""
        response = await integration_client.post("/tools/calculator/init")

        assert response.status_code == 400
        body = response.json()
        assert body["title"] == "Bad Request"
        assert body["status"] == 400

    @pytest.mark.asyncio
    async def test_post_init_mcp_tool_returns_400(
        self, integration_client: AsyncClient
    ) -> None:
        """POST /tools/{name}/init for MCP tool returns 400 ProblemDetail."""
        response = await integration_client.post("/tools/api/init")

        assert response.status_code == 400
        body = response.json()
        assert body["status"] == 400

    @pytest.mark.asyncio
    async def test_post_init_duplicate_returns_409(
        self, integration_client: AsyncClient
    ) -> None:
        """POST same tool twice returns 409 Conflict on second call."""
        import asyncio

        # Use an event to keep the first job in-progress
        hold = asyncio.Event()

        async def slow_init(*args: object, **kwargs: object) -> None:
            await hold.wait()

        with (
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                side_effect=slow_init,
            ),
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("./data")
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            # First call succeeds
            response1 = await integration_client.post("/tools/knowledge_base/init")
            assert response1.status_code == 201

            # Second call conflicts (job still in-progress)
            response2 = await integration_client.post("/tools/knowledge_base/init")

            # Release the held job
            hold.set()
            await asyncio.sleep(0.01)

        assert response2.status_code == 409
        body = response2.json()
        assert body["title"] == "Conflict"

    @pytest.mark.asyncio
    async def test_post_init_with_force_param(
        self, integration_client: AsyncClient
    ) -> None:
        """POST with force=true passes through to the init job."""
        with (
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("./data")
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            response = await integration_client.post(
                "/tools/knowledge_base/init?force=true"
            )

        assert response.status_code == 201
        assert response.json()["force"] is True

    @pytest.mark.asyncio
    async def test_health_and_init_coexist(
        self, integration_client: AsyncClient
    ) -> None:
        """Health and init endpoints both work on the same app."""
        health = await integration_client.get("/health")
        assert health.status_code == 200

        init_404 = await integration_client.post("/tools/missing/init")
        assert init_404.status_code == 404

        health_again = await integration_client.get("/health")
        assert health_again.status_code == 200


@pytest.mark.integration
class TestGetToolInitIntegration:
    """Integration tests for GET /tools/{tool_name}/init (US2)."""

    @pytest.mark.asyncio
    async def test_get_init_no_prior_post_returns_404(
        self, integration_client: AsyncClient
    ) -> None:
        """GET without a prior POST returns 404 ProblemDetail."""
        response = await integration_client.get("/tools/knowledge_base/init")

        assert response.status_code == 404
        body = response.json()
        assert body["title"] == "Not Found"
        assert body["status"] == 404
        assert "application/problem+json" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_get_init_after_post_returns_200(
        self, integration_client: AsyncClient
    ) -> None:
        """GET after POST returns 200 with matching job state."""
        with (
            patch(
                "holodeck.serve.tool_init_manager.initialize_single_tool",
                new_callable=AsyncMock,
            ),
            patch("holodeck.serve.tool_init_manager.SourceResolver") as mock_sr,
        ):
            mock_resolved = MagicMock()
            mock_resolved.local_path = Path("./data")
            mock_resolved.is_remote = False
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resolved)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_sr.resolve_context.return_value = mock_ctx

            # Create the job
            post_resp = await integration_client.post("/tools/knowledge_base/init")
            assert post_resp.status_code == 201

            # Poll the job
            get_resp = await integration_client.get("/tools/knowledge_base/init")

        assert get_resp.status_code == 200
        body = get_resp.json()
        assert body["tool_name"] == "knowledge_base"
        assert body["href"] == "/tools/knowledge_base/init"
        assert "created_at" in body
