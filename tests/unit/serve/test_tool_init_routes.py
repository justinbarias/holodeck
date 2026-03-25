"""Unit tests for tool init route handlers (T027).

Tests cover the POST /tools/{tool_name}/init endpoint including
success responses, error mapping, and ProblemDetail formatting.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from holodeck.serve.models import InitJobProgress, InitJobState

# ---------------------------------------------------------------------------
# Helpers – mirrors test_tool_init_manager.py mock factories
# ---------------------------------------------------------------------------


def _make_init_job(
    tool_name: str = "knowledge_base",
    state: InitJobState = InitJobState.PENDING,
    force: bool = False,
) -> MagicMock:
    """Create a mock InitJob with realistic defaults."""
    job = MagicMock()
    job.tool_name = tool_name
    job.state = state
    job.created_at = datetime(2026, 3, 25, 12, 0, 0)
    job.started_at = None
    job.completed_at = None
    job.message = None
    job.error_detail = None
    job.progress = None
    job.force = force
    return job


def _make_app_with_mock_manager(manager: MagicMock | None = None) -> FastAPI:
    """Create a FastAPI app with the tool init router and a mock manager."""
    from holodeck.serve.tool_init_routes import router

    app = FastAPI()
    app.state.tool_init_manager = manager or MagicMock()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostToolInit:
    """Tests for POST /tools/{tool_name}/init endpoint."""

    def test_success_returns_201(self) -> None:
        """POST with valid vectorstore tool returns 201 Created."""
        manager = MagicMock()
        manager.start_init_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        assert response.status_code == 201

    def test_success_sets_location_header(self) -> None:
        """POST success response includes Location header."""
        manager = MagicMock()
        manager.start_init_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        assert response.headers["location"] == "/tools/knowledge_base/init"

    def test_success_body_matches_init_job_response(self) -> None:
        """POST success returns InitJobResponse-shaped body."""
        manager = MagicMock()
        manager.start_init_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")
        body = response.json()

        assert body["tool_name"] == "knowledge_base"
        assert body["state"] == "pending"
        assert body["href"] == "/tools/knowledge_base/init"
        assert body["force"] is False
        assert "created_at" in body

    def test_force_true_forwarded(self) -> None:
        """POST with force=true query param forwards to manager."""
        manager = MagicMock()
        manager.start_init_job.return_value = _make_init_job(force=True)
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init?force=true")

        manager.start_init_job.assert_called_once_with("knowledge_base", force=True)
        assert response.status_code == 201
        assert response.json()["force"] is True

    def test_force_defaults_false(self) -> None:
        """POST without force param defaults to force=False."""
        manager = MagicMock()
        manager.start_init_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        client.post("/tools/knowledge_base/init")

        manager.start_init_job.assert_called_once_with("knowledge_base", force=False)

    def test_tool_not_found_returns_404(self) -> None:
        """POST for nonexistent tool returns 404 ProblemDetail."""
        from holodeck.serve.tool_init_manager import ToolNotFoundError

        manager = MagicMock()
        manager.start_init_job.side_effect = ToolNotFoundError("not found")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/nonexistent/init")

        assert response.status_code == 404
        body = response.json()
        assert body["title"] == "Not Found"
        assert body["status"] == 404

    def test_not_initializable_returns_400(self) -> None:
        """POST for non-initializable tool type returns 400 ProblemDetail."""
        from holodeck.serve.tool_init_manager import ToolNotInitializableError

        manager = MagicMock()
        manager.start_init_job.side_effect = ToolNotInitializableError("mcp")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/api/init")

        assert response.status_code == 400
        body = response.json()
        assert body["title"] == "Bad Request"
        assert body["status"] == 400

    def test_conflict_returns_409(self) -> None:
        """POST for already-active init job returns 409 ProblemDetail."""
        from holodeck.serve.tool_init_manager import InitJobConflictError

        manager = MagicMock()
        manager.start_init_job.side_effect = InitJobConflictError("active")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        assert response.status_code == 409
        body = response.json()
        assert body["title"] == "Conflict"
        assert body["status"] == 409

    def test_capacity_returns_429(self) -> None:
        """POST at max concurrent jobs returns 429 ProblemDetail."""
        from holodeck.serve.tool_init_manager import InitJobCapacityError

        manager = MagicMock()
        manager.start_init_job.side_effect = InitJobCapacityError("at capacity")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        assert response.status_code == 429
        body = response.json()
        assert body["title"] == "Too Many Requests"
        assert body["status"] == 429

    def test_shutting_down_returns_503(self) -> None:
        """POST during shutdown returns 503 ProblemDetail."""
        from holodeck.serve.tool_init_manager import InitManagerShuttingDownError

        manager = MagicMock()
        manager.start_init_job.side_effect = InitManagerShuttingDownError(
            "shutting down"
        )
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        assert response.status_code == 503
        body = response.json()
        assert body["title"] == "Service Unavailable"
        assert body["status"] == 503

    def test_error_responses_use_problem_json_content_type(self) -> None:
        """Error responses use application/problem+json content type."""
        from holodeck.serve.tool_init_manager import ToolNotFoundError

        manager = MagicMock()
        manager.start_init_job.side_effect = ToolNotFoundError("not found")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/missing/init")

        assert "application/problem+json" in response.headers["content-type"]

    def test_error_response_includes_instance(self) -> None:
        """Error responses include instance field with the request path."""
        from holodeck.serve.tool_init_manager import ToolNotFoundError

        manager = MagicMock()
        manager.start_init_job.side_effect = ToolNotFoundError("not found")
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/my_tool/init")

        body = response.json()
        assert body["instance"] == "/tools/my_tool/init"

    def test_success_with_progress(self) -> None:
        """POST response includes progress when present on job."""
        job = _make_init_job()
        job.state = InitJobState.IN_PROGRESS
        job.progress = InitJobProgress(documents_processed=5, total_documents=10)

        manager = MagicMock()
        manager.start_init_job.return_value = job
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        body = response.json()
        assert body["progress"]["documents_processed"] == 5
        assert body["progress"]["total_documents"] == 10
