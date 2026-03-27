"""Unit tests for tool init route handlers.

Tests cover the POST and GET /tools/{tool_name}/init endpoints including
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

    def test_conflict_409_has_no_location_header(self) -> None:
        """409 Conflict response does NOT include a Location header."""
        from holodeck.serve.tool_init_manager import InitJobConflictError

        manager = MagicMock()
        manager.start_init_job.side_effect = InitJobConflictError(
            "Tool 'kb' already has an active init job (state=pending)"
        )
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/kb/init")

        assert response.status_code == 409
        assert "location" not in response.headers

    def test_conflict_409_detail_includes_tool_name_and_state(self) -> None:
        """409 ProblemDetail detail field contains tool name and current state."""
        from holodeck.serve.tool_init_manager import InitJobConflictError

        manager = MagicMock()
        manager.start_init_job.side_effect = InitJobConflictError(
            "Tool 'knowledge_base' already has an active init job (state=in_progress)"
        )
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.post("/tools/knowledge_base/init")

        body = response.json()
        assert "knowledge_base" in body["detail"]
        assert "in_progress" in body["detail"]

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


class TestGetToolInitStatus:
    """Tests for GET /tools/{tool_name}/init endpoint (US2)."""

    def test_no_job_returns_404(self) -> None:
        """GET for tool with no init job returns 404."""
        manager = MagicMock()
        manager.get_job.return_value = None
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/nonexistent/init")

        assert response.status_code == 404

    def test_no_job_returns_problem_json_content_type(self) -> None:
        """GET 404 response uses application/problem+json content type."""
        manager = MagicMock()
        manager.get_job.return_value = None
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/nonexistent/init")

        assert "application/problem+json" in response.headers["content-type"]

    def test_no_job_body_matches_problem_detail(self) -> None:
        """GET 404 body matches RFC 7807 ProblemDetail schema."""
        manager = MagicMock()
        manager.get_job.return_value = None
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/my_tool/init")

        body = response.json()
        assert body["title"] == "Not Found"
        assert body["status"] == 404
        assert body["detail"] == "No initialization job found for tool 'my_tool'."
        assert body["instance"] == "/tools/my_tool/init"

    def test_existing_job_returns_200(self) -> None:
        """GET for tool with existing init job returns 200."""
        manager = MagicMock()
        manager.get_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")

        assert response.status_code == 200

    def test_existing_job_body_matches_init_job_response(self) -> None:
        """GET 200 body matches InitJobResponse schema."""
        manager = MagicMock()
        manager.get_job.return_value = _make_init_job()
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")
        body = response.json()

        assert body["tool_name"] == "knowledge_base"
        assert body["state"] == "pending"
        assert body["href"] == "/tools/knowledge_base/init"
        assert body["force"] is False
        assert "created_at" in body

    def test_existing_job_with_progress(self) -> None:
        """GET response includes progress when present on job."""
        job = _make_init_job()
        job.state = InitJobState.IN_PROGRESS
        job.progress = InitJobProgress(documents_processed=5, total_documents=10)

        manager = MagicMock()
        manager.get_job.return_value = job
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")
        body = response.json()

        assert body["progress"]["documents_processed"] == 5
        assert body["progress"]["total_documents"] == 10

    def test_in_progress_job_includes_started_at(self) -> None:
        """GET for in-progress job includes started_at, no completed_at."""
        job = _make_init_job(state=InitJobState.IN_PROGRESS)
        job.started_at = datetime(2026, 3, 25, 12, 0, 1)

        manager = MagicMock()
        manager.get_job.return_value = job
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")
        body = response.json()

        assert body["started_at"] is not None
        assert body.get("completed_at") is None

    def test_completed_job_includes_completed_at(self) -> None:
        """GET for completed job includes completed_at timestamp."""
        job = _make_init_job(state=InitJobState.COMPLETED)
        job.started_at = datetime(2026, 3, 25, 12, 0, 1)
        job.completed_at = datetime(2026, 3, 25, 12, 0, 45)
        job.message = "Successfully initialized: 42 documents ingested"

        manager = MagicMock()
        manager.get_job.return_value = job
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")
        body = response.json()

        assert body["completed_at"] is not None
        assert body["state"] == "completed"

    def test_failed_job_includes_error_detail(self) -> None:
        """GET for failed job includes error_detail field."""
        job = _make_init_job(state=InitJobState.FAILED)
        job.started_at = datetime(2026, 3, 25, 12, 0, 1)
        job.completed_at = datetime(2026, 3, 25, 12, 0, 5)
        job.error_detail = "Embedding provider credentials are invalid"

        manager = MagicMock()
        manager.get_job.return_value = job
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools/knowledge_base/init")
        body = response.json()

        assert body["state"] == "failed"
        assert body["error_detail"] == "Embedding provider credentials are invalid"


class TestListTools:
    """Tests for GET /tools endpoint (US3)."""

    def test_returns_200(self) -> None:
        """GET /tools returns 200."""

        manager = MagicMock()
        manager.get_all_tool_statuses.return_value = []
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools")

        assert response.status_code == 200

    def test_empty_tools_returns_empty_list_with_zero_total(self) -> None:
        """GET /tools with no tools returns empty list and total=0."""
        manager = MagicMock()
        manager.get_all_tool_statuses.return_value = []
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools")
        body = response.json()

        assert body["tools"] == []
        assert body["total"] == 0

    def test_response_shape_matches_tool_list_response(self) -> None:
        """GET /tools body has 'tools' list and 'total' integer."""
        from holodeck.serve.models import ToolInfoResponse

        statuses = [
            ToolInfoResponse(
                name="kb",
                type="vectorstore",
                supports_init=True,
                init_status="completed",
            ),
            ToolInfoResponse(
                name="api",
                type="mcp",
                supports_init=False,
                init_status=None,
            ),
        ]
        manager = MagicMock()
        manager.get_all_tool_statuses.return_value = statuses
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools")
        body = response.json()

        assert isinstance(body["tools"], list)
        assert len(body["tools"]) == 2
        assert body["total"] == 2

    def test_tool_info_fields_present(self) -> None:
        """Each tool in the response has name, type, supports_init, init_status."""
        from holodeck.serve.models import ToolInfoResponse

        statuses = [
            ToolInfoResponse(
                name="kb",
                type="vectorstore",
                supports_init=True,
                init_status="pending",
            ),
        ]
        manager = MagicMock()
        manager.get_all_tool_statuses.return_value = statuses
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools")
        tool = response.json()["tools"][0]

        assert tool["name"] == "kb"
        assert tool["type"] == "vectorstore"
        assert tool["supports_init"] is True
        assert tool["init_status"] == "pending"

    def test_non_initializable_tool_has_null_init_status(self) -> None:
        """Non-initializable tools have init_status=null in response."""
        from holodeck.serve.models import ToolInfoResponse

        statuses = [
            ToolInfoResponse(
                name="api",
                type="mcp",
                supports_init=False,
                init_status=None,
            ),
        ]
        manager = MagicMock()
        manager.get_all_tool_statuses.return_value = statuses
        app = _make_app_with_mock_manager(manager)
        client = TestClient(app)

        response = client.get("/tools")
        tool = response.json()["tools"][0]

        assert tool["supports_init"] is False
        assert tool["init_status"] is None
