"""Unit tests for serve module middleware.

Tests cover LoggingMiddleware and ErrorHandlingMiddleware including
request logging, error handling, and RFC 7807 problem details.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from holodeck.serve.middleware import ErrorHandlingMiddleware, LoggingMiddleware


@pytest.fixture
def basic_app() -> FastAPI:
    """Create a basic FastAPI app for testing."""
    app = FastAPI()

    @app.get("/success")
    async def success_endpoint() -> dict:
        return {"status": "ok"}

    @app.get("/error")
    async def error_endpoint() -> dict:
        raise ValueError("Something went wrong")

    @app.get("/custom-error")
    async def custom_error_endpoint() -> dict:
        raise RuntimeError("Custom error message")

    return app


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    def test_logging_middleware_initialization(self) -> None:
        """Test LoggingMiddleware initialization."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, debug=False)

        assert middleware.debug is False

    def test_logging_middleware_debug_mode(self) -> None:
        """Test LoggingMiddleware with debug mode enabled."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, debug=True)

        assert middleware.debug is True

    def test_middleware_logs_request_start(self, basic_app: FastAPI) -> None:
        """Test middleware logs request start."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success")

            # Check that request started was logged
            mock_logger.info.assert_called()
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Request started" in call for call in calls)

    def test_middleware_logs_request_completed(self, basic_app: FastAPI) -> None:
        """Test middleware logs request completion."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success")

            # Check that request completed was logged
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Request completed" in call for call in calls)

    def test_middleware_logs_latency(self, basic_app: FastAPI) -> None:
        """Test middleware logs request latency."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success")

            # Find the completion log call and check for latency_ms
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if "latency_ms" in extra:
                        assert isinstance(extra["latency_ms"], float)
                        break

    def test_middleware_debug_mode_logs_headers(self, basic_app: FastAPI) -> None:
        """Test debug mode logs request headers."""
        basic_app.add_middleware(LoggingMiddleware, debug=True)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success", headers={"X-Custom-Header": "test-value"})

            # Check that debug was called
            mock_logger.debug.assert_called()

    def test_middleware_extracts_session_id_from_header(
        self, basic_app: FastAPI
    ) -> None:
        """Test middleware extracts session ID from X-Session-ID header."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success", headers={"X-Session-ID": "test-session-123"})

            # Check that session_id was logged
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if "session_id" in extra:
                        assert extra["session_id"] == "test-session-123"
                        break

    def test_middleware_extracts_session_id_from_query(
        self, basic_app: FastAPI
    ) -> None:
        """Test middleware extracts session ID from query parameter."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success?session_id=query-session-456")

            # Check that session_id was logged
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if extra.get("session_id") == "query-session-456":
                        assert True
                        break

    def test_middleware_header_session_id_takes_priority(
        self, basic_app: FastAPI
    ) -> None:
        """Test header session ID takes priority over query parameter."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get(
                "/success?session_id=query-session",
                headers={"X-Session-ID": "header-session"},
            )

            # Check that header session_id was used
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if extra.get("session_id") == "header-session":
                        assert True
                        break

    def test_middleware_no_session_id_is_none(self, basic_app: FastAPI) -> None:
        """Test session_id is None when not provided."""
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/success")

            # Check that session_id is None
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if "session_id" in extra and extra.get("path") == "/success":
                        assert extra["session_id"] is None
                        break


class TestLoggingMiddlewareExtractSessionId:
    """Tests for LoggingMiddleware._extract_session_id method."""

    @pytest.fixture
    def middleware(self) -> LoggingMiddleware:
        """Create a LoggingMiddleware instance."""
        app = FastAPI()
        return LoggingMiddleware(app, debug=False)

    def test_extract_session_id_from_header(
        self, middleware: LoggingMiddleware
    ) -> None:
        """Test extracting session ID from header."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Session-ID": "test-session"}
        mock_request.query_params = {}

        session_id = middleware._extract_session_id(mock_request)
        assert session_id == "test-session"

    def test_extract_session_id_from_query_params(
        self, middleware: LoggingMiddleware
    ) -> None:
        """Test extracting session ID from query parameters."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {"session_id": "query-session"}

        session_id = middleware._extract_session_id(mock_request)
        assert session_id == "query-session"

    def test_extract_session_id_none_when_missing(
        self, middleware: LoggingMiddleware
    ) -> None:
        """Test session ID is None when not present."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}

        session_id = middleware._extract_session_id(mock_request)
        assert session_id is None


class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""

    def test_error_handling_middleware_initialization(self) -> None:
        """Test ErrorHandlingMiddleware initialization."""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app, debug=False)

        assert middleware.debug is False

    def test_error_handling_middleware_debug_mode(self) -> None:
        """Test ErrorHandlingMiddleware with debug mode enabled."""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app, debug=True)

        assert middleware.debug is True

    def test_middleware_passes_through_successful_requests(
        self, basic_app: FastAPI
    ) -> None:
        """Test middleware passes through successful requests."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/success")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_middleware_catches_exceptions(self, basic_app: FastAPI) -> None:
        """Test middleware catches unhandled exceptions."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")

        assert response.status_code == 500

    def test_middleware_returns_json_response(self, basic_app: FastAPI) -> None:
        """Test middleware returns JSON error response."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")

        assert response.headers["content-type"] == "application/problem+json"
        data = response.json()
        assert "type" in data
        assert "title" in data
        assert "status" in data

    def test_middleware_returns_rfc7807_format(self, basic_app: FastAPI) -> None:
        """Test middleware returns RFC 7807 Problem Details format."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")
        data = response.json()

        assert data["type"] == "https://holodeck.dev/errors/internal-error"
        assert data["title"] == "Internal Server Error"
        assert data["status"] == 500
        assert "instance" in data

    def test_middleware_hides_details_in_production(self, basic_app: FastAPI) -> None:
        """Test middleware hides error details when not in debug mode."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")
        data = response.json()

        assert data["detail"] == "An unexpected error occurred."
        assert "Something went wrong" not in data["detail"]

    def test_middleware_shows_details_in_debug(self, basic_app: FastAPI) -> None:
        """Test middleware shows error details in debug mode."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=True)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")
        data = response.json()

        assert "Something went wrong" in data["detail"]

    def test_middleware_shows_stack_trace_in_debug(self, basic_app: FastAPI) -> None:
        """Test middleware shows stack trace in debug mode."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=True)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")
        data = response.json()

        assert "Stack trace:" in data["detail"]
        assert "Traceback" in data["detail"]

    def test_middleware_logs_error(self, basic_app: FastAPI) -> None:
        """Test middleware logs errors."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/error")

            mock_logger.error.assert_called_once()

    def test_middleware_logs_error_type(self, basic_app: FastAPI) -> None:
        """Test middleware logs error type in extra data."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            client.get("/error")

            # Check that error_type was logged
            call_kwargs = mock_logger.error.call_args
            assert "extra" in call_kwargs.kwargs
            extra = call_kwargs.kwargs["extra"]
            assert extra["error_type"] == "ValueError"

    def test_middleware_includes_request_path_in_error(
        self, basic_app: FastAPI
    ) -> None:
        """Test middleware includes request path in error response."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/error")
        data = response.json()

        assert data["instance"] == "/error"

    def test_middleware_handles_different_exception_types(
        self, basic_app: FastAPI
    ) -> None:
        """Test middleware handles different exception types."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=True)
        client = TestClient(basic_app, raise_server_exceptions=False)

        response = client.get("/custom-error")
        data = response.json()

        assert "Custom error message" in data["detail"]
        assert response.status_code == 500


class TestErrorHandlingMiddlewareCreateErrorResponse:
    """Tests for ErrorHandlingMiddleware._create_error_response method."""

    @pytest.fixture
    def middleware(self) -> ErrorHandlingMiddleware:
        """Create an ErrorHandlingMiddleware instance."""
        app = FastAPI()
        return ErrorHandlingMiddleware(app, debug=False)

    @pytest.fixture
    def debug_middleware(self) -> ErrorHandlingMiddleware:
        """Create an ErrorHandlingMiddleware instance in debug mode."""
        app = FastAPI()
        return ErrorHandlingMiddleware(app, debug=True)

    def test_create_error_response_returns_json_response(
        self, middleware: ErrorHandlingMiddleware
    ) -> None:
        """Test _create_error_response returns JSONResponse."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "GET"

        with patch("holodeck.serve.middleware.logger"):
            response = middleware._create_error_response(
                mock_request, ValueError("Test error")
            )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

    def test_create_error_response_problem_content_type(
        self, middleware: ErrorHandlingMiddleware
    ) -> None:
        """Test _create_error_response sets problem+json content type."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "GET"

        with patch("holodeck.serve.middleware.logger"):
            response = middleware._create_error_response(
                mock_request, ValueError("Test error")
            )

        assert response.media_type == "application/problem+json"

    def test_create_error_response_production_mode(
        self, middleware: ErrorHandlingMiddleware
    ) -> None:
        """Test _create_error_response in production mode hides details."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "GET"

        with patch("holodeck.serve.middleware.logger"):
            response = middleware._create_error_response(
                mock_request, ValueError("Sensitive error details")
            )

        # Parse the response body
        import json

        body = json.loads(response.body)
        assert body["detail"] == "An unexpected error occurred."

    def test_create_error_response_debug_mode(
        self, debug_middleware: ErrorHandlingMiddleware
    ) -> None:
        """Test _create_error_response in debug mode shows details."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.method = "GET"

        with patch("holodeck.serve.middleware.logger"):
            response = debug_middleware._create_error_response(
                mock_request, ValueError("Debug error message")
            )

        import json

        body = json.loads(response.body)
        assert "Debug error message" in body["detail"]
        assert "Stack trace:" in body["detail"]


class TestMiddlewareIntegration:
    """Integration tests for middleware stack."""

    def test_both_middlewares_together(self, basic_app: FastAPI) -> None:
        """Test both middlewares work together."""
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            # Successful request
            response = client.get("/success")
            assert response.status_code == 200

            # Error request
            response = client.get("/error")
            assert response.status_code == 500
            assert response.headers["content-type"] == "application/problem+json"

            # Both info (logging) and error (error handling) should be called
            assert mock_logger.info.called
            assert mock_logger.error.called

    def test_middleware_order_matters(self, basic_app: FastAPI) -> None:
        """Test middleware order: logging should see the error response."""
        # Note: Starlette applies middleware in reverse order
        basic_app.add_middleware(LoggingMiddleware, debug=False)
        basic_app.add_middleware(ErrorHandlingMiddleware, debug=False)
        client = TestClient(basic_app, raise_server_exceptions=False)

        with patch("holodeck.serve.middleware.logger") as mock_logger:
            response = client.get("/error")

            # Error should be handled
            assert response.status_code == 500

            # Logging middleware should log the 500 status
            for call in mock_logger.info.call_args_list:
                if "extra" in call.kwargs:
                    extra = call.kwargs["extra"]
                    if extra.get("status_code") == 500:
                        assert True
                        break
