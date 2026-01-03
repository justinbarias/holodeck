"""Contract tests for REST API OpenAPI spec compliance.

Tests for:
- T030: OpenAPI spec compliance validation
- Endpoint paths match openapi.yaml
- Request/response schemas match
- Error response formats (RFC 7807)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    pass


# =============================================================================
# OpenAPI Spec Loading
# =============================================================================


@pytest.fixture
def openapi_spec() -> dict:
    """Load the OpenAPI specification from contracts/openapi.yaml."""
    spec_path = "specs/017-agent-local-server/contracts/openapi.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# T030: Contract tests for OpenAPI spec compliance
# =============================================================================


class TestOpenAPIEndpointPaths:
    """Tests that API endpoints match OpenAPI specification."""

    def test_chat_sync_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that POST /agent/{agent_name}/chat is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/agent/{agent_name}/chat" in paths
        assert "post" in paths["/agent/{agent_name}/chat"]

    def test_chat_stream_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that POST /agent/{agent_name}/chat/stream is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/agent/{agent_name}/chat/stream" in paths
        assert "post" in paths["/agent/{agent_name}/chat/stream"]

    def test_delete_session_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that DELETE /sessions/{session_id} is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/sessions/{session_id}" in paths
        assert "delete" in paths["/sessions/{session_id}"]

    def test_health_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that GET /health is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/health" in paths
        assert "get" in paths["/health"]

    def test_health_agent_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that GET /health/agent is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/health/agent" in paths
        assert "get" in paths["/health/agent"]

    def test_ready_endpoint_defined(self, openapi_spec: dict) -> None:
        """Test that GET /ready is defined."""
        paths = openapi_spec.get("paths", {})
        assert "/ready" in paths
        assert "get" in paths["/ready"]


class TestOpenAPIRequestSchemas:
    """Tests that request schemas match OpenAPI specification."""

    def test_chat_request_schema_has_required_fields(self, openapi_spec: dict) -> None:
        """Test ChatRequest schema has required message field."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_request = schemas.get("ChatRequest", {})

        assert "message" in chat_request.get("required", [])

    def test_chat_request_message_constraints(self, openapi_spec: dict) -> None:
        """Test ChatRequest.message has correct constraints."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_request = schemas.get("ChatRequest", {})
        message_prop = chat_request.get("properties", {}).get("message", {})

        assert message_prop.get("type") == "string"
        assert message_prop.get("minLength") == 1
        assert message_prop.get("maxLength") == 10000

    def test_chat_request_session_id_optional(self, openapi_spec: dict) -> None:
        """Test ChatRequest.session_id is optional."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_request = schemas.get("ChatRequest", {})

        # session_id should NOT be in required
        assert "session_id" not in chat_request.get("required", [])

    def test_chat_request_files_constraints(self, openapi_spec: dict) -> None:
        """Test ChatRequest.files has correct constraints."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_request = schemas.get("ChatRequest", {})
        files_prop = chat_request.get("properties", {}).get("files", {})

        assert files_prop.get("type") == "array"
        assert files_prop.get("maxItems") == 10

    def test_file_content_schema_has_required_fields(self, openapi_spec: dict) -> None:
        """Test FileContent schema has required content and mime_type."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        file_content = schemas.get("FileContent", {})

        required = file_content.get("required", [])
        assert "content" in required
        assert "mime_type" in required

    def test_file_content_mime_type_enum(self, openapi_spec: dict) -> None:
        """Test FileContent.mime_type has correct enum values."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        file_content = schemas.get("FileContent", {})
        mime_type_prop = file_content.get("properties", {}).get("mime_type", {})

        expected_mime_types = {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/csv",
            "text/markdown",
        }

        actual_mime_types = set(mime_type_prop.get("enum", []))
        assert actual_mime_types == expected_mime_types


class TestOpenAPIResponseSchemas:
    """Tests that response schemas match OpenAPI specification."""

    def test_chat_response_schema_has_required_fields(self, openapi_spec: dict) -> None:
        """Test ChatResponse schema has required fields."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_response = schemas.get("ChatResponse", {})

        required = chat_response.get("required", [])
        assert "message_id" in required
        assert "content" in required
        assert "session_id" in required
        assert "execution_time_ms" in required

    def test_chat_response_has_tool_calls(self, openapi_spec: dict) -> None:
        """Test ChatResponse schema has tool_calls field."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        chat_response = schemas.get("ChatResponse", {})
        properties = chat_response.get("properties", {})

        assert "tool_calls" in properties

    def test_health_response_schema(self, openapi_spec: dict) -> None:
        """Test HealthResponse schema structure."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        health_response = schemas.get("HealthResponse", {})

        assert "status" in health_response.get("required", [])
        properties = health_response.get("properties", {})
        assert "status" in properties
        assert "agent_name" in properties
        assert "agent_ready" in properties
        assert "active_sessions" in properties


class TestOpenAPIErrorResponses:
    """Tests that error responses follow RFC 7807 ProblemDetail format."""

    def test_problem_detail_schema_exists(self, openapi_spec: dict) -> None:
        """Test ProblemDetail schema is defined."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        assert "ProblemDetail" in schemas

    def test_problem_detail_has_required_fields(self, openapi_spec: dict) -> None:
        """Test ProblemDetail has required type, title, status."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        problem_detail = schemas.get("ProblemDetail", {})

        required = problem_detail.get("required", [])
        assert "type" in required
        assert "title" in required
        assert "status" in required

    def test_problem_detail_has_optional_fields(self, openapi_spec: dict) -> None:
        """Test ProblemDetail has optional detail and instance."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        problem_detail = schemas.get("ProblemDetail", {})
        properties = problem_detail.get("properties", {})

        assert "detail" in properties
        assert "instance" in properties

    def test_chat_sync_400_uses_problem_detail(self, openapi_spec: dict) -> None:
        """Test POST /agent/{name}/chat 400 response uses ProblemDetail."""
        paths = openapi_spec.get("paths", {})
        chat_endpoint = paths.get("/agent/{agent_name}/chat", {}).get("post", {})
        response_400 = chat_endpoint.get("responses", {}).get("400", {})
        content = response_400.get("content", {})

        assert "application/problem+json" in content

    def test_chat_sync_404_uses_problem_detail(self, openapi_spec: dict) -> None:
        """Test POST /agent/{name}/chat 404 response uses ProblemDetail."""
        paths = openapi_spec.get("paths", {})
        chat_endpoint = paths.get("/agent/{agent_name}/chat", {}).get("post", {})
        response_404 = chat_endpoint.get("responses", {}).get("404", {})
        content = response_404.get("content", {})

        assert "application/problem+json" in content

    def test_chat_sync_503_uses_problem_detail(self, openapi_spec: dict) -> None:
        """Test POST /agent/{name}/chat 503 response uses ProblemDetail."""
        paths = openapi_spec.get("paths", {})
        chat_endpoint = paths.get("/agent/{agent_name}/chat", {}).get("post", {})
        response_503 = chat_endpoint.get("responses", {}).get("503", {})
        content = response_503.get("content", {})

        assert "application/problem+json" in content


class TestOpenAPIStreamingEndpoint:
    """Tests for streaming endpoint specification."""

    def test_chat_stream_returns_event_stream(self, openapi_spec: dict) -> None:
        """Test POST /agent/{name}/chat/stream returns text/event-stream."""
        paths = openapi_spec.get("paths", {})
        stream_endpoint = paths.get("/agent/{agent_name}/chat/stream", {}).get(
            "post", {}
        )
        response_200 = stream_endpoint.get("responses", {}).get("200", {})
        content = response_200.get("content", {})

        assert "text/event-stream" in content

    def test_chat_stream_supports_json_body(self, openapi_spec: dict) -> None:
        """Test streaming endpoint accepts application/json body."""
        paths = openapi_spec.get("paths", {})
        stream_endpoint = paths.get("/agent/{agent_name}/chat/stream", {}).get(
            "post", {}
        )
        request_body = stream_endpoint.get("requestBody", {}).get("content", {})

        assert "application/json" in request_body

    def test_chat_stream_supports_multipart(self, openapi_spec: dict) -> None:
        """Test streaming endpoint accepts multipart/form-data body."""
        paths = openapi_spec.get("paths", {})
        stream_endpoint = paths.get("/agent/{agent_name}/chat/stream", {}).get(
            "post", {}
        )
        request_body = stream_endpoint.get("requestBody", {}).get("content", {})

        assert "multipart/form-data" in request_body


class TestOpenAPIMultipartSchema:
    """Tests for multipart form-data schema."""

    def test_chat_request_multipart_schema_exists(self, openapi_spec: dict) -> None:
        """Test ChatRequestMultipart schema is defined."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        assert "ChatRequestMultipart" in schemas

    def test_chat_request_multipart_has_message(self, openapi_spec: dict) -> None:
        """Test ChatRequestMultipart has required message field."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        multipart = schemas.get("ChatRequestMultipart", {})

        assert "message" in multipart.get("required", [])

    def test_chat_request_multipart_files_binary(self, openapi_spec: dict) -> None:
        """Test ChatRequestMultipart.files are binary format."""
        schemas = openapi_spec.get("components", {}).get("schemas", {})
        multipart = schemas.get("ChatRequestMultipart", {})
        files_prop = multipart.get("properties", {}).get("files", {})
        items = files_prop.get("items", {})

        assert items.get("format") == "binary"
