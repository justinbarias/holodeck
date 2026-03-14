"""Tests for MCPToolConfig validation.

Tests the Pydantic validation rules for MCPTool configuration model,
including transport-specific field requirements and value constraints.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.tool import CommandType, MCPTool, TransportType


class TestMCPToolValidConfig:
    """Test valid MCPTool configurations."""

    def test_stdio_minimal_config(self) -> None:
        """Minimal stdio config with required fields only."""
        config = MCPTool(
            name="test_tool",
            description="Test MCP tool",
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-test"],
        )
        assert config.name == "test_tool"
        assert config.transport == TransportType.STDIO
        assert config.command == CommandType.NPX

    def test_stdio_full_config(self) -> None:
        """Stdio config with all optional fields."""
        config = MCPTool(
            name="filesystem",
            description="File operations tool",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-filesystem", "--some-flag"],
            env={"API_KEY": "secret", "DEBUG": "true"},
            env_file=".env.local",
            encoding="utf-8",
            config={"allowed_directories": ["/workspace"]},
            load_tools=True,
            load_prompts=False,
            request_timeout=120,
        )
        assert config.args == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "--some-flag",
        ]
        assert config.env == {"API_KEY": "secret", "DEBUG": "true"}
        assert config.env_file == ".env.local"
        assert config.encoding == "utf-8"
        assert config.config == {"allowed_directories": ["/workspace"]}
        assert config.load_tools is True
        assert config.load_prompts is False
        assert config.request_timeout == 120

    def test_stdio_docker_command(self) -> None:
        """Stdio config with docker command."""
        config = MCPTool(
            name="docker_tool",
            description="Docker MCP tool",
            command=CommandType.DOCKER,
            args=["run", "-i", "--rm", "my-mcp-server:latest"],
        )
        assert config.command == CommandType.DOCKER

    def test_sse_config(self) -> None:
        """Valid SSE transport config."""
        config = MCPTool(
            name="remote_tool",
            description="Remote SSE tool",
            transport=TransportType.SSE,
            url="https://example.com/mcp/sse",
            headers={"Authorization": "Bearer token"},
            timeout=30.0,
            sse_read_timeout=60.0,
        )
        assert config.transport == TransportType.SSE
        assert config.url == "https://example.com/mcp/sse"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_websocket_config(self) -> None:
        """Valid WebSocket transport config."""
        config = MCPTool(
            name="ws_tool",
            description="WebSocket MCP tool",
            transport=TransportType.WEBSOCKET,
            url="wss://example.com/mcp/ws",
        )
        assert config.transport == TransportType.WEBSOCKET
        assert config.url == "wss://example.com/mcp/ws"

    def test_http_config(self) -> None:
        """Valid HTTP transport config."""
        config = MCPTool(
            name="http_tool",
            description="HTTP MCP tool",
            transport=TransportType.HTTP,
            url="https://example.com/mcp/stream",
            terminate_on_close=True,
        )
        assert config.transport == TransportType.HTTP
        assert config.terminate_on_close is True

    def test_localhost_http_allowed(self) -> None:
        """HTTP URLs for localhost can use http:// scheme."""
        config = MCPTool(
            name="local_tool",
            description="Local HTTP tool",
            transport=TransportType.HTTP,
            url="http://localhost:8080/mcp",
        )
        assert config.url == "http://localhost:8080/mcp"

    def test_localhost_127_allowed(self) -> None:
        """HTTP URLs for 127.0.0.1 can use http:// scheme."""
        config = MCPTool(
            name="local_tool",
            description="Local HTTP tool",
            transport=TransportType.HTTP,
            url="http://127.0.0.1:8080/mcp",
        )
        assert config.url == "http://127.0.0.1:8080/mcp"


class TestMCPToolInvalidConfig:
    """Test invalid MCPTool configurations."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_match"),
        [
            pytest.param(
                {"transport": TransportType.STDIO},
                "'command' is required for stdio transport",
                id="stdio_without_command",
            ),
            pytest.param(
                {"transport": TransportType.SSE},
                "'url' is required for sse transport",
                id="sse_without_url",
            ),
            pytest.param(
                {"transport": TransportType.WEBSOCKET},
                "'url' is required for websocket transport",
                id="websocket_without_url",
            ),
            pytest.param(
                {"transport": TransportType.HTTP},
                "'url' is required for http transport",
                id="http_without_url",
            ),
            pytest.param(
                {
                    "transport": TransportType.HTTP,
                    "url": "http://remote-server.com/mcp",
                },
                "must use https://",
                id="remote_http_requires_https",
            ),
            pytest.param(
                {"command": CommandType.NPX, "request_timeout": -1},
                "request_timeout must be positive",
                id="negative_request_timeout",
            ),
            pytest.param(
                {"command": CommandType.NPX, "request_timeout": 0},
                "request_timeout must be positive",
                id="zero_request_timeout",
            ),
            pytest.param(
                {"transport": TransportType.HTTP, "url": "ftp://example.com/mcp"},
                "must use https://",
                id="invalid_url_scheme",
            ),
        ],
    )
    def test_invalid_config_raises_validation_error(
        self, kwargs: dict, expected_match: str
    ) -> None:
        """Test that invalid MCPTool configurations raise ValidationError."""
        with pytest.raises(ValidationError, match=expected_match):
            MCPTool(name="test", description="Test", **kwargs)

    def test_invalid_command_type_raises(self) -> None:
        """Invalid command type is rejected."""
        with pytest.raises(ValidationError):
            MCPTool(
                name="test",
                description="Test",
                command="bash",  # type: ignore[arg-type]
            )


class TestMCPToolDefaults:
    """Test default values for MCPTool."""

    @pytest.mark.parametrize(
        ("field", "expected_value"),
        [
            pytest.param(
                "transport", TransportType.STDIO, id="default_transport_stdio"
            ),
            pytest.param("load_tools", True, id="default_load_tools_true"),
            pytest.param("load_prompts", True, id="default_load_prompts_true"),
            pytest.param("request_timeout", 60, id="default_request_timeout_60"),
            pytest.param("type", "mcp", id="default_type_mcp"),
        ],
    )
    def test_default_values(self, field: str, expected_value: object) -> None:
        """Test that MCPTool defaults are correct."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        assert getattr(config, field) == expected_value
