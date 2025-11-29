"""Tests for MCP plugin factory.

Tests factory function for creating MCP plugins, including:
- Stdio transport plugin creation with mocked SK
- Environment variable resolution
- env_file loading and precedence
- Config passthrough via MCP_CONFIG
"""

import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.errors import ConfigError
from holodeck.models.tool import CommandType, MCPTool, TransportType
from holodeck.tools.mcp.errors import MCPConfigError
from holodeck.tools.mcp.factory import _resolve_env_vars, create_mcp_plugin


@pytest.fixture
def mock_mcp_stdio_plugin() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Fixture providing a mocked MCPStdioPlugin class and instance.

    Yields:
        Tuple of (mock_plugin_class, mock_plugin_instance)
    """
    mock_plugin_class = MagicMock()
    mock_plugin_instance = MagicMock()
    mock_plugin_class.return_value = mock_plugin_instance

    with patch.dict(
        sys.modules,
        {"semantic_kernel.connectors.mcp": MagicMock(MCPStdioPlugin=mock_plugin_class)},
    ):
        yield mock_plugin_class, mock_plugin_instance


@pytest.fixture
def create_plugin_with_mock(
    mock_mcp_stdio_plugin: tuple[MagicMock, MagicMock],
) -> Any:
    """Fixture providing the create_mcp_plugin function with mocked SK.

    Returns:
        The create_mcp_plugin function that uses mocked SK imports.
    """
    # Re-import to pick up the mock
    from holodeck.tools.mcp.factory import create_mcp_plugin as create_plugin

    return create_plugin


class TestCreateMCPPluginNotImplemented:
    """Test factory for transports not yet implemented."""

    def test_sse_transport_not_yet_implemented(self) -> None:
        """SSE transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.SSE,
            url="https://example.com/sse",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"

    def test_websocket_transport_not_yet_implemented(self) -> None:
        """WebSocket transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.WEBSOCKET,
            url="wss://example.com/ws",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"

    def test_http_transport_not_yet_implemented(self) -> None:
        """HTTP transport should raise MCPConfigError (not yet implemented)."""
        config = MCPTool(
            name="test",
            description="Test",
            transport=TransportType.HTTP,
            url="https://example.com/stream",
        )
        with pytest.raises(MCPConfigError) as exc_info:
            create_mcp_plugin(config)
        assert "not yet implemented" in str(exc_info.value)
        assert exc_info.value.field == "transport"


class TestCreateMCPPluginStdio:
    """Test stdio transport plugin creation with mocked SK."""

    def test_stdio_creates_plugin_with_correct_args(
        self,
        mock_mcp_stdio_plugin: tuple[MagicMock, MagicMock],
        create_plugin_with_mock: Any,
    ) -> None:
        """Stdio transport creates MCPStdioPlugin with correct constructor args."""
        mock_plugin_class, mock_plugin_instance = mock_mcp_stdio_plugin

        # User must specify full args including the server package
        config = MCPTool(
            name="filesystem",
            description="File ops",
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
        result = create_plugin_with_mock(config)

        mock_plugin_class.assert_called_once_with(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env=None,
            encoding="utf-8",
        )
        assert result == mock_plugin_instance

    def test_stdio_includes_additional_args(
        self,
        mock_mcp_stdio_plugin: tuple[MagicMock, MagicMock],
        create_plugin_with_mock: Any,
    ) -> None:
        """Stdio transport passes args directly to plugin."""
        mock_plugin_class, _ = mock_mcp_stdio_plugin

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            args=["-y", "test-server", "--verbose"],
        )
        create_plugin_with_mock(config)

        call_args = mock_plugin_class.call_args
        # Args are passed directly as configured by user
        assert call_args.kwargs["args"] == ["-y", "test-server", "--verbose"]

    def test_stdio_docker_command(
        self,
        mock_mcp_stdio_plugin: tuple[MagicMock, MagicMock],
        create_plugin_with_mock: Any,
    ) -> None:
        """Stdio transport uses docker command correctly."""
        mock_plugin_class, _ = mock_mcp_stdio_plugin

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.DOCKER,
            args=["run", "-i", "--rm", "my-container:latest"],
        )
        create_plugin_with_mock(config)

        call_args = mock_plugin_class.call_args
        assert call_args.kwargs["command"] == "docker"

    def test_stdio_custom_encoding(
        self,
        mock_mcp_stdio_plugin: tuple[MagicMock, MagicMock],
        create_plugin_with_mock: Any,
    ) -> None:
        """Stdio transport respects custom encoding."""
        mock_plugin_class, _ = mock_mcp_stdio_plugin

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            encoding="latin-1",
        )
        create_plugin_with_mock(config)

        call_args = mock_plugin_class.call_args
        assert call_args.kwargs["encoding"] == "latin-1"

    def test_stdio_import_error_raises_config_error(self) -> None:
        """Missing SK MCP module raises helpful MCPConfigError."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )

        # Patch the import to raise ImportError by using builtins.__import__
        import builtins

        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "semantic_kernel.connectors.mcp":
                raise ImportError("No module named 'semantic_kernel.connectors.mcp'")
            return original_import(name, globals_, locals_, fromlist, level)

        with patch.object(builtins, "__import__", mock_import):
            with pytest.raises(MCPConfigError) as exc_info:
                create_mcp_plugin(config)

            assert "not installed" in str(exc_info.value)


class TestResolveEnvVars:
    """Test environment variable resolution function."""

    def test_empty_config_returns_empty_dict(self) -> None:
        """Config with no env settings returns empty dict."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
        )
        result = _resolve_env_vars(config)
        assert result == {}

    def test_static_env_values_passed_through(self) -> None:
        """Static env values without ${} are passed through."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env={"KEY1": "value1", "KEY2": "value2"},
        )
        result = _resolve_env_vars(config)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_env_var_substitution(self) -> None:
        """${VAR} patterns are substituted from os.environ."""
        with patch.dict(os.environ, {"MY_SECRET": "secret123"}):
            config = MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                env={"API_KEY": "${MY_SECRET}"},
            )
            result = _resolve_env_vars(config)
            assert result == {"API_KEY": "secret123"}

    def test_multiple_vars_in_same_value(self) -> None:
        """Multiple ${VAR} in same value are all substituted."""
        with patch.dict(os.environ, {"USER": "admin", "HOST": "localhost"}):
            config = MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                env={"CONNECTION": "${USER}@${HOST}"},
            )
            result = _resolve_env_vars(config)
            assert result == {"CONNECTION": "admin@localhost"}

    def test_missing_env_var_raises_config_error(self) -> None:
        """Missing environment variable raises ConfigError (fail-fast)."""
        # Ensure the variable doesn't exist
        env_copy = os.environ.copy()
        if "NONEXISTENT_VAR" in env_copy:
            del env_copy["NONEXISTENT_VAR"]

        with patch.dict(os.environ, env_copy, clear=True):
            config = MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                env={"KEY": "${NONEXISTENT_VAR}"},
            )
            with pytest.raises(ConfigError) as exc_info:
                _resolve_env_vars(config)
            assert "NONEXISTENT_VAR" in str(exc_info.value)

    def test_mixed_static_and_dynamic_env(self) -> None:
        """Mix of static values and ${VAR} patterns works correctly."""
        with patch.dict(os.environ, {"SECRET": "xyz"}):
            config = MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                env={"STATIC": "static_value", "DYNAMIC": "${SECRET}"},
            )
            result = _resolve_env_vars(config)
            assert result == {"STATIC": "static_value", "DYNAMIC": "xyz"}


class TestEnvFileLoading:
    """Test env_file loading functionality."""

    def test_env_file_loaded(self, tmp_path: Path) -> None:
        """Variables from env_file are loaded."""
        env_file = tmp_path / ".env"
        env_file.write_text("VAR1=value1\nVAR2=value2\n")

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file=str(env_file),
        )
        result = _resolve_env_vars(config)
        assert result == {"VAR1": "value1", "VAR2": "value2"}

    def test_env_file_with_comments(self, tmp_path: Path) -> None:
        """Comments and blank lines in env_file are ignored."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\nVAR1=value1\n\n# Another comment\nVAR2=value2\n"
        )

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file=str(env_file),
        )
        result = _resolve_env_vars(config)
        assert result == {"VAR1": "value1", "VAR2": "value2"}

    def test_explicit_env_overrides_env_file(self, tmp_path: Path) -> None:
        """Explicit env values override env_file values."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=from_file\nOTHER=file_value\n")

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file=str(env_file),
            env={"KEY": "from_explicit"},
        )
        result = _resolve_env_vars(config)
        assert result["KEY"] == "from_explicit"
        assert result["OTHER"] == "file_value"

    def test_missing_env_file_raises_config_error(self) -> None:
        """Missing env_file raises ConfigError."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file="/nonexistent/path/.env",
        )
        with pytest.raises(ConfigError) as exc_info:
            _resolve_env_vars(config)
        assert "env_file" in str(exc_info.value) or "Cannot read" in str(exc_info.value)

    def test_env_file_combined_with_substitution(self, tmp_path: Path) -> None:
        """env_file vars combined with ${VAR} substitution in explicit env."""
        env_file = tmp_path / ".env"
        env_file.write_text("BASE_URL=https://api.example.com\n")

        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = MCPTool(
                name="test",
                description="Test",
                command=CommandType.NPX,
                env_file=str(env_file),
                env={"AUTH": "Bearer ${API_KEY}"},
            )
            result = _resolve_env_vars(config)
            assert result == {
                "BASE_URL": "https://api.example.com",
                "AUTH": "Bearer secret123",
            }


class TestConfigPassthrough:
    """Test config passthrough via MCP_CONFIG env var."""

    def test_config_dict_serialized_to_env(self) -> None:
        """Config dict is JSON-serialized to MCP_CONFIG env var."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            config={"allowed_directories": ["/workspace"], "debug": True},
        )
        result = _resolve_env_vars(config)
        assert "MCP_CONFIG" in result
        parsed = json.loads(result["MCP_CONFIG"])
        assert parsed == {"allowed_directories": ["/workspace"], "debug": True}

    def test_config_with_nested_objects(self) -> None:
        """Nested config objects are properly serialized."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            config={
                "database": {"host": "localhost", "port": 5432},
                "features": ["a", "b", "c"],
            },
        )
        result = _resolve_env_vars(config)
        parsed = json.loads(result["MCP_CONFIG"])
        assert parsed["database"]["host"] == "localhost"
        assert parsed["features"] == ["a", "b", "c"]

    def test_config_combined_with_env(self) -> None:
        """Config passthrough works alongside explicit env vars."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env={"OTHER_VAR": "value"},
            config={"setting": "enabled"},
        )
        result = _resolve_env_vars(config)
        assert result["OTHER_VAR"] == "value"
        assert "MCP_CONFIG" in result
        assert json.loads(result["MCP_CONFIG"]) == {"setting": "enabled"}

    def test_empty_config_not_added(self) -> None:
        """Empty/None config does not add MCP_CONFIG."""
        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            config=None,
        )
        result = _resolve_env_vars(config)
        assert "MCP_CONFIG" not in result

    def test_config_combined_with_env_file(self, tmp_path: Path) -> None:
        """Config passthrough works with env_file."""
        env_file = tmp_path / ".env"
        env_file.write_text("FROM_FILE=yes\n")

        config = MCPTool(
            name="test",
            description="Test",
            command=CommandType.NPX,
            env_file=str(env_file),
            config={"from_config": True},
        )
        result = _resolve_env_vars(config)
        assert result["FROM_FILE"] == "yes"
        assert json.loads(result["MCP_CONFIG"]) == {"from_config": True}
