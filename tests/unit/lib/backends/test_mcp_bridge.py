"""Tests for Phase 6 MCP Bridge (T046-T050).

Tests for build_claude_mcp_configs() which translates HoloDeck MCPTool
configurations into Claude Agent SDK McpStdioServerConfig format.
"""

from __future__ import annotations

import json
import logging

import pytest

from holodeck.lib.backends.mcp_bridge import build_claude_mcp_configs
from holodeck.models.tool import CommandType, MCPTool, TransportType

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "filesystem",
    description: str = "File operations",
    command: CommandType = CommandType.NPX,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    env_file: str | None = None,
    config: dict | None = None,
    transport: TransportType = TransportType.STDIO,
) -> MCPTool:
    """Build a minimal MCPTool config for testing."""
    kwargs: dict = {
        "name": name,
        "description": description,
        "type": "mcp",
        "transport": transport,
    }
    if transport == TransportType.STDIO:
        kwargs["command"] = command
    elif transport in (TransportType.SSE, TransportType.WEBSOCKET, TransportType.HTTP):
        kwargs["url"] = "https://example.com/mcp"
    if args is not None:
        kwargs["args"] = args
    if env is not None:
        kwargs["env"] = env
    if env_file is not None:
        kwargs["env_file"] = env_file
    if config is not None:
        kwargs["config"] = config
    return MCPTool(**kwargs)


# ---------------------------------------------------------------------------
# T046: TestBuildClaudeMcpConfigsHappyPath
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildClaudeMcpConfigsHappyPath:
    """Happy path: single stdio tool produces correct output shape."""

    def test_single_stdio_tool_returns_dict_keyed_by_name(self) -> None:
        """Single stdio tool produces a dict with tool name as key."""
        tool = _make_mcp_tool(name="fs_server", command=CommandType.NPX)
        result = build_claude_mcp_configs([tool])

        assert "fs_server" in result
        assert len(result) == 1

    def test_output_has_correct_shape(self) -> None:
        """Output entry contains type, command, and args fields."""
        tool = _make_mcp_tool(
            name="myserver",
            command=CommandType.UVX,
            args=["-y", "some-package"],
        )
        result = build_claude_mcp_configs([tool])

        entry = result["myserver"]
        assert entry["command"] == "uvx"
        assert entry["args"] == ["-y", "some-package"]

    def test_command_is_string_not_enum(self) -> None:
        """Command value is a plain string, not the CommandType enum."""
        tool = _make_mcp_tool(command=CommandType.DOCKER)
        result = build_claude_mcp_configs([tool])

        entry = result["filesystem"]
        assert isinstance(entry["command"], str)
        assert entry["command"] == "docker"


# ---------------------------------------------------------------------------
# T047: TestBuildClaudeMcpConfigsMultiple
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildClaudeMcpConfigsMultiple:
    """Multiple stdio tools produce keyed entries without collisions."""

    def test_two_tools_produce_two_entries(self) -> None:
        """Two distinct tools produce two separate dict entries."""
        tool_a = _make_mcp_tool(name="server_a", command=CommandType.NPX)
        tool_b = _make_mcp_tool(name="server_b", command=CommandType.UVX)
        result = build_claude_mcp_configs([tool_a, tool_b])

        assert len(result) == 2
        assert "server_a" in result
        assert "server_b" in result

    def test_entries_have_independent_configs(self) -> None:
        """Each entry has its own command and args."""
        tool_a = _make_mcp_tool(name="a", command=CommandType.NPX, args=["pkg-a"])
        tool_b = _make_mcp_tool(
            name="b", command=CommandType.DOCKER, args=["run", "img"]
        )
        result = build_claude_mcp_configs([tool_a, tool_b])

        assert result["a"]["command"] == "npx"
        assert result["a"]["args"] == ["pkg-a"]
        assert result["b"]["command"] == "docker"
        assert result["b"]["args"] == ["run", "img"]


# ---------------------------------------------------------------------------
# T048: TestBuildClaudeMcpConfigsEnvResolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildClaudeMcpConfigsEnvResolution:
    """Environment variable resolution: substitution, env_file, config."""

    def test_env_var_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """${VAR} patterns in env values are resolved from os.environ."""
        monkeypatch.setenv("MY_API_KEY", "secret123")
        tool = _make_mcp_tool(
            name="api_server",
            env={"API_KEY": "${MY_API_KEY}"},
        )
        result = build_claude_mcp_configs([tool])

        assert result["api_server"]["env"]["API_KEY"] == "secret123"

    def test_env_file_loading(self, tmp_path: pytest.TempPathFactory) -> None:
        """env_file values are loaded into the env dict."""
        env_file = tmp_path / ".env"  # type: ignore[operator]
        env_file.write_text("LOADED_VAR=from_file\n")

        tool = _make_mcp_tool(
            name="file_server",
            env_file=str(env_file),
        )
        result = build_claude_mcp_configs([tool])

        assert result["file_server"]["env"]["LOADED_VAR"] == "from_file"

    def test_explicit_env_overrides_env_file(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Explicit env values take precedence over env_file values."""
        env_file = tmp_path / ".env"  # type: ignore[operator]
        env_file.write_text("SHARED_KEY=from_file\n")

        tool = _make_mcp_tool(
            name="override_server",
            env_file=str(env_file),
            env={"SHARED_KEY": "from_explicit"},
        )
        result = build_claude_mcp_configs([tool])

        assert result["override_server"]["env"]["SHARED_KEY"] == "from_explicit"

    def test_config_becomes_mcp_config_json(self) -> None:
        """tool.config is serialized to MCP_CONFIG env var as JSON."""
        tool = _make_mcp_tool(
            name="config_server",
            config={"setting": "value", "count": 42},
        )
        result = build_claude_mcp_configs([tool])

        mcp_config = result["config_server"]["env"]["MCP_CONFIG"]
        parsed = json.loads(mcp_config)
        assert parsed == {"setting": "value", "count": 42}


# ---------------------------------------------------------------------------
# T049: TestBuildClaudeMcpConfigsNonStdio
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildClaudeMcpConfigsNonStdio:
    """Non-stdio transports are skipped with a warning."""

    def test_sse_tool_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SSE transport tool is skipped and a warning is logged."""
        tool = _make_mcp_tool(
            name="sse_server",
            transport=TransportType.SSE,
        )
        with caplog.at_level(logging.WARNING):
            result = build_claude_mcp_configs([tool])

        assert len(result) == 0
        assert "sse_server" in caplog.text
        assert "sse" in caplog.text.lower()

    def test_websocket_tool_skipped(self) -> None:
        """WebSocket transport tool is skipped."""
        tool = _make_mcp_tool(
            name="ws_server",
            transport=TransportType.WEBSOCKET,
        )
        result = build_claude_mcp_configs([tool])

        assert len(result) == 0

    def test_mixed_list_returns_only_stdio(self) -> None:
        """Mixed list of stdio and non-stdio returns only stdio entries."""
        stdio_tool = _make_mcp_tool(name="local_server", command=CommandType.NPX)
        sse_tool = _make_mcp_tool(name="remote_sse", transport=TransportType.SSE)
        http_tool = _make_mcp_tool(name="remote_http", transport=TransportType.HTTP)

        result = build_claude_mcp_configs([stdio_tool, sse_tool, http_tool])

        assert len(result) == 1
        assert "local_server" in result


# ---------------------------------------------------------------------------
# T050: TestBuildClaudeMcpConfigsEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildClaudeMcpConfigsEdgeCases:
    """Edge cases: empty list, command=None default."""

    def test_empty_list_returns_empty_dict(self) -> None:
        """Empty tool list returns empty dict."""
        result = build_claude_mcp_configs([])
        assert result == {}

    def test_command_none_defaults_to_npx(self) -> None:
        """When command is None (via model_construct), defaults to 'npx'."""
        tool = MCPTool.model_construct(
            name="no_cmd_server",
            description="Server without command",
            type="mcp",
            transport=TransportType.STDIO,
            command=None,
            args=["-y", "some-package"],
            env=None,
            env_file=None,
            config=None,
            load_tools=True,
            load_prompts=True,
            request_timeout=60,
            is_retrieval=False,
            registry_name=None,
            defer_loading=True,
            encoding=None,
            url=None,
            headers=None,
            timeout=None,
            sse_read_timeout=None,
            terminate_on_close=None,
        )
        result = build_claude_mcp_configs([tool])

        assert result["no_cmd_server"]["command"] == "npx"

    def test_no_args_produces_empty_args(self) -> None:
        """Tool with args=None produces empty list in output."""
        tool = _make_mcp_tool(name="minimal", args=None)
        result = build_claude_mcp_configs([tool])

        assert result["minimal"]["args"] == []

    def test_no_env_omits_env_key(self) -> None:
        """Tool with no env, env_file, or config omits env from output."""
        tool = _make_mcp_tool(name="clean_server")
        result = build_claude_mcp_configs([tool])

        # env should not be present or should be empty
        assert (
            "env" not in result["clean_server"] or result["clean_server"]["env"] == {}
        )
