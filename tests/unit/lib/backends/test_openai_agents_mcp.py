"""Unit tests for holodeck.lib.backends.openai_agents_mcp.

Translates HoloDeck ``MCPTool`` configs into OpenAI Agents SDK MCP server
objects (``MCPServerStdio`` / ``MCPServerSse`` / ``MCPServerStreamableHttp``).
All SDK MCP classes are patched per-test so no subprocess is spawned and no
network call is made; the tests inspect the constructed ``params`` / kwargs.
"""

import logging

import pytest

from holodeck.lib.backends.openai_agents_mcp import build_mcp_servers
from holodeck.models.tool import MCPTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stdio_tool(**extra: object) -> MCPTool:
    return MCPTool(
        name="files",
        description="filesystem server",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
        **extra,  # type: ignore[arg-type]
    )


def _sse_tool(**extra: object) -> MCPTool:
    return MCPTool(
        name="remote",
        description="remote sse server",
        transport="sse",
        url="https://example.com/sse",
        **extra,  # type: ignore[arg-type]
    )


def _http_tool(**extra: object) -> MCPTool:
    return MCPTool(
        name="remote",
        description="remote http server",
        transport="http",
        url="https://example.com/mcp",
        **extra,  # type: ignore[arg-type]
    )


def _ws_tool(**extra: object) -> MCPTool:
    return MCPTool(
        name="ws",
        description="websocket server",
        transport="websocket",
        url="wss://example.com/ws",
        **extra,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStdioTransport:
    def test_builds_stdio_server_with_command_and_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock(name="stdio_server")
        with patch("agents.mcp.MCPServerStdio", return_value=sentinel) as ctor:
            servers = build_mcp_servers([_stdio_tool()], base_dir=None)
        assert servers == [sentinel]
        _, kwargs = ctor.call_args
        params = kwargs["params"]
        assert params["command"] == "npx"
        assert params["args"][0] == "-y"
        # name is forwarded for trace readability
        assert kwargs["name"] == "files"

    def test_resolves_relative_args_against_base_dir(self, tmp_path: object) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        base = Path(tmp_path)  # type: ignore[arg-type]
        with patch("agents.mcp.MCPServerStdio", return_value=MagicMock()) as ctor:
            build_mcp_servers([_stdio_tool()], base_dir=base)
        params = ctor.call_args.kwargs["params"]
        # ``./data`` becomes an absolute path under base_dir.
        assert params["args"][-1] == str((base / "data").resolve())

    def test_env_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("MY_VALUE", "s3cr3t")
        tool = _stdio_tool(env={"SERVER_OPT": "${MY_VALUE}"})
        with patch("agents.mcp.MCPServerStdio", return_value=MagicMock()) as ctor:
            build_mcp_servers([tool], base_dir=None)
        params = ctor.call_args.kwargs["params"]
        assert params["env"]["SERVER_OPT"] == "s3cr3t"


# ---------------------------------------------------------------------------
# sse transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSseTransport:
    def test_builds_sse_server_with_url(self) -> None:
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock(name="sse_server")
        with patch("agents.mcp.MCPServerSse", return_value=sentinel) as ctor:
            servers = build_mcp_servers([_sse_tool()], base_dir=None)
        assert servers == [sentinel]
        params = ctor.call_args.kwargs["params"]
        assert params["url"] == "https://example.com/sse"

    def test_header_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("BEARER", "abc123")
        tool = _sse_tool(headers={"Authorization": "Bearer ${BEARER}"})
        with patch("agents.mcp.MCPServerSse", return_value=MagicMock()) as ctor:
            build_mcp_servers([tool], base_dir=None)
        params = ctor.call_args.kwargs["params"]
        assert params["headers"]["Authorization"] == "Bearer abc123"


# ---------------------------------------------------------------------------
# http transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHttpTransport:
    def test_builds_streamable_http_server_with_url(self) -> None:
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock(name="http_server")
        with patch("agents.mcp.MCPServerStreamableHttp", return_value=sentinel) as ctor:
            servers = build_mcp_servers([_http_tool()], base_dir=None)
        assert servers == [sentinel]
        params = ctor.call_args.kwargs["params"]
        assert params["url"] == "https://example.com/mcp"

    def test_header_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("KEY", "xyz")
        tool = _http_tool(headers={"X-Api-Key": "${KEY}"})
        with patch("agents.mcp.MCPServerStreamableHttp", return_value=MagicMock()) as c:
            build_mcp_servers([tool], base_dir=None)
        params = c.call_args.kwargs["params"]
        assert params["headers"]["X-Api-Key"] == "xyz"


# ---------------------------------------------------------------------------
# websocket transport — skip with warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWebsocketSkipped:
    def test_websocket_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            servers = build_mcp_servers([_ws_tool()], base_dir=None)
        assert servers == []
        assert any(
            "ws" in r.message and "websocket" in r.message.lower()
            for r in caplog.records
        )

    def test_websocket_does_not_break_other_transports(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from unittest.mock import MagicMock, patch

        sentinel = MagicMock(name="sse_server")
        with (
            patch("agents.mcp.MCPServerSse", return_value=sentinel),
            caplog.at_level(logging.WARNING),
        ):
            servers = build_mcp_servers([_ws_tool(), _sse_tool()], base_dir=None)
        # websocket skipped, sse still built
        assert servers == [sentinel]


# ---------------------------------------------------------------------------
# allowed_tools -> static tool filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaticToolFilter:
    def test_allowed_tools_passed_as_static_filter(self) -> None:
        from unittest.mock import MagicMock, patch

        tool = _stdio_tool(config={"allowed_tools": ["read_file", "list_dir"]})
        filter_sentinel = MagicMock(name="filter")
        with (
            patch("agents.mcp.MCPServerStdio", return_value=MagicMock()) as ctor,
            patch(
                "agents.mcp.create_static_tool_filter",
                return_value=filter_sentinel,
            ) as make_filter,
        ):
            build_mcp_servers([tool], base_dir=None)
        make_filter.assert_called_once_with(
            allowed_tool_names=["read_file", "list_dir"]
        )
        assert ctor.call_args.kwargs["tool_filter"] is filter_sentinel

    def test_no_allowed_tools_means_no_filter(self) -> None:
        from unittest.mock import MagicMock, patch

        with (
            patch("agents.mcp.MCPServerStdio", return_value=MagicMock()) as ctor,
            patch("agents.mcp.create_static_tool_filter") as make_filter,
        ):
            build_mcp_servers([_stdio_tool()], base_dir=None)
        make_filter.assert_not_called()
        assert ctor.call_args.kwargs["tool_filter"] is None


# ---------------------------------------------------------------------------
# empty / mixed input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyInput:
    def test_empty_returns_empty_list(self) -> None:
        assert build_mcp_servers([], base_dir=None) == []
        assert build_mcp_servers(None, base_dir=None) == []
