"""MCP adapters bridging HoloDeck MCP tools to the OpenAI Agents SDK.

Translates HoloDeck :class:`~holodeck.models.tool.MCPTool` configs into OpenAI
Agents SDK MCP server objects:

- ``transport: stdio`` -> ``agents.mcp.MCPServerStdio``
- ``transport: sse`` -> ``agents.mcp.MCPServerSse``
- ``transport: http`` -> ``agents.mcp.MCPServerStreamableHttp``
- ``transport: websocket`` -> skipped with a warning (the SDK has no WebSocket
  transport; mirrors the spec-027 skip the Claude path uses in
  :mod:`holodeck.lib.backends.mcp_bridge`).

Per-server ``allowed_tools`` (declared under ``MCPTool.config``) are translated
to ``agents.mcp.create_static_tool_filter(allowed_tool_names=[...])`` and passed
as the server's ``tool_filter``.

Env/header ``${VAR}`` substitution and relative-arg resolution reuse the shared
helpers in :mod:`holodeck.lib.backends.mcp_bridge` — they are not re-implemented
here.

The SDK requires the caller to ``await server.connect()`` before passing a
server to ``Agent(mcp_servers=...)`` and ``await server.cleanup()`` when done;
the backend wires that lifecycle (connect in ``initialize``, cleanup in
``teardown``). This module only constructs the (unconnected) server objects.

Every ``import agents`` happens inside functions to keep the optional SDK import
lazy (SC-005).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from holodeck.config.env_loader import substitute_env_vars
from holodeck.lib.backends.mcp_bridge import _resolve_mcp_env, _resolve_relative_args
from holodeck.models.tool import MCPTool, TransportType

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime SDK import
    from agents.mcp import MCPServer, ToolFilterStatic

logger = logging.getLogger(__name__)


def _build_tool_filter(tool: MCPTool) -> ToolFilterStatic | None:
    """Build a static tool filter for *tool*, or ``None`` when unfiltered.

    The whitelist is read from ``tool.config['allowed_tools']`` (a list of MCP
    tool names). When absent or empty, no filter is applied and the server
    exposes all of its tools.

    Args:
        tool: The MCP tool configuration.

    Returns:
        An ``agents.mcp`` static tool-filter dict, or ``None``.
    """
    if not tool.config:
        return None
    allowed = tool.config.get("allowed_tools")
    if not allowed:
        return None

    from agents.mcp import create_static_tool_filter

    return create_static_tool_filter(allowed_tool_names=list(allowed))


def _substitute_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Apply ``${VAR}`` substitution to each header value.

    Args:
        headers: The raw header map from the MCP config, or ``None``.

    Returns:
        A new map with every value env-substituted (empty when *headers* is
        ``None``).
    """
    if not headers:
        return {}
    return {key: substitute_env_vars(value) for key, value in headers.items()}


def _build_stdio_server(tool: MCPTool, base_dir: Path | None) -> MCPServer:
    """Construct an ``MCPServerStdio`` for a stdio-transport *tool*."""
    from agents.mcp import MCPServerStdio, MCPServerStdioParams

    command = tool.command.value if tool.command else "npx"
    args = _resolve_relative_args(
        tool.args or [], str(base_dir) if base_dir is not None else None
    )
    env = _resolve_mcp_env(tool)

    params: MCPServerStdioParams = {"command": command, "args": args}
    if env:
        params["env"] = env
    if tool.encoding:
        params["encoding"] = tool.encoding

    return MCPServerStdio(
        params=params,
        name=tool.name,
        tool_filter=_build_tool_filter(tool),
    )


def _build_sse_server(tool: MCPTool) -> MCPServer:
    """Construct an ``MCPServerSse`` for an sse-transport *tool*."""
    from agents.mcp import MCPServerSse, MCPServerSseParams

    # ``url`` is guaranteed non-None for sse transport by MCPTool validation.
    params: MCPServerSseParams = {"url": tool.url or ""}
    headers = _substitute_headers(tool.headers)
    if headers:
        params["headers"] = headers
    if tool.timeout is not None:
        params["timeout"] = tool.timeout
    if tool.sse_read_timeout is not None:
        params["sse_read_timeout"] = tool.sse_read_timeout

    return MCPServerSse(
        params=params,
        name=tool.name,
        tool_filter=_build_tool_filter(tool),
    )


def _build_http_server(tool: MCPTool) -> MCPServer:
    """Construct an ``MCPServerStreamableHttp`` for an http-transport *tool*."""
    from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

    # ``url`` is guaranteed non-None for http transport by MCPTool validation.
    params: MCPServerStreamableHttpParams = {"url": tool.url or ""}
    headers = _substitute_headers(tool.headers)
    if headers:
        params["headers"] = headers
    if tool.timeout is not None:
        params["timeout"] = tool.timeout
    if tool.sse_read_timeout is not None:
        params["sse_read_timeout"] = tool.sse_read_timeout
    if tool.terminate_on_close is not None:
        params["terminate_on_close"] = tool.terminate_on_close

    return MCPServerStreamableHttp(
        params=params,
        name=tool.name,
        tool_filter=_build_tool_filter(tool),
    )


def build_mcp_servers(
    mcp_tools: list[MCPTool] | None,
    base_dir: Path | None,
) -> list[MCPServer]:
    """Translate HoloDeck MCP tool configs into SDK MCP server objects.

    Each supported transport maps to its SDK server class; ``websocket`` tools
    are skipped with a warning (the SDK has no WebSocket transport) so a load
    never fails on an unsupported transport. The returned servers are *not* yet
    connected — the backend must ``await server.connect()`` each before passing
    them to ``Agent(mcp_servers=...)`` and ``await server.cleanup()`` at teardown.

    Args:
        mcp_tools: MCP tool configurations from the agent YAML (may be ``None``).
        base_dir: Directory used to resolve relative stdio ``args`` paths
            (typically the agent project root).

    Returns:
        A list of (unconnected) ``agents.mcp`` MCP server objects.
    """
    servers: list[MCPServer] = []
    for tool in mcp_tools or []:
        if tool.transport == TransportType.STDIO:
            servers.append(_build_stdio_server(tool, base_dir))
        elif tool.transport == TransportType.SSE:
            servers.append(_build_sse_server(tool))
        elif tool.transport == TransportType.HTTP:
            servers.append(_build_http_server(tool))
        else:  # TransportType.WEBSOCKET
            logger.warning(
                "Skipping MCP tool '%s': %s transport is not supported by the "
                "openai_agents backend (only stdio, sse, and http are supported)",
                tool.name,
                tool.transport.value,
            )

    return servers
