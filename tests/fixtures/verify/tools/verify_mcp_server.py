"""Offline stdio MCP server for the live verify agents.

The agents start it with ``uvx --with mcp==<locked> python <this file>``,
so it needs no network once uv has cached the ``mcp`` wheel.
"""

from mcp.server.fastmcp import FastMCP

MCP_TOOL_RESULT = "MCP-TOOL-RESULT: "

server = FastMCP("verify-mcp")


@server.tool()
def verify_marker(text: str) -> str:
    """Return a fixed marker plus the given text.

    Args:
        text: Any short string.
    """
    return f"{MCP_TOOL_RESULT}{text}"


if __name__ == "__main__":
    server.run()
