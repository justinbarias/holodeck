"""Click commands for MCP server management.

This module implements the 'holodeck mcp' command group with subcommands
for searching, listing, adding, and removing MCP servers from the
official MCP Registry.
"""

import click


@click.group(name="mcp")
def mcp() -> None:
    """Manage MCP (Model Context Protocol) servers.

    Search the official MCP registry, add servers to your agent configuration,
    and manage installed servers.

    MCP servers extend your agent's capabilities by providing access to
    external tools and data sources. Use 'holodeck mcp search' to discover
    available servers, then 'holodeck mcp add' to install them.

    \b
    EXAMPLES:

        Search for filesystem-related servers:
            holodeck mcp search filesystem

        Add a server to your agent:
            holodeck mcp add io.github.modelcontextprotocol/server-filesystem

        List installed servers:
            holodeck mcp list

        Remove a server:
            holodeck mcp remove filesystem

    For more information, see: https://useholodeck.ai/docs/mcp
    """
    pass


@mcp.command(name="search")
@click.argument("query", required=False)
@click.option(
    "--limit",
    default=25,
    type=click.IntRange(min=1, max=100),
    help="Maximum number of results to return (1-100, default: 25)",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output results as JSON",
)
def search(query: str | None, limit: int, as_json: bool) -> None:
    """Search the MCP registry for available servers.

    QUERY is an optional search term to filter servers by name.
    If not provided, lists all available servers.

    \b
    EXAMPLES:

        Search for filesystem servers:
            holodeck mcp search filesystem

        List all servers (first page):
            holodeck mcp search

        Get results as JSON:
            holodeck mcp search --json
    """
    # TODO: Implement in Phase 3 (T008-T013)
    click.echo("mcp search: Not yet implemented")


@mcp.command(name="list")
@click.option(
    "--agent",
    "agent_file",
    default="agent.yaml",
    type=click.Path(),
    help="Path to agent configuration file (default: agent.yaml)",
)
@click.option(
    "-g",
    "--global",
    "global_only",
    is_flag=True,
    help="Show only global configuration",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show both agent and global configurations",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output results as JSON",
)
def list_cmd(
    agent_file: str,
    global_only: bool,
    show_all: bool,
    as_json: bool,
) -> None:
    """List installed MCP servers.

    By default, shows servers from the agent configuration in the current
    directory. Use -g to show global servers, or --all for both.

    \b
    EXAMPLES:

        List servers in agent.yaml:
            holodeck mcp list

        List global servers:
            holodeck mcp list -g

        List all servers with source labels:
            holodeck mcp list --all
    """
    # TODO: Implement in Phase 5 (T022-T027)
    click.echo("mcp list: Not yet implemented")


@mcp.command(name="add")
@click.argument("server", required=True)
@click.option(
    "--agent",
    "agent_file",
    default="agent.yaml",
    type=click.Path(),
    help="Path to agent configuration file (default: agent.yaml)",
)
@click.option(
    "-g",
    "--global",
    "global_install",
    is_flag=True,
    help="Add to global configuration (~/.holodeck/config.yaml)",
)
@click.option(
    "--version",
    "server_version",
    default="latest",
    help="Server version to install (default: latest)",
)
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse", "http"]),
    help="Transport type (default: stdio)",
)
@click.option(
    "--name",
    "custom_name",
    default=None,
    help="Custom name for the server (overrides default short name)",
)
def add(
    server: str,
    agent_file: str,
    global_install: bool,
    server_version: str,
    transport: str,
    custom_name: str | None,
) -> None:
    """Add an MCP server to your configuration.

    SERVER is the server name from the MCP registry (e.g.,
    io.github.modelcontextprotocol/server-filesystem).

    By default, adds to agent.yaml in the current directory.
    Use -g to add to global configuration (~/.holodeck/config.yaml).

    \b
    EXAMPLES:

        Add filesystem server to agent:
            holodeck mcp add io.github.modelcontextprotocol/server-filesystem

        Add to global config:
            holodeck mcp add io.github.modelcontextprotocol/server-github -g

        Add specific version:
            holodeck mcp add io.github.example/server --version 1.2.0
    """
    # TODO: Implement in Phase 4 (T014-T021)
    click.echo("mcp add: Not yet implemented")


@mcp.command(name="remove")
@click.argument("server", required=True)
@click.option(
    "--agent",
    "agent_file",
    default="agent.yaml",
    type=click.Path(),
    help="Path to agent configuration file (default: agent.yaml)",
)
@click.option(
    "-g",
    "--global",
    "global_remove",
    is_flag=True,
    help="Remove from global configuration",
)
def remove(
    server: str,
    agent_file: str,
    global_remove: bool,
) -> None:
    """Remove an MCP server from your configuration.

    SERVER is the name of the server to remove (e.g., 'filesystem').

    By default, removes from agent.yaml in the current directory.
    Use -g to remove from global configuration.

    \b
    EXAMPLES:

        Remove from agent config:
            holodeck mcp remove filesystem

        Remove from global config:
            holodeck mcp remove github -g
    """
    # TODO: Implement in Phase 6 (T028-T032)
    click.echo("mcp remove: Not yet implemented")
