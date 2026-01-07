"""Click commands for MCP server management.

This module implements the 'holodeck mcp' command group with subcommands
for searching, listing, adding, and removing MCP servers from the
official MCP Registry.
"""

import json
import re
from pathlib import Path

import click

from holodeck.config.loader import (
    add_mcp_server_to_agent,
    add_mcp_server_to_global,
    get_mcp_servers_from_agent,
    get_mcp_servers_from_global,
    remove_mcp_server_from_agent,
    remove_mcp_server_from_global,
)
from holodeck.lib.errors import (
    ConfigError,
    DuplicateServerError,
    FileNotFoundError,
    RegistryAPIError,
    RegistryConnectionError,
    ServerNotFoundError,
)
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.models.registry import RegistryServer, SearchResult
from holodeck.models.tool import MCPTool
from holodeck.services.mcp_registry import (
    SUPPORTED_REGISTRY_TYPES,
    MCPRegistryClient,
    find_stdio_package,
    registry_to_mcp_tool,
)

logger = get_logger(__name__)

# --- Helper Functions for Search Command ---


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: The text to truncate.
        max_len: Maximum length including ellipsis.

    Returns:
        Truncated text with ellipsis if exceeded max_len.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _get_transports(server: RegistryServer) -> str:
    """Get comma-separated transport types from server packages.

    Args:
        server: Registry server with package information.

    Returns:
        Comma-separated transport types, or "stdio" if none found.
    """
    transports: set[str] = set()
    for pkg in server.packages:
        transports.add(pkg.transport.type)
    return ", ".join(sorted(transports)) or "stdio"


def _get_transport_list(server: RegistryServer) -> list[str]:
    """Get list of transport types for JSON output.

    Args:
        server: Registry server with package information.

    Returns:
        Sorted list of transport types, or ["stdio"] if none found.
    """
    transports: set[str] = set()
    for pkg in server.packages:
        transports.add(pkg.transport.type)
    return sorted(transports) if transports else ["stdio"]


# --- Output Formatters for Search Command ---


def _get_version_display(server: RegistryServer) -> str:
    """Get version display string for table output.

    Shows single version if only one, or latest version with count for multiple.

    Args:
        server: Registry server with versions.

    Returns:
        Version display string (e.g., "1.0.0" or "1.0.0 (+2)").
    """
    if server.versions:
        latest = server.versions[0].version or server.version or "-"
        if len(server.versions) == 1:
            return latest
        # Show latest version with additional count
        return f"{latest} (+{len(server.versions) - 1})"
    return server.version or "-"


def _output_table(result: SearchResult) -> None:
    """Format search results as a table.

    Args:
        result: Search result from the MCP registry.
    """
    if not result.servers:
        click.echo("No servers found.")
        return

    # Calculate column widths based on content
    name_width = min(40, max(len(s.name) for s in result.servers))
    version_width = 12
    desc_width = 35

    # Header
    click.echo(
        f"{'NAME':<{name_width}}  {'VERSION':<{version_width}}  "
        f"{'DESCRIPTION':<{desc_width}}  TRANSPORT"
    )
    click.echo("-" * (name_width + version_width + desc_width + 18))

    # Rows
    for server in result.servers:
        name = _truncate(server.name, name_width)
        version = _get_version_display(server)
        desc = _truncate(server.description, desc_width)
        transports = _get_transports(server)
        click.echo(
            f"{name:<{name_width}}  {version:<{version_width}}  "
            f"{desc:<{desc_width}}  {transports}"
        )


def _format_version_for_json(server: RegistryServer) -> list[dict[str, object]]:
    """Format version details for JSON output.

    Args:
        server: Registry server with versions.

    Returns:
        List of version detail dictionaries.
    """
    if not server.versions:
        # Fallback if versions not populated
        return [{"version": server.version}]

    versions_output: list[dict[str, object]] = []
    for v in server.versions:
        version_info: dict[str, object] = {
            "version": v.version,
            "packages": [
                {
                    "registry_type": p.registry_type,
                    "identifier": p.identifier,
                    "version": p.version,
                    "transport": p.transport.type,
                }
                for p in v.packages
            ],
        }
        # Add metadata if available
        if v.meta:
            version_info["published_at"] = (
                v.meta.published_at.isoformat() if v.meta.published_at else None
            )
            version_info["is_latest"] = v.meta.is_latest
            version_info["status"] = v.meta.status
        versions_output.append(version_info)

    return versions_output


def _output_json(result: SearchResult) -> None:
    """Format search results as JSON.

    Args:
        result: Search result from the MCP registry.
    """
    output = {
        "servers": [
            {
                "name": s.name,
                "description": s.description,
                "transports": _get_transport_list(s),
                "versions": _format_version_for_json(s),
            }
            for s in result.servers
        ],
        "total_count": result.total_count,
        "has_more": result.next_cursor is not None,
    }
    click.echo(json.dumps(output, indent=2))


# --- Helper Functions for List Command ---

# Regex to extract version from package specifier (e.g., @package@1.0.0 -> 1.0.0)
VERSION_PATTERN = re.compile(r"@(\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?(?:\+[\w.]+)?)$")


def _extract_version_from_args(mcp_tool: MCPTool) -> str:
    """Extract version from MCP tool args.

    Parses the args list to find version specifiers like:
    - @modelcontextprotocol/server-filesystem@1.0.0 -> 1.0.0
    - package-name@2.3.4-beta -> 2.3.4-beta

    Args:
        mcp_tool: MCPTool instance

    Returns:
        Version string or "-" if not found
    """
    if not mcp_tool.args:
        return "-"

    for arg in mcp_tool.args:
        match = VERSION_PATTERN.search(arg)
        if match:
            return match.group(1)

    return "-"


def _list_output_table(servers: list[tuple[MCPTool, str]], show_source: bool) -> None:
    """Format installed servers list as a table.

    Args:
        servers: List of (MCPTool, source) tuples where source is "agent" or "global"
        show_source: Whether to show SOURCE column (for --all mode)
    """
    if not servers:
        click.echo(
            "No MCP servers configured. "
            "Use 'holodeck mcp search' to find available servers."
        )
        return

    # Calculate column widths based on content
    name_width = min(25, max(len(s[0].name) for s in servers))
    version_width = 12
    transport_width = 10
    desc_width = 40 if not show_source else 30

    # Header
    header = (
        f"{'NAME':<{name_width}}  {'VERSION':<{version_width}}  "
        f"{'TRANSPORT':<{transport_width}}  {'DESCRIPTION':<{desc_width}}"
    )
    if show_source:
        header += "  SOURCE"
    click.echo(header)

    # Separator
    sep_width = name_width + version_width + transport_width + desc_width + 8
    if show_source:
        sep_width += 8
    click.echo("-" * sep_width)

    # Rows
    for mcp_tool, source in servers:
        name = _truncate(mcp_tool.name, name_width)
        version = _extract_version_from_args(mcp_tool)
        transport = mcp_tool.transport.value if mcp_tool.transport else "stdio"
        desc = _truncate(mcp_tool.description, desc_width)

        row = (
            f"{name:<{name_width}}  {version:<{version_width}}  "
            f"{transport:<{transport_width}}  {desc:<{desc_width}}"
        )
        if show_source:
            row += f"  {source}"
        click.echo(row)


def _list_output_json(servers: list[tuple[MCPTool, str]], show_source: bool) -> None:
    """Format installed servers list as JSON.

    Args:
        servers: List of (MCPTool, source) tuples
        show_source: Whether to include source field
    """
    output_servers = []
    for mcp_tool, source in servers:
        server_dict: dict[str, object] = {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "version": _extract_version_from_args(mcp_tool),
            "transport": mcp_tool.transport.value if mcp_tool.transport else "stdio",
        }

        # Include additional fields if present
        if mcp_tool.command:
            server_dict["command"] = mcp_tool.command.value
        if mcp_tool.args:
            server_dict["args"] = mcp_tool.args
        if mcp_tool.registry_name:
            server_dict["registry_name"] = mcp_tool.registry_name
        if show_source:
            server_dict["source"] = source

        output_servers.append(server_dict)

    output = {
        "servers": output_servers,
        "total_count": len(output_servers),
    }
    click.echo(json.dumps(output, indent=2))


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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress INFO logging output",
)
def search(
    query: str | None, limit: int, as_json: bool, verbose: bool, quiet: bool
) -> None:
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
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)
    logger.debug(f"MCP search command invoked: query={query}, limit={limit}")

    try:
        with MCPRegistryClient() as client:
            result = client.search(query=query, limit=limit)

            if as_json:
                _output_json(result)
            else:
                _output_table(result)

            # Show pagination hint if more results available
            if result.next_cursor and not as_json:
                click.echo(f"\n{result.total_count} total results. More available.")

    except RegistryConnectionError as e:
        click.secho(f"Error: Registry unavailable - {e}", fg="red", err=True)
        raise SystemExit(1) from e
    except RegistryAPIError as e:
        msg = f"Error: Registry service error - {e}"
        click.secho(msg, fg="red", err=True)
        raise SystemExit(1) from e


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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress INFO logging output",
)
def list_cmd(
    agent_file: str,
    global_only: bool,
    show_all: bool,
    as_json: bool,
    verbose: bool,
    quiet: bool,
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
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)
    logger.debug(
        f"MCP list command invoked: agent_file={agent_file}, global_only={global_only}"
    )

    servers: list[tuple[MCPTool, str]] = []

    try:
        if show_all:
            # Show both agent and global servers
            agent_path = Path(agent_file)
            if agent_path.exists():
                agent_servers = get_mcp_servers_from_agent(agent_path)
                servers.extend((s, "agent") for s in agent_servers)

            global_servers = get_mcp_servers_from_global()
            servers.extend((s, "global") for s in global_servers)

        elif global_only:
            # Show only global servers
            global_servers = get_mcp_servers_from_global()
            servers.extend((s, "global") for s in global_servers)

        else:
            # Default: show agent servers
            agent_path = Path(agent_file)
            agent_servers = get_mcp_servers_from_agent(agent_path)
            servers.extend((s, "agent") for s in agent_servers)

        # Output results
        if as_json:
            _list_output_json(servers, show_source=show_all)
        else:
            _list_output_table(servers, show_source=show_all)

    except FileNotFoundError as e:
        if show_all:
            # For --all, continue with global servers if agent not found
            global_servers = get_mcp_servers_from_global()
            servers = [(s, "global") for s in global_servers]
            if as_json:
                _list_output_json(servers, show_source=True)
            else:
                _list_output_table(servers, show_source=True)
        else:
            click.secho(f"Error: {e.message}", fg="red", err=True)
            raise SystemExit(1) from e
    except ConfigError as e:
        click.secho(f"Error: {e.message}", fg="red", err=True)
        raise SystemExit(1) from e


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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress INFO logging output",
)
def add(
    server: str,
    agent_file: str,
    global_install: bool,
    server_version: str,
    transport: str,
    custom_name: str | None,
    verbose: bool,
    quiet: bool,
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
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)
    logger.debug(f"MCP add command invoked: server={server}, version={server_version}")

    try:
        # 1. Fetch server from registry
        with MCPRegistryClient() as client:
            registry_server = client.get_server(server, server_version)

        # 2. Find STDIO package (HoloDeck only supports stdio transport)
        stdio_pkg = find_stdio_package(registry_server)
        if stdio_pkg is None:
            available = {p.transport.type for p in registry_server.packages}
            click.secho(
                f"Error: Server '{server}' does not support stdio transport.\n"
                f"Available transports: {', '.join(sorted(available))}\n"
                "HoloDeck currently only supports stdio transport.",
                fg="red",
                err=True,
            )
            raise SystemExit(1)

        # 3. Validate registry type is supported
        if stdio_pkg.registry_type not in SUPPORTED_REGISTRY_TYPES:
            click.secho(
                f"Error: Server uses unsupported package type "
                f"'{stdio_pkg.registry_type}'.\n"
                f"Supported types: {', '.join(sorted(SUPPORTED_REGISTRY_TYPES))}.",
                fg="red",
                err=True,
            )
            raise SystemExit(1)

        # 4. Convert to MCPTool (pass specific package)
        mcp_tool = registry_to_mcp_tool(registry_server, package=stdio_pkg)

        # 5. Apply custom name if provided
        if custom_name:
            mcp_tool = mcp_tool.model_copy(update={"name": custom_name})

        # 6. Add to config (agent or global)
        if global_install:
            add_mcp_server_to_global(mcp_tool)
            target_display = "~/.holodeck/config.yaml"
        else:
            agent_path = Path(agent_file)
            add_mcp_server_to_agent(agent_path, mcp_tool)
            target_display = agent_file

        # 7. Success message
        click.secho(f"Added '{mcp_tool.name}' to {target_display}", fg="green")

        # 8. Display required environment variables
        env_vars = stdio_pkg.environment_variables
        if env_vars:
            click.echo("\nRequired environment variables:")
            for ev in env_vars:
                required_marker = " (required)" if ev.required else " (optional)"
                desc = f" - {ev.description}" if ev.description else ""
                click.echo(f"  {ev.name}{required_marker}{desc}")
            click.echo("\nSet these in your .env file or shell environment.")

    except RegistryConnectionError as e:
        click.secho(f"Error: Registry unavailable - {e}", fg="red", err=True)
        raise SystemExit(1) from e
    except RegistryAPIError as e:
        click.secho(f"Error: Registry error - {e}", fg="red", err=True)
        raise SystemExit(1) from e
    except ServerNotFoundError as e:
        click.secho(
            f"Error: Server '{server}' not found in registry", fg="red", err=True
        )
        raise SystemExit(1) from e
    except DuplicateServerError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        click.secho(f"Error: {e.message}", fg="red", err=True)
        raise SystemExit(1) from e
    except ConfigError as e:
        click.secho(f"Error: {e.message}", fg="red", err=True)
        raise SystemExit(1) from e


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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress INFO logging output",
)
def remove(
    server: str,
    agent_file: str,
    global_remove: bool,
    verbose: bool,
    quiet: bool,
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
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)
    logger.debug(f"MCP remove command invoked: server={server}, global={global_remove}")

    try:
        if global_remove:
            remove_mcp_server_from_global(server)
            target_display = "~/.holodeck/config.yaml"
        else:
            agent_path = Path(agent_file)
            remove_mcp_server_from_agent(agent_path, server)
            target_display = agent_file

        click.secho(f"Removed '{server}' from {target_display}", fg="green")

    except ServerNotFoundError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1) from e

    except FileNotFoundError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1) from e

    except ConfigError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1) from e
