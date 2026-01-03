"""CLI command for serving agents via HTTP.

Implements the 'holodeck serve' command for exposing agents via HTTP with
AG-UI or REST protocol support.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.lib.errors import ConfigError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.models.config import ExecutionConfig

if TYPE_CHECKING:
    from holodeck.models.agent import Agent
    from holodeck.serve.models import ProtocolType

logger = get_logger(__name__)


@click.command()
@click.argument(
    "agent_config",
    type=click.Path(exists=True),
    default="agent.yaml",
    required=False,
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to listen on (default: 8000)",
)
@click.option(
    "--host",
    "-h",
    type=str,
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1 for local-only access)",
)
@click.option(
    "--protocol",
    type=click.Choice(["ag-ui", "rest"]),
    default="ag-ui",
    help="Protocol to use (default: ag-ui)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug logging",
)
@click.option(
    "--cors-origins",
    type=str,
    default="http://localhost:3000",
    help="Comma-separated list of allowed CORS origins (default: http://localhost:3000)",
)
def serve(
    agent_config: str,
    port: int,
    host: str,
    protocol: str,
    debug: bool,
    cors_origins: str,
) -> None:
    """Start an HTTP server exposing an agent.

    AGENT_CONFIG is the path to the agent.yaml configuration file.

    Example:

        holodeck serve examples/weather-agent.yaml

        holodeck serve examples/assistant.yaml --port 9000 --protocol ag-ui

    The server exposes the agent via HTTP with the specified protocol.

    Protocols:

        ag-ui   AG-UI protocol (streaming SSE events)
        rest    REST API (JSON request/response)

    Options:

        --port / -p         Port to listen on (default: 8000)
        --host / -h         Host to bind to (default: 127.0.0.1)
        --protocol          Protocol to use: ag-ui or rest (default: ag-ui)
        --debug             Enable debug logging
        --cors-origins      Comma-separated CORS origins (default: *)
    """
    # Setup logging
    setup_logging(verbose=debug, quiet=not debug)

    logger.info(
        f"Serve command invoked: config={agent_config}, "
        f"port={port}, host={host}, protocol={protocol}, debug={debug}"
    )

    try:
        # Load agent configuration
        from holodeck.config.context import agent_base_dir
        from holodeck.config.loader import ConfigLoader

        logger.debug(f"Loading agent configuration from {agent_config}")
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

        # Set the base directory context for resolving relative paths in tools
        agent_dir = str(Path(agent_config).parent.resolve())
        agent_base_dir.set(agent_dir)
        logger.debug(f"Set agent_base_dir context: {agent_base_dir.get()}")

        # Resolve execution config with 6-level priority hierarchy
        # CLI flags > agent.yaml > project config > user config > env vars > defaults
        # Note: serve command doesn't have CLI flags for execution config yet
        cli_config = ExecutionConfig()  # Empty - no CLI overrides

        # Load project-level config (same directory as agent.yaml)
        project_config = loader.load_project_config(agent_dir)
        project_execution = project_config.execution if project_config else None

        # Load user-level config (~/.holodeck/)
        user_config = loader.load_global_config()
        user_execution = user_config.execution if user_config else None

        resolved_config = loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=agent.execution,
            project_config=project_execution,
            user_config=user_execution,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        logger.debug(
            f"Resolved execution config: llm_timeout={resolved_config.llm_timeout}, "
            f"file_timeout={resolved_config.file_timeout}"
        )

        # Parse CORS origins
        origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

        # Map protocol string to ProtocolType
        from holodeck.serve.models import ProtocolType

        protocol_type = ProtocolType.AG_UI if protocol == "ag-ui" else ProtocolType.REST

        # Create and run server
        asyncio.run(
            _run_server(
                agent=agent,
                host=host,
                port=port,
                protocol=protocol_type,
                cors_origins=origins,
                debug=debug,
                execution_config=resolved_config,
            )
        )

    except ConfigError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        click.secho("Error: Failed to load agent configuration", fg="red", err=True)
        click.echo(f"  {str(e)}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl+C)")
        click.echo()
        click.secho("Server stopped.", fg="yellow")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)


async def _run_server(
    agent: Agent,
    host: str,
    port: int,
    protocol: ProtocolType,
    cors_origins: list[str],
    debug: bool,
    execution_config: ExecutionConfig,
) -> None:
    """Run the HTTP server.

    Args:
        agent: Loaded Agent configuration.
        host: Host to bind to.
        port: Port to listen on.
        protocol: Protocol type (AG-UI or REST).
        cors_origins: List of allowed CORS origins.
        debug: Enable debug mode.
        execution_config: Resolved execution configuration.
    """
    import uvicorn

    from holodeck.serve.server import AgentServer

    # Create server
    server = AgentServer(
        agent_config=agent,
        protocol=protocol,
        host=host,
        port=port,
        cors_origins=cors_origins,
        debug=debug,
        execution_config=execution_config,
    )

    # Create app
    app = server.create_app()

    # Start server lifecycle
    await server.start()

    # Display startup info
    _display_startup_info(agent, protocol, host, port)

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )
    server_instance = uvicorn.Server(config)

    try:
        await server_instance.serve()
    finally:
        await server.stop()


def _display_startup_info(
    agent: Agent,
    protocol: ProtocolType,
    host: str,
    port: int,
) -> None:
    """Display server startup information.

    Args:
        agent: Agent configuration.
        protocol: Protocol type.
        host: Host the server is bound to.
        port: Port the server is listening on.
    """
    from holodeck.serve.models import ProtocolType

    click.echo()
    click.secho("=" * 60, fg="cyan")
    click.secho("  HoloDeck Agent Server", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo()
    click.echo(f"  Agent:    {agent.name}")
    click.echo(f"  Protocol: {protocol.value}")
    click.echo(f"  URL:      http://{host}:{port}")
    click.echo()
    click.secho("  Endpoints:", bold=True)

    if protocol == ProtocolType.AG_UI:
        click.echo("    POST /awp        AG-UI protocol endpoint")
    else:
        click.echo("    POST /chat       Chat endpoint (sync)")
        click.echo("    POST /stream     Chat endpoint (streaming)")

    click.echo("    GET  /health     Health check")
    click.echo("    GET  /ready      Readiness check")
    click.echo()
    click.secho("  Press Ctrl+C to stop", fg="yellow")
    click.secho("=" * 60, fg="cyan")
    click.echo()
