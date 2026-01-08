"""CLI command for serving agents via HTTP.

Implements the 'holodeck serve' command for exposing agents via HTTP with
AG-UI or REST protocol support.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.lib.errors import ConfigError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.observability import (
    ObservabilityContext,
    initialize_observability,
    shutdown_observability,
)
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
    verbose: bool,
    quiet: bool,
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
        --verbose / -v      Enable verbose debug logging
        --quiet / -q        Suppress INFO logging output
        --cors-origins      Comma-separated CORS origins (default: *)
    """
    # Initialize observability context (will be set if observability enabled)
    obs_context: ObservabilityContext | None = None

    try:
        # Load agent configuration FIRST to check observability setting
        from holodeck.config.context import agent_base_dir
        from holodeck.config.loader import ConfigLoader

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)

        # Determine logging strategy: OTel replaces setup_logging when enabled
        if agent.observability and agent.observability.enabled:
            # Enable console exporter for serve command (structured OTel output)
            from holodeck.models.observability import ConsoleExporterConfig

            if agent.observability.exporters.console is None:
                agent.observability.exporters.console = ConsoleExporterConfig(
                    enabled=True
                )
            else:
                agent.observability.exporters.console.enabled = True

            # OTel handles all logging - skip setup_logging
            obs_context = initialize_observability(
                agent.observability, agent.name, verbose=verbose, quiet=quiet
            )
        else:
            # Traditional logging
            setup_logging(verbose=verbose, quiet=quiet)

        logger.info(
            f"Serve command invoked: config={agent_config}, "
            f"port={port}, host={host}, protocol={protocol}, verbose={verbose}"
        )
        logger.debug(f"Loading agent configuration from {agent_config}")
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

        # Determine if observability is enabled for span creation
        observability_enabled = obs_context is not None

        # Create and run server
        asyncio.run(
            _run_server(
                agent=agent,
                host=host,
                port=port,
                protocol=protocol_type,
                cors_origins=origins,
                verbose=verbose,
                execution_config=resolved_config,
                observability_enabled=observability_enabled,
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
    finally:
        # Shutdown observability if it was initialized
        if obs_context:
            shutdown_observability(obs_context)


async def _run_server(
    agent: Agent,
    host: str,
    port: int,
    protocol: ProtocolType,
    cors_origins: list[str],
    verbose: bool,
    execution_config: ExecutionConfig,
    observability_enabled: bool = False,
) -> None:
    """Run the HTTP server.

    Args:
        agent: Loaded Agent configuration.
        host: Host to bind to.
        port: Port to listen on.
        protocol: Protocol type (AG-UI or REST).
        cors_origins: List of allowed CORS origins.
        verbose: Enable verbose debug logging.
        execution_config: Resolved execution configuration.
        observability_enabled: Enable OpenTelemetry per-request tracing.
    """
    # Create parent span for serve command if observability is enabled
    if observability_enabled:
        from holodeck.lib.observability import get_tracer

        tracer = get_tracer(__name__)
        span_context: Any = tracer.start_as_current_span("holodeck.cli.serve")
    else:
        span_context = nullcontext()

    with span_context:
        import uvicorn

        from holodeck.serve.server import AgentServer

        # Create server
        server = AgentServer(
            agent_config=agent,
            protocol=protocol,
            host=host,
            port=port,
            cors_origins=cors_origins,
            debug=verbose,
            execution_config=execution_config,
            observability_enabled=observability_enabled,
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
            log_level="debug" if verbose else "info",
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
        click.echo(
            "    POST /awp                               AG-UI protocol endpoint"
        )
    else:
        click.echo(f"    POST /agent/{agent.name}/chat               Sync chat (JSON)")
        click.echo(f"    POST /agent/{agent.name}/chat/stream        Stream (SSE)")
        click.echo(f"    POST /agent/{agent.name}/chat/multipart     Sync (multipart)")
        click.echo(
            f"    POST /agent/{agent.name}/chat/stream/multipart  Stream (multipart)"
        )
        click.echo("    DELETE /sessions/{session_id}               Delete session")
        click.echo("    GET  /docs                                  OpenAPI docs")

    click.echo("    GET  /health                           Health check")
    click.echo("    GET  /ready                            Readiness check")
    click.echo()
    click.secho("  Press Ctrl+C to stop", fg="yellow")
    click.secho("=" * 60, fg="cyan")
    click.echo()
