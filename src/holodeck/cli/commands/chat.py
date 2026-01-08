"""CLI command for interactive chat with agents.

Implements the 'holodeck chat' command for multi-turn conversations with agents
including message validation, tool execution streaming, and optional observability.
"""

import asyncio
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import click

from holodeck.chat import ChatSessionManager
from holodeck.chat.progress import ChatProgressIndicator
from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import AgentInitializationError, ConfigError, ExecutionError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.observability import (
    ObservabilityContext,
    initialize_observability,
    shutdown_observability,
)
from holodeck.models.agent import Agent
from holodeck.models.chat import ChatConfig
from holodeck.models.config import ExecutionConfig

logger = get_logger(__name__)


class ChatSpinnerThread(threading.Thread):
    """Background thread for displaying animated spinner during agent execution."""

    def __init__(self, progress: ChatProgressIndicator) -> None:
        """Initialize spinner thread.

        Args:
            progress: ChatProgressIndicator instance for spinner animation.
        """
        super().__init__(daemon=True)
        self.progress = progress
        self._stop_event = threading.Event()
        self._running = False

    def run(self) -> None:
        """Run spinner animation loop."""
        self._running = True
        while not self._stop_event.is_set():
            line = self.progress.get_spinner_line()
            if line:
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()
            time.sleep(0.1)  # 10 FPS update rate
        self._running = False

    def stop(self) -> None:
        """Stop spinner animation and clear spinner line."""
        self._stop_event.set()
        if self._running:
            # Clear spinner line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()


@click.command()
@click.argument("agent_config", type=click.Path(exists=True), default="agent.yaml")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed logging and tool execution (parameters, internal state)",
)
@click.option(
    "--quiet/--no-quiet",
    "-q/-Q",
    default=False,
    help="Suppress INFO logging output. Use -q or --quiet to hide logs.",
)
@click.option(
    "--observability",
    "-o",
    is_flag=True,
    help="Enable OpenTelemetry tracing and metrics",
)
@click.option(
    "--max-messages",
    "-m",
    type=int,
    default=50,
    help="Maximum conversation messages before warning",
)
@click.option(
    "--force-ingest",
    "-f",
    is_flag=True,
    help="Force re-ingestion of all vector store source files",
)
def chat(
    agent_config: str,
    verbose: bool,
    quiet: bool,
    observability: bool,
    max_messages: int,
    force_ingest: bool,
) -> None:
    """Start an interactive chat session with an agent.

    AGENT_CONFIG is the path to the agent.yaml configuration file.

    Example:

        holodeck chat examples/weather-agent.yaml

        holodeck chat examples/assistant.yaml --verbose --max-messages 100

    Chat Session Commands:

        Type 'exit' or 'quit' to end the session.
        Press Ctrl+C to interrupt.

    Options:

        --verbose / -v      Show detailed tool execution parameters and results
        --quiet / -q        Suppress logging output (enabled by default)
        --observability / -o    Enable OpenTelemetry tracing for debugging
        --max-messages / -m     Set max messages before context warning (default: 50)
    """
    # Initialize observability context (will be set if observability enabled)
    obs_context: ObservabilityContext | None = None
    effective_quiet = quiet and not verbose

    try:
        # Load agent configuration FIRST to check observability setting
        from holodeck.config.context import agent_base_dir

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)

        # Determine logging strategy: OTel replaces setup_logging when enabled
        if agent.observability and agent.observability.enabled:
            # OTel handles all logging - skip setup_logging
            # Console exporter not enabled by default (only serve enables it)
            obs_context = initialize_observability(
                agent.observability, agent.name, verbose=verbose, quiet=quiet
            )
        else:
            # Traditional logging
            setup_logging(verbose=verbose, quiet=effective_quiet)

        logger.info(
            f"Chat command invoked: config={agent_config}, "
            f"verbose={verbose}, quiet={quiet}, observability={observability}, "
            f"max_messages={max_messages}, force_ingest={force_ingest}"
        )
        logger.debug(f"Loading agent configuration from {agent_config}")
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

        # Set the base directory context for resolving relative paths in tools
        agent_dir = str(Path(agent_config).parent.resolve())
        agent_base_dir.set(agent_dir)
        logger.debug(f"Set agent_base_dir context: {agent_base_dir.get()}")

        # Resolve execution config with 6-level priority hierarchy
        # CLI flags > agent.yaml > project config > user config > env vars > defaults
        cli_config = ExecutionConfig(
            verbose=verbose if verbose else None,
            quiet=quiet if quiet else None,  # Set to True if True, else None
        )

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
            f"Resolved execution config: verbose={resolved_config.verbose}, "
            f"quiet={resolved_config.quiet}, llm_timeout={resolved_config.llm_timeout}"
        )

        # Run async chat session
        logger.debug("Starting chat session runtime")
        asyncio.run(
            _run_chat_session(
                agent=agent,
                agent_config_path=Path(agent_config),
                verbose=resolved_config.verbose or False,
                quiet=resolved_config.quiet or False,
                enable_observability=observability,
                max_messages=max_messages,
                force_ingest=force_ingest,
                llm_timeout=resolved_config.llm_timeout,
                observability_enabled=obs_context is not None,
            )
        )

        # Normal exit (user typed exit/quit)
        logger.info("Chat session ended normally")
        sys.exit(0)

    except ConfigError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        click.secho("Error: Failed to load agent configuration", fg="red", err=True)
        click.echo(f"  {str(e)}", err=True)
        sys.exit(1)
    except AgentInitializationError as e:
        logger.error(f"Agent initialization error: {e}", exc_info=True)
        click.secho("Error: Failed to initialize agent", fg="red", err=True)
        click.echo(f"  {str(e)}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Chat interrupted by user (Ctrl+C)")
        click.echo()
        click.secho("Goodbye!", fg="yellow")
        sys.exit(130)
    except ExecutionError as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)
    finally:
        # Shutdown observability if it was initialized
        if obs_context:
            shutdown_observability(obs_context)


async def _run_chat_session(
    agent: Agent,
    agent_config_path: Path,
    verbose: bool,
    quiet: bool,
    enable_observability: bool,
    max_messages: int,
    force_ingest: bool = False,
    llm_timeout: int | None = None,
    observability_enabled: bool = False,
) -> None:
    """Run the interactive chat session.

    Args:
        agent: Loaded Agent configuration
        agent_config_path: Path to agent.yaml file
        verbose: Enable detailed tool execution display
        quiet: Suppress logging output
        enable_observability: Enable OpenTelemetry tracing
        max_messages: Maximum messages before warning
        force_ingest: Force re-ingestion of vector store source files
        llm_timeout: LLM API call timeout in seconds
        observability_enabled: Whether OTel tracing is enabled

    Raises:
        KeyboardInterrupt: When user interrupts (Ctrl+C)
    """
    # Create parent span for chat command if observability is enabled
    if observability_enabled:
        from holodeck.lib.observability import get_tracer

        tracer = get_tracer(__name__)
        span_context: Any = tracer.start_as_current_span("holodeck.cli.chat")
    else:
        span_context = nullcontext()

    with span_context:
        # Initialize session manager
        try:
            chat_config = ChatConfig(
                agent_config_path=Path(agent_config_path),
                verbose=verbose,
                enable_observability=enable_observability,
                max_messages=max_messages,
                force_ingest=force_ingest,
                llm_timeout=llm_timeout,
            )
            session_manager = ChatSessionManager(
                agent_config=agent,
                config=chat_config,
            )
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}", exc_info=True)
            raise AgentInitializationError(agent.name, str(e)) from e

        # Start session
        try:
            logger.debug("Starting chat session")
            await session_manager.start()
        except Exception as e:
            logger.error(f"Failed to start session: {e}", exc_info=True)
            raise AgentInitializationError(agent.name, str(e)) from e

        try:
            # Display welcome message
            click.secho(f"\nStarting chat with {agent.name}...", fg="green", bold=True)
            click.echo("Type 'exit' or 'quit' to end session.")
            click.echo()

            # Initialize progress indicator
            progress = ChatProgressIndicator(
                max_messages=max_messages,
                quiet=quiet,
                verbose=verbose,
            )

            # REPL loop
            while True:
                try:
                    # Get user input
                    user_input = click.prompt("You", default="").strip()

                    # Check for exit commands
                    if user_input.lower() in ("exit", "quit"):
                        click.secho("Goodbye!", fg="yellow")
                        break

                    # Skip empty messages (validation handled in session)
                    if not user_input:
                        continue

                    # Start spinner (always show, regardless of quiet mode)
                    spinner = None
                    if sys.stdout.isatty():
                        spinner = ChatSpinnerThread(progress)
                        spinner.start()

                    try:
                        logger.debug(f"Processing user message: {user_input[:50]}...")
                        response = await session_manager.process_message(user_input)

                        # Stop spinner
                        if spinner:
                            spinner.stop()
                            spinner.join()

                        # Display agent response
                        if response:
                            # Update progress
                            progress.update(response)

                            # Display response with status
                            if verbose:
                                click.echo(progress.get_status_panel())
                                click.echo(f"Agent: {response.content}\n")
                            else:
                                # Inline status
                                status = progress.get_status_inline()
                                click.echo(f"Agent: {response.content} {status}\n")

                            logger.debug(
                                f"Agent responded with {len(response.tool_executions)} "
                                f"tool executions"
                            )

                        # Check for context limit warning
                        if session_manager.should_warn_context_limit():
                            click.secho(
                                "⚠️  Approaching context limit. Consider a new session.",
                                fg="yellow",
                            )
                            click.echo()

                    except Exception as e:
                        # Stop spinner on error
                        if spinner:
                            spinner.stop()
                            spinner.join()

                        # Display error but continue session (don't crash)
                        logger.warning(f"Error processing message: {e}")
                        click.secho(f"Error: {str(e)}", fg="red")
                        click.echo()

                except EOFError:
                    # Handle Ctrl+D
                    click.echo()
                    click.secho("Goodbye!", fg="yellow")
                    break

        except KeyboardInterrupt:
            # Handle Ctrl+C
            click.echo()
            click.secho("Goodbye!", fg="yellow")
            raise
        finally:
            # Cleanup
            try:
                logger.debug("Terminating chat session")
                await session_manager.terminate()
            except Exception as e:
                logger.warning(f"Error during session cleanup: {e}")
