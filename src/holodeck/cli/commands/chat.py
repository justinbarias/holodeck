"""CLI command for interactive chat with agents.

Implements the 'holodeck chat' command for multi-turn conversations with agents
including message validation, tool execution streaming, and optional observability.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

import click

from holodeck.chat import ChatSessionManager
from holodeck.chat.progress import ChatProgressIndicator
from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import AgentInitializationError, ConfigError, ExecutionError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.models.agent import Agent
from holodeck.models.chat import ChatConfig

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
@click.argument("agent_config", type=click.Path(exists=True))
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed tool execution (parameters, internal state)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=True,
    help="Suppress logging output (enabled by default)",
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
def chat(
    agent_config: str,
    verbose: bool,
    quiet: bool,
    observability: bool,
    max_messages: int,
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
    # Reconfigure logging based on CLI flags
    setup_logging(verbose=verbose, quiet=quiet)

    logger.info(
        f"Chat command invoked: config={agent_config}, "
        f"verbose={verbose}, quiet={quiet}, observability={observability}, "
        f"max_messages={max_messages}"
    )

    try:
        # Load agent configuration
        from holodeck.config.context import agent_base_dir

        logger.debug(f"Loading agent configuration from {agent_config}")
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

        # Set the base directory context for resolving relative paths in tools
        agent_base_dir.set(str(Path(agent_config).parent.resolve()))
        logger.debug(f"Set agent_base_dir context: {agent_base_dir.get()}")

        # Run async chat session
        logger.debug("Starting chat session runtime")
        asyncio.run(
            _run_chat_session(
                agent=agent,
                agent_config_path=Path(agent_config),
                verbose=verbose,
                quiet=quiet,
                enable_observability=observability,
                max_messages=max_messages,
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


async def _run_chat_session(
    agent: Agent,
    agent_config_path: Path,
    verbose: bool,
    quiet: bool,
    enable_observability: bool,
    max_messages: int,
) -> None:
    """Run the interactive chat session.

    Args:
        agent: Loaded Agent configuration
        verbose: Enable detailed tool execution display
        quiet: Suppress logging output
        enable_observability: Enable OpenTelemetry tracing
        max_messages: Maximum messages before warning

    Raises:
        KeyboardInterrupt: When user interrupts (Ctrl+C)
    """
    # Initialize session manager
    try:
        chat_config = ChatConfig(
            agent_config_path=Path(agent_config_path),  # Placeholder path for session
            verbose=verbose,
            enable_observability=enable_observability,
            max_messages=max_messages,
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
