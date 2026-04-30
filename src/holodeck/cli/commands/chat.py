"""CLI command for interactive chat with agents.

Implements the 'holodeck chat' command for multi-turn conversations with agents
including message validation, tool execution streaming, and optional observability.
"""

import asyncio
import contextlib
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import click

from holodeck.chat import ChatSessionManager, LiveComposer, ToolsPanel
from holodeck.chat.executor import AgentExecutor, AgentResponse
from holodeck.chat.progress import ChatProgressIndicator
from holodeck.lib.backends.base import ToolEvent
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

_RENDER_INTERVAL_SEC = 0.1


async def _drain_tool_events(
    queue: asyncio.Queue[ToolEvent],
    panel: ToolsPanel,
    stop_event: asyncio.Event,
) -> None:
    """Forward tool events into *panel* until *stop_event* is set.

    Uses a short timeout on each ``get`` so the loop exits promptly when
    the chat turn ends, even if no further events arrive.
    """
    while not stop_event.is_set():
        try:
            evt = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        panel.apply(evt)


def _drain_remaining(queue: asyncio.Queue[ToolEvent], panel: ToolsPanel) -> None:
    """Apply any events buffered between the last drainer tick and now."""
    while True:
        try:
            evt = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        panel.apply(evt)


async def _paint_panel_loop(
    panel: ToolsPanel,
    progress: ChatProgressIndicator,
    composer: LiveComposer,
    stop_event: asyncio.Event,
) -> None:
    """Drive panel repaints via *composer* at 10 fps until *stop_event*.

    The composer keeps the panel pinned below the streaming-text cursor,
    so this task can run for the full turn — text writes interleave with
    panel repaints without trampling each other.
    """
    while not stop_event.is_set():
        lines = list(panel.render_lines())
        spinner = progress.get_spinner_line()
        if spinner:
            lines.append(spinner)
        await composer.update_panel(lines)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_RENDER_INTERVAL_SEC)
        except asyncio.TimeoutError:
            continue
    # One last paint so end-of-turn state (e.g. final tool just finished)
    # is reflected before the composer tears the panel down.
    lines = list(panel.render_lines())
    spinner = progress.get_spinner_line()
    if spinner:
        lines.append(spinner)
    await composer.update_panel(lines)


async def _ensure_executor_ready(executor: AgentExecutor) -> None:
    """Eagerly initialise backend + session so ``tool_event_queue`` is live.

    Mirrors :func:`holodeck.serve.protocols.agui.AGUIProtocol.handle_request`'s
    eager-init pattern.  Gracefully degrades if the executor doesn't expose
    the private hook (e.g. test mocks).
    """
    ensure = getattr(executor, "_ensure_backend_and_session", None)
    if callable(ensure) and asyncio.iscoroutinefunction(ensure):
        await ensure()


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
        # Load agent config and resolve execution config in one call
        from holodeck.config.loader import load_agent_with_config

        cli_config = ExecutionConfig(
            verbose=verbose if verbose else None,
            quiet=quiet if quiet else None,
        )

        agent, resolved_config, _loader = load_agent_with_config(
            agent_config, cli_config=cli_config
        )

        # Determine logging strategy: OTel replaces setup_logging when enabled
        if agent.observability and agent.observability.enabled:
            obs_context = initialize_observability(
                agent.observability, agent.name, verbose=verbose, quiet=quiet
            )
        else:
            setup_logging(verbose=verbose, quiet=effective_quiet)

        logger.info(
            f"Chat command invoked: config={agent_config}, "
            f"verbose={verbose}, quiet={quiet}, observability={observability}, "
            f"max_messages={max_messages}, force_ingest={force_ingest}"
        )
        logger.debug(f"Loading agent configuration from {agent_config}")
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

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

                    try:
                        logger.debug(f"Processing user message: {user_input[:50]}...")

                        # Eager-init so the backend's tool_event_queue is
                        # live before we spawn the drainer (mirrors the
                        # AGUI protocol path in holodeck serve).
                        executor = session_manager._executor
                        if executor is not None:
                            await _ensure_executor_ready(executor)
                        queue = (
                            executor.tool_event_queue if executor is not None else None
                        )

                        panel = ToolsPanel()
                        composer = LiveComposer()
                        stop_event = asyncio.Event()
                        drain_task: asyncio.Task[None] | None = None
                        if queue is not None:
                            drain_task = asyncio.create_task(
                                _drain_tool_events(queue, panel, stop_event)
                            )

                        await composer.begin()
                        click.echo("Agent: ", nl=False)
                        sys.stdout.flush()
                        paint_task: asyncio.Task[None] = asyncio.create_task(
                            _paint_panel_loop(panel, progress, composer, stop_event)
                        )

                        start_time = time.time()
                        chunks: list[str] = []
                        try:
                            async for (
                                chunk
                            ) in session_manager.process_message_streaming(user_input):
                                await composer.write_text(chunk)
                                chunks.append(chunk)
                        finally:
                            stop_event.set()
                            await paint_task
                            if drain_task is not None:
                                try:
                                    await asyncio.wait_for(drain_task, timeout=0.25)
                                except asyncio.TimeoutError:
                                    drain_task.cancel()
                                    with contextlib.suppress(asyncio.CancelledError):
                                        await drain_task
                            if queue is not None:
                                _drain_remaining(queue, panel)
                            await composer.end()

                        elapsed = time.time() - start_time

                        click.echo()  # newline after streamed content

                        # Build minimal AgentResponse for progress tracking.
                        # Token usage and tool details unavailable via streaming.
                        response = AgentResponse(
                            content="".join(chunks),
                            tool_executions=[],
                            tokens_used=None,
                            execution_time=elapsed,
                        )

                        # Update progress
                        progress.update(response)
                        progress.set_active_snapshot(panel.snapshot())

                        # Display status
                        if verbose:
                            click.echo(progress.get_status_panel())
                        else:
                            status = progress.get_status_inline()
                            click.echo(f"{status}\n")

                        logger.debug(f"Streamed response in {elapsed:.2f}s")

                        # Check for context limit warning
                        if session_manager.should_warn_context_limit():
                            click.secho(
                                "⚠️  Approaching context limit. Consider a new session.",
                                fg="yellow",
                            )
                            click.echo()

                    except Exception as e:
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
