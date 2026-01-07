"""CLI command for executing agent test cases.

Implements the 'holodeck test' command for running test suites against agents
with evaluation metrics and report generation.
"""

import asyncio
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import click

from holodeck.lib.errors import ConfigError, EvaluationError, ExecutionError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.observability import (
    ObservabilityContext,
    initialize_observability,
    shutdown_observability,
)
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.lib.test_runner.progress import ProgressIndicator
from holodeck.lib.test_runner.reporter import generate_markdown_report
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import TestReport, TestResult

logger = get_logger(__name__)


class SpinnerThread(threading.Thread):
    """Background thread for spinner animation."""

    def __init__(self, progress: ProgressIndicator) -> None:
        """Initialize spinner thread.

        Args:
            progress: ProgressIndicator instance
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
                # Use \r to overwrite line, flush to ensure display
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()
            time.sleep(0.1)
        self._running = False

    def stop(self) -> None:
        """Stop spinner animation."""
        self._stop_event.set()
        if self._running:
            # Clear spinner line
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()


@click.command()
@click.argument("agent_config", type=click.Path(exists=True), default="agent.yaml")
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Path to save test report file (JSON or Markdown)",
)
@click.option(
    "--format",
    type=click.Choice(["json", "markdown"]),
    default=None,
    help="Report format (auto-detect from extension if not specified)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output with debug information",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress output (summary still shown)",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="LLM execution timeout in seconds",
)
@click.option(
    "--force-ingest",
    "-f",
    is_flag=True,
    help="Force re-ingestion of all vector store source files",
)
def test(
    agent_config: str,
    output: str | None,
    format: str | None,
    verbose: bool,
    quiet: bool,
    timeout: int | None,
    force_ingest: bool,
) -> None:
    """Execute agent test cases with evaluation metrics.

    Runs test cases defined in the agent configuration file and displays
    pass/fail status with evaluation metric scores.

    AGENT_CONFIG is the path to the agent.yaml configuration file.
    """
    # Initialize observability context (will be set if observability enabled)
    obs_context: ObservabilityContext | None = None
    effective_quiet = quiet and not verbose

    start_time = time.time()

    try:
        # Create execution config from CLI options
        cli_config = None
        if timeout is not None or verbose or quiet:
            cli_config = ExecutionConfig(
                llm_timeout=timeout,
                file_timeout=None,
                download_timeout=None,
                cache_enabled=None,
                cache_dir=None,
                verbose=verbose or None,
                quiet=quiet or None,
            )

        # Load agent config FIRST to check observability setting
        from holodeck.config.context import agent_base_dir
        from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
        from holodeck.config.loader import ConfigLoader

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
            f"Test command invoked: config={agent_config}, "
            f"verbose={verbose}, quiet={quiet}, timeout={timeout}, "
            f"force_ingest={force_ingest}"
        )
        logger.debug(f"Loading agent configuration from {agent_config}")
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

        # Set the base directory context for resolving relative paths in tools
        agent_dir = str(Path(agent_config).parent.resolve())
        agent_base_dir.set(agent_dir)
        logger.debug(f"Set agent_base_dir context: {agent_base_dir.get()}")

        # Resolve execution config once (CLI > agent.yaml > project > user > defaults)
        project_config = loader.load_project_config(agent_dir)
        project_execution = project_config.execution if project_config else None
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

        # Get total test count
        total_tests = len(agent.test_cases) if agent.test_cases else 0
        logger.info(f"Found {total_tests} test cases to execute")

        # Initialize progress indicator
        progress = ProgressIndicator(
            total_tests=total_tests, quiet=quiet, verbose=verbose
        )

        # Initialize spinner thread
        spinner_thread = SpinnerThread(progress)

        # Define callbacks
        def on_test_start(test_case: TestCaseModel) -> None:
            """Start spinner for new test."""
            progress.start_test(test_case.name or "Test")
            if not spinner_thread.is_alive() and not quiet and sys.stdout.isatty():
                # Start thread if not running (it handles its own loop)
                # Note: We can't restart a thread, so we might need a new instance
                # or just keep it running and use events.
                # Simpler approach: Start it once before execution and use flags.
                pass

        def progress_callback(result: TestResult) -> None:
            """Update progress indicator and display progress line."""
            # Temporarily stop spinner to print result
            # In a real implementation with threading, we might need a lock
            # But here we just clear the line in the main thread (via update)

            # Update progress state
            progress.update(result)

            # Get formatted result line
            progress_line = progress.get_progress_line()

            if progress_line:  # Only print if not empty (respects quiet mode)
                # Clear current line (handled by \r in spinner)
                sys.stdout.write("\r" + " " * 60 + "\r")
                click.echo(progress_line)

        # Initialize executor with pre-loaded configs to avoid duplicate I/O
        logger.debug("Initializing test executor")
        executor = TestExecutor(
            agent_config_path=agent_config,
            progress_callback=progress_callback,
            on_test_start=on_test_start,
            force_ingest=force_ingest,
            agent_config=agent,
            resolved_execution_config=resolved_config,
        )

        # Run tests asynchronously
        logger.info("Starting test execution")

        # Start spinner thread
        if not quiet and sys.stdout.isatty():
            spinner_thread.start()

        # Determine if observability is enabled for span creation
        observability_enabled = obs_context is not None

        async def run_tests_with_cleanup() -> TestReport:
            """Execute tests and ensure proper cleanup of MCP plugins."""
            # Create parent span for test command if observability is enabled
            if observability_enabled:
                from holodeck.lib.observability import get_tracer

                tracer = get_tracer(__name__)
                span_context: Any = tracer.start_as_current_span("holodeck.cli.test")
            else:
                span_context = nullcontext()

            with span_context:
                try:
                    return await executor.execute_tests()
                finally:
                    await executor.shutdown()

        # Run tests
        try:
            report = asyncio.run(run_tests_with_cleanup())
        finally:
            # Ensure spinner stops
            if spinner_thread.is_alive():
                spinner_thread.stop()
                spinner_thread.join()

        elapsed_time = time.time() - start_time
        logger.info(
            f"Test execution completed in {elapsed_time:.2f}s - "
            f"{report.summary.passed} passed, {report.summary.failed} failed"
        )

        # Display summary (always shown, even in quiet mode)
        summary_text = progress.get_summary()
        click.echo(summary_text)

        # Save report if output specified
        if output:
            logger.debug(f"Saving report to {output} (format={format})")
            _save_report(report, output, format)
            logger.info(f"Report saved successfully to {output}")

        # Exit with appropriate code
        if report.summary.failed > 0:
            logger.info("Exiting with failure status (failed tests)")
            sys.exit(1)
        else:
            logger.info("Exiting with success status (all tests passed)")
            sys.exit(0)

    except ConfigError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        click.echo(f"Configuration Error: {e}", err=True)
        sys.exit(2)
    except ExecutionError as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        click.echo(f"Execution Error: {e}", err=True)
        sys.exit(3)
    except EvaluationError as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        click.echo(f"Evaluation Error: {e}", err=True)
        sys.exit(4)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(3)
    finally:
        # Shutdown observability if it was initialized
        if obs_context:
            shutdown_observability(obs_context)


def _save_report(report: TestReport, output: str, format: str | None) -> None:
    """Save test report to file in specified format.

    Args:
        report: TestReport instance to save
        output: Output file path
        format: Report format (json/markdown). If None, auto-detect from extension.
    """
    output_path = Path(output)

    # Determine format if not specified
    if format is None:
        if output.endswith(".json"):
            format = "json"
        elif output.endswith(".md") or output.endswith(".markdown"):
            format = "markdown"
        else:
            format = "json"  # Default to JSON
        logger.debug(f"Auto-detected report format: {format}")

    # Generate report content
    logger.debug(f"Generating {format} report")
    if format == "json":
        # Use pydantic's model_dump_json method
        content = report.model_dump_json(indent=2)
    else:  # markdown
        # Generate markdown format using reporter module
        content = generate_markdown_report(report)

    # Write to file (overwrites if exists)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        click.echo(f"Report saved to {output}")
    except OSError as e:
        logger.error(f"Failed to write report to {output}: {e}")
        raise
