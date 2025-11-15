"""CLI command for executing agent test cases.

Implements the 'holodeck test' command for running test suites against agents
with evaluation metrics and report generation.
"""

import asyncio
import sys
import time
from pathlib import Path

import click

from holodeck.lib.errors import ConfigError, EvaluationError, ExecutionError
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.test_runner.executor import TestExecutor
from holodeck.lib.test_runner.progress import ProgressIndicator
from holodeck.models.config import ExecutionConfig
from holodeck.models.test_result import TestReport, TestResult

logger = get_logger(__name__)


@click.command()
@click.argument("agent_config", type=click.Path(exists=True))
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
def test(
    agent_config: str,
    output: str | None,
    format: str | None,
    verbose: bool,
    quiet: bool,
    timeout: int | None,
) -> None:
    """Execute agent test cases with evaluation metrics.

    Runs test cases defined in the agent configuration file and displays
    pass/fail status with evaluation metric scores.

    AGENT_CONFIG is the path to the agent.yaml configuration file.
    """
    # Reconfigure logging based on CLI flags
    setup_logging(verbose=verbose, quiet=quiet)

    logger.info(
        f"Test command invoked: config={agent_config}, "
        f"verbose={verbose}, quiet={quiet}, timeout={timeout}"
    )

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

        # Load agent config to get test count for progress indicator
        from holodeck.config.loader import ConfigLoader

        logger.debug(f"Loading agent configuration from {agent_config}")
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)
        logger.info(f"Agent configuration loaded successfully: {agent.name}")

        # Get total test count
        total_tests = len(agent.test_cases) if agent.test_cases else 0
        logger.info(f"Found {total_tests} test cases to execute")

        # Initialize progress indicator
        progress = ProgressIndicator(
            total_tests=total_tests, quiet=quiet, verbose=verbose
        )

        # Define progress callback
        def progress_callback(result: TestResult) -> None:
            """Update progress indicator and display progress line."""
            progress.update(result)
            progress_line = progress.get_progress_line()
            if progress_line:  # Only print if not empty (respects quiet mode)
                click.echo(progress_line)

        # Initialize executor with progress callback
        logger.debug("Initializing test executor")
        executor = TestExecutor(
            agent_config_path=agent_config,
            execution_config=cli_config,
            progress_callback=progress_callback,
        )

        # Run tests asynchronously
        logger.info("Starting test execution")
        report = asyncio.run(executor.execute_tests())
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
        # Generate markdown format manually
        content = _generate_markdown_report(report)

    # Write to file
    try:
        output_path.write_text(content)
        click.echo(f"Report saved to {output}")
    except OSError as e:
        logger.error(f"Failed to write report to {output}: {e}")
        raise


def _generate_markdown_report(report: TestReport) -> str:
    """Generate markdown formatted report from TestReport.

    Args:
        report: TestReport instance

    Returns:
        Markdown formatted report string
    """
    lines: list[str] = []

    # Header
    lines.append(f"# Test Report: {report.agent_name}")
    lines.append(f"**Config**: {report.agent_config_path}")
    lines.append(f"**Timestamp**: {report.timestamp}")
    lines.append(f"**HoloDeck Version**: {report.holodeck_version}")
    lines.append("")

    # Summary
    summary = report.summary
    lines.append("## Summary")
    lines.append(f"- **Total Tests**: {summary.total_tests}")
    lines.append(f"- **Passed**: {summary.passed}")
    lines.append(f"- **Failed**: {summary.failed}")
    lines.append(f"- **Pass Rate**: {summary.pass_rate:.1f}%")
    lines.append(f"- **Total Duration**: {summary.total_duration_ms / 1000:.2f}s")
    lines.append("")

    # Results table
    if report.results:
        lines.append("## Test Results")
        lines.append("| Test | Status | Duration | Errors |")
        lines.append("|------|--------|----------|--------|")

        for result in report.results:
            status = "✓ Pass" if result.passed else "✗ Fail"
            duration = f"{result.execution_time_ms}ms"
            errors = len(result.errors)
            lines.append(f"| {result.test_name} | {status} | {duration} | {errors} |")

    return "\n".join(lines)
