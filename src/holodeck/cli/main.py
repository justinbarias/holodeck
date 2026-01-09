"""Main Click CLI group for HoloDeck.

This module defines the main CLI entry point and registers all available
commands (init, etc.). It's the root command group that all subcommands attach to.
"""

import os
from pathlib import Path

# =============================================================================
# CRITICAL: Set telemetry env vars BEFORE any library imports.
# - DEEPEVAL_TELEMETRY_OPT_OUT: Prevents deepeval from setting a TracerProvider
# - SK env var: Enables Semantic Kernel telemetry
# =============================================================================
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault(
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "true"
)

from importlib.metadata import PackageNotFoundError, version

import click
from dotenv import load_dotenv

try:
    __version__ = version("holodeck-ai")
except PackageNotFoundError:
    # Package not installed, development mode
    __version__ = "0.0.0.dev0"


def _load_dotenv_files() -> None:
    """Load .env files from current directory and user home.

    Priority (highest to lowest):
    1. Shell environment variables (never overwritten)
    2. .env in CWD (project-level config)
    3. ~/.holodeck/.env (user-level defaults)

    With override=False, the first value set wins. So we load
    project .env first, then home .env fills any remaining gaps.
    """
    # Load project-level .env first (higher priority of .env files)
    project_env = Path.cwd() / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)

    # Load user-level .env second (fills gaps, never overrides)
    user_env = Path.home() / ".holodeck" / ".env"
    if user_env.exists():
        load_dotenv(user_env, override=False)


# Load environment variables before CLI initialization
_load_dotenv_files()


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="holodeck")
@click.pass_context
def main(ctx: click.Context) -> None:
    """HoloDeck - Experimentation platform for AI agents.

    Commands:
        init   Initialize a new agent project
        test   Run agent test cases
        chat   Interactive chat session with an agent
        serve  Start an HTTP server exposing an agent

    Initialize and manage AI agent projects with YAML configuration.
    """
    # Show help if no command is provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Import and register commands
from holodeck.cli.commands.chat import chat  # noqa: E402, F401
from holodeck.cli.commands.config import config  # noqa: E402, F401
from holodeck.cli.commands.init import init  # noqa: E402, F401
from holodeck.cli.commands.mcp import mcp  # noqa: E402, F401
from holodeck.cli.commands.serve import serve  # noqa: E402, F401
from holodeck.cli.commands.test import test  # noqa: E402, F401

# Register commands
main.add_command(init)
main.add_command(test)
main.add_command(chat)
main.add_command(config)
main.add_command(mcp)
main.add_command(serve)


if __name__ == "__main__":
    main()
