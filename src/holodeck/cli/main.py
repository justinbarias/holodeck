"""Main Click CLI group for HoloDeck.

This module defines the main CLI entry point and registers all available
commands (init, etc.). It's the root command group that all subcommands attach to.
"""

import os
from pathlib import Path

# =============================================================================
# CRITICAL: Set telemetry/library env vars BEFORE any library imports.
# - DEEPEVAL_TELEMETRY_OPT_OUT: Prevents deepeval from setting a TracerProvider
# - SK env var: Enables Semantic Kernel telemetry
# - GRPC_VERBOSITY: Silence gRPC C-core INFO logs — notably the
#   "FD from fork parent still in poll list" lines grpcio prints straight to
#   stderr after a fork (multiprocessing / pytest-xdist) when a channel exists
#   (OTLP gRPC exporter, qdrant gRPC, deepeval). These bypass Python logging,
#   so the C-core verbosity knob is the only way to quiet them. setdefault
#   preserves an explicit operator override (e.g. GRPC_VERBOSITY=DEBUG).
# =============================================================================
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault(
    "SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "true"
)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from importlib.metadata import PackageNotFoundError, version

import click
from dotenv import load_dotenv

from holodeck.cli.lazy_group import LazyGroup

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


# Each command module loads only when that command runs (or ``--help`` lists
# it): `holodeck init` must not import the evaluation, SDK and server stacks.
_COMMANDS = {
    name: f"holodeck.cli.commands.{name}:{name}"
    for name in (
        "chat",
        "config",
        "deploy",
        "init",
        "mcp",
        "serve",
        "test",
        "vectorstore",
        "worker",
    )
}


@click.group(cls=LazyGroup, lazy_subcommands=_COMMANDS, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="holodeck")
@click.pass_context
def main(ctx: click.Context) -> None:
    """HoloDeck - Experimentation platform for AI agents.

    Commands:
        init    Initialize a new agent project
        test    Run agent test cases
        chat    Interactive chat session with an agent
        serve   Start an HTTP server exposing an agent
        deploy  Build and deploy agent containers
        worker  Run a Temporal worker serving agents as activities
        vectorstore  Inspect and migrate retrieval collections

    Initialize and manage AI agent projects with YAML configuration.
    """
    # Show help if no command is provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


if __name__ == "__main__":
    main()
