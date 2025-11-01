"""Main Click CLI group for HoloDeck.

This module defines the main CLI entry point and registers all available
commands (init, etc.). It's the root command group that all subcommands attach to.
"""

import click

__version__ = "0.1.0"


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="holodeck")
@click.pass_context
def main(ctx: click.Context) -> None:
    """HoloDeck - Experimentation platform for AI agents.

    Initialize and manage AI agent projects with YAML configuration.
    """
    # Show help if no command is provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Import and register commands
from holodeck.cli.commands.init import init  # noqa: E402, F401
from holodeck.cli.commands.test import test  # noqa: E402, F401

# Register commands
main.add_command(init)
main.add_command(test)


if __name__ == "__main__":
    main()
