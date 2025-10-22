"""Click command for initializing new HoloDeck projects.

This module implements the 'holodeck init' command which creates a new
project directory with templates, configuration, and example files.
"""

import click


@click.command(name="init")
@click.argument("project_name")
@click.option(
    "--template",
    default="conversational",
    type=click.Choice(["conversational", "research", "customer-support"]),
    help="Project template to use",
)
@click.option(
    "--description",
    default=None,
    help="Brief description of the agent",
)
@click.option(
    "--author",
    default=None,
    help="Project creator name",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing project directory",
)
def init(
    project_name: str,
    template: str,
    description: str | None,
    author: str | None,
    force: bool,
) -> None:
    """Initialize a new HoloDeck agent project.

    Creates a new project directory with configuration, examples, and test cases.

    Example:

        holodeck init my-agent

        holodeck init my-agent --template research --author "Jane Doe"
    """
    # Placeholder - implementation will be in T036-T039
    click.echo(f"Initializing project: {project_name}")
    click.echo(f"Template: {template}")
    if description:
        click.echo(f"Description: {description}")
    if author:
        click.echo(f"Author: {author}")
    if force:
        click.echo("Force mode enabled")
