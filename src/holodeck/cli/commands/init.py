"""Click command for initializing new HoloDeck projects.

This module implements the 'holodeck init' command which creates a new
project directory with templates, configuration, and example files.
"""

from pathlib import Path

import click

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.cli.utils.project_init import ProjectInitializer
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.project_config import ProjectInitInput


def validate_template(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate template parameter and provide helpful error messages.

    Args:
        ctx: Click context
        param: Click parameter
        value: Template name provided by user

    Returns:
        The validated template name

    Raises:
        click.BadParameter: If template is invalid
    """
    available = TemplateRenderer.list_available_templates()
    if value not in available:
        raise click.BadParameter(
            f"Unknown template '{value}'. Available templates: {', '.join(available)}"
        )
    return value


@click.command(name="init")
@click.argument("project_name")
@click.option(
    "--template",
    default="conversational",
    type=str,
    callback=validate_template,
    help="Project template: conversational (default), research, or customer-support",
)
@click.option(
    "--description",
    default=None,
    help="Brief description of what the agent does",
)
@click.option(
    "--author",
    default=None,
    help="Name of the project creator or organization",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing project directory without prompting",
)
def init(
    project_name: str,
    template: str,
    description: str | None,
    author: str | None,
    force: bool,
) -> None:
    """Initialize a new HoloDeck agent project.

    Creates a new project directory with all required configuration files,
    example instructions, tools templates, test cases, and data files.

    The generated project includes agent.yaml (main configuration), instructions/
    (system prompts), tools/ (custom function templates), data/ (sample datasets),
    and tests/ (evaluation test cases).

    TEMPLATES:

        conversational  - General-purpose conversational agent (default)
        research        - Research/analysis agent with vector search examples
        customer-support - Customer support agent with function tools

    EXAMPLES:

        Basic project with default (conversational) template:

            holodeck init my-chatbot

        Research-focused agent with metadata:

            holodeck init research-agent --template research \\
                --description "Research paper analysis and summarization" \\
                --author "Data Team"

        Customer support agent:

            holodeck init support-bot --template customer-support \\
                --description "Intelligent customer support chatbot" \\
                --author "Support Team"

        Overwrite existing project:

            holodeck init my-agent --force

    For more information, see: https://holodeck.ai/docs/getting-started
    """
    try:
        # Get current working directory as output directory
        output_dir = Path.cwd()

        # Check if project directory already exists (unless force)
        project_dir = output_dir / project_name
        if project_dir.exists() and not force:
            # Prompt user for confirmation
            if click.confirm(
                f"Project directory '{project_name}' already exists. "
                "Do you want to overwrite it?",
                default=False,
            ):
                force = True
            else:
                click.echo("Initialization cancelled.")
                return

        # Create project initialization input
        init_input = ProjectInitInput(
            project_name=project_name,
            template=template,
            description=description,
            author=author,
            output_dir=str(output_dir),
            overwrite=force,
        )

        # Initialize project
        initializer = ProjectInitializer()
        result = initializer.initialize(init_input)

        # Handle result
        if result.success:
            # Display success message
            click.echo()  # Blank line for readability
            click.secho("✓ Project initialized successfully!", fg="green", bold=True)
            click.echo()
            click.echo(f"Project: {result.project_name}")
            click.echo(f"Location: {result.project_path}")
            click.echo(f"Template: {result.template_used}")
            click.echo(f"Time: {result.duration_seconds:.2f}s")

            # Show created files (first 10, then summary)
            if result.files_created:
                click.echo()
                click.echo("Files created:")
                # Show key files first (config, instructions, tools, data)
                key_files = [
                    f
                    for f in result.files_created
                    if "agent.yaml" in f
                    or "system-prompt" in f
                    or "tools" in f
                    or "data" in f
                ]
                for file_path in key_files[:5]:
                    click.echo(f"  • {file_path}")
                if len(result.files_created) > 5:
                    remaining = len(result.files_created) - 5
                    click.echo(f"  ... and {remaining} more file(s)")

            click.echo()
            click.echo("Next steps:")
            click.echo(f"  1. cd {result.project_name}")
            click.echo("  2. Edit agent.yaml to configure your agent")
            click.echo("  3. Edit instructions/system-prompt.md to customize behavior")
            click.echo("  4. Add tools in tools/ directory")
            click.echo("  5. Update test_cases in agent.yaml")
            click.echo("  6. Run tests with: holodeck test agent.yaml")
            click.echo()
        else:
            # Display error message
            click.secho("✗ Project initialization failed", fg="red", bold=True)
            click.echo()
            for error in result.errors:
                click.secho(f"Error: {error}", fg="red")
            click.echo()
            raise click.Abort()

    except KeyboardInterrupt as e:
        # Handle Ctrl+C gracefully with cleanup
        click.echo()
        click.secho("Initialization cancelled by user.", fg="yellow")
        raise click.Abort() from e

    except (ValidationError, InitError) as e:
        # Handle known errors
        click.secho(f"Error: {str(e)}", fg="red")
        raise click.Abort() from e

    except Exception as e:
        # Handle unexpected errors
        click.secho(f"Unexpected error: {str(e)}", fg="red")
        raise click.Abort() from e
