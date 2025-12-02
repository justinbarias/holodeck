"""Click command group for HoloDeck configuration management.

This module implements the 'holodeck config' command group and its subcommands.
"""

import click

from holodeck.config.manager import ConfigManager


@click.group(name="config")
def config() -> None:
    """Manage HoloDeck configuration."""
    pass


@config.command(name="init")
@click.option(
    "-g",
    "--global",
    "global_config",
    is_flag=True,
    help="Initialize global configuration in ~/.holodeck/config.yaml",
)
@click.option(
    "-p",
    "--project",
    "project_config",
    is_flag=True,
    help="Initialize project configuration in config.yaml",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing configuration file without prompting",
)
def init(
    global_config: bool,
    project_config: bool,
    force: bool,
) -> None:
    """Initialize HoloDeck global or project configuration.

    Creates a new configuration file with default settings. By default, this command
    will prompt you to choose between global (~/.holodeck/config.yaml) or project
    (config.yaml) configuration initialization.

    EXAMPLES:

        Initialize global configuration:
            holodeck config init -g

        Initialize project configuration:
            holodeck config init -p

        Overwrite existing configuration:
            holodeck config init -g --force

    For more information, see: https://useholodeck.ai/docs/config
    """
    # Determine which config to initialize
    if not global_config and not project_config:
        # Prompt user to choose
        choice = click.prompt(
            "Initialize global (~/.holodeck/config.yaml) or "
            "project (config.yaml) configuration?",
            type=click.Choice(["g", "p"]),
            default="g",
        )
        if choice == "g":
            global_config = True
        else:
            project_config = True

    # Determine config file path
    config_path, config_type = ConfigManager.get_config_path(
        global_config, project_config
    )

    # Check if config file already exists
    # Check if config file already exists
    if (
        config_path.exists()
        and not force
        and not click.confirm(
            f"Configuration file '{config_path}' already exists. "
            "Do you want to overwrite it?",
            default=False,
        )
    ):
        click.echo("Initialization cancelled.")
        return

    try:
        # Create default configuration
        default_config = ConfigManager.create_default_config()

        # Generate YAML content
        yaml_content = ConfigManager.generate_config_content(default_config)

        # Write to file
        ConfigManager.write_config(config_path, yaml_content)

        click.secho(
            f"✓ {config_type.capitalize()} configuration initialized successfully!",
            fg="green",
            bold=True,
        )
        click.echo(f"Configuration saved to: {config_path}")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Edit the configuration file to customize settings")
        click.echo("  2. Use the configuration in your agent projects")
        click.echo()

    except Exception as e:
        click.secho(f"✗ Failed to initialize configuration: {str(e)}", fg="red")
        raise click.Abort() from e
