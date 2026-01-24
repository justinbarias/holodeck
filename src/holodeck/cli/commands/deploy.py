"""CLI commands for deploying HoloDeck agents.

Implements the 'holodeck deploy' command group for building container images
and deploying agents to cloud providers.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from holodeck.lib.errors import ConfigError, DeploymentError, DockerNotAvailableError
from holodeck.lib.logging_config import get_logger, setup_logging

if TYPE_CHECKING:
    from holodeck.deploy.builder import BuildResult
    from holodeck.models.agent import Agent
    from holodeck.models.deployment import DeploymentConfig

logger = get_logger(__name__)


@click.group(name="deploy", invoke_without_command=True)
@click.pass_context
def deploy(ctx: click.Context) -> None:
    """Deploy HoloDeck agents to container registries and cloud providers.

    Subcommands:

        build   Build a container image for the agent

    Example:

        holodeck deploy build

        holodeck deploy build agent.yaml --dry-run
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # If no subcommand is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@deploy.command()
@click.argument(
    "agent_config",
    type=click.Path(exists=True),
    default="agent.yaml",
    required=False,
)
@click.option(
    "--tag",
    type=str,
    default=None,
    help="Custom tag for the image (overrides tag_strategy)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Build without using cache",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose debug logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress output",
)
def build(
    agent_config: str,
    tag: str | None,
    no_cache: bool,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Build a container image for the agent.

    AGENT_CONFIG is the path to the agent.yaml configuration file.

    Generates a Dockerfile from the agent configuration and builds
    a container image using Docker.

    Example:

        holodeck deploy build

        holodeck deploy build agent.yaml --tag v1.0.0

        holodeck deploy build --dry-run
    """
    # Setup logging based on verbosity
    if not quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    try:
        # Load agent configuration using ConfigLoader (same as chat, test, serve)
        from holodeck.config.loader import ConfigLoader

        agent_path = Path(agent_config).resolve()
        agent_dir = agent_path.parent

        if not quiet:
            click.echo(f"Loading agent configuration from {agent_config}...")

        # Load and validate agent using standard ConfigLoader
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(agent_config)
        agent_name = agent.name

        # Get deployment configuration from agent (merged by ConfigLoader)
        if not agent.deployment:
            raise ConfigError(
                field="deployment",
                message="No 'deployment' section found in agent configuration",
            )
        deployment_config = agent.deployment

        # Determine tag
        if tag:
            # Custom tag from CLI overrides config
            image_tag = tag
        else:
            # Use tag strategy from config
            from holodeck.deploy.builder import generate_tag

            image_tag = generate_tag(
                deployment_config.registry.tag_strategy,
                deployment_config.registry.custom_tag,
            )

        # Full image name
        registry_url = deployment_config.registry.url
        repository = deployment_config.registry.repository
        image_name = f"{registry_url}/{repository}"
        full_image_name = f"{image_name}:{image_tag}"

        if not quiet:
            click.echo()
            click.secho("Build Configuration:", bold=True)
            click.echo(f"  Agent:     {agent_name}")
            click.echo(f"  Image:     {full_image_name}")
            click.echo(f"  Protocol:  {deployment_config.protocol.value}")
            click.echo(f"  Port:      {deployment_config.port}")
            click.echo()

        if dry_run:
            click.secho("[DRY RUN] Would build image:", fg="yellow")
            click.echo(f"  Image: {full_image_name}")

            # Show generated Dockerfile
            dockerfile_content = _generate_dockerfile_content(
                agent, deployment_config, image_tag
            )
            click.echo()
            click.secho("Generated Dockerfile:", bold=True)
            for line in dockerfile_content.split("\n"):
                click.echo(f"  {line}")

            click.echo()
            click.secho("[DRY RUN] No image was built", fg="yellow")
            sys.exit(0)

        # Create build context
        if not quiet:
            click.echo("Preparing build context...")

        build_dir = _prepare_build_context(
            agent, deployment_config, agent_dir, image_tag
        )

        try:
            # Initialize builder
            if not quiet:
                click.echo("Connecting to Docker...")

            from holodeck.deploy.builder import ContainerBuilder, get_oci_labels

            builder = ContainerBuilder()

            # Generate OCI labels
            labels = get_oci_labels(
                agent_name=agent_name,
                version=image_tag,
            )

            # Build image
            if not quiet:
                click.echo(f"Building image {full_image_name}...")
                click.echo()

            build_kwargs: dict[str, Any] = {}
            if no_cache:
                build_kwargs["nocache"] = True

            result = builder.build(
                build_context=str(build_dir),
                image_name=image_name,
                tag=image_tag,
                labels=labels,
                **build_kwargs,
            )

            # Display build logs if verbose
            if verbose and result.log_lines:
                click.secho("Build Output:", bold=True)
                for line in result.log_lines:
                    if line.strip():
                        click.echo(f"  {line}")
                click.echo()

            # Success message
            _display_build_success(result, quiet)

        finally:
            # Cleanup build context
            shutil.rmtree(build_dir, ignore_errors=True)

    except DockerNotAvailableError as e:
        logger.error(f"Docker not available: {e}")
        click.secho("Error: Docker is not available", fg="red", err=True)
        click.echo(str(e), err=True)
        sys.exit(3)

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        click.secho("Error: Configuration error", fg="red", err=True)
        click.echo(f"  {e.message}", err=True)
        sys.exit(2)

    except DeploymentError as e:
        logger.error(f"Deployment error: {e}")
        click.secho(f"Error: {e.operation} failed", fg="red", err=True)
        click.echo(f"  {e.message}", err=True)
        sys.exit(3)

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(3)


def _generate_dockerfile_content(
    agent: Agent,
    deployment_config: DeploymentConfig,
    version: str,
) -> str:
    """Generate Dockerfile content for the agent.

    Args:
        agent: Loaded Agent configuration model
        deployment_config: Deployment configuration
        version: Version/tag for OCI labels

    Returns:
        Generated Dockerfile content
    """
    from holodeck.deploy.dockerfile import generate_dockerfile
    from holodeck.models.tool import VectorstoreTool

    # Collect instruction files
    instruction_files: list[str] = []
    if agent.instructions.file:
        instruction_files.append(agent.instructions.file)

    # Collect data directories (from vectorstore tools, etc.)
    data_directories: list[str] = []
    if agent.tools:
        for tool in agent.tools:
            if isinstance(tool, VectorstoreTool):
                source_path = tool.source
                if source_path:
                    # Get parent directory if it's a file
                    path = Path(source_path)
                    if path.suffix:
                        # It's a file, add parent directory
                        parent = str(path.parent)
                        if parent and parent != ".":
                            data_directories.append(parent + "/")
                    else:
                        data_directories.append(str(path) + "/")

    # Remove duplicates
    data_directories = list(set(data_directories))

    return generate_dockerfile(
        agent_name=agent.name,
        port=deployment_config.port,
        protocol=deployment_config.protocol.value,
        version=version,
        instruction_files=instruction_files if instruction_files else None,
        data_directories=data_directories if data_directories else None,
        environment=deployment_config.environment or None,
    )


def _prepare_build_context(
    agent: Agent,
    deployment_config: DeploymentConfig,
    agent_dir: Path,
    version: str,
) -> Path:
    """Prepare build context directory with all required files.

    Args:
        agent: Loaded Agent configuration model
        deployment_config: Deployment configuration
        agent_dir: Directory containing agent.yaml
        version: Version for Dockerfile labels

    Returns:
        Path to temporary build context directory
    """
    from holodeck.models.tool import VectorstoreTool

    # Create temporary build directory
    build_dir = Path(tempfile.mkdtemp(prefix="holodeck-build-"))

    # Generate and write Dockerfile
    dockerfile_content = _generate_dockerfile_content(agent, deployment_config, version)
    (build_dir / "Dockerfile").write_text(dockerfile_content)

    # Copy agent.yaml
    shutil.copy2(agent_dir / "agent.yaml", build_dir / "agent.yaml")

    # Copy instruction files
    if agent.instructions.file:
        instruction_file = agent.instructions.file
        src_path = agent_dir / instruction_file
        if src_path.exists():
            dst_path = build_dir / instruction_file
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    # Copy data directories
    if agent.tools:
        for tool in agent.tools:
            if isinstance(tool, VectorstoreTool):
                source_path = tool.source
                if source_path:
                    src = agent_dir / source_path
                    if src.exists():
                        dst = build_dir / source_path
                        if src.is_dir():
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)

    # Create entrypoint script
    entrypoint_content = """#!/bin/bash
set -e

# Start the HoloDeck agent server
exec holodeck serve /app/agent.yaml \\
    --host 0.0.0.0 \\
    --port "${HOLODECK_PORT:-8080}" \\
    --protocol "${HOLODECK_PROTOCOL:-rest}"
"""
    (build_dir / "entrypoint.sh").write_text(entrypoint_content)

    return build_dir


def _display_build_success(result: BuildResult, quiet: bool) -> None:
    """Display build success message.

    Args:
        result: Build result with image details
        quiet: If True, only show minimal output
    """
    if quiet:
        click.echo(result.full_name)
        return

    click.echo()
    click.secho("=" * 60, fg="green")
    click.secho("  Build Successful!", fg="green", bold=True)
    click.secho("=" * 60, fg="green")
    click.echo()
    click.echo(f"  Image:    {result.full_name}")
    click.echo(f"  ID:       {result.image_id[:19]}...")
    click.echo()
    click.secho("  Next steps:", bold=True)
    click.echo(f"    Run locally:  docker run -p 8080:8080 {result.full_name}")
    click.echo(f"    Push to registry:  docker push {result.full_name}")
    click.echo()
