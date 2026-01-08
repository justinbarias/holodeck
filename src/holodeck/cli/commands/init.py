"""Click command for initializing new HoloDeck projects.

This module implements the 'holodeck init' command which creates a new
project directory with templates, configuration, and example files.
"""

from pathlib import Path

import click

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.cli.utils.project_init import ProjectInitializer
from holodeck.cli.utils.wizard import (
    WizardCancelledError,
    is_interactive,
    run_wizard,
)
from holodeck.lib.logging_config import get_logger, setup_logging
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.project_config import ProjectInitInput
from holodeck.models.wizard_config import (
    VALID_EVALS,
    VALID_LLM_PROVIDERS,
    VALID_MCP_SERVERS,
    VALID_VECTOR_STORES,
    ProviderConfig,
    WizardResult,
    get_default_evals,
    get_default_mcp_servers,
)

logger = get_logger(__name__)


def validate_template(
    ctx: click.Context,  # noqa: ARG001
    param: click.Parameter,  # noqa: ARG001
    value: str,
) -> str:
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


def _parse_comma_arg(value: str | None) -> list[str]:
    """Parse a comma-separated argument into a list.

    Args:
        value: Comma-separated string or None.

    Returns:
        List of stripped, non-empty values.
    """
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@click.command(name="init")
@click.option(
    "--name",
    "project_name",
    default=None,
    help="Agent/project name (required in non-interactive mode)",
)
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
@click.option(
    "--llm",
    type=click.Choice(sorted(VALID_LLM_PROVIDERS)),
    default=None,
    help="LLM provider (skips interactive prompt)",
)
@click.option(
    "--vectorstore",
    type=click.Choice(sorted(VALID_VECTOR_STORES)),
    default=None,
    help="Vector store (skips interactive prompt)",
)
@click.option(
    "--evals",
    "evals_arg",
    default=None,
    help="Comma-separated evaluation metrics (skips interactive prompt)",
)
@click.option(
    "--mcp",
    "mcp_arg",
    default=None,
    help="Comma-separated MCP servers (skips interactive prompt)",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Skip all interactive prompts (use defaults or flag values)",
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
    help="Suppress INFO logging output",
)
def init(
    project_name: str | None,
    template: str,
    description: str | None,
    author: str | None,
    force: bool,
    llm: str | None,
    vectorstore: str | None,
    evals_arg: str | None,
    mcp_arg: str | None,
    non_interactive: bool,
    verbose: bool,
    quiet: bool,
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

    INTERACTIVE MODE (default):

        When run without --non-interactive, the wizard prompts for:
        - Agent name
        - LLM provider (Ollama, OpenAI, Azure OpenAI, Anthropic)
        - Vector store (ChromaDB, Qdrant, In-Memory)
        - Evaluation metrics
        - MCP servers

    NON-INTERACTIVE MODE:

        Use --non-interactive with --name to skip prompts and use defaults:

            holodeck init --name my-agent --non-interactive

        Or override specific values:

            holodeck init --name my-agent --llm openai --vectorstore qdrant

    EXAMPLES:

        Basic project with interactive wizard:

            holodeck init

        Quick setup with defaults (no prompts):

            holodeck init --name my-agent --non-interactive

        Custom LLM and vector store:

            holodeck init --name my-agent --llm openai --vectorstore qdrant

        Full customization without prompts:

            holodeck init --name my-agent --llm anthropic \\
                --vectorstore chromadb --evals rag-faithfulness,rag-answer_relevancy \\
                --mcp brave-search,memory --non-interactive

    For more information, see: https://useholodeck.ai/docs/getting-started
    """
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)
    logger.debug(
        f"Init command invoked: project_name={project_name}, template={template}, "
        f"non_interactive={non_interactive}"
    )

    try:
        # Get current working directory as output directory
        output_dir = Path.cwd()

        # Parse comma-separated arguments
        evals_list = _parse_comma_arg(evals_arg)
        mcp_list = _parse_comma_arg(mcp_arg)

        # Validate evals if provided
        if evals_list:
            invalid_evals = [e for e in evals_list if e not in VALID_EVALS]
            if invalid_evals:
                valid = ", ".join(sorted(VALID_EVALS))
                invalid_str = ", ".join(invalid_evals)
                click.secho(
                    f"Warning: Invalid eval(s): {invalid_str}. Valid: {valid}",
                    fg="yellow",
                )
                evals_list = [e for e in evals_list if e in VALID_EVALS]

        # Validate MCP servers if provided
        if mcp_list:
            invalid_mcp = [s for s in mcp_list if s not in VALID_MCP_SERVERS]
            if invalid_mcp:
                valid = ", ".join(sorted(VALID_MCP_SERVERS))
                click.secho(
                    f"Warning: Invalid MCP server(s): {', '.join(invalid_mcp)}. "
                    f"Valid options: {valid}",
                    fg="yellow",
                )
                mcp_list = [s for s in mcp_list if s in VALID_MCP_SERVERS]

        # Determine if we should run wizard
        if non_interactive or not is_interactive():
            # Non-interactive mode: --name is required
            if not project_name:
                click.secho(
                    "Error: --name is required in non-interactive mode",
                    fg="red",
                )
                raise click.Abort()

            # Use defaults or flag values
            selected_llm = llm or "ollama"

            # Create provider config for providers that require endpoint
            provider_config = None
            if selected_llm == "azure_openai":
                # Use env var placeholders for Azure OpenAI
                provider_config = ProviderConfig(
                    endpoint="${AZURE_OPENAI_ENDPOINT}",
                )

            wizard_result = WizardResult(
                agent_name=project_name,
                template=template,
                llm_provider=selected_llm,
                provider_config=provider_config,
                vector_store=vectorstore or "chromadb",
                evals=evals_list if evals_list else get_default_evals(),
                mcp_servers=mcp_list if mcp_list else get_default_mcp_servers(),
            )
        else:
            # Interactive mode: run wizard
            # Skip template prompt if --template was provided (not default)
            wizard_result = run_wizard(
                skip_agent_name=project_name is not None,
                skip_template=template != "conversational",
                skip_llm=llm is not None,
                skip_vectorstore=vectorstore is not None,
                skip_evals=evals_arg is not None,
                skip_mcp=mcp_arg is not None,
                agent_name_default=project_name,
                template_default=template,
                llm_default=llm or "ollama",
                vectorstore_default=vectorstore or "chromadb",
                evals_defaults=evals_list if evals_list else None,
                mcp_defaults=mcp_list if mcp_list else None,
            )

        # Use agent_name from wizard result as project name
        final_project_name = wizard_result.agent_name

        # Check if project directory already exists (unless force)
        project_dir = output_dir / final_project_name
        if project_dir.exists() and not force:
            # Prompt user for confirmation
            if click.confirm(
                f"Project directory '{final_project_name}' already exists. "
                "Do you want to overwrite it?",
                default=False,
            ):
                force = True
            else:
                click.echo("Initialization cancelled.")
                return

        # Create project initialization input
        init_input = ProjectInitInput(
            project_name=final_project_name,
            template=wizard_result.template,
            description=description,
            author=author,
            output_dir=str(output_dir),
            overwrite=force,
            agent_name=wizard_result.agent_name,
            llm_provider=wizard_result.llm_provider,
            provider_config=wizard_result.provider_config,
            vector_store=wizard_result.vector_store,
            evals=wizard_result.evals,
            mcp_servers=wizard_result.mcp_servers,
        )

        # Initialize project
        initializer = ProjectInitializer()
        result = initializer.initialize(init_input)

        # Handle result
        if result.success:
            # Display success message
            click.echo()  # Blank line for readability
            click.secho("Project initialized successfully!", fg="green", bold=True)
            click.echo()
            click.echo(f"Project: {result.project_name}")
            click.echo(f"Location: {result.project_path}")
            click.echo(f"Template: {result.template_used}")
            click.echo()
            click.echo("Configuration:")
            click.echo(f"  Agent Name: {wizard_result.agent_name}")
            click.echo(f"  Template: {wizard_result.template}")
            click.echo(f"  LLM Provider: {wizard_result.llm_provider}")
            click.echo(f"  Vector Store: {wizard_result.vector_store}")
            click.echo(f"  Evals: {', '.join(wizard_result.evals) or 'none'}")
            click.echo(
                f"  MCP Servers: {', '.join(wizard_result.mcp_servers) or 'none'}"
            )
            click.echo()
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
                    click.echo(f"  - {file_path}")
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
            click.secho("Project initialization failed", fg="red", bold=True)
            click.echo()
            for error in result.errors:
                click.secho(f"Error: {error}", fg="red")
            click.echo()
            raise click.Abort()

    except WizardCancelledError as e:
        # Handle wizard cancellation gracefully
        click.echo()
        click.secho("Wizard cancelled.", fg="yellow")
        raise click.Abort() from e

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
