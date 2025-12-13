"""Interactive wizard utilities for holodeck init command.

This module provides the interactive prompts and wizard flow
using InquirerPy for the holodeck init command.
"""

import sys

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from holodeck.lib.validation import validate_agent_name as _validate_agent_name
from holodeck.models.wizard_config import (
    EVAL_CHOICES,
    LLM_PROVIDER_CHOICES,
    MCP_SERVER_CHOICES,
    VECTOR_STORE_CHOICES,
    LLMProviderChoice,
    ProviderConfig,
    WizardResult,
    get_default_evals,
    get_default_mcp_servers,
    get_template_choices,
)


class WizardCancelledError(Exception):
    """Raised when user cancels the wizard (Ctrl+C).

    This exception is raised when the user presses Ctrl+C during
    any interactive prompt in the wizard flow. The caller should
    handle this exception to clean up any partial state.
    """

    pass


def is_interactive() -> bool:
    """Check if terminal supports interactive prompts.

    Checks whether both stdin and stdout are connected to a TTY
    (terminal). This is used to determine if the wizard can run
    interactively or should fall back to non-interactive mode.

    Returns:
        True if stdin and stdout are both TTYs, False otherwise.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_agent_name(default: str | None = None) -> str:
    """Display agent name input prompt.

    Prompts the user to enter an agent name with validation.
    The agent name must start with a letter and contain only
    alphanumeric characters, hyphens, and underscores.

    Args:
        default: Pre-filled agent name value (optional).

    Returns:
        The validated agent name entered by the user.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    from InquirerPy.validator import ValidationError as InquirerValidationError

    def validate_name(name: str) -> bool:
        """Validate agent name format using shared validator.

        Args:
            name: The name to validate.

        Returns:
            True if valid.

        Raises:
            InquirerValidationError: If validation fails.
        """
        try:
            _validate_agent_name(name)
            return True
        except ValueError as e:
            raise InquirerValidationError(message=str(e)) from e

    result: str = inquirer.text(
        message="Enter agent name:",
        default=default or "",
        validate=validate_name,
    ).execute()

    return result


def _prompt_template(default: str = "conversational") -> str:
    """Display template selection prompt.

    Shows a list of available project templates with descriptions.
    The user can select one template using arrow keys and Enter.

    Args:
        default: Pre-selected template value (default: "conversational").

    Returns:
        The value of the selected template.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    template_choices = get_template_choices()
    choices = [
        Choice(
            value=choice.value,
            name=f"{choice.display_name} - {choice.description}",
        )
        for choice in template_choices
    ]

    result: str = inquirer.select(
        message="Select agent template:",
        choices=choices,
        default=default,
    ).execute()

    return result


def _prompt_llm_provider(default: str = "ollama") -> str:
    """Display LLM provider selection prompt.

    Shows a list of available LLM providers with descriptions.
    The user can select one provider using arrow keys and Enter.

    Args:
        default: Pre-selected provider value (default: "ollama").

    Returns:
        The value of the selected LLM provider.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    choices = [
        Choice(
            value=choice.value,
            name=f"{choice.display_name} - {choice.description}",
        )
        for choice in LLM_PROVIDER_CHOICES
    ]

    result: str = inquirer.select(
        message="Select LLM provider:",
        choices=choices,
        default=default,
    ).execute()

    return result


def _get_provider_choice(provider: str) -> LLMProviderChoice | None:
    """Get the LLMProviderChoice for a given provider value.

    Args:
        provider: The provider identifier (e.g., 'azure_openai').

    Returns:
        The matching LLMProviderChoice or None if not found.
    """
    for choice in LLM_PROVIDER_CHOICES:
        if choice.value == provider:
            return choice
    return None


def _prompt_endpoint(provider_choice: LLMProviderChoice) -> str:
    """Display endpoint URL input prompt for providers that require it.

    Prompts the user to enter an API endpoint URL. If the user presses
    Enter without input, returns an environment variable placeholder.

    Args:
        provider_choice: The LLM provider choice with endpoint_env_var.

    Returns:
        The endpoint URL or environment variable placeholder.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    env_var = provider_choice.endpoint_env_var or "ENDPOINT_URL"
    default_placeholder = f"${{{env_var}}}"

    result: str = inquirer.text(
        message=f"Enter {provider_choice.display_name} endpoint URL "
        f"(press Enter for {default_placeholder}):",
        default="",
    ).execute()

    # If empty, return placeholder
    if not result.strip():
        return default_placeholder

    return result.strip()


def _prompt_provider_config(provider: str) -> ProviderConfig | None:
    """Prompt for provider-specific configuration based on selected provider.

    For providers that require additional configuration (like Azure OpenAI),
    this function prompts for endpoint URL. For other providers, returns None.

    Args:
        provider: The selected LLM provider identifier.

    Returns:
        ProviderConfig with collected settings, or None for providers
        that don't require additional configuration.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    provider_choice = _get_provider_choice(provider)
    if not provider_choice:
        return None

    # Only prompt for providers that require endpoint
    if not provider_choice.requires_endpoint:
        return None

    # Prompt for endpoint
    endpoint = _prompt_endpoint(provider_choice)

    return ProviderConfig(
        endpoint=endpoint,
    )


def _prompt_vectorstore(default: str = "chromadb") -> str:
    """Display vector store selection prompt.

    Shows a list of available vector stores with descriptions.
    The user can select one store using arrow keys and Enter.
    If "in-memory" is selected, displays a warning about data loss.

    Args:
        default: Pre-selected store value (default: "chromadb").

    Returns:
        The value of the selected vector store.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    choices = [
        Choice(
            value=choice.value,
            name=f"{choice.display_name} - {choice.description}",
        )
        for choice in VECTOR_STORE_CHOICES
    ]

    result: str = inquirer.select(
        message="Select vector store:",
        choices=choices,
        default=default,
    ).execute()

    # Display warning for in-memory selection
    if result == "in-memory":
        import click

        click.secho(
            "Note: In-memory storage is ephemeral. Data will be lost on restart.",
            fg="yellow",
        )

    return result


def _prompt_evals(defaults: list[str] | None = None) -> list[str]:
    """Display evaluation metrics multi-selection prompt.

    Shows a checkbox list of available evaluation metrics.
    The user can toggle selections with space and confirm with Enter.
    Default metrics (faithfulness and answer_relevancy) are pre-selected.

    Args:
        defaults: Metric identifiers to pre-select. If None, uses
            the default metrics (rag-faithfulness, rag-answer_relevancy).

    Returns:
        List of selected evaluation metric identifiers.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    if defaults is None:
        defaults = get_default_evals()

    choices = [
        Choice(
            value=eval_choice.value,
            name=f"{eval_choice.display_name} - {eval_choice.description}",
            enabled=eval_choice.value in defaults,
        )
        for eval_choice in EVAL_CHOICES
    ]

    result: list[str] = inquirer.checkbox(
        message="Select evaluation metrics (space to toggle, enter to confirm):",
        choices=choices,
    ).execute()

    return result


def _prompt_mcp_servers(defaults: list[str] | None = None) -> list[str]:
    """Display MCP server multi-selection prompt.

    Shows a checkbox list of available MCP servers with descriptions.
    The user can toggle selections with space and confirm with Enter.
    Default servers (brave-search, memory, sequentialthinking) are pre-selected.

    Args:
        defaults: Server identifiers to pre-select. If None, uses
            the default servers.

    Returns:
        List of selected MCP server identifiers.

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C.
    """
    if defaults is None:
        defaults = get_default_mcp_servers()

    choices = [
        Choice(
            value=server.value,
            name=f"{server.display_name} - {server.description}",
            enabled=server.value in defaults,
        )
        for server in MCP_SERVER_CHOICES
    ]

    result: list[str] = inquirer.checkbox(
        message="Select MCP servers (space to toggle, enter to confirm):",
        choices=choices,
    ).execute()

    return result


def run_wizard(
    skip_agent_name: bool = False,
    skip_template: bool = False,
    skip_llm: bool = False,
    skip_provider_config: bool = False,
    skip_vectorstore: bool = False,
    skip_evals: bool = False,
    skip_mcp: bool = False,
    agent_name_default: str | None = None,
    template_default: str = "conversational",
    llm_default: str = "ollama",
    provider_config_default: ProviderConfig | None = None,
    vectorstore_default: str = "chromadb",
    evals_defaults: list[str] | None = None,
    mcp_defaults: list[str] | None = None,
) -> WizardResult:
    """Run interactive configuration wizard.

    Prompts user for agent name, template, LLM provider, provider-specific config,
    vector store, evaluation metrics, and MCP server selections. Skips
    prompts for values provided via CLI flags (when skip_* is True).

    Args:
        skip_agent_name: Skip agent name prompt (use agent_name_default).
        skip_template: Skip template prompt (use template_default).
        skip_llm: Skip LLM prompt (use llm_default).
        skip_provider_config: Skip provider config prompts
            (use provider_config_default).
        skip_vectorstore: Skip vectorstore prompt (use vectorstore_default).
        skip_evals: Skip evals prompt (use evals_defaults).
        skip_mcp: Skip MCP prompt (use mcp_defaults).
        agent_name_default: Default agent name value.
        template_default: Default template value (default: "conversational").
        llm_default: Default LLM provider value (default: "ollama").
        provider_config_default: Default provider config (endpoint, deployment name).
        vectorstore_default: Default vector store value (default: "chromadb").
        evals_defaults: Default evaluation metrics list.
        mcp_defaults: Default MCP server list.

    Returns:
        WizardResult with all validated selections.

    Raises:
        WizardCancelledError: If user cancels with Ctrl+C at any prompt.
    """
    try:
        # Step 1: Agent name
        if skip_agent_name and agent_name_default:
            agent_name = agent_name_default
        else:
            agent_name = _prompt_agent_name(default=agent_name_default)

        # Step 2: Template selection
        if skip_template:
            template = template_default
        else:
            template = _prompt_template(default=template_default)

        # Step 3: LLM provider
        if skip_llm:
            llm_provider = llm_default
        else:
            llm_provider = _prompt_llm_provider(default=llm_default)

        # Step 3b: Provider-specific configuration (e.g., Azure endpoint)
        if skip_provider_config:
            provider_config = provider_config_default
        else:
            provider_config = _prompt_provider_config(llm_provider)

        # Step 4: Vector store
        if skip_vectorstore:
            vector_store = vectorstore_default
        else:
            vector_store = _prompt_vectorstore(default=vectorstore_default)

        # Step 5: Evaluation metrics
        if skip_evals:
            evals = (
                evals_defaults if evals_defaults is not None else get_default_evals()
            )
        else:
            evals = _prompt_evals(defaults=evals_defaults)

        # Step 6: MCP servers
        if skip_mcp:
            mcp_servers = (
                mcp_defaults if mcp_defaults is not None else get_default_mcp_servers()
            )
        else:
            mcp_servers = _prompt_mcp_servers(defaults=mcp_defaults)

        # Create and validate result
        return WizardResult(
            agent_name=agent_name,
            template=template,
            llm_provider=llm_provider,
            provider_config=provider_config,
            vector_store=vector_store,
            evals=evals,
            mcp_servers=mcp_servers,
        )

    except KeyboardInterrupt as e:
        raise WizardCancelledError("Wizard cancelled by user") from e
