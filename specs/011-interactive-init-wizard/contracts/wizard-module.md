# Module Contract: Interactive Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Overview

Wizard module orchestrates the interactive configuration flow for `holodeck init`.

## Module: `holodeck.cli.utils.wizard`

### Public Functions

#### `run_wizard`

```python
def run_wizard(
    skip_llm: bool = False,
    skip_vectorstore: bool = False,
    skip_mcp: bool = False,
    llm_default: str = "ollama",
    vectorstore_default: str = "chromadb",
    mcp_defaults: list[str] | None = None,
) -> WizardResult:
    """Run interactive configuration wizard.

    Prompts user for LLM provider, vector store, and MCP server selections.
    Skips prompts for values provided via CLI flags.

    Args:
        skip_llm: Skip LLM prompt (use llm_default)
        skip_vectorstore: Skip vectorstore prompt (use vectorstore_default)
        skip_mcp: Skip MCP prompt (use mcp_defaults)
        llm_default: Default LLM provider value
        vectorstore_default: Default vector store value
        mcp_defaults: Default MCP server list

    Returns:
        WizardResult with all selections

    Raises:
        WizardCancelledError: If user cancels (Ctrl+C)
        MCPRegistryError: If MCP registry is unreachable
    """
```

#### `is_interactive`

```python
def is_interactive() -> bool:
    """Check if terminal supports interactive prompts.

    Returns:
        True if stdin and stdout are TTY, False otherwise
    """
```

### Internal Functions

#### `_prompt_llm_provider`

```python
def _prompt_llm_provider(default: str = "ollama") -> str:
    """Display LLM provider selection prompt.

    Args:
        default: Pre-selected provider value

    Returns:
        Selected provider identifier

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:
```
? Select LLM provider: (Use arrow keys)
> Ollama (local) - Local LLM inference, no API key required
  OpenAI - GPT-4, GPT-3.5-turbo via OpenAI API
  Azure OpenAI - OpenAI models via Azure deployment
  Anthropic Claude - Claude 3.5, Claude 3 via Anthropic API
```

#### `_prompt_vectorstore`

```python
def _prompt_vectorstore(default: str = "chromadb") -> str:
    """Display vector store selection prompt.

    Args:
        default: Pre-selected store value

    Returns:
        Selected store identifier

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:
```
? Select vector store: (Use arrow keys)
> ChromaDB (default) - Embedded vector database with local persistence
  Redis - Production-grade vector store with Redis Stack
  In-Memory - Ephemeral storage for development/testing
```

**Note**: When "In-Memory" is selected, display warning:
```
Note: In-memory storage is ephemeral. Data will be lost on restart.
```

#### `_prompt_mcp_servers`

```python
def _prompt_mcp_servers(
    available_servers: list[MCPServerChoice],
    defaults: list[str] | None = None,
) -> list[str]:
    """Display MCP server multi-selection prompt.

    Args:
        available_servers: List of server choices from registry
        defaults: Package identifiers to pre-select

    Returns:
        List of selected server package identifiers

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:
```
? Select MCP servers (space to toggle, enter to confirm):
  [X] Filesystem - File system access
  [X] Memory - Key-value memory storage
  [X] Sequential Thinking - Structured reasoning
  [ ] GitHub - GitHub repository access
  [ ] Brave Search - Web search capabilities
  [ ] Postgres - PostgreSQL database access
  ...
```

#### `_fetch_mcp_servers`

```python
def _fetch_mcp_servers() -> list[MCPServerChoice]:
    """Fetch available MCP servers from registry.

    Returns:
        List of server choices for display

    Raises:
        MCPRegistryError: If registry is unreachable
    """
```

### Data Classes

#### `WizardResult`

```python
from pydantic import BaseModel, Field

class WizardResult(BaseModel):
    """Final selections from wizard."""
    llm_provider: str = Field(..., description="Selected LLM provider")
    vector_store: str = Field(..., description="Selected vector store")
    mcp_servers: list[str] = Field(..., description="Selected MCP server packages")
```

### Exceptions

```python
class WizardCancelledError(Exception):
    """Raised when user cancels wizard."""
    pass
```

## Integration with Init Command

```python
# In init.py command handler

from holodeck.cli.utils.wizard import run_wizard, is_interactive, WizardCancelledError

@click.command(name="init")
@click.option("--llm", type=click.Choice(["ollama", "openai", "azure_openai", "anthropic"]))
@click.option("--vectorstore", type=click.Choice(["chromadb", "redis", "in-memory"]))
@click.option("--mcp", type=str, help="Comma-separated MCP servers")
@click.option("--non-interactive", is_flag=True)
def init(
    project_name: str,
    llm: str | None,
    vectorstore: str | None,
    mcp: str | None,
    non_interactive: bool,
    # ... other options
) -> None:
    try:
        # Determine if we should run wizard
        if non_interactive or not is_interactive():
            wizard_result = WizardResult(
                llm_provider=llm or "ollama",
                vector_store=vectorstore or "chromadb",
                mcp_servers=_parse_mcp_arg(mcp),
            )
        else:
            wizard_result = run_wizard(
                skip_llm=llm is not None,
                skip_vectorstore=vectorstore is not None,
                skip_mcp=mcp is not None,
                llm_default=llm or "ollama",
                vectorstore_default=vectorstore or "chromadb",
                mcp_defaults=_parse_mcp_arg(mcp) if mcp else None,
            )

        # Proceed with project initialization
        init_input = ProjectInitInput(
            project_name=project_name,
            llm_provider=wizard_result.llm_provider,
            vector_store=wizard_result.vector_store,
            mcp_servers=wizard_result.mcp_servers,
            # ... other fields
        )

        initializer = ProjectInitializer()
        result = initializer.initialize(init_input)
        # ... handle result

    except WizardCancelledError:
        click.echo("\nWizard cancelled.")
        raise click.Abort()

    except MCPRegistryError as e:
        click.secho(f"Error: {e}", fg="red")
        raise click.Abort()
```

## InquirerPy Usage

```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

def _prompt_llm_provider(default: str = "ollama") -> str:
    choices = [
        Choice(
            value=choice.value,
            name=f"{choice.display_name} - {choice.description}",
        )
        for choice in LLM_PROVIDER_CHOICES
    ]

    result = inquirer.select(
        message="Select LLM provider:",
        choices=choices,
        default=default,
    ).execute()

    return result

def _prompt_mcp_servers(
    available_servers: list[MCPServerChoice],
    defaults: list[str] | None = None,
) -> list[str]:
    defaults = defaults or DEFAULT_MCP_SERVERS

    choices = [
        Choice(
            value=server.value,
            name=f"{server.display_name} - {server.description}",
            enabled=server.value in defaults or server.enabled,
        )
        for server in available_servers
    ]

    result = inquirer.checkbox(
        message="Select MCP servers (space to toggle, enter to confirm):",
        choices=choices,
        validate=lambda r: True,  # Empty selection allowed
    ).execute()

    return result
```

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ holodeck init my-agent                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select LLM provider:                                      │
│   > Ollama (local) - Local LLM inference, no API key        │
│     OpenAI - GPT-4, GPT-3.5-turbo via OpenAI API           │
│     Azure OpenAI - OpenAI models via Azure deployment       │
│     Anthropic Claude - Claude 3.5, Claude 3                 │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select vector store:                                      │
│   > ChromaDB (default) - Embedded vector database           │
│     Redis - Production-grade vector store                   │
│     In-Memory - Ephemeral storage                          │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select MCP servers (space to toggle):                     │
│   [X] Filesystem - File system access                       │
│   [X] Memory - Key-value memory storage                     │
│   [X] Sequential Thinking - Structured reasoning            │
│   [ ] GitHub - GitHub repository access                     │
│   [ ] Brave Search - Web search capabilities                │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ✓ Project initialized successfully!                        │
│                                                             │
│ Project: my-agent                                           │
│ LLM Provider: ollama                                        │
│ Vector Store: chromadb                                      │
│ MCP Servers: filesystem, memory, sequential-thinking        │
└─────────────────────────────────────────────────────────────┘
```

## Testing Requirements

1. **Unit Tests** (`tests/unit/test_wizard.py`):
   - Test each prompt function with mocked InquirerPy
   - Test `is_interactive()` under different conditions
   - Test `WizardResult` validation

2. **Integration Tests** (`tests/integration/test_init_wizard.py`):
   - Test full wizard flow with subprocess
   - Test non-interactive mode with flags
   - Test Ctrl+C cancellation handling
   - Test MCP registry error handling

3. **Fixtures**:
   - Mock MCP registry response data
   - Mock InquirerPy prompts for automated testing
