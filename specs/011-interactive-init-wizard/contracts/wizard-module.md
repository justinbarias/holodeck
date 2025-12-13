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
    skip_agent_name: bool = False,
    skip_llm: bool = False,
    skip_vectorstore: bool = False,
    skip_evals: bool = False,
    skip_mcp: bool = False,
    agent_name_default: str | None = None,
    llm_default: str = "ollama",
    vectorstore_default: str = "chromadb",
    evals_defaults: list[str] | None = None,
    mcp_defaults: list[str] | None = None,
) -> WizardResult:
    """Run interactive configuration wizard.

    Prompts user for agent name, LLM provider, vector store,
    evaluation metrics, and MCP server selections.
    Skips prompts for values provided via CLI flags.

    Args:
        skip_agent_name: Skip agent name prompt (use agent_name_default)
        skip_llm: Skip LLM prompt (use llm_default)
        skip_vectorstore: Skip vectorstore prompt (use vectorstore_default)
        skip_evals: Skip evals prompt (use evals_defaults)
        skip_mcp: Skip MCP prompt (use mcp_defaults)
        agent_name_default: Default agent name value
        llm_default: Default LLM provider value
        vectorstore_default: Default vector store value
        evals_defaults: Default evaluation metrics list
        mcp_defaults: Default MCP server list

    Returns:
        WizardResult with all selections

    Raises:
        WizardCancelledError: If user cancels (Ctrl+C)
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

#### `_prompt_agent_name`

```python
def _prompt_agent_name(default: str | None = None) -> str:
    """Display agent name input prompt.

    Args:
        default: Pre-filled agent name (optional)

    Returns:
        Validated agent name

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:

```
? Enter agent name: my-agent
```

**Validation**: Alphanumeric, hyphens, and underscores only.

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
> Ollama (local) - Local LLM inference, gpt-oss:20b
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
> ChromaDB (default) - Embedded vector database, http://localhost:8000
  Redis - Production-grade vector store with Redis Stack
  In-Memory - Ephemeral storage for development/testing
```

**Note**: When "In-Memory" is selected, display warning:

```
Note: In-memory storage is ephemeral. Data will be lost on restart.
```

#### `_prompt_evals`

```python
def _prompt_evals(
    defaults: list[str] | None = None,
) -> list[str]:
    """Display evaluation metrics multi-selection prompt.

    Args:
        defaults: Metric identifiers to pre-select

    Returns:
        List of selected eval metric identifiers

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:

```
? Select evaluation metrics (space to toggle, enter to confirm):
  [X] RAG Faithfulness - Measures if response is grounded in context
  [X] RAG Answer Relevancy - Measures if response answers the question
  [ ] RAG Context Precision - Measures precision of retrieved context
  [ ] RAG Context Recall - Measures recall of retrieved context
```

#### `_prompt_mcp_servers`

```python
def _prompt_mcp_servers(
    defaults: list[str] | None = None,
) -> list[str]:
    """Display MCP server multi-selection prompt.

    Args:
        defaults: Server identifiers to pre-select

    Returns:
        List of selected server identifiers

    Raises:
        KeyboardInterrupt: If user cancels
    """
```

**Prompt Display**:

```
? Select MCP servers (space to toggle, enter to confirm):
  [X] Brave Search - Web search capabilities
  [X] Memory - Key-value memory storage
  [X] Sequential Thinking - Structured reasoning
  [ ] Filesystem - File system access
  [ ] GitHub - GitHub repository access
  [ ] PostgreSQL - PostgreSQL database access
```

### Data Classes

#### `WizardResult`

```python
from pydantic import BaseModel, Field

class WizardResult(BaseModel):
    """Final selections from wizard."""
    agent_name: str = Field(..., description="Agent name")
    llm_provider: str = Field(..., description="Selected LLM provider")
    vector_store: str = Field(..., description="Selected vector store")
    evals: list[str] = Field(..., description="Selected evaluation metrics")
    mcp_servers: list[str] = Field(..., description="Selected MCP server identifiers")
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
@click.option("--name", type=str, help="Agent name")
@click.option("--llm", type=click.Choice(["ollama", "openai", "azure_openai", "anthropic"]))
@click.option("--vectorstore", type=click.Choice(["chromadb", "redis", "in-memory"]))
@click.option("--evals", type=str, help="Comma-separated eval metrics")
@click.option("--mcp", type=str, help="Comma-separated MCP servers")
@click.option("--non-interactive", is_flag=True)
def init(
    name: str | None,
    llm: str | None,
    vectorstore: str | None,
    evals: str | None,
    mcp: str | None,
    non_interactive: bool,
    # ... other options
) -> None:
    try:
        # Determine if we should run wizard
        if non_interactive or not is_interactive():
            if not name:
                raise click.UsageError("--name is required in non-interactive mode")
            wizard_result = WizardResult(
                agent_name=name,
                llm_provider=llm or "ollama",
                vector_store=vectorstore or "chromadb",
                evals=_parse_comma_arg(evals) if evals else ["rag-faithfulness", "rag-answer_relevancy"],
                mcp_servers=_parse_comma_arg(mcp) if mcp else ["brave-search", "memory", "sequentialthinking"],
            )
        else:
            wizard_result = run_wizard(
                skip_agent_name=name is not None,
                skip_llm=llm is not None,
                skip_vectorstore=vectorstore is not None,
                skip_evals=evals is not None,
                skip_mcp=mcp is not None,
                agent_name_default=name,
                llm_default=llm or "ollama",
                vectorstore_default=vectorstore or "chromadb",
                evals_defaults=_parse_comma_arg(evals) if evals else None,
                mcp_defaults=_parse_comma_arg(mcp) if mcp else None,
            )

        # Proceed with project initialization
        init_input = ProjectInitInput(
            project_name=wizard_result.agent_name,
            agent_name=wizard_result.agent_name,
            llm_provider=wizard_result.llm_provider,
            vector_store=wizard_result.vector_store,
            evals=wizard_result.evals,
            mcp_servers=wizard_result.mcp_servers,
            # ... other fields
        )

        initializer = ProjectInitializer()
        result = initializer.initialize(init_input)
        # ... handle result

    except WizardCancelledError:
        click.echo("\nWizard cancelled.")
        raise click.Abort()
```

## InquirerPy Usage

```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import EmptyInputValidator
import re

def _prompt_agent_name(default: str | None = None) -> str:
    def validate_name(name: str) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))

    result = inquirer.text(
        message="Enter agent name:",
        default=default or "",
        validate=lambda x: validate_name(x) or "Use alphanumeric, hyphens, underscores only",
    ).execute()

    return result

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

def _prompt_evals(
    defaults: list[str] | None = None,
) -> list[str]:
    defaults = defaults or ["rag-faithfulness", "rag-answer_relevancy"]

    choices = [
        Choice(
            value=eval_choice.value,
            name=f"{eval_choice.display_name} - {eval_choice.description}",
            enabled=eval_choice.value in defaults or eval_choice.is_default,
        )
        for eval_choice in EVAL_CHOICES
    ]

    result = inquirer.checkbox(
        message="Select evaluation metrics (space to toggle, enter to confirm):",
        choices=choices,
        validate=lambda r: True,  # Empty selection allowed
    ).execute()

    return result

def _prompt_mcp_servers(
    defaults: list[str] | None = None,
) -> list[str]:
    defaults = defaults or ["brave-search", "memory", "sequentialthinking"]

    choices = [
        Choice(
            value=server.value,
            name=f"{server.display_name} - {server.description}",
            enabled=server.value in defaults or server.is_default,
        )
        for server in MCP_SERVER_CHOICES
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
│ holodeck init                                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Enter agent name: my-agent                                │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select LLM provider:                                      │
│   > Ollama (local) - Local LLM inference, gpt-oss:20b       │
│     OpenAI - GPT-4, GPT-3.5-turbo via OpenAI API           │
│     Azure OpenAI - OpenAI models via Azure deployment       │
│     Anthropic Claude - Claude 3.5, Claude 3                 │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select vector store:                                      │
│   > ChromaDB (default) - Embedded database, localhost:8000  │
│     Redis - Production-grade vector store                   │
│     In-Memory - Ephemeral storage                          │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select evaluation metrics (space to toggle):              │
│   [X] RAG Faithfulness - Measures groundedness              │
│   [X] RAG Answer Relevancy - Measures response relevance    │
│   [ ] RAG Context Precision - Measures context precision    │
│   [ ] RAG Context Recall - Measures context recall          │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ? Select MCP servers (space to toggle):                     │
│   [X] Brave Search - Web search capabilities                │
│   [X] Memory - Key-value memory storage                     │
│   [X] Sequential Thinking - Structured reasoning            │
│   [ ] Filesystem - File system access                       │
│   [ ] GitHub - GitHub repository access                     │
│   [ ] PostgreSQL - PostgreSQL database access               │
└─────────────────────────────────────────────────────────────┘
                           │ [Enter]
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ✓ Project initialized successfully!                        │
│                                                             │
│ Project: my-agent                                           │
│ LLM Provider: ollama (gpt-oss:20b)                         │
│ Vector Store: chromadb (http://localhost:8000)             │
│ Evals: rag-faithfulness, rag-answer_relevancy              │
│ MCP Servers: brave-search, memory, sequentialthinking     │
└─────────────────────────────────────────────────────────────┘
```

## Testing Requirements

1. **Unit Tests** (`tests/unit/test_wizard.py`):

   - Test each prompt function with mocked InquirerPy
   - Test `is_interactive()` under different conditions
   - Test `WizardResult` validation
   - Test agent name validation

2. **Integration Tests** (`tests/integration/test_init_wizard.py`):

   - Test full wizard flow with subprocess
   - Test non-interactive mode with flags
   - Test Ctrl+C cancellation handling

3. **Fixtures**:
   - Default values for testing
   - Mock InquirerPy prompts for automated testing
