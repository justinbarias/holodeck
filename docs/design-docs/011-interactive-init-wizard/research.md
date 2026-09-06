# Research: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Research Tasks

### 1. Predefined Configuration Options

**Context**: The wizard presents users with predefined options for LLM providers, vector stores, evals, and MCP servers without requiring external API calls.

**Decision**: Use hardcoded lists of options with sensible defaults

**Rationale**:

- No network dependency - wizard works offline
- Faster user experience - no API latency
- Simpler implementation - no error handling for network failures
- Consistent experience - options don't change unexpectedly

**Default Configurations**:

| Category     | Default                                              | Other Options                             |
| ------------ | ---------------------------------------------------- | ----------------------------------------- |
| LLM Provider | Ollama (gpt-oss:20b)                                 | OpenAI, Azure OpenAI, Anthropic           |
| Vector Store | ChromaDB (http://localhost:8000)                     | Redis, In-Memory                          |
| Evals        | rag-faithfulness, rag-answer_relevancy               | rag-context_precision, rag-context_recall |
| MCP Servers  | brave-search[web-search], memory, sequentialthinking | filesystem, github, postgres              |

---

### 2. Interactive CLI Library Selection

**Context**: Click's native `prompt` with `multiple=True` doesn't work well. Need library for multi-select prompts.

**Decision**: Use **InquirerPy** for interactive prompts

**Rationale**:

1. Built on prompt_toolkit (cross-platform, including Windows)
2. Native `checkbox` prompt for multi-select with pre-selection support
3. Native `select` prompt for single-selection
4. Native `text` prompt for agent name input with validation
5. Active maintenance and good documentation
6. No Click conflicts - works alongside Click decorators

**Installation**: `inquirerpy` (already follows project dependency patterns)

**Key Features Required**:

| Feature       | InquirerPy Support               | Usage                      |
| ------------- | -------------------------------- | -------------------------- |
| Text input    | `inquirer.text()`                | Agent name                 |
| Single-select | `inquirer.select()`              | LLM provider, Vector store |
| Multi-select  | `inquirer.checkbox()`            | Evals, MCP servers         |
| Pre-selection | `Choice(value, enabled=True)`    | Default evals, MCP servers |
| Descriptions  | `Choice(value, name="Display")`  | Show option descriptions   |
| Validation    | `validate=lambda r: len(r) >= 1` | Ensure selections made     |

**Example Integration**:

```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import EmptyInputValidator

# Text input for agent name
agent_name = inquirer.text(
    message="Enter agent name:",
    validate=EmptyInputValidator("Agent name cannot be empty"),
).execute()

# Single-select for LLM provider
provider = inquirer.select(
    message="Select LLM provider:",
    choices=[
        Choice("ollama", name="Ollama (default, gpt-oss:20b) - Local LLM inference"),
        Choice("openai", name="OpenAI - GPT-4, GPT-3.5-turbo"),
        Choice("azure_openai", name="Azure OpenAI - Azure-hosted models"),
        Choice("anthropic", name="Anthropic Claude - Claude 3.5, Claude 3"),
    ],
    default="ollama",
).execute()

# Multi-select for evals
evals = inquirer.checkbox(
    message="Select evaluation metrics (space to toggle, enter to confirm):",
    choices=[
        Choice("rag-faithfulness", name="RAG Faithfulness", enabled=True),
        Choice("rag-answer_relevancy", name="RAG Answer Relevancy", enabled=True),
        Choice("rag-context_precision", name="RAG Context Precision"),
        Choice("rag-context_recall", name="RAG Context Recall"),
    ],
).execute()

# Multi-select for MCP servers
servers = inquirer.checkbox(
    message="Select MCP servers (space to toggle, enter to confirm):",
    choices=[
        Choice("brave-search", name="Brave Search - Web search capabilities", enabled=True),
        Choice("memory", name="Memory - Key-value storage", enabled=True),
        Choice("sequentialthinking", name="Sequential Thinking - Structured reasoning", enabled=True),
        Choice("filesystem", name="Filesystem - File system access"),
        Choice("github", name="GitHub - Repository access"),
    ],
).execute()
```

**Alternatives Considered**:

- **click-prompt**: Less active, not as well documented
- **questionary**: Good alternative, but InquirerPy has more customization
- **python-inquirer**: Windows support is experimental
- **Raw Click prompts**: Doesn't support multi-select checkbox UI

**Sources**:

- [InquirerPy PyPI](https://pypi.org/project/inquirerpy/)
- [InquirerPy Checkbox Docs](https://inquirerpy.readthedocs.io/en/latest/pages/prompts/checkbox.html)
- [InquirerPy Select Docs](https://inquirerpy.readthedocs.io/en/latest/pages/prompts/list.html)
- [InquirerPy Text Docs](https://inquirerpy.readthedocs.io/en/latest/pages/prompts/input.html)
- [Click Prompts Documentation](https://click.palletsprojects.com/en/stable/prompts/)

---

### 3. Terminal Interactivity Detection

**Context**: Need to detect when terminal doesn't support interactive prompts (per FR-015).

**Decision**: Use `sys.stdin.isatty()` for detection, fall back to defaults

**Rationale**:

- Standard Python approach, no additional dependencies
- Click also uses this pattern internally
- Works across platforms

**Implementation Pattern**:

```python
import sys

def is_interactive() -> bool:
    """Check if terminal supports interactive prompts."""
    return sys.stdin.isatty() and sys.stdout.isatty()
```

**Fallback Behavior**:

- When non-interactive: Use all defaults without prompting
- Log info message explaining defaults were used
- Same behavior as `--non-interactive` flag

**Alternatives Considered**:

- Environment variable check only: Insufficient, doesn't detect piped input
- prompt_toolkit's detection: Overkill for this use case

---

### 4. Non-Interactive Mode CLI Design

**Context**: Need CLI flags for scripted/CI usage (FR-009).

**Decision**: Add flags `--name`, `--llm`, `--vectorstore`, `--evals`, `--mcp`, and `--non-interactive`

**Rationale**:

- Follows existing CLI patterns in holodeck
- Clear, explicit flag names matching wizard prompts
- `--non-interactive` provides explicit opt-out from prompts

**Flag Specification**:

| Flag                | Type               | Default                                | Description                              |
| ------------------- | ------------------ | -------------------------------------- | ---------------------------------------- |
| `--name`            | String             | None                                   | Agent name (required in non-interactive) |
| `--llm`             | Choice             | ollama                                 | LLM provider selection                   |
| `--vectorstore`     | Choice             | chromadb                               | Vector store selection                   |
| `--evals`           | String (comma-sep) | rag-faithfulness,rag-answer_relevancy  | Evaluation metrics                       |
| `--mcp`             | String (comma-sep) | brave-search,memory,sequentialthinking | MCP servers                              |
| `--non-interactive` | Flag               | False                                  | Skip all prompts, use defaults/flags     |

**Validation**:

- Invalid `--llm` or `--vectorstore` values: Error with valid options listed
- Invalid `--evals` or `--mcp` values: Warning + skip invalid, continue with valid
- Missing `--name` in non-interactive: Error requiring name

**Alternatives Considered**:

- JSON config file input: Over-engineering for this use case
- Environment variables only: Less discoverable than CLI flags

---

### 5. Clean Cancellation (Ctrl+C) Handling

**Context**: Per FR-012, no partial files should remain if user cancels mid-wizard.

**Decision**: Wrap wizard in try/except with cleanup, use existing `ProjectInitializer` cleanup pattern

**Rationale**:

- Existing `ProjectInitializer` already handles cleanup on failure
- InquirerPy raises `KeyboardInterrupt` on Ctrl+C
- Defer file creation until all prompts complete

**Implementation Pattern**:

```python
try:
    # Collect all wizard inputs first (no file I/O)
    wizard_result = run_wizard()

    # Only create files after all inputs collected
    initializer = ProjectInitializer()
    initializer.initialize(wizard_result)

except KeyboardInterrupt:
    click.echo("\nWizard cancelled.")
    # No cleanup needed - files weren't created yet
    raise click.Abort()
```

**Key Insight**: Separate prompt collection phase from file creation phase.

**Alternatives Considered**:

- Transactional file system: Overkill, complex
- Temp directory + atomic move: Unnecessary complexity

---

### 6. Evaluation Metrics Configuration

**Context**: Users need to select which evaluation metrics to enable for their agent.

**Decision**: Provide predefined list of RAG-focused evaluation metrics

**Available Metrics**:

| Metric                  | Description                                           | Default |
| ----------------------- | ----------------------------------------------------- | ------- |
| `rag-faithfulness`      | Measures if response is grounded in retrieved context | Yes     |
| `rag-answer_relevancy`  | Measures if response answers the question             | Yes     |
| `rag-context_precision` | Measures precision of retrieved context               | No      |
| `rag-context_recall`    | Measures recall of retrieved context                  | No      |

**Rationale**:

- RAG metrics are most relevant for agent evaluation
- Faithfulness and relevancy are the core quality indicators
- Additional metrics available for advanced users

---

### 7. Existing Codebase Integration Points

**Context**: Understanding where wizard logic should integrate.

**Analysis**:

**Current `holodeck init` Flow**:

1. `init.py` parses CLI args → `ProjectInitInput`
2. `ProjectInitializer.initialize()` validates and creates files
3. Uses `TemplateRenderer` for Jinja2 template processing

**Integration Points**:

1. **`src/holodeck/cli/commands/init.py`**: Add wizard invocation before `ProjectInitializer`
2. **`src/holodeck/cli/utils/wizard.py`** (new): Wizard logic and prompt definitions
3. **`src/holodeck/models/wizard_config.py`** (new): Wizard state and result models

**Existing Models to Extend**:

- `LLMProvider` / `ProviderEnum` in `models/llm.py` - already has all 4 providers
- `DatabaseConfig` in `models/tool.py` - has chromadb, redis, in-memory
- `EvaluationConfig` in `models/evaluation.py` - has metrics configuration
- `MCPTool` in `models/tool.py` - MCP server configuration

**Template Updates Required**:

- Update `agent.yaml.j2` templates to accept wizard selections
- Add placeholder variables for agent name, LLM config, vectorstore, evals, MCP tools

---

## Summary of Decisions

| Area                  | Decision                                                                    | Key Dependency             |
| --------------------- | --------------------------------------------------------------------------- | -------------------------- |
| Configuration Options | Predefined hardcoded lists                                                  | None                       |
| Interactive Prompts   | InquirerPy                                                                  | New dependency             |
| TTY Detection         | `sys.stdin.isatty()`                                                        | Standard library           |
| CLI Flags             | `--name`, `--llm`, `--vectorstore`, `--evals`, `--mcp`, `--non-interactive` | Click (existing)           |
| Cancellation          | Prompt-first, then file creation                                            | Existing cleanup patterns  |
| Evals                 | Predefined RAG metrics list                                                 | Existing evaluation models |

## New Dependencies

```toml
# Add to pyproject.toml dependencies
"inquirerpy>=0.3.4,<0.4.0"  # Interactive CLI prompts
```
