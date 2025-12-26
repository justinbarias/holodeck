# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HoloDeck** is an open-source experimentation platform for building, testing, and deploying AI agents through pure YAML configuration. The project enables teams to go from hypothesis to production API in minutes without writing code.

**Core Value Proposition:** No-code agent definition. Users define agents, tools, evaluations, and deployments entirely through YAML files.

**Current Status:** Early development (v0.1 in progress)

- CLI and configuration infrastructure: **Complete**
- Agent execution engine: **Complete**
- Evaluation framework: **Complete**
- Chat interface: **Complete**
- Deployment engine: **Planned**

**Technology Stack:**

- **Language:** Python 3.10+
- **Package Manager:** UV (fast, modern replacement for pip/Poetry)
- **Framework:** Microsoft Semantic Kernel (agent framework)
- **CLI:** Click
- **Configuration:** Pydantic v2 + YAML
- **Testing:** Pytest with async support
- **Evaluation:** Azure AI Evaluation + DeepEval + NLP metrics

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer (holodeck)                      │
│  ├─ init: Project scaffolding with templates                 │
│  ├─ test: Test runner with multimodal support                │
│  ├─ chat: Interactive chat session                           │
│  └─ config: Configuration wizard                             │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Configuration Management                        │
│  ├─ ConfigLoader: YAML parsing with env substitution         │
│  ├─ ConfigValidator: Schema validation via Pydantic          │
│  ├─ ConfigMerge: Merge defaults + user config                │
│  └─ EnvLoader: .env file loading (project + user)            │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Pydantic Models (Schema)                     │
│  ├─ AgentConfig: Agent configuration schema                  │
│  ├─ LLMProvider: LLM provider settings (OpenAI, Azure, etc.) │
│  ├─ ToolUnion: Tool definitions (5 types)                    │
│  ├─ EvaluationConfig: Metrics and thresholds                 │
│  └─ TestCaseModel: Test cases with multimodal file support   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Engine (Semantic Kernel)             │
│  ├─ LLM Execution: Multi-provider support                    │
│  ├─ Tool Execution: MCP, function, vectorstore, prompt       │
│  ├─ Memory Management: Conversation history                  │
│  └─ Streaming: Real-time response streaming                  │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Framework                            │
│  ├─ NLP Metrics: F1, BLEU, ROUGE, METEOR                     │
│  ├─ Azure AI Metrics: Groundedness, Relevance, Coherence     │
│  ├─ DeepEval Metrics: G-Eval, Faithfulness, Answer Relevancy │
│  └─ Test Runner: Orchestrates test execution & evaluation    │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Patterns

1. **Configuration-Driven Design**: All agent behavior defined via YAML with Pydantic validation
2. **Plugin Architecture**: Tool system supports 5 types (vectorstore, function, MCP, prompt, plugin)
3. **MCP for APIs**: External API integrations must use MCP servers, not custom API tool types
4. **Multimodal Testing**: File processor handles images (OCR), PDFs, Office documents
5. **Evaluation Flexibility**: Three-level model configuration (global, per-evaluation, per-metric)
6. **Streaming Architecture**: Real-time streaming with async/await throughout

## Project Structure

```
holodeck/
├── src/holodeck/
│   ├── __init__.py                 # Package entry point with version
│   ├── cli/                        # Command-line interface
│   │   ├── main.py                 # CLI entry point (holodeck command)
│   │   ├── commands/               # CLI commands (init, test, chat, config)
│   │   ├── utils/                  # CLI utilities (project_init, wizard)
│   │   └── exceptions.py           # CLI-specific exceptions
│   │
│   ├── config/                     # Configuration management
│   │   ├── loader.py               # YAML configuration loader
│   │   ├── validator.py            # Configuration validation logic
│   │   ├── merge.py                # Configuration merging (defaults + user)
│   │   ├── env_loader.py           # Environment variable loading
│   │   ├── defaults.py             # Default configuration values
│   │   ├── context.py              # Configuration context management
│   │   ├── manager.py              # Central configuration manager
│   │   └── schema.py               # JSON schema definitions
│   │
│   ├── models/                     # Pydantic data models
│   │   ├── config.py               # Base configuration models
│   │   ├── agent.py                # Agent configuration model
│   │   ├── llm.py                  # LLM provider models
│   │   ├── tool.py                 # Tool configuration models
│   │   ├── evaluation.py           # Evaluation metrics models
│   │   ├── test_case.py            # Test case models
│   │   ├── test_result.py          # Test result models
│   │   ├── chat.py                 # Chat message models
│   │   └── wizard_config.py        # Configuration wizard models
│   │
│   ├── lib/                        # Core library utilities
│   │   ├── errors.py               # Custom exception hierarchy
│   │   ├── template_engine.py      # Jinja2 template rendering
│   │   ├── file_processor.py       # Multimodal file processing (OCR, PDF)
│   │   ├── vector_store.py         # Vector store integrations
│   │   ├── evaluators/             # Evaluation framework
│   │   │   ├── base.py             # Abstract evaluator base class
│   │   │   ├── nlp_metrics.py      # NLP metrics (F1, BLEU, ROUGE, METEOR)
│   │   │   ├── azure_ai.py         # Azure AI evaluation metrics
│   │   │   └── deepeval/           # DeepEval LLM-as-judge evaluators
│   │   └── test_runner/            # Test execution framework
│   │       ├── executor.py         # Main test execution orchestrator
│   │       ├── agent_factory.py    # Agent instantiation from config
│   │       └── reporter.py         # Test result reporting
│   │
│   ├── chat/                       # Chat interface
│   │   ├── session.py              # Chat session management
│   │   ├── streaming.py            # Streaming response handling
│   │   └── executor.py             # Chat execution engine
│   │
│   ├── tools/                      # Tool implementations
│   │   ├── vectorstore_tool.py     # Vector store search tool
│   │   └── mcp/                    # MCP (Model Context Protocol) tools
│   │
│   └── templates/                  # Project templates for `holodeck init`
│       ├── conversational/         # Conversational agent template
│       ├── customer-support/       # Customer support template
│       └── research/               # Research assistant template
│
├── tests/
│   ├── unit/                       # Unit tests (isolated, fast)
│   ├── integration/                # Integration tests (cross-component)
│   ├── fixtures/                   # Test fixtures and sample data
│   └── conftest.py                 # Pytest configuration
│
├── docs/                           # Documentation (MkDocs)
├── specs/                          # Feature specifications (spec-kit)
├── VISION.md                       # Product vision and roadmap
├── AGENTS.md                       # Comprehensive AI agent documentation
└── pyproject.toml                  # Project metadata and dependencies
```

## Development Setup

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# Initialize project (creates venv, installs deps, sets up pre-commit)
make init

# Activate virtual environment
source .venv/bin/activate

# Verify installation
holodeck --version
```

### Dependency Management

```bash
make install-dev              # Install all dependencies including dev
make install-prod             # Production only
uv add <package>              # Add a new dependency
uv add --dev <package>        # Add a development dependency
uv remove <package>           # Remove a dependency
make update-deps              # Update all dependencies
```

### Environment Configuration

HoloDeck loads environment variables from (priority order):

1. Shell environment variables (highest priority)
2. `.env` in current directory (project-level)
3. `~/.holodeck/.env` (user-level defaults)

## Common Development Commands

### Testing

```bash
make test                # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-coverage      # With coverage report (htmlcov/index.html)
make test-failed        # Re-run failed tests only
make test-parallel      # Parallel execution
```

### Code Quality

```bash
make format             # Format with Black + Ruff
make format-check       # Check formatting (CI-safe)
make lint               # Run Ruff + Bandit
make lint-fix           # Auto-fix linting issues
make type-check         # MyPy type checking
make security           # Safety + Bandit + detect-secrets
make ci                 # Run complete CI pipeline locally
```

### Pre-commit

```bash
make install-hooks      # Install pre-commit hooks
make pre-commit         # Run hooks on all files
```

## Code Quality Standards

### Python Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

**Enforced by Tooling:**

- **Formatting:** Black (88 character line length)
- **Linting:** Ruff (comprehensive rule set)
- **Type Checking:** MyPy (strict mode)
- **Security:** Bandit, Safety, detect-secrets
- **Target:** Python 3.10+

### Type Hints

Always use type hints for function parameters and return values:

```python
def process_data(data: dict[str, Any]) -> list[str]:
    """Process data and return results.

    Args:
        data: Input data dictionary

    Returns:
        List of processed strings

    Raises:
        ValueError: If data is invalid
    """
    if not data:
        raise ValueError("data cannot be empty")
    return [str(v) for v in data.values()]
```

### Docstrings (PEP 257)

```python
def calculate_score(
    prediction: str, reference: str, threshold: float = 0.8
) -> float:
    """Calculate similarity score between prediction and reference.

    Args:
        prediction: The predicted output from the agent
        reference: The ground truth reference text
        threshold: Minimum score to consider a match (default: 0.8)

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        ValueError: If prediction or reference is empty

    Example:
        >>> calculate_score("hello world", "hello earth")
        0.73
    """
```

### Error Handling

Use custom exception hierarchy from `holodeck.lib.errors`:

```python
from holodeck.lib.errors import (
    HoloDeckError,
    ConfigError,
    ValidationError,
    ToolError,
    EvaluationError,
)

def load_config(path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not Path(path).exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
```

### Testing Standards

```python
import pytest
from holodeck.config import ConfigLoader
from holodeck.lib.errors import ConfigError


@pytest.mark.unit
def test_load_valid_config(tmp_path):
    """Test loading a valid configuration file."""
    # Arrange
    config_file = tmp_path / "agent.yaml"
    config_file.write_text("name: test-agent\nmodel:\n  provider: openai")

    # Act
    loader = ConfigLoader()
    config = loader.load(str(config_file))

    # Assert
    assert config["name"] == "test-agent"
    assert config["model"]["provider"] == "openai"


@pytest.mark.unit
def test_load_missing_config():
    """Test loading a non-existent configuration file."""
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="not found"):
        loader.load("nonexistent.yaml")
```

**Test Markers:**

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (cross-component)
- `@pytest.mark.slow`: Slow-running tests

**Test Structure (AAA Pattern):**

1. **Arrange:** Set up test data and preconditions
2. **Act:** Execute the code under test
3. **Assert:** Verify the expected outcome

## Key Patterns

### Configuration Loading

```python
from holodeck.config.loader import ConfigLoader
from holodeck.models.agent import Agent

# Load YAML configuration
loader = ConfigLoader()
raw_config = loader.load("agent.yaml")  # Returns dict with env vars resolved

# Validate and parse with Pydantic
agent = Agent(**raw_config)  # Raises ValidationError if invalid
```

### Tool Loading (Discriminated Union)

```python
from holodeck.models.tool import (
    VectorStoreConfig,
    FunctionConfig,
    MCPConfig,
    PromptConfig,
    ToolUnion,
)

tool_config: ToolUnion = agent.tools[0]

if isinstance(tool_config, VectorStoreConfig):
    tool = create_vectorstore_tool(tool_config)
elif isinstance(tool_config, MCPConfig):
    tool = create_mcp_tool(tool_config)
# etc.
```

### Async/Await Pattern

Use async/await for all I/O operations:

```python
import asyncio

async def invoke_agent(agent, message: str) -> str:
    """Invoke agent with a message and return response."""
    response = await agent.invoke_async(message)
    return str(response)

async def run_test_cases(agent, test_cases: list[str]) -> list[str]:
    """Run multiple test cases concurrently."""
    tasks = [invoke_agent(agent, test) for test in test_cases]
    return await asyncio.gather(*tasks)
```

## Agent Configuration Schema

```yaml
name: string # Required: Agent identifier
description: string # Optional: Human-readable description

model: # Required: LLM provider configuration
  provider: openai | azure_openai | anthropic | ollama
  name: string # Model name (e.g., "gpt-4o")
  temperature: float # 0.0 to 2.0 (default: 0.7)
  max_tokens: int # Max output tokens

instructions: # Required: System instructions
  file: path # Path to instruction file
  # OR
  inline: string # Inline instruction text

tools: # Optional: List of tools
  - name: string
    type: vectorstore | function | mcp | prompt
    # Type-specific configuration...

evaluations: # Optional: Evaluation configuration
  default_model: { ... } # Global model for all metrics
  metrics:
    - metric: groundedness | relevance | f1_score | bleu | geval
      threshold: float
      model: { ... } # Per-metric model override

test_cases: # Optional: Test scenarios
  - input: string
    ground_truth: string
    expected_tools: [string]
    files: [...] # Multimodal file inputs
```

## Workflow

This project uses **spec-kit** for feature development:

1. **Create spec:** `/speckit.specify`
2. **Clarify:** `/speckit.clarify`
3. **Plan:** `/speckit.plan`
4. **Create tasks:** `/speckit.tasks`
5. **Plan todo list** - Read all files in `specs/<spec_name>/`:
   - spec.md, plan.md, tasks.md, data-model.md
   - research.md, quickstart.md (if exist)
   - contracts/\*.md (if any)
6. **Implement** - Follow the todo list
7. **Run code quality checks** after each task:

When planning for tasks or implementing any feature, always extract context from the relevant spec-kit files in `specs/`.

```bash
make format             # Format with Black + Ruff
make lint               # Run Ruff + Bandit
make lint-fix           # Auto-fix linting issues
make type-check         # MyPy type checking
make security           # Safety + Bandit + detect-secrets
```

Always `source .venv/bin/activate` before running Python commands.

## Git Commit Guidelines

When generating commit messages:

- **Do NOT attribute Claude Code** in commit messages
- Do NOT include "Generated with Claude Code" or similar attributions
- Write clean, conventional commit messages focused on the changes

## Do's and Don'ts

### DO's

1. **DO use Pydantic models** for all configuration
2. **DO use async/await** for I/O operations
3. **DO use type hints** everywhere (enforced by MyPy)
4. **DO handle errors** with custom exceptions from `lib/errors.py`
5. **DO write comprehensive tests** (80%+ coverage)
6. **DO follow DRY principle** - extract common logic
7. **DO use MCP** for external API integrations
8. **DO run code quality checks** before committing

### DON'Ts

1. **DON'T use print()** - use Click's echo() in CLI, logging elsewhere
2. **DON'T hardcode configuration** - use env vars and YAML
3. **DON'T ignore type checking errors** - fix them
4. **DON'T write synchronous I/O** in async functions
5. **DON'T catch broad exceptions** without re-raising
6. **DON'T commit** without running tests
7. **DON'T use mutable default arguments**:

   ```python
   # BAD
   def my_func(items: list = []): pass

   # GOOD
   def my_func(items: list | None = None):
       items = items or []
   ```

8. **DON'T skip docstrings** - all public functions need them

## Key Design Constraints

1. **No-Code First**: Users configure agents via YAML, not Python
2. **MCP for APIs**: External API integrations must use MCP servers
3. **OpenTelemetry Native**: Observability follows GenAI semantic conventions
4. **Evaluation Flexibility**: Support model configuration at global, run, and metric levels
5. **Multimodal Testing**: First-class support for images, PDFs, Office docs

## Additional Resources

- **VISION.md**: Product vision and feature specifications
- **AGENTS.md**: Comprehensive documentation for AI agents
- **specs/**: Feature specifications (spec-kit format)

### External References

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

**Remember:** HoloDeck is about enabling no-code agent development. Every feature should be configurable through YAML without requiring Python code.

## Active Technologies
- N/A (uses existing vector store infrastructure) (015-vectorstore-reranking)

## Recent Changes
- 015-vectorstore-reranking: Added Python 3.10+
