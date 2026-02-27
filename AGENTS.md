# AGENTS.md

**Comprehensive Documentation for AI Agents Working on HoloDeck**

This file provides detailed guidance for AI agents working with the HoloDeck codebase. It covers architecture, patterns, conventions, and quality standards.

---

## Project Overview

**HoloDeck** is an open-source experimentation platform for building, testing, and deploying AI agents through pure YAML configuration. The project enables teams to go from hypothesis to production API in minutes without writing code.

**Core Value Proposition:** No-code agent definition. Users define agents, tools, evaluations, and deployments entirely through YAML files.

**Current Status:** Early development (v0.1 in progress)

- CLI and configuration infrastructure: **Complete**
- Agent execution engine: **Complete**
- Evaluation framework: **Complete**
- Chat interface: **Complete**
- Multi-backend abstraction layer: **Complete**
- Deployment engine: **Planned**

**Technology Stack:**

- **Language:** Python 3.10+
- **Package Manager:** UV (fast, modern replacement for pip/Poetry)
- **Agent Backends:** Multi-backend architecture
  - Semantic Kernel (OpenAI, Azure OpenAI, Ollama)
  - Claude Agent SDK (native Anthropic — first-class citizen)
- **CLI:** Click
- **Configuration:** Pydantic v2 + YAML
- **Testing:** Pytest with async support
- **Evaluation:** Azure AI Evaluation + DeepEval + NLP metrics

---

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
│  ├─ LLMProvider: LLM provider settings (all providers)       │
│  ├─ ClaudeConfig: Claude Agent SDK settings                  │
│  ├─ ToolUnion: Tool definitions (6 types)                    │
│  ├─ EvaluationConfig: Metrics and thresholds                 │
│  └─ TestCaseModel: Test cases with multimodal file support   │
└─────────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Backend Abstraction Layer (auto-routed)            │
│  ├─ BackendSelector: Routes by model.provider                │
│  ├─ AgentBackend / AgentSession: Provider-agnostic protocols │
│  ├─ ExecutionResult: Unified response model                  │
│  └─ ContextGenerator: Contextual embeddings protocol         │
├─────────────────────────────┬───────────────────────────────┤
│   SK Backend                │   Claude Backend              │
│   (OpenAI, Azure, Ollama)   │   (Anthropic — first-class)   │
│  ├─ ChatCompletionAgent     │  ├─ Claude Agent SDK          │
│  ├─ SK Tool Plugins         │  ├─ Tool Adapters + MCP Bridge│
│  └─ SK Memory / History     │  ├─ OTel Bridge               │
│                             │  └─ Startup Validators        │
└─────────────────────────────┴───────────────────────────────┘
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

**1. Configuration-Driven Design**

- All agent behavior defined via YAML
- Pydantic models enforce schema validation
- Environment variable substitution for secrets

**2. Plugin Architecture**

- Tool system supports 6 types: vectorstore, function, MCP, prompt, plugin, hierarchical_document
- MCP protocol for standardized integrations (design decision: APIs use MCP, not custom types)
- Claude tool adapter bridge wraps HoloDeck tools as SDK-compatible MCP tools
- Dynamic tool loading from configuration

**3. Multi-Backend Abstraction**

- Protocol-driven: `AgentBackend`, `AgentSession`, `ContextGenerator` define provider-agnostic interfaces
- `BackendSelector.select()` auto-routes by `model.provider`:
  - OpenAI / Azure OpenAI / Ollama → `SKBackend` (Semantic Kernel)
  - Anthropic → `ClaudeBackend` (Claude Agent SDK — first-class citizen)
- Tool adapters bridge HoloDeck tools (VectorStore, HierarchicalDocument) to SDK MCP format
- MCP bridge translates HoloDeck MCP configs to Claude SDK subprocess configs
- OTel bridge translates observability config to subprocess environment variables
- Startup validators surface configuration errors before spawning Claude subprocess

**4. Multimodal Testing**

- File processor handles images (OCR), PDFs, Office documents
- Test cases can include multiple files with metadata
- MarkItDown library for document-to-markdown conversion

**5. Evaluation Flexibility**

- Three-level model configuration: global, per-evaluation, per-metric
- Support for AI-powered metrics (Azure AI, DeepEval) and NLP metrics
- Async evaluation execution for performance

**6. Streaming Architecture**

- Real-time streaming for chat interface
- Progress indicators for test execution
- Async/await throughout for non-blocking I/O

---

## Directory Structure

```
holodeck/
├── src/holodeck/
│   ├── __init__.py                 # Package entry point with version
│   ├── cli/                        # Command-line interface
│   │   ├── main.py                 # CLI entry point (holodeck command)
│   │   ├── commands/               # CLI commands
│   │   │   ├── init.py             # Project initialization (templates)
│   │   │   ├── test.py             # Test runner command
│   │   │   ├── chat.py             # Interactive chat command
│   │   │   └── config.py           # Configuration wizard
│   │   ├── utils/                  # CLI utilities
│   │   │   ├── project_init.py     # Project scaffolding logic
│   │   │   └── wizard.py           # Interactive configuration wizard
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
│   │   ├── tool.py                 # Tool configuration models (6 types)
│   │   ├── claude_config.py        # Claude Agent SDK configuration models
│   │   ├── evaluation.py           # Evaluation metrics models
│   │   ├── test_case.py            # Test case models
│   │   ├── test_result.py          # Test result models
│   │   ├── chat.py                 # Chat message models
│   │   ├── token_usage.py          # Token tracking models
│   │   ├── tool_event.py           # Tool execution event models
│   │   ├── tool_execution.py       # Tool execution state models
│   │   ├── project_config.py       # Project metadata model
│   │   ├── template_manifest.py    # Template manifest model
│   │   └── wizard_config.py        # Configuration wizard models
│   │
│   ├── lib/                        # Core library utilities
│   │   ├── errors.py               # Custom exception hierarchy
│   │   ├── exceptions.py           # Legacy exceptions (to be consolidated)
│   │   ├── template_engine.py      # Jinja2 template rendering
│   │   ├── file_processor.py       # Multimodal file processing (OCR, PDF)
│   │   ├── vector_store.py         # Vector store integrations (Semantic Kernel)
│   │   ├── text_chunker.py         # Text chunking for embeddings
│   │   ├── validation.py           # Validation utilities
│   │   ├── logging_config.py       # Logging configuration
│   │   ├── logging_utils.py        # Logging utilities
│   │   ├── tool_initializer.py     # Shared tool init (both backends)
│   │   ├── instruction_resolver.py # Shared instruction loading
│   │   ├── claude_context_generator.py  # Claude SDK context generation
│   │   ├── llm_context_generator.py     # Generic LLM context generation
│   │   │
│   │   ├── backends/               # Multi-backend abstraction layer
│   │   │   ├── __init__.py         # Public API exports
│   │   │   ├── base.py             # Protocols: AgentBackend, AgentSession, ContextGenerator
│   │   │   ├── selector.py         # BackendSelector (routes by provider)
│   │   │   ├── sk_backend.py       # Semantic Kernel backend (OpenAI/Azure/Ollama)
│   │   │   ├── claude_backend.py   # Claude Agent SDK backend (Anthropic)
│   │   │   ├── tool_adapters.py    # Wraps HD tools as SDK MCP tools
│   │   │   ├── mcp_bridge.py       # Translates MCP configs for Claude SDK
│   │   │   ├── otel_bridge.py      # Translates OTel config for Claude subprocess
│   │   │   └── validators.py       # Startup validators (Node.js, credentials, etc.)
│   │   │
│   │   ├── evaluators/             # Evaluation framework
│   │   │   ├── __init__.py         # Evaluator exports
│   │   │   ├── base.py             # Abstract evaluator base class
│   │   │   ├── nlp_metrics.py      # NLP metrics (F1, BLEU, ROUGE, METEOR)
│   │   │   ├── azure_ai.py         # Azure AI evaluation metrics
│   │   │   ├── param_spec.py       # Parameter specification helpers
│   │   │   └── deepeval/           # DeepEval LLM-as-judge evaluators
│   │   │       ├── __init__.py     # DeepEval exports
│   │   │       ├── base.py         # DeepEval base evaluator
│   │   │       ├── config.py       # DeepEval model configuration
│   │   │       ├── errors.py       # DeepEval exceptions
│   │   │       ├── geval.py        # G-Eval custom criteria evaluator
│   │   │       ├── faithfulness.py           # Faithfulness evaluator
│   │   │       ├── answer_relevancy.py       # Answer relevancy evaluator
│   │   │       ├── contextual_precision.py   # Contextual precision evaluator
│   │   │       ├── contextual_recall.py      # Contextual recall evaluator
│   │   │       └── contextual_relevancy.py   # Contextual relevancy evaluator
│   │   │
│   │   └── test_runner/            # Test execution framework
│   │       ├── __init__.py         # Test runner exports
│   │       ├── executor.py         # Main test execution orchestrator
│   │       ├── agent_factory.py    # Agent instantiation from config
│   │       ├── progress.py         # Real-time progress indicators
│   │       ├── reporter.py         # Test result reporting
│   │       └── eval_kwargs_builder.py  # Evaluation parameter builder
│   │
│   ├── chat/                       # Chat interface
│   │   ├── __init__.py             # Chat exports
│   │   ├── session.py              # Chat session management
│   │   ├── message.py              # Message models
│   │   ├── streaming.py            # Streaming response handling
│   │   ├── progress.py             # Chat progress indicators
│   │   └── executor.py             # Chat execution engine
│   │
│   ├── tools/                      # Tool implementations
│   │   ├── __init__.py             # Tool exports
│   │   ├── vectorstore_tool.py     # Vector store search tool
│   │   └── mcp/                    # MCP (Model Context Protocol) tools
│   │       ├── __init__.py         # MCP exports
│   │       ├── factory.py          # MCP tool factory
│   │       ├── utils.py            # MCP utilities
│   │       └── errors.py           # MCP exceptions
│   │
│   └── templates/                  # Project templates for `holodeck init`
│       ├── __init__.py             # Template exports
│       ├── _static/                # Shared static files
│       ├── conversational/         # Conversational agent template
│       ├── customer-support/       # Customer support template
│       └── research/               # Research assistant template
│
├── tests/
│   ├── unit/                       # Unit tests (isolated, fast)
│   │   └── lib/
│   │       └── backends/           # Backend abstraction tests
│   │           ├── test_base.py            # ExecutionResult, protocol tests
│   │           ├── test_claude_backend.py  # ClaudeBackend tests
│   │           ├── test_sk_backend.py      # SKBackend tests
│   │           ├── test_selector.py        # BackendSelector routing tests
│   │           ├── test_tool_adapters.py   # Tool adapter tests
│   │           ├── test_mcp_bridge.py      # MCP bridge tests
│   │           ├── test_otel_bridge.py     # OTel bridge tests
│   │           └── test_validators.py      # Validator tests
│   ├── integration/                # Integration tests (cross-component)
│   ├── fixtures/                   # Test fixtures and sample data
│   └── conftest.py                 # Pytest configuration
│
├── docs/                           # Documentation (MkDocs)
├── specs/                          # Feature specifications (spec-kit)
├── .github/workflows/              # CI/CD pipelines
├── VISION.md                       # Product vision and roadmap
├── README.md                       # User-facing documentation
├── CLAUDE.md                       # AI agent instructions
├── AGENTS.md                       # This file
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Dependency lock file
└── Makefile                        # Development workflow commands
```

---

## Development Setup

### Prerequisites

- Python 3.10+
- UV package manager (https://astral.sh/uv)

### Initial Setup

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

### Dependency Management with UV

UV is 10-100x faster than pip/Poetry and provides flexible dependency overrides.

```bash
# Install all dependencies including dev
make install-dev
# or: uv sync --all-extras

# Install production dependencies only
make install-prod
# or: uv sync --no-dev

# Add a new dependency
uv add <package>

# Add a development dependency
uv add --dev <package>

# Remove a dependency
uv remove <package>

# Update all dependencies
make update-deps
# or: uv lock --upgrade && uv sync --all-extras

# Install specific extras (vector stores)
uv sync --extra pinecone
uv sync --extra postgres
uv sync --extra qdrant
uv sync --extra chromadb
uv sync --extra vectorstores  # All vector stores
```

### Environment Configuration

HoloDeck loads environment variables from two locations (priority order):

1. Shell environment variables (highest priority, never overwritten)
2. `.env` in current directory (project-level config)
3. `~/.holodeck/.env` (user-level defaults)

Example `.env` file:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=sk-...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
ANTHROPIC_API_KEY=...

# Vector Store Connections
REDIS_URL=redis://localhost:6379
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/vectors

# Observability
LANGSMITH_API_KEY=...
DATADOG_API_KEY=...
```

---

## Common Development Commands

### Testing

**IMPORTANT**: Always run tests with parallel execution (`-n auto`) for faster results.

```bash
make test                # Run all tests (parallel)
make test-unit          # Unit tests only (parallel)
make test-integration   # Integration tests only (parallel)
make test-coverage      # With coverage report (htmlcov/index.html)
make test-failed        # Re-run failed tests only
make test-parallel      # Parallel execution (same as make test)

# When running pytest directly, ALWAYS use -n auto:
pytest tests/unit/ -n auto              # Run unit tests in parallel
pytest tests/unit/lib/ -n auto -v       # Verbose with parallel
pytest tests/ -n auto -q                # Quiet mode with parallel

# Run specific test file
pytest tests/unit/test_config.py -n auto -v

# Run specific test function
pytest tests/unit/test_config.py::test_load_yaml -v

# Run tests matching pattern
pytest -k "test_agent" -n auto -v
```

### Code Quality

```bash
# Format code with Black + Ruff
make format
# or: uv run black . && uv run ruff check --fix .

# Check formatting (CI-safe, no changes)
make format-check
# or: uv run black --check .

# Run linting (Ruff + Bandit)
make lint
# or: uv run ruff check . && uv run bandit -r src/

# Auto-fix linting issues
make lint-fix
# or: uv run ruff check --fix .

# Type checking with MyPy
make type-check
# or: uv run mypy src/

# Security scanning (Safety + Bandit + detect-secrets)
make security
# or: uv run pip-audit && uv run bandit -r src/ && uv run detect-secrets scan

# Run all code quality checks
make ci
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make install-hooks
# or: uv run pre-commit install

# Run pre-commit on all files
make pre-commit
# or: uv run pre-commit run --all-files

# Update pre-commit hooks to latest versions
uv run pre-commit autoupdate
```

### Cleanup

```bash
# Remove temporary files/caches
make clean

# Deep clean including venv
make clean-all
```

---

## Code Quality Standards

### Python Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

**Enforced by Tooling:**

- **Formatting:** Black (88 character line length)
- **Linting:** Ruff (comprehensive rule set)
- **Type Checking:** MyPy (strict mode)
- **Security:** Bandit, Safety, detect-secrets
- **Target:** Python 3.10+

### Formatting Rules (Black + Ruff)

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces
# Quotes: Double quotes for strings (Black preference)
# Imports: Sorted by isort (Ruff I rule)

# Example:
from typing import Any

from pydantic import BaseModel, Field

from holodeck.config import ConfigLoader
from holodeck.models import Agent


class MyModel(BaseModel):
    """Concise one-line summary.

    Detailed description with examples and usage notes.

    Attributes:
        field_name: Description of the field
    """

    field_name: str = Field(
        default=..., description="Field description for schema"
    )
```

### Linting Rules (Ruff)

Enabled rule sets:

- **E/W:** pycodestyle errors and warnings
- **F:** pyflakes (undefined names, unused imports)
- **I:** isort (import sorting)
- **B:** flake8-bugbear (common bugs)
- **C4:** flake8-comprehensions (better list/dict comprehensions)
- **UP:** pyupgrade (modern Python syntax)
- **N:** pep8-naming (naming conventions)
- **SIM:** flake8-simplify (code simplification)
- **S:** flake8-bandit (security issues)

Exceptions for tests:

- S101: Allow `assert` in tests
- S603/S607: Allow subprocess in tests

### Type Checking (MyPy)

Strict type checking enforced:

```python
# Always use type hints
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


# Type hints for class attributes
class Agent:
    name: str
    model: LLMProvider
    tools: list[ToolUnion] | None = None

    def __init__(self, name: str, model: LLMProvider) -> None:
        self.name = name
        self.model = model
```

MyPy settings (from pyproject.toml):

- `disallow_untyped_defs = true`: All functions must have type hints
- `strict_optional = true`: Strict None checking
- `warn_return_any = true`: Warn when returning Any
- `warn_unreachable = true`: Warn about unreachable code

### Docstring Standards (PEP 257)

```python
def calculate_score(
    prediction: str, reference: str, threshold: float = 0.8
) -> float:
    """Calculate similarity score between prediction and reference.

    This function uses fuzzy matching to compute a normalized similarity
    score between 0.0 and 1.0. Scores above the threshold indicate a match.

    Args:
        prediction: The predicted output from the agent
        reference: The ground truth reference text
        threshold: Minimum score to consider a match (default: 0.8)

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        ValueError: If prediction or reference is empty
        TypeError: If inputs are not strings

    Example:
        >>> calculate_score("hello world", "hello earth")
        0.73

    Note:
        This implementation uses Levenshtein distance for matching.
    """
    if not prediction or not reference:
        raise ValueError("prediction and reference must be non-empty")
    # Implementation...
```

**Docstring Format:**

- One-line summary (imperative mood: "Calculate", not "Calculates")
- Blank line
- Detailed description (optional)
- Args section (for parameters)
- Returns section (for return value)
- Raises section (for exceptions)
- Example section (optional, but recommended)
- Note/Warning sections (optional)

### Error Handling

```python
# Use custom exception hierarchy
from holodeck.lib.errors import (
    HoloDeckError,
    ConfigError,
    ValidationError,
    EvaluationError,
)

# Raise specific exceptions with context
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
    # Arrange
    loader = ConfigLoader()

    # Act & Assert
    with pytest.raises(ConfigError, match="not found"):
        loader.load("nonexistent.yaml")


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("hello", "HELLO"),
        ("", ""),
        ("MiXeD", "MIXED"),
    ],
)
def test_uppercase_conversion(input_text, expected):
    """Test uppercase conversion with various inputs."""
    assert input_text.upper() == expected
```

**Test Markers:**

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (cross-component)
- `@pytest.mark.slow`: Slow-running tests

**Test Naming:**

- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

**Test Structure (AAA Pattern):**

1. **Arrange:** Set up test data and preconditions
2. **Act:** Execute the code under test
3. **Assert:** Verify the expected outcome

---

## Key Patterns and Conventions

### 1. Configuration Loading Pattern

All agent configurations follow this flow:

```python
from holodeck.config.loader import ConfigLoader
from holodeck.models.agent import Agent

# Load YAML configuration
loader = ConfigLoader()
raw_config = loader.load("agent.yaml")  # Returns dict with env vars resolved

# Validate and parse with Pydantic
agent = Agent(**raw_config)  # Raises ValidationError if invalid

# Access typed fields
print(agent.name)  # str
print(agent.model.provider)  # "openai" | "azure_openai" | "anthropic" | "ollama"
print(agent.tools)  # list[ToolUnion] | None
```

### 2. Tool Loading Pattern

Tools are loaded dynamically based on configuration:

```python
from holodeck.models.tool import (
    VectorStoreConfig,
    FunctionConfig,
    MCPConfig,
    PromptConfig,
    ToolUnion,
)

# Tool types (discriminated union via Pydantic)
tool_config: ToolUnion = agent.tools[0]

if isinstance(tool_config, VectorStoreConfig):
    # Load vector store tool
    tool = create_vectorstore_tool(tool_config)
elif isinstance(tool_config, FunctionConfig):
    # Load custom function tool
    tool = load_function_tool(tool_config)
elif isinstance(tool_config, MCPConfig):
    # Load MCP tool
    tool = create_mcp_tool(tool_config)
elif isinstance(tool_config, PromptConfig):
    # Create AI-powered prompt tool
    tool = create_prompt_tool(tool_config)
```

### 3. Evaluation Pattern

Evaluations support three-level model configuration:

```python
from holodeck.lib.evaluators import GEvalEvaluator
from holodeck.models.evaluation import EvaluationConfig

# Level 1: Global default model (applies to all metrics)
eval_config = EvaluationConfig(
    default_model={
        "provider": "openai",
        "name": "gpt-4o-mini",
        "temperature": 0.0,
    },
    metrics=[
        # Level 2: Per-metric override
        {
            "metric": "groundedness",
            "threshold": 4.0,
            "model": {
                "provider": "openai",
                "name": "gpt-4o",  # Use more powerful model for critical metrics
                "temperature": 0.0,
            },
        },
        # Level 3: Uses global default_model
        {"metric": "relevance", "threshold": 4.0},
    ],
)

# Create evaluator with resolved model
evaluator = GEvalEvaluator(
    name="Groundedness",
    criteria="Evaluate if the response is grounded in context",
    model_config=resolved_model,
)
```

### 4. Async/Await Pattern

Use async/await for I/O operations:

```python
import asyncio
from semantic_kernel.agents import ChatCompletionAgent

async def invoke_agent(agent: ChatCompletionAgent, message: str) -> str:
    """Invoke agent with a message and return response."""
    response = await agent.invoke_async(message)
    return str(response)


async def run_test_cases(agent: ChatCompletionAgent, test_cases: list[str]) -> list[str]:
    """Run multiple test cases concurrently."""
    tasks = [invoke_agent(agent, test) for test in test_cases]
    results = await asyncio.gather(*tasks)
    return results


# Usage
results = asyncio.run(run_test_cases(agent, ["test 1", "test 2", "test 3"]))
```

### 5. Error Handling Pattern

Use custom exception hierarchy:

```python
from holodeck.lib.errors import (
    HoloDeckError,
    ConfigError,
    ValidationError,
    ToolError,
    EvaluationError,
)

try:
    config = loader.load("agent.yaml")
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    raise
except ValidationError as e:
    logger.error(f"Validation error: {e}")
    raise
except HoloDeckError as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### 6. Logging Pattern

Use structured logging with context:

```python
import logging
from holodeck.lib.logging_config import setup_logging

# Setup logging (call once at application start)
setup_logging(level=logging.INFO)

# Get logger for module
logger = logging.getLogger(__name__)

# Log with context
logger.info("Loading configuration", extra={"config_path": path})
logger.warning("Missing optional field", extra={"field": "description"})
logger.error("Failed to load config", extra={"error": str(e)}, exc_info=True)
```

### 7. File Processing Pattern

Handle multimodal file inputs:

```python
from holodeck.lib.file_processor import FileProcessor
from holodeck.models.test_case import FileInputConfig

processor = FileProcessor()

# Process image (OCR)
file_config = FileInputConfig(
    path="tests/fixtures/receipt.jpg", type="image"
)
content = await processor.process_file(file_config)

# Process PDF (specific pages)
file_config = FileInputConfig(
    path="tests/fixtures/contract.pdf",
    type="pdf",
    pages=[1, 2, 3],
)
content = await processor.process_file(file_config)

# Process Excel (specific sheet and range)
file_config = FileInputConfig(
    path="tests/fixtures/data.xlsx",
    type="excel",
    sheet="Q4 Summary",
    range="A1:E100",
)
content = await processor.process_file(file_config)
```

### 8. Backend Selection Pattern

Use `BackendSelector` to route agents to the correct backend — never instantiate backends directly:

```python
from holodeck.lib.backends import BackendSelector, ExecutionResult

# Auto-routes by model.provider (anthropic → Claude, others → SK)
backend = await BackendSelector.select(agent, tool_instances, mode="test")
await backend.initialize()

# Single-turn (test runner)
result: ExecutionResult = await backend.invoke_once("What is our refund policy?")

# Multi-turn (chat)
session = await backend.create_session()
r1 = await session.send("Hello")
r2 = await session.send("Tell me more")
await session.close()

await backend.teardown()
```

### 9. Claude Backend Initialization Flow

`ClaudeBackend.initialize()` performs these steps in order:

1. **Node.js validation** — `validate_nodejs()` checks `node` is on PATH
2. **Credentials validation** — `validate_credentials()` checks API key / OAuth / Bedrock / Vertex / Foundry
3. **Embedding provider validation** — `validate_embedding_provider()` ensures external embedding config when using vectorstore tools
4. **Tool filtering warning** — `validate_tool_filtering()` warns if tool_filtering set (Claude SDK manages selection natively)
5. **Tool initialization** — `tool_initializer.initialize_tools()` creates vectorstore/hierarchical-doc tool instances
6. **Tool adapters** — `create_tool_adapters()` wraps HoloDeck tools as SDK-compatible MCP tools
7. **SDK server** — `build_holodeck_sdk_server()` bundles adapters into in-process MCP server
8. **External MCP** — `build_claude_mcp_configs()` translates MCP tool configs to SDK format
9. **OTel env vars** — `translate_observability()` configures OpenTelemetry for subprocess
10. **Build options** — `build_options()` assembles `ClaudeAgentOptions` from all above
11. **Working directory / response format validation**

### 10. ContextGenerator Resolution Chain

When initializing hierarchical document tools, the context generator is resolved via a 5-tier priority chain:

1. **Caller-provided** `context_generator` (highest priority)
2. **Caller-provided** `chat_service` → wrapped in `LLMContextGenerator`
3. **Tool config** `context_model` → creates chat service → `LLMContextGenerator`
4. **Anthropic provider** → `ClaudeSDKContextGenerator` (uses `claude-haiku-4-5` via SDK `query()`)
5. **None** → graceful degradation (chunks without contextual embeddings)

---

## Agent Configuration Schema

### Complete YAML Schema

```yaml
# agent.yaml
name: string # Required: Agent identifier
description: string # Optional: Human-readable description
author: string # Optional: Agent author

model: # Required: LLM provider configuration
  provider: openai | azure_openai | anthropic | ollama
  name: string # Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514")
  temperature: float # 0.0 to 2.0 (default: 0.7)
  max_tokens: int # Max output tokens (optional)
  top_p: float # Nucleus sampling (optional)
  auth_provider: string # Anthropic only: api_key | oauth_token | bedrock | vertex | foundry
  # Provider-specific fields
  endpoint: string # Azure OpenAI endpoint (azure_openai only)
  api_version: string # Azure API version (azure_openai only)

embedding_provider: # Required when provider=anthropic + vectorstore/hierarchical_document tools
  provider: openai | azure_openai | ollama
  name: string # Embedding model name
  # Provider-specific fields (endpoint, api_version, etc.)

instructions: # Required: System instructions
  file: path # Path to instruction file (exclusive with inline)
  # OR
  inline: string # Inline instruction text (exclusive with file)

response_format: # Optional: Structured output schema
  type: json_object | json_schema
  # For json_schema:
  schema:
    type: object
    properties: { ... }

tools: # Optional: List of tools
  # Vector Store Tool
  - name: string
    type: vectorstore
    provider: redis | postgres | qdrant | chromadb | pinecone
    connection: string # Connection string
    source: path # Data source directory
    embedding_model: string # Embedding model name
    description: string # Tool description

  # Custom Function Tool
  - name: string
    type: function
    file: path # Python file path
    function: string # Function name
    description: string # Tool description
    parameters: # Function parameters
      param_name:
        type: string | float | int | boolean | array | object
        description: string

  # MCP Tool
  - name: string
    type: mcp
    server: string # MCP server (package or path)
    transport: stdio | sse # Transport protocol
    config: # Server-specific config
      key: value

  # Prompt Tool (AI-powered semantic function)
  - name: string
    type: prompt
    description: string # Tool description
    template: string # Jinja2 template (inline)
    # OR
    file: path # Path to template file
    parameters: # Template parameters
      param_name:
        type: string
        description: string
    model: # Optional: Override model for this tool
      provider: openai
      name: gpt-4o-mini
      temperature: 0.3

claude: # Optional: Claude Agent SDK configuration (anthropic provider only)
  working_directory: path # Scope file access; subprocess cwd
  permission_mode: manual | acceptEdits | acceptAll  # Default: manual
  max_turns: int # Max agent loop iterations
  extended_thinking: # Deep reasoning
    enabled: boolean
    budget_tokens: int # 1000-100000
  web_search: boolean # Built-in web search (default: false)
  bash: # Shell command execution
    enabled: boolean
    excluded_commands: [string]
    allow_unsafe: boolean
  file_system: # File access settings
    read: boolean
    write: boolean
    edit: boolean
  subagents: # Parallel sub-agent execution
    enabled: boolean
    max_parallel: int # 1-16
  allowed_tools: [string] # Explicit tool allowlist (null = all configured)

evaluations: # Optional: Evaluation configuration
  default_model: # Optional: Global model for all metrics
    provider: openai
    name: gpt-4o-mini
    temperature: 0.0

  metrics: # List of metrics to evaluate
    # AI-powered metric
    - metric: groundedness | relevance | coherence | safety
      threshold: float # Minimum passing score
      enabled: boolean # Enable/disable metric
      model: # Optional: Per-metric model override
        provider: openai
        name: gpt-4o
        temperature: 0.0

    # NLP metric
    - metric: f1_score | bleu | rouge | meteor
      threshold: float
      enabled: boolean

    # Custom G-Eval metric
    - metric: geval
      name: string # Metric name
      criteria: string # Evaluation criteria
      threshold: float
      model: { ... } # Model configuration

test_cases: # Optional: Test scenarios
  - name: string # Optional: Test case name
    input: string # Required: Test input query
    ground_truth: string # Optional: Expected output
    expected_tools: list[string] # Optional: Expected tools used

    files: # Optional: Multimodal file inputs
      # Image input
      - path: path # File path
        type: image # File type
        description: string # Optional: File description

      # PDF input
      - path: path
        type: pdf
        pages: list[int] # Optional: Specific pages
        extract_images: boolean # Optional: Extract embedded images
        ocr: boolean # Optional: Apply OCR to scanned pages

      # Excel input
      - path: path
        type: excel
        sheet: string # Optional: Sheet name
        range: string # Optional: Cell range (e.g., "A1:E100")

      # URL input
      - url: string # Remote file URL
        type: pdf | image | csv
        cache: boolean # Optional: Cache for reuse

    evaluations: # Optional: Per-test-case evaluations
      - f1_score
      - bleu
      - groundedness

execution: # Optional: Test execution configuration
  parallel: boolean # Run tests in parallel (default: false)
  timeout_ms: int # Timeout per test (milliseconds)
  retry_on_failure: int # Retry count on failure
```

### Example: Customer Support Agent

```yaml
name: "customer-support-agent"
description: "Handles customer inquiries with empathy and accuracy"
author: "support-team"

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7
  max_tokens: 500

instructions:
  file: instructions/system-prompt.md

response_format:
  type: json_schema
  schema:
    type: object
    properties:
      answer:
        type: string
        description: "The customer support response"
      sentiment:
        type: string
        enum: ["positive", "neutral", "negative"]
      requires_escalation:
        type: boolean

tools:
  - name: search_knowledge_base
    type: vectorstore
    provider: redis
    connection: "${REDIS_URL}"
    source: data/faqs/
    embedding_model: text-embedding-3-small
    description: "Search customer FAQ database"

  - name: check_order_status
    type: function
    file: tools/orders.py
    function: get_order_status
    description: "Retrieve order status by order ID"
    parameters:
      order_id:
        type: string
        description: "The order ID to look up"

  - name: create_support_ticket
    type: mcp
    server: "@holodeck/mcp-jira"
    config:
      api_key: "${JIRA_API_KEY}"
      project: "SUPPORT"

evaluations:
  default_model:
    provider: openai
    name: gpt-4o-mini
    temperature: 0.0

  metrics:
    - metric: groundedness
      threshold: 4.0
      model:
        provider: openai
        name: gpt-4o
        temperature: 0.0

    - metric: relevance
      threshold: 4.0

    - metric: geval
      name: "Empathy"
      criteria: "Evaluate if the response demonstrates empathy and understanding"
      threshold: 4.0

test_cases:
  - name: "Basic FAQ handling"
    input: "What is your return policy?"
    expected_tools: ["search_knowledge_base"]
    ground_truth: "You can return items within 30 days with a receipt"
    evaluations:
      - groundedness
      - relevance
      - f1_score

  - name: "Order status check"
    input: "Where is my order #12345?"
    expected_tools: ["check_order_status"]
    ground_truth: "Your order shipped on Jan 15 and arrives Jan 18"
    evaluations:
      - relevance
      - f1_score

  - name: "Complex issue requiring escalation"
    input: "I received a damaged item and need a refund urgently"
    expected_tools: ["search_knowledge_base", "create_support_ticket"]
    evaluations:
      - empathy
      - groundedness

execution:
  parallel: false
  timeout_ms: 10000
  retry_on_failure: 1
```

### Example: Claude-Native Agent (Minimal)

```yaml
name: "claude-support-agent"
model:
  provider: anthropic
  name: claude-sonnet-4-20250514

instructions:
  inline: "You are a helpful customer support agent."

claude:
  permission_mode: manual
```

### Example: Claude-Native Agent (Full)

```yaml
name: "claude-research-agent"
description: "Research assistant powered by Claude with extended thinking"

model:
  provider: anthropic
  name: claude-sonnet-4-20250514
  temperature: 0.3
  auth_provider: api_key  # api_key | oauth_token | bedrock | vertex | foundry

# Anthropic cannot generate embeddings — external provider required for vectorstore tools
embedding_provider:
  provider: ollama
  name: nomic-embed-text:latest

instructions:
  file: instructions/system-prompt.md

claude:
  working_directory: ./workspace
  permission_mode: manual
  max_turns: 25
  extended_thinking:
    enabled: true
    budget_tokens: 10000
  web_search: true
  bash:
    enabled: false
  file_system:
    read: true
    write: false
    edit: false
  subagents:
    enabled: true
    max_parallel: 4
  allowed_tools: null  # null = all configured tools

tools:
  - name: search_knowledge
    type: vectorstore
    source: data/knowledge_base/
    description: "Search the knowledge base"
    database:
      provider: chromadb
      connection_string: http://localhost:8000

  - name: brave_search
    type: mcp
    description: "Web search via Brave"
    command: npx
    args: ["-y", "@brave/brave-search-mcp-server"]
```

**Authentication Methods:**

| `auth_provider` | Credential Source                          |
| --------------- | ------------------------------------------ |
| `api_key`       | `ANTHROPIC_API_KEY` env var (default)      |
| `oauth_token`   | `CLAUDE_CODE_OAUTH_TOKEN` env var          |
| `bedrock`       | AWS Bedrock — sets `CLAUDE_CODE_USE_BEDROCK=1` |
| `vertex`        | Google Vertex — sets `CLAUDE_CODE_USE_VERTEX=1`|
| `foundry`       | Anthropic Foundry — sets `CLAUDE_CODE_USE_FOUNDRY=1` |

**Prerequisites for Claude Backend:**

- Node.js 18+ on PATH (Claude Agent SDK runs as a subprocess)
- `ANTHROPIC_API_KEY` or equivalent credential
- Separate `embedding_provider` when using vectorstore or hierarchical_document tools

---

### Adding a New Backend

To add a new backend (e.g., for a new LLM provider):

1. **Implement the protocols** in `src/holodeck/lib/backends/my_backend.py`:

```python
from holodeck.lib.backends.base import AgentBackend, AgentSession, ExecutionResult

class MySession:
    """Stateful multi-turn session."""

    async def send(self, message: str) -> ExecutionResult:
        # Implement single-turn invocation
        ...

    async def send_streaming(self, message: str):
        # Implement streaming response
        ...

    async def close(self) -> None:
        # Release session resources
        ...

class MyBackend:
    """Backend for MyProvider."""

    async def initialize(self) -> None:
        # Validate config, prepare resources
        ...

    async def invoke_once(self, message, context=None) -> ExecutionResult:
        # Stateless single-turn invocation
        ...

    async def create_session(self) -> MySession:
        # Create new stateful session
        ...

    async def teardown(self) -> None:
        # Release all resources
        ...
```

2. **Register in `BackendSelector`** (`src/holodeck/lib/backends/selector.py`):

```python
elif agent.model.provider == "my_provider":
    return MyBackend(agent, tool_instances, mode, allow_side_effects)
```

3. **Add tests** in `tests/unit/lib/backends/test_my_backend.py`

---

## HoloDeck CLI Usage

### The `test` Command

The `holodeck test` command executes test cases defined in your agent configuration and evaluates responses against specified metrics.

```bash
# Basic usage (uses agent.yaml in current directory)
holodeck test

# Specify a different agent config
holodeck test path/to/agent.yaml

# With options
holodeck test agent.yaml --verbose --output report.md --format markdown
```

**Command Options:**

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `--output PATH`             | Save test report to file (JSON or Markdown)                  |
| `--format [json\|markdown]` | Report format (auto-detects from extension if not specified) |
| `--verbose, -v`             | Enable verbose output with debug information                 |
| `--quiet, -q`               | Suppress progress output (summary still shown)               |
| `--timeout SECONDS`         | LLM execution timeout in seconds                             |
| `--force-ingest, -f`        | Force re-ingestion of all vector store source files          |

**Exit Codes:**

| Code | Meaning                  |
| ---- | ------------------------ |
| 0    | All tests passed         |
| 1    | One or more tests failed |
| 2    | Configuration error      |
| 3    | Execution error          |
| 4    | Evaluation error         |

### Test Case Configuration

Test cases are defined in the `test_cases` section of `agent.yaml`:

```yaml
test_cases:
  - name: "Basic greeting" # Optional: Test identifier
    input: "Hello, how are you?" # Required: User query/prompt
    ground_truth: "Greeting response" # Optional: Expected output for comparison
    expected_tools: # Optional: Tools expected to be called
      - search_knowledge_base
      - get_user_context
    files: # Optional: Multimodal file inputs
      - path: "./data/image.png"
        type: image
        description: "Product image"
    retrieval_context: # Optional: RAG context for RAG metrics
      - "Retrieved chunk 1..."
      - "Retrieved chunk 2..."
    evaluations: # Optional: Per-test metric overrides
      - type: standard
        metric: bleu
        threshold: 0.6
```

**File Input Types:**

| Type         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `image`      | Images (PNG, JPG) - processed via OCR                     |
| `pdf`        | PDF documents                                             |
| `text`       | Plain text files                                          |
| `excel`      | Excel spreadsheets (supports `sheet` and `range` options) |
| `word`       | Word documents                                            |
| `powerpoint` | PowerPoint presentations (supports `pages` option)        |
| `csv`        | CSV files                                                 |

### Evaluation Metrics Configuration

HoloDeck supports three types of evaluation metrics:

#### 1. Standard NLP Metrics (`type: standard`)

Traditional text comparison metrics that don't require an LLM:

```yaml
evaluations:
  metrics:
    # BLEU - Precision-focused n-gram matching (0.0-1.0)
    - type: standard
      metric: bleu
      threshold: 0.5

    # ROUGE - Recall-focused overlap (variants: rouge1, rouge2, rougeL)
    - type: standard
      metric: rouge
      threshold: 0.6

    # METEOR - Translation quality with synonym handling
    - type: standard
      metric: meteor
      threshold: 0.7
```

**Available Standard Metrics:**

| Metric   | Description                                                 | Score Range | Use Case                            |
| -------- | ----------------------------------------------------------- | ----------- | ----------------------------------- |
| `bleu`   | Precision of n-gram matches (uses SacreBLEU with smoothing) | 0.0-1.0     | Machine translation, exact matching |
| `rouge`  | Recall of n-gram overlaps (rouge1, rouge2, rougeL variants) | 0.0-1.0     | Summarization quality               |
| `meteor` | Synonym-aware matching with stemming                        | 0.0-1.0     | Semantic similarity                 |

#### 2. G-Eval Custom Criteria (`type: geval`)

LLM-as-judge evaluation with custom natural language criteria:

```yaml
evaluations:
  model: # Default LLM for all metrics
    provider: ollama
    name: gpt-oss:20b
    temperature: 0.0
  metrics:
    - type: geval
      name: Professionalism # Custom metric name
      criteria: | # Natural language criteria
        Evaluate if the response uses professional language,
        avoids slang, and maintains a respectful tone.
      evaluation_steps: # Optional: Explicit evaluation steps
        - "Check if the language is formal and professional"
        - "Verify no slang or casual expressions are used"
        - "Assess the overall respectful tone"
      evaluation_params: # Fields to include in evaluation
        - actual_output # Agent's response
        - input # User's query
        - expected_output # Ground truth (if provided)
      threshold: 0.7
      strict_mode: false # If true, binary scoring (1.0 or 0.0)
```

**Valid `evaluation_params`:**

- `input` - User's query
- `actual_output` - Agent's response
- `expected_output` - Ground truth reference
- `context` - Additional context
- `retrieval_context` - Retrieved RAG chunks

#### 3. RAG Pipeline Metrics (`type: rag`)

Specialized metrics for evaluating Retrieval-Augmented Generation:

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o
  metrics:
    # Faithfulness - Detects hallucinations
    - type: rag
      metric_type: faithfulness
      threshold: 0.8
      include_reason: true

    # Answer Relevancy - Response relevance to query
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    # Contextual Relevancy - Retrieved chunks relevance
    - type: rag
      metric_type: contextual_relevancy
      threshold: 0.6

    # Contextual Precision - Ranking quality of chunks
    - type: rag
      metric_type: contextual_precision
      threshold: 0.7

    # Contextual Recall - Retrieval completeness
    - type: rag
      metric_type: contextual_recall
      threshold: 0.6
```

**RAG Metric Types:**

| Metric Type            | Description                                                       | Required Fields                                 |
| ---------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| `faithfulness`         | Detects hallucinations by comparing response to retrieval context | `input`, `actual_output`, `retrieval_context`   |
| `answer_relevancy`     | Measures if response addresses the query                          | `input`, `actual_output`                        |
| `contextual_relevancy` | Evaluates if retrieved chunks are relevant to query               | `input`, `retrieval_context`                    |
| `contextual_precision` | Assesses ranking quality of retrieved chunks                      | `input`, `expected_output`, `retrieval_context` |
| `contextual_recall`    | Measures retrieval completeness                                   | `input`, `expected_output`, `retrieval_context` |

### Running Tests

```bash
# Run tests with progress indicator
holodeck test agent.yaml

# Verbose output for debugging
holodeck test agent.yaml -v

# Save detailed report
holodeck test agent.yaml --output results/report.md --format markdown

# Force vector store re-ingestion
holodeck test agent.yaml --force-ingest

# Quiet mode (summary only)
holodeck test agent.yaml -q --output results.json
```

---

## Common Development Tasks

### Adding a New CLI Command

1. Create command file: `src/holodeck/cli/commands/my_command.py`

```python
import click

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def my_command(config_path: str, verbose: bool) -> None:
    """Command description."""
    if verbose:
        click.echo(f"Processing {config_path}...")
    # Implementation...
```

2. Register command in `src/holodeck/cli/main.py`:

```python
from holodeck.cli.commands.my_command import my_command

main.add_command(my_command)
```

3. Add tests: `tests/unit/cli/commands/test_my_command.py`

### Adding a New Evaluation Metric

1. Implement evaluator in `src/holodeck/lib/evaluators/my_metric.py`:

```python
from holodeck.lib.evaluators.base import BaseEvaluator

class MyMetricEvaluator(BaseEvaluator):
    """Custom evaluation metric."""

    async def evaluate(
        self, *, input: str, actual_output: str, **kwargs
    ) -> dict[str, Any]:
        """Evaluate the metric."""
        score = self._compute_score(actual_output)
        return {
            "metric": "my_metric",
            "score": score,
            "threshold": self.threshold,
            "passed": score >= self.threshold,
        }

    def _compute_score(self, text: str) -> float:
        """Compute metric score."""
        # Implementation...
        return 0.0
```

2. Export in `src/holodeck/lib/evaluators/__init__.py`:

```python
from holodeck.lib.evaluators.my_metric import MyMetricEvaluator

__all__ = [..., "MyMetricEvaluator"]
```

3. Add to evaluation config in `src/holodeck/models/evaluation.py`:

```python
# Add to MetricType enum
class MetricType(str, Enum):
    # ... existing metrics
    MY_METRIC = "my_metric"
```

4. Add tests: `tests/unit/lib/evaluators/test_my_metric.py`

### Adding a New Tool Type

1. Define tool config model in `src/holodeck/models/tool.py`:

```python
class MyToolConfig(BaseModel):
    """Configuration for my custom tool."""
    type: Literal["my_tool"] = "my_tool"
    name: str
    config: dict[str, Any]

# Add to ToolUnion
ToolUnion = Annotated[
    VectorStoreConfig
    | FunctionConfig
    | MCPConfig
    | PromptConfig
    | MyToolConfig,  # Add here
    Field(discriminator="type"),
]
```

2. Implement tool factory in `src/holodeck/tools/my_tool.py`:

```python
from semantic_kernel.functions import KernelFunction
from holodeck.models.tool import MyToolConfig

def create_my_tool(config: MyToolConfig) -> KernelFunction:
    """Create tool from configuration."""
    # Implementation...
    return tool
```

3. Add tests: `tests/unit/tools/test_my_tool.py`

### Adding a New Template

1. Create template directory: `src/holodeck/templates/my_template/`

```
my_template/
├── manifest.yaml          # Template metadata
├── agent.yaml             # Agent configuration
├── instructions/
│   └── system-prompt.md   # System instructions
├── tools/
│   └── custom_tool.py     # Custom tools (optional)
├── tests/
│   └── test-cases.yaml    # Test scenarios
└── data/                  # Sample data (optional)
```

2. Create manifest: `src/holodeck/templates/my_template/manifest.yaml`

```yaml
name: "my_template"
display_name: "My Template"
description: "Description of what this template provides"
version: "1.0.0"
author: "Your Name"
tags:
  - "tag1"
  - "tag2"
```

3. Register in template list (template discovery is automatic via directory scanning)

---

## Do's and Don'ts

### DO's

1. **DO use Pydantic models for all configuration**
   - Leverage Pydantic's validation and serialization
   - Use Field() with descriptions for better error messages

2. **DO use async/await for I/O operations**
   - Enables concurrent execution for better performance
   - Required by Semantic Kernel agent framework

3. **DO use type hints everywhere**
   - Enforced by MyPy in strict mode
   - Improves code quality and IDE support

4. **DO handle errors with custom exceptions**
   - Use the exception hierarchy in `lib/errors.py`
   - Provide context in error messages

5. **DO write comprehensive tests**
   - Unit tests for isolated logic
   - Integration tests for cross-component interactions
   - Aim for 80%+ code coverage

6. **DO follow the DRY principle**
   - Extract common logic into utilities
   - Avoid code duplication

7. **DO use structured logging**
   - Include context in log messages
   - Use appropriate log levels

8. **DO validate environment variables**
   - Check for required env vars at startup
   - Provide clear error messages if missing

9. **DO use MCP for API integrations**
   - Design decision: External APIs should use MCP, not custom types

10. **DO use `BackendSelector` for provider routing**
    - Never instantiate backends directly
    - Use protocol types (`AgentBackend`, `AgentSession`, `ExecutionResult`) in function signatures

11. **DO run code quality checks before committing**
    - `make format`, `make lint`, `make type-check`
    - Pre-commit hooks enforce this automatically

### DON'Ts

1. **DON'T use print() for output**
   - Use Click's echo() in CLI commands
   - Use logging for debug/info messages

2. **DON'T hardcode configuration values**
   - Use environment variables for secrets
   - Use YAML configuration for settings

3. **DON'T ignore type checking errors**
   - Fix MyPy errors, don't use `# type: ignore` unless absolutely necessary
   - Document why type ignore is needed if used

4. **DON'T write synchronous I/O in async functions**
   - Use `aiofiles` for file I/O
   - Use async HTTP clients

5. **DON'T catch broad exceptions without re-raising**
   - Catch specific exceptions when possible
   - Re-raise or convert to custom exceptions with context

6. **DON'T commit without running tests**
   - Pre-commit hooks run automatically
   - Run `make test` before pushing

7. **DON'T add dependencies without justification**
   - Evaluate if functionality can be achieved with existing deps
   - Document why dependency is needed in PR

8. **DON'T use mutable default arguments**

   ```python
   # BAD
   def my_func(items: list = []):
       pass

   # GOOD
   def my_func(items: list | None = None):
       if items is None:
           items = []
   ```

9. **DON'T skip docstrings**
   - All public functions, classes, and modules need docstrings
   - Follow Google docstring style

10. **DON'T mix testing and implementation code**
    - Keep tests in `tests/` directory
    - Use fixtures in `tests/fixtures/`

---

## Git Commit Guidelines

When generating commit messages:

- **Do NOT attribute Claude Code** in commit messages
- Do NOT include "Generated with Claude Code" or similar attributions
- Write clean, conventional commit messages focused on the changes

**Commit Message Format:**

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Formatting, missing semi colons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

---

## Workflow for New Features

HoloDeck uses **spec-kit** for feature development. Follow this workflow:

### 1. Create a Specification

```bash
# Create spec for new feature
/speckit.specify
```

This creates: `specs/<spec_name>/spec.md`

### 2. Clarify the Specification

```bash
# Clarify requirements and edge cases
/speckit.clarify
```

Updates: `specs/<spec_name>/spec.md` with clarifications

### 3. Create a Plan

```bash
# Create implementation plan
/speckit.plan
```

This creates:

- `specs/<spec_name>/plan.md`: High-level implementation plan
- `specs/<spec_name>/data-model.md`: Data structures (if needed)
- `specs/<spec_name>/contracts/*.md`: API contracts (if needed)

### 4. Create Tasks

```bash
# Break down plan into tasks
/speckit.tasks
```

This creates: `specs/<spec_name>/tasks.md`

### 5. Plan Todo List

Before implementing:

1. Read ALL files in `specs/<spec_name>/`:
   - spec.md
   - plan.md
   - tasks.md
   - data-model.md
   - research.md (if exists)
   - quickstart.md (if exists)
   - contracts/\*.md (if exist)

2. Create a detailed todo list for the specific task(s) you're working on

3. Use the TodoWrite tool to track progress (if multiple tasks)

### 6. Implement

Follow the todo list and implement the feature.

### 7. Run Code Quality Checks

After completing each task:

```bash
make format             # Format with Black + Ruff
make format-check       # Check formatting (CI-safe)
make lint               # Run Ruff + Bandit
make lint-fix           # Auto-fix linting issues
make type-check         # MyPy type checking
make security           # Safety + Bandit + detect-secrets
make test               # Run all tests
```

### 8. Commit

```bash
git add .
git commit -m "feat: <description>"
```

---

## Troubleshooting

### Common Issues

**1. Import errors after adding new dependencies**

```bash
# Sync dependencies
uv sync --all-extras

# Verify installation
uv run python -c "import <package>"
```

**2. MyPy type checking errors**

```bash
# Check specific file
uv run mypy src/holodeck/my_file.py

# Common fixes:
# - Add type hints to function parameters and return values
# - Import types from typing module
# - Use Optional[T] for nullable values
# - Add "-> None" for functions without return value
```

**3. Test failures**

```bash
# Run specific test with verbose output
uv run pytest tests/unit/test_file.py::test_function -vv

# Check test output and stack trace
# Common fixes:
# - Update test fixtures
# - Mock external dependencies
# - Check for async/await issues
```

**4. Pre-commit hook failures**

```bash
# Run hooks manually to see errors
uv run pre-commit run --all-files

# Common fixes:
# - Format code: make format
# - Fix linting: make lint-fix
# - Fix type errors: make type-check
```

**5. UV dependency resolution conflicts**

```bash
# Clear cache and re-sync
rm -rf .venv uv.lock
uv sync --all-extras

# Check for incompatible version constraints in pyproject.toml
```

---

## Key Design Constraints

1. **No-Code First**: Users configure agents via YAML, not Python
2. **Claude Code First-Class Citizen**: Native backend via Claude Agent SDK; future development prioritizes Claude-native capabilities (hooks, tools, subagents)
3. **Protocol-Driven Backends**: All backends implement `AgentBackend`/`AgentSession` protocols; consumers never depend on provider-specific types
4. **MCP for APIs**: External API integrations must use MCP servers, not custom API tool types
5. **OpenTelemetry Native**: Observability follows GenAI semantic conventions
6. **Evaluation Flexibility**: Support model configuration at global, run, and metric levels
7. **Multimodal Testing**: First-class support for images, PDFs, Office docs

---

## Additional Resources

### Documentation

- **VISION.md**: Product vision and feature specifications
- **README.md**: User-facing documentation and quick start
- **CLAUDE.md**: AI agent instructions (this file's sibling)
- **specs/**: Feature specifications (spec-kit format)

### External References

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/claude-code-sdk-docs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Azure AI Evaluation](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### Key Dependencies

- **UV**: https://astral.sh/uv (package manager)
- **Semantic Kernel**: https://github.com/microsoft/semantic-kernel
- **Claude Agent SDK**: https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/claude-code-sdk-docs
- **Pydantic**: https://docs.pydantic.dev/
- **Click**: https://click.palletsprojects.com/
- **Pytest**: https://docs.pytest.org/
- **Black**: https://black.readthedocs.io/
- **Ruff**: https://docs.astral.sh/ruff/
- **MyPy**: https://mypy.readthedocs.io/

---

## Summary

This AGENTS.md file provides comprehensive documentation for AI agents working on HoloDeck:

1. **Project Overview**: Understanding the vision and current state
2. **Architecture**: System design and key patterns
3. **Directory Structure**: Where to find code and how it's organized
4. **Development Setup**: Getting started with UV and dependencies
5. **Code Quality Standards**: Python style, formatting, linting, type checking
6. **Key Patterns**: Configuration loading, tool loading, evaluations, async/await
7. **Agent Configuration Schema**: Complete YAML reference
8. **HoloDeck CLI Usage**: Test command, options, exit codes, evaluation metrics
9. **Common Development Tasks**: Adding commands, metrics, tools, templates
10. **Do's and Don'ts**: Best practices and anti-patterns
11. **Git Commit Guidelines**: Conventional commit format and types
12. **Workflow**: spec-kit process for new features
13. **Key Design Constraints**: Core architectural principles
14. **Troubleshooting**: Common issues and solutions

When working on HoloDeck, always:

- Follow the spec-kit workflow for new features
- Run code quality checks before committing
- Write comprehensive tests
- Use type hints and docstrings
- Follow the DRY principle
- Consult VISION.md for feature specifications

**Remember:** HoloDeck is about enabling no-code agent development. Every feature should be configurable through YAML without requiring Python code.

## Active Technologies
- Claude Agent SDK (Node.js subprocess, native Anthropic backend)
- Semantic Kernel (Python, OpenAI/Azure/Ollama backend)
- Node.js 18+ (required for Claude Agent SDK subprocess)

## Recent Changes
- 021-claude-agent-sdk: Multi-backend abstraction layer with Claude Agent SDK as first-class citizen. Added `lib/backends/` package (8 modules), `models/claude_config.py`, `tool_initializer.py`, `instruction_resolver.py`, `claude_context_generator.py`. Protocol-driven architecture with `AgentBackend`/`AgentSession`/`ContextGenerator` interfaces, auto-routing via `BackendSelector`, tool adapters, MCP/OTel bridges, startup validators.
- 019-deploy-command: Added Python 3.10+
