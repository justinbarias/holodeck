# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoloDeck is an open-source experimentation platform for building, testing, and deploying AI agents through YAML configuration. The project is in early development (pre-v0.1) with the CLI and configuration infrastructure now functional.

**Key Principle**: No-code agent definition. Users should define agents, tools, evaluations, and deployments entirely through YAML files without writing code.

## Architecture Vision

The platform is designed around three core engines:

1. **Agent Engine**: Manages LLM interactions, tool execution, memory, and vector stores
2. **Evaluation Framework**: Runs AI-powered metrics (groundedness, relevance) and NLP metrics (F1, BLEU, ROUGE)
3. **Deployment Engine**: Converts agents to production FastAPI endpoints with Docker/cloud deployment

### Current Architecture (Implemented)

The configuration and CLI layer is now functional:

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer (holodeck)                  │
│  ├─ init: Project scaffolding with templates             │
│  ├─ test: Test runner (placeholder)                      │
│  ├─ chat: Interactive chat (placeholder)                 │
│  └─ deploy: Deployment (placeholder)                     │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Configuration Management                    │
│  ├─ ConfigLoader: YAML parsing                          │
│  ├─ ConfigValidator: Schema validation                  │
│  ├─ ConfigMerge: Merge defaults + user config           │
│  └─ EnvLoader: Environment variable substitution        │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 Pydantic Models                          │
│  ├─ AgentConfig: Agent configuration schema              │
│  ├─ LLMConfig: LLM provider settings                     │
│  ├─ ToolConfig: Tool definitions (5 types)               │
│  ├─ EvaluationConfig: Metrics and thresholds            │
│  └─ TestCaseConfig: Test cases with file support        │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Agent Engine (TODO)                    │
│  ├─ LLM Execution                                        │
│  ├─ Tool Execution                                       │
│  ├─ Memory Management                                    │
│  └─ Vector Store Integration                             │
└─────────────────────────────────────────────────────────┘
```

### Tool & Plugin System

HoloDeck supports multiple tool types that extend agent capabilities:

- **Vector Search Tools**: Redis/Postgres-backed semantic search
- **Custom Function Tools**: Python functions loaded from `tools/*.py`
- **MCP (Model Context Protocol) Tools**: Standardized integrations (filesystem, GitHub, databases, custom servers)
- **Prompt-Based Tools**: AI-powered semantic functions defined via templates (inline or file-based)
- **Plugin Packages**: Pre-built plugin collections installed via registry

Critical design decision: API integrations should use MCP, not custom API tool types.

### Evaluation System

Evaluations can specify models at three levels:

- Global default for all metrics
- Per-evaluation-run model configuration
- Per-metric model override (e.g., GPT-4o for critical metrics, GPT-4o-mini for others)

AI-powered metrics follow Azure AI Evaluation patterns. NLP metrics don't require LLM calls.

### Test Cases with Multimodal Support

Test cases support rich file inputs:

- **Images**: JPG, PNG with OCR
- **Documents**: PDF (full or page ranges), Word, PowerPoint (slide selection)
- **Data**: Excel (sheet/range selection), CSV, text files
- **Mixed media**: Multiple files per test case
- **Remote files**: URL-based inputs with caching

Each test can validate `expected_tools` usage and compare against `ground_truth` using evaluation metrics.

### Observability

Native OpenTelemetry integration following [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

- Automatic trace/metric/log instrumentation
- Standard attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`
- Support for Jaeger, Prometheus, Datadog, Honeycomb, LangSmith
- Built-in cost tracking and alerting

## Development Setup

```bash
# Initialize project (creates venv, installs deps, sets up pre-commit)
make init

# Activate virtual environment
source .venv/bin/activate

# Install dependencies manually with Poetry
make install-dev  # Development dependencies (uses poetry install)
make install-prod # Production only (uses poetry install --only main)

# Alternative: Direct Poetry commands
poetry install              # Install all dependencies
poetry install --only main  # Production only
poetry install --only dev   # Dev dependencies only
```

## Common Development Commands

```bash
# Testing
make test                # Run all tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-coverage      # With coverage report (HTML: htmlcov/index.html)
make test-failed        # Re-run failed tests only
make test-parallel      # Parallel execution (requires pytest-xdist)

# Code Quality
make format             # Format with Black + Ruff
make format-check       # Check formatting (CI-safe)
make lint               # Run Ruff + Bandit
make lint-fix           # Auto-fix linting issues
make type-check         # MyPy type checking
make security           # Safety + Bandit + detect-secrets

# Pre-commit
make install-hooks      # Install pre-commit hooks
make pre-commit         # Run hooks on all files

# CI Pipeline
make ci                 # Run complete CI pipeline locally
make ci-github          # CI with GitHub Actions output format

# Cleanup
make clean              # Remove temporary files/caches
make clean-all          # Deep clean including venv
```

## Python Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Key conventions enforced by tooling:

- **Formatting**: Black (88 char line length)
- **Linting**: Ruff (pycodestyle, pyflakes, isort, flake8-bugbear, pyupgrade, pep8-naming, flake8-simplify, flake8-bandit)
- **Type Checking**: MyPy with strict settings
- **Security**: Bandit, Safety, detect-secrets
- **Target**: Python 3.13+

Additional requirements from existing CLAUDE.md:

- Clear, concise docstrings (PEP 257)
- Type hints using `typing` module
- Break down complex functions
- Handle edge cases explicitly
- Algorithm code should include approach explanations

## Project Structure

```
holodeck/
├── src/holodeck/
│   ├── __init__.py        # Package entry point with version and exports
│   ├── cli/               # Command-line interface
│   │   ├── main.py        # CLI entry point (holodeck command)
│   │   ├── commands/      # CLI commands (init, test, chat, deploy)
│   │   │   └── init.py    # Project initialization command
│   │   ├── utils/         # CLI utilities
│   │   │   └── project_init.py  # Project scaffolding logic
│   │   └── exceptions.py  # CLI-specific exceptions
│   ├── config/            # Configuration management
│   │   ├── loader.py      # YAML configuration loader
│   │   ├── schema.py      # Configuration schema definitions
│   │   ├── validator.py   # Configuration validation logic
│   │   ├── merge.py       # Configuration merging (defaults + user)
│   │   ├── env_loader.py  # Environment variable loading
│   │   └── defaults.py    # Default configuration values
│   ├── models/            # Pydantic data models
│   │   ├── config.py      # Base configuration models
│   │   ├── agent.py       # Agent configuration model
│   │   ├── llm.py         # LLM provider models
│   │   ├── tool.py        # Tool configuration models
│   │   ├── evaluation.py  # Evaluation metrics models
│   │   ├── test_case.py   # Test case models
│   │   ├── project_config.py    # Project metadata model
│   │   └── template_manifest.py # Template manifest model
│   ├── lib/               # Core library utilities
│   │   ├── errors.py      # Custom exception hierarchy
│   │   ├── exceptions.py  # Legacy exceptions (to be consolidated)
│   │   └── template_engine.py   # Jinja2 template rendering
│   └── templates/         # Project templates for `holodeck init`
│       ├── __init__.py
│       ├── _static/       # Shared static files
│       ├── conversational/      # Conversational agent template
│       │   ├── agent.yaml
│       │   ├── instructions/
│       │   ├── tools/
│       │   ├── tests/
│       │   └── data/
│       ├── customer-support/    # Customer support template
│       │   └── [same structure]
│       └── research/            # Research assistant template
│           └── [same structure]
├── tests/
│   ├── unit/              # Unit tests (24 test files)
│   ├── integration/       # Integration tests (10 test files)
│   ├── fixtures/          # Test fixtures and sample data
│   └── conftest.py        # Pytest configuration
├── docs/                  # Documentation (MkDocs)
├── specs/                 # Feature specifications (spec-kit)
├── .github/workflows/     # CI/CD pipelines
├── VISION.md              # Product vision and feature specs
├── README.md              # User-facing documentation
├── CLAUDE.md              # This file
└── pyproject.toml         # Project metadata and dependencies
```

## Configuration Files

- `pyproject.toml`: All tool configuration (Black, Ruff, MyPy, Pytest) and dependencies
  - Package name: `holodeck-ai`
  - Python: 3.13+
  - CLI entry point: `holodeck` command
  - Dev dependencies: pytest, black, ruff, mypy, pre-commit, bandit, safety
- `Makefile`: 30+ development workflow commands
- `.pre-commit-config.yaml`: Pre-commit hooks
- `.secrets.baseline`: Detect-secrets baseline for security scanning

## Test Organization

- **Test Structure**:
  - `tests/unit/`: Unit tests (24 test files) - Fast, isolated tests
  - `tests/integration/`: Integration tests (10 test files) - Cross-component tests
  - `tests/fixtures/`: Shared test fixtures and sample data
  - `tests/conftest.py`: Pytest configuration and fixtures
- **Test Markers**:
  - `@pytest.mark.unit`: Unit tests
  - `@pytest.mark.integration`: Integration tests
  - `@pytest.mark.slow`: Slow-running tests
- **Naming Conventions**:
  - Test files: `test_*.py`
  - Test functions: `test_*`
  - Test classes: `Test*`
- **Coverage**: Minimum 80% required
- **Best Practices**:
  - Use `pytest.mark.parametrize` for data-driven tests
  - Mock external dependencies in unit tests
  - Use fixtures for common test setup

## Agent Configuration Schema (Target)

When implementing, agent YAML files will follow this structure:

```yaml
name: string
description: string
model:
  provider: openai|azure_openai|anthropic
  name: string
  temperature: float
  max_tokens: int
instructions:
  file: path # OR
  inline: string
tools: [] # vectorstore|function|mcp|prompt|plugin types
evaluations:
  model: {} # Global eval model
  metrics: [] # Per-metric configuration
test_cases: [] # With multimodal file support
observability:
  opentelemetry: {}
  cost_tracking: {}
```

## Key Design Constraints

1. **No-Code First**: Users configure agents via YAML, not Python
2. **MCP for APIs**: External API integrations must use MCP servers, not custom API tool types
3. **OpenTelemetry Native**: Observability follows GenAI semantic conventions from day one
4. **Evaluation Flexibility**: Support model configuration at global, run, and metric levels
5. **Multimodal Testing**: First-class support for images, PDFs, Office docs in test cases

## Dependencies

### Production Dependencies (Currently Installed)

- **Core**:
  - `pydantic>=2.0.0`: Configuration validation and data models
  - `pyyaml>=6.0.0`: YAML parsing
  - `click>=8.0.0`: CLI framework
  - `python-dotenv>=1.0.0`: Environment variable loading
  - `jinja2>=3.0.0`: Template engine
  - `jsonschema>=4.0.0`: JSON schema validation
- **Build/Documentation**:
  - `mkdocs-material>=9.6.22`: Documentation site
  - `twine>=6.2.0`: Package publishing
- **Utilities**:
  - `python-dateutil>=2.8.0`: Date parsing
  - `requests>=2.32.5`: HTTP client
  - `urllib3>=2.5.0`: HTTP library
  - `cryptography>=46.0.2`: Cryptographic utilities

### Development Dependencies

- **Testing**: pytest, pytest-cov, pytest-asyncio, pytest-mock
- **Code Quality**: black, ruff, mypy, bandit, safety, detect-secrets
- **Tooling**: pre-commit, tox, poetry
- **Type Stubs**: types-PyYAML
- **Documentation**: mkdocstrings[python]

### Planned Dependencies (Not Yet Integrated)

- Semantic Kernel: Agent framework and vector store abstractions
- FastAPI: API deployment
- Azure AI Evaluation: Evaluation metrics
- OpenTelemetry: Observability instrumentation

The project uses **Poetry** for dependency management via `pyproject.toml`. Poetry handles all dependency resolution, virtual environment management, and package building.

## Implementation Status

**Current State**: Early development (v0.1 in progress)

### Phase 1: CLI & Configuration System (In Progress)

- ✅ Vision and architecture defined (VISION.md)
- ✅ Development environment and tooling configured
- ✅ CLI infrastructure (`holodeck` command entry point)
- ✅ `holodeck init` command with template scaffolding
  - Conversational agent template
  - Customer support agent template
  - Research assistant template
- ✅ Configuration system (loader, validator, schema, merge)
- ✅ Pydantic models for all configuration types:
  - Agent configuration (`AgentConfig`)
  - LLM provider models (`LLMConfig`, `OpenAIConfig`, etc.)
  - Tool models (`ToolConfig`, `VectorStoreConfig`, `MCPConfig`, etc.)
  - Evaluation models (`EvaluationConfig`, `MetricConfig`)
  - Test case models (`TestCaseConfig`, `FileInputConfig`)
  - Project metadata (`ProjectConfig`)
- ✅ Environment variable support (`.env` loading)
- ✅ Template engine (Jinja2-based rendering)
- ✅ Custom exception hierarchy (`HoloDeckError`, `ConfigError`, `ValidationError`)

### Phase 2: Core Features (Not Started)

- ⏳ Agent Engine (execution runtime)
  - LLM provider integrations
  - Tool execution framework
  - Memory and context management
  - Vector store integrations
- ⏳ Evaluation Framework
  - AI-powered metrics (groundedness, relevance, coherence)
  - NLP metrics (F1, BLEU, ROUGE, METEOR)
  - Test runner and reporting
- ⏳ Deployment Engine
  - FastAPI endpoint generation
  - Docker containerization
  - Cloud deployment (Azure, AWS, GCP)

### Phase 3: Advanced Features (Not Started)

- ⏳ Multi-agent orchestration patterns
- ⏳ OpenTelemetry instrumentation
- ⏳ Plugin system and registry
- ⏳ Web UI (no-code editor)

When implementing features, refer to VISION.md for detailed specifications and examples.

## Workflow

This project uses spec-kit.

All work should use the following workflow:

- Create a spec /speckit.specify
- Clarify the spec /speckit.clarify
- Create a plan /speckit.plan
- Create the tasks /speckit.tasks
- Implement each task using /speckit.implement

Every time you finish a task, always run the code quality make commands:

```bash
make format             # Format with Black + Ruff
make format-check       # Check formatting (CI-safe)
make lint               # Run Ruff + Bandit
make lint-fix           # Auto-fix linting issues
make type-check         # MyPy type checking
make security           # Safety + Bandit + detect-secrets
```

Always source .venv before running python commands.

When finished, run pre-commit on all changed files.

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

## DRY

Always follow the DRY (dont repeat yourself) principle.
