This file provides guidance to Github Copilot when working with code in this repository.

## Project Overview

HoloDeck is an open-source experimentation platform for building, testing, and deploying AI agents through YAML configuration. The project is currently in early development (pre-v0.1) with no implementation code yet - only vision and architecture documentation.

**Key Principle**: No-code agent definition. Users should define agents, tools, evaluations, and deployments entirely through YAML files without writing code.

## Architecture Vision

The platform is designed around three core engines:

1. **Agent Engine**: Manages LLM interactions, tool execution, memory, and vector stores
2. **Evaluation Framework**: Runs AI-powered metrics (groundedness, relevance) and NLP metrics (F1, BLEU, ROUGE)
3. **Deployment Engine**: Converts agents to production FastAPI endpoints with Docker/cloud deployment

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

# Install dependencies manually
make install-dev  # Development dependencies
make install-prod # Production only
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
- **Target**: Python 3.10+

Additional requirements from existing CLAUDE.md:

- Clear, concise docstrings (PEP 257)
- Type hints using `typing` module
- Break down complex functions
- Handle edge cases explicitly
- Algorithm code should include approach explanations

## Project Structure

```
holodeck/
├── src/holodeck/          # Main package (empty - implementation pending)
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── conftest.py        # Pytest configuration
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── .github/workflows/     # CI/CD pipelines
├── VISION.md              # Product vision and feature specs
├── README.md              # User-facing documentation
├── CLAUDE.md              # This file
└── pyproject.toml         # Project metadata and dependencies
```

## Configuration Files

- `pyproject.toml`: All tool configuration (Black, Ruff, MyPy, Pytest)
- `Makefile`: 30+ development workflow commands
- `.pre-commit-config.yaml`: Pre-commit hooks (if exists)
- `.secrets.baseline`: Detect-secrets baseline

## Test Organization

- Tests marked with `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Test files: `test_*.py`
- Test functions: `test_*`
- Minimum coverage: 80%
- Use `pytest.mark.parametrize` for data-driven tests

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

Core dependencies (see `pyproject.toml`):

- Semantic Kernel: Agent framework and vector store abstractions
- FastAPI: API deployment
- Azure AI Evaluation: Evaluation metrics
- Pydantic: Configuration validation
- OpenTelemetry: Observability instrumentation

The project uses Poetry for dependency management but supports standard pip installation.

## Implementation Status

**Current State**: Pre-implementation (v0.1 roadmap)

- ✅ Vision and architecture defined (VISION.md)
- ✅ Development environment and tooling configured
- ⏳ Core agent engine (not started)
- ⏳ Evaluation framework (not started)
- ⏳ Deployment engine (not started)

When implementing features, refer to VISION.md for detailed specifications and examples.
