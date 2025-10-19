# HoloDeck

A robust Python tool for collecting and analyzing code metrics with best practices for quality, security, and testing.

## Features

- Modern Python 3.14+ project structure
- Comprehensive testing with pytest
- Code quality enforcement with Black, Ruff, and MyPy
- Pre-commit hooks for automated quality checks
- GitHub Actions CI/CD pipeline
- Security scanning with Bandit and Safety
- VSCode integration with optimized settings
- Comprehensive Makefile with 30+ commands for development workflow

## Getting Started

### Prerequisites

- Python 3.14 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd holodeck
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
make install
# or
pip install -e ".[dev]"
pre-commit install
```

4. Copy the environment template and configure:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

```
holodeck/
├── src/
│   └── holodeck/
│       ├── __init__.py
│       ├── core/
│       ├── utils/
│       └── models/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/
├── scripts/
├── pipelines/
│   └── pipeline.yml
├── .vscode/
│   └── settings.json
├── pyproject.toml
├── .gitignore
├── .env.example
└── Makefile
```

## Development

### Quick Start

```bash
# Initialize project (create venv, install deps, setup hooks)
make init

# Run all tests
make test

# Run unit tests only
make test-unit

# Run tests with full coverage report
make test-coverage

# Format code
make format

# Run linters
make lint

# Run complete CI pipeline locally
make ci
```

## Available Commands

Run `make help` to see all available commands. Key commands include:

### Installation & Setup

- `make init` - Initialize project (create venv, install deps, setup pre-commit)
- `make install` or `make install-dev` - Install development dependencies
- `make install-prod` - Install production dependencies only
- `make install-hooks` - Install pre-commit hooks
- `make update-deps` - Update all dependencies to latest versions

### Managing Dependencies

This project supports adding dependencies via Poetry:

```bash
# Add a production dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Remove a dependency
poetry remove package-name
```

Poetry will automatically update `pyproject.toml` and install the package.

### Testing

- `make test` - Run all tests
- `make test-unit` - Run unit tests only
- `make test-integration` - Run integration tests only
- `make test-coverage` - Run tests with full coverage report
- `make test-failed` - Re-run only failed tests
- `make test-parallel` - Run tests in parallel (requires pytest-xdist)

### Code Quality

- `make format` - Format code with Black and isort
- `make format-check` - Check code formatting without changes
- `make lint` - Run all linters (Ruff, Bandit)
- `make lint-fix` - Auto-fix linting issues
- `make type-check` - Run type checking with MyPy
- `make security` - Run security checks (Safety, Bandit, detect-secrets)
- `make pre-commit` - Run pre-commit hooks on all files

### CI/CD

- `make ci` - Run complete CI pipeline locally
- `make ci-azure` - Run CI checks formatted for Azure DevOps
- `make ci-fast` - Run fast CI checks (no coverage, parallel tests)

### Build & Documentation

- `make build` - Build distribution packages
- `make build-check` - Build and validate package
- `make docs` - Build documentation
- `make docs-serve` - Serve documentation locally

### Cleanup

- `make clean` - Clean temporary files and caches
- `make clean-all` - Deep clean including virtual environment

### Development Helpers

- `make run` - Run the application
- `make shell` - Start Python shell with project context
- `make debug` - Run with debugger
- `make git-clean` - Clean git repository (remove untracked files)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test && make lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
