# Contributing to HoloDeck

Thank you for your interest in contributing to HoloDeck! This guide will help you get started with development, testing, and submitting your changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style Guide](#code-style-guide)
- [Commit Message Format](#commit-message-format)
- [Pull Request Workflow](#pull-request-workflow)
- [Pre-commit Hooks](#pre-commit-hooks)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment manager (venv)

### Initial Setup

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/holodeck.git
cd holodeck
```

2. **Initialize the project**:

```bash
make init
```

This command will:

- Create a Python virtual environment (`.venv`)
- Install all development dependencies
- Set up pre-commit hooks

3. **Activate the virtual environment** (if not already done by `make init`):

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### Manual Setup (if `make init` doesn't work)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
make install-hooks
```

## Running Tests

### Run All Tests

```bash
make test
```

### Run Unit Tests Only

```bash
make test-unit
```

### Run Integration Tests Only

```bash
make test-integration
```

### Run Tests with Coverage Report

```bash
make test-coverage
```

This generates:

- Terminal summary
- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`

**Coverage Requirements**: Minimum 80% coverage on all modules is enforced.

### Run Failed Tests Only

Quickly re-run tests that failed in the last run:

```bash
make test-failed
```

### Run Tests in Parallel

For faster test execution (requires pytest-xdist):

```bash
make test-parallel
```

## Code Style Guide

HoloDeck follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with tooling enforcement:

### Formatting

Code is automatically formatted using **Black** (88 character line length):

```bash
make format
```

### Check Formatting Without Changes

Verify code is properly formatted without modifying files:

```bash
make format-check
```

### Linting

Code quality is enforced using **Ruff**:

```bash
make lint
```

To auto-fix linting issues:

```bash
make lint-fix
```

**Ruff includes**:

- pycodestyle (E, W)
- pyflakes (F)
- isort (I) - import sorting
- flake8-bugbear (B) - bug detection
- pyupgrade (UP) - Python syntax modernization
- pep8-naming (N) - naming convention checks
- flake8-simplify (SIM) - code simplification
- flake8-bandit (S) - security checks

### Type Checking

Type hints are required. Validate with **MyPy**:

```bash
make type-check
```

**Type checking rules**:

- Full type coverage required (disallow untyped defs)
- No `Any` types where avoidable
- Strict optional checking enabled
- Strict equality checking enabled

### Security Checks

Run security scanning before committing:

```bash
make security
```

This includes:

- **Safety**: Known vulnerability detection
- **Bandit**: Security issue scanning
- **detect-secrets**: Hardcoded secret detection

### Code Style Checklist

Before committing, ensure:

- [ ] Code is formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make type-check`
- [ ] Tests pass: `make test-coverage`
- [ ] Security checks pass: `make security`

## Commit Message Format

Follow this format for commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

Must be one of:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code formatting changes (no functional changes)
- **refactor**: Code refactoring without feature changes
- **perf**: Performance improvements
- **test**: Test additions or modifications
- **chore**: Build, dependency, or tooling changes
- **ci**: CI/CD configuration changes

### Scope

Optional. Scope of the change:

- `config`: Configuration loading and validation
- `models`: Pydantic data models
- `cli`: Command-line interface
- `loader`: YAML configuration loader
- `tests`: Test infrastructure
- `docs`: Documentation
- `core`: Core engine functionality

### Subject

- Use imperative mood: "add" not "added" or "adds"
- Don't capitalize first letter
- No period at the end
- Limit to 50 characters

### Body

- Optional. Explain what and why, not how.
- Wrap at 72 characters
- Separate from subject with blank line
- Use bullet points for multiple changes

### Footer

- Optional. Reference issues: `Closes #123`
- Reference related PRs: `Refs #456`

### Examples

```
feat(config): add support for environment variable substitution

Support ${VAR_NAME} pattern in YAML configurations.
Variables are substituted before schema validation.

Closes #42
```

```
fix(models): correct validation for vector_field XOR vector_fields

Previously allowed both vector_field and vector_fields to be
specified simultaneously, violating the constraint.

Fixes #89
```

```
docs: update installation instructions for Python 3.10
```

## Pull Request Workflow

### Before Creating a PR

1. **Create a feature branch** from `main`:

```bash
git checkout -b feat/your-feature-name
```

2. **Implement your changes**:

   - Write tests first (TDD approach)
   - Implement feature
   - Ensure all tests pass

3. **Run full validation**:

```bash
make ci
```

This runs:

- Code formatting checks
- Linting
- Type checking
- Test suite with coverage
- Security scanning

4. **Commit with proper messages**:

```bash
git add .
git commit -m "feat(scope): descriptive message"
```

5. **Push to your fork**:

```bash
git push origin feat/your-feature-name
```

### Creating a PR

1. Visit the repository on GitHub
2. Click "New Pull Request"
3. Select your branch as the source
4. Fill in the PR template:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe how you tested this change

## Checklist

- [ ] Tests pass locally (`make test`)
- [ ] Code formatted (`make format`)
- [ ] Type checking passes (`make type-check`)
- [ ] Coverage ≥80%
- [ ] Updated relevant documentation
```

### PR Requirements

Before a PR can be merged:

✅ All checks must pass:

- Formatting check
- Linting
- Type checking
- Test coverage (≥80%)
- Security scanning

✅ At least one review approval

✅ No merge conflicts

### Code Review Process

- Reviewers will check for:

  - Code quality and style consistency
  - Test coverage
  - Performance implications
  - Documentation accuracy
  - Security considerations

- Address feedback and push updates (do not force-push)
- Re-request review after changes
- We aim to review within 1-2 business days

## Pre-commit Hooks

Pre-commit hooks automatically run before each commit to catch issues early.

### Install Pre-commit Hooks

```bash
make install-hooks
```

### Hooks Included

- **black**: Code formatting
- **ruff**: Linting and import sorting
- **mypy**: Type checking
- **detect-secrets**: Secret detection
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure newline at EOF

### Bypass Pre-commit (Use Carefully!)

```bash
git commit --no-verify
```

Only use this in exceptional circumstances and ensure you run the checks manually.

## Project Architecture

For understanding the codebase structure, see:

- `docs/architecture/` - Architecture documentation
- `docs/api/models.md` - Data model documentation
- `docs/api/config-loader.md` - ConfigLoader API reference

## Common Issues

### Virtual Environment Not Activated

**Problem**: `command not found: pytest`

**Solution**:

```bash
source .venv/bin/activate
```

### Type Checking Fails

**Problem**: MyPy reports type errors

**Solution**:

1. Add type hints to function parameters and returns
2. For unavoidable types: `# type: ignore` (use sparingly)
3. Check `Any` usage - prefer specific types

### Tests Fail After Changes

**Problem**: Tests pass locally but fail in CI

**Solution**:

1. Run full test suite: `make test`
2. Run with coverage: `make test-coverage`
3. Check for system-specific issues (paths, line endings)

### Security Scan Fails

**Problem**: Security checks report issues

**Solution**:

1. Run locally: `make security`
2. Fix hardcoded secrets or security vulnerabilities
3. Update `.secrets.baseline` if necessary (carefully!)

## Development Tips

### Running Specific Tests

```bash
# Run specific test file
pytest tests/unit/test_config_loader.py -v

# Run specific test function
pytest tests/unit/test_config_loader.py::test_load_agent_yaml -v

# Run tests matching pattern
pytest -k "validation" -v
```

### Debugging

Enable verbose output:

```bash
pytest -vv --tb=long tests/unit/test_config_loader.py
```

Use Python debugger:

```bash
pytest --pdb tests/unit/test_config_loader.py
```

### Local Documentation

Build and serve documentation locally:

```bash
mkdocs serve
```

Visit `http://localhost:8000` to view.

## Questions?

- **Issues**: Open a GitHub issue with details
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` for answers
- **Code examples**: See `docs/examples/` for configuration examples

## Thank You

Thank you for contributing to HoloDeck! Your efforts help make this project better for everyone.
