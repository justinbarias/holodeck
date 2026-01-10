# Project variables
PYTHON := python
PROJECT_NAME := your_package
SRC_DIR := src
TEST_DIR := tests
MIN_COVERAGE := 80

# Color output using tput (more portable)
RED := $(shell tput setaf 1 2>/dev/null)
GREEN := $(shell tput setaf 2 2>/dev/null)
YELLOW := $(shell tput setaf 3 2>/dev/null)
NC := $(shell tput sgr0 2>/dev/null)

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help install install-dev install-prod test test-unit test-integration \
        test-coverage test-parallel test-integration-parallel test-unit-parallel \
        lint format type-check security clean clean-all \
        pre-commit ci ci-github build docs run docker-build docker-run \
        check-python check-uv init

#################################
# Help and Verification Targets #
#################################

help: ## Show this help message
	@echo '$(GREEN)Available targets:$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ''
	@echo '$(GREEN)CI Pipeline Targets:$(NC)'
	@echo '  $(YELLOW)ci$(NC)                   Run all CI checks (for local testing)'
	@echo '  $(YELLOW)ci-github$(NC)             Run CI checks formatted for Github'

check-python: ## Verify Python installation
	@which $(PYTHON) > /dev/null || (echo "$(RED)Python not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python found: $$($(PYTHON) --version)$(NC)"

check-uv: ## Verify uv is installed
	@which uv > /dev/null || (echo "$(RED)uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)" && exit 1)
	@echo "$(GREEN)✓ uv found: $$(uv --version)$(NC)"

init: check-python check-uv ## Initialize project (create venv, install deps, setup pre-commit)
	@echo "$(GREEN)Initializing project...$(NC)"
	uv venv
	@echo "$(YELLOW)Virtual environment created. Activate with: source .venv/bin/activate$(NC)"
	$(MAKE) install-dev
	$(MAKE) install-hooks

#############################
# Installation Targets      #
#############################

install: install-dev ## Install all dependencies (alias for install-dev)

install-dev: check-uv ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	uv sync --all-extras
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

install-prod: check-uv ## Install production dependencies only
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	uv sync --no-dev
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-hooks: ## Install pre-commit hooks
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	uv run pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

update-deps: ## Update all dependencies to latest versions
	@echo "$(GREEN)Updating dependencies...$(NC)"
	uv lock --upgrade
	uv sync --all-extras
	uv run pre-commit autoupdate
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

#############################
# Testing Targets           #
#############################

test: ## Run all tests with coverage
	@echo "$(GREEN)Running all tests...$(NC)"
	uv run pytest -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	uv run pytest tests/unit -v --tb=short

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	uv run pytest tests/integration -v --tb=short

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	uv run pytest \
		--cov=$(SRC_DIR) \
		--cov-branch \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		-v
	@echo "$(GREEN)✓ Coverage report generated (HTML: htmlcov/index.html)$(NC)"

test-failed: ## Re-run only failed tests
	uv run pytest --lf -v

test-watch: ## Run tests in watch mode (requires pytest-watch)
	uv run pytest_watch -- -v

test-parallel: ## Run all tests in parallel (requires pytest-xdist)
	@echo "$(GREEN)Running all tests in parallel...$(NC)"
	uv run pytest -n auto -v

test-integration-parallel: ## Run integration tests in parallel
	@echo "$(GREEN)Running integration tests in parallel...$(NC)"
	uv run pytest tests/integration -n auto -v

test-unit-parallel: ## Run unit tests in parallel
	@echo "$(GREEN)Running unit tests in parallel...$(NC)"
	uv run pytest tests/unit -n auto -v

#############################
# Code Quality Targets      #
#############################

lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	@echo "Running ruff (includes security checks)..."
	uv run ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-fix: ## Run linters with auto-fix
	@echo "$(GREEN)Auto-fixing linting issues...$(NC)"
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Linting issues fixed$(NC)"

format: ## Format code with black and ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	uv run black $(SRC_DIR) $(TEST_DIR)
	uv run ruff check --select I --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(GREEN)Checking code format...$(NC)"
	uv run black --check --diff $(SRC_DIR) $(TEST_DIR)
	uv run ruff check --select I $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Format check complete$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	uv run mypy $(SRC_DIR) --ignore-missing-imports --no-error-summary --install-types --non-interactive
	@echo "$(GREEN)✓ Type checking complete$(NC)"

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	@echo "Checking for known vulnerabilities..."
	uv run safety check --json
	@echo "Scanning for security issues with Ruff..."
	uv run ruff check $(SRC_DIR) --select S
	@echo "Scanning for security issues with Bandit..."
	uv run bandit -r $(SRC_DIR)
	@echo "Checking for hardcoded secrets..."
	uv run detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)✓ Security checks complete$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	uv run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks complete$(NC)"

#############################
# CI Pipeline Targets       #
#############################

ci: ## Run all CI checks (for local testing)
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)       Running CI Pipeline Locally     $(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@$(MAKE) clean
	@echo "\n$(YELLOW)→ Step 1/7: Checking environment...$(NC)"
	@$(MAKE) check-python
	@echo "\n$(YELLOW)→ Step 2/7: Installing dependencies...$(NC)"
	@$(MAKE) install-dev
	@echo "\n$(YELLOW)→ Step 3/7: Checking code format...$(NC)"
	@$(MAKE) format-check
	@echo "\n$(YELLOW)→ Step 4/7: Running linters...$(NC)"
	@$(MAKE) lint
	@echo "\n$(YELLOW)→ Step 5/7: Type checking...$(NC)"
	@$(MAKE) type-check
	@echo "\n$(YELLOW)→ Step 6/7: Running tests with coverage...$(NC)"
	@$(MAKE) test-coverage
	@echo "\n$(YELLOW)→ Step 7/7: Security scanning...$(NC)"
	@$(MAKE) security
	@echo "\n$(GREEN)========================================$(NC)"
	@echo "$(GREEN)    ✓ All CI checks passed!            $(NC)"
	@echo "$(GREEN)========================================$(NC)"

ci-github: ## Run CI checks with GitHub Actions formatted output
	@echo "::group::🔧 Environment Setup"
	@$(MAKE) check-python
	@echo "::endgroup::"
	@echo "::group::📦 Installing Dependencies"
	@$(MAKE) install-dev
	@echo "::endgroup::"
	@echo "::group::🎨 Code Formatting Check"
	@$(MAKE) format-check || (echo "::error title=Format Check Failed::Code formatting check failed. Run 'make format' locally to fix." && exit 1)
	@echo "::endgroup::"
	@echo "::group::🔍 Linting"
	@$(MAKE) lint || (echo "::error title=Linting Failed::Linting checks failed. Run 'make lint-fix' locally to auto-fix issues." && exit 1)
	@echo "::endgroup::"
	@echo "::group::📝 Type Checking"
	@$(MAKE) type-check || echo "::warning title=Type Check Warning::Type checking failed. Review mypy output above."
	@echo "::endgroup::"
	@echo "::group::🧪 Running Tests with Coverage"
	@uv run pytest \
		--cov=$(SRC_DIR) \
		--cov-branch \
		--cov-report=term \
		--cov-report=xml:coverage.xml \
		--cov-report=html:htmlcov \
		--cov-report=json:coverage.json \
		--cov-fail-under=$(MIN_COVERAGE) \
		--junitxml=test-results.xml \
		-v || (echo "::error title=Tests Failed::Tests or coverage requirements failed." && exit 1)
	@echo "COVERAGE=$$(uv run python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered'])")" >> $$GITHUB_OUTPUT 2>/dev/null || true
	@echo "::notice title=Coverage Report::Coverage is $$(uv run python -c "import json; print(f\"{json.load(open('coverage.json'))['totals']['percent_covered']:.2f}%\")" 2>/dev/null || echo "unknown")"
	@echo "::endgroup::"
	@echo "::group::🔒 Security Scanning"
	@$(MAKE) security || echo "::warning title=Security Scan::Potential security issues found. Review the security report."
	@echo "::endgroup::"
	@echo "::notice title=✅ CI Complete::All CI checks completed successfully!"

ci-github-annotations: ## Generate GitHub annotations from test/lint results
	@echo "::group::📊 Generating GitHub Annotations"
	@# Parse pytest results for failures
	@if [ -f test-results.xml ]; then \
		uv run python -c "import xml.etree.ElementTree as ET; \
		tree = ET.parse('test-results.xml'); \
		root = tree.getroot(); \
		failures = root.findall('.//failure'); \
		[print(f\"::error file={tc.get('classname').replace('.', '/')}.py,line=1,title=Test Failed::{tc.get('name')} failed\") for tc in failures]" 2>/dev/null || true; \
	fi
	@# Parse ruff output for issues
	@uv run ruff check $(SRC_DIR) $(TEST_DIR) --output-format=json 2>/dev/null | \
		uv run python -c "import sys, json; \
		issues = json.load(sys.stdin) if sys.stdin.isatty() == False else []; \
		[print(f\"::warning file={i['filename']},line={i['location']['row']},col={i['location']['column']},title={i['code']}::{i['message']}\") for i in issues]" 2>/dev/null || true
	@# Parse mypy output for type errors
	@uv run mypy $(SRC_DIR) --ignore-missing-imports --no-error-summary --install-types --non-interactive 2>&1 | \
		grep -E "^[^:]+:[0-9]+:" | \
		while read line; do \
			file=$$(echo "$$line" | cut -d: -f1); \
			lineno=$$(echo "$$line" | cut -d: -f2); \
			msg=$$(echo "$$line" | cut -d: -f3-); \
			echo "::warning file=$$file,line=$$lineno,title=Type Error::$$msg"; \
		done || true
	@echo "::endgroup::"

ci-github-matrix: ## Run tests for specific Python version (use with PYTHON_VERSION env var)
	@echo "::notice title=Python Version::Running tests with Python $${PYTHON_VERSION:-$(PYTHON)}"
	@echo "::group::🐍 Python $${PYTHON_VERSION:-$(PYTHON)} Tests"
	@$(MAKE) test-coverage
	@echo "::endgroup::"

ci-github-fast: ## Fast CI for GitHub Actions (parallel tests, no coverage thresholds)
	@echo "::group::⚡ Fast CI Mode"
	@echo "::notice::Running in fast mode - no coverage requirements"
	@$(MAKE) format-check || echo "::warning::Format check failed"
	@$(MAKE) lint || echo "::warning::Linting failed"
	@uv run pytest -n auto --tb=short -q
	@echo "::endgroup::"
	@echo "::notice title=⚡ Fast CI Complete::Quick checks completed"

ci-github-pr: ## CI specifically for pull requests with PR comments
	@echo "::group::🔀 Pull Request Validation"
	@echo "## 🤖 Automated PR Check Results" > pr-comment.md
	@echo "" >> pr-comment.md
	@echo "### 📊 Code Quality" >> pr-comment.md
	@echo "" >> pr-comment.md
	@# Format check
	@echo -n "- **Format Check:** " >> pr-comment.md
	@($(MAKE) format-check > /dev/null 2>&1 && echo "✅ Passed" >> pr-comment.md) || \
		(echo "❌ Failed - run \`make format\` locally" >> pr-comment.md)
	@# Lint check
	@echo -n "- **Linting:** " >> pr-comment.md
	@($(MAKE) lint > /dev/null 2>&1 && echo "✅ Passed" >> pr-comment.md) || \
		(echo "⚠️ Issues found - run \`make lint-fix\` locally" >> pr-comment.md)
	@# Type check
	@echo -n "- **Type Check:** " >> pr-comment.md
	@($(MAKE) type-check > /dev/null 2>&1 && echo "✅ Passed" >> pr-comment.md) || \
		(echo "⚠️ Type issues found" >> pr-comment.md)
	@echo "" >> pr-comment.md
	@echo "### 🧪 Test Results" >> pr-comment.md
	@echo "" >> pr-comment.md
	@# Run tests and capture results
	@if uv run pytest --cov=$(SRC_DIR) --cov-report=json:coverage.json --junitxml=test-results.xml -q; then \
		echo "✅ All tests passed!" >> pr-comment.md; \
	else \
		echo "❌ Some tests failed - see details below" >> pr-comment.md; \
	fi
	@echo "" >> pr-comment.md
	@# Add coverage badge
	@if [ -f coverage.json ]; then \
		COV=$$(uv run python -c "import json; print(f\"{json.load(open('coverage.json'))['totals']['percent_covered']:.1f}\")"); \
		echo "**Coverage:** $$COV%" >> pr-comment.md; \
		if (( $$(echo "$$COV >= $(MIN_COVERAGE)" | bc -l) )); then \
			echo "![Coverage](https://img.shields.io/badge/coverage-$$COV%25-brightgreen)" >> pr-comment.md; \
		else \
			echo "![Coverage](https://img.shields.io/badge/coverage-$$COV%25-red)" >> pr-comment.md; \
		fi; \
	fi
	@echo "" >> pr-comment.md
	@echo "---" >> pr-comment.md
	@echo "*Run \`make ci\` locally to validate all checks before pushing.*" >> pr-comment.md
	@echo "::set-output name=pr-comment-file::pr-comment.md"
	@echo "::endgroup::"

ci-fast: ## Run fast CI checks (no coverage, parallel tests)
	@echo "$(GREEN)Running fast CI checks...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	uv run pytest -n auto --tb=short
	@echo "$(GREEN)✓ Fast CI complete$(NC)"

#############################
# Build and Release Targets #
#############################

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	uv build
	@echo "$(GREEN)✓ Package built (see dist/ directory)$(NC)"

build-check: build ## Build and check package
	uv run twine check dist/*
	@echo "$(GREEN)✓ Package validation complete$(NC)"

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && $(MAKE) clean && $(MAKE) html
	@echo "$(GREEN)✓ Documentation built (see docs/_build/html/index.html)$(NC)"

docs-serve: ## Serve documentation locally
	cd docs/_build/html && uv run python -m http.server

#############################
# Docker Targets            #
#############################

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	docker run --rm -it $(PROJECT_NAME):latest

docker-test: ## Run tests in Docker
	docker run --rm $(PROJECT_NAME):latest make test

#############################
# Cleanup Targets           #
#############################

clean: ## Clean temporary files and caches
	@echo "$(GREEN)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov/
	rm -rf .tox/ .hypothesis/
	rm -f .coverage.* test-results.xml bandit-report.json
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Deep clean including virtual environment
	@echo "$(RED)Removing virtual environment and all generated files...$(NC)"
	rm -rf .venv/ venv/ env/
	rm -rf dist/ build/
	rm -rf docs/_build/
	rm -rf *.egg-info
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

#############################
# Development Helpers       #
#############################

run: ## Run the application
	uv run python -m $(PROJECT_NAME)

shell: ## Start Python shell with project context
	uv run python -c "from $(PROJECT_NAME) import *; import IPython; IPython.embed()"

debug: ## Run with debugger
	uv run python -m pdb -m $(PROJECT_NAME)

profile: ## Profile the application
	uv run python -m cProfile -o profile.stats -m $(PROJECT_NAME)
	uv run python -m pstats profile.stats

requirements: ## Generate requirements.txt from uv.lock
	uv export --format requirements-txt > requirements.txt

#############################
# Git Helpers               #
#############################

git-clean: ## Clean git repository (remove untracked files)
	git clean -fdx -e .venv -e .env

git-stats: ## Show git statistics
	@echo "$(GREEN)Git Statistics:$(NC)"
	@echo "Contributors: $$(git shortlog -sn | wc -l)"
	@echo "Total commits: $$(git rev-list --all --count)"
	@echo "Files changed: $$(git diff --stat HEAD~1 | tail -1)"
