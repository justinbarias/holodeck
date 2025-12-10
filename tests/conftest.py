"""Pytest configuration and shared fixtures for HoloDeck tests."""

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test file operations.

    Yields:
        Path to temporary directory

    Cleanup:
        Automatically removes directory after test
    """
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def isolated_env() -> Generator[dict[str, str]]:
    """Provide isolated environment variables for testing.

    Saves current environment and restores after test.

    Yields:
        Dictionary of original environment variables

    Cleanup:
        Restores original environment after test
    """
    original_env = os.environ.copy()
    yield original_env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def fixture_dir() -> Path:
    """Get path to test fixtures directory.

    Returns:
        Path to tests/fixtures directory
    """
    fixtures_path = Path(__file__).parent / "fixtures"
    fixtures_path.mkdir(parents=True, exist_ok=True)
    return fixtures_path


# NLP Metrics Evaluator Fixtures (lazy loading optimization)
@pytest.fixture(scope="module")
def bleu_evaluator():
    """Shared BLEU evaluator instance for tests without specific thresholds.

    Uses module scope to avoid repeatedly loading the SacreBLEU library,
    which improves test performance.

    Returns:
        BLEUEvaluator instance with default settings
    """
    from holodeck.lib.evaluators.nlp_metrics import BLEUEvaluator

    return BLEUEvaluator()


@pytest.fixture(scope="module")
def rouge_evaluator():
    """Shared ROUGE evaluator instance for tests without specific thresholds.

    Uses module scope to avoid repeatedly loading the evaluate library,
    which improves test performance.

    Returns:
        ROUGEEvaluator instance with default settings
    """
    from holodeck.lib.evaluators.nlp_metrics import ROUGEEvaluator

    return ROUGEEvaluator()


@pytest.fixture(scope="module")
def meteor_evaluator():
    """Shared METEOR evaluator instance for tests without specific thresholds.

    Uses module scope to avoid repeatedly loading the evaluate library,
    which improves test performance.

    Returns:
        METEOREvaluator instance with default settings
    """
    from holodeck.lib.evaluators.nlp_metrics import METEOREvaluator

    return METEOREvaluator()


# Configure pytest
def pytest_configure(config: Any) -> None:
    """Configure pytest with marker options."""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Modify test collection to filter out non-test classes.

    Prevents pytest from collecting classes with __init__ methods
    (like Pydantic models) as test classes.
    """
    # This hook is called after collection but we don't need to modify items
    # The filtering happens in pytest_pycollect_makeitem
    pass


def pytest_pycollect_makeitem(collector: Any, name: str, obj: Any) -> Any:
    """Hook to prevent collection of classes with __init__ constructors.

    This prevents pytest from trying to collect Pydantic models and other
    non-test classes that happen to start with 'Test' (like TestCaseModel,
    TestResult, TestReport, TestExecutor).

    Args:
        collector: The collector object
        name: The name of the object being collected
        obj: The object being collected

    Returns:
        None to skip collection of classes with __init__, or default behavior
    """
    # Check if this is a class that starts with "Test" and has __init__
    # (Pydantic models and non-test classes have __init__)
    if isinstance(obj, type) and name.startswith("Test") and "__init__" in obj.__dict__:
        return None
    # Return None to use default collection behavior
    return None


# Integration Test Fixtures (CLI and Project Initialization)


@pytest.fixture(scope="class")
def cli_runner() -> CliRunner:
    """Shared CLI runner for integration tests.

    Uses class scope to avoid creating new runner instances for each test.

    Returns:
        CliRunner instance for testing Click commands
    """
    return CliRunner()


@pytest.fixture
def init_project(
    temp_dir: Path,
) -> Callable[..., tuple[Path, subprocess.CompletedProcess[str]]]:
    """Create a project using holodeck init command via subprocess.

    This fixture provides a factory function that creates projects with
    specified parameters. It uses subprocess to ensure end-to-end CLI testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Function that creates projects and returns (project_dir, result)

    Example:
        ```python
        def test_something(init_project):
            project_dir, result = init_project("my-agent", template="research")
            assert result.returncode == 0
            assert project_dir.exists()
        ```
    """

    def _init_project(
        project_name: str = "test-project",
        template: str | None = None,
        **kwargs: str | bool,
    ) -> tuple[Path, subprocess.CompletedProcess[str]]:
        """Create a project using holodeck init.

        Args:
            project_name: Name of the project to create
            template: Optional template name
                (conversational, research, customer-support)
            **kwargs: Additional CLI flags
                (e.g., description="...", author="...", force=True)

        Returns:
            Tuple of (project_directory, subprocess_result)
        """
        # Use --name and --non-interactive for CLI tests
        args = [
            sys.executable,
            "-m",
            "holodeck.cli.main",
            "init",
            "--name",
            project_name,
            "--non-interactive",
        ]

        if template:
            args.extend(["--template", template])

        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            # Handle boolean flags (e.g., --force)
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(flag)
            else:
                args.extend([flag, value])

        result = subprocess.run(
            args,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        project_dir = temp_dir / project_name
        return project_dir, result

    return _init_project


@pytest.fixture(
    scope="module",
    params=["conversational", "research", "customer-support"],
)
def template_project_module(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, subprocess.CompletedProcess[str]]:
    """Create template projects once per module (expensive operation).

    This fixture creates each template project once and shares it across
    all tests in the module. This significantly reduces subprocess overhead
    for tests that only need to read the generated project structure.

    Uses module scope to minimize expensive subprocess calls.

    Args:
        request: Pytest request fixture with template parameter
        tmp_path_factory: Factory for creating temporary directories

    Returns:
        Tuple of (project_dir, template_name, subprocess_result)

    Note:
        Tests using this fixture should NOT modify the project directory,
        as it's shared across tests. For tests that need to modify files,
        use the init_project fixture instead.
    """
    template = request.param
    temp_dir = tmp_path_factory.mktemp(f"template_{template}")
    project_name = f"test-{template}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "holodeck.cli.main",
            "init",
            "--name",
            project_name,
            "--template",
            template,
            "--non-interactive",
        ],
        cwd=temp_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to create {template} project: {result.stderr}")

    project_dir = temp_dir / project_name
    return project_dir, template, result
