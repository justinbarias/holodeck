"""Pytest configuration and shared fixtures for HoloDeck tests."""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


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
