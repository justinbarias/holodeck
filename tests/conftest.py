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
