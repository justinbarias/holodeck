"""Test fixtures for project initialization tests.

Provides helper functions and fixtures for creating and managing
temporary test projects during integration and unit tests.
"""

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_project_dir(temp_dir: Path) -> Generator[Path]:
    """Create a temporary directory for test project creation.

    This fixture extends the temp_dir fixture from conftest.py
    with project-specific initialization and cleanup.

    Args:
        temp_dir: Base temporary directory from parent conftest

    Yields:
        Path to project directory ready for use

    Cleanup:
        Automatically removes directory after test
    """
    project_dir = temp_dir / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    yield project_dir
    # Cleanup handled by parent temp_dir fixture


@pytest.fixture
def valid_project_config() -> dict:
    """Fixture providing valid project configuration for testing.

    Returns:
        Dictionary with valid ProjectInitInput data
    """
    return {
        "project_name": "test-agent",
        "template": "conversational",
        "description": "Test agent project",
        "author": "Test Developer",
    }


@pytest.fixture
def valid_template_manifest() -> dict:
    """Fixture providing valid template manifest for testing.

    Returns:
        Dictionary with valid TemplateManifest data
    """
    return {
        "name": "conversational",
        "display_name": "Conversational Agent",
        "description": "AI assistant for conversations",
        "category": "conversational-ai",
        "version": "1.0.0",
        "variables": {
            "project_name": {
                "type": "string",
                "description": "Project name",
                "required": True,
            }
        },
        "defaults": {
            "model.provider": "openai",
            "model.temperature": 0.7,
        },
        "files": {
            "agent.yaml": {
                "path": "agent.yaml",
                "template": True,
                "required": True,
            },
            "instructions/system-prompt.md": {
                "path": "instructions/system-prompt.md",
                "template": True,
                "required": True,
            },
        },
    }
