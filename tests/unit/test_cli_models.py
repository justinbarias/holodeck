"""Tests for CLI-specific Pydantic models.

Tests verify that ProjectInitInput, ProjectInitResult, and TemplateManifest
models validate correctly and enforce constraints.
"""

from typing import Any

import pytest
from pydantic import ValidationError


@pytest.mark.unit
def test_project_init_input_model_exists() -> None:
    """Test that ProjectInitInput model can be imported."""
    from holodeck.cli.models import ProjectInitInput

    assert ProjectInitInput is not None


@pytest.mark.unit
def test_project_init_result_model_exists() -> None:
    """Test that ProjectInitResult model can be imported."""
    from holodeck.cli.models import ProjectInitResult

    assert ProjectInitResult is not None


@pytest.mark.unit
def test_template_manifest_model_exists() -> None:
    """Test that TemplateManifest model can be imported."""
    from holodeck.cli.models import TemplateManifest

    assert TemplateManifest is not None


@pytest.mark.unit
def test_project_init_input_valid_creation() -> None:
    """Test that ProjectInitInput can be created with valid data."""
    from holodeck.cli.models import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "conversational",
    }
    model = ProjectInitInput(**data)

    assert model.project_name == "test-project"
    assert model.template == "conversational"


@pytest.mark.unit
def test_project_init_input_missing_project_name() -> None:
    """Test that ProjectInitInput requires project_name."""
    from holodeck.cli.models import ProjectInitInput

    data = {
        "template": "conversational",
    }
    with pytest.raises(ValidationError):
        ProjectInitInput(**data)


@pytest.mark.unit
def test_project_init_input_missing_template() -> None:
    """Test that ProjectInitInput requires template."""
    from holodeck.cli.models import ProjectInitInput

    data = {
        "project_name": "test-project",
    }
    with pytest.raises(ValidationError):
        ProjectInitInput(**data)


@pytest.mark.unit
def test_project_init_result_valid_creation() -> None:
    """Test that ProjectInitResult can be created with valid data."""
    from holodeck.cli.models import ProjectInitResult

    data = {
        "success": True,
        "project_name": "test-project",
        "project_path": "/path/to/test-project",
        "template_used": "conversational",
        "files_created": ["agent.yaml", "instructions/system-prompt.md"],
        "warnings": [],
        "errors": [],
        "duration_seconds": 2.5,
    }
    result = ProjectInitResult(**data)

    assert result.success is True
    assert result.project_name == "test-project"
    assert len(result.files_created) == 2


@pytest.mark.unit
def test_template_manifest_valid_creation() -> None:
    """Test that TemplateManifest can be created with valid data."""
    from holodeck.cli.models import TemplateManifest

    data: dict[str, Any] = {
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
        "defaults": {"model_provider": "openai"},
        "files": {
            "agent.yaml": {
                "path": "agent.yaml",
                "template": True,
                "required": True,
            }
        },
    }
    manifest = TemplateManifest(**data)

    assert manifest.name == "conversational"
    assert manifest.display_name == "Conversational Agent"


@pytest.mark.unit
def test_project_init_input_optional_fields() -> None:
    """Test that ProjectInitInput allows optional fields."""
    from holodeck.cli.models import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "conversational",
        "description": "Test project",
        "author": "Test Author",
    }
    model = ProjectInitInput(**data)

    assert model.description == "Test project"
    assert model.author == "Test Author"
