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
    from holodeck.models.project_config import ProjectInitInput

    assert ProjectInitInput is not None


@pytest.mark.unit
def test_project_init_result_model_exists() -> None:
    """Test that ProjectInitResult model can be imported."""
    from holodeck.models.project_config import ProjectInitResult

    assert ProjectInitResult is not None


@pytest.mark.unit
def test_template_manifest_model_exists() -> None:
    """Test that TemplateManifest model can be imported."""
    from holodeck.models.template_manifest import TemplateManifest

    assert TemplateManifest is not None


@pytest.mark.unit
def test_project_init_input_valid_creation() -> None:
    """Test that ProjectInitInput can be created with valid data."""
    from holodeck.models.project_config import ProjectInitInput

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
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "template": "conversational",
    }
    with pytest.raises(ValidationError):
        ProjectInitInput(**data)


@pytest.mark.unit
def test_project_init_input_missing_template() -> None:
    """Test that ProjectInitInput requires template."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "test-project",
    }
    with pytest.raises(ValidationError):
        ProjectInitInput(**data)


@pytest.mark.unit
def test_project_init_result_valid_creation() -> None:
    """Test that ProjectInitResult can be created with valid data."""
    from holodeck.models.project_config import ProjectInitResult

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
    from holodeck.models.template_manifest import TemplateManifest

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
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "conversational",
        "description": "Test project",
        "author": "Test Author",
    }
    model = ProjectInitInput(**data)

    assert model.description == "Test project"
    assert model.author == "Test Author"


@pytest.mark.unit
def test_project_init_input_empty_project_name() -> None:
    """Test that ProjectInitInput rejects empty project name."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "",
        "template": "conversational",
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "cannot be empty" in str(exc_info.value).lower()


@pytest.mark.unit
def test_project_init_input_project_name_too_long() -> None:
    """Test that ProjectInitInput rejects overly long project name."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "a" * 65,  # Over 64 character limit
        "template": "conversational",
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "64 characters" in str(exc_info.value)


@pytest.mark.unit
def test_project_init_input_project_name_starts_with_digit() -> None:
    """Test that ProjectInitInput rejects project name starting with digit."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "123project",
        "template": "conversational",
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "cannot start with a digit" in str(exc_info.value)


@pytest.mark.unit
def test_project_init_input_project_name_invalid_chars() -> None:
    """Test that ProjectInitInput rejects project names with invalid characters."""
    from holodeck.models.project_config import ProjectInitInput

    invalid_names = [
        "my project",  # space
        "my@project",  # @
        "my.project",  # dot
        "my/project",  # slash
        "my#project",  # hash
    ]

    for invalid_name in invalid_names:
        data = {
            "project_name": invalid_name,
            "template": "conversational",
        }
        with pytest.raises(ValidationError):
            ProjectInitInput(**data)


@pytest.mark.unit
def test_project_init_input_project_name_valid_special_chars() -> None:
    """Test that ProjectInitInput accepts hyphens and underscores."""
    from holodeck.models.project_config import ProjectInitInput

    valid_names = [
        "my-project",
        "my_project",
        "my-project_1",
        "MyProject",
    ]

    for valid_name in valid_names:
        data = {
            "project_name": valid_name,
            "template": "conversational",
        }
        model = ProjectInitInput(**data)
        assert model.project_name == valid_name


@pytest.mark.unit
def test_project_init_input_invalid_template() -> None:
    """Test that ProjectInitInput rejects unknown template."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "invalid-template",
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "unknown template" in str(exc_info.value).lower()


@pytest.mark.unit
def test_project_init_input_description_too_long() -> None:
    """Test that ProjectInitInput rejects overly long description."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "conversational",
        "description": "x" * 1001,  # Over 1000 character limit
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "1000 characters" in str(exc_info.value)


@pytest.mark.unit
def test_project_init_input_author_too_long() -> None:
    """Test that ProjectInitInput rejects overly long author name."""
    from holodeck.models.project_config import ProjectInitInput

    data = {
        "project_name": "test-project",
        "template": "conversational",
        "author": "x" * 257,  # Over 256 character limit
    }
    with pytest.raises(ValidationError) as exc_info:
        ProjectInitInput(**data)
    assert "256 characters" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_version,reason",
    [
        ("1", "no_dots"),
        ("1.0.0.0", "too_many_dots"),
        ("1.a.0", "non_integer"),
        ("1.0.b", "non_integer"),
        ("a.b.c", "non_integer"),
    ],
    ids=["no_dots", "too_many_dots", "non_int_minor", "non_int_patch", "all_non_int"],
)
def test_template_manifest_invalid_version_formats(
    invalid_version: str, reason: str
) -> None:
    """Test that TemplateManifest rejects various invalid semantic version formats."""
    from holodeck.models.template_manifest import TemplateManifest

    data: dict[str, Any] = {
        "name": "conversational",
        "display_name": "Conversational Agent",
        "description": "AI assistant for conversations",
        "category": "conversational-ai",
        "version": invalid_version,
    }
    with pytest.raises(ValidationError) as exc_info:
        TemplateManifest(**data)

    # Verify error message mentions semver validation
    if reason in ["no_dots", "too_many_dots"]:
        assert "semver" in str(exc_info.value).lower()
