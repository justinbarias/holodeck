"""Unit tests for template discovery and template engine functionality.

Tests for:
- Template discovery function (T053)
- ProjectInitializer template validation (T054)
- CLI template validation callback (T055)
"""

import tempfile

import click
import pytest
from pydantic import ValidationError as PydanticValidationError

from holodeck.cli.commands.init import validate_template
from holodeck.cli.exceptions import ValidationError
from holodeck.cli.utils.project_init import ProjectInitializer
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.project_config import ProjectInitInput


class TestTemplateDiscovery:
    """Test template discovery function (T053)."""

    def test_list_available_templates_returns_list(self) -> None:
        """Verify list_available_templates returns a list of template names."""
        templates = TemplateRenderer.list_available_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    @pytest.mark.parametrize(
        "template_name",
        [
            pytest.param("conversational", id="conversational"),
            pytest.param("research", id="research"),
            pytest.param("customer-support", id="customer_support"),
        ],
    )
    def test_list_available_templates_includes_template(
        self, template_name: str
    ) -> None:
        """Verify expected templates are discoverable."""
        templates = TemplateRenderer.list_available_templates()
        assert template_name in templates

    def test_list_available_templates_sorted(self) -> None:
        """Verify templates are returned in sorted order."""
        templates = TemplateRenderer.list_available_templates()
        assert templates == sorted(templates)


class TestProjectInitializerTemplateValidation:
    """Test ProjectInitializer template validation with discovery (T054)."""

    def test_project_initializer_uses_discovered_templates(self) -> None:
        """Verify ProjectInitializer uses dynamically discovered templates."""
        initializer = ProjectInitializer()
        assert hasattr(initializer, "available_templates")
        assert isinstance(initializer.available_templates, set)
        assert len(initializer.available_templates) > 0

    @pytest.mark.parametrize(
        "template_name",
        [
            pytest.param("conversational", id="conversational"),
            pytest.param("research", id="research"),
            pytest.param("customer-support", id="customer_support"),
        ],
    )
    def test_project_initializer_accepts_valid_template(
        self, template_name: str
    ) -> None:
        """Verify ProjectInitializer accepts discovered templates."""
        initializer = ProjectInitializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_data = ProjectInitInput(
                project_name="test-project",
                template=template_name,
                description="Test",
                author="",
                output_dir=tmpdir,
                overwrite=False,
            )

            try:
                initializer.validate_inputs(input_data)
            except ValidationError as e:
                # Should succeed or fail on overwrite, not template
                assert "template" not in str(e).lower()

    def test_project_initializer_rejects_invalid_template(self) -> None:
        """Verify ProjectInitInput rejects unknown templates at Pydantic level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PydanticValidationError) as exc_info:
                ProjectInitInput(
                    project_name="test-project",
                    template="invalid-template-xyz",
                    description="Test",
                    author="",
                    output_dir=tmpdir,
                    overwrite=False,
                )

            error_msg = str(exc_info.value).lower()
            assert "unknown template" in error_msg or "valid template" in error_msg

    def test_project_initializer_shows_available_templates_in_error(self) -> None:
        """Verify error message lists available templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PydanticValidationError) as exc_info:
                ProjectInitInput(
                    project_name="test-project",
                    template="invalid",
                    description="Test",
                    author="",
                    output_dir=tmpdir,
                    overwrite=False,
                )

            error_msg = str(exc_info.value)
            assert "conversational" in error_msg
            assert "research" in error_msg
            assert "customer-support" in error_msg


class TestCLITemplateValidation:
    """Test CLI template validation callback (T055)."""

    @pytest.mark.parametrize(
        "template_name",
        [
            pytest.param("conversational", id="conversational"),
            pytest.param("research", id="research"),
            pytest.param("customer-support", id="customer_support"),
        ],
    )
    def test_cli_validate_template_accepts_valid(self, template_name: str) -> None:
        """Verify CLI accepts valid template parameters."""
        result = validate_template(None, None, template_name)
        assert result == template_name

    def test_cli_validate_template_rejects_invalid(self) -> None:
        """Verify CLI rejects invalid template."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_template(None, None, "invalid-template")

        error_msg = str(exc_info.value).lower()
        assert "unknown template" in error_msg or "available" in error_msg

    def test_cli_validate_template_error_lists_options(self) -> None:
        """Verify CLI error message lists available templates."""
        with pytest.raises(click.BadParameter) as exc_info:
            validate_template(None, None, "xyz")

        error_msg = str(exc_info.value)
        assert "conversational" in error_msg
        assert "research" in error_msg
        assert "customer-support" in error_msg
