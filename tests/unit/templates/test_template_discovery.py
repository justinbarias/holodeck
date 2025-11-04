"""Unit tests for template discovery and template engine functionality.

Tests for:
- Template discovery function (T053)
- ProjectInitializer template validation (T054)
- CLI template validation callback (T055)
"""

import pytest

from holodeck.cli.exceptions import ValidationError
from holodeck.cli.utils.project_init import ProjectInitializer
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.project_config import ProjectInitInput


class TestTemplateDiscovery:
    """Test template discovery function (T053)."""

    def test_list_available_templates_returns_list(self) -> None:
        """Verify list_available_templates returns a list of template names.

        Test case T053: Template discovery function
        """
        templates = TemplateRenderer.list_available_templates()

        # Should return a list
        assert isinstance(templates, list)

        # Should not be empty
        assert len(templates) > 0

    def test_list_available_templates_includes_conversational(self) -> None:
        """Verify conversational template is discoverable.

        Test case T053: Conversational template in discovery
        """
        templates = TemplateRenderer.list_available_templates()

        assert "conversational" in templates

    def test_list_available_templates_includes_research(self) -> None:
        """Verify research template is discoverable.

        Test case T053: Research template in discovery
        """
        templates = TemplateRenderer.list_available_templates()

        assert "research" in templates

    def test_list_available_templates_includes_customer_support(self) -> None:
        """Verify customer-support template is discoverable.

        Test case T053: Customer-support template in discovery
        """
        templates = TemplateRenderer.list_available_templates()

        assert "customer-support" in templates

    def test_list_available_templates_sorted(self) -> None:
        """Verify templates are returned in sorted order.

        Test case T053: Template discovery returns sorted list
        """
        templates = TemplateRenderer.list_available_templates()

        # Should be sorted
        assert templates == sorted(templates)


class TestProjectInitializerTemplateValidation:
    """Test ProjectInitializer template validation with discovery (T054)."""

    def test_project_initializer_uses_discovered_templates(self) -> None:
        """Verify ProjectInitializer uses dynamically discovered templates.

        Test case T054: ProjectInitializer loads templates from discovery
        """
        initializer = ProjectInitializer()

        # Should have available_templates set from discovery
        assert hasattr(initializer, "available_templates")
        assert isinstance(initializer.available_templates, set)
        assert len(initializer.available_templates) > 0

    def test_project_initializer_accepts_valid_templates(self) -> None:
        """Verify ProjectInitializer accepts all discovered templates.

        Test case T054: Valid template validation passes
        """
        import tempfile

        initializer = ProjectInitializer()

        # Should accept all discovered templates
        with tempfile.TemporaryDirectory() as tmpdir:
            for template in ["conversational", "research", "customer-support"]:
                input_data = ProjectInitInput(
                    project_name="test-project",
                    template=template,
                    description="Test",
                    author="",
                    output_dir=tmpdir,
                    overwrite=False,
                )

                # Should not raise for valid templates when dir exists
                try:
                    initializer.validate_inputs(input_data)
                except ValidationError as e:
                    # Should succeed or fail on overwrite, not template
                    assert "template" not in str(e).lower()

    def test_project_initializer_rejects_invalid_template(self) -> None:
        """Verify ProjectInitInput rejects unknown templates.

        Test case T054: Invalid template validation fails at Pydantic level
        """
        import tempfile

        from pydantic import ValidationError as PydanticValidationError

        # Should raise ValidationError at Pydantic level for invalid template
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
        """Verify error message lists available templates.

        Test case T054: Error message includes template list from discovery
        """
        import tempfile

        from pydantic import ValidationError as PydanticValidationError

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

            # Should mention available templates
            assert "conversational" in error_msg
            assert "research" in error_msg
            assert "customer-support" in error_msg


class TestCLITemplateValidation:
    """Test CLI template validation callback (T055)."""

    def test_cli_validate_template_accepts_valid_template(self) -> None:
        """Verify CLI accepts valid template parameter.

        Test case T055: Valid template passes CLI validation
        """
        from holodeck.cli.commands.init import validate_template

        # Should return the template name unchanged
        result = validate_template(None, None, "conversational")
        assert result == "conversational"

    def test_cli_validate_template_accepts_research_template(self) -> None:
        """Verify CLI accepts research template.

        Test case T055: Research template passes CLI validation
        """
        from holodeck.cli.commands.init import validate_template

        result = validate_template(None, None, "research")
        assert result == "research"

    def test_cli_validate_template_accepts_support_template(self) -> None:
        """Verify CLI accepts customer-support template.

        Test case T055: Customer-support template passes CLI validation
        """
        from holodeck.cli.commands.init import validate_template

        result = validate_template(None, None, "customer-support")
        assert result == "customer-support"

    def test_cli_validate_template_rejects_invalid(self) -> None:
        """Verify CLI rejects invalid template.

        Test case T055: Invalid template fails CLI validation
        """
        import click

        from holodeck.cli.commands.init import validate_template

        with pytest.raises(click.BadParameter) as exc_info:
            validate_template(None, None, "invalid-template")

        error_msg = str(exc_info.value).lower()
        assert "unknown template" in error_msg or "available" in error_msg

    def test_cli_validate_template_error_lists_options(self) -> None:
        """Verify CLI error message lists available templates.

        Test case T055: Error message includes template options
        """
        import click

        from holodeck.cli.commands.init import validate_template

        with pytest.raises(click.BadParameter) as exc_info:
            validate_template(None, None, "xyz")

        error_msg = str(exc_info.value)

        # Should mention available templates
        assert "conversational" in error_msg
        assert "research" in error_msg
        assert "customer-support" in error_msg
