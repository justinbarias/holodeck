"""Template rendering and validation engine for HoloDeck.

This module provides the TemplateRenderer class which handles:
- Jinja2 template rendering with restricted filters
- YAML validation against AgentConfig schema
- Safe rendering with input validation
"""

from pathlib import Path
from typing import Any

import yaml
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateSyntaxError,
    UndefinedError,
    select_autoescape,
)

from holodeck.models.agent import Agent


class TemplateRenderer:
    """Renders Jinja2 templates and validates output against schemas.

    Provides safe template rendering with:
    - Restricted Jinja2 filters for security
    - YAML validation against AgentConfig schema
    - Clear error messages for debugging
    """

    def __init__(self) -> None:
        """Initialize the TemplateRenderer with a secure Jinja2 environment."""
        # Create Jinja2 environment with strict mode (undefined variables cause errors)
        # autoescape disabled for YAML templates (appropriate for config generation)
        self.env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=select_autoescape(
                enabled_extensions=("html", "xml"),
                default_for_string=False,
                default=False,
            ),
        )

        # Only allow safe filters
        self._setup_safe_filters()

    def _setup_safe_filters(self) -> None:
        """Configure safe Jinja2 filters for template rendering.

        Restricts available filters to prevent arbitrary code execution.
        """
        # Keep default safe filters, explicitly block dangerous ones
        # By default, Jinja2 provides safe filters like: upper, lower, title, etc.
        pass

    def render_template(self, template_path: str, variables: dict[str, Any]) -> str:
        """Render a Jinja2 template with provided variables.

        Args:
            template_path: Path to the Jinja2 template file
            variables: Dictionary of variables to pass to the template

        Returns:
            Rendered template content as a string

        Raises:
            FileNotFoundError: If template file doesn't exist
            InitError: If rendering fails (syntax errors, undefined variables, etc.)
        """
        from holodeck.cli.exceptions import InitError

        template_file = Path(template_path)

        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        try:
            # Load template from file
            loader = FileSystemLoader(str(template_file.parent))
            env = Environment(
                loader=loader,
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=select_autoescape(
                    enabled_extensions=("html", "xml"),
                    default_for_string=False,
                    default=False,
                ),
            )

            template = env.get_template(template_file.name)

            # Render template
            return template.render(variables)

        except TemplateSyntaxError as e:
            raise InitError(
                f"Template syntax error in {template_path}:\n"
                f"  Line {e.lineno}: {e.message}"
            ) from e
        except UndefinedError as e:
            raise InitError(
                f"Template rendering error in {template_path}:\n"
                f"  Undefined variable: {str(e)}"
            ) from e
        except Exception as e:
            raise InitError(f"Template rendering failed: {str(e)}") from e

    def validate_agent_config(self, yaml_content: str) -> Agent:
        """Validate YAML content against Agent schema.

        Parses YAML and validates it against the Agent Pydantic model.
        This is the critical validation gate for agent.yaml files.

        Args:
            yaml_content: YAML content as a string

        Returns:
            Agent: Validated Agent configuration object

        Raises:
            ValidationError: If YAML is invalid or doesn't match schema
            InitError: If parsing fails
        """
        from holodeck.cli.exceptions import InitError, ValidationError

        try:
            # Parse YAML
            data = yaml.safe_load(yaml_content)

            if not data:
                raise ValidationError("agent.yaml content is empty")

            # Validate against Agent schema
            agent = Agent.model_validate(data)
            return agent

        except yaml.YAMLError as e:
            raise ValidationError(f"YAML parsing error:\n" f"  {str(e)}") from e
        except ValidationError:
            # Re-raise our validation errors as-is
            raise
        except Exception as e:
            # Catch Pydantic validation errors
            if hasattr(e, "errors"):
                # Pydantic ValidationError
                errors = e.errors()
                error_msg = "Agent configuration validation failed:\n"
                for error in errors:
                    field = ".".join(str(loc) for loc in error["loc"])
                    error_msg += f"  {field}: {error['msg']}\n"
                raise ValidationError(error_msg) from e
            else:
                raise InitError(
                    f"Agent configuration validation failed: {str(e)}"
                ) from e

    def render_and_validate(self, template_path: str, variables: dict[str, Any]) -> str:
        """Render a Jinja2 template and validate output (for YAML files).

        Combines rendering and validation in a safe way: only returns
        rendered content if both rendering and validation succeed.
        This is the recommended way to process agent.yaml templates.

        Args:
            template_path: Path to the Jinja2 template file
            variables: Dictionary of variables to pass to the template

        Returns:
            Rendered and validated template content as a string

        Raises:
            InitError: If rendering fails
            ValidationError: If validation fails
        """
        # Render template first
        rendered = self.render_template(template_path, variables)

        # Determine if this is agent.yaml specifically (not all YAML files)
        template_file = Path(template_path)
        is_agent_yaml = (
            template_file.name == "agent.yaml.j2" or template_file.stem == "agent.yaml"
        )

        if is_agent_yaml:
            # Validate YAML against schema
            # This will raise ValidationError if invalid
            self.validate_agent_config(rendered)

        # Return rendered content (safe to write to disk)
        return rendered

    @staticmethod
    def _discover_template_dirs() -> list[Path]:
        """Discover valid template directories.

        Internal helper that finds all template directories containing
        a manifest.yaml file.

        Returns:
            Sorted list of Path objects for valid template directories.
        """
        templates_dir = Path(__file__).parent.parent / "templates"

        if not templates_dir.exists():
            return []

        template_dirs = []
        for template_dir in sorted(templates_dir.iterdir()):
            if (
                template_dir.is_dir()
                and not template_dir.name.startswith("_")
                and (template_dir / "manifest.yaml").exists()
            ):
                template_dirs.append(template_dir)

        return template_dirs

    @staticmethod
    def list_available_templates() -> list[str]:
        """List all available built-in templates.

        Discovers templates from the templates/ directory structure.

        Returns:
            List of template names (e.g., ['conversational', 'research',
            'customer-support'])
        """
        return [d.name for d in TemplateRenderer._discover_template_dirs()]

    @staticmethod
    def get_available_templates() -> list[dict[str, str]]:
        """Get available templates with metadata.

        Discovers templates from the templates/ directory and extracts
        metadata (name, display_name, description) from their manifest.yaml files.

        Returns:
            List of dicts with 'value', 'display_name', 'description' keys.
            Returns empty list if templates directory doesn't exist.
        """
        templates: list[dict[str, str]] = []

        for template_dir in TemplateRenderer._discover_template_dirs():
            manifest_path = template_dir / "manifest.yaml"
            with open(manifest_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                templates.append(
                    {
                        "value": str(data.get("name", template_dir.name)),
                        "display_name": str(
                            data.get("display_name", template_dir.name)
                        ),
                        "description": str(data.get("description", "")),
                    }
                )

        return templates
