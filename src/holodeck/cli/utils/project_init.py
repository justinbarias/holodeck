"""Project initialization logic for holodeck init command.

This module provides the ProjectInitializer class which handles the core logic
of creating new agent projects, including validation, template loading, and
directory structure creation.
"""

import contextlib
import os
import re
import shutil
import time
from pathlib import Path

import yaml

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.lib.validation import sanitize_tool_name
from holodeck.models.project_config import ProjectInitInput, ProjectInitResult
from holodeck.models.template_manifest import TemplateManifest
from holodeck.models.wizard_config import (
    LLM_PROVIDER_CHOICES,
    MCP_SERVER_CHOICES,
    VECTOR_STORE_CHOICES,
)


def get_model_for_provider(provider: str) -> str:
    """Get the default model for an LLM provider.

    Args:
        provider: LLM provider identifier (e.g., 'ollama', 'openai').

    Returns:
        Default model name for the provider.
    """
    for choice in LLM_PROVIDER_CHOICES:
        if choice.value == provider:
            return choice.default_model
    return "gpt-oss:20b"  # Fallback to Ollama default


def get_mcp_server_config(server_id: str) -> dict[str, str]:
    """Get configuration for an MCP server.

    Args:
        server_id: MCP server identifier (e.g., 'brave-search', 'memory').

    Returns:
        Dictionary with server configuration (name, package, command).
    """
    for server in MCP_SERVER_CHOICES:
        if server.value == server_id:
            return {
                "name": sanitize_tool_name(server.value),
                "display_name": server.display_name,
                "description": server.description,
                "package": server.package_identifier,
                "command": server.command,
            }
    return {
        "name": sanitize_tool_name(server_id),
        "display_name": server_id,
        "description": "",
        "package": server_id,
        "command": "npx",
    }


def get_vectorstore_endpoint(store: str) -> str | None:
    """Get the default endpoint for a vector store.

    Args:
        store: Vector store identifier (e.g., 'chromadb', 'qdrant').

    Returns:
        Default endpoint URL or None if not applicable.
    """
    for choice in VECTOR_STORE_CHOICES:
        if choice.value == store:
            return choice.default_endpoint
    return None


def get_provider_api_key_env_var(provider: str) -> str | None:
    """Get the API key environment variable name for an LLM provider.

    Args:
        provider: LLM provider identifier (e.g., 'openai', 'azure_openai').

    Returns:
        Environment variable name for API key, or None if not required.
    """
    for choice in LLM_PROVIDER_CHOICES:
        if choice.value == provider:
            return choice.api_key_env_var
    return None


def get_provider_endpoint_env_var(provider: str) -> str | None:
    """Get the endpoint environment variable name for an LLM provider.

    Args:
        provider: LLM provider identifier (e.g., 'azure_openai').

    Returns:
        Environment variable name for endpoint, or None if not required.
    """
    for choice in LLM_PROVIDER_CHOICES:
        if choice.value == provider:
            return choice.endpoint_env_var
    return None


class ProjectInitializer:
    """Handles project initialization logic.

    Provides methods to:
    - Validate user inputs (project name, template, permissions)
    - Load and validate template manifests
    - Initialize new agent projects with all required files
    """

    # Valid project name pattern: alphanumeric, hyphens, underscores, no leading digits
    PROJECT_NAME_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_-]*$"
    MAX_PROJECT_NAME_LENGTH = 64

    def __init__(self) -> None:
        """Initialize the ProjectInitializer."""
        self.template_renderer = TemplateRenderer()
        # Get available templates from discovery function
        self.available_templates = set(TemplateRenderer.list_available_templates())

    def validate_inputs(self, input_data: ProjectInitInput) -> None:
        """Validate user inputs for project initialization.

        Checks:
        - Project name format (alphanumeric, hyphens, underscores, no leading digits)
        - Project name is not empty and within length limits
        - Template exists in available templates
        - Output directory is writable
        - Project directory doesn't already exist (unless overwrite is True)

        Args:
            input_data: ProjectInitInput with user-provided values

        Raises:
            ValidationError: If any validation checks fail
        """
        project_name = input_data.project_name.strip()

        # Check project name is not empty
        if not project_name:
            raise ValidationError("Project name cannot be empty")

        # Check project name length
        if len(project_name) > self.MAX_PROJECT_NAME_LENGTH:
            raise ValidationError(
                f"Project name cannot exceed {self.MAX_PROJECT_NAME_LENGTH} characters"
            )

        # Check project name format
        if not re.match(self.PROJECT_NAME_PATTERN, project_name):
            raise ValidationError(
                f"Invalid project name: '{project_name}'. "
                "Project names must start with a letter or underscore, "
                "and contain only alphanumeric characters, hyphens, and underscores."
            )

        # Check template exists
        if input_data.template not in self.available_templates:
            templates_list = ", ".join(sorted(self.available_templates))
            raise ValidationError(
                f"Unknown template: '{input_data.template}'. "
                f"Available templates: {templates_list}"
            )

        # Check output directory is writable
        output_dir = Path(input_data.output_dir)
        if not output_dir.exists():
            raise ValidationError(f"Output directory does not exist: {output_dir}")

        if not output_dir.is_dir():
            raise ValidationError(f"Output path is not a directory: {output_dir}")

        try:
            # Test write permissions by attempting to check access
            if not os.access(str(output_dir), os.W_OK):
                raise ValidationError(f"Output directory is not writable: {output_dir}")
        except OSError as e:
            raise ValidationError(f"Cannot access output directory: {e}") from e

        # Check project directory doesn't already exist (unless force)
        project_dir = output_dir / project_name
        if project_dir.exists() and not input_data.overwrite:
            raise ValidationError(
                f"Project directory already exists: {project_dir}. "
                "Use --force to overwrite."
            )

    def load_template(self, template_name: str) -> TemplateManifest:
        """Load and validate a template manifest.

        Loads the manifest.yaml file from a template directory and validates
        it against the TemplateManifest schema.

        Args:
            template_name: Name of the template (e.g., 'conversational')

        Returns:
            TemplateManifest: Parsed and validated template manifest

        Raises:
            FileNotFoundError: If template or manifest file not found
            InitError: If manifest cannot be parsed or validated
        """
        # Get template directory
        # Templates are bundled in src/holodeck/templates/
        template_dir = Path(__file__).parent.parent.parent / "templates" / template_name

        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        manifest_path = template_dir / "manifest.yaml"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Template manifest not found: {manifest_path}")

        try:
            with open(manifest_path) as f:
                manifest_data = yaml.safe_load(f)

            if not manifest_data:
                raise InitError(f"Template manifest is empty: {manifest_path}")

            # Validate against TemplateManifest schema
            manifest = TemplateManifest.model_validate(manifest_data)
            return manifest

        except yaml.YAMLError as e:
            raise InitError(f"Template manifest contains invalid YAML: {e}") from e
        except Exception as e:
            if isinstance(e, ValidationError | InitError):
                raise
            raise InitError(f"Failed to load template manifest: {e}") from e

    def initialize(self, input_data: ProjectInitInput) -> ProjectInitResult:
        """Initialize a new agent project.

        Creates a new project directory with all required files and templates.
        Follows all-or-nothing semantics: either the entire project is created
        successfully, or no files are created and the directory is cleaned up.

        Args:
            input_data: ProjectInitInput with validated user inputs

        Returns:
            ProjectInitResult: Result of initialization with status and metadata

        Raises:
            InitError: If initialization fails (will attempt cleanup)
        """
        start_time = time.time()
        project_name = input_data.project_name.strip()
        output_dir = Path(input_data.output_dir)
        project_dir = output_dir / project_name

        files_created = []

        try:
            # Validate inputs first
            self.validate_inputs(input_data)

            # Load template manifest
            template = self.load_template(input_data.template)

            # Create project directory
            if project_dir.exists() and input_data.overwrite:
                # Remove existing directory if force flag is set
                shutil.rmtree(project_dir)

            project_dir.mkdir(parents=True, exist_ok=False)
            files_created.append(str(project_dir))

            # Prepare provider-specific config
            provider_config = input_data.provider_config
            endpoint_env_var = get_provider_endpoint_env_var(input_data.llm_provider)

            # Determine endpoint value
            llm_endpoint = None
            if provider_config and provider_config.endpoint:
                llm_endpoint = provider_config.endpoint
            elif endpoint_env_var:
                # Use environment variable placeholder as default
                llm_endpoint = f"${{{endpoint_env_var}}}"

            # Prepare template variables
            template_vars = {
                "project_name": project_name,
                "description": input_data.description or "TODO: Add agent description",
                "author": input_data.author or "",
                # Wizard configuration fields
                "agent_name": input_data.agent_name,
                "llm_provider": input_data.llm_provider,
                "llm_model": get_model_for_provider(input_data.llm_provider),
                "llm_endpoint": llm_endpoint,
                "llm_api_key_env_var": get_provider_api_key_env_var(
                    input_data.llm_provider
                ),
                "vector_store": input_data.vector_store,
                "vector_store_endpoint": get_vectorstore_endpoint(
                    input_data.vector_store
                ),
                "evals": input_data.evals,
                "mcp_servers": [
                    get_mcp_server_config(s) for s in input_data.mcp_servers
                ],
            }

            # Add template-specific defaults from manifest
            if template.defaults:
                template_vars.update(template.defaults)

            # Create files from template
            template_dir = (
                Path(__file__).parent.parent.parent / "templates" / input_data.template
            )

            # Process each file in the template manifest
            if template.files:
                for file_spec in template.files.values():
                    if not file_spec.required:
                        continue

                    file_path = project_dir / file_spec.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    if file_spec.template:
                        # Render Jinja2 template
                        template_file = template_dir / f"{file_spec.path}.j2"
                        if not template_file.exists():
                            # Try without .j2 extension
                            template_file = template_dir / file_spec.path

                        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
                            # Validate YAML files against schema
                            content = self.template_renderer.render_and_validate(
                                str(template_file), template_vars
                            )
                        else:
                            # Render non-YAML files normally
                            content = self.template_renderer.render_template(
                                str(template_file), template_vars
                            )

                        file_path.write_text(content)
                    else:
                        # Copy static files directly
                        source_file = template_dir / file_spec.path
                        if source_file.exists():
                            shutil.copy2(source_file, file_path)

                    files_created.append(str(file_path.relative_to(output_dir)))

            # Also copy .gitignore if it exists
            gitignore_src = template_dir / ".gitignore"
            if gitignore_src.exists():
                gitignore_dst = project_dir / ".gitignore"
                shutil.copy2(gitignore_src, gitignore_dst)
                files_created.append(str(gitignore_dst.relative_to(output_dir)))

            duration = time.time() - start_time

            return ProjectInitResult(
                success=True,
                project_name=project_name,
                project_path=str(project_dir),
                template_used=input_data.template,
                files_created=files_created,
                warnings=[],
                errors=[],
                duration_seconds=duration,
            )

        except (ValidationError, InitError) as e:
            # Clean up partial directory on error
            if project_dir.exists():
                with contextlib.suppress(Exception):
                    shutil.rmtree(project_dir)

            duration = time.time() - start_time

            return ProjectInitResult(
                success=False,
                project_name=project_name,
                project_path=str(project_dir),
                template_used=input_data.template,
                files_created=[],
                warnings=[],
                errors=[str(e)],
                duration_seconds=duration,
            )

        except Exception as e:
            # Clean up partial directory on unexpected error
            if project_dir.exists():
                with contextlib.suppress(Exception):
                    shutil.rmtree(project_dir)

            duration = time.time() - start_time

            return ProjectInitResult(
                success=False,
                project_name=project_name,
                project_path=str(project_dir),
                template_used=input_data.template,
                files_created=[],
                warnings=[],
                errors=[f"Unexpected error: {str(e)}"],
                duration_seconds=duration,
            )
