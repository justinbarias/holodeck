"""Unit tests for ProjectInitializer class and related functionality.

Tests cover:
- T012: ProjectInitializer.validate_inputs()
- T013: ProjectInitializer.load_template()
- T014: ProjectInitializer.initialize()
- T017: Error handling
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.cli.utils.project_init import ProjectInitializer
from holodeck.models.project_config import ProjectInitInput, ProjectInitResult


class TestProjectInitializerValidateInputs:
    """Tests for ProjectInitializer.validate_inputs() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.initializer = ProjectInitializer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_validate_inputs_with_valid_name(self):
        """Test validation accepts valid project names."""
        valid_names = [
            "my-project",
            "my_project",
            "myproject123",
            "project-1",
            "a",
            "project_with_underscores",
            "project-with-hyphens",
        ]
        for name in valid_names:
            input_data = ProjectInitInput(
                project_name=name,
                template="conversational",
                output_dir=self.temp_dir,
            )
            # Should not raise
            self.initializer.validate_inputs(input_data)

    def test_validate_inputs_with_invalid_special_chars(self):
        """Test validation rejects names with invalid characters."""
        invalid_names = [
            "my project",  # space
            "my@project",  # @
            "my.project",  # .
            "my!project",  # !
            "my/project",  # /
            "my\\project",  # backslash
            "my$project",  # $
        ]
        for name in invalid_names:
            # ProjectInitInput model will raise Pydantic ValidationError
            from pydantic import ValidationError as PydanticValidationError

            with pytest.raises(PydanticValidationError):
                ProjectInitInput(
                    project_name=name,
                    template="conversational",
                    output_dir=self.temp_dir,
                )

    def test_validate_inputs_with_leading_digit(self):
        """Test validation rejects names starting with digits."""
        invalid_names = ["1project", "9my-project", "0test"]
        for name in invalid_names:
            # ProjectInitInput model will raise Pydantic ValidationError
            from pydantic import ValidationError as PydanticValidationError

            with pytest.raises(PydanticValidationError):
                ProjectInitInput(
                    project_name=name,
                    template="conversational",
                    output_dir=self.temp_dir,
                )

    def test_validate_inputs_with_invalid_template(self):
        """Test validation rejects invalid template names."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            ProjectInitInput(
                project_name="my-project",
                template="nonexistent-template",
                output_dir=self.temp_dir,
            )

    def test_validate_inputs_with_existing_directory_no_force(self):
        """Test validation rejects existing directory without force flag."""
        existing_dir = Path(self.temp_dir) / "existing-project"
        existing_dir.mkdir()

        input_data = ProjectInitInput(
            project_name="existing-project",
            template="conversational",
            output_dir=str(existing_dir.parent),
            overwrite=False,
        )
        with pytest.raises(ValidationError):
            self.initializer.validate_inputs(input_data)

    def test_validate_inputs_with_existing_directory_with_force(self):
        """Test validation accepts existing directory with force flag."""
        existing_dir = Path(self.temp_dir) / "existing-project"
        existing_dir.mkdir()

        input_data = ProjectInitInput(
            project_name="existing-project",
            template="conversational",
            output_dir=str(existing_dir.parent),
            overwrite=True,
        )
        # Should not raise
        self.initializer.validate_inputs(input_data)

    def test_validate_inputs_with_non_writable_directory(self):
        """Test validation checks directory write permissions."""
        # Create a read-only directory (on supported systems)
        readonly_dir = Path(self.temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)

        try:
            input_data = ProjectInitInput(
                project_name="my-project",
                template="conversational",
                output_dir=str(readonly_dir),
            )
            with pytest.raises(ValidationError):
                self.initializer.validate_inputs(input_data)
        finally:
            readonly_dir.chmod(0o755)

    def test_validate_inputs_with_empty_name(self):
        """Test validation rejects empty project names."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            ProjectInitInput(
                project_name="",
                template="conversational",
                output_dir=self.temp_dir,
            )

    def test_validate_inputs_with_long_name(self):
        """Test validation handles very long project names."""
        long_name = "a" * 256
        # Should handle gracefully (either accept or reject with clear message)
        # ProjectInitInput model will raise Pydantic ValidationError for long names
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            ProjectInitInput(
                project_name=long_name,
                template="conversational",
                output_dir=self.temp_dir,
            )


class TestProjectInitializerLoadTemplate:
    """Tests for ProjectInitializer.load_template() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.initializer = ProjectInitializer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_load_template_with_valid_manifest(self):
        """Test loading a template with valid manifest.yaml."""
        # Create a test template directory
        template_dir = Path(self.temp_dir) / "test-template"
        template_dir.mkdir()

        manifest_data = {
            "name": "test-template",
            "display_name": "Test Template",
            "description": "A test template",
            "category": "test",
            "version": "1.0.0",
            "variables": {
                "project_name": {
                    "type": "string",
                    "description": "Project name",
                    "required": True,
                }
            },
            "defaults": {"model.provider": "openai"},
            "files": {
                "agent.yaml": {"path": "agent.yaml", "template": True, "required": True}
            },
        }

        manifest_file = template_dir / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(manifest_data, f)

        # Load the manifest
        manifest = self.initializer.load_template(str(template_dir))
        assert manifest.name == "test-template"
        assert manifest.version == "1.0.0"

    def test_load_template_with_missing_manifest(self):
        """Test loading a template without manifest.yaml."""
        template_dir = Path(self.temp_dir) / "missing-manifest"
        template_dir.mkdir()

        with pytest.raises((FileNotFoundError, InitError)):
            self.initializer.load_template(str(template_dir))

    def test_load_template_with_malformed_yaml(self):
        """Test loading a template with malformed manifest.yaml."""
        template_dir = Path(self.temp_dir) / "malformed-yaml"
        template_dir.mkdir()

        manifest_file = template_dir / "manifest.yaml"
        with open(manifest_file, "w") as f:
            f.write("invalid: yaml: content:\n  - bad")

        with pytest.raises((yaml.YAMLError, InitError, ValidationError)):
            self.initializer.load_template(str(template_dir))

    def test_load_template_with_missing_required_fields(self):
        """Test loading a template with missing required manifest fields."""
        template_dir = Path(self.temp_dir) / "incomplete-manifest"
        template_dir.mkdir()

        # Missing 'version' field
        manifest_data = {
            "name": "incomplete",
            "display_name": "Incomplete",
            "description": "Test",
            "category": "test",
            "variables": {},
        }

        manifest_file = template_dir / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(manifest_data, f)

        with pytest.raises((ValidationError, InitError)):
            self.initializer.load_template(str(template_dir))


class TestProjectInitializerInitialize:
    """Tests for ProjectInitializer.initialize() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.initializer = ProjectInitializer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("holodeck.cli.utils.project_init.ProjectInitializer.load_template")
    @patch("holodeck.cli.utils.project_init.TemplateRenderer")
    def test_initialize_with_successful_creation(self, mock_renderer, mock_load):
        """Test successful project initialization."""
        # Set up mocks
        manifest = MagicMock()
        manifest.files = {}
        mock_load.return_value = manifest

        input_data = ProjectInitInput(
            project_name="test-project",
            template="conversational",
            output_dir=self.temp_dir,
        )

        # Initialize
        result = self.initializer.initialize(input_data)

        assert result.success is True
        assert result.project_name == "test-project"
        assert Path(result.project_path).exists()

    def test_initialize_with_directory_exists_no_force(self):
        """Test initialization fails when directory exists without force."""
        existing_dir = Path(self.temp_dir) / "existing"
        existing_dir.mkdir()

        input_data = ProjectInitInput(
            project_name="existing",
            template="conversational",
            output_dir=str(existing_dir.parent),
            overwrite=False,
        )

        # Should return failure result
        result = self.initializer.initialize(input_data)
        assert result.success is False
        assert "already exists" in str(result.errors).lower()

    def test_initialize_with_partial_cleanup_on_error(self):
        """Test that partial directories are cleaned up on failure.

        Note: This is a basic unit test. Integration tests should verify
        actual file creation and cleanup behavior with real templates.
        """
        # Create an input with a non-writable output dir to trigger error
        readonly_temp = Path(self.temp_dir) / "readonly_parent"
        readonly_temp.mkdir()
        readonly_temp.chmod(0o555)

        try:
            input_data = ProjectInitInput(
                project_name="failing-project",
                template="conversational",
                output_dir=str(readonly_temp),
            )

            # Should return failure result
            result = self.initializer.initialize(input_data)
            assert result.success is False
        finally:
            readonly_temp.chmod(0o755)

    @patch("holodeck.cli.utils.project_init.ProjectInitializer.load_template")
    @patch("holodeck.cli.utils.project_init.TemplateRenderer")
    def test_initialize_returns_correct_metadata(self, mock_renderer, mock_load):
        """Test that initialize returns correct ProjectInitResult."""
        # Set up mocks
        manifest = MagicMock()
        manifest.files = {}
        mock_load.return_value = manifest

        input_data = ProjectInitInput(
            project_name="test-project",
            template="conversational",
            output_dir=self.temp_dir,
            description="Test description",
            author="Test Author",
        )

        result = self.initializer.initialize(input_data)

        assert isinstance(result, ProjectInitResult)
        assert result.project_name == "test-project"
        assert result.template_used == "conversational"
        assert result.success is True
        assert result.duration_seconds >= 0
        assert isinstance(result.files_created, list)
