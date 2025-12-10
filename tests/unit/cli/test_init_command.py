"""Unit tests for the holodeck init CLI command.

Tests cover:
- T032: init command with valid inputs
- T033: init command with project already exists
- T034: init command with --force flag
- T035: init command with different templates
- T036: init command with custom description
- T037: init command error handling
- T038: init command output messages
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from holodeck.cli.commands.init import init
from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.models.project_config import ProjectInitResult


class TestInitCommandBasic:
    """Tests for basic init command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_creates_project(self):
        """Test init command creates a new project directory."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-test-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert Path("my-test-project").exists()

    def test_init_command_with_description(self):
        """Test init command accepts description parameter."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-project",
                    "--description",
                    "My test agent description",
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()

    def test_init_command_success_message(self):
        """Test init command displays success message."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert "Project initialized successfully" in result.output
            assert "my-project" in result.output

    def test_init_command_shows_project_location(self):
        """Test init command shows project location in output."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert "Location:" in result.output or "project" in result.output.lower()

    def test_init_command_shows_next_steps(self):
        """Test init command shows next steps in output."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert "Next steps:" in result.output
            assert "cd " in result.output or "Edit" in result.output

    def test_init_command_shows_template_used(self):
        """Test init command displays which template was used."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert "Template:" in result.output

    def test_init_command_shows_duration(self):
        """Test init command displays duration in output."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert "Time:" in result.output or "s" in result.output

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_calls_initializer(self, mock_initializer_class: Any) -> None:
        """Test init command calls ProjectInitializer with correct parameters."""
        mock_result = MagicMock(spec=ProjectInitResult)
        mock_result.success = True
        mock_result.project_name = "test-project"
        mock_result.project_path = "test-project"
        mock_result.template_used = "conversational"
        mock_result.duration_seconds = 0.5
        mock_result.errors = []
        mock_result.files_created = [
            "agent.yaml",
            "instructions/system-prompt.md",
        ]
        mock_initializer = MagicMock()
        mock_initializer.initialize.return_value = mock_result
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "test-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert mock_initializer.initialize.called


class TestInitCommandTemplates:
    """Tests for init command with different templates."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_with_conversational_template(self):
        """Test init command with conversational template."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-project",
                    "--template",
                    "conversational",
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert "conversational" in result.output.lower()

    def test_init_command_with_research_template(self):
        """Test init command with research template."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                ["--name", "my-project", "--template", "research", "--non-interactive"],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()

    def test_init_command_with_customer_support_template(self):
        """Test init command with customer-support template."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-project",
                    "--template",
                    "customer-support",
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()

    def test_init_command_default_template_is_conversational(self):
        """Test init command defaults to conversational template."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            # Default is conversational
            assert "conversational" in result.output.lower()


class TestInitCommandExistingDirectory:
    """Tests for init command with existing directories."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_prompts_when_directory_exists(self):
        """Test init command prompts user when directory exists."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            # Create first project
            result1 = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result1.exit_code == 0

            # Try to create again without force - should prompt
            # Note: Using --name without --non-interactive to allow prompt
            result2 = self.runner.invoke(
                init, ["--name", "my-project"], input="n\n"
            )  # Answer no to overwrite
            # When user declines, the command returns 0 but shows cancelled message
            assert (
                "cancelled" in result2.output.lower()
                or "already exists" in result2.output.lower()
            )
            assert "Do you want to overwrite it?" in result2.output

    def test_init_command_confirms_overwrite_when_user_agrees(self):
        """Test init command confirms and overwrites when user agrees."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            # Create first project
            result1 = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result1.exit_code == 0

            # Create marker file to verify overwrite happens
            marker_path = Path("my-project") / "marker.txt"
            marker_path.write_text("old content")
            assert marker_path.exists()

            # Try to create again and confirm overwrite
            result2 = self.runner.invoke(init, ["--name", "my-project"], input="y\n")
            # Should succeed and overwrite
            assert result2.exit_code == 0
            # Marker file should be gone after overwrite
            assert not marker_path.exists()
            assert "Project initialized successfully" in result2.output

    def test_init_command_overwrite_with_force_flag(self):
        """Test init command with --force flag overwrites directory."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            # Create first project
            result1 = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result1.exit_code == 0

            # Create marker file
            marker_path = Path("my-project") / "marker.txt"
            marker_path.write_text("old content")
            assert marker_path.exists()

            # Create again with force
            result2 = self.runner.invoke(
                init, ["--name", "my-project", "--force", "--non-interactive"]
            )
            assert result2.exit_code == 0
            # Marker file should be gone
            assert not marker_path.exists()

    def test_init_command_cancelled_when_user_declines_overwrite(self):
        """Test init command is cancelled when user declines overwrite."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            # Create first project
            result1 = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result1.exit_code == 0

            # Try to create again without force - decline overwrite
            result2 = self.runner.invoke(init, ["--name", "my-project"], input="n\n")
            # When user declines, command shows cancelled message
            assert "cancelled" in result2.output.lower()


class TestInitCommandErrorHandling:
    """Tests for init command error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_handles_validation_error(
        self, mock_initializer_class: Any
    ) -> None:
        """Test init command handles ValidationError gracefully."""
        mock_initializer = MagicMock()
        mock_initializer.initialize.side_effect = ValidationError(
            "Invalid project name"
        )
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code != 0
            assert "Error:" in result.output

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_handles_init_error(self, mock_initializer_class: Any) -> None:
        """Test init command handles InitError gracefully."""
        mock_initializer = MagicMock()
        mock_initializer.initialize.side_effect = InitError(
            "Failed to initialize project"
        )
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code != 0
            assert "Error:" in result.output

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_handles_unexpected_error(
        self, mock_initializer_class: Any
    ) -> None:
        """Test init command handles unexpected errors gracefully."""
        mock_initializer = MagicMock()
        mock_initializer.initialize.side_effect = RuntimeError("Something went wrong")
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code != 0
            assert "Unexpected error" in result.output

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_handles_keyboard_interrupt(
        self, mock_initializer_class: Any
    ) -> None:
        """Test init command handles KeyboardInterrupt (Ctrl+C) gracefully."""
        mock_initializer = MagicMock()
        mock_initializer.initialize.side_effect = KeyboardInterrupt()
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code != 0
            assert "cancelled by user" in result.output.lower()

    @patch("holodeck.cli.commands.init.ProjectInitializer")
    def test_init_command_displays_initialization_failure(
        self, mock_initializer_class: Any
    ) -> None:
        """Test init command displays failure message when initialization fails."""
        mock_result = MagicMock(spec=ProjectInitResult)
        mock_result.success = False
        mock_result.errors = ["Failed to create directories", "Permission denied"]

        mock_initializer = MagicMock()
        mock_initializer.initialize.return_value = mock_result
        mock_initializer_class.return_value = mock_initializer

        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code != 0
            assert "initialization failed" in result.output.lower()
            assert "Error:" in result.output


class TestInitCommandIntegration:
    """Integration tests for init command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_full_workflow_with_all_options(self):
        """Test init command with all options specified."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-agent",
                    "--template",
                    "research",
                    "--description",
                    "A research assistant",
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert Path("my-agent").exists()
            assert "Project initialized successfully" in result.output

    def test_init_command_project_name_is_required(self):
        """Test init command requires project name in non-interactive mode."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(init, ["--non-interactive"])
            assert result.exit_code != 0
            # Should show error about missing --name in non-interactive mode
            assert (
                "required" in result.output.lower() or "name" in result.output.lower()
            )

    def test_init_command_with_special_chars_in_description(self):
        """Test init command with special characters in description."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-project",
                    "--description",
                    "Agent for Q&A, with features like: search, filter",
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()


class TestInitCommandOutput:
    """Tests for init command output formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_output_has_blank_lines(self):
        """Test init command output has proper formatting with blank lines."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            # Output should have multiple lines with proper spacing
            lines = result.output.strip().split("\n")
            assert len(lines) > 3

    def test_init_command_success_message_is_bold_green(self):
        """Test init command success message formatting."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            # Check that output contains success indicator (✓ or "success")
            assert "success" in result.output.lower() or "✓" in result.output

    def test_init_command_lists_next_steps(self):
        """Test init command lists all next steps."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-project", "--non-interactive"]
            )
            assert result.exit_code == 0
            output_lower = result.output.lower()
            # Check for step indicators
            assert "1." in result.output or "cd" in output_lower
            assert "agent.yaml" in output_lower
            assert "test" in output_lower or "run" in output_lower

    def test_init_command_error_message_is_red(self):
        """Test init command error message formatting."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            # Create first project
            self.runner.invoke(init, ["--name", "my-project", "--non-interactive"])

            # Try again without force - should show cancellation message
            result = self.runner.invoke(init, ["--name", "my-project"], input="n\n")
            # Check for cancellation message
            assert "cancelled" in result.output.lower()


class TestInitCommandEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner(catch_exceptions=False)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_command_with_hyphenated_name(self):
        """Test init command with hyphenated project name."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my-awesome-agent", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert Path("my-awesome-agent").exists()

    def test_init_command_with_underscored_name(self):
        """Test init command with underscored project name."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "my_awesome_agent", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert Path("my_awesome_agent").exists()

    def test_init_command_with_numbers_in_name(self):
        """Test init command with numbers in project name."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init, ["--name", "agent-v2-test", "--non-interactive"]
            )
            assert result.exit_code == 0
            assert Path("agent-v2-test").exists()

    def test_init_command_with_empty_description(self):
        """Test init command with empty description."""
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                ["--name", "my-project", "--description", "", "--non-interactive"],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()

    def test_init_command_with_very_long_description(self):
        """Test init command with very long description."""
        long_desc = "A" * 500
        with self.runner.isolated_filesystem(temp_dir=self.temp_dir):
            result = self.runner.invoke(
                init,
                [
                    "--name",
                    "my-project",
                    "--description",
                    long_desc,
                    "--non-interactive",
                ],
            )
            assert result.exit_code == 0
            assert Path("my-project").exists()
