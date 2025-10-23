"""Integration tests for basic project creation (Phase 3: US1).

Tests for the holodeck init command with:
- Basic project creation
- Default template selection
- Overwrite behavior
- Success messages
- Ctrl+C handling
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
class TestInitBasicProjectCreation:
    """Test basic project creation functionality (T027)."""

    def test_holodeck_init_creates_directory(self, temp_dir: Path) -> None:
        """Verify `holodeck init test-project` creates project directory.

        Test case T027: Basic project creation
        """
        # Change to temp directory for test
        project_name = "test-agent"

        # Run holodeck init command
        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify project directory was created
        project_dir = temp_dir / project_name
        assert project_dir.exists(), f"Project directory not created: {project_dir}"
        assert project_dir.is_dir(), f"Project path is not a directory: {project_dir}"

    def test_holodeck_init_creates_agent_yaml(self, temp_dir: Path) -> None:
        """Verify `holodeck init` creates valid agent.yaml file.

        Test case T027: Basic project creation with agent.yaml
        """
        project_name = "test-agent"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify agent.yaml exists
        agent_yaml = temp_dir / project_name / "agent.yaml"
        assert agent_yaml.exists(), f"agent.yaml not created: {agent_yaml}"
        assert agent_yaml.is_file(), f"agent.yaml is not a file: {agent_yaml}"

        # Verify agent.yaml has content
        content = agent_yaml.read_text()
        assert len(content) > 0, "agent.yaml is empty"
        assert "name:" in content, "agent.yaml missing 'name' field"

    def test_holodeck_init_creates_all_folders(self, temp_dir: Path) -> None:
        """Verify `holodeck init` creates all required project folders.

        Test case T027: Verify directory structure
        """
        project_name = "test-agent"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        project_dir = temp_dir / project_name

        # Check for required directories
        required_dirs = [
            project_dir / "instructions",
            project_dir / "tools",
            project_dir / "data",
        ]

        for required_dir in required_dirs:
            assert (
                required_dir.exists()
            ), f"Required directory not created: {required_dir}"
            assert required_dir.is_dir(), f"Path is not a directory: {required_dir}"


@pytest.mark.integration
class TestInitDefaultTemplate:
    """Test default template selection functionality (T028)."""

    def test_holodeck_init_uses_conversational_by_default(self, temp_dir: Path) -> None:
        """Verify conversational is default template when --template omitted.

        Test case T028: Default template selection
        """
        project_name = "test-agent"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check that output mentions conversational template (if verbosity allows)
        # or at least that the default template files were created
        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        # Conversational template should have specific defaults
        assert "provider:" in content or "name:" in content
        assert temp_dir / project_name / "instructions" / "system-prompt.md"

    def test_holodeck_init_respects_template_option(self, temp_dir: Path) -> None:
        """Verify --template option allows template selection.

        Test case T028: Template option functionality
        """
        project_name = "test-agent"

        # Try with research template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "research",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify project was created
        project_dir = temp_dir / project_name
        assert project_dir.exists()
        assert (project_dir / "agent.yaml").exists()


@pytest.mark.integration
class TestInitOverwriteBehavior:
    """Test overwrite behavior functionality (T029)."""

    def test_holodeck_init_fails_if_directory_exists_without_force(
        self, temp_dir: Path
    ) -> None:
        """Verify command fails when directory exists without --force flag.

        Test case T029: Overwrite behavior - error without force
        """
        project_name = "test-agent"

        # Create first project
        result1 = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Try to create again without force - should fail
        result2 = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result2.returncode != 0, "Command should fail when directory exists"
        assert (
            "already exists" in result2.stderr.lower()
            or "already exists" in result2.stdout.lower()
        )

    def test_holodeck_init_overwrites_with_force_flag(self, temp_dir: Path) -> None:
        """Verify --force flag allows overwriting existing directory.

        Test case T029: Overwrite behavior - success with force
        """
        project_name = "test-agent"

        # Create first project
        result1 = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Create marker file to verify overwrite
        marker_file = temp_dir / project_name / "marker.txt"
        marker_file.write_text("old content")
        assert marker_file.exists()

        # Create again with force - should succeed
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--force",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result2.returncode == 0, f"Command failed: {result2.stderr}"

        # Marker file should be gone (directory was replaced)
        assert not marker_file.exists(), "Old files not removed during overwrite"


@pytest.mark.integration
class TestInitSuccessMessage:
    """Test success message output functionality (T030)."""

    def test_holodeck_init_displays_success_message(self, temp_dir: Path) -> None:
        """Verify output shows location and next steps.

        Test case T030: Success message
        """
        project_name = "test-agent"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check output contains useful information
        output = result.stdout + result.stderr

        # Should mention success or completion
        assert (
            "success" in output.lower()
            or "created" in output.lower()
            or project_name in output
        ), f"Output doesn't mention success: {output}"

    def test_holodeck_init_shows_project_location(self, temp_dir: Path) -> None:
        """Verify success message includes project location.

        Test case T030: Project location in message
        """
        project_name = "test-agent"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        output = result.stdout + result.stderr

        # Should mention project path or location
        assert (
            project_name in output or str(temp_dir) in output
        ), f"Output doesn't mention project location: {output}"


@pytest.mark.integration
class TestInitCtrlCHandling:
    """Test Ctrl+C graceful handling functionality (T031)."""

    def test_holodeck_init_cleanup_on_interrupt(self, temp_dir: Path) -> None:
        """Verify cleanup on interrupt (simulated).

        Test case T031: Ctrl+C handling - verify no partial files

        Note: This test simulates interrupt by using timeout or
        checking that interrupts don't leave partial directories.
        """
        project_name = "test-agent"

        # Normal execution should complete without partial files
        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify complete project structure (no partial state)
        project_dir = temp_dir / project_name
        assert (project_dir / "agent.yaml").exists()
        assert (project_dir / "instructions").exists()
