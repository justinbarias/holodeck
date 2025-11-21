"""Integration tests for basic project creation (Phase 3: US1).

Tests for the holodeck init command with:
- Basic project creation
- Default template selection
- Overwrite behavior
- Success messages
- Ctrl+C handling

Refactored to use init_project fixture and eliminate redundant subprocess calls.
"""

import pytest


@pytest.mark.integration
class TestInitBasicProjectCreation:
    """Test basic project creation functionality (T027).

    Consolidates directory, agent.yaml, and folder structure tests to use
    shared init_project fixture instead of creating projects multiple times.
    """

    @pytest.mark.parametrize(
        "check_type,path_getter,validation",
        [
            ("directory", lambda p: p, lambda p: p.exists() and p.is_dir()),
            (
                "agent_yaml",
                lambda p: p / "agent.yaml",
                lambda p: p.exists() and p.is_file() and "name:" in p.read_text(),
            ),
            (
                "instructions_dir",
                lambda p: p / "instructions",
                lambda p: p.exists() and p.is_dir(),
            ),
            ("tools_dir", lambda p: p / "tools", lambda p: p.exists() and p.is_dir()),
            ("data_dir", lambda p: p / "data", lambda p: p.exists() and p.is_dir()),
        ],
    )
    def test_holodeck_init_creates_structure(
        self, init_project, check_type, path_getter, validation
    ) -> None:
        """Verify `holodeck init` creates complete project structure.

        Test case T027: Basic project creation
        - Creates project directory
        - Creates valid agent.yaml file
        - Creates all required folders (instructions, tools, data)

        Uses parameterization to test all structure components with
        single subprocess call.
        """
        project_dir, result = init_project("test-agent")

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Get target path and validate
        target_path = path_getter(project_dir)
        assert validation(
            target_path
        ), f"Validation failed for {check_type}: {target_path}"


@pytest.mark.integration
class TestInitDefaultTemplate:
    """Test default template selection functionality (T028)."""

    def test_holodeck_init_uses_conversational_by_default(self, init_project) -> None:
        """Verify conversational is default template when --template omitted.

        Test case T028: Default template selection
        """
        project_dir, result = init_project("test-agent")

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check that the default template files were created
        agent_yaml = project_dir / "agent.yaml"
        assert agent_yaml.exists(), "agent.yaml not created"

        content = agent_yaml.read_text()
        # Conversational template should have basic configuration
        assert "provider:" in content or "name:" in content
        assert (
            project_dir / "instructions" / "system-prompt.md"
        ).exists(), "system-prompt.md not created"

    def test_holodeck_init_respects_template_option(self, init_project) -> None:
        """Verify --template option allows template selection.

        Test case T028: Template option functionality
        """
        # Try with research template
        project_dir, result = init_project("test-agent", template="research")

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify project was created
        assert project_dir.exists(), "Project directory not created"
        assert (project_dir / "agent.yaml").exists(), "agent.yaml not created"


@pytest.mark.integration
class TestInitOverwriteBehavior:
    """Test overwrite behavior functionality (T029)."""

    def test_holodeck_init_fails_if_directory_exists_without_force(
        self, init_project
    ) -> None:
        """Verify command fails when directory exists without --force flag.

        Test case T029: Overwrite behavior - error without force
        """
        project_name = "test-agent"

        # Create first project
        project_dir, result1 = init_project(project_name)
        assert result1.returncode == 0, f"First init failed: {result1.stderr}"

        # Try to create again without force - should fail
        _, result2 = init_project(project_name)

        assert result2.returncode != 0, "Command should fail when directory exists"
        assert (
            "already exists" in result2.stderr.lower()
            or "already exists" in result2.stdout.lower()
        ), f"Error message doesn't mention 'already exists': {result2.stderr}"

    def test_holodeck_init_overwrites_with_force_flag(self, init_project) -> None:
        """Verify --force flag allows overwriting existing directory.

        Test case T029: Overwrite behavior - success with force
        """
        project_name = "test-agent"

        # Create first project
        project_dir, result1 = init_project(project_name)
        assert result1.returncode == 0, f"First init failed: {result1.stderr}"

        # Create marker file to verify overwrite
        marker_file = project_dir / "marker.txt"
        marker_file.write_text("old content")
        assert marker_file.exists(), "Marker file not created"

        # Create again with force - should succeed
        project_dir, result2 = init_project(project_name, force=True)

        assert result2.returncode == 0, f"Command failed with --force: {result2.stderr}"

        # Marker file should be gone (directory was replaced)
        assert not marker_file.exists(), "Old files not removed during overwrite"


@pytest.mark.integration
class TestInitSuccessMessage:
    """Test success message output functionality (T030)."""

    def test_holodeck_init_displays_success_message(self, init_project) -> None:
        """Verify output shows success indication.

        Test case T030: Success message
        """
        project_name = "test-agent"

        project_dir, result = init_project(project_name)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check output contains useful information
        output = result.stdout + result.stderr

        # Should mention success or completion
        assert (
            "success" in output.lower()
            or "created" in output.lower()
            or project_name in output
        ), f"Output doesn't mention success: {output}"

    def test_holodeck_init_shows_project_location(self, init_project) -> None:
        """Verify success message includes project information.

        Test case T030: Project location in message
        """
        project_name = "test-agent"

        project_dir, result = init_project(project_name)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        output = result.stdout + result.stderr

        # Should mention project name or path
        assert project_name in output, f"Output doesn't mention project name: {output}"


@pytest.mark.integration
class TestInitCtrlCHandling:
    """Test Ctrl+C graceful handling functionality (T031)."""

    def test_holodeck_init_cleanup_on_interrupt(self, init_project) -> None:
        """Verify cleanup on interrupt (simulated).

        Test case T031: Ctrl+C handling - verify no partial files

        Note: This test verifies that normal completion results in a complete
        project structure. Actual interrupt testing would require signal handling
        or timeout simulation which is not easily testable in subprocess.
        """
        project_name = "test-agent"

        # Normal execution should complete without partial files
        project_dir, result = init_project(project_name)

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify complete project structure (no partial state)
        assert (project_dir / "agent.yaml").exists(), "agent.yaml missing"
        assert (project_dir / "instructions").exists(), "instructions dir missing"
        assert (project_dir / "tools").exists(), "tools dir missing"
        assert (project_dir / "data").exists(), "data dir missing"
