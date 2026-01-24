"""Unit tests for the holodeck deploy CLI command.

Tests cover:
- Deploy command group and build subcommand options
- Build command execution with mocked dependencies
- Dry run mode
- Error handling for ConfigError, DeploymentError, DockerNotAvailableError
- Helper functions for dockerfile generation and build context preparation
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from pytest import CaptureFixture

from holodeck.cli.commands.deploy import (
    _display_build_success,
    _generate_dockerfile_content,
    _prepare_build_context,
    build,
    deploy,
)
from holodeck.lib.errors import ConfigError, DeploymentError, DockerNotAvailableError
from holodeck.models.deployment import (
    AWSAppRunnerConfig,
    CloudProvider,
    CloudTargetConfig,
    DeploymentConfig,
    ProtocolType,
    RegistryConfig,
    TagStrategy,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_agent_config(tmp_path: Path) -> Path:
    """Create a temporary agent.yaml config file with deployment section."""
    agent_file = tmp_path / "agent.yaml"
    agent_file.write_text(
        """
name: test-agent
description: A test agent

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: You are a helpful assistant.

deployment:
  registry:
    url: ghcr.io
    repository: test-org/test-agent
    tag_strategy: latest
  target:
    provider: aws
    aws:
      region: us-east-1
  protocol: rest
  port: 8080
"""
    )
    return agent_file


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent configuration."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.description = "A test agent"
    agent.tools = None

    # Mock instructions
    agent.instructions = MagicMock()
    agent.instructions.file = None
    agent.instructions.inline = "You are a helpful assistant."

    # Mock deployment config
    deployment = DeploymentConfig(
        registry=RegistryConfig(
            url="ghcr.io",
            repository="test-org/test-agent",
            tag_strategy=TagStrategy.LATEST,
        ),
        target=CloudTargetConfig(
            provider=CloudProvider.AWS,
            aws=AWSAppRunnerConfig(region="us-east-1"),
        ),
        protocol=ProtocolType.REST,
        port=8080,
    )
    agent.deployment = deployment

    return agent


@pytest.fixture
def mock_build_result() -> MagicMock:
    """Create a mock build result."""
    result = MagicMock()
    result.image_id = "sha256:abc123def456789012345678901234567890"
    result.image_name = "ghcr.io/test-org/test-agent"
    result.tag = "latest"
    result.full_name = "ghcr.io/test-org/test-agent:latest"
    result.log_lines = ["Step 1/5: FROM python:3.10-slim", "Step 2/5: WORKDIR /app"]
    return result


class TestDeployCommandGroup:
    """Tests for deploy command group."""

    def test_deploy_group_shows_help(self, runner: CliRunner) -> None:
        """Test deploy group shows help when invoked without subcommand."""
        result = runner.invoke(deploy)
        assert result.exit_code == 0
        assert "build" in result.output
        assert "Build a container image" in result.output

    def test_deploy_group_help_option(self, runner: CliRunner) -> None:
        """Test deploy group --help option."""
        result = runner.invoke(deploy, ["--help"])
        assert result.exit_code == 0
        assert "Deploy HoloDeck agents" in result.output


class TestBuildCommandOptions:
    """Tests for build command options and parameters."""

    def test_build_command_help(self, runner: CliRunner) -> None:
        """Test build command help shows all options."""
        result = runner.invoke(build, ["--help"])
        assert result.exit_code == 0
        assert "--tag" in result.output
        assert "--no-cache" in result.output
        assert "--dry-run" in result.output
        assert "--verbose" in result.output
        assert "--quiet" in result.output

    def test_build_command_file_not_found(self, runner: CliRunner) -> None:
        """Test build command with non-existent config file."""
        result = runner.invoke(build, ["nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output


class TestBuildCommandExecution:
    """Tests for build command execution with mocked dependencies."""

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_success(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_build_result: MagicMock,
    ) -> None:
        """Test successful build command execution."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_get_labels.return_value = {"org.opencontainers.image.title": "test-agent"}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_build_result
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 0
        assert "Build Successful" in result.output
        mock_builder.build.assert_called_once()

    @patch("holodeck.cli.commands.deploy._generate_dockerfile_content")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_dry_run(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_dockerfile: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
    ) -> None:
        """Test build command dry run mode."""
        mock_load_agent.return_value = mock_agent
        mock_generate_dockerfile.return_value = "FROM python:3.10-slim\nWORKDIR /app"

        result = runner.invoke(build, [str(temp_agent_config), "--dry-run"])

        assert result.exit_code == 0
        assert "[DRY RUN]" in result.output
        assert "No image was built" in result.output
        assert "Generated Dockerfile" in result.output

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_custom_tag(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_build_result: MagicMock,
    ) -> None:
        """Test build command with custom tag overrides config."""
        mock_load_agent.return_value = mock_agent
        mock_get_labels.return_value = {}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_build_result.tag = "v1.0.0"
        mock_build_result.full_name = "ghcr.io/test-org/test-agent:v1.0.0"
        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_build_result
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config), "--tag", "v1.0.0"])

        assert result.exit_code == 0
        # Custom tag should be used, generate_tag should NOT be called
        mock_generate_tag.assert_not_called()
        # Verify the tag was passed to build
        build_call = mock_builder.build.call_args
        assert build_call.kwargs["tag"] == "v1.0.0"

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_quiet_mode(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_build_result: MagicMock,
    ) -> None:
        """Test build command quiet mode outputs only image name."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_get_labels.return_value = {}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_build_result
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config), "--quiet"])

        assert result.exit_code == 0
        # Quiet mode should just output the image name
        assert mock_build_result.full_name in result.output
        # Should NOT have verbose output
        assert "Build Configuration" not in result.output

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_verbose_shows_logs(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
        mock_build_result: MagicMock,
    ) -> None:
        """Test build command verbose mode shows build logs."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_get_labels.return_value = {}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_build_result
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config), "--verbose"])

        assert result.exit_code == 0
        assert "Build Output" in result.output
        assert "Step 1/5" in result.output


class TestBuildCommandErrorHandling:
    """Tests for build command error handling."""

    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_config_error(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
    ) -> None:
        """Test build command handles ConfigError gracefully."""
        mock_load_agent.side_effect = ConfigError("deployment", "Invalid configuration")

        result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 2
        assert "Configuration error" in result.output

    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_missing_deployment_section(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
    ) -> None:
        """Test build command error when deployment section is missing."""
        mock_agent.deployment = None
        mock_load_agent.return_value = mock_agent

        result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 2
        assert "deployment" in result.output.lower()

    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_docker_not_available(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_prepare_context: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
    ) -> None:
        """Test build command handles DockerNotAvailableError."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        with (
            patch(
                "holodeck.deploy.builder.ContainerBuilder",
                side_effect=DockerNotAvailableError("init"),
            ),
            patch("holodeck.cli.commands.deploy.shutil.rmtree"),
        ):
            result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 3
        assert "Docker is not available" in result.output

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_deployment_error(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
    ) -> None:
        """Test build command handles DeploymentError."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_get_labels.return_value = {}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_builder = MagicMock()
        mock_builder.build.side_effect = DeploymentError(
            operation="build", message="Docker build failed"
        )
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 3
        assert "build failed" in result.output

    @patch("holodeck.cli.commands.deploy.shutil.rmtree")
    @patch("holodeck.cli.commands.deploy._prepare_build_context")
    @patch("holodeck.deploy.builder.ContainerBuilder")
    @patch("holodeck.deploy.builder.get_oci_labels")
    @patch("holodeck.deploy.builder.generate_tag")
    @patch("holodeck.config.loader.ConfigLoader.load_agent_yaml")
    @patch("holodeck.cli.commands.deploy.setup_logging")
    def test_build_command_unexpected_error(
        self,
        _mock_setup_logging: MagicMock,
        mock_load_agent: MagicMock,
        mock_generate_tag: MagicMock,
        mock_get_labels: MagicMock,
        mock_builder_class: MagicMock,
        mock_prepare_context: MagicMock,
        _mock_rmtree: MagicMock,
        runner: CliRunner,
        temp_agent_config: Path,
        mock_agent: MagicMock,
    ) -> None:
        """Test build command handles unexpected errors."""
        mock_load_agent.return_value = mock_agent
        mock_generate_tag.return_value = "latest"
        mock_get_labels.return_value = {}
        mock_prepare_context.return_value = Path("/tmp/build")  # noqa: S108

        mock_builder = MagicMock()
        mock_builder.build.side_effect = RuntimeError("Unexpected error")
        mock_builder_class.return_value = mock_builder

        result = runner.invoke(build, [str(temp_agent_config)])

        assert result.exit_code == 3
        assert "Unexpected error" in result.output


class TestGenerateDockerfileContent:
    """Tests for _generate_dockerfile_content helper function."""

    def test_generate_dockerfile_basic(self, mock_agent: MagicMock) -> None:
        """Test basic dockerfile generation."""
        with patch("holodeck.deploy.dockerfile.generate_dockerfile") as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            result = _generate_dockerfile_content(
                mock_agent, mock_agent.deployment, "v1.0.0"
            )

            assert result == "FROM python:3.10-slim"
            mock_gen.assert_called_once()

    def test_generate_dockerfile_with_instruction_file(
        self, mock_agent: MagicMock
    ) -> None:
        """Test dockerfile generation includes instruction files."""
        mock_agent.instructions.file = "instructions.md"

        with patch("holodeck.deploy.dockerfile.generate_dockerfile") as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            _generate_dockerfile_content(mock_agent, mock_agent.deployment, "v1.0.0")

            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs["instruction_files"] == ["instructions.md"]

    def test_generate_dockerfile_with_vectorstore_tool(
        self, mock_agent: MagicMock
    ) -> None:
        """Test dockerfile generation includes data directories from vectorstores."""
        from holodeck.models.tool import VectorstoreTool

        vectorstore_tool = MagicMock(spec=VectorstoreTool)
        vectorstore_tool.source = "data/documents"
        mock_agent.tools = [vectorstore_tool]

        with patch("holodeck.deploy.dockerfile.generate_dockerfile") as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            _generate_dockerfile_content(mock_agent, mock_agent.deployment, "v1.0.0")

            call_kwargs = mock_gen.call_args.kwargs
            assert "data/documents/" in call_kwargs["data_directories"]


class TestPrepareBuildContext:
    """Tests for _prepare_build_context helper function."""

    def test_prepare_build_context_creates_dockerfile(
        self, tmp_path: Path, mock_agent: MagicMock
    ) -> None:
        """Test build context includes generated Dockerfile."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text("name: test")

        with patch(
            "holodeck.cli.commands.deploy._generate_dockerfile_content"
        ) as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim\nWORKDIR /app"

            build_dir = _prepare_build_context(
                mock_agent, mock_agent.deployment, agent_dir, "v1.0.0"
            )

            try:
                dockerfile = build_dir / "Dockerfile"
                assert dockerfile.exists()
                assert "FROM python:3.10-slim" in dockerfile.read_text()
            finally:
                import shutil

                shutil.rmtree(build_dir, ignore_errors=True)

    def test_prepare_build_context_copies_agent_yaml(
        self, tmp_path: Path, mock_agent: MagicMock
    ) -> None:
        """Test build context copies agent.yaml."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text("name: test-agent")

        with patch(
            "holodeck.cli.commands.deploy._generate_dockerfile_content"
        ) as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            build_dir = _prepare_build_context(
                mock_agent, mock_agent.deployment, agent_dir, "v1.0.0"
            )

            try:
                agent_yaml = build_dir / "agent.yaml"
                assert agent_yaml.exists()
                assert "test-agent" in agent_yaml.read_text()
            finally:
                import shutil

                shutil.rmtree(build_dir, ignore_errors=True)

    def test_prepare_build_context_creates_entrypoint(
        self, tmp_path: Path, mock_agent: MagicMock
    ) -> None:
        """Test build context creates entrypoint script."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text("name: test")

        with patch(
            "holodeck.cli.commands.deploy._generate_dockerfile_content"
        ) as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            build_dir = _prepare_build_context(
                mock_agent, mock_agent.deployment, agent_dir, "v1.0.0"
            )

            try:
                entrypoint = build_dir / "entrypoint.sh"
                assert entrypoint.exists()
                content = entrypoint.read_text()
                assert "#!/bin/bash" in content
                assert "holodeck serve" in content
            finally:
                import shutil

                shutil.rmtree(build_dir, ignore_errors=True)

    def test_prepare_build_context_copies_instruction_file(
        self, tmp_path: Path, mock_agent: MagicMock
    ) -> None:
        """Test build context copies instruction files."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text("name: test")
        (agent_dir / "instructions.md").write_text("# Instructions")

        mock_agent.instructions.file = "instructions.md"

        with patch(
            "holodeck.cli.commands.deploy._generate_dockerfile_content"
        ) as mock_gen:
            mock_gen.return_value = "FROM python:3.10-slim"

            build_dir = _prepare_build_context(
                mock_agent, mock_agent.deployment, agent_dir, "v1.0.0"
            )

            try:
                instructions = build_dir / "instructions.md"
                assert instructions.exists()
                assert "# Instructions" in instructions.read_text()
            finally:
                import shutil

                shutil.rmtree(build_dir, ignore_errors=True)


class TestDisplayBuildSuccess:
    """Tests for _display_build_success helper function."""

    def test_display_build_success_full_output(
        self, mock_build_result: MagicMock, capsys: CaptureFixture[str]
    ) -> None:
        """Test full build success message display."""
        _display_build_success(mock_build_result, quiet=False)

        captured = capsys.readouterr()
        output = captured.out

        assert "Build Successful" in output
        assert mock_build_result.full_name in output
        assert "docker run" in output
        assert "docker push" in output

    def test_display_build_success_quiet_mode(
        self, mock_build_result: MagicMock, capsys: CaptureFixture[str]
    ) -> None:
        """Test quiet mode only outputs image name."""
        _display_build_success(mock_build_result, quiet=True)

        captured = capsys.readouterr()
        output = captured.out

        assert output.strip() == mock_build_result.full_name
        assert "Build Successful" not in output
        assert "Next steps" not in output
