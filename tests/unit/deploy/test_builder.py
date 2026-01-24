"""Unit tests for ContainerBuilder.

TDD tests - written before implementation to drive the design.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

if TYPE_CHECKING:
    from holodeck.deploy.builder import ContainerBuilder


class TestGenerateTag:
    """Tests for tag generation based on strategy."""

    def test_generate_tag_git_sha(self) -> None:
        """Test git SHA tag generation."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0, stdout="abc1234def5678901234567890123456789abcde\n"
            )

            tag = generate_tag(TagStrategy.GIT_SHA)

            assert tag == "abc1234"  # First 7 chars of SHA
            mock_run.assert_called_once()
            # Verify git rev-parse was called
            assert "git" in mock_run.call_args[0][0]
            assert "rev-parse" in mock_run.call_args[0][0]

    def test_generate_tag_git_sha_short(self) -> None:
        """Test git SHA returns 7 character short hash."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0, stdout="1234567890abcdef1234567890abcdef12345678\n"
            )

            tag = generate_tag(TagStrategy.GIT_SHA)

            assert len(tag) == 7
            assert tag == "1234567"

    def test_generate_tag_git_tag(self) -> None:
        """Test git tag strategy."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="v1.2.3\n")

            tag = generate_tag(TagStrategy.GIT_TAG)

            assert tag == "v1.2.3"
            mock_run.assert_called_once()
            assert "describe" in mock_run.call_args[0][0]

    def test_generate_tag_latest(self) -> None:
        """Test latest tag strategy."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        tag = generate_tag(TagStrategy.LATEST)

        assert tag == "latest"

    def test_generate_tag_custom(self) -> None:
        """Test custom tag strategy."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        tag = generate_tag(TagStrategy.CUSTOM, custom_tag="my-custom-tag")

        assert tag == "my-custom-tag"

    def test_generate_tag_custom_missing_raises(self) -> None:
        """Test custom tag without value raises error."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.models.deployment import TagStrategy

        with pytest.raises(ValueError, match="custom_tag is required"):
            generate_tag(TagStrategy.CUSTOM)

    def test_generate_tag_git_sha_not_in_repo(self) -> None:
        """Test git SHA when not in a git repository."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.lib.errors import DeploymentError
        from holodeck.models.deployment import TagStrategy

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=128, stdout="", stderr="fatal: not a git repository"
            )

            with pytest.raises(DeploymentError, match="not a git repository"):
                generate_tag(TagStrategy.GIT_SHA)

    def test_generate_tag_git_tag_no_tags(self) -> None:
        """Test git tag when no tags exist."""
        from holodeck.deploy.builder import generate_tag
        from holodeck.lib.errors import DeploymentError
        from holodeck.models.deployment import TagStrategy

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=128, stdout="", stderr="fatal: No names found"
            )

            with pytest.raises(DeploymentError, match="No git tags found"):
                generate_tag(TagStrategy.GIT_TAG)


class TestContainerBuilderInit:
    """Tests for ContainerBuilder initialization."""

    def test_init_success(self) -> None:
        """Test successful initialization with Docker available."""
        from holodeck.deploy.builder import ContainerBuilder

        with patch("docker.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_from_env.return_value = mock_client

            builder = ContainerBuilder()

            assert builder.client == mock_client
            mock_from_env.assert_called_once()

    def test_init_docker_not_available(self) -> None:
        """Test initialization when Docker is not available."""
        from docker.errors import DockerException
        from holodeck.deploy.builder import ContainerBuilder
        from holodeck.lib.errors import DockerNotAvailableError

        with patch("docker.from_env") as mock_from_env:
            mock_from_env.side_effect = DockerException("Cannot connect to Docker")

            with pytest.raises(DockerNotAvailableError):
                ContainerBuilder()

    def test_init_docker_connection_refused(self) -> None:
        """Test initialization when Docker daemon is not running."""
        from docker.errors import DockerException
        from holodeck.deploy.builder import ContainerBuilder
        from holodeck.lib.errors import DockerNotAvailableError

        with patch("docker.from_env") as mock_from_env:
            mock_from_env.side_effect = DockerException(
                "Error while fetching server API version"
            )

            with pytest.raises(DockerNotAvailableError):
                ContainerBuilder()


class TestContainerBuilderBuild:
    """Tests for ContainerBuilder.build() method."""

    @pytest.fixture
    def mock_docker_client(self) -> MagicMock:
        """Create a mock Docker client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def builder(self, mock_docker_client: MagicMock) -> ContainerBuilder:
        """Create a ContainerBuilder with mocked client."""
        from holodeck.deploy.builder import ContainerBuilder

        with patch("docker.from_env") as mock_from_env:
            mock_from_env.return_value = mock_docker_client
            return ContainerBuilder()

    def test_build_success(
        self, builder: ContainerBuilder, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful image build."""
        # Create a mock build context
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "Dockerfile").write_text("FROM python:3.10-slim\n")
        (build_dir / "agent.yaml").write_text("name: test\n")

        # Mock the build method to return an image and logs
        mock_image = MagicMock()
        mock_image.id = "sha256:abc123"
        mock_image.tags = ["test-org/test-agent:abc1234"]

        # Mock streaming build output
        mock_docker_client.images.build.return_value = (
            mock_image,
            [{"stream": "Step 1/5 : FROM python:3.10-slim\n"}],
        )

        result = builder.build(
            build_context=str(build_dir),
            image_name="test-org/test-agent",
            tag="abc1234",
        )

        assert result.image_id == "sha256:abc123"
        assert result.image_name == "test-org/test-agent"
        assert result.tag == "abc1234"
        mock_docker_client.images.build.assert_called_once()

    def test_build_with_labels(
        self, builder: ContainerBuilder, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test build with OCI labels."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "Dockerfile").write_text("FROM python:3.10-slim\n")

        mock_image = MagicMock()
        mock_image.id = "sha256:def456"
        mock_image.tags = ["test-org/test-agent:v1.0"]
        mock_docker_client.images.build.return_value = (mock_image, [])

        labels = {
            "org.opencontainers.image.title": "test-agent",
            "org.opencontainers.image.version": "v1.0",
        }

        builder.build(
            build_context=str(build_dir),
            image_name="test-org/test-agent",
            tag="v1.0",
            labels=labels,
        )

        # Verify labels were passed to build
        call_kwargs = mock_docker_client.images.build.call_args[1]
        assert call_kwargs["labels"] == labels

    def test_build_yields_log_lines(
        self, builder: ContainerBuilder, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test that build yields streaming log lines."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "Dockerfile").write_text("FROM python:3.10-slim\n")

        mock_image = MagicMock()
        mock_image.id = "sha256:ghi789"
        mock_image.tags = ["test-org/test-agent:latest"]

        # Mock streaming output with multiple log lines
        mock_docker_client.images.build.return_value = (
            mock_image,
            [
                {"stream": "Step 1/5 : FROM python:3.10-slim\n"},
                {"stream": " ---> abc123\n"},
                {"stream": "Step 2/5 : WORKDIR /app\n"},
            ],
        )

        result = builder.build(
            build_context=str(build_dir),
            image_name="test-org/test-agent",
            tag="latest",
        )

        # Log lines should be captured in result
        assert len(result.log_lines) == 3
        assert "Step 1/5" in result.log_lines[0]

    def test_build_error_handling(
        self, builder: ContainerBuilder, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test build error is wrapped in DeploymentError."""
        from docker.errors import BuildError
        from holodeck.lib.errors import DeploymentError

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "Dockerfile").write_text("FROM python:3.10-slim\n")

        mock_docker_client.images.build.side_effect = BuildError(
            reason="Build failed", build_log=[]
        )

        with pytest.raises(DeploymentError, match="build"):
            builder.build(
                build_context=str(build_dir),
                image_name="test-org/test-agent",
                tag="latest",
            )

    def test_build_missing_context(
        self, builder: ContainerBuilder, mock_docker_client: MagicMock
    ) -> None:
        """Test build with missing context directory."""
        from holodeck.lib.errors import DeploymentError

        with pytest.raises(DeploymentError, match="Build context not found"):
            builder.build(
                build_context="/nonexistent/path",
                image_name="test-org/test-agent",
                tag="latest",
            )


class TestGetOCILabels:
    """Tests for OCI label generation."""

    def test_get_oci_labels_basic(self) -> None:
        """Test basic OCI label generation."""
        from holodeck.deploy.builder import get_oci_labels

        labels = get_oci_labels(
            agent_name="my-agent",
            version="v1.0.0",
        )

        assert labels["org.opencontainers.image.title"] == "my-agent"
        assert labels["org.opencontainers.image.version"] == "v1.0.0"
        assert "org.opencontainers.image.created" in labels
        assert labels["com.holodeck.managed"] == "true"

    def test_get_oci_labels_with_source(self) -> None:
        """Test OCI labels with source URL."""
        from holodeck.deploy.builder import get_oci_labels

        labels = get_oci_labels(
            agent_name="my-agent",
            version="v1.0.0",
            source_sha="abc1234def5678901234567890123456789abcde",
        )

        assert labels["org.opencontainers.image.source"] == "abc1234"

    def test_get_oci_labels_timestamp_format(self) -> None:
        """Test OCI labels created timestamp is ISO 8601."""
        from datetime import datetime

        from holodeck.deploy.builder import get_oci_labels

        labels = get_oci_labels(
            agent_name="my-agent",
            version="v1.0.0",
        )

        # Should be valid ISO 8601
        created = labels["org.opencontainers.image.created"]
        try:
            datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {created}")

    def test_get_oci_labels_always_has_managed(self) -> None:
        """Test that com.holodeck.managed is always set to 'true'."""
        from holodeck.deploy.builder import get_oci_labels

        labels = get_oci_labels(
            agent_name="test",
            version="1.0",
        )

        assert labels["com.holodeck.managed"] == "true"


class TestBuildResult:
    """Tests for BuildResult dataclass."""

    def test_build_result_attributes(self) -> None:
        """Test BuildResult has expected attributes."""
        from holodeck.deploy.builder import BuildResult

        result = BuildResult(
            image_id="sha256:abc123",
            image_name="test-org/test-agent",
            tag="v1.0.0",
            full_name="test-org/test-agent:v1.0.0",
            log_lines=["Step 1/5 : FROM python:3.10-slim"],
        )

        assert result.image_id == "sha256:abc123"
        assert result.image_name == "test-org/test-agent"
        assert result.tag == "v1.0.0"
        assert result.full_name == "test-org/test-agent:v1.0.0"
        assert len(result.log_lines) == 1

    def test_build_result_from_image(self) -> None:
        """Test BuildResult creation from Docker image object."""
        from holodeck.deploy.builder import BuildResult

        mock_image = MagicMock()
        mock_image.id = "sha256:xyz789"
        mock_image.tags = ["my-org/my-agent:latest"]

        result = BuildResult.from_image(
            image=mock_image,
            image_name="my-org/my-agent",
            tag="latest",
            log_lines=["Build complete"],
        )

        assert result.image_id == "sha256:xyz789"
        assert result.image_name == "my-org/my-agent"
        assert result.tag == "latest"
        assert result.full_name == "my-org/my-agent:latest"
