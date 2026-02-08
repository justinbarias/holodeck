"""Container image builder for HoloDeck agents.

This module provides functionality to build container images from
HoloDeck agent configurations using the Docker SDK.
"""

from __future__ import annotations

import subprocess  # nosec B404
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker
from docker.errors import BuildError, DockerException

from holodeck.lib.errors import DeploymentError, DockerNotAvailableError
from holodeck.models.deployment import TagStrategy

if TYPE_CHECKING:
    from docker.models.images import Image


@dataclass
class BuildResult:
    """Result of a container image build operation.

    Attributes:
        image_id: The SHA256 ID of the built image
        image_name: The repository/image name
        tag: The image tag
        full_name: Full image reference (name:tag)
        log_lines: Build log output lines
    """

    image_id: str
    image_name: str
    tag: str
    full_name: str
    log_lines: list[str] = field(default_factory=list)

    @classmethod
    def from_image(
        cls,
        image: Image,
        image_name: str,
        tag: str,
        log_lines: list[str] | None = None,
    ) -> BuildResult:
        """Create BuildResult from a Docker image object.

        Args:
            image: Docker image object from build
            image_name: Repository/image name
            tag: Image tag
            log_lines: Optional build log lines

        Returns:
            BuildResult instance
        """
        image_id = image.id or ""
        return cls(
            image_id=image_id,
            image_name=image_name,
            tag=tag,
            full_name=f"{image_name}:{tag}",
            log_lines=log_lines or [],
        )


def generate_tag(strategy: TagStrategy, custom_tag: str | None = None) -> str:
    """Generate an image tag based on the specified strategy.

    Args:
        strategy: Tag generation strategy (git_sha, git_tag, latest, custom)
        custom_tag: Custom tag value when strategy is CUSTOM

    Returns:
        Generated tag string

    Raises:
        ValueError: If custom strategy is used without providing custom_tag
        DeploymentError: If git commands fail (not in repo, no tags, etc.)

    Example:
        >>> generate_tag(TagStrategy.LATEST)
        'latest'
        >>> generate_tag(TagStrategy.CUSTOM, custom_tag="v1.0.0")
        'v1.0.0'
    """
    if strategy == TagStrategy.LATEST:
        return "latest"

    if strategy == TagStrategy.CUSTOM:
        if not custom_tag:
            raise ValueError("custom_tag is required when using CUSTOM strategy")
        return custom_tag

    if strategy == TagStrategy.GIT_SHA:
        result = subprocess.run(  # noqa: S603  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DeploymentError(
                operation="tag_generation",
                message="Failed to get git SHA: not a git repository",
            )
        # Return first 7 characters of SHA
        return result.stdout.strip()[:7]

    if strategy == TagStrategy.GIT_TAG:
        result = subprocess.run(  # noqa: S603  # nosec B603 B607
            ["git", "describe", "--tags", "--abbrev=0"],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DeploymentError(
                operation="tag_generation",
                message="No git tags found. Create a tag first: git tag v1.0.0",
            )
        return result.stdout.strip()

    # Should not reach here, but handle gracefully
    raise ValueError(f"Unknown tag strategy: {strategy}")


def get_oci_labels(
    agent_name: str,
    version: str,
    source_sha: str | None = None,
) -> dict[str, str]:
    """Generate OCI-compliant container image labels.

    Args:
        agent_name: Name of the agent for image title
        version: Version string for the image
        source_sha: Optional git SHA for source tracking

    Returns:
        Dictionary of OCI labels

    Example:
        >>> labels = get_oci_labels("my-agent", "v1.0.0")
        >>> labels["org.opencontainers.image.title"]
        'my-agent'
    """
    created = datetime.now(timezone.utc).isoformat()

    labels = {
        "org.opencontainers.image.title": agent_name,
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.created": created,
        "com.holodeck.managed": "true",
    }

    if source_sha:
        # Use first 7 characters of SHA
        labels["org.opencontainers.image.source"] = source_sha[:7]

    return labels


class ContainerBuilder:
    """Builder for HoloDeck agent container images.

    Uses the Docker SDK to build container images from agent configurations.
    Handles Docker daemon connection, build execution, and error handling.

    Example:
        >>> builder = ContainerBuilder()
        >>> result = builder.build(
        ...     build_context="./build",
        ...     image_name="my-org/my-agent",
        ...     tag="v1.0.0",
        ... )
        >>> print(result.full_name)
        'my-org/my-agent:v1.0.0'
    """

    def __init__(self) -> None:
        """Initialize the container builder.

        Connects to the Docker daemon using the environment configuration.

        Raises:
            DockerNotAvailableError: If Docker daemon is not available
        """
        try:
            self.client = docker.from_env()  # type: ignore[attr-defined]
        except DockerException as e:
            raise DockerNotAvailableError(operation="init") from e

    def build(
        self,
        build_context: str,
        image_name: str,
        tag: str,
        labels: dict[str, str] | None = None,
        dockerfile: str = "Dockerfile",
        platform: str = "linux/amd64",
        **build_kwargs: Any,
    ) -> BuildResult:
        """Build a container image from the specified context.

        Args:
            build_context: Path to the build context directory
            image_name: Repository/image name for the built image
            tag: Tag for the built image
            labels: Optional OCI labels to apply
            dockerfile: Path to Dockerfile relative to context
            platform: Target platform for the image (default: linux/amd64)
            **build_kwargs: Additional arguments passed to Docker build

        Returns:
            BuildResult with image details and build logs

        Raises:
            DeploymentError: If build context doesn't exist or build fails
        """
        context_path = Path(build_context)
        if not context_path.exists():
            raise DeploymentError(
                operation="build",
                message=f"Build context not found: {build_context}",
            )

        full_tag = f"{image_name}:{tag}"

        try:
            image, build_logs = self.client.images.build(
                path=str(context_path),
                tag=full_tag,
                dockerfile=dockerfile,
                labels=labels or {},
                rm=True,  # Remove intermediate containers
                platform=platform,
                pull=True,  # Always pull base image to get correct platform
                **build_kwargs,
            )

            # Extract log lines from build output
            log_lines: list[str] = []
            for log_entry in build_logs:
                # Docker SDK returns dict[str, Any] for log entries
                if isinstance(log_entry, dict):
                    if "stream" in log_entry:
                        stream_val = log_entry["stream"]
                        if isinstance(stream_val, str):
                            log_lines.append(stream_val.rstrip("\n"))
                    elif "error" in log_entry:
                        log_lines.append(f"ERROR: {log_entry['error']}")

            return BuildResult.from_image(
                image=image,
                image_name=image_name,
                tag=tag,
                log_lines=log_lines,
            )

        except BuildError as e:
            raise DeploymentError(
                operation="build",
                message=f"Docker build failed: {e.msg}",
            ) from e
        except DockerException as e:
            raise DeploymentError(
                operation="build",
                message=f"Docker error during build: {e}",
            ) from e
