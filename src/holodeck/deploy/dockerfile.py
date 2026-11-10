"""Dockerfile generation for HoloDeck agents.

This module provides functionality to generate Dockerfiles for
containerizing HoloDeck agents with proper OCI labels and configuration.

Every generated image installs an exact HoloDeck version with the extras its
agent uses, constrained to that release's lock file (spec 042, FR-029).
"""

import re
from datetime import UTC, datetime

from jinja2 import Template

# Names of the pin files in the build context, and the path where the image
# keeps the package manifest (``uv pip freeze`` of the finished install).
CONSTRAINTS_CONTEXT_FILE = "holodeck-constraints.txt"
OVERRIDES_CONTEXT_FILE = "holodeck-overrides.txt"
MANIFEST_IMAGE_PATH = "/opt/holodeck/manifest.txt"

_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]*$")
_EXTRA_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_WHEEL_PATTERN = re.compile(r"^[A-Za-z0-9_.+!-]+\.whl$")

# Jinja2 template for generating Dockerfiles
HOLODECK_DOCKERFILE_TEMPLATE = """\
# HoloDeck Agent Container
# Auto-generated Dockerfile for {{ agent_name }}
# Generated at: {{ created }}

FROM {{ base_image }}

# OCI Labels for container metadata
LABEL org.opencontainers.image.title="{{ agent_name }}"
LABEL org.opencontainers.image.version="{{ version }}"
LABEL org.opencontainers.image.created="{{ created }}"
LABEL org.opencontainers.image.source="{{ source_url }}"
LABEL com.holodeck.managed="true"
LABEL com.holodeck.version="{{ holodeck_version }}"
LABEL com.holodeck.extras="{{ extras | join(',') }}"

# Switch to root for file operations
USER root

# Set working directory
WORKDIR /app

{% if needs_nodejs %}
# Install Node.js (required for Claude Agent SDK)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\
    && apt-get install -y --no-install-recommends nodejs \\
    && rm -rf /var/lib/apt/lists/*
{% endif %}

{% if holodeck_wheel %}
# Development build: install the wheel from the build context (not on PyPI)
COPY {{ holodeck_wheel }} /tmp/wheels/{{ holodeck_wheel }}
{% endif %}
COPY {{ constraints_file }} /tmp/holodeck-constraints.txt
COPY {{ overrides_file }} /tmp/holodeck-overrides.txt

# Install the exact HoloDeck version with this agent's extras. Dependency
# versions are constrained to that release's lock; no prerelease resolution.
# The package manifest of the result is kept in the image.
RUN uv pip install --system --no-cache \\
    --constraint /tmp/holodeck-constraints.txt \\
    --override /tmp/holodeck-overrides.txt \\
    "{{ requirement }}" \\
    && mkdir -p /opt/holodeck \\
    && { echo "{{ manifest_header }}"; uv pip freeze --system; } \\
        > {{ manifest_path }} \\
    && rm -rf /tmp/holodeck-constraints.txt /tmp/holodeck-overrides.txt /tmp/wheels

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy agent configuration
COPY agent.yaml /app/agent.yaml

{% if instruction_files %}
# Copy instruction files (read-only at runtime)
{% for file in instruction_files %}
COPY --chown=root:root {{ file }} /app/{{ file }}
RUN chmod a-w /app/{{ file }}
{% endfor %}
{% endif %}

{% if data_directories %}
# Copy data directories (read-only at runtime)
{% for dir in data_directories %}
COPY --chown=root:root {{ dir }} /app/{{ dir }}
RUN chmod -R a-w /app/{{ dir }}
{% endfor %}
{% endif %}

# Set environment variables
ENV HOLODECK_PORT="{{ port }}"
ENV HOLODECK_PROTOCOL="{{ protocol }}"
ENV HOLODECK_AGENT_CONFIG="/app/agent.yaml"
{% if environment %}
{% for key, value in environment.items() %}
ENV {{ key }}="{{ value }}"
{% endfor %}
{% endif %}

# Expose the configured port
EXPOSE {{ port }}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{{ port }}/health || exit 1

# Spec 034 P2a — writable scratch dir for SDK + tool subprocesses.
# Mounted as tmpfs by the ACA deployer.
RUN mkdir -p /var/holodeck/work && chown holodeck:holodeck /var/holodeck/work

USER holodeck

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
"""


def generate_dockerfile(
    agent_name: str,
    port: int,
    protocol: str,
    *,
    holodeck_version: str,
    holodeck_wheel: str | None = None,
    base_image: str = "ghcr.io/justinbarias/holodeck-base:latest",
    version: str = "0.0.0",
    source_url: str = "",
    instruction_files: list[str] | None = None,
    data_directories: list[str] | None = None,
    environment: dict[str, str] | None = None,
    needs_nodejs: bool = False,
    extras: list[str] | None = None,
) -> str:
    """Generate a Dockerfile for a HoloDeck agent.

    The build context must hold ``holodeck-constraints.txt`` and
    ``holodeck-overrides.txt`` (the pin files shipped with the HoloDeck version
    being installed) and, for a development build, the wheel named by
    ``holodeck_wheel``.

    Args:
        agent_name: Name of the agent for labeling
        port: Port to expose
        protocol: Protocol type (rest, ag-ui, both)
        holodeck_version: Exact HoloDeck version the image installs
        holodeck_wheel: File name of a wheel in the build context. Set for a
            development version that is not on PyPI; the wheel is installed
            instead of ``holodeck-ai==holodeck_version``.
        base_image: Base Docker image to use
        version: Version for OCI label
        source_url: Source URL for OCI label
        instruction_files: List of instruction file paths to copy
        data_directories: List of data directories to copy
        environment: Environment variables to set
        needs_nodejs: Whether to install Node.js (required for Claude Agent SDK)
        extras: holodeck-ai extras to install (e.g., ["chromadb"]); rendered
            sorted and without duplicates

    Returns:
        Generated Dockerfile content as a string

    Raises:
        ValueError: If the version, wheel name, or an extra name has characters
            that do not belong in a requirement.

    Example:
        >>> dockerfile = generate_dockerfile(
        ...     agent_name="my-agent",
        ...     port=8080,
        ...     protocol="rest",
        ...     holodeck_version="1.2.3",
        ...     instruction_files=["instructions.md"],
        ... )
        >>> print(dockerfile[:50])
        # HoloDeck Agent Container
        # Auto-generated Doc...
    """
    extras_list = sorted(set(extras or []))
    _validate_install(holodeck_version, holodeck_wheel, extras_list)

    # jinja2 types Template.__new__ as returning Any; annotate to keep render() typed.
    template: Template = Template(HOLODECK_DOCKERFILE_TEMPLATE)

    # Generate ISO 8601 timestamp
    created = datetime.now(UTC).isoformat()

    return template.render(
        agent_name=agent_name,
        port=port,
        protocol=protocol,
        base_image=base_image,
        version=version,
        source_url=source_url,
        created=created,
        instruction_files=instruction_files or [],
        data_directories=data_directories or [],
        environment=environment or {},
        needs_nodejs=needs_nodejs,
        extras=extras_list,
        holodeck_version=holodeck_version,
        holodeck_wheel=holodeck_wheel,
        requirement=_requirement(holodeck_version, holodeck_wheel, extras_list),
        manifest_header=(
            f"# holodeck-ai {holodeck_version}; extras: "
            f"{','.join(extras_list) or 'none'}"
        ),
        constraints_file=CONSTRAINTS_CONTEXT_FILE,
        overrides_file=OVERRIDES_CONTEXT_FILE,
        manifest_path=MANIFEST_IMAGE_PATH,
    )


def _validate_install(version: str, wheel: str | None, extras: list[str]) -> None:
    """Reject values that would change the meaning of the install line."""
    if not _VERSION_PATTERN.match(version):
        raise ValueError(f"Invalid HoloDeck version: {version!r}")
    if wheel is not None and not _WHEEL_PATTERN.match(wheel):
        raise ValueError(f"Invalid wheel file name: {wheel!r}")
    for extra in extras:
        if not _EXTRA_PATTERN.match(extra):
            raise ValueError(f"Invalid extra name: {extra!r}")


def _requirement(version: str, wheel: str | None, extras: list[str]) -> str:
    """Return the requirement the image installs: an exact pin or the wheel."""
    suffix = f"[{','.join(extras)}]" if extras else ""
    if wheel is not None:
        return f"/tmp/wheels/{wheel}{suffix}"  # noqa: S108  # nosec B108
    return f"holodeck-ai{suffix}=={version}"
