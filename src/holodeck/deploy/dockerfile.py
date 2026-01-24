"""Dockerfile generation for HoloDeck agents.

This module provides functionality to generate Dockerfiles for
containerizing HoloDeck agents with proper OCI labels and configuration.
"""

from datetime import datetime, timezone

from jinja2 import Template

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

# Set working directory
WORKDIR /app

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy agent configuration
COPY agent.yaml /app/agent.yaml

{% if instruction_files %}
# Copy instruction files
{% for file in instruction_files %}
COPY {{ file }} /app/{{ file }}
{% endfor %}
{% endif %}

{% if data_directories %}
# Copy data directories
{% for dir in data_directories %}
COPY {{ dir }} /app/{{ dir }}
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

# Switch to non-root user (holodeck user from base image)
USER holodeck

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
"""


def generate_dockerfile(
    agent_name: str,
    port: int,
    protocol: str,
    *,
    base_image: str = "ghcr.io/justinbarias/holodeck-base:latest",
    version: str = "0.0.0",
    source_url: str = "",
    instruction_files: list[str] | None = None,
    data_directories: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    """Generate a Dockerfile for a HoloDeck agent.

    Args:
        agent_name: Name of the agent for labeling
        port: Port to expose
        protocol: Protocol type (rest, ag-ui, both)
        base_image: Base Docker image to use
        version: Version for OCI label
        source_url: Source URL for OCI label
        instruction_files: List of instruction file paths to copy
        data_directories: List of data directories to copy
        environment: Environment variables to set

    Returns:
        Generated Dockerfile content as a string

    Example:
        >>> dockerfile = generate_dockerfile(
        ...     agent_name="my-agent",
        ...     port=8080,
        ...     protocol="rest",
        ...     instruction_files=["instructions.md"],
        ... )
        >>> print(dockerfile[:50])
        # HoloDeck Agent Container
        # Auto-generated Doc...
    """
    template = Template(HOLODECK_DOCKERFILE_TEMPLATE)

    # Generate ISO 8601 timestamp
    created = datetime.now(timezone.utc).isoformat()

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
    )
