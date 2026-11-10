# HoloDeck Base Image
# This Dockerfile creates the base image for HoloDeck agent containers.
# It provides:
# - Python 3.13 runtime
# - UV package manager (fast pip replacement)
# - HoloDeck CLI installed at one exact version
# - Non-root user for security
# - Health check endpoint
#
# Reproducibility (spec 042, FR-029):
# - The Python image is pinned by digest. The tag is only a readable hint: a
#   tag moves when Debian or Python ship a patch, a digest does not.
# - HoloDeck is installed as an exact version, never "latest" and never with
#   --prerelease. Dependency versions are constrained to the
#   constraints.txt that ships inside that HoloDeck release (generated from
#   its uv.lock), so one HoloDeck version always resolves the same packages.
# - /opt/holodeck/manifest.txt lists every installed package.
#
# Usage:
#   docker build --build-arg HOLODECK_VERSION=0.7.1 \
#     -t holodeck-ai/base:0.7.1 -f docker/Dockerfile docker
#
# Build Arguments:
#   HOLODECK_VERSION - Exact holodeck-ai release to install (required)
#   PYTHON_IMAGE     - Python base image, pinned by digest

ARG PYTHON_IMAGE=python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
FROM ${PYTHON_IMAGE}

# Exact holodeck-ai version from PyPI. The build fails when it is empty.
ARG HOLODECK_VERSION=""

# OCI Labels for container metadata
LABEL org.opencontainers.image.title="HoloDeck Base Image"
LABEL org.opencontainers.image.description="Base image for HoloDeck AI agents"
LABEL org.opencontainers.image.vendor="HoloDeck AI"
LABEL org.opencontainers.image.licenses="MIT"
LABEL com.holodeck.managed="true"
LABEL com.holodeck.version="${HOLODECK_VERSION}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOLODECK_PORT=8080 \
    HOLODECK_PROTOCOL=rest \
    HOLODECK_AGENT_CONFIG=/app/agent.yaml

# Install system dependencies
# bubblewrap (bwrap) is required by the Claude Agent SDK when
# CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 (HoloDeck default).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# Install UV from the pinned official image (no unpinned curl | sh)
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /uvx /usr/local/bin/

# Create non-root user for security
RUN groupadd --gid 1000 holodeck \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home holodeck

# Set working directory
WORKDIR /app

# Install holodeck-ai at the exact version, constrained to its own lock.
# Step 1 installs the package without dependencies so its shipped
# constraints.txt and overrides.txt are on disk; step 2 installs the
# dependencies against them.
# find_spec locates the package without importing it (dependencies are absent).
RUN set -eu; \
    test -n "${HOLODECK_VERSION}" \
      || { echo "HOLODECK_VERSION build argument is required (exact release, e.g. 0.7.1)" >&2; exit 1; }; \
    uv pip install --system --no-cache --no-deps "holodeck-ai==${HOLODECK_VERSION}"; \
    CONSTRAINTS="$(python -c "import importlib.util as u; print(u.find_spec('holodeck').submodule_search_locations[0] + '/deploy/constraints.txt')")"; \
    OVERRIDES="$(dirname "${CONSTRAINTS}")/overrides.txt"; \
    test -f "${CONSTRAINTS}" -a -f "${OVERRIDES}" \
      || { echo "holodeck-ai ${HOLODECK_VERSION} ships no deploy/constraints.txt or overrides.txt; use a release that includes them" >&2; exit 1; }; \
    uv pip install --system --no-cache --constraint "${CONSTRAINTS}" --override "${OVERRIDES}" "holodeck-ai==${HOLODECK_VERSION}"; \
    mkdir -p /opt/holodeck; \
    { echo "# holodeck-ai ${HOLODECK_VERSION}; extras: none"; uv pip freeze --system; } > /opt/holodeck/manifest.txt

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Change ownership to holodeck user
RUN chown -R holodeck:holodeck /app

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${HOLODECK_PORT}/health || exit 1

# Switch to non-root user
USER holodeck

# Default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
