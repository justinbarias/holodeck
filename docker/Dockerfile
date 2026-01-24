# HoloDeck Base Image
# This Dockerfile creates the base image for HoloDeck agent containers.
# It provides:
# - Python 3.10+ runtime
# - UV package manager (fast pip replacement)
# - HoloDeck CLI installed
# - Non-root user for security
# - Health check endpoint
#
# Usage:
#   docker build -t holodeck-ai/base:latest -f docker/Dockerfile .
#
# Build Arguments:
#   HOLODECK_VERSION - Version of holodeck-ai to install (default: latest)

FROM python:3.10-slim

# Build argument for holodeck version (empty = latest from PyPI)
ARG HOLODECK_VERSION=""

# OCI Labels for container metadata
LABEL org.opencontainers.image.title="HoloDeck Base Image"
LABEL org.opencontainers.image.description="Base image for HoloDeck AI agents"
LABEL org.opencontainers.image.vendor="HoloDeck AI"
LABEL org.opencontainers.image.licenses="MIT"
LABEL com.holodeck.managed="true"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOLODECK_PORT=8080 \
    HOLODECK_PROTOCOL=rest \
    HOLODECK_AGENT_CONFIG=/app/agent.yaml

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && chmod +x /usr/local/bin/uv

# Create non-root user for security
RUN groupadd --gid 1000 holodeck \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home holodeck

# Set working directory
WORKDIR /app

# Install holodeck package using UV
# --prerelease=allow needed for azure-ai-agents dependency
RUN uv pip install --system --no-cache --prerelease=allow "holodeck-ai${HOLODECK_VERSION:+==$HOLODECK_VERSION}"

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
