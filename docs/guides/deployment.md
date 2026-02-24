# Deployment Guide

This guide covers deploying HoloDeck agents as containerized applications, including building images with the CLI and DIY manual deployment approaches.

## Overview

HoloDeck provides two deployment approaches:

- **CLI-based** (`holodeck deploy build`) - Automated image building from agent configuration
- **DIY Manual** - Build and deploy using the HoloDeck base image directly

Both approaches create OCI-compliant container images that run your agent as an HTTP server.

!!! warning "Claude Agent SDK Not Yet Supported"
    Container deployment currently supports **Semantic Kernel backends only** (OpenAI, Azure OpenAI, Ollama). Agents configured with `provider: anthropic` cannot be deployed as containers yet because `holodeck serve` — which the container entrypoint relies on — does not yet support the Claude Agent SDK backend. Claude agent deployment is planned for a future release.

## Quick Start

```bash
# Build a container image from your agent configuration
holodeck deploy build agent.yaml

# Run the built image locally
docker run -p 8080:8080 ghcr.io/myorg/my-agent:abc1234

# Preview build without executing
holodeck deploy build agent.yaml --dry-run
```

---

## Deploy Build Command

The `holodeck deploy build` command generates a Dockerfile and builds a container image from your agent configuration.

### CLI Reference

```bash
holodeck deploy build <agent_config> [OPTIONS]
```

### Arguments

| Argument       | Description                           | Default      |
| -------------- | ------------------------------------- | ------------ |
| `agent_config` | Path to agent.yaml configuration file | `agent.yaml` |

### Options

| Option        | Description                                   | Default |
| ------------- | --------------------------------------------- | ------- |
| `--tag`       | Custom tag for the image (overrides strategy) | None    |
| `--no-cache`  | Build without using Docker cache              | `false` |
| `--dry-run`   | Show what would be built without executing    | `false` |
| `--verbose, -v` | Enable verbose debug logging                | `false` |
| `--quiet, -q` | Suppress progress output                      | `false` |

### Exit Codes

| Code | Meaning                         |
| ---- | ------------------------------- |
| 0    | Build successful                |
| 2    | Configuration error             |
| 3    | Docker or deployment error      |

### Examples

```bash
# Basic build (uses agent.yaml in current directory)
holodeck deploy build

# Build with custom tag
holodeck deploy build agent.yaml --tag v1.0.0

# Preview Dockerfile without building
holodeck deploy build --dry-run

# Build with verbose output
holodeck deploy build agent.yaml -v

# Build without cache (force fresh build)
holodeck deploy build --no-cache
```

---

## Configuration

Add a `deployment` section to your `agent.yaml` to configure container builds:

```yaml
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  file: instructions/system.md

deployment:
  runtime: container
  registry:
    url: ghcr.io
    repository: myorg/my-agent
    tag_strategy: git_sha
  target:
    provider: aws
    aws:
      region: us-east-1
  protocol: rest
  port: 8080
  environment:
    LOG_LEVEL: info
```

### Registry Configuration

| Field                    | Description                              | Required | Default   |
| ------------------------ | ---------------------------------------- | -------- | --------- |
| `url`                    | Registry URL (e.g., `ghcr.io`, `docker.io`) | Yes    | -         |
| `repository`             | Repository name (e.g., `org/agent-name`) | Yes      | -         |
| `tag_strategy`           | Tag generation strategy                  | No       | `git_sha` |
| `custom_tag`             | Custom tag (when strategy is `custom`)   | No       | -         |
| `credentials_env_prefix` | Env var prefix for registry credentials  | No       | -         |

### Tag Strategies

| Strategy   | Description                                | Example Output |
| ---------- | ------------------------------------------ | -------------- |
| `git_sha`  | First 7 characters of current commit SHA   | `abc1234`      |
| `git_tag`  | Latest git tag                             | `v1.2.0`       |
| `latest`   | Always uses "latest"                       | `latest`       |
| `custom`   | Uses `custom_tag` value                    | `my-tag`       |

### Protocol Options

| Protocol | Description                                      |
| -------- | ------------------------------------------------ |
| `rest`   | REST API with JSON/SSE responses                 |
| `ag-ui`  | AG-UI protocol for CopilotKit and similar tools  |
| `both`   | Exposes both protocols                           |

### Environment Variables

Add custom environment variables to the container:

```yaml
deployment:
  environment:
    LOG_LEVEL: debug
    CUSTOM_API_KEY: "${CUSTOM_API_KEY}"  # From shell environment
```

---

## Registry Push

!!! warning "Under Construction"
    The `holodeck deploy push` command is not yet implemented (planned for P2).
    For now, use Docker directly to push images.

### Planned Features

- Automatic registry authentication
- Support for multiple registries:
    - GitHub Container Registry (ghcr.io)
    - Docker Hub (docker.io)
    - Amazon ECR
    - Google Artifact Registry
    - Azure Container Registry

### Manual Push (Current Workaround)

```bash
# Build the image
holodeck deploy build agent.yaml

# Login to registry
docker login ghcr.io -u USERNAME

# Push the image
docker push ghcr.io/myorg/my-agent:abc1234
```

---

## Cloud Deployment

!!! warning "Under Construction"
    The `holodeck deploy run` command is not yet implemented (planned for P3).
    Cloud deployment configuration is available for future use.

### Planned Cloud Providers

| Provider | Service                | Status  |
| -------- | ---------------------- | ------- |
| AWS      | App Runner             | Planned |
| GCP      | Cloud Run              | Planned |
| Azure    | Container Apps         | Planned |

### AWS App Runner Configuration

```yaml
deployment:
  target:
    provider: aws
    aws:
      region: us-east-1
      cpu: 1                    # vCPU: 1, 2, or 4
      memory: 2048              # MB: 2048, 3072, 4096, 8192, 12288
      ecr_role_arn: arn:aws:iam::123456789012:role/AppRunnerECRAccess
      health_check_path: /health
      auto_scaling_min: 1
      auto_scaling_max: 5
```

### GCP Cloud Run Configuration

```yaml
deployment:
  target:
    provider: gcp
    gcp:
      project_id: my-gcp-project
      region: us-central1
      memory: 512Mi            # e.g., 512Mi, 1Gi, 2Gi
      cpu: 1
      concurrency: 80
      timeout: 300             # seconds
      min_instances: 0
      max_instances: 100
```

### Azure Container Apps Configuration

```yaml
deployment:
  target:
    provider: azure
    azure:
      subscription_id: 12345678-1234-1234-1234-123456789012
      resource_group: my-resource-group
      environment_name: my-container-env
      location: eastus
      cpu: 0.5                 # 0.25, 0.5, 0.75, 1.0, etc.
      memory: 1Gi
      ingress_external: true
      min_replicas: 0
      max_replicas: 10
```

---

## DIY Deployment

Build and deploy containers manually without the HoloDeck CLI.

### Base Image

HoloDeck provides a pre-built base image with all dependencies:

```
ghcr.io/justinbarias/holodeck-base:latest
```

The base image includes:

- Python 3.10+ with HoloDeck installed
- Health check utilities (`curl`)
- Non-root `holodeck` user for security
- Working directory at `/app`

### Required Files

Your build context needs these files:

```
build-context/
├── Dockerfile          # Container definition
├── agent.yaml          # Agent configuration
├── entrypoint.sh       # Startup script
└── instructions/       # Instruction files (if using file reference)
    └── system.md
```

### Dockerfile Template

```dockerfile
# HoloDeck Agent Container
FROM ghcr.io/justinbarias/holodeck-base:latest

# OCI Labels for container metadata
LABEL org.opencontainers.image.title="my-agent"
LABEL org.opencontainers.image.version="v1.0.0"
LABEL org.opencontainers.image.created="2025-01-01T00:00:00Z"
LABEL com.holodeck.managed="true"

# Switch to root for file operations
USER root

# Set working directory
WORKDIR /app

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy agent configuration
COPY agent.yaml /app/agent.yaml

# Copy instruction files (if any)
COPY instructions/ /app/instructions/

# Set environment variables
ENV HOLODECK_PORT="8080"
ENV HOLODECK_PROTOCOL="rest"
ENV HOLODECK_AGENT_CONFIG="/app/agent.yaml"

# Expose the configured port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Fix ownership and switch to non-root user
RUN chown -R holodeck:holodeck /app
USER holodeck

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Entrypoint Script

Create `entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Start the HoloDeck agent server
exec holodeck serve /app/agent.yaml \
    --host 0.0.0.0 \
    --port "${HOLODECK_PORT:-8080}" \
    --protocol "${HOLODECK_PROTOCOL:-rest}"
```

### OCI Labels

Standard OCI labels for container metadata:

| Label                               | Description               |
| ----------------------------------- | ------------------------- |
| `org.opencontainers.image.title`    | Agent name                |
| `org.opencontainers.image.version`  | Image version/tag         |
| `org.opencontainers.image.created`  | ISO 8601 build timestamp  |
| `org.opencontainers.image.source`   | Source URL or git SHA     |
| `com.holodeck.managed`              | Marker for HoloDeck images|

### Environment Variables

| Variable                  | Description                 | Default  |
| ------------------------- | --------------------------- | -------- |
| `HOLODECK_PORT`           | Port for the agent server   | `8080`   |
| `HOLODECK_PROTOCOL`       | Protocol type               | `rest`   |
| `HOLODECK_AGENT_CONFIG`   | Path to agent configuration | `/app/agent.yaml` |

### Build and Run

```bash
# Build the image
docker build -t my-agent:v1.0.0 .

# Run locally
docker run -p 8080:8080 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  my-agent:v1.0.0

# Run with custom port
docker run -p 3000:3000 \
  -e HOLODECK_PORT=3000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  my-agent:v1.0.0
```

### Docker Compose Example

```yaml
# docker-compose.yml
services:
  agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - HOLODECK_PORT=8080
      - HOLODECK_PROTOCOL=rest
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    restart: unless-stopped
```

```bash
# Start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f agent

# Stop
docker compose down
```

---

## Complete Examples

### Minimal Build Example

```yaml
# agent.yaml
name: simple-assistant
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "You are a helpful assistant."

deployment:
  registry:
    url: ghcr.io
    repository: myorg/simple-assistant
    tag_strategy: latest
  target:
    provider: aws
    aws:
      region: us-east-1
  port: 8080
```

```bash
holodeck deploy build agent.yaml
# Output: ghcr.io/myorg/simple-assistant:latest
```

### Build with Git SHA Tag

```yaml
# agent.yaml
name: research-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  file: instructions/research.md
tools:
  - type: vectorstore
    name: search_docs
    store: chroma
    collection: papers
    source: data/papers/

deployment:
  registry:
    url: ghcr.io
    repository: myorg/research-agent
    tag_strategy: git_sha
  target:
    provider: gcp
    gcp:
      project_id: my-project
      region: us-central1
  protocol: ag-ui
  port: 8080
```

```bash
holodeck deploy build agent.yaml -v
# Output: ghcr.io/myorg/research-agent:abc1234
```

### DIY with Custom Dockerfile

```dockerfile
# Dockerfile
FROM ghcr.io/justinbarias/holodeck-base:latest

USER root
WORKDIR /app

# Install additional dependencies
RUN pip install custom-package

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

COPY agent.yaml /app/agent.yaml
COPY instructions/ /app/instructions/
COPY data/ /app/data/

ENV HOLODECK_PORT="8080"
ENV HOLODECK_PROTOCOL="rest"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

RUN chown -R holodeck:holodeck /app
USER holodeck

ENTRYPOINT ["/app/entrypoint.sh"]
```

---

## Troubleshooting

### Docker Not Available

```
Error: Docker is not available
```

**Solution:** Ensure Docker Desktop is running or the Docker daemon is started:

```bash
# macOS/Windows: Start Docker Desktop

# Linux
sudo systemctl start docker
```

### No Deployment Section

```
Error: Configuration error
  No 'deployment' section found in agent configuration
```

**Solution:** Add a `deployment` section to your `agent.yaml`:

```yaml
deployment:
  registry:
    url: ghcr.io
    repository: myorg/my-agent
  target:
    provider: aws
    aws:
      region: us-east-1
```

### Git SHA Strategy Fails

```
Error: tag_generation failed
  Failed to get git SHA: not a git repository
```

**Solution:** Either initialize a git repository or use a different tag strategy:

```yaml
deployment:
  registry:
    tag_strategy: latest  # or use --tag flag
```

### Invalid Repository Name

```
Error: Invalid repository name: My-Agent
  Must contain only lowercase letters, numbers, '.', '_', '/', '-'
```

**Solution:** Use lowercase repository names:

```yaml
deployment:
  registry:
    repository: myorg/my-agent  # lowercase only
```

### Health Check Fails

If the container starts but health checks fail:

1. Check if the agent can start successfully:
   ```bash
   docker run -it --rm my-agent:latest holodeck serve --help
   ```

2. Verify required environment variables are set:
   ```bash
   docker run -p 8080:8080 \
     -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
     my-agent:latest
   ```

3. Check container logs:
   ```bash
   docker logs <container-id>
   ```

---

## Next Steps

- See [Agent Server Guide](serve.md) for detailed server configuration
- See [Agent Configuration Guide](agent-configuration.md) for agent.yaml reference
- See [Observability Guide](observability.md) for monitoring and tracing
