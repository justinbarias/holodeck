# HoloDeck Base Image

The HoloDeck base image provides a pre-configured runtime environment for deploying HoloDeck AI agents as containers.

## Image Location

```
ghcr.io/justinbarias/holodeck-base
```

## Features

- Python 3.10+ runtime
- UV package manager (fast pip replacement)
- HoloDeck CLI pre-installed
- Non-root user for security (`holodeck` user, UID 1000)
- Health check endpoint support
- Multi-platform support (amd64, arm64)

## Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest build from main branch |
| `<sha>` | Specific commit build (e.g., `a1b2c3d`) |
| `<version>` | Release version (e.g., `0.1.0`, `1.0.0`) |
| `<major>.<minor>` | Minor version (e.g., `0.1`, `1.0`) |

### Versioning Strategy

- **latest**: Always points to the most recent build from the `main` branch. Suitable for development and testing.
- **SHA tags**: Useful for pinning to a specific commit for reproducibility.
- **Semantic versioning**: Use these tags for production deployments. Tags like `0.1.0` are immutable.

## Usage

### Basic Usage

Mount your agent configuration and run:

```bash
docker run -v $(pwd)/agent.yaml:/app/agent.yaml \
  -p 8080:8080 \
  ghcr.io/justinbarias/holodeck-base:latest
```

### With Environment Variables

```bash
docker run -v $(pwd)/agent.yaml:/app/agent.yaml \
  -e HOLODECK_PORT=3000 \
  -e HOLODECK_PROTOCOL=rest \
  -e HOLODECK_LOG_LEVEL=debug \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -p 3000:3000 \
  ghcr.io/justinbarias/holodeck-base:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  agent:
    image: ghcr.io/justinbarias/holodeck-base:latest
    volumes:
      - ./agent.yaml:/app/agent.yaml:ro
      - ./instructions.md:/app/instructions.md:ro
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
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOLODECK_PORT` | `8080` | Port the server listens on |
| `HOLODECK_PROTOCOL` | `rest` | Protocol type: `rest`, `ag-ui`, or `both` |
| `HOLODECK_AGENT_CONFIG` | `/app/agent.yaml` | Path to agent configuration file |
| `HOLODECK_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warning`, `error` |

### LLM Provider API Keys

Depending on your agent configuration, you may need to provide API keys:

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OLLAMA_HOST` | Ollama (for self-hosted models) |

## Building Locally

### Using the Build Script

```bash
# Build with default tag (latest)
./scripts/build-base-image.sh

# Build with specific tag
./scripts/build-base-image.sh --tag 0.1.0

# Build with specific HoloDeck version
./scripts/build-base-image.sh --tag 0.1.0 --version 0.1.0

# Build and push to registry
./scripts/build-base-image.sh --push --tag 0.1.0

# Build for specific platform
./scripts/build-base-image.sh --platform linux/arm64

# Build without cache
./scripts/build-base-image.sh --no-cache
```

### Using Docker Directly

```bash
# From project root
docker build -t ghcr.io/justinbarias/holodeck-base:test -f docker/Dockerfile docker/

# With specific HoloDeck version
docker build --build-arg HOLODECK_VERSION=0.1.0 \
  -t ghcr.io/justinbarias/holodeck-base:0.1.0 \
  -f docker/Dockerfile docker/
```

## Manual Testing

Follow these steps to verify the base image works correctly:

### 1. Build the Image

```bash
./scripts/build-base-image.sh --tag test
```

### 2. Verify Image Created

```bash
docker images | grep holodeck-base
```

### 3. Create a Sample Agent Configuration

```bash
mkdir -p /tmp/holodeck-test
cat > /tmp/holodeck-test/agent.yaml << 'EOF'
name: test-agent
description: Test agent for base image validation

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7

instructions:
  inline: "You are a helpful assistant."
EOF
```

### 4. Run the Container

```bash
docker run -d --name holodeck-test \
  -v /tmp/holodeck-test/agent.yaml:/app/agent.yaml \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -p 8080:8080 \
  ghcr.io/justinbarias/holodeck-base:test
```

### 5. Check Container Logs

```bash
docker logs holodeck-test
```

### 6. Test Health Endpoint

```bash
# Note: Requires 'holodeck serve' command to be implemented
curl http://localhost:8080/health
```

### 7. Cleanup

```bash
docker stop holodeck-test && docker rm holodeck-test
rm -rf /tmp/holodeck-test
```

> **Note:** Full automated testing is blocked until the `holodeck serve` command is implemented. The steps above are for manual verification of the image build and container startup.

## Troubleshooting

### Container Exits Immediately

Check that your agent configuration file is mounted correctly:

```bash
docker run -it ghcr.io/justinbarias/holodeck-base:latest ls -la /app/
```

### Permission Denied Errors

The container runs as user `holodeck` (UID 1000). Ensure mounted files are readable:

```bash
chmod 644 agent.yaml
```

### Health Check Failing

Verify the server is running on the expected port:

```bash
docker exec <container_id> curl -f http://localhost:8080/health
```

### Authentication to Registry

To push images, authenticate with GitHub Container Registry:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

## GitHub Actions

The image is automatically built and published by the `build-base-image.yml` workflow:

- **On push to main**: Builds and pushes with `latest` and SHA tags
- **On release**: Builds and pushes with semantic version tags
- **Manual trigger**: Use workflow dispatch with optional version input

To manually trigger a build:

1. Go to Actions > Build Base Image
2. Click "Run workflow"
3. Optionally specify a version tag

## File Structure

```
docker/
├── Dockerfile        # Base image definition
├── entrypoint.sh     # Container entrypoint script
└── README.md         # This file

scripts/
└── build-base-image.sh  # Local build helper script
```

## Contributing

When modifying the base image:

1. Test locally using the build script
2. Verify the container starts and health check passes
3. Update this README if adding new features or environment variables
4. Submit a PR with changes to `docker/` directory
