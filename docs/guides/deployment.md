# Deployment Guide

Package a HoloDeck agent as a container and ship it to the cloud with `holodeck deploy`.

## Quick start

Build an image from a Claude-backend agent and run it on Azure Container Apps.

```yaml
# agent.yaml
name: my-agent
model:
  provider: anthropic
  name: claude-sonnet-4-5
instructions:
  inline: "You are a helpful assistant."

deployment:
  registry:
    url: ghcr.io
    repository: myorg/my-agent
    tag_strategy: git_sha
  target:
    provider: azure
    azure:
      subscription_id: "${AZURE_SUBSCRIPTION_ID}"
      resource_group: my-resource-group
      environment_name: my-container-env
      location: eastus
  port: 8080
```

```bash
holodeck deploy build          # builds ghcr.io/myorg/my-agent:abc1234
holodeck deploy run            # rolls a revision on Azure Container Apps
# → live URL, e.g. https://my-agent.<region>.azurecontainerapps.io
```

```bash
curl https://my-agent.<region>.azurecontainerapps.io/health
# {"status":"healthy",...}  → HTTP 200
```

!!! info "Backend support"
    `holodeck deploy` is fully supported on the Claude backend (`provider: anthropic`) and Ollama. On the OpenAI Agents backend (`provider: openai` / `azure_openai`) deploy is **on the roadmap** — see the [OpenAI Backend guide](openai-backend.md).

## How it works

`holodeck deploy build` generates a Dockerfile from your agent config, builds an OCI image on the HoloDeck base image, and tags it per your `tag_strategy`. The container entrypoint runs `holodeck serve`, so the image is just your agent behind an HTTP server (see the [Serve guide](serve.md)). `holodeck deploy run` then provisions or updates the target cloud service from the `deployment.target` block. Prefer the CLI for the happy path; drop to the [DIY approach](#diy-deployment) when you need a custom Dockerfile or base image.

## Deploy build

```bash
holodeck deploy build <agent_config> [OPTIONS]
```

| Option          | Description                                   | Default      |
| --------------- | --------------------------------------------- | ------------ |
| `agent_config`  | Path to agent.yaml                            | `agent.yaml` |
| `--tag`         | Custom tag (overrides strategy)               | None         |
| `--no-cache`    | Build without Docker cache                    | `false`      |
| `--wheel`       | HoloDeck wheel to install instead of the PyPI release (needed for a development version) | None |
| `--base-image`  | Base image reference                          | `ghcr.io/justinbarias/holodeck-base:<version>` (`:latest` with `--wheel`) |
| `--no-pull`     | Use the base image already in the local Docker daemon | `false` |
| `--dry-run`     | Show what would be built without executing    | `false`      |
| `--verbose, -v` | Verbose debug logging                         | `false`      |
| `--quiet, -q`   | Suppress progress output                      | `false`      |

Exit codes: `0` success · `2` configuration error · `3` Docker/deployment error.

```bash
holodeck deploy build --dry-run            # preview Dockerfile
holodeck deploy build agent.yaml --tag v1.0.0
holodeck deploy build --no-cache           # force fresh build
holodeck deploy build --wheel dist/holodeck_ai-0.7.2.dev64-py3-none-any.whl   # development build
```

---

## Reproducible images

Every image installs one exact HoloDeck version, with dependency versions taken from that release's lock file. Two builds of the same HoloDeck version resolve the same packages.

| What | How it is pinned |
| --- | --- |
| HoloDeck | `holodeck-ai[extras]==X.Y.Z`, the version of the `holodeck` CLI that runs `deploy build`. No `--prerelease` flag. |
| Dependencies | `--constraint` with `constraints.txt`, shipped inside the wheel and generated from `uv.lock` (`make constraints`). `overrides.txt` carries the `[tool.uv] override-dependencies` the lock was resolved with. `deploy build` copies both into the build context. |
| Base image | `ghcr.io/justinbarias/holodeck-base:X.Y.Z`, the base image of the same release. The Python image under it is pinned by digest in [docker/Dockerfile](https://github.com/justinbarias/holodeck/blob/main/docker/Dockerfile); uv is pinned by version and digest. |
| Extras | Chosen from the agent file (below), written sorted into the Dockerfile and the `com.holodeck.extras` label. |

A development version such as `0.7.2.dev61` is not on PyPI. `deploy build` then needs a wheel: build one with `uv build --wheel`, keep it in `./dist` or pass `--wheel`. The image installs that wheel and the constraints inside it. Without a wheel the build stops with that instruction before it starts Docker.

### Image manifest

Each build writes `.holodeck/builds/<tag>.json` next to `agent.yaml`:

- HoloDeck version, install source (`pypi` or `wheel`, with the wheel's SHA-256), extras, and the SHA-256 of the constraints and overrides files.
- The base image reference, its registry digest, and its image ID. A base image built locally has no registry digest; `digest` is `null` and the image ID identifies it.
- The built image's reference and image ID. `digest` stays `null` until you push the image; read it from the registry then.
- Every installed package as `name==version`, sorted.

The same package list is inside the image at `/opt/holodeck/manifest.txt` (`docker run --rm --entrypoint cat <image> /opt/holodeck/manifest.txt`). The manifest has no timestamps, so two builds of the same inputs give the same file except for the image ID.

### Which extras an image gets

`holodeck deploy build` installs the extras the agent file needs. Serving images never get `eval`, `nlp` or `optimizer`: an image runs `holodeck serve`, not `holodeck test`.

| The agent file has | Extra |
| --- | --- |
| `model.provider: openai` or `azure_openai` | `openai-agents` |
| `model.provider: anthropic` or `ollama`, with `observability.enabled` and `traces.enabled` | `claude-otel` |
| Vector store or hierarchical document tool, `database.provider` | `qdrant`, `chromadb`, `postgres`, `pinecone`, `weaviate`, `azure-ai-search`, or `azure-cosmos` (for `azure-cosmos-nosql`); `in-memory` needs none |
| Tool `source` that starts with `az://` or `s3://` | `azure-blob` or `s3` |
| `hierarchical_document` tool with `keyword_index.provider: opensearch` | `opensearch` |
| `observability.exporters.prometheus` or `azure_monitor` enabled | `otel-prometheus` or `otel-azmon` |

A database named by reference and not resolved from global configuration adds no extra. The `temporal` extra (for `holodeck worker`) is not chosen from the agent file: `deploy build` generates a `holodeck serve` image only.

To add an extra by hand, build your own image (see [DIY deployment](#diy-deployment)) from the base image of the HoloDeck version you run, and install with the constraints that ship in the package:

```dockerfile
RUN CONSTRAINTS="$(python -c "import importlib.util as u; print(u.find_spec('holodeck').submodule_search_locations[0] + '/deploy')")" \
    && uv pip install --system --no-cache \
        --constraint "${CONSTRAINTS}/constraints.txt" --override "${CONSTRAINTS}/overrides.txt" \
        "holodeck-ai[temporal]==0.7.1"
```

---

## Deployment configuration

Add a `deployment` section to `agent.yaml`:

```yaml
deployment:
  runtime: container
  registry:
    url: ghcr.io
    repository: myorg/my-agent
    tag_strategy: git_sha
  target:
    provider: azure
    azure:
      subscription_id: "${AZURE_SUBSCRIPTION_ID}"
      resource_group: my-resource-group
      environment_name: my-container-env
      location: eastus
  protocol: rest
  port: 8080
  environment:
    LOG_LEVEL: info
    CUSTOM_API_KEY: "${CUSTOM_API_KEY}"   # from shell environment
```

### Registry

| Field                    | Description                                 | Required | Default   |
| ------------------------ | ------------------------------------------- | -------- | --------- |
| `url`                    | Registry URL (`ghcr.io`, `docker.io`, …)    | Yes      | -         |
| `repository`             | Repository name (`org/agent-name`, lowercase) | Yes    | -         |
| `tag_strategy`           | `git_sha`, `git_tag`, `latest`, `custom`    | No       | `git_sha` |
| `custom_tag`             | Custom tag (when strategy is `custom`)      | No       | -         |
| `credentials_env_prefix` | Env var prefix for registry credentials     | No       | -         |

Tag strategies: `git_sha` (first 7 chars of HEAD → `abc1234`), `git_tag` (latest git tag → `v1.2.0`), `latest`, `custom` (uses `custom_tag`).

### Protocol and port

`protocol` selects `rest`, `ag-ui`, or `both`. `port` sets the container's exposed port (default `8080`).

### Cloud targets

| Provider | Service        | `target.provider` |
| -------- | -------------- | ----------------- |
| Azure    | Container Apps | `azure`           |
| AWS      | App Runner     | `aws`             |
| GCP      | Cloud Run      | `gcp`             |

```yaml
# AWS App Runner
target:
  provider: aws
  aws:
    region: us-east-1
    cpu: 1                 # 1, 2, or 4 vCPU
    memory: 2048           # MB
    ecr_role_arn: arn:aws:iam::123456789012:role/AppRunnerECRAccess
    health_check_path: /health
    auto_scaling_min: 1
    auto_scaling_max: 5
```

```yaml
# GCP Cloud Run
target:
  provider: gcp
  gcp:
    project_id: my-gcp-project
    region: us-central1
    memory: 512Mi
    cpu: 1
    concurrency: 80
    timeout: 300
    min_instances: 0
    max_instances: 100
```

```yaml
# Azure Container Apps
target:
  provider: azure
  azure:
    subscription_id: "${AZURE_SUBSCRIPTION_ID}"
    resource_group: my-resource-group
    environment_name: my-container-env
    location: eastus
    cpu: 0.5               # 0.25, 0.5, 0.75, 1.0, …
    memory: 1Gi
    ingress_external: true
    min_replicas: 0
    max_replicas: 10
```

---

## Registry push

Authenticate and push with Docker:

```bash
holodeck deploy build agent.yaml
docker login ghcr.io -u USERNAME
docker push ghcr.io/myorg/my-agent:abc1234
```

---

## DIY deployment

Build and deploy containers manually without the CLI.

### Base image

```
ghcr.io/justinbarias/holodeck-base:<holodeck-version>
```

Use the tag of the HoloDeck release you run (for example `0.7.1`), not `latest`. Includes Python 3.13 with that HoloDeck version installed, its package list in `/opt/holodeck/manifest.txt`, `curl` for health checks, a non-root `holodeck` user, and `/app` as the working directory.

### Build context

```
build-context/
├── Dockerfile          # Container definition
├── agent.yaml          # Agent configuration
├── entrypoint.sh       # Startup script
└── instructions/       # Instruction files (if using file reference)
    └── system.md
```

### Dockerfile

```dockerfile
FROM ghcr.io/justinbarias/holodeck-base:0.7.1

LABEL org.opencontainers.image.title="my-agent"
LABEL org.opencontainers.image.version="v1.0.0"
LABEL com.holodeck.managed="true"

USER root
WORKDIR /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
COPY agent.yaml /app/agent.yaml
COPY instructions/ /app/instructions/

ENV HOLODECK_PORT="8080"
ENV HOLODECK_PROTOCOL="rest"
ENV HOLODECK_AGENT_CONFIG="/app/agent.yaml"

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

RUN chown -R holodeck:holodeck /app
USER holodeck
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Entrypoint

```bash
#!/bin/bash
set -e
exec holodeck serve /app/agent.yaml \
    --host 0.0.0.0 \
    --port "${HOLODECK_PORT:-8080}" \
    --protocol "${HOLODECK_PROTOCOL:-rest}"
```

### Container reference

| Variable                | Description                 | Default           |
| ----------------------- | --------------------------- | ----------------- |
| `HOLODECK_PORT`         | Agent server port           | `8080`            |
| `HOLODECK_PROTOCOL`     | Protocol type               | `rest`            |
| `HOLODECK_AGENT_CONFIG` | Path to agent configuration | `/app/agent.yaml` |

OCI labels: `org.opencontainers.image.title` (agent name), `.version`, `.created` (ISO 8601), `.source`, `com.holodeck.managed` (marker for HoloDeck images), and, on images from `deploy build`, `com.holodeck.version` and `com.holodeck.extras`.

### Build and run

```bash
docker build -t my-agent:v1.0.0 .

docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  my-agent:v1.0.0
```

```yaml
# docker-compose.yml
services:
  agent:
    build: { context: ., dockerfile: Dockerfile }
    ports: ["8080:8080"]
    environment:
      - HOLODECK_PORT=8080
      - HOLODECK_PROTOCOL=rest
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    restart: unless-stopped
```

---

## Troubleshooting

### Docker not available

```
Error: Docker is not available
```

Ensure the Docker daemon is running (`sudo systemctl start docker` on Linux, or start Docker Desktop).

### No deployment section

```
Error: Configuration error
  No 'deployment' section found in agent configuration
```

Add a `deployment` section with `registry` and `target` to `agent.yaml`.

### Git SHA strategy fails

```
Error: tag_generation failed
  Failed to get git SHA: not a git repository
```

Initialize a git repository, or set `tag_strategy: latest` (or pass `--tag`).

### Invalid repository name

```
Error: Invalid repository name: My-Agent
  Must contain only lowercase letters, numbers, '.', '_', '/', '-'
```

Use a lowercase repository name (`myorg/my-agent`).

### Health check fails

1. Verify the agent starts: `docker run -it --rm my-agent:latest holodeck serve --help`
2. Confirm required credentials are passed (e.g. `-e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"`).
3. Inspect logs: `docker logs <container-id>`.

### Deploying an OpenAI-provider agent fails

Deploy is not yet supported on the OpenAI Agents backend. Use a Claude-backend agent (`provider: anthropic`) for now; see the [OpenAI Backend guide](openai-backend.md).

---

## Next steps

- [Agent Server Guide](serve.md) — server configuration and endpoints
- [Agent Configuration Guide](agent-configuration.md) — agent.yaml reference
- [Observability Guide](observability.md) — monitoring and tracing
