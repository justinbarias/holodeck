# Implementation Plan: HoloDeck Deploy Command

**Branch**: `019-deploy-command` | **Date**: 2026-01-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/019-deploy-command/spec.md`

## Summary

Implement a deployment pipeline for HoloDeck agents that enables developers to containerize and deploy their agents to cloud platforms (AWS App Runner, Google Cloud Run, Azure Container Apps) through a single CLI command. The implementation uses docker-py SDK for container image building/pushing and official cloud provider Python SDKs for deployment orchestration.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- `docker>=7.0.0` - Container image build/push
- `boto3>=1.42.0` - AWS App Runner deployment (optional)
- `google-cloud-run>=0.13.0` - GCP Cloud Run deployment (optional)
- `azure-mgmt-appcontainers>=4.0.0` - Azure Container Apps deployment (optional)
- `azure-identity>=1.15.0` - Azure authentication (optional)
- `jinja2` - Dockerfile template generation (already a dependency)
- `click` - CLI framework (already a dependency)
- `pydantic` - Configuration models (already a dependency)

**Storage**: N/A (stateless CLI tool, uses cloud provider state)
**Testing**: pytest with markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
**Target Platform**: Linux, macOS, Windows (CLI tool)
**Project Type**: Single project (extends existing HoloDeck CLI)
**Performance Goals**: Image build <5 min for typical agents; deployment completion <10 min
**Constraints**: Requires Docker installed; cloud SDKs are optional extras
**Scale/Scope**: Single agent deployment per command invocation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. No-Code-First Agent Definition
**Status**: PASS
- Deployment configuration defined entirely in YAML (`deployment:` section of config.yaml)
- No Python code required from users to deploy agents
- Dockerfile generated automatically from agent configuration

### II. MCP for API Integrations
**Status**: N/A
- This feature does not integrate with external APIs at agent runtime
- Cloud provider SDKs are used for deployment orchestration, not agent tools

### III. Test-First with Multimodal Support
**Status**: PASS (with plan)
- Unit tests for Pydantic models (DeploymentConfig, RegistryConfig, CloudTargetConfig)
- Integration tests for container build (requires Docker)
- Contract tests for cloud deployer interfaces
- Mock-based tests for cloud SDK interactions

### IV. OpenTelemetry-Native Observability
**Status**: PASS (with plan)
- Deployment operations will emit spans: `holodeck.deploy.build`, `holodeck.deploy.push`, `holodeck.deploy.run`
- Progress events logged with structured attributes
- Deployed containers inherit agent's observability config via environment variables

### V. Evaluation Flexibility with Model Overrides
**Status**: N/A
- Deployment feature does not involve LLM evaluation metrics

### Architecture Constraint: Three Decoupled Engines
**Status**: PASS
- Deploy command implements the **Deployment Engine** (third engine)
- No coupling to Agent Engine or Evaluation Framework internals
- Uses agent.yaml as contract between engines

## Project Structure

### Documentation (this feature)

```text
specs/019-deploy-command/
├── plan.md              # This file
├── research.md          # Phase 0 output (completed)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── deployment-config-schema.yaml
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/holodeck/
├── cli/
│   └── commands/
│       └── deploy.py              # NEW: Deploy command group
├── deploy/                        # NEW: Deployment engine package
│   ├── __init__.py
│   ├── builder.py                 # Container image builder (docker-py)
│   ├── pusher.py                  # Registry push operations
│   ├── deployers/                 # Cloud provider deployers
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract deployer interface
│   │   ├── aws_apprunner.py       # AWS App Runner implementation
│   │   ├── gcp_cloudrun.py        # Google Cloud Run implementation
│   │   └── azure_containerapps.py # Azure Container Apps implementation
│   ├── dockerfile.py              # Dockerfile generation (Jinja2)
│   └── config.py                  # Deployment config resolution
├── models/
│   └── deployment.py              # NEW: Pydantic models for deployment config
└── lib/
    └── errors.py                  # ADD: DeploymentError exception

docker/                            # NEW: Base image definition
├── Dockerfile                     # HoloDeck base image
└── entrypoint.sh                  # Container entrypoint script

tests/
├── unit/
│   └── deploy/
│       ├── test_builder.py
│       ├── test_dockerfile.py
│       └── test_models.py
├── integration/
│   └── deploy/
│       ├── test_build_image.py
│       └── test_push_image.py
└── fixtures/
    └── deploy/
        ├── sample_agent/          # Sample agent for testing
        └── mock_responses/        # Mock cloud SDK responses
```

**Structure Decision**: Extends existing single-project structure. New `deploy/` package follows the pattern of existing `serve/`, `chat/`, and `lib/test_runner/` packages. Cloud deployers use a base class pattern similar to `lib/evaluators/base.py`.

## Component Design

### 1. CLI Command Structure (`cli/commands/deploy.py`)

```python
@click.group(name="deploy", invoke_without_command=True)
@click.argument("agent_config", type=click.Path(exists=True), default="agent.yaml")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
@click.pass_context
def deploy(ctx, agent_config, dry_run, verbose, quiet):
    """Build, push, and deploy a HoloDeck agent."""
    ctx.ensure_object(dict)
    ctx.obj["agent_config"] = agent_config
    ctx.obj["dry_run"] = dry_run

    if ctx.invoked_subcommand is None:
        # Full pipeline: build -> push -> run
        ctx.invoke(build)
        ctx.invoke(push)
        ctx.invoke(run)

@deploy.command()
def build(): ...

@deploy.command()
def push(): ...

@deploy.command()
def run(): ...

@deploy.command()
def status(): ...

@deploy.command()
def destroy(): ...
```

### 2. Pydantic Models (`models/deployment.py`)

```python
class RegistryConfig(BaseModel):
    url: str = Field(description="Registry URL (e.g., docker.io, ghcr.io)")
    repository: str = Field(description="Image repository name")
    tag_strategy: Literal["semver", "git-sha", "timestamp", "custom"] = "git-sha"
    custom_tag: str | None = None
    credentials_env_prefix: str | None = None

class AWSAppRunnerConfig(BaseModel):
    region: str = "us-east-1"
    cpu: str = "1 vCPU"
    memory: str = "2 GB"
    ecr_role_arn: str
    health_check_path: str = "/health"

class GCPCloudRunConfig(BaseModel):
    project_id: str
    region: str = "us-central1"
    memory: str = "512Mi"
    cpu: str = "1000m"
    concurrency: int = 80
    min_instances: int = 0
    max_instances: int = 100

class AzureContainerAppsConfig(BaseModel):
    subscription_id: str
    resource_group: str
    environment_name: str
    location: str = "eastus"
    cpu: str = "0.5"
    memory: str = "1Gi"
    min_replicas: int = 1
    max_replicas: int = 10

class CloudTargetConfig(BaseModel):
    provider: Literal["aws", "gcp", "azure"]
    aws: AWSAppRunnerConfig | None = None
    gcp: GCPCloudRunConfig | None = None
    azure: AzureContainerAppsConfig | None = None

class DeploymentConfig(BaseModel):
    runtime: Literal["docker", "auto"] = "auto"
    registry: RegistryConfig
    target: CloudTargetConfig
    protocol: Literal["rest", "ag-ui", "both"] = "rest"
    port: int = 8080
    environment: dict[str, str] = Field(default_factory=dict)
```

### 3. Cloud Deployer Interface (`deploy/deployers/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Generator

class BaseDeployer(ABC):
    @abstractmethod
    def deploy(
        self,
        service_name: str,
        image_uri: str,
        port: int,
        env_vars: dict[str, str],
        health_check_path: str = "/health"
    ) -> dict:
        """Deploy container and return {url, status, arn/id}."""
        pass

    @abstractmethod
    def get_status(self, service_id: str) -> dict:
        """Get deployment status and URL."""
        pass

    @abstractmethod
    def destroy(self, service_id: str) -> None:
        """Tear down deployment."""
        pass

    @abstractmethod
    def stream_logs(self, service_id: str) -> Generator[str, None, None]:
        """Stream deployment logs (optional, may raise NotImplementedError)."""
        pass
```

### 4. HoloDeck Base Image (`docker/Dockerfile`)

```dockerfile
FROM python:3.10-slim

# Install UV package manager
RUN pip install uv

# Create non-root user
RUN groupadd -r holodeck && useradd -r -g holodeck holodeck

WORKDIR /app

# Install holodeck package
RUN uv pip install --system holodeck

# Switch to non-root user
USER holodeck

# Default environment
ENV HOLODECK_PORT=8080
ENV HOLODECK_PROTOCOL=rest

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${HOLODECK_PORT}/health')"

EXPOSE 8080

# Entrypoint runs holodeck serve
ENTRYPOINT ["holodeck", "serve"]
CMD ["/app/agent.yaml", "--host", "0.0.0.0"]
```

## Complexity Tracking

> No constitution violations requiring justification.

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Multi-cloud support | Separate deployer classes | Clean separation, testable independently |
| Optional dependencies | pyproject.toml extras | Users install only what they need |
| Docker-only runtime | docker-py SDK | Structured API, auto-credentials; Podman/nerdctl deferred to v2 |

## Key Implementation Notes

1. **Dependency Installation**: Cloud SDKs are optional extras. Commands check for availability and provide clear error messages if missing.

2. **Credential Handling**: Follow each cloud provider's standard credential chain. Document required environment variables in error messages.

3. **Progress Display**: Use spinner thread pattern from test runner. Respect `--quiet` flag for CI/CD.

4. **Error Messages**: Include remediation hints (e.g., "Run `pip install holodeck[deploy-aws]` to enable AWS deployment").

5. **Idempotency**: `deploy run` should update existing deployments rather than failing on duplicates.

6. **State Tracking**: Store deployment metadata (service ARN/ID, URL) in `.holodeck/deployments.json` for `status` and `destroy` commands.

## Next Steps

1. Generate `data-model.md` with entity details
2. Generate `contracts/deployment-config-schema.yaml`
3. Generate `quickstart.md` with usage examples
4. Run `/speckit.tasks` to create task breakdown
