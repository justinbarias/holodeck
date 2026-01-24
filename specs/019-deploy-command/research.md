# Research: HoloDeck Deploy Command

**Feature Branch**: `019-deploy-command`
**Date**: 2026-01-24

## Executive Summary

This document captures research findings for implementing the `holodeck deploy` command, covering container image building, registry push operations, and cloud platform deployment.

---

## 1. Container Image Building

### Decision: docker-py SDK

**Chosen Approach**: Use the `docker` Python SDK (docker-py) exclusively for container image building and pushing.

**Rationale**:
- Structured JSON output for parsing build/push progress
- Automatic credential management from `~/.docker/config.json`
- Clean exception hierarchy (`BuildError`, `APIError`, `ImageNotFound`)
- Type hints and IDE support
- No subprocess overhead or output parsing

**Trade-off**: Docker-only support (no Podman/nerdctl). Users must have Docker installed.

### Implementation Pattern

```python
import docker
from docker.errors import BuildError, APIError, DockerException

class ContainerBuilder:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except DockerException as e:
            raise RuntimeError(
                "Docker not available. Install Docker Desktop or Docker Engine.\n"
                "See: https://docs.docker.com/get-docker/"
            ) from e

    def build(
        self,
        context_path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        build_args: dict | None = None,
        labels: dict | None = None
    ) -> Generator[str, None, None]:
        """Build image with streaming output."""
        try:
            _, logs = self.client.images.build(
                path=context_path,
                tag=tag,
                dockerfile=dockerfile,
                buildargs=build_args or {},
                labels=labels or {},
                pull=True,
                rm=True
            )
            for log in logs:
                if 'stream' in log:
                    yield log['stream'].strip()
                elif 'error' in log:
                    raise BuildError(log['error'], logs)
        except BuildError as e:
            raise RuntimeError(f"Build failed: {e}") from e

    def push(
        self,
        repository: str,
        tag: str,
        auth_config: dict | None = None
    ) -> Generator[str, None, None]:
        """Push image with streaming output."""
        for line in self.client.images.push(
            repository=repository,
            tag=tag,
            stream=True,
            decode=True,
            auth_config=auth_config
        ):
            if 'status' in line:
                yield line['status']
            if 'error' in line:
                raise RuntimeError(f"Push failed: {line['error']}")
```

### Dockerfile Generation

Use Jinja2 templates for generating Dockerfiles dynamically:

```python
from jinja2 import Template

HOLODECK_DOCKERFILE_TEMPLATE = """
FROM {{ base_image }}

LABEL org.opencontainers.image.title="{{ agent_name }}"
LABEL org.opencontainers.image.version="{{ version }}"
LABEL org.opencontainers.image.created="{{ build_timestamp }}"
LABEL org.opencontainers.image.source="{{ git_sha }}"
LABEL com.holodeck.managed="true"

WORKDIR /app

# Copy agent configuration and artifacts
COPY agent.yaml .
{% for file in instruction_files %}
COPY {{ file }} ./{{ file }}
{% endfor %}
{% for dir in data_dirs %}
COPY {{ dir }}/ ./{{ dir }}/
{% endfor %}

# Configure serve command
ENV HOLODECK_AGENT_CONFIG=/app/agent.yaml
ENV HOLODECK_PORT={{ port }}
ENV HOLODECK_PROTOCOL={{ protocol }}

EXPOSE {{ port }}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{{ port }}/health')"

ENTRYPOINT ["holodeck", "serve", "/app/agent.yaml", "--host", "0.0.0.0", "--port", "{{ port }}", "--protocol", "{{ protocol }}"]
"""
```

### Registry Authentication

The SDK automatically uses credentials from `~/.docker/config.json`. For programmatic auth:

| Registry | Username | Password/Token |
|----------|----------|----------------|
| Docker Hub | Docker Hub username | Personal Access Token |
| GHCR | GitHub username | `GITHUB_TOKEN` or PAT |
| ACR | Service Principal ID | Service Principal Secret |
| GAR | `_json_key` | JSON key content |
| ECR | AWS Access Key ID | AWS Secret Access Key |

**Environment Variable Pattern**:
```
HOLODECK_REGISTRY_USERNAME
HOLODECK_REGISTRY_PASSWORD
```

---

## 2. Cloud Deployment SDKs

### AWS App Runner (boto3)

**Package**: `boto3` (AWS SDK for Python)
**Version**: 1.42.0+

**Key Methods**:
- `create_service()` - Create new service
- `update_service()` - Update existing service
- `describe_service()` - Get status and URL
- `delete_service()` - Remove service

**Implementation Pattern**:

```python
import boto3
from botocore.exceptions import ClientError

class AWSAppRunnerDeployer:
    def __init__(self, region: str = 'us-east-1'):
        self.client = boto3.client('apprunner', region_name=region)

    def deploy(
        self,
        service_name: str,
        image_uri: str,
        ecr_role_arn: str,
        cpu: str = '1 vCPU',
        memory: str = '2 GB',
        port: int = 8080,
        env_vars: dict | None = None,
        health_check_path: str = '/health'
    ) -> dict:
        config = {
            'ServiceName': service_name,
            'SourceConfiguration': {
                'ImageRepository': {
                    'ImageIdentifier': image_uri,
                    'ImageRepositoryType': 'ECR',
                    'ImageConfiguration': {
                        'RuntimeEnvironmentVariables': env_vars or {},
                        'Port': str(port)
                    }
                },
                'AuthenticationConfiguration': {
                    'AccessRoleArn': ecr_role_arn
                },
                'AutoDeploymentsEnabled': False
            },
            'InstanceConfiguration': {
                'Cpu': cpu,
                'Memory': memory
            },
            'HealthCheckConfiguration': {
                'Protocol': 'HTTP',
                'Path': health_check_path,
                'Interval': 5,
                'Timeout': 2,
                'HealthyThreshold': 1,
                'UnhealthyThreshold': 3
            }
        }

        response = self.client.create_service(**config)
        return {
            'arn': response['Service']['ServiceArn'],
            'url': response['Service']['ServiceUrl'],
            'status': response['Service']['Status']
        }

    def get_status(self, service_arn: str) -> dict:
        response = self.client.describe_service(ServiceArn=service_arn)
        return {
            'status': response['Service']['Status'],
            'url': response['Service']['ServiceUrl']
        }

    def destroy(self, service_arn: str) -> None:
        self.client.delete_service(ServiceArn=service_arn)
```

**Required IAM Permissions**:
- `apprunner:CreateService`
- `apprunner:UpdateService`
- `apprunner:DescribeService`
- `apprunner:DeleteService`
- `ecr:GetDownloadUrlForLayer`
- `ecr:BatchGetImage`
- `iam:PassRole`

---

### Google Cloud Run (google-cloud-run)

**Package**: `google-cloud-run`
**Version**: 0.13.0+

**Key Methods**:
- `ServicesClient.create_service()` - Create service (returns LRO)
- `ServicesClient.update_service()` - Update service
- `ServicesClient.get_service()` - Get status
- `ServicesClient.delete_service()` - Delete service

**Implementation Pattern**:

```python
from google.cloud import run_v2
from google.cloud.run_v2.types import Service, RevisionTemplate, Container

class GCPCloudRunDeployer:
    def __init__(self, project_id: str, region: str = 'us-central1'):
        self.project_id = project_id
        self.region = region
        self.client = run_v2.ServicesClient()

    def deploy(
        self,
        service_name: str,
        image_uri: str,
        memory: str = '512Mi',
        cpu: str = '1000m',
        concurrency: int = 80,
        timeout_seconds: int = 300,
        env_vars: dict | None = None,
        min_instances: int = 0,
        max_instances: int = 100
    ) -> dict:
        parent = f'projects/{self.project_id}/locations/{self.region}'

        container = Container(
            image=image_uri,
            env=[run_v2.EnvVar(name=k, value=v) for k, v in (env_vars or {}).items()],
            resources=run_v2.ResourceRequirements(limits={'cpu': cpu, 'memory': memory}),
            ports=[run_v2.ContainerPort(container_port=8080)]
        )

        template = RevisionTemplate(
            containers=[container],
            timeout=run_v2.Duration(seconds=timeout_seconds),
            max_instance_request_concurrency=concurrency,
            scaling=run_v2.RevisionScaling(
                min_instance_count=min_instances,
                max_instance_count=max_instances
            )
        )

        service = Service(
            template=template,
            ingress=run_v2.Service.IngressTraffic.INGRESS_TRAFFIC_ALL
        )

        request = run_v2.CreateServiceRequest(
            parent=parent,
            service=service,
            service_id=service_name
        )

        operation = self.client.create_service(request=request)
        result = operation.result(timeout=600)

        return {
            'name': result.name,
            'url': result.uri,
            'status': 'RUNNING'
        }

    def get_status(self, service_name: str) -> dict:
        name = f'projects/{self.project_id}/locations/{self.region}/services/{service_name}'
        service = self.client.get_service(request={'name': name})
        return {
            'url': service.uri,
            'latest_revision': service.latest_ready_revision
        }

    def destroy(self, service_name: str) -> None:
        name = f'projects/{self.project_id}/locations/{self.region}/services/{service_name}'
        operation = self.client.delete_service(request={'name': name})
        operation.result(timeout=600)
```

**Required IAM Roles**:
- `roles/run.developer`
- `roles/iam.serviceAccountUser`

---

### Azure Container Apps (azure-mgmt-appcontainers)

**Package**: `azure-mgmt-appcontainers`
**Version**: 4.0.0+

**Note**: Use `azure-mgmt-appcontainers` (not `azure-mgmt-containerinstance`) for serverless Container Apps.

**Key Methods**:
- `container_apps.begin_create_or_update()` - Create/update (returns LRO poller)
- `container_apps.get()` - Get status
- `container_apps.begin_delete()` - Delete

**Implementation Pattern**:

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
from azure.mgmt.appcontainers.models import (
    ContainerApp, Template, Container, ContainerResources,
    Ingress, Configuration, EnvironmentVar, TrafficWeight
)

class AzureContainerAppsDeployer:
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        environment_name: str,
        location: str = 'eastus'
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.environment_name = environment_name
        self.location = location
        self.client = ContainerAppsAPIClient(
            DefaultAzureCredential(),
            subscription_id
        )

    def deploy(
        self,
        app_name: str,
        image_uri: str,
        port: int = 8080,
        cpu: str = '0.5',
        memory: str = '1Gi',
        env_vars: dict | None = None,
        ingress_external: bool = True,
        min_replicas: int = 1,
        max_replicas: int = 10
    ) -> dict:
        container = Container(
            name=app_name,
            image=image_uri,
            resources=ContainerResources(cpu=cpu, memory=memory),
            env=[EnvironmentVar(name=k, value=v) for k, v in (env_vars or {}).items()]
        )

        ingress = Ingress(
            external=ingress_external,
            target_port=port,
            traffic=[TrafficWeight(percentage=100, latest_revision=True)]
        )

        container_app = ContainerApp(
            location=self.location,
            template=Template(containers=[container]),
            configuration=Configuration(ingress=ingress),
            managed_environment_id=f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.App/managedEnvironments/{self.environment_name}'
        )

        poller = self.client.container_apps.begin_create_or_update(
            resource_group_name=self.resource_group,
            container_app_name=app_name,
            container_app_envelope=container_app
        )

        result = poller.result()

        return {
            'name': result.name,
            'url': f"https://{result.configuration.ingress.fqdn}" if result.configuration.ingress else None,
            'status': result.provisioning_state
        }

    def get_status(self, app_name: str) -> dict:
        app = self.client.container_apps.get(
            resource_group_name=self.resource_group,
            container_app_name=app_name
        )
        return {
            'status': app.provisioning_state,
            'url': f"https://{app.configuration.ingress.fqdn}" if app.configuration.ingress else None
        }

    def destroy(self, app_name: str) -> None:
        self.client.container_apps.begin_delete(
            resource_group_name=self.resource_group,
            container_app_name=app_name
        ).result()
```

**Required Azure RBAC**:
- `Contributor` on resource group
- `AcrPull` on container registry

---

## 3. Authentication Patterns

### Credential Discovery Order

| Platform | Priority 1 | Priority 2 | Priority 3 |
|----------|------------|------------|------------|
| **AWS** | IAM Role (EC2/Lambda) | `~/.aws/credentials` | Environment vars |
| **GCP** | ADC / Metadata server | `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON |
| **Azure** | Managed Identity | `DefaultAzureCredential` chain | Environment vars |

### Environment Variables

```bash
# AWS
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# GCP
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=my-project

# Azure
AZURE_SUBSCRIPTION_ID=...
AZURE_TENANT_ID=...
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
```

---

## 4. HoloDeck CLI Patterns

Based on codebase analysis, the deploy command should follow these patterns:

### Command Structure
- Use `@click.group()` for `deploy` with subcommands
- Register subcommands: `build`, `push`, `run`, `status`, `destroy`
- Default behavior (no subcommand) runs full pipeline

### Common Options
```python
@click.option("--verbose", "-v", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
@click.option("--dry-run", is_flag=True)
```

### Error Handling
- Add `DeploymentError` to `lib/errors.py`
- Exit codes: 0=success, 2=config error, 3=execution error

### Progress Indicators
- Use existing `ProgressIndicator` pattern from test runner
- Spinner thread for long-running operations
- TTY detection for CI compatibility

### Async Pattern
- Use `asyncio.run()` in Click command
- Cloud SDK operations are sync (use threading for progress)

---

## 5. Dependencies

### New Dependencies (Optional Extras)

```toml
[project.optional-dependencies]
deploy = [
    "docker>=7.0.0",
]
deploy-aws = [
    "boto3>=1.42.0",
]
deploy-gcp = [
    "google-cloud-run>=0.13.0",
]
deploy-azure = [
    "azure-mgmt-appcontainers>=4.0.0",
    "azure-identity>=1.15.0",
]
deploy-all = [
    "holodeck[deploy,deploy-aws,deploy-gcp,deploy-azure]",
]
```

### Base Image Dependencies
The HoloDeck base image needs:
- Python 3.10+ slim base
- UV package manager
- holodeck package
- FastAPI + Uvicorn (for serve command)

---

## 6. Decisions Summary

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Container runtime | docker-py SDK only | Structured API, auto-credentials, clean errors |
| Cloud SDKs | boto3, google-cloud-run, azure-mgmt-appcontainers | Official SDKs, well-maintained, async support |
| Dockerfile generation | Jinja2 templates | Flexible, readable, maintainable |
| Registry auth | Environment variables + docker config | Standard patterns, secure |
| CLI structure | Click group with subcommands | Consistent with existing HoloDeck CLI |
| Progress display | Spinner thread + ProgressIndicator | Consistent UX, CI-compatible |

---

## 7. Open Questions Resolved

1. **Q: docker-py vs subprocess?** → docker-py SDK for structured API
2. **Q: Cloud CLIs vs SDKs?** → Python SDKs for better integration
3. **Q: Multi-runtime support?** → Docker-only for v1 (simplicity)
4. **Q: Credential management?** → Environment variables + native credential helpers
