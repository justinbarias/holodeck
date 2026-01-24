# Data Model: HoloDeck Deploy Command

**Feature Branch**: `019-deploy-command`
**Date**: 2026-01-24

## Entity Relationship Diagram

```
┌─────────────────────┐
│   DeploymentConfig  │
├─────────────────────┤
│ runtime             │
│ registry ──────────────────┐
│ target ────────────────────┼──┐
│ protocol            │      │  │
│ port                │      │  │
│ environment         │      │  │
└─────────────────────┘      │  │
                             │  │
┌─────────────────────┐      │  │
│   RegistryConfig    │◄─────┘  │
├─────────────────────┤         │
│ url                 │         │
│ repository          │         │
│ tag_strategy        │         │
│ custom_tag          │         │
│ credentials_env     │         │
└─────────────────────┘         │
                                │
┌─────────────────────┐         │
│  CloudTargetConfig  │◄────────┘
├─────────────────────┤
│ provider            │
│ aws ────────────────────┐
│ gcp ────────────────────┼──┐
│ azure ──────────────────┼──┼──┐
└─────────────────────┘   │  │  │
                          │  │  │
┌─────────────────────┐   │  │  │
│ AWSAppRunnerConfig  │◄──┘  │  │
├─────────────────────┤      │  │
│ region              │      │  │
│ cpu                 │      │  │
│ memory              │      │  │
│ ecr_role_arn        │      │  │
│ health_check_path   │      │  │
└─────────────────────┘      │  │
                             │  │
┌─────────────────────┐      │  │
│ GCPCloudRunConfig   │◄─────┘  │
├─────────────────────┤         │
│ project_id          │         │
│ region              │         │
│ memory              │         │
│ cpu                 │         │
│ concurrency         │         │
│ min_instances       │         │
│ max_instances       │         │
└─────────────────────┘         │
                                │
┌─────────────────────┐         │
│AzureContainerApps   │◄────────┘
│      Config         │
├─────────────────────┤
│ subscription_id     │
│ resource_group      │
│ environment_name    │
│ location            │
│ cpu                 │
│ memory              │
│ min_replicas        │
│ max_replicas        │
└─────────────────────┘

┌─────────────────────┐
│  ContainerImage     │ (Runtime entity)
├─────────────────────┤
│ name                │
│ tag                 │
│ image_id            │
│ size_bytes          │
│ created_at          │
│ labels              │
└─────────────────────┘

┌─────────────────────┐
│  CloudDeployment    │ (Runtime entity)
├─────────────────────┤
│ provider            │
│ service_id          │
│ service_name        │
│ url                 │
│ status              │
│ created_at          │
│ updated_at          │
└─────────────────────┘
```

---

## Entity Definitions

### DeploymentConfig

**Purpose**: Root configuration for the deployment pipeline, added to agent.yaml under `deployment:` key.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `runtime` | `Literal["docker", "auto"]` | No | `"auto"` | Container runtime to use. `auto` detects Docker. |
| `registry` | `RegistryConfig` | Yes | - | Container registry configuration |
| `target` | `CloudTargetConfig` | Yes | - | Cloud deployment target configuration |
| `protocol` | `Literal["rest", "ag-ui", "both"]` | No | `"rest"` | Protocol(s) exposed by the deployed container |
| `port` | `int` | No | `8080` | Port the container listens on |
| `environment` | `dict[str, str]` | No | `{}` | Environment variables injected at runtime |

**Validation Rules**:
- `port` must be between 1 and 65535
- `environment` keys must be valid environment variable names (alphanumeric + underscore)

---

### RegistryConfig

**Purpose**: Configuration for the container registry where images are pushed.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `str` | Yes | - | Registry URL (e.g., `docker.io`, `ghcr.io`, `123456.dkr.ecr.us-east-1.amazonaws.com`) |
| `repository` | `str` | Yes | - | Repository name within the registry |
| `tag_strategy` | `Literal["semver", "git-sha", "timestamp", "custom"]` | No | `"git-sha"` | Strategy for generating image tags |
| `custom_tag` | `str \| None` | No | `None` | Custom tag value (required if `tag_strategy` is `custom`) |
| `credentials_env_prefix` | `str \| None` | No | `None` | Environment variable prefix for credentials (e.g., `DOCKER` → `DOCKER_USERNAME`, `DOCKER_PASSWORD`) |

**Validation Rules**:
- `url` must be a valid hostname (no protocol prefix)
- `repository` must match pattern `^[a-z0-9]+(?:[._-][a-z0-9]+)*$`
- If `tag_strategy` is `custom`, `custom_tag` is required
- `custom_tag` must match pattern `^[a-zA-Z0-9][a-zA-Z0-9._-]*$`

**Tag Generation**:
- `semver`: Reads version from `pyproject.toml` or defaults to `0.1.0`
- `git-sha`: Uses first 8 characters of current git commit SHA
- `timestamp`: Uses ISO format `YYYYMMDD-HHMMSS`
- `custom`: Uses value from `custom_tag` field

---

### CloudTargetConfig

**Purpose**: Discriminated union for cloud provider configuration. Exactly one provider config should be populated.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | `Literal["aws", "gcp", "azure"]` | Yes | - | Cloud provider identifier |
| `aws` | `AWSAppRunnerConfig \| None` | No | `None` | AWS App Runner configuration |
| `gcp` | `GCPCloudRunConfig \| None` | No | `None` | Google Cloud Run configuration |
| `azure` | `AzureContainerAppsConfig \| None` | No | `None` | Azure Container Apps configuration |

**Validation Rules**:
- Exactly one of `aws`, `gcp`, or `azure` must be populated
- The populated config must match the `provider` field

---

### AWSAppRunnerConfig

**Purpose**: Configuration specific to AWS App Runner deployments.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `region` | `str` | No | `"us-east-1"` | AWS region for deployment |
| `cpu` | `str` | No | `"1 vCPU"` | CPU allocation (`0.25 vCPU`, `0.5 vCPU`, `1 vCPU`, `2 vCPU`, `4 vCPU`) |
| `memory` | `str` | No | `"2 GB"` | Memory allocation (`512 MB`, `1 GB`, `2 GB`, `3 GB`, `4 GB`) |
| `ecr_role_arn` | `str` | Yes | - | IAM role ARN for ECR access |
| `health_check_path` | `str` | No | `"/health"` | HTTP path for health checks |
| `auto_scaling_min` | `int` | No | `1` | Minimum number of instances |
| `auto_scaling_max` | `int` | No | `25` | Maximum number of instances |

**Validation Rules**:
- `region` must be a valid AWS region code
- `cpu` must be one of the allowed values
- `memory` must be compatible with selected CPU (App Runner constraints)
- `ecr_role_arn` must match ARN pattern `^arn:aws:iam::\d{12}:role/.+$`
- `auto_scaling_min` >= 1, `auto_scaling_max` <= 25

---

### GCPCloudRunConfig

**Purpose**: Configuration specific to Google Cloud Run deployments.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `project_id` | `str` | Yes | - | GCP project ID |
| `region` | `str` | No | `"us-central1"` | GCP region for deployment |
| `memory` | `str` | No | `"512Mi"` | Memory allocation (`128Mi` to `32Gi`) |
| `cpu` | `str` | No | `"1000m"` | CPU allocation in millicores (`1000m` = 1 CPU) |
| `concurrency` | `int` | No | `80` | Maximum concurrent requests per instance |
| `timeout_seconds` | `int` | No | `300` | Request timeout (1-3600 seconds) |
| `min_instances` | `int` | No | `0` | Minimum instances (0 enables scale-to-zero) |
| `max_instances` | `int` | No | `100` | Maximum instances |

**Validation Rules**:
- `project_id` must match GCP project ID pattern `^[a-z][a-z0-9-]{4,28}[a-z0-9]$`
- `region` must be a valid GCP region
- `memory` must be a valid Kubernetes quantity
- `concurrency` must be between 1 and 1000
- `timeout_seconds` must be between 1 and 3600
- `min_instances` >= 0, `max_instances` <= 1000

---

### AzureContainerAppsConfig

**Purpose**: Configuration specific to Azure Container Apps deployments.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `subscription_id` | `str` | Yes | - | Azure subscription ID |
| `resource_group` | `str` | Yes | - | Azure resource group name |
| `environment_name` | `str` | Yes | - | Container Apps environment name |
| `location` | `str` | No | `"eastus"` | Azure region |
| `cpu` | `str` | No | `"0.5"` | CPU cores (`0.25`, `0.5`, `0.75`, `1`, `1.25`, `1.5`, `1.75`, `2`) |
| `memory` | `str` | No | `"1Gi"` | Memory allocation (must be 2x CPU in Gi) |
| `ingress_external` | `bool` | No | `True` | Whether ingress is externally accessible |
| `min_replicas` | `int` | No | `1` | Minimum replicas |
| `max_replicas` | `int` | No | `10` | Maximum replicas |

**Validation Rules**:
- `subscription_id` must be a valid UUID
- `resource_group` must match pattern `^[-\w._()]+$`
- `cpu` must be one of the allowed values
- `memory` must be 2x the CPU value in Gi (e.g., `0.5` CPU → `1Gi` memory)
- `min_replicas` >= 0, `max_replicas` <= 300

---

### ContainerImage (Runtime Entity)

**Purpose**: Represents a built container image in the local Docker daemon. Not persisted to YAML.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Full image name including registry |
| `tag` | `str` | Image tag |
| `image_id` | `str` | Docker image ID (SHA256 hash) |
| `size_bytes` | `int` | Image size in bytes |
| `created_at` | `datetime` | Image creation timestamp |
| `labels` | `dict[str, str]` | Image labels (OCI standard + custom) |

**Labels Applied**:
- `org.opencontainers.image.title` - Agent name
- `org.opencontainers.image.version` - Version tag
- `org.opencontainers.image.created` - Build timestamp (ISO 8601)
- `org.opencontainers.image.source` - Git SHA
- `com.holodeck.managed` - Always `"true"`

---

### CloudDeployment (Runtime Entity)

**Purpose**: Represents a deployed instance on a cloud platform. Persisted to `.holodeck/deployments.json` for lifecycle management.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Cloud provider (`aws`, `gcp`, `azure`) |
| `service_id` | `str` | Provider-specific service identifier (ARN, name, resource ID) |
| `service_name` | `str` | Human-readable service name |
| `url` | `str` | Public URL of the deployed service |
| `status` | `str` | Deployment status (`PROVISIONING`, `RUNNING`, `FAILED`, `DELETED`) |
| `created_at` | `datetime` | Initial deployment timestamp |
| `updated_at` | `datetime` | Last update timestamp |
| `image_uri` | `str` | Deployed container image URI |
| `config_hash` | `str` | Hash of deployment config for change detection |

**Status Transitions**:
```
PROVISIONING → RUNNING (success)
PROVISIONING → FAILED (error)
RUNNING → UPDATING (during update)
UPDATING → RUNNING (update success)
UPDATING → FAILED (update error)
RUNNING → DELETED (destroy command)
```

---

## YAML Configuration Examples

### Minimal AWS Deployment

```yaml
deployment:
  registry:
    url: 123456789012.dkr.ecr.us-east-1.amazonaws.com
    repository: my-agents
  target:
    provider: aws
    aws:
      ecr_role_arn: arn:aws:iam::123456789012:role/AppRunnerECRRole
```

### Full GCP Deployment

```yaml
deployment:
  runtime: docker
  protocol: both
  port: 8080
  registry:
    url: us-docker.pkg.dev
    repository: my-project/agents/customer-support
    tag_strategy: semver
  target:
    provider: gcp
    gcp:
      project_id: my-gcp-project
      region: us-central1
      memory: 1Gi
      cpu: 1000m
      concurrency: 100
      min_instances: 1
      max_instances: 50
  environment:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
    LOG_LEVEL: INFO
```

### Azure Deployment with Custom Tag

```yaml
deployment:
  registry:
    url: myacr.azurecr.io
    repository: holodeck-agents
    tag_strategy: custom
    custom_tag: production-v2.1.0
  target:
    provider: azure
    azure:
      subscription_id: 12345678-1234-1234-1234-123456789012
      resource_group: holodeck-prod
      environment_name: holodeck-env
      location: eastus
      cpu: "1"
      memory: 2Gi
      min_replicas: 2
      max_replicas: 20
```

---

## State File Format

**Location**: `.holodeck/deployments.json`

```json
{
  "version": "1.0",
  "deployments": {
    "my-agent": {
      "provider": "aws",
      "service_id": "arn:aws:apprunner:us-east-1:123456789012:service/my-agent/abc123",
      "service_name": "my-agent",
      "url": "https://abc123.us-east-1.awsapprunner.com",
      "status": "RUNNING",
      "created_at": "2026-01-24T10:30:00Z",
      "updated_at": "2026-01-24T10:35:00Z",
      "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-agents:abc12345",
      "config_hash": "sha256:1234567890abcdef"
    }
  }
}
```
